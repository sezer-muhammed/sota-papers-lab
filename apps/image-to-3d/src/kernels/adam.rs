//! Multi-group Adam over the SoA store: plans, learning-rate schedules, step
//! plans, sparsity modes, bf16 moments, LazyCatchUp. Owner: optimizer (E5).
//! Kernels: `cuda/adam.cuh` (device helpers, also spliced into preprocess for
//! fused Adam), `cuda/adam.cu`. Spec: spec-train 4.
//!
//! Buffer contract: reads `Grads` (Write sink) and `Gaussians::last_visible`
//! (Overwrite mode: g = 0 unless `last_visible[i] == t`), updates parameters
//! and [`Moments`] in place, zeroes accumulated gradients it steps. Starts
//! with the overflow check. The host-side plan ([`Adam::plan`]) goes into
//! `IterParams::adam`.
//!
//! Write modes and the zero-after bit: see [`Adam::plan`]. Sparsity: `Dense`
//! (PyTorch), `VisibleOnly` (skip Gaussians not rendered since the group's
//! previous step), `LazyCatchUp` (VisibleOnly plus the closed-form catch-up of
//! the skipped steps, [`Adam::flush`] before anything reads every Gaussian).

use anyhow::ensure;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use spl_core::Result;
use spl_gpu::cudarc::driver::{
    CudaFunction, CudaSlice, CudaStream, DevicePtr, DeviceRepr, LaunchConfig, PushKernelArg,
    ValidAsZeroBits,
};
use ts_rs::TS;

use crate::frame::IterParams;
use crate::gpu::{
    COMMON_CUH, CudaFile, CudaResultExt, Defines, Gpu, KernelSource, SPLAT_COMMON_CUH,
};
use crate::kernels::layout::STATUS_LEN;
use crate::model::{Gaussians, Grads, Group, round_cap};

pub const ADAM_CUH: CudaFile = CudaFile {
    name: "adam.cuh",
    text: include_str!("../../cuda/adam.cuh"),
};

pub const SOURCE: KernelSource = KernelSource {
    module: "adam",
    files: &[
        COMMON_CUH,
        SPLAT_COMMON_CUH,
        ADAM_CUH,
        CudaFile {
            name: "adam.cu",
            text: include_str!("../../cuda/adam.cu"),
        },
    ],
    kernels: &["adam_step", "adam_invisible", "adam_catchup"],
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export, export_to = "image-to-3d/")]
pub enum MomentPrecision {
    #[default]
    F32,
    /// Both moments bf16 with stochastic rounding.
    Bf16Stochastic,
    /// m bf16 (stochastic rounding), v f32.
    Bf16MomentF32Var,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export, export_to = "image-to-3d/")]
pub enum Sparsity {
    /// Exact PyTorch semantics: every Gaussian steps, invisible ones with g = 0.
    #[default]
    Dense,
    /// Skip Gaussians not visible in the accumulation window (Taming).
    VisibleOnly,
    /// VisibleOnly plus the closed-form catch-up of skipped steps (4.6).
    LazyCatchUp,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct AdamConfig {
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub bias_correction: bool,
    pub moments: MomentPrecision,
    pub sparsity: Sparsity,
    /// Phase 3b: Adam inside preprocess_bwd (kept only if ≥ 5% faster).
    pub fused_in_backward: bool,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-15,
            bias_correction: true,
            moments: MomentPrecision::F32,
            sparsity: Sparsity::Dense,
            fused_in_backward: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export, export_to = "image-to-3d/")]
pub enum LrSchedule {
    Constant(f32),
    /// lr(t) = scale·exp(lerp(ln init, ln final, clamp((t − hold_until)/(max_steps − hold_until), 0, 1))),
    /// scale = scene extent when `scale_by_extent`.
    ExpDecay {
        init: f32,
        #[serde(rename = "final")]
        final_: f32,
        max_steps: u32,
        hold_until: u32,
        scale_by_extent: bool,
    },
    /// PPISP: linear warm-up from `start_factor`·base, then exponential decay to `final_factor`·base.
    WarmupExp {
        base: f32,
        warmup_steps: u32,
        start_factor: f32,
        decay_steps: u32,
        final_factor: f32,
    },
}

impl LrSchedule {
    /// Learning rate at iteration `t` (f64: the host computes step sizes in f64).
    pub fn at(&self, t: u32, extent: f32) -> f64 {
        match *self {
            LrSchedule::Constant(lr) => lr as f64,
            LrSchedule::ExpDecay {
                init,
                final_,
                max_steps,
                hold_until,
                scale_by_extent,
            } => {
                let span = max_steps.saturating_sub(hold_until).max(1) as f64;
                let f = ((t.saturating_sub(hold_until)) as f64 / span).clamp(0.0, 1.0);
                let lr = ((init as f64).ln() * (1.0 - f) + (final_ as f64).ln() * f).exp();
                if scale_by_extent {
                    lr * extent as f64
                } else {
                    lr
                }
            }
            LrSchedule::WarmupExp {
                base,
                warmup_steps,
                start_factor,
                decay_steps,
                final_factor,
            } => {
                let base = base as f64;
                if t < warmup_steps {
                    let f = t as f64 / warmup_steps.max(1) as f64;
                    base * (start_factor as f64 + (1.0 - start_factor as f64) * f)
                } else {
                    let f = ((t - warmup_steps) as f64 / decay_steps.max(1) as f64).clamp(0.0, 1.0);
                    base * (final_factor as f64).powf(f)
                }
            }
        }
    }
}

/// Piecewise "step every n" plan: `[(from_iteration, every)]` sorted by
/// `from_iteration`; gradients accumulate between steps. A group is due at t
/// when the entry in force at t (largest `from ≤ t`) has `t % every == 0`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct StepPlan(pub Vec<(u32, u32)>);

impl StepPlan {
    pub fn every_iteration() -> Self {
        Self(vec![(0, 1)])
    }

    pub fn due(&self, t: u32) -> bool {
        let every = self
            .0
            .iter()
            .take_while(|(from, _)| *from <= t)
            .last()
            .map_or(1, |&(_, every)| every.max(1));
        t.is_multiple_of(every)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct GroupConfig {
    pub lr: LrSchedule,
    pub plan: StepPlan,
}

/// One iteration's optimizer plan as the kernels read it (inside `IterParams`). 84 B.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AdamIterPlan {
    /// η(t) / (1 − β1^k) per group.
    pub step_size: [f32; 6],
    /// 1 / sqrt(1 − β2^k) per group.
    pub bc2_sqrt_rcp: [f32; 6],
    /// Bit g: group g steps this iteration.
    pub due_mask: u32,
    /// Bit g: the raster overwrites (instead of accumulating) group g's gradient.
    pub overwrite_mask: u32,
    /// Flat element range prefix over due groups (×n).
    pub elem_prefix: [u32; 7],
}

const _: () = assert!(std::mem::size_of::<AdamIterPlan>() == 84);
// SAFETY: #[repr(C)] f32/u32 fields, no padding (size asserted); zero is valid.
unsafe impl DeviceRepr for AdamIterPlan {}
unsafe impl ValidAsZeroBits for AdamIterPlan {}

pub struct AdamKernels {
    pub step: CudaFunction,
    pub invisible: CudaFunction,
    pub catchup: CudaFunction,
}

impl AdamKernels {
    /// f32 moments, or both bf16 with stochastic rounding.
    pub fn load(gpu: &Gpu, bf16_moments: bool) -> Result<Self> {
        Self::with_precision(
            gpu,
            if bf16_moments {
                MomentPrecision::Bf16Stochastic
            } else {
                MomentPrecision::F32
            },
        )
    }

    pub fn with_precision(gpu: &Gpu, moments: MomentPrecision) -> Result<Self> {
        let k = gpu.kernels(
            &SOURCE,
            &Defines::new().set("ADAM_MOMENTS_BF16", moments.define()),
        )?;
        Ok(Self {
            step: k.get("adam_step")?,
            invisible: k.get("adam_invisible")?,
            catchup: k.get("adam_catchup")?,
        })
    }
}

impl MomentPrecision {
    /// `ADAM_MOMENTS_BF16` of adam.cuh.
    pub const fn define(self) -> u8 {
        match self {
            MomentPrecision::F32 => 0,
            MomentPrecision::Bf16Stochastic => 1,
            MomentPrecision::Bf16MomentF32Var => 2,
        }
    }

    /// f32 words per element of a moment buffer.
    pub const fn words(self) -> usize {
        match self {
            MomentPrecision::Bf16Stochastic => 1,
            MomentPrecision::F32 | MomentPrecision::Bf16MomentF32Var => 2,
        }
    }
}

impl Sparsity {
    const fn code(self) -> u32 {
        match self {
            Sparsity::Dense => 0,
            Sparsity::VisibleOnly => 1,
            Sparsity::LazyCatchUp => 2,
        }
    }
}

/// `AdamIterPlan::overwrite_mask` bit `ZERO_AFTER_SHIFT + g`: the kernel clears
/// group g's gradient after stepping it (the next iteration accumulates).
pub const ZERO_AFTER_SHIFT: u32 = 8;
/// Segments per step plan the device understands (`PLAN_SEGS`).
pub const PLAN_SEGS: usize = 4;
/// LazyCatchUp ring of per-step (step size, bias correction) per group (`LR_HIST`).
pub const LR_HIST: usize = 8192;
const THREADS: u32 = 256;
/// Resident 256-thread blocks per SM for the persistent grid (1536 threads on sm_89).
const BLOCKS_PER_SM: u32 = 6;

/// Fixed per run (adam.cuh `AdamHyper`). 24 B.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AdamHyper {
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub sparsity: u32,
    /// 1 − β rounded once from f64 (PyTorch's `value = 1 - beta`).
    pub one_minus_beta1: f32,
    pub one_minus_beta2: f32,
}

const _: () = assert!(std::mem::size_of::<AdamHyper>() == 24);
// SAFETY: #[repr(C)] f32/u32 fields, no padding (size asserted).
unsafe impl DeviceRepr for AdamHyper {}

/// Every group's step plan as the kernels read it (adam.cuh `StepPlansDev`). 240 B.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StepPlansDev {
    pub from: [[u32; PLAN_SEGS]; 6],
    pub every: [[u32; PLAN_SEGS]; 6],
    pub n_seg: [u32; 6],
    /// Groups with equal plans share an id (one LazyCatchUp stamp array each).
    pub plan_id: [u32; 6],
}

const _: () = assert!(std::mem::size_of::<StepPlansDev>() == 240);
// SAFETY: #[repr(C)] u32 arrays, no padding (size asserted).
unsafe impl DeviceRepr for StepPlansDev {}

impl StepPlansDev {
    /// Packs the plans; returns the number of distinct plans.
    pub fn new(groups: &[GroupConfig; 6]) -> Result<(Self, usize)> {
        let mut d = Self::default();
        let mut distinct: Vec<&StepPlan> = Vec::new();
        for (g, cfg) in groups.iter().enumerate() {
            let segs = &cfg.plan.0;
            ensure!(
                segs.len() <= PLAN_SEGS,
                "step plan of group {g} has {} segments, at most {PLAN_SEGS} are supported",
                segs.len()
            );
            ensure!(
                segs.windows(2).all(|w| w[0].0 < w[1].0),
                "step plan of group {g} is not sorted by iteration"
            );
            for (i, &(from, every)) in segs.iter().enumerate() {
                d.from[g][i] = from;
                d.every[g][i] = every.max(1);
            }
            d.n_seg[g] = segs.len() as u32;
            let id = match distinct.iter().position(|p| **p == cfg.plan) {
                Some(i) => i,
                None => {
                    distinct.push(&cfg.plan);
                    distinct.len() - 1
                }
            };
            d.plan_id[g] = id as u32;
        }
        Ok((d, distinct.len()))
    }
}

impl StepPlan {
    /// Largest due iteration before `t` (0 = none; iterations start at 1).
    pub fn prev_due(&self, t: u32) -> u32 {
        (1..t).rev().find(|&x| self.due(x)).unwrap_or(0)
    }

    /// Due iterations in `(a, b]`.
    pub fn due_count(&self, a: u32, b: u32) -> u32 {
        (a.saturating_add(1)..=b).filter(|&x| self.due(x)).count() as u32
    }
}

/// Device pointers of every group, by value as a kernel argument
/// (`GroupPtrs` in adam.cuh). 144 B.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GroupPtrs {
    pub param: [u64; 6],
    pub grad: [u64; 6],
    pub moments: [u64; 6],
}

// SAFETY: plain #[repr(C)] array of device addresses.
unsafe impl DeviceRepr for GroupPtrs {}

fn dptr<T>(s: &CudaSlice<T>, stream: &CudaStream) -> u64 {
    s.device_ptr(stream).0
}

impl GroupPtrs {
    /// Parameters and moments; gradients when `grads` is given (0 otherwise).
    pub fn new(gpu: &Gpu, g: &Gaussians, grads: Option<&Grads>, m: &Moments) -> Self {
        let s = &gpu.stream;
        let mut p = Self::default();
        for grp in Group::ALL {
            let i = grp.index();
            p.param[i] = dptr(g.param(grp), s);
            p.grad[i] = grads.map_or(0, |gr| dptr(gr.get(grp), s));
            p.moments[i] = dptr(&m.groups[i], s);
        }
        p
    }
}

/// Adam moments, per group `float2 (m, v)` at the parameter's element index
/// (f32), or two bf16 packed in a u32 (bf16 modes).
pub struct Moments {
    pub precision: MomentPrecision,
    /// Per group, [`MomentPrecision::words`] f32 words per element.
    pub groups: [CudaSlice<f32>; 6],
}

impl Moments {
    pub fn alloc(gpu: &Gpu, cap: usize, precision: MomentPrecision) -> Result<Self> {
        let s = &gpu.stream;
        let z = |g: Group| {
            s.alloc_zeros::<f32>(cap * g.floats() * precision.words())
                .cuda("alloc Adam moments")
        };
        Ok(Self {
            precision,
            groups: [
                z(Group::Means)?,
                z(Group::Scales)?,
                z(Group::Rotations)?,
                z(Group::Opacities)?,
                z(Group::Sh0)?,
                z(Group::ShRest)?,
            ],
        })
    }
}

/// What a fused-Adam sink needs on the device: the plan lives in IterParams,
/// so this only carries the SH-rest options (spec-raster 5.5).
pub struct AdamDevice {
    /// Store only (g_raw, d) per visible Gaussian; SH Adam recomputes dL/df.
    pub sh_rank1: bool,
    /// Accumulate SH-rest gradients and step every 16 iterations.
    pub sh_accumulate: bool,
}

/// LazyCatchUp state: stamps per distinct plan and the per-step LR ring.
pub struct LazyState {
    /// `[n_plans][cap]`: iteration up to which each Gaussian's state is current.
    /// Compaction moves these with the Gaussians (new Gaussians: the current t).
    pub last_update: CudaSlice<u32>,
    /// `[6][LR_HIST]` float2 (step size, bias correction) by group step index.
    pub lr_hist: CudaSlice<f32>,
    pub n_plans: usize,
    /// Scratch for `flush` (outside the graph): IterParams and a zero status.
    params: CudaSlice<IterParams>,
    status: CudaSlice<u32>,
}

/// Moments per group, step counts, LR history (LazyCatchUp), kernels.
pub struct Adam {
    pub cfg: AdamConfig,
    pub groups: [GroupConfig; 6],
    pub extent: f32,
    /// Steps taken per group (PyTorch keeps one counter per tensor).
    pub steps: [u32; 6],
    pub moments: Moments,
    pub kernels: AdamKernels,
    pub plans: StepPlansDev,
    pub lazy: Option<LazyState>,
    /// MCMC regularizer weights folded per Gaussian: (λ_o / n, λ_s / (3n)); 0 = off.
    pub reg: [f32; 2],
    pub cap: u32,
    grid: u32,
}

impl Adam {
    pub fn new(
        gpu: &Gpu,
        capacity: usize,
        cfg: AdamConfig,
        groups: [GroupConfig; 6],
        extent: f32,
    ) -> Result<Self> {
        let cap = round_cap(capacity.max(1));
        let (plans, n_plans) = StepPlansDev::new(&groups)?;
        let s = &gpu.stream;
        let lazy = if cfg.sparsity == Sparsity::LazyCatchUp {
            Some(LazyState {
                last_update: s
                    .alloc_zeros::<u32>(n_plans * cap)
                    .cuda("alloc LazyCatchUp stamps")?,
                lr_hist: s
                    .alloc_zeros::<f32>(6 * LR_HIST * 2)
                    .cuda("alloc LazyCatchUp LR history")?,
                n_plans,
                params: s.alloc_zeros::<IterParams>(1).cuda("alloc flush params")?,
                status: s
                    .alloc_zeros::<u32>(STATUS_LEN)
                    .cuda("alloc flush status")?,
            })
        } else {
            None
        };
        Ok(Self {
            kernels: AdamKernels::with_precision(gpu, cfg.moments)?,
            moments: Moments::alloc(gpu, cap, cfg.moments)?,
            cfg,
            groups,
            extent,
            steps: [0; 6],
            plans,
            lazy,
            reg: [0.0; 2],
            cap: cap as u32,
            grid: gpu.multiprocessors() * BLOCKS_PER_SM,
        })
    }

    fn hyper(&self) -> AdamHyper {
        AdamHyper {
            beta1: self.cfg.beta1,
            beta2: self.cfg.beta2,
            eps: self.cfg.eps,
            sparsity: self.cfg.sparsity.code(),
            one_minus_beta1: one_minus(self.cfg.beta1),
            one_minus_beta2: one_minus(self.cfg.beta2),
        }
    }

    /// Host side: which groups step at t, their step sizes and bias corrections
    /// (f64), write modes. Advances step counts of due groups. Pure CPU.
    ///
    /// Write mode: Overwrite when the group steps at t and stepped at t − 1 (no
    /// pending accumulation), else Accumulate. Zero-after when the group steps
    /// at t but not at t + 1 (the raster accumulates next).
    pub fn plan(&mut self, t: u32) -> AdamIterPlan {
        plan_for(&self.groups, &self.cfg, self.extent, &mut self.steps, t)
    }

    /// Non-fused path: one launch over all due groups (LazyCatchUp: the catch-up
    /// of the Gaussians that step first).
    pub fn step(
        &mut self,
        gpu: &Gpu,
        g: &mut Gaussians,
        grads: &mut Grads,
        params: &crate::frame::IterParamsDev,
    ) -> Result<()> {
        ensure!(
            g.cap <= self.cap && grads.cap <= self.cap,
            "Adam capacity {} is smaller than the store's {}",
            self.cap,
            g.cap
        );
        let ptrs = GroupPtrs::new(gpu, g, Some(grads), &self.moments);
        let h = self.hyper();
        let cfg = LaunchConfig {
            grid_dim: (self.grid, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        if let Some(lz) = &mut self.lazy {
            let flush = 0u32;
            let mut l = gpu.stream.launch_builder(&self.kernels.catchup);
            l.arg(&params.params)
                .arg(&ptrs)
                .arg(&h)
                .arg(&self.plans)
                .arg(&g.last_visible)
                .arg(&mut lz.last_update)
                .arg(&lz.lr_hist)
                .arg(&self.cap)
                .arg(&flush)
                .arg(&params.status);
            // SAFETY: adam_catchup(const IterParams*, GroupPtrs, AdamHyper, StepPlansDev,
            // const u32* last_visible, u32* last_update [n_plans][cap], const float2* lr_hist
            // [6][LR_HIST], u32 n_cap, u32 flush, const u32* status); every pointer in
            // `ptrs` addresses a buffer of `cap` Gaussians (checked above).
            unsafe { l.launch(cfg) }.cuda("launch adam_catchup")?;
        }
        let reg = Float2 {
            x: self.reg[0],
            y: self.reg[1],
        };
        let hist = self
            .lazy
            .as_ref()
            .map_or(0, |lz| dptr(&lz.lr_hist, &gpu.stream));
        let mut l = gpu.stream.launch_builder(&self.kernels.step);
        l.arg(&params.params)
            .arg(&ptrs)
            .arg(&h)
            .arg(&self.plans)
            .arg(&g.last_visible)
            .arg(&self.cap)
            .arg(&reg)
            .arg(&hist)
            .arg(&params.status);
        // SAFETY: adam_step(const IterParams*, GroupPtrs, AdamHyper, StepPlansDev,
        // const u32* last_visible, u32 n_cap, float2 reg, float2* lr_hist (or null),
        // const u32* status); buffers hold `cap` Gaussians, the kernel stops at
        // IterParams.frame.n_gaussians.
        unsafe { l.launch(cfg) }.map(drop).cuda("launch adam_step")
    }

    /// Dense mode after fused Adam in the backward: steps the Gaussians not
    /// rendered this iteration (g = 0).
    pub fn step_invisible(
        &mut self,
        gpu: &Gpu,
        g: &mut Gaussians,
        params: &crate::frame::IterParamsDev,
    ) -> Result<()> {
        ensure!(
            g.cap <= self.cap,
            "Adam capacity {} < store {}",
            self.cap,
            g.cap
        );
        let ptrs = GroupPtrs::new(gpu, g, None, &self.moments);
        let h = self.hyper();
        let mut l = gpu.stream.launch_builder(&self.kernels.invisible);
        l.arg(&params.params)
            .arg(&ptrs)
            .arg(&h)
            .arg(&self.plans)
            .arg(&g.last_visible)
            .arg(&self.cap)
            .arg(&params.status);
        // SAFETY: adam_invisible(const IterParams*, GroupPtrs, AdamHyper, StepPlansDev,
        // const u32* last_visible, u32 n_cap, const u32* status); gradients are not read.
        unsafe {
            l.launch(LaunchConfig {
                grid_dim: (self.grid, 1, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(drop)
        .cuda("launch adam_invisible")
    }

    pub fn zero_moments(&mut self, gpu: &Gpu, group: Group) -> Result<()> {
        gpu.stream
            .memset_zeros(&mut self.moments.groups[group.index()])
            .cuda("zero Adam moments")
    }

    /// LazyCatchUp: bring every Gaussian to its dense-equivalent state (after
    /// the step of iteration `t`; before densify, Morton, snapshots, export).
    /// A no-op in the other modes. Outside the graph (uploads its own params).
    pub fn flush(&mut self, gpu: &Gpu, g: &mut Gaussians, t: u32) -> Result<()> {
        let ptrs = GroupPtrs::new(gpu, g, None, &self.moments);
        let h = self.hyper();
        let Some(lz) = &mut self.lazy else {
            return Ok(());
        };
        let mut p = IterParams::default();
        p.frame.iteration = t;
        p.frame.n_gaussians = g.n;
        gpu.stream
            .memcpy_htod(std::slice::from_ref(&p), &mut lz.params)
            .cuda("upload flush params")?;
        let flush = 1u32;
        let mut l = gpu.stream.launch_builder(&self.kernels.catchup);
        l.arg(&lz.params)
            .arg(&ptrs)
            .arg(&h)
            .arg(&self.plans)
            .arg(&g.last_visible)
            .arg(&mut lz.last_update)
            .arg(&lz.lr_hist)
            .arg(&self.cap)
            .arg(&flush)
            .arg(&lz.status);
        // SAFETY: adam_catchup as in `step`, flush = 1; gradients are not read.
        unsafe {
            l.launch(LaunchConfig {
                grid_dim: (self.grid, 1, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(drop)
        .cuda("launch adam_catchup (flush)")
    }

    /// Compaction and Morton sorting move these.
    pub fn moments_mut(&mut self) -> &mut Moments {
        &mut self.moments
    }
}

/// 1 − β as PyTorch computes it: in f64 from the decimal value (the shortest
/// representation of the f32, so 0.9 → 0.1, not 0.10000002), rounded once.
fn one_minus(beta: f32) -> f32 {
    let b: f64 = beta.to_string().parse().unwrap_or(beta as f64);
    (1.0 - b) as f32
}

/// [`Adam::plan`] on bare state (shared with the CPU tests).
fn plan_for(
    groups: &[GroupConfig; 6],
    cfg: &AdamConfig,
    extent: f32,
    steps: &mut [u32; 6],
    t: u32,
) -> AdamIterPlan {
    let mut p = AdamIterPlan::default();
    let (b1, b2) = (cfg.beta1 as f64, cfg.beta2 as f64);
    for (g, gc) in groups.iter().enumerate() {
        let floats = Group::ALL[g].floats() as u32;
        p.elem_prefix[g + 1] = p.elem_prefix[g];
        if !gc.plan.due(t) {
            continue;
        }
        steps[g] += 1;
        let k = steps[g] as i32;
        let lr = gc.lr.at(t, extent);
        let (step, bc2) = if cfg.bias_correction {
            (lr / (1.0 - b1.powi(k)), 1.0 / (1.0 - b2.powi(k)).sqrt())
        } else {
            (lr, 1.0)
        };
        p.step_size[g] = step as f32;
        p.bc2_sqrt_rcp[g] = bc2 as f32;
        p.due_mask |= 1 << g;
        if t <= 1 || gc.plan.due(t - 1) {
            p.overwrite_mask |= 1 << g;
        }
        if !gc.plan.due(t + 1) {
            p.overwrite_mask |= 1 << (ZERO_AFTER_SHIFT + g as u32);
        }
        p.elem_prefix[g + 1] += floats;
    }
    p
}

/// `float2` kernel argument.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Float2 {
    x: f32,
    y: f32,
}

// SAFETY: #[repr(C)] pair of f32.
unsafe impl DeviceRepr for Float2 {}

#[cfg(test)]
mod tests;
