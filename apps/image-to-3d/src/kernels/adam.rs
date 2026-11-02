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
//! Implemented: config types, [`StepPlan::due`], [`LrSchedule::at`]. STUBS: the
//! device optimizer.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use spl_core::Result;
use spl_gpu::cudarc::driver::{CudaFunction, CudaSlice, DeviceRepr, ValidAsZeroBits};
use ts_rs::TS;

use crate::gpu::{
    COMMON_CUH, CudaFile, Defines, Gpu, KernelSource, SPLAT_COMMON_CUH, not_implemented,
};
use crate::model::{Gaussians, Grads, Group};

const OWNER: &str = "optimizer (E5)";

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

/// Adam moments, per group `float2 (m, v)` at the parameter's element index
/// (f32), or two bf16 packed in a u32 (bf16 modes).
pub struct Moments {
    pub precision: MomentPrecision,
    /// Per group, 2 f32 words per element (f32) or 1 (bf16).
    pub groups: [CudaSlice<f32>; 6],
}

/// What a fused-Adam sink needs on the device: the plan lives in IterParams,
/// so this only carries the SH-rest options (spec-raster 5.5).
pub struct AdamDevice {
    /// Store only (g_raw, d) per visible Gaussian; SH Adam recomputes dL/df.
    pub sh_rank1: bool,
    /// Accumulate SH-rest gradients and step every 16 iterations.
    pub sh_accumulate: bool,
}

pub struct AdamKernels {
    pub step: CudaFunction,
    pub invisible: CudaFunction,
    pub catchup: CudaFunction,
}

impl AdamKernels {
    pub fn load(gpu: &Gpu, bf16_moments: bool) -> Result<Self> {
        let k = gpu.kernels(
            &SOURCE,
            &Defines::new().set("ADAM_MOMENTS_BF16", u8::from(bf16_moments)),
        )?;
        Ok(Self {
            step: k.get("adam_step")?,
            invisible: k.get("adam_invisible")?,
            catchup: k.get("adam_catchup")?,
        })
    }
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
}

impl Adam {
    pub fn new(
        _gpu: &Gpu,
        _capacity: usize,
        _cfg: AdamConfig,
        _groups: [GroupConfig; 6],
        _extent: f32,
    ) -> Result<Self> {
        Err(not_implemented("Adam::new", OWNER))
    }

    /// Host side: which groups step at t, their step sizes and bias corrections
    /// (f64), write modes. Advances step counts of due groups. Pure CPU.
    pub fn plan(&mut self, _t: u32) -> AdamIterPlan {
        AdamIterPlan::default()
    }

    /// Non-fused path: one launch over all due groups.
    pub fn step(
        &mut self,
        _gpu: &Gpu,
        _g: &mut Gaussians,
        _grads: &mut Grads,
        _params: &crate::frame::IterParamsDev,
    ) -> Result<()> {
        Err(not_implemented("Adam::step", OWNER))
    }

    pub fn zero_moments(&mut self, _gpu: &Gpu, _group: Group) -> Result<()> {
        Err(not_implemented("Adam::zero_moments", OWNER))
    }

    /// LazyCatchUp: bring every Gaussian to its dense-equivalent state.
    pub fn flush(&mut self, _gpu: &Gpu, _g: &mut Gaussians, _t: u32) -> Result<()> {
        Err(not_implemented("Adam::flush", OWNER))
    }

    /// Compaction and Morton sorting move these.
    pub fn moments_mut(&mut self) -> &mut Moments {
        &mut self.moments
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FastGS plan for SH rest: every 16 to 15k, every 32 to 20k, every 64 after.
    #[test]
    fn step_plan_is_piecewise() {
        let p = StepPlan(vec![(0, 16), (15_001, 32), (20_001, 64)]);
        assert!(p.due(16) && !p.due(17) && p.due(14_992) && !p.due(15_000));
        assert!(p.due(15_040) && !p.due(15_024));
        assert!(p.due(20_032) && !p.due(20_064) && p.due(20_096));
        assert!(StepPlan::every_iteration().due(7));
        assert!(StepPlan(vec![]).due(3));
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-6 * b.abs()
    }

    #[test]
    fn means_lr_decays_exponentially_from_init_to_final() {
        let s = LrSchedule::ExpDecay {
            init: 1.6e-4,
            final_: 1.6e-6,
            max_steps: 30_000,
            hold_until: 0,
            scale_by_extent: true,
        };
        assert!(close(s.at(0, 2.0), 3.2e-4));
        assert!(close(s.at(15_000, 1.0), 1.6e-5));
        assert!(close(s.at(40_000, 1.0), 1.6e-6));
        let held = LrSchedule::ExpDecay {
            init: 1.0,
            final_: 0.01,
            max_steps: 100,
            hold_until: 50,
            scale_by_extent: false,
        };
        assert_eq!(held.at(30, 1.0), 1.0);
        assert!(close(held.at(75, 1.0), 0.1));
    }

    #[test]
    fn warmup_then_decay() {
        let s = LrSchedule::WarmupExp {
            base: 2e-3,
            warmup_steps: 500,
            start_factor: 0.01,
            decay_steps: 29_500,
            final_factor: 0.01,
        };
        assert!(close(s.at(0, 1.0), 2e-5));
        assert!(close(s.at(500, 1.0), 2e-3));
        assert!(close(s.at(30_000, 1.0), 2e-5));
    }
}
