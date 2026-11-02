//! Densification strategies (ADC, FastGS VCD/VCP, MCMC), their statistics and
//! view scores, opacity clamp/reset. Owner: densify (E6). Kernels:
//! `cuda/densify.cu`, `cuda/compact.cu`, `cuda/mcmc.cu`; compaction and Morton
//! sorting live in `model.rs` (same owner). Spec: spec-train 5.
//!
//! Buffer contract: preprocess_bwd accumulates [`DensifyStats`] at p when the
//! strategy `wants_stats(t)`; `SplatRaster::count_masked` fills [`ViewScores`]
//! (through [`scores_add`]); `after_step` runs after the optimizer step of
//! iteration t (never before), may change n, and zeroes the statistics it
//! consumed. Every decision has a CPU mirror here ([`plan_cpu`],
//! [`relocation_cpu`]) that the GPU tests compare against bit for bit.
//!
//! Deviations from the references (all deliberate, spec-train 5.3/5.6):
//! FastGS's prune mask uses the pre-step indexing; MCMC also zeroes the
//! moments of relocation destinations; in-place compaction moves whole
//! column batches through the slab (the chunked `dst ≤ src` scheme is unsafe
//! for tail entries).

use serde::{Deserialize, Serialize};
use spl_core::Result;
use spl_gpu::cudarc::driver::{
    CudaFunction, CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits,
};

use super::RenderMode;
use super::adam::Adam;
use super::layout::RNG_VIEW_SAMPLE;
use super::loss::{ErrorMap, Loss, resample_target};
use super::raster::{FrameSource, SplatRaster};
use crate::frame::{IterParams, IterParamsDev};
use crate::gpu::{
    COMMON_CUH, CudaFile, CudaResultExt, Defines, Gpu, KernelSource, SPLAT_COMMON_CUH,
};
use crate::model::{
    Column, Compactor, EmitPlan, EmitResult, Gaussians, Grads, Group, PerGaussianArrays, TILE,
    Uint2, round_cap,
};
use crate::rng::Philox;
use crate::trainer::SceneGpu;

pub const DENSIFY_SOURCE: KernelSource = KernelSource {
    module: "densify",
    files: &[
        COMMON_CUH,
        SPLAT_COMMON_CUH,
        CudaFile {
            name: "densify.cu",
            text: include_str!("../../cuda/densify.cu"),
        },
    ],
    kernels: &[
        "vcd_mask",
        "scores_add",
        "minmax_f32",
        "densify_plan",
        "gumbel_keys",
        "vcp_mark",
        "opacity_clamp",
    ],
};

pub const COMPACT_SOURCE: KernelSource = KernelSource {
    module: "compact",
    files: &[
        COMMON_CUH,
        SPLAT_COMMON_CUH,
        CudaFile {
            name: "compact.cu",
            text: include_str!("../../cuda/compact.cu"),
        },
    ],
    kernels: &[
        "scan_u32",
        "emit_count",
        "emit_dst",
        "slab_load",
        "slab_scatter",
        "slab_gather",
        "slab_store",
        "split_children",
        "aabb_reduce",
        "morton_codes",
        "radix_hist",
        "radix_scatter",
    ],
};

pub const MCMC_SOURCE: KernelSource = KernelSource {
    module: "mcmc",
    files: &[
        COMMON_CUH,
        SPLAT_COMMON_CUH,
        CudaFile {
            name: "mcmc.cu",
            text: include_str!("../../cuda/mcmc.cu"),
        },
    ],
    kernels: &[
        "mcmc_weights",
        "fscan_reduce",
        "fscan_partials",
        "fscan_apply",
        "flag_count",
        "flag_list",
        "mcmc_draw",
        "mcmc_adjust",
        "copy_rows",
        "mcmc_noise",
    ],
};

/// Per Gaussian since the last densify step (spec-train 5.1); zeroed after each step.
pub struct DensifyStats {
    pub vis_count: CudaSlice<f32>,
    /// Σ |g_ndc|, g_ndc = (∂L/∂u·W/2, ∂L/∂v·H/2).
    pub grad_sum: CudaSlice<f32>,
    /// AbsGS: Σ |(Σ_p |∂L_p/∂u|·W/2, Σ_p |∂L_p/∂v|·H/2)|, only when requested.
    pub grad_abs_sum: Option<CudaSlice<f32>>,
}

impl DensifyStats {
    pub fn alloc(gpu: &Gpu, capacity: usize, abs_grad: bool) -> Result<Self> {
        let cap = round_cap(capacity.max(1));
        let s = &gpu.stream;
        Ok(Self {
            vis_count: s.alloc_zeros::<f32>(cap).cuda("alloc vis_count")?,
            grad_sum: s.alloc_zeros::<f32>(cap).cuda("alloc grad_sum")?,
            grad_abs_sum: if abs_grad {
                Some(s.alloc_zeros::<f32>(cap).cuda("alloc grad_abs_sum")?)
            } else {
                None
            },
        })
    }
}

/// FastGS multi-view scores, zeroed at the start of each scoring round.
pub struct ViewScores {
    pub count: CudaSlice<u32>,
    pub score: CudaSlice<f32>,
}

impl ViewScores {
    pub fn alloc(gpu: &Gpu, capacity: usize) -> Result<Self> {
        let cap = round_cap(capacity.max(1));
        let s = &gpu.stream;
        Ok(Self {
            count: s.alloc_zeros::<u32>(cap).cuda("alloc view counts")?,
            score: s.alloc_zeros::<f32>(cap).cuda("alloc view scores")?,
        })
    }
}

/// Everything a strategy may touch after the optimizer step.
pub struct DensifyCx<'a> {
    pub gpu: &'a Gpu,
    pub g: &'a mut Gaussians,
    pub grads: &'a mut Grads,
    pub adam: &'a mut Adam,
    pub stats: &'a mut DensifyStats,
    pub raster: &'a mut dyn SplatRaster,
    pub loss: &'a mut Loss,
    pub scene: &'a SceneGpu,
    /// Scene extent r (spec-train 8.5).
    pub extent: f32,
    pub res_scale: u32,
    pub rng: Philox,
    /// IterParams per scene view index for scoring renders; filled by the
    /// trainer only when `wants_views(t)` (empty otherwise).
    pub views: &'a [IterParams],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct DensifyOutcome {
    pub changed: bool,
    pub n_before: u32,
    pub n_after: u32,
    pub cloned: u32,
    pub split: u32,
    pub pruned: u32,
    pub relocated: u32,
    /// Tail entries dropped because the capacity was full (never silent).
    pub capped: u32,
}

pub trait Strategy: Send {
    /// Called after the optimizer step of iteration t. May change n.
    fn after_step(&mut self, cx: &mut DensifyCx<'_>, t: u32) -> Result<DensifyOutcome>;
    fn wants_abs_grad(&self) -> bool;
    /// The raster accumulates DensifyStats only when true.
    fn wants_stats(&self, t: u32) -> bool;
    /// The step at t renders training views (FastGS scoring): the trainer
    /// fills `DensifyCx::views`.
    fn wants_views(&self, _t: u32) -> bool {
        false
    }
}

/// `start < t < end` and `t % every == 0`: the densify iterations of every
/// strategy (ADC/FastGS-Big: 600..=14900 step 100; FastGS: 1000..=14500 step 500).
pub fn window_due(t: u32, start: u32, end: u32, every: u32) -> bool {
    t > start && t < end && every > 0 && t.is_multiple_of(every)
}

/// 3DGS / Faster-GS adaptive density control (spec-train 5.2).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdcConfig {
    /// Exclusive bounds: densify at `start < t < end`, `t % every == 0`.
    pub start: u32,
    pub end: u32,
    pub every: u32,
    /// Average NDC gradient threshold.
    pub grad_thresh: f32,
    /// Clone when max scale ≤ percent_dense·r, split otherwise.
    pub percent_dense: f32,
    pub min_opacity: f32,
    /// Opacity reset to 0.01 at multiples of this, before `end`.
    pub reset_every: u32,
    /// World-scale prune `max(s) > 0.1·r` only after this iteration.
    pub big_after: u32,
}

impl Default for AdcConfig {
    fn default() -> Self {
        Self {
            start: 500,
            end: 15_000,
            every: 100,
            grad_thresh: 2e-4,
            percent_dense: 0.01,
            min_opacity: 0.005,
            reset_every: 3000,
            big_after: 3000,
        }
    }
}

/// FastGS VCD/VCP (spec-train 5.3); code defaults, per-scene overrides in the presets.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FastGsConfig {
    pub start: u32,
    pub end: u32,
    /// 500 (base) or 100 (Big).
    pub every: u32,
    /// K sampled views per scoring round.
    pub views: u32,
    /// τ on the min-max normalized L1 map.
    pub loss_thresh: f32,
    /// Candidate iff floor(count / K) > τ_d.
    pub tau_d: u32,
    pub grad_thresh: f32,
    pub grad_abs_thresh: f32,
    /// Clone vs split by max scale ≤ dense·r.
    pub dense: f32,
    pub min_opacity: f32,
    pub reset_every: u32,
    /// Opacity clamped to ≤ this at every densify step (moments zeroed).
    pub clamp_opacity: f32,
    /// Final VCP prunes at `final_prune_from, + every, ..., ≤ until`.
    pub final_prune_from: u32,
    pub final_prune_every: u32,
    pub final_prune_until: u32,
    pub final_min_opacity: f32,
    pub final_score: f32,
    pub score_at_full_res: bool,
}

impl Default for FastGsConfig {
    fn default() -> Self {
        Self {
            start: 500,
            end: 15_000,
            every: 500,
            views: 10,
            loss_thresh: 0.1,
            tau_d: 5,
            grad_thresh: 2e-4,
            grad_abs_thresh: 0.0012,
            dense: 0.001,
            min_opacity: 0.005,
            reset_every: 3000,
            clamp_opacity: 0.8,
            final_prune_from: 18_000,
            final_prune_every: 3000,
            final_prune_until: 27_000,
            final_min_opacity: 0.1,
            final_score: 0.9,
            score_at_full_res: false,
        }
    }
}

impl FastGsConfig {
    /// FastGS-Big: densify every 100 (per-scene thresholds come from the presets).
    pub fn big() -> Self {
        Self {
            every: 100,
            ..Self::default()
        }
    }

    pub fn final_prune_due(&self, t: u32) -> bool {
        t >= self.final_prune_from
            && t <= self.final_prune_until
            && (t - self.final_prune_from).is_multiple_of(self.final_prune_every.max(1))
    }
}

/// 3DGS-MCMC with Faster-GS's fused kernels (spec-train 5.6).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct McmcConfig {
    pub start: u32,
    pub end: u32,
    pub every: u32,
    pub cap_max: u32,
    pub min_opacity: f32,
    /// n_target = min(cap_max, floor(grow·n)).
    pub grow: f32,
    /// Noise factor on the means LR (5e5).
    pub noise_lr: f32,
    pub opacity_reg: f32,
    pub scale_reg: f32,
    /// Zero moments of destinations too (the reference resets only sources).
    pub reset_dead_moments: bool,
}

impl Default for McmcConfig {
    fn default() -> Self {
        Self {
            start: 500,
            end: 25_000,
            every: 100,
            cap_max: 1_000_000,
            min_opacity: 0.005,
            grow: 1.05,
            noise_lr: 5e5,
            opacity_reg: 0.01,
            scale_reg: 0.01,
            reset_dead_moments: true,
        }
    }
}

/// Per-scene MCMC caps for Mip-NeRF 360 (Faster-GS, after Steiner et al.).
pub const MCMC_CAP_MAX_MIP360: &[(&str, u32)] = &[
    ("bicycle", 6_131_954),
    ("bonsai", 1_244_819),
    ("counter", 1_222_956),
    ("flowers", 3_636_448),
    ("garden", 5_834_784),
    ("kitchen", 1_852_335),
    ("room", 1_593_376),
    ("stump", 4_961_797),
    ("treehill", 3_783_761),
];

// ---------------------------------------------------------------- device plan rules

pub const PLAN_ADC: u32 = 0;
pub const PLAN_FASTGS: u32 = 1;
pub const PLAN_FINAL: u32 = 2;

/// ln 1.6: split children's log-scale offset (`LN_1_6` in compact.cu).
pub const LN_1_6: f32 = 0.470_003_63;

/// `DensifyRules` of densify.cu (64 B): thresholds in parameter space.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DensifyRules {
    pub mode: u32,
    pub grad_thresh: f32,
    pub grad_abs_thresh: f32,
    /// ln(dense · r): clone iff max log scale ≤ this.
    pub clone_log_scale: f32,
    /// ln(0.1 · r) when the world-size prune is active, else +inf.
    pub big_log_scale: f32,
    pub min_opacity_logit: f32,
    /// K.
    pub views: f32,
    /// Candidate iff floor(count / K) > tau.
    pub tau: f32,
    pub final_opacity_logit: f32,
    pub final_score: f32,
    pub pad: [u32; 6],
}

// SAFETY: plain #[repr(C)] scalars, mirrored by `DensifyRules` in densify.cu.
unsafe impl DeviceRepr for DensifyRules {}
// SAFETY: all-zero bits are a valid value.
unsafe impl ValidAsZeroBits for DensifyRules {}
const _: () = assert!(std::mem::size_of::<DensifyRules>() == 64);

pub fn logit(p: f32) -> f32 {
    (p / (1.0 - p)).ln()
}

impl DensifyRules {
    pub fn adc(c: &AdcConfig, extent: f32, t: u32) -> Self {
        Self {
            mode: PLAN_ADC,
            grad_thresh: c.grad_thresh,
            clone_log_scale: (c.percent_dense * extent).ln(),
            big_log_scale: if t > c.big_after {
                (0.1 * extent).ln()
            } else {
                f32::INFINITY
            },
            min_opacity_logit: logit(c.min_opacity),
            ..Default::default()
        }
    }

    /// `res_scale`: the scoring resolution divisor; footprint counts scale
    /// with pixel count, so τ_d becomes τ_d / s² (spec-train 5.4).
    pub fn fastgs(c: &FastGsConfig, extent: f32, t: u32, res_scale: u32) -> Self {
        let s = if c.score_at_full_res {
            1
        } else {
            res_scale.max(1)
        };
        Self {
            mode: PLAN_FASTGS,
            grad_thresh: c.grad_thresh,
            grad_abs_thresh: c.grad_abs_thresh,
            clone_log_scale: (c.dense * extent).ln(),
            big_log_scale: if t > c.reset_every {
                (0.1 * extent).ln()
            } else {
                f32::INFINITY
            },
            min_opacity_logit: logit(c.min_opacity),
            views: c.views as f32,
            tau: c.tau_d as f32 / (s * s) as f32,
            ..Default::default()
        }
    }

    pub fn fastgs_final(c: &FastGsConfig) -> Self {
        Self {
            mode: PLAN_FINAL,
            final_opacity_logit: logit(c.final_min_opacity),
            final_score: c.final_score,
            views: c.views as f32,
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------- CPU reference (bit-exact mirror)

/// Host inputs of one plan (compact, n entries).
#[derive(Clone, Debug, Default)]
pub struct PlanInputs {
    pub log_scale: Vec<f32>,
    pub quat: Vec<f32>,
    pub opacity: Vec<f32>,
    pub vis_count: Vec<f32>,
    pub grad_sum: Vec<f32>,
    pub grad_abs_sum: Option<Vec<f32>>,
    pub count: Vec<u32>,
    pub score: Vec<f32>,
}

/// What `densify_plan` writes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PlanOut {
    pub out_count: Vec<u32>,
    pub clone_flag: Vec<u8>,
    pub split_flag: Vec<u8>,
    pub priority: Vec<f32>,
    pub prune_candidates: u32,
}

fn avg(sum: f32, cnt: f32) -> f32 {
    if cnt > 0.0 { sum / cnt } else { 0.0 }
}

/// (min, max) of the scores as densify.cu's prune_score sees them.
pub fn score_range(score: &[f32]) -> (f32, f32) {
    score
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &x| {
            (a.min(x), b.max(x))
        })
}

pub fn prune_score(s: f32, (lo, hi): (f32, f32)) -> f32 {
    if hi > lo { (s - lo) / (hi - lo) } else { 0.0 }
}

/// `densify_plan` on the host, decision for decision.
pub fn plan_cpu(x: &PlanInputs, r: &DensifyRules) -> PlanOut {
    let n = x.opacity.len();
    let mm = score_range(&x.score);
    let mut out = PlanOut {
        out_count: vec![1; n],
        clone_flag: vec![0; n],
        split_flag: vec![0; n],
        priority: vec![0.0; n],
        prune_candidates: 0,
    };
    for i in 0..n {
        let ls = x.log_scale[3 * i]
            .max(x.log_scale[3 * i + 1])
            .max(x.log_scale[3 * i + 2]);
        let o = x.opacity[i];
        if r.mode == PLAN_FINAL {
            out.out_count[i] = u32::from(
                !(o < r.final_opacity_logit || prune_score(x.score[i], mm) > r.final_score),
            );
            continue;
        }
        let cnt = x.vis_count[i];
        let g = avg(x.grad_sum[i], cnt);
        let small = ls <= r.clone_log_scale;
        let (oc, cl, sp, pri);
        if r.mode == PLAN_ADC {
            let q = &x.quat[4 * i..4 * i + 4];
            let q2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
            let dead = o < r.min_opacity_logit || q2 < 1e-8;
            let hot = g >= r.grad_thresh;
            let kill = dead || ls > r.big_log_scale;
            let kill_child = dead || ls - LN_1_6 > r.big_log_scale;
            oc = u32::from(!(hot && !small) && !kill);
            cl = hot && small && !kill;
            sp = hot && !small && !kill_child;
            pri = g;
        } else {
            let imp = (x.count[i] as f32 / r.views).floor();
            let c = imp > r.tau;
            let ga = x.grad_abs_sum.as_ref().map_or(0.0, |a| avg(a[i], cnt));
            cl = c && small && g >= r.grad_thresh;
            sp = c && !small && ga >= r.grad_abs_thresh;
            let p = !sp && (o < r.min_opacity_logit || ls > r.big_log_scale);
            oc = if sp {
                0
            } else if p {
                2
            } else {
                1
            };
            out.prune_candidates += u32::from(p);
            pri = if sp { ga } else { g };
        }
        out.out_count[i] = oc;
        out.clone_flag[i] = u8::from(cl);
        out.split_flag[i] = u8::from(sp);
        out.priority[i] = pri;
    }
    out
}

/// Emission order on the host: for each source row, where it goes
/// (survivors, then clones, then split children), as `(src, dst, is_new, child)`.
pub fn emission_cpu(oc: &[u32], cf: &[u8], sf: &[u8]) -> (Vec<(usize, usize, bool, u8)>, usize) {
    let surv = oc.iter().filter(|&&c| c != 0).count();
    let clones = cf.iter().filter(|&&c| c != 0).count();
    let mut moves = Vec::new();
    let (mut k, mut c, mut s) = (0, surv, surv + clones);
    for i in 0..oc.len() {
        if oc[i] != 0 {
            moves.push((i, k, false, 0));
            k += 1;
        }
        if cf[i] != 0 {
            moves.push((i, c, true, 0));
            c += 1;
        }
        if sf[i] != 0 {
            moves.push((i, s, true, 1));
            moves.push((i, s + 1, true, 2));
            s += 2;
        }
    }
    (moves, s)
}

/// Binomial coefficients C(n, k) as an `n_max × n_max` f32 table (gsplat's `binoms`).
pub fn binomial_table(n_max: usize) -> Vec<f32> {
    let mut t = vec![0f32; n_max * n_max];
    for n in 0..n_max {
        let mut c = 1f64;
        for k in 0..=n {
            t[n * n_max + k] = c as f32;
            c = c * (n - k) as f64 / (k + 1) as f64;
        }
    }
    t
}

/// The MCMC relocation of one source drawn `draws` times (N = draws + 1,
/// capped at n_max) in f64: (o', ln coefficient on the scale). The kernel
/// evaluates the same sum in f32 (as gsplat does), which cancels for large N.
pub fn relocation_cpu(o: f64, draws: u32, binoms: &[f32], n_max: usize, min_o: f64) -> (f64, f64) {
    let nn = ((draws + 1) as usize).min(n_max);
    let o_new = 1.0 - (1.0 - o).powf(1.0 / nn as f64);
    let mut denom = 0.0;
    for a in 1..=nn {
        for k in 0..a {
            let sign = if k % 2 == 1 { -1.0 } else { 1.0 };
            denom += binoms[(a - 1) * n_max + k] as f64 * sign / ((k + 1) as f64).sqrt()
                * o_new.powi(k as i32 + 1);
        }
    }
    let coeff = o / denom;
    let o_new = o_new.clamp(min_o, 1.0 - f32::EPSILON as f64);
    (o_new, coeff.ln())
}

/// K distinct indices of `0..n`, uniform without replacement (partial
/// Fisher-Yates driven by Philox RNG_VIEW_SAMPLE at iteration `t`).
pub fn sample_views(rng: &Philox, n: usize, k: usize, t: u32) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    let k = k.min(n);
    for j in 0..k {
        let u = rng.uniform(j as u32, t, RNG_VIEW_SAMPLE);
        let r = j + ((u * (n - j) as f32) as usize).min(n - j - 1);
        idx.swap(j, r);
    }
    idx.truncate(k);
    idx
}

// ---------------------------------------------------------------- kernels and launchers

pub struct DensifyKernels {
    pub vcd_mask: CudaFunction,
    pub scores_add: CudaFunction,
    pub minmax_f32: CudaFunction,
    pub densify_plan: CudaFunction,
    pub gumbel_keys: CudaFunction,
    pub vcp_mark: CudaFunction,
    pub opacity_clamp: CudaFunction,
}

impl DensifyKernels {
    pub fn load(gpu: &Gpu) -> Result<Self> {
        let k = gpu.kernels(&DENSIFY_SOURCE, &Defines::new())?;
        Ok(Self {
            vcd_mask: k.get("vcd_mask")?,
            scores_add: k.get("scores_add")?,
            minmax_f32: k.get("minmax_f32")?,
            densify_plan: k.get("densify_plan")?,
            gumbel_keys: k.get("gumbel_keys")?,
            vcp_mark: k.get("vcp_mark")?,
            opacity_clamp: k.get("opacity_clamp")?,
        })
    }
}

pub struct McmcKernels {
    pub weights: CudaFunction,
    pub fscan_reduce: CudaFunction,
    pub fscan_partials: CudaFunction,
    pub fscan_apply: CudaFunction,
    pub flag_count: CudaFunction,
    pub flag_list: CudaFunction,
    pub draw: CudaFunction,
    pub adjust: CudaFunction,
    pub copy_rows: CudaFunction,
    pub noise: CudaFunction,
}

impl McmcKernels {
    pub fn load(gpu: &Gpu) -> Result<Self> {
        let k = gpu.kernels(&MCMC_SOURCE, &Defines::new())?;
        Ok(Self {
            weights: k.get("mcmc_weights")?,
            fscan_reduce: k.get("fscan_reduce")?,
            fscan_partials: k.get("fscan_partials")?,
            fscan_apply: k.get("fscan_apply")?,
            flag_count: k.get("flag_count")?,
            flag_list: k.get("flag_list")?,
            draw: k.get("mcmc_draw")?,
            adjust: k.get("mcmc_adjust")?,
            copy_rows: k.get("copy_rows")?,
            noise: k.get("mcmc_noise")?,
        })
    }
}

fn grid(n: u32, per: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(per).max(1), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// After a count-mode raster pass: scatter `count_v` (v-space) into `acc`
/// with the view weight and zero `count_v`. For `SplatRaster::count_masked`.
pub fn scores_add(
    gpu: &Gpu,
    k: &DensifyKernels,
    count_v: &mut CudaSlice<u32>,
    vis_idx: &CudaSlice<u32>,
    status: &CudaSlice<u32>,
    view_weight: f32,
    acc: &mut ViewScores,
) -> Result<()> {
    let mut l = gpu.stream.launch_builder(&k.scores_add);
    l.arg(&*count_v)
        .arg(vis_idx)
        .arg(status)
        .arg(&view_weight)
        .arg(&mut acc.count)
        .arg(&mut acc.score);
    let blocks = gpu.multiprocessors() * 8;
    // SAFETY: scores_add(count_v, vis_idx, status, w, count_p, score_p): grid-stride over
    // status[ST_N_VIS] <= V_cap entries; vis_idx < N_cap = acc capacity.
    unsafe {
        l.launch(LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .cuda("scores_add")
    .map(drop)
}

/// mask = normalized error > τ over `n_pixels`.
pub fn vcd_mask(
    gpu: &Gpu,
    k: &DensifyKernels,
    err: &ErrorMap,
    tau: f32,
    n_pixels: u32,
    mask: &mut CudaSlice<u8>,
) -> Result<()> {
    let mut l = gpu.stream.launch_builder(&k.vcd_mask);
    l.arg(&err.err)
        .arg(&err.minmax)
        .arg(&tau)
        .arg(&n_pixels)
        .arg(mask);
    // SAFETY: vcd_mask(err, minmax[2], tau, n_pixels, mask), one thread per pixel.
    unsafe { l.launch(grid(n_pixels, 256)) }
        .cuda("vcd_mask")
        .map(drop)
}

/// The MCMC position noise (spec-train 5.6), after the optimizer step; reads
/// n, the iteration, means_lr, noise_scale and the seed from `params`.
pub fn mcmc_noise(
    gpu: &Gpu,
    k: &McmcKernels,
    params: &IterParamsDev,
    g: &mut Gaussians,
) -> Result<()> {
    let mut l = gpu.stream.launch_builder(&k.noise);
    l.arg(&params.params)
        .arg(&g.log_scale)
        .arg(&g.quat)
        .arg(&g.opacity)
        .arg(&mut g.mean)
        .arg(&params.status);
    // SAFETY: mcmc_noise(IterParams*, log_scale, float4* quat, opacity, mean, status):
    // threads >= params.n_gaussians (<= g.cap) return.
    unsafe { l.launch(grid(g.cap, 256)) }
        .cuda("mcmc_noise")
        .map(drop)
}

/// Device scratch shared by every strategy, sized to the store's capacity.
pub struct Work {
    pub k: DensifyKernels,
    pub plan: EmitPlan,
    pub compactor: Compactor,
    /// [0]: prune candidates.
    pub counters: CudaSlice<u32>,
    /// Sortable (min, max) keys of the view scores.
    pub minmax: CudaSlice<u32>,
}

impl Work {
    pub fn new(gpu: &Gpu, cap: usize) -> Result<Self> {
        let s = &gpu.stream;
        Ok(Self {
            k: DensifyKernels::load(gpu)?,
            plan: EmitPlan::alloc(gpu, cap)?,
            compactor: Compactor::new(gpu, cap)?,
            counters: s.alloc_zeros::<u32>(4).cuda("alloc densify counters")?,
            minmax: s.alloc_zeros::<u32>(2).cuda("alloc score min/max")?,
        })
    }

    /// Sortable (min, max) of `x[..n]` into `self.minmax`.
    pub fn score_minmax(&mut self, gpu: &Gpu, x: &CudaSlice<f32>, n: u32) -> Result<()> {
        let s = &gpu.stream;
        s.memcpy_htod(&[u32::MAX, 0], &mut self.minmax)
            .cuda("init min/max")?;
        let mut l = s.launch_builder(&self.k.minmax_f32);
        l.arg(x).arg(&n).arg(&mut self.minmax);
        let blocks = n.div_ceil(256).clamp(1, gpu.multiprocessors() * 8);
        // SAFETY: minmax_f32(x, n, out[2]), grid-stride.
        unsafe {
            l.launch(LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .cuda("minmax_f32")
        .map(drop)
    }

    /// `densify_plan` over the n live Gaussians; returns the prune-candidate count (host sync).
    pub fn plan(
        &mut self,
        gpu: &Gpu,
        g: &Gaussians,
        stats: &DensifyStats,
        scores: Option<&ViewScores>,
        rules: &DensifyRules,
    ) -> Result<u32> {
        let s = &gpu.stream;
        let n = g.n;
        self.plan.n_before = n;
        s.memset_zeros(&mut self.counters)
            .cuda("zero densify counters")?;
        let null = 0u64;
        let mut l = s.launch_builder(&self.k.densify_plan);
        l.arg(&g.log_scale)
            .arg(&g.quat)
            .arg(&g.opacity)
            .arg(&stats.vis_count)
            .arg(&stats.grad_sum);
        match &stats.grad_abs_sum {
            Some(a) => l.arg(a),
            None => l.arg(&null),
        };
        match scores {
            Some(v) => l.arg(&v.count).arg(&v.score),
            None => l.arg(&null).arg(&null),
        };
        l.arg(&self.minmax)
            .arg(rules)
            .arg(&n)
            .arg(&mut self.plan.out_count)
            .arg(&mut self.plan.clone_flag)
            .arg(&mut self.plan.split_flag)
            .arg(&mut self.plan.priority)
            .arg(&mut self.counters);
        // SAFETY: densify_plan(log_scale, float4* quat, opacity, vis_count, grad_sum,
        // grad_abs_sum | null, count_p | null, score_p | null, score_minmax, DensifyRules, n,
        // out_count, clone_flag, split_flag, priority, counters); null inputs are only read in
        // modes that need them (abs: FastGS with stats; scores: FastGS modes).
        unsafe { l.launch(grid(n, 256)) }.cuda("densify_plan")?;
        let c = s
            .clone_dtoh(&self.counters.slice(0..1))
            .cuda("read prune candidates")?;
        Ok(c[0])
    }

    /// In-densify VCP (spec-train 5.3 step 6): Gumbel-top-k over all n with
    /// weights 1 / (1e-6 + 1 − prune_score); drawn candidates are pruned.
    pub fn vcp(
        &mut self,
        gpu: &Gpu,
        score: &CudaSlice<f32>,
        k: u32,
        rng: Philox,
        t: u32,
    ) -> Result<()> {
        let n = self.plan.n_before;
        if k == 0 || n == 0 {
            return Ok(());
        }
        let s = gpu.stream.clone();
        {
            let [k0, _] = &mut self.compactor.keys;
            let [v0, _] = &mut self.compactor.vals;
            let seed = Uint2::from(rng.seed);
            let mut l = s.launch_builder(&self.k.gumbel_keys);
            l.arg(score)
                .arg(&self.minmax)
                .arg(&seed)
                .arg(&t)
                .arg(&n)
                .arg(k0)
                .arg(v0);
            // SAFETY: gumbel_keys(score_p, minmax, uint2 seed, iteration, n, keys, idx), one per Gaussian.
            unsafe { l.launch(grid(n, 256)) }.cuda("gumbel_keys")?;
        }
        self.compactor.sort(gpu, n, 32)?;
        let mut l = s.launch_builder(&self.k.vcp_mark);
        l.arg(&self.compactor.vals[0])
            .arg(&k)
            .arg(&mut self.plan.out_count);
        // SAFETY: vcp_mark(sorted_idx, k <= n, out_count), one thread per drawn rank.
        unsafe { l.launch(grid(k, 256)) }.cuda("vcp_mark").map(drop)
    }

    /// ô ← min(ô, logit_max) for the n live Gaussians; zero the opacity moments.
    pub fn opacity_clamp(
        &self,
        gpu: &Gpu,
        g: &mut Gaussians,
        adam: &mut Adam,
        logit_max: f32,
    ) -> Result<()> {
        let n = g.n;
        let buf = &mut adam.moments.groups[Group::Opacities.index()];
        let words = (buf.len() / g.cap as usize).max(1) as u32;
        let mut l = gpu.stream.launch_builder(&self.k.opacity_clamp);
        l.arg(&mut g.opacity)
            .arg(buf)
            .arg(&words)
            .arg(&logit_max)
            .arg(&n);
        // SAFETY: opacity_clamp(opacity, uint32_t* moments, words, logit_max, n): moments
        // hold `words` u32 per Gaussian for the opacity group.
        unsafe { l.launch(grid(n, 256)) }
            .cuda("opacity_clamp")
            .map(drop)
    }
}

fn zero_stats(gpu: &Gpu, st: &mut DensifyStats) -> Result<()> {
    let s = &gpu.stream;
    s.memset_zeros(&mut st.vis_count).cuda("zero vis_count")?;
    s.memset_zeros(&mut st.grad_sum).cuda("zero grad_sum")?;
    if let Some(a) = st.grad_abs_sum.as_mut() {
        s.memset_zeros(a).cuda("zero grad_abs_sum")?;
    }
    Ok(())
}

/// Emits `w.plan` over parameters, gradients, moments and statistics.
fn emit_all(cx: &mut DensifyCx<'_>, w: &mut Work, t: u32) -> Result<EmitResult> {
    let s = cx.gpu.stream.clone();
    let cap = cx.g.cap;
    let mut extra = PerGaussianArrays::new();
    extra.add_grads(&s, cx.grads);
    extra.add_moments(&s, &mut cx.adam.moments, cap);
    extra.add_stats(&s, cx.stats);
    w.compactor
        .emit(cx.gpu, &mut w.plan, cx.g, &extra, cx.rng, t)
}

fn outcome(r: &EmitResult) -> DensifyOutcome {
    DensifyOutcome {
        changed: true,
        n_before: r.n_before,
        n_after: r.n_after,
        cloned: r.clones,
        split: r.splits,
        pruned: r.n_before - r.survivors - r.splits,
        relocated: 0,
        capped: r.capped,
    }
}

// ---------------------------------------------------------------- ADC

struct Adc {
    cfg: AdcConfig,
    work: Option<Work>,
}

impl Strategy for Adc {
    fn after_step(&mut self, cx: &mut DensifyCx<'_>, t: u32) -> Result<DensifyOutcome> {
        let c = &self.cfg;
        let densify = window_due(t, c.start, c.end, c.every);
        let reset = t < c.end && t.is_multiple_of(c.reset_every.max(1));
        if !densify && !reset {
            return Ok(DensifyOutcome::default());
        }
        let w = match &mut self.work {
            Some(w) => w,
            None => self.work.insert(Work::new(cx.gpu, cx.g.cap as usize)?),
        };
        let mut out = DensifyOutcome::default();
        if densify {
            let rules = DensifyRules::adc(c, cx.extent, t);
            w.plan(cx.gpu, cx.g, cx.stats, None, &rules)?;
            out = outcome(&emit_all(cx, w, t)?);
            zero_stats(cx.gpu, cx.stats)?;
        }
        if reset {
            w.opacity_clamp(cx.gpu, cx.g, cx.adam, logit(0.01))?;
            out.changed = true;
        }
        Ok(out)
    }

    fn wants_abs_grad(&self) -> bool {
        false
    }

    fn wants_stats(&self, t: u32) -> bool {
        t < self.cfg.end
    }
}

// ---------------------------------------------------------------- FastGS

/// Per-round scoring buffers (target, error map, mask).
struct Scoring {
    scores: ViewScores,
    target: CudaSlice<f32>,
    err: ErrorMap,
    mask: CudaSlice<u8>,
}

impl Scoring {
    fn new(gpu: &Gpu, cap: usize, max_px: usize) -> Result<Self> {
        let s = &gpu.stream;
        Ok(Self {
            scores: ViewScores::alloc(gpu, cap)?,
            target: s.alloc_zeros::<f32>(3 * max_px).cuda("alloc VCD target")?,
            err: ErrorMap {
                err: s.alloc_zeros::<f32>(max_px).cuda("alloc VCD error")?,
                minmax: s.alloc_zeros::<f32>(2).cuda("alloc VCD min/max")?,
                sums: s.alloc_zeros::<f32>(2).cuda("alloc VCD sums")?,
            },
            mask: s.alloc_zeros::<u8>(max_px).cuda("alloc VCD mask")?,
        })
    }
}

struct FastGs {
    cfg: FastGsConfig,
    work: Option<Work>,
    scoring: Option<Scoring>,
}

impl FastGs {
    /// spec-train 5.4: K distinct training views; per view render, error map,
    /// mask, count pass weighted by E_j = 0.8·L1 + 0.2·(1 − SSIM). ErrorMap
    /// contract (E4): `sums = [mean L1, mean SSIM]` of the view.
    fn score(&mut self, cx: &mut DensifyCx<'_>, t: u32) -> Result<()> {
        let gpu = cx.gpu;
        let max_px = cx
            .views
            .iter()
            .map(|p| p.frame.render_w as usize * p.frame.render_h as usize)
            .max()
            .unwrap_or(1);
        let sc = match &mut self.scoring {
            Some(s) if s.mask.len() >= max_px => s,
            slot => slot.insert(Scoring::new(gpu, cx.g.cap as usize, max_px)?),
        };
        let w = self.work.as_ref().expect("work allocated before scoring");
        gpu.stream
            .memset_zeros(&mut sc.scores.count)
            .cuda("zero view counts")?;
        gpu.stream
            .memset_zeros(&mut sc.scores.score)
            .cuda("zero view scores")?;
        let train = &cx.scene.train;
        for j in sample_views(&cx.rng, train.len(), self.cfg.views as usize, t) {
            let Some(p) = cx.views.get(train[j]) else {
                continue;
            };
            let px = p.frame.render_w * p.frame.render_h;
            cx.raster.forward(
                gpu,
                cx.g,
                &cx.scene.cams,
                FrameSource::Eager(p),
                RenderMode::Infer,
            )?;
            resample_target(
                gpu,
                &cx.loss.kernels,
                cx.raster.params(),
                &cx.scene.images,
                &mut sc.target,
            )?;
            cx.loss.error_map(
                gpu,
                cx.raster.params(),
                cx.raster.image(),
                &sc.target,
                &mut sc.err,
            )?;
            let sums = gpu
                .stream
                .clone_dtoh(&sc.err.sums.slice(0..2))
                .cuda("read view loss")?;
            let e = 0.8 * sums[0] + 0.2 * (1.0 - sums[1]);
            vcd_mask(gpu, &w.k, &sc.err, self.cfg.loss_thresh, px, &mut sc.mask)?;
            cx.raster.count_masked(
                gpu,
                cx.g,
                &cx.scene.cams,
                FrameSource::Eager(p),
                &sc.mask,
                e,
                &mut sc.scores,
            )?;
        }
        Ok(())
    }
}

impl Strategy for FastGs {
    fn after_step(&mut self, cx: &mut DensifyCx<'_>, t: u32) -> Result<DensifyOutcome> {
        let c = self.cfg.clone();
        let densify = window_due(t, c.start, c.end, c.every);
        let final_prune = c.final_prune_due(t);
        if !densify && !final_prune {
            return Ok(DensifyOutcome::default());
        }
        if self.work.is_none() {
            self.work = Some(Work::new(cx.gpu, cx.g.cap as usize)?);
        }
        self.score(cx, t)?;
        let w = self.work.as_mut().expect("allocated above");
        let sc = self.scoring.as_ref().expect("allocated by score");
        w.score_minmax(cx.gpu, &sc.scores.score, cx.g.n)?;
        let rules = if densify {
            DensifyRules::fastgs(&c, cx.extent, t, cx.res_scale)
        } else {
            DensifyRules::fastgs_final(&c)
        };
        let candidates = w.plan(cx.gpu, cx.g, cx.stats, Some(&sc.scores), &rules)?;
        if densify {
            w.vcp(cx.gpu, &sc.scores.score, candidates / 2, cx.rng, t)?;
        }
        let out = outcome(&emit_all(cx, w, t)?);
        if densify {
            w.opacity_clamp(cx.gpu, cx.g, cx.adam, logit(c.clamp_opacity))?;
            zero_stats(cx.gpu, cx.stats)?;
            if t.is_multiple_of(c.reset_every.max(1)) {
                w.opacity_clamp(cx.gpu, cx.g, cx.adam, logit(0.01))?;
            }
        }
        Ok(out)
    }

    fn wants_abs_grad(&self) -> bool {
        true
    }

    fn wants_stats(&self, t: u32) -> bool {
        t < self.cfg.end
    }

    fn wants_views(&self, t: u32) -> bool {
        window_due(t, self.cfg.start, self.cfg.end, self.cfg.every) || self.cfg.final_prune_due(t)
    }
}

// ---------------------------------------------------------------- MCMC

/// Device scratch of the relocation and growth rounds.
pub struct McmcWork {
    pub k: McmcKernels,
    pub compact: crate::model::CompactKernels,
    pub w: CudaSlice<f32>,
    pub cdf: CudaSlice<f32>,
    pub dead: CudaSlice<u32>,
    pub dead_list: CudaSlice<u32>,
    pub src: CudaSlice<u32>,
    pub draws: CudaSlice<u32>,
    pub fpart: CudaSlice<f32>,
    pub upart: CudaSlice<u32>,
    pub total: CudaSlice<u32>,
    pub binoms: CudaSlice<f32>,
    pub cols: CudaSlice<Column>,
    pub n_max: u32,
}

/// N cap of the relocation sums (spec-train 5.6).
pub const MCMC_N_MAX: u32 = 50;

impl McmcWork {
    pub fn new(gpu: &Gpu, cap: usize) -> Result<Self> {
        let cap = round_cap(cap.max(1));
        let s = &gpu.stream;
        let tiles = cap.div_ceil(TILE as usize);
        let zf = |len: usize, what: &str| s.alloc_zeros::<f32>(len).cuda(what);
        let zu = |len: usize, what: &str| s.alloc_zeros::<u32>(len).cuda(what);
        Ok(Self {
            k: McmcKernels::load(gpu)?,
            compact: crate::model::CompactKernels::load(gpu)?,
            w: zf(cap, "alloc MCMC weights")?,
            cdf: zf(cap, "alloc MCMC cdf")?,
            dead: zu(cap, "alloc MCMC dead flags")?,
            dead_list: zu(cap, "alloc MCMC dead list")?,
            src: zu(cap, "alloc MCMC sources")?,
            draws: zu(cap, "alloc MCMC draws")?,
            fpart: zf(tiles, "alloc MCMC partials")?,
            upart: zu(tiles, "alloc MCMC partials")?,
            total: zu(1, "alloc MCMC total")?,
            binoms: s
                .clone_htod(&binomial_table(MCMC_N_MAX as usize))
                .cuda("upload binomials")?,
            cols: s
                .alloc_zeros::<Column>(crate::model::MAX_COLUMNS)
                .cuda("alloc MCMC columns")?,
            n_max: MCMC_N_MAX,
        })
    }

    /// cdf of the weights over `[0, n)` (weights from `mcmc_weights`).
    fn weights_and_cdf(
        &mut self,
        gpu: &Gpu,
        g: &Gaussians,
        alive_only: bool,
        min_opacity: f32,
    ) -> Result<()> {
        let s = &gpu.stream;
        let n = g.n;
        let tiles = n.div_ceil(TILE);
        let dl = logit(min_opacity);
        let ao = u32::from(alive_only);
        let mut l = s.launch_builder(&self.k.weights);
        l.arg(&g.opacity)
            .arg(&g.quat)
            .arg(&n)
            .arg(&dl)
            .arg(&ao)
            .arg(&mut self.w)
            .arg(&mut self.dead);
        // SAFETY: mcmc_weights(opacity, float4* quat, n, dead_logit, alive_only, w, dead), per Gaussian.
        unsafe { l.launch(grid(n, 256)) }.cuda("mcmc_weights")?;
        let mut l = s.launch_builder(&self.k.fscan_reduce);
        l.arg(&self.w).arg(&n).arg(&mut self.fpart);
        // SAFETY: fscan_reduce(w, n, part[tiles]), one block of 256 per 2048-item tile.
        unsafe { l.launch(tile_cfg(tiles)) }.cuda("fscan_reduce")?;
        let mut l = s.launch_builder(&self.k.fscan_partials);
        l.arg(&mut self.fpart).arg(&tiles);
        // SAFETY: fscan_partials(part, tiles), one block of 1024.
        unsafe { l.launch(one_block()) }.cuda("fscan_partials")?;
        let mut l = s.launch_builder(&self.k.fscan_apply);
        l.arg(&self.w).arg(&n).arg(&self.fpart).arg(&mut self.cdf);
        // SAFETY: fscan_apply(w, n, part, cdf), same tiling as fscan_reduce.
        unsafe { l.launch(tile_cfg(tiles)) }
            .cuda("fscan_apply")
            .map(drop)
    }

    /// The ordered dead list; returns its length (host sync).
    fn dead_list(&mut self, gpu: &Gpu, n: u32) -> Result<u32> {
        let s = &gpu.stream;
        let tiles = n.div_ceil(TILE);
        let mut l = s.launch_builder(&self.k.flag_count);
        l.arg(&self.dead).arg(&n).arg(&mut self.upart);
        // SAFETY: flag_count(flags, n, part[tiles]), one block per tile.
        unsafe { l.launch(tile_cfg(tiles)) }.cuda("flag_count")?;
        let mut l = s.launch_builder(&self.compact.scan_u32);
        l.arg(&mut self.upart).arg(&tiles).arg(&mut self.total);
        // SAFETY: scan_u32(data, count, total), one block of 1024.
        unsafe { l.launch(one_block()) }.cuda("scan_u32")?;
        let mut l = s.launch_builder(&self.k.flag_list);
        l.arg(&self.dead)
            .arg(&n)
            .arg(&self.upart)
            .arg(&mut self.dead_list);
        // SAFETY: flag_list(flags, n, scanned part, list), same tiling; list holds <= n entries.
        unsafe { l.launch(tile_cfg(tiles)) }.cuda("flag_list")?;
        Ok(s.clone_dtoh(&self.total).cuda("read dead count")?[0])
    }

    /// m draws ∝ cdf over [0, n), the histogram, and the (o', s') adjustment.
    #[allow(clippy::too_many_arguments)]
    fn draw_and_adjust(
        &mut self,
        gpu: &Gpu,
        g: &mut Gaussians,
        m: u32,
        lane: u32,
        rng: Philox,
        t: u32,
        min_o: f32,
    ) -> Result<()> {
        let s = &gpu.stream;
        let n = g.n;
        s.memset_zeros(&mut self.draws).cuda("zero draws")?;
        let seed = Uint2::from(rng.seed);
        let mut l = s.launch_builder(&self.k.draw);
        l.arg(&self.cdf)
            .arg(&n)
            .arg(&m)
            .arg(&seed)
            .arg(&t)
            .arg(&lane)
            .arg(&mut self.src)
            .arg(&mut self.draws);
        // SAFETY: mcmc_draw(cdf, n >= 1, m <= cap, uint2 seed, iteration, lane, src, draws).
        unsafe { l.launch(grid(m, 256)) }.cuda("mcmc_draw")?;
        let mut l = s.launch_builder(&self.k.adjust);
        l.arg(&self.draws)
            .arg(&self.binoms)
            .arg(&self.n_max)
            .arg(&n)
            .arg(&min_o)
            .arg(&mut g.opacity)
            .arg(&mut g.log_scale);
        // SAFETY: mcmc_adjust(draws, binoms[n_max^2], n_max, n, min_opacity, opacity, log_scale).
        unsafe { l.launch(grid(n, 256)) }
            .cuda("mcmc_adjust")
            .map(drop)
    }

    /// Row copies src[j] → dst (list or base + j) over every column.
    #[allow(clippy::too_many_arguments)]
    fn copy(
        &mut self,
        gpu: &Gpu,
        cols: &[Column],
        m: u32,
        dst_list: bool,
        dst_base: u32,
        zero_dst: bool,
    ) -> Result<()> {
        let s = &gpu.stream;
        s.memcpy_htod(cols, &mut self.cols.slice_mut(..cols.len()))
            .cuda("upload MCMC columns")?;
        let maxw = cols.iter().map(|c| c.words).max().unwrap_or(1);
        let gx = (m * maxw).div_ceil(256).clamp(1, 256);
        let null = 0u64;
        let (zs, zd) = (1u32, u32::from(zero_dst));
        let table = self.cols.slice(..cols.len());
        let mut l = s.launch_builder(&self.k.copy_rows);
        l.arg(&table).arg(&self.src);
        if dst_list {
            l.arg(&self.dead_list);
        } else {
            l.arg(&null);
        }
        l.arg(&dst_base).arg(&m).arg(&zs).arg(&zd);
        // SAFETY: copy_rows(cols, src_list, dst_list | null, dst_base, m, zero_src, zero_dst): one
        // column per blockIdx.y; sources and destinations are disjoint (alive vs dead, or
        // [0, n) vs [n, n + m)) and below cap.
        unsafe {
            l.launch(LaunchConfig {
                grid_dim: (gx, cols.len() as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .cuda("copy_rows")
        .map(drop)
    }
}

fn tile_cfg(tiles: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (tiles.max(1), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn one_block() -> LaunchConfig {
    LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1024, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// One MCMC round (spec-train 5.6): relocate the dead, then grow toward
/// `min(cap_max, floor(grow · n))`. Returns (relocated, added).
pub fn mcmc_round(
    gpu: &Gpu,
    mw: &mut McmcWork,
    cfg: &McmcConfig,
    g: &mut Gaussians,
    extra: &PerGaussianArrays<'_>,
    rng: Philox,
    t: u32,
) -> Result<(u32, u32)> {
    let s = gpu.stream.clone();
    let n = g.n;
    if n == 0 {
        return Ok((0, 0));
    }
    let mut cols = crate::model::gaussian_columns(&s, g);
    cols.extend_from_slice(&extra.cols);
    // 1. Relocation.
    mw.weights_and_cdf(gpu, g, true, cfg.min_opacity)?;
    let n_dead = mw.dead_list(gpu, n)?;
    let mut relocated = 0;
    if n_dead > 0 && n_dead < n {
        mw.draw_and_adjust(gpu, g, n_dead, 0, rng, t, cfg.min_opacity)?;
        // The reference resets only the sources' moments (reset_dead_moments = false).
        mw.copy(gpu, &cols, n_dead, true, 0, cfg.reset_dead_moments)?;
        relocated = n_dead;
    }
    // 2. Growth.
    // The references compute int(1.05 · n) in f64 from the decimal; the config is f32.
    let grow = (cfg.grow as f64 * 1e6).round() / 1e6;
    let target = (cfg.cap_max.min(g.cap) as f64).min((grow * n as f64).floor()) as u32;
    let add = target.saturating_sub(n);
    if add > 0 {
        mw.weights_and_cdf(gpu, g, false, cfg.min_opacity)?;
        mw.draw_and_adjust(gpu, g, add, 1, rng, t, cfg.min_opacity)?;
        mw.copy(gpu, &cols, add, false, n, true)?;
        g.n = n + add;
    }
    Ok((relocated, add))
}

struct Mcmc {
    cfg: McmcConfig,
    work: Option<McmcWork>,
}

impl Strategy for Mcmc {
    fn after_step(&mut self, cx: &mut DensifyCx<'_>, t: u32) -> Result<DensifyOutcome> {
        let c = &self.cfg;
        if !window_due(t, c.start, c.end, c.every) {
            return Ok(DensifyOutcome::default());
        }
        let mw = match &mut self.work {
            Some(w) => w,
            None => self.work.insert(McmcWork::new(cx.gpu, cx.g.cap as usize)?),
        };
        let s = cx.gpu.stream.clone();
        let cap = cx.g.cap;
        let n_before = cx.g.n;
        let mut extra = PerGaussianArrays::new();
        extra.add_grads(&s, cx.grads);
        extra.add_moments(&s, &mut cx.adam.moments, cap);
        let (relocated, added) = mcmc_round(cx.gpu, mw, c, cx.g, &extra, cx.rng, t)?;
        Ok(DensifyOutcome {
            changed: true,
            n_before,
            n_after: cx.g.n,
            cloned: added,
            relocated,
            ..Default::default()
        })
    }

    fn wants_abs_grad(&self) -> bool {
        false
    }

    fn wants_stats(&self, _t: u32) -> bool {
        false
    }
}

pub fn adc(cfg: AdcConfig) -> Box<dyn Strategy> {
    Box::new(Adc { cfg, work: None })
}

pub fn fastgs(cfg: FastGsConfig) -> Box<dyn Strategy> {
    Box::new(FastGs {
        cfg,
        work: None,
        scoring: None,
    })
}

pub fn mcmc(cfg: McmcConfig) -> Box<dyn Strategy> {
    Box::new(Mcmc { cfg, work: None })
}

#[cfg(test)]
mod gpu_tests;

#[cfg(test)]
mod tests {
    use super::*;

    fn count(c: impl Fn(u32) -> bool) -> usize {
        (1..=30_000).filter(|&t| c(t)).count()
    }

    /// The densify step sets of spec-train 5.2, 5.3 and 6.1.
    #[test]
    fn schedules_match_the_reference_sets() {
        let a = AdcConfig::default();
        assert_eq!(count(|t| window_due(t, a.start, a.end, a.every)), 144);
        assert!(
            window_due(600, a.start, a.end, a.every) && window_due(14_900, a.start, a.end, a.every)
        );
        let f = FastGsConfig::default();
        assert_eq!(count(|t| window_due(t, f.start, f.end, f.every)), 28);
        assert!(
            window_due(1000, f.start, f.end, f.every)
                && window_due(14_500, f.start, f.end, f.every)
        );
        let b = FastGsConfig::big();
        assert_eq!(count(|t| window_due(t, b.start, b.end, b.every)), 144);
        assert_eq!(count(|t| f.final_prune_due(t)), 4);
        assert!(
            f.final_prune_due(18_000) && f.final_prune_due(27_000) && !f.final_prune_due(30_000)
        );
        let m = McmcConfig::default();
        assert_eq!(count(|t| window_due(t, m.start, m.end, m.every)), 244);
        let resets = (1..=30_000)
            .filter(|t| *t < a.end && t % a.reset_every == 0)
            .count();
        assert_eq!(resets, 4);
    }

    #[test]
    fn mcmc_caps_cover_mip360() {
        assert_eq!(MCMC_CAP_MAX_MIP360.len(), 9);
        assert!(
            MCMC_CAP_MAX_MIP360
                .iter()
                .any(|&(s, c)| s == "garden" && c == 5_834_784)
        );
    }
}
