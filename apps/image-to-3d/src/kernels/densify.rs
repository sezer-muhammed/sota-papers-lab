//! Densification strategies (ADC, FastGS VCD/VCP, MCMC), their statistics and
//! view scores, opacity clamp/reset. Owner: densify (E6). Kernels:
//! `cuda/densify.cu`, `cuda/compact.cu`, `cuda/mcmc.cu`; compaction and Morton
//! sorting live in `model.rs` (same owner). Spec: spec-train 5.
//!
//! Buffer contract: preprocess_bwd accumulates [`DensifyStats`] at p when the
//! strategy `wants_stats(t)`; `SplatRaster::count_masked` fills [`ViewScores`];
//! `after_step` runs after the optimizer step of iteration t (never before),
//! may change n, and zeroes the statistics it consumed.
//!
//! Implemented: configs with the spec's defaults, the schedule predicates,
//! statistics allocation. STUBS: the device work in `after_step`.

use serde::{Deserialize, Serialize};
use spl_core::Result;
use spl_gpu::cudarc::driver::CudaSlice;

use super::adam::Adam;
use super::loss::Loss;
use super::raster::SplatRaster;
use crate::gpu::{
    COMMON_CUH, CudaFile, CudaResultExt, Gpu, KernelSource, SPLAT_COMMON_CUH, not_implemented,
};
use crate::model::{Gaussians, Grads, round_cap};
use crate::rng::Philox;
use crate::trainer::SceneGpu;

const OWNER: &str = "densify (E6)";

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
        "densify_plan",
        "gumbel_keys",
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
        "emit_scan",
        "slab_load",
        "slab_scatter",
        "morton_codes",
        "gather_words",
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
    kernels: &["mcmc_relocate", "mcmc_add", "mcmc_noise"],
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

struct Adc(AdcConfig);
struct FastGs(FastGsConfig);
struct Mcmc(McmcConfig);

impl Strategy for Adc {
    fn after_step(&mut self, _cx: &mut DensifyCx<'_>, t: u32) -> Result<DensifyOutcome> {
        let c = &self.0;
        if !window_due(t, c.start, c.end, c.every)
            && !(t < c.end && t.is_multiple_of(c.reset_every.max(1)))
        {
            return Ok(DensifyOutcome::default());
        }
        Err(not_implemented("Adc::after_step", OWNER))
    }

    fn wants_abs_grad(&self) -> bool {
        false
    }

    fn wants_stats(&self, t: u32) -> bool {
        t < self.0.end
    }
}

impl Strategy for FastGs {
    fn after_step(&mut self, _cx: &mut DensifyCx<'_>, t: u32) -> Result<DensifyOutcome> {
        let c = &self.0;
        if !window_due(t, c.start, c.end, c.every) && !c.final_prune_due(t) {
            return Ok(DensifyOutcome::default());
        }
        Err(not_implemented("FastGs::after_step", OWNER))
    }

    fn wants_abs_grad(&self) -> bool {
        true
    }

    fn wants_stats(&self, t: u32) -> bool {
        t < self.0.end
    }
}

impl Strategy for Mcmc {
    fn after_step(&mut self, _cx: &mut DensifyCx<'_>, t: u32) -> Result<DensifyOutcome> {
        let c = &self.0;
        if !window_due(t, c.start, c.end, c.every) {
            return Ok(DensifyOutcome::default());
        }
        Err(not_implemented("Mcmc::after_step", OWNER))
    }

    fn wants_abs_grad(&self) -> bool {
        false
    }

    fn wants_stats(&self, _t: u32) -> bool {
        false
    }
}

pub fn adc(cfg: AdcConfig) -> Box<dyn Strategy> {
    Box::new(Adc(cfg))
}

pub fn fastgs(cfg: FastGsConfig) -> Box<dyn Strategy> {
    Box::new(FastGs(cfg))
}

pub fn mcmc(cfg: McmcConfig) -> Box<dyn Strategy> {
    Box::new(Mcmc(cfg))
}

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
