//! Training presets and schedules. Owner: trainer (E7). Spec: spec-train 4.2,
//! 6 (presets, per-scene FastGS overrides, scaling rule, SH ramp, DashGaussian).
//!
//! A [`Preset`] holds every setting of a run at T = 30k; [`Preset::scaled`]
//! applies the T ≠ 30k rule; [`Schedule`] answers per-iteration questions.
//! Implemented (pure CPU, tested); `DashSchedule::from_spectra` is a stub.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use spl_core::Result;
use ts_rs::TS;

use crate::gpu::not_implemented;
use crate::kernels::RasterVariant;
use crate::kernels::adam::{
    AdamConfig, GroupConfig, LrSchedule, MomentPrecision, Sparsity, StepPlan,
};
use crate::kernels::densify::{AdcConfig, FastGsConfig, McmcConfig};
use crate::kernels::loss::LossConfig;

/// The presets of spec-train 6.2.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema, TS,
)]
#[serde(rename_all = "snake_case")]
#[ts(export, export_to = "image-to-3d/")]
pub enum PresetName {
    /// Apples-to-apples kernel speed vs Faster-GS (ADC, dense Adam every iteration).
    FasterGs,
    /// FastGS parity (VCD/VCP base, FastGS step plan).
    Fastgs,
    /// FastGS-Big parity.
    FastgsBig,
    /// 3DGS-MCMC: quality ceiling and dense-init fallback.
    Mcmc,
    /// Ours: FastGS-Big with generic thresholds; target ≥ 27.9 dB in ≤ 6 min.
    #[default]
    Quality,
    /// Ours: COLMAP-free fast target, fewer iterations, resolution warm-up.
    Fast,
    /// Quality plus the geometry regularisers (`mesh::geom`): for meshing.
    Geometry,
}

/// Any preset field, for ablations. `None` keeps the preset's value.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct PresetOverrides {
    /// FastGS per-scene table key (bicycle, garden, truck, drjohnson, ...).
    #[serde(default)]
    #[ts(optional)]
    pub scene: Option<String>,
    #[serde(default)]
    #[ts(optional)]
    pub grad_abs_thresh: Option<f32>,
    #[serde(default)]
    #[ts(optional)]
    pub dense: Option<f32>,
    #[serde(default)]
    #[ts(optional)]
    pub loss_thresh: Option<f32>,
    /// SH-rest LR = highfeature_lr / 20.
    #[serde(default)]
    #[ts(optional)]
    pub highfeature_lr: Option<f32>,
    #[serde(default)]
    #[ts(optional)]
    pub compact_box_beta: Option<f32>,
    #[serde(default)]
    #[ts(optional)]
    pub sparsity: Option<Sparsity>,
    #[serde(default)]
    #[ts(optional)]
    pub moments: Option<MomentPrecision>,
    #[serde(default)]
    #[ts(optional)]
    pub warmup: Option<bool>,
    #[serde(default)]
    #[ts(optional)]
    pub mcmc_cap_max: Option<u32>,
    #[serde(default)]
    #[ts(optional)]
    pub tile_h: Option<u8>,
    #[serde(default)]
    #[ts(optional)]
    pub chain_k: Option<u8>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum DensifyPreset {
    Adc(AdcConfig),
    FastGs(FastGsConfig),
    Mcmc(McmcConfig),
}

/// Every setting of a run.
#[derive(Clone, Debug, PartialEq)]
pub struct Preset {
    pub name: PresetName,
    pub iterations: u32,
    pub densify: DensifyPreset,
    /// In `Group::ALL` order.
    pub groups: [GroupConfig; 6],
    pub adam: AdamConfig,
    pub loss: LossConfig,
    pub variant: RasterVariant,
    /// SH degree +1 every this many iterations (max 3).
    pub sh_interval: u32,
    /// Morton re-sort right after these iterations' hooks.
    pub morton_at: Vec<u32>,
    /// DashGaussian resolution warm-up.
    pub warmup: bool,
    /// FastGS compact box β (1 = off).
    pub compact_box_beta: f32,
    /// Opacity-compensated 2D filter (antialiased rasterization).
    pub antialiased: bool,
    /// Post-training prune of o < this.
    pub final_prune_opacity: f32,
}

/// Default LR of every group (3DGS 2024 code = Faster-GS), T_lr = 30k.
fn groups(opacity_lr: f32, sh_rest_lr: f32, main: StepPlan, sh_rest: StepPlan) -> [GroupConfig; 6] {
    let c = |lr: f32, plan: &StepPlan| GroupConfig {
        lr: LrSchedule::Constant(lr),
        plan: plan.clone(),
    };
    [
        GroupConfig {
            lr: LrSchedule::ExpDecay {
                init: 1.6e-4,
                final_: 1.6e-6,
                max_steps: 30_000,
                hold_until: 0,
                scale_by_extent: true,
            },
            plan: main.clone(),
        },
        c(5e-3, &main),
        c(1e-3, &main),
        c(opacity_lr, &main),
        c(2.5e-3, &main),
        c(sh_rest_lr, &sh_rest),
    ]
}

fn fastgs_plans() -> (StepPlan, StepPlan) {
    (
        StepPlan(vec![(0, 1), (15_001, 32), (20_001, 64)]),
        StepPlan(vec![(0, 16), (15_001, 32), (20_001, 64)]),
    )
}

/// FastGS `highfeature_lr` code default.
pub const FASTGS_HIGHFEATURE_LR: f32 = 0.005;

impl Preset {
    /// The preset at T = 30k (spec-train 6.1-6.2).
    pub fn new(name: PresetName) -> Self {
        let every = StepPlan::every_iteration;
        let (main, sh) = fastgs_plans();
        let base = Self {
            name,
            iterations: 30_000,
            densify: DensifyPreset::Adc(AdcConfig::default()),
            groups: groups(0.025, 1.25e-4, every(), every()),
            adam: AdamConfig::default(),
            loss: LossConfig::default(),
            variant: RasterVariant::default(),
            sh_interval: 1000,
            morton_at: vec![5000, 10_000, 15_000],
            warmup: false,
            compact_box_beta: 1.0,
            antialiased: false,
            final_prune_opacity: 1.0 / 255.0,
        };
        match name {
            PresetName::FasterGs => base,
            PresetName::Fastgs => Self {
                densify: DensifyPreset::FastGs(FastGsConfig::default()),
                groups: groups(0.025, FASTGS_HIGHFEATURE_LR / 20.0, main, sh),
                compact_box_beta: 0.5,
                ..base
            },
            PresetName::FastgsBig => Self {
                densify: DensifyPreset::FastGs(FastGsConfig::big()),
                groups: groups(0.025, FASTGS_HIGHFEATURE_LR / 20.0, main, sh),
                compact_box_beta: 0.5,
                ..base
            },
            PresetName::Mcmc => Self {
                densify: DensifyPreset::Mcmc(McmcConfig::default()),
                groups: groups(0.05, 1.25e-4, every(), every()),
                morton_at: vec![5000, 10_000, 15_000, 20_000, 25_000],
                ..base
            },
            // Generic thresholds: FastGS-Big medians as the starting point (6.2).
            PresetName::Quality | PresetName::Geometry => Self {
                densify: DensifyPreset::FastGs(FastGsConfig {
                    grad_abs_thresh: 4e-4,
                    dense: 0.001,
                    loss_thresh: 0.1,
                    ..FastGsConfig::big()
                }),
                groups: groups(0.025, 0.02 / 20.0, main, sh),
                compact_box_beta: 0.5,
                ..base
            },
            // T ablated over {10k, 15k, 20k}; 15k until measured.
            PresetName::Fast => Self {
                densify: DensifyPreset::FastGs(FastGsConfig::default()),
                groups: groups(0.025, FASTGS_HIGHFEATURE_LR / 20.0, main, sh),
                compact_box_beta: 0.5,
                warmup: true,
                ..base
            }
            .scaled(15_000),
        }
    }

    /// The preset rescaled to `total` iterations: every iteration constant
    /// × total/30000, rounded to the nearest multiple of 10; counts (K, τ_d,
    /// every-16/32/64 step cadences) stay (spec-train 6.2).
    pub fn scaled(&self, total: u32) -> Self {
        // Relative to the preset's own length: Fast is already at 15k.
        let from = self.iterations;
        let s = |x: u32| scale_between(x, from, total);
        let mut p = self.clone();
        p.iterations = total;
        p.sh_interval = s(p.sh_interval).max(10);
        p.morton_at = p.morton_at.iter().map(|&x| s(x)).collect();
        p.densify = match &p.densify {
            DensifyPreset::Adc(c) => DensifyPreset::Adc(AdcConfig {
                start: s(c.start),
                end: s(c.end),
                every: s(c.every).max(10),
                reset_every: s(c.reset_every).max(10),
                big_after: s(c.big_after),
                ..c.clone()
            }),
            DensifyPreset::FastGs(c) => DensifyPreset::FastGs(FastGsConfig {
                start: s(c.start),
                end: s(c.end),
                every: s(c.every).max(10),
                reset_every: s(c.reset_every).max(10),
                final_prune_from: s(c.final_prune_from),
                final_prune_every: s(c.final_prune_every).max(10),
                final_prune_until: s(c.final_prune_until),
                ..c.clone()
            }),
            DensifyPreset::Mcmc(c) => DensifyPreset::Mcmc(McmcConfig {
                start: s(c.start),
                end: s(c.end),
                every: s(c.every).max(10),
                ..c.clone()
            }),
        };
        for g in &mut p.groups {
            if let LrSchedule::ExpDecay {
                max_steps,
                hold_until,
                ..
            } = &mut g.lr
            {
                *max_steps = s(*max_steps);
                *hold_until = s(*hold_until);
            }
            for (from, _) in &mut g.plan.0 {
                if *from > 0 {
                    // Boundaries like 15001 scale from 15000 and keep the "+1".
                    *from = s(*from - 1) + 1;
                }
            }
        }
        p
    }

    /// Applies `o` (and the FastGS per-scene table when `o.scene` names a
    /// known scene and the preset is FastGS parity).
    pub fn with_overrides(mut self, o: &PresetOverrides) -> Self {
        let big = self.name == PresetName::FastgsBig;
        let table = match (self.name, &o.scene) {
            (PresetName::Fastgs | PresetName::FastgsBig, Some(scene)) => {
                fastgs_scene_override(scene, big)
            }
            _ => None,
        };
        let mut hf = None;
        if let DensifyPreset::FastGs(c) = &mut self.densify {
            if let Some(t) = &table {
                c.grad_abs_thresh = t.grad_abs_thresh;
                c.dense = t.dense;
                c.loss_thresh = t.loss_thresh;
                hf = Some(t.highfeature_lr);
                self.groups[4].lr = LrSchedule::Constant(t.lowfeature_lr);
                self.compact_box_beta = t.mult;
            }
            if let Some(v) = o.grad_abs_thresh {
                c.grad_abs_thresh = v;
            }
            if let Some(v) = o.dense {
                c.dense = v;
            }
            if let Some(v) = o.loss_thresh {
                c.loss_thresh = v;
            }
        }
        if let Some(v) = o.highfeature_lr.or(hf) {
            self.groups[5].lr = LrSchedule::Constant(v / 20.0);
        }
        if let (DensifyPreset::Mcmc(c), Some(cap)) = (&mut self.densify, o.mcmc_cap_max) {
            c.cap_max = cap;
        }
        if let Some(v) = o.compact_box_beta {
            self.compact_box_beta = v;
        }
        if let Some(v) = o.sparsity {
            self.adam.sparsity = v;
        }
        if let Some(v) = o.moments {
            self.adam.moments = v;
        }
        if let Some(v) = o.warmup {
            self.warmup = v;
        }
        if let Some(v) = o.tile_h {
            self.variant.tile_h = v;
        }
        if let Some(v) = o.chain_k {
            self.variant.chain_k = v;
        }
        self
    }
}

/// `x · total / 30000`, rounded to the nearest multiple of 10.
pub fn scale_iter(x: u32, total: u32) -> u32 {
    scale_between(x, 30_000, total)
}

/// `x · to / from`, rounded to the nearest multiple of 10 (`x` when equal).
pub fn scale_between(x: u32, from: u32, to: u32) -> u32 {
    if from == to || from == 0 {
        return x;
    }
    let v = x as f64 * to as f64 / from as f64;
    ((v / 10.0).round() * 10.0) as u32
}

/// FastGS per-scene hyperparameters (`train_base.sh` / `train_big.sh`, spec-train 6.3).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FastGsSceneOverride {
    pub highfeature_lr: f32,
    pub lowfeature_lr: f32,
    pub dense: f32,
    pub grad_abs_thresh: f32,
    pub loss_thresh: f32,
    pub mult: f32,
}

pub fn fastgs_scene_override(scene: &str, big: bool) -> Option<FastGsSceneOverride> {
    let d = FastGsSceneOverride {
        highfeature_lr: FASTGS_HIGHFEATURE_LR,
        lowfeature_lr: 0.0025,
        dense: 0.001,
        grad_abs_thresh: 0.0012,
        loss_thresh: 0.1,
        mult: 0.5,
    };
    let pick = |base: f32, big_value: f32| if big { big_value } else { base };
    Some(match scene {
        "bicycle" => FastGsSceneOverride {
            grad_abs_thresh: pick(0.0012, 0.0008),
            ..d
        },
        "flowers" => FastGsSceneOverride {
            dense: 0.005,
            grad_abs_thresh: pick(0.0015, 0.001),
            ..d
        },
        "garden" => FastGsSceneOverride {
            highfeature_lr: 0.02,
            loss_thresh: 0.06,
            grad_abs_thresh: pick(0.0008, 0.0003),
            ..d
        },
        "stump" => FastGsSceneOverride {
            dense: 0.004,
            grad_abs_thresh: pick(0.0015, 0.001),
            ..d
        },
        "treehill" => FastGsSceneOverride {
            dense: 0.01,
            grad_abs_thresh: pick(0.002, 0.0018),
            ..d
        },
        "room" | "counter" => FastGsSceneOverride {
            highfeature_lr: 0.02,
            grad_abs_thresh: pick(0.0008, 0.0004),
            ..d
        },
        "kitchen" | "bonsai" => FastGsSceneOverride {
            highfeature_lr: 0.02,
            grad_abs_thresh: pick(0.0006, 0.0002),
            ..d
        },
        "truck" => FastGsSceneOverride {
            highfeature_lr: 0.04,
            grad_abs_thresh: pick(0.0009, 0.0004),
            mult: 0.7,
            ..d
        },
        "train" => FastGsSceneOverride {
            highfeature_lr: 0.042,
            grad_abs_thresh: pick(0.0015, 0.0004),
            dense: pick(0.01, 0.015),
            mult: 0.7,
            ..d
        },
        "playroom" => FastGsSceneOverride {
            highfeature_lr: 0.0015,
            dense: 0.003,
            grad_abs_thresh: pick(0.0012, 0.0005),
            mult: 0.7,
            ..d
        },
        "drjohnson" => FastGsSceneOverride {
            highfeature_lr: 0.0025,
            lowfeature_lr: pick(0.0025, 0.0005),
            grad_abs_thresh: pick(0.0012, 0.0005),
            dense: pick(0.013, 0.005),
            mult: 0.7,
            ..d
        },
        _ => return None,
    })
}

/// Per-iteration answers for a resolved preset (all iterations already scaled).
#[derive(Clone, Debug, PartialEq)]
pub struct Schedule {
    pub total: u32,
    pub sh_interval: u32,
    pub morton_at: Vec<u32>,
    pub means_hold_until: u32,
    pub dash: Option<DashSchedule>,
}

impl Schedule {
    pub fn from_preset(p: &Preset, total: u32) -> Self {
        let p = if p.iterations == total {
            p.clone()
        } else {
            p.scaled(total)
        };
        Self {
            total,
            sh_interval: p.sh_interval,
            morton_at: p.morton_at,
            means_hold_until: 0,
            dash: None,
        }
    }

    /// Degree 0 until `sh_interval`, +1 per interval, max 3 (6.4).
    pub fn sh_degree(&self, t: u32) -> u32 {
        (t / self.sh_interval.max(1)).min(3)
    }

    /// Integer render downscale (1 when the warm-up is off).
    pub fn res_scale(&self, t: u32) -> u32 {
        self.dash.as_ref().map_or(1, |d| d.scale_at(t))
    }

    pub fn means_hold_until(&self) -> u32 {
        self.means_hold_until
    }

    pub fn morton_due(&self, t: u32) -> bool {
        self.morton_at.contains(&t)
    }
}

/// Per-image magnitude spectrum `|fftshift(FFT2(I))|` (one channel mean).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MagnitudeSpectrum {
    pub width: u32,
    pub height: u32,
    pub data: Vec<f32>,
}

/// DashGaussian frequency-based resolution schedule (spec-train 6.5).
#[derive(Clone, Debug, PartialEq)]
pub struct DashSchedule {
    /// (begin iteration b_i, scale s_i), b_0 = 0, s_0 = s_max, s_last = 1.
    pub levels: Vec<(u32, f32)>,
    /// Full resolution from here on (the densify end).
    pub t_res: u32,
}

impl DashSchedule {
    /// Steps 1-3 of 6.5 (energy-matched levels from the spectra). STUB (E7).
    pub fn from_spectra(_spectra: &[MagnitudeSpectrum], _t_res: u32) -> Result<Self> {
        Err(not_implemented(
            "DashSchedule::from_spectra",
            "trainer (E7)",
        ))
    }

    /// Step 4: within [b_i, b_{i+1}) interpolate in pixel count from s_{i−1}
    /// to s_i, render at the integer floor, 1 from `t_res` on.
    pub fn scale_at(&self, t: u32) -> u32 {
        if t >= self.t_res || self.levels.is_empty() {
            return 1;
        }
        let i = self.levels.iter().rposition(|&(b, _)| b <= t).unwrap_or(0);
        let s = if i == 0 {
            self.levels[0].1
        } else {
            let (b, s_i) = self.levels[i];
            let s_prev = self.levels[i - 1].1;
            let b_next = self.levels.get(i + 1).map_or(self.t_res, |l| l.0);
            let f = (t - b) as f32 / (b_next - b).max(1) as f32;
            let inv2 = f * (1.0 / (s_i * s_i) - 1.0 / (s_prev * s_prev)) + 1.0 / (s_prev * s_prev);
            1.0 / inv2.sqrt()
        };
        (s.floor() as u32).max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::densify::window_due;
    use crate::model::Group;

    fn assert_lr(lr: &LrSchedule, want: f32) {
        let LrSchedule::Constant(v) = lr else {
            panic!("{lr:?}")
        };
        assert!((v - want).abs() <= 1e-6 * want, "{v} vs {want}");
    }

    #[test]
    fn presets_carry_the_reference_settings() {
        let f = Preset::new(PresetName::FasterGs);
        assert!(matches!(f.densify, DensifyPreset::Adc(_)));
        assert!(
            f.groups
                .iter()
                .all(|g| g.plan == StepPlan::every_iteration())
        );
        assert_lr(&f.groups[Group::Opacities.index()].lr, 0.025);
        assert_lr(&f.groups[Group::ShRest.index()].lr, 1.25e-4);
        let g = Preset::new(PresetName::Fastgs);
        assert_lr(&g.groups[Group::ShRest.index()].lr, 2.5e-4);
        assert!(
            g.groups[Group::ShRest.index()].plan.due(16)
                && !g.groups[Group::ShRest.index()].plan.due(8)
        );
        assert_eq!(
            Preset::new(PresetName::Mcmc).groups[Group::Opacities.index()].lr,
            LrSchedule::Constant(0.05)
        );
        assert_eq!(f.adam.eps, 1e-15);
        let fast = Preset::new(PresetName::Fast);
        assert_eq!((fast.iterations, fast.sh_interval), (15_000, 500));
        // Rescaling a pre-scaled preset is relative to its own length.
        assert_eq!(fast.scaled(7_500).sh_interval, 250);
    }

    #[test]
    fn scaling_rule_rounds_to_tens_and_keeps_counts() {
        assert_eq!(scale_iter(15_000, 15_000), 7500);
        assert_eq!(scale_iter(500, 10_000), 170);
        assert_eq!(scale_iter(1000, 30_000), 1000);
        let p = Preset::new(PresetName::Fastgs).scaled(15_000);
        let DensifyPreset::FastGs(c) = &p.densify else {
            panic!()
        };
        assert_eq!(
            (c.start, c.end, c.every, c.views, c.tau_d),
            (250, 7500, 250, 10, 5)
        );
        assert_eq!(
            (1..=15_000)
                .filter(|&t| window_due(t, c.start, c.end, c.every))
                .count(),
            28
        );
        assert_eq!(p.sh_interval, 500);
        let sh = &p.groups[Group::ShRest.index()].plan;
        assert_eq!(sh.0, vec![(0, 16), (7501, 32), (10_001, 64)]);
        let LrSchedule::ExpDecay { max_steps, .. } = p.groups[0].lr else {
            panic!()
        };
        assert_eq!(max_steps, 15_000);
    }

    #[test]
    fn fastgs_scene_table_applies() {
        let o = PresetOverrides {
            scene: Some("garden".into()),
            ..Default::default()
        };
        let p = Preset::new(PresetName::FastgsBig).with_overrides(&o);
        let DensifyPreset::FastGs(c) = &p.densify else {
            panic!()
        };
        assert_eq!((c.grad_abs_thresh, c.loss_thresh), (0.0003, 0.06));
        assert_lr(&p.groups[Group::ShRest.index()].lr, 0.001);
        let t = fastgs_scene_override("truck", false).unwrap();
        assert_eq!((t.mult, t.highfeature_lr), (0.7, 0.04));
        assert!(fastgs_scene_override("unknown", true).is_none());
        // Generic presets ignore the table.
        let q = Preset::new(PresetName::Quality).with_overrides(&o);
        assert_lr(&q.groups[Group::ShRest.index()].lr, 0.001);
        let DensifyPreset::FastGs(c) = &q.densify else {
            panic!()
        };
        assert_eq!(c.grad_abs_thresh, 4e-4);
    }

    #[test]
    fn sh_ramp_and_morton() {
        let s = Schedule::from_preset(&Preset::new(PresetName::FasterGs), 30_000);
        assert_eq!(
            [
                s.sh_degree(999),
                s.sh_degree(1000),
                s.sh_degree(2500),
                s.sh_degree(9000)
            ],
            [0, 1, 2, 3]
        );
        assert!(s.morton_due(5000) && !s.morton_due(5001));
        assert_eq!(s.res_scale(10), 1);
    }

    #[test]
    fn dash_schedule_interpolates_in_pixel_count() {
        let d = DashSchedule {
            levels: vec![(0, 4.0), (100, 2.0), (200, 1.0)],
            t_res: 300,
        };
        assert_eq!(d.scale_at(50), 4);
        assert_eq!(d.scale_at(100), 4, "starts at s_(i-1)");
        // Halfway between 1/16 and 1/4 of the pixels: s = 1/sqrt(0.15625) = 2.53.
        assert_eq!(d.scale_at(150), 2);
        assert_eq!(d.scale_at(299), 1);
        assert_eq!(d.scale_at(300), 1);
    }
}
