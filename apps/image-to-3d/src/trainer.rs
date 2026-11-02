//! The training loop: one iteration as a capturable sequence, the hooks after
//! it, progress, metrics and the run. Owner: trainer (E7).
//! Spec: spec-train 1 (iteration order), 2.1, 11-12; spec-raster 1.3, 1.5, 8.
//!
//! One iteration (host side, spec-train 12):
//!
//! ```text
//! view = order.next(); s = schedule.res_scale(t); deg = schedule.sh_degree(t)
//! plan = adam.plan(t); ring.push(IterParams)            // never overwrites an in-flight slot
//! key = (n, size, deg, plan shape, stats, abs_grad)     // CUDA-graph capture point:
//! graph = cache[key] or capture(record_iteration)       //   re-capture only when key changes
//! graph.launch()
//! hooks(t): densify / prune / reset / Morton / final prune / snapshot / eval
//! every 100 t: lagging read of loss sums and status (overflow → grow, replay)
//! ```
//!
//! `record_iteration` (captured, one stream): raster.forward (frame_begin …
//! raster_fwd) → resample_target → loss.forward_backward → raster.backward
//! (raster_bwd → preprocess_bwd → sink) → adam.step → [mcmc.noise] →
//! raster.end_frame. No host sync inside.
//!
//! The loop and its bookkeeping are real; the stages it calls are stubs
//! until their owners land them, so `Trainer::new` currently fails at the
//! first one (init_gaussians).

use std::time::Instant;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use spl_core::{CancellationToken, Result, bail};
use spl_gpu::cudarc::driver::CudaSlice;
use ts_rs::TS;

use crate::frame::{
    CameraTable, FrameParams, FrameRing, GraphKey, IterParams, IterationGraph, PinnedStatus,
};
use crate::gpu::{CudaResultExt, Gpu, not_implemented};
use crate::io::images::{ImageCache, Rgb8Image};
use crate::io::scene::{InitKind, Scene, init_gaussians};
use crate::kernels::adam::{Adam, AdamIterPlan};
use crate::kernels::densify::{self, DensifyCx, DensifyOutcome, DensifyStats, Strategy};
use crate::kernels::layout::{EPS2D_DEFAULT, FLAG_AA, FLAG_ABS_GRAD, FLAG_DENSIFY_STATS};
use crate::kernels::loss::{Loss, resample_target};
use crate::kernels::raster::{FrameSource, GradTargets, SplatRaster};
use crate::kernels::{GradSink, RenderMode};
use crate::model::{Gaussians, GaussiansHost, Grads, plan_capacity};
use crate::presets::{DensifyPreset, Preset, PresetName, PresetOverrides, Schedule};
use crate::rng::Philox;

const OWNER: &str = "trainer (E7)";

/// What to train and how (spec-train 11).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct TrainConfig {
    #[serde(default)]
    pub preset: PresetName,
    /// Overrides the preset's iteration count; every schedule rescales.
    #[serde(default)]
    #[ts(optional)]
    pub iterations: Option<u32>,
    #[serde(default)]
    #[ts(type = "number")]
    pub seed: u64,
    /// Any preset field, for ablations.
    #[serde(default)]
    pub overrides: PresetOverrides,
    /// Live snapshot cadence for the viewer; off when absent.
    #[serde(default)]
    #[ts(optional)]
    pub snapshot_every_s: Option<f32>,
    /// Test-view PSNR every n iterations; off when absent.
    #[serde(default)]
    #[ts(optional)]
    pub eval_every: Option<u32>,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            preset: PresetName::Quality,
            iterations: None,
            seed: 0,
            overrides: PresetOverrides::default(),
            snapshot_every_s: None,
            eval_every: None,
        }
    }
}

impl TrainConfig {
    /// The preset with overrides applied and scaled to the iteration count.
    pub fn resolve(&self) -> Preset {
        let p = Preset::new(self.preset).with_overrides(&self.overrides);
        match self.iterations {
            Some(t) if t != p.iterations => p.scaled(t),
            _ => p,
        }
    }
}

/// The scene on the device: base poses and the training images.
pub struct SceneGpu {
    pub cams: CameraTable,
    pub images: ImageCache,
    /// Scene view indices the cache holds, in training order.
    pub train: Vec<usize>,
}

/// Epoch-shuffled view order (Fisher-Yates per epoch, like the Inria
/// "viewpoint stack"), driven by Philox so a replay draws the same views.
#[derive(Clone, Debug)]
pub struct ViewOrder {
    views: Vec<usize>,
    order: Vec<usize>,
    pos: usize,
    epoch: u32,
    rng: Philox,
}

impl ViewOrder {
    pub fn new(views: Vec<usize>, seed: u64) -> Self {
        Self {
            order: Vec::new(),
            views,
            pos: 0,
            epoch: 0,
            rng: Philox::new(seed ^ 0x5649_4557),
        }
    }

    pub fn next_view(&mut self) -> usize {
        if self.pos == self.order.len() {
            self.order = self.views.clone();
            for i in (1..self.order.len()).rev() {
                let r = self.rng.draw4(
                    i as u32,
                    self.epoch,
                    crate::kernels::layout::RNG_VIEW_SAMPLE,
                )[0];
                let j = ((r as u64 * (i as u64 + 1)) >> 32) as usize;
                self.order.swap(i, j);
            }
            self.epoch += 1;
            self.pos = 0;
        }
        self.pos += 1;
        self.order[self.pos - 1]
    }
}

/// Live progress, at most every 0.5 s.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct TrainProgress {
    pub t: u32,
    pub total: u32,
    pub n_gaussians: u32,
    pub elapsed_s: f64,
    pub iterations_per_s: f64,
    /// Mean loss over the last read window (lagging).
    pub loss: Option<f64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct StepReport {
    pub t: u32,
    pub n: u32,
    pub res_scale: u32,
    pub recaptured: bool,
    pub overflow_retry: bool,
    pub densify: Option<DensifyOutcome>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct TrainSummary {
    pub iterations: u32,
    /// From the first launch to the last completion, incl. densify, scoring,
    /// Morton, captures; excl. decode, init, eval, export (spec-train 7.5).
    pub train_s: f64,
    pub final_gaussians: u32,
    #[ts(type = "number")]
    pub peak_vram_bytes: u64,
    pub graph_captures: u32,
    pub densify_steps: u32,
    pub capped: u32,
}

pub struct Trainer {
    pub gpu: Gpu,
    pub cfg: TrainConfig,
    pub preset: Preset,
    pub schedule: Schedule,
    pub scene: Scene,
    pub scene_gpu: SceneGpu,
    pub g: Gaussians,
    pub grads: Grads,
    pub adam: Adam,
    pub loss: Loss,
    pub raster: Box<dyn SplatRaster>,
    pub strategy: Box<dyn Strategy>,
    pub stats: DensifyStats,
    pub ring: FrameRing,
    pub status: PinnedStatus,
    pub graph: Option<IterationGraph>,
    /// The resampled f32 target of the current iteration.
    pub target: CudaSlice<f32>,
    pub order: ViewOrder,
    pub rng: Philox,
    /// Last completed iteration.
    pub t: u32,
    pub total: u32,
    pub summary: TrainSummary,
}

impl Trainer {
    pub fn new(
        gpu: &Gpu,
        scene: Scene,
        raster: Box<dyn SplatRaster>,
        cfg: TrainConfig,
    ) -> Result<Self> {
        let preset = cfg.resolve();
        let total = preset.iterations;
        let schedule = Schedule::from_preset(&preset, total);
        let kind = match preset.densify {
            DensifyPreset::Mcmc(_) => InitKind::Mcmc,
            _ => InitKind::Standard,
        };
        let host = init_gaussians(&scene.points, kind)?;
        let scene_gpu = SceneGpu {
            cams: CameraTable::upload(gpu, &scene.views)?,
            images: ImageCache::upload(gpu, &scene, &scene.train)?,
            train: scene.train.clone(),
        };
        // The raster and the images are resident already, so free memory is what
        // the store, moments, gradients and slab may use.
        let (free, _) = gpu.mem_info()?;
        let preset_cap = match &preset.densify {
            DensifyPreset::Mcmc(c) => c.cap_max as u64,
            _ => u64::MAX,
        };
        let capacity = plan_capacity(free, 0, 0, preset_cap, false).max(host.len() as u64) as usize;
        let mut g = Gaussians::alloc(gpu, capacity, preset.antialiased)?;
        g.upload(gpu, &host)?;
        let strategy = match &preset.densify {
            DensifyPreset::Adc(c) => densify::adc(c.clone()),
            DensifyPreset::FastGs(c) => densify::fastgs(c.clone()),
            DensifyPreset::Mcmc(c) => densify::mcmc(c.clone()),
        };
        let max_px = scene
            .views
            .iter()
            .map(|v| v.width as usize * v.height as usize)
            .max()
            .unwrap_or(1);
        let (max_w, max_h) = scene
            .views
            .iter()
            .fold((1, 1), |(w, h), v| (w.max(v.width), h.max(v.height)));
        Ok(Self {
            gpu: gpu.clone(),
            grads: Grads::alloc(gpu, capacity)?,
            adam: Adam::new(
                gpu,
                capacity,
                preset.adam.clone(),
                preset.groups.clone(),
                scene.extent,
            )?,
            loss: Loss::new(gpu, max_w, max_h, preset.loss)?,
            stats: DensifyStats::alloc(gpu, capacity, strategy.wants_abs_grad())?,
            ring: FrameRing::new(gpu, FrameRing::DEFAULT_SLOTS)?,
            status: PinnedStatus::new(gpu)?,
            graph: None,
            target: gpu
                .stream
                .alloc_zeros::<f32>(3 * max_px)
                .cuda("alloc target")?,
            order: ViewOrder::new(scene.train.clone(), cfg.seed),
            rng: Philox::new(cfg.seed),
            t: 0,
            total,
            summary: TrainSummary::default(),
            scene_gpu,
            strategy,
            raster,
            g,
            schedule,
            preset,
            scene,
            cfg,
        })
    }

    /// Host-written parameters of iteration `t` (one ring slot).
    pub fn iter_params(
        &self,
        t: u32,
        view: usize,
        sh_degree: u32,
        res_scale: u32,
        adam: AdamIterPlan,
    ) -> IterParams {
        let v = &self.scene.views[view];
        let entry = self
            .scene_gpu
            .images
            .entry(view)
            .copied()
            .unwrap_or_default();
        let mut flags = 0;
        if self.preset.antialiased {
            flags |= FLAG_AA;
        }
        if self.strategy.wants_stats(t) {
            flags |= FLAG_DENSIFY_STATS;
            if self.strategy.wants_abs_grad() {
                flags |= FLAG_ABS_GRAD;
            }
        }
        let means_lr = self.preset.groups[0].lr.at(t, self.scene.extent) as f32;
        IterParams {
            frame: FrameParams {
                iteration: t,
                cam_index: view as u32,
                n_gaussians: self.g.n,
                sh_degree,
                bg: [0.0; 3],
                eps2d: EPS2D_DEFAULT,
                flags,
                adam_lr_scale: 1.0,
                bias_c1: 1.0,
                bias_c2: 1.0,
                render_w: v.width / res_scale.max(1),
                render_h: v.height / res_scale.max(1),
                box_beta: self.preset.compact_box_beta,
                pad: 0,
            },
            target_offset: entry.offset,
            target_w: entry.width,
            target_h: entry.height,
            res_scale,
            means_lr,
            noise_scale: match &self.preset.densify {
                DensifyPreset::Mcmc(c) => means_lr * c.noise_lr,
                _ => 0.0,
            },
            seed: self.rng.seed,
            adam,
            ..Default::default()
        }
    }

    /// One iteration plus its hooks.
    pub fn step(&mut self) -> Result<StepReport> {
        let t = self.t + 1;
        let view = self.order.next_view();
        let res_scale = self.schedule.res_scale(t);
        let sh_degree = self.schedule.sh_degree(t);
        let plan = self.adam.plan(t);
        let params = self.iter_params(t, view, sh_degree, res_scale, plan);
        self.ring.push(&params)?;

        // CUDA-graph capture point: graphs are keyed by everything that changes
        // a launch configuration; a new key re-captures (spec-train 12).
        let key = GraphKey {
            n_gaussians: self.g.n,
            render_w: params.frame.render_w,
            render_h: params.frame.render_h,
            sh_degree,
            due_mask: plan.due_mask,
            stats: params.frame.flags & FLAG_DENSIFY_STATS != 0,
            abs_grad: params.frame.flags & FLAG_ABS_GRAD != 0,
        };
        let recaptured = self.graph.as_ref().is_none_or(|g| g.key != key);
        if recaptured {
            // E7: self.graph = Some(IterationGraph::capture(&self.gpu.stream, key, record_iteration)?)
            // and count it in summary.graph_captures; until then the sequence runs eagerly.
            self.graph = None;
        }
        self.record_iteration()?;

        let densify = self.hooks(t)?;
        self.t = t;
        Ok(StepReport {
            t,
            n: self.g.n,
            res_scale,
            recaptured,
            overflow_retry: false,
            densify,
        })
    }

    /// The captured sequence of one iteration (eager until graphs land).
    fn record_iteration(&mut self) -> Result<()> {
        let gpu = &self.gpu;
        self.raster.forward(
            gpu,
            &mut self.g,
            &self.scene_gpu.cams,
            FrameSource::Ring(&self.ring),
            RenderMode::Train,
        )?;
        let io = self.raster.loss_io();
        // u8 → f32 planar at the render size: a copy at scale 1, the tent filter in warm-up.
        resample_target(
            gpu,
            &self.loss.kernels,
            io.params,
            &self.scene_gpu.images,
            &mut self.target,
        )?;
        self.loss
            .forward_backward(gpu, io.params, io.image, &self.target, &mut io.grads.d_rgb)?;
        let want_stats = self.strategy.wants_stats(self.t + 1);
        self.raster.backward(
            gpu,
            &mut self.g,
            GradTargets {
                sink: GradSink::Write(&mut self.grads),
                camera: None,
                stats: want_stats.then_some(&mut self.stats),
            },
        )?;
        self.adam
            .step(gpu, &mut self.g, &mut self.grads, self.raster.params())?;
        self.raster.end_frame(gpu, &self.status)
    }

    /// Everything after the optimizer step of iteration `t` (spec-train 1).
    fn hooks(&mut self, t: u32) -> Result<Option<DensifyOutcome>> {
        let mut cx = DensifyCx {
            gpu: &self.gpu,
            g: &mut self.g,
            grads: &mut self.grads,
            adam: &mut self.adam,
            stats: &mut self.stats,
            raster: self.raster.as_mut(),
            loss: &mut self.loss,
            scene: &self.scene_gpu,
            extent: self.scene.extent,
            res_scale: self.schedule.res_scale(t),
            rng: self.rng,
        };
        let outcome = self.strategy.after_step(&mut cx, t)?;
        let densify = outcome.changed.then_some(outcome);
        if let Some(d) = densify {
            self.summary.densify_steps += 1;
            self.summary.capped += d.capped;
            self.graph = None;
        }
        if self.schedule.morton_due(t) {
            return Err(not_implemented("Morton re-sort hook", OWNER));
        }
        Ok(densify)
    }

    /// The whole run; checks `cancel` every 50 iterations and calls `progress`
    /// at most every 0.5 s.
    pub fn run(
        &mut self,
        cancel: &CancellationToken,
        progress: &mut dyn FnMut(&TrainProgress),
    ) -> Result<TrainSummary> {
        let started = Instant::now();
        let mut last = Instant::now();
        while self.t < self.total {
            if self.t.is_multiple_of(50) && cancel.is_cancelled() {
                bail!("cancelled");
            }
            self.step()?;
            if last.elapsed().as_secs_f64() >= 0.5 || self.t == self.total {
                last = Instant::now();
                let elapsed = started.elapsed().as_secs_f64();
                progress(&TrainProgress {
                    t: self.t,
                    total: self.total,
                    n_gaussians: self.g.n,
                    elapsed_s: elapsed,
                    iterations_per_s: self.t as f64 / elapsed.max(1e-9),
                    loss: None,
                });
            }
        }
        self.gpu.sync()?;
        self.summary.iterations = self.t;
        self.summary.train_s = started.elapsed().as_secs_f64();
        self.summary.final_gaussians = self.g.n;
        Ok(self.summary.clone())
    }

    /// An 8-bit eval render of scene view `view` (Infer mode, eager).
    pub fn render(&mut self, _view: usize) -> Result<Rgb8Image> {
        Err(not_implemented("Trainer::render", OWNER))
    }

    pub fn snapshot(&mut self) -> Result<GaussiansHost> {
        self.g.download(&self.gpu)
    }

    /// LazyCatchUp flush, prune o < 1/255, Morton order; returns the final cloud.
    pub fn finish(mut self) -> Result<(GaussiansHost, TrainSummary)> {
        self.adam.flush(&self.gpu, &mut self.g, self.t)?;
        Err(not_implemented(
            "Trainer::finish (final prune, Morton)",
            OWNER,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn view_order_is_a_seeded_permutation_per_epoch() {
        let views: Vec<usize> = (10..30).collect();
        let mut a = ViewOrder::new(views.clone(), 7);
        let mut b = ViewOrder::new(views.clone(), 7);
        let first: Vec<usize> = (0..20).map(|_| a.next_view()).collect();
        let mut sorted = first.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, views, "each epoch visits every view once");
        assert_eq!(first, (0..20).map(|_| b.next_view()).collect::<Vec<_>>());
        let second: Vec<usize> = (0..20).map(|_| a.next_view()).collect();
        assert_ne!(first, second, "epochs reshuffle");
        let mut c = ViewOrder::new(views, 8);
        assert_ne!(first, (0..20).map(|_| c.next_view()).collect::<Vec<_>>());
    }

    #[test]
    fn config_resolves_presets_and_iterations() {
        let cfg = TrainConfig {
            preset: PresetName::Fastgs,
            iterations: Some(10_000),
            ..Default::default()
        };
        let p = cfg.resolve();
        assert_eq!(p.iterations, 10_000);
        let DensifyPreset::FastGs(c) = &p.densify else {
            panic!()
        };
        assert_eq!(c.every, 170);
        let json = serde_json::to_value(TrainConfig::default()).unwrap();
        assert_eq!(json["preset"], "quality");
        let parsed: TrainConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(parsed, TrainConfig::default());
    }
}
