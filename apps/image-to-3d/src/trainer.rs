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
//! Graphs are cached per [`GraphKey`] (step-plan shapes alternate) and
//! dropped whenever densification, Morton sorting or a raster regrow moves
//! buffers. Overflow is read from the lagging pinned status mirror.

use std::time::Instant;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use spl_core::{CancellationToken, Result, bail};
use spl_gpu::cudarc::driver::CudaSlice;
use ts_rs::TS;

use crate::frame::{
    CameraTable, FrameParams, FrameRing, GraphKey, IterParams, IterationGraph, PinnedStatus,
};
use crate::gpu::{CudaResultExt, Gpu};
use crate::io::images::{ImageCache, Rgb8Image};
use crate::io::scene::{InitKind, Scene, init_gaussians};
use crate::kernels::adam::{Adam, AdamIterPlan, MomentPrecision};
use crate::kernels::densify::{self, DensifyCx, DensifyOutcome, DensifyStats, Strategy};
use crate::kernels::layout::{
    EPS2D_DEFAULT, FLAG_AA, FLAG_ABS_GRAD, FLAG_DENSIFY_STATS, OVF_CHAINS, OVF_INSTANCES,
    OVF_VISIBLE,
};
use crate::kernels::loss::{Loss, resample_target};
use crate::kernels::raster::{FrameSource, GpuRaster, GradTargets, SplatRaster};
use crate::kernels::{Capacity, FrameBuffers, GradSink, RenderMode};
use crate::mesh::geom::{GeomTrain, GeometryConfig};
use crate::model::{Gaussians, GaussiansHost, Grads, plan_capacity};
use crate::presets::{DensifyPreset, Preset, PresetName, PresetOverrides, Schedule};
use crate::rng::Philox;

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
    /// Gaussian capacity ceiling; free VRAM decides when absent.
    #[serde(default)]
    #[ts(optional)]
    pub max_gaussians: Option<u32>,
    /// Geometry regularisers (on by default with the Geometry preset).
    #[serde(default)]
    #[ts(optional)]
    pub geometry: Option<GeometryConfig>,
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
            max_gaussians: None,
            geometry: None,
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

    /// The geometry regularisers this run uses, if any.
    pub fn geometry(&self) -> Option<GeometryConfig> {
        self.geometry
            .clone()
            .or_else(|| (self.preset == PresetName::Geometry).then(GeometryConfig::default))
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
    /// Largest `total − free` device memory seen (whole device, sampled).
    #[ts(type = "number")]
    pub peak_vram_bytes: u64,
    pub graph_captures: u32,
    pub densify_steps: u32,
    pub capped: u32,
    /// Raster buffers grown after an overflow (each loses the poisoned iterations).
    #[serde(default)]
    pub raster_regrows: u32,
    /// Most Gaussians alive at once.
    #[serde(default)]
    pub peak_gaussians: u32,
    /// Fixed Gaussian capacity the run allocated (spec-train 5.10).
    #[serde(default)]
    pub capacity: u32,
    /// Mean training loss over the last progress window.
    #[serde(default)]
    #[ts(optional)]
    pub final_loss: Option<f64>,
}

/// Instances the raster starts with; it grows to 1.25x the need on overflow.
const INITIAL_INSTANCES: u32 = 1 << 24;
/// Gaussian capacity ceiling (spec-train 5.10 picks less when VRAM is short).
const MAX_GAUSSIANS: u64 = 8 << 20;
/// Graphs kept per Gaussian count (step-plan shapes alternate, e.g. SH-rest every 16).
const MAX_GRAPHS: usize = 8;

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
    /// `None` only while it is being regrown.
    pub raster: Option<GpuRaster>,
    /// Geometry regularisers (Geometry preset).
    pub geometry: Option<GeomTrain>,
    pub strategy: Box<dyn Strategy>,
    pub stats: DensifyStats,
    pub ring: FrameRing,
    pub status: PinnedStatus,
    /// Captured iterations by key (cleared whenever buffers move).
    pub graphs: Vec<IterationGraph>,
    /// Run every iteration eagerly instead of through graphs (debugging, profiles).
    pub eager: bool,
    /// The resampled f32 target of the current iteration.
    pub target: CudaSlice<f32>,
    pub order: ViewOrder,
    pub rng: Philox,
    /// Last completed iteration.
    pub t: u32,
    pub total: u32,
    pub summary: TrainSummary,
    max_size: (u32, u32),
    loss_window: u32,
}

impl Trainer {
    pub fn new(gpu: &Gpu, scene: Scene, cfg: TrainConfig) -> Result<Self> {
        let mut preset = cfg.resolve();
        let total = preset.iterations;
        let mut schedule = Schedule::from_preset(&preset, total);
        crate::dash::install(&mut preset, &mut schedule, &scene)?;
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
        let max_px = scene
            .views
            .iter()
            .map(|v| v.width as usize * v.height as usize)
            .max()
            .unwrap_or(1);
        let max_size = scene
            .views
            .iter()
            .fold((1, 1), |(w, h), v| (w.max(v.width), h.max(v.height)));
        // Images are resident; the raster's reserve is estimated at a 4M-Gaussian
        // frame plus room for the instance buffers to grow (spec-train 5.10).
        let (free, _) = gpu.mem_info()?;
        let variant = preset.variant;
        // A capacity ceiling below 4M shrinks the probe frame and the 1 GiB
        // growth room alike (the shared-GPU plans of `crate::vram`;
        // unchanged when absent).
        let probe_n = cfg
            .max_gaussians
            .map_or(4 << 20, |m| m.clamp(1 << 16, 4 << 20));
        let probe = Capacity::new(probe_n, INITIAL_INSTANCES, max_size.0, max_size.1, &variant);
        let reserve = FrameBuffers::vram_bytes(probe, &variant).total()
            + ((1u64 << 30) * probe_n as u64 / (4 << 20));
        let preset_cap = match &preset.densify {
            DensifyPreset::Mcmc(c) => c.cap_max as u64,
            _ => MAX_GAUSSIANS,
        }
        .min(cfg.max_gaussians.map_or(u64::MAX, u64::from));
        let bf16 = preset.adam.moments != MomentPrecision::F32;
        // 10% of the free memory stays spare: other processes share the device.
        let capacity = plan_capacity(free - free / 10, 0, reserve, preset_cap, bf16)
            .max(host.len().next_multiple_of(1024) as u64) as usize;
        tracing::info!(
            "free VRAM {} MiB, raster reserve {} MiB: capacity {capacity} Gaussians",
            free >> 20,
            reserve >> 20
        );
        let mut g = Gaussians::alloc(gpu, capacity, preset.antialiased)?;
        g.upload(gpu, &host)?;
        let strategy = match &preset.densify {
            DensifyPreset::Adc(c) => densify::adc(c.clone()),
            DensifyPreset::FastGs(c) => densify::fastgs(c.clone()),
            DensifyPreset::Mcmc(c) => densify::mcmc(c.clone()),
        };
        let cap = Capacity::new(
            capacity as u32,
            INITIAL_INSTANCES,
            max_size.0,
            max_size.1,
            &variant,
        );
        let raster = GpuRaster::new(gpu, cap, variant, strategy.wants_abs_grad())?;
        let geometry = match cfg.geometry() {
            Some(c) => {
                let mut geo = GeomTrain::new(gpu, c, &variant, cap, scene.extent, total)?;
                geo.enable_multiview(gpu, &scene.views, &scene_gpu.images)?;
                Some(geo)
            }
            None => None,
        };
        let mut t = Self {
            gpu: gpu.clone(),
            grads: Grads::alloc(gpu, capacity)?,
            adam: Adam::new(
                gpu,
                capacity,
                preset.adam.clone(),
                preset.groups.clone(),
                scene.extent,
            )?,
            loss: Loss::new(gpu, max_size.0, max_size.1, preset.loss)?,
            stats: DensifyStats::alloc(gpu, capacity, strategy.wants_abs_grad())?,
            ring: FrameRing::new(gpu, FrameRing::DEFAULT_SLOTS)?,
            status: PinnedStatus::new(gpu)?,
            graphs: Vec::new(),
            eager: false,
            raster: Some(raster),
            geometry,
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
            g,
            schedule,
            preset,
            scene,
            cfg,
            max_size,
            loss_window: 0,
        };
        t.summary.capacity = t.g.cap;
        t.gpu.sync()?;
        t.sample_vram()?;
        Ok(t)
    }

    fn raster(&mut self) -> &mut GpuRaster {
        self.raster.as_mut().expect("raster present outside regrow")
    }

    fn sample_vram(&mut self) -> Result<()> {
        let (free, total) = self.gpu.mem_info()?;
        self.summary.peak_vram_bytes = self.summary.peak_vram_bytes.max(total - free);
        Ok(())
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
        let overflow_retry = self.check_overflow()?;
        let t = self.t + 1;
        let view = self.order.next_view();
        let res_scale = self.schedule.res_scale(t);
        let sh_degree = self.schedule.sh_degree(t);
        let plan = self.adam.plan(t);
        let params = self.iter_params(t, view, sh_degree, res_scale, plan);
        self.ring.push(&params)?;

        // CUDA-graph capture point: graphs are keyed by everything that changes
        // a launch configuration or a launch argument (spec-train 12).
        let key = GraphKey {
            n_gaussians: self.g.n,
            render_w: params.frame.render_w,
            render_h: params.frame.render_h,
            sh_degree,
            due_mask: plan.due_mask,
            stats: params.frame.flags & FLAG_DENSIFY_STATS != 0,
            abs_grad: params.frame.flags & FLAG_ABS_GRAD != 0,
        };
        let mut recaptured = false;
        if self.eager {
            self.record_iteration(t)?;
        } else {
            if self
                .graphs
                .first()
                .is_some_and(|g| g.key.n_gaussians != key.n_gaussians)
                || self.graphs.len() >= MAX_GRAPHS
            {
                self.graphs.clear();
            }
            let i = match self.graphs.iter().position(|g| g.key == key) {
                Some(i) => i,
                None => {
                    let stream = self.gpu.stream.clone();
                    let graph =
                        IterationGraph::capture(&stream, key, |_| self.record_iteration(t))?;
                    self.summary.graph_captures += 1;
                    recaptured = true;
                    self.graphs.push(graph);
                    self.graphs.len() - 1
                }
            };
            self.graphs[i].replay()?;
        }
        self.ring.enqueued(&self.gpu.stream)?;
        self.loss_window += 1;

        let densify = self.hooks(t)?;
        self.t = t;
        Ok(StepReport {
            t,
            n: self.g.n,
            res_scale,
            recaptured,
            overflow_retry,
            densify,
        })
    }

    /// Grows the raster when a finished iteration reported an overflow (the
    /// poisoned iterations changed nothing; spec-raster 1.5).
    fn check_overflow(&mut self) -> Result<bool> {
        if self.status.read()?.overflow == 0 {
            return Ok(false);
        }
        self.gpu.sync()?;
        let st = self
            .raster
            .as_ref()
            .expect("raster present")
            .frame
            .iter
            .read_status(&self.gpu)?;
        if st.overflow == 0 {
            self.status.clear();
            return Ok(false);
        }
        let old = self.raster().capacity();
        let mut instances = old.instances;
        if st.overflow & (OVF_INSTANCES | OVF_CHAINS) != 0 {
            let need = (st.n_instances as u64 * 5 / 4).max(old.instances as u64 * 5 / 4);
            instances = need.min(u32::MAX as u64 / 2) as u32;
        }
        if st.overflow & OVF_VISIBLE != 0 {
            bail!("visible-set overflow with V_cap = N_cap ({})", old.visible);
        }
        tracing::info!(
            "raster overflow {:#x} at iteration {}: instances {} -> {}",
            st.overflow,
            st.overflow_iteration,
            old.instances,
            instances
        );
        self.graphs.clear();
        self.raster = None;
        let cap = Capacity::new(
            old.gaussians,
            instances,
            self.max_size.0,
            self.max_size.1,
            &self.preset.variant,
        );
        let mut r = GpuRaster::new(
            &self.gpu,
            cap,
            self.preset.variant,
            self.strategy.wants_abs_grad(),
        )?;
        r.set_iteration_counter(&self.gpu, self.ring.pushed as u32)?;
        self.raster = Some(r);
        self.status.clear();
        self.summary.raster_regrows += 1;
        self.gpu.sync()?;
        self.sample_vram()?;
        Ok(true)
    }

    /// The captured sequence of one iteration: no host sync, no allocation.
    fn record_iteration(&mut self, t: u32) -> Result<()> {
        let gpu = &self.gpu;
        let raster = self.raster.as_mut().expect("raster present");
        raster.forward(
            gpu,
            &mut self.g,
            &self.scene_gpu.cams,
            FrameSource::Ring(&self.ring),
            RenderMode::Train,
        )?;
        let io = raster.loss_io();
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
        if let Some(geo) = &mut self.geometry {
            geo.after_forward(gpu, raster, &self.g, Some(&self.scene_gpu.images))?;
        }
        let want_stats = self.strategy.wants_stats(t);
        raster.backward(
            gpu,
            &mut self.g,
            GradTargets {
                sink: GradSink::Write(&mut self.grads),
                camera: None,
                stats: want_stats.then_some(&mut self.stats),
            },
        )?;
        if let Some(geo) = &mut self.geometry {
            geo.after_backward(gpu, raster, &self.g, &mut self.grads)?;
        }
        self.adam
            .step(gpu, &mut self.g, &mut self.grads, raster.params())?;
        raster.end_frame(gpu, &self.status)
    }

    /// Everything after the optimizer step of iteration `t` (spec-train 1).
    fn hooks(&mut self, t: u32) -> Result<Option<DensifyOutcome>> {
        let wants_views = self.strategy.wants_views(t);
        let morton = self.schedule.morton_due(t);
        let views: Vec<IterParams> = if wants_views {
            let (deg, res) = (self.schedule.sh_degree(t), self.schedule.res_scale(t));
            (0..self.scene.views.len())
                .map(|v| self.iter_params(t, v, deg, res, AdamIterPlan::default()))
                .collect()
        } else {
            Vec::new()
        };
        let res_scale = self.schedule.res_scale(t);
        let raster = self.raster.as_mut().expect("raster present");
        let mut cx = DensifyCx {
            gpu: &self.gpu,
            g: &mut self.g,
            grads: &mut self.grads,
            adam: &mut self.adam,
            stats: &mut self.stats,
            raster,
            loss: &mut self.loss,
            scene: &self.scene_gpu,
            extent: self.scene.extent,
            res_scale,
            rng: self.rng,
            views: &views,
        };
        let outcome = self.strategy.after_step(&mut cx, t)?;
        // Right after emission when both fall on t (spec-train 5.9).
        if morton {
            self.strategy.morton_sort(&mut cx, t)?;
        }
        let densify = outcome.changed.then_some(outcome);
        self.summary.peak_gaussians = self.summary.peak_gaussians.max(self.g.n);
        if let Some(d) = densify {
            self.summary.densify_steps += 1;
            self.summary.capped += d.capped;
        }
        if densify.is_some() || morton {
            // Buffers moved: a captured graph would replay the old order.
            self.graphs.clear();
        }
        if densify.is_some() || wants_views {
            self.sample_vram()?;
        }
        Ok(densify)
    }

    /// Mean loss since the last read (host sync).
    pub fn read_loss(&mut self) -> Result<Option<f64>> {
        if self.loss_window == 0 {
            return Ok(None);
        }
        let v = self.loss.take_sums(&self.gpu, self.loss_window)?;
        self.loss_window = 0;
        Ok(Some(v.loss))
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
                let loss = self.read_loss()?;
                self.sample_vram()?;
                let elapsed = started.elapsed().as_secs_f64();
                self.summary.final_loss = loss.or(self.summary.final_loss);
                progress(&TrainProgress {
                    t: self.t,
                    total: self.total,
                    n_gaussians: self.g.n,
                    elapsed_s: elapsed,
                    iterations_per_s: self.t as f64 / elapsed.max(1e-9),
                    loss,
                });
            }
        }
        self.gpu.sync()?;
        let st = self.status.read()?;
        if st.overflow != 0 {
            tracing::warn!(
                "the last iterations overflowed the raster ({:#x})",
                st.overflow
            );
        }
        self.summary.iterations = self.t;
        self.summary.train_s = started.elapsed().as_secs_f64();
        self.summary.final_gaussians = self.g.n;
        Ok(self.summary.clone())
    }

    /// An 8-bit eval render of scene view `view` at full resolution and the
    /// current SH degree (Infer mode, eager).
    pub fn render(&mut self, view: usize) -> Result<Rgb8Image> {
        let t = self.t.max(1);
        let deg = self.schedule.sh_degree(t);
        let mut params = self.iter_params(t, view, deg, 1, AdamIterPlan::default());
        params.frame.flags &= !(FLAG_DENSIFY_STATS | FLAG_ABS_GRAD);
        let (w, h) = (params.frame.render_w, params.frame.render_h);
        let gpu = self.gpu.clone();
        let cams = &self.scene_gpu.cams;
        let raster = self.raster.as_mut().expect("raster present");
        raster.forward(
            &gpu,
            &mut self.g,
            cams,
            FrameSource::Eager(&params),
            RenderMode::Infer,
        )?;
        let st = raster.frame.iter.read_status(&gpu)?;
        if st.overflow != 0 {
            // Eval renders see every Gaussian at once; grow and retry once.
            self.status.clear();
            self.force_regrow(st.n_instances)?;
            return self.render(view);
        }
        let px = (w * h) as usize;
        let planar = gpu
            .stream
            .clone_dtoh(&self.raster().out.rgb.slice(0..3 * px))
            .cuda("download render")?;
        Ok(planar_to_rgb8(&planar, w, h))
    }

    fn force_regrow(&mut self, need: u32) -> Result<()> {
        self.gpu.sync()?;
        let old = self.raster().capacity();
        self.graphs.clear();
        self.raster = None;
        let cap = Capacity::new(
            old.gaussians,
            (need as u64 * 5 / 4).max(old.instances as u64 * 5 / 4) as u32,
            self.max_size.0,
            self.max_size.1,
            &self.preset.variant,
        );
        let mut r = GpuRaster::new(
            &self.gpu,
            cap,
            self.preset.variant,
            self.strategy.wants_abs_grad(),
        )?;
        r.set_iteration_counter(&self.gpu, self.ring.pushed as u32)?;
        self.raster = Some(r);
        self.summary.raster_regrows += 1;
        Ok(())
    }

    pub fn snapshot(&mut self) -> Result<GaussiansHost> {
        self.g.download(&self.gpu)
    }

    /// LazyCatchUp flush and the final prune of o < `final_prune_opacity`;
    /// returns the final cloud (in training order).
    pub fn finish(mut self) -> Result<(GaussiansHost, TrainSummary)> {
        self.adam.flush(&self.gpu, &mut self.g, self.t)?;
        let mut host = self.g.download(&self.gpu)?;
        host.sh_degree = self.schedule.sh_degree(self.t.max(1));
        let keep: Vec<usize> = (0..host.len())
            .filter(|&i| sigmoid(host.opacities[i]) >= self.preset.final_prune_opacity)
            .collect();
        let host = select(&host, &keep);
        self.summary.final_gaussians = host.len() as u32;
        Ok((host, self.summary))
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// The Gaussians at `keep`, in that order.
pub fn select(g: &GaussiansHost, keep: &[usize]) -> GaussiansHost {
    let pick = |v: &[f32], k: usize| -> Vec<f32> {
        keep.iter()
            .flat_map(|&i| v[i * k..(i + 1) * k].iter().copied())
            .collect()
    };
    GaussiansHost {
        means: pick(&g.means, 3),
        scales: pick(&g.scales, 3),
        rotations: pick(&g.rotations, 4),
        opacities: pick(&g.opacities, 1),
        sh0: pick(&g.sh0, 3),
        sh_rest: pick(&g.sh_rest, 45),
        sh_degree: g.sh_degree,
        antialiased: g.antialiased,
    }
}

/// f32 planar `[3][H][W]` → interleaved u8 like `torchvision.save_image`
/// (`floor(clamp(x, 0, 1)·255 + 0.5)`).
pub fn planar_to_rgb8(planar: &[f32], w: u32, h: u32) -> Rgb8Image {
    let px = (w * h) as usize;
    let mut img = Rgb8Image::new(w, h);
    for i in 0..px {
        for c in 0..3 {
            let x = planar[c * px + i];
            let x = if x.is_nan() { 0.0 } else { x.clamp(0.0, 1.0) };
            img.data[i * 3 + c] = (x * 255.0 + 0.5).floor() as u8;
        }
    }
    img
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

#[cfg(test)]
#[path = "trainer_gpu_tests.rs"]
mod gpu_tests;
