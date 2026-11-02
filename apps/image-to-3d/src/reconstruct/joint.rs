//! Stage C for unposed input: the Phase 3 [`Trainer`] plus joint per-image
//! SE(3) and shared-focal refinement, PPISP and the scale-shift-invariant
//! depth prior (Report section 3, Stage C; spec-train 10). Owner:
//! mvp-pipeline.
//!
//! The trainer's own iteration (`Trainer::step`, CUDA graphs) runs whenever
//! none of the three is active; otherwise [`Joint::step`] runs the same
//! sequence eagerly with the additions (host-chosen frame indices and
//! learning rates):
//!
//! ```text
//! raster.forward → resample_target → [ppisp.forward] → loss → [ppisp.backward]
//! → [depth: gather → ssi fit → λ∂L/∂d → scatter into d_extra]
//! → raster.backward (+ CameraGrads) → adam.step → [pose/focal Adam] → [ppisp step]
//! → end_frame → densify hooks
//! ```
//!
//! Pose updates follow spec-train 10.1 (per-image Adam on the twist in
//! `CameraTable::pose_delta`, the first training view frozen as the gauge;
//! one log-focal per intrinsics entry). Test-time refinement ([`TestViews`])
//! optimizes held-out poses on frozen Gaussians with the same kernels.

use spl_core::{Result, bail};
use spl_gpu::cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use crate::frame::{CameraTable, IterParams};
use crate::gpu::{CudaFile, CudaResultExt, Defines, Gpu, KernelSource, Kernels};
use crate::init::InitKernels;
use crate::init::depth_prior::{DepthPrior, DepthPriorSchedule, SsiInputs};
use crate::io::images::{ImageCache, Rgb8Image};
use crate::io::scene::View;
use crate::kernels::adam::AdamIterPlan;
use crate::kernels::densify::{DensifyCx, DensifyOutcome};
use crate::kernels::layout::{FLAG_ABS_GRAD, FLAG_DENSIFY_STATS, OVF_CHAINS, OVF_INSTANCES};
use crate::kernels::loss::resample_target;
use crate::kernels::preprocess::PreprocessKernels;
use crate::kernels::raster::{FrameSource, GpuRaster, GradTargets, SplatRaster};
use crate::kernels::{CameraGrads, Capacity, GradSink, RenderMode};
use crate::model::GaussiansHost;
use crate::ppisp::{NovelView, Ppisp, PpispConfig};
use crate::trainer::{Trainer, planar_to_rgb8, select};

pub const SOURCE: KernelSource = KernelSource {
    module: "reconstruct",
    files: &[CudaFile {
        name: "reconstruct.cu",
        text: include_str!("../../cuda/reconstruct.cu"),
    }],
    kernels: &[
        "recon_pose_adam",
        "recon_focal_adam",
        "recon_depth_gather",
        "recon_depth_scatter",
    ],
};

const BETAS: (f32, f32) = (0.9, 0.999);
const EPS: f32 = 1e-15;

/// Joint camera refinement (spec-train 10.1). Iterations are absolute.
#[derive(Clone, Debug, PartialEq)]
pub struct PoseRefine {
    pub start: u32,
    pub end: u32,
    /// Rotation LR (radians per step, Adam).
    pub lr_rot: f32,
    /// Translation LR in units of the scene extent.
    pub lr_trans: f32,
    /// Log-focal LR; 0 keeps the focal fixed.
    pub lr_focal: f32,
}

/// Depth prior: weight `init` at `start`, linear to 0 at `end`.
#[derive(Clone, Debug, PartialEq)]
pub struct DepthPriorCfg {
    pub init: f32,
    pub end: u32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct JointConfig {
    pub pose: Option<PoseRefine>,
    pub ppisp: Option<PpispConfig>,
    pub depth: Option<DepthPriorCfg>,
}

/// Device priors of every training view at the engine's resolution.
pub struct DepthPriors {
    pub width: u32,
    pub height: u32,
    /// `[views][h][w]` engine depth (0 where unweighted) and 0/1 weights.
    pub prior: CudaSlice<f32>,
    pub weight: CudaSlice<f32>,
}

impl DepthPriors {
    pub fn upload(
        gpu: &Gpu,
        width: u32,
        height: u32,
        prior: &[f32],
        weight: &[f32],
    ) -> Result<Self> {
        Ok(Self {
            width,
            height,
            prior: gpu.stream.clone_htod(prior).cuda("upload depth priors")?,
            weight: gpu.stream.clone_htod(weight).cuda("upload depth weights")?,
        })
    }
}

struct DepthState {
    priors: DepthPriors,
    schedule: DepthPriorSchedule,
    fit: DepthPrior,
    k: InitKernels,
    small_d: CudaSlice<f32>,
    small_g: CudaSlice<f32>,
    /// Last SSI loss read (lagging; for logs).
    last_loss: Option<f32>,
}

/// Per-image twist and per-intrinsics log-focal optimizer state of one
/// camera table.
pub struct CamOpt {
    pub grads: CameraGrads,
    m: CudaSlice<f32>,
    v: CudaSlice<f32>,
    steps: Vec<u32>,
    intr0: CudaSlice<f32>,
    focal: CudaSlice<f32>,
    focal_steps: Vec<u32>,
}

impl CamOpt {
    pub fn new(gpu: &Gpu, cams: &CameraTable) -> Result<Self> {
        let s = &gpu.stream;
        let n = cams.n_images.max(1) as usize;
        let intr = s.clone_dtoh(&cams.intr).cuda("download intrinsics")?;
        let flat: Vec<f32> = intr.iter().flat_map(|i| [i.fx, i.fy, i.cx, i.cy]).collect();
        Ok(Self {
            grads: CameraGrads::alloc(
                gpu,
                cams.n_images,
                cams.n_intr,
                PreprocessKernels::cam_partial_blocks(gpu),
            )?,
            m: s.alloc_zeros(6 * n).cuda("alloc pose m")?,
            v: s.alloc_zeros(6 * n).cuda("alloc pose v")?,
            steps: vec![0; n],
            intr0: s.clone_htod(&flat).cuda("upload intr0")?,
            focal: s
                .alloc_zeros(3 * cams.n_intr.max(1) as usize)
                .cuda("alloc focal state")?,
            focal_steps: vec![0; cams.n_intr.max(1) as usize],
        })
    }

    /// One Adam step of image `image`'s twist (zeroes its gradient row).
    #[allow(clippy::too_many_arguments)]
    pub fn step_pose(
        &mut self,
        gpu: &Gpu,
        k: &Kernels,
        cams: &mut CameraTable,
        image: u32,
        lr_rot: f32,
        lr_trans: f32,
        frozen: bool,
    ) -> Result<()> {
        let steps = &mut self.steps[image as usize];
        if !frozen {
            *steps += 1;
        }
        let t = (*steps).max(1) as i32;
        let (b1, b2) = BETAS;
        let (bc1, bc2) = (1.0 - b1.powi(t), 1.0 - b2.powi(t));
        let fz = frozen as i32;
        let f = k.get("recon_pose_adam")?;
        let mut l = gpu.stream.launch_builder(&f);
        l.arg(&mut self.grads.pose)
            .arg(&mut cams.pose_delta)
            .arg(&mut self.m)
            .arg(&mut self.v)
            .arg(&image)
            .arg(&lr_rot)
            .arg(&lr_trans)
            .arg(&b1)
            .arg(&b2)
            .arg(&bc1)
            .arg(&bc2)
            .arg(&EPS)
            .arg(&fz);
        // SAFETY: recon_pose_adam(grad, delta, m, v [n][6], image < n, ...), 6 threads.
        unsafe { l.launch(one_block(32)) }.cuda("recon_pose_adam")?;
        Ok(())
    }

    /// One Adam step of intrinsics `id`'s log focal at render scale (sx, sy).
    #[allow(clippy::too_many_arguments)]
    pub fn step_focal(
        &mut self,
        gpu: &Gpu,
        k: &Kernels,
        cams: &mut CameraTable,
        id: u32,
        sx: f32,
        sy: f32,
        lr: f32,
    ) -> Result<()> {
        let frozen = lr == 0.0;
        let steps = &mut self.focal_steps[id as usize];
        if !frozen {
            *steps += 1;
        }
        let t = (*steps).max(1) as i32;
        let (b1, b2) = BETAS;
        let (bc1, bc2) = (1.0 - b1.powi(t), 1.0 - b2.powi(t));
        let fz = frozen as i32;
        let f = k.get("recon_focal_adam")?;
        let mut l = gpu.stream.launch_builder(&f);
        l.arg(&mut self.grads.intr)
            .arg(&mut cams.intr)
            .arg(&self.intr0)
            .arg(&mut self.focal)
            .arg(&id)
            .arg(&sx)
            .arg(&sy)
            .arg(&lr)
            .arg(&b1)
            .arg(&b2)
            .arg(&bc1)
            .arg(&bc2)
            .arg(&EPS)
            .arg(&fz);
        // SAFETY: recon_focal_adam(grad [n][4], intr [n] Intr = 4 f32, intr0 [n][4],
        // state [n][3], id < n, ...), one thread.
        unsafe { l.launch(one_block(32)) }.cuda("recon_focal_adam")?;
        Ok(())
    }
}

fn one_block(threads: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// What one joint step did (for logs).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct JointStep {
    pub t: u32,
    pub eager: bool,
    pub densify: Option<DensifyOutcome>,
}

pub struct Joint {
    pub tr: Trainer,
    pub cfg: JointConfig,
    k: Kernels,
    cam: CamOpt,
    ppisp: Option<(Ppisp, CudaSlice<f32>)>,
    depth: Option<DepthState>,
    /// Training view whose pose is the gauge.
    pub frozen: u32,
    max_size: (u32, u32),
    loss_window: u32,
    pub eager_steps: u32,
}

impl Joint {
    /// Wraps a trainer whose Gaussians are already initialized. `priors`
    /// are needed when `cfg.depth` is set.
    pub fn new(mut tr: Trainer, cfg: JointConfig, priors: Option<DepthPriors>) -> Result<Self> {
        let gpu = tr.gpu.clone();
        let k = gpu.kernels(&SOURCE, &Defines::new())?;
        let max_size = tr
            .scene
            .views
            .iter()
            .fold((1, 1), |(w, h), v| (w.max(v.width), h.max(v.height)));
        let max_px = max_size.0 as usize * max_size.1 as usize;
        let cam = CamOpt::new(&gpu, &tr.scene_gpu.cams)?;
        let ppisp = match &cfg.ppisp {
            Some(c) => {
                let n_cams = tr.scene_gpu.cams.n_intr as usize;
                let p = Ppisp::new(&gpu, n_cams, tr.scene.views.len(), max_px, c.clone())?;
                let buf = gpu
                    .stream
                    .alloc_zeros::<f32>(3 * max_px)
                    .cuda("alloc ppisp out")?;
                Some((p, buf))
            }
            None => None,
        };
        let depth = match (&cfg.depth, priors) {
            (Some(d), Some(priors)) => {
                let px = priors.width as usize * priors.height as usize;
                // The expected-depth channel the SSI loss reads and pushes into.
                let mut variant = tr.preset.variant;
                variant.extra_ch = 1;
                rebuild_raster(&mut tr, variant)?;
                Some(DepthState {
                    schedule: DepthPriorSchedule {
                        init: d.init,
                        final_: 0.0,
                        until: d.end,
                    },
                    fit: DepthPrior::alloc(&gpu)?,
                    k: InitKernels::load(&gpu)?,
                    small_d: gpu.stream.alloc_zeros(px).cuda("alloc depth small")?,
                    small_g: gpu.stream.alloc_zeros(px).cuda("alloc depth grad")?,
                    priors,
                    last_loss: None,
                })
            }
            (Some(_), None) => bail!("the depth prior needs priors"),
            _ => None,
        };
        let frozen = tr.scene.train.first().copied().unwrap_or(0) as u32;
        Ok(Self {
            tr,
            cfg,
            k,
            cam,
            ppisp,
            depth,
            frozen,
            max_size,
            loss_window: 0,
            eager_steps: 0,
        })
    }

    fn pose_active(&self, t: u32) -> bool {
        self.cfg
            .pose
            .as_ref()
            .is_some_and(|p| t >= p.start && t < p.end)
    }

    fn depth_active(&self, t: u32) -> bool {
        self.depth
            .as_ref()
            .is_some_and(|d| d.schedule.weight(t) > 0.0)
    }

    /// One iteration (eager with the Phase 4 terms, else the trainer's graph).
    pub fn step(&mut self) -> Result<JointStep> {
        let t = self.tr.t + 1;
        // Depth prior finished: drop the extra channel (and its cost).
        if self.depth.is_some() && !self.depth_active(t) {
            self.depth = None;
            let mut v = self.tr.preset.variant;
            v.extra_ch = 0;
            rebuild_raster(&mut self.tr, v)?;
        }
        if !self.pose_active(t) && self.ppisp.is_none() && self.depth.is_none() {
            let r = self.tr.step()?;
            return Ok(JointStep {
                t: r.t,
                eager: false,
                densify: r.densify,
            });
        }
        self.check_overflow()?;
        let tr = &mut self.tr;
        let view = tr.order.next_view();
        let res_scale = tr.schedule.res_scale(t);
        let sh_degree = tr.schedule.sh_degree(t);
        let plan = tr.adam.plan(t);
        let params = tr.iter_params(t, view, sh_degree, res_scale, plan);
        tr.ring.push(&params)?;
        self.record(t, view, &params)?;
        let tr = &mut self.tr;
        tr.ring.enqueued(&tr.gpu.stream)?;
        self.loss_window += 1;
        self.eager_steps += 1;
        let densify = self.hooks(t)?;
        self.tr.t = t;
        Ok(JointStep {
            t,
            eager: true,
            densify,
        })
    }

    fn record(&mut self, t: u32, view: usize, params: &IterParams) -> Result<()> {
        let pose_on = self.pose_active(t);
        let lambda = self.depth.as_ref().map_or(0.0, |d| d.schedule.weight(t));
        let (w, h) = (params.frame.render_w, params.frame.render_h);
        let base = &self.tr.scene.views[view];
        let (sx, sy) = (w as f32 / base.width as f32, h as f32 / base.height as f32);
        let intr_id = self.intr_of(view);
        let tr = &mut self.tr;
        let gpu = tr.gpu.clone();
        let raster = tr.raster.as_mut().expect("raster present");
        raster.forward(
            &gpu,
            &mut tr.g,
            &tr.scene_gpu.cams,
            FrameSource::Ring(&tr.ring),
            RenderMode::Train,
        )?;
        {
            let io = raster.loss_io();
            resample_target(
                &gpu,
                &tr.loss.kernels,
                io.params,
                &tr.scene_gpu.images,
                &mut tr.target,
            )?;
            match &mut self.ppisp {
                Some((p, out)) => {
                    p.forward(&gpu, intr_id as i32, view as i32, io.image, out, w, h)?;
                    tr.loss.forward_backward(
                        &gpu,
                        io.params,
                        out,
                        &tr.target,
                        &mut io.grads.d_rgb,
                    )?;
                    p.backward_inplace(
                        &gpu,
                        intr_id as i32,
                        view as i32,
                        io.image,
                        &mut io.grads.d_rgb,
                        w,
                        h,
                    )?;
                }
                None => tr.loss.forward_backward(
                    &gpu,
                    io.params,
                    io.image,
                    &tr.target,
                    &mut io.grads.d_rgb,
                )?,
            }
        }
        if let Some(d) = &mut self.depth {
            let s = &gpu.stream;
            let d_extra = raster
                .pixel_grads
                .d_extra
                .as_mut()
                .ok_or_else(|| spl_core::anyhow!("depth prior: raster has no depth channel"))?;
            s.memset_zeros(d_extra).cuda("zero d_extra")?;
            if lambda > 0.0 {
                let (pw, ph) = (d.priors.width as i32, d.priors.height as i32);
                let px = (pw * ph) as usize;
                let extra = raster
                    .out
                    .extra
                    .as_ref()
                    .ok_or_else(|| spl_core::anyhow!("depth prior: no depth output"))?;
                let (wi, hi) = (w as i32, h as i32);
                let f = self.k.get("recon_depth_gather")?;
                let mut l = s.launch_builder(&f);
                l.arg(extra)
                    .arg(&wi)
                    .arg(&hi)
                    .arg(&mut d.small_d)
                    .arg(&pw)
                    .arg(&ph);
                // SAFETY: depth [h][w] (render ≤ capacity pixels), out [ph][pw].
                unsafe { l.launch(LaunchConfig::for_num_elems(px as u32)) }
                    .cuda("recon_depth_gather")?;
                let o = view * px;
                let x = SsiInputs {
                    depth: d.small_d.slice(0..px),
                    prior: d.priors.prior.slice(o..o + px),
                    weight: d.priors.weight.slice(o..o + px),
                };
                d.fit.forward(&gpu, &d.k, &x)?;
                s.memset_zeros(&mut d.small_g).cuda("zero depth grad")?;
                let mut g = d.small_g.slice_mut(0..px);
                d.fit.backward(&gpu, &d.k, &x, lambda, &mut g)?;
                let d_extra = raster.pixel_grads.d_extra.as_mut().expect("checked above");
                let f = self.k.get("recon_depth_scatter")?;
                let mut l = s.launch_builder(&f);
                l.arg(&d.small_g)
                    .arg(&pw)
                    .arg(&ph)
                    .arg(d_extra)
                    .arg(&wi)
                    .arg(&hi);
                // SAFETY: g [ph][pw], grad [h][w].
                unsafe { l.launch(LaunchConfig::for_num_elems(px as u32)) }
                    .cuda("recon_depth_scatter")?;
            }
        }
        let want_stats = tr.strategy.wants_stats(t);
        raster.backward(
            &gpu,
            &mut tr.g,
            GradTargets {
                sink: GradSink::Write(&mut tr.grads),
                camera: pose_on.then_some(&mut self.cam.grads),
                stats: want_stats.then_some(&mut tr.stats),
            },
        )?;
        tr.adam
            .step(&gpu, &mut tr.g, &mut tr.grads, raster.params())?;
        if let Some(p) = self.cfg.pose.as_ref().filter(|_| pose_on) {
            let extent = tr.scene.extent;
            self.cam.step_pose(
                &gpu,
                &self.k,
                &mut tr.scene_gpu.cams,
                view as u32,
                p.lr_rot,
                p.lr_trans * extent,
                view as u32 == self.frozen,
            )?;
            self.cam.step_focal(
                &gpu,
                &self.k,
                &mut tr.scene_gpu.cams,
                intr_id,
                sx,
                sy,
                p.lr_focal,
            )?;
        }
        if let Some((p, _)) = &mut self.ppisp {
            p.regularize(&gpu)?;
            p.step(&gpu)?;
        }
        raster.end_frame(&gpu, &tr.status)
    }

    fn intr_of(&self, view: usize) -> u32 {
        let v = &self.tr.scene.views[view];
        // CameraTable::pack: one entry per distinct (camera, size), first seen.
        let mut keys: Vec<(u32, u32, u32)> = Vec::new();
        for u in &self.tr.scene.views {
            let k = (u.camera, u.width, u.height);
            if !keys.contains(&k) {
                keys.push(k);
            }
        }
        keys.iter()
            .position(|k| *k == (v.camera, v.width, v.height))
            .unwrap_or(0) as u32
    }

    /// `Trainer::hooks` (densify, prune, Morton) for an eager iteration.
    fn hooks(&mut self, t: u32) -> Result<Option<DensifyOutcome>> {
        let tr = &mut self.tr;
        let wants_views = tr.strategy.wants_views(t);
        let morton = tr.schedule.morton_due(t);
        let views: Vec<IterParams> = if wants_views {
            let (deg, res) = (tr.schedule.sh_degree(t), tr.schedule.res_scale(t));
            (0..tr.scene.views.len())
                .map(|v| tr.iter_params(t, v, deg, res, AdamIterPlan::default()))
                .collect()
        } else {
            Vec::new()
        };
        let res_scale = tr.schedule.res_scale(t);
        let raster = tr.raster.as_mut().expect("raster present");
        let mut cx = DensifyCx {
            gpu: &tr.gpu,
            g: &mut tr.g,
            grads: &mut tr.grads,
            adam: &mut tr.adam,
            stats: &mut tr.stats,
            raster,
            loss: &mut tr.loss,
            scene: &tr.scene_gpu,
            extent: tr.scene.extent,
            res_scale,
            rng: tr.rng,
            views: &views,
        };
        let outcome = tr.strategy.after_step(&mut cx, t)?;
        if morton {
            tr.strategy.morton_sort(&mut cx, t)?;
        }
        let densify = outcome.changed.then_some(outcome);
        tr.summary.peak_gaussians = tr.summary.peak_gaussians.max(tr.g.n);
        if let Some(d) = densify {
            tr.summary.densify_steps += 1;
            tr.summary.capped += d.capped;
        }
        if densify.is_some() || morton {
            tr.graphs.clear();
        }
        Ok(densify)
    }

    /// `Trainer::check_overflow` for eager iterations: grow the raster.
    fn check_overflow(&mut self) -> Result<()> {
        if self.tr.status.read()?.overflow == 0 {
            return Ok(());
        }
        self.tr.gpu.sync()?;
        let st = self
            .tr
            .raster
            .as_ref()
            .expect("raster present")
            .frame
            .iter
            .read_status(&self.tr.gpu)?;
        self.tr.status.clear();
        if st.overflow & (OVF_INSTANCES | OVF_CHAINS) != 0 {
            self.regrow(st.n_instances)?;
        } else if st.overflow != 0 {
            bail!("raster overflow {:#x}", st.overflow);
        }
        Ok(())
    }

    fn regrow(&mut self, need: u32) -> Result<()> {
        let tr = &mut self.tr;
        tr.gpu.sync()?;
        let old = tr.raster.as_ref().expect("raster present").capacity();
        let instances = (need as u64 * 5 / 4)
            .max(old.instances as u64 * 5 / 4)
            .min(u32::MAX as u64 / 2) as u32;
        tracing::info!("raster regrow: instances {} -> {instances}", old.instances);
        tr.graphs.clear();
        tr.raster = None;
        let cap = Capacity::new(
            old.gaussians,
            instances,
            self.max_size.0,
            self.max_size.1,
            &tr.preset.variant,
        );
        let mut r = GpuRaster::new(
            &tr.gpu,
            cap,
            tr.preset.variant,
            tr.strategy.wants_abs_grad(),
        )?;
        r.set_iteration_counter(&tr.gpu, tr.ring.pushed as u32)?;
        tr.raster = Some(r);
        tr.summary.raster_regrows += 1;
        Ok(())
    }

    /// Mean training loss since the last read (host sync).
    pub fn read_loss(&mut self) -> Result<Option<f64>> {
        let own = if self.loss_window > 0 {
            let v = self.tr.loss.take_sums(&self.tr.gpu, self.loss_window)?;
            self.loss_window = 0;
            Some(v.loss)
        } else {
            None
        };
        if let Some(d) = &mut self.depth {
            d.last_loss = Some(d.fit.read_fit(&self.tr.gpu)?.loss);
        }
        Ok(self.tr.read_loss()?.or(own))
    }

    pub fn depth_loss(&self) -> Option<f32> {
        self.depth.as_ref().and_then(|d| d.last_loss)
    }

    /// Refined world-to-camera poses and intrinsics of the training views.
    pub fn refined_views(&self) -> Result<Vec<View>> {
        refined_views(&self.tr.gpu, &self.tr.scene_gpu.cams, &self.tr.scene.views)
    }

    /// LazyCatchUp flush and the final opacity prune (like `Trainer::finish`,
    /// without consuming the trainer: evaluation still renders afterwards).
    pub fn export_cloud(&mut self) -> Result<GaussiansHost> {
        let tr = &mut self.tr;
        tr.adam.flush(&tr.gpu, &mut tr.g, tr.t)?;
        let mut host = tr.g.download(&tr.gpu)?;
        host.sh_degree = tr.schedule.sh_degree(tr.t.max(1));
        let keep: Vec<usize> = (0..host.len())
            .filter(|&i| 1.0 / (1.0 + (-host.opacities[i]).exp()) >= tr.preset.final_prune_opacity)
            .collect();
        let host = select(&host, &keep);
        tr.summary.final_gaussians = host.len() as u32;
        tr.summary.iterations = tr.t;
        Ok(host)
    }

    /// IterParams rendering `cam_index` of an arbitrary camera table at
    /// `w × h` with target `entry` (eval, test-time refinement).
    fn params_for(
        &self,
        cam_index: usize,
        w: u32,
        h: u32,
        target: Option<(u64, u32, u32)>,
    ) -> IterParams {
        let tr = &self.tr;
        let t = tr.t.max(1);
        let mut p = tr.iter_params(t, 0, tr.schedule.sh_degree(t), 1, AdamIterPlan::default());
        p.frame.flags &= !(FLAG_DENSIFY_STATS | FLAG_ABS_GRAD);
        p.frame.cam_index = cam_index as u32;
        p.frame.render_w = w;
        p.frame.render_h = h;
        p.res_scale = 1;
        if let Some((off, tw, th)) = target {
            p.target_offset = off;
            p.target_w = tw;
            p.target_h = th;
        }
        p
    }

    /// 8-bit render of `cam_index` of `cams` (full resolution, Infer mode;
    /// PPISP's per-camera response with an identity frame correction).
    pub fn render(
        &mut self,
        cams: &CameraTable,
        cam_index: usize,
        w: u32,
        h: u32,
    ) -> Result<Rgb8Image> {
        let params = self.params_for(cam_index, w, h, None);
        let gpu = self.tr.gpu.clone();
        for attempt in 0..2 {
            let raster = self.tr.raster.as_mut().expect("raster present");
            raster.forward(
                &gpu,
                &mut self.tr.g,
                cams,
                FrameSource::Eager(&params),
                RenderMode::Infer,
            )?;
            let st = raster.frame.iter.read_status(&gpu)?;
            if st.overflow != 0 && attempt == 0 {
                self.tr.status.clear();
                self.regrow(st.n_instances)?;
                continue;
            }
            break;
        }
        let px = (w * h) as usize;
        let raster = self.tr.raster.as_mut().expect("raster present");
        let planar = match &mut self.ppisp {
            Some((p, out)) => {
                p.forward_novel(&gpu, 0, NovelView::Identity, &raster.out.rgb, out, w, h)?;
                gpu.stream
                    .clone_dtoh(&out.slice(0..3 * px))
                    .cuda("download render")?
            }
            None => gpu
                .stream
                .clone_dtoh(&raster.out.rgb.slice(0..3 * px))
                .cuda("download render")?,
        };
        Ok(planar_to_rgb8(&planar, w, h))
    }

    /// Test-time pose refinement of every held-out view: `iters` steps each
    /// on frozen Gaussians, pose gradients only.
    pub fn refine_test_poses(
        &mut self,
        tv: &mut TestViews,
        iters: u32,
        lr_rot: f32,
        lr_trans: f32,
    ) -> Result<()> {
        let gpu = self.tr.gpu.clone();
        let extent = self.tr.scene.extent;
        for i in 0..tv.views.len() {
            let e = tv.images.entries[i];
            let v = &tv.views[i];
            let params = self.params_for(i, v.width, v.height, Some((e.offset, e.width, e.height)));
            for it in 0..iters {
                // Exponential decay to 1 % over the run.
                let decay = 0.01f32.powf(it as f32 / iters.max(1) as f32);
                let tr = &mut self.tr;
                let raster = tr.raster.as_mut().expect("raster present");
                raster.forward(
                    &gpu,
                    &mut tr.g,
                    &tv.cams,
                    FrameSource::Eager(&params),
                    RenderMode::Train,
                )?;
                let io = raster.loss_io();
                resample_target(
                    &gpu,
                    &tr.loss.kernels,
                    io.params,
                    &tv.images,
                    &mut tr.target,
                )?;
                tr.loss.forward_backward(
                    &gpu,
                    io.params,
                    io.image,
                    &tr.target,
                    &mut io.grads.d_rgb,
                )?;
                if let Some(d) = raster.pixel_grads.d_extra.as_mut() {
                    gpu.stream.memset_zeros(d).cuda("zero d_extra")?;
                }
                raster.backward(
                    &gpu,
                    &mut tr.g,
                    GradTargets {
                        sink: GradSink::Write(&mut tr.grads),
                        camera: Some(&mut tv.opt.grads),
                        stats: None,
                    },
                )?;
                tv.opt.step_pose(
                    &gpu,
                    &self.k,
                    &mut tv.cams,
                    i as u32,
                    lr_rot * decay,
                    lr_trans * extent * decay,
                    false,
                )?;
                // Intrinsics stay shared with training: drop their gradient.
                gpu.stream
                    .memset_zeros(&mut tv.opt.grads.intr)
                    .cuda("zero intr grads")?;
            }
            let st = self
                .tr
                .raster
                .as_ref()
                .expect("raster present")
                .frame
                .iter
                .read_status(&gpu)?;
            if st.overflow != 0 {
                self.tr.status.clear();
                self.regrow(st.n_instances)?;
            }
        }
        self.tr.loss.take_sums(&gpu, 1).map(drop)
    }
}

/// Swaps the raster for one of `variant` with the same capacity.
fn rebuild_raster(tr: &mut Trainer, variant: crate::kernels::RasterVariant) -> Result<()> {
    tr.gpu.sync()?;
    let old = tr.raster.as_ref().expect("raster present").capacity();
    tr.graphs.clear();
    tr.raster = None;
    let max_size = tr
        .scene
        .views
        .iter()
        .fold((1, 1), |(w, h), v| (w.max(v.width), h.max(v.height)));
    let cap = Capacity::new(
        old.gaussians,
        old.instances,
        max_size.0,
        max_size.1,
        &variant,
    );
    let mut r = GpuRaster::new(&tr.gpu, cap, variant, tr.strategy.wants_abs_grad())?;
    r.set_iteration_counter(&tr.gpu, tr.ring.pushed as u32)?;
    tr.raster = Some(r);
    tr.preset.variant = variant;
    Ok(())
}

/// `views` with the device pose deltas and intrinsics folded in.
pub fn refined_views(gpu: &Gpu, cams: &CameraTable, views: &[View]) -> Result<Vec<View>> {
    let delta = gpu
        .stream
        .clone_dtoh(&cams.pose_delta)
        .cuda("download pose deltas")?;
    let intr = gpu
        .stream
        .clone_dtoh(&cams.intr)
        .cuda("download intrinsics")?;
    let (_, ids) = intr_ids(views);
    Ok(views
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let (rd, td) = se3_exp(&delta[6 * i..6 * i + 6]);
            let mut out = v.clone();
            for r in 0..3 {
                for c in 0..3 {
                    out.r_cw[r][c] = (0..3).map(|k| rd[r][k] * v.r_cw[k][c]).sum();
                }
                out.t_cw[r] = (0..3).map(|k| rd[r][k] * v.t_cw[k]).sum::<f64>() + td[r];
            }
            let k = intr[ids[i]];
            out.fx = k.fx as f64;
            out.fy = k.fy as f64;
            out
        })
        .collect())
}

/// Intrinsics entry of each view, as `CameraTable::pack` assigns them.
fn intr_ids(views: &[View]) -> (usize, Vec<usize>) {
    let mut keys: Vec<(u32, u32, u32)> = Vec::new();
    let ids = views
        .iter()
        .map(|v| {
            let k = (v.camera, v.width, v.height);
            match keys.iter().position(|x| *x == k) {
                Some(i) => i,
                None => {
                    keys.push(k);
                    keys.len() - 1
                }
            }
        })
        .collect();
    (keys.len(), ids)
}

/// SE(3) exponential of (ρ, φ) like `se3_exp` in `preprocess.cu`: R = exp(φ^),
/// t = V ρ.
pub fn se3_exp(xi: &[f32]) -> ([[f64; 3]; 3], [f64; 3]) {
    let rho = [xi[0] as f64, xi[1] as f64, xi[2] as f64];
    let phi = [xi[3] as f64, xi[4] as f64, xi[5] as f64];
    let th2 = phi.iter().map(|x| x * x).sum::<f64>();
    let th = th2.sqrt();
    let (a, b, c) = if th < 1e-8 {
        (1.0, 0.5, 1.0 / 6.0)
    } else {
        (
            th.sin() / th,
            (1.0 - th.cos()) / th2,
            (th - th.sin()) / (th2 * th),
        )
    };
    let k = [
        [0.0, -phi[2], phi[1]],
        [phi[2], 0.0, -phi[0]],
        [-phi[1], phi[0], 0.0],
    ];
    let mut k2 = [[0.0; 3]; 3];
    for r in 0..3 {
        for col in 0..3 {
            k2[r][col] = (0..3).map(|m| k[r][m] * k[m][col]).sum();
        }
    }
    let id = |r: usize, col: usize| if r == col { 1.0 } else { 0.0 };
    let mut rm = [[0.0; 3]; 3];
    let mut v = [[0.0; 3]; 3];
    for r in 0..3 {
        for col in 0..3 {
            rm[r][col] = id(r, col) + a * k[r][col] + b * k2[r][col];
            v[r][col] = id(r, col) + b * k[r][col] + c * k2[r][col];
        }
    }
    let t = std::array::from_fn(|r| (0..3).map(|m| v[r][m] * rho[m]).sum());
    (rm, t)
}

/// Held-out views on the device: their own camera table (poses from the
/// engine, intrinsics shared with training) and images.
pub struct TestViews {
    pub views: Vec<View>,
    pub cams: CameraTable,
    pub images: ImageCache,
    pub opt: CamOpt,
}

impl TestViews {
    pub fn new(gpu: &Gpu, views: Vec<View>, images: &[Rgb8Image]) -> Result<Self> {
        let cams = CameraTable::upload(gpu, &views)?;
        let images = ImageCache::from_images(gpu, (0..views.len()).collect(), images, 0.0)?;
        let opt = CamOpt::new(gpu, &cams)?;
        Ok(Self {
            views,
            cams,
            images,
            opt,
        })
    }

    /// Copies the training run's refined intrinsics (one shared camera).
    pub fn share_intrinsics(&mut self, gpu: &Gpu, train: &CameraTable) -> Result<()> {
        let intr = gpu
            .stream
            .clone_dtoh(&train.intr)
            .cuda("download intrinsics")?;
        let mine = gpu
            .stream
            .clone_dtoh(&self.cams.intr)
            .cuda("download test intrinsics")?;
        let shared: Vec<_> = mine.iter().map(|_| intr[0]).collect();
        gpu.stream
            .memcpy_htod(&shared, &mut self.cams.intr)
            .cuda("upload test intrinsics")?;
        for (v, k) in self.views.iter_mut().zip(&shared) {
            v.fx = k.fx as f64;
            v.fy = k.fy as f64;
        }
        Ok(())
    }

    pub fn refined(&self, gpu: &Gpu) -> Result<Vec<View>> {
        refined_views(gpu, &self.cams, &self.views)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn se3_exp_small_and_finite_angles() {
        let (r, t) = se3_exp(&[0.1, -0.2, 0.3, 0.0, 0.0, 0.0]);
        assert_eq!(r, [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]);
        assert!((t[0] - 0.1).abs() < 1e-7 && (t[2] - 0.3).abs() < 1e-7);
        let a = std::f32::consts::FRAC_PI_2;
        let (r, _) = se3_exp(&[0.0, 0.0, 0.0, 0.0, 0.0, a]);
        // Rotation about z by 90°: x → y.
        assert!((r[1][0] - 1.0).abs() < 1e-6 && (r[0][1] + 1.0).abs() < 1e-6);
        let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
            - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
        assert!((det - 1.0).abs() < 1e-9);
    }

    #[test]
    fn intr_ids_follow_pack_order() {
        let v = |camera, width| View {
            name: String::new(),
            camera,
            r_cw: [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
            t_cw: [0.; 3],
            fx: 1.,
            fy: 1.,
            cx: 0.,
            cy: 0.,
            width,
            height: 10,
            image_path: Default::default(),
        };
        let views = [v(0, 10), v(1, 10), v(0, 10), v(0, 20)];
        let (n, ids) = intr_ids(&views);
        assert_eq!(n, 3);
        assert_eq!(ids, vec![0, 1, 0, 2]);
        let (base, intr) = CameraTable::pack(&views);
        assert_eq!(intr.len(), n);
        assert!(base.iter().zip(&ids).all(|(b, &i)| b.intr_id as usize == i));
    }
}
