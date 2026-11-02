//! The blocking reconstruction pipeline (Report section 3, Phase 4):
//! keyframes → VGGT-Ω (poses, intrinsics, depth, confidence) → Stage B init
//! and prune-first → joint splat optimization → exports. Intermediate
//! results go out as [`Event`]s: the cameras and the point cloud as soon as the
//! engine finishes, splat snapshots on a cadence. Owner: mvp-pipeline.
//!
//! GPU memory: the engine is dropped before the trainer allocates (the
//! trainer sizes its fixed capacity from free VRAM), unless the caller keeps
//! it ([`PipelineConfig::keep_engine`]).

use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use anyhow::ensure;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use spl_core::{CancellationToken, Context, Result, bail};
use ts_rs::TS;

use super::ba::{BaConfig, bundle_adjust};
use super::geom::{GeoRefine, consistent_points, refine_poses};
use super::joint::{DepthPriorCfg, DepthPriors, Joint, JointConfig, PoseRefine, TestViews};
use super::prune::{PruneReport, prune_first};
use super::scene::{Posed, holdout_split, spread_stats, subsample, training_scene, voxel_merge};
use crate::eval::{ImageMetrics, image_metrics};
use crate::export::colmap::{write_colmap, write_transforms_json};
use crate::export::ply::write_inria_ply;
use crate::export::spz::{SpzOptions, write_spz};
use crate::gpu::Gpu;
use crate::init::{DepthFrames, InitKernels, InitOptions, init_from_depth};
use crate::io::images::{ImageCache, Rgb8Image};
use crate::io::scene::{View, extent};
use crate::model::GaussiansHost;
use crate::ppisp::PpispConfig;
use crate::presets::{DensifyPreset, PresetName, PresetOverrides};
use crate::trainer::{TrainConfig, Trainer, ViewOrder};
use crate::vggt::{VggtConfig, VggtEngine};

/// User-facing speed/quality trade-off.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export, export_to = "image-to-3d/")]
pub enum ReconPreset {
    /// Target: ≥ 26.5 dB on Mip-NeRF 360 in ≤ 90 s end to end.
    Fast,
    /// Target: ≥ 27.4 dB on Mip-NeRF 360 in ≤ 5 min end to end.
    #[default]
    Quality,
}

/// Every knob of a run; [`PipelineConfig::preset`] fills the defaults.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct PipelineConfig {
    pub preset: ReconPreset,
    /// Training iterations (the preset's when absent).
    #[ts(optional)]
    pub iterations: Option<u32>,
    #[ts(type = "number")]
    pub seed: u64,
    /// Gaussians Stage B creates.
    pub init_target: u32,
    /// Above 0: skip the voxel init and seed this many engine points with
    /// the trainer's kNN init instead (the COLMAP-style sparse start).
    pub init_points: u32,
    /// Fraction of low-confidence pixels Stage B drops (0-1).
    pub init_conf_drop: f32,
    /// Prune-first pass: views, error threshold, fraction removed (0 = off).
    pub prune_views: u32,
    pub prune_tau: f32,
    pub prune_fraction: f32,
    /// Joint pose/focal refinement; LRs are Adam step sizes (rotation in
    /// radians, translation in scene extents, log focal).
    pub pose_refine: bool,
    pub pose_start: u32,
    /// Absolute end; the densification end when absent.
    #[ts(optional)]
    pub pose_end: Option<u32>,
    pub pose_lr_rot: f32,
    pub pose_lr_trans: f32,
    pub focal_lr: f32,
    /// Pose and focal LRs decay exponentially to this factor at the end.
    pub pose_lr_final: f32,
    pub ppisp: bool,
    /// Above 0: a pose bootstrap of this many iterations (sparse init, poses
    /// only kept) before the real run, whose init then uses the engine depth
    /// re-unprojected with the refined cameras.
    pub bootstrap_iters: u32,
    /// Bootstrap pose LR (rotation; translation the same × extent).
    pub bootstrap_lr: f32,
    /// The bootstrap trains at 1/this resolution.
    #[serde(default = "default_one")]
    pub bootstrap_scale: u32,
    /// Depth prior weight at iteration 0 (decays linearly to 0); 0 = off.
    pub depth_weight: f32,
    /// Fraction of the run over which the depth weight decays.
    pub depth_until: f32,
    /// Fraction of lowest-confidence engine pixels the depth prior ignores.
    pub depth_conf_drop: f32,
    /// Live snapshot cadence in seconds (0 = none).
    pub snapshot_every_s: f32,
    /// Rescale the world so training cameras sit this far (mean) from their
    /// centroid; 0 keeps the engine's units.
    pub world_spread: f32,
    /// Frames per engine pass; more are registered in chunks (12 GB fits
    /// about 200 frames at 512×336 when idle).
    pub engine_max_frames: u32,
    /// Keep the engine resident after posing (faster next run, less VRAM).
    pub keep_engine: bool,
    /// Trainer overrides (ablations).
    #[serde(default)]
    pub overrides: PresetOverrides,
    /// Geometric pose refinement before training ([`geom::refine_poses`]):
    /// Jacobi iterations (0 = off), samples per frame, per-frame depth scale.
    #[serde(default)]
    pub geo_iters: u32,
    #[serde(default = "default_geo_samples")]
    pub geo_samples: u32,
    #[serde(default = "default_true")]
    pub geo_scale: bool,
    /// Above 0: init points must agree with this many neighbour depth maps
    /// ([`geom::consistent_points`]); 0 = every confident pixel.
    #[serde(default)]
    pub init_min_views: u32,
    /// Relative depth tolerance of that check.
    #[serde(default = "default_init_tau")]
    pub init_tau: f32,
    /// Above 0: engine points are merged per voxel of this fraction of the
    /// camera extent (coarser with distance, [`voxel_merge`]) before the
    /// `init_points` cap; `init_oversample` × `init_points` candidates.
    #[serde(default)]
    pub init_voxel: f32,
    #[serde(default = "default_oversample")]
    pub init_oversample: u32,
    /// Bundle adjustment on KLT tracks before training ([`ba::bundle_adjust`]):
    /// LM iterations (0 = off), keypoints per frame, neighbours per keypoint.
    #[serde(default)]
    pub ba_iters: u32,
    #[serde(default = "default_ba_keypoints")]
    pub ba_keypoints: u32,
    #[serde(default = "default_ba_neighbours")]
    pub ba_neighbours: u32,
    /// Also refine the shared focal in the BA.
    #[serde(default = "default_true")]
    pub ba_focal: bool,
    /// Above 0: a second BA round without observations off by more (half-res px).
    #[serde(default)]
    pub ba_reject_px: f32,
    /// One focal per image instead of one per camera.
    #[serde(default)]
    pub per_image_focal: bool,
}

fn default_ba_keypoints() -> u32 {
    1024
}

fn default_ba_neighbours() -> u32 {
    6
}

fn default_one() -> u32 {
    1
}

fn default_oversample() -> u32 {
    8
}

fn default_geo_samples() -> u32 {
    2000
}

fn default_true() -> bool {
    true
}

fn default_init_tau() -> f32 {
    0.01
}

impl PipelineConfig {
    /// Defaults measured on Mip-NeRF 360 garden (Fast): sparse kNN init from
    /// engine points beats the voxel init by 0.25 dB; per-image pose Adam at
    /// 1e-3 over the whole run (decay to 0.1x) beats 1e-4 by 1.5 dB; the
    /// depth prior is neutral (kept at the design's 0.1 → 0 over 30 %).
    pub fn preset(p: ReconPreset) -> Self {
        let base = Self {
            preset: p,
            iterations: None,
            seed: 0,
            init_target: 600_000,
            init_points: 150_000,
            init_conf_drop: 0.2,
            prune_views: 10,
            prune_tau: 0.5,
            prune_fraction: 0.0,
            pose_refine: true,
            pose_start: 0,
            pose_end: Some(u32::MAX),
            pose_lr_rot: 1e-3,
            pose_lr_trans: 1e-3,
            focal_lr: 1e-4,
            pose_lr_final: 0.1,
            ppisp: false,
            bootstrap_iters: 0,
            bootstrap_lr: 3e-3,
            bootstrap_scale: 1,
            depth_weight: 0.1,
            depth_until: 0.3,
            depth_conf_drop: 0.3,
            snapshot_every_s: 8.0,
            world_spread: 0.0,
            engine_max_frames: 200,
            keep_engine: false,
            overrides: PresetOverrides::default(),
            geo_iters: 0,
            geo_samples: default_geo_samples(),
            geo_scale: true,
            init_min_views: 0,
            init_tau: default_init_tau(),
            init_voxel: 0.0,
            init_oversample: default_oversample(),
            per_image_focal: false,
            ba_iters: 0,
            ba_keypoints: default_ba_keypoints(),
            ba_neighbours: default_ba_neighbours(),
            ba_focal: true,
            ba_reject_px: 0.0,
        };
        match p {
            // Pose bootstrap: +0.4 dB on garden for 17 s.
            ReconPreset::Quality => Self {
                bootstrap_iters: 5000,
                ..base
            },
            ReconPreset::Fast => base,
        }
    }

    pub fn train_config(&self) -> TrainConfig {
        TrainConfig {
            preset: match self.preset {
                ReconPreset::Fast => PresetName::Fast,
                ReconPreset::Quality => PresetName::Quality,
            },
            iterations: self.iterations,
            seed: self.seed,
            overrides: self.overrides.clone(),
            ..Default::default()
        }
    }
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self::preset(ReconPreset::Quality)
    }
}

/// Frames at both resolutions, in capture (or name) order.
pub struct Frames {
    pub names: Vec<String>,
    /// Engine input (any size; the engine resizes to 512 long side).
    pub engine: Vec<Rgb8Image>,
    /// Trainer resolution; all the same size.
    pub trainer: Vec<Rgb8Image>,
}

/// Intermediate results for the caller (artifacts, progress).
pub enum Event {
    Progress(f64, String),
    Log(String),
    /// `cameras.json` of the engine's poses (all frames).
    Cameras(Vec<u8>),
    /// INRIA PLY of the engine's point cloud.
    Points(Vec<u8>, u32),
    /// SPZ snapshot at iteration `t` with `n` Gaussians.
    Snapshot {
        spz: Vec<u8>,
        t: u32,
        n: u32,
    },
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct StageTimes {
    pub engine_load_s: f64,
    pub engine_s: f64,
    pub engine_forward_ms: f32,
    pub setup_s: f64,
    pub init_s: f64,
    pub prune_s: f64,
    #[serde(default)]
    pub bootstrap_s: f64,
    #[serde(default)]
    pub geo_s: f64,
    #[serde(default)]
    pub ba_s: f64,
    pub train_s: f64,
    pub export_s: f64,
    /// From the call to the exports being encoded.
    pub total_s: f64,
}

pub struct Reconstruction {
    pub cloud: GaussiansHost,
    pub spz: Vec<u8>,
    /// Refined training cameras.
    pub views: Vec<View>,
    pub times: StageTimes,
    pub prune: PruneReport,
    pub init_gaussians: u32,
    pub iterations: u32,
    pub eager_steps: u32,
    pub peak_vram_bytes: u64,
    /// Kept alive for evaluation; `None` after [`Reconstruction::release`].
    pub joint: Option<Joint>,
    /// Held-out frames (benchmark protocol), engine-posed.
    pub test: Option<TestViews>,
    pub test_images: Vec<Rgb8Image>,
}

impl Reconstruction {
    /// Every format next to each other: `splat.spz`, `splat.ply`,
    /// `cameras.json`, `transforms.json`, `colmap/sparse/0/*.txt`.
    pub fn write_exports(&self, dir: &std::path::Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join("splat.spz"), &self.spz)?;
        let mut ply = Vec::new();
        write_inria_ply(&self.cloud, &mut ply)?;
        std::fs::write(dir.join("splat.ply"), ply)?;
        std::fs::write(dir.join("cameras.json"), cameras_json(&self.views)?)?;
        let mut tj = Vec::new();
        write_transforms_json(&self.views, &mut tj)?;
        std::fs::write(dir.join("transforms.json"), tj)?;
        let sparse = dir.join("colmap").join("sparse").join("0");
        std::fs::create_dir_all(&sparse)?;
        write_colmap(&self.views, &self.cloud, &sparse, true)
    }

    /// Mean metrics of the held-out views at their current poses; renders
    /// go to `out_dir/<name>.png` when given.
    pub fn evaluate_test(
        &mut self,
        out_dir: Option<&std::path::Path>,
    ) -> Result<(ImageMetrics, Vec<ImageMetrics>)> {
        let (Some(joint), Some(tv)) = (self.joint.as_mut(), self.test.as_ref()) else {
            bail!("no held-out views");
        };
        if let Some(d) = out_dir {
            std::fs::create_dir_all(d)?;
        }
        let mut per = Vec::new();
        for (i, v) in tv.views.iter().enumerate() {
            let img = joint.render(&tv.cams, i, v.width, v.height)?;
            if let Some(d) = out_dir {
                let stem = std::path::Path::new(&v.name)
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| format!("{i:05}"));
                img.save_png(&d.join(format!("{stem}.png")))?;
            }
            per.push(image_metrics(&img, &self.test_images[i])?);
        }
        let n = per.len().max(1) as f64;
        let mean = ImageMetrics {
            psnr: per.iter().map(|m| m.psnr).sum::<f64>() / n,
            ssim: per.iter().map(|m| m.ssim).sum::<f64>() / n,
            ssim_valid: per.iter().map(|m| m.ssim_valid).sum::<f64>() / n,
            lpips_vgg: None,
            lpips_vgg_inria: None,
        };
        Ok((mean, per))
    }

    /// Mean metrics of every `every`-th training view (fit diagnostic).
    pub fn evaluate_train(&mut self, every: usize) -> Result<ImageMetrics> {
        let Some(joint) = self.joint.as_mut() else {
            bail!("released");
        };
        let views: Vec<(usize, u32, u32)> = joint
            .tr
            .scene
            .views
            .iter()
            .enumerate()
            .step_by(every.max(1))
            .map(|(i, v)| (i, v.width, v.height))
            .collect();
        let mut per = Vec::new();
        for (i, w, h) in views {
            let cams = std::mem::replace(
                &mut joint.tr.scene_gpu.cams,
                crate::frame::CameraTable::upload(&joint.tr.gpu, &[])?,
            );
            let img = joint.render(&cams, i, w, h);
            joint.tr.scene_gpu.cams = cams;
            let gt = joint.tr.scene_gpu.images.download(&joint.tr.gpu, i)?;
            per.push(image_metrics(&img?, &gt)?);
        }
        let n = per.len().max(1) as f64;
        Ok(ImageMetrics {
            psnr: per.iter().map(|m| m.psnr).sum::<f64>() / n,
            ssim: per.iter().map(|m| m.ssim).sum::<f64>() / n,
            ssim_valid: per.iter().map(|m| m.ssim_valid).sum::<f64>() / n,
            lpips_vgg: None,
            lpips_vgg_inria: None,
        })
    }

    /// Test-time pose refinement of the held-out views (frozen Gaussians).
    pub fn refine_test(&mut self, iters: u32, lr_rot: f32, lr_trans: f32) -> Result<f64> {
        self.refine_test_at(1, iters, lr_rot, lr_trans)
    }

    /// [`Self::refine_test`] at `1/scale` render resolution.
    pub fn refine_test_at(
        &mut self,
        scale: u32,
        iters: u32,
        lr_rot: f32,
        lr_trans: f32,
    ) -> Result<f64> {
        let (Some(joint), Some(tv)) = (self.joint.as_mut(), self.test.as_mut()) else {
            bail!("no held-out views");
        };
        let t = Instant::now();
        joint.refine_test_poses_at(tv, scale, iters, lr_rot, lr_trans)?;
        joint.tr.gpu.sync()?;
        Ok(t.elapsed().as_secs_f64())
    }

    pub fn release(&mut self) {
        self.test = None;
        self.joint = None;
    }
}

/// Camera export: the refined training views, OpenCV world-to-camera.
pub fn cameras_json(views: &[View]) -> Result<Vec<u8>> {
    let frames: Vec<_> = views
        .iter()
        .map(|v| {
            serde_json::json!({
                "name": v.name,
                "width": v.width,
                "height": v.height,
                "fx": v.fx, "fy": v.fy, "cx": v.cx, "cy": v.cy,
                "w2c": [
                    [v.r_cw[0][0], v.r_cw[0][1], v.r_cw[0][2], v.t_cw[0]],
                    [v.r_cw[1][0], v.r_cw[1][1], v.r_cw[1][2], v.t_cw[1]],
                    [v.r_cw[2][0], v.r_cw[2][1], v.r_cw[2][2], v.t_cw[2]],
                ],
            })
        })
        .collect();
    Ok(serde_json::to_vec_pretty(&serde_json::json!({
        "convention": "opencv w2c (x right, y down, z forward); world = first frame of the engine; model-scale units",
        "frames": frames,
    }))?)
}

type EngineSlot = Mutex<Option<(PathBuf, VggtEngine)>>;

fn engine_slot() -> &'static EngineSlot {
    static SLOT: OnceLock<EngineSlot> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(None))
}

/// Drops this module's resident engine.
pub fn unload_engine() {
    if let Ok(mut s) = engine_slot().lock() {
        *s = None;
    }
}

/// Engine chunks of exactly `max` frame indices when `n > max`: chunk 0 is
/// an even sample of all frames; every later chunk is `anchors` frames of
/// chunk 0 (its first frame first) plus up to `max − anchors` new frames,
/// padded with more chunk-0 frames so every chunk has the same size (one
/// engine workspace serves them all).
pub fn chunk_plan(n: usize, max: usize, anchors: usize) -> Vec<Vec<usize>> {
    if n <= max {
        return vec![(0..n).collect()];
    }
    let anchors = anchors.clamp(2, max - 1);
    let first = crate::pose::sample_indices(n, max);
    let mut in_first = vec![false; n];
    first.iter().for_each(|&i| in_first[i] = true);
    let rest: Vec<usize> = (0..n).filter(|&i| !in_first[i]).collect();
    let anc: Vec<usize> = crate::pose::sample_indices(max, anchors)
        .into_iter()
        .map(|k| first[k])
        .collect();
    let mut out = vec![first.clone()];
    for piece in rest.chunks(max - anchors) {
        let mut c = anc.clone();
        c.extend_from_slice(piece);
        for &f in &first {
            if c.len() >= max {
                break;
            }
            if !c.contains(&f) {
                c.push(f);
            }
        }
        out.push(c);
    }
    out
}

/// Runs the engine on every frame (in [`chunk_plan`] chunks of at most
/// `max_frames`, registered to the first chunk by a similarity on the shared
/// frames' camera centres). Returns the host results, load and wall seconds
/// and the summed forward GPU time.
pub fn pose_frames(
    weights: &std::path::Path,
    frames: &Frames,
    keep: bool,
    max_frames: usize,
    cancel: &CancellationToken,
    events: &mut dyn FnMut(Event),
) -> Result<(Posed, f64, f64, f32)> {
    // The pose op's engine would double the weights on the device.
    crate::pose::unload_engine();
    let t0 = Instant::now();
    let mut slot = engine_slot()
        .lock()
        .map_err(|_| spl_core::anyhow!("engine lock poisoned"))?;
    let engine = match slot.take() {
        Some((p, e)) if p == weights => slot.insert((p, e)),
        _ => {
            let nn = Arc::new(spl_nn::Nn::open()?);
            let e = VggtEngine::load(nn, weights, VggtConfig::default())
                .with_context(|| format!("load VGGT-Omega from {}", weights.display()))?;
            slot.insert((weights.to_path_buf(), e))
        }
    };
    let load_s = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let e = &mut engine.1;
    let n = frames.engine.len();
    let plan = chunk_plan(n, max_frames.max(2), 32);
    let n_chunks = plan.len();
    let mut fwd = 0.0f32;
    let mut out: Option<Posed> = None;
    let mut done = vec![false; n];
    for (ci, chunk) in plan.iter().enumerate() {
        let imgs: Vec<Rgb8Image> = chunk.iter().map(|&i| frames.engine[i].clone()).collect();
        let prepared = e.preprocess(&imgs)?;
        let mut progress = |stage: &str, f: f32| {
            let g = (ci as f32 + f) / n_chunks as f32;
            events(Event::Progress(
                0.02 + 0.13 * g as f64,
                format!("engine {}/{n_chunks}: {stage}", ci + 1),
            ));
        };
        let cams = e.run(&prepared, cancel, &mut progress)?;
        let dense = e.download(&prepared.geom)?;
        fwd += e.times.total_ms;
        let part = Posed {
            names: chunk.iter().map(|&i| frames.names[i].clone()).collect(),
            cams,
            prepared,
            dense,
            seed: None,
        };
        let all = out.get_or_insert_with(|| Posed::empty(n, &part));
        all.merge_chunk(&part, chunk, &mut done)?;
    }
    if !keep {
        *slot = None;
    }
    drop(slot);
    let mut posed = out.ok_or_else(|| spl_core::anyhow!("no frames"))?;
    posed.names = frames.names.clone();
    Ok((posed, load_s, t1.elapsed().as_secs_f64(), fwd))
}

/// Moves each camera of `test` with its nearest (by centre) camera of
/// `before`: `T' = T · T_nn⁻¹ · T_nn'` with `T_nn'` the same camera in
/// `after` (world-to-camera transforms; `before[k]` and `after[k]` are one
/// camera). Keeps the engine's relative pose to the neighbour.
pub fn carry_poses(test: &[View], before: &[View], after: &[View]) -> Vec<View> {
    type M = [[f64; 3]; 3];
    let mul = |a: &M, b: &M| -> M {
        std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
    };
    let mv = |a: &M, v: &[f64; 3]| -> [f64; 3] {
        std::array::from_fn(|i| (0..3).map(|k| a[i][k] * v[k]).sum())
    };
    let tr = |a: &M| -> M { std::array::from_fn(|i| std::array::from_fn(|j| a[j][i])) };
    test.iter()
        .map(|v| {
            let c = v.center();
            let nn = before
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let d = |x: &View| {
                        let p = x.center();
                        (0..3).map(|k| (p[k] - c[k]).powi(2)).sum::<f64>()
                    };
                    d(a).total_cmp(&d(b))
                })
                .map(|(k, _)| k);
            let Some(k) = nn else {
                return v.clone();
            };
            let (b, a) = (&before[k], &after[k]);
            // M = T_b⁻¹ · T_a: R = R_bᵀ R_a, t = R_bᵀ (t_a − t_b).
            let rbt = tr(&b.r_cw);
            let rm = mul(&rbt, &a.r_cw);
            let tm = mv(&rbt, &std::array::from_fn(|i| a.t_cw[i] - b.t_cw[i]));
            // T' = T · M: R = R R_m, t = R t_m + t.
            let mut out = v.clone();
            out.r_cw = mul(&v.r_cw, &rm);
            let rt = mv(&v.r_cw, &tm);
            out.t_cw = std::array::from_fn(|i| rt[i] + v.t_cw[i]);
            out
        })
        .collect()
}

/// Area downscale to `w`×`h`.
fn downscale(img: Rgb8Image, w: u32, h: u32) -> Rgb8Image {
    let r = crate::ingest::resize_area_cpu(
        &crate::ingest::Rgb8 {
            width: img.width,
            height: img.height,
            data: img.data,
        },
        w,
        h,
    );
    Rgb8Image {
        width: r.width,
        height: r.height,
        data: r.data,
    }
}

/// Pose bootstrap: a short sparse-init run with joint pose refinement;
/// returns the refined training views.
#[allow(clippy::too_many_arguments)]
fn bootstrap(
    gpu: &Gpu,
    posed: &Posed,
    train: &[usize],
    imgs: &[Rgb8Image],
    (tw, th): (u32, u32),
    focal: (f64, f64),
    cfg: &PipelineConfig,
    cancel: &CancellationToken,
) -> Result<Vec<View>> {
    let cache = ImageCache::from_images(gpu, (0..train.len()).collect(), imgs, 0.0)?;
    let points = posed.points(train, cfg.init_conf_drop, 100_000);
    let mut scene = training_scene(posed.views(train, tw, th, focal), points)?;
    let all: Vec<usize> = scene.train.clone();
    scene.train = Vec::new();
    let tcfg = TrainConfig {
        preset: PresetName::Fast,
        iterations: Some(cfg.bootstrap_iters),
        seed: cfg.seed ^ 0xB007,
        ..Default::default()
    };
    let mut tr = Trainer::new(gpu, scene, tcfg)?;
    tr.scene.train = all.clone();
    tr.scene_gpu.train = all.clone();
    tr.scene_gpu.images = cache;
    tr.order = ViewOrder::new(all, cfg.seed ^ 0xB007);
    let total = tr.total;
    let jc = JointConfig {
        pose: Some(PoseRefine {
            start: 0,
            end: total,
            lr_rot: cfg.bootstrap_lr,
            lr_trans: cfg.bootstrap_lr,
            lr_focal: cfg.focal_lr,
            final_factor: 0.1,
        }),
        ppisp: None,
        depth: None,
    };
    let mut joint = Joint::new(tr, jc, None)?;
    while joint.tr.t < total {
        if joint.tr.t.is_multiple_of(50) && cancel.is_cancelled() {
            bail!("cancelled");
        }
        joint.step()?;
    }
    joint.tr.gpu.sync()?;
    joint.refined_views()
}

/// The densification end of the resolved preset (pose refinement stops there).
fn densify_end(tr: &Trainer) -> u32 {
    match &tr.preset.densify {
        DensifyPreset::FastGs(c) => c.end,
        DensifyPreset::Adc(c) => c.end,
        DensifyPreset::Mcmc(c) => c.end,
    }
}

/// Everything after ingest. `holdout_every` > 0 holds out every n-th frame
/// (posed with the rest, never trained on) for the benchmark protocol.
pub fn reconstruct(
    weights: &std::path::Path,
    frames: Frames,
    cfg: &PipelineConfig,
    holdout_every: usize,
    cancel: &CancellationToken,
    events: &mut dyn FnMut(Event),
) -> Result<Reconstruction> {
    reconstruct_with(weights, frames, cfg, holdout_every, cancel, events, None)
}

/// Edits the engine's results in place (see [`reconstruct_with`]).
pub type PosedHook<'a> = dyn FnMut(&mut Posed) -> Result<()> + 'a;

/// [`reconstruct`] with a hook that may edit the engine's results before
/// anything uses them (diagnostics: reference poses).
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_with(
    weights: &std::path::Path,
    frames: Frames,
    cfg: &PipelineConfig,
    holdout_every: usize,
    cancel: &CancellationToken,
    events: &mut dyn FnMut(Event),
    adjust: Option<&mut PosedHook<'_>>,
) -> Result<Reconstruction> {
    let started = Instant::now();
    let n = frames.names.len();
    ensure!(n >= 2, "need at least two frames, got {n}");
    let (tw, th) = (frames.trainer[0].width, frames.trainer[0].height);
    ensure!(
        frames
            .trainer
            .iter()
            .all(|f| f.width == tw && f.height == th),
        "all frames must have the same size"
    );
    let mut times = StageTimes::default();

    // A. Engine.
    events(Event::Progress(0.02, "engine".into()));
    let (mut posed, load_s, engine_s, fwd_ms) = pose_frames(
        weights,
        &frames,
        cfg.keep_engine,
        cfg.engine_max_frames as usize,
        cancel,
        events,
    )?;
    if let Some(f) = adjust {
        f(&mut posed)?;
    }
    times.engine_load_s = load_s;
    times.engine_s = engine_s;
    times.engine_forward_ms = fwd_ms;
    events(Event::Log(format!(
        "engine: {n} frames at {}x{} in {engine_s:.2} s (load {load_s:.2} s, forward {fwd_ms:.0} ms)",
        posed.cams.width, posed.cams.height
    )));
    let (train, test) = holdout_split(n, holdout_every);
    ensure!(train.len() >= 2, "need at least two training frames");
    if cfg.world_spread > 0.0 {
        let s = posed.spread(&train);
        if s > 0.0 && s.is_finite() {
            posed.rescale(cfg.world_spread / s);
        }
    }
    if cfg.ba_iters > 0 {
        // Every frame takes part (as COLMAP would); training frame 0 is the gauge.
        let order: Vec<usize> = train.iter().chain(&test).copied().collect();
        let imgs: Vec<&Rgb8Image> = order.iter().map(|&i| &frames.trainer[i]).collect();
        let rep = bundle_adjust(
            &mut posed,
            &order,
            &imgs,
            &BaConfig {
                iters: cfg.ba_iters,
                keypoints: cfg.ba_keypoints,
                neighbours: cfg.ba_neighbours,
                focal: cfg.ba_focal,
                reject_px: cfg.ba_reject_px as f64,
                ..Default::default()
            },
        );
        times.ba_s = rep.track_s + rep.solve_s;
        events(Event::Log(format!(
            "bundle adjustment: {} frames, {} points, {} observations ({} kept, {} tracks tried), rms {:.3} → {:.3} px, inliers {:.3} → {:.3}, focal ×{:.4}, median rotation change {:.3}°, {} iterations, track {:.1} s + solve {:.1} s",
            rep.frames,
            rep.points,
            rep.observations,
            rep.observations_kept,
            rep.tracks_tried,
            rep.rms_before,
            rep.rms_after,
            rep.inliers_before,
            rep.inliers_after,
            rep.focal_scale,
            rep.median_rot_change_deg,
            rep.iterations,
            rep.track_s,
            rep.solve_s
        )));
    }
    // The engine's training cameras: held-out cameras are carried from them.
    let engine_train = {
        let f = posed.shared_focal(&train, tw, th);
        posed.views(&train, tw, th, f)
    };
    if cfg.geo_iters > 0 {
        let rep = refine_poses(
            &mut posed,
            &train,
            &GeoRefine {
                iters: cfg.geo_iters,
                samples: cfg.geo_samples as usize,
                scale: cfg.geo_scale,
                ..Default::default()
            },
        );
        times.geo_s = rep.seconds;
        events(Event::Log(format!(
            "geometric refinement: {} frames, |r| {:.4} → {:.4}, inliers {:.2} → {:.2}, median rotation change {:.3}°, scale {:.3}-{:.3}, {:.1} s",
            rep.frames,
            rep.rms_before,
            rep.rms_after,
            rep.inliers_before,
            rep.inliers_after,
            rep.median_rot_change_deg,
            rep.scale_min,
            rep.scale_max,
            rep.seconds
        )));
    }
    let focal = posed.shared_focal(&train, tw, th);
    {
        let all: Vec<usize> = (0..n).collect();
        events(Event::Cameras(cameras_json(
            &posed.views(&all, tw, th, focal),
        )?));
        let pts = crate::pose::point_cloud(&posed.prepared, &posed.dense, 50.0, 1_000_000);
        let mut ply = Vec::new();
        crate::pose::write_points_ply(&pts, &posed.cams, &mut ply)?;
        events(Event::Points(ply, pts.len() as u32));
    }
    if cancel.is_cancelled() {
        bail!("cancelled");
    }

    // B. Trainer, Stage B init, prune-first.
    let t_setup = Instant::now();
    events(Event::Progress(0.16, "initializing".into()));
    let gpu = Gpu::open()?;
    let mut focal = focal;
    if cfg.bootstrap_iters > 0 {
        let t = Instant::now();
        let imgs: Vec<Rgb8Image> = train.iter().map(|&i| frames.trainer[i].clone()).collect();
        let k = cfg.bootstrap_scale.max(1);
        let (bw, bh) = ((tw / k).max(1), (th / k).max(1));
        let imgs: Vec<Rgb8Image> = if k > 1 {
            imgs.into_iter().map(|i| downscale(i, bw, bh)).collect()
        } else {
            imgs
        };
        let (sx, sy) = (bw as f64 / tw as f64, bh as f64 / th as f64);
        let views = bootstrap(
            &gpu,
            &posed,
            &train,
            &imgs,
            (bw, bh),
            (focal.0 * sx, focal.1 * sy),
            cfg,
            cancel,
        )?;
        posed.update_cameras(&train, &views, bw, bh);
        focal = (views[0].fx / sx, views[0].fy / sy);
        times.bootstrap_s = t.elapsed().as_secs_f64();
        events(Event::Log(format!(
            "pose bootstrap: {} iterations in {:.1} s",
            cfg.bootstrap_iters, times.bootstrap_s
        )));
    }
    let mut train_views = posed.views(&train, tw, th, focal);
    if cfg.per_image_focal {
        for (k, v) in train_views.iter_mut().enumerate() {
            v.camera = k as u32;
        }
    }
    let test_views = posed.views(&test, tw, th, focal);
    let train_imgs: Vec<Rgb8Image> = train.iter().map(|&i| frames.trainer[i].clone()).collect();
    let test_imgs: Vec<Rgb8Image> = test.iter().map(|&i| frames.trainer[i].clone()).collect();
    drop(frames);
    // Images first: the trainer sizes its capacity from what is left.
    let cache = ImageCache::from_images(&gpu, (0..train.len()).collect(), &train_imgs, 0.0)?;
    drop(train_imgs);
    let seed_points = if let Some(p) = posed.seed.take() {
        p
    } else if cfg.init_min_views > 0 {
        let t = Instant::now();
        let pts = consistent_points(
            &posed,
            &train,
            cfg.init_conf_drop,
            4 * cfg.init_points.max(4096) as usize / train.len().max(1) + 64,
            cfg.init_tau as f64,
            cfg.init_min_views as usize,
            cfg.init_points.max(4096) as usize,
        );
        events(Event::Log(format!(
            "consistent init points: {} in {:.2} s",
            pts.xyz.len(),
            t.elapsed().as_secs_f64()
        )));
        pts
    } else if cfg.init_voxel > 0.0 {
        let cap = cfg.init_points.max(4096) as usize;
        let cand = posed.points(
            &train,
            cfg.init_conf_drop,
            cap * cfg.init_oversample.max(1) as usize,
        );
        let (c, r) = extent(train_views.iter().map(View::center));
        let merged = voxel_merge(&cand, c, r, cfg.init_voxel);
        events(Event::Log(format!(
            "voxel-merged init: {} candidates → {} voxels",
            cand.xyz.len(),
            merged.xyz.len()
        )));
        subsample(merged, cap)
    } else {
        posed.points(
            &train,
            cfg.init_conf_drop,
            cfg.init_points.max(4096) as usize,
        )
    };
    {
        let (c, r) = extent(train_views.iter().map(View::center));
        let (med, far) = spread_stats(&seed_points, c, r);
        events(Event::Log(format!(
            "init points: {}, median distance {med:.3} extents, {:.1} % beyond",
            seed_points.xyz.len(),
            100.0 * far
        )));
    }
    let mut scene = training_scene(train_views, seed_points)?;
    let all_train = scene.train.clone();
    scene.train = Vec::new();
    let tcfg = cfg.train_config();
    let mut tr = Trainer::new(&gpu, scene, tcfg)?;
    tr.scene.train = all_train.clone();
    tr.scene_gpu.train = all_train.clone();
    tr.scene_gpu.images = cache;
    tr.order = ViewOrder::new(all_train, cfg.seed);
    times.setup_s = t_setup.elapsed().as_secs_f64();

    let t_init = Instant::now();
    if cfg.init_points == 0 {
        let kinit = InitKernels::load(&gpu)?;
        let depth_host = posed.depth_frames(&train);
        let df = DepthFrames::upload(&gpu, &depth_host)?;
        drop(depth_host);
        let target = cfg.init_target.min((tr.g.cap as f32 / 1.15) as u32).max(1);
        let opts = InitOptions {
            target,
            conf_percentile: cfg.init_conf_drop,
            ..Default::default()
        };
        let rep = init_from_depth(&gpu, &kinit, &df, &opts, &mut tr.g)?;
        drop(df);
        gpu.sync()?;
        times.init_s = t_init.elapsed().as_secs_f64();
        events(Event::Log(format!(
            "stage B: {} Gaussians (voxel {:.4}, conf ≥ {:.3}) in {:.0} ms",
            rep.n_gaussians, rep.voxel, rep.conf_threshold, rep.ms_total
        )));
    } else {
        times.init_s = t_init.elapsed().as_secs_f64();
        events(Event::Log(format!(
            "init: {} engine points, kNN scales",
            tr.g.n
        )));
    }
    let init_gaussians = tr.g.n;
    let t_prune = Instant::now();
    let prune = prune_first(
        &mut tr,
        cfg.prune_views as usize,
        cfg.prune_tau,
        cfg.prune_fraction,
    )?;
    times.prune_s = t_prune.elapsed().as_secs_f64();
    if prune.after != prune.before {
        events(Event::Log(format!(
            "prune-first: {} → {} Gaussians over {} views",
            prune.before, prune.after, prune.views
        )));
    }

    // C. Joint optimization.
    let total = tr.total;
    let dens_end = densify_end(&tr).min(total);
    let joint_cfg = JointConfig {
        pose: cfg.pose_refine.then(|| PoseRefine {
            start: cfg.pose_start,
            end: cfg.pose_end.unwrap_or(dens_end).min(total),
            lr_rot: cfg.pose_lr_rot,
            lr_trans: cfg.pose_lr_trans,
            lr_focal: cfg.focal_lr,
            final_factor: cfg.pose_lr_final,
        }),
        ppisp: cfg.ppisp.then(PpispConfig::default),
        depth: (cfg.depth_weight > 0.0).then(|| DepthPriorCfg {
            init: cfg.depth_weight,
            end: ((total as f32 * cfg.depth_until) as u32).max(1),
        }),
    };
    let priors = if joint_cfg.depth.is_some() {
        let (d, w) = posed.depth_priors(&train, cfg.depth_conf_drop);
        Some(DepthPriors::upload(
            &gpu,
            posed.cams.width as u32,
            posed.cams.height as u32,
            &d,
            &w,
        )?)
    } else {
        None
    };
    drop(posed);
    let mut joint = Joint::new(tr, joint_cfg, priors)?;
    let t_train = Instant::now();
    let mut last_progress = Instant::now();
    let mut last_snap = Instant::now();
    let (snap_tx, snap_rx) = std::sync::mpsc::sync_channel::<(GaussiansHost, u32)>(1);
    let (done_tx, done_rx) = std::sync::mpsc::channel::<(Vec<u8>, u32, u32)>();
    let encoder = std::thread::spawn(move || {
        while let Ok((g, t)) = snap_rx.recv() {
            let opts = SpzOptions {
                sh_degree: Some(1),
                ..Default::default()
            };
            if let Ok(spz) = write_spz(&g, &opts) {
                let _ = done_tx.send((spz, t, g.len() as u32));
            }
        }
    });
    while joint.tr.t < total {
        if joint.tr.t.is_multiple_of(50) && cancel.is_cancelled() {
            bail!("cancelled");
        }
        joint.step()?;
        let t = joint.tr.t;
        if last_progress.elapsed().as_secs_f64() >= 0.5 || t == total {
            last_progress = Instant::now();
            let loss = joint.read_loss()?;
            let el = t_train.elapsed().as_secs_f64();
            events(Event::Progress(
                0.18 + 0.78 * t as f64 / total as f64,
                format!(
                    "training {t}/{total}: {} Gaussians, {:.0} it/s, loss {}",
                    joint.tr.g.n,
                    t as f64 / el.max(1e-9),
                    loss.map_or("-".into(), |l| format!("{l:.4}"))
                ),
            ));
        }
        if cfg.snapshot_every_s > 0.0
            && last_snap.elapsed().as_secs_f32() >= cfg.snapshot_every_s
            && t < total
        {
            last_snap = Instant::now();
            let mut g = joint.tr.snapshot()?;
            g.sh_degree = joint.tr.schedule.sh_degree(t.max(1));
            // Skipped when the encoder is still busy with the last one.
            let _ = snap_tx.try_send((g, t));
        }
        while let Ok((spz, st, n)) = done_rx.try_recv() {
            events(Event::Snapshot { spz, t: st, n });
        }
    }
    drop(snap_tx);
    let _ = encoder.join();
    joint.tr.gpu.sync()?;
    times.train_s = t_train.elapsed().as_secs_f64();

    // D. Exports.
    let t_export = Instant::now();
    events(Event::Progress(0.97, "exporting".into()));
    let cloud = joint.export_cloud()?;
    let spz = write_spz(&cloud, &SpzOptions::default())?;
    let views = joint.refined_views()?;
    times.export_s = t_export.elapsed().as_secs_f64();
    times.total_s = started.elapsed().as_secs_f64();
    // Held-out cameras follow their nearest training camera's refinement
    // (training information only), then share the refined intrinsics.
    let test = if test_views.is_empty() {
        None
    } else {
        let carried = carry_poses(&test_views, &engine_train, &views);
        let mut tv = TestViews::new(&gpu, carried, &test_imgs)?;
        tv.share_intrinsics(&gpu, &joint.tr.scene_gpu.cams)?;
        Some(tv)
    };
    let peak = joint.tr.summary.peak_vram_bytes;
    Ok(Reconstruction {
        iterations: joint.tr.t,
        eager_steps: joint.eager_steps,
        cloud,
        spz,
        views,
        times,
        prune,
        init_gaussians,
        peak_vram_bytes: peak,
        joint: Some(joint),
        test,
        test_images: test_imgs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_resolve() {
        let q = PipelineConfig::preset(ReconPreset::Quality);
        assert_eq!(q.train_config().resolve().iterations, 30_000);
        let f = PipelineConfig::preset(ReconPreset::Fast);
        assert_eq!(f.train_config().resolve().iterations, 15_000);
        let json = serde_json::to_string(&f).unwrap();
        let back: PipelineConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back, f);
    }

    #[test]
    fn chunk_plan_covers_every_frame_with_equal_chunks() {
        assert_eq!(chunk_plan(10, 20, 4), vec![(0..10).collect::<Vec<_>>()]);
        let plan = chunk_plan(301, 180, 32);
        assert!(plan.iter().all(|c| c.len() == 180));
        let mut seen = vec![false; 301];
        plan.iter().flatten().for_each(|&i| seen[i] = true);
        assert!(seen.iter().all(|&s| s));
        for c in &plan[1..] {
            assert_eq!(c[0], 0, "the reference frame anchors every chunk");
            let shared = c.iter().filter(|i| plan[0].contains(i)).count();
            assert!(shared >= 32);
            let mut d = c.clone();
            d.sort_unstable();
            d.dedup();
            assert_eq!(d.len(), c.len(), "no duplicates");
        }
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn carried_cameras_keep_their_relative_pose() {
        let rz = |d: f64| -> [[f64; 3]; 3] {
            let (s, c) = d.to_radians().sin_cos();
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
        };
        let view = |r, t| View {
            name: String::new(),
            camera: 0,
            r_cw: r,
            t_cw: t,
            fx: 1.,
            fy: 1.,
            cx: 0.,
            cy: 0.,
            width: 1,
            height: 1,
            image_path: PathBuf::new(),
        };
        let before = vec![
            view(rz(0.0), [0.0, 0.0, 0.0]),
            view(rz(0.0), [10.0, 0.0, 0.0]),
        ];
        // The world moved: every camera got the same world transform G
        // (T' = T·G); a test camera near camera 0 must get it too.
        let g_r = rz(5.0);
        let g_t = [0.3, -0.2, 0.1];
        let apply = |v: &View| {
            let mut o = v.clone();
            for i in 0..3 {
                for j in 0..3 {
                    o.r_cw[i][j] = (0..3).map(|k| v.r_cw[i][k] * g_r[k][j]).sum();
                }
                o.t_cw[i] = (0..3).map(|k| v.r_cw[i][k] * g_t[k]).sum::<f64>() + v.t_cw[i];
            }
            o
        };
        let after: Vec<View> = before.iter().map(apply).collect();
        let test = vec![view(rz(20.0), [0.5, 0.1, 0.0])];
        let got = carry_poses(&test, &before, &after);
        let want = apply(&test[0]);
        for i in 0..3 {
            assert!((got[0].t_cw[i] - want.t_cw[i]).abs() < 1e-12);
            for j in 0..3 {
                assert!((got[0].r_cw[i][j] - want.r_cw[i][j]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn cameras_json_lists_frames() {
        let v = View {
            name: "a.jpg".into(),
            camera: 0,
            r_cw: [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
            t_cw: [1., 2., 3.],
            fx: 10.,
            fy: 11.,
            cx: 5.,
            cy: 4.,
            width: 10,
            height: 8,
            image_path: PathBuf::new(),
        };
        let j: serde_json::Value = serde_json::from_slice(&cameras_json(&[v]).unwrap()).unwrap();
        assert_eq!(j["frames"][0]["w2c"][1][3], 2.0);
        assert_eq!(j["frames"][0]["fy"], 11.0);
    }
}
