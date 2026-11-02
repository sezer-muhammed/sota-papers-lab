//! The app's ops and their wire types (also the UI's TypeScript bindings in
//! `web/bindings/image-to-3d/`). Owner: trainer (E7).
//!
//! - `image_to_3d.train`: a COLMAP-posed dataset (a folder under the host's
//!   dataset root, or an uploaded zip) → splat artifacts (.ply, .spz), camera
//!   exports, metrics. `dry_run` loads and validates the dataset only.
//! - `image_to_3d.render`: one view of a trained splat → PNG.
//! - `image_to_3d.mesh` (in `mesh::op`): splat + cameras → textured GLB; the
//!   train op's `mesh` option runs the same mesher after training.
//!

use std::path::{Component, Path, PathBuf};

use anyhow::ensure;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use spl_core::{
    ArtifactKind, ArtifactRef, Context, DataDir, GpuUse, Op, Result, RunCx, anyhow, bail,
};
use ts_rs::TS;

use crate::eval::{ImageMetrics, evaluate};
use crate::export::colmap::write_colmap;
use crate::export::ply::{read_inria_ply, write_inria_ply};
use crate::export::spz::{SpzOptions, read_spz, write_spz};
use crate::export::{MEDIA_PLY, MEDIA_SPZ, MEDIA_ZIP};
use crate::frame::{CameraTable, FrameParams, IterParams};
use crate::gpu::{CudaResultExt, Gpu};
use crate::io::images::Rgb8Image;
use crate::io::scene::{LoadOptions, Scene, View, find_dataset_root, load_colmap_scene};
use crate::kernels::layout::{EPS2D_DEFAULT, FLAG_AA};
use crate::kernels::raster::{FrameSource, GpuRaster, SplatRaster};
use crate::kernels::{Capacity, RasterVariant, RenderMode};
use crate::mesh::op::MEDIA_GLB;
use crate::mesh::pipeline::{MeshConfig, MeshResult, MeshSummary};
use crate::model::{Gaussians, GaussiansHost};
use crate::presets::DensifyPreset;
use crate::trainer::{TrainConfig, TrainProgress, TrainSummary, Trainer, planar_to_rgb8};
use crate::vram;

fn yes() -> bool {
    true
}

fn default_exports() -> Vec<ExportFormat> {
    vec![ExportFormat::Ply, ExportFormat::Spz]
}

fn default_sh_degree() -> u32 {
    3
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export, export_to = "image-to-3d/")]
pub enum ExportFormat {
    /// INRIA 3DGS PLY (62 floats per Gaussian).
    Ply,
    /// Niantic SPZ v3 (gzip): about 10x smaller, what web viewers load.
    Spz,
    /// COLMAP sparse model (bin) of the cameras, zipped.
    Colmap,
}

#[derive(Debug, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct TrainInput {
    /// Dataset folder relative to the host's dataset root (`$SPL_DATASETS`,
    /// else `<data dir>/datasets`), e.g. "mipnerf360/garden". Give this or `zip`.
    #[serde(default)]
    #[ts(optional)]
    pub dataset: Option<String>,
    /// Artifact id of an uploaded zip holding a COLMAP-posed dataset
    /// (`sparse/0/` plus an image folder). Give this or `dataset`.
    #[serde(default)]
    #[ts(optional)]
    pub zip: Option<String>,
    #[serde(default)]
    pub load: LoadOptions,
    #[serde(default)]
    pub train: TrainConfig,
    /// Render the held-out views and report PSNR / SSIM.
    #[serde(default = "yes")]
    pub eval: bool,
    #[serde(default = "default_exports")]
    pub export: Vec<ExportFormat>,
    /// Only load and validate the dataset (no GPU work).
    #[serde(default)]
    pub dry_run: bool,
    /// Also extract a textured mesh (GLB) from the trained splats and the
    /// training views (use the Geometry preset for the best surface).
    #[serde(default)]
    #[ts(optional)]
    pub mesh: Option<MeshConfig>,
}

/// What was loaded.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct SceneInfo {
    pub views: u32,
    pub train_views: u32,
    pub test_views: u32,
    /// Distinct COLMAP cameras.
    pub cameras: u32,
    pub points: u32,
    /// Size of the first view (pixels).
    pub width: u32,
    pub height: u32,
    pub extent: f32,
    /// Largest |principal point − image centre| over views, pixels.
    pub max_pp_offset: f32,
    pub load_s: f64,
}

impl SceneInfo {
    fn of(scene: &Scene, load_s: f64) -> Self {
        let mut cams: Vec<u32> = scene.views.iter().map(|v| v.camera).collect();
        cams.sort_unstable();
        cams.dedup();
        Self {
            views: scene.views.len() as u32,
            train_views: scene.train.len() as u32,
            test_views: scene.test.len() as u32,
            cameras: cams.len() as u32,
            points: scene.points.xyz.len() as u32,
            width: scene.views.first().map_or(0, |v| v.width),
            height: scene.views.first().map_or(0, |v| v.height),
            extent: scene.extent,
            max_pp_offset: scene.max_pp_offset,
            load_s,
        }
    }
}

#[derive(Debug, Serialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct TrainOutput {
    pub scene: SceneInfo,
    #[ts(optional)]
    pub summary: Option<TrainSummary>,
    /// Mean over the test views.
    #[ts(optional)]
    pub eval: Option<ImageMetrics>,
    #[ts(optional)]
    pub ply: Option<ArtifactRef>,
    #[ts(optional)]
    pub spz: Option<ArtifactRef>,
    #[ts(optional)]
    pub colmap: Option<ArtifactRef>,
    /// What the run gave up because other processes held GPU memory
    /// (fewer Gaussians, smaller images); empty on the quality path.
    pub gpu_memory: Vec<String>,
    /// Seconds spent waiting for GPU memory.
    pub gpu_wait_s: f64,
    #[ts(optional)]
    pub mesh: Option<ArtifactRef>,
    #[ts(optional)]
    pub mesh_summary: Option<MeshSummary>,
    /// Why the requested mesh is missing; the splat outputs are still valid.
    #[ts(optional)]
    pub mesh_error: Option<String>,
}

pub struct Train;

impl Op for Train {
    const ID: &'static str = "image_to_3d.train";
    const SUMMARY: &'static str =
        "Train Gaussian splats on a COLMAP-posed dataset (folder or uploaded zip)";
    const GPU: GpuUse = GpuUse::Exclusive;

    type Input = TrainInput;
    type Output = TrainOutput;

    async fn run(&self, cx: &RunCx, input: TrainInput) -> Result<TrainOutput> {
        let root = dataset_root(cx, &input).await?;
        let load = input.load.clone();
        cx.progress(0.0, "loading dataset");
        let (scene, load_s) = tokio::task::spawn_blocking({
            let root = root.clone();
            move || {
                let started = std::time::Instant::now();
                load_colmap_scene(&root, &load).map(|s| (s, started.elapsed().as_secs_f64()))
            }
        })
        .await??;
        let info = SceneInfo::of(&scene, load_s);
        cx.metric("views", info.views as f64, "");
        cx.metric("train_views", info.train_views as f64, "");
        cx.metric("test_views", info.test_views as f64, "");
        cx.metric("init_points", info.points as f64, "");
        cx.metric("extent", info.extent as f64, "");
        cx.metric("load_time", load_s, "s");
        cx.log(format!(
            "{} views ({} train / {} test) at {}x{}, {} points, extent {:.3}",
            info.views,
            info.train_views,
            info.test_views,
            info.width,
            info.height,
            info.points,
            info.extent
        ));
        if input.dry_run {
            cx.progress(1.0, "dataset ok");
            return Ok(TrainOutput {
                scene: info,
                summary: None,
                eval: None,
                ply: None,
                spz: None,
                colmap: None,
                gpu_memory: Vec::new(),
                gpu_wait_s: 0.0,
                mesh: None,
                mesh_summary: None,
                mesh_error: None,
            });
        }

        let mut cfg = input.train.clone();
        let cancel = cx.cancel.clone();
        let do_eval = input.eval;
        let mesh_cfg = input.mesh.clone();
        let mesh_views: Vec<View> = scene
            .train
            .iter()
            .map(|&i| scene.views[i].clone())
            .collect();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Progress>();
        let views = scene.views.clone();
        let load = input.load.clone();
        let job = tokio::task::spawn_blocking(move || -> Result<Trained> {
            // Resolution and capacity for the free memory (the quality path
            // when it is plentiful); waits when even the smallest misses.
            let (w, h) = scene
                .views
                .iter()
                .fold((1, 1), |(w, h), v| (w.max(v.width), h.max(v.height)));
            let preset = cfg.resolve();
            let cap = match &preset.densify {
                DensifyPreset::Mcmc(c) => c.cap_max as u64,
                _ => 8 << 20,
            }
            .min(cfg.max_gaussians.map_or(u64::MAX, u64::from));
            let (fit, waited) = vram::wait_until(
                &vram::WaitPolicy::from_env(),
                &cancel,
                vram::free_now,
                |f| {
                    Ok(vram::fit_trainer(
                        f,
                        256 << 20,
                        w,
                        h,
                        scene.train.len(),
                        cap,
                        &preset,
                    ))
                },
                &mut |ws: &vram::WaitState| {
                    let _ = tx.send(Progress::Waiting(ws.message(), ws.data("refine")));
                },
            )?;
            let mut coexist = vram::Coexist {
                notes: Vec::new(),
                waited_s: waited,
            };
            let (free, _) = vram::device_memory()?;
            if let Some(note) = vram::trainer_note(&fit, w, h, free) {
                let _ = tx.send(Progress::Log(note.clone()));
                coexist.notes.push(note);
            }
            cfg.max_gaussians = fit.max_gaussians.or(cfg.max_gaussians);
            let scene = if fit.scale < 1.0 {
                let mut load = load;
                load.max_width = Some(vram::scaled(w, h, fit.scale).0);
                load_colmap_scene(&root, &load)?
            } else {
                scene
            };
            let r = train_blocking(scene, cfg, do_eval, &mesh_cfg, &mesh_views, &cancel, &tx);
            vram::release_quietly();
            r.map(|t| Trained { coexist, ..t })
        });
        while let Some(p) = rx.recv().await {
            let p = match p {
                Progress::Train(p) => p,
                Progress::Waiting(msg, data) => {
                    cx.progress_with(0.0, msg, data);
                    continue;
                }
                Progress::Log(msg) => {
                    cx.log(msg);
                    continue;
                }
            };
            let loss = p.loss.map_or(String::new(), |l| format!(", loss {l:.4}"));
            cx.progress(
                p.t as f64 / p.total.max(1) as f64,
                format!(
                    "iteration {}/{}: {} Gaussians, {:.0} it/s{loss}",
                    p.t, p.total, p.n_gaussians, p.iterations_per_s
                ),
            );
        }
        let Trained {
            cloud,
            summary,
            eval,
            coexist,
            mesh,
        } = job.await??;
        let (mesh, mesh_error) = match mesh {
            Some(Ok(m)) => (Some(m), None),
            Some(Err(e)) => {
                let msg = format!("{e:#}");
                cx.log(format!("mesh failed, keeping the splat: {msg}"));
                (None, Some(msg))
            }
            None => (None, None),
        };
        if let Some(m) = &mesh {
            for n in &m.summary.notes {
                cx.log(format!("mesh: {n}"));
            }
        }
        cx.metric("train_time", summary.train_s, "s");
        cx.metric("final_gaussians", summary.final_gaussians as f64, "");
        cx.metric(
            "peak_vram",
            summary.peak_vram_bytes as f64 / (1u64 << 20) as f64,
            "MiB",
        );
        if let Some(m) = &eval {
            cx.metric("psnr", m.psnr, "dB");
            cx.metric("ssim", m.ssim, "");
        }
        let mut out = TrainOutput {
            scene: info,
            summary: Some(summary),
            eval,
            ply: None,
            spz: None,
            colmap: None,
            gpu_memory: coexist.notes,
            gpu_wait_s: coexist.waited_s,
            mesh: None,
            mesh_summary: None,
            mesh_error,
        };
        if let Some(m) = mesh {
            cx.metric("mesh_time", m.summary.total_s, "s");
            cx.metric("mesh_triangles", m.summary.triangles as f64, "");
            out.mesh = Some(
                cx.artifacts
                    .put(m.glb, ArtifactKind::Mesh, MEDIA_GLB)
                    .await?,
            );
            out.mesh_summary = Some(m.summary);
        }
        for fmt in &input.export {
            match fmt {
                ExportFormat::Ply => {
                    let mut bytes = Vec::new();
                    write_inria_ply(&cloud, &mut bytes)?;
                    out.ply = Some(
                        cx.artifacts
                            .put(bytes, ArtifactKind::Splat, MEDIA_PLY)
                            .await?,
                    );
                }
                ExportFormat::Spz => {
                    let bytes = write_spz(&cloud, &SpzOptions::default())?;
                    out.spz = Some(
                        cx.artifacts
                            .put(bytes, ArtifactKind::Splat, MEDIA_SPZ)
                            .await?,
                    );
                }
                ExportFormat::Colmap => {
                    let (views, cloud) = (views.clone(), cloud.clone());
                    let bytes =
                        tokio::task::spawn_blocking(move || colmap_zip(&views, &cloud)).await??;
                    out.colmap = Some(
                        cx.artifacts
                            .put(bytes, ArtifactKind::Cameras, MEDIA_ZIP)
                            .await?,
                    );
                }
            }
        }
        Ok(out)
    }
}

struct Trained {
    cloud: GaussiansHost,
    summary: TrainSummary,
    eval: Option<ImageMetrics>,
    coexist: vram::Coexist,
    /// The mesh, or why it failed (never fails the run).
    mesh: Option<Result<MeshResult>>,
}

/// Messages from the training thread.
enum Progress {
    Train(TrainProgress),
    /// Waiting for GPU memory: message, progress data.
    Waiting(String, serde_json::Value),
    Log(String),
}

/// Trains, evaluates and downloads; every device buffer is gone on return.
fn train_blocking(
    scene: Scene,
    cfg: TrainConfig,
    do_eval: bool,
    mesh_cfg: &Option<crate::mesh::pipeline::MeshConfig>,
    mesh_views: &[View],
    cancel: &spl_core::CancellationToken,
    tx: &tokio::sync::mpsc::UnboundedSender<Progress>,
) -> Result<Trained> {
    let gpu = Gpu::open()?;
    let mut trainer = Trainer::new(&gpu, scene, cfg)?;
    let summary = trainer.run(cancel, &mut |p| {
        let _ = tx.send(Progress::Train(p.clone()));
    })?;
    let eval = if do_eval {
        Some(evaluate(&mut trainer, None)?.mean)
    } else {
        None
    };
    let (cloud, mut s) = trainer.finish()?;
    s.train_s = summary.train_s;
    let mesh = mesh_cfg
        .as_ref()
        .map(|c| crate::mesh::pipeline::build(&gpu, &cloud, mesh_views, c));
    Ok(Trained {
        cloud,
        summary: s,
        eval,
        coexist: vram::Coexist::default(),
        mesh,
    })
}

/// The COLMAP sparse model (bin) of the cameras plus points from the splat
/// means, zipped as `sparse/0/*.bin`.
fn colmap_zip(views: &[View], cloud: &GaussiansHost) -> Result<Vec<u8>> {
    let dir = tempfile::tempdir().context("temp dir for the COLMAP export")?;
    let sparse = dir.path().join("sparse").join("0");
    std::fs::create_dir_all(&sparse)?;
    write_colmap(views, cloud, &sparse, false)?;
    let mut z = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    let opts = zip::write::SimpleFileOptions::default();
    for name in ["cameras.bin", "images.bin", "points3D.bin"] {
        let path = sparse.join(name);
        if path.is_file() {
            z.start_file(format!("sparse/0/{name}"), opts)?;
            std::io::Write::write_all(&mut z, &std::fs::read(&path)?)?;
        }
    }
    Ok(z.finish()?.into_inner())
}

/// A pinhole camera in COLMAP conventions (world → camera, x right, y down).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct RenderCamera {
    /// World-to-camera rotation, row-major.
    pub r_cw: [f64; 9],
    pub t_cw: [f64; 3],
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct RenderInput {
    /// Artifact id of a trained splat (.ply or .spz).
    pub splat: String,
    pub camera: RenderCamera,
    #[serde(default = "default_sh_degree")]
    pub sh_degree: u32,
    /// Linear RGB in [0, 1].
    #[serde(default)]
    pub background: [f32; 3],
}

#[derive(Debug, Serialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct RenderOutput {
    /// 8-bit PNG.
    pub image: ArtifactRef,
    pub render_ms: f64,
}

pub struct Render;

impl Op for Render {
    const ID: &'static str = "image_to_3d.render";
    const SUMMARY: &'static str = "Render one view of a trained splat";
    const GPU: GpuUse = GpuUse::Exclusive;

    type Input = RenderInput;
    type Output = RenderOutput;

    async fn run(&self, cx: &RunCx, input: RenderInput) -> Result<RenderOutput> {
        ensure!(input.sh_degree <= 3, "sh_degree must be 0..=3");
        let c = &input.camera;
        ensure!(
            c.width > 0 && c.height > 0 && c.width <= 8192 && c.height <= 8192,
            "image size must be 1..=8192 per side"
        );
        let bytes = cx.artifacts.get(&input.splat).await?;
        let camera = input.camera.clone();
        let (sh, bg) = (input.sh_degree, input.background);
        let (img, render_ms) = tokio::task::spawn_blocking(move || -> Result<(Rgb8Image, f64)> {
            let cloud = if bytes.starts_with(b"ply") {
                read_inria_ply(&bytes)?
            } else {
                read_spz(&bytes)?
            };
            render_cloud(&cloud, &camera, sh, bg)
        })
        .await??;
        let mut png = Vec::new();
        image::write_buffer_with_format(
            &mut std::io::Cursor::new(&mut png),
            &img.data,
            img.width,
            img.height,
            image::ExtendedColorType::Rgb8,
            image::ImageFormat::Png,
        )
        .context("encode PNG")?;
        let image = cx
            .artifacts
            .put(png, ArtifactKind::Image, "image/png")
            .await?;
        Ok(RenderOutput { image, render_ms })
    }
}

/// One eager Infer render of `cloud` (blocking; opens the GPU).
pub fn render_cloud(
    cloud: &GaussiansHost,
    c: &RenderCamera,
    sh_degree: u32,
    background: [f32; 3],
) -> Result<(Rgb8Image, f64)> {
    let gpu = Gpu::open()?;
    let n = cloud.len();
    ensure!(n > 0, "the splat is empty");
    let cap = n.next_multiple_of(1024);
    let mut g = Gaussians::alloc(&gpu, cap, cloud.antialiased)?;
    g.upload(&gpu, cloud)?;
    let view = View {
        name: "render".into(),
        camera: 0,
        r_cw: [
            [c.r_cw[0], c.r_cw[1], c.r_cw[2]],
            [c.r_cw[3], c.r_cw[4], c.r_cw[5]],
            [c.r_cw[6], c.r_cw[7], c.r_cw[8]],
        ],
        t_cw: c.t_cw,
        fx: c.fx,
        fy: c.fy,
        cx: c.cx,
        cy: c.cy,
        width: c.width,
        height: c.height,
        image_path: PathBuf::new(),
    };
    let cams = CameraTable::upload(&gpu, std::slice::from_ref(&view))?;
    let variant = RasterVariant::default();
    let mut instances = (n as u64 * 16).clamp(1 << 20, 1 << 26) as u32;
    let params = IterParams {
        frame: FrameParams {
            iteration: 1,
            n_gaussians: n as u32,
            sh_degree: sh_degree.min(cloud.sh_degree),
            bg: background,
            eps2d: EPS2D_DEFAULT,
            flags: if cloud.antialiased { FLAG_AA } else { 0 },
            render_w: c.width,
            render_h: c.height,
            box_beta: 1.0,
            ..Default::default()
        },
        ..Default::default()
    };
    for _ in 0..4 {
        let cap = Capacity::new(cap as u32, instances, c.width, c.height, &variant);
        let mut raster = GpuRaster::new(&gpu, cap, variant, false)?;
        let started = std::time::Instant::now();
        raster.forward(
            &gpu,
            &mut g,
            &cams,
            FrameSource::Eager(&params),
            RenderMode::Infer,
        )?;
        let st = raster.frame.iter.read_status(&gpu)?;
        let ms = started.elapsed().as_secs_f64() * 1e3;
        if st.overflow == 0 {
            let px = (c.width * c.height) as usize;
            let planar = gpu
                .stream
                .clone_dtoh(&raster.out.rgb.slice(0..3 * px))
                .cuda("download render")?;
            return Ok((planar_to_rgb8(&planar, c.width, c.height), ms));
        }
        instances = (st.n_instances as u64 * 5 / 4)
            .max(instances as u64 * 2)
            .min(1 << 30) as u32;
    }
    bail!("render: raster buffers kept overflowing")
}

/// The dataset directory a train run reads: a validated name under the
/// dataset root, or an uploaded zip extracted (once) under the data dir.
async fn dataset_root(cx: &RunCx, input: &TrainInput) -> Result<PathBuf> {
    dataset_dir(cx, &input.dataset, &input.zip).await
}

/// [`dataset_root`] for any op that takes a `dataset` folder or a `zip` id.
pub(crate) async fn dataset_dir(
    cx: &RunCx,
    dataset: &Option<String>,
    zip: &Option<String>,
) -> Result<PathBuf> {
    match (dataset, zip) {
        (Some(name), None) => {
            let base = match std::env::var_os("SPL_DATASETS") {
                Some(p) => PathBuf::from(p),
                None => data_dir(cx)?.join("datasets"),
            };
            let root = base.join(safe_relative(name)?);
            ensure!(
                root.is_dir(),
                "dataset {name:?} not found under {}",
                base.display()
            );
            Ok(root)
        }
        (None, Some(id)) => {
            ensure!(
                !id.is_empty() && id.chars().all(|c| c.is_ascii_alphanumeric()),
                "zip must be an artifact id"
            );
            let dest = data_dir(cx)?.join("image-to-3d").join("datasets").join(id);
            let marker = dest.join(".complete");
            if !marker.is_file() {
                let bytes = cx.artifacts.get(id).await?;
                cx.progress(0.0, "extracting dataset");
                let d = dest.clone();
                tokio::task::spawn_blocking(move || extract_zip(&bytes, &d)).await??;
                std::fs::write(&marker, b"").context("mark extracted dataset")?;
            }
            find_dataset_root(&dest, 3)
                .with_context(|| "the zip has no COLMAP model (sparse/0/cameras.bin or .txt)")
        }
        _ => bail!("give exactly one of `dataset` and `zip`"),
    }
}

fn data_dir(cx: &RunCx) -> Result<PathBuf> {
    cx.resources
        .get::<DataDir>()
        .map(|d| d.0.clone())
        .ok_or_else(|| anyhow!("this host has no data directory"))
}

/// `name` as a relative path of normal components only (no `..`, no root).
fn safe_relative(name: &str) -> Result<PathBuf> {
    let p = Path::new(name);
    ensure!(
        !name.is_empty() && p.components().all(|c| matches!(c, Component::Normal(_))),
        "dataset must be a relative path like \"mipnerf360/garden\", got {name:?}"
    );
    Ok(p.to_path_buf())
}

/// Extracts a zip into `dest`; entries that would escape it are rejected.
fn extract_zip(bytes: &[u8], dest: &Path) -> Result<()> {
    let mut zip = zip::ZipArchive::new(std::io::Cursor::new(bytes))
        .context("the upload is not a zip file")?;
    std::fs::create_dir_all(dest).with_context(|| format!("create {}", dest.display()))?;
    for i in 0..zip.len() {
        let mut f = zip.by_index(i)?;
        let rel = f
            .enclosed_name()
            .with_context(|| format!("zip entry {:?} escapes the archive", f.name()))?;
        let out = dest.join(rel);
        if f.is_dir() {
            std::fs::create_dir_all(&out)?;
            continue;
        }
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut w =
            std::fs::File::create(&out).with_context(|| format!("create {}", out.display()))?;
        std::io::copy(&mut f, &mut w).with_context(|| format!("extract {}", out.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    #[test]
    fn empty_and_minimal_inputs_parse() {
        let t: TrainInput = serde_json::from_str(r#"{"dataset": "mipnerf360/garden"}"#).unwrap();
        assert!(t.eval && !t.dry_run);
        assert_eq!(t.export, vec![ExportFormat::Ply, ExportFormat::Spz]);
        assert_eq!(t.load.test_every, 8);
        let r: RenderInput = serde_json::from_str(
            r#"{"splat": "ab", "camera": {"r_cw": [1,0,0,0,1,0,0,0,1], "t_cw": [0,0,0],
                "fx": 100, "fy": 100, "cx": 50, "cy": 40, "width": 100, "height": 80}}"#,
        )
        .unwrap();
        assert_eq!((r.sh_degree, r.background), (3, [0.0; 3]));
    }

    #[test]
    fn dataset_names_cannot_escape_the_root() {
        assert!(safe_relative("mipnerf360/garden").is_ok());
        for bad in ["", "../etc", "/etc", "a/../../b", "./x"] {
            assert!(safe_relative(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn zips_extract_and_reject_traversal() {
        let dir = tempfile::tempdir().unwrap();
        let mut z = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        let opts = zip::write::SimpleFileOptions::default();
        z.start_file("scene/sparse/0/cameras.txt", opts).unwrap();
        z.write_all(b"1 PINHOLE 4 4 1 1 2 2\n").unwrap();
        let bytes = z.finish().unwrap().into_inner();
        extract_zip(&bytes, dir.path()).unwrap();
        assert!(dir.path().join("scene/sparse/0/cameras.txt").is_file());

        let mut z = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        z.start_file("../evil.txt", opts).unwrap();
        z.write_all(b"x").unwrap();
        let bytes = z.finish().unwrap().into_inner();
        let sub = dir.path().join("sub");
        assert!(extract_zip(&bytes, &sub).is_err());
        assert!(!dir.path().join("evil.txt").exists());
    }
}
