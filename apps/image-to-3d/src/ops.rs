//! The app's ops and their wire types (also the UI's TypeScript bindings in
//! `web/bindings/image-to-3d/`). Owner: trainer (E7).
//!
//! - `image_to_3d.train`: a COLMAP-posed dataset (a folder under the host's
//!   dataset root, or an uploaded zip) → splat artifacts (.ply, .spz), camera
//!   exports, metrics. `dry_run` loads and validates the dataset only.
//! - `image_to_3d.render`: one view of a trained splat → PNG.
//!
//! Dataset loading is real; training and rendering return "not implemented"
//! until the kernel owners land their stages (see DESIGN.md).

use std::path::{Component, Path, PathBuf};

use anyhow::ensure;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use spl_core::{ArtifactRef, Context, DataDir, Op, Result, RunCx, anyhow, bail};
use ts_rs::TS;

use crate::eval::ImageMetrics;
use crate::gpu::{Gpu, not_implemented};
use crate::io::scene::{LoadOptions, Scene, find_dataset_root, load_colmap_scene};
use crate::kernels::raster::GpuRaster;
use crate::kernels::{Capacity, RasterVariant};
use crate::trainer::{TrainConfig, TrainSummary, Trainer};

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
}

pub struct Train;

impl Op for Train {
    const ID: &'static str = "image_to_3d.train";
    const SUMMARY: &'static str =
        "Train Gaussian splats on a COLMAP-posed dataset (folder or uploaded zip)";

    type Input = TrainInput;
    type Output = TrainOutput;

    async fn run(&self, cx: &RunCx, input: TrainInput) -> Result<TrainOutput> {
        let root = dataset_root(cx, &input).await?;
        let load = input.load.clone();
        cx.progress(0.0, "loading dataset");
        let (scene, load_s) = tokio::task::spawn_blocking(move || {
            let started = std::time::Instant::now();
            load_colmap_scene(&root, &load).map(|s| (s, started.elapsed().as_secs_f64()))
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
            });
        }

        let cfg = input.train.clone();
        let cancel = cx.cancel.clone();
        let summary = tokio::task::spawn_blocking(move || -> Result<TrainSummary> {
            let gpu = Gpu::open()?;
            let preset = cfg.resolve();
            let (w, h) = scene
                .views
                .iter()
                .fold((1, 1), |(w, h), v| (w.max(v.width), h.max(v.height)));
            // E7: size from VRAM (spec-train 5.10) and grow on overflow (spec-raster 1.5).
            let cap = Capacity::new(1 << 22, 1 << 24, w, h, &preset.variant);
            let raster = GpuRaster::new(&gpu, cap, RasterVariant { ..preset.variant }, true)?;
            let mut trainer = Trainer::new(&gpu, scene, Box::new(raster), cfg)?;
            trainer.run(&cancel, &mut |_| {})
        })
        .await??;
        cx.metric("train_time", summary.train_s, "s");
        cx.metric("final_gaussians", summary.final_gaussians as f64, "");
        Err(not_implemented(
            "image_to_3d.train exports and eval",
            "trainer (E7)",
        ))
    }
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

    type Input = RenderInput;
    type Output = RenderOutput;

    async fn run(&self, cx: &RunCx, input: RenderInput) -> Result<RenderOutput> {
        ensure!(input.sh_degree <= 3, "sh_degree must be 0..=3");
        let c = &input.camera;
        ensure!(
            c.width > 0 && c.height > 0 && c.width <= 8192 && c.height <= 8192,
            "image size must be 1..=8192 per side"
        );
        let _bytes = cx.artifacts.get(&input.splat).await?;
        Err(not_implemented(
            "image_to_3d.render (PLY/SPZ read + raster_fwd)",
            "raster (E3)",
        ))
    }
}

/// The dataset directory a train run reads: a validated name under the
/// dataset root, or an uploaded zip extracted (once) under the data dir.
async fn dataset_root(cx: &RunCx, input: &TrainInput) -> Result<PathBuf> {
    match (&input.dataset, &input.zip) {
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
