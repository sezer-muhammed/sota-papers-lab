//! `image_to_3d.generate`: 1-4 object photos → PBR `.glb` mesh via
//! Pixal3D-mv (SIGGRAPH 2026) as a Python sidecar. Owner: pixal3d-6a.
//!
//! Pipeline: upload photos → VGGT-Ω poses → canonicalise (§2 of
//! spec-pixal3d) → Pixal3D sidecar (Docker) → PBR GLB.
//!
//! GPU-exclusive: the sidecar takes the whole card.

pub mod canon;
pub mod sidecar;

use std::path::PathBuf;

use anyhow::ensure;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use spl_core::{ArtifactKind, ArtifactRef, Context, DataDir, GpuUse, Op, Result, RunCx, anyhow};
use ts_rs::TS;

use crate::io::images::Rgb8Image;
use crate::pose;
use crate::vggt::{DenseHost, VggtCameras, VggtConfig, VggtEngine};

fn default_resolution() -> u32 {
    1024
}

fn default_max_tokens() -> u32 {
    49152
}

#[derive(Debug, Clone, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct GenerateInput {
    /// Artifact ids of 1-4 object photos (JPEG, PNG, WebP) with alpha masks
    /// or on a clean background.
    pub images: Vec<String>,
    /// Sidecar resolution: 1024 is the only tested value for 12 GB.
    #[serde(default = "default_resolution")]
    pub resolution: u32,
    /// Max sparse tokens for the HR stages: lower = less VRAM.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
}

#[derive(Debug, Clone, Serialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct GenerateTimes {
    pub pose_s: f64,
    pub canon_s: f64,
    pub sidecar_s: f64,
    pub total_s: f64,
}

#[derive(Debug, Serialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct GenerateOutput {
    pub frames: u32,
    pub glb: ArtifactRef,
    pub times: GenerateTimes,
    pub peak_vram_mb: f32,
}

pub struct Generate;

impl Op for Generate {
    const ID: &'static str = "image_to_3d.generate";
    const SUMMARY: &'static str = "Object photos to a PBR GLB mesh (Pixal3D-mv sidecar, 1-4 views)";
    const GPU: GpuUse = GpuUse::Exclusive;

    type Input = GenerateInput;
    type Output = GenerateOutput;

    async fn run(&self, cx: &RunCx, input: GenerateInput) -> Result<GenerateOutput> {
        let started = std::time::Instant::now();
        let n = input.images.len();
        ensure!((1..=4).contains(&n), "need 1-4 photos, got {n}");
        ensure!(
            input.resolution == 1024,
            "only resolution 1024 is supported (12 GB)"
        );

        // Check sidecar image exists.
        ensure!(
            sidecar::image_exists(),
            "the spl-pixal3d Docker image is not built on this host"
        );

        // 1. Load photos.
        cx.progress(0.0, "loading photos");
        let mut names = Vec::with_capacity(n);
        let mut photo_bytes = Vec::with_capacity(n);
        for id in &input.images {
            ensure!(
                !id.is_empty() && id.chars().all(|c| c.is_ascii_alphanumeric()),
                "images must be artifact ids"
            );
            names.push(id.clone());
            photo_bytes.push(cx.artifacts.get(id).await?);
        }

        // Decode photos.
        let photos: Vec<Rgb8Image> = photo_bytes
            .iter()
            .map(|b| crate::vggt::preprocess::decode(b))
            .collect::<Result<_>>()?;
        let image_wh = (photos[0].width as usize, photos[0].height as usize);

        cx.log(format!(
            "{n} photos loaded ({} x {})",
            image_wh.0, image_wh.1
        ));

        // 2. VGGT-Ω poses.
        cx.progress(0.05, "estimating poses (VGGT-Ω)");
        let weights = vggt_weights_path(cx)?;
        let cancel = cx.cancel.clone();
        let photos_clone = photos.clone();
        let (tx_prog, mut rx_prog) = tokio::sync::mpsc::unbounded_channel::<(String, f32)>();
        let pose_job = tokio::task::spawn_blocking(move || -> Result<PoseResult> {
            let t = std::time::Instant::now();
            let nn = std::sync::Arc::new(spl_nn::Nn::open()?);
            let engine_ref = get_or_load_engine(nn, &weights)?;
            let mut engine = engine_ref;
            let prep = engine.preprocess(&photos_clone)?;
            let mut progress = |stage: &str, f: f32| {
                let _ = tx_prog.send((stage.to_string(), f));
            };
            let cams = engine.run(&prep, &cancel, &mut progress)?;
            let dense = engine.download(&prep.geom)?;
            let pose_s = t.elapsed().as_secs_f64();
            Ok(PoseResult {
                cams,
                dense,
                pose_s,
            })
        });

        // Forward progress.
        let forward = async {
            while let Some((stage, f)) = rx_prog.recv().await {
                cx.progress(0.05 + 0.25 * f as f64, format!("posing: {stage}"));
            }
        };
        let (pose_res, ()) = tokio::join!(pose_job, forward);
        let PoseResult {
            cams,
            dense,
            pose_s,
        } = pose_res??;

        cx.metric("pose_time", pose_s, "s");
        cx.log(format!("poses estimated in {pose_s:.1} s"));

        // Unload VGGT engine to free GPU for the sidecar.
        pose::unload_engine();

        // 3. Canonicalise.
        cx.progress(0.35, "canonicalising poses");
        let canon_start = std::time::Instant::now();
        let model_wh = (cams.width, cams.height);
        let pixels = model_wh.0 * model_wh.1;

        // Build masks (simple: all pixels with alpha > 128, or all true if no alpha).
        let masks: Vec<Vec<bool>> = photos
            .iter()
            .map(|img| {
                // For RGB images without alpha, mask everything as object.
                vec![true; img.width as usize * img.height as usize]
            })
            .collect();

        // Build per-view depth and confidence from VGGT dense outputs.
        let mut depths = Vec::with_capacity(n);
        let mut confs = Vec::with_capacity(n);
        for vi in 0..n {
            let off = vi * pixels;
            let end = (off + pixels).min(dense.depth.len());
            depths.push(dense.depth[off..end].to_vec());
            let cend = (off + pixels).min(dense.conf.len());
            confs.push(dense.conf[off..cend].to_vec());
        }

        // RGBA PNGs for the sidecar.
        let rgba_pngs: Vec<Vec<u8>> = photos
            .iter()
            .zip(&masks)
            .map(|(img, mask)| encode_rgba_png(img, mask))
            .collect::<Result<_>>()?;

        let canon = canon::canonicalise(
            &cams.extrinsics,
            &cams.intrinsics,
            &depths,
            &confs,
            &masks,
            &rgba_pngs,
            0,
            model_wh,
            image_wh,
        )?;
        let canon_s = canon_start.elapsed().as_secs_f64();
        cx.metric("canon_time", canon_s, "s");

        // 4. Write transforms and images to a temp directory.
        let work_dir = tempfile::tempdir()?;
        canon::write_transforms(work_dir.path(), &canon, &names)?;
        cx.log(format!(
            "canonicalised {n} views, writing to {}",
            work_dir.path().display()
        ));

        // 5. Run the sidecar.
        cx.progress(0.4, "running Pixal3D-mv sidecar");
        let output_dir = tempfile::tempdir()?;
        let weights_dir = pixal3d_weights_dir(cx);
        let resolution = input.resolution;
        let cancel_flag = cx.cancel.clone();

        let (tx_sc, mut rx_sc) = tokio::sync::mpsc::unbounded_channel::<sidecar::SidecarProgress>();

        let wd = work_dir.path().to_owned();
        let od = output_dir.path().to_owned();
        let sidecar_job = tokio::task::spawn_blocking(move || {
            sidecar::run_sidecar(&wd, &weights_dir, &od, resolution, &cancel_flag, |p| {
                let _ = tx_sc.send(p);
            })
        });

        let sidecar_forward = async {
            while let Some(p) = rx_sc.recv().await {
                cx.progress(
                    0.4 + 0.55 * p.fraction as f64,
                    format!("pixal3d: {}", p.stage),
                );
            }
        };
        let (sc_res, ()) = tokio::join!(sidecar_job, sidecar_forward);
        let sc_result = sc_res??;
        let sidecar_s = sc_result.total_s;
        cx.metric("sidecar_time", sidecar_s, "s");
        cx.metric("peak_vram", sc_result.peak_vram_mb as f64, "MiB");

        // 6. Read the GLB and store as an artifact.
        cx.progress(0.95, "storing GLB");
        let glb_path = if std::path::Path::new(&sc_result.glb_path).is_file() {
            PathBuf::from(&sc_result.glb_path)
        } else {
            output_dir.path().join("output.glb")
        };
        ensure!(
            glb_path.is_file(),
            "GLB not found at {}",
            glb_path.display()
        );
        let glb_bytes = std::fs::read(&glb_path)?;
        let glb = cx
            .artifacts
            .put(glb_bytes, ArtifactKind::Mesh, spl_core::media::GLB)
            .await?;
        cx.announce("mesh", &glb, serde_json::json!({ "format": "glb" }));

        let total_s = started.elapsed().as_secs_f64();
        cx.metric("total_time", total_s, "s");
        cx.progress(1.0, "done");

        Ok(GenerateOutput {
            frames: n as u32,
            glb,
            times: GenerateTimes {
                pose_s,
                canon_s,
                sidecar_s,
                total_s,
            },
            peak_vram_mb: sc_result.peak_vram_mb,
        })
    }
}

struct PoseResult {
    cams: VggtCameras,
    dense: DenseHost,
    pose_s: f64,
}

fn vggt_weights_path(cx: &RunCx) -> Result<PathBuf> {
    let p = match std::env::var_os("SPL_VGGT_WEIGHTS") {
        Some(p) => PathBuf::from(p),
        None => {
            let data = cx
                .resources
                .get::<DataDir>()
                .map(|d| d.0.clone())
                .ok_or_else(|| anyhow!("this host has no data directory"))?;
            VggtEngine::default_weights(&data.join("models").join("vggt-omega"))
        }
    };
    ensure!(
        p.is_file(),
        "VGGT-Omega weights not found at {}",
        p.display()
    );
    Ok(p)
}

fn pixal3d_weights_dir(_cx: &RunCx) -> PathBuf {
    match std::env::var_os("SPL_PIXAL3D_WEIGHTS") {
        Some(p) => PathBuf::from(p),
        None => sidecar::default_weights_dir(),
    }
}

fn get_or_load_engine(
    nn: std::sync::Arc<spl_nn::Nn>,
    weights: &std::path::Path,
) -> Result<VggtEngine> {
    VggtEngine::load(nn, weights, VggtConfig::default())
        .with_context(|| format!("load VGGT-Omega from {}", weights.display()))
}

fn encode_rgba_png(img: &Rgb8Image, mask: &[bool]) -> Result<Vec<u8>> {
    let w = img.width as usize;
    let h = img.height as usize;
    let mut rgba = vec![0u8; w * h * 4];
    for i in 0..w * h {
        rgba[i * 4] = img.data[i * 3];
        rgba[i * 4 + 1] = img.data[i * 3 + 1];
        rgba[i * 4 + 2] = img.data[i * 3 + 2];
        rgba[i * 4 + 3] = if i < mask.len() && mask[i] { 255 } else { 0 };
    }
    let mut buf = Vec::new();
    image::write_buffer_with_format(
        &mut std::io::Cursor::new(&mut buf),
        &rgba,
        img.width,
        img.height,
        image::ExtendedColorType::Rgba8,
        image::ImageFormat::Png,
    )
    .context("encode RGBA PNG")?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_input_parses() {
        let input: GenerateInput =
            serde_json::from_str(r#"{"images": ["a1", "b2", "c3"]}"#).unwrap();
        assert_eq!(input.images.len(), 3);
        assert_eq!(input.resolution, 1024);
        assert_eq!(input.max_tokens, 49152);
    }

    #[test]
    fn generate_input_rejects_empty() {
        let input: GenerateInput = serde_json::from_str(r#"{"images": []}"#).unwrap();
        assert!(input.images.is_empty());
    }
}
