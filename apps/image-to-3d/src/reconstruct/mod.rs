//! `image_to_3d.reconstruct`: photos or a phone video → Gaussian splats,
//! with no COLMAP (Report sections 3 and 9, Phase 4). Owner: mvp-pipeline.
//!
//! Stages: A0 ingest ([`crate::ingest`]: decode, EXIF, video keyframes) →
//! A VGGT-Ω engine ([`crate::vggt`]: poses, intrinsics, depth, confidence)
//! → B Stage B init ([`crate::init`]: confidence filter, voxel downsample)
//! and the prune-first pass ([`prune`]) → C joint optimization ([`joint`]:
//! the Phase 3 trainer plus per-image SE(3) and shared-focal refinement,
//! PPISP, the SSI depth prior) → exports (`.spz`, `.ply`, `cameras.json`,
//! COLMAP text, `transforms.json`).
//!
//! Announced while running: `cameras` and `points` once the engine is done
//! (seconds), then `splat` snapshots every [`PipelineConfig::snapshot_every_s`].
//! GPU-exclusive. The VGGT-Ω weights are under the FAIR non-commercial
//! research licence (see [`crate::pose`]).

pub mod joint;
pub mod pipeline;
pub mod prune;
pub mod scene;

#[cfg(test)]
mod gpu_tests;

use std::path::{Path, PathBuf};

use anyhow::ensure;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use spl_core::{ArtifactKind, ArtifactRef, Context, DataDir, GpuUse, Op, Result, RunCx, anyhow};
use ts_rs::TS;

use crate::export::ply::write_inria_ply;
use crate::export::{MEDIA_JSON, MEDIA_PLY, MEDIA_SPZ, MEDIA_ZIP};
use crate::ingest::{IngestConfig, IngestResult, ingest_images, ingest_video};
use crate::io::images::Rgb8Image;
use crate::vggt::VggtEngine;
pub use pipeline::{Event, Frames, PipelineConfig, ReconPreset, StageTimes, reconstruct};

fn default_max_frames() -> u32 {
    200
}

fn default_long_side() -> u32 {
    1600
}

#[derive(Debug, Clone, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct ReconstructInput {
    /// Artifact ids of photos (JPEG, PNG, WebP) in capture order.
    #[serde(default)]
    pub images: Vec<String>,
    /// Artifact id of a zip of photos (sorted by name).
    #[serde(default)]
    #[ts(optional)]
    pub zip: Option<String>,
    /// Artifact id of a video (MP4, MOV, WebM): sharp keyframes spread
    /// evenly in time.
    #[serde(default)]
    #[ts(optional)]
    pub video: Option<String>,
    /// A folder of photos under the host's dataset root, e.g.
    /// "mipnerf360/garden/images_4".
    #[serde(default)]
    #[ts(optional)]
    pub dataset: Option<String>,
    #[serde(default)]
    pub preset: ReconPreset,
    /// At most this many frames (keyframes for a video, an even sample of photos).
    #[serde(default = "default_max_frames")]
    pub max_frames: u32,
    /// Trainer images: longest side (never upscaled).
    #[serde(default = "default_long_side")]
    pub long_side: u32,
    /// Every knob (ablations); the preset's defaults when absent.
    #[serde(default)]
    #[ts(optional)]
    pub config: Option<PipelineConfig>,
}

#[derive(Debug, Serialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct ReconstructOutput {
    pub frames: u32,
    pub gaussians: u32,
    pub iterations: u32,
    pub spz: ArtifactRef,
    pub ply: ArtifactRef,
    pub cameras: ArtifactRef,
    pub colmap: ArtifactRef,
    pub transforms: ArtifactRef,
    pub times: StageTimes,
    pub ingest_s: f64,
    /// From the start of the run to the last export stored.
    pub wall_s: f64,
}

pub struct Reconstruct;

impl Op for Reconstruct {
    const ID: &'static str = "image_to_3d.reconstruct";
    const SUMMARY: &'static str =
        "Gaussian splats from unposed photos or a video (VGGT-Omega poses, joint refinement)";
    const GPU: GpuUse = GpuUse::Exclusive;

    type Input = ReconstructInput;
    type Output = ReconstructOutput;

    async fn run(&self, cx: &RunCx, input: ReconstructInput) -> Result<ReconstructOutput> {
        let started = std::time::Instant::now();
        ensure!(input.max_frames >= 2, "max_frames must be at least 2");
        let mut cfg = input
            .config
            .clone()
            .unwrap_or_else(|| PipelineConfig::preset(input.preset));
        let video = input.video.is_some();
        if input.config.is_none() && video {
            // Phone video: auto exposure and white balance drift.
            cfg.ppisp = true;
        }
        cx.progress(0.0, "reading input");
        let ingest_cfg = IngestConfig {
            trainer_long_side: input.long_side,
            max_frames: input.max_frames,
            ..Default::default()
        };
        let ingested = ingest(cx, &input, &ingest_cfg).await?;
        let ingest_s = started.elapsed().as_secs_f64();
        cx.log(format!(
            "ingest: {} frames kept of {} decoded in {ingest_s:.2} s",
            ingested.frames.len(),
            ingested.stats.decoded
        ));
        cx.metric("ingest_time", ingest_s, "s");
        let frames = to_frames(ingested);
        let n_frames = frames.names.len() as u32;
        let weights = weights_path(cx)?;
        let cancel = cx.cancel.clone();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
        let job = tokio::task::spawn_blocking(move || -> Result<_> {
            let mut send = |e: Event| {
                let _ = tx.send(e);
            };
            let r = reconstruct(&weights, frames, &cfg, 0, &cancel, &mut send)?;
            let mut ply = Vec::new();
            write_inria_ply(&r.cloud, &mut ply)?;
            let cameras = pipeline::cameras_json(&r.views)?;
            let mut transforms = Vec::new();
            crate::export::colmap::write_transforms_json(&r.views, &mut transforms)?;
            let colmap = colmap_zip(&r)?;
            Ok((
                r.spz,
                ply,
                cameras,
                transforms,
                colmap,
                r.times,
                r.iterations,
                r.cloud.len(),
            ))
        });
        let relay = async {
            while let Some(e) = rx.recv().await {
                if let Err(err) = relay_event(cx, e).await {
                    cx.log(format!("could not store an intermediate result: {err:#}"));
                }
            }
        };
        let (res, ()) = tokio::join!(job, relay);
        let (spz, ply, cameras, transforms, colmap, times, iterations, n) = res??;
        cx.progress(0.98, "storing exports");
        let spz = cx
            .artifacts
            .put(spz, ArtifactKind::Splat, MEDIA_SPZ)
            .await?;
        cx.announce(
            "splat",
            &spz,
            serde_json::json!({ "gaussians": n, "final": true }),
        );
        let ply = cx
            .artifacts
            .put(ply, ArtifactKind::Splat, MEDIA_PLY)
            .await?;
        let cameras = cx
            .artifacts
            .put(cameras, ArtifactKind::Cameras, MEDIA_JSON)
            .await?;
        let transforms = cx
            .artifacts
            .put(transforms, ArtifactKind::Cameras, MEDIA_JSON)
            .await?;
        let colmap = cx
            .artifacts
            .put(colmap, ArtifactKind::Cameras, MEDIA_ZIP)
            .await?;
        let wall_s = started.elapsed().as_secs_f64();
        cx.metric("frames", n_frames as f64, "");
        cx.metric("gaussians", n as f64, "");
        cx.metric("engine_time", times.engine_s, "s");
        cx.metric("train_time", times.train_s, "s");
        cx.metric("wall_time", wall_s, "s");
        cx.progress(1.0, "done");
        Ok(ReconstructOutput {
            frames: n_frames,
            gaussians: n as u32,
            iterations,
            spz,
            ply,
            cameras,
            colmap,
            transforms,
            times,
            ingest_s,
            wall_s,
        })
    }
}

async fn relay_event(cx: &RunCx, e: Event) -> Result<()> {
    match e {
        Event::Progress(f, msg) => cx.progress(f, msg),
        Event::Log(msg) => cx.log(msg),
        Event::Cameras(json) => {
            let a = cx
                .artifacts
                .put(json, ArtifactKind::Cameras, MEDIA_JSON)
                .await?;
            cx.announce("cameras", &a, serde_json::Value::Null);
        }
        Event::Points(ply, n) => {
            let a = cx
                .artifacts
                .put(ply, ArtifactKind::Splat, MEDIA_PLY)
                .await?;
            cx.announce("points", &a, serde_json::json!({ "points": n }));
        }
        Event::Snapshot { spz, t, n } => {
            let a = cx
                .artifacts
                .put(spz, ArtifactKind::Splat, MEDIA_SPZ)
                .await?;
            cx.announce_snapshot(
                "splat",
                &a,
                serde_json::json!({ "iteration": t, "gaussians": n }),
            );
        }
    }
    Ok(())
}

fn colmap_zip(r: &pipeline::Reconstruction) -> Result<Vec<u8>> {
    use std::io::Write;
    let dir = tempfile::tempdir()?;
    let sparse = dir.path().join("sparse").join("0");
    std::fs::create_dir_all(&sparse)?;
    crate::export::colmap::write_colmap(&r.views, &r.cloud, &sparse, true)?;
    let mut z = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    let opts = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    let mut files: Vec<PathBuf> = std::fs::read_dir(&sparse)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();
    files.sort();
    for f in files {
        let name = format!(
            "sparse/0/{}",
            f.file_name().and_then(|n| n.to_str()).unwrap_or("")
        );
        z.start_file(name, opts)?;
        z.write_all(&std::fs::read(&f)?)?;
    }
    Ok(z.finish()?.into_inner())
}

fn to_rgb(f: crate::ingest::Rgb8) -> Rgb8Image {
    Rgb8Image {
        width: f.width,
        height: f.height,
        data: f.data,
    }
}

/// Ingested keyframes → engine and trainer images.
pub fn to_frames(r: IngestResult) -> Frames {
    let mut out = Frames {
        names: Vec::with_capacity(r.frames.len()),
        engine: Vec::with_capacity(r.frames.len()),
        trainer: Vec::with_capacity(r.frames.len()),
    };
    for f in r.frames {
        out.names.push(f.source);
        out.engine.push(to_rgb(f.engine));
        out.trainer.push(to_rgb(f.trainer));
    }
    out
}

fn is_artifact_id(id: &str) -> bool {
    !id.is_empty() && id.chars().all(|c| c.is_ascii_alphanumeric())
}

fn is_image_name(name: &str) -> bool {
    let l = name.to_ascii_lowercase();
    [".jpg", ".jpeg", ".png", ".webp"]
        .iter()
        .any(|e| l.ends_with(e))
}

/// Image files of a folder, sorted by name.
pub fn image_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("read {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(is_image_name)
        })
        .collect();
    files.sort();
    Ok(files)
}

async fn ingest(cx: &RunCx, input: &ReconstructInput, cfg: &IngestConfig) -> Result<IngestResult> {
    let given = [
        !input.images.is_empty(),
        input.zip.is_some(),
        input.video.is_some(),
        input.dataset.is_some(),
    ];
    ensure!(
        given.iter().filter(|&&g| g).count() == 1,
        "give exactly one of `images`, `zip`, `video` and `dataset`"
    );
    let max = input.max_frames as usize;
    if let Some(id) = &input.video {
        ensure!(is_artifact_id(id), "video must be an artifact id");
        let mut tmp = None;
        let path = match cx.artifacts.local_path(id).filter(|p| p.is_file()) {
            Some(p) => p,
            None => {
                let f = tempfile::NamedTempFile::new()?;
                std::fs::write(f.path(), cx.artifacts.get(id).await?)?;
                let p = f.path().to_path_buf();
                tmp = Some(f);
                p
            }
        };
        // `ingest_video`'s future is not `Send` (it holds a std mpsc
        // receiver across awaits): drive it on a blocking thread.
        let (cfg, cancel) = (cfg.clone(), cx.cancel.clone());
        let r = tokio::task::spawn_blocking(move || {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?
                .block_on(ingest_video(&path, &cfg, &cancel))
        })
        .await?;
        drop(tmp);
        return r;
    }
    let dir = tempfile::tempdir()?;
    let paths: Vec<PathBuf> = if !input.images.is_empty() {
        let mut out = Vec::new();
        for (i, k) in crate::pose::sample_indices(input.images.len(), max)
            .into_iter()
            .enumerate()
        {
            let id = &input.images[k];
            ensure!(is_artifact_id(id), "images must be artifact ids");
            let p = match cx.artifacts.local_path(id).filter(|p| p.is_file()) {
                Some(p) => p,
                None => {
                    let p = dir.path().join(format!("{i:05}_{id}"));
                    std::fs::write(&p, cx.artifacts.get(id).await?)?;
                    p
                }
            };
            out.push(p);
        }
        out
    } else if let Some(id) = &input.zip {
        ensure!(is_artifact_id(id), "zip must be an artifact id");
        let bytes = cx.artifacts.get(id).await?;
        let root = dir.path().to_path_buf();
        tokio::task::spawn_blocking(move || -> Result<Vec<PathBuf>> {
            let mut z = zip::ZipArchive::new(std::io::Cursor::new(bytes))
                .context("the upload is not a zip file")?;
            let mut names: Vec<String> = z
                .file_names()
                .filter_map(|n| n.ok().map(|n| n.into_owned()))
                .filter(|n| is_image_name(n) && !n.starts_with("__MACOSX"))
                .collect();
            names.sort();
            let mut out = Vec::new();
            for (i, k) in crate::pose::sample_indices(names.len(), max)
                .into_iter()
                .enumerate()
            {
                let mut f = z.by_name(&names[k])?;
                let base = Path::new(&names[k])
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("frame.jpg")
                    .to_string();
                let p = root.join(format!("{i:05}_{base}"));
                let mut w = std::fs::File::create(&p)?;
                std::io::copy(&mut f, &mut w)?;
                out.push(p);
            }
            Ok(out)
        })
        .await??
    } else {
        let name = input.dataset.as_deref().unwrap_or_default();
        ensure!(
            Path::new(name)
                .components()
                .all(|c| matches!(c, std::path::Component::Normal(_))),
            "dataset must be a relative path"
        );
        let base = match std::env::var_os("SPL_DATASETS") {
            Some(p) => PathBuf::from(p),
            None => cx
                .resources
                .get::<DataDir>()
                .map(|d| d.0.join("datasets"))
                .ok_or_else(|| anyhow!("this host has no data directory"))?,
        };
        let files = image_files(&base.join(name))?;
        crate::pose::sample_indices(files.len(), max)
            .into_iter()
            .map(|i| files[i].clone())
            .collect()
    };
    ensure!(!paths.is_empty(), "no images found");
    let cfg = cfg.clone();
    let cancel = cx.cancel.clone();
    let r = tokio::task::spawn_blocking(move || ingest_images(&paths, &cfg, &cancel)).await??;
    drop(dir);
    Ok(r)
}

/// `$SPL_VGGT_WEIGHTS`, else `<data dir>/models/vggt-omega/…`.
fn weights_path(cx: &RunCx) -> Result<PathBuf> {
    let p = match std::env::var_os("SPL_VGGT_WEIGHTS") {
        Some(p) => PathBuf::from(p),
        None => VggtEngine::default_weights(
            &cx.resources
                .get::<DataDir>()
                .map(|d| d.0.clone())
                .ok_or_else(|| anyhow!("this host has no data directory"))?
                .join("models")
                .join("vggt-omega"),
        ),
    };
    ensure!(
        p.is_file(),
        "VGGT-Omega weights not found at {} (set SPL_VGGT_WEIGHTS)",
        p.display()
    );
    Ok(p)
}
