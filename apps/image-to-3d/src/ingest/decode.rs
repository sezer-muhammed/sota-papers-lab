//! Photo ingest: decode, EXIF orientation and focal prior, GPU resize.
//! Owner: ingest (A0).

use std::path::{Path, PathBuf};
use std::time::Instant;

use image::{DynamicImage, ImageDecoder, ImageReader};
use rayon::prelude::*;
use spl_core::{CancellationToken, Context, Result, bail};

use super::gpu::{IngestGpu, Resident};
use super::{
    FocalPrior, IngestConfig, IngestFrame, IngestResult, IngestStats, Rgb8, SharpnessSample, exif,
    fit_size, median_prior,
};

/// Decodes one JPEG, PNG or WebP with its EXIF orientation applied; returns
/// the pixels and the 35 mm focal prior when the file has one.
pub fn decode_image(path: &Path) -> Result<(Rgb8, Option<FocalPrior>)> {
    let mut decoder = ImageReader::open(path)
        .with_context(|| format!("open {}", path.display()))?
        .with_guessed_format()
        .with_context(|| format!("read {}", path.display()))?
        .into_decoder()
        .with_context(|| format!("decode {}", path.display()))?;
    let focal = decoder
        .exif_metadata()
        .ok()
        .flatten()
        .map(|e| exif::parse_focal(&e))
        .and_then(|f| f.focal_35mm)
        .map(|focal_35mm| FocalPrior { focal_35mm });
    let orientation = decoder
        .orientation()
        .unwrap_or(image::metadata::Orientation::NoTransforms);
    let mut img = DynamicImage::from_decoder(decoder)
        .with_context(|| format!("decode {}", path.display()))?;
    img.apply_orientation(orientation);
    let rgb = img.into_rgb8();
    Ok((
        Rgb8 {
            width: rgb.width(),
            height: rgb.height(),
            data: rgb.into_raw(),
        },
        focal,
    ))
}

/// Scores and resizes one decoded frame on the GPU.
pub(super) fn finish_frame(
    g: &mut IngestGpu,
    cfg: &IngestConfig,
    img: &Rgb8,
) -> Result<(f64, Rgb8, Rgb8)> {
    let r = g.upload(img.width, img.height, &img.data)?;
    let sharpness = g.sharpness(r)?;
    let (engine, trainer) = resize_both(g, cfg, r, false)?;
    Ok((sharpness, engine, trainer))
}

/// Resizes the resident (or kept) frame to the engine and trainer sizes.
pub(super) fn resize_both(
    g: &mut IngestGpu,
    cfg: &IngestConfig,
    r: Resident,
    kept: bool,
) -> Result<(Rgb8, Rgb8)> {
    let (ew, eh) = fit_size(r.width, r.height, cfg.engine_long_side, cfg.engine_multiple);
    let (tw, th) = fit_size(r.width, r.height, cfg.trainer_long_side, 1);
    if kept {
        Ok((g.resize_kept(r, ew, eh)?, g.resize_kept(r, tw, th)?))
    } else {
        Ok((g.resize(r, ew, eh)?, g.resize(r, tw, th)?))
    }
}

/// Photo ingest. Decodes in parallel on the CPU (a chunk per rayon pool size,
/// to bound memory), resizes on the GPU; keeps every photo in input order.
/// Blocking: call from `spawn_blocking`.
pub fn ingest_images(
    paths: &[PathBuf],
    cfg: &IngestConfig,
    cancel: &CancellationToken,
) -> Result<IngestResult> {
    let started = Instant::now();
    let mut g = IngestGpu::open()?;
    let mut frames = Vec::with_capacity(paths.len());
    let mut sharpness = Vec::with_capacity(paths.len());
    let chunk = rayon::current_num_threads().max(1);
    for (c, group) in paths.chunks(chunk).enumerate() {
        if cancel.is_cancelled() {
            bail!("cancelled");
        }
        let decoded: Vec<Result<(Rgb8, Option<FocalPrior>)>> =
            group.par_iter().map(|p| decode_image(p)).collect();
        for (k, (path, d)) in group.iter().zip(decoded).enumerate() {
            let (img, focal_prior) = d?;
            let index = (c * chunk + k) as u32;
            let (s, engine, trainer) = finish_frame(&mut g, cfg, &img)?;
            sharpness.push(SharpnessSample {
                index,
                timestamp_s: None,
                sharpness: s,
            });
            frames.push(IngestFrame {
                source: path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default(),
                index,
                timestamp_s: None,
                sharpness: s,
                source_size: (img.width, img.height),
                focal_prior,
                engine,
                trainer,
            });
        }
    }
    let seconds = started.elapsed().as_secs_f64();
    let n = frames.len() as u32;
    Ok(IngestResult {
        focal_prior: median_prior(&frames),
        frames,
        sharpness,
        stats: IngestStats {
            decoded: n,
            kept: n,
            seconds,
            decode_fps: n as f64 / seconds.max(1e-9),
            gpu_seconds: g.seconds,
        },
    })
}
