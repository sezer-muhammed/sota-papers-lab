//! The `nvs-colmapfree` protocol (Report section 6) and the video end-to-end
//! run, on the GPU host (`#[ignore]`), driven by environment variables:
//!
//! | Variable | Default | Meaning |
//! |---|---|---|
//! | `SPL_RECON_SCENE` | `mipnerf360/garden` | dataset under `$SPL_DATASETS` |
//! | `SPL_RECON_IMAGES` | `images_4` | protocol image folder |
//! | `SPL_RECON_VIDEO` | none | a video file instead of the image folder |
//! | `SPL_RECON_PRESET` | `quality` | `fast` or `quality` |
//! | `SPL_RECON_CONFIG` | none | `PipelineConfig` JSON patch (ablations) |
//! | `SPL_RECON_HOLDOUT` | 8 | every n-th frame held out (0 = none) |
//! | `SPL_RECON_TTR` | 200 | test-time pose refinement iterations per view |
//! | `SPL_RECON_MAX_FRAMES` | 300 | frames (video keyframes) |
//! | `SPL_RECON_TAG` | `<scene>-<preset>` | output folder name |
//! | `SPL_VGGT_WEIGHTS` | — | engine weights |
//!
//! End to end = from the first JPEG read (or the video probe) to
//! `splat.spz` written to disk. Writes `/out/image-to-3d/reconstruct/<tag>/`:
//! `result.json`, `renders_before/` and `renders/` (held-out views before and
//! after test-time refinement, scored by `gs-metrics` for LPIPS), exports.

use std::path::PathBuf;
use std::time::Instant;

use spl_core::CancellationToken;

use super::pipeline::{Event, PipelineConfig, ReconPreset, reconstruct};
use super::{image_files, to_frames};
use crate::ingest::{IngestConfig, ingest_images, ingest_video};

fn env(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

fn merge(base: &mut serde_json::Value, patch: serde_json::Value) {
    match (base, patch) {
        (serde_json::Value::Object(b), serde_json::Value::Object(p)) => {
            for (k, v) in p {
                merge(b.entry(k).or_insert(serde_json::Value::Null), v);
            }
        }
        (b, p) => *b = p,
    }
}

#[test]
#[ignore = "GPU, weights and dataset: run on the GPU host"]
fn gpu_reconstruct_protocol() {
    let scene_name = env("SPL_RECON_SCENE", "mipnerf360/garden");
    let preset_name = env("SPL_RECON_PRESET", "quality");
    let preset: ReconPreset =
        serde_json::from_value(serde_json::Value::String(preset_name.clone())).expect("preset");
    let mut cfg_json = serde_json::to_value(PipelineConfig::preset(preset)).unwrap();
    if let Ok(p) = std::env::var("SPL_RECON_CONFIG") {
        merge(
            &mut cfg_json,
            serde_json::from_str(&p).expect("SPL_RECON_CONFIG"),
        );
    }
    let cfg: PipelineConfig = serde_json::from_value(cfg_json).expect("config");
    let holdout: usize = env("SPL_RECON_HOLDOUT", "8").parse().unwrap();
    let ttr: u32 = env("SPL_RECON_TTR", "200").parse().unwrap();
    let max_frames: u32 = env("SPL_RECON_MAX_FRAMES", "300").parse().unwrap();
    let video = std::env::var("SPL_RECON_VIDEO").ok().map(PathBuf::from);
    let short = scene_name.rsplit('/').next().unwrap().to_string();
    let tag = env("SPL_RECON_TAG", &format!("{short}-{preset_name}"));
    let out = PathBuf::from("/out/image-to-3d/reconstruct").join(&tag);
    std::fs::create_dir_all(&out).unwrap();
    let weights = PathBuf::from(env(
        "SPL_VGGT_WEIGHTS",
        "/models/vggt-omega/vggt_omega_1b_512.safetensors",
    ));
    let cancel = CancellationToken::new();

    // Ingest: the clock starts here (JPEG files or the video on disk).
    let t0 = Instant::now();
    let ingest_cfg = IngestConfig {
        // Never downscale the protocol images (images_4 / images_2).
        trainer_long_side: 100_000,
        max_frames,
        ..Default::default()
    };
    let ingested = match &video {
        Some(v) => tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(ingest_video(v, &ingest_cfg, &cancel))
            .unwrap(),
        None => {
            let dir = PathBuf::from(env("SPL_DATASETS", "/datasets"))
                .join(&scene_name)
                .join(env("SPL_RECON_IMAGES", "images_4"));
            let files = image_files(&dir).unwrap();
            ingest_images(&files, &ingest_cfg, &cancel).unwrap()
        }
    };
    let ingest_s = t0.elapsed().as_secs_f64();
    let stats = ingested.stats;
    let frames = to_frames(ingested);
    let n_frames = frames.names.len();
    eprintln!(
        "{tag}: {n_frames} frames ({}x{}) ingested in {ingest_s:.2} s (decoded {})",
        frames.trainer[0].width, frames.trainer[0].height, stats.decoded
    );

    let mut first_points_s = None;
    let mut snapshots = Vec::new();
    let mut last_log = Instant::now();
    let mut on_event = |e: Event| match e {
        Event::Progress(_, msg) => {
            if last_log.elapsed().as_secs_f64() > 10.0 {
                last_log = Instant::now();
                eprintln!("[{:7.1}s] {msg}", t0.elapsed().as_secs_f64());
            }
        }
        Event::Log(msg) => eprintln!("[{:7.1}s] {msg}", t0.elapsed().as_secs_f64()),
        Event::Cameras(_) => {}
        Event::Points(_, n) => {
            first_points_s = Some(t0.elapsed().as_secs_f64());
            eprintln!(
                "[{:7.1}s] points announced ({n})",
                t0.elapsed().as_secs_f64()
            );
        }
        Event::Snapshot { spz, t, n } => {
            snapshots.push((t0.elapsed().as_secs_f64(), t, n, spz.len()));
        }
    };
    let mut r = reconstruct(&weights, frames, &cfg, holdout, &cancel, &mut on_event).unwrap();
    std::fs::write(out.join("splat.spz"), &r.spz).unwrap();
    let e2e_s = t0.elapsed().as_secs_f64();
    eprintln!(
        "{tag}: END TO END {e2e_s:.1} s (ingest {ingest_s:.1}, engine {:.1} + load {:.1}, setup {:.1}, init {:.2}, prune {:.2}, train {:.1}, export {:.1}); {} Gaussians, {} eager steps",
        r.times.engine_s,
        r.times.engine_load_s,
        r.times.setup_s,
        r.times.init_s,
        r.times.prune_s,
        r.times.train_s,
        r.times.export_s,
        r.cloud.len(),
        r.eager_steps
    );
    r.write_exports(&out.join("exports")).unwrap();

    let mut result = serde_json::json!({
        "tag": tag,
        "scene": scene_name,
        "video": video.as_ref().map(|v| v.display().to_string()),
        "preset": preset_name,
        "config": cfg,
        "frames": n_frames,
        "holdout_every": holdout,
        "e2e_s": e2e_s,
        "ingest_s": ingest_s,
        "times": r.times,
        "first_points_s": first_points_s,
        "snapshots": snapshots.iter().map(|s| serde_json::json!({"at_s": s.0, "t": s.1, "n": s.2, "bytes": s.3})).collect::<Vec<_>>(),
        "init_gaussians": r.init_gaussians,
        "prune": {"before": r.prune.before, "after": r.prune.after},
        "final_gaussians": r.cloud.len(),
        "iterations": r.iterations,
        "eager_steps": r.eager_steps,
        "peak_vram_mib": r.peak_vram_bytes >> 20,
        "spz_bytes": r.spz.len(),
    });
    if r.test.is_some() {
        let (before, _) = r.evaluate_test(Some(&out.join("renders_before"))).unwrap();
        eprintln!(
            "{tag}: held-out BEFORE test-time refinement: PSNR {:.3} SSIM {:.4}",
            before.psnr, before.ssim
        );
        let ttr_s = r.refine_test(ttr, 1e-3, 1e-3).unwrap();
        let (after, per) = r.evaluate_test(Some(&out.join("renders"))).unwrap();
        eprintln!(
            "{tag}: held-out AFTER {ttr}-iteration test-time refinement ({ttr_s:.1} s): PSNR {:.3} SSIM {:.4}",
            after.psnr, after.ssim
        );
        let names: Vec<String> = r
            .test
            .as_ref()
            .unwrap()
            .views
            .iter()
            .map(|v| v.name.clone())
            .collect();
        result["before"] = serde_json::to_value(before).unwrap();
        result["after"] = serde_json::to_value(after).unwrap();
        result["ttr_iters"] = ttr.into();
        result["ttr_s"] = ttr_s.into();
        result["per_view_after"] = names
            .iter()
            .zip(&per)
            .map(|(n, m)| serde_json::json!({"name": n, "psnr": m.psnr, "ssim": m.ssim}))
            .collect::<Vec<_>>()
            .into();
    }
    std::fs::write(
        out.join("result.json"),
        serde_json::to_vec_pretty(&result).unwrap(),
    )
    .unwrap();
    eprintln!("{}", serde_json::to_string(&result).unwrap());
}
