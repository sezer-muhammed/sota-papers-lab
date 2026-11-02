//! The `nvs-colmapfree` protocol (Report section 6) and the video end-to-end
//! run, on the GPU host (`#[ignore]`), driven by environment variables:
//!
//! | Variable | Default | Meaning |
//! |---|---|---|
//! | `SPL_RECON_SCENE` | `mipnerf360/garden` | dataset under `$SPL_DATASETS` |
//! | `SPL_RECON_IMAGES` | `images_4` | protocol image folder |
//! | `SPL_RECON_VIDEO` | none | a video file instead of the image folder |
//! | `SPL_RECON_PRESET` | `quality` | `fast` or `quality` |
//! | `SPL_RECON_CONFIG` | none | `PipelineConfig` JSON patch, or an array of patches run one after another on the same ingest (ablations; tags get `-<i>`) |
//! | `SPL_RECON_HOLDOUT` | 8 | every n-th frame held out (0 = none) |
//! | `SPL_RECON_TTR` | 200 | test-time pose refinement iterations per view |
//! | `SPL_RECON_TTR_LR` | 3e-3 | its rotation LR (translation: same × extent); bicycle Fast: 1e-3 21.47 dB, 3e-3 22.25, 1e-2 20.06 |
//! | `SPL_RECON_MAX_FRAMES` | all | frames (video keyframes) |
//! | `SPL_RECON_TAG` | `<scene>-<preset>` | output folder name |
//! | `SPL_VGGT_WEIGHTS` | — | engine weights |
//! | `SPL_RECON_GT_POSES` | 0 | 1 = replace the engine's cameras by the dataset's COLMAP cameras, Sim(3)-aligned into the engine's frame (diagnostic: isolates pose error) |
//!
//! End to end = from the first JPEG read (or the video probe) to
//! `splat.spz` written to disk. Writes `/out/image-to-3d/reconstruct/<tag>/`:
//! `result.json`, `renders_before/` and `renders/` (held-out views before and
//! after test-time refinement, scored by `gs-metrics` for LPIPS), exports.

use std::path::PathBuf;
use std::time::Instant;

use spl_core::CancellationToken;

use super::align::{M3, V3, poses_into_frame};
use super::pipeline::{Event, Frames, PipelineConfig, ReconPreset, reconstruct_with};
use super::scene::Posed;
use super::{image_files, to_frames};
use crate::ingest::{IngestConfig, ingest_images, ingest_video};
use crate::io::scene::{LoadOptions, load_colmap_scene};

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
    let patches: Vec<serde_json::Value> = match std::env::var("SPL_RECON_CONFIG") {
        Ok(p) => match serde_json::from_str(&p).expect("SPL_RECON_CONFIG") {
            serde_json::Value::Array(a) => a,
            v => vec![v],
        },
        Err(_) => vec![serde_json::json!({})],
    };
    let max_frames: u32 = env("SPL_RECON_MAX_FRAMES", "100000").parse().unwrap();
    let video = std::env::var("SPL_RECON_VIDEO").ok().map(PathBuf::from);
    let short = scene_name.rsplit('/').next().unwrap().to_string();
    let tag = env("SPL_RECON_TAG", &format!("{short}-{preset_name}"));
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
            let all = image_files(&dir).unwrap();
            let files: Vec<PathBuf> = crate::pose::sample_indices(all.len(), max_frames as usize)
                .into_iter()
                .map(|i| all[i].clone())
                .collect();
            ingest_images(&files, &ingest_cfg, &cancel).unwrap()
        }
    };
    let ingest_s = t0.elapsed().as_secs_f64();
    let stats = ingested.stats;
    let frames = to_frames(ingested);
    eprintln!(
        "{tag}: {} frames ({}x{}) ingested in {ingest_s:.2} s (decoded {})",
        frames.names.len(),
        frames.trainer[0].width,
        frames.trainer[0].height,
        stats.decoded
    );
    let n = patches.len();
    for (i, patch) in patches.into_iter().enumerate() {
        let mut cfg_json = serde_json::to_value(PipelineConfig::preset(preset)).unwrap();
        merge(&mut cfg_json, patch);
        let cfg: PipelineConfig = serde_json::from_value(cfg_json).expect("config");
        let t = if n == 1 {
            tag.clone()
        } else {
            format!("{tag}-{i}")
        };
        let f = Frames {
            names: frames.names.clone(),
            engine: frames.engine.clone(),
            trainer: frames.trainer.clone(),
        };
        run_one(
            &t,
            &scene_name,
            &preset_name,
            video.as_ref(),
            cfg,
            f,
            ingest_s,
            &cancel,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn run_one(
    tag: &str,
    scene_name: &str,
    preset_name: &str,
    video: Option<&PathBuf>,
    cfg: PipelineConfig,
    frames: Frames,
    ingest_s: f64,
    cancel: &CancellationToken,
) {
    let holdout: usize = env("SPL_RECON_HOLDOUT", "8").parse().unwrap();
    let ttr: u32 = env("SPL_RECON_TTR", "200").parse().unwrap();
    let ttr_lr: f32 = env("SPL_RECON_TTR_LR", "3e-3").parse().unwrap();
    let out = PathBuf::from("/out/image-to-3d/reconstruct").join(tag);
    std::fs::create_dir_all(&out).unwrap();
    let weights = PathBuf::from(env(
        "SPL_VGGT_WEIGHTS",
        "/models/vggt-omega/vggt_omega_1b_512.safetensors",
    ));
    let n_frames = frames.names.len();
    // The clock continues from ingest (measured once per process).
    let t0 = Instant::now() - std::time::Duration::from_secs_f64(ingest_s);
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
    let gt_poses = env("SPL_RECON_GT_POSES", "0") == "1";
    let mut hook = |p: &mut Posed| -> spl_core::Result<()> {
        let root = PathBuf::from(env("SPL_DATASETS", "/datasets")).join(scene_name);
        let load = LoadOptions {
            images_dir: env("SPL_RECON_IMAGES", "images_4"),
            ..Default::default()
        };
        let scene = load_colmap_scene(&root, &load)?;
        let by_name: std::collections::HashMap<&str, &crate::io::scene::View> =
            scene.views.iter().map(|v| (v.name.as_str(), v)).collect();
        let mut idx = Vec::new();
        let mut ours: Vec<(M3, V3)> = Vec::new();
        let mut gt: Vec<(M3, V3)> = Vec::new();
        for (i, n) in p.names.iter().enumerate() {
            if let Some(v) = by_name.get(n.as_str()) {
                let (r, t) = p.pose(i);
                ours.push((r.map(|row| row.map(f64::from)), t.map(f64::from)));
                gt.push((v.r_cw, v.t_cw));
                idx.push((i, v.fx / v.width as f64, v.fy / v.height as f64));
            }
        }
        anyhow::ensure!(idx.len() == p.names.len(), "COLMAP misses frames");
        let (w, h) = (p.cams.width as f64, p.cams.height as f64);
        for ((i, fxn, fyn), (r, t)) in idx.into_iter().zip(poses_into_frame(&ours, &gt)) {
            let e = &mut p.cams.extrinsics[i];
            for k in 0..3 {
                e[4 * k] = r[k][0] as f32;
                e[4 * k + 1] = r[k][1] as f32;
                e[4 * k + 2] = r[k][2] as f32;
                e[4 * k + 3] = t[k] as f32;
            }
            p.cams.intrinsics[i][0] = (fxn * w) as f32;
            p.cams.intrinsics[i][4] = (fyn * h) as f32;
        }
        eprintln!("diagnostic: engine cameras replaced by aligned COLMAP cameras");
        Ok(())
    };
    let adjust: Option<&mut super::pipeline::PosedHook<'_>> =
        if gt_poses { Some(&mut hook) } else { None };
    let mut r = reconstruct_with(
        &weights,
        frames,
        &cfg,
        holdout,
        cancel,
        &mut on_event,
        adjust,
    )
    .unwrap();
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
    let train = r.evaluate_train(8).unwrap();
    eprintln!(
        "{tag}: training views (every 8th): PSNR {:.3} SSIM {:.4}",
        train.psnr, train.ssim
    );

    let mut result = serde_json::json!({
        "tag": tag,
        "scene": scene_name,
        "video": video.map(|v| v.display().to_string()),
        "preset": preset_name,
        "gt_poses": gt_poses,
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
        "train_views": train,
    });
    if r.test.is_some() {
        let (before, _) = r.evaluate_test(Some(&out.join("renders_before"))).unwrap();
        eprintln!(
            "{tag}: held-out BEFORE test-time refinement: PSNR {:.3} SSIM {:.4}",
            before.psnr, before.ssim
        );
        let ttr_s = r.refine_test(ttr, ttr_lr, ttr_lr).unwrap();
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
        result["ttr_lr"] = ttr_lr.into();
        result["ttr_s"] = ttr_s.into();
        result["per_view_after"] = names
            .iter()
            .zip(&per)
            .map(|(n, m)| serde_json::json!({"name": n, "psnr": m.psnr, "ssim": m.ssim}))
            .collect::<Vec<_>>()
            .into();
    }
    r.release();
    std::fs::write(
        out.join("result.json"),
        serde_json::to_vec_pretty(&result).unwrap(),
    )
    .unwrap();
    eprintln!("{}", serde_json::to_string(&result).unwrap());
}
