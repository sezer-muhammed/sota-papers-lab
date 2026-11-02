//! Full training runs on a dataset (GPU host only, `#[ignore]`), driven by
//! environment variables so the benchmark protocol can call them:
//!
//! | Variable | Default | Meaning |
//! |---|---|---|
//! | `SPL_TRAIN_SCENE` | `mipnerf360/garden` | dataset under `$SPL_DATASETS` |
//! | `SPL_TRAIN_IMAGES` | `images_4` | protocol image folder |
//! | `SPL_TRAIN_PRESET` | `quality` | preset name (snake case) |
//! | `SPL_TRAIN_ITERS` | preset | iteration override |
//! | `SPL_TRAIN_EAGER` | 0 | 1 = no CUDA graphs |
//! | `SPL_TRAIN_MAX_GAUSSIANS` | VRAM | capacity ceiling |
//! | `SPL_TRAIN_TAG` | `<scene>-<preset>` | output folder name |
//!
//! Writes `/out/image-to-3d/train/<tag>/`: `renders/<view>.png` (test views,
//! scored by `gs-metrics`), `result.json`, `profile.json` (per-stage ms of
//! eager iterations on the final model), `splat.spz`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use spl_core::CancellationToken;
use spl_gpu::cudarc::driver::CudaEvent;

use super::*;
use crate::eval::evaluate;
use crate::export::spz::{SpzOptions, write_spz};
use crate::io::scene::{LoadOptions, load_colmap_scene};
use crate::kernels::{preprocess, raster as rk, sort};

fn env(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

#[test]
#[ignore = "GPU and dataset: run on the GPU host"]
fn gpu_train_dataset() {
    let scene_name = env("SPL_TRAIN_SCENE", "mipnerf360/garden");
    let preset: PresetName = serde_json::from_value(serde_json::Value::String(env(
        "SPL_TRAIN_PRESET",
        "quality",
    )))
    .expect("preset name");
    let iterations = std::env::var("SPL_TRAIN_ITERS")
        .ok()
        .map(|v| v.parse::<u32>().expect("SPL_TRAIN_ITERS"));
    let short = scene_name.rsplit('/').next().unwrap().to_string();
    let tag = env(
        "SPL_TRAIN_TAG",
        &format!("{short}-{}", env("SPL_TRAIN_PRESET", "quality")),
    );
    let out = PathBuf::from("/out/image-to-3d/train").join(&tag);
    std::fs::create_dir_all(&out).unwrap();
    let root = PathBuf::from(env("SPL_DATASETS", "/datasets")).join(&scene_name);
    let load = LoadOptions {
        images_dir: env("SPL_TRAIN_IMAGES", "images_4"),
        ..Default::default()
    };
    let t0 = Instant::now();
    let scene = load_colmap_scene(&root, &load).unwrap();
    let load_s = t0.elapsed().as_secs_f64();
    eprintln!(
        "{tag}: {} views ({} train / {} test), {} points, {}x{}, load {load_s:.1}s",
        scene.views.len(),
        scene.train.len(),
        scene.test.len(),
        scene.points.xyz.len(),
        scene.views[0].width,
        scene.views[0].height
    );
    let cfg = TrainConfig {
        preset,
        iterations,
        max_gaussians: std::env::var("SPL_TRAIN_MAX_GAUSSIANS")
            .ok()
            .map(|v| v.parse().expect("SPL_TRAIN_MAX_GAUSSIANS")),
        overrides: PresetOverrides {
            scene: Some(short.clone()),
            ..Default::default()
        },
        ..Default::default()
    };
    let gpu = Gpu::open().unwrap();
    let (free, _) = gpu.mem_info().unwrap();
    eprintln!("free VRAM at start: {} MiB", free >> 20);
    let t1 = Instant::now();
    let mut trainer = Trainer::new(&gpu, scene, cfg).unwrap();
    let setup_s = t1.elapsed().as_secs_f64();
    eprintln!(
        "capacity {} Gaussians, raster {:?}, setup {setup_s:.1}s",
        trainer.g.cap,
        trainer.raster.as_ref().unwrap().capacity()
    );
    trainer.eager = env("SPL_TRAIN_EAGER", "0") == "1";
    let mut last_log = 0u32;
    let summary = trainer
        .run(&CancellationToken::new(), &mut |p| {
            if p.t >= last_log + 1000 || p.t == p.total {
                last_log = p.t;
                eprintln!(
                    "t {:>6}/{}  n {:>8}  {:>6.1} it/s  loss {:?}",
                    p.t, p.total, p.n_gaussians, p.iterations_per_s, p.loss
                );
            }
        })
        .unwrap();
    eprintln!("{summary:?}");
    let t2 = Instant::now();
    let report = evaluate(&mut trainer, Some(&out.join("renders"))).unwrap();
    let eval_s = t2.elapsed().as_secs_f64();
    eprintln!(
        "{tag}: PSNR {:.3}  SSIM {:.4} (valid {:.4})  eval {eval_s:.1}s",
        report.mean.psnr, report.mean.ssim, report.mean.ssim_valid
    );
    let profile = profile_iterations(&mut trainer, 30);
    eprintln!("profile (ms): {profile:#?}");
    let (cloud, summary) = trainer.finish().unwrap();
    let spz = write_spz(&cloud, &SpzOptions::default()).unwrap();
    std::fs::write(out.join("splat.spz"), &spz).unwrap();
    let result = serde_json::json!({
        "tag": tag,
        "scene": scene_name,
        "preset": preset,
        "iterations": summary.iterations,
        "train_s": summary.train_s,
        "setup_s": setup_s,
        "load_s": load_s,
        "final_gaussians": summary.final_gaussians,
        "peak_vram_mib": summary.peak_vram_bytes as f64 / (1u64 << 20) as f64,
        "graph_captures": summary.graph_captures,
        "densify_steps": summary.densify_steps,
        "raster_regrows": summary.raster_regrows,
        "final_loss": summary.final_loss,
        "psnr": report.mean.psnr,
        "ssim": report.mean.ssim,
        "ssim_valid": report.mean.ssim_valid,
        "per_view": report.per_view,
        "spz_bytes": spz.len(),
        "profile_ms": profile,
    });
    std::fs::write(
        out.join("result.json"),
        serde_json::to_string_pretty(&result).unwrap(),
    )
    .unwrap();
    write_profile(&out.join("profile.json"), &result["profile_ms"]);
    assert!(
        report.mean.psnr > 15.0,
        "PSNR {:.2} is broken",
        report.mean.psnr
    );
}

fn write_profile(path: &Path, v: &serde_json::Value) {
    std::fs::write(path, serde_json::to_string_pretty(v).unwrap()).unwrap();
}

/// Mean ms per stage over `n` eager training iterations at full resolution
/// and degree 3 on the current model (events between the stage functions),
/// with an early-phase optimizer plan (every group due: the dense worst case).
/// The iterations really step the model (the run is over anyway).
fn profile_iterations(tr: &mut Trainer, n: u32) -> std::collections::BTreeMap<String, f64> {
    const STAGES: [&str; 10] = [
        "frame_begin",
        "preprocess_fwd",
        "sort_depth",
        "bin_tiles",
        "raster_fwd",
        "resample_target",
        "loss_fwd_bwd",
        "raster_bwd",
        "preprocess_bwd",
        "adam",
    ];
    let gpu = tr.gpu.clone();
    let s = gpu.stream.clone();
    let ev = || -> CudaEvent {
        gpu.ctx
            .new_event(Some(
                spl_gpu::cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT,
            ))
            .unwrap()
    };
    let events: Vec<CudaEvent> = (0..=STAGES.len()).map(|_| ev()).collect();
    let mut acc = vec![0f64; STAGES.len()];
    let mut total = 0f64;
    let t_end = tr.t;
    let mut counted = 0u32;
    for i in 0..n + 3 {
        let t = t_end + 1 + i;
        let view = tr.order.next_view();
        let plan = tr.adam.plan(16);
        assert_eq!(plan.due_mask, 0x3f, "profile wants every group due");
        let params = tr.iter_params(t, view, 3, 1, plan);
        tr.ring.push(&params).unwrap();
        let raster = tr.raster.as_mut().unwrap();
        let k = &raster.kernels;
        events[0].record(&s).unwrap();
        preprocess::frame_begin(k, &s, &tr.ring, &tr.scene_gpu.cams, &mut raster.frame).unwrap();
        events[1].record(&s).unwrap();
        preprocess::preprocess_fwd(k, &s, &mut tr.g, &mut raster.frame, &mut raster.g2d).unwrap();
        events[2].record(&s).unwrap();
        sort::sort_depth(k, &s, &mut raster.sorter, &mut raster.frame).unwrap();
        events[3].record(&s).unwrap();
        sort::bin_tiles(k, &s, &mut raster.sorter, &mut raster.frame).unwrap();
        events[4].record(&s).unwrap();
        rk::raster_fwd(k, &s, &mut raster.frame, RenderMode::Train, &mut raster.out).unwrap();
        events[5].record(&s).unwrap();
        resample_target(
            &gpu,
            &tr.loss.kernels,
            &raster.frame.iter,
            &tr.scene_gpu.images,
            &mut tr.target,
        )
        .unwrap();
        events[6].record(&s).unwrap();
        tr.loss
            .forward_backward(
                &gpu,
                &raster.frame.iter,
                &raster.out.rgb,
                &tr.target,
                &mut raster.pixel_grads.d_rgb,
            )
            .unwrap();
        events[7].record(&s).unwrap();
        rk::raster_bwd(
            k,
            &s,
            &mut raster.frame,
            &raster.out,
            &raster.pixel_grads,
            &mut raster.g2d,
        )
        .unwrap();
        events[8].record(&s).unwrap();
        preprocess::preprocess_bwd(
            k,
            &s,
            &mut tr.g,
            &raster.frame,
            &raster.g2d,
            GradSink::Write(&mut tr.grads),
            None,
            None,
        )
        .unwrap();
        events[9].record(&s).unwrap();
        tr.adam
            .step(&gpu, &mut tr.g, &mut tr.grads, &raster.frame.iter)
            .unwrap();
        events[10].record(&s).unwrap();
        raster.end_frame(&gpu, &tr.status).unwrap();
        tr.ring.enqueued(&s).unwrap();
        s.synchronize().unwrap();
        if i < 3 {
            continue; // warm-up
        }
        counted += 1;
        for j in 0..STAGES.len() {
            acc[j] += events[j].elapsed_ms(&events[j + 1]).unwrap() as f64;
        }
        total += events[0].elapsed_ms(&events[STAGES.len()]).unwrap() as f64;
    }
    tr.t = t_end + n + 3;
    let mut out: std::collections::BTreeMap<String, f64> = STAGES
        .iter()
        .zip(&acc)
        .map(|(k, v)| (k.to_string(), v / counted as f64))
        .collect();
    out.insert("total".into(), total / counted as f64);
    out.insert("n_gaussians".into(), tr.g.n as f64);
    let st = tr
        .raster
        .as_ref()
        .unwrap()
        .frame
        .iter
        .read_status(&gpu)
        .unwrap();
    out.insert("n_visible_last".into(), st.n_visible as f64);
    out.insert("n_instances_last".into(), st.n_instances as f64);
    out
}
