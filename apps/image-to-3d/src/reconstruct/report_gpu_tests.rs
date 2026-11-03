//! Keyframe policy A/B on a video made from a posed dataset, on the GPU
//! host (`#[ignore]`). Owner: web-results.
//!
//! The video holds only training images (index % 8 ≠ 0 of the dataset
//! folder); the held-out set is the dataset's every-8th images, the same for
//! every policy, interleaved with the keyframes so `holdout_every` picks
//! exactly them. Pose accuracy compares the refined training cameras with
//! the dataset's COLMAP cameras (pairwise relative rotation and translation
//! direction errors, AUC up to 3°, 5°, 30° as in VGGT).
//!
//! | Variable | Meaning |
//! |---|---|
//! | `SPL_KF_VIDEO` | the video |
//! | `SPL_KF_SEGMENTS` | JSON `[[start_s, image name], …]` of the video's timeline |
//! | `SPL_KF_MODE` | `motion` (default) or `time` |
//! | `SPL_KF_MAX` | keyframes (default 48) |
//! | `SPL_RECON_SCENE`, `SPL_RECON_IMAGES`, `SPL_RECON_PRESET` | as `gpu_tests` |
//! | `SPL_KF_TAG` | output folder under `/out/image-to-3d/keyframes/` |
//!
//! Writes `result.json` (with the capture and result reports) and the
//! worst-view thumbnails there.

use std::path::PathBuf;
use std::time::Instant;

use spl_core::CancellationToken;

use super::{capture_report, evaluate};
use crate::ingest::{
    IngestConfig, KeyframeMode, Rgb8, ingest_images, ingest_video, resize_area_cpu,
};
use crate::io::images::Rgb8Image;
use crate::io::scene::{LoadOptions, load_colmap_scene};
use crate::reconstruct::align::M3;
use crate::reconstruct::pipeline::{Event, Frames, PipelineConfig, ReconPreset, reconstruct};
use crate::reconstruct::{image_files, to_frames};

fn env(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

type Pose = (M3, [f64; 3]);

fn mm(a: &M3, b: &M3) -> M3 {
    let mut o = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    o
}

fn transpose(a: &M3) -> M3 {
    [0, 1, 2].map(|i| [0, 1, 2].map(|j| a[j][i]))
}

fn centre(p: &Pose) -> [f64; 3] {
    [0, 1, 2].map(|j| -(0..3).map(|i| p.0[i][j] * p.1[i]).sum::<f64>())
}

fn angle(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let n = (a.iter().map(|v| v * v).sum::<f64>() * b.iter().map(|v| v * v).sum::<f64>()).sqrt();
    (d / n.max(1e-12)).clamp(-1.0, 1.0).acos().to_degrees()
}

/// Pairwise max(rotation, translation-direction) errors in degrees.
fn pair_errors(ours: &[Pose], gt: &[Pose], same: &[usize]) -> Vec<f64> {
    let mut e = Vec::new();
    for i in 0..ours.len() {
        for j in i + 1..ours.len() {
            if same[i] == same[j] {
                continue;
            }
            let ra = mm(&ours[i].0, &transpose(&ours[j].0));
            let rb = mm(&gt[i].0, &transpose(&gt[j].0));
            let d = mm(&ra, &transpose(&rb));
            let rot = ((d[0][0] + d[1][1] + d[2][2] - 1.0) / 2.0)
                .clamp(-1.0, 1.0)
                .acos()
                .to_degrees();
            // Direction from i to j in i's camera frame: frame-independent.
            let dir = |p: &[Pose]| {
                let c = centre(&p[j]);
                let r = &p[i].0;
                let t = p[i].1;
                [0, 1, 2].map(|k| r[k][0] * c[0] + r[k][1] * c[1] + r[k][2] * c[2] + t[k])
            };
            e.push(rot.max(angle(dir(ours), dir(gt))));
        }
    }
    e
}

fn auc(errors: &[f64], max_deg: u32) -> f64 {
    if errors.is_empty() {
        return 0.0;
    }
    let n = errors.len() as f64;
    (1..=max_deg)
        .map(|t| errors.iter().filter(|&&e| e < t as f64).count() as f64 / n)
        .sum::<f64>()
        / max_deg as f64
        * 100.0
}

#[test]
#[ignore = "GPU, weights, dataset and a video: run on the GPU host"]
fn gpu_keyframe_policy_ab() {
    let video = PathBuf::from(env("SPL_KF_VIDEO", ""));
    let segments: Vec<(f64, String)> =
        serde_json::from_str(&std::fs::read_to_string(env("SPL_KF_SEGMENTS", "")).unwrap())
            .unwrap();
    let mode: KeyframeMode =
        serde_json::from_value(serde_json::Value::String(env("SPL_KF_MODE", "motion"))).unwrap();
    let max: u32 = env("SPL_KF_MAX", "48").parse().unwrap();
    let scene_name = env("SPL_RECON_SCENE", "mipnerf360/garden");
    let images = env("SPL_RECON_IMAGES", "images_4");
    let preset_name = env("SPL_RECON_PRESET", "fast");
    let preset: ReconPreset =
        serde_json::from_value(serde_json::Value::String(preset_name.clone())).unwrap();
    let tag = env("SPL_KF_TAG", "kf");
    let out = PathBuf::from("/out/image-to-3d/keyframes").join(&tag);
    std::fs::create_dir_all(&out).unwrap();
    let cancel = CancellationToken::new();
    let root = PathBuf::from(env("SPL_DATASETS", "/datasets")).join(&scene_name);

    let t0 = Instant::now();
    let cfg = IngestConfig {
        trainer_long_side: 100_000,
        max_frames: max,
        keyframes: mode,
        ..Default::default()
    };
    let ingested = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(ingest_video(&video, &cfg, &cancel))
        .unwrap();
    let ingest_s = t0.elapsed().as_secs_f64();
    let capture_stats = ingested.capture.clone();
    let source_of = |t: f64| -> String {
        let k = segments.partition_point(|s| s.0 <= t + 1e-6).max(1) - 1;
        segments[k].1.clone()
    };
    let sources: Vec<String> = ingested
        .frames
        .iter()
        .map(|f| source_of(f.timestamp_s.unwrap_or(0.0)))
        .collect();
    let unique = {
        let mut u = sources.clone();
        u.sort();
        u.dedup();
        u.len()
    };
    let kept = ingested.frames.len();
    let train = to_frames(ingested);
    let (tw, th) = (train.trainer[0].width, train.trainer[0].height);

    // Held-out photos: the dataset's every-8th images at the video's size.
    let files = image_files(&root.join(&images)).unwrap();
    let test_files: Vec<PathBuf> = files.iter().step_by(8).cloned().collect();
    let every = 1 + kept.div_ceil(test_files.len());
    let n_test = kept.div_ceil(every - 1).min(test_files.len());
    let picked: Vec<PathBuf> = crate::pose::sample_indices(test_files.len(), n_test)
        .into_iter()
        .map(|i| test_files[i].clone())
        .collect();
    let tests = to_frames(ingest_images(&picked, &IngestConfig::default(), &cancel).unwrap());
    let fit = |img: &Rgb8Image, w: u32, h: u32| {
        let r = resize_area_cpu(
            &Rgb8 {
                width: img.width,
                height: img.height,
                data: img.data.clone(),
            },
            w,
            h,
        );
        Rgb8Image {
            width: r.width,
            height: r.height,
            data: r.data,
        }
    };
    let mut frames = Frames {
        names: Vec::new(),
        engine: Vec::new(),
        trainer: Vec::new(),
    };
    let (mut ti, mut ki) = (0, 0);
    let mut i = 0;
    while ki < kept {
        if i % every == 0 && ti < n_test {
            frames.names.push(tests.names[ti].clone());
            frames.engine.push(tests.engine[ti].clone());
            frames.trainer.push(fit(&tests.trainer[ti], tw, th));
            ti += 1;
        } else {
            assert!(i % every != 0, "held-out slots and test photos disagree");
            frames.names.push(train.names[ki].clone());
            frames.engine.push(train.engine[ki].clone());
            frames.trainer.push(train.trainer[ki].clone());
            ki += 1;
        }
        i += 1;
    }
    let name_of: std::collections::HashMap<String, String> = train
        .names
        .iter()
        .cloned()
        .zip(sources.iter().cloned())
        .collect();

    let weights = PathBuf::from(env(
        "SPL_VGGT_WEIGHTS",
        "/models/vggt-omega/vggt_omega_1b_512.safetensors",
    ));
    let (mut cams_json, mut pts_ply) = (Vec::new(), Vec::new());
    let mut on_event = |e: Event| match e {
        Event::Cameras(j) => cams_json = j,
        Event::Points(p, _) => pts_ply = p,
        Event::Log(m) => eprintln!("{m}"),
        _ => {}
    };
    let t1 = Instant::now();
    let mut r = reconstruct(
        &weights,
        frames,
        &PipelineConfig::preset(preset),
        every,
        &cancel,
        &mut on_event,
    )
    .unwrap();
    let recon_s = t1.elapsed().as_secs_f64();
    let capture = capture_report(&capture_stats, &cams_json, &pts_ply);

    // Pose accuracy of the refined training cameras.
    let scene = load_colmap_scene(
        &root,
        &LoadOptions {
            images_dir: images.clone(),
            ..Default::default()
        },
    )
    .unwrap();
    let gt_by: std::collections::HashMap<&str, Pose> = scene
        .views
        .iter()
        .map(|v| (v.name.as_str(), (v.r_cw, v.t_cw)))
        .collect();
    let (mut ours, mut gt, mut same) = (Vec::new(), Vec::new(), Vec::new());
    let mut ids: Vec<String> = Vec::new();
    for v in &r.views {
        let src = &name_of[&v.name];
        if let Some(g) = gt_by.get(src.as_str()) {
            let id = ids.iter().position(|s| s == src).unwrap_or_else(|| {
                ids.push(src.clone());
                ids.len() - 1
            });
            ours.push((v.r_cw, v.t_cw));
            gt.push(*g);
            same.push(id);
        }
    }
    let errs = pair_errors(&ours, &gt, &same);

    let ev = evaluate(&mut r, &preset_name).unwrap();
    for (k, jpg) in ev.thumbs.iter().enumerate() {
        std::fs::write(out.join(format!("worst-{k}.jpg")), jpg).unwrap();
    }
    if let Some(spz) = &ev.clean_spz {
        eprintln!("clean spz: {} bytes", spz.len());
    }
    let result = serde_json::json!({
        "tag": tag,
        "mode": mode,
        "video": video.display().to_string(),
        "preset": preset_name,
        "max_frames": max,
        "kept": kept,
        "unique_sources": unique,
        "held_out": n_test,
        "holdout_every": every,
        "ingest_s": ingest_s,
        "recon_s": recon_s,
        "gaussians": r.cloud.len(),
        "auc3": auc(&errs, 3),
        "auc5": auc(&errs, 5),
        "auc30": auc(&errs, 30),
        "pairs": errs.len(),
        "capture": capture,
        "quality": ev.report,
    });
    r.release();
    std::fs::write(
        out.join("result.json"),
        serde_json::to_vec_pretty(&result).unwrap(),
    )
    .unwrap();
    eprintln!("RESULT {}", serde_json::to_string(&result).unwrap());
}

#[test]
fn auc_of_perfect_and_bad_pairs() {
    assert_eq!(auc(&[0.0, 0.0], 30), 100.0);
    assert_eq!(auc(&[90.0], 30), 0.0);
    let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let p = vec![(id, [0.0, 0.0, 0.0]), (id, [-1.0, 0.0, 0.0])];
    assert_eq!(pair_errors(&p, &p, &[0, 1]), vec![0.0]);
}
