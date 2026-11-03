//! GPU tests of Stage A0. All `#[ignore]`: run on the GPU host with
//! `cargo test -p spl-app-image-to-3d ingest -- --ignored --nocapture`.
//! The video tests need `ffmpeg` on PATH and skip without it.
//! `SPL_INGEST_BENCH=<clip>` adds the throughput run on a real clip.

use std::time::{Duration, Instant};

use spl_core::CancellationToken;

use super::gpu::IngestGpu;
use super::tests::{jpeg_with_exif, synth_video, tiff_orientation_focal};
use super::*;

fn noise(w: u32, h: u32, seed: u32) -> Rgb8 {
    let mut s = seed.wrapping_mul(2_654_435_761).max(1);
    let data = (0..w * h * 3)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s >> 24) as u8
        })
        .collect();
    Rgb8 {
        width: w,
        height: h,
        data,
    }
}

#[test]
#[ignore]
fn gpu_sharpness_and_resize_match_the_cpu_reference() {
    let mut g = IngestGpu::open().unwrap();
    for (w, h, dw, dh) in [(97, 61, 31, 20), (640, 360, 512, 288), (1000, 750, 100, 75)] {
        let img = noise(w, h, w + h);
        let r = g.upload(w, h, &img.data).unwrap();
        let (gs, cs) = (g.sharpness(r).unwrap(), sharpness_cpu(&img));
        assert!((gs - cs).abs() <= 1e-4 * cs, "{w}x{h}: gpu {gs} cpu {cs}");
        let (gr, cr) = (g.resize(r, dw, dh).unwrap(), resize_area_cpu(&img, dw, dh));
        let worst = gr
            .data
            .iter()
            .zip(&cr.data)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap();
        assert!(worst <= 1, "{w}x{h}->{dw}x{dh}: max diff {worst}");
        println!("{w}x{h}: sharpness gpu {gs:.6} cpu {cs:.6}; resize max diff {worst}");
    }
}

#[test]
#[ignore]
fn gpu_ingests_photos_with_orientation_and_focal() {
    let dir = tempfile::tempdir().unwrap();
    let mut paths = Vec::new();
    for (i, f35) in [24u16, 26, 28].iter().enumerate() {
        let p = dir.path().join(format!("{i}.jpg"));
        std::fs::write(
            &p,
            jpeg_with_exif(1200, 800, &tiff_orientation_focal(6, *f35)),
        )
        .unwrap();
        paths.push(p);
    }
    let r = ingest_images(&paths, &IngestConfig::default(), &CancellationToken::new()).unwrap();
    assert_eq!(r.frames.len(), 3);
    let f = &r.frames[0];
    assert_eq!(f.source_size, (800, 1200));
    assert_eq!((f.engine.width, f.engine.height), (336, 512));
    assert_eq!((f.trainer.width, f.trainer.height), (800, 1200));
    assert_eq!(r.focal_prior, Some(FocalPrior { focal_35mm: 26.0 }));
    println!("photos: {:?}", r.stats);
}

#[tokio::test]
#[ignore]
async fn gpu_video_keyframes_pick_the_sharp_frame_of_each_bin() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("s.mp4");
    if !synth_video(&p, 640, 360, 30, 4) {
        return;
    }
    let cfg = IngestConfig {
        max_frames: 12,
        sample_fps: None,
        keyframes: KeyframeMode::Time,
        ..IngestConfig::default()
    };
    let r = ingest_video(&p, &cfg, &CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(r.stats.decoded, 120);
    assert_eq!(r.sharpness.len(), 120);
    assert_eq!(r.frames.len(), 12);
    for (b, f) in r.frames.iter().enumerate() {
        assert_eq!(f.index % 5, 2, "bin {b} kept blurred frame {}", f.index);
        assert_eq!(f.index / 10, b as u32, "frame {} outside bin {b}", f.index);
        let t = f.timestamp_s.unwrap();
        assert!((t - f.index as f64 / 30.0).abs() < 1e-9);
        assert_eq!((f.engine.width, f.engine.height), (512, 288));
        assert_eq!((f.trainer.width, f.trainer.height), (640, 360));
    }
    let sharp: f64 = r.frames.iter().map(|f| f.sharpness).sum::<f64>() / 12.0;
    let blurred: Vec<f64> = r
        .sharpness
        .iter()
        .filter(|s| s.index % 5 != 2)
        .map(|s| s.sharpness)
        .collect();
    let blurred = blurred.iter().sum::<f64>() / blurred.len() as f64;
    println!(
        "kept {:?}; mean sharpness kept {sharp:.5} vs blurred {blurred:.5}; {:?}",
        r.frames.iter().map(|f| f.index).collect::<Vec<_>>(),
        r.stats
    );
    assert!(sharp > 2.0 * blurred);

    // Sampling at 10 fps of a 30 fps clip: every third frame, timestamps i/10.
    let time = IngestConfig {
        keyframes: KeyframeMode::Time,
        ..IngestConfig::default()
    };
    let r = ingest_video(&p, &time, &CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(r.stats.decoded, 40);
    assert_eq!(r.frames.len(), 40, "fewer frames than bins keeps all");
}

#[tokio::test]
#[ignore]
async fn gpu_video_motion_keyframes_keep_sharp_frames_along_the_path() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("s.mp4");
    if !synth_video(&p, 640, 360, 30, 4) {
        return;
    }
    let cfg = IngestConfig {
        max_frames: 12,
        sample_fps: None,
        ..IngestConfig::default()
    };
    assert_eq!(cfg.keyframes, KeyframeMode::Motion);
    let r = ingest_video(&p, &cfg, &CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(r.stats.decoded, 120);
    let kept: Vec<u32> = r.frames.iter().map(|f| f.index).collect();
    println!("kept {kept:?}; capture {:?}", r.capture);
    assert!(kept.len() >= 10 && kept.len() <= 12, "{kept:?}");
    assert!(kept.windows(2).all(|w| w[0] < w[1]));
    assert!(
        kept.iter().all(|i| i % 5 == 2),
        "a blurred frame was kept: {kept:?}"
    );
    assert!(kept[0] < 20 && *kept.last().unwrap() >= 100, "{kept:?}");
    for f in &r.frames {
        assert_eq!((f.engine.width, f.engine.height), (512, 288));
        assert_eq!((f.trainer.width, f.trainer.height), (640, 360));
    }
    assert_eq!(r.capture.keyframes, "motion");
    assert!(
        (r.capture.blurry_share - 0.8).abs() < 0.05,
        "{:?}",
        r.capture
    );
    assert_eq!(r.capture.kept_blurry_share, 0.0);
}

#[tokio::test]
#[ignore]
async fn gpu_video_cancel_kills_ffmpeg_promptly() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("long.mp4");
    if !synth_video(&p, 1920, 1080, 30, 20) {
        return;
    }
    let cancel = CancellationToken::new();
    let c = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(500)).await;
        c.cancel();
    });
    let cfg = IngestConfig {
        sample_fps: None,
        ..IngestConfig::default()
    };
    let t = Instant::now();
    let err = ingest_video(&p, &cfg, &cancel).await.unwrap_err();
    let took = t.elapsed();
    println!("cancelled after {took:?}: {err}");
    assert!(err.to_string().contains("cancelled"), "{err}");
    assert!(took < Duration::from_secs(3), "{took:?}");
}

/// Throughput on a real clip: `SPL_INGEST_BENCH=/path/clip.mp4`, optional
/// `SPL_INGEST_FPS` (default 10, `all` = every frame) and
/// `SPL_INGEST_HWACCEL=cuda`.
#[tokio::test]
#[ignore]
async fn gpu_ingest_throughput_bench() {
    let Ok(clip) = std::env::var("SPL_INGEST_BENCH") else {
        eprintln!("SPL_INGEST_BENCH not set; skipping");
        return;
    };
    let sample_fps = match std::env::var("SPL_INGEST_FPS").as_deref() {
        Ok("all") => None,
        Ok(v) => Some(v.parse().unwrap()),
        Err(_) => Some(10.0),
    };
    let ffmpeg_input_args = match std::env::var("SPL_INGEST_HWACCEL") {
        Ok(h) => vec!["-hwaccel".into(), h],
        Err(_) => Vec::new(),
    };
    let cfg = IngestConfig {
        sample_fps,
        ffmpeg_input_args,
        ..IngestConfig::default()
    };
    let info = probe_video(std::path::Path::new(&clip)).await.unwrap();
    let r = ingest_video(std::path::Path::new(&clip), &cfg, &CancellationToken::new())
        .await
        .unwrap();
    println!(
        "BENCH {clip}: {}x{} {:.1}s @{:.2}fps, sample {:?}, hwaccel {:?}: decoded {} kept {} in {:.2}s \
         = {:.1} frames/s ({:.2}x realtime), gpu {:.2}s ({:.2} ms/frame)",
        info.width,
        info.height,
        info.duration_s,
        info.fps,
        cfg.sample_fps,
        cfg.ffmpeg_input_args,
        r.stats.decoded,
        r.stats.kept,
        r.stats.seconds,
        r.stats.decode_fps,
        info.duration_s / r.stats.seconds,
        r.stats.gpu_seconds,
        1e3 * r.stats.gpu_seconds / r.stats.decoded as f64,
    );
}
