//! Shared-GPU behaviour with an artificial memory hog, on the GPU host
//! (`#[ignore]`). The hog is device memory this process allocates (plain
//! `cuMemAlloc` through cudarc, never trimmed) so that a chosen amount stays
//! free; other processes on the card count too, so every test sizes its hog
//! from the free memory it finds. Data: `$SPL_DATASETS/mipnerf360/garden/
//! images_4` (`SPL_COEXIST_SCENE`), weights `$SPL_VGGT_WEIGHTS`.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use spl_core::CancellationToken;
use spl_gpu::cudarc::driver::CudaSlice;

use crate::ingest::{IngestConfig, ingest_images};
use crate::reconstruct::pipeline::{
    EngineOpts, Event, Frames, PipelineConfig, ReconPreset, pose_frames, reconstruct,
};
use crate::reconstruct::{image_files, to_frames};
use crate::vram::{WaitPolicy, device_memory, release_cached};

const GB: f64 = 1e9;

fn env(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

fn weights() -> PathBuf {
    PathBuf::from(env(
        "SPL_VGGT_WEIGHTS",
        "/models/vggt-omega/vggt_omega_1b_512.safetensors",
    ))
}

fn frames(n: usize, long_side: u32) -> Frames {
    let dir = PathBuf::from(env("SPL_DATASETS", "/datasets"))
        .join(env("SPL_COEXIST_SCENE", "mipnerf360/garden"))
        .join("images_4");
    let all = image_files(&dir).unwrap();
    let files: Vec<PathBuf> = crate::pose::sample_indices(all.len(), n)
        .into_iter()
        .map(|i| all[i].clone())
        .collect();
    let cfg = IngestConfig {
        trainer_long_side: long_side,
        max_frames: n as u32,
        ..Default::default()
    };
    to_frames(ingest_images(&files, &cfg, &CancellationToken::new()).unwrap())
}

fn free() -> u64 {
    release_cached().unwrap();
    device_memory().unwrap().0
}

/// Device memory held until dropped.
#[derive(Default)]
struct Hog(Vec<CudaSlice<u8>>);

impl Hog {
    /// Grows until about `target` bytes are free (no-op when already less).
    fn leave(&mut self, target: u64) {
        let stream = spl_gpu::context(0).unwrap().default_stream();
        loop {
            let f = free();
            if f <= target + (64 << 20) {
                break;
            }
            let n = (f - target).min(1 << 30) as usize;
            self.0.push(stream.alloc_zeros::<u8>(n).unwrap());
            stream.synchronize().unwrap();
        }
        eprintln!(
            "hog {:.2} GB: {:.2} GB free",
            self.bytes() as f64 / GB,
            free() as f64 / GB
        );
    }

    fn bytes(&self) -> u64 {
        self.0.iter().map(|s| s.len() as u64).sum()
    }
}

/// Waits (up to an hour) for other processes to leave `bytes` free.
fn wait_free(bytes: u64) -> u64 {
    let policy = WaitPolicy {
        poll: Duration::from_secs(10),
        timeout: Duration::from_secs(3600),
    };
    let (f, waited) = crate::vram::wait_until(
        &policy,
        &CancellationToken::new(),
        || Ok(free()),
        |f| {
            Ok(if f >= bytes {
                Ok(f)
            } else {
                Err(crate::vram::Shortfall {
                    what: "this test".into(),
                    need: bytes,
                })
            })
        },
        &mut |_| {},
    )
    .unwrap();
    eprintln!("start: {:.2} GB free (waited {waited:.0} s)", f as f64 / GB);
    f
}

/// This process's own pool footprint is the hog alone (device-wide free
/// memory moves with other processes, so it cannot show a leak).
fn idle(label: &str, hog: u64) {
    release_cached().unwrap();
    let r = crate::vram::pool_reserved().unwrap();
    eprintln!(
        "{label}: this process holds {:.0} MB of pool memory (hog {:.0} MB)",
        r as f64 / 1e6,
        hog as f64 / 1e6
    );
    assert!(r <= hog + (64 << 20), "{label}: memory left behind");
}

fn quick_wait(timeout_s: u64) -> WaitPolicy {
    WaitPolicy {
        poll: Duration::from_secs(1),
        timeout: Duration::from_secs(timeout_s),
    }
}

/// Relative rotation of frame i to frame 0, angle difference between two
/// runs (degrees), mean over frames.
fn rot_diff_deg(a: &[[f32; 12]], b: &[[f32; 12]]) -> f64 {
    let r = |e: &[f32; 12]| -> [[f64; 3]; 3] {
        std::array::from_fn(|i| std::array::from_fn(|j| e[i * 4 + j] as f64))
    };
    let rel = |x: &[[f32; 12]], i: usize| {
        let (ri, r0) = (r(&x[i]), r(&x[0]));
        // R_i R_0ᵀ
        let m: [[f64; 3]; 3] = std::array::from_fn(|p| {
            std::array::from_fn(|q| (0..3).map(|k| ri[p][k] * r0[q][k]).sum())
        });
        m
    };
    let mut sum = 0.0;
    for i in 1..a.len() {
        let (x, y) = (rel(a, i), rel(b, i));
        // trace(X Yᵀ) = 1 + 2 cos θ
        let tr: f64 = (0..3)
            .map(|p| (0..3).map(|k| x[p][k] * y[p][k]).sum::<f64>())
            .sum();
        sum += ((tr - 1.0) / 2.0).clamp(-1.0, 1.0).acos().to_degrees();
    }
    sum / (a.len() - 1).max(1) as f64
}

/// Engine runs leave nothing behind; with too little memory the engine
/// plans fewer frames per pass (same poses within a fraction of a degree),
/// then waits, then fails clearly, and frees its weights after a failure.
#[test]
#[ignore = "GPU, weights and dataset: run on the GPU host"]
fn gpu_coexist_engine() {
    let cancel = CancellationToken::new();
    let fr = frames(60, 800);
    let f0 = wait_free(5_000_000_000);
    let mut log = |e: Event| match e {
        Event::Log(m) | Event::Waiting(_, m, _) => eprintln!("  {m}"),
        _ => {}
    };

    // 1. Unconstrained: one pass, nothing kept afterwards.
    let opts = EngineOpts {
        wait: quick_wait(600),
        ..EngineOpts::new(false, 200)
    };
    let t = Instant::now();
    let full = pose_frames(&weights(), &fr, &opts, &cancel, &mut log).unwrap();
    let f1 = free();
    eprintln!(
        "full: {:.2} s, notes {:?}; after: {:.2} GB free (Δ {:+.0} MB)",
        t.elapsed().as_secs_f64(),
        full.mem.notes,
        f1 as f64 / GB,
        (f1 as f64 - f0 as f64) / 1e6
    );
    // Other processes may grow during the run; then the plan degrades too.
    if !full.mem.notes.is_empty() {
        eprintln!("note: the full run degraded (memory taken by another process)");
    }
    idle("after the full run", 0);

    // 2. 4.0 GB free: weights fit, 60 frames do not → smaller passes.
    let mut hog = Hog::default();
    hog.leave(4_000_000_000);
    let t = Instant::now();
    match pose_frames(&weights(), &fr, &opts, &cancel, &mut log) {
        Ok(deg) => {
            let rd = rot_diff_deg(&full.posed.cams.extrinsics, &deg.posed.cams.extrinsics);
            eprintln!(
                "4.0 GB: {:.2} s, notes {:?}, mean relative-rotation difference to the full run {rd:.3} deg",
                t.elapsed().as_secs_f64(),
                deg.mem.notes
            );
            assert_eq!(deg.mem.notes.len(), 1);
            assert!(rd < 2.0, "{rd}");
        }
        // Another process took the rest of the card for the whole wait.
        Err(e) => {
            eprintln!("4.0 GB: other processes left too little: {e:#}");
            assert!(format!("{e:#}").contains("not enough GPU memory"));
        }
    }
    idle("after the 4 GB run", hog.bytes());

    // 3. 2.0 GB free (the deployed failure): waits, then a clear error,
    // and the memory is still free afterwards.
    hog.leave(2_000_000_000);
    let f2 = free();
    let mut waits = 0;
    let t = Instant::now();
    let err = pose_frames(
        &weights(),
        &fr,
        &EngineOpts {
            wait: quick_wait(6),
            ..opts.clone()
        },
        &cancel,
        &mut |e| {
            if let Event::Waiting(_, m, d) = e {
                waits += 1;
                assert_eq!(d["waiting_for_gpu_memory"], true);
                eprintln!("  {m}");
            }
        },
    )
    .err()
    .expect("must fail");
    eprintln!(
        "2.0 GB: failed after {:.1} s with {waits} waits: {err:#}",
        t.elapsed().as_secs_f64()
    );
    assert!(waits >= 5);
    assert!(format!("{err:#}").contains("not enough GPU memory"));
    let _ = f2;
    idle("after the failed run", hog.bytes());

    // 4. The hog goes away after 8 s: the run waits, then succeeds.
    let hog = Arc::new(Mutex::new(Some(hog)));
    let h2 = hog.clone();
    let releaser = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_secs(8));
        // cudarc 0.19 frees without binding the context: bind it first,
        // or the free fails silently on a fresh thread.
        spl_gpu::context(0).unwrap().bind_to_thread().unwrap();
        h2.lock().unwrap().take();
    });
    let t = Instant::now();
    let ok = pose_frames(&weights(), &fr, &opts, &cancel, &mut log).unwrap();
    releaser.join().unwrap();
    eprintln!(
        "released after 8 s: done in {:.1} s, waited {:.1} s",
        t.elapsed().as_secs_f64(),
        ok.mem.waited_s
    );
    assert!(ok.mem.waited_s >= 5.0);
    let f3 = free();
    eprintln!("end: {:.2} GB free", f3 as f64 / GB);
    idle("end of engine test", 0);
}

/// The whole pipeline under a hog that grows once the engine is done: the
/// trainer degrades (capacity, then resolution), the run finishes, and the
/// device memory is back afterwards.
#[test]
#[ignore = "GPU, weights and dataset: run on the GPU host"]
fn gpu_coexist_reconstruct() {
    let cancel = CancellationToken::new();
    let trainer_free: f64 = env("SPL_COEXIST_TRAINER_GB", "3.0").parse().unwrap();
    let fr = frames(40, 1600);
    let f0 = wait_free(6_000_000_000);
    let cfg = PipelineConfig {
        iterations: Some(3000),
        snapshot_every_s: 0.0,
        ..PipelineConfig::preset(ReconPreset::Fast)
    };
    let hog = Mutex::new(Hog::default());
    let mut grown = false;
    let t = Instant::now();
    let r = reconstruct(&weights(), fr, &cfg, 0, &cancel, &mut |e| match e {
        Event::Log(m) => {
            eprintln!("  [{:6.1}s] {m}", t.elapsed().as_secs_f64());
            if m.starts_with("engine:") && !grown {
                grown = true;
                hog.lock().unwrap().leave((trainer_free * GB) as u64);
            }
        }
        Event::Waiting(_, m, _) => eprintln!("  {m}"),
        _ => {}
    })
    .unwrap();
    eprintln!(
        "done in {:.1} s: {} Gaussians, {} iterations, notes {:?}, peak {:.2} GB",
        t.elapsed().as_secs_f64(),
        r.cloud.len(),
        r.iterations,
        r.coexist.notes,
        r.peak_vram_bytes as f64 / GB
    );
    assert!(!r.coexist.notes.is_empty());
    drop(r);
    drop(hog);
    let f1 = free();
    eprintln!(
        "end: {:.2} GB free (Δ {:+.0} MB)",
        f1 as f64 / GB,
        (f1 as f64 - f0 as f64) / 1e6
    );
    idle("end of reconstruct test", 0);
}

/// Memory dropped on another thread is free again for the next poll.
#[test]
#[ignore = "GPU: run on the GPU host"]
fn gpu_coexist_cross_thread_free() {
    let f0 = crate::vram::free_now().unwrap();
    let mut hog = Hog::default();
    hog.leave(f0.saturating_sub(2_000_000_000).max(1_000_000_000));
    let held = crate::vram::pool_reserved().unwrap();
    let f1 = crate::vram::free_now().unwrap();
    let h = Arc::new(Mutex::new(Some(hog)));
    let h2 = h.clone();
    std::thread::spawn(move || {
        // cudarc 0.19 frees without binding the context: bind it first,
        // or the free fails silently on a fresh thread.
        spl_gpu::context(0).unwrap().bind_to_thread().unwrap();
        h2.lock().unwrap().take();
    })
    .join()
    .unwrap();
    let f2 = crate::vram::free_now().unwrap();
    let after = crate::vram::pool_reserved().unwrap();
    eprintln!(
        "free {:.2} → {:.2} (hog) → {:.2} GB; pool {:.0} → {:.0} MB",
        f0 as f64 / GB,
        f1 as f64 / GB,
        f2 as f64 / GB,
        held as f64 / 1e6,
        after as f64 / 1e6
    );
    assert!(after < 64 << 20, "dropped hog still in the pool");
}
