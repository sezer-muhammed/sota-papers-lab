//! Phase 2 (attention schedule, SageAttention, FP8 GEMMs) checks and the
//! measurements behind `tools/vggt/PHASE2.md`.
//!
//! - `gpu_p2_eval_dump`: extrinsics per schedule on the held-out scenes of
//!   `spec-attn-schedule.md` (`$SPL_P2_LIST/<scene>/frames.txt`, written by
//!   `tools/vggt/p2_eval.py list`) and on the ETH3D / Sintel subsets, for
//!   `p2_eval.py score`.
//! - `gpu_p2_bench`: time and memory per (schedule, N).
//! - `gpu_p2_deviation`: pose-encoding and depth distance of each schedule
//!   from the dense native run.

use std::path::PathBuf;
use std::sync::Arc;

use spl_core::CancellationToken;
use spl_nn::Nn;

use super::integration::{datasets, decode_all, free_bytes, image_files, scene_images, write_npy};
use super::tests::{env_path, weights_file};
use super::*;
use crate::pose;
use trunk::DEPTH;

fn schedules() -> Vec<(String, AttnSchedule)> {
    std::env::var("SPL_P2_SCHEDULES")
        .unwrap_or_else(|_| "dense,ours+sage".into())
        .split(',')
        .map(|s| {
            let s = s.trim();
            (s.to_string(), AttnSchedule::named(s).unwrap())
        })
        .collect()
}

fn engine_with(schedule: AttnSchedule) -> VggtEngine {
    let nn = Arc::new(Nn::open().expect("CUDA"));
    let cfg = VggtConfig {
        schedule,
        fp8: true,
        fp8_blocks: fp8_blocks(),
        sage_pv16: pv16(),
        ..VggtConfig::default()
    };
    VggtEngine::load(nn, &weights_file(), cfg).expect("load engine")
}

/// `$SPL_P2_PV=f16`: SageAttention2++ FP16 P·V accumulation.
fn pv16() -> bool {
    std::env::var("SPL_P2_PV").as_deref() == Ok("f16")
}

/// `$SPL_P2_FP8`: FP8 blocks as a hex mask (bit i frame block i, 24 + i
/// inter-frame block i), or `all`.
fn fp8_blocks() -> trunk::Fp8Blocks {
    match std::env::var("SPL_P2_FP8").as_deref() {
        Ok("all") => trunk::Fp8Blocks((1 << 48) - 1),
        Ok(h) if !h.is_empty() => {
            trunk::Fp8Blocks(u64::from_str_radix(h.trim_start_matches("0x"), 16).unwrap())
        }
        _ => trunk::Fp8Blocks::NONE,
    }
}

#[test]
fn phase2_schedule_matches_the_spec() {
    let s = AttnSchedule::phase2(true);
    for i in 0..24 {
        let want = if REGISTER_LAYERS.contains(&i) {
            InterAttn::Register
        } else if FRAME_LAYERS.contains(&i) {
            InterAttn::Frame
        } else if MERGE4_LAYERS.contains(&i) {
            InterAttn::Merge4
        } else {
            InterAttn::Sage
        };
        assert_eq!(s.0[i], want, "layer {i}");
    }
    assert_eq!(
        AttnSchedule::phase2(false)
            .0
            .iter()
            .filter(|&&a| a == InterAttn::Global)
            .count(),
        19 - 3 - 4
    );
    let w = AttnSchedule::phase2_wide(false);
    assert_eq!(w.0[8], InterAttn::Merge4);
    assert_eq!(w.0[22], InterAttn::Merge4);
    assert_eq!(AttnSchedule::named("dense").unwrap(), AttnSchedule::dense());
    assert!(AttnSchedule::named("nope").is_err());
    for (name, _) in [("ours+sage", 0), ("wide+sage", 0), ("sage", 0), ("ours", 0)] {
        let s = AttnSchedule::named(name).unwrap();
        assert!(super::trunk_schedule(&s).is_ok(), "{name}");
    }
}

#[test]
fn sage_workspaces_are_counted() {
    let g = trunk::TrunkGeom {
        frames: 200,
        grid_h: 21,
        grid_w: 32,
    };
    let dense = trunk::Trunk::scratch_bytes(&g);
    let s = super::trunk_schedule(&AttnSchedule::phase2(true)).unwrap();
    let p2 = trunk::Trunk::scratch_bytes_for(&g, &s);
    // Dense sage (Q, K, V at one byte each) plus merge4 (Q, pooled K/V).
    let extra = p2 - dense;
    let tokens = 200 * 689;
    assert!(
        extra > 4 * 16 * 64 * tokens && extra < 5 * 16 * 64 * tokens,
        "{extra}"
    );
    let d = super::trunk_schedule(&AttnSchedule::dense()).unwrap();
    assert_eq!(trunk::Trunk::scratch_bytes_for(&g, &d), dense);
}

/// Image lists written by `p2_eval.py list`: `<root>/<scene path>/frames.txt`.
fn eval_lists() -> Vec<(String, Vec<PathBuf>)> {
    let root = env_path("SPL_P2_LIST", "target/p2-list");
    let mut out = Vec::new();
    let mut stack = vec![root.clone()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|n| n == "frames.txt") {
                let scene = d.strip_prefix(&root).unwrap().to_string_lossy().to_string();
                let files = std::fs::read_to_string(&p)
                    .unwrap()
                    .lines()
                    .map(|l| PathBuf::from(l.trim()))
                    .collect();
                out.push((scene, files));
            }
        }
    }
    out.sort();
    out
}

/// The ETH3D / Sintel subsets of `gpu_dump_eval_predictions`.
fn subset_lists() -> Vec<(String, Vec<PathBuf>)> {
    let frames = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tools/vggt/frames");
    let mut out = Vec::new();
    for ds in ["eth3d", "sintel"] {
        let map: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(
            &std::fs::read(frames.join(format!("{ds}_subset.json"))).unwrap(),
        )
        .unwrap();
        for (scene, idx) in map {
            let root = datasets().join("vggt-eval").join(ds).join(&scene);
            if !root.is_dir() {
                continue;
            }
            let names: Vec<String> = std::fs::read_to_string(root.join("frames.txt"))
                .unwrap()
                .lines()
                .map(|l| l.trim().to_string())
                .collect();
            let files = idx
                .as_array()
                .unwrap()
                .iter()
                .map(|i| {
                    root.join("images")
                        .join(&names[i.as_u64().unwrap() as usize])
                })
                .collect();
            out.push((format!("{ds}/{scene}"), files));
        }
    }
    out
}

/// Extrinsics (and depth for the subsets) per schedule:
/// `$SPL_P2_PRED/<schedule>/<scene>/extrinsics.npy`.
#[test]
#[ignore]
fn gpu_p2_eval_dump() {
    let out = env_path("SPL_P2_PRED", "target/p2-pred");
    let lists = eval_lists();
    let subsets = subset_lists();
    let cancel = CancellationToken::new();
    let mut e = engine_with(AttnSchedule::dense());
    for (name, sched) in schedules() {
        e.cfg.schedule = sched;
        for (scene, files) in lists.iter().chain(&subsets) {
            let p = e.preprocess(&decode_all(files)).unwrap();
            let cams = e.run(&p, &cancel, &mut |_, _| {}).unwrap();
            let g = p.geom;
            let dir = out.join(&name).join(scene);
            let ex: Vec<f32> = cams.extrinsics.iter().flatten().copied().collect();
            write_npy(&dir.join("extrinsics.npy"), &[g.frames, 3, 4], &ex);
            if subsets.iter().any(|(s, _)| s == scene) {
                let d = e.download(&g).unwrap();
                write_npy(
                    &dir.join("depth.npy"),
                    &[g.frames, g.height, g.width],
                    &d.depth,
                );
            }
            println!(
                "p2 eval: {name} {scene} {} frames {}x{} trunk {:.1} ms total {:.1} ms",
                g.frames, g.width, g.height, e.times.trunk_ms, e.times.total_ms
            );
        }
    }
}

/// RMS distance of the pose encodings and the relative depth error of
/// each schedule from the dense native run (one scene, `$SPL_P2_N` frames).
#[test]
#[ignore]
fn gpu_p2_deviation() {
    let n: usize = std::env::var("SPL_P2_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(50);
    let scene = std::env::var("SPL_P2_SCENE").unwrap_or("mipnerf360/kitchen".into());
    let files = image_files(&scene_images(&datasets().join(&scene)));
    let pick: Vec<PathBuf> = pose::sample_indices(files.len(), n)
        .iter()
        .map(|&i| files[i].clone())
        .collect();
    let cancel = CancellationToken::new();
    let mut e = engine_with(AttnSchedule::dense());
    let p = e.preprocess(&decode_all(&pick)).unwrap();
    let base = e.run(&p, &cancel, &mut |_, _| {}).unwrap();
    let base_d = e.download(&p.geom).unwrap();
    for (name, sched) in schedules() {
        e.cfg.schedule = sched;
        let c = e.run(&p, &cancel, &mut |_, _| {}).unwrap();
        let d = e.download(&p.geom).unwrap();
        let enc: f64 = c
            .pose_enc
            .iter()
            .zip(&base.pose_enc)
            .map(|(a, b)| {
                a.iter()
                    .zip(b)
                    .map(|(x, y)| ((x - y) as f64).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .sum::<f64>()
            / n as f64;
        let (num, den) = d
            .depth
            .iter()
            .zip(&base_d.depth)
            .fold((0f64, 0f64), |(a, b), (x, y)| {
                (a + ((x - y) as f64).powi(2), b + (*y as f64).powi(2))
            });
        println!(
            "p2 deviation: {scene} N={n} {name}: pose_enc |Δ| {enc:.4e}, depth rel {:.3e}, trunk {:.1} ms",
            (num / den).sqrt(),
            e.times.trunk_ms
        );
    }
}

/// Median GPU time and memory per (schedule, N) on `$SPL_P2_SCENE`.
#[test]
#[ignore]
fn gpu_p2_bench() {
    let ns: Vec<usize> = std::env::var("SPL_P2_NS")
        .unwrap_or_else(|_| "2,10,50,100,200".into())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let scene = std::env::var("SPL_P2_SCENE").unwrap_or("mipnerf360/kitchen".into());
    let files = image_files(&scene_images(&datasets().join(&scene)));
    let total = spl_nn::cudarc::driver::result::mem_get_info().unwrap().1;
    let cancel = CancellationToken::new();
    for (name, sched) in schedules() {
        let free0 = free_bytes();
        let mut e = engine_with(sched);
        for &n in &ns {
            let pick: Vec<PathBuf> = pose::sample_indices(files.len(), n)
                .iter()
                .map(|&i| files[i].clone())
                .collect();
            let p = e.preprocess(&decode_all(&pick)).unwrap();
            let plan = e.plan(p.geom).unwrap();
            let reps = if n <= 50 { 10 } else { 3 };
            let mut t = Vec::new();
            let mut min_free = usize::MAX;
            for k in 0..reps + 2 {
                e.run(&p, &cancel, &mut |_, _| {}).unwrap();
                min_free = min_free.min(free_bytes());
                if k >= 2 {
                    t.push((e.times.total_ms, e.times.clone()));
                }
            }
            t.sort_by(|a, b| a.0.total_cmp(&b.0));
            let (med, st) = &t[t.len() / 2];
            println!(
                "p2 bench: {name} N={n} {}x{}: median {med:.1} ms (min {:.1}, {reps} runs) trunk {:.1} camera {:.1} dense {:.1} graph {} offload {} release {}; engine {:.0} MiB (free-memory delta), card {:.0} MiB of {:.0}",
                p.geom.width,
                p.geom.height,
                t[0].0,
                st.trunk_ms,
                st.camera_ms,
                st.dense_ms,
                st.graph,
                plan.offload,
                plan.release_trunk,
                (free0 - min_free) as f64 / 1048576.0,
                (total - min_free) as f64 / 1048576.0,
                total as f64 / 1048576.0
            );
        }
    }
}

/// Per-block FP8 error on the calibration scenes of the schedule search
/// (bicycle, garden, stump, kitchen; `$SPL_P2_N` frames each): every
/// non-register block alone in FP8, then all of them, measured against
/// the same schedule with bf16 GEMMs. Prints one line per block for the
/// greedy selection in `PHASE2.md`.
#[test]
#[ignore]
fn gpu_p2_fp8_sensitivity() {
    let n: usize = std::env::var("SPL_P2_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30);
    let scenes = std::env::var("SPL_P2_CAL").unwrap_or(
        "mipnerf360/bicycle,mipnerf360/garden,mipnerf360/stump,mipnerf360/kitchen".into(),
    );
    let (name, sched) = schedules().pop().unwrap();
    let mut e = engine_with(sched);
    let cancel = CancellationToken::new();
    let inputs: Vec<Prepared> = scenes
        .split(',')
        .map(|s| {
            let files = image_files(&scene_images(&datasets().join(s)));
            let pick: Vec<PathBuf> = pose::sample_indices(files.len(), n)
                .iter()
                .map(|&i| files[i].clone())
                .collect();
            e.preprocess(&decode_all(&pick)).unwrap()
        })
        .collect();
    let run = |e: &mut VggtEngine, mask: trunk::Fp8Blocks| -> Vec<(Vec<[f32; 9]>, Vec<f32>)> {
        e.cfg.fp8_blocks = mask;
        inputs
            .iter()
            .map(|p| {
                let c = e.run(p, &cancel, &mut |_, _| {}).unwrap();
                let d = e.download(&p.geom).unwrap();
                (c.pose_enc, d.depth)
            })
            .collect()
    };
    let base = run(&mut e, trunk::Fp8Blocks::NONE);
    let score = |out: &[(Vec<[f32; 9]>, Vec<f32>)]| -> (f64, f64) {
        let mut enc = 0.0;
        let mut dep = 0.0;
        for ((pe, d), (bpe, bd)) in out.iter().zip(&base) {
            enc += pe
                .iter()
                .zip(bpe)
                .map(|(a, b)| {
                    a.iter()
                        .zip(b)
                        .map(|(x, y)| ((x - y) as f64).powi(2))
                        .sum::<f64>()
                        .sqrt()
                })
                .sum::<f64>()
                / pe.len() as f64;
            let (num, den) = d.iter().zip(bd).fold((0f64, 0f64), |(a, b), (x, y)| {
                (a + ((x - y) as f64).powi(2), b + (*y as f64).powi(2))
            });
            dep += (num / den).sqrt();
        }
        let k = out.len() as f64;
        (enc / k, dep / k)
    };
    println!("p2 fp8: schedule {name}, {n} frames x {}", inputs.len());
    for (fam, count) in [("frame", DEPTH), ("inter", DEPTH)] {
        for i in 0..count {
            if fam == "inter" && REGISTER_LAYERS.contains(&i) {
                continue;
            }
            let m = if fam == "frame" {
                trunk::Fp8Blocks::NONE.with_frame(i)
            } else {
                trunk::Fp8Blocks::NONE.with_inter(i)
            };
            let (enc, dep) = score(&run(&mut e, m));
            println!("p2 fp8 block {fam} {i}: pose_enc |Δ| {enc:.4e} depth rel {dep:.3e}");
        }
    }
    let (enc, dep) = score(&run(&mut e, trunk::Fp8Blocks((1 << 48) - 1)));
    println!("p2 fp8 block all: pose_enc |Δ| {enc:.4e} depth rel {dep:.3e}");
    let (enc, dep) = score(&run(&mut e, trunk::Fp8Blocks::NONE));
    println!("p2 fp8 block none (repeat): pose_enc |Δ| {enc:.4e} depth rel {dep:.3e}");
}
