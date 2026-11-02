//! GPU tests of the commercial engine (DA3-BASE on ONNX Runtime). Owner:
//! commercial. `#[ignore]`d; run on the GPU host in the CUDA container with
//! ONNX Runtime and cuDNN on the loader path (`tools/onnx/README.md`):
//!
//! - `gpu_da3_dump_eval_predictions`: depth and extrinsics on the vggt-ref
//!   evaluation subsets, scored by `tools/vggt/eval_subset.py --predictions`.
//! - `gpu_da3_bench_scaling`: forward time and card memory at N = 3..200
//!   frames of a Mip-NeRF 360 scene (`$SPL_DA3_SCALING_N`,
//!   `$SPL_DA3_SCALING_SCENE`).

use std::path::{Path, PathBuf};

use spl_core::CancellationToken;

use super::LicenceProfile;
use super::da3::Da3Engine;
use crate::io::images::Rgb8Image;
use crate::pose;

fn env_path(var: &str, default: &str) -> PathBuf {
    std::env::var_os(var)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(default))
}

fn datasets() -> PathBuf {
    env_path("SPL_DATASETS", "/datasets")
}

fn engine() -> Da3Engine {
    Da3Engine::load(
        &env_path("SPL_DA3_DIR", "/models/da3-base"),
        LicenceProfile::Commercial,
    )
    .expect("DA3-BASE graphs and ONNX Runtime")
}

/// Writes a little-endian f32 `.npy` in C order.
fn write_npy(path: &Path, shape: &[usize], data: &[f32]) {
    assert_eq!(shape.iter().product::<usize>(), data.len());
    let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    let mut header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}), }}",
        dims.join(", ")
    );
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut b = b"\x93NUMPY\x01\x00".to_vec();
    b.extend((header.len() as u16).to_le_bytes());
    b.extend(header.as_bytes());
    for v in data {
        b.extend(v.to_le_bytes());
    }
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, b).unwrap();
}

fn image_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| {
            let e = p
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            matches!(e.as_str(), "jpg" | "jpeg" | "png")
        })
        .collect();
    files.sort();
    files
}

fn decode_all(files: &[PathBuf]) -> Vec<Rgb8Image> {
    use rayon::prelude::*;
    files
        .par_iter()
        .map(|f| crate::vggt::preprocess::decode(&std::fs::read(f).unwrap()).unwrap())
        .collect()
}

fn free_mib() -> f64 {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output();
    out.ok()
        .and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse().ok())
        .unwrap_or(f64::NAN)
}

#[test]
#[ignore]
fn gpu_da3_dump_eval_predictions() {
    let out = env_path("SPL_DA3_PRED", "target/da3-eval-pred");
    let frames = Path::new(env!("CARGO_MANIFEST_DIR")).join("tools/vggt/frames");
    let mut e = engine();
    let cancel = CancellationToken::new();
    for ds in ["eth3d", "sintel"] {
        let map: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(
            &std::fs::read(frames.join(format!("{ds}_subset.json"))).unwrap(),
        )
        .unwrap();
        for (scene, idx) in map {
            let root = datasets().join("vggt-eval").join(ds).join(&scene);
            if !root.is_dir() {
                println!("eval: {} missing, skipped", root.display());
                continue;
            }
            let names: Vec<String> = std::fs::read_to_string(root.join("frames.txt"))
                .unwrap()
                .lines()
                .map(|l| l.trim().to_string())
                .collect();
            let files: Vec<PathBuf> = idx
                .as_array()
                .unwrap()
                .iter()
                .map(|i| {
                    root.join("images")
                        .join(&names[i.as_u64().unwrap() as usize])
                })
                .collect();
            let p = e.preprocess(&decode_all(&files)).unwrap();
            // Warm-up (first use of the bucket), then the timed run.
            e.run(&p, &cancel, &mut |_, _| {}).unwrap();
            let (cams, dense) = e.run(&p, &cancel, &mut |_, _| {}).unwrap();
            let g = p.geom;
            let dir = out.join(ds).join(&scene);
            write_npy(
                &dir.join("depth.npy"),
                &[g.frames, g.height, g.width],
                &dense.depth,
            );
            let ex: Vec<f32> = cams.extrinsics.iter().flatten().copied().collect();
            write_npy(&dir.join("extrinsics.npy"), &[g.frames, 3, 4], &ex);
            assert!(dense.depth.iter().all(|d| d.is_finite() && *d > 0.0));
            println!(
                "eval: {ds}/{scene} {} frames {}x{} in {:.1} ms (backbone {:.1}, head {:.1}) -> {}",
                g.frames,
                g.width,
                g.height,
                e.times.forward_ms,
                e.times.trunk_ms,
                e.times.dense_ms,
                dir.display()
            );
        }
    }
}

#[test]
#[ignore]
fn gpu_da3_bench_scaling() {
    let ns: Vec<usize> = std::env::var("SPL_DA3_SCALING_N")
        .unwrap_or_else(|_| "3,10,32,50,100,200".into())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let scene = std::env::var("SPL_DA3_SCALING_SCENE").unwrap_or("mipnerf360/kitchen".into());
    let root = datasets().join(&scene);
    let dir = ["images_2", "images_4", "images"]
        .iter()
        .map(|d| root.join(d))
        .find(|d| d.is_dir())
        .unwrap();
    let files = image_files(&dir);
    let used0 = free_mib();
    let mut e = engine();
    let cancel = CancellationToken::new();
    println!(
        "scaling: {scene} ({}), {} images; card {used0:.0} MiB used before",
        dir.display(),
        files.len()
    );
    for &n in &ns {
        assert!(n <= files.len(), "{scene} has {} images", files.len());
        let pick = pose::sample_indices(files.len(), n);
        let chosen: Vec<PathBuf> = pick.iter().map(|&i| files[i].clone()).collect();
        let t = std::time::Instant::now();
        let p = e.preprocess(&decode_all(&chosen)).unwrap();
        let prep_ms = t.elapsed().as_secs_f64() * 1e3;
        let reps = if n <= 50 { 10 } else { 3 };
        let mut runs = Vec::new();
        let mut peak = 0.0f64;
        for k in 0..reps + 2 {
            e.forward(&p, &cancel, &mut |_, _| {}).unwrap();
            peak = peak.max(free_mib());
            if k >= 2 {
                runs.push(e.times.clone());
            }
        }
        runs.sort_by(|a, b| a.forward_ms.total_cmp(&b.forward_ms));
        let med = &runs[runs.len() / 2];
        println!(
            "scaling N={n} {}x{}: median {:.1} ms (min {:.1}, {reps} runs) backbone {:.1} head {:.1}; decode+preprocess {prep_ms:.0} ms; card {peak:.0} MiB used (+{:.0})",
            p.geom.width,
            p.geom.height,
            med.forward_ms,
            runs[0].forward_ms,
            med.trunk_ms,
            med.dense_ms,
            peak - used0
        );
    }
}
