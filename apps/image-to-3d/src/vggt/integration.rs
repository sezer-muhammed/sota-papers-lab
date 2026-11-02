//! Integration tests of the assembled engine (trunk + heads + runtime) and
//! the `image_to_3d.pose` op. Owner: integration. GPU tests are `#[ignore]`
//! and run on the GPU host like [`super::tests`]:
//!
//! - `gpu_bench_scaling`: time and device memory at N = 2..200 frames of a
//!   Mip-NeRF 360 scene (`$SPL_VGGT_SCALING_N`, `$SPL_VGGT_SCALING_SCENE`).
//! - `gpu_dump_eval_predictions`: our depth and extrinsics on the vggt-ref
//!   evaluation subsets, scored by `tools/vggt/eval_subset.py --predictions`.
//! - `gpu_pose_op_vs_colmap`: the pose op on a Mip-NeRF 360 scene, poses
//!   against COLMAP after a Sim(3) alignment (ATE, RPE, rotation error).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use spl_core::{ArtifactStore, DataDir, EventSink, Op, Resources, RunCx};
use spl_nn::Nn;

use super::tests::{engine, env_path, weights_file};
use super::*;
use crate::io::colmap::ColmapModel;
use crate::pose;

// ---------------------------------------------------------------- helpers

/// Writes a little-endian f32 `.npy` in C order.
fn write_npy(path: &Path, shape: &[usize], data: &[f32]) {
    assert_eq!(shape.iter().product::<usize>(), data.len());
    let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    let tuple = if dims.len() == 1 {
        format!("{},", dims[0])
    } else {
        dims.join(", ")
    };
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({tuple}), }}");
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

fn datasets() -> PathBuf {
    env_path("SPL_DATASETS", "/datasets")
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

/// The first of `images_4`, `images_2`, `images` that exists in a scene.
fn scene_images(scene: &Path) -> PathBuf {
    ["images_4", "images_2", "images"]
        .iter()
        .map(|d| scene.join(d))
        .find(|d| d.is_dir())
        .unwrap_or_else(|| panic!("no image folder in {}", scene.display()))
}

fn decode_all(files: &[PathBuf]) -> Vec<crate::io::images::Rgb8Image> {
    use rayon::prelude::*;
    files
        .par_iter()
        .map(|f| preprocess::decode(&std::fs::read(f).unwrap()).unwrap())
        .collect()
}

fn free_bytes() -> usize {
    spl_nn::cudarc::driver::result::mem_get_info().unwrap().0
}

// ------------------------------------------------------- Sim(3) alignment

type V3 = [f64; 3];
type M3 = [[f64; 3]; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn mv(m: &M3, v: V3) -> V3 {
    std::array::from_fn(|i| dot(m[i], v))
}

fn mm(a: &M3, b: &M3) -> M3 {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

fn tr(a: &M3) -> M3 {
    std::array::from_fn(|i| std::array::from_fn(|j| a[j][i]))
}

fn mean(p: &[V3]) -> V3 {
    let n = p.len() as f64;
    std::array::from_fn(|k| p.iter().map(|x| x[k]).sum::<f64>() / n)
}

/// Angle of a rotation matrix in degrees.
fn angle_deg(r: &M3) -> f64 {
    ((r[0][0] + r[1][1] + r[2][2] - 1.0) / 2.0)
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees()
}

/// Eigenvector of the largest eigenvalue of a symmetric 4×4 (Jacobi).
#[allow(clippy::needless_range_loop)]
fn top_eigvec4(mut a: [[f64; 4]; 4]) -> [f64; 4] {
    let mut v = [[0.0; 4]; 4];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _ in 0..64 {
        let off: f64 = (0..4)
            .flat_map(|i| (0..4).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[i][j] * a[i][j])
            .sum();
        if off < 1e-30 {
            break;
        }
        for p in 0..4 {
            for q in p + 1..4 {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..4 {
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..4 {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for row in v.iter_mut() {
                    let (vkp, vkq) = (row[p], row[q]);
                    row[p] = c * vkp - s * vkq;
                    row[q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let best = (0..4).max_by(|&i, &j| a[i][i].total_cmp(&a[j][j])).unwrap();
    std::array::from_fn(|k| v[k][best])
}

/// Similarity `b ≈ s R a + t` minimising the squared error (Horn's
/// quaternion method; Umeyama without reflections).
fn sim3(a: &[V3], b: &[V3]) -> (f64, M3, V3) {
    let (ma, mb) = (mean(a), mean(b));
    let mut s = [[0.0f64; 3]; 3];
    let mut va = 0.0;
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (sub(*x, ma), sub(*y, mb));
        va += dot(x, x);
        for i in 0..3 {
            for j in 0..3 {
                s[i][j] += x[i] * y[j];
            }
        }
    }
    let [[sxx, sxy, sxz], [syx, syy, syz], [szx, szy, szz]] = s;
    let n = [
        [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
        [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
        [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
        [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
    ];
    let [w, x, y, z] = top_eigvec4(n);
    let r = [
        [
            w * w + x * x - y * y - z * z,
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            w * w - x * x + y * y - z * z,
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            w * w - x * x - y * y + z * z,
        ],
    ];
    let num: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| dot(sub(*y, mb), mv(&r, sub(*x, ma))))
        .sum();
    let scale = num / va;
    let rm = mv(&r, ma);
    let t = std::array::from_fn(|k| mb[k] - scale * rm[k]);
    (scale, r, t)
}

/// Pose comparison after aligning `ours` to `gt` (world-to-camera R, t).
#[derive(Debug)]
struct PoseErrors {
    /// RMS camera-centre error, in gt units and relative to the gt
    /// trajectory's RMS radius.
    ate: f64,
    ate_rel: f64,
    /// Consecutive-pair relative pose errors: rotation (deg, mean) and
    /// translation (gt units, RMS, after the alignment's scale).
    rpe_rot: f64,
    rpe_trans: f64,
    rpe_trans_rel: f64,
    /// Absolute orientation error after alignment (deg, mean / max).
    rot_mean: f64,
    rot_max: f64,
}

fn pose_errors(ours: &[(M3, V3)], gt: &[(M3, V3)]) -> PoseErrors {
    let center = |(r, t): &(M3, V3)| -> V3 {
        let c = mv(&tr(r), *t);
        [-c[0], -c[1], -c[2]]
    };
    let co: Vec<V3> = ours.iter().map(center).collect();
    let cg: Vec<V3> = gt.iter().map(center).collect();
    let (s, ra, ta) = sim3(&co, &cg);
    let n = co.len() as f64;
    let mg = mean(&cg);
    let radius = (cg
        .iter()
        .map(|c| dot(sub(*c, mg), sub(*c, mg)))
        .sum::<f64>()
        / n)
        .sqrt();
    let ate = (co
        .iter()
        .zip(&cg)
        .map(|(a, b)| {
            let p = mv(&ra, *a);
            let d = sub([s * p[0] + ta[0], s * p[1] + ta[1], s * p[2] + ta[2]], *b);
            dot(d, d)
        })
        .sum::<f64>()
        / n)
        .sqrt();
    // Camera-to-world rotations in the gt frame.
    let c2w_o: Vec<M3> = ours.iter().map(|(r, _)| mm(&ra, &tr(r))).collect();
    let c2w_g: Vec<M3> = gt.iter().map(|(r, _)| tr(r)).collect();
    let rot: Vec<f64> = c2w_o
        .iter()
        .zip(&c2w_g)
        .map(|(o, g)| angle_deg(&mm(&tr(g), o)))
        .collect();
    let (mut rr, mut rt) = (Vec::new(), Vec::new());
    for i in 1..ours.len() {
        // Relative motion i-1 → i in camera i-1's frame.
        let rel = |c2w: &[M3], c: &[V3], k: f64| -> (M3, V3) {
            let d = sub(c[i], c[i - 1]);
            let d = [k * d[0], k * d[1], k * d[2]];
            (mm(&tr(&c2w[i - 1]), &c2w[i]), mv(&tr(&c2w[i - 1]), d))
        };
        let co_g: Vec<V3> = co.iter().map(|c| mv(&ra, *c)).collect();
        let (r_o, t_o) = rel(&c2w_o, &co_g, s);
        let (r_g, t_g) = rel(&c2w_g, &cg, 1.0);
        rr.push(angle_deg(&mm(&tr(&r_g), &r_o)));
        rt.push(dot(sub(t_o, t_g), sub(t_o, t_g)));
    }
    let rpe_trans = (rt.iter().sum::<f64>() / rt.len().max(1) as f64).sqrt();
    PoseErrors {
        ate,
        ate_rel: ate / radius,
        rpe_rot: rr.iter().sum::<f64>() / rr.len().max(1) as f64,
        rpe_trans,
        rpe_trans_rel: rpe_trans / radius,
        rot_mean: rot.iter().sum::<f64>() / n,
        rot_max: rot.iter().copied().fold(0.0, f64::max),
    }
}

fn rot_z(deg: f64) -> M3 {
    let (s, c) = deg.to_radians().sin_cos();
    [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
}

fn rot_x(deg: f64) -> M3 {
    let (s, c) = deg.to_radians().sin_cos();
    [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
}

// ---------------------------------------------------------------- CPU

#[test]
fn sim3_recovers_a_similarity() {
    let a: Vec<V3> = (0..12)
        .map(|i| {
            let f = i as f64;
            [f.sin() * 3.0, (0.7 * f).cos(), 0.3 * f - 1.0]
        })
        .collect();
    let r = mm(&rot_z(37.0), &rot_x(-112.0));
    let b: Vec<V3> = a
        .iter()
        .map(|p| {
            let q = mv(&r, *p);
            [2.5 * q[0] + 1.0, 2.5 * q[1] - 4.0, 2.5 * q[2] + 0.5]
        })
        .collect();
    let (s, ra, t) = sim3(&a, &b);
    assert!((s - 2.5).abs() < 1e-9, "{s}");
    assert!(angle_deg(&mm(&tr(&ra), &r)) < 1e-6);
    assert!((t[0] - 1.0).abs() < 1e-9 && (t[1] + 4.0).abs() < 1e-9 && (t[2] - 0.5).abs() < 1e-9);
}

#[test]
fn pose_errors_vanish_under_a_similarity() {
    // gt cameras on a circle looking inwards; ours = gt in another frame.
    let gt: Vec<(M3, V3)> = (0..10)
        .map(|i| {
            let r = mm(&rot_x(10.0 * i as f64), &rot_z(36.0 * i as f64));
            let c = [
                3.0 * (0.6 * i as f64).cos(),
                3.0 * (0.6 * i as f64).sin(),
                0.2,
            ];
            let t = mv(&r, c);
            (r, [-t[0], -t[1], -t[2]])
        })
        .collect();
    let (s, a, off) = (0.4, mm(&rot_z(-50.0), &rot_x(20.0)), [0.3, -2.0, 1.0]);
    let ours: Vec<(M3, V3)> = gt
        .iter()
        .map(|(r, t)| {
            // World change x' = s A x + off: R' = R Aᵀ, t' = s t − R' off.
            let r2 = mm(r, &tr(&a));
            let ro = mv(&r2, off);
            (r2, [s * t[0] - ro[0], s * t[1] - ro[1], s * t[2] - ro[2]])
        })
        .collect();
    let e = pose_errors(&ours, &gt);
    assert!(
        e.ate < 1e-9 && e.rot_max < 1e-6 && e.rpe_rot < 1e-6 && e.rpe_trans < 1e-9,
        "{e:?}"
    );
    let mut noisy = ours.clone();
    noisy[3].0 = mm(&rot_z(2.0), &noisy[3].0);
    let e = pose_errors(&noisy, &gt);
    assert!(e.rot_max > 1.5 && e.rot_max < 2.5, "{e:?}");
}

#[test]
fn npy_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("x.npy");
    let v: Vec<f32> = (0..24).map(|i| i as f32 * 0.5 - 3.0).collect();
    write_npy(&p, &[2, 3, 4], &v);
    let (s, w) = super::tests::npy(&p);
    assert_eq!(s, vec![2, 3, 4]);
    assert_eq!(v, w);
}

// ---------------------------------------------------------------- GPU

/// Median GPU time and device memory per N. Memory: the engine's share
/// of the device (free before loading the weights minus free after the
/// run, weights included) next to the whole card's use.
#[test]
#[ignore]
fn gpu_bench_scaling() {
    let ns: Vec<usize> = std::env::var("SPL_VGGT_SCALING_N")
        .unwrap_or_else(|_| "2,10,50,100,200".into())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let scene = std::env::var("SPL_VGGT_SCALING_SCENE").unwrap_or("mipnerf360/kitchen".into());
    let files = image_files(&scene_images(&datasets().join(&scene)));
    let nn = Arc::new(Nn::open().expect("CUDA"));
    let free0 = free_bytes();
    let total = spl_nn::cudarc::driver::result::mem_get_info().unwrap().1;
    let mut e = VggtEngine::load(nn, &weights_file(), VggtConfig::default()).unwrap();
    let after_load = free0 - free_bytes();
    println!(
        "scaling: {scene}, {} images; weights + kernels {:.0} MiB",
        files.len(),
        after_load as f64 / 1048576.0
    );
    let cancel = CancellationToken::new();
    for &n in &ns {
        assert!(n <= files.len(), "{scene} has {} images", files.len());
        let pick = pose::sample_indices(files.len(), n);
        let chosen: Vec<PathBuf> = pick.iter().map(|&i| files[i].clone()).collect();
        let p = e.preprocess(&decode_all(&chosen)).unwrap();
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
            "scaling N={n} {}x{}: median {med:.1} ms (min {:.1}, {reps} runs) trunk {:.1} camera {:.1} dense {:.1} wall {:.1} graph {} offload {} chunk {}; engine {:.0} MiB, card {:.0} MiB of {:.0}",
            p.geom.width,
            p.geom.height,
            t[0].0,
            st.trunk_ms,
            st.camera_ms,
            st.dense_ms,
            st.wall_ms,
            st.graph,
            plan.offload,
            plan.dense_chunk,
            (free0 - min_free) as f64 / 1048576.0,
            (total - min_free) as f64 / 1048576.0,
            total as f64 / 1048576.0
        );
    }
}

/// Depth and extrinsics on the evaluation subsets of the vggt-ref tools,
/// written as `$SPL_VGGT_PRED/<dataset>/<scene>/{depth,extrinsics}.npy`.
#[test]
#[ignore]
fn gpu_dump_eval_predictions() {
    let out = env_path("SPL_VGGT_PRED", "target/vggt-eval-pred");
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
            let cams = e.run(&p, &cancel, &mut |_, _| {}).unwrap();
            let d = e.download(&p.geom).unwrap();
            let g = p.geom;
            let dir = out.join(ds).join(&scene);
            write_npy(
                &dir.join("depth.npy"),
                &[g.frames, g.height, g.width],
                &d.depth,
            );
            let ex: Vec<f32> = cams.extrinsics.iter().flatten().copied().collect();
            write_npy(&dir.join("extrinsics.npy"), &[g.frames, 3, 4], &ex);
            println!(
                "eval: {ds}/{scene} {} frames {}x{} in {:.1} ms -> {}",
                g.frames,
                g.width,
                g.height,
                e.times.total_ms,
                dir.display()
            );
        }
    }
}

/// End to end: `image_to_3d.pose` on a Mip-NeRF 360 scene's photos
/// (`$SPL_POSE_SCENE`, `$SPL_POSE_FRAMES`), cameras compared with the
/// scene's COLMAP model after a Sim(3) alignment.
#[test]
#[ignore]
fn gpu_pose_op_vs_colmap() {
    let scene = std::env::var("SPL_POSE_SCENE").unwrap_or("mipnerf360/garden".into());
    let frames: u32 = std::env::var("SPL_POSE_FRAMES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    let root = datasets().join(&scene);
    let images = scene_images(&root);
    let rel_images = images.strip_prefix(datasets()).unwrap().to_string_lossy();
    let gt = ColmapModel::read_dir(&root.join("sparse/0")).unwrap();

    let store_dir = tempfile::tempdir().unwrap();
    let store = Arc::new(spl_data::LocalStore::new(store_dir.path()));
    let mut res = Resources::default();
    // The op finds the weights at `<data dir>/models/vggt-omega/` (the
    // container mounts them at /models).
    let weights = weights_file();
    let data_dir = weights
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    res.insert(Arc::new(DataDir(data_dir.to_path_buf())));
    let cx = RunCx::new("pose-e2e".into(), store.clone(), res, EventSink::default());
    let input: pose::PoseInput = serde_json::from_value(serde_json::json!({
        "dataset": rel_images,
        "max_frames": frames,
    }))
    .unwrap();
    let rt = tokio::runtime::Runtime::new().unwrap();
    let out = rt.block_on(pose::Pose.run(&cx, input)).unwrap();
    let doc: serde_json::Value =
        serde_json::from_slice(&rt.block_on(store.get(&out.cameras.id)).unwrap()).unwrap();
    assert!(rt.block_on(store.get(&out.points.id)).unwrap().len() > 1000);
    assert!(rt.block_on(store.get(&out.colmap.id)).unwrap().len() > 100);

    let (mut ours, mut want) = (Vec::new(), Vec::new());
    for f in doc["frames"].as_array().unwrap() {
        let name = f["name"].as_str().unwrap();
        let w2c: Vec<Vec<f64>> = serde_json::from_value(f["w2c"].clone()).unwrap();
        let Some(g) = gt.images.values().find(|i| i.name == name) else {
            println!("pose: {name} not in the COLMAP model");
            continue;
        };
        ours.push((
            std::array::from_fn(|i| std::array::from_fn(|j| w2c[i][j])),
            [w2c[0][3], w2c[1][3], w2c[2][3]],
        ));
        want.push((g.rotation(), g.tvec));
    }
    let e = pose_errors(&ours, &want);
    println!(
        "pose op {scene} ({rel_images}): {} frames {}x{}, {} points; forward {:.1} ms (trunk {:.1}, heads {:.1}), load {:.2} s, decode {:.2} s, wall {:.2} s",
        out.frames,
        out.width,
        out.height,
        out.point_count,
        out.times.forward_ms,
        out.times.trunk_ms,
        out.times.heads_ms,
        out.times.load_s,
        out.times.decode_s,
        out.times.wall_s
    );
    println!(
        "pose vs COLMAP ({} matched): ATE {:.4} ({:.2}% of the trajectory radius), RPE rot {:.3} deg, RPE trans {:.4} ({:.2}%), rotation error mean {:.3} max {:.3} deg",
        ours.len(),
        e.ate,
        100.0 * e.ate_rel,
        e.rpe_rot,
        e.rpe_trans,
        100.0 * e.rpe_trans_rel,
        e.rot_mean,
        e.rot_max
    );
    assert_eq!(ours.len(), out.frames as usize);
    assert!(e.ate_rel < 0.1 && e.rot_mean < 5.0, "{e:?}");
}
