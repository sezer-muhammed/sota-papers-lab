//! Runtime tests. CPU tests run everywhere; `gpu_*` and `host_*` tests are
//! `#[ignore]` and run on the GPU host with the converted weights
//! (`$SPL_VGGT_WEIGHTS`) and the oracle dumps of `scripts/vggt_oracle.py`
//! (`$SPL_VGGT_ORACLE/<set>/{bf16,fp32}/*.npy`). Owner: runtime.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::preprocess::{self, ResizeMode, aspect_crop, pil_coeffs, round_half_even, target_size};
use super::weights::{Dtype, parse_header};
use super::workspace::{Geometry, Plan};
use super::*;
use crate::pose;

// ---------------------------------------------------------------- helpers

/// A little `.npy` reader: f32 / f16 / f64, little-endian, C order.
fn npy(path: &Path) -> (Vec<usize>, Vec<f32>) {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert_eq!(&b[..6], b"\x93NUMPY");
    let (hl, start) = if b[6] == 1 {
        (u16::from_le_bytes([b[8], b[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12)
    };
    let h = std::str::from_utf8(&b[start..start + hl]).unwrap();
    assert!(h.contains("'fortran_order': False"), "{h}");
    let shape: Vec<usize> = h
        .split("'shape': (")
        .nth(1)
        .unwrap()
        .split(')')
        .next()
        .unwrap()
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let d = &b[start + hl..];
    let v = if h.contains("'<f4'") {
        d.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    } else if h.contains("'<f2'") {
        d.as_chunks::<2>()
            .0
            .iter()
            .map(|c| spl_nn::host::f16_to_f32(u16::from_le_bytes(*c)))
            .collect()
    } else if h.contains("'<f8'") {
        d.as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c) as f32)
            .collect()
    } else {
        panic!("unsupported npy dtype: {h}")
    };
    (shape, v)
}

/// RMS of `a − b` relative to the RMS of `b`, and max |a − b|.
fn rel(a: &[f32], b: &[f32]) -> (f64, f64) {
    assert_eq!(a.len(), b.len());
    let (mut se, mut sb, mut mx) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        let d = x as f64 - y as f64;
        se += d * d;
        sb += (y as f64) * (y as f64);
        mx = mx.max(d.abs());
    }
    ((se / sb.max(1e-30)).sqrt(), mx)
}

fn env_path(var: &str, default: &str) -> PathBuf {
    std::env::var_os(var)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(default))
}

fn oracle_dir() -> PathBuf {
    env_path("SPL_VGGT_ORACLE", "/oracle")
}

fn weights_file() -> PathBuf {
    env_path(
        "SPL_VGGT_WEIGHTS",
        "/models/vggt-omega/vggt_omega_1b_512.safetensors",
    )
}

/// The oracle's preprocessed input `[N,3,H,W]` in [0, 1] back to RGB8 HWC
/// (exact: the values are k / 255).
fn prepared_from_oracle(set: &str) -> Prepared {
    let (s, v) = npy(&oracle_dir().join(set).join("bf16/input.npy"));
    let (n, h, w) = (s[0], s[2], s[3]);
    let mut pixels = vec![0u8; n * h * w * 3];
    for f in 0..n {
        for c in 0..3 {
            for p in 0..h * w {
                let x = v[(f * 3 + c) * h * w + p] * 255.0;
                pixels[(f * h * w + p) * 3 + c] = x.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    Prepared {
        geom: Geometry::new(n, h, w),
        pixels,
    }
}

// ---------------------------------------------------------------- CPU

#[test]
fn rounding_is_half_even() {
    assert_eq!(round_half_even(2.5), 2.0);
    assert_eq!(round_half_even(3.5), 4.0);
    assert_eq!(round_half_even(-0.5), -0.0);
    assert_eq!(round_half_even(2.4999), 2.0);
}

#[test]
fn target_sizes_match_load_fn() {
    // Spec §2 examples.
    assert_eq!(target_size(3.0 / 4.0, 512, ResizeMode::MaxSize), (384, 512));
    assert_eq!(target_size(2.0 / 3.0, 512, ResizeMode::MaxSize), (336, 512));
    assert_eq!(target_size(4.0 / 3.0, 512, ResizeMode::MaxSize), (512, 384));
    assert_eq!(
        target_size(3.0 / 4.0, 512, ResizeMode::Balanced),
        (448, 592)
    );
    // Truck (979×546) → 512×288.
    assert_eq!(
        target_size(546.0 / 979.0, 512, ResizeMode::MaxSize),
        (288, 512)
    );
    assert_eq!(target_size(0.01, 512, ResizeMode::MaxSize), (16, 512));
}

#[test]
fn aspect_crop_limits() {
    assert_eq!(aspect_crop(1000, 100), (250, 0, 200, 100));
    assert_eq!(aspect_crop(100, 1000), (0, 400, 100, 200));
    assert_eq!(aspect_crop(640, 480), (0, 0, 640, 480));
}

#[test]
fn pil_coefficients_are_normalised() {
    for (i, o) in [(979, 512), (512, 979), (100, 37), (37, 37)] {
        for (x0, k) in pil_coeffs(i, o) {
            let s: i64 = k.iter().map(|&v| v as i64).sum();
            assert!((s - (1 << 22)).abs() <= k.len() as i64, "{i}->{o}: {s}");
            assert!(x0 + k.len() <= i);
        }
    }
}

#[test]
fn resize_keeps_constant_images_and_identity() {
    let (w, h) = (37, 23);
    let src: Vec<u8> = (0..w * h * 3).map(|i| (i * 7 % 251) as u8).collect();
    assert_eq!(preprocess::resize_bicubic(&src, w, h, w, h), src);
    let flat = vec![200u8; w * h * 3];
    let out = preprocess::resize_bicubic(&flat, w, h, 16, 64);
    assert!(out.iter().all(|&v| v == 200));
}

/// Values from PIL 11 (`Image.resize((4, 1), BICUBIC)` of a 9×1 ramp
/// `[0, 30, ..., 240]`, RGB with equal channels): a downscale with
/// antialiasing support 4.5.
#[test]
fn resize_matches_pil_downscale() {
    let src: Vec<u8> = (0..9).flat_map(|i| [(i * 30) as u8; 3]).collect();
    let out = preprocess::resize_bicubic(&src, 9, 1, 4, 1);
    let got: Vec<u8> = out.chunks(3).map(|c| c[0]).collect();
    // Computed with the reference formula in f64 (see pil_coeffs):
    // centres 1.125, 3.375, 5.625, 7.875 on a linear ramp → ≈ 30·(c − 0.5).
    let want = [24u8, 86, 154, 214];
    for (g, w) in got.iter().zip(want) {
        assert!((*g as i32 - w as i32).abs() <= 2, "{got:?} vs {want:?}");
    }
}

#[test]
fn prepare_resizes_every_frame_to_the_first() {
    let mk = |w: u32, h: u32| crate::io::images::Rgb8Image {
        width: w,
        height: h,
        data: vec![128; (w * h * 3) as usize],
    };
    let p = preprocess::prepare(
        &[mk(979, 546), mk(546, 979), mk(1000, 560)],
        512,
        ResizeMode::MaxSize,
    )
    .unwrap();
    assert_eq!(p.geom, Geometry::new(3, 288, 512));
    assert_eq!(p.geom.tokens(), 17 + 32 * 18);
    assert_eq!(p.pixels.len(), 3 * 288 * 512 * 3);
}

fn write_st(entries: &[(&str, &str, Vec<usize>, Vec<u8>)], meta: &str) -> Vec<u8> {
    let mut off = 0;
    let mut h = String::from("{");
    let mut data = Vec::new();
    for (n, d, s, b) in entries {
        h.push_str(&format!(
            "\"{n}\":{{\"dtype\":\"{d}\",\"shape\":{s:?},\"data_offsets\":[{off},{}]}},",
            off + b.len()
        ));
        off += b.len();
        data.extend_from_slice(b);
    }
    h.push_str(&format!("\"__metadata__\":{{{meta}}}}}"));
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(h.as_bytes());
    out.extend_from_slice(&data);
    out
}

#[test]
fn safetensors_header_parses_and_validates() {
    let f = write_st(
        &[
            ("a", "BF16", vec![2, 2], vec![0; 8]),
            ("__pad.0", "U8", vec![248], vec![0; 248]),
            ("b", "F32", vec![3], vec![0; 12]),
        ],
        "\"device_bytes\":\"256\"",
    );
    let h = parse_header(&f, f.len()).unwrap();
    assert_eq!(h.device_bytes, 256);
    assert_eq!(h.tensors["a"].dtype, Dtype::Bf16);
    assert_eq!(h.tensors["b"].start, 256);
    // Wrong size for the shape.
    let bad = write_st(&[("a", "F32", vec![3], vec![0; 8])], "");
    assert!(parse_header(&bad, bad.len()).is_err());
    // Truncated data.
    assert!(parse_header(&f, f.len() - 1).is_err());
}

#[test]
fn schedule_defaults_to_the_official_layers() {
    let s = AttnSchedule::dense();
    let reg: Vec<usize> = (0..24).filter(|&i| s.0[i] == InterAttn::Register).collect();
    assert_eq!(reg, REGISTER_LAYERS);
    let mut p2 = s;
    p2.0[0] = InterAttn::Sage;
    assert!(super::trunk_schedule(&p2).is_err());
    assert!(super::trunk_schedule(&s).is_ok());
}

#[test]
fn plan_chunks_and_offload() {
    let g = Geometry::new(20, 336, 512);
    let p = Plan::new(g, 8, 10 << 30, 1 << 30, 200 << 20).unwrap();
    let c: Vec<_> = p.chunks().collect();
    assert_eq!(c, vec![0..8, 8..16, 16..20]);
    assert!(!p.offload);
    // 300 frames: the cache alone is 4 × 300 × 689 × 2048 × 2 B ≈ 3.4 GB.
    let big = Geometry::new(300, 336, 512);
    let tight = Plan::new(big, 8, 4_400 << 20, 600 << 20, 400 << 20).unwrap();
    assert!(tight.offload);
    assert!(Plan::new(big, 8, 2 << 30, 600 << 20, 400 << 20).is_err());
}

#[test]
fn sample_indices_are_even_and_bounded() {
    assert_eq!(pose::sample_indices(5, 10), vec![0, 1, 2, 3, 4]);
    assert_eq!(pose::sample_indices(10, 1), vec![0]);
    let s = pose::sample_indices(100, 4);
    assert_eq!(s.first(), Some(&0));
    assert_eq!(s.last(), Some(&99));
    assert!(s.windows(2).all(|w| w[0] < w[1]));
}

#[test]
fn quaternion_round_trip() {
    // 90° about z.
    let r = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    let q = pose::mat_to_quat(r);
    let h = std::f64::consts::FRAC_1_SQRT_2;
    for (a, b) in q.iter().zip([h, 0.0, 0.0, h]) {
        assert!((a - b).abs() < 1e-6, "{q:?}");
    }
    // 180° about x: the trace branch with a negative trace.
    let q = pose::mat_to_quat([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]);
    assert!((q[1].abs() - 1.0).abs() < 1e-6, "{q:?}");
}

fn tiny_run() -> (Prepared, DenseHost, VggtCameras) {
    let g = Geometry::new(2, 16, 32);
    let n = g.frames * g.pixels();
    let prepared = Prepared {
        geom: g,
        pixels: (0..n * 3).map(|i| i as u8).collect(),
    };
    let dense = DenseHost {
        depth: vec![2.0; n],
        conf: (0..n).map(|i| i as f32).collect(),
        world: (0..n * 3).map(|i| i as f32 * 0.01).collect(),
    };
    let cams = VggtCameras {
        pose_enc: vec![[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]; 2],
        extrinsics: vec![[1.0, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]; 2],
        intrinsics: vec![[30.0, 0.0, 16.0, 0.0, 30.0, 8.0, 0.0, 0.0, 1.0]; 2],
        width: 32,
        height: 16,
    };
    (prepared, dense, cams)
}

#[test]
fn point_cloud_keeps_confident_points_and_caps_count() {
    let (p, d, _) = tiny_run();
    let n = p.geom.frames * p.geom.pixels();
    let all = pose::point_cloud(&p, &d, 0.0, usize::MAX);
    assert_eq!(all.len(), n);
    let half = pose::point_cloud(&p, &d, 50.0, usize::MAX);
    assert!((half.len() as i64 - n as i64 / 2).abs() <= 1);
    let capped = pose::point_cloud(&p, &d, 0.0, 100);
    assert!(
        capped.len() <= 100 && capped.len() >= 90,
        "{}",
        capped.len()
    );
    assert_eq!(all[5].rgb, [15, 16, 17]);
}

#[test]
fn exports_are_well_formed() {
    let (p, d, c) = tiny_run();
    let pts = pose::point_cloud(&p, &d, 90.0, usize::MAX);
    let mut ply = Vec::new();
    pose::write_points_ply(&pts, &c, &mut ply).unwrap();
    let head = String::from_utf8_lossy(&ply[..400]);
    assert!(head.contains(&format!("element vertex {}", pts.len())));
    let end = ply.windows(11).position(|w| w == b"end_header\n").unwrap() + 11;
    assert_eq!(ply.len() - end, pts.len() * 17 * 4);
    let names = vec!["a.jpg".to_string(), "b c.jpg".to_string()];
    let (cams, imgs, points) = pose::colmap_text(&names, &c, &pts);
    assert!(cams.contains("1 PINHOLE 32 16 30 30 16 8"));
    assert!(imgs.contains("2 1 0 0 0 0.5 0 0 2 b_c.jpg"));
    assert_eq!(
        points.lines().filter(|l| !l.starts_with('#')).count(),
        pts.len()
    );
}

// ---------------------------------------------------------------- GPU host

/// The oracle's preprocessed tensors equal our CPU preprocessing of the
/// same files (decode + crop + PIL bicubic). Needs the dataset at
/// `$SPL_DATASETS` and the oracle dumps.
#[test]
#[ignore]
fn host_preprocess_matches_oracle() {
    let datasets = env_path("SPL_DATASETS", "/datasets");
    let dir = datasets.join("tandt/truck/images");
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    files.sort();
    let frame = preprocess::decode(&std::fs::read(&files[0]).unwrap()).unwrap();
    let ours = preprocess::prepare(&[frame], 512, ResizeMode::MaxSize).unwrap();
    let want = prepared_from_oracle("t1");
    assert_eq!(ours.geom, want.geom);
    let diff: Vec<i32> = ours
        .pixels
        .iter()
        .zip(&want.pixels)
        .map(|(&a, &b)| a as i32 - b as i32)
        .collect();
    let max = diff.iter().map(|d| d.abs()).max().unwrap();
    let off = diff.iter().filter(|&&d| d != 0).count();
    println!(
        "preprocess vs PIL: max |Δ| {max} LSB, {off} of {} samples differ",
        diff.len()
    );
    assert!(max <= 1, "max |Δ| {max}");
    assert!((off as f64) < 1e-3 * diff.len() as f64);
}

fn engine() -> VggtEngine {
    let nn = Arc::new(Nn::open().expect("CUDA"));
    VggtEngine::load(nn, &weights_file(), VggtConfig::default()).expect("load engine")
}

#[test]
#[ignore]
fn gpu_weights_upload_and_views() {
    let nn = Nn::open().unwrap();
    let t = std::time::Instant::now();
    let w = Weights::load(&nn, &[weights_file()]).unwrap();
    println!(
        "weights: {:.3} GB on device, bulk upload {:.2} s, load {:.2} s",
        w.device_bytes() as f64 / 1e9,
        w.upload_seconds(),
        t.elapsed().as_secs_f64()
    );
    for name in [
        "aggregator.frame_blocks.3.attn.qkv.weight",
        "aggregator.patch_embed.blocks.0.norm1.weight",
        "aggregator.inter_frame_blocks.23.mlp.fc2.bias",
    ] {
        let host = w.host_f32(name).unwrap();
        let dev: Vec<f32> = match w.info(name).unwrap().dtype {
            Dtype::Bf16 => nn
                .stream
                .clone_dtoh(&w.bf16(name).unwrap())
                .unwrap()
                .iter()
                .map(|&b| spl_nn::host::bf16_to_f32(b))
                .collect(),
            _ => nn.stream.clone_dtoh(&w.f32(name).unwrap()).unwrap(),
        };
        assert_eq!(host, dev, "{name}");
    }
    // Masked QKV biases are gone; head tensors stay on the host.
    assert!(!w.has("aggregator.frame_blocks.0.attn.qkv.bias"));
    assert!(w.bf16("camera_head.trunk.0.attn.qkv.weight").is_err());
    let b = w.host_f32("camera_head.trunk.0.attn.qkv.bias").unwrap();
    assert!(
        b[2048..4096].iter().all(|&v| v == 0.0),
        "k bias must be masked"
    );
    assert_eq!(
        w.host_f32("aggregator.patch_embed.rope_embed.periods")
            .unwrap()[1],
        1.3359375
    );
}

#[test]
#[ignore]
fn gpu_normalize_matches_torch_formula() {
    let mut e = engine();
    let p = prepared_from_oracle("t1");
    let (_, want) = npy(&oracle_dir().join("t1/bf16/input.npy"));
    let g = p.geom;
    let nn = e.nn.clone();
    let ws = e.workspace(g).unwrap();
    nn.stream.memcpy_htod(&p.pixels, &mut ws.pixels).unwrap();
    let f = e.normalize.clone();
    let ws = e.workspaces.get_mut(&g).unwrap();
    super::normalize_launch(&nn, &f, ws, g).unwrap();
    let got: Vec<f32> = nn.stream.clone_dtoh(&ws.images).unwrap();
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];
    let hw = g.pixels();
    for (i, (&a, &x)) in got.iter().zip(&want).enumerate() {
        let c = (i / hw) % 3;
        assert_eq!(a, (x - mean[c]) / std[c], "element {i}");
    }
}

struct Parity {
    set: &'static str,
}

impl Parity {
    fn run(&self, e: &mut VggtEngine) {
        let p = prepared_from_oracle(self.set);
        let g = p.geom;
        let cancel = CancellationToken::new();
        let cams = e.run(&p, &cancel, &mut |_, _| {}).unwrap();
        let dense = e.download(&g).unwrap();
        let o = oracle_dir().join(self.set);
        let ld = |prec: &str, n: &str| npy(&o.join(prec).join(n)).1;
        let pe: Vec<f32> = cams.pose_enc.iter().flatten().copied().collect();
        let report = |name: &str, ours: &[f32], gate: f64| {
            let b = ld("bf16", name);
            let f = ld("fp32", name);
            let (e_b, m_b) = rel(ours, &b);
            let (e_f, _) = rel(ours, &f);
            let (floor, _) = rel(&b, &f);
            println!(
                "{} {name}: rel vs bf16 {e_b:.2e} (max {m_b:.2e}), vs fp32 {e_f:.2e}, official bf16-vs-fp32 {floor:.2e}",
                self.set
            );
            assert!(
                e_b < gate || e_f <= 1.5 * floor.max(1e-6),
                "{name} out of budget"
            );
        };
        report("pose_enc.npy", &pe, 1e-2);
        report("depth.npy", &dense.depth, 1e-2);
        report("depth_conf.npy", &dense.conf, 1e-2);
        // Layer by layer: the cached levels hold [F_i | G_i] for i = 4, 23.
        let ws = e.workspace_of(&g).unwrap();
        for (k, i) in [(0usize, 4usize), (3, 23)] {
            let path = o.join("bf16").join(format!("F_{i}.npy"));
            if !path.is_file() {
                continue;
            }
            if let Some(c) = ws.cache[k].as_ref() {
                let raw: Vec<u16> = e.nn.stream.clone_dtoh(c).unwrap();
                let t = g.tokens();
                let (mut fv, mut gv) = (Vec::new(), Vec::new());
                for r in 0..g.frames * t {
                    let row = &raw[r * 2048..(r + 1) * 2048];
                    fv.extend(row[..1024].iter().map(|&b| spl_nn::host::bf16_to_f32(b)));
                    gv.extend(row[1024..].iter().map(|&b| spl_nn::host::bf16_to_f32(b)));
                }
                for (name, v) in [(format!("F_{i}.npy"), fv), (format!("G_{i}.npy"), gv)] {
                    let (eb, _) = rel(&v, &ld("bf16", &name));
                    println!("{} {name}: rel vs bf16 {eb:.2e}", self.set);
                    assert!(eb < 1e-2, "{name}: {eb}");
                }
            }
        }
    }
}

#[test]
#[ignore]
fn gpu_parity_with_oracle() {
    let mut e = engine();
    for set in ["t1", "t2", "g8", "t32"] {
        if oracle_dir().join(set).is_dir() {
            Parity { set }.run(&mut e);
        }
    }
}

/// Median of 10 runs after warm-up per set; prints stage times.
#[test]
#[ignore]
fn gpu_bench_engine() {
    let mut e = engine();
    for set in ["t1", "t2", "g8", "t32"] {
        if !oracle_dir().join(set).is_dir() {
            continue;
        }
        let p = prepared_from_oracle(set);
        let cancel = CancellationToken::new();
        for _ in 0..3 {
            e.run(&p, &cancel, &mut |_, _| {}).unwrap();
        }
        let mut t = Vec::new();
        let mut last = StageTimes::default();
        for _ in 0..10 {
            e.run(&p, &cancel, &mut |_, _| {}).unwrap();
            t.push(e.times.total_ms);
            last = e.times.clone();
        }
        t.sort_by(f32::total_cmp);
        let torch = std::fs::read_to_string(oracle_dir().join(set).join("bf16/time.json")).ok();
        println!(
            "{set}: N={} {}x{}: median {:.1} ms (min {:.1}); last: trunk {:.1} camera {:.1} dense {:.1} upload {:.2} wall {:.1} graph {}; torch {}",
            p.geom.frames,
            p.geom.width,
            p.geom.height,
            t[5],
            t[0],
            last.trunk_ms,
            last.camera_ms,
            last.dense_ms,
            last.upload_ms,
            last.wall_ms,
            last.graph,
            torch.unwrap_or_default()
        );
    }
}
