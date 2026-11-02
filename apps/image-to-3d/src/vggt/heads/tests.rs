//! Heads tests (owner: heads). CPU: weight re-layouts and the pose
//! conversion. GPU (`#[ignore]`, run in the CUDA container with the model
//! directory mounted):
//!
//! - `gpu_heads_parity_n2`: both heads on the official layer cache of 2
//!   frames (`capture.py` → `heads_n2.npz`), every tapped intermediate and
//!   the outputs against the official fp32 heads, plus the noise floor of a
//!   bf16 layer cache (official heads on bf16-rounded inputs).
//! - `gpu_heads_chunked_n9`: 9 frames from a bf16 cache, dense chunks 8 + 1,
//!   world points.
//! - `bench_heads`: CUDA-event timings per head and per dense stage.
//!
//! Inputs: `$SPL_VGGT_WEIGHTS` (default
//! `~/spl-work/models/vggt-omega/vggt_omega_1b_512.safetensors`, the
//! runtime's converted file; head tensors F32) and `$SPL_VGGT_HEADS_DUMPS`
//! (default `/out/eng-heads`). Tests skip with a note when inputs are
//! missing.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use spl_core::{Context, Result, bail};
use spl_nn::Nn;
use spl_nn::cudarc::driver::{CudaSlice, sys::CUevent_flags};
use spl_nn::host;

use super::camera::CameraHead;
use super::dense::DenseHead;
use super::{
    DIM, DeviceCams, Geometry, HeadSource, Kernels, SPECIAL, deconv_rows, encoding_to_camera, ohwi,
};
use crate::testkit::npz::{NpyData, Npz};

// ------------------------------------------------------------------ CPU --

#[test]
fn ohwi_moves_input_channels_last() {
    // o = 2, i = 3, k = 2: w[o][i][t] = 100 o + 10 i + t.
    let w: Vec<f32> = (0..2)
        .flat_map(|o| (0..3).flat_map(move |i| (0..4).map(move |t| (100 * o + 10 * i + t) as f32)))
        .collect();
    let r = ohwi(&w, 2, 3, 2);
    // r[o][t][i]
    assert_eq!(r[(4 + 2) * 3 + 1], 112.0);
    assert_eq!(r[3 * 3 + 2], 23.0);
}

#[test]
fn deconv_rows_are_tap_major() {
    // i = 2, o = 3, s = 2: w[i][o][t] = 100 i + 10 o + t → r[t][o][i].
    let w: Vec<f32> = (0..2)
        .flat_map(|i| (0..3).flat_map(move |o| (0..4).map(move |t| (100 * i + 10 * o + t) as f32)))
        .collect();
    let r = deconv_rows(&w, 2, 3, 2);
    assert_eq!(r[(3 * 3 + 2) * 2 + 1], 123.0);
}

#[test]
fn identity_pose_and_centred_intrinsics() {
    let fov = 2.0 * (0.5f32).atan(); // tan(fov/2) = 0.5
    let (e, k) = encoding_to_camera(&[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 2.0, fov, fov], 336, 512);
    assert_eq!(
        e,
        [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 3.0]
    );
    assert!((k[0] - 512.0).abs() < 1e-3 && (k[4] - 336.0).abs() < 1e-3);
    assert_eq!((k[2], k[5]), (256.0, 168.0));
}

// --------------------------------------------------------------- inputs --

/// Minimal safetensors reader (F32 / BF16 tensors, read on demand).
struct SafeTensors {
    path: PathBuf,
    base: u64,
    meta: HashMap<String, (String, Vec<usize>, u64, u64)>,
}

impl SafeTensors {
    fn open(path: &Path) -> Result<Self> {
        let mut f = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let mut len = [0u8; 8];
        f.read_exact(&mut len)?;
        let n = u64::from_le_bytes(len);
        let mut hdr = vec![0u8; n as usize];
        f.read_exact(&mut hdr)?;
        let v: serde_json::Value = serde_json::from_slice(&hdr)?;
        let mut meta = HashMap::new();
        for (k, t) in v.as_object().context("safetensors header")? {
            if k == "__metadata__" {
                continue;
            }
            let dtype = t["dtype"].as_str().unwrap_or("").to_string();
            let shape = t["shape"]
                .as_array()
                .context("shape")?
                .iter()
                .map(|d| d.as_u64().unwrap_or(0) as usize)
                .collect();
            let off = t["data_offsets"].as_array().context("offsets")?;
            meta.insert(
                k.clone(),
                (
                    dtype,
                    shape,
                    off[0].as_u64().unwrap_or(0),
                    off[1].as_u64().unwrap_or(0),
                ),
            );
        }
        Ok(Self {
            path: path.to_path_buf(),
            base: 8 + n,
            meta,
        })
    }
}

impl HeadSource for SafeTensors {
    fn host_f32(&self, name: &str) -> Result<Vec<f32>> {
        let (dtype, _, a, b) = self
            .meta
            .get(name)
            .with_context(|| format!("tensor {name} missing"))?;
        let mut f = File::open(&self.path)?;
        f.seek(SeekFrom::Start(self.base + a))?;
        let mut raw = vec![0u8; (b - a) as usize];
        f.read_exact(&mut raw)?;
        Ok(match dtype.as_str() {
            "F32" => raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect(),
            "BF16" => raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| host::bf16_to_f32(u16::from_le_bytes(*c)))
                .collect(),
            d => bail!("{name}: dtype {d} not supported"),
        })
    }

    fn shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self
            .meta
            .get(name)
            .with_context(|| format!("tensor {name} missing"))?
            .1
            .clone())
    }
}

fn weights_path() -> PathBuf {
    std::env::var_os("SPL_VGGT_WEIGHTS")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var("HOME").unwrap_or_default();
            PathBuf::from(home).join("spl-work/models/vggt-omega/vggt_omega_1b_512.safetensors")
        })
}

fn dumps_dir() -> PathBuf {
    std::env::var_os("SPL_VGGT_HEADS_DUMPS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/out/eng-heads"))
}

struct Setup {
    nn: Nn,
    k: Arc<Kernels>,
    camera: CameraHead,
    dense: DenseHead,
}

fn setup() -> Option<Setup> {
    let wp = weights_path();
    if !wp.exists() {
        println!("skip: weights {} not found", wp.display());
        return None;
    }
    let src = SafeTensors::open(&wp).expect("weights");
    let nn = Nn::open().expect("GPU");
    let k = Kernels::new(&nn).expect("kernels");
    let camera = CameraHead::new(&nn, k.clone(), &src).expect("camera head");
    let dense = DenseHead::new(&nn, k.clone(), &src).expect("dense head");
    Some(Setup {
        nn,
        k,
        camera,
        dense,
    })
}

fn load_npz(name: &str) -> Option<Npz> {
    let p = dumps_dir().join(name);
    match std::fs::read(&p) {
        Ok(b) => Some(Npz::from_bytes(&b).expect("npz")),
        Err(_) => {
            println!("skip: {} not found (run capture.py)", p.display());
            None
        }
    }
}

fn f32s(z: &Npz, name: &str) -> Vec<f32> {
    if let Ok(a) = z.get(&format!("{name}_f16"))
        && let NpyData::U16(v) = &a.data
    {
        return v.iter().map(|&h| host::f16_to_f32(h)).collect();
    }
    let a = z.get(name).unwrap_or_else(|_| panic!("{name} missing"));
    match &a.data {
        NpyData::F32(v) => v.clone(),
        _ => a.to_f64().into_iter().map(|x| x as f32).collect(),
    }
}

fn shape(z: &Npz, name: &str) -> Vec<usize> {
    z.get(name)
        .or_else(|_| z.get(&format!("{name}_f16")))
        .unwrap()
        .shape
        .clone()
}

/// Layer caches as the trunk stores them: bf16 `[N·T][2048]`.
fn caches(nn: &Nn, z: &Npz, bf16_keys: bool) -> [CudaSlice<u16>; 4] {
    [4, 11, 17, 23].map(|i| {
        let bits: Vec<u16> = if bf16_keys {
            match &z.get(&format!("L{i}_bf16")).unwrap().data {
                NpyData::U16(v) => v.clone(),
                _ => panic!("L{i}_bf16 not uint16"),
            }
        } else {
            host::to_bf16(&f32s(z, &format!("L{i}")))
        };
        nn.stream.clone_htod(&bits).unwrap()
    })
}

fn f16s(nn: &Nn, d: &CudaSlice<u16>) -> Vec<f32> {
    nn.stream
        .clone_dtoh(d)
        .unwrap()
        .into_iter()
        .map(host::f16_to_f32)
        .collect()
}

/// NCHW → NHWC for comparing against our activations.
fn nhwc(v: &[f32], s: &[usize]) -> Vec<f32> {
    let (n, c, h, w) = (s[0], s[1], s[2], s[3]);
    let mut o = vec![0f32; v.len()];
    for ni in 0..n {
        for ci in 0..c {
            for y in 0..h {
                for x in 0..w {
                    o[((ni * h + y) * w + x) * c + ci] = v[((ni * c + ci) * h + y) * w + x];
                }
            }
        }
    }
    o
}

struct Rel {
    rel_rms: f64,
    max_abs: f64,
}

fn rel(got: &[f32], want: &[f32]) -> Rel {
    assert_eq!(got.len(), want.len(), "length mismatch");
    let (mut se, mut sw, mut mx) = (0f64, 0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        let d = g as f64 - w as f64;
        assert!(d.is_finite(), "non-finite {g} vs {w}");
        se += d * d;
        sw += (w as f64) * (w as f64);
        mx = mx.max(d.abs());
    }
    Rel {
        rel_rms: (se / sw.max(1e-30)).sqrt(),
        max_abs: mx,
    }
}

fn report(name: &str, got: &[f32], want: &[f32], floor: Option<&[f32]>, limit: f64) -> f64 {
    let e = rel(got, want);
    let fl = floor.map(|f| rel(f, want).rel_rms);
    println!(
        "{name:>18}: rel_rms {:.3e} max_abs {:.3e}{}",
        e.rel_rms,
        e.max_abs,
        fl.map(|f| format!("  (bf16-cache floor {f:.3e})"))
            .unwrap_or_default()
    );
    assert!(
        e.rel_rms <= limit,
        "{name}: rel_rms {:.3e} > {limit:.1e}",
        e.rel_rms
    );
    e.rel_rms
}

/// Median |a − b| / b.
fn median_abs_rel(got: &[f32], want: &[f32]) -> f64 {
    let mut r: Vec<f64> = got
        .iter()
        .zip(want)
        .map(|(&g, &w)| ((g - w) / w).abs() as f64)
        .collect();
    r.sort_by(|a, b| a.partial_cmp(b).unwrap());
    r[r.len() / 2]
}

/// Angle (degrees) between rotations of two row-major 3×4 extrinsics.
fn rot_err_deg(a: &[f32], b: &[f32]) -> f64 {
    let mut tr = 0f64;
    for i in 0..3 {
        for j in 0..3 {
            tr += a[i * 4 + j] as f64 * b[i * 4 + j] as f64;
        }
    }
    ((tr - 1.0) / 2.0).clamp(-1.0, 1.0).acos().to_degrees()
}

fn dir_err_deg(a: [f32; 3], b: [f32; 3]) -> f64 {
    let d: f64 = (0..3).map(|i| a[i] as f64 * b[i] as f64).sum();
    let na: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    if na < 1e-6 || nb < 1e-6 {
        return 0.0;
    }
    (d / (na * nb)).clamp(-1.0, 1.0).acos().to_degrees()
}

struct Outputs {
    pose: Vec<f32>,
    extr: Vec<f32>,
    intr: Vec<f32>,
    depth: Vec<f32>,
    conf: Vec<f32>,
    world: Vec<f32>,
}

fn run_heads(s: &Setup, cache: &[CudaSlice<u16>; 4], g: &Geometry, chunk: usize) -> Outputs {
    let nn = &s.nn;
    let n = g.frames;
    let mut cams = DeviceCams::new(nn, n).unwrap();
    let pose = s.camera.run(nn, &cache[3].slice(..), g, &mut cams).unwrap();
    let px = g.height * g.width;
    let mut depth = nn.stream.alloc_zeros::<u16>(n * px).unwrap();
    let mut conf = nn.stream.alloc_zeros::<u16>(n * px).unwrap();
    let mut world = nn.stream.alloc_zeros::<f32>(n * px * 3).unwrap();
    let rows = g.tokens() * DIM;
    for f0 in (0..n).step_by(chunk) {
        let c = chunk.min(n - f0);
        let gc = g.with_frames(c);
        let layers = [0, 1, 2, 3].map(|i| cache[i].slice(f0 * rows..(f0 + c) * rows));
        let e = cams.extrinsics.slice(f0 * 12..(f0 + c) * 12);
        let k = cams.intrinsics.slice(f0 * 9..(f0 + c) * 9);
        s.dense
            .run_chunk(
                nn,
                layers,
                &gc,
                &mut depth.slice_mut(f0 * px..(f0 + c) * px),
                &mut conf.slice_mut(f0 * px..(f0 + c) * px),
                Some((&e, &k)),
                Some(&mut world.slice_mut(f0 * px * 3..(f0 + c) * px * 3)),
            )
            .unwrap();
    }
    nn.sync().unwrap();
    Outputs {
        pose: pose.concat(),
        extr: nn.stream.clone_dtoh(&cams.extrinsics).unwrap(),
        intr: nn.stream.clone_dtoh(&cams.intrinsics).unwrap(),
        depth: f16s(nn, &depth),
        conf: f16s(nn, &conf),
        world: nn.stream.clone_dtoh(&world).unwrap(),
    }
}

fn geometry(z: &Npz, frames: usize) -> Geometry {
    let hw = f32s(z, "image_hw");
    Geometry::new(frames, hw[0] as usize, hw[1] as usize)
}

/// Pose / depth gates of spec §8 against the official outputs.
fn gates(z: &Npz, o: &Outputs, n: usize) {
    let e = f32s(z, "extrinsics");
    let pe = f32s(z, "pose_enc");
    // Noise floor of the bf16 layer cache when the dump has it.
    let floor_pe = z
        .get("pose_enc_bf16in")
        .ok()
        .map(|_| f32s(z, "pose_enc_bf16in"));
    let t = |v: &[f32], f: usize| [v[f * 12 + 3], v[f * 12 + 7], v[f * 12 + 11]];
    let norm = |a: [f32; 3]| a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    let tmax = (0..n).map(|f| norm(t(&e, f))).fold(0.0, f64::max);
    let (mut rot, mut dir, mut fov, mut fov_floor) = (0f64, 0f64, 0f64, 0f64);
    for f in 0..n {
        rot = rot.max(rot_err_deg(
            &o.extr[f * 12..f * 12 + 12],
            &e[f * 12..f * 12 + 12],
        ));
        // The direction of a near-zero translation (frame 0 is the world
        // frame) is noise: only frames with |T| ≥ 5% of the largest.
        if norm(t(&e, f)) >= 0.05 * tmax {
            dir = dir.max(dir_err_deg(t(&o.extr, f), t(&e, f)));
        }
        for j in 7..9 {
            fov = fov.max(
                ((o.pose[f * 9 + j] - pe[f * 9 + j]) as f64)
                    .abs()
                    .to_degrees(),
            );
            if let Some(fl) = &floor_pe {
                fov_floor =
                    fov_floor.max(((fl[f * 9 + j] - pe[f * 9 + j]) as f64).abs().to_degrees());
            }
        }
    }
    let dmed = median_abs_rel(&o.depth, &f32s(z, "depth"));
    let cmed = median_abs_rel(&o.conf, &f32s(z, "depth_conf"));
    println!(
        "gates: rot {rot:.4}° (<0.1) tdir {dir:.4}° (<0.1) fov {fov:.4}° (<0.05; bf16-cache floor {fov_floor:.4}°) depth med abs-rel {:.4}% (<0.5) conf med {:.4}% (<1)",
        100.0 * dmed,
        100.0 * cmed
    );
    assert!(
        rot < 0.1 && dir < 0.1 && dmed < 5e-3 && cmed < 1e-2,
        "spec §8 gates failed"
    );
    // fov: the spec's absolute gate, or 1.5× the bf16-cache floor when that
    // is larger (§8: "no worse than 1.5× the bf16-vs-fp32 distance").
    assert!(
        fov < 0.05f64.max(1.5 * fov_floor) || (floor_pe.is_none() && fov < 0.06),
        "fov gate failed: {fov:.4}°"
    );
}

// ------------------------------------------------------------------ GPU --

#[test]
#[ignore]
fn gpu_heads_parity_n2() {
    let Some(s) = setup() else { return };
    let Some(z) = load_npz("heads_n2.npz") else {
        return;
    };
    let n = shape(&z, "L23")[0];
    let g = geometry(&z, n);
    println!("geometry {g:?}");
    let cache = caches(&s.nn, &z, false);
    s.k.taps.enable(true);
    let o = run_heads(&s, &cache, &g, 8);
    s.k.taps.enable(false);
    let taps: HashMap<String, Vec<f32>> = s.k.taps.take().into_iter().collect();

    // Layer by layer (targets: rel_rms ≤ 1e-2 against the official fp32 heads).
    let mut worst = 0f64;
    let mut cmp = |name: &str, nchw: bool| {
        let want = f32s(&z, name);
        let want = if nchw {
            nhwc(&want, &shape(&z, name))
        } else {
            want
        };
        worst = worst.max(report(name, &taps[name], &want, None, 1e-2));
    };
    cmp("cam_token_norm", false);
    for i in 0..4 {
        cmp(&format!("cam_block{i}"), false);
    }
    for k in 0..4 {
        cmp(&format!("dense_resize{k}"), true);
    }
    for k in 0..4 {
        cmp(&format!("dense_rn{k}"), true);
    }
    for r in (1..=4).rev() {
        cmp(&format!("dense_refine{r}"), true);
    }

    let fl = |k: &str| f32s(&z, &format!("{k}_bf16in"));
    report(
        "pose_enc",
        &o.pose,
        &f32s(&z, "pose_enc"),
        Some(&fl("pose_enc")),
        1e-2,
    );
    report(
        "extrinsics",
        &o.extr,
        &f32s(&z, "extrinsics"),
        Some(&fl("extrinsics")),
        1e-2,
    );
    report(
        "intrinsics",
        &o.intr,
        &f32s(&z, "intrinsics"),
        Some(&fl("intrinsics")),
        1e-2,
    );
    report(
        "depth",
        &o.depth,
        &f32s(&z, "depth"),
        Some(&fl("depth")),
        1e-2,
    );
    report(
        "depth_conf",
        &o.conf,
        &f32s(&z, "depth_conf"),
        Some(&fl("depth_conf")),
        1e-2,
    );
    report(
        "world_points",
        &o.world,
        &f32s(&z, "world_points"),
        Some(&fl("world_points")),
        1e-2,
    );
    gates(&z, &o, n);
    println!("worst intermediate rel_rms {worst:.3e}");
}

#[test]
#[ignore]
fn gpu_heads_chunked_n9() {
    let Some(s) = setup() else { return };
    let Some(z) = load_npz("heads_n9.npz") else {
        return;
    };
    let n = shape(&z, "L23_bf16")[0];
    let g = geometry(&z, n);
    let cache = caches(&s.nn, &z, true);
    let o = run_heads(&s, &cache, &g, 8);
    report("pose_enc", &o.pose, &f32s(&z, "pose_enc"), None, 1e-2);
    report("depth", &o.depth, &f32s(&z, "depth"), None, 1e-2);
    report("depth_conf", &o.conf, &f32s(&z, "depth_conf"), None, 1e-2);
    report(
        "world_points",
        &o.world,
        &f32s(&z, "world_points"),
        None,
        2e-2,
    );
    gates(&z, &o, n);
    // Chunking must not change results: frame 8 alone vs inside a chunk of 9.
    let o9 = run_heads(&s, &cache, &g, 16);
    let px = g.height * g.width;
    let e = rel(&o.depth[8 * px..], &o9.depth[8 * px..]);
    println!("chunk 8+1 vs 9: depth rel_rms {:.3e}", e.rel_rms);
    assert!(e.rel_rms < 1e-3);
}

fn time_ms(nn: &Nn, reps: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..3 {
        f();
    }
    nn.sync().unwrap();
    let ev = || {
        nn.ctx
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .unwrap()
    };
    let mut v = Vec::new();
    for _ in 0..reps {
        let (a, b) = (ev(), ev());
        a.record(&nn.stream).unwrap();
        f();
        b.record(&nn.stream).unwrap();
        v.push(a.elapsed_ms(&b).unwrap() as f64);
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn smi() -> String {
    std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu,memory.used,clocks.sm,clocks.mem,power.draw",
            "--format=csv,noheader",
        ])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "nvidia-smi unavailable".into())
}

/// Median of 10 CUDA-event timings per head at 512×384 (4:3) and 512×336
/// (3:2) for N = 8, 32 (dense head in chunks of 8, world points on).
#[test]
#[ignore]
fn bench_heads() {
    let Some(s) = setup() else { return };
    let nn = &s.nn;
    println!("nvidia-smi before: {}", smi());
    for (h, w) in [(384usize, 512usize), (336, 512)] {
        for n in [8usize, 32] {
            let g = Geometry::new(n, h, w);
            let rows = n * g.tokens() * DIM;
            // Unit-variance random caches (timing does not depend on values).
            let mut seed = 1u64;
            let vals: Vec<u16> = (0..rows)
                .map(|_| {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    host::bf16(((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 3.4)
                })
                .collect();
            let cache: [CudaSlice<u16>; 4] =
                std::array::from_fn(|_| nn.stream.clone_htod(&vals).unwrap());
            let mut cams = DeviceCams::new(nn, n).unwrap();
            let cam_ms = time_ms(nn, 10, || {
                s.camera
                    .enqueue(nn, &cache[3].slice(..), &g, &mut cams)
                    .unwrap()
            });
            let px = h * w;
            let c = 8.min(n);
            let gc = g.with_frames(c);
            let mut depth = nn.stream.alloc_zeros::<u16>(c * px).unwrap();
            let mut conf = nn.stream.alloc_zeros::<u16>(c * px).unwrap();
            let mut world = nn.stream.alloc_zeros::<f32>(c * px * 3).unwrap();
            let r = g.tokens() * DIM;
            let chunk_ms = time_ms(nn, 10, || {
                let layers = [0, 1, 2, 3].map(|i| cache[i].slice(..c * r));
                let e = cams.extrinsics.slice(..c * 12);
                let k = cams.intrinsics.slice(..c * 9);
                s.dense
                    .run_chunk(
                        nn,
                        layers,
                        &gc,
                        &mut depth.slice_mut(..),
                        &mut conf.slice_mut(..),
                        Some((&e, &k)),
                        Some(&mut world.slice_mut(..)),
                    )
                    .unwrap()
            });
            let chunks = n.div_ceil(8);
            println!(
                "bench_heads {h}x{w} N={n}: camera {cam_ms:.3} ms, dense {chunk_ms:.3} ms/chunk of {c} → {:.3} ms, total {:.3} ms ({:.3} ms/frame)",
                chunk_ms * chunks as f64,
                cam_ms + chunk_ms * chunks as f64,
                (cam_ms + chunk_ms * chunks as f64) / n as f64
            );
            let _ = SPECIAL;
        }
    }
    println!("nvidia-smi after: {}", smi());
}
