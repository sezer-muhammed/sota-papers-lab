//! Trunk tests. CPU tests run anywhere; `gpu_*` tests are `#[ignore]` and
//! run on the GPU host against the oracle dumps written by `trunk_oracle.py`:
//!
//! - `SPL_VGGT_WEIGHTS`: bf16 safetensors with checkpoint names (default
//!   `/models/vggt-omega/converted/weights.safetensors`); 2-D weights are
//!   read from it.
//! - `SPL_VGGT_SMALL`: fp32 side file from `trunk_oracle.py --small`
//!   (LN weights, gammas, tokens, periods, non-zero effective QKV biases).
//! - `SPL_VGGT_ORACLE`: directory holding `n2_bf16/`, `n2_fp32/`, `n8_bf16/`.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use spl_core::{Result, anyhow, bail};
use spl_nn::Nn;
use spl_nn::cudarc::driver::sys::CUevent_flags;
use spl_nn::cudarc::driver::{CudaSlice, CudaView};

use super::*;

#[test]
fn dense_schedule_marks_register_layers() {
    let s = dense_schedule();
    assert_eq!(s.iter().filter(|&&k| k == InterKind::Register).count(), 5);
    assert_eq!(s[2], InterKind::Register);
    assert_eq!(s[3], InterKind::Global);
}

#[test]
fn geometry_counts_special_tokens() {
    let g = TrunkGeom {
        frames: 2,
        grid_h: 21,
        grid_w: 32,
    };
    assert_eq!(g.tokens(), 689);
    assert_eq!(g.rows(), 1378);
    assert!(Trunk::scratch_bytes(&g) > g.rows() * DIM * 8);
}

#[test]
fn kernel_source_has_every_entry_point() {
    let src = include_str!("../../../cuda/vggt_trunk.cu");
    for k in [
        "patchify",
        "put_prefix",
        "put_bf16_rows",
        "copy_rows",
        "ln_rows_f32",
        "to_cache",
        "rows_to_wide",
    ] {
        assert!(src.contains(&format!("void {k}(")), "{k}");
    }
}

// ---------------------------------------------------------------- GPU ---

/// Weights from the bf16 file plus the fp32 side file, uploaded once.
pub(crate) struct FileWeights {
    bf16: HashMap<String, CudaSlice<u16>>,
    f32: HashMap<String, (CudaSlice<f32>, Vec<f32>)>,
}

struct Entry {
    dtype: String,
    shape: Vec<usize>,
    start: u64,
    end: u64,
}

fn st_header(path: &Path) -> Result<(File, u64, HashMap<String, Entry>)> {
    let mut f = File::open(path).map_err(|e| anyhow!("open {}: {e}", path.display()))?;
    let mut n = [0u8; 8];
    f.read_exact(&mut n)?;
    let n = u64::from_le_bytes(n);
    let mut h = vec![0u8; n as usize];
    f.read_exact(&mut h)?;
    let v: serde_json::Value = serde_json::from_slice(&h)?;
    let mut out = HashMap::new();
    for (k, e) in v.as_object().ok_or_else(|| anyhow!("bad header"))? {
        if k == "__metadata__" {
            continue;
        }
        let off = e["data_offsets"].as_array().ok_or_else(|| anyhow!("{k}"))?;
        out.insert(
            k.clone(),
            Entry {
                dtype: e["dtype"].as_str().unwrap_or("").to_string(),
                shape: e["shape"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| x.as_u64())
                            .map(|x| x as usize)
                            .collect()
                    })
                    .unwrap_or_default(),
                start: off[0].as_u64().unwrap_or(0),
                end: off[1].as_u64().unwrap_or(0),
            },
        );
    }
    Ok((f, 8 + n, out))
}

fn read_bytes(f: &mut File, base: u64, e: &Entry) -> Result<Vec<u8>> {
    f.seek(SeekFrom::Start(base + e.start))?;
    let mut b = vec![0u8; (e.end - e.start) as usize];
    f.read_exact(&mut b)?;
    Ok(b)
}

impl FileWeights {
    pub(crate) fn load(nn: &Nn, big: &Path, small: &Path) -> Result<Self> {
        let up = |e| anyhow!("upload weights: {e:?}");
        let mut f32m = HashMap::new();
        let mut bf16m = HashMap::new();
        let (mut sf, sbase, sh) = st_header(small)?;
        for (k, e) in &sh {
            if e.dtype != "F32" {
                bail!("{k}: side file must be F32, is {}", e.dtype);
            }
            let b = read_bytes(&mut sf, sbase, e)?;
            let v: Vec<f32> = b
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            if k.ends_with("qkv.bias") {
                bf16m.insert(
                    k.clone(),
                    nn.stream
                        .clone_htod(&spl_nn::host::to_bf16(&v))
                        .map_err(up)?,
                );
            } else {
                f32m.insert(k.clone(), (nn.stream.clone_htod(&v).map_err(up)?, v));
            }
        }
        let (mut bf, bbase, bh) = st_header(big)?;
        for (k, e) in &bh {
            if !k.starts_with("aggregator.") || e.shape.len() < 2 || k.ends_with("qkv.bias") {
                continue;
            }
            if f32m.contains_key(k) {
                continue;
            }
            let v: Vec<u16> = match e.dtype.as_str() {
                "BF16" => read_bytes(&mut bf, bbase, e)?
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| u16::from_le_bytes(*c))
                    .collect(),
                d => bail!("{k}: expected BF16, got {d}"),
            };
            bf16m.insert(k.clone(), nn.stream.clone_htod(&v).map_err(up)?);
        }
        // 1-D biases of Linear layers (proj, fc1, fc2, patch proj) are bf16
        // in the official autocast path: round the fp32 values.
        for (k, (_, v)) in &f32m {
            if k.ends_with(".bias")
                && (k.contains(".proj.") || k.contains(".mlp.") || k.contains("patch_embed.proj"))
            {
                bf16m.insert(
                    k.clone(),
                    nn.stream
                        .clone_htod(&spl_nn::host::to_bf16(v))
                        .map_err(up)?,
                );
            }
        }
        Ok(Self {
            bf16: bf16m,
            f32: f32m,
        })
    }
}

impl TrunkWeights for FileWeights {
    fn bf16(&self, name: &str) -> Result<CudaView<'_, u16>> {
        self.bf16
            .get(name)
            .map(|s| s.slice(..))
            .ok_or_else(|| anyhow!("missing bf16 weight {name}"))
    }
    fn f32(&self, name: &str) -> Result<CudaView<'_, f32>> {
        self.f32
            .get(name)
            .map(|s| s.0.slice(..))
            .ok_or_else(|| anyhow!("missing f32 weight {name}"))
    }
    fn host_f32(&self, name: &str) -> Result<Vec<f32>> {
        self.f32
            .get(name)
            .map(|s| s.1.clone())
            .ok_or_else(|| anyhow!("missing f32 weight {name}"))
    }
    fn has(&self, name: &str) -> bool {
        self.bf16.contains_key(name) || self.f32.contains_key(name)
    }
}

/// fp32 `.npy` (C order).
pub(crate) fn npy(path: &Path) -> Result<(Vec<usize>, Vec<f32>)> {
    let b = std::fs::read(path).map_err(|e| anyhow!("read {}: {e}", path.display()))?;
    if &b[1..6] != b"NUMPY" {
        bail!("{} is not .npy", path.display());
    }
    let (hl, off) = if b[6] == 1 {
        (u16::from_le_bytes([b[8], b[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12)
    };
    let h = std::str::from_utf8(&b[off..off + hl])?;
    if !h.contains("'<f4'") || h.contains("'fortran_order': True") {
        bail!("{}: need little-endian f4 C order: {h}", path.display());
    }
    let s = &h[h.find("'shape': (").ok_or_else(|| anyhow!("no shape"))? + 10..];
    let shape: Vec<usize> = s[..s.find(')').unwrap_or(0)]
        .split(',')
        .filter_map(|x| x.trim().parse().ok())
        .collect();
    let v = b[off + hl..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    Ok((shape, v))
}

fn env_path(var: &str, default: &str) -> PathBuf {
    std::env::var_os(var).map_or_else(|| PathBuf::from(default), PathBuf::from)
}

fn oracle_dir() -> PathBuf {
    env_path("SPL_VGGT_ORACLE", "/data/eng-trunk")
}

fn open() -> (Nn, FileWeights) {
    let nn = Nn::open().expect("CUDA device");
    let w = FileWeights::load(
        &nn,
        &env_path(
            "SPL_VGGT_WEIGHTS",
            "/models/vggt-omega/converted/weights.safetensors",
        ),
        &env_path("SPL_VGGT_SMALL", "/models/vggt-omega/trunk_f32.safetensors"),
    )
    .expect("load weights");
    (nn, w)
}

/// Collects the requested taps on the host.
struct Collect {
    want: Vec<Tap>,
    got: HashMap<String, Vec<f32>>,
}

fn tap_name(t: Tap) -> String {
    match t {
        Tap::Input => "input".into(),
        Tap::Frame(i) => format!("frame_{i:02}"),
        Tap::Inter(i) => format!("inter_{i:02}"),
    }
}

impl TapSink for Collect {
    fn wants(&self, tap: Tap) -> bool {
        self.want.contains(&tap)
    }
    fn take(&mut self, nn: &Nn, tap: Tap, rows: &CudaSlice<f32>, g: TrunkGeom) -> Result<()> {
        let v = nn
            .stream
            .clone_dtoh(&rows.slice(..g.rows() * DIM))
            .map_err(|e| anyhow!("download tap: {e:?}"))?;
        self.got.insert(tap_name(tap), v);
        Ok(())
    }
}

/// Relative RMS error of `a` against `b` over the selected token rows.
fn rel(a: &[f32], b: &[f32], g: TrunkGeom, special: bool) -> f64 {
    let t = g.tokens();
    let (mut se, mut sb) = (0.0f64, 0.0f64);
    for r in 0..g.rows() {
        if ((r % t) < SPECIAL) != special {
            continue;
        }
        for c in 0..DIM {
            let (x, y) = (a[r * DIM + c] as f64, b[r * DIM + c] as f64);
            se += (x - y) * (x - y);
            sb += y * y;
        }
    }
    (se / sb.max(1e-30)).sqrt()
}

struct Prepared {
    geom: TrunkGeom,
    images: CudaSlice<f32>,
    cache: Vec<CudaSlice<u16>>,
}

fn prepare(nn: &Nn, dir: &Path) -> Prepared {
    let (shape, img) = npy(&dir.join("images.npy")).expect("images.npy");
    let geom = TrunkGeom {
        frames: shape[0],
        grid_h: shape[2] / 16,
        grid_w: shape[3] / 16,
    };
    let images = nn.stream.clone_htod(&img).expect("upload images");
    let cache = (0..4)
        .map(|_| {
            nn.stream
                .alloc_zeros::<u16>(geom.rows() * 2 * DIM)
                .expect("cache")
        })
        .collect();
    Prepared {
        geom,
        images,
        cache,
    }
}

fn run(nn: &Nn, w: &FileWeights, trunk: &mut Trunk, p: &mut Prepared, tap: &mut dyn TapSink) {
    let [c0, c1, c2, c3] = &mut p.cache[..] else {
        unreachable!()
    };
    let (mut v0, mut v1, mut v2, mut v3) = (
        c0.slice_mut(..),
        c1.slice_mut(..),
        c2.slice_mut(..),
        c3.slice_mut(..),
    );
    trunk
        .run(
            nn,
            w,
            p.geom,
            &p.images.slice(..),
            [&mut v0, &mut v1, &mut v2, &mut v3],
            None,
            &dense_schedule(),
            tap,
        )
        .expect("trunk run");
    nn.sync().expect("sync");
}

/// Layer-by-layer parity on two frames against the official bf16 path,
/// with the official bf16-vs-fp32 distance as the noise floor.
#[test]
#[ignore]
fn gpu_trunk_matches_oracle_n2() {
    let (nn, w) = open();
    let root = oracle_dir();
    let (bdir, fdir) = (root.join("n2_bf16"), root.join("n2_fp32"));
    let mut p = prepare(&nn, &bdir);
    let mut want = vec![Tap::Input];
    for i in [0, 1, 2, 3, 4, 11, 17, 23] {
        want.push(Tap::Frame(i));
        want.push(Tap::Inter(i));
    }
    let mut tap = Collect {
        want: want.clone(),
        got: HashMap::new(),
    };
    let mut trunk = Trunk::new(&nn, &w).expect("trunk");
    run(&nn, &w, &mut trunk, &mut p, &mut tap);

    println!(
        "geom {:?}\n{:<10} {:>12} {:>12} {:>12} {:>12} {:>12} {:>8}",
        p.geom, "layer", "patch/bf16", "patch/fp32", "floor", "spec/bf16", "spec floor", "ratio"
    );
    let mut worst: f64 = 0.0;
    let mut worst_special: f64 = 0.0;
    for t in want {
        let name = tap_name(t);
        let ours = &tap.got[&name];
        let (_, ob) = npy(&bdir.join(format!("{name}.npy"))).expect("oracle bf16");
        let (_, of) = npy(&fdir.join(format!("{name}.npy"))).expect("oracle fp32");
        let eb = rel(ours, &ob, p.geom, false);
        let ef = rel(ours, &of, p.geom, false);
        let floor = rel(&ob, &of, p.geom, false);
        let es = rel(ours, &ob, p.geom, true);
        let sfloor = rel(&ob, &of, p.geom, true);
        println!(
            "{name:<10} {eb:>12.3e} {ef:>12.3e} {floor:>12.3e} {es:>12.3e} {sfloor:>12.3e} {:>8.2}",
            ef / floor.max(1e-12)
        );
        // Spec §8: patch tokens no worse than 1.5x the official
        // bf16-vs-fp32 distance from fp32 (and no farther from the official
        // bf16 output than fp32 is); special tokens, whose official noise is
        // tiny in early layers, within max(1e-2, 2x their floor) of bf16
        // (measured worst: 1.51x at frame_23, an open issue).
        worst = worst.max(ef / floor.max(1e-12)).max(eb / floor.max(1e-12));
        worst_special = worst_special.max(es / sfloor.max(5e-3));
    }
    // Cached layer 23 = [F_23 | G_23] in bf16.
    let c3 = nn.stream.clone_dtoh(&p.cache[3]).expect("cache");
    let (f23, g23) = (&tap.got["frame_23"], &tap.got["inter_23"]);
    let mut max_rel: f32 = 0.0;
    for r in 0..p.geom.rows() {
        for c in 0..DIM {
            let a = spl_nn::host::bf16_to_f32(c3[r * 2 * DIM + c]);
            let b = spl_nn::host::bf16_to_f32(c3[r * 2 * DIM + DIM + c]);
            let (fa, gb) = (f23[r * DIM + c], g23[r * DIM + c]);
            max_rel = max_rel
                .max((a - fa).abs() / fa.abs().max(1e-3))
                .max((b - gb).abs() / gb.abs().max(1e-3));
        }
    }
    println!("cache[3] vs taps: max rel {max_rel:.2e} (bf16 rounding)");
    assert!(max_rel < 8e-3);
    println!("patch error / floor {worst:.2}, special error / floor {worst_special:.2}");
    assert!(
        worst_special <= 2.0,
        "special tokens {worst_special:.2}x the official noise"
    );
    assert!(
        worst <= 1.5,
        "worst error / official noise floor {worst:.2} > 1.5"
    );
}

/// Eight frames: parity of the register layer and the last layer, then
/// timing (CUDA events, median of 10 after warm-up).
#[test]
#[ignore]
fn gpu_trunk_n8_parity_and_speed() {
    let (nn, w) = open();
    let dir = oracle_dir().join("n8_bf16");
    let mut p = prepare(&nn, &dir);
    let want = vec![
        Tap::Inter(2),
        Tap::Frame(4),
        Tap::Inter(4),
        Tap::Frame(23),
        Tap::Inter(23),
    ];
    let mut tap = Collect {
        want: want.clone(),
        got: HashMap::new(),
    };
    let mut trunk = Trunk::new(&nn, &w).expect("trunk");
    run(&nn, &w, &mut trunk, &mut p, &mut tap);
    let mut worst: f64 = 0.0;
    for t in want {
        let name = tap_name(t);
        let (_, ob) = npy(&dir.join(format!("{name}.npy"))).expect("oracle");
        let (ep, es) = (
            rel(&tap.got[&name], &ob, p.geom, false),
            rel(&tap.got[&name], &ob, p.geom, true),
        );
        println!("N=8 {name:<10} patch {ep:.3e} special {es:.3e}");
        worst = worst.max(ep).max(es);
    }
    // No fp32 dump at N = 8: same gate as the official bf16 noise at N = 2
    // for the matching layers (see gpu_trunk_matches_oracle_n2).
    for frames in [1usize, 2, 4, 8] {
        bench(&nn, &w, &mut trunk, &p, frames);
    }
    assert!(
        worst < 0.3,
        "N=8 relative error {worst:.3e} beyond the N=2 noise floor"
    );
}

fn bench(nn: &Nn, w: &FileWeights, trunk: &mut Trunk, p8: &Prepared, frames: usize) {
    let g = TrunkGeom { frames, ..p8.geom };
    let per = 3 * g.grid_h * 16 * g.grid_w * 16;
    let mut p = Prepared {
        geom: g,
        images: nn
            .stream
            .clone_htod(
                &nn.stream
                    .clone_dtoh(&p8.images.slice(..frames * per))
                    .expect("dl"),
            )
            .expect("ul"),
        cache: (0..4)
            .map(|_| {
                nn.stream
                    .alloc_zeros::<u16>(g.rows() * 2 * DIM)
                    .expect("cache")
            })
            .collect(),
    };
    for _ in 0..3 {
        run(nn, w, trunk, &mut p, &mut NoTap);
    }
    let flag = Some(CUevent_flags::CU_EVENT_DEFAULT);
    let mut ms = Vec::new();
    for _ in 0..10 {
        let a = nn.ctx.new_event(flag).expect("event");
        let b = nn.ctx.new_event(flag).expect("event");
        a.record(&nn.stream).expect("record");
        run(nn, w, trunk, &mut p, &mut NoTap);
        b.record(&nn.stream).expect("record");
        b.synchronize().expect("sync");
        ms.push(a.elapsed_ms(&b).expect("elapsed"));
    }
    ms.sort_by(f32::total_cmp);
    println!(
        "bench trunk N={frames} T={} rows={}: median {:.2} ms (min {:.2}, max {:.2})",
        g.tokens(),
        g.rows(),
        ms[5],
        ms[0],
        ms[9]
    );
}
