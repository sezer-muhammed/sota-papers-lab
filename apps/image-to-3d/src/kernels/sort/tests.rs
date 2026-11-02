//! Tests of the sort and binning kernels. Owner: sort (E2).
//!
//! - CPU: the launch constants agree with the `.cu` files, tile key widths, a
//!   bit-exact CPU mirror of the span rule (`span`) against brute-force pixel
//!   coverage.
//! - GPU (`#[ignore]`; run on the host with
//!   `apps/image-to-3d/scripts/gpu-test.sh <checkout> sort::`): the radix
//!   sorter against a stable CPU sort (spec-raster 3.2's cases), CUDA-graph
//!   replay with changing device counts, `count_tiles` + `sort_depth` +
//!   `bin_tiles` against the CPU mirror on mixed scenes (both tile heights,
//!   odd sizes, 1×1, empty, a tile with > 65,535 Gaussians), instance overflow,
//!   and the gsplat oracle (`isect`, staged).

use std::collections::BTreeSet;
use std::f32::consts::LN_2;

use spl_gpu::cudarc::driver::CudaSlice;
use spl_gpu::cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};

use super::*;
use crate::frame::{FrameCam, IterParams};
use crate::kernels::layout::{
    BUCKET, KBIN_ABS, KBIN_REL, LOG2E, MAX_TILES, OVF_INSTANCES, ST_N_CHAINS, ST_N_TILES_HIT,
    ST_OVERFLOW, STATUS_LEN, TILE_W,
};
use crate::kernels::{Capacity, RasterVariant, SplatRec};
use crate::testkit::npz::{NpyData, Npz};
use crate::testkit::oracle;

// ------------------------------------------------------------------ helpers

/// SplitMix64: deterministic test data.
pub(super) struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Uniform in the open interval (0, 1).
    pub fn f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 + 0.5) / (1u64 << 24) as f32
    }

    pub fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.f32()
    }

    pub fn below(&mut self, n: u32) -> u32 {
        ((self.u32() as u64 * n as u64) >> 32) as u32
    }
}

pub(super) fn gpu() -> Gpu {
    Gpu::open().expect("open GPU")
}

fn upload_u32(gpu: &Gpu, host: &[u32], dev: &mut CudaSlice<u32>) {
    if !host.is_empty() {
        gpu.stream
            .memcpy_htod(host, &mut dev.slice_mut(0..host.len()))
            .unwrap();
    }
}

fn download_u32(gpu: &Gpu, dev: &CudaSlice<u32>, n: usize) -> Vec<u32> {
    if n == 0 {
        return Vec::new();
    }
    gpu.stream.clone_dtoh(&dev.slice(0..n)).unwrap()
}

/// Stable CPU sort of (key, value) by key.
fn cpu_sort<K: Copy + Ord>(keys: &[K], vals: &[u32]) -> (Vec<K>, Vec<u32>) {
    let mut idx: Vec<usize> = (0..keys.len()).collect();
    idx.sort_by_key(|&i| keys[i]);
    (
        idx.iter().map(|&i| keys[i]).collect(),
        idx.iter().map(|&i| vals[i]).collect(),
    )
}

// ------------------------------------------------------------------ CPU mirror of span (binning.cu)

/// binning.cu's `span::Span`, computed with the same IEEE operations (the CUDA
/// side uses explicit `__f*_rn` intrinsics, so the results agree bit for bit).
#[derive(Clone, Copy, Debug)]
pub(super) struct CpuSpan {
    mx: f32,
    my: f32,
    a: f32,
    b: f32,
    inv_a: f32,
    kb: f32,
    det: f32,
    ratio: f32,
    px0: i32,
    px1: i32,
    py0: i32,
    py1: i32,
}

#[allow(clippy::neg_cmp_op_on_partial_ord)]
pub(super) fn span_setup(r: &SplatRec, w: u32, h: u32) -> Option<CpuSpan> {
    let a = r.ca * (-2.0 * LN_2);
    let b = r.cb * -LN_2;
    let c = r.cc * (-2.0 * LN_2);
    let k = r.pthr * (2.0 * LN_2);
    let kb = k.mul_add(1.0 + KBIN_REL, KBIN_ABS);
    let det = a.mul_add(c, -(b * b));
    if !(det > 0.0 && kb > 0.0 && a > 0.0 && c > 0.0) {
        return None;
    }
    let inv_det = 1.0 / det;
    let x_ext = (kb * (c * inv_det)).sqrt();
    let y_ext = (kb * (a * inv_det)).sqrt();
    let ratio = (b * x_ext) * (1.0 / c);
    let fx0 = ((r.mx - x_ext) - 0.5).ceil().max(0.0);
    let fx1 = ((r.mx + x_ext) - 0.5).floor().min(w as f32 - 1.0);
    let fy0 = ((r.my - y_ext) - 0.5).ceil().max(0.0);
    let fy1 = ((r.my + y_ext) - 0.5).floor().min(h as f32 - 1.0);
    if !(fx0 <= fx1 && fy0 <= fy1) {
        return None;
    }
    Some(CpuSpan {
        mx: r.mx,
        my: r.my,
        a,
        b,
        inv_a: 1.0 / a,
        kb,
        det,
        ratio,
        px0: fx0 as i32,
        px1: fx1 as i32,
        py0: fy0 as i32,
        py1: fy1 as i32,
    })
}

fn band_x_interval(s: &CpuSpan, y0: f32, y1: f32) -> (f32, f32) {
    let ak = s.a * s.kb;
    let yl = (s.my + s.ratio).max(y0).min(y1);
    let dl = yl - s.my;
    let ql = (-s.det).mul_add(dl * dl, ak).max(0.0).sqrt();
    let xl = s.mx + ((-(s.b * dl)) - ql) * s.inv_a;
    let yr = (s.my - s.ratio).max(y0).min(y1);
    let dr = yr - s.my;
    let qr = (-s.det).mul_add(dr * dr, ak).max(0.0).sqrt();
    let xr = s.mx + ((-(s.b * dr)) + qr) * s.inv_a;
    (xl, xr)
}

/// Tiles of one record in emit order (rows top to bottom, columns left to right).
// `!(a <= b)` on purpose: NaN behaves exactly as in the CUDA source.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
pub(super) fn cpu_tiles(r: &SplatRec, w: u32, h: u32, tile_h: u32) -> Vec<u32> {
    let Some(s) = span_setup(r, w, h) else {
        return Vec::new();
    };
    let th = tile_h as i32;
    let grid_w = w.div_ceil(TILE_W);
    let mut out = Vec::new();
    for ty in s.py0 / th..=s.py1 / th {
        let yb0 = (ty * th).max(s.py0) as f32 + 0.5;
        let yb1 = (ty * th + th - 1).min(s.py1) as f32 + 0.5;
        let (xl, xr) = band_x_interval(&s, yb0, yb1);
        let c0 = (xl - 0.5).ceil().max(s.px0 as f32);
        let c1 = (xr - 0.5).floor().min(s.px1 as f32);
        if !(c0 <= c1) {
            continue;
        }
        let (tx0, tx1) = (c0 as u32 / TILE_W, c1 as u32 / TILE_W);
        out.extend((tx0..=tx1).map(|tx| ty as u32 * grid_w + tx));
    }
    out
}

/// Tiles holding a pixel centre with q(c − m) ≤ k (k = 2 ln(255 o) from the
/// stored pthr), by brute force in f64: what any correct binning must cover.
fn brute_force_tiles(r: &SplatRec, w: u32, h: u32, tile_h: u32) -> BTreeSet<u32> {
    let a = r.ca as f64 / (-0.5 * LOG2E as f64);
    let b = r.cb as f64 / -(LOG2E as f64);
    let c = r.cc as f64 / (-0.5 * LOG2E as f64);
    let k = 2.0 * std::f64::consts::LN_2 * r.pthr as f64;
    let det = a * c - b * b;
    let mut out = BTreeSet::new();
    if !(k > 0.0 && det > 0.0) {
        return out;
    }
    let (xe, ye) = ((k * c / det).sqrt() + 1.0, (k * a / det).sqrt() + 1.0);
    let clampi = |v: f64, hi: u32| v.clamp(0.0, hi as f64 - 1.0) as u32;
    let (mx, my) = (r.mx as f64, r.my as f64);
    if mx + xe < 0.0 || my + ye < 0.0 || mx - xe > w as f64 || my - ye > h as f64 {
        return out;
    }
    let grid_w = w.div_ceil(TILE_W);
    for py in clampi((my - ye).floor(), h)..=clampi((my + ye).ceil(), h) {
        for px in clampi((mx - xe).floor(), w)..=clampi((mx + xe).ceil(), w) {
            let (dx, dy) = (px as f64 + 0.5 - mx, py as f64 + 0.5 - my);
            if a * dx * dx + 2.0 * b * dx * dy + c * dy * dy <= k {
                out.insert((py / tile_h) * grid_w + px / TILE_W);
            }
        }
    }
    out
}

/// A record from a screen-space Gaussian: covariance (σ1, σ2, θ) + 0.3 px²
/// dilation, effective opacity o, depth z (spec-raster 1.4's encoding).
pub(super) fn make_rec(mx: f32, my: f32, s1: f32, s2: f32, theta: f32, o: f32, z: f32) -> SplatRec {
    let (sn, cs) = (theta as f64).sin_cos();
    let (v1, v2) = ((s1 as f64).powi(2), (s2 as f64).powi(2));
    let sxx = cs * cs * v1 + sn * sn * v2 + 0.3;
    let syy = sn * sn * v1 + cs * cs * v2 + 0.3;
    let sxy = cs * sn * (v1 - v2);
    rec_from_cov(mx, my, [sxx, sxy, syy], o, z)
}

/// A record from a (dilated) 2D covariance [Σxx, Σxy, Σyy].
pub(super) fn rec_from_cov(mx: f32, my: f32, cov: [f64; 3], o: f32, z: f32) -> SplatRec {
    let [sxx, sxy, syy] = cov;
    let det = sxx * syy - sxy * sxy;
    let (a, b, c) = (syy / det, -sxy / det, sxx / det);
    let l2e = LOG2E as f64;
    SplatRec {
        mx,
        my,
        ca: (-0.5 * l2e * a) as f32,
        cb: (-l2e * b) as f32,
        cc: (-0.5 * l2e * c) as f32,
        o,
        r: 0.5,
        g: 0.5,
        b: 0.5,
        pthr: (255.0 * o as f64).log2() as f32,
        z,
        flags: 0,
    }
}

/// A random record over a w × h image: log-uniform sizes from sub-pixel to
/// ~40 px, a few percent near the opacity threshold, some off-screen.
fn random_rec(rng: &mut Rng, w: u32, h: u32) -> SplatRec {
    let mx = rng.range(-60.0, w as f32 + 60.0);
    let my = rng.range(-60.0, h as f32 + 60.0);
    let s1 = (rng.range(0.3f32.ln(), 40f32.ln())).exp();
    let s2 = s1 * rng.range(0.05, 1.0);
    let theta = rng.range(0.0, std::f32::consts::PI);
    let o = if rng.below(10) < 3 {
        rng.range(1.0 / 255.0, 0.02)
    } else {
        rng.range(0.02, 1.0)
    };
    let z = rng.range(0.2, 60.0);
    make_rec(mx, my, s1, s2, theta, o, z)
}

// ------------------------------------------------------------------ CPU tests

/// Every `#define NAME VALUE` of a source (first definition wins).
fn cuda_define(text: &str, name: &str) -> u32 {
    text.lines()
        .filter_map(|l| l.trim_start().strip_prefix("#define "))
        .find_map(|rest| {
            let mut w = rest.split_whitespace();
            (w.next() == Some(name)).then(|| w.next().unwrap_or("").to_string())
        })
        .unwrap_or_else(|| panic!("#define {name} missing"))
        .trim_matches(|c| c == '(' || c == ')')
        .trim_end_matches('u')
        .parse()
        .unwrap_or_else(|_| panic!("#define {name} is not a number"))
}

#[test]
fn launch_constants_match_cuda() {
    let bin = include_str!("../../../cuda/binning.cu");
    let radix = include_str!("../../../cuda/radix.cu");
    assert_eq!(cuda_define(bin, "BIN_THREADS"), BIN_THREADS);
    assert_eq!(cuda_define(bin, "SCAN_THREADS"), SCAN_THREADS);
    assert_eq!(cuda_define(bin, "SCAN_IPT"), SCAN_IPT);
    assert_eq!(cuda_define(bin, "HIST_THREADS"), HIST_THREADS);
    assert_eq!(cuda_define(bin, "BIN_SCAN_THREADS"), BIN_SCAN_THREADS);
    assert_eq!(cuda_define(bin, "SC_DONE") as usize + 1, SC_STATE);
    assert_eq!(cuda_define(radix, "OS_HIST_DONE") as usize + 1, OS_STATE);
    assert_eq!(
        MAX_TILES % BIN_SCAN_THREADS,
        0,
        "bin_scan: whole tiles per thread"
    );
}

#[test]
fn tile_key_bits_cover_the_grid() {
    assert_eq!(tile_key_bits(1), 1);
    assert_eq!(tile_key_bits(2), 1);
    assert_eq!(tile_key_bits(256), 8);
    assert_eq!(tile_key_bits(257), 9);
    assert_eq!(tile_key_bits(82 * 53), 13);
    assert_eq!(tile_key_bits(MAX_TILES), 13);
    assert_eq!(u16_passes(13, 8), vec![(0, 8), (8, 5)]);
    assert_eq!(u16_passes(8, 8), vec![(0, 8)]);
    assert!(RadixConfig::default().validate().is_ok());
    assert!(
        RadixConfig {
            ipt32: 40,
            ..Default::default()
        }
        .validate()
        .is_err()
    );
}

/// The span rule is conservative: every pixel centre inside the opacity-aware
/// ellipse lies in a binned tile (spec-raster 2.8), on random records of all
/// sizes, both tile heights.
#[test]
fn span_covers_every_contributing_pixel() {
    let mut rng = Rng::new(7);
    let (w, h) = (97u32, 61u32);
    let mut extra = 0usize;
    let mut total = 0usize;
    for i in 0..4000 {
        let r = random_rec(&mut rng, w, h);
        let th = if i % 2 == 0 { 16 } else { 12 };
        let tiles: BTreeSet<u32> = cpu_tiles(&r, w, h, th).into_iter().collect();
        let need = brute_force_tiles(&r, w, h, th);
        assert!(
            need.is_subset(&tiles),
            "record {i} {r:?}: {need:?} ⊄ {tiles:?}"
        );
        assert_eq!(
            tiles.len(),
            cpu_tiles(&r, w, h, th).len(),
            "duplicate tiles"
        );
        extra += tiles.len() - need.len();
        total += tiles.len();
    }
    println!("span: {total} binned tiles, {extra} without a contributing pixel centre");
    assert!(
        extra * 20 < total,
        "the span rule is too loose: {extra} of {total}"
    );
}

// ------------------------------------------------------------------ GPU: radix sorter

/// The launch bounds declared ahead of the definitions reach the PTX
/// (`.maxntid`), so register allocation targets the real block sizes.
#[test]
#[ignore]
fn gpu_kernels_carry_their_launch_bounds() {
    let gpu = gpu();
    let cfg = RadixConfig::default();
    let units = [
        (&RADIX_SOURCE, cfg.defines()),
        (&BINNING_SOURCE, RasterVariant::default().defines()),
    ];
    let expect = [
        ("radix_hist32", cfg.hist_threads),
        ("onesweep_u32", cfg.threads),
        ("onesweep_u16", cfg.threads),
        ("count_tiles", BIN_THREADS),
        ("scan_tiles", SCAN_THREADS),
        ("emit", BIN_THREADS),
        ("tile_hist", HIST_THREADS),
        ("bin_scan", BIN_SCAN_THREADS),
    ];
    let ptx: String = units
        .iter()
        .map(|(src, d)| {
            crate::gpu::compile_ptx(src, d, gpu.arch())
                .unwrap()
                .to_src()
        })
        .collect();
    for (name, threads) in expect {
        let at = ptx
            .find(&format!(".entry {name}("))
            .unwrap_or_else(|| panic!("{name} not in the PTX"));
        let body = &ptx[at..at + ptx[at..].find('{').unwrap()];
        assert!(
            body.contains(&format!(".maxntid {threads}, 1, 1")),
            "{name}: no .maxntid {threads} in {body}"
        );
    }
    println!("{} kernels carry their launch bounds", expect.len());
}

struct SortBufs<K> {
    k: [CudaSlice<K>; 2],
    v: [CudaSlice<u32>; 2],
    n: CudaSlice<u32>,
}

fn sort_bufs<K: spl_gpu::cudarc::driver::DeviceRepr + spl_gpu::cudarc::driver::ValidAsZeroBits>(
    gpu: &Gpu,
    cap: usize,
) -> SortBufs<K> {
    let s = &gpu.stream;
    SortBufs {
        k: [s.alloc_zeros(cap).unwrap(), s.alloc_zeros(cap).unwrap()],
        v: [s.alloc_zeros(cap).unwrap(), s.alloc_zeros(cap).unwrap()],
        n: s.alloc_zeros(1).unwrap(),
    }
}

#[derive(Clone, Copy, Debug)]
enum Dist {
    Uniform,
    FewValues,
    AllEqual,
    MaxKey,
    Ascending,
    Descending,
    FloatDepth,
}

fn keys_for(rng: &mut Rng, n: usize, bits: u32, dist: Dist) -> Vec<u32> {
    let mask = if bits == 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    };
    let few: Vec<u32> = (0..5).map(|_| rng.u32() & mask).collect();
    let c = rng.u32() & mask;
    (0..n)
        .map(|i| match dist {
            Dist::Uniform => rng.u32() & mask,
            Dist::FewValues => few[rng.below(5) as usize],
            Dist::AllEqual => c,
            Dist::MaxKey => {
                if rng.below(2) == 0 {
                    mask
                } else {
                    rng.u32() & mask
                }
            }
            Dist::Ascending => ((i as u64 * mask as u64) / n.max(1) as u64) as u32,
            Dist::Descending => mask - ((i as u64 * mask as u64) / n.max(1) as u64) as u32,
            Dist::FloatDepth => rng.range(0.2, 60.0).to_bits() & mask,
        })
        .collect()
}

fn check_u32_case(
    gpu: &Gpu,
    sorter: &mut RadixSorter,
    b: &mut SortBufs<u32>,
    keys: &[u32],
    bits: u32,
    what: &str,
) {
    let n = keys.len();
    let vals: Vec<u32> = (0..n as u32).collect();
    upload_u32(gpu, keys, &mut b.k[0]);
    upload_u32(gpu, &vals, &mut b.v[0]);
    upload_u32(gpu, &[n as u32], &mut b.n);
    let nv = b.n.slice(..);
    let [k0, k1] = &mut b.k;
    let [v0, v1] = &mut b.v;
    let r = sorter
        .sort_u32(&gpu.stream, [k0, k1], [v0, v1], &nv, bits, None)
        .unwrap();
    assert_eq!(r, bits.div_ceil(8) as usize % 2);
    let (ek, ev) = cpu_sort(keys, &vals);
    assert_eq!(download_u32(gpu, &b.k[r], n), ek, "{what}: keys");
    assert_eq!(
        download_u32(gpu, &b.v[r], n),
        ev,
        "{what}: values (stability)"
    );
}

/// spec-raster 3.2's cases: key widths 1-32, n around partition boundaries up
/// to 3M, uniform, few-valued, all-equal, max-key, sorted and float-depth
/// keys; one sorter for all (its self-resetting state survives every size).
#[test]
#[ignore]
fn gpu_sort_u32_matches_stable_cpu_sort() {
    let gpu = gpu();
    let cap = 3_000_000usize;
    let mut sorter = RadixSorter::new(&gpu, cap as u32).unwrap();
    let mut b = sort_bufs::<u32>(&gpu, cap);
    let mut rng = Rng::new(1);
    let part = sorter.config.part32() as usize;
    let small = [
        0,
        1,
        2,
        31,
        33,
        part - 1,
        part,
        part + 1,
        2 * part + 7,
        100_000,
    ];
    let dists = [
        Dist::Uniform,
        Dist::FewValues,
        Dist::AllEqual,
        Dist::MaxKey,
        Dist::Ascending,
        Dist::Descending,
    ];
    let mut cases = 0;
    for bits in [1, 7, 8, 13, 16, 24, 32] {
        for &n in &small {
            for d in dists {
                let keys = keys_for(&mut rng, n, bits, d);
                check_u32_case(
                    &gpu,
                    &mut sorter,
                    &mut b,
                    &keys,
                    bits,
                    &format!("{bits} bits n {n} {d:?}"),
                );
                cases += 1;
            }
        }
        for n in [1_000_000, cap] {
            for d in [Dist::Uniform, Dist::FewValues] {
                let keys = keys_for(&mut rng, n, bits, d);
                check_u32_case(
                    &gpu,
                    &mut sorter,
                    &mut b,
                    &keys,
                    bits,
                    &format!("{bits} bits n {n} {d:?}"),
                );
                cases += 1;
            }
        }
    }
    let keys = keys_for(&mut rng, cap, 32, Dist::FloatDepth);
    check_u32_case(&gpu, &mut sorter, &mut b, &keys, 32, "float depths");
    println!("{} u32 sorts agree with the stable CPU sort", cases + 1);
}

/// The full spec size: 24M items through one sorter (u32 keys) and the 13-bit
/// tile sort, n read from the device.
#[test]
#[ignore]
fn gpu_sort_24m_items() {
    let gpu = gpu();
    let n = 24_000_000usize;
    let mut sorter = RadixSorter::new(&gpu, n as u32).unwrap();
    let mut rng = Rng::new(24);
    {
        let mut b = sort_bufs::<u32>(&gpu, n);
        let keys = keys_for(&mut rng, n, 32, Dist::FloatDepth);
        check_u32_case(&gpu, &mut sorter, &mut b, &keys, 32, "24M float depths");
    }
    let mut b = sort_bufs::<u16>(&gpu, n);
    let keys: Vec<u16> = (0..n).map(|_| rng.below(4346) as u16).collect();
    check_u16_case(&gpu, &mut sorter, &mut b, &keys, 13, 8, "24M tile keys");
    println!("24M items sorted (u32 and u16)");
}

/// CPU digit offsets of a two-pass u16 sort, the layout bin_scan writes.
fn cpu_u16_offsets(keys: &[u16], lo_bits: u32) -> Vec<u32> {
    let mut out = vec![0u32; 2 * RADIX_BINS as usize];
    for &k in keys {
        out[(k as u32 & ((1 << lo_bits) - 1)) as usize] += 1;
        out[RADIX_BINS as usize + (k as u32 >> lo_bits) as usize] += 1;
    }
    for p in 0..2 {
        let mut run = 0;
        for d in 0..RADIX_BINS as usize {
            let c = out[p * RADIX_BINS as usize + d];
            out[p * RADIX_BINS as usize + d] = run;
            run += c;
        }
    }
    out
}

fn check_u16_case(
    gpu: &Gpu,
    sorter: &mut RadixSorter,
    b: &mut SortBufs<u16>,
    keys: &[u16],
    bits: u32,
    lo_bits: u32,
    what: &str,
) {
    let s = &gpu.stream;
    let n = keys.len();
    let vals: Vec<u32> = (0..n as u32).collect();
    if n > 0 {
        s.memcpy_htod(keys, &mut b.k[0].slice_mut(0..n)).unwrap();
    }
    upload_u32(gpu, &vals, &mut b.v[0]);
    upload_u32(gpu, &[n as u32], &mut b.n);
    let offsets = s.clone_htod(&cpu_u16_offsets(keys, lo_bits)).unwrap();
    let nv = b.n.slice(..);
    let ov = offsets.slice(..);
    let [k0, k1] = &mut b.k;
    let [v0, v1] = &mut b.v;
    let r = sorter
        .sort_u16_with_hist(s, [k0, k1], [v0, v1], &nv, bits, lo_bits, &ov, None)
        .unwrap();
    let (ek, ev) = cpu_sort(keys, &vals);
    if n > 0 {
        let gk = s.clone_dtoh(&b.k[r].slice(0..n)).unwrap();
        assert_eq!(gk, ek, "{what}: keys");
    }
    assert_eq!(
        download_u32(gpu, &b.v[r], n),
        ev,
        "{what}: values (stability)"
    );
}

/// u16 keys with offsets from a histogram (the tile sort): 1-16 bits, first
/// digit 6-8 bits, sizes around partition boundaries.
#[test]
#[ignore]
fn gpu_sort_u16_with_hist_matches_stable_cpu_sort() {
    let gpu = gpu();
    let cap = 2_000_000usize;
    let mut sorter = RadixSorter::new(&gpu, cap as u32).unwrap();
    let mut b = sort_bufs::<u16>(&gpu, cap);
    let mut rng = Rng::new(2);
    let part = sorter.config.part16() as usize;
    let mut cases = 0;
    for bits in [1u32, 5, 8, 9, 13, 16] {
        for lo in [8u32, 7, 6] {
            if bits > lo + 8 {
                continue;
            }
            for n in [
                0,
                1,
                33,
                part - 1,
                part,
                part + 1,
                3 * part + 5,
                250_000,
                cap,
            ] {
                for d in [Dist::Uniform, Dist::FewValues, Dist::AllEqual, Dist::MaxKey] {
                    let keys: Vec<u16> = keys_for(&mut rng, n, bits, d)
                        .into_iter()
                        .map(|k| k as u16)
                        .collect();
                    check_u16_case(
                        &gpu,
                        &mut sorter,
                        &mut b,
                        &keys,
                        bits,
                        lo,
                        &format!("{bits}/{lo} bits n {n} {d:?}"),
                    );
                    cases += 1;
                }
            }
        }
    }
    println!("{cases} u16 sorts agree with the stable CPU sort");
}

/// Both sorts captured once in a CUDA graph and replayed with different
/// device counts and data: no memset or host value is baked into the graph.
#[test]
#[ignore]
fn gpu_sort_replays_in_a_cuda_graph() {
    let gpu = gpu();
    // Captured slices must not record cudarc's per-use events.
    unsafe { gpu.ctx.disable_event_tracking() };
    let cap = 400_000usize;
    let mut sorter = RadixSorter::new(&gpu, cap as u32).unwrap();
    let mut b32 = sort_bufs::<u32>(&gpu, cap);
    let mut b16 = sort_bufs::<u16>(&gpu, cap);
    let mut offsets = gpu
        .stream
        .alloc_zeros::<u32>(2 * RADIX_BINS as usize)
        .unwrap();
    gpu.sync().unwrap();
    let s = gpu.stream.clone();
    s.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        .unwrap();
    let (r32, r16) = {
        let n32 = b32.n.slice(..);
        let [k0, k1] = &mut b32.k;
        let [v0, v1] = &mut b32.v;
        let r32 = sorter
            .sort_u32(&s, [k0, k1], [v0, v1], &n32, 32, None)
            .unwrap();
        let n16 = b16.n.slice(..);
        let ov = offsets.slice(..);
        let [k0, k1] = &mut b16.k;
        let [v0, v1] = &mut b16.v;
        let r16 = sorter
            .sort_u16_with_hist(&s, [k0, k1], [v0, v1], &n16, 13, 8, &ov, None)
            .unwrap();
        (r32, r16)
    };
    let graph = s
        .end_capture(CUgraphInstantiate_flags(0))
        .unwrap()
        .expect("captured graph");
    let mut rng = Rng::new(3);
    for (i, n) in [100_000usize, 7, 0, cap, 6144, 250_001, 1]
        .into_iter()
        .enumerate()
    {
        let k32 = keys_for(&mut rng, n, 32, Dist::FloatDepth);
        let k16: Vec<u16> = (0..n).map(|_| rng.below(4346) as u16).collect();
        let vals: Vec<u32> = (0..n as u32).collect();
        upload_u32(&gpu, &k32, &mut b32.k[0]);
        upload_u32(&gpu, &vals, &mut b32.v[0]);
        upload_u32(&gpu, &[n as u32], &mut b32.n);
        if n > 0 {
            s.memcpy_htod(&k16, &mut b16.k[0].slice_mut(0..n)).unwrap();
        }
        upload_u32(&gpu, &vals, &mut b16.v[0]);
        upload_u32(&gpu, &[n as u32], &mut b16.n);
        s.memcpy_htod(&cpu_u16_offsets(&k16, 8), &mut offsets)
            .unwrap();
        graph.launch().unwrap();
        gpu.sync().unwrap();
        let (ek, ev) = cpu_sort(&k32, &vals);
        assert_eq!(
            download_u32(&gpu, &b32.k[r32], n),
            ek,
            "replay {i}: u32 keys"
        );
        assert_eq!(
            download_u32(&gpu, &b32.v[r32], n),
            ev,
            "replay {i}: u32 values"
        );
        let (ek, ev) = cpu_sort(&k16, &vals);
        if n > 0 {
            assert_eq!(
                s.clone_dtoh(&b16.k[r16].slice(0..n)).unwrap(),
                ek,
                "replay {i}: u16 keys"
            );
        }
        assert_eq!(
            download_u32(&gpu, &b16.v[r16], n),
            ev,
            "replay {i}: u16 values"
        );
    }
    drop(graph);
    unsafe { gpu.ctx.enable_event_tracking() };
    println!("graph of 1 histogram + 6 onesweep passes replayed 7 times");
}

// ------------------------------------------------------------------ GPU: binning

/// Everything a binning run produced, downloaded.
pub(super) struct BinOut {
    pub status: Vec<u32>,
    pub n_tiles: Vec<u32>,
    pub sorted_v: Vec<u32>,
    pub inst_off: Vec<u32>,
    pub keys: Vec<u16>,
    pub vals: Vec<u32>,
    pub tile_start: Vec<u32>,
    pub chain_off: Vec<u32>,
    pub digit_offsets: Vec<u32>,
}

pub(super) fn frame_cam(w: u32, h: u32, tile_h: u32) -> FrameCam {
    let (grid_w, grid_h) = (w.div_ceil(TILE_W), h.div_ceil(tile_h));
    FrameCam {
        width: w,
        height: h,
        grid_w,
        grid_h,
        n_tiles: grid_w * grid_h,
        ..Default::default()
    }
}

/// Uploads records (v order = index order) and their depth keys, sets
/// status[ST_N_VIS], then count_tiles → sort_depth → bin_tiles.
pub(super) fn upload_frame(gpu: &Gpu, f: &mut FrameBuffers, cam: &FrameCam, recs: &[SplatRec]) {
    let s = &gpu.stream;
    f.iter
        .upload(gpu, &IterParams::default(), Some(cam))
        .unwrap();
    let n = recs.len();
    if n > 0 {
        s.memcpy_htod(recs, &mut f.rec.slice_mut(0..n)).unwrap();
    }
    let keys: Vec<u32> = recs.iter().map(|r| r.z.to_bits()).collect();
    let vals: Vec<u32> = (0..n as u32).collect();
    upload_u32(gpu, &keys, &mut f.depth_key[0]);
    upload_u32(gpu, &vals, &mut f.depth_val[0]);
    let mut status = vec![0u32; STATUS_LEN];
    status[ST_N_VIS] = n as u32;
    s.memcpy_htod(&status, &mut f.iter.status).unwrap();
}

pub(super) fn download_bin(gpu: &Gpu, f: &FrameBuffers, nt: usize) -> BinOut {
    let s = &gpu.stream;
    let status = s.clone_dtoh(&f.iter.status).unwrap();
    let n = status[ST_N_VIS] as usize;
    let i = status[ST_N_INST] as usize;
    let ok = status[ST_OVERFLOW] == 0;
    let ni = if ok { i } else { 0 };
    BinOut {
        n_tiles: download_u32(gpu, &f.n_tiles, n),
        sorted_v: download_u32(gpu, &f.depth_val[f.depth_sorted], n),
        inst_off: download_u32(gpu, &f.inst_off, n + 1),
        keys: if ni > 0 {
            s.clone_dtoh(&f.tile_key[f.tile_sorted].slice(0..ni))
                .unwrap()
        } else {
            Vec::new()
        },
        vals: download_u32(gpu, &f.inst_v[f.tile_sorted], ni),
        tile_start: download_u32(gpu, &f.tile_start, nt + 1),
        chain_off: download_u32(gpu, &f.chain_off, nt + 1),
        digit_offsets: download_u32(gpu, &f.tile_digit_offsets, 2 * RADIX_BINS as usize),
        status,
    }
}

/// The whole binning on the CPU: depth order, offsets, per-tile lists.
pub(super) struct CpuBin {
    pub n_tiles: Vec<u32>,
    pub order: Vec<u32>,
    pub inst_off: Vec<u32>,
    pub keys: Vec<u16>,
    pub vals: Vec<u32>,
    pub tile_start: Vec<u32>,
    pub chain_off: Vec<u32>,
    pub tiles_hit: u32,
    pub digit_offsets: Vec<u32>,
}

pub(super) fn cpu_bin(recs: &[SplatRec], w: u32, h: u32, tile_h: u32, chain_k: u32) -> CpuBin {
    let nt = (w.div_ceil(TILE_W) * h.div_ceil(tile_h)) as usize;
    let tiles: Vec<Vec<u32>> = recs.iter().map(|r| cpu_tiles(r, w, h, tile_h)).collect();
    let n_tiles: Vec<u32> = tiles.iter().map(|t| t.len() as u32).collect();
    let keys: Vec<u32> = recs.iter().map(|r| r.z.to_bits()).collect();
    let (_, order) = cpu_sort(&keys, &(0..recs.len() as u32).collect::<Vec<_>>());
    let mut inst_off = Vec::with_capacity(recs.len() + 1);
    let mut run = 0u32;
    let mut lists: Vec<Vec<u32>> = vec![Vec::new(); nt];
    for &v in &order {
        inst_off.push(run);
        run += n_tiles[v as usize];
        for &t in &tiles[v as usize] {
            lists[t as usize].push(v);
        }
    }
    inst_off.push(run);
    let mut out_keys = Vec::with_capacity(run as usize);
    let mut out_vals = Vec::with_capacity(run as usize);
    let (mut tile_start, mut chain_off) = (Vec::with_capacity(nt + 1), Vec::with_capacity(nt + 1));
    let (mut ti, mut ci, mut hit) = (0u32, 0u32, 0u32);
    let mut counts = vec![0u32; nt];
    for (t, l) in lists.iter().enumerate() {
        tile_start.push(ti);
        chain_off.push(ci);
        ti += l.len() as u32;
        ci += (l.len() as u32).div_ceil(BUCKET * chain_k);
        hit += u32::from(!l.is_empty());
        counts[t] = l.len() as u32;
        out_keys.extend(std::iter::repeat_n(t as u16, l.len()));
        out_vals.extend_from_slice(l);
    }
    tile_start.push(ti);
    chain_off.push(ci);
    let sorted_keys: Vec<u16> = (0..nt)
        .flat_map(|t| std::iter::repeat_n(t as u16, counts[t] as usize))
        .collect();
    CpuBin {
        n_tiles,
        order,
        inst_off,
        keys: out_keys,
        vals: out_vals,
        tile_start,
        chain_off,
        tiles_hit: hit,
        digit_offsets: cpu_u16_offsets(&sorted_keys, TILE_LO_BITS),
    }
}

fn compare_bin(name: &str, g: &BinOut, c: &CpuBin) {
    assert_eq!(g.status[ST_OVERFLOW], 0, "{name}: overflow");
    assert_eq!(
        g.n_tiles, c.n_tiles,
        "{name}: n_tiles (count rule, CPU mirror)"
    );
    assert_eq!(g.sorted_v, c.order, "{name}: depth order");
    assert_eq!(g.inst_off, c.inst_off, "{name}: inst_off");
    assert_eq!(g.status[ST_N_INST] as usize, c.vals.len(), "{name}: N_INST");
    assert_eq!(
        g.status[ST_N_CHAINS],
        *c.chain_off.last().unwrap(),
        "{name}: N_CHAINS"
    );
    assert_eq!(g.status[ST_N_TILES_HIT], c.tiles_hit, "{name}: N_TILES_HIT");
    assert_eq!(g.tile_start, c.tile_start, "{name}: tile_start");
    assert_eq!(g.chain_off, c.chain_off, "{name}: chain_off");
    assert_eq!(g.keys, c.keys, "{name}: sorted tile keys");
    if g.vals != c.vals {
        let i = g
            .vals
            .iter()
            .zip(&c.vals)
            .position(|(a, b)| a != b)
            .unwrap();
        panic!(
            "{name}: sorted instance values differ first at {i} (tile {}): {} vs {}",
            c.keys[i], g.vals[i], c.vals[i]
        );
    }
    assert_eq!(
        g.digit_offsets, c.digit_offsets,
        "{name}: tile digit offsets"
    );
}

pub(super) fn bin_frame(
    gpu: &Gpu,
    sorter: &mut RadixSorter,
    variant: RasterVariant,
    recs: &[SplatRec],
    w: u32,
    h: u32,
    i_cap: u32,
) -> (FrameBuffers, SplatKernels) {
    let k = SplatKernels::load(gpu, variant).unwrap();
    let cap = Capacity::new(recs.len().max(1) as u32, i_cap.max(1), w, h, &variant);
    let mut f = FrameBuffers::alloc(gpu, cap, &variant).unwrap();
    let cam = frame_cam(w, h, variant.tile_h as u32);
    upload_frame(gpu, &mut f, &cam, recs);
    count_tiles(&k, &gpu.stream, &mut f).unwrap();
    sort_depth(&k, &gpu.stream, sorter, &mut f).unwrap();
    bin_tiles(&k, &gpu.stream, sorter, &mut f).unwrap();
    gpu.sync().unwrap();
    (f, k)
}

fn mixed_scene(rng: &mut Rng, n: usize, w: u32, h: u32) -> Vec<SplatRec> {
    let mut recs: Vec<SplatRec> = (0..n).map(|_| random_rec(rng, w, h)).collect();
    // Depth ties (stability: v order), whole-image splats (warp path, > 4096
    // tiles at 1297×840), thin diagonals, edge and corner splats.
    for i in (0..n).step_by(97) {
        recs[i].z = 7.25;
    }
    let (fw, fh) = (w as f32, h as f32);
    recs.push(make_rec(fw * 0.5, fh * 0.5, 3000.0, 2500.0, 0.3, 0.9, 1.0));
    recs.push(make_rec(fw * 0.3, fh * 0.6, 900.0, 4.0, 0.7, 0.5, 2.0));
    recs.push(make_rec(-3.0, -3.0, 12.0, 12.0, 0.0, 0.8, 3.0));
    recs.push(make_rec(fw + 2.0, fh - 1.0, 30.0, 2.0, 1.2, 0.8, 7.25));
    recs.push(make_rec(
        fw * 0.5,
        fh * 0.5,
        0.2,
        0.2,
        0.0,
        1.0 / 255.0,
        4.0,
    ));
    recs
}

/// count_tiles + sort_depth + bin_tiles against the CPU mirror, bit for bit:
/// n_tiles, depth order, offsets, sorted (tile, v) lists, ranges, chains,
/// status words, digit offsets. One sorter for every case.
#[test]
#[ignore]
fn gpu_bin_tiles_matches_cpu_reference() {
    let gpu = gpu();
    let mut sorter = RadixSorter::new(&gpu, 8_000_000).unwrap();
    let mut rng = Rng::new(11);
    let d = RasterVariant::default();
    let cases: Vec<(&str, RasterVariant, u32, u32, Vec<SplatRec>)> = vec![
        (
            "mixed 1297x840",
            d,
            1297,
            840,
            mixed_scene(&mut rng, 30_000, 1297, 840),
        ),
        (
            "mixed 1297x840 tile_h 12",
            RasterVariant { tile_h: 12, ..d },
            1297,
            840,
            mixed_scene(&mut rng, 30_000, 1297, 840),
        ),
        ("odd 67x53", d, 67, 53, mixed_scene(&mut rng, 600, 67, 53)),
        ("1x1", d, 1, 1, mixed_scene(&mut rng, 64, 1, 1)),
        ("empty", d, 640, 480, Vec::new()),
        (
            "chain_k 1",
            RasterVariant { chain_k: 1, ..d },
            300,
            200,
            mixed_scene(&mut rng, 3000, 300, 200),
        ),
        ("crowded tile", d, 160, 160, {
            // > 65,535 Gaussians in one tile, plus a background.
            let mut r: Vec<SplatRec> = (0..70_000)
                .map(|i| {
                    make_rec(
                        40.5,
                        40.5,
                        0.6,
                        0.5,
                        0.0,
                        0.5,
                        1.0 + (i % 1000) as f32 * 0.01,
                    )
                })
                .collect();
            r.extend((0..2000).map(|_| random_rec(&mut rng, 160, 160)));
            r
        }),
    ];
    for (name, variant, w, h, recs) in cases {
        let cpu = cpu_bin(&recs, w, h, variant.tile_h as u32, variant.chain_k as u32);
        let i_cap = (cpu.vals.len() as u32).max(1);
        let (f, _k) = bin_frame(&gpu, &mut sorter, variant, &recs, w, h, i_cap);
        let nt = (w.div_ceil(TILE_W) * h.div_ceil(variant.tile_h as u32)) as usize;
        let g = download_bin(&gpu, &f, nt);
        compare_bin(name, &g, &cpu);
        let max_tile = (0..nt)
            .map(|t| cpu.tile_start[t + 1] - cpu.tile_start[t])
            .max()
            .unwrap_or(0);
        println!(
            "{name}: V {} I {} chains {} tiles hit {}/{nt}, longest tile {max_tile}",
            recs.len(),
            cpu.vals.len(),
            cpu.chain_off[nt],
            cpu.tiles_hit
        );
    }
}

/// I over capacity: OVF_INSTANCES with the true total, nothing after scan_tiles
/// writes (tile buffers untouched, no chains), and the same sorter then bins
/// the frame correctly at a grown capacity (the replay of spec-raster 1.5).
#[test]
#[ignore]
fn gpu_instance_overflow_is_a_no_op_then_replays() {
    let gpu = gpu();
    let mut sorter = RadixSorter::new(&gpu, 2_000_000).unwrap();
    let mut rng = Rng::new(5);
    let (w, h) = (640, 480);
    let v = RasterVariant::default();
    let recs = mixed_scene(&mut rng, 20_000, w, h);
    let cpu = cpu_bin(&recs, w, h, 16, 4);
    let need = cpu.vals.len() as u32;
    let k = SplatKernels::load(&gpu, v).unwrap();
    let mut f = FrameBuffers::alloc(
        &gpu,
        Capacity::new(recs.len() as u32, need - 1, w, h, &v),
        &v,
    )
    .unwrap();
    let sentinel = vec![0xBEEFu16; (need - 1) as usize];
    for b in &mut f.tile_key {
        gpu.stream.memcpy_htod(&sentinel, b).unwrap();
    }
    upload_frame(&gpu, &mut f, &frame_cam(w, h, 16), &recs);
    count_tiles(&k, &gpu.stream, &mut f).unwrap();
    sort_depth(&k, &gpu.stream, &mut sorter, &mut f).unwrap();
    bin_tiles(&k, &gpu.stream, &mut sorter, &mut f).unwrap();
    let st = gpu.stream.clone_dtoh(&f.iter.status).unwrap();
    assert_eq!(st[ST_OVERFLOW], OVF_INSTANCES);
    assert_eq!(st[ST_N_INST], need, "true total reported");
    assert_eq!(st[ST_N_CHAINS], 0, "bin_scan must not run");
    for b in &f.tile_key {
        assert!(
            gpu.stream
                .clone_dtoh(b)
                .unwrap()
                .iter()
                .all(|&x| x == 0xBEEF),
            "emit or the tile sort wrote"
        );
    }
    // Host: grow to 1.25x the need and replay; the sorter state is consistent.
    let (f2, _k) = bin_frame(&gpu, &mut sorter, v, &recs, w, h, need + need / 4);
    compare_bin(
        "replay after overflow",
        &download_bin(&gpu, &f2, (40 * 30) as usize),
        &cpu,
    );
    // And the small buffers still work for a frame that fits.
    let small = &recs[..2000];
    let cpu_small = cpu_bin(small, w, h, 16, 4);
    let (f3, _k) = bin_frame(&gpu, &mut sorter, v, small, w, h, need - 1);
    compare_bin(
        "small frame",
        &download_bin(&gpu, &f3, (40 * 30) as usize),
        &cpu_small,
    );
    println!(
        "overflow at I_cap {} (need {need}): no-op, replay exact",
        need - 1
    );
}

// ------------------------------------------------------------------ GPU: gsplat oracle (staged)

/// gsplat's own binning (`isect_tiles` + `isect_offset_encode`, ORACLE.md) as
/// the reference for the sort machinery: given gsplat's per-Gaussian tile sets
/// (rectangles from radii), our stable depth sort, scan, histogram, bin_scan
/// and two-pass tile sort must reproduce gsplat's 64-bit-key order and tile
/// offsets exactly. Staged: the first run writes `<case>.isect.in.npz` under
/// `/out/image-to-3d/oracle`; run `~/spl-work/bin/gs-oracle batch
/// ~/spl-work/out/image-to-3d/oracle` on the host, then run the test again.
#[test]
#[ignore]
fn gpu_isect_matches_gsplat_oracle() {
    let gpu = gpu();
    let dir = oracle::stage_dir();
    std::fs::create_dir_all(&dir).unwrap();
    let mut sorter = RadixSorter::new(&gpu, 8_000_000).unwrap();
    let cases = [
        ("isect_small", 256usize, 64u32, 48u32, 2u64),
        ("isect_garden", 200_000, 1297, 840, 3),
    ];
    let mut compared = 0;
    for (name, n, w, h, seed) in cases {
        let mut rng = Rng::new(seed);
        let mut means = Vec::with_capacity(2 * n);
        let mut radii = Vec::with_capacity(2 * n);
        let mut depths = Vec::with_capacity(n);
        for i in 0..n {
            means.push(rng.range(-20.0, w as f32 + 20.0));
            means.push(rng.range(-20.0, h as f32 + 20.0));
            let r = (rng.range(0.0, 4.0f32).exp() as i32).max(1);
            let culled = i % 31 == 0;
            radii.push(if culled { 0 } else { r });
            radii.push(if culled {
                0
            } else {
                (r as f32 * rng.range(0.3, 1.0)) as i32 + 1
            });
            // Ties every 13th (stability) and exact float duplicates.
            depths.push(if i % 13 == 0 {
                3.5
            } else {
                rng.range(0.2, 60.0)
            });
        }
        let mut z = Npz::new();
        z.f32("means2d", &[1, n, 2], means.clone()).unwrap();
        z.insert("radii", &[1, n, 2], NpyData::I32(radii.clone()))
            .unwrap();
        z.f32("depths", &[1, n], depths.clone()).unwrap();
        z.i32_scalar("width", w as i32).unwrap();
        z.i32_scalar("height", h as i32).unwrap();
        z.i32_scalar("tile_size", 16).unwrap();
        let input = oracle::case_path(&dir, name, "isect", "in");
        if !input.exists() || Npz::read(&input).map(|old| old != z).unwrap_or(true) {
            oracle::stage(&dir, name, "isect", &z).unwrap();
        }
        let Ok(out) = oracle::load_out(&dir, name, "isect") else {
            println!(
                "{name}: staged {}; run gs-oracle batch on the host and re-run",
                input.display()
            );
            continue;
        };
        let to_i64 = |k: &str| -> Vec<i64> {
            out.get(k)
                .unwrap()
                .to_f64()
                .iter()
                .map(|&x| x as i64)
                .collect()
        };
        let ids = to_i64("isect_ids");
        let flat = to_i64("flatten_ids");
        let offsets = to_i64("isect_offsets");
        let tpg = to_i64("tiles_per_gauss");
        // gsplat's tile set per Gaussian, from its own output.
        let mut sets: Vec<Vec<u32>> = vec![Vec::new(); n];
        for (&id, &g) in ids.iter().zip(&flat) {
            sets[g as usize].push((id >> 32) as u32);
        }
        for (g, s) in sets.iter_mut().enumerate() {
            s.sort_unstable();
            assert_eq!(s.len() as i64, tpg[g], "{name}: tiles_per_gauss of {g}");
        }
        // Ours: GPU depth sort, scan, emission of gsplat's sets in our order,
        // GPU histogram, bin_scan and tile sort.
        let v = RasterVariant::default();
        let k = SplatKernels::load(&gpu, v).unwrap();
        let i_total = flat.len() as u32;
        let cap = Capacity::new(n as u32, i_total.max(1), w, h, &v);
        let mut f = FrameBuffers::alloc(&gpu, cap, &v).unwrap();
        let s = &gpu.stream;
        let cam = frame_cam(w, h, 16);
        f.iter
            .upload(&gpu, &IterParams::default(), Some(&cam))
            .unwrap();
        let keys: Vec<u32> = depths.iter().map(|d| d.to_bits()).collect();
        upload_u32(&gpu, &keys, &mut f.depth_key[0]);
        upload_u32(
            &gpu,
            &(0..n as u32).collect::<Vec<_>>(),
            &mut f.depth_val[0],
        );
        let counts: Vec<u32> = sets.iter().map(|s| s.len() as u32).collect();
        upload_u32(&gpu, &counts, &mut f.n_tiles);
        let mut status = vec![0u32; STATUS_LEN];
        status[ST_N_VIS] = n as u32;
        s.memcpy_htod(&status, &mut f.iter.status).unwrap();
        sort_depth(&k, s, &mut sorter, &mut f).unwrap();
        scan_tiles(&k, s, &mut sorter, &mut f).unwrap();
        gpu.sync().unwrap();
        let order = download_u32(&gpu, &f.depth_val[f.depth_sorted], n);
        let inst_off = download_u32(&gpu, &f.inst_off, n + 1);
        let mut ek = vec![0u16; i_total as usize];
        let mut ev = vec![0u32; i_total as usize];
        for (kk, &g) in order.iter().enumerate() {
            for (j, &t) in sets[g as usize].iter().enumerate() {
                ek[inst_off[kk] as usize + j] = t as u16;
                ev[inst_off[kk] as usize + j] = g;
            }
        }
        if i_total > 0 {
            s.memcpy_htod(&ek, &mut f.tile_key[0].slice_mut(0..i_total as usize))
                .unwrap();
            s.memcpy_htod(&ev, &mut f.inst_v[0].slice_mut(0..i_total as usize))
                .unwrap();
        }
        tile_hist(&k, s, &sorter, &mut f).unwrap();
        bin_scan(&k, s, &mut f).unwrap();
        sort_tiles(&k, s, &mut sorter, &mut f).unwrap();
        gpu.sync().unwrap();
        let nt = cam.n_tiles as usize;
        let g = download_bin(&gpu, &f, nt);
        assert_eq!(g.status[ST_N_INST], i_total, "{name}: instances");
        let ours: Vec<i64> = g.vals.iter().map(|&v| v as i64).collect();
        assert_eq!(
            ours, flat,
            "{name}: sorted Gaussian ids vs gsplat flatten_ids"
        );
        let ours_keys: Vec<i64> = g.keys.iter().map(|&t| t as i64).collect();
        let gs_keys: Vec<i64> = ids.iter().map(|&id| id >> 32).collect();
        assert_eq!(
            ours_keys, gs_keys,
            "{name}: sorted tile ids vs gsplat isect_ids >> 32"
        );
        let ours_off: Vec<i64> = g.tile_start[..nt].iter().map(|&x| x as i64).collect();
        assert_eq!(
            ours_off, offsets,
            "{name}: tile_start vs gsplat isect_offsets"
        );
        // gsplat's depth word: equal to ours for every instance.
        for (&id, &gid) in ids.iter().zip(&flat) {
            assert_eq!(
                (id & 0xffff_ffff) as u32,
                keys[gid as usize],
                "{name}: depth bits"
            );
        }
        println!("{name}: N {n}, {i_total} intersections, {nt} tiles: identical to gsplat");
        compared += 1;
    }
    if compared < cases.len() {
        println!(
            "gsplat oracle: {compared}/{} cases compared (others staged)",
            cases.len()
        );
    }
}
