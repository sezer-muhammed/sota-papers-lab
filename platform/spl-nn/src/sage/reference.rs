//! CPU oracles for the sage kernels: bf16 and FP8 E4M3 conversions, the
//! key permutation, the KV-merge geometry, LayerNorm, attention in f64 and
//! error metrics. The tests and the benchmark compare the GPU against them.

use super::{FrameLayout, KvMerge, Plan};

/// bf16 bits -> f32 (exact).
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// f32 -> bf16 bits, round to nearest even (NaN stays NaN).
pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return ((bits >> 16) | 0x40) as u16;
    }
    let round = 0x7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(round) >> 16) as u16
}

/// E4M3 byte -> f32 (0x7f and 0xff are NaN).
pub fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = ((b >> 3) & 0xf) as i32;
    let man = (b & 7) as f32;
    if exp == 15 && b & 7 == 7 {
        return f32::NAN;
    }
    let mag = if exp == 0 {
        man * (2.0f32).powi(-9)
    } else {
        (1.0 + man / 8.0) * (2.0f32).powi(exp - 7)
    };
    sign * mag
}

/// f32 -> E4M3 byte: round to nearest even, saturating at ±448 like
/// `cvt.rn.satfinite.e4m3x2.f32`.
pub fn f32_to_e4m3(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7f;
    }
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    let a = x.abs();
    if a >= 448.0 {
        return sign | 0x7e;
    }
    // Every E4M3 value is k * 2^(e - 3) for an integer k in [8, 15] (normal)
    // or k * 2^-9 for k in [0, 7] (subnormal); round a to that grid.
    let e = if a < (2.0f32).powi(-6) {
        -6
    } else {
        a.log2().floor() as i32
    };
    let step = (2.0f64).powi(e - 3);
    let k = (a as f64 / step).round_ties_even() as u32;
    let (mut e, mut k) = (e, k);
    if k == 16 {
        e += 1;
        k = 8;
    }
    if e == -6 && k < 8 {
        return sign | k as u8;
    }
    if e > 8 || (e == 8 && k > 14) {
        return sign | 0x7e;
    }
    sign | (((e + 7) as u8) << 3) | (k - 8) as u8
}

/// Position of key `k` (0..16) inside its 16-key group of the FP8 V tile;
/// mirrors `sg_vpos`.
pub fn vpos(k: usize) -> usize {
    ((k >> 1) & 3) * 4 + ((k >> 3) << 1) + (k & 1)
}

/// The tokens (within one batch item) merged key `j` pools.
pub fn members(plan: &Plan, j: usize) -> Vec<usize> {
    match plan.merge {
        KvMerge::Dense => vec![j],
        KvMerge::Pool { stride, layout } => cell_members(&layout, stride, j),
    }
}

fn cell_members(layout: &FrameLayout, s: usize, j: usize) -> Vec<usize> {
    let per = layout.merged_per_frame(s);
    let (f, loc) = (j / per, j % per);
    let base = f * layout.tokens_per_frame();
    if loc < layout.special {
        return vec![base + loc];
    }
    let cell = loc - layout.special;
    let mw = layout.grid_w.div_ceil(s);
    let (y0, x0) = ((cell / mw) * s, (cell % mw) * s);
    let mut out = Vec::new();
    for y in y0..(y0 + s).min(layout.grid_h) {
        for x in x0..(x0 + s).min(layout.grid_w) {
            out.push(base + layout.special + y * layout.grid_w + x);
        }
    }
    out
}

/// LayerNorm over one row in place (biased variance, f32 like PyTorch).
pub fn layer_norm(x: &mut [f32], w: &[f32], b: &[f32], eps: f32) {
    let n = x.len() as f64;
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = x.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    let rstd = 1.0 / (var + eps as f64).sqrt();
    for (i, v) in x.iter_mut().enumerate() {
        *v = ((*v as f64 - mean) * rstd * w[i] as f64 + b[i] as f64) as f32;
    }
}

/// Mean-pools `k` and `v` (`[tokens][d]`, one batch item and head) the way
/// `plan` merges them. Returns pooled K, pooled V and the log2 member count
/// of every merged key.
pub fn merge_kv(plan: &Plan, k: &[f32], v: &[f32], d: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut km = vec![0f32; plan.tk * d];
    let mut vm = vec![0f32; plan.tk * d];
    let mut bias = vec![0f32; plan.tk];
    for j in 0..plan.tk {
        let m = members(plan, j);
        for &t in &m {
            for c in 0..d {
                km[j * d + c] += k[t * d + c];
                vm[j * d + c] += v[t * d + c];
            }
        }
        let n = m.len() as f32;
        for c in 0..d {
            km[j * d + c] /= n;
            vm[j * d + c] /= n;
        }
        bias[j] = n.log2();
    }
    (km, vm, bias)
}

/// Softmax attention in f64 for one batch item and head: `q` is
/// `[tq][d]`, `k` and `v` are `[tk][d]`, `bias_log2` adds `ln(2) · b_j` to
/// key j's logits. Returns `[tq][d]`.
pub fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    bias_log2: Option<&[f32]>,
    d: usize,
    scale: f32,
) -> Vec<f32> {
    let tq = q.len() / d;
    let tk = k.len() / d;
    let mut out = vec![0f32; tq * d];
    let mut logits = vec![0f64; tk];
    for i in 0..tq {
        let qi = &q[i * d..(i + 1) * d];
        let mut mx = f64::NEG_INFINITY;
        for (j, l) in logits.iter_mut().enumerate() {
            let kj = &k[j * d..(j + 1) * d];
            let dot: f64 = qi.iter().zip(kj).map(|(&a, &b)| a as f64 * b as f64).sum();
            *l = dot * scale as f64
                + bias_log2.map_or(0.0, |b| b[j] as f64 * std::f64::consts::LN_2);
            mx = mx.max(*l);
        }
        let mut sum = 0.0;
        let mut acc = vec![0f64; d];
        for (j, l) in logits.iter().enumerate() {
            let p = (l - mx).exp();
            sum += p;
            for (c, a) in acc.iter_mut().enumerate() {
                *a += p * v[j * d + c] as f64;
            }
        }
        for c in 0..d {
            out[i * d + c] = (acc[c] / sum) as f32;
        }
    }
    out
}

/// How far `ours` is from `reference`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Errors {
    /// Cosine similarity of the flattened tensors.
    pub cos: f64,
    /// Σ|a − r| / Σ|r|.
    pub rel_l1: f64,
    /// ‖a − r‖₂ / ‖r‖₂.
    pub rel_l2: f64,
    /// max |a − r|.
    pub max_abs: f64,
    /// max |r|, for scale.
    pub ref_max: f64,
}

impl Errors {
    pub fn of(ours: &[f32], reference: &[f32]) -> Self {
        assert_eq!(ours.len(), reference.len());
        let (mut dot, mut na, mut nr, mut l1, mut l1r, mut l2, mut mx, mut rmax) =
            (0f64, 0f64, 0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
        for (&a, &r) in ours.iter().zip(reference) {
            let (a, r) = (a as f64, r as f64);
            dot += a * r;
            na += a * a;
            nr += r * r;
            l1 += (a - r).abs();
            l1r += r.abs();
            l2 += (a - r) * (a - r);
            mx = mx.max((a - r).abs());
            rmax = rmax.max(r.abs());
        }
        Self {
            cos: dot / (na.sqrt() * nr.sqrt()).max(1e-300),
            rel_l1: l1 / l1r.max(1e-300),
            rel_l2: (l2 / nr.max(1e-300)).sqrt(),
            max_abs: mx,
            ref_max: rmax,
        }
    }
}

/// splitmix64: deterministic test data without a dependency.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal (Box-Muller).
    pub fn normal(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
    }
}
