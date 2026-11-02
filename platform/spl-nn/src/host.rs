//! Host-side bf16 and FP8 E4M3 conversions, plus plain CPU reference
//! implementations of every kernel in this crate. The references accumulate
//! in f64 and serve as test oracles here and in the engine's own tests.

/// f32 → bf16 bits, round to nearest even (NaN stays NaN).
pub fn bf16(v: f32) -> u16 {
    let b = v.to_bits();
    if v.is_nan() {
        return ((b >> 16) | 0x40) as u16;
    }
    let round = 0x7fff + ((b >> 16) & 1);
    (b.wrapping_add(round) >> 16) as u16
}

pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// Rounds through bf16.
pub fn round_bf16(v: f32) -> f32 {
    bf16_to_f32(bf16(v))
}

pub fn to_bf16(v: &[f32]) -> Vec<u16> {
    v.iter().map(|&x| bf16(x)).collect()
}

pub fn from_bf16(v: &[u16]) -> Vec<f32> {
    v.iter().map(|&x| bf16_to_f32(x)).collect()
}

/// f32 → IEEE fp16 bits, round to nearest even; overflow gives ±inf.
pub fn f16(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let a = b & 0x7fff_ffff;
    if a > 0x7f80_0000 {
        return sign | 0x7e00;
    }
    if a >= 0x477f_f000 {
        // ≥ 65520 rounds past 65504.
        return sign | 0x7c00;
    }
    if a < 0x3880_0000 {
        // Below 2^-14: subnormal, in units of 2^-24 (exact in f64).
        let x = f32::from_bits(a) as f64 * 2f64.powi(24);
        let f = x.floor();
        let d = x - f;
        let r = if d > 0.5 || (d == 0.5 && f % 2.0 == 1.0) {
            f + 1.0
        } else {
            f
        };
        return sign | r as u16;
    }
    let e = (a >> 23) as i32 - 127;
    let (hi, rem) = ((a & 0x7f_ffff) >> 13, a & 0x1fff);
    let mut h = (((e + 15) as u32) << 10) | hi;
    if rem > 0x1000 || (rem == 0x1000 && hi & 1 == 1) {
        h += 1; // a mantissa carry correctly bumps the exponent
    }
    sign | h as u16
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f32;
    match e {
        0 => sign * m * 2f32.powi(-24),
        31 if m == 0.0 => sign * f32::INFINITY,
        31 => f32::NAN,
        _ => sign * (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    }
}

/// Largest finite FP8 E4M3 value (the `fn` variant: no infinities).
pub const FP8_MAX: f32 = 448.0;

/// FP8 E4M3 bits → f32 (torch.float8_e4m3fn).
pub fn fp8_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((b >> 3) & 0xf) as i32;
    let m = (b & 7) as f32;
    if e == 15 && b & 7 == 7 {
        return f32::NAN;
    }
    let mag = if e == 0 {
        m / 8.0 * 2f32.powi(-6)
    } else {
        (1.0 + m / 8.0) * 2f32.powi(e - 7)
    };
    sign * mag
}

/// f32 → FP8 E4M3 bits, round to nearest even, saturating at ±448 like
/// PTX `cvt.rn.satfinite.e4m3x2.f32`.
pub fn fp8(v: f32) -> u8 {
    if v.is_nan() {
        return 0x7f;
    }
    let sign = if v.is_sign_negative() { 0x80u8 } else { 0 };
    let a = v.abs();
    if a >= FP8_MAX {
        return sign | 0x7e;
    }
    // Positive codes 0x00..=0x7e are increasing in value.
    let (mut lo, mut hi) = (0u8, 0x7eu8);
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if fp8_to_f32(mid) <= a {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let (dl, dh) = (a - fp8_to_f32(lo), fp8_to_f32(hi) - a);
    let code = if dl < dh || (dl == dh && lo & 1 == 0) {
        lo
    } else {
        hi
    };
    sign | code
}

pub fn from_fp8(v: &[u8]) -> Vec<f32> {
    v.iter().map(|&x| fp8_to_f32(x)).collect()
}

/// erf with |error| < 1.2e-7 (Numerical Recipes `erfc` Chebyshev fit).
pub fn erf(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let poly = -z * z - 1.265_512_23
        + t * (1.000_023_68
            + t * (0.374_091_96
                + t * (0.096_784_18
                    + t * (-0.186_288_06
                        + t * (0.278_868_07
                            + t * (-1.135_203_98
                                + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77))))))));
    let erfc = t * poly.exp();
    if x >= 0.0 { 1.0 - erfc } else { erfc - 1.0 }
}

pub fn gelu_erf(x: f64) -> f64 {
    0.5 * x * (1.0 + erf(x * std::f64::consts::FRAC_1_SQRT_2))
}

pub fn gelu_tanh(x: f64) -> f64 {
    let u = 0.797_884_560_802_865_4 * (x + 0.044715 * x * x * x);
    0.5 * x * (1.0 + u.tanh())
}

/// `y[r][n] = Σ_k x[r][k] · w[n][k] + bias[n]` in f64.
pub fn linear(x: &[f32], w: &[f32], bias: Option<&[f32]>, k: usize, n: usize) -> Vec<f64> {
    let rows = x.len() / k;
    let mut y = vec![0.0f64; rows * n];
    for r in 0..rows {
        let xr = &x[r * k..(r + 1) * k];
        for o in 0..n {
            let wr = &w[o * k..(o + 1) * k];
            let mut acc = bias.map_or(0.0, |b| b[o] as f64);
            for i in 0..k {
                acc += xr[i] as f64 * wr[i] as f64;
            }
            y[r * n + o] = acc;
        }
    }
    y
}

/// LayerNorm of each `c`-wide row in f64.
pub fn layer_norm(x: &[f32], c: usize, gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f64> {
    let mut y = vec![0.0f64; x.len()];
    for (xr, yr) in x.chunks(c).zip(y.chunks_mut(c)) {
        let mean = xr.iter().map(|&v| v as f64).sum::<f64>() / c as f64;
        let var = xr.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / c as f64;
        let rstd = 1.0 / (var + eps as f64).sqrt();
        for i in 0..c {
            yr[i] = (xr[i] as f64 - mean) * rstd * gamma[i] as f64 + beta[i] as f64;
        }
    }
    y
}

/// RMSNorm of each `c`-wide row in f64 (`gamma` scales, no bias).
pub fn rms_norm(x: &[f32], c: usize, gamma: &[f32], eps: f32) -> Vec<f64> {
    let mut y = vec![0.0f64; x.len()];
    for (xr, yr) in x.chunks(c).zip(y.chunks_mut(c)) {
        let ms = xr.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / c as f64;
        let r = 1.0 / (ms + eps as f64).sqrt();
        for i in 0..c {
            yr[i] = xr[i] as f64 * r * gamma[i] as f64;
        }
    }
    y
}

/// Per-row FP8 quantisation exactly as the kernels do it, in f32:
/// `scale = amax / 448`, `q = fp8(v · (448 / amax))`; an all-zero row gets
/// scale 1.
pub fn quant_rows(y: &[f32], c: usize) -> (Vec<u8>, Vec<f32>) {
    let mut q = Vec::with_capacity(y.len());
    let mut scales = Vec::with_capacity(y.len() / c);
    for row in y.chunks(c) {
        let amax = row.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let (scale, inv) = if amax > 0.0 {
            (amax / FP8_MAX, FP8_MAX / amax)
        } else {
            (1.0, 1.0)
        };
        scales.push(scale);
        q.extend(row.iter().map(|&v| fp8(v * inv)));
    }
    (q, scales)
}

/// Per-output-channel FP8 quantisation of `[n][k]` weights: one scale per
/// row (`amax / 448`). Returns the FP8 bytes and the scales.
pub fn quant_weights(w: &[f32], k: usize) -> (Vec<u8>, Vec<f32>) {
    quant_rows(w, k)
}

/// Softmax attention per head in f64. `q`, `k`, `v` are `[heads][len][d]`
/// for one sequence; returns `[len][heads·d]`.
pub fn attention(q: &[f32], k: &[f32], v: &[f32], heads: usize, d: usize, scale: f64) -> Vec<f64> {
    let len = q.len() / (heads * d);
    let mut o = vec![0.0f64; len * heads * d];
    let mut s = vec![0.0f64; len];
    for h in 0..heads {
        let base = h * len * d;
        for i in 0..len {
            let qi = &q[base + i * d..base + (i + 1) * d];
            let mut mx = f64::NEG_INFINITY;
            for j in 0..len {
                let kj = &k[base + j * d..base + (j + 1) * d];
                let dot: f64 = qi.iter().zip(kj).map(|(&a, &b)| a as f64 * b as f64).sum();
                s[j] = dot * scale;
                mx = mx.max(s[j]);
            }
            let mut sum = 0.0;
            for sj in s.iter_mut() {
                *sj = (*sj - mx).exp();
                sum += *sj;
            }
            let out = &mut o[(i * heads + h) * d..(i * heads + h + 1) * d];
            for j in 0..len {
                let p = s[j] / sum;
                let vj = &v[base + j * d..base + (j + 1) * d];
                for (oo, &vv) in out.iter_mut().zip(vj) {
                    *oo += p * vv as f64;
                }
            }
        }
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_rounds_to_nearest_even() {
        assert_eq!(bf16(1.0), 0x3f80);
        assert_eq!(
            bf16_to_f32(bf16(1.0 + 1.0 / 256.0)),
            1.0,
            "tie rounds to even"
        );
        assert_eq!(
            bf16_to_f32(bf16(1.0 + 3.0 / 256.0)),
            1.0 + 4.0 / 256.0,
            "tie rounds up to even"
        );
        assert_eq!(bf16(-2.5), 0xc020);
        assert!(bf16_to_f32(bf16(f32::NAN)).is_nan());
        assert_eq!(bf16_to_f32(bf16(f32::INFINITY)), f32::INFINITY);
    }

    #[test]
    fn f16_rounds_to_nearest_even() {
        assert_eq!(f16(1.0), 0x3c00);
        assert_eq!(f16(-2.0), 0xc000);
        assert_eq!(f16(65504.0), 0x7bff);
        assert_eq!(f16(65519.0), 0x7bff);
        assert_eq!(f16(65520.0), 0x7c00);
        assert_eq!(f16(2f32.powi(-24)), 1);
        assert_eq!(f16(2f32.powi(-25)), 0, "tie to even (0)");
        assert_eq!(f16(3.0 * 2f32.powi(-25)), 2, "tie to even (2)");
        assert_eq!(f16(1.0 + 2f32.powi(-11)), 0x3c00, "tie to even");
        assert_eq!(f16(1.0 + 3.0 * 2f32.powi(-11)), 0x3c02, "tie up to even");
        assert_eq!(f16(2047.5), 0x6800, "carry into the exponent");
        for h in (0u16..0x7c00).step_by(7) {
            assert_eq!(f16(f16_to_f32(h)), h, "{h:#x}");
        }
    }

    #[test]
    fn fp8_table_matches_the_format() {
        assert_eq!(fp8_to_f32(0x7e), 448.0);
        assert_eq!(fp8_to_f32(0x08), 2f32.powi(-6), "smallest normal");
        assert_eq!(fp8_to_f32(0x01), 2f32.powi(-9), "smallest subnormal");
        assert_eq!(fp8_to_f32(0x38), 1.0);
        assert!(fp8_to_f32(0x7f).is_nan());
        assert_eq!(fp8_to_f32(0xb8), -1.0);
        for b in 0u8..=0x7e {
            assert_eq!(fp8(fp8_to_f32(b)), b, "round trip {b:#x}");
            assert_eq!(fp8(-fp8_to_f32(b)), b | 0x80, "negative {b:#x}");
        }
    }

    #[test]
    fn fp8_rounds_and_saturates() {
        assert_eq!(fp8(1000.0), 0x7e);
        assert_eq!(fp8(-1e9), 0xfe);
        assert_eq!(fp8(f32::INFINITY), 0x7e);
        // 1.0625 sits halfway between 1.0 (0x38, even) and 1.125 (0x39).
        assert_eq!(fp8(1.0625), 0x38);
        // 1.1875 sits halfway between 1.125 (0x39) and 1.25 (0x3a, even).
        assert_eq!(fp8(1.1875), 0x3a);
        assert_eq!(fp8(1.07), 0x39);
        assert_eq!(fp8(2f32.powi(-11)), 0, "below half the smallest subnormal");
    }

    #[test]
    fn erf_and_gelu_match_known_values() {
        assert!((erf(0.5) - 0.520_499_877_813_046_5).abs() < 2e-7);
        assert!((erf(-1.5) + 0.966_105_146_475_310_7).abs() < 2e-7);
        assert!((gelu_erf(1.0) - 0.841_344_746_068_542_9).abs() < 2e-7);
        assert!((gelu_tanh(1.0) - 0.841_191_990_607_477).abs() < 1e-9);
        assert_eq!(gelu_erf(0.0), 0.0);
    }

    #[test]
    fn quant_rows_uses_amax_scaling() {
        let (q, s) = quant_rows(&[0.5, -2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 4);
        assert_eq!(s, vec![2.0 / 448.0, 1.0]);
        assert_eq!(fp8_to_f32(q[1]), -448.0);
        assert_eq!(fp8_to_f32(q[0]), 112.0);
        assert!(q[4..].iter().all(|&b| b == 0));
    }

    #[test]
    fn attention_reference_is_a_convex_mix() {
        // One head, two tokens, d = 2: outputs lie between the two values.
        let q = [1.0, 0.0, 0.0, 1.0];
        let k = [1.0, 0.0, 0.0, 1.0];
        let v = [1.0, 2.0, 3.0, 4.0];
        let o = attention(&q, &k, &v, 1, 2, 1.0);
        let e = std::f64::consts::E;
        let p0 = e / (e + 1.0);
        assert!((o[0] - (p0 * 1.0 + (1.0 - p0) * 3.0)).abs() < 1e-12);
        assert!((o[3] - ((1.0 - p0) * 2.0 + p0 * 4.0)).abs() < 1e-12);
    }
}
