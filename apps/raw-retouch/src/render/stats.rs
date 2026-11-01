//! Turns the stats kernel's per-candidate accumulators into [`Stats`].

use super::layout::*;
use crate::types::Stats;

/// Bins of `Stats::luma_hist`.
const HIST_BINS: usize = 64;

/// `counts` is one candidate's `PART_U32` record summed over chunks, `sums`
/// its `N_SUMS` sums.
pub(crate) fn finalize(counts: &[u32], sums: &[f64]) -> Stats {
    let luma = &counts[..LUMA_BINS];
    let chroma = &counts[LUMA_BINS..LUMA_BINS + CHROMA_BINS];
    let c = &counts[LUMA_BINS + CHROMA_BINS..PART_U32];
    let n = c[C_PX] as f64;
    if n == 0.0 {
        return Stats {
            luma_hist: vec![0.0; HIST_BINS],
            structure: 1.0,
            ..Stats::default()
        };
    }
    let share = |k: usize| (c[k] as f64 / n) as f32;
    let per = LUMA_BINS / HIST_BINS;
    let luma_hist: Vec<f32> = luma
        .chunks_exact(per)
        .map(|b| (b.iter().map(|&v| v as f64).sum::<f64>() / n) as f32)
        .collect();
    let entropy = -luma_hist
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| p as f64 * (p as f64).log2())
        .sum::<f64>();

    // Colorfulness (Hasler & Süsstrunk 2003) on 0-255 sRGB.
    let mean = |k: usize| sums[k] / n;
    let var = |s: usize, s2: usize| (mean(s2) - mean(s) * mean(s)).max(0.0);
    let sigma = (var(S_RG, S_RG2) + var(S_YB, S_YB2)).sqrt();
    let mu = (mean(S_RG).powi(2) + mean(S_YB).powi(2)).sqrt();
    let colorfulness = 255.0 * (sigma + 0.3 * mu);

    let cast_n = c[C_CAST] as f64;
    let (cast_a, cast_b) = if cast_n > 0.0 {
        (sums[S_CA] / cast_n, sums[S_CB] / cast_n)
    } else {
        (0.0, 0.0)
    };
    let structure = if c[C_SBLK] > 0 {
        sums[S_STRUCT] / c[C_SBLK] as f64
    } else {
        1.0
    };
    let local_contrast = if c[C_LCBLK] > 0 {
        sums[S_LC] / c[C_LCBLK] as f64
    } else {
        0.0
    };
    let lp = |q: f64| percentile(luma, n, q, 1.0);
    Stats {
        luma_hist,
        luma_mean: mean(S_LUMA) as f32,
        luma_p01: lp(0.01),
        luma_p05: lp(0.05),
        luma_p50: lp(0.50),
        luma_p95: lp(0.95),
        luma_p99: lp(0.99),
        clip_hi: share(C_HI),
        clip_hi_new: share(C_HI_NEW),
        clip_lo: share(C_LO),
        colorfulness: colorfulness as f32,
        chroma_mean: mean(S_CHROMA) as f32,
        chroma_p95: percentile(chroma, n, 0.95, CHROMA_MAX as f64),
        out_of_gamut: share(C_OOG),
        cast_a: cast_a as f32,
        cast_b: cast_b as f32,
        structure: structure.clamp(0.0, 1.0) as f32,
        local_contrast: local_contrast as f32,
        entropy: entropy as f32,
    }
}

/// The `q` quantile of a histogram over [0, `max`), interpolated inside the bin.
fn percentile(hist: &[u32], n: f64, q: f64, max: f64) -> f32 {
    let target = q * n;
    let mut below = 0.0;
    for (i, &count) in hist.iter().enumerate() {
        let count = count as f64;
        if count > 0.0 && below + count >= target {
            let within = ((target - below) / count).clamp(0.0, 1.0);
            return ((i as f64 + within) / hist.len() as f64 * max) as f32;
        }
        below += count;
    }
    max as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(f: impl Fn(&mut Vec<u32>, &mut Vec<f64>)) -> Stats {
        let mut u = vec![0u32; PART_U32];
        let mut s = vec![0f64; N_SUMS];
        f(&mut u, &mut s);
        finalize(&u, &s)
    }

    #[test]
    fn uniform_luma_has_linear_percentiles_and_full_entropy() {
        let st = record(|u, s| {
            for b in &mut u[..LUMA_BINS] {
                *b = 10;
            }
            u[LUMA_BINS] = 10 * LUMA_BINS as u32;
            u[LUMA_BINS + CHROMA_BINS + C_PX] = 10 * LUMA_BINS as u32;
            s[S_LUMA] = 0.5 * 10.0 * LUMA_BINS as f64;
        });
        assert!((st.luma_p50 - 0.5).abs() < 1e-3);
        assert!((st.luma_p05 - 0.05).abs() < 1e-3);
        assert!((st.luma_p99 - 0.99).abs() < 1e-3);
        assert!((st.luma_mean - 0.5).abs() < 1e-6);
        assert!((st.luma_hist.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert_eq!(st.luma_hist.len(), 64);
        assert!((st.entropy - 6.0).abs() < 1e-4);
        assert!(st.chroma_p95 < 0.4 / CHROMA_BINS as f32 + 1e-6);
    }

    #[test]
    fn shares_casts_and_colorfulness() {
        let st = record(|u, s| {
            let c = LUMA_BINS + CHROMA_BINS;
            u[0] = 100;
            u[LUMA_BINS] = 100;
            u[c + C_PX] = 100;
            u[c + C_HI] = 25;
            u[c + C_HI_NEW] = 5;
            u[c + C_LO] = 10;
            u[c + C_OOG] = 1;
            u[c + C_CAST] = 50;
            u[c + C_SBLK] = 4;
            u[c + C_LCBLK] = 2;
            s[S_CA] = 50.0 * 0.01;
            s[S_CB] = -50.0 * 0.02;
            s[S_STRUCT] = 3.0;
            s[S_LC] = 0.1;
            // rg constant 0.1, yb constant 0: colorfulness = 0.3 * 0.1 * 255.
            s[S_RG] = 100.0 * 0.1;
            s[S_RG2] = 100.0 * 0.01;
        });
        assert_eq!(st.clip_hi, 0.25);
        assert_eq!(st.clip_hi_new, 0.05);
        assert_eq!(st.clip_lo, 0.1);
        assert_eq!(st.out_of_gamut, 0.01);
        assert!((st.cast_a - 0.01).abs() < 1e-7 && (st.cast_b + 0.02).abs() < 1e-7);
        assert_eq!(st.structure, 0.75);
        assert!((st.local_contrast - 0.05).abs() < 1e-7);
        assert!((st.colorfulness - 0.3 * 0.1 * 255.0).abs() < 1e-3);
        assert_eq!(st.entropy, 0.0);
    }

    #[test]
    fn empty_record_is_harmless() {
        let st = record(|_, _| {});
        assert_eq!(st.luma_hist.len(), 64);
        assert_eq!(st.structure, 1.0);
        assert_eq!(st.luma_mean, 0.0);
    }
}
