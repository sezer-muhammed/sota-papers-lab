//! A deterministic "Auto" first guess from the neutral render's statistics,
//! the way Lightroom's Auto button works. Owner: search.
//!
//! Closed-form rules, not a search: exposure moves the median toward a
//! midtone, highlights and shadows recover what sits near either end,
//! whites and blacks stretch the histogram, colour gets a mild vibrance
//! boost and white balance stays as shot unless the cast is strong. Auto
//! seeds the search, centres its over-editing penalty and is always among
//! the finalists, so it aims to be safe rather than bold.

use crate::objective::{adapted_p50, darkness, luma_quantile, noise_factor, smoothstep};
use crate::search::{bounds, clamp_params, sane_temperature, warmed};
use crate::types::{Params, Scene, Stats};

/// Median luma auto aims for, about where Lightroom Auto lands.
const TARGET_P50: f32 = 0.45;
/// EV per doubling of median luma: roughly the inverse of a base tone
/// curve's midtone slope (log-log), damped so auto undershoots rather than
/// overshoots; the search finishes the job.
const EXPOSURE_GAIN: f32 = 1.5;
/// How much of an exposure change (in log2 units) reaches the top and the
/// bottom of the histogram: tone curves compress both ends.
const SLOPE_HIGH: f32 = 0.35;
const SLOPE_LOW: f32 = 0.6;
/// Highlights run from this (always a little recovery, as experts do)...
const HIGHLIGHTS_MIN: f32 = -10.0;
/// ...down by this much more when p99 nears white or new clipping shows.
const HIGHLIGHTS_RANGE: f32 = 60.0;
/// New clipping at which highlights recovery is at full strength.
const CLIP_FULL: f32 = 0.01;
/// Pixels blown in the RAW render white whatever the sliders do, so the
/// top of the histogram they fill says nothing about what can be
/// recovered: above this share, highlights recovery reads the brightest
/// percentile below them instead of p99...
const RAW_CLIPPED_SOME: f32 = 0.002;
/// ...and above this one, whites never pull the blown area down to grey.
const RAW_CLIPPED_TOP: f32 = 0.01;
/// Luma change at p99 per highlights unit, to predict where whites land.
const HIGHLIGHTS_EFFECT: f32 = 0.001;
const SHADOWS_MIN: f32 = 5.0;
const SHADOWS_RANGE: f32 = 45.0;
/// Shadow lift kept on a high-ISO file (noise lifts with it).
const SHADOWS_NOISY: f32 = 0.5;
const WHITE_POINT: f32 = 0.97;
/// Whites lift the white end at most this far above where exposure puts
/// it (beyond what Highlights takes off): a
/// blue sky or a grey backdrop that tops the histogram clips a channel
/// long before its luma reaches white.
const WHITES_STRETCH: f32 = 0.06;
const WHITES_GAIN: f32 = 250.0;
const BLACK_POINT: f32 = 0.03;
/// Night scenes may keep true blacks.
const BLACK_POINT_DARK: f32 = 0.01;
const BLACKS_GAIN: f32 = 350.0;
/// Up to 15 contrast on a flat (low p05..p95 spread) render.
const CONTRAST_MAX: f32 = 15.0;
const CONTRAST_SPREAD: (f32, f32) = (0.80, 0.55);
/// Vibrance 25 on dull scenes down to 15 on colourful ones; saturation 5 to 0.
const VIBRANCE: (f32, f32) = (25.0, 15.0);
const SATURATION_MAX: f32 = 5.0;
const COLORFUL: (f32, f32) = (15.0, 50.0);
const CHROMA_HOT: (f32, f32) = (0.15, 0.20);
/// Casts (Oklab a/b of near-neutral pixels) beyond these count as strong.
const CAST_TOL_A: f32 = 0.015;
const CAST_TOL_WARM: f32 = 0.03;
const CAST_TOL_COOL: f32 = 0.015;
/// Warm shift (mired) per unit of Oklab b, and tint per unit of Oklab a;
/// only `WB_SHARE` of a strong cast's excess is corrected.
const MIRED_PER_CAST_B: f32 = 2000.0;
const TINT_PER_CAST_A: f32 = 1500.0;
const WB_SHARE: f32 = 0.5;

/// Auto for this photo, inside its search box.
pub fn auto_params(neutral: &Stats, scene: &Scene) -> Params {
    let guess = auto_from_stats(
        neutral,
        scene.info.as_shot_temperature,
        scene.info.as_shot_tint,
        scene.info.iso,
    );
    let (lo, hi) = bounds(scene);
    clamp_params(&guess, &lo, &hi)
}

/// Auto from the neutral render's stats and the as-shot white balance,
/// within Lightroom's legal ranges but not yet clamped to a photo's search
/// box (the objective uses it this way as the over-editing baseline).
pub fn auto_from_stats(
    neutral: &Stats,
    as_shot_temperature: f32,
    as_shot_tint: f32,
    iso: Option<u32>,
) -> Params {
    let or = |x: f32, fallback: f32| if x.is_finite() { x } else { fallback };
    let get = |x: f32, fallback: f32| or(x, fallback).clamp(0.0, 1.0);
    let p01 = get(neutral.luma_p01, 0.02);
    let p05 = get(neutral.luma_p05, 0.06).max(p01);
    let p50 = get(neutral.luma_p50, 0.42).max(p05);
    let p95 = get(neutral.luma_p95, 0.88).max(p50);
    let p99 = get(neutral.luma_p99, 0.96).max(p95);
    let clip_new = get(neutral.clip_hi_new, 0.0);
    // Pixels clipped at neutral that were not newly clipped: blown in the RAW.
    let raw_share = (get(neutral.clip_hi, 0.0) - clip_new).max(0.0);
    let dark = darkness(p50);

    let target = adapted_p50(TARGET_P50, p50);
    let exposure = EXPOSURE_GAIN * (target / p50.max(0.02)).log2();
    let moved = |p: f32, slope: f32| (p.max(1e-4) * (exposure * slope).exp2()).min(1.0);
    let top = if raw_share > RAW_CLIPPED_SOME {
        luma_quantile(neutral, (0.99 - raw_share).clamp(0.90, 0.99)).unwrap_or(p95)
    } else {
        p99
    };
    let p99e = moved(p99, SLOPE_HIGH);
    let p05e = moved(p05, SLOPE_LOW);
    let p01e = moved(p01, SLOPE_LOW);

    let near_white = smoothstep(0.90, 1.0, moved(top, SLOPE_HIGH));
    let recover = near_white.max((clip_new / CLIP_FULL).min(1.0));
    let highlights = HIGHLIGHTS_MIN - HIGHLIGHTS_RANGE * recover;
    let lift = smoothstep(0.12, 0.02, p05e);
    let noise = noise_factor(iso);
    let shadows = (SHADOWS_MIN + SHADOWS_RANGE * lift) * (1.0 - (1.0 - SHADOWS_NOISY) * noise);

    let p99h = p99e + HIGHLIGHTS_EFFECT * highlights;
    let white_point = WHITE_POINT.min(p99e + WHITES_STRETCH);
    let mut whites = (WHITES_GAIN * (white_point - p99h)).clamp(-30.0, 40.0);
    if raw_share >= RAW_CLIPPED_TOP {
        whites = whites.max(0.0);
    }
    let black_point = BLACK_POINT + (BLACK_POINT_DARK - BLACK_POINT) * dark;
    let mut blacks = (BLACKS_GAIN * (black_point - p01e)).clamp(-30.0, 25.0);
    if blacks > 0.0 {
        blacks *= 1.0 - dark;
    }
    let spread = p95 - p05;
    let contrast = CONTRAST_MAX * smoothstep(CONTRAST_SPREAD.0, CONTRAST_SPREAD.1, spread);

    let colorful = ((or(neutral.colorfulness, 30.0) - COLORFUL.0) / (COLORFUL.1 - COLORFUL.0))
        .clamp(0.0, 1.0)
        .max(smoothstep(
            CHROMA_HOT.0,
            CHROMA_HOT.1,
            or(neutral.chroma_p95, 0.1),
        ));
    let vibrance = VIBRANCE.0 + (VIBRANCE.1 - VIBRANCE.0) * colorful;
    let saturation = SATURATION_MAX * (1.0 - colorful);

    // Correct only part of a strong cast; warm casts get the most room.
    let (cast_a, cast_b) = (or(neutral.cast_a, 0.0), or(neutral.cast_b, 0.0));
    let excess_b = (cast_b - CAST_TOL_WARM).max(0.0) - (-cast_b - CAST_TOL_COOL).max(0.0);
    let excess_a = cast_a.signum() * (cast_a.abs() - CAST_TOL_A).max(0.0);
    let as_shot_temperature = sane_temperature(as_shot_temperature);
    let as_shot_tint = or(as_shot_tint, 0.0);
    let temperature = warmed(as_shot_temperature, -WB_SHARE * MIRED_PER_CAST_B * excess_b);
    let tint = as_shot_tint - WB_SHARE * TINT_PER_CAST_A * excess_a;

    legal(Params {
        temperature,
        tint,
        exposure,
        contrast,
        highlights,
        shadows,
        whites,
        blacks,
        vibrance,
        saturation,
        ..Params::neutral(temperature, tint)
    })
}

/// Clamped to Lightroom's ranges but not rounded: the search quantizes
/// whatever it renders.
fn legal(p: Params) -> Params {
    p.clamped()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::warm_shift;

    fn stats(p01: f32, p05: f32, p50: f32, p95: f32, p99: f32) -> Stats {
        Stats {
            luma_p01: p01,
            luma_p05: p05,
            luma_p50: p50,
            luma_p95: p95,
            luma_p99: p99,
            colorfulness: 30.0,
            chroma_p95: 0.12,
            ..Stats::default()
        }
    }

    fn auto(s: &Stats) -> Params {
        auto_from_stats(s, 5200.0, 8.0, Some(200))
    }

    #[test]
    fn exposure_moves_the_median_toward_a_midtone() {
        let normal = auto(&stats(0.03, 0.08, 0.44, 0.85, 0.93));
        assert!(normal.exposure.abs() < 0.15, "{normal:?}");
        let dark = auto(&stats(0.01, 0.03, 0.22, 0.6, 0.75));
        assert!(dark.exposure > 0.8 && dark.exposure < 2.0, "{dark:?}");
        // A high-key scene keeps its key (the camera meters it bright on
        // purpose); highlights recovery handles the top.
        let bright = auto(&stats(0.2, 0.35, 0.72, 0.97, 0.995));
        assert!(
            bright.exposure < 0.05 && bright.exposure > -0.3,
            "{bright:?}"
        );
        assert!(bright.highlights < -50.0, "{bright:?}");
        let overexposed = auto(&stats(0.15, 0.3, 0.62, 0.95, 0.99));
        assert!(overexposed.exposure < -0.2, "{overexposed:?}");
        // A night scene is lifted, but not to a daylight midtone.
        let night = auto(&stats(0.0, 0.01, 0.07, 0.4, 0.9));
        assert!(night.exposure > 1.0 && night.exposure < 3.0, "{night:?}");
    }

    #[test]
    fn highlights_recover_near_clipping() {
        let calm = auto(&stats(0.03, 0.08, 0.44, 0.75, 0.82));
        assert!((-20.0..=-10.0).contains(&calm.highlights), "{calm:?}");
        let hot = auto(&stats(0.03, 0.08, 0.44, 0.95, 0.995));
        assert!(
            (-70.0..=-20.0).contains(&hot.highlights) && hot.highlights < -50.0,
            "{hot:?}"
        );
        let mut clipping = stats(0.03, 0.08, 0.44, 0.8, 0.85);
        clipping.clip_hi_new = 0.02;
        clipping.clip_hi = 0.02;
        let clipped = auto(&clipping);
        assert!(clipped.highlights <= -69.0, "{clipped:?}");
    }

    #[test]
    fn blown_raw_highlights_are_not_pulled_to_grey() {
        // 3 % blown in the RAW: p99 is white whatever happens.
        let mut blown = stats(0.03, 0.08, 0.44, 0.75, 1.0);
        blown.clip_hi = 0.03;
        let b = auto(&blown);
        assert!(b.highlights > -20.0, "{b:?}");
        assert!(b.whites >= 0.0, "{b:?}");
        // The same p99 from recoverable highlights gets full recovery.
        let recoverable = auto(&stats(0.03, 0.08, 0.44, 0.75, 1.0));
        assert!(recoverable.highlights < -60.0, "{recoverable:?}");
        // 2 % blown, but bright recoverable clouds just below them: recover
        // those (read from the histogram), without greying the blown part.
        let mut clouds = blown.clone();
        clouds.clip_hi = 0.02;
        clouds.luma_hist = vec![0.0; 64];
        clouds.luma_hist[20] = 0.93;
        clouds.luma_hist[61] = 0.05;
        clouds.luma_hist[63] = 0.02;
        let c = auto(&clouds);
        assert!(c.highlights < -50.0 && c.whites >= 0.0, "{c:?}");
    }

    #[test]
    fn shadows_lift_when_p05_is_low() {
        let open = auto(&stats(0.08, 0.15, 0.45, 0.85, 0.94));
        assert!(open.shadows < 10.0, "{open:?}");
        let deep = auto(&stats(0.0, 0.015, 0.42, 0.85, 0.94));
        assert!((40.0..=50.0).contains(&deep.shadows), "{deep:?}");
        let noisy = auto_from_stats(
            &stats(0.0, 0.015, 0.42, 0.85, 0.94),
            5200.0,
            8.0,
            Some(12800),
        );
        assert!(noisy.shadows < deep.shadows * 0.6, "{noisy:?}");
    }

    #[test]
    fn whites_and_blacks_stretch_the_histogram() {
        let flat = auto(&stats(0.10, 0.2, 0.45, 0.7, 0.8));
        assert!(flat.whites > 10.0 && flat.blacks < -10.0, "{flat:?}");
        assert!(flat.contrast > 10.0, "{flat:?}");
        let punchy = auto(&stats(0.0, 0.02, 0.45, 0.97, 1.0));
        assert!(punchy.contrast < 1.0, "{punchy:?}");
        assert!(punchy.blacks > 0.0, "{punchy:?}");
    }

    #[test]
    fn colour_and_white_balance_stay_gentle() {
        let base = auto(&stats(0.03, 0.08, 0.44, 0.85, 0.93));
        assert!((15.0..=25.0).contains(&base.vibrance));
        assert!((0.0..=5.0).contains(&base.saturation));
        assert_eq!(base.temperature, 5200.0);
        assert_eq!(base.tint, 8.0);
        let mut loud = stats(0.03, 0.08, 0.44, 0.85, 0.93);
        loud.colorfulness = 70.0;
        loud.chroma_p95 = 0.22;
        let loud = auto(&loud);
        assert_eq!((loud.vibrance, loud.saturation), (15.0, 0.0));

        // A strong yellow cast is partly corrected toward cooler...
        let mut yellow = stats(0.03, 0.08, 0.44, 0.85, 0.93);
        yellow.cast_b = 0.05;
        let y = auto(&yellow);
        let shift = warm_shift(5200.0, y.temperature);
        assert!(shift < -10.0 && shift > -40.0, "{shift}");
        // ...a mild warm one is kept, and a green cast moves tint to magenta.
        yellow.cast_b = 0.02;
        assert_eq!(auto(&yellow).temperature, 5200.0);
        let mut green = stats(0.03, 0.08, 0.44, 0.85, 0.93);
        green.cast_a = -0.03;
        assert!(auto(&green).tint > 15.0);
    }

    #[test]
    fn broken_stats_still_give_legal_params() {
        let mut s = stats(f32::NAN, 0.0, f32::NAN, 2.0, -1.0);
        s.colorfulness = f32::INFINITY;
        let p = auto_from_stats(&s, f32::NAN, f32::NAN, Some(0));
        assert!(p.to_array().iter().all(|v| v.is_finite()), "{p:?}");
        assert_eq!(p, legal(p));
        let black = auto(&Stats::default());
        assert!(black.exposure > 0.0 && black.exposure <= 5.0);
    }
}
