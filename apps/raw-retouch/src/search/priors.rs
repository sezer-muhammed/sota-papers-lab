//! Expert priors for the settings beyond the Basic panel: the search box and
//! how far an edit may stray from a look's typical value, per slider, in
//! Lightroom units. Owner: search.
//!
//! The numbers come from expert XMPs (`research_notes/Automatic RAW photo
//! retouching/gap_expert_slider_priors.md`; per-band statistics recomputed
//! from PPR10K's 3 x 11,161 target XMPs):
//!
//! - **Tone Curve, parametric.** PPR10K's experts (PV2012) almost never use
//!   it (0.1 to 0.4 % of photos): the Basic tone sliders do that job. FiveK's
//!   (PV2003, no Highlights/Shadows/Whites sliders) use it on about half
//!   their photos as an S-curve, Shadows -14 to -23, Darks -7 to -20, Lights
//!   +12 to +19, Highlights +10 to +41, standard deviations 13 to 38;
//!   VeraRetouch's table gives 7 to 16. So: centred on the look (zero for
//!   natural), a box of +-30, and standard deviations of 8 to 10 that keep
//!   it from doing the Basic sliders' work.
//! - **Presence.** Clarity: expert a +4.4 (p50 +5, std 1.8) on 87 % of
//!   photos, b +0.9 (p95 +7, p99 +13), c almost never. Texture: c +1.5 (0 to
//!   3) on 80 % of the photos that carry it. Dehaze: under 1 % of photos,
//!   about +11 to +17 when used. Small positive typical values, narrow
//!   priors; dehaze only on hazy scenes (`auto`).
//! - **Color Mixer.** Used on 94 to 96 % of PPR10K's portraits, but as
//!   small per-band moves, mostly on skin: orange saturation -5 to -15 and
//!   luminance +7 to +33, yellow saturation -5 to -18 and hue -12, green hue
//!   +6; other bands' standard deviations are 2 to 15 with medians of zero.
//!   Portrait habits do not transfer to other scenes, so `auto` applies the
//!   skin move only where skin-like colours are, scaled down.
//! - **Calibration.** 55 to 88 % of PPR10K's photos, nearly all as a
//!   saturation boost: Blue Primary Saturation +44 (a, mode +50), +5 (b),
//!   +23 (c); Red +8 and Green +14 (c); hues within a few units; Shadows
//!   Tint almost never. Small boxes, zero typical for `auto`, the boost kept
//!   for the vivid look.
//!
//! The search varies what the render statistics can judge. The Color
//! Mixer's hues, Shadows Tint and the primaries' hues are not searched
//! (`searched: false` in `PARAM_SPECS`, so `search::bounds` holds them at
//! zero): no statistic sees a hue move, and a guessed one risks skin. Their
//! rows here stay narrow (+-10 on the skin bands, +-15 elsewhere; bands sit
//! 35 to 55 degrees apart and +-100 moves a colour about one band, so no
//! setting inside them could carry one band's colours past its neighbour's)
//! for whoever turns them on.

use crate::types::Params;

/// One slider's prior.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SliderPrior {
    pub(crate) key: &'static str,
    /// The search box in Lightroom units, on Lightroom's step grid, around 0.
    pub(crate) lo: f32,
    pub(crate) hi: f32,
    /// Distance from the look's typical value that costs one unit of prior.
    pub(crate) std: f32,
}

const fn prior(key: &'static str, lo: f32, hi: f32, std: f32) -> SliderPrior {
    SliderPrior { key, lo, hi, std }
}

/// Indices into [`Params::to_array`] (checked against [`PARAM_SPECS`] by a test).
pub(crate) const CURVE_HIGHLIGHTS: usize = 10;
pub(crate) const CURVE_LIGHTS: usize = 11;
pub(crate) const CURVE_DARKS: usize = 12;
pub(crate) const CURVE_SHADOWS: usize = 13;
pub(crate) const TEXTURE: usize = 14;
pub(crate) const CLARITY: usize = 15;
pub(crate) const DEHAZE: usize = 16;
/// First of the eight Color Mixer hue, saturation and luminance sliders, in
/// band order (red, orange, yellow, green, aqua, blue, purple, magenta).
pub(crate) const HUE: usize = 17;
pub(crate) const SAT: usize = 25;
pub(crate) const LUM: usize = 33;
#[allow(dead_code, reason = "not searched; tests check the table by it")]
pub(crate) const SHADOW_TINT: usize = 41;
pub(crate) const RED_HUE: usize = 42;
pub(crate) const RED_SAT: usize = 43;
#[allow(dead_code, reason = "not searched; tests check the table by it")]
pub(crate) const GREEN_HUE: usize = 44;
pub(crate) const GREEN_SAT: usize = 45;
#[allow(dead_code, reason = "not searched; tests check the table by it")]
pub(crate) const BLUE_HUE: usize = 46;
pub(crate) const BLUE_SAT: usize = 47;
/// The first field with a prior, and one past the last: everything between
/// the Basic panel and Detail.
pub(crate) const FIRST: usize = Params::BASIC;
pub(crate) const END: usize = 48;
pub(crate) const N: usize = END - FIRST;

/// In [`PARAM_SPECS`] order from [`FIRST`].
pub(crate) const PRIORS: [SliderPrior; N] = [
    // Tone Curve.
    prior("curve_highlights", -30.0, 30.0, 10.0),
    prior("curve_lights", -30.0, 30.0, 8.0),
    prior("curve_darks", -30.0, 30.0, 10.0),
    prior("curve_shadows", -30.0, 30.0, 8.0),
    // Presence.
    prior("texture", -20.0, 30.0, 6.0),
    prior("clarity", -25.0, 40.0, 7.0),
    prior("dehaze", -15.0, 35.0, 5.0),
    // Color Mixer hue: red, orange, yellow, green, aqua, blue, purple, magenta.
    prior("hue_red", -10.0, 10.0, 4.0),
    prior("hue_orange", -10.0, 10.0, 4.0),
    prior("hue_yellow", -10.0, 10.0, 5.0),
    prior("hue_green", -15.0, 15.0, 5.0),
    prior("hue_aqua", -15.0, 15.0, 5.0),
    prior("hue_blue", -15.0, 15.0, 5.0),
    prior("hue_purple", -15.0, 15.0, 5.0),
    prior("hue_magenta", -15.0, 15.0, 5.0),
    // Saturation.
    prior("saturation_red", -25.0, 25.0, 6.0),
    prior("saturation_orange", -30.0, 15.0, 6.0),
    prior("saturation_yellow", -35.0, 25.0, 10.0),
    prior("saturation_green", -35.0, 30.0, 10.0),
    prior("saturation_aqua", -30.0, 30.0, 8.0),
    prior("saturation_blue", -30.0, 35.0, 10.0),
    prior("saturation_purple", -40.0, 25.0, 10.0),
    prior("saturation_magenta", -40.0, 25.0, 10.0),
    // Luminance.
    prior("luminance_red", -20.0, 20.0, 6.0),
    prior("luminance_orange", -15.0, 30.0, 8.0),
    prior("luminance_yellow", -25.0, 30.0, 8.0),
    prior("luminance_green", -30.0, 25.0, 8.0),
    prior("luminance_aqua", -25.0, 25.0, 6.0),
    prior("luminance_blue", -35.0, 20.0, 8.0),
    prior("luminance_purple", -20.0, 20.0, 6.0),
    prior("luminance_magenta", -20.0, 20.0, 6.0),
    // Calibration.
    prior("shadow_tint", -10.0, 10.0, 3.0),
    prior("red_primary_hue", -10.0, 10.0, 4.0),
    prior("red_primary_saturation", -15.0, 25.0, 6.0),
    prior("green_primary_hue", -10.0, 15.0, 4.0),
    prior("green_primary_saturation", -15.0, 30.0, 8.0),
    prior("blue_primary_hue", -15.0, 10.0, 4.0),
    prior("blue_primary_saturation", -15.0, 50.0, 10.0),
];

/// Squared distance of `p` from `centre` over the fields with a prior, in
/// units of their standard deviation, through a Huber so a far setting
/// pulls back steadily: 1 per standard deviation squared up to two of
/// them, linear beyond.
pub(crate) fn cost(p: &[f32; Params::COUNT], centre: &[f32; Params::COUNT]) -> f32 {
    PRIORS
        .iter()
        .enumerate()
        .map(|(k, s)| {
            let d = ((p[FIRST + k] - centre[FIRST + k]) / s.std).abs();
            if d <= 2.0 { d * d } else { 4.0 * d - 4.0 }
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::PARAM_SPECS;

    #[test]
    fn priors_follow_the_spec_table() {
        assert_eq!(PARAM_SPECS[END].key, "sharpening", "Detail follows");
        for (k, s) in PRIORS.iter().enumerate() {
            let spec = &PARAM_SPECS[FIRST + k];
            assert_eq!(s.key, spec.key);
            assert!(spec.min <= s.lo && s.lo <= 0.0 && 0.0 <= s.hi && s.hi <= spec.max);
            assert_eq!(spec.quantize(s.lo), s.lo);
            assert_eq!(spec.quantize(s.hi), s.hi);
            assert!(s.std > 0.0 && s.std < s.hi - s.lo);
        }
        let key = |i: usize| PARAM_SPECS[i].key;
        assert_eq!(key(CURVE_HIGHLIGHTS), "curve_highlights");
        assert_eq!(key(CURVE_LIGHTS), "curve_lights");
        assert_eq!(key(CURVE_DARKS), "curve_darks");
        assert_eq!(key(CURVE_SHADOWS), "curve_shadows");
        assert_eq!(
            (key(TEXTURE), key(CLARITY), key(DEHAZE)),
            ("texture", "clarity", "dehaze")
        );
        assert_eq!((key(HUE), key(HUE + 7)), ("hue_red", "hue_magenta"));
        assert_eq!(
            (key(SAT), key(SAT + 7)),
            ("saturation_red", "saturation_magenta")
        );
        assert_eq!(
            (key(LUM), key(LUM + 7)),
            ("luminance_red", "luminance_magenta")
        );
        assert_eq!(key(SHADOW_TINT), "shadow_tint");
        assert_eq!(
            (key(RED_HUE), key(RED_SAT)),
            ("red_primary_hue", "red_primary_saturation")
        );
        assert_eq!(
            (key(GREEN_HUE), key(GREEN_SAT)),
            ("green_primary_hue", "green_primary_saturation")
        );
        assert_eq!(
            (key(BLUE_HUE), key(BLUE_SAT)),
            ("blue_primary_hue", "blue_primary_saturation")
        );
        // The search varies the Basic panel and every field with a prior
        // but the hues: the Color Mixer's, Shadows Tint and the primaries'
        // (and, in the black-and-white look only, the gray mixer).
        let hues = [SHADOW_TINT, RED_HUE, GREEN_HUE, BLUE_HUE];
        for (i, spec) in PARAM_SPECS.iter().enumerate() {
            let hue = (HUE..SAT).contains(&i) || hues.contains(&i);
            let gray = spec.key.starts_with("gray_");
            assert_eq!(
                spec.searched,
                i < Params::BASIC || ((FIRST..END).contains(&i) && !hue) || gray,
                "{}",
                spec.key
            );
        }
    }

    #[test]
    fn cost_is_quadratic_near_and_linear_far() {
        let centre = Params::neutral(5000.0, 0.0).to_array();
        let mut p = centre;
        assert_eq!(cost(&p, &centre), 0.0);
        p[CLARITY] = PRIORS[CLARITY - FIRST].std;
        assert!((cost(&p, &centre) - 1.0).abs() < 1e-6);
        p[CLARITY] *= 2.0;
        assert!((cost(&p, &centre) - 4.0).abs() < 1e-5);
        p[CLARITY] *= 2.0;
        assert!((cost(&p, &centre) - 12.0).abs() < 1e-4);
        // Basic sliders, Detail and Lens are not counted.
        let mut q = centre;
        q[2] = 3.0;
        q[48] = 150.0;
        assert_eq!(cost(&q, &centre), 0.0);
    }
}
