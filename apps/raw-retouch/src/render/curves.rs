//! What each slider does. The film-like base curve and the shape of every
//! Basic-panel slider live here on the CPU: [`pack`] turns a parameter set
//! into matrices and two lookup tables, and the kernels only apply them. All
//! tone shapes work in EV relative to middle grey (0.18 scene-linear) after
//! exposure, so they behave the same on every photo and at every exposure.

use std::sync::LazyLock;

use spl_core::{Result, anyhow};

use super::layout::*;
use super::{BASELINE_EV, CameraColor};
use super::{color_mixer, presence, tone_curve};
use crate::types::{PARAM_SPECS, Params};

/// log2(0.18).
pub(crate) const LOG2_GREY: f64 = -2.473_931_188_332_412;
/// White in display EV above middle grey: log2(1 / 0.18).
const WHITE_EV: f64 = -LOG2_GREY;

// Base curve. Log-log slope deep in the shadows and through the midtones
// (Adobe-like default contrast: 0.18 renders at 0.46 sRGB, a stop above grey
// at about 174/255, a stop below at about 78/255), the toe-to-mid hand-over,
// and how far above raw clip at exposure 0 the shoulder lands on white: the
// highlights roll into white with zero slope instead of clipping.
const TOE_SLOPE: f64 = 1.10;
const MID_SLOPE: f64 = 1.25;
const TOE_CENTER: f64 = -2.5;
const TOE_WIDTH: f64 = 1.0;
const WHITE_MARGIN: f64 = 0.4;

// Contrast: log-space S-curve around middle grey (slope gain at grey, and
// the EV scale beyond which it only shifts tones).
const CONTRAST_UP: f64 = 0.40;
const CONTRAST_DOWN: f64 = 0.35;
const CONTRAST_WIDTH: f64 = 2.5;
// Whites and Blacks are image-adaptive, as in Lightroom: they move the
// photo's own bright and dark ends (its 99th and 1st luminance percentiles,
// after exposure, Highlights and Shadows), whatever EV those sit at.
// Whites: EV shift at +-100, full from WHITES_ABOVE below the bright end up,
// fading out WHITES_BELOW below it.
const WHITES_UP: f64 = 1.0;
const WHITES_DOWN: f64 = 1.4;
const WHITES_BELOW: f64 = 2.25;
const WHITES_ABOVE: f64 = 0.5;
// Blacks: EV shift at +-100, full at the dark end, gone BLACKS_ABOVE above it.
const BLACKS_UP: f64 = 1.25;
const BLACKS_DOWN: f64 = 2.0;
const BLACKS_BELOW: f64 = 0.5;
const BLACKS_ABOVE: f64 = 2.5;
// Where the ends may sit (scene EV relative to grey at exposure 0): a photo
// with a hazy floor still gets Blacks in its shadows, not in its midtones.
const DARK_MIN: f64 = -8.0;
const DARK_MAX: f64 = -2.0;
const BRIGHT_MIN: f64 = 0.5;
// Highlights: EV moved at +-100 for bright base regions (logistic around
// HIGHLIGHTS_CENTER EV above grey; about -0.1 EV at grey for -100).
const HIGHLIGHTS_DOWN: f64 = 1.75;
const HIGHLIGHTS_UP: f64 = 1.25;
const HIGHLIGHTS_CENTER: f64 = 2.0;
const HIGHLIGHTS_WIDTH: f64 = 0.7;
// Shadows: EV moved at +-100 for dark base regions, fading out toward the
// midtones and again near the noise floor so blacks stay black.
const SHADOWS_UP: f64 = 1.75;
const SHADOWS_DOWN: f64 = 1.5;
const SHADOWS_CENTER: f64 = -2.25;
const SHADOWS_WIDTH: f64 = 0.75;
const SHADOWS_FLOOR: f64 = -9.0;
const SHADOWS_FLOOR_WIDTH: f64 = 1.0;
/// Vibrance -100 removes this share of the chroma of muted colours.
const VIBRANCE_DOWN: f32 = 0.8;

fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// The neutral (all sliders at zero) tone curve, built once.
pub(crate) struct BaseCurve {
    /// Scene EV (relative to grey) where the curve reaches white.
    pub(crate) white: f64,
    /// Display-linear output, sampled every `FINE_DU` from `FINE_U0` to `white`.
    v: Vec<f64>,
}

const FINE_U0: f64 = -24.0;
const FINE_DU: f64 = 1.0 / 64.0;

pub(crate) static BASE: LazyLock<BaseCurve> = LazyLock::new(BaseCurve::new);

impl BaseCurve {
    fn new() -> Self {
        let white = BASELINE_EV as f64 + WHITE_EV + WHITE_MARGIN;
        let toe =
            |u: f64| TOE_SLOPE + (MID_SLOPE - TOE_SLOPE) * sigmoid((u - TOE_CENTER) / TOE_WIDTH);
        let slope = |u: f64, start: f64| toe(u) * (1.0 - smoothstep(start, white, u));
        let integral = |a: f64, b: f64, start: f64| {
            let n = 512;
            let h = (b - a) / n as f64;
            let mut acc = slope(a, start) + slope(b, start);
            for i in 1..n {
                acc += if i % 2 == 1 { 4.0 } else { 2.0 } * slope(a + i as f64 * h, start);
            }
            acc * h / 3.0
        };
        // Where the shoulder starts so that the curve lands exactly on white.
        let (mut lo, mut hi) = (-2.0, white);
        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            if integral(0.0, white, mid) > WHITE_EV {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        let start = 0.5 * (lo + hi);
        let n = ((white - FINE_U0) / FINE_DU).ceil() as usize + 1;
        let zero = (-FINE_U0 / FINE_DU).round() as usize;
        let mut w = vec![0.0; n];
        for i in (0..zero).rev() {
            let u = FINE_U0 + i as f64 * FINE_DU;
            w[i] = w[i + 1] - integral(u, u + FINE_DU, start);
        }
        for i in zero + 1..n {
            let u = FINE_U0 + i as f64 * FINE_DU;
            w[i] = w[i - 1] + integral(u - FINE_DU, u.min(white), start);
        }
        // Stored display-linear: lookups need no exp2, and at 1/64 EV steps
        // linear interpolation is exact to 2e-5 relative.
        let v = w.iter().map(|&w| (0.18 * w.exp2()).min(1.0)).collect();
        Self { white, v }
    }

    /// Display-linear output (0 to 1) for scene EV `u` relative to grey.
    pub(crate) fn eval(&self, u: f64) -> f64 {
        if u >= self.white {
            return 1.0;
        }
        let f = (u - FINE_U0) / FINE_DU;
        if f <= 0.0 {
            return self.v[0] * (TOE_SLOPE * (u - FINE_U0)).exp2();
        }
        let i = (f as usize).min(self.v.len() - 2);
        let a = f - i as f64;
        self.v[i] + a * (self.v[i + 1] - self.v[i])
    }
}

/// A photo's own dark and bright ends in scene EV relative to grey at
/// exposure 0: where Whites and Blacks act.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Anchors {
    pub(crate) dark: f64,
    pub(crate) bright: f64,
}

impl Anchors {
    /// From the 1st and 99th percentiles of log2 luminance (1 = raw clip).
    pub(crate) fn from_log2(p01: f32, p99: f32) -> Self {
        let ev = |l: f32| l as f64 + BASELINE_EV as f64 - LOG2_GREY;
        Self {
            dark: ev(p01).clamp(DARK_MIN, DARK_MAX),
            bright: ev(p99).clamp(BRIGHT_MIN, BASELINE_EV as f64 + WHITE_EV),
        }
    }
}

fn scale(v: f32, up: f64, down: f64) -> f64 {
    let v = v as f64 / 100.0;
    v * if v >= 0.0 { up } else { down }
}

fn contrast_of(contrast: f64, u: f64) -> f64 {
    u + contrast * CONTRAST_WIDTH * (u / CONTRAST_WIDTH).tanh()
}

/// Contrast: a monotonic remap of scene EV before the base curve. Both run
/// on each pixel's largest and smallest channel, so like Lightroom's
/// Contrast it also makes colours richer (or flatter).
#[derive(Debug, Clone, Copy)]
pub(crate) struct ToneSliders {
    contrast: f64,
}

impl ToneSliders {
    pub(crate) fn new(p: &Params) -> Self {
        Self {
            contrast: scale(p.contrast, CONTRAST_UP, CONTRAST_DOWN),
        }
    }

    pub(crate) fn apply(&self, u: f64) -> f64 {
        contrast_of(self.contrast, u)
    }
}

/// Whites and Blacks: EV added to a pixel by its own luminance (scene EV
/// relative to grey after exposure and Highlights/Shadows), applied to all
/// channels alike. Brightness, not the brightest channel, decides, so a
/// vivid mid-tone colour is not treated as a white.
#[derive(Debug, Clone, Copy)]
pub(crate) struct EndSliders {
    whites: f64,
    blacks: f64,
    white_at: f64,
    black_at: f64,
}

impl EndSliders {
    pub(crate) fn new(p: &Params, anchors: &Anchors, local: &LocalSliders) -> Self {
        // The photo's ends as this parameter set renders them (their base
        // sits at the end, so Highlights/Shadows moved them too).
        let rendered = |a: f64| {
            let u = a + p.exposure as f64;
            u + local.delta(u)
        };
        let mut s = Self {
            whites: scale(p.whites, WHITES_UP, WHITES_DOWN),
            blacks: scale(p.blacks, BLACKS_UP, BLACKS_DOWN),
            white_at: rendered(anchors.bright),
            black_at: 0.0,
        };
        s.black_at = s.whites_of(rendered(anchors.dark));
        s
    }

    fn whites_of(&self, u: f64) -> f64 {
        u + self.whites
            * smoothstep(
                self.white_at - WHITES_BELOW,
                self.white_at + WHITES_ABOVE,
                u,
            )
    }

    pub(crate) fn apply(&self, u: f64) -> f64 {
        let u = self.whites_of(u);
        u + self.blacks
            * (1.0
                - smoothstep(
                    self.black_at - BLACKS_BELOW,
                    self.black_at + BLACKS_ABOVE,
                    u,
                ))
    }

    pub(crate) fn delta(&self, u: f64) -> f64 {
        self.apply(u) - u
    }
}

/// Highlights and Shadows: EV added to a pixel whose base layer sits at
/// `b` EV relative to grey (after exposure).
#[derive(Debug, Clone, Copy)]
pub(crate) struct LocalSliders {
    highlights: f64,
    shadows: f64,
}

impl LocalSliders {
    pub(crate) fn new(p: &Params) -> Self {
        let h = p.highlights as f64 / 100.0;
        let s = p.shadows as f64 / 100.0;
        Self {
            highlights: h * if h >= 0.0 {
                HIGHLIGHTS_UP
            } else {
                HIGHLIGHTS_DOWN
            },
            shadows: s * if s >= 0.0 { SHADOWS_UP } else { SHADOWS_DOWN },
        }
    }

    pub(crate) fn delta(&self, b: f64) -> f64 {
        let hi = sigmoid((b - HIGHLIGHTS_CENTER) / HIGHLIGHTS_WIDTH);
        let lo = sigmoid((SHADOWS_CENTER - b) / SHADOWS_WIDTH)
            * sigmoid((b - SHADOWS_FLOOR) / SHADOWS_FLOOR_WIDTH);
        self.highlights * hi + self.shadows * lo
    }
}

/// Per-photo colour reference: the as-shot white balance as the renderer
/// sees it, and the as-shot camera -> Rec.2020 matrix the base layer uses.
#[derive(Debug, Clone)]
pub(crate) struct Reference {
    pub(crate) t0: f32,
    pub(crate) tint0: f32,
    wb_ref: [f32; 3],
    pub(crate) m0: [[f32; 3]; 3],
}

fn finite3(v: &[f32; 3]) -> bool {
    v.iter().all(|x| x.is_finite())
}

fn finite33(m: &[[f32; 3]; 3]) -> bool {
    m.iter().all(finite3)
}

impl Reference {
    /// The white-balance delta is taken against `wb_for(as-shot)` rather than
    /// the decoder's multipliers, so neutral sliders reproduce the developed
    /// image exactly even where temperature/tint round trips are inexact.
    pub(crate) fn new(
        color: &dyn CameraColor,
        t0: f32,
        tint0: f32,
        wb_as_shot: [f32; 3],
    ) -> Result<Self> {
        let (t0, tint0) = if t0.is_finite() && tint0.is_finite() {
            (t0, tint0)
        } else {
            (5500.0, 0.0)
        };
        let m0 = color.rec2020_from_cam(t0);
        if !finite33(&m0) {
            return Err(anyhow!(
                "camera colour matrix for {t0} K is not finite: {m0:?}"
            ));
        }
        let wb = color.wb_for(t0, tint0);
        let wb_ref = if finite3(&wb) && wb.iter().all(|&x| x > 0.0) {
            wb
        } else if finite3(&wb_as_shot) && wb_as_shot.iter().all(|&x| x > 0.0) {
            wb_as_shot
        } else {
            [1.0; 3]
        };
        Ok(Self {
            t0,
            tint0,
            wb_ref,
            m0,
        })
    }

    /// Rec.2020 luminance of as-shot camera RGB: what the base layer smooths.
    pub(crate) fn luminance_row(&self) -> [f32; 3] {
        const Y: [f32; 3] = [0.262_700_2, 0.677_998_07, 0.059_301_717];
        let m = &self.m0;
        std::array::from_fn(|j| Y[0] * m[0][j] + Y[1] * m[1][j] + Y[2] * m[2][j])
    }

    /// Legal, finite parameters: non-finite values fall back to neutral and
    /// everything is clamped to Lightroom's ranges (widened to include the
    /// as-shot white balance, so neutral is always reachable).
    pub(crate) fn sanitize(&self, p: &Params) -> Params {
        let pick = |v: f32, neutral: f32, lo: f32, hi: f32| {
            if v.is_finite() {
                v.clamp(lo, hi)
            } else {
                neutral
            }
        };
        let slider = |v: f32| pick(v, 0.0, -100.0, 100.0);
        // Beyond the Basic panel: legal range, Lightroom's default for NaN.
        let mut rest = p.to_array();
        for (v, s) in rest.iter_mut().zip(&PARAM_SPECS).skip(Params::BASIC) {
            *v = pick(*v, s.neutral.unwrap_or(0.0), s.min, s.max);
        }
        Params {
            temperature: pick(
                p.temperature,
                self.t0,
                self.t0.min(2000.0),
                self.t0.max(50000.0),
            ),
            tint: pick(
                p.tint,
                self.tint0,
                self.tint0.min(-150.0),
                self.tint0.max(150.0),
            ),
            exposure: pick(p.exposure, 0.0, -5.0, 5.0),
            contrast: slider(p.contrast),
            highlights: slider(p.highlights),
            shadows: slider(p.shadows),
            whites: slider(p.whites),
            blacks: slider(p.blacks),
            vibrance: slider(p.vibrance),
            saturation: slider(p.saturation),
            ..Params::from_array(rest)
        }
    }
}

/// Writes one parameter set in the kernels' layout (`CAND_STRIDE` floats).
pub(crate) fn pack(
    p: &Params,
    color: &dyn CameraColor,
    reference: &Reference,
    anchors: &Anchors,
    out: &mut [f32],
) {
    let p = reference.sanitize(p);
    let out = &mut out[..CAND_STRIDE];
    out.fill(0.0);

    let wb = color.wb_for(p.temperature, p.tint);
    let delta: [f32; 3] = std::array::from_fn(|i| {
        let d = wb[i] / reference.wb_ref[i];
        if d.is_finite() && d > 0.0 { d } else { 1.0 }
    });
    let m = color.rec2020_from_cam(p.temperature);
    let mut m = if finite33(&m) { m } else { reference.m0 };
    color_mixer::calibrate(&p, &mut m);
    // Clipped pixels get one multiplier for all channels, the largest, so
    // they stay neutral and at least as bright as before.
    let neutral = delta.iter().copied().fold(1.0f32, f32::max);
    for i in 0..3 {
        for j in 0..3 {
            out[P_A1 + 3 * i + j] = m[i][j] * delta[j];
            out[P_A2 + 3 * i + j] = m[i][j] * neutral;
        }
    }
    let ev = p.exposure as f64 + BASELINE_EV as f64;
    out[P_GAIN] = ev.exp2() as f32;
    out[P_BSHIFT] = (ev - LOG2_GREY) as f32;
    out[P_VIB] = p.vibrance / 100.0 * if p.vibrance < 0.0 { VIBRANCE_DOWN } else { 1.0 };
    out[P_SAT] = 1.0 + p.saturation / 100.0;

    let local = LocalSliders::new(&p);
    let tone = ToneSliders::new(&p);
    let ends = EndSliders::new(&p, anchors, &local);
    let base = &*BASE;
    let du = (TONE_U1 - TONE_U0) as f64 / (TONE_N - 1) as f64;
    for (i, v) in out[TONE_OFF..TONE_OFF + TONE_N].iter_mut().enumerate() {
        *v = base.eval(tone.apply(TONE_U0 as f64 + i as f64 * du)) as f32;
    }
    let db = (LOCAL_B1 - LOCAL_B0) as f64 / (LOCAL_N - 1) as f64;
    for (i, v) in out[LOCAL_OFF..LOCAL_OFF + LOCAL_N].iter_mut().enumerate() {
        *v = local.delta(LOCAL_B0 as f64 + i as f64 * db) as f32;
    }
    for (i, v) in out[ENDS_OFF..ENDS_OFF + LOCAL_N].iter_mut().enumerate() {
        *v = ends.delta(LOCAL_B0 as f64 + i as f64 * db) as f32;
    }
    presence::pack(&p, &mut out[P_PRES..P_PRES + P_PRES_N]);
    tone_curve::pack(&p, &mut out[CURVE_OFF..CURVE_OFF + CURVE_LEN]);
    color_mixer::pack(&p, &mut out[MIXER_OFF..MIXER_OFF + MIXER_LEN]);
}

/// sRGB encoding of a display-linear value.
#[cfg(test)]
pub(crate) fn oetf(v: f64) -> f64 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.003_130_8 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A photo whose ends sit where Lightroom's default rendering puts them:
    /// raw clip and deep shadow.
    const WIDE: Anchors = Anchors {
        dark: -6.0,
        bright: BASELINE_EV as f64 + WHITE_EV,
    };

    /// What a grey pixel's EV goes through before the base curve: Whites and
    /// Blacks (by luminance), then Contrast.
    struct Chain(EndSliders, ToneSliders);

    impl Chain {
        fn apply(&self, u: f64) -> f64 {
            self.1.apply(self.0.apply(u))
        }
    }

    fn tone(p: &Params, a: &Anchors) -> Chain {
        Chain(
            EndSliders::new(p, a, &LocalSliders::new(p)),
            ToneSliders::new(p),
        )
    }

    /// Scene EV -> sRGB-encoded output for given sliders (grey pixel, no local).
    fn render_ev(p: &Params, u: f64) -> f64 {
        oetf(BASE.eval(tone(p, &WIDE).apply(u)))
    }

    fn neutral() -> Params {
        Params::neutral(5500.0, 0.0)
    }

    fn with(f: impl Fn(&mut Params)) -> Params {
        let mut p = neutral();
        f(&mut p);
        p
    }

    #[test]
    fn base_curve_anchors() {
        let b = &*BASE;
        assert!(
            (oetf(b.eval(0.0)) - 0.46).abs() < 0.01,
            "grey -> {}",
            oetf(b.eval(0.0))
        );
        // Raw clip at exposure 0 is near white but the curve still rolls.
        let clip = BASELINE_EV as f64 + WHITE_EV;
        let at_clip = oetf(b.eval(clip)) * 255.0;
        assert!(at_clip > 253.0 && at_clip < 255.0, "raw clip -> {at_clip}");
        assert_eq!(b.eval(b.white), 1.0);
        assert!(oetf(b.eval(-6.0)) * 255.0 > 3.0 && oetf(b.eval(-6.0)) * 255.0 < 8.0);
        let one_up = oetf(b.eval(1.0)) * 255.0;
        let one_down = oetf(b.eval(-1.0)) * 255.0;
        assert!((165.0..185.0).contains(&one_up), "+1 EV -> {one_up}");
        assert!((70.0..90.0).contains(&one_down), "-1 EV -> {one_down}");
    }

    #[test]
    fn base_curve_is_smooth_and_monotonic() {
        let b = &*BASE;
        let mut prev = 0.0;
        let mut prev_slope = f64::NAN;
        let mut u = -20.0;
        while u < b.white + 1.0 {
            let v = b.eval(u);
            assert!(v >= prev, "not monotonic at {u}");
            // Log-log slope never jumps (no kinks), and lands on zero at white.
            let slope = ((b.eval(u + 0.01).max(1e-12)).log2() - v.max(1e-12).log2()) / 0.01;
            if prev_slope.is_finite() {
                assert!(
                    (slope - prev_slope).abs() < 0.05,
                    "kink at {u}: {prev_slope} -> {slope}"
                );
            }
            prev = v;
            prev_slope = slope;
            u += 0.01;
        }
    }

    #[test]
    fn sliders_at_zero_change_nothing() {
        let t = tone(&neutral(), &WIDE);
        let l = LocalSliders::new(&neutral());
        for i in -200..80 {
            let u = i as f64 * 0.1;
            assert_eq!(t.apply(u), u);
            assert_eq!(l.delta(u), 0.0);
        }
    }

    /// Every tone mapping stays strictly increasing for any slider mix and
    /// any photo, so no setting can invert tones. (The flattest mix,
    /// Contrast -100 with Whites -100 and Blacks +100 on a flat photo, still
    /// keeps a slope above 0.05.)
    #[test]
    fn tone_sliders_never_invert_tones() {
        let vals = [-100.0, -50.0, 0.0, 50.0, 100.0];
        let photos = [
            WIDE,
            Anchors::from_log2(-4.0, -1.0),
            Anchors::from_log2(-12.0, -4.0),
        ];
        for a in photos {
            for c in vals {
                for w in vals {
                    for b in vals {
                        for h in [-100.0, 0.0, 100.0] {
                            let t = tone(
                                &with(|p| {
                                    p.contrast = c;
                                    p.whites = w;
                                    p.blacks = b;
                                    p.highlights = h;
                                    p.shadows = -h;
                                    p.exposure = h / 100.0;
                                }),
                                &a,
                            );
                            let mut prev = f64::NEG_INFINITY;
                            for i in -400..200 {
                                let u = t.apply(i as f64 * 0.05);
                                assert!(
                                    u > prev + 0.05 * 0.05,
                                    "{a:?} c{c} w{w} b{b} h{h} at {}",
                                    i as f64 * 0.05
                                );
                                prev = u;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Blacks and Whites act on the photo's own ends: on a hazy photo whose
    /// darkest tones sit 2.5 EV under grey, Blacks -100 still deepens them a
    /// lot while grey barely moves.
    #[test]
    fn whites_and_blacks_follow_the_photo() {
        let hazy = Anchors::from_log2(-5.5, -1.5);
        let n = neutral();
        let dark_end = hazy.dark;
        let out = |p: &Params, u: f64| oetf(BASE.eval(tone(p, &hazy).apply(u))) * 255.0;
        let crushed = out(&with(|p| p.blacks = -100.0), dark_end);
        assert!(
            crushed < 0.6 * out(&n, dark_end),
            "{crushed} vs {}",
            out(&n, dark_end)
        );
        assert!((out(&with(|p| p.blacks = -100.0), 0.5) - out(&n, 0.5)).abs() < 3.0);
        let bright_end = hazy.bright;
        assert!(out(&with(|p| p.whites = 100.0), bright_end) > out(&n, bright_end) + 15.0);
        // The ends move with exposure: the same tones keep the same treatment.
        let e = with(|p| {
            p.exposure = 1.0;
            p.blacks = -100.0;
        });
        let e0 = with(|p| p.exposure = 1.0);
        assert!(out(&e, dark_end + 1.0) < 0.6 * out(&e0, dark_end + 1.0));
        // Anchors stay inside sane bounds for degenerate photos.
        let black = Anchors::from_log2(-14.0, -14.0);
        assert_eq!((black.dark, black.bright), (DARK_MIN, BRIGHT_MIN));
    }

    /// The base layer's mapping b -> b + delta(b) stays increasing, so
    /// highlights and shadows compress or expand regions without reversing them.
    #[test]
    fn local_sliders_never_invert_tones() {
        let vals = [-100.0, -50.0, 0.0, 50.0, 100.0];
        for h in vals {
            for s in vals {
                let l = LocalSliders::new(&with(|p| {
                    p.highlights = h;
                    p.shadows = s;
                }));
                let mut prev = f64::NEG_INFINITY;
                let mut min_slope = f64::INFINITY;
                for i in -320..160 {
                    let b = i as f64 * 0.05;
                    let m = b + l.delta(b);
                    if prev.is_finite() {
                        min_slope = min_slope.min((m - prev) / 0.05);
                    }
                    prev = m;
                }
                assert!(min_slope > 0.3, "h{h} s{s}: slope {min_slope}");
            }
        }
    }

    #[test]
    fn sliders_move_the_right_way() {
        let out = |p: &Params, u: f64| render_ev(p, u);
        let n = neutral();
        let monotone = |name: &str, f: &dyn Fn(f32) -> Params, u: f64, up: bool| {
            let mut prev = if up { -1.0 } else { 2.0 };
            for v in [-100.0, -50.0, 0.0, 50.0, 100.0] {
                let o = out(&f(v), u);
                if up {
                    assert!(o >= prev, "{name} not increasing at {u} EV");
                } else {
                    assert!(o <= prev, "{name} not decreasing at {u} EV");
                }
                prev = o;
            }
        };
        // Contrast: darker shadows and brighter highlights, grey fixed.
        monotone("contrast", &|v| with(|p| p.contrast = v), 1.0, true);
        monotone("contrast", &|v| with(|p| p.contrast = v), -2.0, false);
        let c = with(|p| p.contrast = 100.0);
        assert!((out(&c, 0.0) - out(&n, 0.0)).abs() < 1e-9);
        // Whites move the top, barely the midtones; Blacks move the bottom.
        monotone("whites", &|v| with(|p| p.whites = v), 2.5, true);
        let w = with(|p| p.whites = 100.0);
        assert!((out(&w, 0.0) - out(&n, 0.0)).abs() < 0.002);
        assert!(out(&with(|p| p.whites = -100.0), BASE.white) < 0.95);
        monotone("blacks", &|v| with(|p| p.blacks = v), -5.0, true);
        assert!((out(&with(|p| p.blacks = -100.0), 0.0) - out(&n, 0.0)).abs() < 0.002);
        assert!(out(&with(|p| p.blacks = -100.0), -6.0) * 255.0 < 2.5);

        // Highlights pull bright regions down by up to about 1.5-2 EV and
        // leave grey nearly alone; Shadows lift dark regions by up to ~1.75 EV.
        let l = |p: Params, b: f64| LocalSliders::new(&p).delta(b);
        let hl = with(|p| p.highlights = -100.0);
        assert!(l(hl, 3.5) < -1.5 && l(hl, 3.5) > -2.0);
        assert!(l(hl, 0.0).abs() < 0.12);
        assert!(l(with(|p| p.highlights = 100.0), 3.0) > 0.8);
        let sh = with(|p| p.shadows = 100.0);
        assert!(l(sh, -4.0) > 1.5 && l(sh, -4.0) < 2.0);
        assert!(l(sh, 0.0) < 0.1);
        assert!(l(sh, -12.0) < 0.1, "noise floor stays put");
        assert!(l(with(|p| p.shadows = -100.0), -4.0) < -1.2);
    }

    struct Fixed;

    impl CameraColor for Fixed {
        fn rec2020_from_cam(&self, temperature: f32) -> [[f32; 3]; 3] {
            let k = temperature / 5500.0;
            [
                [1.2, -0.1, -0.1],
                [-0.05, 1.1 * k, -0.05 * k],
                [0.0, -0.2, 1.2],
            ]
        }
        fn wb_for(&self, temperature: f32, tint: f32) -> [f32; 3] {
            [
                2.0 * temperature / 5500.0,
                1.0,
                1.5 * 5500.0 / temperature + tint / 1000.0,
            ]
        }
    }

    #[test]
    fn neutral_pack_is_the_as_shot_render() {
        let r = Reference::new(&Fixed, 5000.0, 3.0, [2.0, 1.0, 1.6]).unwrap();
        let mut out = vec![0.0; CAND_STRIDE];
        pack(&Params::neutral(5000.0, 3.0), &Fixed, &r, &WIDE, &mut out);
        let m = Fixed.rec2020_from_cam(5000.0);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(out[P_A1 + 3 * i + j], m[i][j]);
                assert_eq!(out[P_A2 + 3 * i + j], m[i][j]);
            }
        }
        assert!((out[P_GAIN] - BASELINE_EV.exp2()).abs() < 1e-6);
        assert_eq!(out[P_VIB], 0.0);
        assert_eq!(out[P_SAT], 1.0);
        assert!(
            out[LOCAL_OFF..LOCAL_OFF + LOCAL_N]
                .iter()
                .chain(&out[ENDS_OFF..ENDS_OFF + LOCAL_N])
                .all(|&v| v == 0.0)
        );
        let du = (TONE_U1 - TONE_U0) as f64 / (TONE_N - 1) as f64;
        for (i, &v) in out[TONE_OFF..TONE_OFF + TONE_N].iter().enumerate() {
            assert_eq!(v, BASE.eval(TONE_U0 as f64 + i as f64 * du) as f32);
        }

        // Garbage in, legal render out.
        let mut bad = Params::neutral(f32::NAN, f32::INFINITY);
        bad.exposure = 99.0;
        bad.contrast = f32::NAN;
        pack(&bad, &Fixed, &r, &WIDE, &mut out);
        assert!(out.iter().all(|v| v.is_finite()));
        assert!((out[P_GAIN] - (5.0 + BASELINE_EV).exp2()).abs() < 1e-3);
    }

    #[test]
    fn warmer_temperature_boosts_red_relative_to_blue() {
        let r = Reference::new(&Fixed, 5000.0, 0.0, [2.0, 1.0, 1.6]).unwrap();
        let mut out = vec![0.0; CAND_STRIDE];
        pack(&Params::neutral(6500.0, 0.0), &Fixed, &r, &WIDE, &mut out);
        let m = Fixed.rec2020_from_cam(6500.0);
        // delta red > 1 > delta blue for the Fixed model's warmer setting.
        assert!(out[P_A1] / m[0][0] > 1.0);
        assert!(out[P_A1 + 8] / m[2][2] < 1.0);
        // Clipped pixels get one multiplier for all channels.
        let k = out[P_A2] / m[0][0];
        assert!((out[P_A2 + 4] / m[1][1] - k).abs() < 1e-5);
    }
}
