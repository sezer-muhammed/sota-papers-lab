//! Color Mixer (HSL) and Calibration: packs the candidate's MIXER region for
//! `cuda/color_mixer.cu`. Owner: colour mixer.
//!
//! [`pack`] is called by `curves::pack` for every parameter set (thousands
//! per photo, on several threads), so it must stay cheap and write its whole
//! region: the buffer is reused.
//!
//! The Color Mixer works on Oklab hue, chroma and lightness. Its eight bands
//! sit at the Oklab hues of sRGB's red, orange, yellow, green, aqua, blue,
//! purple and magenta ([`BAND_HUES`]) and blend smoothly into their
//! neighbours, so every hue answers to one or two bands and setting all
//! eight alike acts on every hue alike. Per band:
//! - Hue moves the band's colours toward the next band (+) or the previous
//!   one (-), at +-100 by 60% of the way, at most 30 degrees. Bands pushed
//!   toward each other keep a gap, and the shift is smoothed over hue, so
//!   the hue map never folds over (no hue reversals or banding).
//! - Saturation scales chroma: 0 at -100, twice at +100.
//! - Luminance scales Oklab lightness by up to 2x (+100) or 1/2x (-100) at
//!   black, rolling off to no change at white, weighted by chroma so greys
//!   never change.
//!
//! Every change fades in with chroma (nearly fully on by an Oklab chroma of
//! 0.05 for hue and saturation and 0.1 for luminance; skin, foliage and sky
//! sit around 0.05 to 0.15), so near neutrals (noise, a white wall's or a
//! grey card's faint cast) keep their colour and tone. The kernel only
//! interpolates tables over hue that [`tables`] bakes here; [`apply`] is the
//! same per-pixel step on the CPU, for the tests. With every Color Mixer
//! slider at 0 the kernel skips it, so neutral renders cost nothing extra
//! and stay bit-identical.
//!
//! Calibration's red, green and blue primaries change the camera -> working
//! space matrix ([`calibrate`], called by `curves::pack` before white
//! balance is folded in), as Lightroom's Calibration panel moves its
//! profile's primaries: each Rec.2020 primary is turned in hue and scaled in
//! saturation, less that change in proportion to lightness, so white stays
//! white. It costs the kernels nothing and acts on the whole image
//! consistently; working-space primaries (rather than the camera's own,
//! which are imaginary and far apart) make it behave alike on every camera.
//! Its shadows tint adds green (-) or magenta (+) to the shadows per pixel,
//! fading out by mid-grey.

use super::layout::MIXER_LEN;
use crate::types::Params;

/// Index of the "Color Mixer is on" flag in the MIXER region.
pub(crate) const MIX_ON: usize = 31;
/// Shadows tint: Oklab (a, b) per unit of lightness at full shadow weight.
pub(crate) const MIX_TINT: usize = 32;
/// Hue tables: `MIX_N + 1` bins of (hue shift in radians, chroma gain,
/// log2 lightness scale).
pub(crate) const MIX_LUT: usize = 36;
/// Hue bins over the full circle (5 degrees each).
pub(crate) const MIX_N: usize = 72;
/// Chroma at which hue and saturation are half on (they fade in as the
/// fourth power of chroma).
#[cfg(test)]
pub(crate) const MIX_CN: f32 = 0.025;
/// Chroma at which luminance is half on.
#[cfg(test)]
pub(crate) const MIX_CL: f32 = 0.05;
/// Oklab lightness by which the shadows tint has faded out.
#[cfg(test)]
pub(crate) const MIX_TINT_L1: f32 = 0.75;

/// Oklab hue (degrees) at the centre of each band: red, orange, yellow,
/// green, aqua, blue, purple, magenta. Each is the hue of sRGB's colour at
/// that HSV hue (0, 30, 60, 120, 180, 240, 270, 300), between full and half
/// saturation, where Oklab bends a little.
pub(crate) const BAND_HUES: [f32; 8] = [24.0, 58.0, 108.0, 143.0, 195.0, 266.0, 298.0, 328.0];

/// Hue at +-100: this fraction of the way to the neighbouring band...
const HUE_REACH: f32 = 0.6;
/// ...but at most this many degrees.
const HUE_MAX: f32 = 30.0;
/// Two bands pushed toward each other stay at least this fraction of their
/// original distance apart.
const HUE_MIN_GAP: f32 = 0.15;
/// Chroma gain added at Saturation +100 (-100 always removes all chroma).
const SAT_UP: f32 = 1.0;
/// Luminance +-100 scales dark colours' Oklab lightness by 2^+-LUM_STOPS.
const LUM_STOPS: f32 = 1.0;

/// Shadows tint +100 points this way in Oklab (degrees): the magenta of
/// Lightroom's Tint, the direction across the Planckian locus at D65.
const TINT_HUE: f32 = 330.0;
/// Oklab chroma per unit of lightness at +-100 and full shadow weight.
const TINT_STRENGTH: f32 = 0.06;

/// Calibration: each primary's chroma turn at +-100 (degrees; red, green,
/// blue), which turns a strong colour of that primary by roughly 7 (green)
/// to 18 (blue) degrees of Oklab hue, and its chroma scale, 1 + v *
/// PRIMARY_SAT.
const PRIMARY_HUE: [f64; 3] = [12.0, 16.0, 8.0];
const PRIMARY_SAT: f64 = 0.4;

/// Linear Rec.2020 -> LMS (Oklab's M1 for Rec.2020 primaries).
const LMS_FROM_2020: [[f64; 3]; 3] = [
    [0.616_755_787_2, 0.360_198_399_4, 0.023_045_813_4],
    [0.265_133_064_0, 0.635_839_364_1, 0.099_027_571_8],
    [0.100_102_634_2, 0.203_906_519_4, 0.695_990_846_4],
];
/// LMS (after the cube root) -> Oklab (Oklab's M2).
const LAB_FROM_LMS: [[f64; 3]; 3] = [
    [0.210_454_255_3, 0.793_617_785_0, -0.004_072_046_8],
    [1.977_998_495_1, -2.428_592_205_0, 0.450_593_709_9],
    [0.025_904_037_1, 0.782_771_766_2, -0.808_675_766_0],
];

type M3 = [[f64; 3]; 3];

fn mul(a: &M3, b: &M3) -> M3 {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

#[cfg(test)]
fn mul_v(a: &M3, v: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| (0..3).map(|k| a[i][k] * v[k]).sum())
}

/// The bands' hue, saturation and luminance sliders / 100.
struct Bands {
    hue: [f32; 8],
    sat: [f32; 8],
    lum: [f32; 8],
}

impl Bands {
    fn of(p: &Params) -> Self {
        let unit = |v: [f32; 8]| v.map(|x| (x / 100.0).clamp(-1.0, 1.0));
        Self {
            hue: unit([
                p.hue_red,
                p.hue_orange,
                p.hue_yellow,
                p.hue_green,
                p.hue_aqua,
                p.hue_blue,
                p.hue_purple,
                p.hue_magenta,
            ]),
            sat: unit([
                p.saturation_red,
                p.saturation_orange,
                p.saturation_yellow,
                p.saturation_green,
                p.saturation_aqua,
                p.saturation_blue,
                p.saturation_purple,
                p.saturation_magenta,
            ]),
            lum: unit([
                p.luminance_red,
                p.luminance_orange,
                p.luminance_yellow,
                p.luminance_green,
                p.luminance_aqua,
                p.luminance_blue,
                p.luminance_purple,
                p.luminance_magenta,
            ]),
        }
    }

    fn is_neutral(&self) -> bool {
        self.hue
            .iter()
            .chain(&self.sat)
            .chain(&self.lum)
            .all(|&v| v == 0.0)
    }
}

/// Degrees from band `i`'s centre to the next band's.
fn gap(i: usize) -> f32 {
    (BAND_HUES[(i + 1) % 8] - BAND_HUES[i]).rem_euclid(360.0)
}

fn smoothstep(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// The Color Mixer as functions of Oklab hue, at hue `360 k / MIX_N`
/// degrees for bin k: hue shift (degrees), chroma gain, and the log2 of the
/// lightness scale at black.
pub(crate) fn tables(p: &Params) -> [[f32; 3]; MIX_N] {
    let bands = Bands::of(p);
    // Each band's hue displacement, then a minimum gap between bands pushed
    // toward each other. Bands move one way each, so the pairs that need
    // the fix never share a band.
    let mut shift: [f32; 8] = std::array::from_fn(|i| {
        let v = bands.hue[i];
        let reach = if v >= 0.0 { gap(i) } else { gap((i + 7) % 8) };
        v * (HUE_REACH * reach).min(HUE_MAX)
    });
    for i in 0..8 {
        let j = (i + 1) % 8;
        let sep = gap(i) + shift[j] - shift[i];
        let min = HUE_MIN_GAP * gap(i);
        if sep < min {
            let fix = 0.5 * (min - sep);
            shift[i] -= fix;
            shift[j] += fix;
        }
    }
    let gain = bands
        .sat
        .map(|s| 1.0 + s * if s > 0.0 { SAT_UP } else { 1.0 });
    let lum = bands.lum.map(|l| l * LUM_STOPS);

    let mut out = [[0.0f32; 3]; MIX_N];
    for (k, row) in out.iter_mut().enumerate() {
        let h = k as f32 * (360.0 / MIX_N as f32);
        let i = BAND_HUES.iter().rposition(|&c| c <= h).unwrap_or(7);
        let j = (i + 1) % 8;
        let x = (h - BAND_HUES[i]).rem_euclid(360.0) / gap(i);
        // The hue shift interpolates linearly between centres, which keeps
        // the hue map monotonic (smoothed below); saturation and luminance
        // ease in and out, flat at each band's centre.
        let e = smoothstep(x);
        *row = [
            shift[i] + x * (shift[j] - shift[i]),
            gain[i] + e * (gain[j] - gain[i]),
            lum[i] + e * (lum[j] - lum[i]),
        ];
    }
    // Smooth the hue shift around the circle (binomial, sigma one bin). A
    // weighted average of a monotonic hue map stays monotonic.
    let raw: [f32; MIX_N] = std::array::from_fn(|k| out[k][0]);
    for (k, row) in out.iter_mut().enumerate() {
        let at = |d: usize| raw[(k + MIX_N - 2 + d) % MIX_N];
        row[0] = (at(0) + 4.0 * at(1) + 6.0 * at(2) + 4.0 * at(3) + at(4)) / 16.0;
    }
    out
}

/// Calibration's primaries applied to `m`, the camera -> linear Rec.2020
/// matrix (row-major) at `p`'s white balance, as a working-space matrix in
/// front of it. In linear Oklab axes (lightness l and chroma a, b; white has
/// no chroma), each Rec.2020 primary's chroma turns by its hue slider and
/// scales by its saturation slider. A colour moves by its share of each
/// primary minus its lightness times that primary's change, so white (and
/// every grey) stays exactly where it was. All zero leaves `m` untouched.
pub(crate) fn calibrate(p: &Params, m: &mut [[f32; 3]; 3]) {
    let hue = [p.red_primary_hue, p.green_primary_hue, p.blue_primary_hue];
    let sat = [
        p.red_primary_saturation,
        p.green_primary_saturation,
        p.blue_primary_saturation,
    ];
    if hue.iter().chain(&sat).all(|&v| v == 0.0) {
        return;
    }
    let to_lab = mul(&LAB_FROM_LMS, &LMS_FROM_2020);
    let Some(from_lab) = crate::color::invert(&to_lab) else {
        return;
    };
    // In linear Oklab, from Rec.2020: to_lab plus each primary's chroma
    // change times (its own share - the colour's lightness).
    let mut moved = to_lab;
    for j in 0..3 {
        let (a, b) = (to_lab[1][j], to_lab[2][j]);
        let h = (f64::from(hue[j]) / 100.0).clamp(-1.0, 1.0);
        let s = 1.0 + (f64::from(sat[j]) / 100.0).clamp(-1.0, 1.0) * PRIMARY_SAT;
        let (sin, cos) = (h * PRIMARY_HUE[j]).to_radians().sin_cos();
        let d = [s * (cos * a - sin * b) - a, s * (sin * a + cos * b) - b];
        for (row, d) in moved[1..].iter_mut().zip(d) {
            for (k, v) in row.iter_mut().enumerate() {
                let own = if k == j { 1.0 } else { 0.0 };
                *v += d * (own - to_lab[0][k]);
            }
        }
    }
    let c = mul(&from_lab, &moved);
    let out = mul(&c, &m.map(|row| row.map(f64::from)));
    if out.iter().flatten().all(|v| v.is_finite()) {
        *m = out.map(|row| row.map(|v| v as f32));
    }
}

/// Writes `p`'s Color Mixer and Calibration into `out` (the MIXER region,
/// `MIXER_LEN` floats) in the layout `cuda/color_mixer.cu` documents.
pub(crate) fn pack(p: &Params, out: &mut [f32]) {
    let out = &mut out[..MIXER_LEN];
    out.fill(0.0);
    let values = [
        p.hue_red,
        p.hue_orange,
        p.hue_yellow,
        p.hue_green,
        p.hue_aqua,
        p.hue_blue,
        p.hue_purple,
        p.hue_magenta,
        p.saturation_red,
        p.saturation_orange,
        p.saturation_yellow,
        p.saturation_green,
        p.saturation_aqua,
        p.saturation_blue,
        p.saturation_purple,
        p.saturation_magenta,
        p.luminance_red,
        p.luminance_orange,
        p.luminance_yellow,
        p.luminance_green,
        p.luminance_aqua,
        p.luminance_blue,
        p.luminance_purple,
        p.luminance_magenta,
        p.shadow_tint,
        p.red_primary_hue,
        p.red_primary_saturation,
        p.green_primary_hue,
        p.green_primary_saturation,
        p.blue_primary_hue,
        p.blue_primary_saturation,
    ];
    for (o, v) in out.iter_mut().zip(values) {
        *o = v / 100.0;
    }
    let tint = (p.shadow_tint / 100.0).clamp(-1.0, 1.0) * TINT_STRENGTH;
    if tint != 0.0 {
        let (sin, cos) = TINT_HUE.to_radians().sin_cos();
        out[MIX_TINT] = tint * cos;
        out[MIX_TINT + 1] = tint * sin;
    }
    let on = !Bands::of(p).is_neutral();
    out[MIX_ON] = if on { 1.0 } else { 0.0 };
    let lut = &mut out[MIX_LUT..MIX_LUT + 3 * (MIX_N + 1)];
    if !on {
        for bin in lut.as_chunks_mut::<3>().0 {
            bin[1] = 1.0;
        }
        return;
    }
    let t = tables(p);
    for (k, bin) in lut.as_chunks_mut::<3>().0.iter_mut().enumerate() {
        let [shift, gain, e] = t[k % MIX_N];
        *bin = [shift.to_radians(), gain, e];
    }
}

/// The kernel's per-pixel step (`color_mixer_apply`) on the CPU, for tests:
/// Oklab (L, a, b) in, Oklab out.
#[cfg(test)]
pub(crate) fn apply(m: &[f32], lab: [f32; 3]) -> [f32; 3] {
    let [mut l, mut a, mut b] = lab;
    if m[MIX_ON] != 0.0 {
        let c2 = a * a + b * b;
        if c2 > 1e-14 {
            let mut f = b.atan2(a) * (MIX_N as f32 / std::f32::consts::TAU);
            if f < 0.0 {
                f += MIX_N as f32;
            }
            let f = f.clamp(0.0, MIX_N as f32);
            let i = (f as usize).min(MIX_N - 1);
            let t = f - i as f32;
            let at = |c: usize| {
                let (x0, x1) = (m[MIX_LUT + 3 * i + c], m[MIX_LUT + 3 * i + 3 + c]);
                x0 + t * (x1 - x0)
            };
            let fade = |c0: f32| {
                let q = (c2 / (c0 * c0)).powi(2);
                q / (1.0 + q)
            };
            let wn = fade(MIX_CN);
            let k = 1.0 + wn * (at(1) - 1.0);
            let (s, c) = (wn * at(0)).sin_cos();
            let (u, v) = (k * c, k * s);
            (a, b) = (u * a - v * b, v * a + u * b);
            let e = at(2) * fade(MIX_CL);
            if e != 0.0 {
                let x = l.clamp(0.0, 1.0);
                let g = e.exp2();
                l = x * g / (1.0 + x * (g - 1.0));
            }
        }
    }
    let (ta, tb) = (m[MIX_TINT], m[MIX_TINT + 1]);
    if ta != 0.0 || tb != 0.0 {
        let x = l.max(0.0);
        let w = x * (1.0 - smoothstep(x / MIX_TINT_L1));
        a += w * ta;
        b += w * tb;
    }
    [l, a, b]
}

#[cfg(test)]
mod tests;
