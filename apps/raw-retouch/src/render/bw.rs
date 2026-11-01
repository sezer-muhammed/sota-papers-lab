//! Black & White (Lightroom's B&W treatment) and its gray mixer: packs the
//! candidate's BW region for `cuda/bw.cu`. Owner: bw.
//!
//! Lightroom-like: the grey of a pixel is its linear luminance (Rec.2020 Y,
//! after white balance and the camera matrix, before exposure), brightened
//! or darkened per hue band by the gray mixer, and then every tone stage
//! (Exposure, Highlights/Shadows, Whites/Blacks, Presence, the tone curves,
//! Contrast) works on that grey. White balance still matters, as in
//! Lightroom: it decides which band a colour falls in.
//!
//! The eight bands sit at the Color Mixer's Oklab hues
//! ([`color_mixer::BAND_HUES`]) and blend into their neighbours the same
//! way (eased between centres, flat at each one), so setting all eight
//! alike moves every hue alike. A slider at +-100 moves a fully colourful
//! pixel of its band by +-[`BW_STOPS`] stops; the move fades in with
//! colourfulness (Oklab C / L, half on at [`BW_S0`]), so greys keep their
//! tone exactly and a muted colour moves less than a vivid one. It fades
//! out in deep shadows too, where a pixel's hue is colour noise. The
//! kernel only interpolates the table [`pack`] bakes; [`apply`] is the
//! same per-pixel step on the CPU, for tests.
//!
//! The output carries no colour: render_core clears Oklab a/b, the gamut
//! mapping turns an exact grey into three equal channels, the 8-bit
//! preview dithers a grey pixel's channels alike, and [`grey8`] /
//! [`grey16`] equalize whatever Detail's rounding leaves (at most a step).

use super::color_mixer::{BAND_HUES, MIX_N};
use super::layout::BW_LEN;
use crate::types::{Params, Rgb8Image, Rgb16Image};

/// 1 in black and white.
pub(crate) const BW_ON: usize = 0;
/// 1 when any gray mixer slider is set (else the mix is plain luminance).
pub(crate) const BW_MIX: usize = 1;
/// The hue table: `BW_N + 1` bins of stops at full colourfulness.
pub(crate) const BW_LUT: usize = 8;
/// Hue bins over the full circle (5 degrees each), as the Color Mixer's.
pub(crate) const BW_N: usize = MIX_N;
/// Stops a fully colourful pixel moves at +-100.
pub(crate) const BW_STOPS: f32 = 2.0;
/// Colourfulness (Oklab C / L) at which the mixer is half on.
#[cfg(test)]
pub(crate) const BW_S0: f32 = 0.12;
/// Scene luminance after exposure (middle grey 0.18) below which the mixer
/// is off, and from which it is fully on.
#[cfg(test)]
pub(crate) const BW_DARK0: f32 = 0.001;
#[cfg(test)]
pub(crate) const BW_DARK1: f32 = 0.006;

const _: () = assert!(BW_LUT + BW_N < BW_LEN);

/// The gray mixer's sliders / 100, red to magenta.
pub(crate) fn sliders(p: &Params) -> [f32; 8] {
    [
        p.gray_red,
        p.gray_orange,
        p.gray_yellow,
        p.gray_green,
        p.gray_aqua,
        p.gray_blue,
        p.gray_purple,
        p.gray_magenta,
    ]
    .map(|v| (v / 100.0).clamp(-1.0, 1.0))
}

fn smoothstep(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// Stops at full colourfulness as a function of Oklab hue, at hue
/// `360 k / BW_N` degrees for bin k.
pub(crate) fn table(p: &Params) -> [f32; BW_N] {
    let stops = sliders(p).map(|v| v * BW_STOPS);
    std::array::from_fn(|k| {
        let h = k as f32 * (360.0 / BW_N as f32);
        let i = BAND_HUES.iter().rposition(|&c| c <= h).unwrap_or(7);
        let j = (i + 1) % 8;
        let gap = (BAND_HUES[j] - BAND_HUES[i]).rem_euclid(360.0);
        let x = (h - BAND_HUES[i]).rem_euclid(360.0) / gap;
        stops[i] + smoothstep(x) * (stops[j] - stops[i])
    })
}

/// Writes `p`'s Black & White settings into `out` (the BW region,
/// `BW_LEN` floats) in the layout `cuda/bw.cu` documents. All zero in
/// colour, so the kernel skips the stage.
pub(crate) fn pack(p: &Params, out: &mut [f32]) {
    let out = &mut out[..BW_LEN];
    out.fill(0.0);
    if !p.is_bw() {
        return;
    }
    out[BW_ON] = 1.0;
    if sliders(p).iter().all(|&v| v == 0.0) {
        return;
    }
    out[BW_MIX] = 1.0;
    let t = table(p);
    for (k, v) in out[BW_LUT..=BW_LUT + BW_N].iter_mut().enumerate() {
        *v = t[k % BW_N];
    }
}

/// Every pixel's channels set to their mean: the last word on black and
/// white after Detail, whose luma/chroma round trip can leave a channel a
/// step off.
pub(crate) fn grey8(img: &mut Rgb8Image) {
    for px in img.data.as_chunks_mut::<3>().0 {
        let m = ((u16::from(px[0]) + u16::from(px[1]) + u16::from(px[2]) + 1) / 3) as u8;
        *px = [m; 3];
    }
}

/// [`grey8`] for 16-bit images.
pub(crate) fn grey16(img: &mut Rgb16Image) {
    for px in img.data.as_chunks_mut::<3>().0 {
        let m = ((u32::from(px[0]) + u32::from(px[1]) + u32::from(px[2]) + 1) / 3) as u16;
        *px = [m; 3];
    }
}

/// Linear Rec.2020 to Oklab, the kernel's matrices.
#[cfg(test)]
fn oklab(r: f32, g: f32, b: f32) -> [f32; 3] {
    let l = (0.616_755_8 * r + 0.360_198_4 * g + 0.023_045_813 * b).cbrt();
    let m = (0.265_133_06 * r + 0.635_839_4 * g + 0.099_027_57 * b).cbrt();
    let s = (0.100_102_63 * r + 0.203_906_52 * g + 0.695_990_85 * b).cbrt();
    [
        0.210_454_26 * l + 0.793_617_8 * m - 0.004_072_047 * s,
        1.977_998_5 * l - 2.428_592_2 * m + 0.450_593_7 * s,
        0.025_904_037 * l + 0.782_771_77 * m - 0.808_675_77 * s,
    ]
}

/// The kernel's `bw_apply` on the CPU, for tests: linear Rec.2020 in (at
/// exposure `gain`), the grey out, or the pixel unchanged in colour.
#[cfg(test)]
pub(crate) fn apply(b: &[f32], gain: f32, rgb: [f32; 3]) -> [f32; 3] {
    if b[BW_ON] == 0.0 {
        return rgb;
    }
    let [x, y, z] = rgb;
    let mut lum = 0.262_700_2 * x + 0.677_998_07 * y + 0.059_301_717 * z;
    if b[BW_MIX] != 0.0 && lum > 0.0 {
        let [l, a, bb] = oklab(x, y, z);
        let c2 = a * a + bb * bb;
        if c2 > 1e-14 && l > 1e-6 {
            let mut f = bb.atan2(a) * (BW_N as f32 / std::f32::consts::TAU);
            if f < 0.0 {
                f += BW_N as f32;
            }
            let f = f.clamp(0.0, BW_N as f32);
            let i = (f as usize).min(BW_N - 1);
            let t = f - i as f32;
            let e = b[BW_LUT + i] + t * (b[BW_LUT + i + 1] - b[BW_LUT + i]);
            let q = c2 / (l * l * BW_S0 * BW_S0);
            let w = q / (1.0 + q) * smoothstep((lum * gain - BW_DARK0) / (BW_DARK1 - BW_DARK0));
            lum *= (w * e).exp2();
        }
    }
    [lum; 3]
}

#[cfg(test)]
mod gpu_tests;

#[cfg(test)]
mod tests {
    use super::*;

    fn bw(f: impl FnOnce(&mut Params)) -> Params {
        let mut p = Params {
            bw: 1.0,
            ..Params::neutral(5500.0, 0.0)
        };
        f(&mut p);
        p
    }

    fn packed(p: &Params) -> Vec<f32> {
        let mut out = vec![f32::NAN; BW_LEN];
        pack(p, &mut out);
        out
    }

    /// Sample colours, linear Rec.2020 around middle grey: red, orange
    /// (skin), yellow, green (foliage), aqua, blue (sky), purple, magenta.
    const COLOURS: [[f32; 3]; 8] = [
        [0.40, 0.06, 0.05],
        [0.36, 0.17, 0.09],
        [0.30, 0.27, 0.04],
        [0.08, 0.22, 0.05],
        [0.05, 0.22, 0.22],
        [0.06, 0.12, 0.40],
        [0.16, 0.08, 0.36],
        [0.34, 0.07, 0.26],
    ];

    #[test]
    fn defines_match_cuda() {
        let src = include_str!("../../cuda/bw.cu");
        let get = |name: &str| -> f32 {
            let line = src
                .lines()
                .find_map(|l| l.strip_prefix(&format!("#define {name} ")))
                .unwrap_or_else(|| panic!("{name}"));
            line.trim()
                .trim_start_matches('(')
                .trim_end_matches(')')
                .trim_end_matches('f')
                .parse()
                .unwrap()
        };
        assert_eq!(get("BW_ON") as usize, BW_ON);
        assert_eq!(get("BW_MIX") as usize, BW_MIX);
        assert_eq!(get("BW_LUT") as usize, BW_LUT);
        assert_eq!(get("BW_N") as usize, BW_N);
        assert_eq!(get("BW_S0"), BW_S0);
        assert_eq!(get("BW_DARK0"), BW_DARK0);
        assert_eq!(get("BW_DARK1"), BW_DARK1);
    }

    #[test]
    fn colour_packs_to_nothing() {
        let p = Params {
            gray_blue: -40.0,
            ..Params::neutral(5500.0, 0.0)
        };
        assert!(packed(&p).iter().all(|&v| v == 0.0));
        assert_eq!(apply(&packed(&p), 1.0, [0.3, 0.2, 0.1]), [0.3, 0.2, 0.1]);
    }

    #[test]
    fn neutral_mixer_is_luminance_and_greys_stay_exact() {
        let p = bw(|_| {});
        let b = packed(&p);
        assert_eq!((b[BW_ON], b[BW_MIX]), (1.0, 0.0));
        let [y, ..] = apply(&b, 1.0, [0.4, 0.2, 0.1]);
        let want = 0.262_700_2 * 0.4 + 0.677_998_07 * 0.2 + 0.059_301_717 * 0.1;
        assert!((y - want).abs() < 1e-7);
        // A strong mixer leaves every grey exactly where plain luminance
        // puts it, which is the grey itself.
        let strong = packed(&bw(|p| {
            p.gray_red = 100.0;
            p.gray_blue = -100.0;
            p.gray_green = 60.0;
        }));
        for v in [0.0, 0.001, 0.05, 0.18, 0.9, 1.0] {
            let out = apply(&strong, 1.0, [v; 3]);
            assert_eq!(out, apply(&b, 1.0, [v; 3]));
            assert!((out[0] - v).abs() <= 1e-6 * v, "{v}: {out:?}");
        }
    }

    #[test]
    fn each_band_moves_its_colours_monotonically_and_mostly_alone() {
        for (band, rgb) in COLOURS.iter().enumerate() {
            let mut last = 0.0;
            for v in [-100.0, -50.0, 0.0, 50.0, 100.0] {
                let mut p = bw(|_| {});
                let mut a = p.to_array();
                a[Params::COUNT - 8 + band] = v;
                p = Params::from_array(a);
                let [y, g, b] = apply(&packed(&p), 1.0, *rgb);
                assert!(y == g && g == b);
                assert!(y > last, "band {band} at {v}: {y} <= {last}");
                last = y;
            }
            // At +100 its own colour brightens by well over half a stop;
            // the colour two bands away barely moves.
            let mut a = bw(|_| {}).to_array();
            a[Params::COUNT - 8 + band] = 100.0;
            let b = packed(&Params::from_array(a));
            let plain = packed(&bw(|_| {}));
            let own = apply(&b, 1.0, *rgb)[0] / apply(&plain, 1.0, *rgb)[0];
            assert!(own > 1.5, "band {band}: {own}");
            let far = COLOURS[(band + 4) % 8];
            let other = apply(&b, 1.0, far)[0] / apply(&plain, 1.0, far)[0];
            assert!((other - 1.0).abs() < 0.05, "band {band}: {other}");
        }
    }

    #[test]
    fn equal_sliders_move_every_hue_alike_and_shadows_are_spared() {
        let all = bw(|p| {
            p.gray_red = 50.0;
            p.gray_orange = 50.0;
            p.gray_yellow = 50.0;
            p.gray_green = 50.0;
            p.gray_aqua = 50.0;
            p.gray_blue = 50.0;
            p.gray_purple = 50.0;
            p.gray_magenta = 50.0;
        });
        let t = table(&all);
        assert!(t.iter().all(|&v| (v - 0.5 * BW_STOPS).abs() < 1e-6));
        // A dark noisy pixel keeps its plain luminance.
        let b = packed(&all);
        let plain = packed(&bw(|_| {}));
        let dark = [0.0008, 0.0001, 0.0002];
        assert_eq!(apply(&b, 1.0, dark), apply(&plain, 1.0, dark));
    }

    #[test]
    fn grey_out_equalizes_channels() {
        let mut img = Rgb8Image {
            width: 2,
            height: 1,
            data: vec![10, 11, 10, 255, 254, 255],
        };
        grey8(&mut img);
        assert_eq!(img.data, [10, 10, 10, 255, 255, 255]);
        let mut img = Rgb16Image {
            width: 1,
            height: 1,
            data: vec![1000, 1001, 1001],
        };
        grey16(&mut img);
        assert_eq!(img.data, [1001; 3]);
    }
}
