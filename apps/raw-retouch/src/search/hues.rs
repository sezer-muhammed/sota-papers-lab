//! Which colours a photo holds, per Color Mixer band, measured once on the
//! CPU from the linear search image. Owner: search.
//!
//! The render statistics are global (colourfulness, chroma percentiles,
//! gamut), so they cannot tell a sky from a face. This profile can: it
//! decides which bands the search varies at all (a band with no pixels
//! stays at zero), where the look's colour moves apply, and how much the
//! skin-protection and band-chroma terms of the objective weigh.
//!
//! Colours are read in Oklab at the as-shot white balance: hue angle
//! `atan2(b, a)` and colourfulness `C / L`, which exposure does not change
//! (Oklab scales L, a and b alike when light scales). Bands are centred on
//! Lightroom's colour names as Oklab hues and overlap with raised-cosine
//! weights that sum to one, the way the Color Mixer blends neighbouring
//! bands. `cuda/color_mixer.cu` owns the rendered bands; these centres only
//! need to agree with it roughly.

use super::priors::{HUE, LUM, SAT};
use crate::types::{LinearImage, Params, Scene};

pub(crate) const BANDS: usize = 8;
pub(crate) const RED: usize = 0;
pub(crate) const ORANGE: usize = 1;
pub(crate) const YELLOW: usize = 2;
pub(crate) const GREEN: usize = 3;
pub(crate) const AQUA: usize = 4;
pub(crate) const BLUE: usize = 5;
pub(crate) const PURPLE: usize = 6;
pub(crate) const MAGENTA: usize = 7;

/// Band centres as Oklab hue angles in degrees: red, orange, yellow, green,
/// aqua, blue, purple, magenta (sRGB red sits at 29, yellow 110, green
/// 142, cyan 195, blue 264, magenta 328; skin at 40 to 70).
pub(crate) const CENTRES: [f32; BANDS] = [20.0, 55.0, 95.0, 140.0, 195.0, 250.0, 290.0, 330.0];

/// Colourfulness (Oklab C / L) below which a pixel counts as grey, and from
/// which it counts fully.
const GREY: (f32, f32) = (0.03, 0.10);
/// Skin and other skin-like surfaces (wood, sand, tan): Oklab hue 45 +- 25
/// degrees, as the renderer's Vibrance protects them, at moderate
/// colourfulness.
const SKIN_HUE: f32 = 45.0;
const SKIN_HALF: f32 = 25.0;
const SKIN_EDGE: f32 = 8.0;
const SKIN_CHROMA: (f32, f32, f32, f32) = (0.03, 0.06, 0.30, 0.40);
/// Linear luminance (1.0 = clip) below which colour is mostly noise, and the
/// brightest channel above which it is clipped away.
const DARK: f32 = 0.003;
const CLIPPED: f32 = 0.98;
/// Pixels measured at most.
const SAMPLES: usize = 16_384;
/// Band share (colour-weighted, of all pixels) from which a band is present
/// at all and from which it is fully present.
const PRESENT: (f32, f32) = (0.01, 0.06);
/// Band share from which the search varies a band's sliders.
const SEARCHED: f32 = 0.02;
/// Skin share from which skin is fully present.
const SKIN_FULL: (f32, f32) = (0.01, 0.05);

/// A photo's colours by Color Mixer band.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Hues {
    /// Share of all pixels in each band, each pixel weighted by how
    /// colourful it is (0 for grey, 1 from C / L 0.1) and by its band weights.
    pub mass: [f32; BANDS],
    /// Mean colourfulness (C / L) of each band's colourful pixels.
    pub chroma: [f32; BANDS],
    /// Share of all pixels that look like skin.
    pub skin: f32,
    /// How the skin-like pixels spread over the bands (summing to 1, or all 0).
    pub skin_bands: [f32; BANDS],
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    crate::objective::smoothstep(e0, e1, x)
}

/// Every band's weight at Oklab hue `deg`: raised-cosine overlap between
/// the two nearest centres, summing to 1.
pub(crate) fn band_weights(deg: f32) -> [f32; BANDS] {
    let mut w = [0.0; BANDS];
    if !deg.is_finite() {
        return w;
    }
    let h = deg.rem_euclid(360.0);
    for b in 0..BANDS {
        let lo = CENTRES[b];
        let hi = if b + 1 < BANDS {
            CENTRES[b + 1]
        } else {
            CENTRES[0] + 360.0
        };
        let x = if h < lo { h + 360.0 } else { h };
        if (lo..hi).contains(&x) {
            let t = (x - lo) / (hi - lo);
            let to_next = 0.5 - 0.5 * (std::f32::consts::PI * t).cos();
            w[b] = 1.0 - to_next;
            w[(b + 1) % BANDS] = to_next;
            break;
        }
    }
    w
}

/// Linear Rec.2020 to Oklab, the renderer's matrices (`cuda/render.cu`).
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

impl Hues {
    /// The scene's search image at its as-shot white balance.
    pub fn of(scene: &Scene) -> Self {
        let t = crate::search::sane_temperature(scene.info.as_shot_temperature);
        Self::measure(&scene.search, &scene.color.rec2020_from_cam(t))
    }

    /// `img` is linear camera RGB (white balanced, 1.0 = clip) and `m` its
    /// camera -> linear Rec.2020 matrix.
    pub fn measure(img: &LinearImage, m: &[[f32; 3]; 3]) -> Self {
        let pixels = img.width * img.height;
        if pixels == 0 || img.data.len() < pixels * 3 || !m.iter().flatten().all(|v| v.is_finite())
        {
            return Self::default();
        }
        let step = pixels.div_ceil(SAMPLES).max(1);
        let (mut n, mut skin, mut skin_w) = (0.0f64, 0.0f64, [0.0f64; BANDS]);
        let (mut mass, mut chroma) = ([0.0f64; BANDS], [0.0f64; BANDS]);
        for px in img.data[..pixels * 3]
            .as_chunks::<3>()
            .0
            .iter()
            .step_by(step)
        {
            n += 1.0;
            if !px.iter().all(|v| v.is_finite())
                || px.iter().copied().fold(0.0, f32::max) >= CLIPPED
            {
                continue;
            }
            let rgb =
                [0, 1, 2].map(|i| (m[i][0] * px[0] + m[i][1] * px[1] + m[i][2] * px[2]).max(0.0));
            let y = 0.2627 * rgb[0] + 0.678 * rgb[1] + 0.0593 * rgb[2];
            if y < DARK {
                continue;
            }
            let [l, a, b] = oklab(rgb[0], rgb[1], rgb[2]);
            if l <= 1e-6 {
                continue;
            }
            let sat = a.hypot(b) / l;
            let colourful = smoothstep(GREY.0, GREY.1, sat);
            if colourful <= 0.0 {
                continue;
            }
            let deg = b.atan2(a).to_degrees();
            let weights = band_weights(deg);
            for (k, w) in weights.iter().enumerate() {
                let w = f64::from(w * colourful);
                mass[k] += w;
                chroma[k] += w * f64::from(sat);
            }
            let off = ((deg - SKIN_HUE + 180.0).rem_euclid(360.0) - 180.0).abs();
            let (c0, c1, c2, c3) = SKIN_CHROMA;
            let s = smoothstep(SKIN_HALF + SKIN_EDGE, SKIN_HALF - SKIN_EDGE, off)
                * smoothstep(c0, c1, sat)
                * smoothstep(c3, c2, sat);
            if s > 0.0 {
                skin += f64::from(s);
                for (sw, w) in skin_w.iter_mut().zip(weights) {
                    *sw += f64::from(s * w);
                }
            }
        }
        if n == 0.0 {
            return Self::default();
        }
        let skin_total: f64 = skin_w.iter().sum();
        Self {
            mass: mass.map(|v| (v / n) as f32),
            chroma: std::array::from_fn(|k| {
                if mass[k] > 0.0 {
                    (chroma[k] / mass[k]) as f32
                } else {
                    0.0
                }
            }),
            skin: (skin / n) as f32,
            skin_bands: skin_w.map(|v| {
                if skin_total > 0.0 {
                    (v / skin_total) as f32
                } else {
                    0.0
                }
            }),
        }
    }

    /// How present band `b` is, 0 (absent) to 1.
    pub(crate) fn presence(&self, b: usize) -> f32 {
        smoothstep(PRESENT.0, PRESENT.1, self.mass[b])
    }

    /// Whether the search varies band `b`'s sliders.
    pub(crate) fn searched(&self, b: usize) -> bool {
        self.mass[b] >= SEARCHED
    }

    /// How present skin is, 0 (none) to 1.
    pub(crate) fn skin_presence(&self) -> f32 {
        smoothstep(SKIN_FULL.0, SKIN_FULL.1, self.skin)
    }

    /// `p` (in [`Params::to_array`] order) with the hue, saturation and
    /// luminance of every band the search leaves alone back at zero: a
    /// colour the photo does not hold gets no setting.
    pub(crate) fn fix(&self, p: &mut [f32; Params::COUNT]) {
        for b in (0..BANDS).filter(|&b| !self.searched(b)) {
            for first in [HUE, SAT, LUM] {
                p[first + b] = 0.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const IDENTITY: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    fn hue_of(rgb: [f32; 3]) -> f32 {
        let [_, a, b] = oklab(rgb[0], rgb[1], rgb[2]);
        b.atan2(a).to_degrees().rem_euclid(360.0)
    }

    fn image(pixels: &[([f32; 3], usize)]) -> LinearImage {
        let data: Vec<f32> = pixels
            .iter()
            .flat_map(|(rgb, n)| std::iter::repeat_n(*rgb, *n).flatten())
            .collect();
        LinearImage {
            width: data.len() / 3,
            height: 1,
            data,
        }
    }

    #[test]
    fn weights_overlap_smoothly_and_sum_to_one() {
        for deg in (0..720).map(|d| d as f32 * 0.5) {
            let w = band_weights(deg);
            assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-5, "{deg}: {w:?}");
            assert!(w.iter().filter(|v| **v > 0.0).count() <= 2);
        }
        for (b, c) in CENTRES.iter().enumerate() {
            assert!((band_weights(*c)[b] - 1.0).abs() < 1e-6);
        }
        // Halfway between magenta and red, across 360.
        let w = band_weights(355.0);
        assert!(
            (w[7] - 0.5).abs() < 1e-5 && (w[0] - 0.5).abs() < 1e-5,
            "{w:?}"
        );
        assert_eq!(band_weights(f32::NAN), [0.0; BANDS]);
    }

    #[test]
    fn primaries_land_in_their_bands() {
        // Rec.2020-linear stand-ins for an sRGB-like red, yellow, green, cyan,
        // blue and magenta.
        let dominant = |rgb: [f32; 3]| {
            let w = band_weights(hue_of(rgb));
            (0..BANDS).max_by(|a, b| w[*a].total_cmp(&w[*b])).unwrap()
        };
        assert_eq!(dominant([0.6, 0.05, 0.03]), 0);
        assert_eq!(
            dominant([0.55, 0.22, 0.08]),
            1,
            "orange: {}",
            hue_of([0.55, 0.22, 0.08])
        );
        assert_eq!(
            dominant([0.5, 0.45, 0.03]),
            2,
            "yellow: {}",
            hue_of([0.5, 0.45, 0.03])
        );
        assert_eq!(
            dominant([0.05, 0.4, 0.04]),
            3,
            "green: {}",
            hue_of([0.05, 0.4, 0.04])
        );
        assert_eq!(
            dominant([0.04, 0.35, 0.4]),
            4,
            "aqua: {}",
            hue_of([0.04, 0.35, 0.4])
        );
        assert_eq!(
            dominant([0.05, 0.12, 0.5]),
            5,
            "blue: {}",
            hue_of([0.05, 0.12, 0.5])
        );
        assert_eq!(
            dominant([0.4, 0.05, 0.4]),
            7,
            "magenta: {}",
            hue_of([0.4, 0.05, 0.4])
        );
    }

    #[test]
    fn profile_finds_sky_foliage_and_skin() {
        let sky = [0.08, 0.16, 0.35];
        let leaves = [0.05, 0.12, 0.02];
        let skin = [0.30, 0.18, 0.12];
        let grey = [0.1, 0.1, 0.1];
        let hues = Hues::measure(
            &image(&[(sky, 400), (leaves, 300), (skin, 100), (grey, 200)]),
            &IDENTITY,
        );
        let blue = hues.mass[4] + hues.mass[5];
        let green = hues.mass[2] + hues.mass[3];
        assert!(blue > 0.3 && blue < 0.41, "{hues:?}");
        assert!(green > 0.2 && green < 0.31, "{hues:?}");
        assert!(hues.skin > 0.08 && hues.skin < 0.11, "{hues:?}");
        assert!(hues.skin_bands[1] > 0.5, "skin is mostly orange: {hues:?}");
        assert!((hues.skin_bands.iter().sum::<f32>() - 1.0).abs() < 1e-4);
        assert_eq!(hues.skin_presence(), 1.0);
        assert!(hues.searched(5) && !hues.searched(7), "{hues:?}");
        assert_eq!(hues.presence(7), 0.0);
        assert!(hues.chroma[5] > 0.1, "{hues:?}");

        // Grey, black, clipped and broken pixels hold no colour.
        let none = Hues::measure(
            &image(&[
                (grey, 50),
                ([0.0; 3], 50),
                ([1.0, 0.99, 0.2], 50),
                ([f32::NAN, 0.1, 0.1], 5),
            ]),
            &IDENTITY,
        );
        assert_eq!((none.mass, none.skin), ([0.0; BANDS], 0.0));
        assert_eq!(none.skin_presence(), 0.0);
        assert_eq!(
            Hues::measure(&LinearImage::default(), &IDENTITY),
            Hues::default()
        );
    }
}
