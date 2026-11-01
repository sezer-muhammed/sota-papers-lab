//! An analytic stand-in for the GPU renderer so the search can be tested on
//! the CPU. A scene is a sorted set of linear luminances (logistic in log
//! space) plus a few colour numbers; the "render" is a tone pipeline that
//! reacts to every slider the way a real one does (exposure shifts the
//! percentiles, highlights pull the top and the clipping down, saturation
//! raises chroma and gamut excursions, white balance moves the cast, the
//! tone curve bends its region, clarity and texture raise local contrast,
//! dehaze deepens the blacks, a band's saturation raises chroma by the
//! band's share of the image) and costs microseconds.

use spl_core::Result;

use super::hues::BANDS;
use super::priors::{
    BLUE_SAT, CLARITY, CURVE_DARKS, CURVE_HIGHLIGHTS, CURVE_LIGHTS, CURVE_SHADOWS, DEHAZE,
    GREEN_SAT, LUM, RED_SAT, SAT, TEXTURE,
};
use super::{Evaluator, Hues, warm_shift};
use crate::color::ColorModel;
use crate::objective::smoothstep;
use crate::types::{LinearImage, Params, RawInfo, Scene, Stats};

const SAMPLES: usize = 64;
/// Baseline exposure: the RAW clip point lands exactly on the curve's white.
const BASE_GAIN: f32 = 1.5;

pub(crate) struct Synthetic {
    pub(crate) as_shot_temperature: f32,
    pub(crate) as_shot_tint: f32,
    pub(crate) iso: Option<u32>,
    lum: Vec<f32>,
    raw_clipped: f32,
    chroma: f32,
    cast: (f32, f32),
    /// Coloured samples: linear Rec.2020 tints (luminance 1) by sample index.
    tints: Vec<Option<[f32; 3]>>,
    /// The colour profile the scene's search image measures.
    hues: Hues,
    /// Fail every call after this many (None = never), to test error paths.
    pub(crate) fail_after: Option<usize>,
    /// Return NaN stats for parameter sets with exposure above this.
    pub(crate) nan_above: Option<f32>,
    pub(crate) calls: usize,
}

impl Synthetic {
    /// Linear luminance with this `median` (1.0 = RAW clip) and `spread`
    /// (stops per logistic unit; 0.9 is an ordinary scene, 1.2 backlight),
    /// mean Oklab `chroma` and an as-shot grey `cast` (Oklab a, b).
    pub(crate) fn new(median: f32, spread: f32, chroma: f32, cast: (f32, f32)) -> Self {
        let lum: Vec<f32> = (0..SAMPLES)
            .map(|i| {
                let q = (i as f32 + 0.5) / SAMPLES as f32;
                (median * (spread * (q / (1.0 - q)).ln()).exp2()).min(1.0)
            })
            .collect();
        let raw_clipped = lum.iter().filter(|v| **v >= 1.0).count() as f32 / SAMPLES as f32;
        Self {
            as_shot_temperature: 5200.0,
            as_shot_tint: 8.0,
            iso: Some(400),
            lum,
            raw_clipped,
            chroma,
            cast,
            tints: vec![None; SAMPLES],
            hues: Hues::default(),
            fail_after: None,
            nan_above: None,
            calls: 0,
        }
    }

    /// An ordinary daylight scene, slightly underexposed, with a bright sky
    /// that clips at neutral settings.
    pub(crate) fn daylight() -> Self {
        Self::new(0.07, 0.95, 0.05, (0.002, 0.006))
    }

    /// Daylight with colour: a blue sky in the bright quarter (below the
    /// clipped samples), foliage in the dark third and some skin between.
    pub(crate) fn colourful() -> Self {
        Self::daylight().with_tints(&[
            (44..58, [0.55, 1.05, 2.4]),
            (0..18, [0.7, 1.25, 0.35]),
            (26..34, [1.9, 0.75, 0.5]),
        ])
    }

    /// The samples in each range tinted (a linear Rec.2020 colour scaled to
    /// the sample's luminance), so the scene's image holds those colours.
    pub(crate) fn with_tints(mut self, tints: &[(std::ops::Range<usize>, [f32; 3])]) -> Self {
        for (range, rgb) in tints {
            let y = 0.2627 * rgb[0] + 0.678 * rgb[1] + 0.0593 * rgb[2];
            for i in range.clone() {
                self.tints[i] = Some(rgb.map(|c| c / y));
            }
        }
        self.hues = Hues::of(&self.scene());
        self
    }

    /// A scene whose search image holds these luminances (grey or tinted
    /// pixels), so `bounds` reads the same key this stand-in renders.
    pub(crate) fn scene(&self) -> Scene {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let color = ColorModel::new(identity, None);
        let m = color.rec2020_from_cam(self.as_shot_temperature);
        let cam_from =
            crate::color::invert(&m.map(|r| r.map(f64::from))).expect("invertible camera matrix");
        let data = self
            .lum
            .iter()
            .zip(&self.tints)
            .flat_map(|(v, tint)| match tint {
                None => [*v; 3],
                Some(rgb) => {
                    let rgb = rgb.map(|c| f64::from(c * *v));
                    let cam: [f64; 3] =
                        std::array::from_fn(|r| (0..3).map(|c| cam_from[r][c] * rgb[c]).sum());
                    cam.map(|c| (c as f32).clamp(0.0, 0.97))
                }
            })
            .collect();
        Scene {
            info: RawInfo {
                make: "Sony".into(),
                model: "ILCE-7RM3".into(),
                lens: None,
                width: 7952,
                height: 5304,
                orientation: 1,
                iso: self.iso,
                exposure_time_s: Some(0.004),
                f_number: Some(8.0),
                focal_length_mm: Some(35.0),
                captured_at: None,
                as_shot_temperature: self.as_shot_temperature,
                as_shot_tint: self.as_shot_tint,
                raw_clipped_fraction: self.raw_clipped,
            },
            color,
            wb_as_shot: [2.0, 1.0, 1.6],
            search: LinearImage {
                width: 8,
                height: SAMPLES / 8,
                data,
            },
            display: LinearImage::default(),
        }
    }

    pub(crate) fn render(&self, p: &Params) -> Stats {
        if self.nan_above.is_some_and(|ev| p.exposure > ev) {
            return Stats {
                luma_p50: f32::NAN,
                ..Stats::default()
            };
        }
        let gain = BASE_GAIN * p.exposure.exp2();
        let white2 = BASE_GAIN * BASE_GAIN;
        let (h, s, w, b, c) = (
            p.highlights / 100.0,
            p.shadows / 100.0,
            p.whites / 100.0,
            p.blacks / 100.0,
            p.contrast / 100.0,
        );
        let a = p.to_array();
        let curve = [
            a[CURVE_SHADOWS],
            a[CURVE_DARKS],
            a[CURVE_LIGHTS],
            a[CURVE_HIGHLIGHTS],
        ]
        .map(|v| v / 100.0);
        let dehaze = a[DEHAZE] / 100.0;
        // Band luminance moves the mean by each band's share.
        let band_lum: f32 = (0..BANDS)
            .map(|k| self.hues.mass[k] * a[LUM + k] / 100.0)
            .sum();
        let tone = |lum: f32| {
            let y = lum * gain;
            let t = (y * (1.0 + y / white2) / (1.0 + y)).min(1.0);
            let mut v = t.max(0.0).powf(1.0 / 2.2);
            v += h * 0.18 * smoothstep(0.45, 1.0, v);
            v += s * 0.16 * (1.0 - smoothstep(0.05, 0.55, v)) * smoothstep(0.0, 0.04, v);
            v += w * 0.12 * smoothstep(0.6, 1.0, v);
            v += b * 0.08 * (1.0 - smoothstep(0.0, 0.3, v));
            v = 0.45 + (v - 0.45) * (1.0 + 0.4 * c);
            // Tone curve regions: a bump over each quarter of the range.
            for (r, amount) in curve.iter().enumerate() {
                let centre = 0.125 + 0.25 * r as f32;
                v += amount * 0.08 * (1.0 - ((v - centre) / 0.25).abs()).max(0.0);
            }
            v -= dehaze * 0.08 * (1.0 - smoothstep(0.0, 0.5, v));
            v += 0.1 * band_lum;
            v.clamp(0.0, 1.0)
        };
        let out: Vec<(f32, bool)> = self.lum.iter().map(|l| (tone(*l), *l >= 1.0)).collect();
        let mut sorted: Vec<f32> = out.iter().map(|(v, _)| *v).collect();
        sorted.sort_by(f32::total_cmp);
        let n = SAMPLES as f32;
        let pct = |q: f32| {
            let x = (q * n - 0.5).clamp(0.0, n - 1.0);
            let i = x.floor() as usize;
            let j = (i + 1).min(SAMPLES - 1);
            sorted[i] + (sorted[j] - sorted[i]) * (x - i as f32)
        };
        let p50 = pct(0.5);
        let mean = sorted.iter().sum::<f32>() / n;

        // Chroma and colourfulness scale with brightness, as they do on
        // display-encoded output.
        let vibrance = p.vibrance / 100.0;
        let saturation = 1.0 + p.saturation / 100.0;
        let room = (1.0 - self.chroma / 0.2).clamp(0.0, 1.0);
        let lightness = (mean / 0.4).clamp(0.05, 2.0);
        let bands: f32 = (0..BANDS)
            .map(|k| self.hues.mass[k] * a[SAT + k] / 100.0)
            .sum();
        let calibration = 0.004 * a[BLUE_SAT] + 0.002 * (a[RED_SAT] + a[GREEN_SAT]);
        let common = saturation
            * (1.0 + 0.15 * c + 0.15 * dehaze)
            * (1.0 + bands + calibration).max(0.1)
            * lightness;
        let chroma_mean = self.chroma * common * (1.0 + 0.6 * vibrance * room);
        let chroma_p95 = 2.4 * self.chroma * common * (1.0 + 0.25 * vibrance);

        // A colourful pixel's brightest channel clips before its luma does.
        let channel_max = |v: f32| v + 0.5 * chroma_mean;
        let share = |f: &dyn Fn(f32, bool) -> bool| {
            out.iter().filter(|(v, raw)| f(*v, *raw)).count() as f32 / n
        };
        let clip_hi = share(&|v, _| channel_max(v) >= 254.0 / 255.0);
        let clip_hi_new = share(&|v, raw| !raw && channel_max(v) >= 254.0 / 255.0);
        let clip_lo = share(&|v, _| v <= 2.0 / 255.0);

        let mut hist = vec![0.0f32; 64];
        for v in &sorted {
            hist[((v * 64.0) as usize).min(63)] += 1.0 / n;
        }
        let std = (sorted.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n).sqrt();
        let entropy = -hist
            .iter()
            .filter(|p| **p > 0.0)
            .map(|p| p * p.log2())
            .sum::<f32>();
        let structure =
            0.86 - 2.0 * clip_lo - 2.5 * clip_hi_new + 0.03 * (-h).max(0.0) + 0.02 * s.max(0.0)
                - 0.04 * c.abs()
                - 0.02 * (h.max(0.0) + (-s).max(0.0));

        Stats {
            luma_mean: mean,
            luma_p01: pct(0.01),
            luma_p05: pct(0.05),
            luma_p50: p50,
            luma_p95: pct(0.95),
            luma_p99: pct(0.99),
            luma_hist: hist,
            clip_hi,
            clip_hi_new,
            clip_lo,
            colorfulness: 450.0 * chroma_mean,
            chroma_mean,
            chroma_p95,
            out_of_gamut: 0.002 + 0.8 * (chroma_p95 - 0.16).max(0.0) + 0.3 * clip_hi,
            cast_a: self.cast.0 + 0.0006 * (p.tint - self.as_shot_tint),
            cast_b: self.cast.1 + 0.0005 * warm_shift(self.as_shot_temperature, p.temperature),
            structure: structure.clamp(0.0, 1.0),
            local_contrast: 0.3
                * std
                * (1.0 + 0.25 * c + 0.4 * a[CLARITY] / 100.0 + 0.2 * a[TEXTURE] / 100.0).max(0.1),
            entropy,
        }
    }
}

impl Evaluator for Synthetic {
    fn stats(&mut self, params: &[Params]) -> Result<Vec<Stats>> {
        self.calls += 1;
        if self.fail_after.is_some_and(|n| self.calls > n) {
            spl_core::bail!("CUDA_ERROR_LAUNCH_FAILED (synthetic)");
        }
        Ok(params.iter().map(|p| self.render(p)).collect())
    }
}
