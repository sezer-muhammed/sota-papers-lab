//! Parametric tone curve (Lightroom's Tone Curve panel, Region mode): packs
//! the candidate's CURVE region for `cuda/tone_curve.cu`. Owner: tone curve.
//!
//! The curve lives on the perceptual axis Lightroom draws it on: sRGB-encoded
//! values, 0 to 1, after the Basic panel's tone stage. The splits (25/50/75,
//! [`TONE_CURVE_SPLITS`]) cut it into four regions; each region slider moves
//! the curve at its region's centre by up to [`AMPLITUDE`] and fades out
//! toward the neighbouring centres (half its effect at the splits), so equal
//! settings in every region lift or lower the whole middle evenly. Black and
//! white stay put. The knots are joined by a monotone cubic (Fritsch and
//! Carlson 1980), and every amplitude is under the knot spacing, so no
//! setting can reverse tones.
//!
//! [`pack`] bakes the curve into a table the kernel only interpolates:
//! indexed by the square root of the display-linear value, holding the square
//! root of the output, so the identity is a straight line (exact under
//! interpolation) and the shadows get fine steps. The kernel applies it like
//! the Basic tone stage, to each pixel's largest and smallest channel with
//! the middle one keeping its relative place, so hue holds.
//!
//! [`pack`] is called by `curves::pack` for every parameter set (thousands
//! per photo, on several threads): about 10 microseconds when a slider is
//! set, nothing when all four are zero (the kernel then skips the stage).

use super::layout::CURVE_LEN;
use crate::types::{Params, TONE_CURVE_SPLITS};

/// Entries of the baked table (after the on/off flag in slot 0).
pub(crate) const CURVE_N: usize = 256;
/// How far each region slider at +-100 moves the curve at its region's
/// centre, in sRGB-encoded units: Shadows, Darks, Lights, Highlights. At most
/// the knot spacing (an eighth) minus a margin, so the knots always rise.
pub(crate) const AMPLITUDE: [f64; 4] = [0.085, 0.10, 0.10, 0.085];

const _: () = assert!(CURVE_N < CURVE_LEN);

/// sRGB encoding of a display-linear value.
fn oetf(v: f64) -> f64 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.003_130_8 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// Inverse of [`oetf`].
fn eotf(e: f64) -> f64 {
    let e = e.clamp(0.0, 1.0);
    if e <= 0.040_45 {
        e / 12.92
    } else {
        ((e + 0.055) / 1.055).powf(2.4)
    }
}

/// Nine knots on the encoded axis: black, the four region centres, the
/// three splits and white.
const KNOTS: usize = 9;

/// The parametric curve for one parameter set, on sRGB-encoded values.
#[derive(Debug, Clone)]
pub(crate) struct ParametricCurve {
    x: [f64; KNOTS],
    y: [f64; KNOTS],
    /// Slope at each knot (monotone cubic Hermite).
    m: [f64; KNOTS],
}

impl ParametricCurve {
    /// `None` when every region slider is zero (the identity).
    pub(crate) fn new(p: &Params) -> Option<Self> {
        let sliders = [
            p.curve_shadows,
            p.curve_darks,
            p.curve_lights,
            p.curve_highlights,
        ];
        if sliders.iter().all(|&v| v == 0.0 || !v.is_finite()) {
            return None;
        }
        let d: [f64; 4] = std::array::from_fn(|i| {
            let v = if sliders[i].is_finite() {
                sliders[i].clamp(-100.0, 100.0) as f64 / 100.0
            } else {
                0.0
            };
            AMPLITUDE[i] * v
        });
        let [s1, s2, s3] = TONE_CURVE_SPLITS.map(|(_, s)| s as f64 / 100.0);
        let x = [
            0.0,
            0.5 * s1,
            s1,
            0.5 * (s1 + s2),
            s2,
            0.5 * (s2 + s3),
            s3,
            0.5 * (s3 + 1.0),
            1.0,
        ];
        let offset = [
            0.0,
            d[0],
            0.5 * (d[0] + d[1]),
            d[1],
            0.5 * (d[1] + d[2]),
            d[2],
            0.5 * (d[2] + d[3]),
            d[3],
            0.0,
        ];
        let y: [f64; KNOTS] = std::array::from_fn(|k| x[k] + offset[k]);
        Some(Self {
            x,
            y,
            m: slopes(&x, &y),
        })
    }

    /// Encoded input (0 to 1) -> encoded output (0 to 1).
    pub(crate) fn eval(&self, e: f64) -> f64 {
        let e = e.clamp(0.0, 1.0);
        let k = self.x[1..KNOTS - 1]
            .iter()
            .take_while(|&&xk| xk <= e)
            .count();
        let h = self.x[k + 1] - self.x[k];
        let t = (e - self.x[k]) / h;
        let (t2, t3) = (t * t, t * t * t);
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;
        (h00 * self.y[k] + h10 * h * self.m[k] + h01 * self.y[k + 1] + h11 * h * self.m[k + 1])
            .clamp(0.0, 1.0)
    }

    /// Display-linear in -> display-linear out.
    pub(crate) fn apply_linear(&self, v: f64) -> f64 {
        eotf(self.eval(oetf(v)))
    }
}

/// Fritsch-Carlson slopes: monotone wherever the knots are.
fn slopes(x: &[f64; KNOTS], y: &[f64; KNOTS]) -> [f64; KNOTS] {
    let delta: [f64; KNOTS - 1] = std::array::from_fn(|k| (y[k + 1] - y[k]) / (x[k + 1] - x[k]));
    let mut m = [0.0; KNOTS];
    for k in 1..KNOTS - 1 {
        let (a, b) = (delta[k - 1], delta[k]);
        m[k] = if a * b <= 0.0 {
            0.0
        } else {
            // Weighted harmonic mean (Fritsch and Butland), equal spacing aside.
            let (h0, h1) = (x[k] - x[k - 1], x[k + 1] - x[k]);
            let (w1, w2) = (2.0 * h1 + h0, h1 + 2.0 * h0);
            (w1 + w2) / (w1 / a + w2 / b)
        };
    }
    // Ends: the three-point formula, kept to the side of the secant.
    let end = |h0: f64, h1: f64, d0: f64, d1: f64| {
        let s = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
        if s * d0 <= 0.0 {
            0.0
        } else if d0 * d1 <= 0.0 && s.abs() > 3.0 * d0.abs() {
            3.0 * d0
        } else {
            s
        }
    };
    m[0] = end(x[1] - x[0], x[2] - x[1], delta[0], delta[1]);
    m[KNOTS - 1] = end(
        x[KNOTS - 1] - x[KNOTS - 2],
        x[KNOTS - 2] - x[KNOTS - 3],
        delta[KNOTS - 2],
        delta[KNOTS - 3],
    );
    m
}

/// Writes `p`'s tone curve into `out` (the CURVE region, `CURVE_LEN` floats):
/// slot 0 is 1 when the curve is on, then [`CURVE_N`] entries of
/// sqrt(output) at sqrt(input) = i / (CURVE_N - 1), display-linear.
pub(crate) fn pack(p: &Params, out: &mut [f32]) {
    let out = &mut out[..CURVE_LEN];
    out.fill(0.0);
    let Some(curve) = ParametricCurve::new(p) else {
        return;
    };
    out[0] = 1.0;
    let step = 1.0 / (CURVE_N - 1) as f64;
    for (i, o) in out[1..1 + CURVE_N].iter_mut().enumerate() {
        let u = i as f64 * step;
        *o = curve.apply_linear(u * u).sqrt() as f32;
    }
}

#[cfg(test)]
mod tests;
