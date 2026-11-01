//! Parametric tone curve: packs the candidate's CURVE region for
//! `cuda/tone_curve.cu`. Owner: tone curve.
//!
//! [`pack`] is called by `curves::pack` for every parameter set (thousands
//! per photo, on several threads), so it must stay cheap (tens of
//! microseconds at most) and write its whole region: the buffer is reused.
//!
//! Stub: writes the four region sliders / 100 and the fixed splits; the
//! kernel ignores them. To become: the curve itself, baked into a table the
//! kernel interpolates (Lightroom's region curve over the perceptual value,
//! splits at 25/50/75), with its CPU description and tests here, the way
//! `curves.rs` describes the Basic sliders.

use super::layout::CURVE_LEN;
use crate::types::{Params, TONE_CURVE_SPLITS};

/// Writes `p`'s tone curve into `out` (the CURVE region, `CURVE_LEN` floats).
pub(crate) fn pack(p: &Params, out: &mut [f32]) {
    let out = &mut out[..CURVE_LEN];
    out.fill(0.0);
    out[..4].copy_from_slice(&[
        p.curve_shadows / 100.0,
        p.curve_darks / 100.0,
        p.curve_lights / 100.0,
        p.curve_highlights / 100.0,
    ]);
    for (o, (_, split)) in out[4..7].iter_mut().zip(TONE_CURVE_SPLITS) {
        *o = split / 100.0;
    }
}
