//! Color Mixer (HSL) and Calibration: packs the candidate's MIXER region for
//! `cuda/color_mixer.cu`. Owner: colour mixer.
//!
//! [`pack`] is called by `curves::pack` for every parameter set (thousands
//! per photo, on several threads), so it must stay cheap and write its whole
//! region: the buffer is reused.
//!
//! Calibration's red, green and blue primaries act on the camera matrices
//! here on the CPU ([`calibrate`], called by `curves::pack` before white
//! balance is folded in), which costs the kernels nothing; its shadows tint
//! and the Color Mixer act per pixel in `color_mixer_apply`.
//!
//! Stubs: [`calibrate`] leaves the matrix alone and [`pack`] writes the
//! sliders / 100 in the order `cuda/color_mixer.cu` documents; the kernel
//! ignores them. To become: the primaries' hue and saturation as a change
//! of the camera -> linear Rec.2020 matrix that keeps neutrals neutral (rows
//! still summing to one), whatever the kernel needs precomputed (band
//! weights), and CPU tests of the colour behaviour here.

use super::layout::MIXER_LEN;
use crate::types::Params;

/// Calibration's primaries applied to `m`, the camera -> linear Rec.2020
/// matrix (row-major) at `p`'s white balance. Stub: leaves `m` alone.
pub(crate) fn calibrate(_p: &Params, _m: &mut [[f32; 3]; 3]) {}

/// Writes `p`'s Color Mixer and Calibration into `out` (the MIXER region,
/// `MIXER_LEN` floats).
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
}
