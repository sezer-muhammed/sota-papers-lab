//! Detail (sharpening, noise reduction) and Lens Corrections: what runs on
//! a rendered image after [`crate::render`], and the per-photo values of
//! those settings. Owner: detail (who also writes `src/detail.rs`,
//! `cuda/detail.cu` and `src/lens.rs`; this file only routes to them).
//!
//! Both render paths call in here: [`crate::render::render_full`] calls
//! [`finish_full`] on the full-resolution export, and
//! [`crate::render::RenderEngine::display`] calls [`finish_preview`] on
//! every display-size preview (what people and Claude look at). Images
//! arrive unrotated (sensor orientation), so lens geometry applies as is.
//!
//! The settings are not searched: [`auto_detail`] and [`auto_lens`] set them
//! once per photo from its metadata, and `search::bounds` fixes them in the
//! search box, so every candidate carries them into its previews, export and
//! XMP.
//!
//! Stubs today: both hooks leave the image alone, [`auto_detail`] returns
//! Lightroom's defaults and [`auto_lens`] switches both corrections off, so
//! output is unchanged. What they must become:
//! - [`auto_detail`]: sharpening and noise reduction from ISO and pixel
//!   count (PPR10K and FiveK usage in `research_notes/`), deterministic.
//! - [`auto_lens`]: profile corrections and CA removal on when the file
//!   carries what they need (the lens's correction data or a known profile).
//! - [`finish_full`] / [`finish_preview`]: Lightroom-like sharpening (amount,
//!   radius, detail, edge masking), luminance and colour noise reduction,
//!   and the lens corrections, on the GPU. The full-resolution render
//!   already applies a fixed capture sharpening and chroma noise reduction
//!   (`render/full.rs`, `Finish`); switch that off there once this replaces
//!   it, so the image is not sharpened twice.

use spl_core::Result;

use crate::types::{Gpu, Params, RawInfo, Rgb8Image, Rgb16Image};

/// Lightroom's Detail panel, in its units (the [`Params`] fields of the
/// same names).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detail {
    /// Sharpening Amount, 0 to 150.
    pub sharpening: f32,
    /// Sharpening Radius in pixels of the full-resolution image, 0.5 to 3.0.
    pub radius: f32,
    /// Sharpening Detail, 0 to 100.
    pub detail: f32,
    /// Sharpening Masking, 0 to 100.
    pub masking: f32,
    /// Luminance noise reduction, 0 to 100.
    pub luminance_nr: f32,
    /// Colour noise reduction, 0 to 100.
    pub color_nr: f32,
}

impl Detail {
    /// Lightroom's defaults for a RAW file.
    pub const LIGHTROOM: Self = Self {
        sharpening: 40.0,
        radius: 1.0,
        detail: 25.0,
        masking: 0.0,
        luminance_nr: 0.0,
        color_nr: 25.0,
    };

    pub fn of(p: &Params) -> Self {
        Self {
            sharpening: p.sharpening,
            radius: p.sharpen_radius,
            detail: p.sharpen_detail,
            masking: p.sharpen_masking,
            luminance_nr: p.luminance_nr,
            color_nr: p.color_nr,
        }
    }

    /// `p` with these Detail settings.
    pub fn apply(&self, p: &Params) -> Params {
        Params {
            sharpening: self.sharpening,
            sharpen_radius: self.radius,
            sharpen_detail: self.detail,
            sharpen_masking: self.masking,
            luminance_nr: self.luminance_nr,
            color_nr: self.color_nr,
            ..*p
        }
    }
}

/// What lens correction to apply to one photo, and what is known about its
/// lens. The lens owner extends this (and decoding) with the correction
/// data the file carries.
#[derive(Debug, Clone, PartialEq)]
pub struct LensCorrection {
    /// Enable Profile Corrections: distortion and vignetting.
    pub profile: bool,
    /// Remove Chromatic Aberration (lateral).
    pub chromatic_aberration: bool,
    pub make: String,
    pub model: String,
    pub lens: Option<String>,
    pub focal_length_mm: Option<f32>,
    pub f_number: Option<f32>,
    /// Full-resolution size, unrotated: the frame the corrections are
    /// defined on. A preview is the same frame scaled down.
    pub width: u32,
    pub height: u32,
}

impl LensCorrection {
    pub fn new(p: &Params, info: &RawInfo) -> Self {
        Self {
            profile: p.lens_profile >= 0.5,
            chromatic_aberration: p.lens_ca >= 0.5,
            make: info.make.clone(),
            model: info.model.clone(),
            lens: info.lens.clone(),
            focal_length_mm: info.focal_length_mm,
            f_number: info.f_number,
            width: info.width,
            height: info.height,
        }
    }

    /// Whether anything is to be corrected.
    pub fn any(&self) -> bool {
        self.profile || self.chromatic_aberration
    }
}

/// Detail settings for this photo. Stub: Lightroom's defaults.
pub fn auto_detail(_info: &RawInfo) -> Detail {
    Detail::LIGHTROOM
}

/// Lens Corrections switches for this photo: (profile, chromatic
/// aberration). Stub: both off.
pub fn auto_lens(_info: &RawInfo) -> (bool, bool) {
    (false, false)
}

/// `p` with this photo's automatic Detail and Lens Corrections settings.
pub fn with_auto(p: &Params, info: &RawInfo) -> Params {
    let (profile, ca) = auto_lens(info);
    let on = |b: bool| if b { 1.0 } else { 0.0 };
    Params {
        lens_profile: on(profile),
        lens_ca: on(ca),
        ..auto_detail(info).apply(p)
    }
}

/// Detail and lens corrections on the full-resolution export (16-bit sRGB,
/// unrotated), in place. Stub: leaves the image alone.
pub fn finish_full(
    _gpu: &Gpu,
    _img: &mut Rgb16Image,
    _detail: &Detail,
    _lens: &LensCorrection,
) -> Result<()> {
    Ok(())
}

/// Detail and lens corrections on a display-size preview (8-bit sRGB,
/// unrotated), in place. `scale` is preview pixels per full-resolution pixel
/// (below 1), so radii in full-resolution pixels can be scaled to match the
/// export. Stub: leaves the image alone.
pub fn finish_preview(
    _gpu: &Gpu,
    _img: &mut Rgb8Image,
    _detail: &Detail,
    _lens: &LensCorrection,
    _scale: f32,
) -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detail_round_trips_through_params() {
        let n = Params::neutral(5200.0, 3.0);
        assert_eq!(Detail::of(&n), Detail::LIGHTROOM);
        let d = Detail {
            sharpening: 70.0,
            radius: 1.2,
            detail: 30.0,
            masking: 20.0,
            luminance_nr: 15.0,
            color_nr: 30.0,
        };
        let p = d.apply(&n);
        assert_eq!(Detail::of(&p), d);
        assert_eq!(p.basic(), n.basic());
    }
}
