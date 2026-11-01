//! Detail (sharpening, noise reduction) and Lens Corrections: the per-photo
//! values of those settings, and where they run around [`crate::render`].
//! Owner: detail (the GPU work lives in [`crate::detail`],
//! `cuda/detail.cu`, [`crate::lens`] and `cuda/lens.cu`; this file routes to
//! them).
//!
//! Lens Corrections act on linear camera RGB before anything is rendered,
//! where the camera's correction data is defined:
//! - [`develop_full`] for the export: lateral chromatic aberration on the
//!   RAW samples (a copy of the frame), then the demosaic, then distortion
//!   and vignetting;
//! - [`correct_scene`] for the search and display images every preview and
//!   statistic is rendered from: the same corrections at preview size, so
//!   previews match the export. The ops keep one corrected scene per photo
//!   and setting.
//!
//! Detail runs last, on the rendered sRGB: [`crate::render::render_full`]
//! calls [`finish_full`] on the full-resolution export, and
//! [`crate::render::RenderEngine::display`] calls [`finish_preview`] on every
//! display-size preview (what people and Claude look at), with radii scaled
//! to the preview. It replaces the fixed capture sharpening and chroma noise
//! reduction the export render used to apply (`render/full.rs`, `Finish`).
//! Images arrive unrotated (sensor orientation).
//!
//! The settings are not searched: [`auto_detail`] and [`auto_lens`] set them
//! once per photo from its metadata ([`with_auto`]), and `search::bounds`
//! fixes them in the search box, so every candidate carries them into its
//! previews, export and XMP. Whatever a [`Params`] says is what renders: a
//! person's own Detail or Lens values override the automatic ones.

use spl_core::Result;

use crate::decode::RawFrame;
use crate::detail::{self, DetailSettings};
use crate::develop;
use crate::lens::{self, LensSettings};
use crate::types::{Gpu, LinearImage, Params, RawInfo, Rgb8Image, Rgb16Image, Scene};

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

    /// Whether any sharpening or noise reduction is asked for (Radius,
    /// Detail and Masking only shape the sharpening).
    pub fn any(&self) -> bool {
        self.sharpening > 0.0 || self.luminance_nr > 0.0 || self.color_nr > 0.0
    }

    /// The same settings as the GPU module takes them.
    pub fn settings(&self) -> DetailSettings {
        DetailSettings {
            sharpen_amount: self.sharpening,
            sharpen_radius: self.radius,
            sharpen_detail: self.detail,
            sharpen_masking: self.masking,
            luminance_nr: self.luminance_nr,
            color_nr: self.color_nr,
        }
    }
}

impl From<DetailSettings> for Detail {
    fn from(s: DetailSettings) -> Self {
        Self {
            sharpening: s.sharpen_amount,
            radius: s.sharpen_radius,
            detail: s.sharpen_detail,
            masking: s.sharpen_masking,
            luminance_nr: s.luminance_nr,
            color_nr: s.color_nr,
        }
    }
}

/// What lens correction to apply to one photo, and what is known about its
/// lens.
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

    /// The switches as [`crate::lens`] takes them.
    pub fn settings(&self) -> LensSettings {
        LensSettings {
            profile: self.profile,
            lateral_ca: self.chromatic_aberration,
        }
    }
}

/// Detail settings for this photo, from its ISO and pixel count
/// ([`detail::auto_detail`]): Lightroom's defaults at base ISO, more noise
/// reduction and gentler sharpening as ISO rises.
pub fn auto_detail(info: &RawInfo) -> Detail {
    let megapixels = info.width as f32 * info.height as f32 / 1e6;
    detail::auto_detail(info.iso, megapixels).into()
}

/// Lens Corrections switches for this photo: (profile, chromatic
/// aberration). Both on for Sony files, which carry their lens's own
/// correction data ([`lens::lens_profile`]); off for everything else, which
/// the renderer cannot correct. A Sony file without the data renders
/// uncorrected, and Lightroom applies its own profile for the lens.
pub fn auto_lens(info: &RawInfo) -> (bool, bool) {
    let sony = info.make.to_ascii_lowercase().contains("sony");
    (sony, sony)
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

/// `scene`'s search and display images with the lens corrections `p`
/// switches on, all of them (chromatic aberration too: the previews come
/// from a half-size superpixel image, which keeps each colour where the
/// lens put it). `None` when nothing changes: no switch is on or the file
/// carries no correction data. A few milliseconds.
pub fn correct_scene(
    gpu: &Gpu,
    scene: &Scene,
    profile: Option<&lens::LensCorrection>,
    p: &Params,
) -> Result<Option<Scene>> {
    let settings = LensCorrection::new(p, &scene.info).settings();
    let Some(profile) = profile.filter(|c| corrects(c, settings)) else {
        return Ok(None);
    };
    Ok(Some(Scene {
        info: scene.info.clone(),
        color: scene.color.clone(),
        wb_as_shot: scene.wb_as_shot,
        search: lens::apply(gpu, &scene.search, profile, settings)?,
        display: lens::apply(gpu, &scene.display, profile, settings)?,
    }))
}

/// Full-resolution demosaic ([`develop::develop_full`]) with the lens
/// corrections `p` switches on: lateral chromatic aberration on a copy of
/// the RAW samples first (where a demosaic cannot hide half of it), then
/// distortion and vignetting on the demosaiced image. Matches
/// [`correct_scene`]'s previews.
pub fn develop_full(
    gpu: &Gpu,
    frame: &RawFrame,
    profile: Option<&lens::LensCorrection>,
    p: &Params,
) -> Result<LinearImage> {
    let settings = LensCorrection::new(p, &frame.info).settings();
    let Some(profile) = profile.filter(|c| corrects(c, settings)) else {
        return develop::develop_full(gpu, frame);
    };
    let mut ca_done = false;
    let full = if settings.lateral_ca && profile.ca.is_some() {
        let mut copy = copy_frame(frame);
        ca_done = lens::correct_mosaic(gpu, &mut copy, profile)?;
        develop::develop_full(gpu, &copy)?
    } else {
        develop::develop_full(gpu, frame)?
    };
    let rest = LensSettings {
        profile: settings.profile,
        lateral_ca: settings.lateral_ca && !ca_done,
    };
    if corrects(profile, rest) {
        lens::apply(gpu, &full, profile, rest)
    } else {
        Ok(full)
    }
}

/// Whether `settings` ask for a correction `profile` has data for.
fn corrects(profile: &lens::LensCorrection, settings: LensSettings) -> bool {
    (settings.profile && (profile.distortion.is_some() || profile.vignetting.is_some()))
        || (settings.lateral_ca && profile.ca.is_some())
}

fn copy_frame(frame: &RawFrame) -> RawFrame {
    RawFrame {
        info: frame.info.clone(),
        color: frame.color.clone(),
        wb_as_shot: frame.wb_as_shot,
        width: frame.width,
        height: frame.height,
        channels: frame.channels,
        cfa: frame.cfa,
        data: frame.data.clone(),
        black: frame.black,
        white: frame.white,
    }
}

/// Detail on the full-resolution export (16-bit sRGB, unrotated), in
/// place. Lens Corrections were applied before rendering ([`develop_full`]).
pub fn finish_full(
    gpu: &Gpu,
    img: &mut Rgb16Image,
    detail: &Detail,
    _lens: &LensCorrection,
) -> Result<()> {
    if detail.any() {
        *img = detail::apply(gpu, img, &detail.settings(), 1.0)?;
    }
    Ok(())
}

/// Detail on a display-size preview (8-bit sRGB, unrotated), in place.
/// `scale` is preview pixels per full-resolution pixel (below 1): radii
/// shrink with it, so the preview shows roughly what the export will. Lens
/// Corrections were applied to the scene the preview is rendered from
/// ([`correct_scene`]).
pub fn finish_preview(
    gpu: &Gpu,
    img: &mut Rgb8Image,
    detail: &Detail,
    _lens: &LensCorrection,
    scale: f32,
) -> Result<()> {
    if detail.any() {
        *img = detail::apply8(gpu, img, &detail.settings(), scale)?;
    }
    Ok(())
}

/// Lens corrections and Detail through the real pipeline on sample RAWs.
/// Run on the CUDA host: `gpu-test <dir> finish::gpu_tests --ignored`.
#[cfg(test)]
mod gpu_tests {
    use std::path::PathBuf;

    use super::*;
    use crate::decode::decode;
    use crate::encode;
    use crate::render::{RenderEngine, render_full};

    fn samples_dir() -> PathBuf {
        std::env::var_os("SPL_RAW_SAMPLES").map_or_else(|| PathBuf::from("/samples"), PathBuf::from)
    }

    fn mad(a: &Rgb8Image, b: &Rgb8Image) -> f64 {
        assert_eq!((a.width, a.height), (b.width, b.height));
        let sum: u64 = a
            .data
            .iter()
            .zip(&b.data)
            .map(|(&x, &y)| u64::from(x.abs_diff(y)))
            .sum();
        sum as f64 / a.data.len() as f64
    }

    /// With the automatic lens corrections on, the display preview (from
    /// [`correct_scene`]) and the export (from [`develop_full`]) show the
    /// same geometry and tones; the uncorrected export does not. Detail off,
    /// so only the lens path is compared.
    #[test]
    #[ignore = "needs the CUDA host and the sample RAWs"]
    fn gpu_lens_preview_matches_export() {
        let gpu = Gpu::open().expect("GPU");
        for file in ["DSC-RX100M3__DSC00734.ARW", "ILCE-6400__DSC00087.ARW"] {
            let bytes = std::fs::read(samples_dir().join(file)).expect("read sample");
            let frame = decode(&bytes).expect("decode");
            let profile = lens::lens_profile(&bytes).expect("Sony lens data");
            let scene = develop::develop_scene(&gpu, &frame).expect("develop_scene");
            let off = Detail {
                sharpening: 0.0,
                luminance_nr: 0.0,
                color_nr: 0.0,
                ..Detail::LIGHTROOM
            };
            let info = &scene.info;
            let p = off.apply(&with_auto(
                &Params::neutral(info.as_shot_temperature, info.as_shot_tint),
                info,
            ));
            assert!(LensCorrection::new(&p, info).any());
            let corrected = correct_scene(&gpu, &scene, Some(&profile), &p)
                .expect("correct_scene")
                .expect("the sample has lens data");
            let shown = RenderEngine::new(&gpu, &corrected)
                .and_then(|mut e| e.display(&p))
                .expect("display");
            let edge = shown.width.max(shown.height);
            let export = |scene: &Scene, full: &LinearImage| {
                let img = render_full(&gpu, scene, full, &p).expect("render_full");
                encode::resize(&encode::to_rgb8(&img), edge)
            };
            let fixed = export(
                &corrected,
                &develop_full(&gpu, &frame, Some(&profile), &p).expect("develop_full"),
            );
            let plain = export(
                &scene,
                &develop_full(&gpu, &frame, None, &p).expect("develop_full"),
            );
            let (same, other) = (mad(&fixed, &shown), mad(&plain, &shown));
            println!(
                "{file}: {}; corrected export vs preview MAD {same:.2}/255, uncorrected {other:.2}",
                profile.summary()
            );
            assert!(same < 2.5, "{file}: corrected export differs: {same}");
            assert!(
                other > 2.0 * same,
                "{file}: the correction should show: {other}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(make: &str, iso: Option<u32>, width: u32, height: u32) -> RawInfo {
        RawInfo {
            make: make.into(),
            model: "ILCE-7M3".into(),
            lens: Some("FE 24-70mm F2.8 GM".into()),
            width,
            height,
            orientation: 1,
            iso,
            exposure_time_s: None,
            f_number: None,
            focal_length_mm: None,
            captured_at: None,
            as_shot_temperature: 5200.0,
            as_shot_tint: 3.0,
            raw_clipped_fraction: 0.0,
        }
    }

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
        assert_eq!(Detail::from(d.settings()), d);
        assert_eq!(
            Detail::from(DetailSettings::default()).settings(),
            Detail::LIGHTROOM.settings()
        );
        assert!(Detail::LIGHTROOM.any());
        assert!(!Detail::from(DetailSettings::OFF).any());
    }

    #[test]
    fn auto_settings_follow_the_metadata() {
        let base = info("SONY", Some(100), 6000, 4000);
        assert_eq!(auto_detail(&base), Detail::LIGHTROOM);
        let night = auto_detail(&info("SONY", Some(6400), 6000, 4000));
        assert!(
            night.luminance_nr > 0.0 && night.color_nr > 25.0,
            "{night:?}"
        );
        assert!(night.sharpening <= 40.0);
        assert!(auto_detail(&info("SONY", Some(100), 9504, 6336)).radius < 1.0);

        assert_eq!(auto_lens(&base), (true, true));
        assert_eq!(auto_lens(&info("Sony", None, 6000, 4000)), (true, true));
        assert_eq!(auto_lens(&info("Canon", None, 6000, 4000)), (false, false));

        let n = Params::neutral(5200.0, 3.0);
        let p = with_auto(&n, &info("SONY", Some(6400), 6000, 4000));
        assert_eq!(p.basic(), n.basic());
        assert_eq!((p.lens_profile, p.lens_ca), (1.0, 1.0));
        assert_eq!(Detail::of(&p), night);
        assert_eq!(
            p.quantized(),
            p,
            "automatic values sit on Lightroom's steps"
        );
        let lens = LensCorrection::new(&p, &base);
        assert!(lens.any());
        assert_eq!(lens.settings(), LensSettings::AUTO);
        let off = with_auto(&n, &info("Canon", Some(100), 6000, 4000));
        assert!(!LensCorrection::new(&off, &base).any());
    }

    #[test]
    fn corrections_need_data_for_their_switch() {
        let vig = [
            11, 0, 320, 768, 1376, 2080, 2816, 3744, 4992, 6528, 8384, 10592,
        ];
        let only_vignetting = lens::LensCorrection::from_sony_tags(None, None, Some(&vig));
        let ca_only = LensSettings {
            profile: false,
            lateral_ca: true,
        };
        assert!(corrects(&only_vignetting, LensSettings::AUTO));
        assert!(!corrects(&only_vignetting, ca_only));
        let off = LensSettings {
            profile: false,
            lateral_ca: false,
        };
        assert!(!corrects(&only_vignetting, off));
    }
}
