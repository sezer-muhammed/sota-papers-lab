//! Lens correction from the camera's own profile, on the GPU
//! (`cuda/lens.cu`). Owner: lens.
//!
//! Sony bodies write the lens's correction splines into every ARW (tags
//! 0x7037 distortion, 0x7035 lateral chromatic aberration and 0x7032
//! vignetting in the RAW IFD) whether or not the camera applied them to its
//! own JPEG; the RAW data is always uncorrected. rawler names the tags but
//! does not apply them, so [`lens_profile`] reads them with rawler's TIFF
//! reader and decodes them the way darktable (`iop/lens.cc`, metadata mode)
//! and RawTherapee (`rtengine/lensmetadata.cc`) do:
//!
//! - each array starts with its knot count `n` (2 to 16; chromatic
//!   aberration holds `2n` values, red then blue), knot `i` sits at radius
//!   `(i + 0.5) / (n - 1)` of the half-diagonal, and values are linearly
//!   interpolated and held flat past the end knots;
//! - distortion is a radial scale `1 + v / 2^14`: the output pixel at radius
//!   `r` reads the source at `r * D(r)`; red and blue read at
//!   `r * D(r) * (1 + v / 2^21)`;
//! - vignetting is a gain `2^(2^(v / 2^13 - 1) - 0.5)` that multiplies the
//!   source pixel.
//!
//! Wiring, in pipeline order:
//!
//! 1. [`lens_profile`] on the file's bytes (a fraction of a millisecond).
//! 2. [`correct_mosaic`] on the decoded frame, before `develop_scene` and
//!    `develop_full`: lateral chromatic aberration belongs on the RAW samples
//!    (measured on the samples, a colour-difference demosaic hides about half
//!    of it, so correcting afterwards over-corrects).
//! 3. [`apply`] (or [`apply_device`] inside `develop_full`) with
//!    `lateral_ca: false` on the demosaiced image and the previews:
//!    distortion and vignetting.
//!
//! [`apply`] works on linear camera RGB ([`LinearImage`]) of any size, since
//! radii are relative to the image's own half-diagonal: the full-resolution
//! demosaic and the search and display previews correct identically.
//! Vignetting gains lift corners above 1.0 where the scene was bright there;
//! values are left unclamped (the renderer accepts them, and clipped pixels
//! stay at or above 1.0 in every channel). Distortion is a resampling warp
//! scaled by [`LensCorrection::zoom`] so that no output pixel reads outside
//! the frame: the result has no empty borders and keeps as much of the frame
//! as that allows.

use std::panic::{AssertUnwindSafe, catch_unwind};

use rawler::formats::tiff::GenericTiffReader;
use rawler::formats::tiff::reader::TiffReader;
use rawler::tags::ExifTag;
use spl_core::{Result, anyhow};
use spl_gpu::cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use crate::decode::RawFrame;
use crate::types::{Gpu, LinearImage};

const SOURCE: &str = include_str!("../cuda/lens.cu");

/// Rows of the GPU lookup table and the radius it spans (relative to the
/// half-diagonal; zoomed-out corners and their source radii stay below it).
const LUT_N: usize = 1024;
const LUT_RMAX: f32 = 1.75;
/// Boundary points per image edge tested when choosing the zoom.
const EDGE_STEPS: usize = 256;

/// One radial curve: values at knots (radius over the half-diagonal),
/// linearly interpolated and held flat outside the end knots.
#[derive(Debug, Clone, PartialEq)]
pub struct Curve {
    knots: Vec<f32>,
    values: Vec<f32>,
}

impl Curve {
    /// Sony's knot layout: `n` values at `(i + 0.5) / (n - 1)`.
    fn sony(values: Vec<f32>) -> Option<Self> {
        let n = values.len();
        if !(2..=16).contains(&n) || values.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let knots = (0..n).map(|i| (i as f32 + 0.5) / (n - 1) as f32).collect();
        Some(Self { knots, values })
    }

    pub fn at(&self, r: f32) -> f32 {
        let (k, v) = (&self.knots, &self.values);
        if r <= k[0] {
            return v[0];
        }
        let i = k.partition_point(|&x| x <= r);
        if i >= k.len() {
            return v[v.len() - 1];
        }
        let t = (r - k[i - 1]) / (k[i] - k[i - 1]);
        v[i - 1] + t * (v[i] - v[i - 1])
    }

    fn is_identity(&self) -> bool {
        self.values.iter().all(|&v| (v - 1.0).abs() < 1e-6)
    }

    fn range(&self) -> (f32, f32) {
        self.values
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)))
    }
}

/// A lens's correction curves, as read from the file. Missing or trivial
/// parts are `None`.
#[derive(Debug, Clone, PartialEq)]
pub struct LensCorrection {
    /// Source radius over output radius, all channels.
    pub distortion: Option<Curve>,
    /// Extra radial scale of red and of blue relative to green.
    pub ca: Option<[Curve; 2]>,
    /// Gain that multiplies the source pixel.
    pub vignetting: Option<Curve>,
}

/// Which corrections to apply: Lightroom's `crs:LensProfileEnable`
/// (distortion and vignetting) and `crs:AutoLateralCA`. When CA was already
/// corrected on the RAW samples ([`correct_mosaic`], the better place), pass
/// `lateral_ca: false` here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LensSettings {
    pub profile: bool,
    pub lateral_ca: bool,
}

impl LensSettings {
    pub const AUTO: Self = Self {
        profile: true,
        lateral_ca: true,
    };
}

/// The camera's embedded correction, or `None` when the file carries none
/// (non-Sony files, A-mount adapters without data, malformed tags).
pub fn lens_profile(bytes: &[u8]) -> Option<LensCorrection> {
    // rawler panics on some malformed TIFF structures; a bad file simply has no profile.
    catch_unwind(AssertUnwindSafe(|| read_sony(bytes))).ok()?
}

fn read_sony(bytes: &[u8]) -> Option<LensCorrection> {
    if bytes.len() < 1024 {
        return None;
    }
    let tiff = GenericTiffReader::new_with_buffer(bytes, 0, 0, None).ok()?;
    let values = |tag: ExifTag| -> Option<Vec<i32>> {
        let ifd = tiff.find_first_ifd_with_tag(tag)?;
        let value = &ifd.get_entry(tag)?.value;
        (0..value.count())
            .map(|i| value.get_i32(i).ok().flatten())
            .collect()
    };
    let correction = LensCorrection::from_sony_tags(
        values(ExifTag::DistortionCorrParams).as_deref(),
        values(ExifTag::ChromaticAberrationCorrParams).as_deref(),
        values(ExifTag::VignettingCorrParams).as_deref(),
    );
    (!correction.is_empty()).then_some(correction)
}

impl LensCorrection {
    /// Decodes the three Sony arrays (each led by its knot count); any of
    /// them may be absent or invalid.
    pub fn from_sony_tags(
        distortion: Option<&[i32]>,
        ca: Option<&[i32]>,
        vignetting: Option<&[i32]>,
    ) -> Self {
        let curve = |raw: Option<&[i32]>, f: &dyn Fn(f32) -> f32| {
            let raw = raw?;
            let n = usize::try_from(*raw.first()?).ok()?;
            let values: Vec<f32> = raw.get(1..1 + n)?.iter().map(|&v| f(v as f32)).collect();
            Curve::sony(values).filter(|c| !c.is_identity())
        };
        let scale = |v: f32| 1.0 + v / 16384.0;
        let distortion = curve(distortion, &scale)
            .filter(|c| matches!(c.range(), (lo, hi) if lo > 0.6 && hi < 1.5));
        let ca = ca.and_then(|raw| {
            let n2 = usize::try_from(*raw.first()?).ok()?;
            if n2 % 2 != 0 || raw.len() < 1 + n2 {
                return None;
            }
            let n = n2 / 2;
            let fine = |v: i32| 1.0 + v as f32 / 2_097_152.0;
            let red = Curve::sony(raw[1..1 + n].iter().map(|&v| fine(v)).collect())?;
            let blue = Curve::sony(raw[1 + n..1 + n2].iter().map(|&v| fine(v)).collect())?;
            let sane = |c: &Curve| matches!(c.range(), (lo, hi) if lo > 0.95 && hi < 1.05);
            (sane(&red) && sane(&blue) && !(red.is_identity() && blue.is_identity()))
                .then_some([red, blue])
        });
        let gain = |v: f32| (2f32.powf(v / 8192.0 - 1.0) - 0.5).exp2();
        let vignetting = curve(vignetting, &gain)
            .filter(|c| matches!(c.range(), (lo, hi) if lo > 0.5 && hi < 16.0));
        Self {
            distortion,
            ca,
            vignetting,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.distortion.is_none() && self.ca.is_none() && self.vignetting.is_none()
    }

    /// Source radius over output radius of channel `c` (0 = R, 1 = G, 2 = B)
    /// at normalized output radius `r`, before zoom.
    fn scale(&self, settings: LensSettings, c: usize, r: f32) -> f32 {
        let d = match (&self.distortion, settings.profile) {
            (Some(curve), true) => curve.at(r),
            _ => 1.0,
        };
        let ca = match (&self.ca, settings.lateral_ca, c) {
            (Some([red, _]), true, 0) => red.at(r),
            (Some([_, blue]), true, 2) => blue.at(r),
            _ => 1.0,
        };
        d * ca
    }

    fn gain(&self, settings: LensSettings, r: f32) -> f32 {
        match (&self.vignetting, settings.profile) {
            (Some(curve), true) => curve.at(r),
            _ => 1.0,
        }
    }

    fn warps(&self, settings: LensSettings) -> bool {
        (settings.profile && self.distortion.is_some())
            || (settings.lateral_ca && self.ca.is_some())
    }

    fn does_anything(&self, settings: LensSettings) -> bool {
        self.warps(settings) || (settings.profile && self.vignetting.is_some())
    }

    /// Magnification of the corrected image for a `width` x `height` frame:
    /// the smallest one (largest field of view) at which every output pixel,
    /// in every channel, reads inside the source frame. 1 without a warp.
    pub fn zoom(&self, width: usize, height: usize, settings: LensSettings) -> f32 {
        if !self.warps(settings) || width < 2 || height < 2 {
            return 1.0;
        }
        let (cx, cy) = ((width - 1) as f32 / 2.0, (height - 1) as f32 / 2.0);
        let inv_half_diag = 1.0 / (width as f32 / 2.0).hypot(height as f32 / 2.0);
        let tol = 1e-3;
        // `u` is 1 / zoom: the output offset from the centre is multiplied by it.
        let inside = |u: f32| {
            (0..=EDGE_STEPS).all(|i| {
                let t = i as f32 / EDGE_STEPS as f32;
                let (ex, ey) = (cx * (2.0 * t - 1.0), cy * (2.0 * t - 1.0));
                [(ex, -cy), (ex, cy), (-cx, ey), (cx, ey)]
                    .into_iter()
                    .all(|(dx, dy)| {
                        let (dx, dy) = (dx * u, dy * u);
                        let r = dx.hypot(dy) * inv_half_diag;
                        (0..3).all(|c| {
                            let s = self.scale(settings, c, r);
                            (dx * s).abs() <= cx + tol && (dy * s).abs() <= cy + tol
                        })
                    })
            })
        };
        let (mut lo, mut hi) = (0.5f32, 2.0f32);
        if !inside(lo) {
            return 1.0 / lo;
        }
        if inside(hi) {
            return 1.0 / hi;
        }
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            if inside(mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        1.0 / lo
    }

    /// (S_r, S_g, S_b, gain) at LUT_N radii on [0, LUT_RMAX].
    fn lut(&self, settings: LensSettings) -> Vec<f32> {
        (0..LUT_N)
            .flat_map(|i| {
                let r = i as f32 * LUT_RMAX / (LUT_N - 1) as f32;
                [
                    self.scale(settings, 0, r),
                    self.scale(settings, 1, r),
                    self.scale(settings, 2, r),
                    self.gain(settings, r),
                ]
            })
            .collect()
    }

    /// One line for logs: what the profile does at the corner.
    pub fn summary(&self) -> String {
        let corner = 1.0;
        let mut parts = Vec::new();
        if let Some(d) = &self.distortion {
            let (lo, hi) = d.range();
            parts.push(format!(
                "distortion {:+.2}% at the corner (scale {lo:.3}..{hi:.3})",
                (d.at(corner) / d.at(0.0) - 1.0) * 100.0
            ));
        }
        if let Some([red, blue]) = &self.ca {
            parts.push(format!(
                "CA red {:+.3}% blue {:+.3}% at the corner",
                (red.at(corner) - 1.0) * 100.0,
                (blue.at(corner) - 1.0) * 100.0
            ));
        }
        if let Some(v) = &self.vignetting {
            parts.push(format!(
                "vignetting {:+.2} EV at the corner",
                v.at(corner).log2()
            ));
        }
        if parts.is_empty() {
            "no correction".to_string()
        } else {
            parts.join(", ")
        }
    }
}

/// Corrects `img` (linear camera RGB). Returns a copy when there is nothing
/// to do. GPU memory: the image twice (about 1.5 GB at 61 MP).
pub fn apply(
    gpu: &Gpu,
    img: &LinearImage,
    correction: &LensCorrection,
    settings: LensSettings,
) -> Result<LinearImage> {
    if img.data.len() != 3 * img.width * img.height {
        return Err(anyhow!(
            "lens: image is malformed: {}x{} with {} samples",
            img.width,
            img.height,
            img.data.len()
        ));
    }
    if !correction.does_anything(settings) || img.width < 2 || img.height < 2 {
        return Ok(img.clone());
    }
    let src = gpu
        .stream
        .clone_htod(&img.data)
        .map_err(gpu_err("upload image"))?;
    let dst = apply_device(gpu, &src, img.width, img.height, correction, settings)?
        .ok_or_else(|| anyhow!("lens: correction unexpectedly empty"))?;
    drop(src);
    let data = gpu
        .stream
        .clone_dtoh(&dst)
        .map_err(gpu_err("download image"))?;
    Ok(LinearImage {
        width: img.width,
        height: img.height,
        data,
    })
}

/// [`apply`] on an image already on the GPU (`width * height` interleaved
/// RGB floats), for use inside a develop pipeline without a host round
/// trip. `None` when there is nothing to do: keep using `src`.
pub fn apply_device(
    gpu: &Gpu,
    src: &CudaSlice<f32>,
    width: usize,
    height: usize,
    correction: &LensCorrection,
    settings: LensSettings,
) -> Result<Option<CudaSlice<f32>>> {
    if src.len() < 3 * width * height {
        return Err(anyhow!(
            "lens: buffer holds {} floats, {width}x{height} needs {}",
            src.len(),
            3 * width * height
        ));
    }
    if !correction.does_anything(settings) || width < 2 || height < 2 {
        return Ok(None);
    }
    if i32::try_from(3 * width * height).is_err() {
        return Err(anyhow!("lens: image too large: {width}x{height}"));
    }
    let module = gpu.module("lens", SOURCE)?;
    let kernel = module
        .load_function("lens_kernel")
        .map_err(|e| anyhow!("load kernel lens_kernel: {e:?}"))?;
    let s = &gpu.stream;
    let lut = s
        .clone_htod(&correction.lut(settings))
        .map_err(gpu_err("upload lens table"))?;
    // SAFETY: lens_kernel writes every element of `dst` before anything reads it.
    let mut dst =
        unsafe { s.alloc::<f32>(3 * width * height) }.map_err(gpu_err("allocate image"))?;
    let (w, h, n) = (width as i32, height as i32, LUT_N as i32);
    let inv_half_diag = 1.0 / (width as f32 / 2.0).hypot(height as f32 / 2.0);
    let inv_zoom = 1.0 / correction.zoom(width, height, settings);
    let mut launch = s.launch_builder(&kernel);
    launch
        .arg(src)
        .arg(&mut dst)
        .arg(&w)
        .arg(&h)
        .arg(&lut)
        .arg(&n)
        .arg(&LUT_RMAX)
        .arg(&inv_half_diag)
        .arg(&inv_zoom);
    let cfg = LaunchConfig {
        grid_dim: (width.div_ceil(32) as u32, height.div_ceil(8) as u32, 1),
        block_dim: (32, 8, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: arguments match `lens_kernel` in cuda/lens.cu; `src` and `dst`
    // hold width * height RGB pixels and `lut` LUT_N rows of four floats.
    unsafe { launch.launch(cfg) }.map_err(gpu_err("lens_kernel"))?;
    Ok(Some(dst))
}

/// Lateral chromatic aberration corrected on the RAW samples, before
/// demosaicing: the right place for it. RCD (like any colour-difference
/// demosaic) rebuilds red and blue from green's edges, which hides about half
/// of the misalignment in the result and leaves a residue that a later
/// per-channel warp over-corrects (measured on the samples: see
/// `gpu_lens_ca_direction`). Call it on the decoded frame before
/// `develop_scene`/`develop_full`, then [`apply`] with `lateral_ca: false`.
/// Returns whether the frame changed.
pub fn correct_mosaic(
    gpu: &Gpu,
    frame: &mut RawFrame,
    correction: &LensCorrection,
) -> Result<bool> {
    let only_ca = LensSettings {
        profile: false,
        lateral_ca: true,
    };
    let (w, h) = (frame.width, frame.height);
    if correction.ca.is_none() || w < 4 || h < 4 {
        return Ok(false);
    }
    if !matches!(frame.channels, 1 | 3) || frame.data.len() != w * h * frame.channels {
        return Err(anyhow!(
            "lens: RAW frame is malformed: {w}x{h}x{} with {} samples",
            frame.channels,
            frame.data.len()
        ));
    }
    if i32::try_from(frame.data.len()).is_err() {
        return Err(anyhow!("lens: RAW frame too large: {w}x{h}"));
    }
    let module = gpu.module("lens", SOURCE)?;
    let name = if frame.channels == 1 {
        "ca_mosaic_kernel"
    } else {
        "ca_rgb_kernel"
    };
    let kernel = module
        .load_function(name)
        .map_err(|e| anyhow!("load kernel {name}: {e:?}"))?;
    let s = &gpu.stream;
    let lut = s
        .clone_htod(&correction.lut(only_ca))
        .map_err(gpu_err("upload lens table"))?;
    let src = s.clone_htod(&frame.data).map_err(gpu_err("upload RAW"))?;
    // SAFETY: the kernel writes every element of `dst` before anything reads it.
    let mut dst = unsafe { s.alloc::<u16>(frame.data.len()) }.map_err(gpu_err("allocate RAW"))?;
    let (wi, hi, n) = (w as i32, h as i32, LUT_N as i32);
    let cfa = frame
        .cfa
        .iter()
        .enumerate()
        .fold(0i32, |acc, (k, &c)| acc | (i32::from(c & 3) << (2 * k)));
    let inv_half_diag = 1.0 / (w as f32 / 2.0).hypot(h as f32 / 2.0);
    let mut launch = s.launch_builder(&kernel);
    launch.arg(&src).arg(&mut dst).arg(&wi).arg(&hi);
    if frame.channels == 1 {
        launch.arg(&cfa);
    }
    launch.arg(&lut).arg(&n).arg(&LUT_RMAX).arg(&inv_half_diag);
    let cfg = LaunchConfig {
        grid_dim: (w.div_ceil(32) as u32, h.div_ceil(8) as u32, 1),
        block_dim: (32, 8, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: arguments match `ca_mosaic_kernel` (Bayer, with the packed
    // pattern) or `ca_rgb_kernel` (three channels) in cuda/lens.cu; `src` and
    // `dst` hold the frame's samples and `lut` LUT_N rows of four floats.
    unsafe { launch.launch(cfg) }.map_err(gpu_err(name))?;
    s.memcpy_dtoh(&dst, &mut frame.data)
        .map_err(gpu_err("download RAW"))?;
    Ok(true)
}

fn gpu_err<E: std::fmt::Debug>(what: &'static str) -> impl Fn(E) -> spl_core::Error {
    move |e| anyhow!("GPU lens {what}: {e:?}")
}

#[cfg(test)]
mod gpu_tests;

#[cfg(test)]
mod tests {
    use super::*;

    /// The A6400 sample's arrays (11 knots).
    const DIST: [i32; 17] = [
        11, 29, 29, 58, 144, 259, 417, 633, 891, 1222, 1639, 2156, 0, 0, 0, 0, 0,
    ];
    const CA: [i32; 33] = [
        22, -1280, -1408, -1536, -1536, -1664, -1536, -1536, -1536, -1664, -1664, -1664, 0, -128,
        -256, -384, -512, -640, -640, -768, -896, -1024, -1152, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    const VIG: [i32; 17] = [
        11, 0, 320, 768, 1376, 2080, 2816, 3744, 4992, 6528, 8384, 10592, 0, 0, 0, 0, 0,
    ];
    /// The RX100 III's strong barrel distortion.
    const BARREL: [i32; 17] = [
        11, 1265, 1218, 1083, 860, 562, 209, -179, -580, -972, -1353, -1725, 0, 0, 0, 0, 0,
    ];

    fn a6400() -> LensCorrection {
        LensCorrection::from_sony_tags(Some(&DIST), Some(&CA), Some(&VIG))
    }

    #[test]
    fn sony_arrays_decode_like_darktable() {
        let c = a6400();
        let d = c.distortion.as_ref().expect("distortion");
        // Knot i at (i + 0.5) / 10; flat before the first and after the last.
        assert!((d.at(0.0) - (1.0 + 29.0 / 16384.0)).abs() < 1e-6);
        assert!((d.at(0.25) - (1.0 + 58.0 / 16384.0)).abs() < 1e-6);
        assert!((d.at(0.3) - (1.0 + (58.0 + 144.0) / 2.0 / 16384.0)).abs() < 1e-6);
        assert!((d.at(2.0) - (1.0 + 2156.0 / 16384.0)).abs() < 1e-6);
        let [red, blue] = c.ca.as_ref().expect("ca");
        assert!((red.at(0.0) - (1.0 - 1280.0 / 2_097_152.0)).abs() < 1e-7);
        assert!((blue.at(1.05) - (1.0 - 1152.0 / 2_097_152.0)).abs() < 1e-7);
        let v = c.vignetting.as_ref().expect("vignetting");
        assert!((v.at(0.0) - 1.0).abs() < 1e-6, "no gain at the centre");
        // 10592 -> 2^(2^(10592/8192 - 1) - 0.5) = +0.72 EV at the corner.
        assert!((v.at(1.05).log2() - 0.7248).abs() < 1e-3, "{}", v.at(1.05));
        assert!(v.at(0.5) < v.at(0.8), "gain rises toward the corner");
    }

    #[test]
    fn missing_or_bad_arrays_are_dropped() {
        let none = LensCorrection::from_sony_tags(None, None, None);
        assert!(none.is_empty());
        // Count larger than the array, odd CA count, all-zero (identity) arrays.
        let bad = LensCorrection::from_sony_tags(
            Some(&[16, 1, 2, 3]),
            Some(&[21, 0, 0, 0]),
            Some(&[11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        );
        assert!(bad.is_empty(), "{bad:?}");
        let only_vig = LensCorrection::from_sony_tags(None, None, Some(&VIG));
        assert!(only_vig.distortion.is_none() && only_vig.vignetting.is_some());
        assert_eq!(only_vig.zoom(6000, 4000, LensSettings::AUTO), 1.0);
        assert!(lens_profile(&[0u8; 4096]).is_none());
        assert!(lens_profile(b"tiny").is_none());
    }

    /// Every boundary pixel reads inside the frame at the chosen zoom, and
    /// a slightly wider view would read outside.
    #[test]
    fn zoom_leaves_no_empty_border() {
        for (dist, (w, h)) in [
            (&DIST, (6000, 4000)),
            (&BARREL, (5472, 3648)),
            (&BARREL, (3648, 5472)),
        ] {
            let c = LensCorrection::from_sony_tags(Some(dist), Some(&CA), None);
            let z = c.zoom(w, h, LensSettings::AUTO);
            let (cx, cy) = ((w - 1) as f32 / 2.0, (h - 1) as f32 / 2.0);
            let ihd = 1.0 / (w as f32 / 2.0).hypot(h as f32 / 2.0);
            let worst = |z: f32| {
                let mut worst = f32::MIN;
                for i in 0..=1000 {
                    let t = i as f32 / 1000.0;
                    for (dx, dy) in [
                        (cx * (2.0 * t - 1.0), -cy),
                        (cx * (2.0 * t - 1.0), cy),
                        (-cx, cy * (2.0 * t - 1.0)),
                        (cx, cy * (2.0 * t - 1.0)),
                    ] {
                        let (dx, dy) = (dx / z, dy / z);
                        let r = dx.hypot(dy) * ihd;
                        for ch in 0..3 {
                            let s = c.scale(LensSettings::AUTO, ch, r);
                            worst = worst.max((dx * s).abs() - cx).max((dy * s).abs() - cy);
                        }
                    }
                }
                worst
            };
            assert!(worst(z) < 0.01, "{w}x{h}: overshoot {} px", worst(z));
            assert!(worst(z * 0.995) > 0.0, "{w}x{h}: zoom {z} is not tight");
            assert!((0.8..1.25).contains(&z), "{z}");
        }
        // Pincushion correction pushes the edges outward: the view must zoom in.
        let pin = LensCorrection::from_sony_tags(Some(&DIST), None, None);
        assert!(pin.zoom(6000, 4000, LensSettings::AUTO) > 1.0);
        let off = LensSettings {
            profile: false,
            lateral_ca: false,
        };
        assert_eq!(pin.zoom(6000, 4000, off), 1.0);
    }

    #[test]
    fn lut_follows_settings() {
        let c = a6400();
        let only_ca = LensSettings {
            profile: false,
            lateral_ca: true,
        };
        let lut = c.lut(only_ca);
        assert_eq!(lut.len(), 4 * LUT_N);
        for row in lut.as_chunks::<4>().0 {
            assert_eq!(row[1], 1.0, "green does not move without the profile");
            assert_eq!(row[3], 1.0, "no vignetting without the profile");
        }
        let full = c.lut(LensSettings::AUTO);
        let last = full.as_chunks::<4>().0[LUT_N - 1];
        assert!(last[1] > 1.1 && last[3] > 1.5, "{last:?}");
        assert!(last[0] < last[1] && last[2] < last[1], "{last:?}");
    }
}
