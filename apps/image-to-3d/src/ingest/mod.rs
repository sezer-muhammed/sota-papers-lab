//! Stage A0 ingest: photos or a phone video → keyframes at the engine and
//! trainer resolutions, with sharpness, timestamps and an EXIF focal prior.
//! Owner: ingest (A0). Design: `reports/Image to 3D.md` section 3, "Stage A0".
//!
//! - Photos ([`ingest_images`]): JPEG, PNG and WebP through the `image` crate,
//!   EXIF orientation applied, EXIF focal length (35 mm equivalent) kept as a
//!   prior for the focal refinement in Stage C.
//! - Video ([`ingest_video`]): `ffprobe` for size, rotation and duration, then
//!   `ffmpeg` decodes packed RGB to a pipe as a child process that dies with
//!   the future (`kill_on_drop`) and stops on the run's cancel token.
//!   Keyframes ([`KeyframeMode::Motion`], the default): every decoded frame
//!   gets a sharpness score (Laplacian variance of luma, one CUDA kernel)
//!   and a global motion estimate on a small luma thumbnail ([`motion`]);
//!   the camera's path is cut into `max_frames` equal slices of motion, each
//!   keeping its sharpest frame, so standing still costs no frames, fast
//!   pans keep coverage, and blurry frames and near duplicates lose. The
//!   picks are resized in a second decode. [`KeyframeMode::Time`] keeps the
//!   older one-pass policy (equal time bins).
//! - Resizing to both resolutions is an area (box) filter on the GPU.
//!
//! The ffmpeg binary must exist in the runtime image: see
//! `apps/image-to-3d/src/ingest/DOCKERFILE_PATCH.md`.

pub mod capture;
mod decode;
pub mod exif;
pub mod gpu;
pub mod motion;
mod video;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use ts_rs::TS;

pub use capture::CaptureStats;

pub use decode::{decode_image, ingest_images};
pub use video::{VideoInfo, ingest_video, probe_video};

/// Target resolutions and keyframe policy.
#[derive(Debug, Clone, PartialEq)]
pub struct IngestConfig {
    /// Engine input: longest side (VGGT-Ω runs at 512×336 for 3:2).
    pub engine_long_side: u32,
    /// Engine dimensions are multiples of this (the ViT patch size).
    pub engine_multiple: u32,
    /// Trainer images: longest side (never upscaled).
    pub trainer_long_side: u32,
    /// Most keyframes a video yields (the engine routes > 300 differently).
    pub max_frames: u32,
    /// Decode rate for videos in frames per second; `None` = every frame.
    pub sample_fps: Option<f64>,
    /// Extra `ffmpeg` input options, e.g. `["-hwaccel", "cuda"]`.
    pub ffmpeg_input_args: Vec<String>,
    /// How video keyframes are spread.
    pub keyframes: KeyframeMode,
}

/// How video keyframes are spread over the clip.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export, export_to = "image-to-3d/")]
pub enum KeyframeMode {
    /// Equal slices of camera motion (two decodes).
    #[default]
    Motion,
    /// Equal slices of time (one decode).
    Time,
}

impl Default for IngestConfig {
    fn default() -> Self {
        Self {
            engine_long_side: 512,
            engine_multiple: 16,
            trainer_long_side: 1600,
            max_frames: 300,
            sample_fps: Some(10.0),
            ffmpeg_input_args: Vec::new(),
            keyframes: KeyframeMode::default(),
        }
    }
}

/// Packed 8-bit RGB, row-major, no padding.
#[derive(Debug, Clone, PartialEq)]
pub struct Rgb8 {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

/// Focal length prior from EXIF.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct FocalPrior {
    /// 35 mm equivalent focal length in mm.
    pub focal_35mm: f32,
}

impl FocalPrior {
    /// Focal length in pixels for an image of `width`×`height` (the 35 mm
    /// equivalent is defined on the 43.27 mm full-frame diagonal, so it holds
    /// for any aspect ratio and crop of the full sensor).
    pub fn focal_px(&self, width: u32, height: u32) -> f32 {
        let diag = ((width as f32).powi(2) + (height as f32).powi(2)).sqrt();
        self.focal_35mm * diag / 43.266_615
    }
}

/// One kept frame.
#[derive(Debug, Clone)]
pub struct IngestFrame {
    /// File name (photos) or `<video>#<frame index>`.
    pub source: String,
    /// Index of the frame among the decoded ones (photos: input order).
    pub index: u32,
    /// Seconds from the start of the video; `None` for photos.
    pub timestamp_s: Option<f64>,
    /// Laplacian variance of luma in [0, 1] at the decoded resolution.
    pub sharpness: f64,
    /// Decoded size after orientation.
    pub source_size: (u32, u32),
    pub focal_prior: Option<FocalPrior>,
    pub engine: Rgb8,
    pub trainer: Rgb8,
}

/// Sharpness of every decoded frame (the keyframe score timeline).
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct SharpnessSample {
    pub index: u32,
    pub timestamp_s: Option<f64>,
    pub sharpness: f64,
}

/// Timings of one ingest.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct IngestStats {
    pub decoded: u32,
    pub kept: u32,
    pub seconds: f64,
    /// Decoded frames per second, end to end.
    pub decode_fps: f64,
    /// Seconds spent in GPU work (upload, sharpness, resize, download).
    pub gpu_seconds: f64,
}

/// What Stage A0 hands to the engine and the trainer.
#[derive(Debug, Clone)]
pub struct IngestResult {
    /// Kept frames in temporal (video) or input (photos) order.
    pub frames: Vec<IngestFrame>,
    /// Sharpness of every decoded frame, in decode order.
    pub sharpness: Vec<SharpnessSample>,
    /// Median of the per-frame priors (photos), or `None`.
    pub focal_prior: Option<FocalPrior>,
    pub stats: IngestStats,
    /// Blur, motion and exposure, for the capture report.
    pub capture: CaptureStats,
}

impl IngestResult {
    pub fn timestamps(&self) -> Vec<Option<f64>> {
        self.frames.iter().map(|f| f.timestamp_s).collect()
    }
}

/// Median of the frames' priors.
pub fn median_prior(frames: &[IngestFrame]) -> Option<FocalPrior> {
    let mut v: Vec<f32> = frames
        .iter()
        .filter_map(|f| f.focal_prior.map(|p| p.focal_35mm))
        .collect();
    if v.is_empty() {
        return None;
    }
    v.sort_by(f32::total_cmp);
    Some(FocalPrior {
        focal_35mm: v[v.len() / 2],
    })
}

/// Fits `w`×`h` into `long_side` keeping the aspect (never upscales), each
/// dimension rounded to the nearest multiple of `multiple` (at least one).
pub fn fit_size(w: u32, h: u32, long_side: u32, multiple: u32) -> (u32, u32) {
    let m = multiple.max(1);
    let scale = (long_side as f64 / w.max(h).max(1) as f64).min(1.0);
    let round = |v: u32| {
        let s = v as f64 * scale / m as f64;
        ((s.round() as u32).max(1) * m).min(v.max(m))
    };
    (round(w), round(h))
}

/// Keyframe bins: `count` equal slices of `[0, duration)`, each keeping its
/// sharpest frame.
#[derive(Debug, Clone)]
pub struct KeyframeBins {
    duration: f64,
    best: Vec<Option<(f64, u32)>>,
}

impl KeyframeBins {
    pub fn new(duration_s: f64, count: u32) -> Self {
        Self {
            duration: duration_s.max(1e-9),
            best: vec![None; count.max(1) as usize],
        }
    }

    pub fn bin_of(&self, t: f64) -> usize {
        let n = self.best.len();
        ((t / self.duration * n as f64).floor().max(0.0) as usize).min(n - 1)
    }

    /// Offers frame `index` at time `t`; returns its bin when it becomes the
    /// bin's best (strictly sharper; ties keep the earlier frame).
    pub fn offer(&mut self, t: f64, sharpness: f64, index: u32) -> Option<usize> {
        let b = self.bin_of(t);
        match self.best[b] {
            Some((s, _)) if s >= sharpness => None,
            _ => {
                self.best[b] = Some((sharpness, index));
                Some(b)
            }
        }
    }

    /// Frame index kept per bin.
    pub fn kept(&self) -> Vec<Option<u32>> {
        self.best.iter().map(|b| b.map(|(_, i)| i)).collect()
    }
}

/// CPU reference of the `sharpness_partials` kernel: variance of the
/// 4-neighbour Laplacian of BT.601 luma in [0, 1] over interior pixels.
pub fn sharpness_cpu(img: &Rgb8) -> f64 {
    let (w, h) = (img.width as usize, img.height as usize);
    if w < 3 || h < 3 {
        return 0.0;
    }
    let luma = |x: usize, y: usize| {
        let p = &img.data[(y * w + x) * 3..];
        (0.299 * p[0] as f64 + 0.587 * p[1] as f64 + 0.114 * p[2] as f64) / 255.0
    };
    let (mut s, mut s2) = (0.0, 0.0);
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let l = luma(x - 1, y) + luma(x + 1, y) + luma(x, y - 1) + luma(x, y + 1)
                - 4.0 * luma(x, y);
            s += l;
            s2 += l * l;
        }
    }
    let n = ((w - 2) * (h - 2)) as f64;
    (s2 / n - (s / n).powi(2)).max(0.0)
}

/// CPU reference of the `resize_area` kernel.
pub fn resize_area_cpu(src: &Rgb8, dw: u32, dh: u32) -> Rgb8 {
    let (sw, sh) = (src.width as usize, src.height as usize);
    let (fx, fy) = (sw as f64 / dw as f64, sh as f64 / dh as f64);
    let mut data = vec![0u8; dw as usize * dh as usize * 3];
    for oy in 0..dh as usize {
        let (y0, y1) = (oy as f64 * fy, oy as f64 * fy + fy);
        for ox in 0..dw as usize {
            let (x0, x1) = (ox as f64 * fx, ox as f64 * fx + fx);
            let mut acc = [0.0f64; 3];
            let mut wsum = 0.0;
            for y in y0.floor() as usize..(y1.ceil() as usize).min(sh) {
                let wy = (y as f64 + 1.0).min(y1) - (y as f64).max(y0);
                if wy <= 0.0 {
                    continue;
                }
                for x in x0.floor() as usize..(x1.ceil() as usize).min(sw) {
                    let wx = (x as f64 + 1.0).min(x1) - (x as f64).max(x0);
                    if wx <= 0.0 {
                        continue;
                    }
                    for (c, a) in acc.iter_mut().enumerate() {
                        *a += wx * wy * src.data[(y * sw + x) * 3 + c] as f64;
                    }
                    wsum += wx * wy;
                }
            }
            for (c, a) in acc.iter().enumerate() {
                data[(oy * dw as usize + ox) * 3 + c] = (a / wsum + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
    }
    Rgb8 {
        width: dw,
        height: dh,
        data,
    }
}

#[cfg(test)]
mod gpu_tests;
#[cfg(test)]
mod tests;
