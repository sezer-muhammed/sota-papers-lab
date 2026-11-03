//! Capture statistics measured during ingest (blur, standing still,
//! exposure clipping) for the capture report. Owner: ingest (A0).

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use ts_rs::TS;

use super::motion::{Scored, Selection, blurry_flags};
use super::{IngestFrame, KeyframeMode, Rgb8};

/// Luma at or below this (of 255) counts as crushed black.
pub const DARK_CLIP: f32 = 3.0;
/// Any channel at or above this counts as blown highlight.
pub const BRIGHT_CLIP: u8 = 252;

/// What ingest saw of the capture.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct CaptureStats {
    /// `motion`, `time` (video keyframes) or `photos`.
    pub keyframes: String,
    /// Frames decoded (video at the sample rate) or photos given.
    pub decoded: u32,
    pub kept: u32,
    /// Share of decoded frames that are blurry next to their neighbours.
    pub blurry_share: f64,
    /// Share of the kept frames that are blurry.
    pub kept_blurry_share: f64,
    /// Video: share of consecutive frames with almost no camera motion.
    #[ts(optional)]
    pub still_share: Option<f64>,
    /// Video: summed global motion (≈ frame widths of image travel).
    #[ts(optional)]
    pub total_motion: Option<f64>,
    /// Video: picks dropped as near duplicates.
    pub duplicates: u32,
    /// Mean share of crushed-black pixels in the kept frames.
    pub dark_clip_share: f64,
    /// Mean share of blown-highlight pixels in the kept frames.
    pub bright_clip_share: f64,
}

/// Shares of crushed-black and blown pixels in one image.
pub fn clipping(img: &Rgb8) -> (f64, f64) {
    let n = (img.width as usize * img.height as usize).max(1);
    let (mut dark, mut bright) = (0usize, 0usize);
    for p in img.data.as_chunks::<3>().0 {
        let l = 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32;
        if l <= DARK_CLIP {
            dark += 1;
        }
        if p.iter().any(|&c| c >= BRIGHT_CLIP) {
            bright += 1;
        }
    }
    (dark as f64 / n as f64, bright as f64 / n as f64)
}

fn mean_clipping(frames: &[IngestFrame]) -> (f64, f64) {
    if frames.is_empty() {
        return (0.0, 0.0);
    }
    let (d, b) = frames
        .iter()
        .map(|f| clipping(&f.engine))
        .fold((0.0, 0.0), |a, c| (a.0 + c.0, a.1 + c.1));
    (d / frames.len() as f64, b / frames.len() as f64)
}

impl CaptureStats {
    pub fn from_video(
        scored: &[Scored],
        sel: &Selection,
        frames: &[IngestFrame],
        mode: KeyframeMode,
    ) -> Self {
        let n = scored.len().max(1) as f64;
        let blur = blurry_flags(scored, 15);
        let kept_blurry = scored
            .iter()
            .zip(&blur)
            .filter(|(s, b)| **b && sel.kept.binary_search(&s.index).is_ok())
            .count();
        let (dark, bright) = mean_clipping(frames);
        Self {
            keyframes: match mode {
                KeyframeMode::Motion => "motion",
                KeyframeMode::Time => "time",
            }
            .into(),
            decoded: scored.len() as u32,
            kept: frames.len() as u32,
            blurry_share: sel.blurry as f64 / n,
            kept_blurry_share: kept_blurry as f64 / frames.len().max(1) as f64,
            still_share: Some(sel.still as f64 / (n - 1.0).max(1.0)),
            total_motion: Some(sel.total_motion),
            duplicates: sel.duplicates as u32,
            dark_clip_share: dark,
            bright_clip_share: bright,
        }
    }

    /// Photos: blur against the whole set (no order to lean on).
    pub fn from_photos(frames: &[IngestFrame]) -> Self {
        let scored: Vec<Scored> = frames
            .iter()
            .map(|f| Scored {
                index: f.index,
                t: 0.0,
                sharpness: f.sharpness,
                motion: 0.0,
            })
            .collect();
        let blur = blurry_flags(&scored, scored.len());
        let blurry = blur.iter().filter(|&&b| b).count();
        let share = blurry as f64 / frames.len().max(1) as f64;
        let (dark, bright) = mean_clipping(frames);
        Self {
            keyframes: "photos".into(),
            decoded: frames.len() as u32,
            kept: frames.len() as u32,
            blurry_share: share,
            kept_blurry_share: share,
            still_share: None,
            total_motion: None,
            duplicates: 0,
            dark_clip_share: dark,
            bright_clip_share: bright,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clipping_counts_black_and_blown_pixels() {
        let img = Rgb8 {
            width: 4,
            height: 1,
            data: vec![0, 0, 0, 255, 10, 10, 128, 128, 128, 2, 2, 2],
        };
        assert_eq!(clipping(&img), (0.5, 0.25));
    }
}
