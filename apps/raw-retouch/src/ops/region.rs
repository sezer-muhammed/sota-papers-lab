//! Zoomed views: a region of the full-resolution render at 1:1, pixel for
//! pixel what the JPEG export holds (Detail and Lens Corrections included).
//! Owner: media.
//!
//! A region costs a crop and a JPEG once its photo and settings have been
//! rendered at full resolution: the process keeps the two most recent
//! renders (a photo's before and after; 72 MB each at 24 MP, 183 MB at
//! 61 MP, 8-bit and upright) and the most recent demosaiced frame (288 MB
//! at 24 MP, 732 MB at 61 MP), so new settings skip the demosaic. An export
//! fills both caches, so zooming into a fresh result renders nothing.

use std::sync::Arc;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use spl_core::{Result, RunCx, bail};
use ts_rs::TS;

use super::cache::{Lru, Photo};
use super::export::full_res_permit;
use super::work::{Span, blocking};
use crate::encode;
use crate::finish::{self, LensCorrection};
use crate::render::render_full;
use crate::types::{Gpu, LinearImage, Params, Rgb8Image};

/// Longest side of a region, in pixels.
pub const MAX_REGION: u32 = 2048;
/// JPEG quality of a region.
const REGION_QUALITY: u8 = 92;

/// A rectangle of the upright full-resolution image, in its pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema, TS)]
#[ts(export, export_to = "raw-retouch/")]
pub struct Region {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl Region {
    /// The region clipped to a `width` x `height` image, or why it cannot be
    /// rendered: empty, larger than [`MAX_REGION`], or outside the image.
    pub fn within(&self, width: u32, height: u32) -> Result<Region> {
        if self.w == 0 || self.h == 0 {
            bail!("region is empty ({}x{})", self.w, self.h);
        }
        if self.w > MAX_REGION || self.h > MAX_REGION {
            bail!(
                "region is {}x{}; at most {MAX_REGION}x{MAX_REGION} per request",
                self.w,
                self.h
            );
        }
        if self.x >= width || self.y >= height {
            bail!(
                "region starts at ({}, {}), outside the {width}x{height} image",
                self.x,
                self.y
            );
        }
        Ok(Region {
            x: self.x,
            y: self.y,
            w: self.w.min(width - self.x),
            h: self.h.min(height - self.y),
        })
    }
}

/// Demosaiced full-resolution frames with their lens corrections, by RAW
/// and lens switches: the most recent.
static FRAMES: Lru<LinearImage> = Lru::new(1);
/// Finished full-resolution renders, upright and 8-bit as the JPEG export
/// holds them, by RAW and settings: the two most recent.
static RENDERS: Lru<Rgb8Image> = Lru::new(2);

fn frame_key(photo: &Photo, params: &Params) -> String {
    let lens = LensCorrection::new(params, &photo.scene.info).settings();
    format!(
        "{}/lens{}{}",
        photo.raw,
        u8::from(lens.profile),
        u8::from(lens.lateral_ca)
    )
}

fn render_key(photo: &Photo, params: &Params) -> String {
    let bits: Vec<u8> = params
        .to_array()
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect();
    format!("{}/{}", photo.raw, blake3::hash(&bits).to_hex())
}

/// `photo` demosaiced at full resolution with the lens corrections
/// `params` switch on ([`finish::develop_full`]), from the cache or now.
/// Call it holding the full-resolution permit.
pub(super) async fn frame(
    cx: &RunCx,
    gpu: &Gpu,
    photo: &Arc<Photo>,
    params: &Params,
    span: Span,
) -> Result<Arc<LinearImage>> {
    FRAMES
        .get_or_load(&frame_key(photo, params), || {
            let (gpu, photo, params) = (gpu.clone(), photo.clone(), *params);
            blocking(cx, span, move |r| {
                r.progress(0.0, "Demosaicing at full resolution");
                finish::develop_full(&gpu, &photo.frame, photo.lens.as_ref(), &params)
            })
        })
        .await
}

/// Keeps an export's 8-bit image for zoomed views of the same settings.
pub(super) fn keep_render(photo: &Photo, params: &Params, image: Rgb8Image) {
    RENDERS.insert(&render_key(photo, params), Arc::new(image));
}

/// A region at 1:1 of `photo` rendered with `params`.
pub(super) struct Tile {
    pub jpeg: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// `region` (upright full-resolution pixels) of `photo` rendered with
/// `params`, exactly as the export has it, as a JPEG. Renders the whole
/// image once per photo and settings (holding the full-resolution permit)
/// and crops it.
pub(super) async fn render(
    cx: &RunCx,
    gpu: &Gpu,
    photo: &Arc<Photo>,
    params: &Params,
    region: Region,
    span: Span,
) -> Result<Tile> {
    let info = &photo.scene.info;
    let (width, height) = if (5..=8).contains(&info.orientation) {
        (info.height, info.width)
    } else {
        (info.width, info.height)
    };
    let region = region.within(width, height)?;
    let image = RENDERS
        .get_or_load(&render_key(photo, params), || async {
            let permit = full_res_permit(cx, span).await?;
            let full = frame(cx, gpu, photo, params, span.part(0.0, 0.5)).await?;
            let (gpu, photo, params) = (gpu.clone(), photo.clone(), *params);
            blocking(cx, span.part(0.5, 1.0), move |r| {
                let _permit = permit;
                r.progress(0.0, "Rendering at full resolution");
                let scene = photo.scene_for(&gpu, &params)?;
                let rendered = render_full(&gpu, &scene, &full, &params)?;
                drop(full);
                let upright = encode::orient16(&rendered, photo.scene.info.orientation);
                drop(rendered);
                Ok(encode::to_rgb8(&upright))
            })
            .await
        })
        .await?;
    blocking(cx, Span::NONE, move |_| {
        let crop = crop(&image, region);
        Ok(Tile {
            jpeg: encode::jpeg(&crop, REGION_QUALITY)?,
            width: region.w,
            height: region.h,
        })
    })
    .await
}

/// `region` of `img`; the region lies inside it.
fn crop(img: &Rgb8Image, region: Region) -> Rgb8Image {
    let (x, y, w, h) = (
        region.x as usize,
        region.y as usize,
        region.w as usize,
        region.h as usize,
    );
    let mut data = Vec::with_capacity(w * h * 3);
    for row in y..y + h {
        let start = (row * img.width + x) * 3;
        data.extend_from_slice(&img.data[start..start + w * 3]);
    }
    Rgb8Image {
        width: w,
        height: h,
        data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regions_clip_to_the_image() {
        let r = |x, y, w, h| Region { x, y, w, h };
        assert_eq!(
            r(10, 20, 300, 200).within(6000, 4000).unwrap(),
            r(10, 20, 300, 200)
        );
        assert_eq!(
            r(5900, 3900, 300, 200).within(6000, 4000).unwrap(),
            r(5900, 3900, 100, 100)
        );
        assert_eq!(
            r(0, 0, 2048, 2048).within(6000, 4000).unwrap(),
            r(0, 0, 2048, 2048)
        );
        assert!(r(0, 0, 0, 10).within(6000, 4000).is_err());
        assert!(r(0, 0, 2049, 10).within(6000, 4000).is_err());
        assert!(r(6000, 0, 10, 10).within(6000, 4000).is_err());
        assert!(r(0, 4000, 10, 10).within(6000, 4000).is_err());
    }

    #[test]
    fn crops_take_the_region_row_by_row() {
        let img = Rgb8Image {
            width: 5,
            height: 4,
            data: (0..60).collect(),
        };
        let c = crop(
            &img,
            Region {
                x: 1,
                y: 2,
                w: 3,
                h: 2,
            },
        );
        assert_eq!((c.width, c.height), (3, 2));
        assert_eq!(
            c.data,
            [
                33, 34, 35, 36, 37, 38, 39, 40, 41, 48, 49, 50, 51, 52, 53, 54, 55, 56
            ]
        );
    }
}
