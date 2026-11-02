//! VGGT-Ω input preprocessing (spec-vggt.md section 2). Owner: runtime.
//!
//! On the CPU: decode (RGBA composited over white), centre-crop to an aspect
//! ratio in [0.5, 2], resize with a bit-exact port of PIL's 8-bit bicubic
//! resampler (`Image.resize(..., BICUBIC)`: separable, support scaled by the
//! downscale factor, 22-bit fixed-point coefficients, horizontal pass then
//! vertical, rounding to u8 between the passes). On the GPU: u8 HWC → f32
//! NCHW with `(x / 255 − mean) / std` in fp32 (`cuda/vggt_runtime.cu`).
//!
//! All frames of one call get the size of the first frame; frames with a
//! different aspect ratio are centre-cropped to it first, so nothing is ever
//! padded (the official loader pads with white, which its evaluation
//! rejects).

use anyhow::ensure;
use rayon::prelude::*;
use spl_core::{Context, Result};

use super::workspace::Geometry;
use crate::io::images::Rgb8Image;

pub const PATCH: usize = 16;

/// How the target size is chosen (`load_fn.py` modes).
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Default,
    serde::Serialize,
    serde::Deserialize,
    schemars::JsonSchema,
    ts_rs::TS,
)]
#[serde(rename_all = "snake_case")]
#[ts(export, export_to = "image-to-3d/")]
pub enum ResizeMode {
    /// Longest side = resolution (the official evaluation).
    #[default]
    MaxSize,
    /// About (resolution / 16)² patches whatever the aspect ratio.
    Balanced,
}

/// Frames at the model's resolution, RGB8 `[N][H][W][3]`, ready to upload.
#[derive(Debug, Clone)]
pub struct Prepared {
    pub geom: Geometry,
    pub pixels: Vec<u8>,
}

/// `np.round` / Python `round`: half to even.
pub fn round_half_even(v: f64) -> f64 {
    v.round_ties_even()
}

/// Crop box `(left, top, width, height)` that brings `h / w` into [0.5, 2].
pub fn aspect_crop(width: usize, height: usize) -> (usize, usize, usize, usize) {
    let a = height as f64 / width.max(1) as f64;
    if a < 0.5 {
        let cw = width.min((round_half_even(height as f64 / 0.5) as usize).max(1));
        ((width - cw) / 2, 0, cw, height)
    } else if a > 2.0 {
        let ch = height.min((round_half_even(width as f64 * 2.0) as usize).max(1));
        (0, (height - ch) / 2, width, ch)
    } else {
        (0, 0, width, height)
    }
}

/// Target `(height, width)` for an image of aspect `a = h / w` (after the
/// aspect crop).
pub fn target_size(a: f64, resolution: usize, mode: ResizeMode) -> (usize, usize) {
    let p = PATCH as f64;
    match mode {
        ResizeMode::MaxSize => {
            let round = |v: f64| ((round_half_even(v / p) as usize) * PATCH).max(PATCH);
            if a >= 1.0 {
                (resolution, round(resolution as f64 / a))
            } else {
                (round(resolution as f64 * a), resolution)
            }
        }
        ResizeMode::Balanced => {
            let tokens = ((resolution / PATCH) * (resolution / PATCH)) as f64;
            let wp = (tokens / a).sqrt();
            let hp = tokens / wp;
            let wp = (round_half_even(wp) as usize).max(1);
            let hp = (round_half_even(hp) as usize).max(1);
            (hp * PATCH, wp * PATCH)
        }
    }
}

fn bicubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

const PRECISION_BITS: u32 = 32 - 8 - 2;

/// PIL's `precompute_coeffs` + `normalize_coeffs_8bpc`: per output pixel
/// the first input index and the fixed-point weights.
pub fn pil_coeffs(in_size: usize, out_size: usize) -> Vec<(usize, Vec<i32>)> {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = 2.0 * filterscale;
    let ss = 1.0 / filterscale;
    (0..out_size)
        .map(|xx| {
            let center = (xx as f64 + 0.5) * scale;
            let xmin = ((center - support + 0.5) as i64).max(0) as usize;
            let xmax = ((center + support + 0.5) as i64).min(in_size as i64) as usize;
            let mut k: Vec<f64> = (xmin..xmax)
                .map(|x| bicubic((x as f64 - center + 0.5) * ss))
                .collect();
            let ww: f64 = k.iter().sum();
            if ww != 0.0 {
                for v in &mut k {
                    *v /= ww;
                }
            }
            let one = (1u32 << PRECISION_BITS) as f64;
            let ki = k
                .iter()
                .map(|&v| {
                    if v < 0.0 {
                        (-0.5 + v * one) as i32
                    } else {
                        (0.5 + v * one) as i32
                    }
                })
                .collect();
            (xmin, ki)
        })
        .collect()
}

#[inline]
fn clip8(ss: i64) -> u8 {
    (ss >> PRECISION_BITS).clamp(0, 255) as u8
}

/// PIL-exact bicubic resize of interleaved RGB8.
pub fn resize_bicubic(src: &[u8], w: usize, h: usize, ow: usize, oh: usize) -> Vec<u8> {
    let half = 1i64 << (PRECISION_BITS - 1);
    let horiz = if ow != w {
        let cx = pil_coeffs(w, ow);
        let mut out = vec![0u8; ow * h * 3];
        out.par_chunks_mut(ow * 3).enumerate().for_each(|(y, row)| {
            let s = &src[y * w * 3..(y + 1) * w * 3];
            for (x, (x0, k)) in cx.iter().enumerate() {
                for c in 0..3 {
                    let mut ss = half;
                    for (i, &kv) in k.iter().enumerate() {
                        ss += s[(x0 + i) * 3 + c] as i64 * kv as i64;
                    }
                    row[x * 3 + c] = clip8(ss);
                }
            }
        });
        out
    } else {
        src.to_vec()
    };
    if oh == h {
        return horiz;
    }
    let cy = pil_coeffs(h, oh);
    let mut out = vec![0u8; ow * oh * 3];
    out.par_chunks_mut(ow * 3).enumerate().for_each(|(y, row)| {
        let (y0, k) = &cy[y];
        for (x, px) in row.iter_mut().enumerate() {
            let mut ss = half;
            for (i, &kv) in k.iter().enumerate() {
                ss += horiz[(y0 + i) * ow * 3 + x] as i64 * kv as i64;
            }
            *px = clip8(ss);
        }
    });
    out
}

/// Crops `img` to `(left, top, w, h)`.
fn crop(img: &Rgb8Image, (l, t, w, h): (usize, usize, usize, usize)) -> Vec<u8> {
    let iw = img.width as usize;
    let mut out = Vec::with_capacity(w * h * 3);
    for y in t..t + h {
        out.extend_from_slice(&img.data[(y * iw + l) * 3..(y * iw + l + w) * 3]);
    }
    out
}

/// Centre crop of a `w × h` image to aspect `a = h / w` (as PIL's integer
/// boxes do: the kept size rounds, the offset floors).
fn crop_to_aspect(w: usize, h: usize, a: f64) -> (usize, usize, usize, usize) {
    let cur = h as f64 / w as f64;
    if (cur - a).abs() < 1e-9 {
        (0, 0, w, h)
    } else if cur > a {
        let ch = ((w as f64 * a).round() as usize).clamp(1, h);
        (0, (h - ch) / 2, w, ch)
    } else {
        let cw = ((h as f64 / a).round() as usize).clamp(1, w);
        ((w - cw) / 2, 0, cw, h)
    }
}

/// §2 for a batch: every frame ends at the first frame's size.
pub fn prepare(frames: &[Rgb8Image], resolution: usize, mode: ResizeMode) -> Result<Prepared> {
    ensure!(!frames.is_empty(), "no frames");
    ensure!(
        resolution.is_multiple_of(PATCH),
        "resolution must be a multiple of 16"
    );
    let f0 = &frames[0];
    let box0 = aspect_crop(f0.width as usize, f0.height as usize);
    let a0 = box0.3 as f64 / box0.2.max(1) as f64;
    let (oh, ow) = target_size(a0, resolution, mode);
    let per: Vec<Vec<u8>> = frames
        .par_iter()
        .enumerate()
        .map(|(i, f)| -> Result<Vec<u8>> {
            let (w, h) = (f.width as usize, f.height as usize);
            ensure!(
                w > 0 && h > 0 && f.data.len() == w * h * 3,
                "frame {i}: bad image"
            );
            let bx = if i == 0 {
                box0
            } else {
                let b = aspect_crop(w, h);
                let ab = b.3 as f64 / b.2 as f64;
                if (ab - a0).abs() < 1e-9 {
                    b
                } else {
                    crop_to_aspect(w, h, a0)
                }
            };
            let px = crop(f, bx);
            Ok(resize_bicubic(&px, bx.2, bx.3, ow, oh))
        })
        .collect::<Result<_>>()?;
    let mut pixels = Vec::with_capacity(frames.len() * ow * oh * 3);
    for p in per {
        pixels.extend_from_slice(&p);
    }
    Ok(Prepared {
        geom: Geometry::new(frames.len(), oh, ow),
        pixels,
    })
}

/// Decodes a JPEG / PNG / WebP to RGB8; RGBA is composited over opaque
/// white like `load_fn._load_rgb_image`.
pub fn decode(bytes: &[u8]) -> Result<Rgb8Image> {
    if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        return crate::io::images::decode_bytes(bytes);
    }
    let img = image::load_from_memory(bytes).context("decode image")?;
    if !img.color().has_alpha() {
        return crate::io::images::decode_bytes_image_crate(bytes);
    }
    let rgba = img.into_rgba8();
    let (width, height) = rgba.dimensions();
    let mut data = Vec::with_capacity(width as usize * height as usize * 3);
    for p in rgba.pixels() {
        let a = p[3] as u32;
        for c in 0..3 {
            // PIL alpha_composite over opaque white, 8-bit.
            let v = (p[c] as u32 * a + 255 * (255 - a) + 127) / 255;
            data.push(v as u8);
        }
    }
    Ok(Rgb8Image {
        width,
        height,
        data,
    })
}
