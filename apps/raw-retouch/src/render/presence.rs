//! Texture, Clarity and Dehaze: the multi-pass part of the renderer, with
//! `cuda/presence.cu`. Owner: presence.
//!
//! These need blurred copies of the image, so they come in two parts:
//! - per-image *planes* (a float4 per pixel, independent of the sliders),
//!   made once by [`preview_planes`] for the search and display previews
//!   when a `RenderEngine` is built (and for the display image of every
//!   full-resolution export), and by [`strip_planes`] for each strip of a
//!   full-resolution export. The kernels (`render8`, `stats`, `full`) hand
//!   each pixel's planes to `presence_apply` in `cuda/presence.cu`;
//! - the sliders, which [`pack`] writes into the candidate at `P_PRES`.
//!
//! Stub: no planes and `presence_apply` does nothing. To become: kernels in
//! `cuda/presence.cu` launched from here (log-luminance blurs at Texture's
//! and Clarity's scales, a dark-channel haze estimate), and the per-pixel
//! application. Keep the search fast: planes are computed once per image,
//! never per candidate. Scale-dependent filters must look the same on the
//! 768 px search image, the 1536 px display image and the full-resolution
//! export (the export can upsample the display planes, as the base layer
//! does, or blur its strips; `render/full.rs` gives each strip a halo of
//! rows that must then cover the blur radius).

use spl_core::Result;
use spl_gpu::cudarc::driver::CudaSlice;

use super::CudaContextExt;
use super::layout::P_PRES_N;
use crate::types::{Gpu, Params};

/// Writes `p`'s Presence sliders into `out` (`P_PRES_N` floats at `P_PRES`).
pub(crate) fn pack(p: &Params, out: &mut [f32]) {
    let out = &mut out[..P_PRES_N];
    out.fill(0.0);
    out[..3].copy_from_slice(&[p.texture / 100.0, p.clarity / 100.0, p.dehaze / 100.0]);
}

/// One image's presence planes on the GPU: four floats per pixel, or none.
pub(crate) struct Planes {
    /// A float4 per pixel when `on`, otherwise a single placeholder float4.
    pub(crate) data: CudaSlice<f32>,
    pub(crate) on: bool,
}

impl Planes {
    pub(crate) fn none(gpu: &Gpu) -> Result<Self> {
        Ok(Self {
            data: gpu
                .stream
                .alloc_zeros::<f32>(4)
                .cuda("alloc presence planes")?,
            on: false,
        })
    }

    /// The kernels' `pres_on` argument.
    pub(crate) fn flag(&self) -> i32 {
        i32::from(self.on)
    }
}

/// Planes for a preview image. `px` is the prepared image (`prep_kernel`: a
/// float4 per pixel of camera RGB, as-shot white balance, and the base
/// layer), `width` x `height`. Stub: none.
pub(crate) fn preview_planes(
    gpu: &Gpu,
    _px: &CudaSlice<f32>,
    _width: usize,
    _height: usize,
) -> Result<Planes> {
    Planes::none(gpu)
}

/// One horizontal strip of a full-resolution export, as `render/full.rs`
/// renders it.
#[allow(
    dead_code,
    reason = "for the presence owner; the stub reads none of it"
)]
pub(crate) struct Strip<'a> {
    /// Camera RGB (3 floats per pixel, as-shot white balance), `rows` rows
    /// of `width` starting at full-image row `y0`, halo rows included.
    pub(crate) rgb: &'a CudaSlice<f32>,
    pub(crate) width: usize,
    pub(crate) rows: usize,
    pub(crate) y0: usize,
    pub(crate) full_height: usize,
    /// The prepared display image and its planes, and display pixels per
    /// full-resolution pixel.
    pub(crate) display_px: &'a CudaSlice<f32>,
    pub(crate) display_planes: &'a Planes,
    pub(crate) display_width: usize,
    pub(crate) display_height: usize,
    pub(crate) sx: f32,
    pub(crate) sy: f32,
}

/// Planes for one full-resolution strip (`rows` x `width`, the strip's own
/// pixel order). Stub: none.
pub(crate) fn strip_planes(gpu: &Gpu, _strip: &Strip) -> Result<Planes> {
    Planes::none(gpu)
}
