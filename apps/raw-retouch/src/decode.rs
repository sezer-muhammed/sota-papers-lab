//! RAW file → [`RawFrame`] with rawler (CPU). Owner: develop.

use spl_core::Result;

use crate::color::ColorModel;
use crate::types::RawInfo;

/// A decoded RAW, cropped to the camera's default crop, still mosaiced (or
/// three-channel for linear RAWs such as Sony's M/S lossless sizes).
pub struct RawFrame {
    /// `as_shot_temperature`/`as_shot_tint` filled in from [`ColorModel`];
    /// `raw_clipped_fraction` is filled in by develop.
    pub info: RawInfo,
    pub color: ColorModel,
    /// As-shot white balance multipliers, green = 1.
    pub wb_as_shot: [f32; 3],
    pub width: usize,
    pub height: usize,
    /// 1 = Bayer mosaic, 3 = linear RGB.
    pub channels: usize,
    /// Bayer colours at (0,0), (0,1), (1,0), (1,1) of the cropped image: 0 = R, 1 = G, 2 = B.
    pub cfa: [u8; 4],
    /// Samples, row-major (interleaved when `channels` is 3), cropped.
    pub data: Vec<u16>,
    /// Black level per CFA position (same order as `cfa`), or per channel for linear RAWs.
    pub black: [f32; 4],
    pub white: f32,
}

/// Decodes any format rawler supports (every Sony ARW variant, DNG, ...).
pub fn decode(bytes: &[u8]) -> Result<RawFrame> {
    let _ = bytes;
    unimplemented!("decode")
}
