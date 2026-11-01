//! Image files: orientation, resizing, JPEG and 16-bit TIFF. Owner: judge.

use spl_core::Result;

use crate::types::{Rgb8Image, Rgb16Image};

/// Applies EXIF orientation (1 to 8) so the result is upright.
pub fn orient(img: &Rgb8Image, orientation: u8) -> Rgb8Image {
    let _ = (img, orientation);
    unimplemented!("orient")
}

pub fn orient16(img: &Rgb16Image, orientation: u8) -> Rgb16Image {
    let _ = (img, orientation);
    unimplemented!("orient16")
}

/// 16 → 8 bits with light dithering.
pub fn to_rgb8(img: &Rgb16Image) -> Rgb8Image {
    let _ = img;
    unimplemented!("to_rgb8")
}

/// Area-average downscale so the long edge is at most `long_edge`; never upscales.
pub fn resize(img: &Rgb8Image, long_edge: usize) -> Rgb8Image {
    let _ = (img, long_edge);
    unimplemented!("resize")
}

/// Baseline sRGB JPEG with 4:4:4 chroma at quality >= 90.
pub fn jpeg(img: &Rgb8Image, quality: u8) -> Result<Vec<u8>> {
    let _ = (img, quality);
    unimplemented!("jpeg")
}

pub fn tiff16(img: &Rgb16Image) -> Result<Vec<u8>> {
    let _ = img;
    unimplemented!("tiff16")
}
