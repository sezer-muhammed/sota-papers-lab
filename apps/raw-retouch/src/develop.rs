//! [`RawFrame`] → linear camera RGB on the GPU (`cuda/develop.cu`). Owner: develop.

use spl_core::Result;

use crate::decode::RawFrame;
use crate::types::{Gpu, LinearImage, Scene};

/// Normalize black/white levels, apply as-shot white balance, clip
/// highlights to neutral, half-size superpixel, then area-downscale to the
/// search and display sizes.
pub fn develop_scene(gpu: &Gpu, frame: &RawFrame) -> Result<Scene> {
    let _ = (gpu, frame);
    unimplemented!("develop_scene")
}

/// Full-resolution demosaic with the same normalization as
/// [`develop_scene`]. Cropped and unrotated.
pub fn develop_full(gpu: &Gpu, frame: &RawFrame) -> Result<LinearImage> {
    let _ = (gpu, frame);
    unimplemented!("develop_full")
}
