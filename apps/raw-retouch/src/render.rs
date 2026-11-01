//! [`Params`] → pixels and [`Stats`] on the GPU (`cuda/render.cu`). Owner: render.

use spl_core::Result;

use crate::types::{Gpu, LinearImage, Params, Rgb8Image, Rgb16Image, Scene, Stats};

/// Keeps one photo's search and display images (and their base layers) on
/// the GPU so a candidate costs about a millisecond.
pub struct RenderEngine {}

impl RenderEngine {
    pub fn new(gpu: &Gpu, scene: &Scene) -> Result<Self> {
        let _ = (gpu, scene);
        unimplemented!("RenderEngine::new")
    }

    /// Renders every parameter set on the search image and measures it, batched.
    pub fn stats(&mut self, params: &[Params]) -> Result<Vec<Stats>> {
        let _ = params;
        unimplemented!("RenderEngine::stats")
    }

    /// Display-size sRGB render, unrotated.
    pub fn display(&mut self, params: &Params) -> Result<Rgb8Image> {
        let _ = params;
        unimplemented!("RenderEngine::display")
    }

    /// Search-size sRGB render, unrotated.
    pub fn search(&mut self, params: &Params) -> Result<Rgb8Image> {
        let _ = params;
        unimplemented!("RenderEngine::search")
    }
}

/// Full-resolution render with output sharpening and chroma noise
/// reduction. Looks the same as [`RenderEngine::display`] for the same params.
pub fn render_full(
    gpu: &Gpu,
    scene: &Scene,
    full: &LinearImage,
    params: &Params,
) -> Result<Rgb16Image> {
    let _ = (gpu, scene, full, params);
    unimplemented!("render_full")
}
