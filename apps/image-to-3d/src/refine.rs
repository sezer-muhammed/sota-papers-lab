//! Phase 4 interfaces, wired now so the preprocess backward can expose camera
//! gradients: pose/focal refinement and PPISP. Owner: trainer (E7) until
//! Phase 4 staffs it. Spec: spec-train 10.
//!
//! STUBS.

use spl_core::Result;

use crate::gpu::{Gpu, not_implemented};
use crate::kernels::CameraGrads;

const OWNER: &str = "Phase 4 (WP-P)";

/// Which camera parameters are optimized (10.1). COLMAP-posed presets keep it Off.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PoseRefine {
    #[default]
    Off,
    /// One shared correction folded into every pose (1-minute challenge).
    Global,
    PerImage {
        focal: bool,
        principal_point: bool,
    },
}

/// Adam over per-image tangents (ρ, φ) and log-focal κ, applied as
/// `T_cw ← exp(Δξ^)·T_cw` with re-zeroed tangents; image 0 frozen.
pub struct PoseOptimizer {
    pub mode: PoseRefine,
}

impl PoseOptimizer {
    pub fn step(&mut self, _gpu: &Gpu, _grads: &CameraGrads, _t: u32) -> Result<()> {
        Err(not_implemented("PoseOptimizer::step", OWNER))
    }
}

/// PPISP: exposure, vignetting, colour homography, CRF between the raster and
/// the loss (10.2).
pub struct Ppisp;

impl Ppisp {
    pub fn new(_gpu: &Gpu, _n_frames: u32, _n_cameras: u32) -> Result<Self> {
        Err(not_implemented("Ppisp::new", OWNER))
    }
}
