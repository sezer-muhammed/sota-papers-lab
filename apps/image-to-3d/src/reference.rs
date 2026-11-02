//! CPU f64 reference rasterizer, written from the specs independently of the
//! CUDA code, validated by central finite differences. Owner: optimizer (E5),
//! who also owns the gsplat oracle fixtures (`tests/fixtures/`, generated with
//! `gs-oracle`). Spec: spec-raster 6.3 layer 1, spec-train 2.1 ("reference
//! rasterizer": naive per-pixel, up to ~10k Gaussians at 128×96).
//!
//! It covers preprocess (projection, filters, compensation, SH, culling),
//! exact tile spans (brute force over pixel centres), blending with gsplat
//! semantics, and the full backward including pose and focal gradients.
//! Kernel owners test against it: E1 (preprocess), E2 (tile lists), E3
//! (images, T, n_contrib, 2D gradients), E1 again for 3D and camera gradients.
//!
//! STUBS: types are final, functions return "not implemented".

use spl_core::Result;

use crate::gpu::not_implemented;
use crate::io::scene::View;
use crate::kernels::BlendSemantics;
use crate::model::GaussiansHost;

const OWNER: &str = "optimizer (E5), reference";

/// Settings the reference shares with a kernel run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RefOptions {
    pub sh_degree: u32,
    pub background: [f64; 3],
    pub eps2d: f64,
    pub near: f64,
    pub far: f64,
    /// Opacity compensation (Mip-Splatting 2D / gsplat "antialiased").
    pub antialiased: bool,
    /// Gradient through the compensation (gsplat oracle mode).
    pub aa_grad: bool,
    pub blend: BlendSemantics,
    pub compact_box_beta: f64,
}

impl Default for RefOptions {
    fn default() -> Self {
        Self {
            sh_degree: 3,
            background: [0.0; 3],
            eps2d: 0.3,
            near: 0.01,
            far: 1e4,
            antialiased: false,
            aa_grad: false,
            blend: BlendSemantics::Gsplat,
            compact_box_beta: 1.0,
        }
    }
}

/// Per-Gaussian preprocess results (culled Gaussians have `visible = false`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RefProjected {
    pub visible: Vec<bool>,
    pub means2d: Vec<[f64; 2]>,
    /// (A, B, C) of the inverse 2D covariance incl. ε (scalar-B convention).
    pub conics: Vec<[f64; 3]>,
    pub depths: Vec<f64>,
    pub opacity_eff: Vec<f64>,
    pub colors: Vec<[f64; 3]>,
    /// Tiles touched (exact, by pixel centres).
    pub n_tiles: Vec<u32>,
}

/// Forward outputs, planar like the kernels'.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RefImage {
    pub width: u32,
    pub height: u32,
    /// [3][H][W], background composited.
    pub rgb: Vec<f64>,
    pub trans: Vec<f64>,
    /// Index + 1 of the last blended Gaussian in depth order; 0 = none.
    pub n_contrib: Vec<u32>,
}

/// Gradients of every parameter group (host layouts) and the camera.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RefGrads {
    pub means: Vec<f64>,
    pub scales: Vec<f64>,
    pub rotations: Vec<f64>,
    pub opacities: Vec<f64>,
    pub sh0: Vec<f64>,
    pub sh_rest: Vec<f64>,
    /// 2D gradients per Gaussian: mx, my, A, B, C, o, r, g, b.
    pub g2d: Vec<[f64; 9]>,
    /// (ρ, φ) left perturbation.
    pub pose: [f64; 6],
    /// fx, fy, cx, cy.
    pub intr: [f64; 4],
}

pub fn project(_g: &GaussiansHost, _view: &View, _opts: &RefOptions) -> Result<RefProjected> {
    Err(not_implemented("reference::project", OWNER))
}

pub fn render(_g: &GaussiansHost, _view: &View, _opts: &RefOptions) -> Result<RefImage> {
    Err(not_implemented("reference::render", OWNER))
}

/// Backward of `Σ d_rgb ⊙ rgb + Σ d_alpha ⊙ (1 − T)`.
pub fn backward(
    _g: &GaussiansHost,
    _view: &View,
    _opts: &RefOptions,
    _d_rgb: &[f64],
    _d_alpha: Option<&[f64]>,
) -> Result<RefGrads> {
    Err(not_implemented("reference::backward", OWNER))
}
