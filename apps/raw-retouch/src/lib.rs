//! `raw_retouch.*`: automatic RAW photo retouching.
//!
//! Decode the RAW (rawler), develop it on the GPU, search Lightroom-style
//! sliders per photo against a constrained objective (CMA-ES over a CUDA
//! renderer), let Claude plan looks and judge the finalists pairwise, then
//! export a full-resolution JPEG and an XMP sidecar. The design follows
//! `reports/Automatic RAW photo retouching.md`: render fast, constrain hard,
//! judge pairwise.
//!
//! Module owners (one per module, so work can proceed in parallel):
//! develop = decode, color, develop, cuda/develop.cu;
//! render = render, cuda/render.cu;
//! search = objective, cmaes, auto, search;
//! judge = judge, encode, xmp;
//! integration = ops.

pub mod auto;
pub mod cmaes;
pub mod color;
pub mod decode;
pub mod develop;
pub mod encode;
pub mod judge;
pub mod objective;
pub mod ops;
pub mod render;
pub mod search;
pub mod types;
pub mod xmp;

use spl_core::Registry;

pub fn register(registry: &mut Registry) {
    registry.op(ops::Inspect).op(ops::Auto).op(ops::Render);
}
