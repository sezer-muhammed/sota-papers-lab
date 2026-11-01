//! `raw_retouch.*`: automatic RAW photo retouching.
//!
//! Decode the RAW (rawler), develop it on the GPU, search Lightroom-style
//! sliders per photo against a constrained objective (CMA-ES over a CUDA
//! renderer), let Claude plan looks and judge the finalists pairwise, then
//! export a full-resolution JPEG and an XMP sidecar. The design follows
//! `reports/Automatic RAW photo retouching.md`: render fast, constrain hard,
//! judge pairwise.
//!
//! Ops ([`ops`]): `raw_retouch.inspect` (metadata and a neutral preview),
//! `raw_retouch.auto` (the whole pipeline, Claude optional) and
//! `raw_retouch.render` (given sliders, as a live preview or the full-size
//! files). A process keeps its four most recently developed RAWs, so the
//! page's slider previews after an auto run skip decoding. See the app's
//! README for inputs, stages and limits.
//!
//! Module owners (one per module, so work can proceed in parallel):
//! develop = decode, color, develop, cuda/develop.cu;
//! render = render, cuda/render.cu;
//! search = objective, cmaes, auto, search;
//! judge = judge, encode, xmp;
//! detail = detail, cuda/detail.cu; lens = lens, cuda/lens.cu;
//! integration = ops.

pub mod auto;
pub mod cmaes;
pub mod color;
pub mod decode;
pub mod detail;
pub mod develop;
pub mod encode;
pub mod judge;
pub mod lens;
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
