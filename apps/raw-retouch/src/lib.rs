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
//! judge = judge, encode (PNG: media), xmp;
//! media = place, RawInfo's date and position (decode), the PNG export
//! and zoomed region renders (ops/region.rs);
//! integration = ops;
//! detail = finish, detail, cuda/detail.cu; lens = lens, cuda/lens.cu.
//! The render owner shares `render` with per-effect owners: tone curve =
//! render/tone_curve.rs, cuda/tone_curve.cu; colour mixer =
//! render/color_mixer.rs, cuda/color_mixer.cu; presence =
//! render/presence.rs, cuda/presence.cu; black and white = render/bw.rs,
//! cuda/bw.cu.

pub mod auto;
pub mod cmaes;
pub mod color;
pub mod decode;
pub mod detail;
pub mod develop;
pub mod encode;
pub mod finish;
pub mod judge;
pub mod lens;
pub mod objective;
pub mod ops;
pub mod place;
pub mod render;
pub mod search;
pub mod types;
pub mod xmp;

use spl_core::Registry;

pub fn register(registry: &mut Registry) {
    registry.op(ops::Inspect).op(ops::Auto).op(ops::Render);
}
