//! `image_to_3d.*`: unposed images or video to Gaussian splats and meshes.
//!
//! Design: `reports/Image to 3D.md` (git-excluded); the Phase 3 splat
//! optimizer follows `reports/image-to-3d/spec-raster.md` (rasterizer) and
//! `reports/image-to-3d/spec-train.md` (everything else). The contract
//! between modules (buffers, ownership, call order per iteration) is
//! `apps/image-to-3d/DESIGN.md`.
//!
//! Ops ([`ops`]): `image_to_3d.train` (COLMAP-posed dataset → splats) and
//! `image_to_3d.render` (one view of a trained splat).
//!
//! Module owners (one per file; the header of each file names it):
//! - preprocess (E1): `kernels/{mod,layout,preprocess,selftest}.rs`,
//!   `cuda/{common,splat_common}.cuh`, `cuda/{preprocess,selftest}.cu`
//! - sort (E2): `kernels/sort.rs`, `cuda/{radix,binning}.cu`
//! - raster (E3): `kernels/raster.rs`, `cuda/{raster_fwd,raster_bwd}.cu`
//! - loss (E4): `kernels/loss.rs`, `eval.rs`, `export/*`, `cuda/{loss,metrics}.cu`
//! - optimizer (E5): `kernels/adam.rs`, `reference.rs`, `cuda/adam.{cuh,cu}`
//! - densify (E6): `kernels/densify.rs`, `model.rs`, `cuda/{densify,compact,mcmc}.cu`
//! - trainer (E7): `gpu.rs`, `frame.rs`, `rng.rs`, `presets.rs`,
//!   `trainer.rs`, `ops.rs`, `refine.rs`, `io/*`, `testkit/*`
//! - ingest (A0): `ingest/*`, `cuda/ingest.cu`

pub mod eval;
pub mod export;
pub mod frame;
pub mod gpu;
pub mod ingest;
pub mod io;
pub mod kernels;
pub mod model;
pub mod ops;
pub mod presets;
pub mod reference;
pub mod refine;
pub mod rng;
pub mod testkit;
pub mod trainer;

use spl_core::Registry;

pub fn register(registry: &mut Registry) {
    registry.op(ops::Train).op(ops::Render);
}
