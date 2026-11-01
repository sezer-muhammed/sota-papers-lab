//! Camera exports: COLMAP `sparse/0` and nerfstudio `transforms.json`
//! (spec-train 9.3-9.4). Owner: loss (E4).
//!
//! COLMAP: one PINHOLE camera per physical camera at the trained image size,
//! images with `qvec` (w ≥ 0), `tvec`, zero 2D points, and up to 500k Gaussian
//! centres with o > 0.5 sampled ∝ o as points3D. The byte layouts are
//! `crate::io::colmap::ColmapModel::{write_bin, write_txt}` (implemented).
//! transforms.json: `transform_matrix = c2w_cv · diag(1, −1, −1, 1)`.
//!
//! STUBS.

use std::io::Write;
use std::path::Path;

use spl_core::Result;

use crate::gpu::not_implemented;
use crate::io::scene::View;
use crate::model::GaussiansHost;

const OWNER: &str = "loss (E4), exports";

pub fn write_colmap(
    _views: &[View],
    _g: &GaussiansHost,
    _sparse_dir: &Path,
    _text: bool,
) -> Result<()> {
    Err(not_implemented("write_colmap", OWNER))
}

pub fn write_transforms_json(_views: &[View], _out: &mut dyn Write) -> Result<()> {
    Err(not_implemented("write_transforms_json", OWNER))
}
