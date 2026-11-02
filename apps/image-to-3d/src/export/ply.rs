//! INRIA 3DGS PLY (spec-train 9.1). Owner: loss (E4).
//!
//! `binary_little_endian 1.0`, one `vertex` element of 62 f32 properties in
//! the order x y z nx ny nz f_dc_0..2 f_rest_0..44 opacity scale_0..2
//! rot_0..3; `f_rest_{c·15 + k}` is coefficient k of channel c
//! (channel-major, the transpose of our `[15][3]`); always the degree-3
//! layout (zeros above the trained degree); normals 0; opacity is the logit,
//! scales are logs, rotation is normalized (w, x, y, z). Viewers treat the
//! file as RDF.
//!
//! STUBS.

use std::io::Write;

use spl_core::Result;

use crate::gpu::not_implemented;
use crate::model::GaussiansHost;

const OWNER: &str = "loss (E4), exports";

/// Floats per vertex in the INRIA layout.
pub const PLY_FLOATS: usize = 62;

pub fn write_inria_ply(_g: &GaussiansHost, _out: &mut dyn Write) -> Result<()> {
    Err(not_implemented("write_inria_ply", OWNER))
}

pub fn read_inria_ply(_bytes: &[u8]) -> Result<GaussiansHost> {
    Err(not_implemented("read_inria_ply", OWNER))
}
