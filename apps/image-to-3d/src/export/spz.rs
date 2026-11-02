//! Niantic SPZ v3 (gzip, default) and v4 (per-stream zstd) (spec-train 9.2).
//! Owner: loss (E4). Format after niantic `spz` (MIT, affd0ec): our own
//! encoder and a decoder for tests.
//!
//! Streams per Gaussian: positions 24-bit fixed point (fractional bits 12 by
//! default, lowered until |p| < 2^(23−fb)), alpha u8, colour u8 from the SH DC
//! coefficient, log scales u8, smallest-three rotation u32, SH coefficients
//! quantized to 5 bits (degree 1) / 4 bits (rest). SPZ is RUB; we write RDF
//! with the (1, −1, −1) flip on positions, quaternion xyz and the SH sign table.
//!
//! STUBS (types are final).

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use spl_core::Result;
use ts_rs::TS;

use crate::gpu::not_implemented;
use crate::model::GaussiansHost;

const OWNER: &str = "loss (E4), exports";

pub const SPZ_MAGIC: u32 = 0x5053_474e;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct SpzOptions {
    /// 3 (gzip, default: what released viewers decode) or 4 (zstd streams).
    pub version: u32,
    /// Truncate SH to this degree (e.g. 1 for live snapshots); None keeps it.
    #[ts(optional)]
    pub sh_degree: Option<u32>,
    pub sh1_bits: u8,
    pub sh_rest_bits: u8,
    /// Position fractional bits; None = 12, lowered automatically when needed.
    #[ts(optional)]
    pub fractional_bits: Option<u8>,
}

impl Default for SpzOptions {
    fn default() -> Self {
        Self {
            version: 3,
            sh_degree: None,
            sh1_bits: 5,
            sh_rest_bits: 4,
            fractional_bits: None,
        }
    }
}

pub fn write_spz(_g: &GaussiansHost, _opts: &SpzOptions) -> Result<Vec<u8>> {
    Err(not_implemented("write_spz", OWNER))
}

pub fn read_spz(_bytes: &[u8]) -> Result<GaussiansHost> {
    Err(not_implemented("read_spz", OWNER))
}
