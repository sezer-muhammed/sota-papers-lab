//! Exports: INRIA PLY, SPZ v3/v4, COLMAP, transforms.json. Owner: loss (E4).
//! Spec: spec-train 9. All CPU, from a [`crate::model::GaussiansHost`] that
//! has been flushed (LazyCatchUp), pruned (o < 1/255) and Morton-ordered.

pub mod colmap;
pub mod ply;
pub mod spz;

/// Media types of the exported files (artifact kinds `splat` / `cameras` come
/// with the platform3d work; until then they are stored as `blob`).
pub const MEDIA_PLY: &str = "application/ply";
pub const MEDIA_SPZ: &str = "application/x-spz";
pub const MEDIA_ZIP: &str = "application/zip";
pub const MEDIA_JSON: &str = "application/json";
