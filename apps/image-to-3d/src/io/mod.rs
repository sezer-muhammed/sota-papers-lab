//! Dataset input: COLMAP models, scenes (views, split, extent, points) and
//! images (decode, downscale, GPU cache). Owner: trainer (E7).

pub mod colmap;
pub mod images;
pub mod scene;

#[cfg(test)]
mod dataset_tests;
