//! Transformer building blocks on CUDA for the native VGGT-Ω engine
//! (`reports/Image to 3D.md`, Section 3, Stage A).
//!
//! Minimal crate root so [`sage`] builds on its own; the full spl-nn root
//! (GEMMs, norms, QKV prologue, FlashAttention, `Nn`) lands from a parallel
//! branch, and the integrator keeps that file and adds `pub mod sage;`.

pub use cudarc;

pub mod sage;
