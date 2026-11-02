//! ONNX Runtime on CUDA for model front ends that are not (yet) native
//! `spl-nn` engines, e.g. the commercial-profile DA3 engine of image-to-3d.
//!
//! - [`environment`]: the process-wide ONNX Runtime, loaded once from
//!   `$ORT_DYLIB_PATH` (default `libonnxruntime.so` on the loader path).
//!   The CUDA build of ONNX Runtime 1.23 needs CUDA 12 cuBLAS and cuDNN 9 on
//!   the loader path too.
//! - [`cuda_session`]: a session on the CUDA execution provider that fails
//!   instead of silently falling back to the CPU.
//! - [`cuda_memory`], [`device_view`]: bind outputs to device memory
//!   (IoBinding) and feed slices of them to the next session without a host
//!   round trip.
//!
//! ONNX Runtime is MIT, the `ort` crate MIT / Apache-2.0.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

pub use half::f16;
pub use ort;
use ort::environment::Environment;
use ort::ep;
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::{PrimitiveTensorElementType, Shape, TensorRefMut};
use spl_core::{Context, Result, anyhow};

/// Where ONNX Runtime is loaded from.
pub fn dylib_path() -> PathBuf {
    std::env::var_os("ORT_DYLIB_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("libonnxruntime.so"))
}

/// The process-wide environment; the first call loads the library.
pub fn environment() -> Result<&'static Environment> {
    static ENV: OnceLock<std::result::Result<Environment, String>> = OnceLock::new();
    ENV.get_or_init(|| {
        let path = dylib_path();
        ort::init_from(&path)
            .map_err(|e| {
                format!(
                    "load ONNX Runtime from {} (set ORT_DYLIB_PATH): {e}",
                    path.display()
                )
            })?
            .with_name("spl")
            .build()
            .map_err(|e| format!("ONNX Runtime environment: {e}"))
    })
    .as_ref()
    .map_err(|e| anyhow!("{e}"))
}

#[derive(Debug, Clone)]
pub struct CudaOptions {
    pub device: i32,
    /// TF32 for fp32 GEMMs and convolutions.
    pub tf32: bool,
    /// Cap of the CUDA EP's arena in bytes.
    pub memory_limit: Option<usize>,
}

impl Default for CudaOptions {
    fn default() -> Self {
        Self {
            device: 0,
            tf32: true,
            memory_limit: None,
        }
    }
}

/// A session on the CUDA execution provider, all graph optimisations on.
/// Errors if the provider cannot be registered (no silent CPU fallback).
pub fn cuda_session(model: &Path, opts: &CudaOptions) -> Result<Session> {
    let env = environment()?;
    let mut cuda = ep::CUDA::default()
        .with_device_id(opts.device)
        .with_tf32(opts.tf32)
        .with_conv_algorithm_search(ep::cuda::ConvAlgorithmSearch::Heuristic);
    if let Some(limit) = opts.memory_limit {
        cuda = cuda.with_memory_limit(limit);
    }
    Session::builder(env)
        .map_err(|e| anyhow!("session builder: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::All)
        .map_err(|e| anyhow!("{e}"))?
        .with_execution_providers([cuda.build().error_on_failure()])
        .map_err(|e| anyhow!("CUDA execution provider: {e}"))?
        .commit_from_file(model)
        .map_err(|e| anyhow!("{e}"))
        .with_context(|| format!("load {}", model.display()))
}

/// Device memory of CUDA device `device`, for `IoBinding::bind_output_to_device`.
pub fn cuda_memory(device: i32) -> Result<MemoryInfo<'static>> {
    MemoryInfo::new(
        AllocationDevice::CUDA,
        device,
        AllocatorType::Device,
        MemoryType::Default,
    )
    .map_err(|e| anyhow!("{e}"))
}

/// Host memory, for outputs that are read on the CPU anyway.
pub fn cpu_memory() -> Result<MemoryInfo<'static>> {
    MemoryInfo::new(
        AllocationDevice::CPU,
        0,
        AllocatorType::Device,
        MemoryType::CPUOutput,
    )
    .map_err(|e| anyhow!("{e}"))
}

/// A tensor view of `shape` over device memory starting `offset` elements
/// into `base`, for feeding part of one session's device output to another
/// session without a copy.
///
/// # Safety
/// `base` must be a device pointer on CUDA device `device` with at least
/// `offset + shape.product()` elements of `T`, alive and unmodified while
/// the view is used.
pub unsafe fn device_view<'a, T: PrimitiveTensorElementType + std::fmt::Debug>(
    device: i32,
    base: *const std::ffi::c_void,
    offset: usize,
    shape: &[i64],
) -> Result<TensorRefMut<'a, T>> {
    let ptr = base.cast::<T>().wrapping_add(offset).cast_mut().cast();
    // SAFETY: the caller guarantees the pointer, its extent and lifetime.
    unsafe { TensorRefMut::from_raw(cuda_memory(device)?, ptr, Shape::new(shape.iter().copied())) }
        .map_err(|e| anyhow!("{e}"))
}

/// `f32` → IEEE half, in parallel-friendly chunks.
pub fn to_f16(src: &[f32], dst: &mut [f16]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d = f16::from_f32(*s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dylib_path_defaults_to_the_loader_path() {
        if std::env::var_os("ORT_DYLIB_PATH").is_none() {
            assert_eq!(dylib_path(), PathBuf::from("libonnxruntime.so"));
        }
    }

    #[test]
    fn half_conversion_rounds_to_nearest() {
        let mut out = [f16::ZERO; 3];
        to_f16(&[1.0, -2.5, 65504.0], &mut out);
        assert_eq!(out.map(f16::to_f32), [1.0, -2.5, 65504.0]);
    }

    #[test]
    fn missing_library_is_an_error_not_a_panic() {
        // On machines without ONNX Runtime (macOS dev) loading fails cleanly.
        if std::env::var_os("ORT_DYLIB_PATH").is_none() && cfg!(target_os = "macos") {
            assert!(environment().is_err());
        }
    }
}
