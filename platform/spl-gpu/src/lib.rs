//! CUDA access for apps, built on `cudarc`.
//!
//! The CUDA driver and NVRTC are loaded when first used, not at link time, so
//! this crate compiles everywhere and only fails at runtime on machines
//! without an NVIDIA driver. [`probe`] turns that failure into a value.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

pub use cudarc;
use cudarc::driver::{CudaContext, CudaModule, sys};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use spl_core::{Context, Result, anyhow};
use ts_rs::TS;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, TS)]
#[ts(export)]
pub struct GpuInfo {
    pub ordinal: u32,
    pub name: String,
    /// `"8.9"` style; pick nvcc/NVRTC targets from this.
    pub compute_capability: String,
    #[ts(type = "number")]
    pub total_memory_bytes: u64,
    pub multiprocessors: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, TS)]
#[ts(export)]
pub struct GpuReport {
    /// `"12.8"` style, or `None` when no driver could be loaded.
    pub driver_version: Option<String>,
    pub devices: Vec<GpuInfo>,
    /// Why no devices were found, when that is the case.
    pub error: Option<String>,
}

/// Lists CUDA devices without ever panicking, even with no driver installed.
pub fn probe() -> GpuReport {
    // Checking first keeps cudarc from printing a panic on driverless hosts;
    // catch_unwind stays as the backstop.
    if !unsafe { sys::is_culib_present() } {
        return GpuReport {
            driver_version: None,
            devices: Vec::new(),
            error: Some("CUDA driver library not found on this machine".into()),
        };
    }
    match catch_unwind(AssertUnwindSafe(probe_inner)) {
        Ok(Ok(report)) => report,
        Ok(Err(e)) => GpuReport {
            driver_version: None,
            devices: Vec::new(),
            error: Some(format!("{e:#}")),
        },
        Err(_) => GpuReport {
            driver_version: None,
            devices: Vec::new(),
            error: Some("CUDA driver library not found on this machine".into()),
        },
    }
}

fn probe_inner() -> Result<GpuReport> {
    let mut raw = 0;
    unsafe { sys::cuDriverGetVersion(&mut raw) }
        .result()
        .map_err(|e| anyhow!("cuDriverGetVersion: {e:?}"))?;
    let driver_version = Some(format!("{}.{}", raw / 1000, (raw % 1000) / 10));

    let count = CudaContext::device_count().map_err(|e| anyhow!("device count: {e:?}"))?;
    let mut devices = Vec::with_capacity(count.max(0) as usize);
    for ordinal in 0..count.max(0) as usize {
        let ctx = context(ordinal)?;
        let (major, minor) = ctx
            .compute_capability()
            .map_err(|e| anyhow!("compute capability: {e:?}"))?;
        devices.push(GpuInfo {
            ordinal: ordinal as u32,
            name: ctx.name().map_err(|e| anyhow!("device name: {e:?}"))?,
            compute_capability: format!("{major}.{minor}"),
            total_memory_bytes: ctx.total_mem().map_err(|e| anyhow!("total mem: {e:?}"))? as u64,
            multiprocessors: ctx
                .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
                .map_err(|e| anyhow!("SM count: {e:?}"))? as u32,
        });
    }
    Ok(GpuReport {
        driver_version,
        devices,
        error: None,
    })
}

pub fn context(ordinal: usize) -> Result<Arc<CudaContext>> {
    if !unsafe { sys::is_culib_present() } {
        return Err(anyhow!(
            "CUDA driver library not found; run this op on a host with an NVIDIA driver"
        ));
    }
    CudaContext::new(ordinal).map_err(|e| anyhow!("open CUDA device {ordinal}: {e:?}"))
}

/// Compiles CUDA C++ source with NVRTC at runtime and loads it.
/// Apps keep kernels in `cuda/*.cu` and pass them in with `include_str!`.
pub fn load_cuda_source(ctx: &Arc<CudaContext>, source: &str) -> Result<Arc<CudaModule>> {
    let ptx = cudarc::nvrtc::compile_ptx(source).context("NVRTC compile failed")?;
    ctx.load_module(ptx)
        .map_err(|e| anyhow!("load CUDA module: {e:?}"))
}
