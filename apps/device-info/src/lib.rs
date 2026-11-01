//! `device_info.*`: what GPUs this host has. The smoke test for a new machine.

use schemars::JsonSchema;
use serde::Deserialize;
use spl_core::{Op, Registry, Result, RunCx};
use spl_gpu::GpuReport;

pub fn register(registry: &mut Registry) {
    registry.op(List);
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct ListInput {}

pub struct List;

impl Op for List {
    const ID: &'static str = "device_info.list";
    const SUMMARY: &'static str = "List CUDA devices, driver version and memory";

    type Input = ListInput;
    type Output = GpuReport;

    async fn run(&self, cx: &RunCx, _input: ListInput) -> Result<GpuReport> {
        let report = tokio::task::spawn_blocking(spl_gpu::probe).await?;
        cx.metric("gpu_count", report.devices.len() as f64, "");
        for gpu in &report.devices {
            cx.metric(
                format!("gpu{}_memory", gpu.ordinal),
                gpu.total_memory_bytes as f64 / (1u64 << 30) as f64,
                "GiB",
            );
        }
        Ok(report)
    }
}

// Re-exported so this app's UI and other tools can name the output type.
pub use spl_gpu::GpuInfo;
