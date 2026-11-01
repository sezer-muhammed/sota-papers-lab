//! `device_info.*`: what GPUs this host has, and whether CUDA actually works
//! on it. The first thing to run on a new machine.

use std::time::Instant;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use spl_core::{Op, Registry, Result, RunCx, anyhow};
use spl_gpu::GpuReport;
use spl_gpu::cudarc::driver::sys::CUevent_flags;
use spl_gpu::cudarc::driver::{LaunchConfig, PushKernelArg};

pub fn register(registry: &mut Registry) {
    registry.op(List).op(Smoke);
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

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SmokeInput {
    /// GPU to test.
    #[serde(default)]
    pub ordinal: u32,
    /// Vector length; the default moves 192 MiB through the kernel.
    #[serde(default = "default_smoke_n")]
    pub n: u32,
}

fn default_smoke_n() -> u32 {
    1 << 24
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct SmokeOutput {
    pub device: String,
    pub n: u32,
    /// NVRTC compile + module load.
    pub compile_ms: f64,
    /// Best of the timed launches, measured with CUDA events.
    pub kernel_ms: f64,
    /// Effective DRAM bandwidth: 12 bytes per element (read x, read y, write y).
    pub bandwidth_gb_per_s: f64,
    pub max_abs_error: f32,
    pub passed: bool,
}

/// Compiles, launches and verifies a SAXPY kernel: proves driver, NVRTC,
/// memory transfer and kernel launch all work on this host.
pub struct Smoke;

impl Op for Smoke {
    const ID: &'static str = "device_info.smoke";
    const SUMMARY: &'static str = "Compile, launch and verify a CUDA kernel end to end";

    type Input = SmokeInput;
    type Output = SmokeOutput;

    async fn run(&self, cx: &RunCx, input: SmokeInput) -> Result<SmokeOutput> {
        cx.progress(0.0, "compiling kernel");
        let output = tokio::task::spawn_blocking(move || smoke(input)).await??;
        cx.metric("compile_time", output.compile_ms, "ms");
        cx.metric("kernel_time", output.kernel_ms, "ms");
        cx.metric("bandwidth", output.bandwidth_gb_per_s, "GB/s");
        cx.metric("max_abs_error", output.max_abs_error as f64, "");
        if !output.passed {
            return Err(anyhow!(
                "GPU result mismatch: max abs error {}",
                output.max_abs_error
            ));
        }
        Ok(output)
    }
}

const TIMED_LAUNCHES: usize = 5;

fn smoke(input: SmokeInput) -> Result<SmokeOutput> {
    let ctx = spl_gpu::context(input.ordinal as usize)?;
    let device = ctx.name().map_err(|e| anyhow!("device name: {e:?}"))?;

    let compile_started = Instant::now();
    let module = spl_gpu::load_cuda_source(&ctx, include_str!("../cuda/smoke.cu"))?;
    let kernel = module
        .load_function("saxpy")
        .map_err(|e| anyhow!("load kernel: {e:?}"))?;
    let compile_ms = compile_started.elapsed().as_secs_f64() * 1e3;

    let stream = ctx.default_stream();
    let n = input.n;
    let a = 2.0f32;
    let x: Vec<f32> = (0..n).map(|i| (i % 1024) as f32 * 0.5).collect();
    let y: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
    let x_dev = stream
        .clone_htod(&x)
        .map_err(|e| anyhow!("upload x: {e:?}"))?;
    let mut y_dev = stream
        .clone_htod(&y)
        .map_err(|e| anyhow!("upload y: {e:?}"))?;
    let n_arg = n as i32;
    let config = LaunchConfig::for_num_elems(n);

    // One verified launch, then timed launches (y keeps accumulating; only the
    // first result is checked).
    let launch = |y_dev: &mut _| -> Result<()> {
        let mut builder = stream.launch_builder(&kernel);
        builder.arg(&a).arg(&x_dev).arg(y_dev).arg(&n_arg);
        // SAFETY: argument types and order match `saxpy` in cuda/smoke.cu.
        unsafe { builder.launch(config) }.map_err(|e| anyhow!("launch: {e:?}"))?;
        Ok(())
    };
    launch(&mut y_dev)?;
    let result = stream
        .clone_dtoh(&y_dev)
        .map_err(|e| anyhow!("download: {e:?}"))?;
    let max_abs_error = x
        .iter()
        .zip(&y)
        .zip(&result)
        .map(|((x, y), r)| (a * x + y - r).abs())
        .fold(0.0, f32::max);

    let mut kernel_ms = f64::INFINITY;
    for _ in 0..TIMED_LAUNCHES {
        let start = stream
            .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| anyhow!("event: {e:?}"))?;
        launch(&mut y_dev)?;
        let end = stream
            .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| anyhow!("event: {e:?}"))?;
        let ms = start
            .elapsed_ms(&end)
            .map_err(|e| anyhow!("elapsed: {e:?}"))?;
        kernel_ms = kernel_ms.min(f64::from(ms));
    }
    let bandwidth_gb_per_s = 12.0 * f64::from(n) / (kernel_ms * 1e-3) / 1e9;

    Ok(SmokeOutput {
        device,
        n,
        compile_ms,
        kernel_ms,
        bandwidth_gb_per_s,
        max_abs_error,
        passed: max_abs_error <= 1e-5,
    })
}

// Re-exported so this app's UI and other tools can name the output type.
pub use spl_gpu::GpuInfo;
