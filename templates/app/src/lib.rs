//! `__SNAKE__.*`: __SUMMARY__
//!
//! Template op: scales a vector on the GPU and checks it against the CPU.
//! Replace it with the app's real ops; keep the shape (typed input/output,
//! heavy work on a blocking thread, metrics through `cx.metric`).

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use spl_core::{Op, Registry, Result, RunCx, anyhow};
use spl_gpu::cudarc::driver::{LaunchConfig, PushKernelArg};

pub fn register(registry: &mut Registry) {
    registry.op(Scale);
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ScaleInput {
    /// Number of elements.
    #[serde(default = "default_n")]
    pub n: u32,
    #[serde(default = "default_factor")]
    pub factor: f32,
}

fn default_n() -> u32 {
    1 << 20
}

fn default_factor() -> f32 {
    2.0
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct ScaleOutput {
    pub max_abs_error: f32,
    pub kernel_ms: f64,
}

pub struct Scale;

impl Op for Scale {
    const ID: &'static str = "__SNAKE__.scale";
    const SUMMARY: &'static str = "Template: scale a vector on the GPU and verify on the CPU";

    type Input = ScaleInput;
    type Output = ScaleOutput;

    async fn run(&self, cx: &RunCx, input: ScaleInput) -> Result<ScaleOutput> {
        // GPU calls block, so keep them off the async runtime.
        let output = tokio::task::spawn_blocking(move || scale_on_gpu(input)).await??;
        cx.metric("kernel_time", output.kernel_ms, "ms");
        cx.metric("max_abs_error", output.max_abs_error as f64, "");
        Ok(output)
    }
}

fn scale_on_gpu(input: ScaleInput) -> Result<ScaleOutput> {
    let ctx = spl_gpu::context(0)?;
    let module = spl_gpu::load_cuda_source(&ctx, include_str!("../cuda/kernels.cu"))?;
    let kernel = module
        .load_function("scale")
        .map_err(|e| anyhow!("load kernel: {e:?}"))?;
    let stream = ctx.default_stream();

    let host: Vec<f32> = (0..input.n).map(|i| i as f32).collect();
    let mut device = stream
        .clone_htod(&host)
        .map_err(|e| anyhow!("upload: {e:?}"))?;

    let started = std::time::Instant::now();
    let n = input.n as i32;
    let mut launch = stream.launch_builder(&kernel);
    launch.arg(&mut device).arg(&input.factor).arg(&n);
    // SAFETY: argument types and order match `scale` in cuda/kernels.cu.
    unsafe { launch.launch(LaunchConfig::for_num_elems(input.n)) }
        .map_err(|e| anyhow!("launch: {e:?}"))?;
    stream.synchronize().map_err(|e| anyhow!("sync: {e:?}"))?;
    let kernel_ms = started.elapsed().as_secs_f64() * 1e3;

    let result = stream
        .clone_dtoh(&device)
        .map_err(|e| anyhow!("download: {e:?}"))?;
    let max_abs_error = host
        .iter()
        .zip(&result)
        .map(|(x, y)| (x * input.factor - y).abs())
        .fold(0.0, f32::max);
    Ok(ScaleOutput {
        max_abs_error,
        kernel_ms,
    })
}
