//! Training loss 0.8·L1 + 0.2·(1 − SSIM) forward and backward, target
//! resampling, the VCD error map, and the eval metric kernels.
//! Owner: loss (E4). Kernels: `cuda/loss.cu`, `cuda/metrics.cu`.
//! Spec: spec-train 3 and 7.4.
//!
//! Buffer contract: reads the raster image (`RenderOut::rgb`, f32 planar
//! `[3][H][W]`) and the resampled target (same layout); writes dL/dimage into
//! `PixelGrads::d_rgb` (and d_alpha when a mask or alpha loss is on); adds
//! Σ|x − y| and Σ S into a 2-float device accumulator. Sizes come from
//! `IterParams.frame.render_w/h`; nothing syncs the host.
//!
//! STUBS: everything except the config types returns "not implemented".

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use spl_core::Result;
use spl_gpu::cudarc::driver::{CudaFunction, CudaSlice};
use ts_rs::TS;

use crate::frame::IterParamsDev;
use crate::gpu::{
    COMMON_CUH, CudaFile, Defines, Gpu, KernelSource, SPLAT_COMMON_CUH, not_implemented,
};
use crate::io::images::ImageCache;

const OWNER: &str = "loss (E4)";

pub const SOURCE: KernelSource = KernelSource {
    module: "loss",
    files: &[
        COMMON_CUH,
        SPLAT_COMMON_CUH,
        CudaFile {
            name: "loss.cu",
            text: include_str!("../../cuda/loss.cu"),
        },
    ],
    kernels: &[
        "resample_target",
        "ssim_fwd",
        "ssim_bwd",
        "ssim_fused",
        "error_map",
    ],
};

pub const METRICS_SOURCE: KernelSource = KernelSource {
    module: "metrics",
    files: &[
        COMMON_CUH,
        SPLAT_COMMON_CUH,
        CudaFile {
            name: "metrics.cu",
            text: include_str!("../../cuda/metrics.cu"),
        },
    ],
    kernels: &["psnr_u8", "ssim_u8"],
};

/// The 11-tap Gaussian (σ = 1.5), normalized; matches fused-ssim's `cGauss` to 1e-9.
pub const SSIM_WINDOW: [f32; 11] = [
    0.001_028_380_1,
    0.007_598_758,
    0.036_000_773,
    0.109_360_69,
    0.213_005_54,
    0.266_011_72,
    0.213_005_54,
    0.109_360_69,
    0.036_000_773,
    0.007_598_758,
    0.001_028_380_1,
];
pub const SSIM_C1: f32 = 0.01 * 0.01;
pub const SSIM_C2: f32 = 0.03 * 0.03;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export, export_to = "image-to-3d/")]
pub enum SsimKernel {
    /// v1: fused-ssim's two-pass design (the reference).
    TwoPass,
    /// v2: one kernel that recomputes the halo (default once it matches v1 to 1e-6).
    #[default]
    Fused,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export, export_to = "image-to-3d/")]
pub enum Precision {
    #[default]
    F32,
    Bf16,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct LossConfig {
    pub lambda_l1: f32,
    pub lambda_dssim: f32,
    pub kernel: SsimKernel,
    /// Storage of v1's partial-derivative planes (bf16 is an ablation).
    pub partials: Precision,
}

impl Default for LossConfig {
    fn default() -> Self {
        Self {
            lambda_l1: 0.8,
            lambda_dssim: 0.2,
            kernel: SsimKernel::Fused,
            partials: Precision::F32,
        }
    }
}

/// Target as kernels see it: u8 RGB interleaved in the image pool.
pub struct Target<'a> {
    pub pool: &'a CudaSlice<u8>,
    pub offset: usize,
    pub width: u32,
    pub height: u32,
}

/// VCD eval-mode output: per-pixel mean abs error, its min/max, the view loss sums.
pub struct ErrorMap {
    pub err: CudaSlice<f32>,
    /// [min, max]
    pub minmax: CudaSlice<f32>,
    /// [Σ|x − y|, Σ S]
    pub sums: CudaSlice<f32>,
}

/// Means since the last read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct LossValue {
    pub l1: f64,
    pub ssim: f64,
    pub loss: f64,
}

pub struct LossKernels {
    pub resample_target: CudaFunction,
    pub ssim_fwd: CudaFunction,
    pub ssim_bwd: CudaFunction,
    pub ssim_fused: CudaFunction,
    pub error_map: CudaFunction,
}

impl LossKernels {
    pub fn load(gpu: &Gpu) -> Result<Self> {
        let k = gpu.kernels(&SOURCE, &Defines::new())?;
        Ok(Self {
            resample_target: k.get("resample_target")?,
            ssim_fwd: k.get("ssim_fwd")?,
            ssim_bwd: k.get("ssim_bwd")?,
            ssim_fused: k.get("ssim_fused")?,
            error_map: k.get("error_map")?,
        })
    }
}

/// Kernels, partial planes sized for the largest image, the 2-float accumulator.
pub struct Loss {
    pub cfg: LossConfig,
    pub kernels: LossKernels,
    pub max_width: u32,
    pub max_height: u32,
    /// v1 partial planes [3][3][H][W] (empty for v2).
    pub partials: Option<CudaSlice<f32>>,
    /// [Σ|x − y|, Σ S], read by `take_sums`.
    pub sums: CudaSlice<f32>,
}

impl Loss {
    pub fn new(_gpu: &Gpu, _max_width: u32, _max_height: u32, _cfg: LossConfig) -> Result<Self> {
        Err(not_implemented("Loss::new", OWNER))
    }

    /// Forward + backward: writes dL/dimage (f32 planar), adds Σ|x−y| and ΣS to
    /// the accumulator. Image size from IterParams; no host sync.
    pub fn forward_backward(
        &mut self,
        _gpu: &Gpu,
        _params: &IterParamsDev,
        _image: &CudaSlice<f32>,
        _target: &CudaSlice<f32>,
        _d_image: &mut CudaSlice<f32>,
    ) -> Result<()> {
        Err(not_implemented("Loss::forward_backward", OWNER))
    }

    /// Eval mode for VCD: error map, min/max, view loss.
    pub fn error_map(
        &mut self,
        _gpu: &Gpu,
        _params: &IterParamsDev,
        _image: &CudaSlice<f32>,
        _target: &CudaSlice<f32>,
        _out: &mut ErrorMap,
    ) -> Result<()> {
        Err(not_implemented("Loss::error_map", OWNER))
    }

    /// Reads and resets the accumulator (host sync).
    pub fn take_sums(&mut self, _gpu: &Gpu, _iterations: u32) -> Result<LossValue> {
        Err(not_implemented("Loss::take_sums", OWNER))
    }
}

/// Resamples the cached u8 target of this iteration to the render size (copy at
/// scale 1, antialiased tent filter otherwise) into f32 planar.
pub fn resample_target(
    _gpu: &Gpu,
    _k: &LossKernels,
    _params: &IterParamsDev,
    _cache: &ImageCache,
    _out: &mut CudaSlice<f32>,
) -> Result<()> {
    Err(not_implemented("resample_target", OWNER))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The window is the normalized σ = 1.5 Gaussian.
    #[test]
    fn ssim_window_is_the_normalized_gaussian() {
        let raw: Vec<f64> = (-5..=5)
            .map(|i| (-(i * i) as f64 / (2.0 * 1.5 * 1.5)).exp())
            .collect();
        let sum: f64 = raw.iter().sum();
        for (w, r) in SSIM_WINDOW.iter().zip(&raw) {
            assert!((*w as f64 - r / sum).abs() < 1e-8, "{w} vs {}", r / sum);
        }
    }
}
