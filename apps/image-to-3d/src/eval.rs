//! Evaluation protocol: test renders, PSNR, SSIM (same and valid), LPIPS
//! later. Owner: loss (E4). Kernels: `cuda/metrics.cu`. Spec: spec-train 7.
//!
//! Renders are clamped and quantized like `torchvision.save_image`
//! (`u8 = floor(x·255 + 0.5)`), compared with the u8 ground truth. Rust PSNR
//! and SSIM must equal the parity scripts to 1e-4 dB / 1e-5 (a test).
//!
//! Implemented: the CPU PSNR used as the reference. STUBS: GPU metrics, the
//! protocol runner.

use std::path::Path;

use anyhow::ensure;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use spl_core::Result;
use ts_rs::TS;

use crate::gpu::{Gpu, not_implemented};
use crate::io::images::Rgb8Image;
use crate::io::scene::Scene;
use crate::trainer::Trainer;

const OWNER: &str = "loss (E4)";

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct ImageMetrics {
    pub psnr: f64,
    /// Zero "same" padding, mean over all pixels (Inria metrics.py).
    pub ssim: f64,
    /// Mean over pixels ≥ 5 from every border (torchmetrics).
    pub ssim_valid: f64,
    /// LPIPS-VGG with inputs mapped to [−1, 1].
    #[ts(optional)]
    pub lpips_vgg: Option<f64>,
    /// LPIPS-VGG fed [0, 1] (Inria lpipsPyTorch convention).
    #[ts(optional)]
    pub lpips_vgg_inria: Option<f64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct EvalReport {
    /// Means over the test views.
    pub mean: ImageMetrics,
    pub per_view: Vec<(String, ImageMetrics)>,
}

/// `−10·log10(mean over 3HW of ((r − g)/255)²)` on u8 values (∞ when equal).
pub fn psnr_u8(render: &Rgb8Image, gt: &Rgb8Image) -> Result<f64> {
    ensure!(
        (render.width, render.height) == (gt.width, gt.height),
        "size mismatch {}x{} vs {}x{}",
        render.width,
        render.height,
        gt.width,
        gt.height
    );
    let se: u64 = render
        .data
        .iter()
        .zip(&gt.data)
        .map(|(a, b)| (*a as i64 - *b as i64).pow(2) as u64)
        .sum();
    let mse = se as f64 / (render.data.len() as f64 * 255.0 * 255.0);
    Ok(-10.0 * mse.log10())
}

/// PSNR, both SSIMs on the GPU (`metrics.cu`).
pub fn image_metrics(_gpu: &Gpu, _render: &Rgb8Image, _gt: &Rgb8Image) -> Result<ImageMetrics> {
    Err(not_implemented("image_metrics", OWNER))
}

/// Renders every test view, saves PNGs under `out_dir` when given, and averages the metrics.
pub fn evaluate(
    _trainer: &mut Trainer,
    _scene: &Scene,
    _out_dir: Option<&Path>,
) -> Result<EvalReport> {
    Err(not_implemented("evaluate", OWNER))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn psnr_of_a_one_level_error_everywhere() {
        let a = Rgb8Image::new(4, 3);
        let mut b = a.clone();
        b.data.iter_mut().for_each(|v| *v = 1);
        // mse = 1/255^2 -> 20 log10(255) = 48.13 dB.
        assert!((psnr_u8(&a, &b).unwrap() - 48.130_803_608_679_1).abs() < 1e-9);
        assert!(psnr_u8(&a, &a).unwrap().is_infinite());
        assert!(psnr_u8(&a, &Rgb8Image::new(3, 3)).is_err());
    }
}
