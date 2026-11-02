//! Evaluation protocol: test renders, PSNR, SSIM (same and valid), LPIPS
//! later. Owner: loss (E4). Kernels: `cuda/metrics.cu`. Spec: spec-train 7.
//!
//! Renders are clamped and quantized like `torchvision.save_image`
//! (`u8 = floor(x·255 + 0.5)`), compared with the u8 ground truth. Rust PSNR
//! and SSIM must equal the parity scripts to 1e-4 dB / 1e-5 (a test).
//!
//! Metrics run on the CPU (f64); `metrics.cu` is still a stub. LPIPS comes
//! from the benchmark harness on the saved PNGs.

use std::path::Path;

use anyhow::{Context, ensure};
use rayon::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use spl_core::Result;
use ts_rs::TS;

use crate::io::images::{Rgb8Image, load_view_image};
use crate::trainer::Trainer;

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

/// PSNR and both SSIMs on the CPU (f64; the GPU `metrics.cu` kernels are
/// still stubs). LPIPS comes from the harness (`gs-metrics`) on saved PNGs.
pub fn image_metrics(render: &Rgb8Image, gt: &Rgb8Image) -> Result<ImageMetrics> {
    let psnr = psnr_u8(render, gt)?;
    let (ssim, ssim_valid) = ssim_u8(render, gt);
    Ok(ImageMetrics {
        psnr,
        ssim,
        ssim_valid,
        lpips_vgg: None,
        lpips_vgg_inria: None,
    })
}

/// SSIM of two same-size u8 images in [0, 1]: 11×11 Gaussian window σ = 1.5,
/// zero "same" padding, C1 = 0.01², C2 = 0.03² (Inria `ssim`). Returns the
/// mean over every pixel and channel, and over pixels ≥ 5 from every border.
pub fn ssim_u8(a: &Rgb8Image, b: &Rgb8Image) -> (f64, f64) {
    const R: usize = 5;
    let (w, h) = (a.width as usize, a.height as usize);
    let mut win = [0f64; 2 * R + 1];
    for (i, v) in win.iter_mut().enumerate() {
        let d = i as f64 - R as f64;
        *v = (-d * d / (2.0 * 1.5 * 1.5)).exp();
    }
    let sum: f64 = win.iter().sum();
    win.iter_mut().for_each(|v| *v /= sum);
    // Separable blur with zero padding.
    let blur = |x: &[f64]| -> Vec<f64> {
        let mut tmp = vec![0f64; w * h];
        for y in 0..h {
            for xx in 0..w {
                let mut acc = 0.0;
                for (k, wk) in win.iter().enumerate() {
                    let sx = xx as isize + k as isize - R as isize;
                    if sx >= 0 && (sx as usize) < w {
                        acc += wk * x[y * w + sx as usize];
                    }
                }
                tmp[y * w + xx] = acc;
            }
        }
        let mut out = vec![0f64; w * h];
        for y in 0..h {
            for xx in 0..w {
                let mut acc = 0.0;
                for (k, wk) in win.iter().enumerate() {
                    let sy = y as isize + k as isize - R as isize;
                    if sy >= 0 && (sy as usize) < h {
                        acc += wk * tmp[sy as usize * w + xx];
                    }
                }
                out[y * w + xx] = acc;
            }
        }
        out
    };
    let (c1, c2) = (0.01f64 * 0.01, 0.03f64 * 0.03);
    let per_channel: Vec<(f64, f64, usize)> = (0..3)
        .into_par_iter()
        .map(|c| {
            let chan = |img: &Rgb8Image| -> Vec<f64> {
                (0..w * h)
                    .map(|i| img.data[i * 3 + c] as f64 / 255.0)
                    .collect()
            };
            let (x, y) = (chan(a), chan(b));
            let xx: Vec<f64> = x.iter().map(|v| v * v).collect();
            let yy: Vec<f64> = y.iter().map(|v| v * v).collect();
            let xy: Vec<f64> = x.iter().zip(&y).map(|(p, q)| p * q).collect();
            let (mx, my) = (blur(&x), blur(&y));
            let (sxx, syy, sxy) = (blur(&xx), blur(&yy), blur(&xy));
            let (mut all, mut valid, mut n_valid) = (0f64, 0f64, 0usize);
            for py in 0..h {
                for px in 0..w {
                    let i = py * w + px;
                    let (m1, m2) = (mx[i], my[i]);
                    let v1 = sxx[i] - m1 * m1;
                    let v2 = syy[i] - m2 * m2;
                    let cv = sxy[i] - m1 * m2;
                    let s = ((2.0 * m1 * m2 + c1) * (2.0 * cv + c2))
                        / ((m1 * m1 + m2 * m2 + c1) * (v1 + v2 + c2));
                    all += s;
                    if py >= R && px >= R && py + R < h && px + R < w {
                        valid += s;
                        n_valid += 1;
                    }
                }
            }
            (all, valid, n_valid)
        })
        .collect();
    let all: f64 = per_channel.iter().map(|c| c.0).sum();
    let valid: f64 = per_channel.iter().map(|c| c.1).sum();
    let n_valid: usize = per_channel.iter().map(|c| c.2).sum();
    (all / (3 * w * h) as f64, valid / n_valid.max(1) as f64)
}

/// Renders every test view, saves `<name>.png` under `out_dir` when given
/// (the harness scores those with LPIPS), and averages the metrics.
pub fn evaluate(trainer: &mut Trainer, out_dir: Option<&Path>) -> Result<EvalReport> {
    if let Some(d) = out_dir {
        std::fs::create_dir_all(d).with_context(|| format!("create {}", d.display()))?;
    }
    let test = trainer.scene.test.clone();
    let mut per_view = Vec::with_capacity(test.len());
    for v in test {
        let render = trainer.render(v)?;
        let view = &trainer.scene.views[v];
        let gt = load_view_image(view)?;
        let m = image_metrics(&render, &gt)?;
        let stem = Path::new(&view.name)
            .file_stem()
            .map_or_else(|| view.name.clone(), |s| s.to_string_lossy().into_owned());
        if let Some(d) = out_dir {
            render.save_png(&d.join(format!("{stem}.png")))?;
        }
        per_view.push((view.name.clone(), m));
    }
    let n = per_view.len().max(1) as f64;
    let mean = ImageMetrics {
        psnr: per_view.iter().map(|(_, m)| m.psnr).sum::<f64>() / n,
        ssim: per_view.iter().map(|(_, m)| m.ssim).sum::<f64>() / n,
        ssim_valid: per_view.iter().map(|(_, m)| m.ssim_valid).sum::<f64>() / n,
        lpips_vgg: None,
        lpips_vgg_inria: None,
    };
    Ok(EvalReport { mean, per_view })
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

    #[test]
    fn ssim_is_one_for_equal_images_and_drops_with_noise() {
        let mut a = Rgb8Image::new(24, 20);
        for (i, v) in a.data.iter_mut().enumerate() {
            *v = ((i * 37) % 251) as u8;
        }
        let (s, sv) = ssim_u8(&a, &a);
        assert!((s - 1.0).abs() < 1e-12 && (sv - 1.0).abs() < 1e-12);
        let mut b = a.clone();
        b.data
            .iter_mut()
            .step_by(7)
            .for_each(|v| *v = v.wrapping_add(40));
        let (s, sv) = ssim_u8(&a, &b);
        assert!(s < 0.99 && sv < 0.99 && s > 0.0);
    }
}
