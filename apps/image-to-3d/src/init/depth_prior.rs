//! Scale-and-shift-invariant depth prior (Report section 3, Stage C; spec-train
//! 10.3): a rendered depth `d` against a prior `D` (VGGT-Ω depth, any scale)
//! with per-pixel weights `w`:
//!
//! (s, t) = argmin Σ w (s·d + t − D)², L = Σ w (s·d + t − D)² / Σ w,
//! ∂L/∂dᵢ = 2 wᵢ s (s·dᵢ + t − Dᵢ) / Σ w (the s, t terms vanish at the optimum).
//!
//! The loss is added with weight λ(step) from [`DepthPriorSchedule`] (0.1 → 0)
//! into the rasterizer's depth-channel pixel gradient (`PixelGrads::d_extra`
//! row of the expected-depth channel), which `ssi_backward` accumulates into.
//! Owner: init-b. Kernels: `ssi_partial`, `ssi_solve`, `ssi_backward` in
//! `cuda/init.cu`.

use anyhow::ensure;
use spl_core::Result;
use spl_gpu::cudarc::driver::{CudaSlice, CudaView, CudaViewMut, LaunchConfig, PushKernelArg};

use super::InitKernels;
use crate::gpu::{CudaResultExt, Gpu};

const BLOCK: u32 = 256;
const SUMS: usize = 6;

/// λ(step): linear from `init` at step 0 to `final_` at `until`, then `final_`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthPriorSchedule {
    pub init: f32,
    pub final_: f32,
    pub until: u32,
}

impl DepthPriorSchedule {
    /// The design-doc recipe: 0.1 decayed to 0 over `max_steps`.
    pub fn report(max_steps: u32) -> Self {
        Self {
            init: 0.1,
            final_: 0.0,
            until: max_steps,
        }
    }

    pub fn weight(&self, step: u32) -> f32 {
        if self.until == 0 || step >= self.until {
            return self.final_;
        }
        let a = step as f32 / self.until as f32;
        self.init + (self.final_ - self.init) * a
    }
}

/// One image's rendered depth, prior and weights (same length).
pub struct SsiInputs<'a> {
    pub depth: CudaView<'a, f32>,
    pub prior: CudaView<'a, f32>,
    pub weight: CudaView<'a, f32>,
}

/// Fit and loss of one evaluation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SsiFit {
    pub scale: f32,
    pub shift: f32,
    /// L without λ.
    pub loss: f32,
    pub weight_sum: f32,
}

/// Scratch for the depth prior of images up to `max_pixels`.
pub struct DepthPrior {
    blocks: u32,
    partial: CudaSlice<f64>,
    /// (s, t, L, W) of the last [`DepthPrior::forward`].
    pub fit: CudaSlice<f32>,
}

impl DepthPrior {
    pub fn alloc(gpu: &Gpu) -> Result<Self> {
        let blocks = gpu.multiprocessors() * 4;
        let s = &gpu.stream;
        Ok(Self {
            blocks,
            partial: s
                .alloc_zeros::<f64>(blocks as usize * SUMS)
                .cuda("alloc ssi partials")?,
            fit: s.alloc_zeros::<f32>(4).cuda("alloc ssi fit")?,
        })
    }

    /// Fits (s, t) and the loss into `self.fit` (device; no sync).
    pub fn forward(&mut self, gpu: &Gpu, k: &InitKernels, x: &SsiInputs) -> Result<()> {
        let (depth, prior, weight) = (&x.depth, &x.prior, &x.weight);
        let n = depth.len();
        ensure!(
            prior.len() == n && weight.len() == n,
            "depth prior: size mismatch"
        );
        let n = n as u32;
        let s = &gpu.stream;
        let f = k.get("ssi_partial")?;
        let mut l = s.launch_builder(&f);
        l.arg(depth)
            .arg(prior)
            .arg(weight)
            .arg(&n)
            .arg(&mut self.partial);
        let cfg = LaunchConfig {
            grid_dim: (self.blocks, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: ssi_partial(d, D, w, n, partial[blocks][6]); grid = blocks.
        unsafe { l.launch(cfg) }.cuda("ssi_partial")?;
        let f = k.get("ssi_solve")?;
        let mut l = s.launch_builder(&f);
        l.arg(&self.partial).arg(&self.blocks).arg(&mut self.fit);
        let one = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: ssi_solve(partial[blocks][6], blocks, out[4]); one block.
        unsafe { l.launch(one) }.cuda("ssi_solve")?;
        Ok(())
    }

    /// `grad += λ ∂L/∂d` using the fit of the last forward.
    pub fn backward(
        &self,
        gpu: &Gpu,
        k: &InitKernels,
        x: &SsiInputs,
        lambda: f32,
        grad: &mut CudaViewMut<f32>,
    ) -> Result<()> {
        let (depth, prior, weight) = (&x.depth, &x.prior, &x.weight);
        let n = depth.len();
        ensure!(
            prior.len() == n && weight.len() == n && grad.len() == n,
            "depth prior: size mismatch"
        );
        if lambda == 0.0 {
            return Ok(());
        }
        let n = n as u32;
        let f = k.get("ssi_backward")?;
        let mut l = gpu.stream.launch_builder(&f);
        l.arg(depth)
            .arg(prior)
            .arg(weight)
            .arg(&n)
            .arg(&self.fit)
            .arg(&lambda)
            .arg(grad);
        // SAFETY: ssi_backward(d, D, w, n, fit[4], lambda, grad[n]); one thread per pixel.
        unsafe { l.launch(LaunchConfig::for_num_elems(n)) }.cuda("ssi_backward")?;
        Ok(())
    }

    /// Downloads the last fit (synchronizes).
    pub fn read_fit(&self, gpu: &Gpu) -> Result<SsiFit> {
        let v = gpu.stream.clone_dtoh(&self.fit).cuda("download ssi fit")?;
        Ok(SsiFit {
            scale: v[0],
            shift: v[1],
            loss: v[2],
            weight_sum: v[3],
        })
    }
}

/// CPU reference: (fit, ∂L/∂d without λ).
pub fn ssi_cpu(d: &[f32], prior: &[f32], w: &[f32]) -> (SsiFit, Vec<f32>) {
    let mut t = [0f64; SUMS];
    for ((&di, &pi), &wi) in d.iter().zip(prior).zip(w) {
        if wi.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) || pi.is_nan() || di.is_nan() {
            continue;
        }
        let (wi, di, pi) = (wi as f64, di as f64, pi as f64);
        t[0] += wi;
        t[1] += wi * di;
        t[2] += wi * di * di;
        t[3] += wi * pi;
        t[4] += wi * di * pi;
        t[5] += wi * pi * pi;
    }
    let [ws, sd, sdd, sp, sdp, spp] = t;
    let (mut s, mut sh, mut l) = (0.0, 0.0, 0.0);
    if ws > 0.0 {
        let var = sdd * ws - sd * sd;
        if var > 1e-12 * sdd * ws {
            s = (sdp * ws - sd * sp) / var;
            sh = (sp - s * sd) / ws;
        } else {
            sh = sp / ws;
        }
        l = ((s * s * sdd + 2.0 * s * sh * sd + sh * sh * ws - 2.0 * s * sdp - 2.0 * sh * sp
            + spp)
            / ws)
            .max(0.0);
    }
    let g = d
        .iter()
        .zip(prior)
        .zip(w)
        .map(|((&di, &pi), &wi)| {
            if ws > 0.0 && wi > 0.0 && !pi.is_nan() && !di.is_nan() {
                (2.0 * wi as f64 * s * (s * di as f64 + sh - pi as f64) / ws) as f32
            } else {
                0.0
            }
        })
        .collect();
    (
        SsiFit {
            scale: s as f32,
            shift: sh as f32,
            loss: l as f32,
            weight_sum: ws as f32,
        },
        g,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let d: Vec<f32> = (0..n)
            .map(|i| 1.0 + ((i * 37) % 101) as f32 * 0.03)
            .collect();
        let p: Vec<f32> = d
            .iter()
            .enumerate()
            .map(|(i, &x)| 2.5 * x - 0.7 + ((i * 13) % 7) as f32 * 0.02)
            .collect();
        let w: Vec<f32> = (0..n)
            .map(|i| {
                if i % 11 == 0 {
                    0.0
                } else {
                    0.5 + (i % 3) as f32
                }
            })
            .collect();
        (d, p, w)
    }

    #[test]
    fn exact_affine_prior_has_zero_loss() {
        let d: Vec<f32> = (0..50).map(|i| 1.0 + i as f32 * 0.1).collect();
        let p: Vec<f32> = d.iter().map(|x| 3.0 * x + 0.5).collect();
        let (fit, g) = ssi_cpu(&d, &p, &[1.0; 50]);
        assert!(
            (fit.scale - 3.0).abs() < 1e-4 && (fit.shift - 0.5).abs() < 1e-3,
            "{fit:?}"
        );
        assert!(fit.loss < 1e-8 && g.iter().all(|x| x.abs() < 1e-4));
    }

    #[test]
    fn gradient_matches_finite_differences() {
        let (d, p, w) = sample(64);
        let (_, g) = ssi_cpu(&d, &p, &w);
        for i in [1usize, 5, 17, 40] {
            let h = 1e-3;
            let mut dp = d.clone();
            dp[i] += h;
            let mut dm = d.clone();
            dm[i] -= h;
            let fd = (ssi_cpu(&dp, &p, &w).0.loss - ssi_cpu(&dm, &p, &w).0.loss) / (2.0 * h);
            assert!(
                (fd - g[i]).abs() < 2e-3 * (1.0 + g[i].abs()),
                "{i}: fd {fd} vs {}",
                g[i]
            );
        }
    }

    #[test]
    fn schedule_decays_to_zero() {
        let s = DepthPriorSchedule::report(1000);
        assert_eq!(s.weight(0), 0.1);
        assert!((s.weight(500) - 0.05).abs() < 1e-7);
        assert_eq!(s.weight(1000), 0.0);
        assert_eq!(s.weight(5000), 0.0);
    }
}
