//! Training loss 0.8·L1 + 0.2·(1 − SSIM) forward and backward, target
//! resampling, the VCD error map, and the eval metric kernels.
//! Owner: loss (E4). Kernels: `cuda/loss.cu`, `cuda/metrics.cu`.
//! Spec: spec-train 3 and 7.4.
//!
//! Buffer contract: reads the raster image (`RenderOut::rgb`, f32 planar
//! `[3][H][W]`) and the resampled target (same layout); writes dL/dimage into
//! `PixelGrads::d_rgb` (and d_alpha when a mask or alpha loss is on); adds
//! mean |x − y| and mean S (per iteration, already divided by N = 3HW) into a
//! 2-float device accumulator. Sizes come from `IterParams.frame.render_w/h`;
//! nothing syncs the host. The eval metrics (`metrics.cu`) are still stubs.

use anyhow::ensure;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use spl_core::Result;
use spl_gpu::cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
use ts_rs::TS;

use crate::frame::IterParamsDev;
use crate::gpu::{
    COMMON_CUH, CudaFile, CudaResultExt, Defines, Gpu, KernelSource, SPLAT_COMMON_CUH,
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
    /// [mean |x − y|, mean S] (f64 accumulator)
    pub sums: CudaSlice<f64>,
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
    /// The default weights (0.8 L1 + 0.2 D-SSIM).
    pub fn load(gpu: &Gpu) -> Result<Self> {
        Self::with_weights(gpu, 0.8, 0.2)
    }

    /// The loss weights are compile-time constants of the unit.
    pub fn with_weights(gpu: &Gpu, lambda_l1: f32, lambda_dssim: f32) -> Result<Self> {
        let d = Defines::new()
            .set("LAMBDA_L1", format!("{lambda_l1:e}f"))
            .set("LAMBDA_DSSIM", format!("{lambda_dssim:e}f"));
        let k = gpu.kernels(&SOURCE, &d)?;
        for name in ["ssim_fused", "error_map"] {
            let blocks = k
                .get(name)?
                .occupancy_max_active_blocks_per_multiprocessor(FUSED_TW * FUSED_TH, 0, None)
                .cuda("SSIM kernel occupancy")?;
            ensure!(
                blocks > 0,
                "{name} cannot run {} threads per block on this GPU",
                FUSED_TW * FUSED_TH
            );
        }
        Ok(Self {
            resample_target: k.get("resample_target")?,
            ssim_fwd: k.get("ssim_fwd")?,
            ssim_bwd: k.get("ssim_bwd")?,
            ssim_fused: k.get("ssim_fused")?,
            error_map: k.get("error_map")?,
        })
    }
}

/// v1 tile (16 × 16 threads) and v2 output tile (32 × 16 threads; NVRTC units
/// carry no `__launch_bounds__`, so `with_weights` checks the block fits).
const V1_TILE: u32 = 16;
const FUSED_TW: u32 = 32;
const FUSED_TH: u32 = 16;

fn grid(w: u32, h: u32, tw: u32, th: u32, z: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (w.div_ceil(tw), h.div_ceil(th), z),
        block_dim: (tw, th, 1),
        shared_mem_bytes: 0,
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
    /// [mean |x − y|, mean S] summed over iterations (f64), read by `take_sums`.
    pub sums: CudaSlice<f64>,
}

impl Loss {
    pub fn new(gpu: &Gpu, max_width: u32, max_height: u32, cfg: LossConfig) -> Result<Self> {
        ensure!(
            cfg.partials == Precision::F32,
            "image-to-3d: bf16 SSIM partials are not implemented (owner: {OWNER})"
        );
        let s = &gpu.stream;
        let px = (max_width as usize * max_height as usize).max(1);
        Ok(Self {
            kernels: LossKernels::with_weights(gpu, cfg.lambda_l1, cfg.lambda_dssim)?,
            partials: match cfg.kernel {
                SsimKernel::TwoPass => {
                    Some(s.alloc_zeros::<f32>(9 * px).cuda("alloc SSIM partials")?)
                }
                SsimKernel::Fused => None,
            },
            sums: s.alloc_zeros::<f64>(2).cuda("alloc loss sums")?,
            cfg,
            max_width,
            max_height,
        })
    }

    /// Forward + backward: writes dL/dimage (f32 planar), adds mean |x−y| and
    /// mean S to the accumulator. Image size from IterParams; no host sync.
    pub fn forward_backward(
        &mut self,
        gpu: &Gpu,
        params: &IterParamsDev,
        image: &CudaSlice<f32>,
        target: &CudaSlice<f32>,
        d_image: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let (w, h) = (self.max_width, self.max_height);
        let s = &gpu.stream;
        match (&self.cfg.kernel, &mut self.partials) {
            (SsimKernel::Fused, _) => {
                let mut l = s.launch_builder(&self.kernels.ssim_fused);
                l.arg(&params.params)
                    .arg(image)
                    .arg(target)
                    .arg(d_image)
                    .arg(&mut self.sums)
                    .arg(&params.status);
                // SAFETY: ssim_fused(const IterParams*, const float* x [3][H][W], const float* y,
                // float* d_x, double* sums [2], const u32* status); the buffers hold the
                // largest render (max_width × max_height), which the grid covers.
                unsafe { l.launch(grid(w, h, FUSED_TW, FUSED_TH, 3)) }.cuda("launch ssim_fused")?;
            }
            (SsimKernel::TwoPass, Some(partials)) => {
                let mut l = s.launch_builder(&self.kernels.ssim_fwd);
                l.arg(&params.params)
                    .arg(image)
                    .arg(target)
                    .arg(&mut *partials)
                    .arg(&mut self.sums)
                    .arg(&params.status);
                // SAFETY: ssim_fwd(params, x, y, float* partials [3][3][H][W], sums, status).
                unsafe { l.launch(grid(w, h, V1_TILE, V1_TILE, 3)) }.cuda("launch ssim_fwd")?;
                let mut l = s.launch_builder(&self.kernels.ssim_bwd);
                l.arg(&params.params)
                    .arg(image)
                    .arg(target)
                    .arg(&*partials)
                    .arg(d_image)
                    .arg(&params.status);
                // SAFETY: ssim_bwd(params, x, y, const float* partials, float* d_x, status).
                unsafe { l.launch(grid(w, h, V1_TILE, V1_TILE, 3)) }.cuda("launch ssim_bwd")?;
            }
            (SsimKernel::TwoPass, None) => unreachable!("two-pass loss without partial planes"),
        }
        Ok(())
    }

    /// Eval mode for VCD: error map, min/max, view loss (outside the graph: it
    /// resets `out.minmax` and `out.sums` from the host).
    pub fn error_map(
        &mut self,
        gpu: &Gpu,
        params: &IterParamsDev,
        image: &CudaSlice<f32>,
        target: &CudaSlice<f32>,
        out: &mut ErrorMap,
    ) -> Result<()> {
        let s = &gpu.stream;
        s.memcpy_htod(&[f32::MAX, 0.0], &mut out.minmax)
            .cuda("reset error-map min/max")?;
        s.memset_zeros(&mut out.sums).cuda("reset error-map sums")?;
        let mut l = s.launch_builder(&self.kernels.error_map);
        l.arg(&params.params)
            .arg(image)
            .arg(target)
            .arg(&mut out.err)
            .arg(&mut out.minmax)
            .arg(&mut out.sums);
        // SAFETY: error_map(const IterParams*, const float* x, const float* y,
        // float* err [H][W], float* minmax [2], double* sums [2]); sizes ≤ max render.
        unsafe { l.launch(grid(self.max_width, self.max_height, FUSED_TW, FUSED_TH, 3)) }
            .map(drop)
            .cuda("launch error_map")
    }

    /// Reads and resets the accumulator (host sync): the means over the last
    /// `iterations` iterations.
    pub fn take_sums(&mut self, gpu: &Gpu, iterations: u32) -> Result<LossValue> {
        let s = &gpu.stream;
        let v = s.clone_dtoh(&self.sums).cuda("download loss sums")?;
        s.memset_zeros(&mut self.sums).cuda("reset loss sums")?;
        let n = iterations.max(1) as f64;
        let (l1, ssim) = (v[0] / n, v[1] / n);
        Ok(LossValue {
            l1,
            ssim,
            loss: self.cfg.lambda_l1 as f64 * l1 + self.cfg.lambda_dssim as f64 * (1.0 - ssim),
        })
    }
}

impl ErrorMap {
    pub fn alloc(gpu: &Gpu, max_width: u32, max_height: u32) -> Result<Self> {
        let s = &gpu.stream;
        Ok(Self {
            err: s
                .alloc_zeros::<f32>((max_width as usize * max_height as usize).max(1))
                .cuda("alloc error map")?,
            minmax: s.alloc_zeros::<f32>(2).cuda("alloc error min/max")?,
            sums: s.alloc_zeros::<f64>(2).cuda("alloc error sums")?,
        })
    }
}

/// Resamples the cached u8 target of this iteration to the render size (copy at
/// scale 1, antialiased tent filter otherwise) into f32 planar. The grid
/// covers the largest cached image.
pub fn resample_target(
    gpu: &Gpu,
    k: &LossKernels,
    params: &IterParamsDev,
    cache: &ImageCache,
    out: &mut CudaSlice<f32>,
) -> Result<()> {
    let (w, h) = cache
        .entries
        .iter()
        .fold((1, 1), |(w, h), e| (w.max(e.width), h.max(e.height)));
    resample_target_raw(gpu, k, params, &cache.pool, w, h, out)
}

/// [`resample_target`] on a bare u8 pool, with the grid sized for `max_w × max_h` renders.
pub fn resample_target_raw(
    gpu: &Gpu,
    k: &LossKernels,
    params: &IterParamsDev,
    pool: &CudaSlice<u8>,
    max_w: u32,
    max_h: u32,
    out: &mut CudaSlice<f32>,
) -> Result<()> {
    let mut l = gpu.stream.launch_builder(&k.resample_target);
    l.arg(&params.params).arg(pool).arg(out).arg(&params.status);
    // SAFETY: resample_target(const IterParams*, const u8* pool, float* target [3][H][W],
    // const u32* status); the target holds max_w × max_h pixels, the render size never
    // exceeds the cached image size, and IterParams.target_offset addresses the pool.
    unsafe { l.launch(grid(max_w, max_h, 32, 8, 1)) }
        .map(drop)
        .cuda("launch resample_target")
}

/// f64 CPU reference of the loss (spec-train 3.1): returns (L, mean |x−y|,
/// mean S, dL/dx), images planar `[3][h][w]`.
pub fn reference_loss(
    x: &[f32],
    y: &[f32],
    w: usize,
    h: usize,
    lambda_l1: f64,
    lambda_dssim: f64,
) -> (f64, f64, f64, Vec<f64>) {
    let hw = w * h;
    let n = (3 * hw) as f64;
    let g: Vec<f64> = SSIM_WINDOW.iter().map(|&v| v as f64).collect();
    // Separable "same" convolution with zero padding.
    let conv = |img: &[f64]| -> Vec<f64> {
        let mut tmp = vec![0.0; hw];
        for r in 0..h {
            for c in 0..w {
                let mut s = 0.0;
                for (d, gd) in g.iter().enumerate() {
                    let cc = c as isize + d as isize - 5;
                    if (0..w as isize).contains(&cc) {
                        s += gd * img[r * w + cc as usize];
                    }
                }
                tmp[r * w + c] = s;
            }
        }
        let mut out = vec![0.0; hw];
        for r in 0..h {
            for c in 0..w {
                let mut s = 0.0;
                for (d, gd) in g.iter().enumerate() {
                    let rr = r as isize + d as isize - 5;
                    if (0..h as isize).contains(&rr) {
                        s += gd * tmp[rr as usize * w + c];
                    }
                }
                out[r * w + c] = s;
            }
        }
        out
    };
    let (c1, c2) = (SSIM_C1 as f64, SSIM_C2 as f64);
    let k = -lambda_dssim / n;
    let (mut l1, mut ssum) = (0.0, 0.0);
    let mut grad = vec![0.0; 3 * hw];
    for ch in 0..3 {
        let xs: Vec<f64> = x[ch * hw..(ch + 1) * hw]
            .iter()
            .map(|&v| v as f64)
            .collect();
        let ys: Vec<f64> = y[ch * hw..(ch + 1) * hw]
            .iter()
            .map(|&v| v as f64)
            .collect();
        let sq = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(p, q)| p * q).collect::<Vec<_>>();
        let (mx, my) = (conv(&xs), conv(&ys));
        let (mxx, myy, mxy) = (
            conv(&sq(&xs, &xs)),
            conv(&sq(&ys, &ys)),
            conv(&sq(&xs, &ys)),
        );
        let (mut pa, mut pb, mut pd) = (vec![0.0; hw], vec![0.0; hw], vec![0.0; hw]);
        for i in 0..hw {
            let (ux, uy) = (mx[i], my[i]);
            let sx = (mxx[i] - ux * ux).max(0.0);
            let sy = (myy[i] - uy * uy).max(0.0);
            let sxy = mxy[i] - ux * uy;
            let a = ux * ux + uy * uy + c1;
            let b = sx + sy + c2;
            let cn = 2.0 * ux * uy + c1;
            let d = 2.0 * sxy + c2;
            let s = cn * d / (a * b);
            if !(-1.0..=1.0).contains(&s) {
                ssum += s.clamp(-1.0, 1.0);
                continue;
            }
            ssum += s;
            let dm1 =
                2.0 * uy * d / (a * b) - 2.0 * uy * cn / (a * b) - 2.0 * ux * cn * d / (a * a * b)
                    + 2.0 * ux * cn * d / (a * b * b);
            pa[i] = k * dm1;
            pb[i] = k * (-cn * d / (a * b * b));
            pd[i] = k * (2.0 * cn / (a * b));
        }
        let (ca, cb, cd) = (conv(&pa), conv(&pb), conv(&pd));
        for i in 0..hw {
            let diff = xs[i] - ys[i];
            l1 += diff.abs();
            let sign = if diff > 0.0 {
                1.0
            } else if diff < 0.0 {
                -1.0
            } else {
                0.0
            };
            grad[ch * hw + i] = ca[i] + 2.0 * xs[i] * cb[i] + ys[i] * cd[i] + lambda_l1 / n * sign;
        }
    }
    let (l1, ssim) = (l1 / n, ssum / n);
    (lambda_l1 * l1 + lambda_dssim * (1.0 - ssim), l1, ssim, grad)
}

#[cfg(test)]
mod tests;
