//! VGGT-Ω camera and dense heads on CUDA (owner: heads).
//!
//! Spec: `reports/image-to-3d/spec-vggt.md` §4 (camera head), §5 (dense
//! head), §9. The trunk leaves the four cached layers `L_i = [F_i | G_i]`
//! (bf16 `[N·T][2048]`, 17 special tokens first in each frame); the heads
//! read them and produce:
//!
//! - [`camera::CameraHead`]: `pose_enc` `[N][9]` plus extrinsics (w2c,
//!   OpenCV) and intrinsics on the device ([`DeviceCams`]);
//! - [`dense::DenseHead`]: depth and confidence as f16 `[n][H][W]` per chunk
//!   of frames (8 in the engine), and optionally world points (f32
//!   `[n][H][W][3]`) unprojected with the [`DeviceCams`] of the chunk.
//!
//! Precision: bf16 GEMM/convolution inputs with fp32 accumulation, fp32
//! LayerNorm statistics and residual stream in the camera trunk, fp32 for
//! the final 256 → 32 projection, exp and unprojection.
//!
//! Kernels: cuBLASLt GEMMs, LayerNorm and residual from `spl_nn`; the
//! convolutions, depth-to-space, bilinear resize, table add, exact GELU and
//! head-dim-128 attention from `spl_nn::conv` / `spl_nn::rowops`; the
//! model-specific tails in `cuda/vggt_heads.cu`.

pub mod camera;
pub mod dense;
pub mod pos_embed;
#[cfg(test)]
mod tests;

use std::sync::Arc;

use spl_core::{Context, Result, bail};
use spl_nn::conv::Conv;
use spl_nn::cudarc::driver::{CudaFunction, CudaSlice};
use spl_nn::gemm::{Gemm, Tuning};
use spl_nn::norm::Norm;
use spl_nn::residual::Residual;
use spl_nn::rowops::RowOps;
use spl_nn::{COMMON_CUH, Nn, host};

/// Special tokens per frame (camera + 16 registers).
pub const SPECIAL: usize = 17;
/// Width of a cached layer, `[F_i | G_i]`.
pub const DIM: usize = 2048;

const SOURCE: &str = include_str!("../../../cuda/vggt_heads.cu");

/// Host access to checkpoint tensors (implemented by the runtime's
/// `Weights`). Names are the checkpoint's; `camera_head.trunk.*.attn.qkv.bias`
/// must be the effective `[b_q, 0, b_v]` (the heads zero the K third anyway).
pub trait HeadSource {
    fn host_f32(&self, name: &str) -> Result<Vec<f32>>;
    fn shape(&self, name: &str) -> Result<Vec<usize>>;
}

/// Frame count and preprocessed resolution (shared with the runtime).
pub use super::workspace::Geometry;

/// Cameras on the device, written by [`camera::CameraHead::run`] and read by
/// the dense head's unprojection.
pub struct DeviceCams {
    pub frames: usize,
    /// f32 `[frames][9]`, the head output (fov already `relu + 0.01`).
    pub pose_enc: CudaSlice<f32>,
    /// f32 `[frames][12]`: `[R | T]` row-major, world-to-camera, OpenCV axes.
    pub extrinsics: CudaSlice<f32>,
    /// f32 `[frames][9]`: K row-major in preprocessed pixels.
    pub intrinsics: CudaSlice<f32>,
}

impl DeviceCams {
    pub fn new(nn: &Nn, frames: usize) -> Result<Self> {
        let a = |n: usize| {
            nn.stream
                .alloc_zeros::<f32>(n.max(1))
                .context("alloc cameras")
        };
        Ok(Self {
            frames,
            pose_enc: a(frames * 9)?,
            extrinsics: a(frames * 12)?,
            intrinsics: a(frames * 9)?,
        })
    }
}

/// `utils/pose_enc.py::encoding_to_camera` on the host: w2c `[R | T]`
/// row-major 3×4 and K row-major 3×3 for an image of `height × width`.
pub fn encoding_to_camera(pe: &[f32; 9], height: usize, width: usize) -> ([f32; 12], [f32; 9]) {
    let (x, y, z, w) = (pe[3], pe[4], pe[5], pe[6]);
    let s = 2.0 / (x * x + y * y + z * z + w * w);
    let e = [
        1.0 - s * (y * y + z * z),
        s * (x * y - z * w),
        s * (x * z + y * w),
        pe[0],
        s * (x * y + z * w),
        1.0 - s * (x * x + z * z),
        s * (y * z - x * w),
        pe[1],
        s * (x * z - y * w),
        s * (y * z + x * w),
        1.0 - s * (x * x + y * y),
        pe[2],
    ];
    let fy = 0.5 * height as f32 / (0.5 * pe[7]).tan();
    let fx = 0.5 * width as f32 / (0.5 * pe[8]).tan();
    let k = [
        fx,
        0.0,
        0.5 * width as f32,
        0.0,
        fy,
        0.5 * height as f32,
        0.0,
        0.0,
        1.0,
    ];
    (e, k)
}

/// The `spl_nn` kernels and the app-local ones, shared by both heads.
pub struct Kernels {
    pub gemm: Gemm,
    pub norm: Norm,
    pub residual: Residual,
    pub rowops: RowOps,
    pub conv: Conv,
    pub(crate) widen: CudaFunction,
    pub(crate) camera_finish: CudaFunction,
    pub(crate) dense_output: CudaFunction,
    /// Debug snapshots of intermediates (off unless enabled; tests only).
    pub taps: Taps,
}

/// Records named intermediates when enabled (a sync and a download each):
/// the layer-by-layer parity tests read them. Disabled, a call costs a
/// relaxed atomic load.
#[derive(Default)]
pub struct Taps {
    on: std::sync::atomic::AtomicBool,
    store: std::sync::Mutex<Vec<(String, Vec<f32>)>>,
}

impl Taps {
    pub fn enable(&self, on: bool) {
        self.on.store(on, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn take(&self) -> Vec<(String, Vec<f32>)> {
        self.store
            .lock()
            .map(|mut s| std::mem::take(&mut *s))
            .unwrap_or_default()
    }

    fn enabled(&self) -> bool {
        self.on.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub(crate) fn bf16(&self, nn: &Nn, name: &str, buf: &CudaSlice<u16>, len: usize) -> Result<()> {
        if self.enabled() {
            let v = nn
                .stream
                .clone_dtoh(&buf.slice(..len))
                .context("tap download")?;
            self.push(name, host::from_bf16(&v));
        }
        Ok(())
    }

    pub(crate) fn f32(&self, nn: &Nn, name: &str, buf: &CudaSlice<f32>, len: usize) -> Result<()> {
        if self.enabled() {
            let v = nn
                .stream
                .clone_dtoh(&buf.slice(..len))
                .context("tap download")?;
            self.push(name, v);
        }
        Ok(())
    }

    fn push(&self, name: &str, v: Vec<f32>) {
        if let Ok(mut s) = self.store.lock() {
            s.push((name.to_string(), v));
        }
    }
}

impl Kernels {
    pub fn new(nn: &Nn) -> Result<Arc<Self>> {
        let m = nn.module("vggt_heads", &[COMMON_CUH, SOURCE].concat())?;
        let f = |n: &str| {
            m.load_function(n)
                .map_err(|e| spl_core::anyhow!("load kernel {n}: {e:?}"))
        };
        Ok(Arc::new(Self {
            gemm: Gemm::new(nn, Tuning::Heuristic)?,
            norm: Norm::new(nn)?,
            residual: Residual::new(nn)?,
            rowops: RowOps::new(nn)?,
            conv: Conv::new(nn)?,
            widen: f("widen_bf16")?,
            camera_finish: f("camera_finish")?,
            dense_output: f("dense_output")?,
            taps: Taps::default(),
        }))
    }
}

/// Reads a tensor and checks its shape.
pub(crate) fn tensor(src: &dyn HeadSource, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let got = src.shape(name)?;
    if got != shape {
        bail!("{name}: shape {got:?}, expected {shape:?}");
    }
    let v = src.host_f32(name)?;
    if v.len() != shape.iter().product::<usize>() {
        bail!("{name}: {} values for shape {shape:?}", v.len());
    }
    Ok(v)
}

pub(crate) fn up_f32(nn: &Nn, v: &[f32]) -> Result<CudaSlice<f32>> {
    nn.stream.clone_htod(v).context("upload f32 weights")
}

pub(crate) fn up_bf16(nn: &Nn, v: &[f32]) -> Result<CudaSlice<u16>> {
    nn.stream
        .clone_htod(&host::to_bf16(v))
        .context("upload bf16 weights")
}

pub(crate) fn alloc<
    T: spl_nn::cudarc::driver::DeviceRepr + spl_nn::cudarc::driver::ValidAsZeroBits,
>(
    nn: &Nn,
    n: usize,
) -> Result<CudaSlice<T>> {
    nn.stream
        .alloc_zeros::<T>(n.max(1))
        .context("alloc head workspace")
}

/// OIHW conv weight → `[O][kh][kw][I]` (what `spl_nn::conv` reads).
pub(crate) fn ohwi(w: &[f32], o: usize, i: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0f32; w.len()];
    for oo in 0..o {
        for ii in 0..i {
            for t in 0..k * k {
                out[(oo * k * k + t) * i + ii] = w[(oo * i + ii) * k * k + t];
            }
        }
    }
    out
}

/// ConvTranspose2d weight `[I][O][s][s]` (kernel = stride) → GEMM weight
/// rows `(ky, kx, o)` over `I`, i.e. `[s·s·O][I]` for a tap-major
/// depth-to-space.
pub(crate) fn deconv_rows(w: &[f32], i: usize, o: usize, s: usize) -> Vec<f32> {
    let mut out = vec![0f32; w.len()];
    for ii in 0..i {
        for oo in 0..o {
            for t in 0..s * s {
                out[(t * o + oo) * i + ii] = w[(ii * o + oo) * s * s + t];
            }
        }
    }
    out
}
