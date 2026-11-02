//! GPU half of Stage A0: upload a decoded frame, score its sharpness, resize
//! it to the engine and trainer sizes. Owner: ingest (A0). Kernels: `cuda/ingest.cu`.
//!
//! One frame at a time on the handle's stream; device buffers grow to the
//! largest frame seen and are reused. Every call ends with a download, so it
//! syncs the stream.

use std::time::Instant;

use spl_core::Result;
use spl_gpu::cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};

use super::Rgb8;
use crate::gpu::{COMMON_CUH, CudaFile, CudaResultExt, Defines, Gpu, KernelSource};

pub const SOURCE: KernelSource = KernelSource {
    module: "ingest",
    files: &[
        COMMON_CUH,
        CudaFile {
            name: "ingest.cu",
            text: include_str!("../../cuda/ingest.cu"),
        },
    ],
    kernels: &["sharpness_partials", "resize_area"],
};

/// Mirrors `SHARP_BLOCK_X/Y` in `cuda/ingest.cu`.
const SHARP_BLOCK: (u32, u32) = (32, 8);
const RESIZE_BLOCK: (u32, u32) = (16, 16);

pub struct IngestGpu {
    pub gpu: Gpu,
    sharpness: CudaFunction,
    resize: CudaFunction,
    src: Option<CudaSlice<u8>>,
    /// The frame [`IngestGpu::keep`] set aside.
    kept: Option<CudaSlice<u8>>,
    dst: Option<CudaSlice<u8>>,
    partials: Option<CudaSlice<f32>>,
    /// Wall seconds spent inside this handle's calls.
    pub seconds: f64,
}

/// A frame resident on the device (the last one [`IngestGpu::upload`] took).
#[derive(Debug, Clone, Copy)]
pub struct Resident {
    pub width: u32,
    pub height: u32,
}

impl IngestGpu {
    pub fn new(gpu: Gpu) -> Result<Self> {
        let k = gpu.kernels(&SOURCE, &Defines::new())?;
        Ok(Self {
            sharpness: k.get("sharpness_partials")?,
            resize: k.get("resize_area")?,
            gpu,
            src: None,
            kept: None,
            dst: None,
            partials: None,
            seconds: 0.0,
        })
    }

    pub fn open() -> Result<Self> {
        Self::new(Gpu::open()?)
    }

    fn ensure_u8(gpu: &Gpu, slot: &mut Option<CudaSlice<u8>>, len: usize) -> Result<()> {
        if slot.as_ref().is_none_or(|s| s.len() < len) {
            *slot = None;
            // SAFETY: every byte read later is written first (upload or kernel output).
            *slot = Some(unsafe { gpu.stream.alloc::<u8>(len) }.cuda("alloc ingest frame")?);
        }
        Ok(())
    }

    /// Uploads packed RGB `data` of `width`×`height`.
    pub fn upload(&mut self, width: u32, height: u32, data: &[u8]) -> Result<Resident> {
        let t = Instant::now();
        let len = width as usize * height as usize * 3;
        if data.len() < len {
            spl_core::bail!("frame buffer is {} bytes, need {len}", data.len());
        }
        Self::ensure_u8(&self.gpu, &mut self.src, len)?;
        let src = self.src.as_mut().expect("allocated above");
        self.gpu
            .stream
            .memcpy_htod(&data[..len], &mut src.slice_mut(..len))
            .cuda("upload frame")?;
        self.seconds += t.elapsed().as_secs_f64();
        Ok(Resident { width, height })
    }

    /// Laplacian variance of the resident frame (see `sharpness_cpu`).
    pub fn sharpness(&mut self, r: Resident) -> Result<f64> {
        let t = Instant::now();
        let (gx, gy) = (
            r.width.div_ceil(SHARP_BLOCK.0),
            r.height.div_ceil(SHARP_BLOCK.1),
        );
        let blocks = (gx * gy) as usize;
        if self.partials.as_ref().is_none_or(|p| p.len() < 2 * blocks) {
            self.partials = Some(
                self.gpu
                    .stream
                    .alloc_zeros::<f32>(2 * blocks)
                    .cuda("alloc sharpness partials")?,
            );
        }
        let partials = self.partials.as_mut().expect("allocated above");
        let src = self.src.as_ref().expect("upload first");
        let (w, h) = (r.width as i32, r.height as i32);
        let cfg = LaunchConfig {
            grid_dim: (gx, gy, 1),
            block_dim: (SHARP_BLOCK.0, SHARP_BLOCK.1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: src holds w·h·3 bytes, partials 2·gx·gy floats; matches the kernel signature.
        unsafe {
            self.gpu
                .stream
                .launch_builder(&self.sharpness)
                .arg(src)
                .arg(&w)
                .arg(&h)
                .arg(&mut *partials)
                .launch(cfg)
        }
        .cuda("launch sharpness_partials")?;
        let mut host = vec![0f32; 2 * blocks];
        self.gpu
            .stream
            .memcpy_dtoh(&partials.slice(..2 * blocks), &mut host)
            .cuda("download sharpness partials")?;
        self.seconds += t.elapsed().as_secs_f64();
        let n = (r.width.saturating_sub(2) as f64) * (r.height.saturating_sub(2) as f64);
        if n <= 0.0 {
            return Ok(0.0);
        }
        let (s, s2) = host
            .as_chunks::<2>()
            .0
            .iter()
            .fold((0f64, 0f64), |(a, b), p| (a + p[0] as f64, b + p[1] as f64));
        Ok((s2 / n - (s / n).powi(2)).max(0.0))
    }

    /// Sets the resident frame aside (a buffer swap, no copy); the next
    /// upload goes to the other buffer. Resize it later with [`Self::resize_kept`].
    pub fn keep(&mut self) {
        std::mem::swap(&mut self.src, &mut self.kept);
    }

    /// Area-resizes the resident frame to `dw`×`dh` and downloads it.
    pub fn resize(&mut self, r: Resident, dw: u32, dh: u32) -> Result<Rgb8> {
        self.resize_from(false, r, dw, dh)
    }

    /// Like [`Self::resize`], for the frame [`Self::keep`] set aside.
    pub fn resize_kept(&mut self, r: Resident, dw: u32, dh: u32) -> Result<Rgb8> {
        self.resize_from(true, r, dw, dh)
    }

    fn resize_from(&mut self, kept: bool, r: Resident, dw: u32, dh: u32) -> Result<Rgb8> {
        let t = Instant::now();
        let len = dw as usize * dh as usize * 3;
        Self::ensure_u8(&self.gpu, &mut self.dst, len)?;
        let dst = self.dst.as_mut().expect("allocated above");
        let src = if kept { &self.kept } else { &self.src };
        let src = src.as_ref().expect("upload (and keep) first");
        let (sw, sh, dwi, dhi) = (r.width as i32, r.height as i32, dw as i32, dh as i32);
        let cfg = LaunchConfig {
            grid_dim: (dw.div_ceil(RESIZE_BLOCK.0), dh.div_ceil(RESIZE_BLOCK.1), 1),
            block_dim: (RESIZE_BLOCK.0, RESIZE_BLOCK.1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: src holds sw·sh·3 bytes, dst at least dw·dh·3; matches the kernel signature.
        unsafe {
            self.gpu
                .stream
                .launch_builder(&self.resize)
                .arg(src)
                .arg(&sw)
                .arg(&sh)
                .arg(&mut *dst)
                .arg(&dwi)
                .arg(&dhi)
                .launch(cfg)
        }
        .cuda("launch resize_area")?;
        let mut data = vec![0u8; len];
        self.gpu
            .stream
            .memcpy_dtoh(&dst.slice(..len), &mut data)
            .cuda("download resized frame")?;
        self.seconds += t.elapsed().as_secs_f64();
        Ok(Rgb8 {
            width: dw,
            height: dh,
            data,
        })
    }
}
