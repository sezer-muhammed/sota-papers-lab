//! GPU TSDF fusion into a sparse voxel hash. Owner: mesh-tsdf.
//!
//! Blocks of 8³ samples are allocated on demand along each pixel's
//! truncation band (`tsdf_alloc`), then every allocated sample is projected
//! into the view and updated (`tsdf_integrate`); see `cuda/mesh_tsdf.cu` and
//! the conventions in [`super`]. Memory: 4 KiB per block (f32 tsdf + f32
//! weight) plus 12 B per hash slot; the hash and pool are fixed at creation
//! and overflow is an error, not silent loss.

use std::sync::OnceLock;

use spl_core::{Result, bail};

macro_rules! ensure {
    ($c:expr, $($t:tt)+) => {
        if !$c {
            bail!($($t)+);
        }
    };
}
use spl_gpu::cudarc::driver::{CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg};

use super::Mesh;
use super::mc::{self, DeviceTables, ExtractOptions};
use crate::gpu::{CudaFile, CudaResultExt, Defines, Gpu, KernelSource, Kernels};

pub const SOURCE: KernelSource = KernelSource {
    module: "mesh_tsdf",
    files: &[CudaFile {
        name: "mesh_tsdf.cu",
        text: include_str!("../../cuda/mesh_tsdf.cu"),
    }],
    kernels: &[
        "tsdf_alloc",
        "tsdf_integrate",
        "mc_count",
        "mc_vertices",
        "mc_faces",
    ],
};

/// Samples per block edge.
pub const BLOCK: u32 = 8;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TsdfParams {
    /// Sample spacing (world units).
    pub voxel: f32,
    /// Truncation distance (2DGS default: 5 voxels).
    pub trunc: f32,
    /// Cap of the per-sample running-mean weight.
    pub max_weight: f32,
    /// World position of sample (0, 0, 0).
    pub origin: [f32; 3],
    /// Depths outside (depth_min, depth_max) are ignored (2DGS `depth_trunc`).
    pub depth_min: f32,
    pub depth_max: f32,
    /// Block pool size.
    pub max_blocks: u32,
    /// Hash slots, a power of two (≥ 2 · max_blocks keeps probes short).
    pub hash_slots: u32,
}

impl TsdfParams {
    /// `res` samples across the cube `[lo, lo + size]³` (the "effective
    /// resolution"), truncation 5 voxels as in 2DGS.
    pub fn for_bounds(lo: [f32; 3], size: f32, res: u32) -> Self {
        let voxel = size / res as f32;
        Self {
            voxel,
            trunc: 5.0 * voxel,
            max_weight: 255.0,
            origin: lo,
            depth_min: 1e-3,
            depth_max: f32::INFINITY,
            max_blocks: 1 << 18,
            hash_slots: 1 << 20,
        }
    }

    pub fn bytes(&self) -> u64 {
        self.max_blocks as u64 * (BLOCK.pow(3) as u64 * 8 + 12) + self.hash_slots as u64 * 12
    }
}

/// A pinhole view of a depth map: world-to-camera pose, COLMAP pixel
/// convention (pixel `(u, v)` covers `[u, u+1) × [v, v+1)`), depth = z_cam.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthCamera {
    pub r_cw: [[f32; 3]; 3],
    pub t_cw: [f32; 3],
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub width: u32,
    pub height: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct TsdfCam {
    r: [f32; 9],
    t: [f32; 3],
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    width: i32,
    height: i32,
    depth_min: f32,
    depth_max: f32,
}

const _: () = assert!(size_of::<TsdfCam>() == 20 * 4);

// SAFETY: #[repr(C)] of f32/i32 only, no padding; matches `TsdfCam` in
// mesh_tsdf.cu field for field.
unsafe impl DeviceRepr for TsdfCam {}

pub struct TsdfVolume {
    gpu: Gpu,
    kernels: Kernels,
    params: TsdfParams,
    keys: CudaSlice<u64>,
    vals: CudaSlice<i32>,
    coords: CudaSlice<i32>,
    tsdf: CudaSlice<f32>,
    weight: CudaSlice<f32>,
    /// [blocks, overflow flags]
    counters: CudaSlice<u32>,
    blocks: u32,
    tables: OnceLock<DeviceTables>,
}

impl TsdfVolume {
    pub fn new(gpu: &Gpu, params: TsdfParams) -> Result<Self> {
        ensure!(
            params.hash_slots.is_power_of_two(),
            "hash_slots must be a power of two"
        );
        ensure!(
            params.voxel > 0.0 && params.trunc > 0.0,
            "voxel and trunc must be positive"
        );
        let s = &gpu.stream;
        let n = params.max_blocks as usize * BLOCK.pow(3) as usize;
        Ok(Self {
            kernels: gpu.kernels(&SOURCE, &Defines::new())?,
            keys: s
                .clone_htod(&vec![u64::MAX; params.hash_slots as usize])
                .cuda("alloc TSDF hash keys")?,
            vals: s
                .clone_htod(&vec![-1i32; params.hash_slots as usize])
                .cuda("alloc TSDF hash values")?,
            coords: s
                .alloc_zeros(params.max_blocks as usize * 3)
                .cuda("alloc TSDF block coords")?,
            tsdf: s.alloc_zeros(n).cuda("alloc TSDF values")?,
            weight: s.alloc_zeros(n).cuda("alloc TSDF weights")?,
            counters: s.alloc_zeros(2).cuda("alloc TSDF counters")?,
            blocks: 0,
            params,
            gpu: Gpu::new(gpu.stream.clone())?,
            tables: OnceLock::new(),
        })
    }

    pub fn params(&self) -> &TsdfParams {
        &self.params
    }

    /// Allocated blocks.
    pub fn blocks(&self) -> u32 {
        self.blocks
    }

    pub(super) fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    pub(super) fn kernels(&self) -> &Kernels {
        &self.kernels
    }

    pub(super) fn hash(&self) -> (&CudaSlice<u64>, &CudaSlice<i32>, u32) {
        (&self.keys, &self.vals, self.params.hash_slots - 1)
    }

    pub(super) fn voxels(&self) -> (&CudaSlice<f32>, &CudaSlice<f32>, &CudaSlice<i32>) {
        (&self.tsdf, &self.weight, &self.coords)
    }

    pub(super) fn device_tables(&self) -> Result<&DeviceTables> {
        if self.tables.get().is_none() {
            let _ = self.tables.set(DeviceTables::upload(self)?);
        }
        Ok(self.tables.get().expect("set above"))
    }

    /// Fuses a host depth map (`width · height`, row-major, z_cam; ≤ 0 or
    /// non-finite = no depth).
    pub fn integrate(&mut self, depth: &[f32], cam: &DepthCamera) -> Result<()> {
        ensure!(
            depth.len() == (cam.width * cam.height) as usize,
            "depth map is {} values, camera is {}x{}",
            depth.len(),
            cam.width,
            cam.height
        );
        let d = self.gpu.stream.clone_htod(depth).cuda("upload depth")?;
        self.integrate_device(&d, cam)
    }

    /// Fuses a depth map already on the device (e.g. rendered median depth).
    pub fn integrate_device(&mut self, depth: &CudaSlice<f32>, cam: &DepthCamera) -> Result<()> {
        ensure!(
            depth.len() >= (cam.width * cam.height) as usize,
            "depth buffer too small"
        );
        let p = self.params;
        let c = TsdfCam {
            r: std::array::from_fn(|i| cam.r_cw[i / 3][i % 3]),
            t: cam.t_cw,
            fx: cam.fx,
            fy: cam.fy,
            cx: cam.cx,
            cy: cam.cy,
            width: cam.width as i32,
            height: cam.height as i32,
            depth_min: p.depth_min,
            depth_max: p.depth_max,
        };
        let s = &self.gpu.stream;
        let [ox, oy, oz] = p.origin;
        let mask = p.hash_slots - 1;
        let f = self.kernels.get("tsdf_alloc")?;
        let mut l = s.launch_builder(&f);
        l.arg(depth)
            .arg(&c)
            .arg(&p.voxel)
            .arg(&p.trunc)
            .arg(&ox)
            .arg(&oy)
            .arg(&oz);
        l.arg(&mut self.keys)
            .arg(&mut self.vals)
            .arg(&mask)
            .arg(&mut self.coords);
        l.arg(&mut self.counters).arg(&p.max_blocks);
        let cfg = LaunchConfig {
            grid_dim: (cam.width.div_ceil(16), cam.height.div_ceil(16), 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: tsdf_alloc(depth[w·h], TsdfCam, voxel, trunc, ox, oy, oz,
        // keys/vals[mask+1], mask, coords[max_blocks·3], counters[2],
        // max_blocks); one thread per pixel, bounds-checked.
        unsafe { l.launch(cfg) }.cuda("tsdf_alloc")?;
        let n = s.clone_dtoh(&self.counters).cuda("read TSDF counters")?;
        if n[1] != 0 {
            bail!(
                "TSDF overflow ({}): {} blocks requested, pool {} / hash {} slots; raise max_blocks/hash_slots or the voxel size",
                if n[1] & 2 != 0 {
                    "hash table full"
                } else {
                    "block pool full"
                },
                n[0],
                p.max_blocks,
                p.hash_slots
            );
        }
        self.blocks = n[0];
        if self.blocks == 0 {
            return Ok(());
        }
        let f = self.kernels.get("tsdf_integrate")?;
        let mut l = s.launch_builder(&f);
        l.arg(depth)
            .arg(&c)
            .arg(&p.voxel)
            .arg(&p.trunc)
            .arg(&p.max_weight);
        l.arg(&ox)
            .arg(&oy)
            .arg(&oz)
            .arg(&self.coords)
            .arg(&mut self.tsdf)
            .arg(&mut self.weight);
        let cfg = LaunchConfig {
            grid_dim: (self.blocks, 1, 1),
            block_dim: (BLOCK.pow(3), 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: tsdf_integrate(depth, TsdfCam, voxel, trunc, max_weight,
        // ox, oy, oz, coords, tsdf, weight); one CUDA block per allocated
        // block (< max_blocks, checked above), 512 samples each.
        unsafe { l.launch(cfg) }.cuda("tsdf_integrate")?;
        Ok(())
    }

    /// Marching cubes over the fused volume.
    pub fn extract(&self, opts: ExtractOptions) -> Result<Mesh> {
        mc::extract(self, opts)
    }
}

#[cfg(test)]
#[path = "tsdf_tests.rs"]
mod tests;
