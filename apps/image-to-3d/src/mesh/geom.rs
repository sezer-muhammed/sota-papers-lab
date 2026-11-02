//! Geometry rendering (median depth, normals) and the Geometry preset's
//! regularisers: depth distortion and depth-normal consistency (2DGS) with
//! PGSR's ray/plane depth and a flattening term on the smallest scale.
//! Owner: mesh-mode. Kernels: `cuda/mesh_geom.cu` (math and conventions in
//! its header).
//!
//! The kernels walk the raster's sorted tile lists after `raster_fwd`, so
//! they see exactly the Gaussians and blend order of the colour image. The
//! alpha-chain gradients land in `g2d` before `raster_bwd` (which
//! accumulates) and flow through `preprocess_bwd` like the colour loss; the
//! normal and depth terms go straight to the quaternion and mean gradients
//! after it.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use spl_core::Result;
use spl_gpu::cudarc::driver::{CudaFunction, CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg};
use ts_rs::TS;

use crate::gpu::{COMMON_CUH, CudaFile, CudaResultExt, Gpu, KernelSource, SPLAT_COMMON_CUH};
use crate::kernels::raster::{GpuRaster, RASTER_COMMON_CUH};
use crate::kernels::{Capacity, RasterVariant};
use crate::model::{Gaussians, Grads};

pub const SOURCE: KernelSource = KernelSource {
    module: "mesh_geom",
    files: &[
        COMMON_CUH,
        SPLAT_COMMON_CUH,
        RASTER_COMMON_CUH,
        CudaFile {
            name: "mesh_geom.cu",
            text: include_str!("../../cuda/mesh_geom.cu"),
        },
    ],
    kernels: &[
        "geom_prep",
        "geom_fwd",
        "geom_pixel",
        "geom_bwd",
        "geom_post",
    ],
};

/// Output channels of `geom_fwd`, planar `[GEOM_CH][P]`.
pub const GEOM_CH: usize = 8;
/// Channel of the median depth (0 where the transmittance never reaches 0.5).
pub const CH_MEDIAN: usize = 0;
pub const CH_ALPHA: usize = 1;
pub const CH_NORMAL: usize = 4;
pub const CH_EXPECTED: usize = 7;
const GPIX_CH: usize = 5;
/// f32 words of one `GeomAux`.
const AUX_WORDS: usize = 6;

/// Weights and schedule of the geometry regularisers (Geometry preset).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct GeometryConfig {
    /// Depth distortion weight (2DGS: 1000 on DTU, 0 on unbounded scenes).
    #[serde(default = "d_dist")]
    pub lambda_dist: f32,
    /// Depth-normal consistency weight (2DGS: 0.05).
    #[serde(default = "d_normal")]
    pub lambda_normal: f32,
    /// Flattening weight on each visible Gaussian's smallest scale (PGSR-style).
    #[serde(default = "d_flat")]
    pub lambda_flat: f32,
    /// Distortion depth mapping m = 1 − near / z, near = this × scene extent.
    #[serde(default = "d_near")]
    pub near_rel: f32,
    /// Fractions of the run where each term switches on (2DGS: 3k and 7k of 30k).
    #[serde(default = "d_dist_from")]
    pub dist_from: f32,
    #[serde(default = "d_normal_from")]
    pub normal_from: f32,
    #[serde(default)]
    pub flat_from: f32,
}

fn d_dist() -> f32 {
    100.0
}
fn d_normal() -> f32 {
    0.05
}
fn d_flat() -> f32 {
    1e-4
}
fn d_near() -> f32 {
    0.06
}
fn d_dist_from() -> f32 {
    0.1
}
fn d_normal_from() -> f32 {
    7.0 / 30.0
}

impl Default for GeometryConfig {
    fn default() -> Self {
        Self {
            lambda_dist: d_dist(),
            lambda_normal: d_normal(),
            lambda_flat: d_flat(),
            near_rel: d_near(),
            dist_from: d_dist_from(),
            normal_from: d_normal_from(),
            flat_from: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GeomParams {
    pub near_m: f32,
    /// Both pixel losses are means over the frame (the kernel divides by P).
    pub lambda_dist: f32,
    pub lambda_normal: f32,
    pub lambda_flat: f32,
    pub dist_from: u32,
    pub normal_from: u32,
    pub flat_from: u32,
    pub v_cap: u32,
}

const _: () = assert!(size_of::<GeomParams>() == 32);
// SAFETY: #[repr(C)] f32/u32 only, no padding; matches `GeomParams` in mesh_geom.cu.
unsafe impl DeviceRepr for GeomParams {}

pub struct GeomKernels {
    prep: CudaFunction,
    fwd: CudaFunction,
    pixel: CudaFunction,
    bwd: CudaFunction,
    post: CudaFunction,
    tile_px: u32,
}

impl GeomKernels {
    pub fn load(gpu: &Gpu, v: &RasterVariant) -> Result<Self> {
        let k = gpu.kernels(&SOURCE, &v.defines())?;
        Ok(Self {
            prep: k.get("geom_prep")?,
            fwd: k.get("geom_fwd")?,
            pixel: k.get("geom_pixel")?,
            bwd: k.get("geom_bwd")?,
            post: k.get("geom_post")?,
            tile_px: v.tile_pixels(),
        })
    }
}

/// Device buffers sized for one raster capacity (visible and pixel caps are
/// fixed for a run; a raster regrow only changes instances).
pub struct GeomBuffers {
    pub cap: Capacity,
    /// `GeomAux` per visible Gaussian.
    pub aux: CudaSlice<f32>,
    /// `[GEOM_CH][P]`.
    pub out: CudaSlice<f32>,
    gpix: Option<CudaSlice<f32>>,
    gz: Option<CudaSlice<f32>>,
    gn: Option<CudaSlice<f32>>,
    /// Running (distortion, normal) loss sums, for logs and tests.
    pub sums: Option<CudaSlice<f32>>,
}

impl GeomBuffers {
    pub fn alloc(gpu: &Gpu, cap: Capacity, train: bool) -> Result<Self> {
        let s = &gpu.stream;
        let vis = cap.visible.max(1) as usize;
        let px = cap.pixels.max(1) as usize;
        let z = |n: usize, what: &str| s.alloc_zeros::<f32>(n).cuda(what);
        Ok(Self {
            cap,
            aux: z(AUX_WORDS * vis, "alloc geom aux")?,
            out: z(GEOM_CH * px, "alloc geom out")?,
            gpix: if train {
                Some(z(GPIX_CH * px, "alloc geom gpix")?)
            } else {
                None
            },
            gz: if train {
                Some(z(vis, "alloc geom gz")?)
            } else {
                None
            },
            gn: if train {
                Some(z(3 * vis, "alloc geom gn")?)
            } else {
                None
            },
            sums: if train {
                Some(z(2, "alloc geom sums")?)
            } else {
                None
            },
        })
    }

    pub fn bytes(cap: Capacity, train: bool) -> u64 {
        let (vis, px) = (cap.visible as u64, cap.pixels as u64);
        4 * (AUX_WORDS as u64 * vis
            + GEOM_CH as u64 * px
            + if train {
                GPIX_CH as u64 * px + 4 * vis + 2
            } else {
                0
            })
    }
}

fn grid_stride(n: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(256).clamp(1, 4096), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// `geom_prep` + `geom_fwd` after a `raster_fwd` of the same frame: fills
/// `b.out` (median depth, alpha, moments, normals, expected depth).
pub fn render(
    gpu: &Gpu,
    k: &GeomKernels,
    raster: &GpuRaster,
    g: &Gaussians,
    b: &mut GeomBuffers,
    gp: GeomParams,
) -> Result<()> {
    let s = &gpu.stream;
    let f = &raster.frame;
    let v_cap = f.cap.visible;
    let null = 0u64;
    {
        let mut l = s.launch_builder(&k.prep);
        l.arg(&f.iter.cam)
            .arg(&f.vis_idx)
            .arg(&g.mean)
            .arg(&g.log_scale)
            .arg(&g.quat)
            .arg(&mut b.aux);
        match (&mut b.gz, &mut b.gn) {
            (Some(gz), Some(gn)) => l.arg(gz).arg(gn),
            _ => l.arg(&null).arg(&null),
        };
        l.arg(&v_cap).arg(&f.iter.status);
        // SAFETY: arguments match `geom_prep`; v < min(status[ST_N_VIS], v_cap),
        // the size of vis_idx, aux (6 words each), gz and gn (3 rows of v_cap).
        unsafe { l.launch(grid_stride(v_cap)) }.cuda("geom_prep")?;
    }
    let mut l = s.launch_builder(&k.fwd);
    l.arg(&f.iter.cam)
        .arg(&f.rec)
        .arg(&b.aux)
        .arg(&f.inst_v[f.tile_sorted])
        .arg(&f.tile_start)
        .arg(&gp)
        .arg(&mut b.out)
        .arg(&f.iter.status);
    let cfg = LaunchConfig {
        grid_dim: (f.cap.tiles.max(1), 1, 1),
        block_dim: (k.tile_px, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: arguments match `geom_fwd`; tiles and the image size come from
    // the device FrameCam (≤ cap.tiles, cap.pixels), records v < v_cap.
    unsafe { l.launch(cfg) }.cuda("geom_fwd")?;
    Ok(())
}

/// The Geometry preset's per-iteration work, captured with the iteration.
pub struct GeomTrain {
    pub cfg: GeometryConfig,
    pub kernels: GeomKernels,
    pub buf: GeomBuffers,
    params: GeomParams,
}

impl GeomTrain {
    pub fn new(
        gpu: &Gpu,
        cfg: GeometryConfig,
        variant: &RasterVariant,
        cap: Capacity,
        extent: f32,
        total: u32,
    ) -> Result<Self> {
        let at = |f: f32| ((f.clamp(0.0, 1.0) * total as f32).round() as u32).max(1);
        let params = GeomParams {
            near_m: cfg.near_rel * extent,
            lambda_dist: cfg.lambda_dist,
            lambda_normal: cfg.lambda_normal,
            lambda_flat: cfg.lambda_flat,
            dist_from: if cfg.lambda_dist > 0.0 {
                at(cfg.dist_from)
            } else {
                u32::MAX
            },
            normal_from: if cfg.lambda_normal > 0.0 {
                at(cfg.normal_from)
            } else {
                u32::MAX
            },
            flat_from: if cfg.lambda_flat > 0.0 {
                at(cfg.flat_from)
            } else {
                u32::MAX
            },
            v_cap: cap.visible,
        };
        Ok(Self {
            kernels: GeomKernels::load(gpu, variant)?,
            buf: GeomBuffers::alloc(gpu, cap, true)?,
            cfg,
            params,
        })
    }

    fn check(&self, raster: &GpuRaster) -> Result<()> {
        let c = raster.frame.cap;
        anyhow::ensure!(
            c.visible <= self.buf.cap.visible && c.pixels <= self.buf.cap.pixels,
            "geometry buffers were sized for {:?}, the raster is {:?}",
            self.buf.cap,
            c
        );
        Ok(())
    }

    /// After `raster_fwd(Train)`, before `raster_bwd`: losses and the
    /// alpha-chain gradients into `g2d`.
    pub fn after_forward(
        &mut self,
        gpu: &Gpu,
        raster: &mut GpuRaster,
        g: &Gaussians,
    ) -> Result<()> {
        self.check(raster)?;
        let mut gp = self.params;
        gp.v_cap = raster.frame.cap.visible;
        let k = &self.kernels;
        render(gpu, k, raster, g, &mut self.buf, gp)?;
        let s = &gpu.stream;
        let f = &raster.frame;
        let b = &mut self.buf;
        let (Some(gpix), Some(gz), Some(gn), Some(sums)) =
            (&mut b.gpix, &mut b.gz, &mut b.gn, &mut b.sums)
        else {
            unreachable!("training buffers are allocated")
        };
        {
            let mut l = s.launch_builder(&k.pixel);
            l.arg(&f.iter.params)
                .arg(&f.iter.cam)
                .arg(&b.out)
                .arg(&gp)
                .arg(&mut *gpix)
                .arg(&mut *sums)
                .arg(&f.iter.status);
            // SAFETY: arguments match `geom_pixel`; pixels < width·height ≤ cap.pixels.
            unsafe { l.launch(grid_stride(f.cap.pixels)) }.cuda("geom_pixel")?;
        }
        let mut l = s.launch_builder(&k.bwd);
        l.arg(&f.iter.params)
            .arg(&f.iter.cam)
            .arg(&f.rec)
            .arg(&b.aux)
            .arg(&f.inst_v[f.tile_sorted])
            .arg(&f.tile_start)
            .arg(&gp)
            .arg(&b.out)
            .arg(&*gpix)
            .arg(&mut raster.g2d.g)
            .arg(&mut *gz)
            .arg(&mut *gn)
            .arg(&f.iter.status);
        let cfg = LaunchConfig {
            grid_dim: (f.cap.tiles.max(1), 1, 1),
            block_dim: (k.tile_px, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: arguments match `geom_bwd`; g2d has 9 rows of v_cap, gz and gn
        // 1 and 3 rows of v_cap, v < v_cap.
        unsafe { l.launch(cfg) }.cuda("geom_bwd")?;
        Ok(())
    }

    /// After `raster.backward` (preprocess_bwd wrote the gradients at p),
    /// before the optimizer step.
    pub fn after_backward(
        &mut self,
        gpu: &Gpu,
        raster: &GpuRaster,
        g: &Gaussians,
        grads: &mut Grads,
    ) -> Result<()> {
        let f = &raster.frame;
        let mut gp = self.params;
        gp.v_cap = f.cap.visible;
        // The flattening loss is a mean over the visible set in PGSR; per
        // Gaussian here (Adam normalises the scale anyway).
        let (Some(gz), Some(gn)) = (&self.buf.gz, &self.buf.gn) else {
            unreachable!("training buffers are allocated")
        };
        let mut l = gpu.stream.launch_builder(&self.kernels.post);
        l.arg(&f.iter.params)
            .arg(&f.iter.cam)
            .arg(&f.vis_idx)
            .arg(&g.mean)
            .arg(&g.log_scale)
            .arg(&g.quat)
            .arg(gz)
            .arg(gn)
            .arg(&gp)
            .arg(&mut grads.mean)
            .arg(&mut grads.log_scale)
            .arg(&mut grads.quat)
            .arg(&f.iter.status);
        // SAFETY: arguments match `geom_post`; v < min(status[ST_N_VIS], v_cap),
        // p = vis_idx[v] < n ≤ the store's and the gradients' capacity.
        unsafe { l.launch(grid_stride(f.cap.visible)) }.cuda("geom_post")?;
        Ok(())
    }

    /// Mean (distortion, normal) loss per frame since the last call (host sync).
    pub fn take_sums(&mut self, gpu: &Gpu, frames: u32) -> Result<(f64, f64)> {
        let Some(s) = &mut self.buf.sums else {
            return Ok((0.0, 0.0));
        };
        let v = gpu.stream.clone_dtoh(&*s).cuda("download geom sums")?;
        gpu.stream.memset_zeros(s).cuda("reset geom sums")?;
        let n = frames.max(1) as f64;
        Ok((v[0] as f64 / n, v[1] as f64 / n))
    }
}

#[cfg(test)]
#[path = "geom_gpu_tests.rs"]
mod gpu_tests;
