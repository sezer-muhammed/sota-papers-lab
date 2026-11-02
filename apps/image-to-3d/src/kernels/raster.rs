//! Tile rasterizer forward and chained per-Gaussian backward, and the
//! [`SplatRaster`] boundary the trainer and densification talk to.
//! Owner: raster (E3). Kernels: `cuda/raster_fwd.cu`, `cuda/raster_bwd.cu`.
//! Spec: spec-raster 4-5.2, spec-train 2.1.
//!
//! Buffer contract:
//! - `raster_fwd` reads `rec`, `inst_v[f.tile_sorted]`, `tile_start`,
//!   `chain_off`; writes [`RenderOut`] and, in Train mode, `ckpt` and
//!   `tile_max_nc` (count mode: `RenderOut::count`).
//! - `raster_bwd` reads the forward's outputs, `ckpt`, `tile_max_nc` and
//!   [`PixelGrads`]; accumulates [`Grad2d`] with per-lane atomics.
//!
//! [`GpuRaster`] composes the stage functions in the order of DESIGN.md; the
//! composition is real, the stages are STUBS until their owners land them.

use spl_core::Result;
use spl_gpu::cudarc::driver::{CudaFunction, CudaSlice, CudaStream};

use super::densify::{DensifyStats, ViewScores};
use super::sort::{RadixSorter, bin_tiles, sort_depth};
use super::{
    CameraGrads, Capacity, FrameBuffers, Grad2d, GradSink, PixelGrads, RasterVariant, RenderMode,
    RenderOut, SplatKernels, preprocess,
};
use crate::frame::{CameraTable, FrameRing, IterParams, IterParamsDev, PinnedStatus};
use crate::gpu::{
    COMMON_CUH, CudaFile, Defines, Gpu, KernelSource, SPLAT_COMMON_CUH, not_implemented,
};
use crate::model::Gaussians;

const OWNER: &str = "raster (E3)";

pub const FWD_SOURCE: KernelSource = KernelSource {
    module: "raster_fwd",
    files: &[
        COMMON_CUH,
        SPLAT_COMMON_CUH,
        CudaFile {
            name: "raster_fwd.cu",
            text: include_str!("../../cuda/raster_fwd.cu"),
        },
    ],
    kernels: &["raster_fwd"],
};

pub const BWD_SOURCE: KernelSource = KernelSource {
    module: "raster_bwd",
    files: &[
        COMMON_CUH,
        SPLAT_COMMON_CUH,
        CudaFile {
            name: "raster_bwd.cu",
            text: include_str!("../../cuda/raster_bwd.cu"),
        },
    ],
    kernels: &["raster_bwd"],
};

pub struct RasterKernels {
    pub fwd: CudaFunction,
    pub bwd: CudaFunction,
}

impl RasterKernels {
    pub fn load(gpu: &Gpu, d: &Defines) -> Result<Self> {
        Ok(Self {
            fwd: gpu.kernels(&FWD_SOURCE, d)?.get("raster_fwd")?,
            bwd: gpu.kernels(&BWD_SOURCE, d)?.get("raster_bwd")?,
        })
    }
}

/// Blend the sorted tile lists (spec-raster 4.2): colour, T, last contributor,
/// optional extra channels; checkpoints every 32·K Gaussians in Train mode.
pub fn raster_fwd(
    _k: &SplatKernels,
    _s: &CudaStream,
    _f: &mut FrameBuffers,
    _mode: RenderMode<'_>,
    _out: &mut RenderOut,
) -> Result<()> {
    Err(not_implemented("raster_fwd", OWNER))
}

/// Chained per-Gaussian backward (spec-raster 5.2) into `g2d`.
pub fn raster_bwd(
    _k: &SplatKernels,
    _s: &CudaStream,
    _f: &FrameBuffers,
    _out: &RenderOut,
    _pg: &PixelGrads,
    _g2d: &mut Grad2d,
) -> Result<()> {
    Err(not_implemented("raster_bwd", OWNER))
}

/// Per-run render options the trainer folds into [`IterParams`] (spec-train 2.1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderOpts {
    /// Active SH degree 0..=3.
    pub sh_degree: u32,
    /// Black for all benchmarks.
    pub background: [f32; 3],
    /// FastGS compact box "mult": 1.0 off, 0.5 Mip360, 0.7 T&T/DB.
    pub compact_box_beta: f32,
    /// Accumulate AbsGS statistics (FastGS split criterion).
    pub abs_grad: bool,
}

impl Default for RenderOpts {
    fn default() -> Self {
        Self {
            sh_degree: 3,
            background: [0.0; 3],
            compact_box_beta: 1.0,
            abs_grad: false,
        }
    }
}

/// Where this frame's parameters come from: the training ring (captured
/// graphs) or one eagerly uploaded [`IterParams`] (eval, VCD scoring, tests).
pub enum FrameSource<'a> {
    Ring(&'a FrameRing),
    Eager(&'a IterParams),
}

/// Where the backward routes gradients (spec-train's `GradSink` struct,
/// expressed with the raster's `GradSink` enum plus the two optional targets).
pub struct GradTargets<'a> {
    pub sink: GradSink<'a>,
    /// Phase 4: per-image pose and focal gradients (wired now, unused in Phase 3).
    pub camera: Option<&'a mut CameraGrads>,
    /// Densification statistics, when the strategy wants them this iteration.
    pub stats: Option<&'a mut DensifyStats>,
}

/// The raster as training sees it (spec-train 2.1). Every method only enqueues
/// work; overflow poisons the iteration (status[ST_OVERFLOW]) instead of failing.
pub trait SplatRaster: Send {
    /// frame_begin → preprocess_fwd → sort_depth → bin_tiles → raster_fwd.
    /// Stamps `g.last_visible[p] = t` for every Gaussian that touches a tile.
    fn forward(
        &mut self,
        gpu: &Gpu,
        g: &mut Gaussians,
        cams: &CameraTable,
        frame: FrameSource<'_>,
        mode: RenderMode<'_>,
    ) -> Result<()>;

    /// Device IterParams, FrameCam and status of the current frame (the loss
    /// and the optimizer read these).
    fn params(&self) -> &IterParamsDev;

    /// The forward's image, f32 planar `[3][H][W]` (the loss input).
    fn image(&self) -> &CudaSlice<f32>;

    /// The forward's outputs (T, n_contrib, extra channels).
    fn output(&self) -> &RenderOut;

    /// Where the loss writes dL/dimage (and dL/dalpha) before `backward`.
    fn pixel_grads(&mut self) -> &mut PixelGrads;

    /// Everything the loss touches at once (a split borrow of the above).
    fn loss_io(&mut self) -> LossIo<'_>;

    /// raster_bwd → preprocess_bwd (→ cam_reduce when `targets.camera` is set).
    fn backward(&mut self, gpu: &Gpu, g: &mut Gaussians, targets: GradTargets<'_>) -> Result<()>;

    /// FastGS counting pass (spec-train 5.4): re-blends exactly like `forward`
    /// and adds, per Gaussian p, the masked pixels it contributes to into
    /// `acc.count[p]` and `view_weight` times that into `acc.score[p]`.
    #[allow(clippy::too_many_arguments)]
    fn count_masked(
        &mut self,
        gpu: &Gpu,
        g: &mut Gaussians,
        cams: &CameraTable,
        frame: FrameSource<'_>,
        mask: &CudaSlice<u8>,
        view_weight: f32,
        acc: &mut ViewScores,
    ) -> Result<()>;

    /// Device status words; `status[ST_OVERFLOW] != 0` means this iteration was a no-op.
    fn overflow_flag(&self) -> &CudaSlice<u32>;

    /// Last launch of an iteration (`frame_end`): status into the pinned
    /// mirror, iteration counter + 1.
    fn end_frame(&mut self, gpu: &Gpu, status_mirror: &PinnedStatus) -> Result<()>;
}

/// The loss's view of one frame: read the params and the image, write the
/// pixel gradients.
pub struct LossIo<'a> {
    pub params: &'a IterParamsDev,
    pub image: &'a CudaSlice<f32>,
    pub grads: &'a mut PixelGrads,
}

/// The production [`SplatRaster`]: owns the kernels, the per-frame buffers and
/// the sorter for one capacity.
pub struct GpuRaster {
    pub kernels: SplatKernels,
    pub frame: FrameBuffers,
    pub out: RenderOut,
    pub pixel_grads: PixelGrads,
    pub g2d: Grad2d,
    pub sorter: RadixSorter,
}

impl GpuRaster {
    pub fn new(gpu: &Gpu, cap: Capacity, v: RasterVariant, abs_grad: bool) -> Result<Self> {
        Ok(Self {
            kernels: SplatKernels::load(gpu, v)?,
            frame: FrameBuffers::alloc(gpu, cap, &v)?,
            out: RenderOut::alloc(gpu, cap, &v, true)?,
            pixel_grads: PixelGrads::alloc(gpu, cap, &v, true)?,
            g2d: Grad2d::alloc(gpu, cap, &v, abs_grad)?,
            sorter: RadixSorter::new(gpu, cap.instances.max(cap.visible))?,
        })
    }
}

impl SplatRaster for GpuRaster {
    fn forward(
        &mut self,
        gpu: &Gpu,
        g: &mut Gaussians,
        cams: &CameraTable,
        frame: FrameSource<'_>,
        mode: RenderMode<'_>,
    ) -> Result<()> {
        let s = &gpu.stream;
        match frame {
            FrameSource::Ring(ring) => {
                preprocess::frame_begin(&self.kernels, s, ring, cams, &mut self.frame)?
            }
            FrameSource::Eager(_) => return Err(not_implemented("eager frame_begin", OWNER)),
        }
        preprocess::preprocess_fwd(&self.kernels, s, g, &mut self.frame, &mut self.g2d)?;
        sort_depth(&self.kernels, s, &mut self.sorter, &mut self.frame)?;
        bin_tiles(&self.kernels, s, &mut self.sorter, &mut self.frame)?;
        raster_fwd(&self.kernels, s, &mut self.frame, mode, &mut self.out)
    }

    fn params(&self) -> &IterParamsDev {
        &self.frame.iter
    }

    fn image(&self) -> &CudaSlice<f32> {
        &self.out.rgb
    }

    fn output(&self) -> &RenderOut {
        &self.out
    }

    fn pixel_grads(&mut self) -> &mut PixelGrads {
        &mut self.pixel_grads
    }

    fn loss_io(&mut self) -> LossIo<'_> {
        LossIo {
            params: &self.frame.iter,
            image: &self.out.rgb,
            grads: &mut self.pixel_grads,
        }
    }

    fn backward(&mut self, gpu: &Gpu, g: &mut Gaussians, targets: GradTargets<'_>) -> Result<()> {
        let s = &gpu.stream;
        raster_bwd(
            &self.kernels,
            s,
            &self.frame,
            &self.out,
            &self.pixel_grads,
            &mut self.g2d,
        )?;
        let GradTargets {
            sink,
            mut camera,
            stats,
        } = targets;
        preprocess::preprocess_bwd(
            &self.kernels,
            s,
            g,
            &self.frame,
            &self.g2d,
            sink,
            camera.as_deref_mut(),
            stats,
        )?;
        if let Some(cam) = camera {
            preprocess::cam_reduce(&self.kernels, s, &self.frame, cam)?;
        }
        Ok(())
    }

    fn count_masked(
        &mut self,
        _gpu: &Gpu,
        _g: &mut Gaussians,
        _cams: &CameraTable,
        _frame: FrameSource<'_>,
        _mask: &CudaSlice<u8>,
        _view_weight: f32,
        _acc: &mut ViewScores,
    ) -> Result<()> {
        Err(not_implemented("GpuRaster::count_masked", OWNER))
    }

    fn overflow_flag(&self) -> &CudaSlice<u32> {
        &self.frame.iter.status
    }

    fn end_frame(&mut self, gpu: &Gpu, status_mirror: &PinnedStatus) -> Result<()> {
        preprocess::frame_end(
            &self.kernels,
            &gpu.stream,
            &mut self.frame,
            status_mirror.device_ptr,
        )
    }
}
