//! Per-iteration camera (`frame_begin`/`frame_end`), preprocess forward and
//! backward, camera-gradient reduction. Owner: preprocess (E1).
//! Kernels: `cuda/preprocess.cu`. Spec: spec-raster 1.2-1.3, 2, 5.3-5.5.
//!
//! Buffer contract (spec-raster 1.4; DESIGN.md has the full table):
//! - `frame_begin` writes `f.iter.params`, `f.iter.cam`, zeroes the status counters.
//! - `preprocess_fwd` writes `vis_mask`, `g.last_visible`, `vis_idx`, `rec`,
//!   `depth_key[0]`, `depth_val[0]`, `n_tiles`, zeroes `g2d[·][v < V]`, sets
//!   `status[ST_N_VIS]`.
//! - `preprocess_bwd` reads `g2d`, `vis_idx`, `rec` and writes the sink, camera
//!   partials and densify statistics.
//!
//! STUBS: the stage functions return "not implemented" until E1 lands them.

use spl_core::Result;
use spl_gpu::cudarc::driver::{CudaFunction, CudaStream};

use super::densify::DensifyStats;
use super::{CameraGrads, FrameBuffers, Grad2d, GradSink, SplatKernels};
use crate::frame::{CameraTable, FrameRing};
use crate::gpu::{
    COMMON_CUH, CudaFile, Defines, Gpu, KernelSource, SPLAT_COMMON_CUH, not_implemented,
};
use crate::model::Gaussians;

const OWNER: &str = "preprocess (E1)";

pub const SOURCE: KernelSource = KernelSource {
    module: "preprocess",
    files: &[
        COMMON_CUH,
        SPLAT_COMMON_CUH,
        CudaFile {
            name: "preprocess.cu",
            text: include_str!("../../cuda/preprocess.cu"),
        },
    ],
    kernels: &[
        "frame_begin",
        "frame_end",
        "preprocess_fwd",
        "preprocess_bwd",
        "cam_reduce",
    ],
};

pub struct PreprocessKernels {
    pub frame_begin: CudaFunction,
    pub frame_end: CudaFunction,
    pub fwd: CudaFunction,
    pub bwd: CudaFunction,
    pub cam_reduce: CudaFunction,
}

impl PreprocessKernels {
    pub fn load(gpu: &Gpu, d: &Defines) -> Result<Self> {
        let k = gpu.kernels(&SOURCE, d)?;
        Ok(Self {
            frame_begin: k.get("frame_begin")?,
            frame_end: k.get("frame_end")?,
            fwd: k.get("preprocess_fwd")?,
            bwd: k.get("preprocess_bwd")?,
            cam_reduce: k.get("cam_reduce")?,
        })
    }
}

/// Copies the ring slot of the current iteration into `f.iter.params` and
/// composes `f.iter.cam` (W = exp(ξ^)·W0, intrinsics scaled to the render size).
pub fn frame_begin(
    _k: &SplatKernels,
    _s: &CudaStream,
    _ring: &FrameRing,
    _cams: &CameraTable,
    _f: &mut FrameBuffers,
) -> Result<()> {
    Err(not_implemented("frame_begin", OWNER))
}

/// Last launch of an iteration: status → pinned mirror, iteration counter + 1.
pub fn frame_end(
    _k: &SplatKernels,
    _s: &CudaStream,
    _f: &mut FrameBuffers,
    _status_mirror: u64,
) -> Result<()> {
    Err(not_implemented("frame_end", OWNER))
}

/// Cull, project, colour, tile span and order-preserving compaction into v-space
/// (spec-raster 2). Takes `g` mutably only to stamp `last_visible[p] = t`, and
/// `g2d` to zero the rows of this frame's visible set (spec-raster §8 left both
/// out of the signature; DESIGN.md "reconciliations").
pub fn preprocess_fwd(
    _k: &SplatKernels,
    _s: &CudaStream,
    _g: &mut Gaussians,
    _f: &mut FrameBuffers,
    _g2d: &mut Grad2d,
) -> Result<()> {
    Err(not_implemented("preprocess_fwd", OWNER))
}

/// Chain rule from g2d to the parameters (spec-raster 5.3), pose and focal
/// partials (5.4) and densify statistics (5.5). `g` is `&mut` because the
/// `FusedAdam` sink updates it in place.
#[allow(clippy::too_many_arguments)]
pub fn preprocess_bwd(
    _k: &SplatKernels,
    _s: &CudaStream,
    _g: &mut Gaussians,
    _f: &FrameBuffers,
    _g2d: &Grad2d,
    _sink: GradSink<'_>,
    _cam: Option<&mut CameraGrads>,
    _stats: Option<&mut DensifyStats>,
) -> Result<()> {
    Err(not_implemented("preprocess_bwd", OWNER))
}

/// Sums `cam.partial` in fixed order into `cam.pose[image]` and `cam.intr[intr_id]`.
pub fn cam_reduce(
    _k: &SplatKernels,
    _s: &CudaStream,
    _f: &FrameBuffers,
    _cam: &mut CameraGrads,
) -> Result<()> {
    Err(not_implemented("cam_reduce", OWNER))
}
