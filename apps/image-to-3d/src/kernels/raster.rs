//! Tile rasterizer forward and chained per-Gaussian backward, and the
//! [`SplatRaster`] boundary the trainer and densification talk to.
//! Owner: raster (E3). Kernels: `cuda/raster_fwd.cu`, `cuda/raster_bwd.cu`,
//! with the decisions both must share in `cuda/raster_common.cuh`.
//! Spec: spec-raster 4-5.2, spec-train 2.1.
//!
//! Buffer contract:
//! - `raster_fwd` reads `rec`, `inst_v[f.tile_sorted]`, `tile_start`,
//!   `chain_off`; writes [`RenderOut`] and, in Train mode, `ckpt` and
//!   `tile_max_nc` (count mode: `RenderOut::count`).
//! - `raster_bwd` reads the forward's outputs, `ckpt`, `tile_max_nc` and
//!   [`PixelGrads`]; accumulates [`Grad2d`] with one atomic set per
//!   (Gaussian, tile).
//!
//! [`GpuRaster`] composes the stage functions in the order of DESIGN.md; the
//! raster stages are implemented, preprocess, sort and binning belong to E1
//! and E2. Kernel tests feed hand-built tile lists (`scene`), compare with a
//! CPU f64 reference and with gsplat through gs-oracle (`tests`), and time
//! the kernels on trained garden models (`bench`).

use anyhow::{anyhow, bail, ensure};
use spl_core::Result;
use spl_gpu::cudarc::driver::{
    CudaFunction, CudaSlice, CudaStream, DevicePtrMut, DeviceRepr, LaunchConfig, PushKernelArg,
    ValidAsZeroBits, result,
};

use super::densify::{DensifyStats, ViewScores};
use super::layout::{BWD_WARPS, MODE_COUNT, MODE_INFER, MODE_TRAIN};
use super::sort::{RadixSorter, bin_tiles, sort_depth};
use super::{
    CameraGrads, Capacity, FrameBuffers, Grad2d, GradSink, PixelGrads, RasterVariant, RenderMode,
    RenderOut, SplatKernels, preprocess,
};
use crate::frame::{CameraTable, FrameRing, IterParams, IterParamsDev, PinnedStatus};
use crate::gpu::{
    COMMON_CUH, CudaFile, CudaResultExt, Defines, Gpu, KernelSource, SPLAT_COMMON_CUH,
    not_implemented,
};
use crate::model::Gaussians;

#[cfg(test)]
mod bench;
#[cfg(test)]
pub(crate) mod scene;
#[cfg(test)]
mod tests;

const OWNER: &str = "raster (E3)";

/// Inline code both raster units share: the skip, clamp and subtile-cull
/// decisions must agree bit for bit between forward and backward.
pub const RASTER_COMMON_CUH: CudaFile = CudaFile {
    name: "raster_common.cuh",
    text: include_str!("../../cuda/raster_common.cuh"),
};

pub const FWD_SOURCE: KernelSource = KernelSource {
    module: "raster_fwd",
    files: &[
        COMMON_CUH,
        SPLAT_COMMON_CUH,
        RASTER_COMMON_CUH,
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
        RASTER_COMMON_CUH,
        CudaFile {
            name: "raster_bwd.cu",
            text: include_str!("../../cuda/raster_bwd.cu"),
        },
    ],
    kernels: &["raster_bwd"],
};

/// Null device pointer for optional kernel arguments.
const NULL: u64 = 0;

/// Stream-ordered memset to zero on a borrowed stream (cudarc's
/// `memset_zeros` needs the stream's `Arc`; stage functions get `&CudaStream`).
fn zero_async<T: DeviceRepr + ValidAsZeroBits>(
    s: &CudaStream,
    dst: &mut CudaSlice<T>,
) -> Result<()> {
    let bytes = dst.num_bytes();
    let (ptr, _record) = dst.device_ptr_mut(s);
    // SAFETY: `ptr` is `dst`'s live allocation of `bytes` bytes (the record
    // keeps it ordered on `s`); all-zero bits are a valid `T`.
    unsafe { result::memset_d8_async(ptr, 0, bytes, s.cu_stream()) }.cuda("memset to zero")
}

pub struct RasterKernels {
    pub fwd: CudaFunction,
    pub bwd: CudaFunction,
    /// Persistent grid of the backward: SMs × resident blocks per SM.
    pub bwd_blocks: u32,
}

impl RasterKernels {
    pub fn load(gpu: &Gpu, d: &Defines) -> Result<Self> {
        let bwd = gpu.kernels(&BWD_SOURCE, d)?.get("raster_bwd")?;
        let per_sm = bwd
            .occupancy_max_active_blocks_per_multiprocessor(BWD_WARPS * 32, 0, None)
            .cuda("raster_bwd occupancy")?
            .max(1);
        Ok(Self {
            fwd: gpu.kernels(&FWD_SOURCE, d)?.get("raster_fwd")?,
            bwd,
            bwd_blocks: per_sm * gpu.multiprocessors(),
        })
    }
}

/// f32 words per tile pixel and chain in `FrameBuffers::ckpt`: (C.rgb, T) as
/// a float4 (a half4 in two words with `ckpt_f16`) plus one f32 per extra
/// channel (layout in `cuda/raster_common.cuh`).
pub fn ckpt_words(v: &RasterVariant) -> u64 {
    (if v.ckpt_f16 { 2 } else { 4 }) + v.extra_ch as u64
}

/// The variant knobs the raster kernels implement. `rec32` (the 32 B fp16
/// record) and `deterministic` (per-instance gradients; needs the tile sort to
/// carry the pre-sort instance index) compile but are refused here until
/// their buffers exist, so nothing silently runs the default instead.
pub fn check_variant(v: &RasterVariant) -> Result<()> {
    v.validate()?;
    if v.rec32 {
        bail!("raster: the 32 B fp16 record (rec32) is not implemented yet ({OWNER})");
    }
    if v.deterministic {
        bail!(
            "raster: DETERMINISTIC needs per-instance gradient buffers and the tile sort's \
             pre-sort index (sort E2 + {OWNER}); not implemented yet"
        );
    }
    Ok(())
}

/// The chain capacity of `f`, after checking that its checkpoint buffer holds
/// [`ckpt_words`] per pixel for every chain.
fn chain_capacity(f: &FrameBuffers) -> Result<u32> {
    let v = &f.variant;
    let c_cap = f.cap.chains(v);
    let need = c_cap as u64 * v.tile_pixels() as u64 * ckpt_words(v);
    ensure!(
        f.ckpt.len() as u64 >= need,
        "raster: the checkpoint buffer holds {} floats; {c_cap} chains × {} px × {} words = {need} \
         are needed (size FrameBuffers::ckpt with raster::ckpt_words)",
        f.ckpt.len(),
        v.tile_pixels(),
        ckpt_words(v)
    );
    Ok(c_cap)
}

/// Blend the sorted tile lists (spec-raster 4.2): colour, T, last contributor,
/// optional extra channels; checkpoints every 32·K Gaussians in Train mode;
/// `RenderOut::count` (zeroed here first) in Count mode.
pub fn raster_fwd(
    k: &SplatKernels,
    s: &CudaStream,
    f: &mut FrameBuffers,
    mode: RenderMode<'_>,
    out: &mut RenderOut,
) -> Result<()> {
    let v = k.variant;
    check_variant(&v)?;
    ensure!(
        f.variant == v,
        "raster_fwd: buffers were allocated for {:?}, kernels for {v:?}",
        f.variant
    );
    let c_cap = chain_capacity(f)?;
    ensure!(
        v.extra_ch == 0 || out.extra.is_some(),
        "raster_fwd: EXTRA_CH = {} needs RenderOut::extra",
        v.extra_ch
    );
    let (mode_id, mask) = match mode {
        RenderMode::Train => (MODE_TRAIN, None),
        RenderMode::Infer => (MODE_INFER, None),
        RenderMode::Count { mask } => (MODE_COUNT, Some(mask)),
    };
    if mode_id == MODE_COUNT {
        let count = out
            .count
            .as_mut()
            .ok_or_else(|| anyhow!("raster_fwd: Count mode needs RenderOut::count"))?;
        zero_async(s, count)?;
    }
    let mut b = s.launch_builder(&k.raster.fwd);
    b.arg(&f.iter.params)
        .arg(&f.iter.cam)
        .arg(&f.rec)
        .arg(&f.inst_v[f.tile_sorted])
        .arg(&f.tile_start)
        .arg(&f.chain_off)
        .arg(&mode_id);
    match mask {
        Some(m) => b.arg(m),
        None => b.arg(&NULL),
    };
    b.arg(&mut out.rgb)
        .arg(&mut out.trans)
        .arg(&mut out.n_contrib);
    match &mut out.extra {
        Some(x) => b.arg(x),
        None => b.arg(&NULL),
    };
    b.arg(&mut f.ckpt).arg(&c_cap).arg(&mut f.tile_max_nc);
    match &mut out.count {
        Some(c) if mode_id == MODE_COUNT => b.arg(c),
        _ => b.arg(&NULL),
    };
    b.arg(&f.iter.status);
    let cfg = LaunchConfig {
        grid_dim: (f.cap.tiles.max(1), 1, 1),
        block_dim: (v.tile_pixels(), 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: arguments match `raster_fwd` in cuda/raster_fwd.cu. Every buffer
    // has its FrameBuffers/RenderOut capacity (tiles ≤ cap.tiles through the
    // device FrameCam, pixels ≤ cap.pixels, chains ≤ c_cap checked above,
    // v < cap.visible); optional pointers are null exactly when the mode or
    // the variant leaves them untouched.
    unsafe { b.launch(cfg) }.cuda("raster_fwd")?;
    Ok(())
}

/// Chained per-Gaussian backward (spec-raster 5.2) into `g2d`. It accumulates
/// (preprocess_fwd zeroes the rows of the frame's visible set) and resets its
/// own chain counter first.
pub fn raster_bwd(
    k: &SplatKernels,
    s: &CudaStream,
    f: &mut FrameBuffers,
    out: &RenderOut,
    pg: &PixelGrads,
    g2d: &mut Grad2d,
) -> Result<()> {
    let v = k.variant;
    check_variant(&v)?;
    let c_cap = chain_capacity(f)?;
    ensure!(
        v.extra_ch == 0 || (out.extra.is_some() && pg.d_extra.is_some() && g2d.extra.is_some()),
        "raster_bwd: EXTRA_CH = {} needs RenderOut::extra, PixelGrads::d_extra and Grad2d::extra",
        v.extra_ch
    );
    zero_async(s, &mut f.chain_ctr)?;
    let v_cap = f.cap.visible;
    let mut b = s.launch_builder(&k.raster.bwd);
    b.arg(&f.iter.params)
        .arg(&f.iter.cam)
        .arg(&f.rec)
        .arg(&f.inst_v[f.tile_sorted])
        .arg(&f.tile_start)
        .arg(&f.chain_off)
        .arg(&f.tile_max_nc)
        .arg(&f.ckpt)
        .arg(&c_cap)
        .arg(&out.rgb)
        .arg(&out.trans)
        .arg(&out.n_contrib);
    match &out.extra {
        Some(x) => b.arg(x),
        None => b.arg(&NULL),
    };
    b.arg(&pg.d_rgb);
    match &pg.d_alpha {
        Some(x) => b.arg(x),
        None => b.arg(&NULL),
    };
    match &pg.d_extra {
        Some(x) => b.arg(x),
        None => b.arg(&NULL),
    };
    b.arg(&v_cap).arg(&mut g2d.g);
    match &mut g2d.abs {
        Some(x) => b.arg(x),
        None => b.arg(&NULL),
    };
    match &mut g2d.extra {
        Some(x) => b.arg(x),
        None => b.arg(&NULL),
    };
    b.arg(&mut f.chain_ctr).arg(&f.iter.status);
    let cfg = LaunchConfig {
        grid_dim: (k.raster.bwd_blocks, 1, 1),
        block_dim: (BWD_WARPS * 32, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: arguments match `raster_bwd` in cuda/raster_bwd.cu; the chain
    // count comes from status[ST_N_CHAINS] (≤ c_cap, the checkpoint capacity
    // checked above), tiles from FrameCam, v < v_cap; optional pointers are
    // null exactly when the variant or the flags leave them unused.
    unsafe { b.launch(cfg) }.cuda("raster_bwd")?;
    Ok(())
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
            // Needs a frame_begin that composes the FrameCam from uploaded
            // params instead of a ring slot (preprocess, E1).
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
            &mut self.frame,
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
        gpu: &Gpu,
        g: &mut Gaussians,
        cams: &CameraTable,
        frame: FrameSource<'_>,
        mask: &CudaSlice<u8>,
        _view_weight: f32,
        _acc: &mut ViewScores,
    ) -> Result<()> {
        // The raster half is real: RenderOut::count[v] holds the masked pixels
        // each visible Gaussian blends into. Scattering it into the p-space
        // ViewScores is densify.cu's `scores_add`, which has no stage function yet.
        self.forward(gpu, g, cams, frame, RenderMode::Count { mask })?;
        Err(not_implemented(
            "scores_add stage (count_masked)",
            "densify (E6)",
        ))
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
