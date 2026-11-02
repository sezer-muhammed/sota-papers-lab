//! Per-iteration camera (`frame_begin`/`frame_end`), preprocess forward and
//! backward, camera-gradient reduction. Owner: preprocess (E1).
//! Kernels: `cuda/preprocess.cu`. Spec: spec-raster 1.2-1.3, 2, 5.3-5.5.
//!
//! Buffer contract (spec-raster 1.4; DESIGN.md has the full table):
//! - `frame_begin` writes `f.iter.params` (from the ring, or uploaded eagerly),
//!   `f.iter.cam`, zeroes the status counters.
//! - `preprocess_fwd` writes `vis_mask`, `g.last_visible`, `vis_idx`, `rec`,
//!   `depth_key[0]`, `depth_val[0]`, `n_tiles`, zeroes `g2d[·][v < V]`, sets
//!   `status[ST_N_VIS]`.
//! - `preprocess_bwd` reads `g2d`, `vis_idx` and the parameters (it recomputes
//!   the forward instead of storing it) and writes the sink, camera partials
//!   and densify statistics.
//!
//! [`span`] is the bit-exact CPU mirror of the tile-span functions in
//! `splat_common.cuh`, for tests of every stage that bins tiles.

use anyhow::ensure;
use spl_core::Result;
use spl_gpu::cudarc::driver::{
    CudaFunction, CudaSlice, CudaStream, DevicePtrMut, DeviceRepr, DriverError, LaunchConfig,
    PushKernelArg, result,
};

use super::densify::DensifyStats;
use super::layout::{CAM_GRAD_N, PRE_THREADS};
use super::{CameraGrads, FrameBuffers, Grad2d, GradSink, SplatKernels};
use crate::frame::{CameraTable, FrameRing};
use crate::gpu::{
    COMMON_CUH, CudaFile, CudaResultExt, Defines, Gpu, KernelSource, SPLAT_COMMON_CUH,
    not_implemented,
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

/// `preprocess_bwd`'s `sink` argument.
const SINK_WRITE: u32 = 0;

/// Most resident blocks per SM a persistent grid uses (bounds the camera partials).
pub const MAX_BLOCKS_PER_SM: u32 = 8;

pub struct PreprocessKernels {
    pub frame_begin: CudaFunction,
    pub frame_end: CudaFunction,
    pub fwd: CudaFunction,
    pub bwd: CudaFunction,
    pub cam_reduce: CudaFunction,
    /// Persistent grid of `preprocess_fwd`: SMs × resident blocks per SM.
    pub fwd_blocks: u32,
    /// Fixed grid of `preprocess_bwd` (= rows of `CameraGrads::partial` it writes).
    pub bwd_blocks: u32,
}

impl PreprocessKernels {
    pub fn load(gpu: &Gpu, d: &Defines) -> Result<Self> {
        let k = gpu.kernels(&SOURCE, d)?;
        let fwd = k.get("preprocess_fwd")?;
        let bwd = k.get("preprocess_bwd")?;
        let resident = |f: &CudaFunction, what: &str| -> Result<u32> {
            let per_sm = f
                .occupancy_max_active_blocks_per_multiprocessor(PRE_THREADS, 0, None)
                .cuda(what)?;
            Ok(gpu.multiprocessors() * per_sm.clamp(1, MAX_BLOCKS_PER_SM))
        };
        Ok(Self {
            fwd_blocks: resident(&fwd, "occupancy of preprocess_fwd")?,
            bwd_blocks: resident(&bwd, "occupancy of preprocess_bwd")?,
            frame_begin: k.get("frame_begin")?,
            frame_end: k.get("frame_end")?,
            fwd,
            bwd,
            cam_reduce: k.get("cam_reduce")?,
        })
    }

    /// Rows `CameraGrads::partial` needs on this device (allocate with this).
    pub fn cam_partial_blocks(gpu: &Gpu) -> u32 {
        gpu.multiprocessors() * MAX_BLOCKS_PER_SM
    }
}

/// Async memset to 0 on `s` (a memset node when captured). cudarc's own
/// `memset_zeros` needs the `Arc<CudaStream>` the stage functions do not get.
fn memset_zero<T: DeviceRepr>(
    s: &CudaStream,
    dst: &mut CudaSlice<T>,
) -> std::result::Result<(), DriverError> {
    let bytes = dst.num_bytes();
    let (ptr, _record) = dst.device_ptr_mut(s);
    // SAFETY: `ptr` addresses `bytes` bytes of `dst`, alive and exclusively
    // borrowed for the call; the write is ordered on `s`.
    unsafe { result::memset_d8_async(ptr, 0, bytes, s.cu_stream()) }
}

fn blocks(n: u32, threads: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.max(1), 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Copies the ring slot of the current iteration into `f.iter.params` and
/// composes `f.iter.cam` (W = exp(ξ^)·W0, intrinsics scaled to the render size).
pub fn frame_begin(
    k: &SplatKernels,
    s: &CudaStream,
    ring: &FrameRing,
    cams: &CameraTable,
    f: &mut FrameBuffers,
) -> Result<()> {
    ensure!(ring.slots > 0, "frame ring has no slots");
    launch_frame_begin(k, s, ring.device_ptr, ring.slots as u32, cams, f)
}

/// [`frame_begin`] for an eagerly uploaded frame (`f.iter.upload`): composes
/// `f.iter.cam` from the `IterParams` already in device memory (eval renders,
/// VCD scoring, tests).
pub fn frame_begin_eager(
    k: &SplatKernels,
    s: &CudaStream,
    cams: &CameraTable,
    f: &mut FrameBuffers,
) -> Result<()> {
    launch_frame_begin(k, s, 0, 0, cams, f)
}

fn launch_frame_begin(
    k: &SplatKernels,
    s: &CudaStream,
    ring: u64,
    slots: u32,
    cams: &CameraTable,
    f: &mut FrameBuffers,
) -> Result<()> {
    let it = &mut f.iter;
    let mut b = s.launch_builder(&k.preprocess.frame_begin);
    b.arg(&ring)
        .arg(&it.iter_counter)
        .arg(&slots)
        .arg(&cams.base)
        .arg(&cams.intr)
        .arg(&cams.pose_delta)
        .arg(&mut it.params)
        .arg(&mut it.cam)
        .arg(&mut it.status);
    // SAFETY: matches `frame_begin` in preprocess.cu; `ring` is a mapped pinned
    // device pointer of `slots` IterParams (or 0 with slots = 0); the camera
    // table holds the image IterParams.cam_index names.
    unsafe { b.launch(blocks(1, 64)) }.cuda("launch frame_begin")?;
    Ok(())
}

/// Last launch of an iteration: status → pinned mirror, iteration counter + 1.
/// `status_mirror` is a mapped pinned device pointer of `STATUS_LEN` words, or 0.
pub fn frame_end(
    k: &SplatKernels,
    s: &CudaStream,
    f: &mut FrameBuffers,
    status_mirror: u64,
) -> Result<()> {
    let it = &mut f.iter;
    let mut b = s.launch_builder(&k.preprocess.frame_end);
    b.arg(&it.params)
        .arg(&mut it.status)
        .arg(&status_mirror)
        .arg(&mut it.iter_counter);
    // SAFETY: matches `frame_end` in preprocess.cu; the mirror holds
    // STATUS_LEN words or is null.
    unsafe { b.launch(blocks(1, 32)) }.cuda("launch frame_end")?;
    Ok(())
}

/// Cull, project, colour, tile span and order-preserving compaction into v-space
/// (spec-raster 2). Takes `g` mutably only to stamp `last_visible[p] = t`, and
/// `g2d` to zero the rows of this frame's visible set (spec-raster §8 left both
/// out of the signature; DESIGN.md "reconciliations").
pub fn preprocess_fwd(
    k: &SplatKernels,
    s: &CudaStream,
    g: &mut Gaussians,
    f: &mut FrameBuffers,
    g2d: &mut Grad2d,
) -> Result<()> {
    ensure!(
        g.cap <= f.cap.gaussians,
        "Gaussian capacity {} exceeds the frame buffers' {}",
        g.cap,
        f.cap.gaussians
    );
    let pk = &k.preprocess;
    let v_cap = f.cap.visible;
    memset_zero(s, &mut f.chunk_ctr).cuda("zero preprocess chunk counter")?;
    memset_zero(s, &mut f.chunk_lookback).cuda("zero preprocess look-back")?;
    let null = 0u64;
    let mut b = s.launch_builder(&pk.fwd);
    b.arg(&f.iter.params)
        .arg(&f.iter.cam)
        .arg(&g.mean)
        .arg(&g.log_scale)
        .arg(&g.quat)
        .arg(&g.opacity)
        .arg(&g.sh_dc)
        .arg(&g.sh_rest);
    match &g.filter3d {
        Some(x) => b.arg(x),
        None => b.arg(&null),
    };
    b.arg(&g.cap)
        .arg(&v_cap)
        .arg(&mut f.vis_mask)
        .arg(&mut g.last_visible)
        .arg(&mut f.vis_idx)
        .arg(&mut f.rec)
        .arg(&mut f.depth_key[0])
        .arg(&mut f.depth_val[0])
        .arg(&mut f.n_tiles)
        .arg(&mut g2d.g);
    match &mut g2d.abs {
        Some(x) => b.arg(x),
        None => b.arg(&null),
    };
    match &mut g2d.extra {
        Some(x) => b.arg(x),
        None => b.arg(&null),
    };
    b.arg(&mut f.chunk_ctr)
        .arg(&mut f.chunk_lookback)
        .arg(&mut f.iter.status);
    // SAFETY: matches `preprocess_fwd` in preprocess.cu. Parameter buffers hold
    // g.cap Gaussians (sh_rest planar with stride g.cap), the per-v buffers and
    // g2d rows hold v_cap entries, vis_mask cap/32 words, chunk_lookback cap/256
    // words (FrameBuffers::alloc sizes them from the same capacity).
    unsafe { b.launch(blocks(pk.fwd_blocks, PRE_THREADS)) }.cuda("launch preprocess_fwd")?;
    Ok(())
}

/// Chain rule from g2d to the parameters (spec-raster 5.3), pose and focal
/// partials (5.4) and densify statistics (5.5). `g` is `&mut` because the
/// `FusedAdam` sink updates it in place.
///
/// Camera partials need `FLAG_CAM_GRAD` in the frame flags and `cam.partial`
/// of at least `k.preprocess.bwd_blocks` rows; statistics need
/// `FLAG_DENSIFY_STATS`.
#[allow(clippy::too_many_arguments)]
pub fn preprocess_bwd(
    k: &SplatKernels,
    s: &CudaStream,
    g: &mut Gaussians,
    f: &FrameBuffers,
    g2d: &Grad2d,
    sink: GradSink<'_>,
    cam: Option<&mut CameraGrads>,
    stats: Option<&mut DensifyStats>,
) -> Result<()> {
    let pk = &k.preprocess;
    let GradSink::Write(grads) = sink else {
        return Err(not_implemented(
            "the FusedAdam sink of preprocess_bwd (waits for adam.cuh's adam_apply)",
            OWNER,
        ));
    };
    ensure!(
        grads.cap == g.cap,
        "gradient capacity {} differs from the Gaussians' {} (planar SH stride)",
        grads.cap,
        g.cap
    );
    let v_cap = f.cap.visible;
    let null = 0u64;
    let mut b = s.launch_builder(&pk.bwd);
    b.arg(&f.iter.params)
        .arg(&f.iter.cam)
        .arg(&g.mean)
        .arg(&g.log_scale)
        .arg(&g.quat)
        .arg(&g.opacity)
        .arg(&g.sh_dc)
        .arg(&g.sh_rest);
    match &g.filter3d {
        Some(x) => b.arg(x),
        None => b.arg(&null),
    };
    b.arg(&g.cap).arg(&v_cap).arg(&f.vis_idx).arg(&g2d.g);
    match &g2d.abs {
        Some(x) => b.arg(x),
        None => b.arg(&null),
    };
    match &g2d.extra {
        Some(x) => b.arg(x),
        None => b.arg(&null),
    };
    b.arg(&SINK_WRITE)
        .arg(&mut grads.mean)
        .arg(&mut grads.log_scale)
        .arg(&mut grads.quat)
        .arg(&mut grads.opacity)
        .arg(&mut grads.sh_dc)
        .arg(&mut grads.sh_rest);
    match cam {
        Some(c) => {
            ensure!(
                c.partial.len() >= (pk.bwd_blocks * CAM_GRAD_N) as usize,
                "CameraGrads::partial holds {} floats, preprocess_bwd writes {} blocks x {CAM_GRAD_N}",
                c.partial.len(),
                pk.bwd_blocks
            );
            b.arg(&mut c.partial)
        }
        None => b.arg(&null),
    };
    match stats {
        Some(st) => {
            b.arg(&mut st.vis_count).arg(&mut st.grad_sum);
            match &mut st.grad_abs_sum {
                Some(x) => b.arg(x),
                None => b.arg(&null),
            }
        }
        None => b.arg(&null).arg(&null).arg(&null),
    };
    b.arg(&f.iter.status);
    // SAFETY: matches `preprocess_bwd` in preprocess.cu; parameter and gradient
    // buffers share capacity g.cap (checked), g2d rows hold v_cap entries,
    // camera partials hold bwd_blocks rows (checked), statistics hold g.cap.
    unsafe { b.launch(blocks(pk.bwd_blocks, PRE_THREADS)) }.cuda("launch preprocess_bwd")?;
    Ok(())
}

/// Adds the per-block partials of the last `preprocess_bwd` (fixed order) to
/// `cam.pose[image]` (ρ, φ) and `cam.intr[intr_id]`. Intrinsics gradients are
/// w.r.t. the normalized intrinsics (fx/W, fy/H, cx/W, cy/H), so frames at any
/// render scale accumulate consistently; divide by the image size for pixel
/// units. The pose optimizer zeroes what it steps.
pub fn cam_reduce(
    k: &SplatKernels,
    s: &CudaStream,
    f: &FrameBuffers,
    cam: &mut CameraGrads,
) -> Result<()> {
    let n_blocks = k.preprocess.bwd_blocks;
    let mut b = s.launch_builder(&k.preprocess.cam_reduce);
    b.arg(&f.iter.cam)
        .arg(&cam.partial)
        .arg(&n_blocks)
        .arg(&mut cam.pose)
        .arg(&mut cam.intr)
        .arg(&f.iter.status);
    // SAFETY: matches `cam_reduce` in preprocess.cu; partial holds n_blocks rows
    // (checked in preprocess_bwd), pose/intr hold the frame's image and intr id.
    unsafe { b.launch(blocks(1, 32 * CAM_GRAD_N)) }.cuda("launch cam_reduce")?;
    Ok(())
}

/// Bit-exact CPU mirror of the tile-span functions of `splat_common.cuh`
/// (spec-raster 2.8). The CUDA side uses only round-to-nearest intrinsics, so
/// IEEE f32 arithmetic here reproduces every count and every tile of
/// `preprocess_fwd`, `emit` and the raster's subtile cull.
pub mod span {
    use crate::kernels::SplatRec;
    use crate::kernels::layout::{CONIC_SCALE_AC, CONIC_SCALE_B, KBIN_ABS, KBIN_REL, TILE_W};

    /// `SpanEllipse` of splat_common.cuh.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct SpanEllipse {
        pub mx: f32,
        pub my: f32,
        pub a: f32,
        pub b: f32,
        pub c: f32,
        pub det: f32,
        pub k: f32,
        pub x_ext: f32,
        pub y_ext: f32,
    }

    pub fn span_ellipse(r: &SplatRec, box_beta: f32) -> SpanEllipse {
        let a = r.ca / CONIC_SCALE_AC;
        let b = r.cb / CONIC_SCALE_B;
        let c = r.cc / CONIC_SCALE_AC;
        let det = a * c - b * b;
        let k = box_beta * (2.0 * std::f32::consts::LN_2) * r.pthr.max(0.0);
        let k = k * (1.0 + KBIN_REL) + KBIN_ABS;
        let kd = k / det;
        SpanEllipse {
            mx: r.mx,
            my: r.my,
            a,
            b,
            c,
            det,
            k,
            x_ext: (kd * c).sqrt(),
            y_ext: (kd * a).sqrt(),
        }
    }

    /// CUDA's fminf/fmaxf: a NaN operand yields the other one.
    fn fmin(a: f32, b: f32) -> f32 {
        a.min(b)
    }

    fn fmax(a: f32, b: f32) -> f32 {
        a.max(b)
    }

    pub fn band_x_interval(e: &SpanEllipse, y0: f32, y1: f32) -> (f32, f32) {
        let sh = (e.b * e.x_ext) / e.c;
        let yl = fmin(fmax(e.my + sh, y0), y1);
        let yr = fmin(fmax(e.my - sh, y0), y1);
        let ak = e.a * e.k;
        let dl = yl - e.my;
        let dr = yr - e.my;
        let sl = fmax(ak - e.det * (dl * dl), 0.0).sqrt();
        let sr = fmax(ak - e.det * (dr * dr), 0.0).sqrt();
        let xl = e.mx + (-(e.b * dl) - sl) / e.a;
        let xr = e.mx + (-(e.b * dr) + sr) / e.a;
        (xl, xr)
    }

    fn axis(c: f32, ext: f32, size: u32) -> (i32, i32) {
        let fs = size as f32;
        let lo = fmin(fmax(((c - ext) - 0.5).ceil(), 0.0), fs) as i32;
        let hi = fmin(fmax(((c + ext) - 0.5).floor(), -1.0), fs - 1.0) as i32;
        (lo, hi)
    }

    /// (px0, px1, py0, py1), or None when empty.
    pub fn span_pixel_box(e: &SpanEllipse, w: u32, h: u32) -> Option<(i32, i32, i32, i32)> {
        let (px0, px1) = axis(e.mx, e.x_ext, w);
        let (py0, py1) = axis(e.my, e.y_ext, h);
        (px0 <= px1 && py0 <= py1).then_some((px0, px1, py0, py1))
    }

    /// Tile columns (tx0, tx1) of tile row `ty`, or None.
    pub fn span_tile_row(
        e: &SpanEllipse,
        ty: i32,
        bx: (i32, i32, i32, i32),
        tile_h: i32,
    ) -> Option<(i32, i32)> {
        let (px0, px1, py0, py1) = bx;
        let r0 = (ty * tile_h).max(py0);
        let r1 = (ty * tile_h + tile_h - 1).min(py1);
        if r0 > r1 {
            return None;
        }
        let (xl, xr) = band_x_interval(e, r0 as f32 + 0.5, r1 as f32 + 0.5);
        let c0 = fmin(fmax((xl - 0.5).ceil(), px0 as f32), px1 as f32 + 1.0) as i32;
        let c1 = fmin(fmax((xr - 0.5).floor(), px0 as f32 - 1.0), px1 as f32) as i32;
        (c0 <= c1).then(|| (c0 / TILE_W as i32, c1 / TILE_W as i32))
    }

    /// Every tile `(tx, ty)` the record spans, row-major (what `emit` writes).
    pub fn tiles(r: &SplatRec, box_beta: f32, w: u32, h: u32, tile_h: u32) -> Vec<(u32, u32)> {
        let e = span_ellipse(r, box_beta);
        let mut out = Vec::new();
        if e.det.is_nan() || e.det <= 0.0 {
            return out;
        }
        let Some(bx) = span_pixel_box(&e, w, h) else {
            return out;
        };
        let th = tile_h as i32;
        for ty in bx.2 / th..=bx.3 / th {
            if let Some((t0, t1)) = span_tile_row(&e, ty, bx, th) {
                out.extend((t0..=t1).map(|tx| (tx as u32, ty as u32)));
            }
        }
        out
    }
}

#[cfg(test)]
#[path = "preprocess_tests.rs"]
mod tests;
