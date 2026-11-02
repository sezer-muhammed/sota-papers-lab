//! Per-iteration parameters and cameras as kernels see them, the mapped
//! pinned ring that feeds them, the status mirror and CUDA-graph capture.
//! Owner: trainer (E7). Spec: spec-raster 1.2-1.3, 1.5, 8; spec-train 2.1, 12.
//!
//! No kernel argument changes between iterations: the host writes iteration
//! t's [`IterParams`] into ring slot `t % RING_SLOTS`, `frame_begin` copies it
//! into [`IterParamsDev`] and composes the [`FrameCam`], and every other kernel
//! reads those device copies. One captured graph therefore replays forever
//! (until n, the render size, the SH degree or the step phase changes).
//!
//! The `#[repr(C)]` structs mirror `cuda/splat_common.cuh` field for field;
//! sizes are asserted at compile time here and on the device by the selftest
//! layout probe.

use std::mem::size_of;
use std::sync::Arc;

use spl_core::Result;
use spl_gpu::cudarc::driver::{CudaGraph, CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits};

use crate::gpu::{CudaResultExt, Gpu, not_implemented};
use crate::io::scene::View;
use crate::kernels::adam::AdamIterPlan;
use crate::kernels::layout::{
    RING_SLOTS, ST_ITER, ST_N_CHAINS, ST_N_INST, ST_N_VIS, ST_OVERFLOW, ST_OVERFLOW_ITER,
    STATUS_LEN,
};

/// Base pose of one image (never modified on device). 64 B.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CamBase {
    /// World to camera rotation, row-major.
    pub r: [f32; 9],
    pub t: [f32; 3],
    pub intr_id: u32,
    /// Size the intrinsics refer to (the cached image).
    pub width: u32,
    pub height: u32,
    pub pad: u32,
}

/// Intrinsics of one physical camera at its images' size. 16 B.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Intr {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
}

/// The raster's per-iteration header (spec-raster 1.3). 64 B.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FrameParams {
    pub iteration: u32,
    pub cam_index: u32,
    pub n_gaussians: u32,
    pub sh_degree: u32,
    pub bg: [f32; 3],
    pub eps2d: f32,
    /// `FLAG_*` bits of `kernels::layout`.
    pub flags: u32,
    pub adam_lr_scale: f32,
    pub bias_c1: f32,
    pub bias_c2: f32,
    /// Render size; frame_begin scales the intrinsics per axis from the CamBase size.
    pub render_w: u32,
    pub render_h: u32,
    /// FastGS compact box (spec-train 6.1): bound k = box_beta·2·ln(255·o); 1.0 = off.
    pub box_beta: f32,
    pub pad: u32,
}

/// The camera of this iteration, composed on the device by `frame_begin`. 128 B.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FrameCam {
    pub r: [f32; 9],
    pub t: [f32; 3],
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub campos: [f32; 3],
    pub near_z: f32,
    pub far_z: f32,
    pub lim_x0: f32,
    pub lim_x1: f32,
    pub lim_y0: f32,
    pub lim_y1: f32,
    pub width: u32,
    pub height: u32,
    pub grid_w: u32,
    pub grid_h: u32,
    pub n_tiles: u32,
    pub image_index: u32,
    pub intr_id: u32,
}

/// One ring slot: the raster header plus what training kernels need. 256 B.
/// (spec-train's `IterParams.camera: CameraGpu` became `frame.cam_index` plus
/// the device-composed [`FrameCam`]: the raster spec's names win.)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct IterParams {
    pub frame: FrameParams,
    /// Byte offset of the target image in the u8 pool (`io::images::ImageCache`).
    pub target_offset: u64,
    pub target_w: u32,
    pub target_h: u32,
    /// Integer warm-up scale; 1 = full resolution.
    pub res_scale: u32,
    pub means_lr: f32,
    /// MCMC position noise factor (0 = off).
    pub noise_scale: f32,
    /// Philox key of the run.
    pub seed: [u32; 2],
    pub pad0: u32,
    pub adam: AdamIterPlan,
    pub pad1: [u32; 17],
}

const _: () = {
    assert!(size_of::<CamBase>() == 64);
    assert!(size_of::<Intr>() == 16);
    assert!(size_of::<FrameParams>() == 64);
    assert!(size_of::<FrameCam>() == 128);
    assert!(size_of::<IterParams>() == 256);
};

// SAFETY: plain #[repr(C)] structs of f32/u32/u64 with no padding the device
// could misread (sizes asserted above, offsets checked by the layout probe);
// all-zero bits are a valid value of each.
unsafe impl DeviceRepr for CamBase {}
unsafe impl DeviceRepr for Intr {}
unsafe impl DeviceRepr for FrameParams {}
unsafe impl DeviceRepr for FrameCam {}
unsafe impl DeviceRepr for IterParams {}
unsafe impl ValidAsZeroBits for CamBase {}
unsafe impl ValidAsZeroBits for Intr {}
unsafe impl ValidAsZeroBits for FrameParams {}
unsafe impl ValidAsZeroBits for FrameCam {}
unsafe impl ValidAsZeroBits for IterParams {}

/// Base poses and intrinsics of a scene, uploaded once (spec-raster 1.2).
pub struct CameraTable {
    pub base: CudaSlice<CamBase>,
    pub intr: CudaSlice<Intr>,
    /// Twist (rho, phi) per image, optimized when pose refinement is on (image 0 frozen).
    pub pose_delta: CudaSlice<f32>,
    pub n_images: u32,
    pub n_intr: u32,
}

impl CameraTable {
    /// One [`CamBase`] per view and one [`Intr`] per distinct (COLMAP camera,
    /// image size), in first-seen order.
    pub fn pack(views: &[View]) -> (Vec<CamBase>, Vec<Intr>) {
        let mut keys: Vec<(u32, u32, u32)> = Vec::new();
        let mut intr = Vec::new();
        let mut base = Vec::with_capacity(views.len());
        for v in views {
            let key = (v.camera, v.width, v.height);
            let id = match keys.iter().position(|k| *k == key) {
                Some(i) => i,
                None => {
                    keys.push(key);
                    intr.push(Intr {
                        fx: v.fx as f32,
                        fy: v.fy as f32,
                        cx: v.cx as f32,
                        cy: v.cy as f32,
                    });
                    keys.len() - 1
                }
            };
            let mut r = [0f32; 9];
            for (i, row) in v.r_cw.iter().enumerate() {
                for (j, x) in row.iter().enumerate() {
                    r[i * 3 + j] = *x as f32;
                }
            }
            base.push(CamBase {
                r,
                t: v.t_cw.map(|x| x as f32),
                intr_id: id as u32,
                width: v.width,
                height: v.height,
                pad: 0,
            });
        }
        (base, intr)
    }

    pub fn upload(gpu: &Gpu, views: &[View]) -> Result<Self> {
        let (base, intr) = Self::pack(views);
        let s = &gpu.stream;
        Ok(Self {
            base: s.clone_htod(&base).cuda("upload camera table")?,
            intr: s.clone_htod(&intr).cuda("upload intrinsics")?,
            pose_delta: s
                .alloc_zeros::<f32>(views.len().max(1) * 6)
                .cuda("alloc pose deltas")?,
            n_images: views.len() as u32,
            n_intr: intr.len() as u32,
        })
    }
}

/// Device copies every kernel reads instead of arguments: [`IterParams`], the
/// composed [`FrameCam`], the status words and the iteration counter.
pub struct IterParamsDev {
    pub params: CudaSlice<IterParams>,
    pub cam: CudaSlice<FrameCam>,
    /// `STATUS_LEN` words, indices `ST_*` of `kernels::layout`.
    pub status: CudaSlice<u32>,
    pub iter_counter: CudaSlice<u32>,
}

impl IterParamsDev {
    pub fn alloc(gpu: &Gpu) -> Result<Self> {
        let s = &gpu.stream;
        Ok(Self {
            params: s.alloc_zeros(1).cuda("alloc IterParams")?,
            cam: s.alloc_zeros(1).cuda("alloc FrameCam")?,
            status: s.alloc_zeros(STATUS_LEN).cuda("alloc status")?,
            iter_counter: s.alloc_zeros(1).cuda("alloc iteration counter")?,
        })
    }

    /// Eager (non-ring) upload, for kernel tests and eval renders outside the
    /// training graph. `cam` is optional: frame_begin normally composes it.
    pub fn upload(&mut self, gpu: &Gpu, params: &IterParams, cam: Option<&FrameCam>) -> Result<()> {
        let s = &gpu.stream;
        s.memcpy_htod(std::slice::from_ref(params), &mut self.params)
            .cuda("upload IterParams")?;
        if let Some(cam) = cam {
            s.memcpy_htod(std::slice::from_ref(cam), &mut self.cam)
                .cuda("upload FrameCam")?;
        }
        Ok(())
    }

    /// Synchronous read of the status words (tests and debugging; the trainer
    /// reads the lagging [`PinnedStatus`] mirror instead).
    pub fn read_status(&self, gpu: &Gpu) -> Result<Status> {
        let words = gpu
            .stream
            .clone_dtoh(&self.status)
            .cuda("download status")?;
        Ok(Status::from_words(&words))
    }
}

/// The status words by name.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Status {
    /// `OVF_*` bits; sticky until the host clears it.
    pub overflow: u32,
    pub n_visible: u32,
    pub n_instances: u32,
    pub n_chains: u32,
    pub iteration: u32,
    pub overflow_iteration: u32,
}

impl Status {
    pub fn from_words(w: &[u32]) -> Self {
        let at = |i: usize| w.get(i).copied().unwrap_or(0);
        Self {
            overflow: at(ST_OVERFLOW),
            n_visible: at(ST_N_VIS),
            n_instances: at(ST_N_INST),
            n_chains: at(ST_N_CHAINS),
            iteration: at(ST_ITER),
            overflow_iteration: at(ST_OVERFLOW_ITER),
        }
    }
}

/// Host-readable mirror of the status words in mapped pinned memory, written
/// by `frame_end` and read one iteration late (spec-raster 1.5). STUB (E7):
/// needs `alloc_pinned_with_flags(CU_MEMHOSTALLOC_DEVICEMAP)` plus
/// `cuMemHostGetDevicePointer` through `cudarc::driver::sys`.
pub struct PinnedStatus {
    pub device_ptr: u64,
}

impl PinnedStatus {
    pub fn new(_gpu: &Gpu) -> Result<Self> {
        Err(not_implemented("PinnedStatus::new", "trainer (E7)"))
    }

    pub fn read(&self) -> Result<Status> {
        Err(not_implemented("PinnedStatus::read", "trainer (E7)"))
    }
}

/// Mapped pinned ring of [`IterParams`] slots (spec-raster 1.3). The host
/// never overwrites a slot before iteration t − RING_SLOTS + 1 has completed.
/// STUB (E7).
pub struct FrameRing {
    pub slots: usize,
    /// Device pointer of slot 0 (mapped pinned memory).
    pub device_ptr: u64,
    /// Iterations pushed so far.
    pub pushed: u64,
}

impl FrameRing {
    pub const DEFAULT_SLOTS: usize = RING_SLOTS as usize;

    pub fn new(_gpu: &Gpu, _slots: usize) -> Result<Self> {
        Err(not_implemented("FrameRing::new", "trainer (E7)"))
    }

    /// Writes iteration `it`'s parameters; blocks only while the slot is still in flight.
    pub fn push(&mut self, _it: &IterParams) -> Result<()> {
        Err(not_implemented("FrameRing::push", "trainer (E7)"))
    }
}

/// One training iteration as a captured CUDA graph (spec-raster 8, E7).
/// Capture with `CU_STREAM_CAPTURE_MODE_THREAD_LOCAL` and event tracking off
/// (`unsafe { ctx.disable_event_tracking() }`), else cudarc records per-slice
/// events into the graph. STUB.
pub struct IterationGraph {
    pub graph: Option<CudaGraph>,
    /// What the graph was captured for: any change re-captures.
    pub key: GraphKey,
}

/// spec-train 12: `(n, scale, degree, plan shape, stats, abs_grad)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct GraphKey {
    pub n_gaussians: u32,
    pub render_w: u32,
    pub render_h: u32,
    pub sh_degree: u32,
    pub due_mask: u32,
    pub stats: bool,
    pub abs_grad: bool,
}

impl IterationGraph {
    pub fn capture(
        _s: &Arc<CudaStream>,
        _key: GraphKey,
        _record: impl FnOnce(&CudaStream) -> Result<()>,
    ) -> Result<Self> {
        Err(not_implemented("IterationGraph::capture", "trainer (E7)"))
    }

    pub fn replay(&self) -> Result<()> {
        Err(not_implemented("IterationGraph::replay", "trainer (E7)"))
    }
}

#[cfg(test)]
mod tests {
    use std::mem::offset_of;

    use super::*;

    #[test]
    fn offsets_follow_the_cuda_layout() {
        assert_eq!(offset_of!(FrameParams, bg), 16);
        assert_eq!(offset_of!(FrameParams, render_w), 48);
        assert_eq!(offset_of!(FrameCam, campos), 64);
        assert_eq!(offset_of!(FrameCam, intr_id), 124);
        assert_eq!(offset_of!(IterParams, target_offset), 64);
        assert_eq!(offset_of!(IterParams, seed), 92);
        assert_eq!(offset_of!(IterParams, adam), 104);
    }

    #[test]
    fn camera_table_shares_intrinsics_per_camera_and_size() {
        let view = |camera, width| View {
            name: format!("{camera}-{width}"),
            camera,
            r_cw: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            t_cw: [1.0, 2.0, 3.0],
            fx: 100.0,
            fy: 101.0,
            cx: 50.0,
            cy: 40.0,
            width,
            height: 80,
            image_path: Default::default(),
        };
        let (base, intr) =
            CameraTable::pack(&[view(1, 100), view(1, 100), view(2, 100), view(1, 50)]);
        assert_eq!(
            base.iter().map(|b| b.intr_id).collect::<Vec<_>>(),
            [0, 0, 1, 2]
        );
        assert_eq!(intr.len(), 3);
        assert_eq!(base[0].r[4], 1.0);
        assert_eq!(base[0].t, [1.0, 2.0, 3.0]);
        assert_eq!(intr[0].fy, 101.0);
    }

    #[test]
    fn status_reads_named_words() {
        let mut w = [0u32; STATUS_LEN];
        w[ST_OVERFLOW] = 2;
        w[ST_N_INST] = 99;
        let s = Status::from_words(&w);
        assert_eq!((s.overflow, s.n_instances, s.n_visible), (2, 99, 0));
    }
}
