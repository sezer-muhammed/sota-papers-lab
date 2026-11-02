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

use std::ffi::c_void;
use std::mem::size_of;
use std::sync::Arc;

use spl_core::Result;
use spl_gpu::cudarc::driver::{
    CudaContext, CudaEvent, CudaGraph, CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits, sys,
};

use crate::gpu::{CudaResultExt, Gpu};
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

/// `count` values of `T` in mapped pinned host memory (`cuMemHostAlloc` with
/// `DEVICEMAP`): the host writes or reads them directly, kernels through
/// `device_ptr`. Zeroed on allocation.
struct MappedPinned<T> {
    host: *mut T,
    device_ptr: u64,
    count: usize,
    ctx: Arc<CudaContext>,
}

// SAFETY: the allocation is plain host memory owned by this value; access is
// serialized by `&mut self` (writes) and the stream order (device side).
unsafe impl<T: Send> Send for MappedPinned<T> {}

impl<T: Copy + ValidAsZeroBits> MappedPinned<T> {
    fn new(gpu: &Gpu, count: usize, what: &str) -> Result<Self> {
        let bytes = size_of::<T>() * count.max(1);
        gpu.ctx.bind_to_thread().cuda(what)?;
        let mut host: *mut c_void = std::ptr::null_mut();
        // SAFETY: plain driver allocation into a local out-pointer.
        unsafe { sys::cuMemHostAlloc(&mut host, bytes, sys::CU_MEMHOSTALLOC_DEVICEMAP) }
            .result()
            .cuda(what)?;
        let mut device_ptr: sys::CUdeviceptr = 0;
        // SAFETY: `host` was just allocated with DEVICEMAP in this context.
        let mapped = unsafe { sys::cuMemHostGetDevicePointer_v2(&mut device_ptr, host, 0) }
            .result()
            .cuda(what);
        if let Err(e) = mapped {
            // SAFETY: frees the allocation made above, which nothing else references.
            unsafe { sys::cuMemFreeHost(host) };
            return Err(e);
        }
        // SAFETY: `host` holds `bytes` writable bytes; all-zero bits are a valid T.
        unsafe { std::ptr::write_bytes(host as *mut u8, 0, bytes) };
        Ok(Self {
            host: host as *mut T,
            device_ptr,
            count: count.max(1),
            ctx: gpu.ctx.clone(),
        })
    }

    fn read(&self, i: usize) -> T {
        assert!(i < self.count);
        // SAFETY: in bounds; volatile because the device writes it behind our back.
        unsafe { std::ptr::read_volatile(self.host.add(i)) }
    }

    fn write(&mut self, i: usize, v: T) {
        assert!(i < self.count);
        // SAFETY: in bounds; the caller guarantees no kernel reads slot i now.
        unsafe { std::ptr::write_volatile(self.host.add(i), v) };
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
    }
}

impl<T> Drop for MappedPinned<T> {
    fn drop(&mut self) {
        // Pending kernels may still read or write the memory.
        let _ = self.ctx.synchronize();
        let _ = self.ctx.bind_to_thread();
        // SAFETY: allocated by cuMemHostAlloc in `new`, freed once.
        unsafe { sys::cuMemFreeHost(self.host as *mut c_void) };
    }
}

/// Host-readable mirror of the status words in mapped pinned memory, written
/// by `frame_end` and read one iteration late (spec-raster 1.5).
pub struct PinnedStatus {
    pub device_ptr: u64,
    words: MappedPinned<u32>,
}

impl PinnedStatus {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let words = MappedPinned::new(gpu, STATUS_LEN, "alloc pinned status mirror")?;
        Ok(Self {
            device_ptr: words.device_ptr,
            words,
        })
    }

    /// The status of the last finished iteration (no sync).
    pub fn read(&self) -> Result<Status> {
        let w: Vec<u32> = (0..STATUS_LEN).map(|i| self.words.read(i)).collect();
        Ok(Status::from_words(&w))
    }

    /// Clears the host copy (after the trainer handled an overflow).
    pub fn clear(&mut self) {
        for i in 0..STATUS_LEN {
            self.words.write(i, 0);
        }
    }
}

/// Mapped pinned ring of [`IterParams`] slots (spec-raster 1.3). The host
/// never overwrites a slot before iteration t − RING_SLOTS + 1 has completed:
/// [`FrameRing::enqueued`] records an event per slot and [`FrameRing::push`]
/// waits on it.
pub struct FrameRing {
    pub slots: usize,
    /// Device pointer of slot 0 (mapped pinned memory).
    pub device_ptr: u64,
    /// Iterations pushed so far.
    pub pushed: u64,
    mem: MappedPinned<IterParams>,
    done: Vec<CudaEvent>,
}

impl FrameRing {
    pub const DEFAULT_SLOTS: usize = RING_SLOTS as usize;

    pub fn new(gpu: &Gpu, slots: usize) -> Result<Self> {
        let mem = MappedPinned::new(gpu, slots, "alloc pinned frame ring")?;
        let done = (0..slots)
            .map(|_| gpu.ctx.new_event(None).cuda("create ring event"))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            slots,
            device_ptr: mem.device_ptr,
            pushed: 0,
            mem,
            done,
        })
    }

    /// Writes the next iteration's parameters; blocks only while the slot is still in flight.
    pub fn push(&mut self, it: &IterParams) -> Result<()> {
        let slot = (self.pushed % self.slots as u64) as usize;
        self.done[slot].synchronize().cuda("wait for ring slot")?;
        self.mem.write(slot, *it);
        self.pushed += 1;
        Ok(())
    }

    /// Marks the last pushed slot as in flight until `s` reaches this point.
    pub fn enqueued(&self, s: &CudaStream) -> Result<()> {
        let slot = ((self.pushed + self.slots as u64 - 1) % self.slots as u64) as usize;
        self.done[slot].record(s).cuda("record ring event")
    }
}

/// One training iteration as a captured CUDA graph (spec-raster 8, E7).
/// Captured with `CU_STREAM_CAPTURE_MODE_THREAD_LOCAL` and cudarc's event
/// tracking off for the capture (else it records per-slice events into the graph).
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
    /// Records `record` on `s` into a graph. Nothing inside may sync, allocate
    /// or copy from pageable memory.
    pub fn capture(
        s: &Arc<CudaStream>,
        key: GraphKey,
        record: impl FnOnce(&CudaStream) -> Result<()>,
    ) -> Result<Self> {
        let ctx = s.context().clone();
        let tracking = ctx.is_event_tracking();
        // SAFETY: one stream does all of the trainer's work, so stream order
        // alone keeps every buffer consistent while tracking is off.
        unsafe { ctx.disable_event_tracking() };
        let begun = s
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .cuda("begin graph capture");
        let recorded = begun.and_then(|_| record(s));
        let ended = s
            .end_capture(sys::CUgraphInstantiate_flags(0))
            .cuda("end graph capture");
        if tracking {
            // SAFETY: restores the context's previous mode.
            unsafe { ctx.enable_event_tracking() };
        }
        recorded?;
        let graph = ended?;
        if let Some(g) = &graph {
            g.upload().cuda("upload graph")?;
        }
        Ok(Self { graph, key })
    }

    pub fn replay(&self) -> Result<()> {
        match &self.graph {
            Some(g) => g.launch().cuda("launch iteration graph"),
            None => Ok(()),
        }
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
