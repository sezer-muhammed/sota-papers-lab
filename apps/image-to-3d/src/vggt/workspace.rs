//! Device buffers for one (frames, resolution) and the memory plan. Owner:
//! runtime.
//!
//! The trunk and the heads run one after the other and share nothing but
//! the cached layers: `cache[k]` is bf16 `[N·T][2048]` = `[F_i | G_i]` for
//! `i = 4, 11, 17, 23` (spec-vggt.md section 3.4). The trunk writes them; the
//! camera head reads `cache[3]`; the dense head reads all four, 8 frames at
//! a time. Activations inside the trunk are reused per layer (the trunk sizes
//! them; [`Scratch`] is the shared region a component can carve instead of
//! allocating).
//!
//! For large N the first three levels (3 × 2.8 MB per frame at 512×336)
//! leave the device after the trunk and come back per dense chunk from
//! pinned host memory ([`Plan::offload`]); the trunk still writes them on
//! the device, so the plan first checks that they fit there.

use std::ops::Range;

use anyhow::ensure;
use spl_core::{Result, anyhow};
use spl_nn::Nn;
use spl_nn::cudarc::driver::{CudaSlice, CudaView, CudaViewMut, PinnedHostSlice};

pub const SPECIAL: usize = 17;
pub const DIM: usize = 1024;
pub const CACHE_DIM: usize = 2 * DIM;
/// Aggregator layers whose `[F | G]` the heads read.
pub const CACHED_LAYERS: [usize; 4] = [4, 11, 17, 23];

/// Frames and resolution of one run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Geometry {
    pub frames: usize,
    pub height: usize,
    pub width: usize,
    pub grid_h: usize,
    pub grid_w: usize,
}

impl Geometry {
    pub fn new(frames: usize, height: usize, width: usize) -> Self {
        Self {
            frames,
            height,
            width,
            grid_h: height / 16,
            grid_w: width / 16,
        }
    }

    pub fn patches(&self) -> usize {
        self.grid_h * self.grid_w
    }

    /// Tokens per frame: camera, 16 registers, patches.
    pub fn tokens(&self) -> usize {
        SPECIAL + self.patches()
    }

    pub fn pixels(&self) -> usize {
        self.height * self.width
    }

    /// The same resolution with another frame count (dense-head chunks).
    pub fn with_frames(&self, frames: usize) -> Self {
        Self { frames, ..*self }
    }

    /// Bytes of one cached level, `[N·T][2048]` bf16.
    pub fn cache_level_bytes(&self) -> usize {
        self.frames * self.tokens() * CACHE_DIM * 2
    }
}

/// Shared device bytes, carved into disjoint typed views for one call.
pub struct Scratch {
    buf: CudaSlice<u8>,
}

impl Scratch {
    pub fn new(nn: &Nn, bytes: usize) -> Result<Self> {
        let buf = nn
            .stream
            .alloc_zeros::<u8>(bytes.max(256))
            .map_err(|e| anyhow!("allocate {bytes} B of scratch: {e:?}"))?;
        Ok(Self { buf })
    }

    pub fn bytes(&self) -> usize {
        self.buf.len()
    }

    pub fn carve(&mut self) -> Carver<'_> {
        Carver {
            rest: Some(self.buf.slice_mut(..)),
        }
    }
}

/// Hands out 256-byte-aligned, non-overlapping views of a [`Scratch`].
pub struct Carver<'a> {
    rest: Option<CudaViewMut<'a, u8>>,
}

impl<'a> Carver<'a> {
    pub fn take<T>(&mut self, len: usize) -> Result<CudaViewMut<'a, T>> {
        let bytes = (len * std::mem::size_of::<T>()).next_multiple_of(256);
        let rest = self
            .rest
            .take()
            .ok_or_else(|| anyhow!("scratch exhausted"))?;
        ensure!(
            rest.len() >= bytes,
            "scratch exhausted: wanted {bytes} B, {} B left",
            rest.len()
        );
        let (head, tail) = rest.split_at_mut_owned(bytes);
        self.rest = Some(tail);
        // SAFETY: `head` holds at least `len` elements of `T` and starts on
        // a 256-byte boundary (the allocation is aligned, every cut is a
        // multiple of 256).
        unsafe { head.transmute_owned::<T>(len) }.ok_or_else(|| anyhow!("scratch view"))
    }
}

/// Owned split/transmute for `CudaViewMut` (cudarc only splits through a
/// reborrow, which would tie every piece to the carver's borrow).
trait OwnedSplit<'a>: Sized {
    fn split_at_mut_owned(self, mid: usize) -> (CudaViewMut<'a, u8>, CudaViewMut<'a, u8>);
    unsafe fn transmute_owned<T>(self, len: usize) -> Option<CudaViewMut<'a, T>>;
}

impl<'a> OwnedSplit<'a> for CudaViewMut<'a, u8> {
    fn split_at_mut_owned(mut self, mid: usize) -> (CudaViewMut<'a, u8>, CudaViewMut<'a, u8>) {
        let (a, b) = self.split_at_mut(mid);
        // SAFETY: `a` and `b` are disjoint parts of `self`, which lives for
        // 'a in the scratch buffer; extending the reborrow to 'a is sound
        // because `self` is consumed here and never used again.
        unsafe {
            (
                std::mem::transmute::<CudaViewMut<'_, u8>, CudaViewMut<'a, u8>>(a),
                std::mem::transmute::<CudaViewMut<'_, u8>, CudaViewMut<'a, u8>>(b),
            )
        }
    }

    unsafe fn transmute_owned<T>(mut self, len: usize) -> Option<CudaViewMut<'a, T>> {
        // SAFETY: as above; the caller vouches for the element type.
        unsafe {
            self.transmute_mut::<T>(len)
                .map(|v| std::mem::transmute::<CudaViewMut<'_, T>, CudaViewMut<'a, T>>(v))
        }
    }
}

/// Where the cached levels live between the trunk and the dense head.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plan {
    pub geom: Geometry,
    /// Frames per dense-head chunk.
    pub dense_chunk: usize,
    /// Levels 0..3 go to pinned host memory after the trunk.
    pub offload: bool,
    /// Estimated device bytes of this workspace (without weights and the
    /// components' own buffers).
    pub device_bytes: usize,
}

impl Plan {
    /// `free` = device bytes available for the workspace after the weights
    /// and the components' buffers (`trunk_bytes`, `head_bytes`).
    pub fn new(
        geom: Geometry,
        dense_chunk: usize,
        free: usize,
        trunk_bytes: usize,
        head_bytes: usize,
    ) -> Result<Self> {
        let base = geom.frames * geom.pixels() * (3 + 3 * 4 + 2 * 2);
        let cache = 4 * geom.cache_level_bytes();
        let need = base + cache + trunk_bytes.max(head_bytes);
        ensure!(
            need <= free,
            "{} frames at {}x{} need {:.2} GB of device memory, {:.2} GB are free",
            geom.frames,
            geom.width,
            geom.height,
            need as f64 / 1e9,
            free as f64 / 1e9
        );
        // The dense head's buffers come on top of the cache; offloading the
        // first three levels frees room for them when the margin is thin.
        let offload = base + cache + trunk_bytes.max(head_bytes) + head_bytes > free;
        Ok(Self {
            geom,
            dense_chunk: dense_chunk.max(1),
            offload,
            device_bytes: need,
        })
    }

    pub fn chunks(&self) -> impl Iterator<Item = Range<usize>> + '_ {
        (0..self.geom.frames)
            .step_by(self.dense_chunk)
            .map(|s| s..(s + self.dense_chunk).min(self.geom.frames))
    }
}

/// Buffers of one geometry, kept between runs of the same geometry.
pub struct Workspace {
    pub geom: Geometry,
    /// RGB8 `[N][H][W][3]` upload target.
    pub pixels: CudaSlice<u8>,
    /// Normalised images, f32 `[N][3][H][W]`.
    pub images: CudaSlice<f32>,
    /// `None` while a level is offloaded (dropped from the device).
    pub cache: [Option<CudaSlice<u16>>; 4],
    /// Pinned host copies of levels 0..3 when offloading.
    pub host_cache: Option<[PinnedHostSlice<u16>; 3]>,
    /// Device staging for one dense chunk of an offloaded level.
    pub stage: Option<[CudaSlice<u16>; 3]>,
    /// f16 bits `[N][H][W]`.
    pub depth: CudaSlice<u16>,
    pub conf: CudaSlice<u16>,
    /// World points f32 `[N][H][W][3]`.
    pub world: CudaSlice<f32>,
    /// f32 `[N · 17][2048]` = `[F_23 | G_23]` of the special tokens: the
    /// camera head's input without the bf16 rounding of the cache.
    pub cam_tokens: CudaSlice<f32>,
    pub plan: Plan,
}

impl Workspace {
    pub fn new(nn: &Nn, plan: Plan) -> Result<Self> {
        let g = plan.geom;
        let s = &nn.stream;
        let alloc_err = |what: &str| {
            let what = what.to_string();
            move |e| anyhow!("allocate {what}: {e:?}")
        };
        let level = g.frames * g.tokens() * CACHE_DIM;
        let cache = [0, 1, 2, 3].map(|_| s.alloc_zeros::<u16>(level));
        let [c0, c1, c2, c3] = cache;
        let cache = [
            Some(c0.map_err(alloc_err("cache"))?),
            Some(c1.map_err(alloc_err("cache"))?),
            Some(c2.map_err(alloc_err("cache"))?),
            Some(c3.map_err(alloc_err("cache"))?),
        ];
        let (host_cache, stage) = if plan.offload {
            let chunk = plan.dense_chunk.min(g.frames) * g.tokens() * CACHE_DIM;
            let mut host = Vec::new();
            let mut st = Vec::new();
            for _ in 0..3 {
                // SAFETY: pinned memory is written by the device before any
                // host read (copies are stream-ordered and synchronised).
                host.push(
                    unsafe { nn.ctx.alloc_pinned::<u16>(level) }
                        .map_err(alloc_err("pinned cache"))?,
                );
                st.push(
                    s.alloc_zeros::<u16>(chunk)
                        .map_err(alloc_err("cache stage"))?,
                );
            }
            let h: [PinnedHostSlice<u16>; 3] = host.try_into().map_err(|_| anyhow!("pinned"))?;
            let st: [CudaSlice<u16>; 3] = st.try_into().map_err(|_| anyhow!("stage"))?;
            (Some(h), Some(st))
        } else {
            (None, None)
        };
        Ok(Self {
            geom: g,
            pixels: s
                .alloc_zeros(g.frames * g.pixels() * 3)
                .map_err(alloc_err("pixels"))?,
            images: s
                .alloc_zeros(g.frames * g.pixels() * 3)
                .map_err(alloc_err("images"))?,
            cache,
            host_cache,
            stage,
            depth: s
                .alloc_zeros(g.frames * g.pixels())
                .map_err(alloc_err("depth"))?,
            conf: s
                .alloc_zeros(g.frames * g.pixels())
                .map_err(alloc_err("conf"))?,
            world: s
                .alloc_zeros(g.frames * g.pixels() * 3)
                .map_err(alloc_err("world"))?,
            cam_tokens: s
                .alloc_zeros(g.frames * crate::vggt::trunk::SPECIAL * CACHE_DIM)
                .map_err(alloc_err("camera tokens"))?,
            plan,
        })
    }

    /// Re-allocates levels an earlier run offloaded.
    pub fn restore_cache(&mut self, nn: &Nn) -> Result<()> {
        let level = self.geom.frames * self.geom.tokens() * CACHE_DIM;
        for c in &mut self.cache {
            if c.is_none() {
                *c = Some(
                    nn.stream
                        .alloc_zeros::<u16>(level)
                        .map_err(|e| anyhow!("allocate cache: {e:?}"))?,
                );
            }
        }
        Ok(())
    }

    /// A level that is on the device.
    pub fn level(&self, k: usize) -> Result<CudaView<'_, u16>> {
        Ok(self.cache[k]
            .as_ref()
            .ok_or_else(|| anyhow!("cache level {k} offloaded"))?
            .slice(..))
    }

    /// Moves levels 0..3 to pinned host memory and frees them on the device
    /// (when the plan offloads).
    pub fn offload_levels(&mut self, nn: &Nn) -> Result<()> {
        let Some(host) = self.host_cache.as_mut() else {
            return Ok(());
        };
        for (h, d) in host.iter_mut().zip(&self.cache[..3]) {
            if let Some(d) = d {
                nn.stream
                    .memcpy_dtoh(d, h)
                    .map_err(|e| anyhow!("offload cache: {e:?}"))?;
            }
        }
        nn.sync()?;
        for c in &mut self.cache[..3] {
            *c = None;
        }
        Ok(())
    }
}

/// All four levels of `cache`, for the trunk to write.
pub fn cache_mut(cache: &mut [Option<CudaSlice<u16>>; 4]) -> Result<[CudaViewMut<'_, u16>; 4]> {
    fn view(s: &mut Option<CudaSlice<u16>>) -> Result<CudaViewMut<'_, u16>> {
        Ok(s.as_mut()
            .ok_or_else(|| anyhow!("cache level offloaded"))?
            .slice_mut(..))
    }
    let [a, b, c, d] = cache;
    Ok([view(a)?, view(b)?, view(c)?, view(d)?])
}
