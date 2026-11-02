//! The splat kernels and the buffers they share. Owner of this file: preprocess
//! (E1), reviewed by every kernel owner; each submodule has its own owner.
//!
//! Spec: reports/image-to-3d/spec-raster.md Section 8 (types, stage functions,
//! buffer table 1.4) and spec-train.md Section 2.1 (the raster boundary). Where
//! the two specs name a raster-owned thing differently, the raster names win;
//! DESIGN.md lists every reconciliation.
//!
//! Stage functions (one file each) enqueue work on one stream, never
//! synchronize, never allocate, and read counts from the device. Order inside a
//! training iteration:
//!
//! `frame_begin → preprocess_fwd → sort_depth → bin_tiles → raster_fwd(Train)
//! → loss → raster_bwd → preprocess_bwd(sink) → [cam_reduce, adam] → frame_end`

pub mod adam;
pub mod densify;
pub mod layout;
pub mod loss;
pub mod preprocess;
pub mod raster;
#[cfg(test)]
mod selftest;
pub mod sort;

use std::mem::size_of;

use anyhow::ensure;
use spl_core::Result;
use spl_gpu::cudarc::driver::{CudaSlice, DeviceRepr, ValidAsZeroBits};

use crate::frame::IterParamsDev;
use crate::gpu::{CudaResultExt, Defines, Gpu, KernelSource};
use layout::{BUCKET, MAX_TILES, TILE_W};

/// One visible Gaussian as binning and raster read it (spec-raster 1.4). 48 B.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SplatRec {
    pub mx: f32,
    pub my: f32,
    /// Conic pre-scaled for exp2: (−½·log2e·A, −log2e·B, −½·log2e·C).
    pub ca: f32,
    pub cb: f32,
    pub cc: f32,
    /// Effective opacity sigmoid(ô)·c2·c3.
    pub o: f32,
    pub r: f32,
    pub g: f32,
    pub b: f32,
    /// log2(255·o) ≥ 0.
    pub pthr: f32,
    pub z: f32,
    /// Bit c: colour channel c was clamped (its gradient is 0).
    pub flags: u32,
}

const _: () = assert!(size_of::<SplatRec>() == 48);
// SAFETY: #[repr(C)] f32/u32 fields, no padding (size asserted); zero is valid.
unsafe impl DeviceRepr for SplatRec {}
unsafe impl ValidAsZeroBits for SplatRec {}

/// Blend semantics (spec-raster 0): gsplat's (oracle parity) or Faster-GS's.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum BlendSemantics {
    #[default]
    Gsplat,
    FasterGs,
}

/// Compile-time kernel variant; one NVRTC module per distinct value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RasterVariant {
    /// 16 (default) or 12.
    pub tile_h: u8,
    /// Buckets per chain in the backward: 1, 2, 4 (default), 8.
    pub chain_k: u8,
    pub ckpt_f16: bool,
    /// 32 B fp16 record instead of 48 B fp32.
    pub rec32: bool,
    /// 0, 1 (expected depth), 4 (depth + normal).
    pub extra_ch: u8,
    pub blend: BlendSemantics,
    pub deterministic: bool,
}

impl Default for RasterVariant {
    fn default() -> Self {
        Self {
            tile_h: 16,
            chain_k: 4,
            ckpt_f16: false,
            rec32: false,
            extra_ch: 0,
            blend: BlendSemantics::Gsplat,
            deterministic: false,
        }
    }
}

impl RasterVariant {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            matches!(self.tile_h, 12 | 16),
            "tile_h must be 12 or 16, got {}",
            self.tile_h
        );
        ensure!(
            matches!(self.chain_k, 1 | 2 | 4 | 8),
            "chain_k must be 1, 2, 4 or 8, got {}",
            self.chain_k
        );
        ensure!(
            matches!(self.extra_ch, 0 | 1 | 4),
            "extra_ch must be 0, 1 or 4, got {}",
            self.extra_ch
        );
        Ok(())
    }

    /// The `-D` defines of `cuda/splat_common.cuh`'s variant block.
    pub fn defines(&self) -> Defines {
        Defines::new()
            .set("TILE_H", self.tile_h)
            .set("CHAIN_K", self.chain_k)
            .set("CKPT_F16", u8::from(self.ckpt_f16))
            .set("REC32", u8::from(self.rec32))
            .set("EXTRA_CH", self.extra_ch)
            .set(
                "BLEND",
                match self.blend {
                    BlendSemantics::Gsplat => layout::BLEND_GSPLAT,
                    BlendSemantics::FasterGs => layout::BLEND_FASTERGS,
                },
            )
            .set("DETERMINISTIC", u8::from(self.deterministic))
    }

    /// Pixels per tile.
    pub fn tile_pixels(&self) -> u32 {
        TILE_W * self.tile_h as u32
    }

    /// Tiles of a `w × h` image.
    pub fn tiles(&self, w: u32, h: u32) -> u32 {
        w.div_ceil(TILE_W) * h.div_ceil(self.tile_h as u32)
    }
}

/// Fixed capacities of every index space (spec-raster 1.4-1.5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Capacity {
    /// N_cap: Gaussians.
    pub gaussians: u32,
    /// V_cap: visible Gaussians (default = N_cap).
    pub visible: u32,
    /// I_cap: (tile, Gaussian) instances.
    pub instances: u32,
    /// T_cap: tiles (≤ MAX_TILES).
    pub tiles: u32,
    /// P_cap: pixels of the largest render.
    pub pixels: u32,
}

impl Capacity {
    /// For renders up to `w × h` with `gaussians` capacity and `instances`
    /// (spec-raster 1.5: grow to 1.25x the reported need on overflow).
    pub fn new(gaussians: u32, instances: u32, w: u32, h: u32, v: &RasterVariant) -> Self {
        Self {
            gaussians,
            visible: gaussians,
            instances,
            tiles: v.tiles(w, h).min(MAX_TILES),
            pixels: w * h,
        }
    }

    /// C_cap: Σ_t ceil(ceil(n_t/32)/K) ≤ I/(32K) + T, so chains never overflow
    /// before instances do.
    pub fn chains(&self, v: &RasterVariant) -> u32 {
        self.instances / (BUCKET * v.chain_k as u32) + self.tiles
    }
}

/// Byte sizes of the per-frame groups of spec-raster 1.5's VRAM table.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RasterVram {
    /// vis_idx, rec, depth key/val ×2, n_tiles, inst_off, g2d (+ abs, extra).
    pub per_visible: u64,
    /// tile keys ×2 (u16), instance values ×2 (u32).
    pub per_instance: u64,
    pub checkpoints: u64,
    /// vis_mask, tiles, pixels (outputs and pixel gradients), status, scratch.
    pub other: u64,
}

impl RasterVram {
    pub fn total(&self) -> u64 {
        self.per_visible + self.per_instance + self.checkpoints + self.other
    }
}

/// Every per-frame buffer of spec-raster 1.4 except the outputs, pixel
/// gradients and g2d (which have their own structs, below).
pub struct FrameBuffers {
    pub cap: Capacity,
    pub variant: RasterVariant,
    /// Device IterParams, FrameCam, status words, iteration counter.
    pub iter: IterParamsDev,
    /// N space: bit p set when Gaussian p is visible this frame.
    pub vis_mask: CudaSlice<u32>,
    /// v → p, strictly increasing (order-preserving compaction).
    pub vis_idx: CudaSlice<u32>,
    pub rec: CudaSlice<SplatRec>,
    /// Ping-pong depth sort buffers; `depth_sorted` names the result.
    pub depth_key: [CudaSlice<u32>; 2],
    pub depth_val: [CudaSlice<u32>; 2],
    pub n_tiles: CudaSlice<u32>,
    /// By depth rank k, V_cap + 1.
    pub inst_off: CudaSlice<u32>,
    /// Ping-pong tile sort buffers; `tile_sorted` names the result.
    pub tile_key: [CudaSlice<u16>; 2],
    pub inst_v: [CudaSlice<u32>; 2],
    /// T_cap + 1 each.
    pub tile_hist: CudaSlice<u32>,
    pub tile_start: CudaSlice<u32>,
    pub chain_off: CudaSlice<u32>,
    pub tile_max_nc: CudaSlice<u32>,
    /// Both u16 radix passes' digit offsets, from bin_scan: [2][RADIX_BINS].
    pub tile_digit_offsets: CudaSlice<u32>,
    /// Chain checkpoints (C, T): [C_cap][tile pixels] float4, or half4 packed in
    /// two f32 words each with `ckpt_f16`.
    pub ckpt: CudaSlice<f32>,
    /// Persistent backward work counter.
    pub chain_ctr: CudaSlice<u32>,
    /// preprocess_fwd compaction: chunk counter and look-back state per 256-chunk.
    pub chunk_ctr: CudaSlice<u32>,
    pub chunk_lookback: CudaSlice<u32>,
    pub depth_sorted: usize,
    pub tile_sorted: usize,
}

impl FrameBuffers {
    pub fn alloc(gpu: &Gpu, cap: Capacity, v: &RasterVariant) -> Result<Self> {
        v.validate()?;
        let s = &gpu.stream;
        let n = cap.gaussians as usize;
        let vis = cap.visible as usize;
        let inst = cap.instances as usize;
        let tiles = cap.tiles as usize;
        let u32s = |len: usize, what: &str| s.alloc_zeros::<u32>(len.max(1)).cuda(what);
        let u16s = |len: usize, what: &str| s.alloc_zeros::<u16>(len.max(1)).cuda(what);
        Ok(Self {
            cap,
            variant: *v,
            iter: IterParamsDev::alloc(gpu)?,
            vis_mask: u32s(n.div_ceil(32), "alloc vis_mask")?,
            vis_idx: u32s(vis, "alloc vis_idx")?,
            rec: s.alloc_zeros::<SplatRec>(vis.max(1)).cuda("alloc rec")?,
            depth_key: [u32s(vis, "alloc depth_key")?, u32s(vis, "alloc depth_key")?],
            depth_val: [u32s(vis, "alloc depth_val")?, u32s(vis, "alloc depth_val")?],
            n_tiles: u32s(vis, "alloc n_tiles")?,
            inst_off: u32s(vis + 1, "alloc inst_off")?,
            tile_key: [u16s(inst, "alloc tile_key")?, u16s(inst, "alloc tile_key")?],
            inst_v: [u32s(inst, "alloc inst_v")?, u32s(inst, "alloc inst_v")?],
            tile_hist: u32s(tiles + 1, "alloc tile_hist")?,
            tile_start: u32s(tiles + 1, "alloc tile_start")?,
            chain_off: u32s(tiles + 1, "alloc chain_off")?,
            tile_max_nc: u32s(tiles, "alloc tile_max_nc")?,
            tile_digit_offsets: u32s(2 * layout::RADIX_BINS as usize, "alloc tile digit offsets")?,
            ckpt: s
                .alloc_zeros::<f32>((Self::ckpt_floats(&cap, v) as usize).max(1))
                .cuda("alloc checkpoints")?,
            chain_ctr: u32s(1, "alloc chain counter")?,
            chunk_ctr: u32s(1, "alloc chunk counter")?,
            chunk_lookback: u32s(n.div_ceil(256), "alloc chunk look-back")?,
            // 4 depth passes (32-bit keys) and 2 tile passes end in buffer 0.
            depth_sorted: 0,
            tile_sorted: 0,
        })
    }

    fn ckpt_floats(cap: &Capacity, v: &RasterVariant) -> u64 {
        let per_pixel = if v.ckpt_f16 { 2 } else { 4 };
        cap.chains(v) as u64 * v.tile_pixels() as u64 * per_pixel
    }

    /// Bytes of everything the raster allocates per frame for `cap` (this
    /// struct plus [`RenderOut`], [`PixelGrads`] and [`Grad2d`]), grouped as in
    /// spec-raster 1.5; for capacity planning.
    pub fn vram_bytes(cap: Capacity, v: &RasterVariant) -> RasterVram {
        let n = cap.gaussians as u64;
        let vis = cap.visible as u64;
        let inst = cap.instances as u64;
        let tiles = cap.tiles as u64;
        let e = v.extra_ch as u64;
        let g2d = 9 + 2 + e;
        RasterVram {
            per_visible: vis * (4 + 48 + 16 + 4 + 4 + 4 * g2d) + 4,
            per_instance: inst * (2 * 2 + 2 * 4),
            checkpoints: Self::ckpt_floats(&cap, v) * 4,
            other: n.div_ceil(32) * 4
                + tiles * 16
                + 4 * 2 * layout::RADIX_BINS as u64
                + n.div_ceil(256) * 4
                + RenderOut::bytes(cap, v)
                + PixelGrads::bytes(cap, v),
        }
    }
}

/// Raster forward outputs, planar CHW (spec-raster 1.4).
pub struct RenderOut {
    /// [3][P], background composited.
    pub rgb: CudaSlice<f32>,
    /// [P] final transmittance (alpha = 1 − T).
    pub trans: CudaSlice<f32>,
    /// [P] index + 1 of the last blended Gaussian in the tile list; 0 = none.
    pub n_contrib: CudaSlice<u32>,
    /// [EXTRA_CH][P] (expected depth Σ w z, later normals).
    pub extra: Option<CudaSlice<f32>>,
    /// Count mode: [V_cap] masked pixels each visible Gaussian blends into.
    pub count: Option<CudaSlice<u32>>,
}

impl RenderOut {
    pub fn alloc(gpu: &Gpu, cap: Capacity, v: &RasterVariant, with_count: bool) -> Result<Self> {
        let s = &gpu.stream;
        let px = cap.pixels.max(1) as usize;
        Ok(Self {
            rgb: s.alloc_zeros::<f32>(3 * px).cuda("alloc out_rgb")?,
            trans: s.alloc_zeros::<f32>(px).cuda("alloc out_T")?,
            n_contrib: s.alloc_zeros::<u32>(px).cuda("alloc out_nc")?,
            extra: match v.extra_ch {
                0 => None,
                e => Some(
                    s.alloc_zeros::<f32>(e as usize * px)
                        .cuda("alloc out_extra")?,
                ),
            },
            count: if with_count {
                Some(
                    s.alloc_zeros::<u32>(cap.visible.max(1) as usize)
                        .cuda("alloc count")?,
                )
            } else {
                None
            },
        })
    }

    fn bytes(cap: Capacity, v: &RasterVariant) -> u64 {
        cap.pixels as u64 * 4 * (3 + 1 + 1 + v.extra_ch as u64)
    }
}

/// Loss gradients w.r.t. the raster outputs, same shapes as [`RenderOut`].
pub struct PixelGrads {
    pub d_rgb: CudaSlice<f32>,
    pub d_alpha: Option<CudaSlice<f32>>,
    pub d_extra: Option<CudaSlice<f32>>,
}

impl PixelGrads {
    pub fn alloc(gpu: &Gpu, cap: Capacity, v: &RasterVariant, with_alpha: bool) -> Result<Self> {
        let s = &gpu.stream;
        let px = cap.pixels.max(1) as usize;
        Ok(Self {
            d_rgb: s.alloc_zeros::<f32>(3 * px).cuda("alloc d_rgb")?,
            d_alpha: if with_alpha {
                Some(s.alloc_zeros::<f32>(px).cuda("alloc d_alpha")?)
            } else {
                None
            },
            d_extra: match v.extra_ch {
                0 => None,
                e => Some(
                    s.alloc_zeros::<f32>(e as usize * px)
                        .cuda("alloc d_extra")?,
                ),
            },
        })
    }

    fn bytes(cap: Capacity, v: &RasterVariant) -> u64 {
        cap.pixels as u64 * 4 * (3 + 1 + v.extra_ch as u64)
    }
}

/// 2D gradients per visible Gaussian, SoA in v-space: `g[9][V_cap]` in the order
/// mx, my, A, B, C, o, r, g, b (scalar-B convention = gsplat `v_conics`);
/// `abs[2][V_cap]` (AbsGS), `extra[E][V_cap]` (depth channel).
/// preprocess_fwd zeroes `[·][v]` for v < V; raster_bwd accumulates.
pub struct Grad2d {
    pub g: CudaSlice<f32>,
    pub abs: Option<CudaSlice<f32>>,
    pub extra: Option<CudaSlice<f32>>,
}

impl Grad2d {
    pub const MX: usize = 0;
    pub const MY: usize = 1;
    pub const A: usize = 2;
    pub const B: usize = 3;
    pub const C: usize = 4;
    pub const O: usize = 5;
    pub const R: usize = 6;
    pub const G: usize = 7;
    pub const BLUE: usize = 8;

    pub fn alloc(gpu: &Gpu, cap: Capacity, v: &RasterVariant, abs: bool) -> Result<Self> {
        let s = &gpu.stream;
        let vis = cap.visible.max(1) as usize;
        Ok(Self {
            g: s.alloc_zeros::<f32>(9 * vis).cuda("alloc g2d")?,
            abs: if abs {
                Some(s.alloc_zeros::<f32>(2 * vis).cuda("alloc g2d abs")?)
            } else {
                None
            },
            extra: match v.extra_ch {
                0 => None,
                e => Some(
                    s.alloc_zeros::<f32>(e as usize * vis)
                        .cuda("alloc g2d extra")?,
                ),
            },
        })
    }
}

/// Per-image pose and per-camera intrinsics gradients (spec-raster 5.4), with
/// the per-block partials `cam_reduce` sums in fixed order.
pub struct CameraGrads {
    /// [n_images][6]: (ρ, φ), left perturbation in the camera frame.
    pub pose: CudaSlice<f32>,
    /// [n_intr][4]: fx, fy, cx, cy.
    pub intr: CudaSlice<f32>,
    /// [blocks][CAM_GRAD_N].
    pub partial: CudaSlice<f32>,
}

impl CameraGrads {
    pub fn alloc(gpu: &Gpu, n_images: u32, n_intr: u32, blocks: u32) -> Result<Self> {
        let s = &gpu.stream;
        Ok(Self {
            pose: s
                .alloc_zeros::<f32>(6 * n_images.max(1) as usize)
                .cuda("alloc pose grads")?,
            intr: s
                .alloc_zeros::<f32>(4 * n_intr.max(1) as usize)
                .cuda("alloc intrinsics grads")?,
            partial: s
                .alloc_zeros::<f32>(layout::CAM_GRAD_N as usize * blocks.max(1) as usize)
                .cuda("alloc camera partials")?,
        })
    }
}

/// Host sum of one image's camera gradients (spec-train 11's `CameraGrads`,
/// renamed: the raster's device struct above owns the name).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CameraGradSum {
    pub d_xi: [f32; 6],
    pub d_f: [f32; 2],
    pub d_c: [f32; 2],
}

/// What `raster_fwd` does this launch.
pub enum RenderMode<'a> {
    /// Checkpoints, n_contrib and tile_max_nc for the backward.
    Train,
    /// Colour and T only (test PSNR, previews, VCD renders).
    Infer,
    /// FastGS VCD/VCP: `RenderOut::count[v]` += masked pixels v blends into.
    Count { mask: &'a CudaSlice<u8> },
}

/// Where preprocess_bwd routes per-Gaussian gradients (spec-raster 5.5).
/// `ShRank1` / `ShAccumulate` are flags inside `adam::AdamDevice` for the SH-rest group.
pub enum GradSink<'a> {
    /// Gradient buffers, visible entries only; Adam treats invisible ones as 0
    /// (`last_visible != t`).
    Write(&'a mut crate::model::Grads),
    /// Phase 3b: apply Adam in place (parameters are the `Gaussians` passed to
    /// preprocess_bwd, which is why it takes them `&mut`).
    FusedAdam {
        m: &'a mut adam::Moments,
        hyper: &'a adam::AdamDevice,
    },
}

/// The CUDA functions of one [`RasterVariant`], loaded per stage.
pub struct SplatKernels {
    pub variant: RasterVariant,
    pub preprocess: preprocess::PreprocessKernels,
    pub binning: sort::BinningKernels,
    pub raster: raster::RasterKernels,
}

impl SplatKernels {
    pub fn load(gpu: &Gpu, v: RasterVariant) -> Result<Self> {
        v.validate()?;
        let d = v.defines();
        Ok(Self {
            variant: v,
            preprocess: preprocess::PreprocessKernels::load(gpu, &d)?,
            binning: sort::BinningKernels::load(gpu, &d)?,
            raster: raster::RasterKernels::load(gpu, &d)?,
        })
    }
}

/// Every NVRTC compilation unit of the app, for the existence tests.
pub const SOURCES: &[KernelSource] = &[
    preprocess::SOURCE,
    sort::RADIX_SOURCE,
    sort::BINNING_SOURCE,
    raster::FWD_SOURCE,
    raster::BWD_SOURCE,
    loss::SOURCE,
    loss::METRICS_SOURCE,
    adam::SOURCE,
    densify::DENSIFY_SOURCE,
    densify::COMPACT_SOURCE,
    densify::MCMC_SOURCE,
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Every kernel a Rust module loads is declared `extern "C" __global__` in
    /// its unit, and every declared kernel is loaded (no dead stubs).
    #[test]
    fn every_listed_kernel_is_declared_in_its_source() {
        for src in SOURCES {
            let text = src.text();
            let mut declared: Vec<&str> = text
                .split("extern \"C\" __global__ void ")
                .skip(1)
                .map(|rest| {
                    // `__launch_bounds__(...)` may sit between `void` and the name.
                    match rest.strip_prefix("__launch_bounds__(") {
                        Some(r) => r.split_once(')').map_or(r, |(_, after)| after),
                        None => rest,
                    }
                    .trim_start()
                })
                .filter_map(|rest| rest.split('(').next())
                .collect();
            declared.sort_unstable();
            let mut listed = src.kernels.to_vec();
            listed.sort_unstable();
            assert_eq!(declared, listed, "{}", src.module);
        }
    }

    #[test]
    fn variant_defines_cover_the_cuda_variant_block() {
        let d = RasterVariant::default().defines();
        for name in [
            "TILE_H",
            "CHAIN_K",
            "CKPT_F16",
            "REC32",
            "EXTRA_CH",
            "BLEND",
            "DETERMINISTIC",
        ] {
            assert!(d.get(name).is_some(), "{name}");
            assert!(
                include_str!("../../cuda/splat_common.cuh").contains(&format!("#ifndef {name}")),
                "{name} has no default in splat_common.cuh"
            );
        }
        assert!(
            RasterVariant {
                chain_k: 3,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert_eq!(RasterVariant::default().tiles(1297, 840), 82 * 53);
        assert_eq!(
            RasterVariant::default().tile_pixels(),
            layout::RASTER_THREADS
        );
    }

    /// spec-raster 1.5: garden at 1/4 resolution, N_cap 6.5M, I_cap 24M, K = 4.
    #[test]
    fn vram_budget_matches_the_spec_table() {
        let v = RasterVariant::default();
        let cap = Capacity {
            gaussians: 6_500_000,
            visible: 6_500_000,
            instances: 24_000_000,
            tiles: MAX_TILES,
            pixels: 1297 * 840,
        };
        let b = FrameBuffers::vram_bytes(cap, &v);
        let gb = |x: u64| x as f64 / 1e9;
        assert!(
            (0.75..0.90).contains(&gb(b.per_visible)),
            "per-v {}",
            gb(b.per_visible)
        );
        assert!(
            (0.27..0.31).contains(&gb(b.per_instance)),
            "per-instance {}",
            gb(b.per_instance)
        );
        assert!(
            (0.78..0.82).contains(&gb(b.checkpoints)),
            "checkpoints {}",
            gb(b.checkpoints)
        );
        assert!(gb(b.other) < 0.17, "other {}", gb(b.other));
        let k1 = FrameBuffers::vram_bytes(cap, &RasterVariant { chain_k: 1, ..v });
        assert!(
            (3.0..3.3).contains(&gb(k1.checkpoints)),
            "K = 1 {}",
            gb(k1.checkpoints)
        );
    }
}
