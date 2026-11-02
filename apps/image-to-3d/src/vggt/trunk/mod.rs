//! VGGT-Ω trunk (spec `reports/image-to-3d/spec-vggt.md` §3): the DINOv3
//! ViT-L/16 patch embedder, token assembly, and the 24 alternating frame /
//! inter-frame blocks, on `spl_nn` bf16 GEMMs and FlashAttention with an
//! fp32 residual stream (the precision of the official autocast path, §3.5).
//!
//! The trunk writes the four cached layers the heads read
//! (`[F_i | G_i]`, i = 4, 11, 17, 23) as bf16 rows into caller-owned
//! buffers and can hand every intermediate stream to a [`Tap`] for tests.
//! Buffers are allocated once per largest geometry and reused across blocks
//! and runs.
//!
//! Inter-frame attention goes through [`InterAttention`], so Phase 2 can swap
//! the dense global kernel for SageAttention or KV merging per layer.

mod aggregator;
mod block;
mod dino;
mod kernels;
pub mod rope;

#[cfg(test)]
mod tests;

use spl_core::{Result, anyhow, bail};
use spl_nn::Nn;
use spl_nn::attention::{AttnShape, FlashAttention};
use spl_nn::cudarc::driver::{CudaSlice, CudaView, CudaViewMut, DevicePtr, DevicePtrMut};
use spl_nn::gemm::{Gemm, Tuning};
use spl_nn::norm::Norm;
use spl_nn::qkv::{HEAD_DIM, QkvPrologue};
use spl_nn::residual::Residual;

pub use block::BlockWeights;

/// Model width.
pub const DIM: usize = 1024;
/// Attention heads (head dim 64).
pub const HEADS: usize = 16;
/// MLP hidden width.
pub const HIDDEN: usize = 4 * DIM;
/// Blocks per family.
pub const DEPTH: usize = 24;
/// Camera + register tokens in front of each frame's patches.
pub const SPECIAL: usize = 17;
/// cls + storage tokens in front of the DINO patches.
pub const DINO_PREFIX: usize = 5;
/// Aggregator layers whose `[F_i | G_i]` the heads read.
pub const CACHED: [usize; 4] = [4, 11, 17, 23];
/// Register-attention layers of the official schedule.
pub const REGISTER_LAYERS: [usize; 5] = [2, 6, 9, 14, 20];
/// LayerNorm epsilon everywhere in the trunk.
pub const LN_EPS: f32 = 1e-5;

/// Weight lookup by checkpoint name (spec §1). 2-D weights and biases are
/// bf16; LayerNorm parameters, LayerScale gammas, tokens and RoPE periods
/// f32. A QKV bias that the checkpoint masks to zero is simply absent.
pub trait TrunkWeights {
    fn bf16(&self, name: &str) -> Result<CudaView<'_, u16>>;
    fn f32(&self, name: &str) -> Result<CudaView<'_, f32>>;
    fn host_f32(&self, name: &str) -> Result<Vec<f32>>;
    fn has(&self, name: &str) -> bool;
}

/// Frames and patch grid of one run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrunkGeom {
    pub frames: usize,
    pub grid_h: usize,
    pub grid_w: usize,
}

impl TrunkGeom {
    pub fn patches(&self) -> usize {
        self.grid_h * self.grid_w
    }

    /// Tokens per frame in the aggregator, `17 + P`.
    pub fn tokens(&self) -> usize {
        SPECIAL + self.patches()
    }

    /// Rows of the residual stream, `N · T`.
    pub fn rows(&self) -> usize {
        self.frames * self.tokens()
    }
}

/// What an inter-frame block attends over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterKind {
    /// All `N · T` tokens as one sequence.
    Global,
    /// Only the 17 special tokens of every frame (`17 N`); patches untouched.
    Register,
}

/// The official schedule: register attention at 2, 6, 9, 14, 20.
pub fn dense_schedule() -> [InterKind; DEPTH] {
    let mut s = [InterKind::Global; DEPTH];
    for i in REGISTER_LAYERS {
        s[i] = InterKind::Register;
    }
    s
}

/// An intermediate stream handed to the tap (fp32 rows, `[N · T][1024]` in
/// the aggregator layout).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tap {
    /// Tokens entering frame block 0 (DINO output + special tokens).
    Input,
    /// After `frame_blocks[i]`.
    Frame(usize),
    /// After `inter_frame_blocks[i]`.
    Inter(usize),
}

/// Observer of intermediate streams; return `true` from `wants` to receive one.
pub trait TapSink {
    fn wants(&self, tap: Tap) -> bool;
    fn take(&mut self, nn: &Nn, tap: Tap, rows: &CudaSlice<f32>, geom: TrunkGeom) -> Result<()>;
}

/// No taps.
pub struct NoTap;

impl TapSink for NoTap {
    fn wants(&self, _: Tap) -> bool {
        false
    }
    fn take(&mut self, _: &Nn, _: Tap, _: &CudaSlice<f32>, _: TrunkGeom) -> Result<()> {
        Ok(())
    }
}

/// Attention over head-major `[16][tokens][64]` bf16 Q/K/V, writing
/// `[tokens][1024]` bf16; `n_seq` sequences of `seq_len` tokens.
pub trait InterAttention {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        nn: &Nn,
        fa: &FlashAttention,
        q: &dyn DevicePtr<u16>,
        k: &dyn DevicePtr<u16>,
        v: &dyn DevicePtr<u16>,
        o: &mut dyn DevicePtrMut<u16>,
        n_seq: usize,
        seq_len: usize,
    ) -> Result<()>;
}

/// FlashAttention-2 bf16 with the measured default tile.
pub struct DenseAttention;

impl InterAttention for DenseAttention {
    fn forward(
        &self,
        nn: &Nn,
        fa: &FlashAttention,
        q: &dyn DevicePtr<u16>,
        k: &dyn DevicePtr<u16>,
        v: &dyn DevicePtr<u16>,
        o: &mut dyn DevicePtrMut<u16>,
        n_seq: usize,
        seq_len: usize,
    ) -> Result<()> {
        let shape = AttnShape {
            seq_len,
            n_seq,
            heads: HEADS,
            o_ld: DIM,
        };
        fa.forward(nn, q, k, v, o, &shape, ATTN_SCALE, None)
    }
}

/// The dense attention used by frame and register blocks.
pub(crate) const DENSE: DenseAttention = DenseAttention;

/// SDPA's default scale, `1 / sqrt(64)`.
pub const ATTN_SCALE: f32 = 0.125;

/// Device buffers sized for `rows` stream rows, reused by every block.
pub(crate) struct Buffers {
    rows: usize,
    /// Aggregator residual stream `[N · T][1024]`.
    x: CudaSlice<f32>,
    /// DINO stream (`N · (5 + P)` rows), then the packed register sequence.
    xd: CudaSlice<f32>,
    /// LayerNorm output, GEMM input.
    xn: CudaSlice<u16>,
    /// QKV projection, MLP hidden, patch im2col (`[rows][4096]`).
    big: CudaSlice<u16>,
    q: CudaSlice<u16>,
    k: CudaSlice<u16>,
    v: CudaSlice<u16>,
    /// Attention output.
    o: CudaSlice<u16>,
    /// proj / fc2 output.
    y: CudaSlice<u16>,
}

impl Buffers {
    fn bytes(rows: usize) -> usize {
        rows * (2 * DIM * 4 + (DIM + HIDDEN + 3 * DIM + 2 * DIM) * 2)
    }

    fn new(nn: &Nn, rows: usize) -> Result<Self> {
        let s = &nn.stream;
        let e = |e| anyhow!("allocate trunk buffers for {rows} rows: {e:?}");
        Ok(Self {
            rows,
            x: s.alloc_zeros(rows * DIM).map_err(e)?,
            xd: s.alloc_zeros(rows * DIM).map_err(e)?,
            xn: s.alloc_zeros(rows * DIM).map_err(e)?,
            big: s.alloc_zeros(rows * HIDDEN).map_err(e)?,
            q: s.alloc_zeros(rows * DIM).map_err(e)?,
            k: s.alloc_zeros(rows * DIM).map_err(e)?,
            v: s.alloc_zeros(rows * DIM).map_err(e)?,
            o: s.alloc_zeros(rows * DIM).map_err(e)?,
            y: s.alloc_zeros(rows * DIM).map_err(e)?,
        })
    }

    /// The two f32 streams and the block scratch, borrowed apart.
    fn split(&mut self) -> (&mut CudaSlice<f32>, &mut CudaSlice<f32>, block::Scratch<'_>) {
        (
            &mut self.x,
            &mut self.xd,
            block::Scratch {
                xn: &mut self.xn,
                big: &mut self.big,
                q: &mut self.q,
                k: &mut self.k,
                v: &mut self.v,
                o: &mut self.o,
                y: &mut self.y,
            },
        )
    }
}

/// The spl-nn and trunk kernels.
pub(crate) struct Ops {
    gemm: Gemm,
    norm: Norm,
    qkv: QkvPrologue,
    fa: FlashAttention,
    res: Residual,
    k: kernels::TrunkKernels,
}

/// RoPE tables uploaded for one grid: (grid, DINO, aggregator).
type RopeTables = ((usize, usize), CudaSlice<f32>, CudaSlice<f32>);

pub struct Trunk {
    ops: Ops,
    bufs: Option<Buffers>,
    /// `[cls + 0·mask, storage 0..3]`, f32 `[5][1024]`.
    dino_prefix: CudaSlice<f32>,
    /// `[camera, register 0..15]` for frame 0 and for the others, f32 `[2][17][1024]`.
    special: CudaSlice<f32>,
    dino_periods: Vec<f32>,
    agg_periods: Vec<f32>,
    /// RoPE tables for the last grid: (grid, DINO, aggregator).
    rope: Option<RopeTables>,
    global: Box<dyn InterAttention + Send + Sync>,
}

impl Trunk {
    pub fn new(nn: &Nn, w: &dyn TrunkWeights) -> Result<Self> {
        let ops = Ops {
            gemm: Gemm::new(nn, Tuning::Heuristic)?,
            norm: Norm::new(nn)?,
            qkv: QkvPrologue::new(nn)?,
            fa: FlashAttention::new(nn)?,
            res: Residual::new(nn)?,
            k: kernels::TrunkKernels::new(nn)?,
        };
        debug_assert_eq!(DIM, HEADS * HEAD_DIM);

        let pe = "aggregator.patch_embed.";
        let cls = w.host_f32(&format!("{pe}cls_token"))?;
        let storage = w.host_f32(&format!("{pe}storage_tokens"))?;
        if cls.len() != DIM || storage.len() != 4 * DIM {
            bail!("DINO cls/storage tokens have unexpected sizes");
        }
        // cls + 0 · mask_token is cls unless the mask holds non-finite values.
        let prefix: Vec<f32> = cls.iter().chain(&storage).copied().collect();

        let cam = w.host_f32("aggregator.camera_token")?;
        let reg = w.host_f32("aggregator.register_token")?;
        if cam.len() != 2 * DIM || reg.len() != 2 * 16 * DIM {
            bail!("camera/register tokens have unexpected sizes");
        }
        let mut special = Vec::with_capacity(2 * SPECIAL * DIM);
        for v in 0..2 {
            special.extend_from_slice(&cam[v * DIM..(v + 1) * DIM]);
            special.extend_from_slice(&reg[v * 16 * DIM..(v + 1) * 16 * DIM]);
        }
        let up = |v: &[f32]| {
            nn.stream
                .clone_htod(v)
                .map_err(|e| anyhow!("upload trunk tokens: {e:?}"))
        };
        Ok(Self {
            ops,
            bufs: None,
            dino_prefix: up(&prefix)?,
            special: up(&special)?,
            dino_periods: w.host_f32(&format!("{pe}rope_embed.periods"))?,
            agg_periods: w.host_f32("aggregator.rope_embed.periods")?,
            rope: None,
            global: Box::new(DenseAttention),
        })
    }

    /// Replaces the global inter-frame attention (Phase 2 schedules).
    pub fn set_global_attention(&mut self, a: Box<dyn InterAttention + Send + Sync>) {
        self.global = a;
    }

    /// Device bytes the trunk allocates for `g` (on top of the weights and
    /// the cache).
    pub fn scratch_bytes(g: &TrunkGeom) -> usize {
        Buffers::bytes(g.rows())
    }

    /// Runs the trunk on `images` (normalised f32 `[N][3][H][W]`, H = 16 h,
    /// W = 16 w) and writes `cache[k]` = bf16 `[N · T][2048]` rows
    /// `[F_i | G_i]` for i = [`CACHED`]`[k]`. With `cam_tokens`, also
    /// writes f32 `[N · 17][2048]` = `[F_23 | G_23]` of the special tokens
    /// (frame-major) for the camera head, which is sensitive to bf16 there. Enqueues work; does not
    /// synchronise unless a tap asks for data.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &mut self,
        nn: &Nn,
        w: &dyn TrunkWeights,
        geom: TrunkGeom,
        images: &CudaView<'_, f32>,
        cache: [&mut CudaViewMut<'_, u16>; 4],
        cam_tokens: Option<&mut CudaViewMut<'_, f32>>,
        sched: &[InterKind; DEPTH],
        tap: &mut dyn TapSink,
    ) -> Result<()> {
        if let Some(c) = &cam_tokens
            && c.len() < geom.frames * SPECIAL * 2 * DIM
        {
            bail!(
                "trunk: cam_tokens holds {} values, needs N · 17 · 2048",
                c.len()
            );
        }
        if geom.frames == 0 || geom.grid_h == 0 || geom.grid_w == 0 {
            bail!("trunk: empty geometry {geom:?}");
        }
        let need = geom.frames * 3 * geom.grid_h * 16 * geom.grid_w * 16;
        if images.len() < need {
            bail!("trunk: images hold {} values, need {need}", images.len());
        }
        for c in &cache {
            if c.len() < geom.rows() * 2 * DIM {
                bail!("trunk: cache buffer too small for {} rows", geom.rows());
            }
        }
        if self.bufs.as_ref().is_none_or(|b| b.rows < geom.rows()) {
            self.bufs = None;
            self.bufs = Some(Buffers::new(nn, geom.rows())?);
        }
        let grid = (geom.grid_h, geom.grid_w);
        if self.rope.as_ref().is_none_or(|r| r.0 != grid) {
            let d = rope::table(geom.grid_h, geom.grid_w, DINO_PREFIX, &self.dino_periods);
            let a = rope::table(geom.grid_h, geom.grid_w, SPECIAL, &self.agg_periods);
            self.rope = Some((grid, d.upload(nn)?, a.upload(nn)?));
        }
        let Some(bufs) = self.bufs.as_mut() else {
            bail!("trunk buffers missing");
        };
        let Some((_, dino_rope, agg_rope)) = self.rope.as_ref() else {
            bail!("trunk RoPE tables missing");
        };
        dino::run(
            nn,
            &self.ops,
            w,
            bufs,
            geom,
            images,
            &self.dino_prefix,
            dino_rope,
        )?;
        aggregator::run(
            nn,
            &self.ops,
            w,
            bufs,
            geom,
            &self.special,
            agg_rope,
            self.global.as_ref(),
            cache,
            cam_tokens,
            sched,
            tap,
        )
    }
}
