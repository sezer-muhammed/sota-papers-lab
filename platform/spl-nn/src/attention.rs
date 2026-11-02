//! FlashAttention-2-style bf16 attention, head dim 64 (`cuda/attention.cu`).
//!
//! Inputs are the [`crate::qkv`] layout, `[heads][n_seq · seq_len][64]`:
//! `n_seq` sequences of `seq_len` tokens each, per head. Frame attention is
//! `n_seq = frames, seq_len = 689`; global attention over the same buffers
//! is `n_seq = 1, seq_len = frames · 689`. The output is token-major,
//! `[n_seq · seq_len][o_ld]` with head `h` in columns `h·64..h·64+64`, which
//! is what the output projection GEMM reads.
//!
//! The kernel family is parameterised by its [`Tile`] (warps per CTA, m16
//! tiles per warp, keys per pipeline stage, Q aliasing, V format). The
//! defaults come from measurements on the RTX 4070 SUPER (`bench_attention`
//! against `scripts/torch_bench.py`, 16 heads, 689-token frames):
//!
//! | Shape | bf16 (ours) | fp16 P·V (ours) | SDPA FlashAttention-2 | SDPA cuDNN |
//! |---|---|---|---|---|
//! | 2 frames | 57 TFLOPS | 73 | 41 | 48 |
//! | 200 frames | 67 | 84 | 53 | 64 |
//! | global, 64 frames (44,096 tokens) | 73 | 94 | 71 | 74 |
//!
//! [`VFormat::F16`] tiles read V in fp16 (written by the QKV prologue) and
//! run P·V on fp16-accumulating tensor cores, which GeForce Ada executes at
//! twice the fp32-accumulate rate; each 64-key partial is folded into fp32,
//! so they are also slightly more accurate than bf16 (P keeps 3 more
//! mantissa bits). They need |V| below about 1000.

use cudarc::driver::sys::CUfunction_attribute;
use cudarc::driver::{
    CudaFunction, DevicePtr, DevicePtrMut, DeviceRepr, LaunchConfig, PushKernelArg,
};
use spl_core::{Result, anyhow, bail};

use crate::norm::check_aligned;
use crate::qkv::HEAD_DIM;
pub use crate::qkv::VFormat;
use crate::{COMMON_CUH, Cuda, Nn, Ptrs, dim};

const SOURCE: &str = include_str!("../cuda/attention.cu");

/// One compiled kernel shape: `warps` warps per CTA, each owning `mt` m16
/// tiles of query rows (so `bm = 16 · warps · mt` rows per CTA), streaming
/// `bn` keys per pipeline stage. `alias_q` stages Q in the second pipeline
/// stage before the key loop starts, which saves the Q tile's shared memory.
/// `v` is the format of V the kernel reads; [`VFormat::F16`] tiles run P·V
/// on fp16-accumulating tensor cores (twice the rate on GeForce Ada).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Tile {
    pub warps: u32,
    pub mt: u32,
    pub bn: u32,
    pub alias_q: bool,
    pub v: VFormat,
}

impl Tile {
    pub const fn new(warps: u32, mt: u32, bn: u32) -> Self {
        Self {
            warps,
            mt,
            bn,
            alias_q: false,
            v: VFormat::Bf16,
        }
    }

    pub const fn aliased(warps: u32, mt: u32, bn: u32) -> Self {
        Self {
            alias_q: true,
            ..Self::new(warps, mt, bn)
        }
    }

    /// The same tile reading fp16 V with fp16-accumulated P·V.
    pub const fn pv16(self) -> Self {
        Self {
            v: VFormat::F16,
            ..self
        }
    }

    /// Query rows per CTA.
    pub const fn bm(self) -> u32 {
        16 * self.warps * self.mt
    }

    /// Dynamic shared memory: two [K | V] stages, plus the Q tile unless it
    /// is aliased onto the second stage.
    pub const fn smem_bytes(self) -> u32 {
        let q = if self.alias_q { 0 } else { self.bm() };
        (q + 4 * self.bn) * HEAD_DIM as u32 * 2
    }

    fn kernel_name(self) -> String {
        let qa = if self.alias_q { "_qa" } else { "" };
        let pv = if self.v == VFormat::F16 { "_pv16" } else { "" };
        format!("attn_w{}_m{}_n{}{qa}{pv}", self.warps, self.mt, self.bn)
    }
}

/// Every compiled shape (`ATTN(...)` lines in cuda/attention.cu).
pub const TILES: [Tile; 15] = [
    Tile::new(4, 1, 64),
    Tile::new(4, 1, 128),
    Tile::new(4, 2, 32),
    Tile::new(4, 2, 64),
    Tile::new(8, 1, 64),
    Tile::new(8, 1, 128),
    Tile::new(2, 2, 64),
    Tile::aliased(4, 1, 64),
    Tile::aliased(4, 2, 64),
    Tile::aliased(8, 1, 64),
    Tile::aliased(2, 2, 64),
    Tile::aliased(4, 1, 64).pv16(),
    Tile::new(4, 2, 64).pv16(),
    Tile::new(8, 1, 64).pv16(),
    Tile::aliased(8, 1, 64).pv16(),
];

/// Problem shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttnShape {
    pub seq_len: usize,
    pub n_seq: usize,
    pub heads: usize,
    /// Elements per output row (`heads · 64` when packed).
    pub o_ld: usize,
}

impl AttnShape {
    /// Frame attention: `frames` sequences of `tokens` each.
    pub fn frames(frames: usize, tokens: usize, heads: usize) -> Self {
        Self {
            seq_len: tokens,
            n_seq: frames,
            heads,
            o_ld: heads * HEAD_DIM,
        }
    }

    /// Global attention: all `frames · tokens` tokens as one sequence.
    pub fn global(frames: usize, tokens: usize, heads: usize) -> Self {
        Self {
            seq_len: frames * tokens,
            n_seq: 1,
            heads,
            o_ld: heads * HEAD_DIM,
        }
    }

    pub fn total_tokens(&self) -> usize {
        self.seq_len * self.n_seq
    }

    /// Useful floating-point work: `4 · L² · 64` per (sequence, head).
    pub fn flops(&self) -> f64 {
        4.0 * (self.seq_len as f64).powi(2) * HEAD_DIM as f64 * (self.n_seq * self.heads) as f64
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct AttnArgs {
    q: u64,
    k: u64,
    v: u64,
    o: u64,
    seq_len: i32,
    n_seq: i32,
    o_ld: i32,
    scale_log2: f32,
}

// SAFETY: plain old data matching `struct AttnArgs` in cuda/attention.cu.
unsafe impl DeviceRepr for AttnArgs {}

pub struct FlashAttention {
    kernels: Vec<(Tile, CudaFunction)>,
}

impl FlashAttention {
    /// Registers per thread and resident CTAs per SM of every compiled tile.
    pub fn occupancy(&self) -> Result<Vec<(Tile, i32, u32)>> {
        self.kernels
            .iter()
            .map(|(t, f)| {
                let regs = f
                    .get_attribute(CUfunction_attribute::CU_FUNC_ATTRIBUTE_NUM_REGS)
                    .cuda("kernel registers")?;
                let ctas = f
                    .occupancy_max_active_blocks_per_multiprocessor(
                        t.warps * 32,
                        t.smem_bytes() as usize,
                        None,
                    )
                    .cuda("occupancy")?;
                Ok((*t, regs, ctas))
            })
            .collect()
    }

    pub fn new(nn: &Nn) -> Result<Self> {
        let src = [COMMON_CUH, SOURCE].concat();
        let module = nn.module("spl_nn_attention", &src)?;
        let mut kernels = Vec::with_capacity(TILES.len());
        for t in TILES {
            let name = t.kernel_name();
            let f = module
                .load_function(&name)
                .map_err(|e| anyhow!("load kernel {name}: {e:?}"))?;
            f.set_attribute(
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                t.smem_bytes() as i32,
            )
            .cuda("attention shared memory size")?;
            // Prefer the largest shared-memory carveout: more CTAs per SM.
            f.set_attribute(
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                100,
            )
            .cuda("attention carveout")?;
            kernels.push((t, f));
        }
        Ok(Self { kernels })
    }

    /// The tile measured fastest for `shape` and V format on Ada (RTX 4070
    /// SUPER, `bench_attention`): frame-length sequences keep `bm = 64`
    /// with Q aliased (three CTAs per SM, little waste in the ragged last
    /// query tile); long ones take 128-row CTAs that halve K/V traffic.
    pub fn default_tile(shape: &AttnShape, v: VFormat) -> Tile {
        match (shape.seq_len <= 8192, v) {
            (true, VFormat::Bf16) => Tile::aliased(4, 1, 64),
            (false, VFormat::Bf16) => Tile::new(4, 2, 64),
            (true, VFormat::F16) => Tile::aliased(4, 1, 64).pv16(),
            (false, VFormat::F16) => Tile::new(8, 1, 64).pv16(),
        }
    }

    /// `o = softmax(q kᵀ · scale) v` for every (sequence, head). `q` and `k`
    /// are bf16; `v` is bf16 or fp16 as `tile.v` says (the default tile
    /// expects bf16).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        nn: &Nn,
        q: &dyn DevicePtr<u16>,
        k: &dyn DevicePtr<u16>,
        v: &dyn DevicePtr<u16>,
        o: &mut dyn DevicePtrMut<u16>,
        shape: &AttnShape,
        scale: f32,
        tile: Option<Tile>,
    ) -> Result<()> {
        let tile = tile.unwrap_or_else(|| Self::default_tile(shape, VFormat::Bf16));
        let Some((_, f)) = self.kernels.iter().find(|(t, _)| *t == tile) else {
            bail!("attention: tile {tile:?} is not compiled");
        };
        if shape.o_ld < shape.heads * HEAD_DIM || !shape.o_ld.is_multiple_of(8) {
            bail!(
                "attention: o_ld {} must be a multiple of 8 and at least heads · 64",
                shape.o_ld
            );
        }
        let total = shape.total_tokens();
        if total == 0 || shape.heads == 0 {
            return Ok(());
        }
        let mut ptrs = Ptrs::new(&nn.stream);
        let n = shape.heads * total * HEAD_DIM;
        let a = AttnArgs {
            q: ptrs.input(q, n, "attention q")?,
            k: ptrs.input(k, n, "attention k")?,
            v: ptrs.input(v, n, "attention v")?,
            o: ptrs.output(
                o,
                (total - 1) * shape.o_ld + shape.heads * HEAD_DIM,
                "attention o",
            )?,
            seq_len: dim(shape.seq_len, "seq_len")?,
            n_seq: dim(shape.n_seq, "n_seq")?,
            o_ld: dim(shape.o_ld, "o_ld")?,
            scale_log2: scale * std::f32::consts::LOG2_E,
        };
        check_aligned(&[a.q, a.k, a.v, a.o])?;
        if shape.n_seq > 65535 || shape.heads > 65535 {
            bail!("attention: n_seq and heads must be at most 65535");
        }
        let cfg = LaunchConfig {
            grid_dim: (
                shape.seq_len.div_ceil(tile.bm() as usize) as u32,
                shape.n_seq as u32,
                shape.heads as u32,
            ),
            block_dim: (tile.warps * 32, 1, 1),
            shared_mem_bytes: tile.smem_bytes(),
        };
        let mut b = nn.stream.launch_builder(f);
        b.arg(&a);
        // SAFETY: `a` matches `AttnArgs`; q/k/v hold heads × total × 64
        // elements, o covers every written row; all 16-byte aligned.
        unsafe { b.launch(cfg) }.cuda("attention")?;
        drop(ptrs);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiles_fit_in_ada_shared_memory() {
        for t in TILES {
            assert!(t.smem_bytes() <= 99 * 1024, "{t:?}");
            assert_eq!((t.bm() * 8) % (t.warps * 32), 0, "{t:?} Q tile load");
            assert_eq!((t.bn * 8) % (t.warps * 32), 0, "{t:?} K/V tile load");
            assert_eq!(t.bn % 16, 0, "{t:?} keys per stage");
        }
        assert_eq!(Tile::new(4, 1, 64).smem_bytes(), 40 * 1024);
        assert_eq!(Tile::aliased(4, 1, 64).smem_bytes(), 32 * 1024);
    }

    #[test]
    fn kernel_names_match_the_source() {
        for t in TILES {
            let line = format!(
                "ATTN({}, {}, {}, {}, {}, {})",
                t.kernel_name(),
                t.warps,
                t.mt,
                t.bn,
                t.alias_q,
                t.v == VFormat::F16
            );
            assert!(SOURCE.contains(&line), "{line}");
            assert!(!t.alias_q || t.bm() <= 2 * t.bn, "{t:?} Q fits a stage");
        }
    }

    #[test]
    fn flops_count_both_matmuls() {
        let s = AttnShape::frames(2, 689, 16);
        assert_eq!(s.flops(), 4.0 * 689.0 * 689.0 * 64.0 * 32.0);
        assert_eq!(AttnShape::global(2, 689, 16).seq_len, 1378);
    }
}
