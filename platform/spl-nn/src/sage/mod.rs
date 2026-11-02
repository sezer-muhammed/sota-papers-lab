//! SageAttention2-style quantised attention for long global sequences
//! (`reports/Image to 3D.md`, Stage A items 5 and 6): the global layers of
//! the VGGT-Ω engine, where one sequence holds every frame's tokens
//! (200 frames × 689 tokens = 137,800 at 512×336).
//!
//! [`SageAttention::prepare`] (`cuda/sage_quant.cu`, three launches) reads
//! bf16 Q, K and V in any `[batch][token][head][dim]`-strided layout, so the
//! packed output of the QKV GEMM works as is, and writes:
//!
//! - **Q:** INT8 with one scale per 16 rows (the rows one warp owns), with
//!   the softmax scale and log2(e) folded in;
//! - **K:** smoothed (the per-channel mean over the sequence is subtracted,
//!   exact under softmax), INT8 with one scale per 64-key block;
//! - **V:** FP8 E4M3 with one scale per channel, optionally smoothed too (the
//!   mean is added back to the output), transposed and key-permuted for the
//!   FP8 `mma.sync`.
//!
//! The per-head QK LayerNorm of VGGT-Ω (`use_qk_norm`) can run in the same
//! pass ([`QkNorm`]); VGGT-Ω's global layers have no RoPE, so the raw QKV
//! GEMM output can go straight in.
//!
//! **KV merging** ([`KvMerge::Pool`], the GPA-VGGT pruning schedule): keys
//! and values of each frame's patch grid are mean-pooled in 2×2 or 4×4
//! cells, each pooled key gets a +log(count) logit bias so it keeps its
//! members' attention mass, camera and register tokens stay unmerged, and
//! queries are never merged. Pooling happens in the same prologue pass.
//!
//! [`SageAttention::attend`] (`cuda/sage_attn.cu`) computes INT8 QKᵀ with
//! `mma.sync m16n8k32 s8`, an exp2 online softmax, and FP8 PV with
//! `mma.sync m16n8k32 e4m3` accumulating in FP32 with two-level
//! accumulation ([`PvAccum::F16`] accumulates each 64-key block in FP16
//! instead, SageAttention2++), and writes bf16 in any strided layout.
//!
//! **Measured** on an RTX 4070 SUPER, 200 frames × 677 tokens of real
//! VGGT-1B global-attention activations, 16 heads of 64 (details and the
//! harness: `tests.rs` `gpu_bench`, `torch_ref.py`, `vggt_capture.py`):
//!
//! | | attention | error vs fp64 (cos / rel L1) |
//! |---|---|---|
//! | FlashAttention-2 2.8.3, bf16 | 1047 ms, 71.7 TFLOPS | 0.999999 / 0.15% |
//! | PyTorch SDPA (cuDNN), bf16 | 1018 ms, 73.8 TFLOPS | 0.999999 / 0.15% |
//! | sage, FP32 PV | 425 ms + 4.3 ms prologue, 177 TOPS | 0.99999 / 0.24-0.44% |
//! | sage, FP16 PV | 358 ms + 4.3 ms prologue, 210 TOPS | 0.99999 / 0.25-0.44% |
//! | sage merge 2×2 / 4×4, FP16 PV | 112 / 33 ms | merging itself: 3.5-40% / 5.4-56% by layer |
//!
//! Merging is a per-layer decision: on VGGT-1B it costs 3.5% (2×2) at
//! global layer 20 but 40% at layer 11, while the kernel adds only its
//! 0.2-0.4% quantisation error to whatever the merge does.
//!
//! Algorithm: SageAttention2 (arXiv 2411.10958) and SageAttention2++ (arXiv
//! 2505.21136), reference code Apache-2.0; KV merging: GPA-VGGT (arXiv
//! 2610.37000). Our own implementation; needs compute capability 8.9 (Ada)
//! or newer.

pub mod reference;
#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

use cudarc::driver::sys::CUfunction_attribute;
use cudarc::driver::{
    CudaFunction, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, LaunchConfig,
    PushKernelArg, SyncOnDrop,
};
use cudarc::nvrtc::{CompileOptions, Ptx};
use spl_core::{Result, anyhow, bail};

/// Query rows per INT8 scale (one `mma` m-tile; mirrors `SG_QGROUP`).
pub const QGROUP: usize = 16;
/// Queries per prologue tile (mirrors `SG_QTILE`).
pub const QTILE: usize = 128;
/// Keys per K/V block and per K scale (mirrors `SG_KBLOCK`).
pub const KBLOCK: usize = 64;
/// Largest E4M3 magnitude; V is scaled to it.
pub const FP8_MAX: f32 = 448.0;

const LN_Q: i32 = 1;
const LN_K: i32 = 2;
const SMOOTH_K: i32 = 4;
const SMOOTH_V: i32 = 8;

/// The NVRTC source: shared helpers, the prologue and the attention kernels.
pub static SOURCE: LazyLock<String> = LazyLock::new(|| {
    [
        include_str!("../../cuda/sage_common.cu"),
        "\n",
        include_str!("../../cuda/sage_quant.cu"),
        "\n",
        include_str!("../../cuda/sage_attn.cu"),
    ]
    .concat()
});

/// The tensor-core peak probes (`cuda/sage_peak.cu`), for benchmarks.
pub static PEAK_SOURCE: LazyLock<String> = LazyLock::new(|| {
    [
        include_str!("../../cuda/sage_common.cu"),
        "\n",
        include_str!("../../cuda/sage_peak.cu"),
    ]
    .concat()
});

trait Cuda<T> {
    fn cuda(self, what: &str) -> Result<T>;
}

impl<T, E: std::fmt::Debug> Cuda<T> for std::result::Result<T, E> {
    fn cuda(self, what: &str) -> Result<T> {
        self.map_err(|e| anyhow!("{what}: {e:?}"))
    }
}

/// How one CTA of the attention kernel is shaped. Rows per CTA are
/// `16 × m_tiles × warps`; keys stream through a `stages`-deep ring of
/// 64-key tiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Tile {
    pub warps: u32,
    pub m_tiles: u32,
    pub stages: u32,
}

impl Tile {
    /// Tuned on an RTX 4070 SUPER at head dim 64 (see the tests' benchmark).
    pub const DEFAULT: Tile = Tile {
        warps: 4,
        m_tiles: 2,
        stages: 3,
    };

    /// Head dim 128: two m-tiles per warp would spill (O alone is 128
    /// registers), so one.
    pub const D128: Tile = Tile {
        warps: 4,
        m_tiles: 1,
        stages: 3,
    };

    pub fn for_head_dim(head_dim: usize) -> Self {
        if head_dim > 64 {
            Self::D128
        } else {
            Self::DEFAULT
        }
    }

    pub fn rows(&self) -> usize {
        16 * (self.m_tiles * self.warps) as usize
    }

    fn shared_bytes(&self, head_dim: usize) -> usize {
        self.rows() * head_dim + self.stages as usize * 2 * KBLOCK * head_dim
    }

    fn validate(&self) -> Result<()> {
        if !matches!(self.warps, 2 | 4 | 8)
            || !matches!(self.m_tiles, 1 | 2 | 4)
            || !(2..=6).contains(&self.stages)
            || !QTILE.is_multiple_of(self.rows())
        {
            bail!("unsupported sage tile {self:?}");
        }
        Ok(())
    }
}

/// How each 64-key block's P·V is accumulated before it is folded into the
/// FP32 output accumulator (two-level accumulation either way).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum PvAccum {
    /// FP8 `mma` with FP32 accumulate (SageAttention2).
    #[default]
    F32,
    /// FP8 `mma` with FP16 accumulate (SageAttention2++): twice the tensor
    /// rate on GeForce Ada. V is quantised to ±[`PV16_V_RANGE`] so a block's
    /// partial sums (64 keys, P ≤ 448) stay below the FP16 maximum.
    F16,
}

/// The FP8 range V is scaled to with [`PvAccum::F16`]: 64 · 448 · 2.25 =
/// 64512 < 65504.
pub const PV16_V_RANGE: f32 = 2.25;

/// Kernel configuration: CTA shape and PV accumulation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Config {
    pub tile: Tile,
    pub pv_accum: PvAccum,
}

impl Config {
    pub fn for_head_dim(head_dim: usize) -> Self {
        Self {
            tile: Tile::for_head_dim(head_dim),
            pv_accum: PvAccum::F32,
        }
    }
}

/// Element strides of a bf16 tensor holding `[batch][token][head][dim]`
/// values with contiguous channels. Every stride and the view's start must
/// be a multiple of 8 elements (16-byte vector access).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Strides {
    pub batch: usize,
    pub token: usize,
    pub head: usize,
}

impl Strides {
    /// `[B][T][H][D]`: a `[tokens][heads × dim]` activation, the layout the
    /// output projection reads.
    pub fn bthd(tokens: usize, heads: usize, head_dim: usize) -> Self {
        Self {
            batch: tokens * heads * head_dim,
            token: heads * head_dim,
            head: head_dim,
        }
    }

    /// `[B][H][T][D]`, the layout PyTorch's SDPA takes.
    pub fn bhtd(tokens: usize, heads: usize, head_dim: usize) -> Self {
        Self {
            batch: heads * tokens * head_dim,
            token: head_dim,
            head: tokens * head_dim,
        }
    }

    /// Q, K or V inside one packed QKV GEMM output `[B][T][3][H][D]`; view
    /// the buffer from element 0, `H × D` or `2 × H × D`.
    pub fn packed_qkv(tokens: usize, heads: usize, head_dim: usize) -> Self {
        Self {
            batch: tokens * 3 * heads * head_dim,
            token: 3 * heads * head_dim,
            head: head_dim,
        }
    }

    /// Elements a tensor of this shape spans.
    fn extent(&self, shape: &Shape, head_dim: usize) -> usize {
        (shape.batch - 1) * self.batch
            + (shape.tokens - 1) * self.token
            + (shape.heads - 1) * self.head
            + head_dim
    }

    fn check(&self, what: &str) -> Result<()> {
        if !self.batch.is_multiple_of(8)
            || !self.token.is_multiple_of(8)
            || !self.head.is_multiple_of(8)
        {
            bail!("{what} strides must be multiples of 8 elements: {self:?}");
        }
        Ok(())
    }
}

/// One attention problem: `tokens` queries and keys per batch item and head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub batch: usize,
    pub heads: usize,
    pub tokens: usize,
}

/// How a frame's tokens are laid out: `special` camera and register tokens,
/// then a `grid_h × grid_w` patch grid in row-major order. VGGT-Ω at
/// 512×336: 17 special tokens and a 21 × 32 grid, 689 tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameLayout {
    pub frames: usize,
    pub special: usize,
    pub grid_h: usize,
    pub grid_w: usize,
}

impl FrameLayout {
    pub fn tokens_per_frame(&self) -> usize {
        self.special + self.grid_h * self.grid_w
    }

    /// Keys per frame after `stride × stride` pooling.
    pub fn merged_per_frame(&self, stride: usize) -> usize {
        self.special + self.grid_h.div_ceil(stride) * self.grid_w.div_ceil(stride)
    }
}

/// Which keys and values attention sees.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvMerge {
    /// Every key: the reference schedule.
    Dense,
    /// Patch keys and values pooled in `stride × stride` cells within each
    /// frame (`stride` 2 or 4), with a +log(count) bias per pooled key.
    Pool { stride: usize, layout: FrameLayout },
}

/// Per-call options.
#[derive(Clone, Copy)]
pub struct Options<'a> {
    /// Defaults to `1 / sqrt(head_dim)`.
    pub softmax_scale: Option<f32>,
    /// Subtract the per-channel mean of K before quantising (exact).
    pub smooth_k: bool,
    /// Subtract the per-channel mean of V before quantising and add it back
    /// to the output (exact).
    pub smooth_v: bool,
    pub qk_norm: Option<QkNorm<'a>>,
}

impl Default for Options<'_> {
    fn default() -> Self {
        Self {
            softmax_scale: None,
            smooth_k: true,
            smooth_v: true,
            qk_norm: None,
        }
    }
}

/// The per-head LayerNorm VGGT-Ω applies to Q and K (`q_norm`, `k_norm`),
/// fused into the prologue. `params` holds `[q weight, q bias, k weight,
/// k bias]`, `head_dim` floats each.
#[derive(Clone, Copy)]
pub struct QkNorm<'a> {
    pub params: &'a CudaSlice<f32>,
    pub eps: f32,
}

/// Mirrors `SgGeom` in `cuda/sage_common.cu`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Geom {
    q_sb: i64,
    q_st: i64,
    q_sh: i64,
    k_sb: i64,
    k_st: i64,
    k_sh: i64,
    v_sb: i64,
    v_st: i64,
    v_sh: i64,
    o_sb: i64,
    o_st: i64,
    o_sh: i64,
    batch: i32,
    heads: i32,
    tokens: i32,
    tq_pad: i32,
    tk: i32,
    tk_pad: i32,
    merge: i32,
    frames: i32,
    frame_tokens: i32,
    special: i32,
    grid_h: i32,
    grid_w: i32,
    merged_frame: i32,
    merged_w: i32,
    q_tiles: i32,
    flags: i32,
    q_mul: f32,
    ln_eps: f32,
    v_range: f32,
    pad_: i32,
}

// SAFETY: plain `#[repr(C)]` data with the field order and types of SgGeom.
unsafe impl DeviceRepr for Geom {}

fn int(v: usize, what: &str) -> Result<i32> {
    i32::try_from(v).map_err(|_| anyhow!("{what} = {v} does not fit in i32"))
}

/// Sizes that follow from a [`Shape`] and a [`KvMerge`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    pub shape: Shape,
    pub merge: KvMerge,
    /// Queries per batch item rounded up to [`QTILE`].
    pub tq_pad: usize,
    /// Keys per batch item after merging.
    pub tk: usize,
    /// `tk` rounded up to [`KBLOCK`].
    pub tk_pad: usize,
}

impl Plan {
    pub fn new(shape: Shape, merge: KvMerge) -> Result<Self> {
        if shape.batch == 0 || shape.heads == 0 || shape.tokens == 0 {
            bail!("empty attention shape {shape:?}");
        }
        let tk = match merge {
            KvMerge::Dense => shape.tokens,
            KvMerge::Pool { stride, layout } => {
                if !matches!(stride, 2 | 4) {
                    bail!("KV merge stride must be 2 or 4, got {stride}");
                }
                if layout.grid_h == 0 || layout.grid_w == 0 {
                    bail!("KV merge needs a patch grid: {layout:?}");
                }
                if layout.frames * layout.tokens_per_frame() != shape.tokens {
                    bail!(
                        "frame layout {layout:?} covers {} tokens, the sequence has {}",
                        layout.frames * layout.tokens_per_frame(),
                        shape.tokens
                    );
                }
                layout.frames * layout.merged_per_frame(stride)
            }
        };
        let plan = Self {
            shape,
            merge,
            tq_pad: shape.tokens.next_multiple_of(QTILE),
            tk,
            tk_pad: tk.next_multiple_of(KBLOCK),
        };
        // Everything the kernels index with i32 must fit.
        int(plan.tq_pad, "padded queries")?;
        int(plan.tk_pad, "padded keys")?;
        int(shape.batch * shape.heads, "batch × heads")?;
        Ok(plan)
    }

    /// Keys after merging per key before: 1 when dense.
    pub fn key_ratio(&self) -> f64 {
        self.tk as f64 / self.shape.tokens as f64
    }

    fn q_tiles(&self) -> usize {
        self.tq_pad / QTILE
    }

    fn kblocks(&self) -> usize {
        self.tk_pad / KBLOCK
    }
}

/// Device buffers for one [`Plan`]: the quantised tensors and their scales.
/// Reuse it across layers and calls of the same shape.
pub struct Workspace {
    plan: Plan,
    head_dim: usize,
    /// `[b][h][tq_pad][D]` INT8 Q.
    pub q8: CudaSlice<i8>,
    /// `[b][h][tq_pad / 16]`, softmax scale × log2(e) folded in.
    pub q_scale: CudaSlice<f32>,
    /// `[b][h][tk_pad][D]` INT8 smoothed (and pooled) K.
    pub k8: CudaSlice<i8>,
    /// `[b][h][tk_pad / 64]`.
    pub k_scale: CudaSlice<f32>,
    /// `[b][h][tk_pad / 64][D][64]` E4M3 V, keys permuted within 16.
    pub v8: CudaSlice<u8>,
    /// `[b][h][4][D]`: K mean, V mean, V scale, 1 / V scale.
    pub stats: CudaSlice<f32>,
    /// `[tk_pad]` log2 of each merged key's member count.
    pub kbias: CudaSlice<f32>,
    part: CudaSlice<f32>,
    geom: Option<Geom>,
}

impl Workspace {
    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    /// Device memory this workspace holds.
    pub fn bytes(&self) -> usize {
        self.q8.num_bytes()
            + self.q_scale.num_bytes()
            + self.k8.num_bytes()
            + self.k_scale.num_bytes()
            + self.v8.num_bytes()
            + self.stats.num_bytes()
            + self.kbias.num_bytes()
            + self.part.num_bytes()
    }
}

/// A bf16 input tensor: any device buffer plus its [`Strides`].
pub struct Tensor<'a> {
    pub data: &'a dyn DevicePtr<u16>,
    pub strides: Strides,
}

/// The bf16 output tensor.
pub struct TensorMut<'a> {
    pub data: &'a mut dyn DevicePtrMut<u16>,
    pub strides: Strides,
}

/// Compiled sage kernels for one head dim and [`Config`] on one stream.
/// Every call enqueues work and returns without synchronising.
pub struct SageAttention {
    stream: Arc<CudaStream>,
    head_dim: usize,
    config: Config,
    prep: CudaFunction,
    finalize: CudaFunction,
    quant_kv: CudaFunction,
    dense: CudaFunction,
    merged: CudaFunction,
}

/// PTX per compile-option set, shared across contexts in the process.
static PTX: LazyLock<Mutex<HashMap<Vec<String>, Ptx>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Compiles `source` with NVRTC for compute_89 (FP8 needs Ada or newer; the
/// driver JITs it for newer GPUs) with extra `-D` options, once per process.
pub fn compile(source: &str, defines: &[String]) -> Result<Ptx> {
    // -default-device: generic lambdas in the kernels count as device code.
    let mut options = vec!["--std=c++17".to_string(), "-default-device".to_string()];
    options.extend(defines.iter().cloned());
    let mut cache = PTX.lock().map_err(|_| anyhow!("PTX cache lock poisoned"))?;
    if let Some(p) = cache.get(&options) {
        return Ok(p.clone());
    }
    let opts = CompileOptions {
        arch: Some("compute_89"),
        options: options.clone(),
        ..Default::default()
    };
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(source, opts)
        .map_err(|e| anyhow!("NVRTC compile of the sage kernels failed: {e}"))?;
    // For SASS inspection: `ptxas -arch=sm_89` + `cuobjdump -sass` on the dump.
    if let Ok(dir) = std::env::var("SAGE_DUMP_PTX") {
        let name = options[2..].join("").replace(['-', '='], "");
        let _ = std::fs::write(format!("{dir}/sage{name}.ptx"), ptx.to_src());
    }
    cache.insert(options, ptx.clone());
    Ok(ptx)
}

/// Fails unless the stream's GPU has FP8 tensor cores (compute 8.9+).
pub fn check_device(stream: &CudaStream) -> Result<()> {
    let cc = stream
        .context()
        .compute_capability()
        .cuda("compute capability")?;
    if cc < (8, 9) {
        bail!(
            "sage attention needs FP8 tensor cores (compute capability 8.9 or newer); this GPU is {}.{}",
            cc.0,
            cc.1
        );
    }
    Ok(())
}

impl SageAttention {
    /// Compiles the kernels for `head_dim` (64 or 128) with
    /// [`Config::for_head_dim`]: FP32 PV accumulation, the tuned tile.
    pub fn new(stream: Arc<CudaStream>, head_dim: usize) -> Result<Self> {
        Self::with_config(stream, head_dim, Config::for_head_dim(head_dim))
    }

    pub fn with_config(stream: Arc<CudaStream>, head_dim: usize, config: Config) -> Result<Self> {
        Self::with_defines(stream, head_dim, config, &[])
    }

    /// [`Self::with_config`] plus extra `-D` options for the kernel source
    /// (tuning experiments: `SG_SPLIT`, `SG_FAST_EXP`).
    pub(crate) fn with_defines(
        stream: Arc<CudaStream>,
        head_dim: usize,
        config: Config,
        extra: &[String],
    ) -> Result<Self> {
        if !matches!(head_dim, 64 | 128) {
            bail!("sage attention supports head dim 64 or 128, got {head_dim}");
        }
        let tile = config.tile;
        tile.validate()?;
        check_device(&stream)?;
        let mut defines = vec![
            format!("-DSG_D={head_dim}"),
            format!("-DSG_NW={}", tile.warps),
            format!("-DSG_MT={}", tile.m_tiles),
            format!("-DSG_STAGES={}", tile.stages),
            format!("-DSG_PV16={}", u8::from(config.pv_accum == PvAccum::F16)),
        ];
        defines.extend(extra.iter().cloned());
        let ptx = compile(&SOURCE, &defines)?;
        let module = stream
            .context()
            .load_module(ptx)
            .cuda("load the sage kernels")?;
        let f = |name: &str| module.load_function(name).cuda(name);
        let me = Self {
            prep: f("sg_prep")?,
            finalize: f("sg_finalize")?,
            quant_kv: f("sg_quant_kv")?,
            dense: f("sg_attn_dense")?,
            merged: f("sg_attn_merge")?,
            stream,
            head_dim,
            config,
        };
        let smem = tile.shared_bytes(head_dim) as i32;
        for func in [&me.dense, &me.merged] {
            func.set_attribute(
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                smem,
            )
            .cuda("allow the sage tile's shared memory")?;
        }
        Ok(me)
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn config(&self) -> Config {
        self.config
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Registers per thread and local (spill) bytes of the attention kernel.
    pub fn attn_resources(&self) -> Result<(i32, i32)> {
        Ok((
            self.dense.num_regs().cuda("registers")?,
            self.dense.local_size_bytes().cuda("local memory")?,
        ))
    }

    /// Allocates the buffers for `shape` with `merge`.
    pub fn workspace(&self, shape: Shape, merge: KvMerge) -> Result<Workspace> {
        let plan = Plan::new(shape, merge)?;
        let (b, h, d) = (shape.batch, shape.heads, self.head_dim);
        let s = &self.stream;
        let zeros_i8 = |n: usize| s.alloc_zeros::<i8>(n).cuda("allocate sage workspace");
        let zeros_f32 = |n: usize| s.alloc_zeros::<f32>(n).cuda("allocate sage workspace");
        Ok(Workspace {
            q8: zeros_i8(b * h * plan.tq_pad * d)?,
            q_scale: zeros_f32(b * h * plan.tq_pad / QGROUP)?,
            k8: zeros_i8(b * h * plan.tk_pad * d)?,
            k_scale: zeros_f32(b * h * plan.kblocks())?,
            v8: s
                .alloc_zeros::<u8>(b * h * plan.tk_pad * d)
                .cuda("allocate sage workspace")?,
            stats: zeros_f32(b * h * 4 * d)?,
            kbias: zeros_f32(plan.tk_pad)?,
            part: zeros_f32(b * h * plan.q_tiles() * 4 * d)?,
            geom: None,
            head_dim: d,
            plan,
        })
    }

    /// Quantises (and merges) Q, K and V into `ws`: three launches.
    pub fn prepare(
        &self,
        ws: &mut Workspace,
        q: Tensor<'_>,
        k: Tensor<'_>,
        v: Tensor<'_>,
        opts: &Options<'_>,
    ) -> Result<()> {
        if ws.head_dim != self.head_dim {
            bail!(
                "workspace was made for head dim {}, kernels are {}",
                ws.head_dim,
                self.head_dim
            );
        }
        let plan = ws.plan;
        let shape = plan.shape;
        let d = self.head_dim;
        for (t, what) in [(&q, "Q"), (&k, "K"), (&v, "V")] {
            t.strides.check(what)?;
            let need = t.strides.extent(&shape, d);
            if t.data.len() < need {
                bail!(
                    "{what} holds {} elements, its shape and strides need {need}",
                    t.data.len()
                );
            }
        }
        if let Some(n) = &opts.qk_norm
            && n.params.len() < 4 * d
        {
            bail!(
                "QK-norm params hold {} floats, need {}",
                n.params.len(),
                4 * d
            );
        }
        let (stride, layout) = match plan.merge {
            KvMerge::Dense => (1, None),
            KvMerge::Pool { stride, layout } => (stride, Some(layout)),
        };
        let mut flags = 0;
        if opts.qk_norm.is_some() {
            flags |= LN_Q | LN_K;
        }
        if opts.smooth_k {
            flags |= SMOOTH_K;
        }
        if opts.smooth_v {
            flags |= SMOOTH_V;
        }
        let scale = opts
            .softmax_scale
            .unwrap_or(1.0 / (self.head_dim as f32).sqrt());
        let lay = layout.unwrap_or(FrameLayout {
            frames: 1,
            special: 0,
            grid_h: 0,
            grid_w: 0,
        });
        let geom = Geom {
            q_sb: q.strides.batch as i64,
            q_st: q.strides.token as i64,
            q_sh: q.strides.head as i64,
            k_sb: k.strides.batch as i64,
            k_st: k.strides.token as i64,
            k_sh: k.strides.head as i64,
            v_sb: v.strides.batch as i64,
            v_st: v.strides.token as i64,
            v_sh: v.strides.head as i64,
            batch: int(shape.batch, "batch")?,
            heads: int(shape.heads, "heads")?,
            tokens: int(shape.tokens, "tokens")?,
            tq_pad: int(plan.tq_pad, "padded queries")?,
            tk: int(plan.tk, "keys")?,
            tk_pad: int(plan.tk_pad, "padded keys")?,
            merge: stride as i32,
            frames: int(lay.frames, "frames")?,
            frame_tokens: int(lay.tokens_per_frame(), "frame tokens")?,
            special: int(lay.special, "special tokens")?,
            grid_h: int(lay.grid_h, "grid height")?,
            grid_w: int(lay.grid_w, "grid width")?,
            merged_frame: int(lay.merged_per_frame(stride), "merged frame")?,
            merged_w: int(lay.grid_w.div_ceil(stride), "merged grid width")?,
            q_tiles: int(plan.q_tiles(), "query tiles")?,
            flags,
            q_mul: scale * std::f32::consts::LOG2_E,
            ln_eps: opts.qk_norm.map_or(1e-5, |n| n.eps),
            v_range: match self.config.pv_accum {
                PvAccum::F32 => FP8_MAX,
                PvAccum::F16 => PV16_V_RANGE,
            },
            ..Default::default()
        };

        let s = &*self.stream;
        let mut guards: Vec<SyncOnDrop<'_>> = Vec::with_capacity(4);
        let (qp, g) = q.data.device_ptr(s);
        guards.push(g);
        let (kp, g) = k.data.device_ptr(s);
        guards.push(g);
        let (vp, g) = v.data.device_ptr(s);
        guards.push(g);
        // Null when no LayerNorm runs: the kernels then never read it.
        let ln: u64 = match &opts.qk_norm {
            Some(n) => {
                let (p, g) = n.params.device_ptr(s);
                guards.push(g);
                p
            }
            None => 0,
        };
        let (b, h) = (shape.batch as u32, shape.heads as u32);

        let cfg = LaunchConfig {
            grid_dim: (plan.q_tiles() as u32, h, b),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut l = s.launch_builder(&self.prep);
        l.arg(&qp)
            .arg(&kp)
            .arg(&vp)
            .arg(&ln)
            .arg(&mut ws.q8)
            .arg(&mut ws.q_scale)
            .arg(&mut ws.part)
            .arg(&geom);
        // SAFETY: argument types match sg_prep; bounds were checked above.
        unsafe { l.launch(cfg) }.cuda("launch sg_prep")?;

        let cfg = LaunchConfig {
            grid_dim: (h, b, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut l = s.launch_builder(&self.finalize);
        l.arg(&ws.part).arg(&mut ws.stats).arg(&geom);
        // SAFETY: argument types match sg_finalize.
        unsafe { l.launch(cfg) }.cuda("launch sg_finalize")?;

        let cfg = LaunchConfig {
            grid_dim: (plan.kblocks() as u32, h, b),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut l = s.launch_builder(&self.quant_kv);
        l.arg(&kp)
            .arg(&vp)
            .arg(&ln)
            .arg(&ws.stats)
            .arg(&mut ws.k8)
            .arg(&mut ws.k_scale)
            .arg(&mut ws.v8)
            .arg(&mut ws.kbias)
            .arg(&geom);
        // SAFETY: argument types match sg_quant_kv; bounds were checked above.
        unsafe { l.launch(cfg) }.cuda("launch sg_quant_kv")?;
        drop(guards);
        ws.geom = Some(geom);
        Ok(())
    }

    /// Attention over the tensors [`Self::prepare`] left in `ws`; writes
    /// `[batch][token][head][dim]` bf16 into `out`.
    pub fn attend(&self, ws: &Workspace, out: TensorMut<'_>) -> Result<()> {
        let Some(mut geom) = ws.geom else {
            bail!("sage workspace has not been prepared");
        };
        let plan = ws.plan;
        out.strides.check("output")?;
        let need = out.strides.extent(&plan.shape, self.head_dim);
        if out.data.len() < need {
            bail!(
                "output holds {} elements, its shape and strides need {need}",
                out.data.len()
            );
        }
        geom.o_sb = out.strides.batch as i64;
        geom.o_st = out.strides.token as i64;
        geom.o_sh = out.strides.head as i64;
        let s = &*self.stream;
        let (op, guard) = out.data.device_ptr_mut(s);
        let func = match plan.merge {
            KvMerge::Dense => &self.dense,
            KvMerge::Pool { .. } => &self.merged,
        };
        let cfg = LaunchConfig {
            grid_dim: (
                (plan.tq_pad / self.config.tile.rows()) as u32,
                plan.shape.heads as u32,
                plan.shape.batch as u32,
            ),
            block_dim: (self.config.tile.warps * 32, 1, 1),
            shared_mem_bytes: self.config.tile.shared_bytes(self.head_dim) as u32,
        };
        let mut l = s.launch_builder(func);
        l.arg(&ws.q8)
            .arg(&ws.q_scale)
            .arg(&ws.k8)
            .arg(&ws.k_scale)
            .arg(&ws.v8)
            .arg(&ws.stats)
            .arg(&ws.kbias)
            .arg(&op)
            .arg(&geom);
        // SAFETY: argument types match sg_attn_*; the workspace was sized by
        // its plan and the output extent was checked above.
        unsafe { l.launch(cfg) }.cuda("launch sage attention")?;
        drop(guard);
        Ok(())
    }

    /// [`Self::prepare`] then [`Self::attend`].
    pub fn forward(
        &self,
        ws: &mut Workspace,
        q: Tensor<'_>,
        k: Tensor<'_>,
        v: Tensor<'_>,
        out: TensorMut<'_>,
        opts: &Options<'_>,
    ) -> Result<()> {
        self.prepare(ws, q, k, v, opts)?;
        self.attend(ws, out)
    }
}

/// Multiply-accumulate operations (×2 = FLOPs) of one attention call: QKᵀ
/// and PV over every (query, key) pair, `4 · B · H · Tq · Tk · D` FLOPs.
pub fn flops(plan: &Plan, head_dim: usize) -> f64 {
    4.0 * (plan.shape.batch * plan.shape.heads) as f64
        * plan.shape.tokens as f64
        * plan.tk as f64
        * head_dim as f64
}
