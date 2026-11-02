//! Small row ops (`cuda/rowops.cu`) for layers the main kernels do not
//! cover:
//!
//! - [`RowOps::activate`]: in-place activation of a bf16 buffer, for exact
//!   GELU after a cuBLASLt GEMM (whose GELU epilogue is the tanh form) on
//!   widths the row epilogue does not compile (e.g. 8192).
//! - [`RowOps::attention_d128`]: softmax attention with head dimension 128
//!   on short sequences (a few thousand tokens), reading Q, K and V directly
//!   from a packed QKV GEMM output; SIMT fp32 math.

use cudarc::driver::{
    CudaFunction, DevicePtr, DevicePtrMut, DeviceRepr, LaunchConfig, PushKernelArg,
};
use spl_core::{Result, bail};

use crate::{Act, COMMON_CUH, Cuda, Nn, Ptrs, dim};

const SOURCE: &str = include_str!("../cuda/rowops.cu");

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct Attn128Args {
    qkv: u64,
    out: u64,
    ld: i32,
    q_off: i32,
    k_off: i32,
    v_off: i32,
    seq: i32,
    out_ld: i32,
    scale_log2: f32,
}

// SAFETY: matches `struct Attn128Args` in cuda/rowops.cu.
unsafe impl DeviceRepr for Attn128Args {}

/// Where Q, K and V live in a packed `[seq][ld]` bf16 buffer: column offsets
/// of head 0; head `h` adds `h · 128`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedQkv {
    pub ld: usize,
    pub q: usize,
    pub k: usize,
    pub v: usize,
}

impl PackedQkv {
    /// `torch` layout of `Linear(C, 3C)`: q, k, v thirds of the row.
    pub fn thirds(c: usize) -> Self {
        Self {
            ld: 3 * c,
            q: 0,
            k: c,
            v: 2 * c,
        }
    }
}

pub struct RowOps {
    act: CudaFunction,
    attn: CudaFunction,
}

impl RowOps {
    pub fn new(nn: &Nn) -> Result<Self> {
        let src = [COMMON_CUH, SOURCE].concat();
        let [act, attn] = nn.functions("spl_nn_rowops", &src, ["activate_bf16", "attn_d128"])?;
        Ok(Self { act, attn })
    }

    /// `x = act(x)` over `len` bf16 elements (`len % 8 == 0`).
    pub fn activate(
        &self,
        nn: &Nn,
        x: &mut dyn DevicePtrMut<u16>,
        len: usize,
        act: Act,
    ) -> Result<()> {
        if !len.is_multiple_of(8) {
            bail!("activate: len ({len}) must be a multiple of 8");
        }
        if len == 0 || act == Act::None {
            return Ok(());
        }
        let mut p = Ptrs::new(&nn.stream);
        let ptr = p.output(x, len, "activate x")?;
        let chunks = (len / 8) as i64;
        let blocks = (len / 8)
            .div_ceil(256)
            .min(nn.multiprocessors() as usize * 16);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let a = act as i32;
        let mut b = nn.stream.launch_builder(&self.act);
        b.arg(&ptr).arg(&chunks).arg(&a);
        // SAFETY: pointer covers `len` elements.
        unsafe { b.launch(cfg) }.cuda("activate")?;
        drop(p);
        Ok(())
    }

    /// `out[t][h·128..] = softmax(q kᵀ · scale) v` per head over one
    /// sequence of `seq` tokens; `out` is `[seq][out_ld]` bf16.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_d128(
        &self,
        nn: &Nn,
        qkv: &dyn DevicePtr<u16>,
        layout: PackedQkv,
        seq: usize,
        heads: usize,
        scale: f32,
        out: &mut dyn DevicePtrMut<u16>,
        out_ld: usize,
    ) -> Result<()> {
        let width = heads * 128;
        if layout.q.max(layout.k).max(layout.v) + width > layout.ld || out_ld < width {
            bail!("attention_d128: heads × 128 does not fit the row strides");
        }
        if !layout.ld.is_multiple_of(2)
            || !out_ld.is_multiple_of(4)
            || layout.k % 2 + layout.v % 2 != 0
        {
            bail!("attention_d128: offsets and strides must be even (out_ld a multiple of 4)");
        }
        if seq == 0 {
            return Ok(());
        }
        let mut p = Ptrs::new(&nn.stream);
        let a = Attn128Args {
            qkv: p.input(qkv, seq * layout.ld, "attention qkv")?,
            out: p.output(out, (seq - 1) * out_ld + width, "attention out")?,
            ld: dim(layout.ld, "ld")?,
            q_off: dim(layout.q, "q offset")?,
            k_off: dim(layout.k, "k offset")?,
            v_off: dim(layout.v, "v offset")?,
            seq: dim(seq, "seq")?,
            out_ld: dim(out_ld, "out_ld")?,
            scale_log2: scale * std::f32::consts::LOG2_E,
        };
        let cfg = LaunchConfig {
            grid_dim: (seq.div_ceil(16) as u32, heads as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = nn.stream.launch_builder(&self.attn);
        b.arg(&a);
        // SAFETY: `a` matches `Attn128Args`; sizes checked.
        unsafe { b.launch(cfg) }.cuda("attention_d128")?;
        drop(p);
        Ok(())
    }
}
