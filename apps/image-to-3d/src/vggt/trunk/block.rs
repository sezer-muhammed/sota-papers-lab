//! One pre-norm `SelfAttentionBlock` (spec §3.1) on fp32 stream rows:
//!
//! ```text
//! LN1 → QKV GEMM → prologue (QK-norm, RoPE, head-major) → attention
//!     → proj GEMM (+bias) → x += ls1 ⊙ · and LN2 (one kernel)
//!     → fc1 GEMM → bias + erf GELU → fc2 GEMM (+bias)
//!     → x += ls2 ⊙ · (and the next block's LN1 when it reads the same rows)
//! ```
//!
//! cuBLASLt's GELU epilogue is the tanh form, so fc1's bias and the exact
//! erf GELU run in `spl_nn::norm::Norm::row_epilogue`.

use spl_core::Result;
use spl_nn::cudarc::driver::{CudaSlice, CudaView};
use spl_nn::gemm::{Epilogue, GemmShape};
use spl_nn::norm::{Branch, LnParams, Quant, Rows, RowsMut};
use spl_nn::qkv::{NormKind, QkNorm, QkvIn, QkvOut, Rope};
use spl_nn::{Act, Nn};

use super::{DIM, HEADS, HIDDEN, InterAttention, LN_EPS, Ops, TrunkWeights};

/// LayerNorm weights, f32 `[1024]`.
pub struct LnWeights<'a> {
    pub gamma: CudaView<'a, f32>,
    pub beta: CudaView<'a, f32>,
}

impl<'a> LnWeights<'a> {
    pub fn load(w: &'a dyn TrunkWeights, prefix: &str) -> Result<Self> {
        Ok(Self {
            gamma: w.f32(&format!("{prefix}.weight"))?,
            beta: w.f32(&format!("{prefix}.bias"))?,
        })
    }

    fn params(&self) -> LnParams<'_> {
        LnParams {
            gamma: &self.gamma,
            beta: &self.beta,
            eps: LN_EPS,
        }
    }
}

/// Views of one block's weights (checkpoint names under `prefix`).
pub struct BlockWeights<'a> {
    pub norm1: LnWeights<'a>,
    pub qkv_w: CudaView<'a, u16>,
    pub qkv_b: Option<CudaView<'a, u16>>,
    /// QK-norm (q weight, q bias, k weight, k bias), f32 `[64]` each.
    pub qk_norm: Option<[CudaView<'a, f32>; 4]>,
    pub proj_w: CudaView<'a, u16>,
    pub proj_b: CudaView<'a, u16>,
    pub ls1: CudaView<'a, f32>,
    pub norm2: LnWeights<'a>,
    pub fc1_w: CudaView<'a, u16>,
    pub fc1_b: CudaView<'a, u16>,
    pub fc2_w: CudaView<'a, u16>,
    pub fc2_b: CudaView<'a, u16>,
    pub ls2: CudaView<'a, f32>,
}

impl<'a> BlockWeights<'a> {
    pub fn load(w: &'a dyn TrunkWeights, prefix: &str) -> Result<Self> {
        let n = |s: &str| format!("{prefix}.{s}");
        let qk_norm = if w.has(&n("attn.q_norm.weight")) {
            Some([
                w.f32(&n("attn.q_norm.weight"))?,
                w.f32(&n("attn.q_norm.bias"))?,
                w.f32(&n("attn.k_norm.weight"))?,
                w.f32(&n("attn.k_norm.bias"))?,
            ])
        } else {
            None
        };
        Ok(Self {
            norm1: LnWeights::load(w, &n("norm1"))?,
            qkv_w: w.bf16(&n("attn.qkv.weight"))?,
            qkv_b: if w.has(&n("attn.qkv.bias")) {
                Some(w.bf16(&n("attn.qkv.bias"))?)
            } else {
                None
            },
            qk_norm,
            proj_w: w.bf16(&n("attn.proj.weight"))?,
            proj_b: w.bf16(&n("attn.proj.bias"))?,
            ls1: w.f32(&n("ls1.gamma"))?,
            norm2: LnWeights::load(w, &n("norm2"))?,
            fc1_w: w.bf16(&n("mlp.fc1.weight"))?,
            fc1_b: w.bf16(&n("mlp.fc1.bias"))?,
            fc2_w: w.bf16(&n("mlp.fc2.weight"))?,
            fc2_b: w.bf16(&n("mlp.fc2.bias"))?,
            ls2: w.f32(&n("ls2.gamma"))?,
        })
    }
}

/// The bf16 work buffers a block uses (all sized for at least `rows`).
pub(crate) struct Scratch<'b> {
    pub xn: &'b mut CudaSlice<u16>,
    pub big: &'b mut CudaSlice<u16>,
    pub q: &'b mut CudaSlice<u16>,
    pub k: &'b mut CudaSlice<u16>,
    pub v: &'b mut CudaSlice<u16>,
    pub o: &'b mut CudaSlice<u16>,
    pub y: &'b mut CudaSlice<u16>,
}

/// How the block is called.
pub(crate) struct Call<'r> {
    /// `n_seq` independent sequences of `seq_len` rows each.
    pub n_seq: usize,
    pub seq_len: usize,
    pub rope: Option<Rope<'r>>,
    /// `scratch.xn` already holds LN1 of the rows (fused into the previous block).
    pub xn_ready: bool,
}

/// Runs the block on the first `n_seq · seq_len` rows of `x` in place. With
/// `next`, also writes that LayerNorm of the updated rows into `scratch.xn`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn forward(
    nn: &Nn,
    ops: &Ops,
    bw: &BlockWeights<'_>,
    x: &mut CudaSlice<f32>,
    s: &mut Scratch<'_>,
    call: &Call<'_>,
    attn: &dyn InterAttention,
    next: Option<&LnWeights<'_>>,
) -> Result<()> {
    let rows = call.n_seq * call.seq_len;
    if !call.xn_ready {
        let p = bw.norm1.params();
        ops.norm.layer_norm(
            nn,
            Rows::F32(&*x),
            rows,
            DIM,
            &p,
            Some(&mut *s.xn),
            Quant::None,
        )?;
    }
    ops.gemm.linear_bf16(
        nn,
        &*s.xn,
        &bw.qkv_w,
        bw.qkv_b.as_ref().map(|b| b as _),
        Epilogue::None,
        &mut *s.big,
        GemmShape::packed(rows, 3 * DIM, DIM),
    )?;
    let qk = bw.qk_norm.as_ref().map(|[qw, qb, kw, kb]| QkNorm {
        kind: NormKind::LayerNorm,
        q_weight: qw,
        q_bias: Some(qb),
        k_weight: kw,
        k_bias: Some(kb),
        eps: LN_EPS,
    });
    ops.qkv.run(
        nn,
        &QkvIn {
            y: &*s.big,
            ld: 3 * DIM,
            tok: None,
            chan: None,
            bias: None,
        },
        rows,
        HEADS,
        qk.as_ref(),
        call.rope.as_ref(),
        QkvOut {
            q: &mut *s.q,
            k: &mut *s.k,
            v: &mut *s.v,
            v_format: attn.v_format(),
        },
    )?;
    attn.forward(
        nn,
        &ops.fa,
        &*s.q,
        &*s.k,
        &*s.v,
        &mut *s.o,
        call.n_seq,
        call.seq_len,
    )?;
    ops.gemm.linear_bf16(
        nn,
        &*s.o,
        &bw.proj_w,
        Some(&bw.proj_b),
        Epilogue::None,
        &mut *s.y,
        GemmShape::packed(rows, DIM, DIM),
    )?;
    {
        let branch = Branch {
            y: &*s.y,
            tok: None,
            chan: None,
            bias: None,
            layer_scale: Some(&bw.ls1),
        };
        let p = bw.norm2.params();
        ops.norm.residual_layer_norm(
            nn,
            RowsMut::F32(&mut *x),
            &branch,
            rows,
            DIM,
            &p,
            Some(&mut *s.xn),
            Quant::None,
        )?;
    }
    ops.gemm.linear_bf16(
        nn,
        &*s.xn,
        &bw.fc1_w,
        None,
        Epilogue::None,
        &mut *s.big,
        GemmShape::packed(rows, HIDDEN, DIM),
    )?;
    ops.norm.row_epilogue(
        nn,
        &mut *s.big,
        rows,
        HIDDEN,
        HIDDEN,
        None,
        None,
        Some(&bw.fc1_b),
        Act::GeluErf,
        true,
        Quant::None,
    )?;
    ops.gemm.linear_bf16(
        nn,
        &*s.big,
        &bw.fc2_w,
        Some(&bw.fc2_b),
        Epilogue::None,
        &mut *s.y,
        GemmShape::packed(rows, DIM, HIDDEN),
    )?;
    let branch = Branch {
        y: &*s.y,
        tok: None,
        chan: None,
        bias: None,
        layer_scale: Some(&bw.ls2),
    };
    match next {
        Some(ln) => {
            let p = ln.params();
            ops.norm.residual_layer_norm(
                nn,
                RowsMut::F32(&mut *x),
                &branch,
                rows,
                DIM,
                &p,
                Some(&mut *s.xn),
                Quant::None,
            )
        }
        None => ops
            .res
            .add(nn, RowsMut::F32(&mut *x), &branch, rows, DIM, DIM),
    }
}
