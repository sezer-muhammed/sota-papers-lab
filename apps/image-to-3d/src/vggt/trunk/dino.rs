//! The patch embedder (spec §3.3): a DINOv3 ViT-L/16 run per frame.
//!
//! `patchify` (im2col) → patch GEMM (+bias) → `[cls, storage 0..3, patches]`
//! rows → 24 blocks with RoPE (prefix 5) → LayerNorm written in fp32
//! straight into the aggregator stream behind the 17 special-token rows
//! (cls and storage outputs are dropped).

use spl_core::Result;
use spl_nn::Nn;
use spl_nn::cudarc::driver::{CudaSlice, CudaView};
use spl_nn::gemm::{Epilogue, GemmShape};
use spl_nn::qkv::{Rope, RopeLayout};

use super::block::{self, BlockWeights, Call, LnWeights};
use super::kernels::RowMap;
use super::{
    Buffers, DENSE, DEPTH, DIM, DINO_PREFIX, LN_EPS, Ops, SPECIAL, TrunkGeom, TrunkWeights,
};

const PE: &str = "aggregator.patch_embed";

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    nn: &Nn,
    ops: &Ops,
    w: &dyn TrunkWeights,
    bufs: &mut Buffers,
    g: TrunkGeom,
    images: &CudaView<'_, f32>,
    prefix: &CudaSlice<f32>,
    rope: &CudaSlice<f32>,
) -> Result<()> {
    let (n, p) = (g.frames, g.patches());
    let len = DINO_PREFIX + p;
    let (x, xd, _, mut s) = bufs.split();

    ops.k
        .patchify(nn, images, &mut *s.big, n, g.grid_h * 16, g.grid_w * 16)?;
    let proj_w = w.bf16(&format!("{PE}.patch_embed.proj.weight"))?;
    let proj_b = w.bf16(&format!("{PE}.patch_embed.proj.bias"))?;
    ops.gemm.linear_bf16(
        nn,
        &*s.big,
        &proj_w,
        Some(&proj_b),
        Epilogue::None,
        &mut *s.o,
        GemmShape::packed(n * p, DIM, 768),
    )?;
    let dm = RowMap::new(len, 0);
    ops.k
        .put_prefix(nn, prefix, &mut *xd, n, DINO_PREFIX, 1, dm)?;
    ops.k
        .put_bf16_rows(nn, &*s.o, &mut *xd, n, p, RowMap::new(len, DINO_PREFIX))?;

    let blocks: Vec<BlockWeights<'_>> = (0..DEPTH)
        .map(|i| BlockWeights::load(w, &format!("{PE}.blocks.{i}")))
        .collect::<Result<_>>()?;
    for i in 0..DEPTH {
        let call = Call {
            n_seq: n,
            seq_len: len,
            rope: Some(Rope {
                table: rope,
                layout: RopeLayout::Dinov3,
                frame_len: len,
            }),
            xn_ready: i > 0,
        };
        let next = blocks.get(i + 1).map(|b| &b.norm1);
        block::forward(nn, ops, &blocks[i], xd, &mut s, &call, &DENSE, next)?;
    }

    let norm = LnWeights::load(w, &format!("{PE}.norm"))?;
    ops.k.ln_rows_f32(
        nn,
        &*xd,
        &mut *x,
        &norm.gamma,
        &norm.beta,
        n,
        p,
        RowMap::new(len, DINO_PREFIX),
        RowMap::new(g.tokens(), SPECIAL),
        LN_EPS,
    )
}
