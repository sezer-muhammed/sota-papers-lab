//! Token assembly and the alternating loop (spec §3.4).
//!
//! Frame `f` holds `[camera, register 0..15, patches]` (frame 0 takes the
//! "first" token variants). Per layer i: `frame_blocks[i]` over each frame
//! with RoPE (prefix 17), then `inter_frame_blocks[i]` either over all
//! `N · T` tokens (global) or only over the `17 N` special tokens gathered
//! into a packed sequence (register; patches untouched). After cached
//! layers the stream goes to the bf16 cache as `[F_i | G_i]`.
//!
//! LN1 of the next block is fused into the previous block's last residual
//! update whenever both run on the same rows (everywhere except around
//! register blocks).

use spl_core::Result;
use spl_nn::Nn;
use spl_nn::cudarc::driver::{CudaSlice, CudaViewMut};
use spl_nn::qkv::{Rope, RopeLayout};

use super::block::{self, BlockWeights, Call};
use super::kernels::RowMap;
use super::{
    Buffers, CACHED, DENSE, DEPTH, DIM, InterAttention, InterKind, Ops, SPECIAL, Tap, TapSink,
    TrunkGeom, TrunkWeights,
};

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    nn: &Nn,
    ops: &Ops,
    w: &dyn TrunkWeights,
    bufs: &mut Buffers,
    g: TrunkGeom,
    special: &CudaSlice<f32>,
    rope: &CudaSlice<f32>,
    global: &dyn InterAttention,
    cache: [&mut CudaViewMut<'_, u16>; 4],
    mut cam_tokens: Option<&mut CudaViewMut<'_, f32>>,
    sched: &[InterKind; DEPTH],
    tap: &mut dyn TapSink,
) -> Result<()> {
    let (n, t) = (g.frames, g.tokens());
    let rows = g.rows();
    let (x, xr, mut s) = bufs.split();
    ops.k
        .put_prefix(nn, special, &mut *x, n, SPECIAL, 2, RowMap::new(t, 0))?;
    if tap.wants(Tap::Input) {
        tap.take(nn, Tap::Input, x, g)?;
    }

    let frame: Vec<BlockWeights<'_>> = (0..DEPTH)
        .map(|i| BlockWeights::load(w, &format!("aggregator.frame_blocks.{i}")))
        .collect::<Result<_>>()?;
    let inter: Vec<BlockWeights<'_>> = (0..DEPTH)
        .map(|i| BlockWeights::load(w, &format!("aggregator.inter_frame_blocks.{i}")))
        .collect::<Result<_>>()?;

    let mut xn_ready = false;
    for i in 0..DEPTH {
        let global_next = sched[i] == InterKind::Global;
        let call = Call {
            n_seq: n,
            seq_len: t,
            rope: Some(Rope {
                table: rope,
                layout: RopeLayout::Dinov3,
                frame_len: t,
            }),
            xn_ready,
        };
        let next = global_next.then_some(&inter[i].norm1);
        block::forward(nn, ops, &frame[i], x, &mut s, &call, &DENSE, next)?;
        let slot = CACHED.iter().position(|&c| c == i);
        if let Some(k) = slot {
            ops.k.to_cache(nn, &*x, &mut *cache[k], rows, 2 * DIM, 0)?;
        }
        if i == DEPTH - 1
            && let Some(c) = cam_tokens.as_deref_mut()
        {
            let tm = RowMap::new(t, 0);
            ops.k.rows_to_wide(nn, &*x, c, n, SPECIAL, tm, 2 * DIM, 0)?;
        }
        if tap.wants(Tap::Frame(i)) {
            tap.take(nn, Tap::Frame(i), x, g)?;
        }

        match sched[i] {
            InterKind::Global => {
                let call = Call {
                    n_seq: 1,
                    seq_len: rows,
                    rope: None,
                    xn_ready: true,
                };
                let following = sched.get(i + 1).map(|_| &frame[i + 1].norm1);
                block::forward(nn, ops, &inter[i], x, &mut s, &call, global, following)?;
                xn_ready = following.is_some();
            }
            InterKind::Register => {
                let (tm, pm) = (RowMap::new(t, 0), RowMap::new(SPECIAL, 0));
                ops.k.copy_rows(nn, &*x, &mut *xr, n, SPECIAL, tm, pm)?;
                let call = Call {
                    n_seq: 1,
                    seq_len: n * SPECIAL,
                    rope: None,
                    xn_ready: false,
                };
                block::forward(nn, ops, &inter[i], xr, &mut s, &call, &DENSE, None)?;
                ops.k.copy_rows(nn, &*xr, &mut *x, n, SPECIAL, pm, tm)?;
                xn_ready = false;
            }
        }
        if let Some(k) = slot {
            ops.k
                .to_cache(nn, &*x, &mut *cache[k], rows, 2 * DIM, DIM)?;
        }
        // The G_23 half of the camera tokens (the F_23 half is written
        // after the frame block above).
        if i == DEPTH - 1
            && let Some(c) = cam_tokens.as_deref_mut()
        {
            let tm = RowMap::new(t, 0);
            ops.k
                .rows_to_wide(nn, &*x, c, n, SPECIAL, tm, 2 * DIM, DIM)?;
        }
        if tap.wants(Tap::Inter(i)) {
            tap.take(nn, Tap::Inter(i), x, g)?;
        }
    }
    Ok(())
}
