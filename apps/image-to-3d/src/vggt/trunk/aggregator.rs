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

use super::block::{self, BlockWeights, Call, Fp8Scratch, Fp8Weights};
use super::kernels::RowMap;
use super::sched::Sage;
use super::{
    Buffers, CACHED, DENSE, DEPTH, DIM, Fp8Blocks, HIDDEN, InterAttention, InterKind, Ops, SPECIAL,
    Tap, TapSink, TrunkGeom, TrunkWeights,
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
    sage: &mut Sage,
    fp8: Fp8Blocks,
    cache: [&mut CudaViewMut<'_, u16>; 4],
    mut cam_tokens: Option<&mut CudaViewMut<'_, f32>>,
    sched: &[InterKind; DEPTH],
    tap: &mut dyn TapSink,
) -> Result<()> {
    let (n, t) = (g.frames, g.tokens());
    let rows = g.rows();
    let (x, xr, [t0, t1], mut s) = bufs.split();
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
    let fp8_of = |on: bool, name: String| on.then(|| Fp8Weights::load(w, &name)).transpose();
    let frame8: Vec<Option<Fp8Weights<'_>>> = (0..DEPTH)
        .map(|i| fp8_of(fp8.frame(i), format!("aggregator.frame_blocks.{i}")))
        .collect::<Result<_>>()?;
    let inter8: Vec<Option<Fp8Weights<'_>>> = (0..DEPTH)
        .map(|i| {
            let on = fp8.inter(i) && sched[i] != InterKind::Register;
            fp8_of(on, format!("aggregator.inter_frame_blocks.{i}"))
        })
        .collect::<Result<_>>()?;
    // Runs one block on the full stream, in FP8 when it has FP8 weights.
    // The FP8 activations borrow `xr` (the DINO stream, idle by now).
    macro_rules! full_block {
        ($bw:expr, $f8:expr, $call:expr, $attn:expr, $next:expr) => {
            match $f8 {
                Some(f8) => {
                    // SAFETY: `xr` holds `rows · 1024` f32, the same bytes
                    // as `rows · 4096` u8, and nothing else uses it during
                    // this block.
                    let mut x8 = unsafe { xr.transmute_mut::<u8>(rows * HIDDEN) }
                        .ok_or_else(|| spl_core::anyhow!("FP8 activation buffer too small"))?;
                    let mut q8 = Fp8Scratch {
                        x8: &mut x8,
                        tok: [&mut *t0, &mut *t1],
                    };
                    block::forward_fp8(nn, ops, $bw, f8, x, &mut s, &mut q8, $call, $attn, $next)
                }
                None => block::forward(nn, ops, $bw, x, &mut s, $call, $attn, $next),
            }
        };
    }

    let mut xn_ready = false;
    for i in 0..DEPTH {
        // Every kind but register reads the full stream, so LN1 fuses.
        let global_next = sched[i] != InterKind::Register;
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
        full_block!(&frame[i], frame8[i].as_ref(), &call, &DENSE, next)?;
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
            kind @ (InterKind::Global
            | InterKind::Frame
            | InterKind::Sage
            | InterKind::Merge2
            | InterKind::Merge4) => {
                // Global blocks have no RoPE; as frame attention they run
                // over each frame's tokens alone.
                let (n_seq, seq_len) = if kind == InterKind::Frame {
                    (n, t)
                } else {
                    (1, rows)
                };
                let call = Call {
                    n_seq,
                    seq_len,
                    rope: None,
                    xn_ready: true,
                };
                let following = sched.get(i + 1).map(|_| &frame[i + 1].norm1);
                let layer;
                let attn: &dyn InterAttention = match kind {
                    InterKind::Global => global,
                    InterKind::Frame => &DENSE,
                    _ => {
                        layer = sage.layer(kind, &g)?;
                        &layer
                    }
                };
                full_block!(&inter[i], inter8[i].as_ref(), &call, attn, following)?;
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
