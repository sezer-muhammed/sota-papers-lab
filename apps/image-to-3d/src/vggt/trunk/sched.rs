//! Phase 2 inter-frame attention (`reports/image-to-3d/spec-attn-schedule.md`):
//! SageAttention (INT8 QKᵀ, FP8 P·V) for global layers, and KV merging
//! (2×2 / 4×4 mean-pooled patch keys and values with a +log(count) logit
//! bias; camera and register tokens and every key of frame 0 stay
//! unmerged) through the same quantised kernel. `Frame` layers need no
//! kernel of their own: the aggregator runs them as frame attention.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use spl_core::{Result, anyhow};
use spl_nn::Nn;
use spl_nn::attention::{FlashAttention, VFormat};
use spl_nn::cudarc::driver::{DevicePtr, DevicePtrMut};
use spl_nn::qkv::HEAD_DIM;
use spl_nn::sage::{
    self, FrameLayout, KvMerge, Options, SageAttention, Shape, Strides, Tensor, TensorMut,
    Workspace,
};

use super::{ATTN_SCALE, DEPTH, HEADS, InterAttention, InterKind, SPECIAL, TrunkGeom};

pub use spl_nn::sage::PvAccum;

/// The KV merge one schedule entry asks for (`None`: not a sage layer).
pub(crate) fn merge_of(kind: InterKind, g: &TrunkGeom) -> Option<KvMerge> {
    let pool = |stride| KvMerge::Pool {
        stride,
        layout: FrameLayout {
            frames: g.frames,
            special: SPECIAL,
            grid_h: g.grid_h,
            grid_w: g.grid_w,
            dense_frames: 1,
        },
    };
    match kind {
        InterKind::Sage => Some(KvMerge::Dense),
        // One frame has nothing to merge: frame 0 stays dense.
        InterKind::Merge2 | InterKind::Merge4 if g.frames == 1 => Some(KvMerge::Dense),
        InterKind::Merge2 => Some(pool(2)),
        InterKind::Merge4 => Some(pool(4)),
        _ => None,
    }
}

fn shape(g: &TrunkGeom) -> Shape {
    Shape {
        batch: 1,
        heads: HEADS,
        tokens: g.rows(),
    }
}

/// Device bytes of one sage workspace (mirrors `SageAttention::workspace`).
pub(crate) fn workspace_bytes(g: &TrunkGeom, merge: KvMerge) -> usize {
    let Ok(p) = sage::Plan::new(shape(g), merge) else {
        return 0;
    };
    let (h, d) = (HEADS, HEAD_DIM);
    let q_tiles = p.tq_pad / sage::QTILE;
    h * p.tq_pad * d
        + h * p.tq_pad / sage::QGROUP * 4
        + 2 * h * p.tk_pad * d
        + h * p.tk_pad / sage::KBLOCK * 4
        + h * 4 * d * 4
        + p.tk_pad * 4
        + h * q_tiles * 4 * d * 4
}

/// Every distinct sage workspace a schedule needs at `g`.
pub(crate) fn merges(sched: &[InterKind; DEPTH], g: &TrunkGeom) -> Vec<KvMerge> {
    let mut out: Vec<KvMerge> = Vec::new();
    for &k in sched {
        if let Some(m) = merge_of(k, g)
            && !out.contains(&m)
        {
            out.push(m);
        }
    }
    out
}

/// The compiled sage kernels and the workspaces of the current geometry.
pub(crate) struct Sage {
    kernels: Option<SageAttention>,
    pv: PvAccum,
    ws: HashMap<(usize, usize, usize, usize), Workspace>,
}

fn ws_key(g: &TrunkGeom, merge: KvMerge) -> (usize, usize, usize, usize) {
    let s = match merge {
        KvMerge::Dense => 1,
        KvMerge::Pool { stride, .. } => stride,
    };
    (g.frames, g.grid_h, g.grid_w, s)
}

impl Sage {
    pub fn new(pv: PvAccum) -> Self {
        Self {
            kernels: None,
            pv,
            ws: HashMap::new(),
        }
    }

    pub fn set_pv_accum(&mut self, pv: PvAccum) {
        if pv != self.pv {
            self.pv = pv;
            self.kernels = None;
        }
    }

    pub fn release(&mut self) {
        self.ws.clear();
    }

    /// Compiles the kernels and allocates the workspaces `sched` needs at
    /// `g` (dropping those of other geometries). Call outside graph capture.
    pub fn prepare(&mut self, nn: &Nn, sched: &[InterKind; DEPTH], g: &TrunkGeom) -> Result<()> {
        let need = merges(sched, g);
        if need.is_empty() {
            return Ok(());
        }
        if self.kernels.is_none() {
            let cfg = sage::Config {
                tile: sage::Tile::for_head_dim(HEAD_DIM),
                pv_accum: self.pv,
            };
            self.kernels = Some(SageAttention::with_config(
                Arc::clone(&nn.stream),
                HEAD_DIM,
                cfg,
            )?);
        }
        let keys: Vec<_> = need.iter().map(|&m| ws_key(g, m)).collect();
        self.ws.retain(|k, _| keys.contains(k));
        let k = self.kernels.as_ref().expect("compiled");
        for m in need {
            if let std::collections::hash_map::Entry::Vacant(e) = self.ws.entry(ws_key(g, m)) {
                e.insert(k.workspace(shape(g), m)?);
            }
        }
        Ok(())
    }

    /// The attention for one sage layer of `kind` at `g`.
    pub fn layer(&mut self, kind: InterKind, g: &TrunkGeom) -> Result<SageLayer<'_>> {
        let merge = merge_of(kind, g).ok_or_else(|| anyhow!("{kind:?} is not a sage layer"))?;
        let kernels = self
            .kernels
            .as_ref()
            .ok_or_else(|| anyhow!("sage kernels not prepared"))?;
        let ws = self
            .ws
            .get_mut(&ws_key(g, merge))
            .ok_or_else(|| anyhow!("sage workspace for {kind:?} at {g:?} not prepared"))?;
        Ok(SageLayer {
            kernels,
            ws: RefCell::new(ws),
        })
    }
}

/// One global layer through SageAttention, reading the head-major bf16
/// Q/K/V the QKV prologue writes (QK-norm already applied).
pub(crate) struct SageLayer<'a> {
    kernels: &'a SageAttention,
    ws: RefCell<&'a mut Workspace>,
}

impl InterAttention for SageLayer<'_> {
    fn v_format(&self) -> VFormat {
        VFormat::Bf16
    }

    fn forward(
        &self,
        _nn: &Nn,
        _fa: &FlashAttention,
        q: &dyn DevicePtr<u16>,
        k: &dyn DevicePtr<u16>,
        v: &dyn DevicePtr<u16>,
        o: &mut dyn DevicePtrMut<u16>,
        n_seq: usize,
        seq_len: usize,
    ) -> Result<()> {
        let mut ws = self.ws.borrow_mut();
        let tokens = n_seq * seq_len;
        if n_seq != 1 || ws.plan().shape.tokens != tokens {
            return Err(anyhow!(
                "sage layer: workspace for {:?}, called with {n_seq} × {seq_len}",
                ws.plan().shape
            ));
        }
        let hm = Strides::bhtd(tokens, HEADS, HEAD_DIM);
        let opts = Options {
            softmax_scale: Some(ATTN_SCALE),
            ..Options::default()
        };
        self.kernels.forward(
            &mut ws,
            Tensor {
                data: q,
                strides: hm,
            },
            Tensor {
                data: k,
                strides: hm,
            },
            Tensor {
                data: v,
                strides: hm,
            },
            TensorMut {
                data: o,
                strides: Strides::bthd(tokens, HEADS, HEAD_DIM),
            },
            &opts,
        )
    }
}
