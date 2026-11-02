//! QKV prologue (`cuda/qkv.cu`): one pass from the fused QKV projection to
//! the attention inputs.
//!
//! Per token and head (head dim 64): split q | k | v, apply the optional
//! dequantisation scales and bias of an FP8 projection, QK-norm (LayerNorm
//! or RMSNorm over the head dim, weights shared across heads), 2D RoPE on q
//! and k, and write Q, K and V as `[heads][tokens][64]` bf16. That layout
//! serves frame attention (each frame is a contiguous run of tokens) and
//! global attention (all frames as one sequence) without another copy.
//!
//! RoPE angles come from a host-built [`RopeTable`] indexed by the token's
//! position inside its frame, so the kernel supports VGGT's layout (y on
//! the first half of the head, x on the second, rotate-half inside each)
//! and DINOv3's (y and x periods interleaved, rotate-half over the whole
//! head) and any per-token angles a checkpoint needs.

use std::f64::consts::PI;

use cudarc::driver::{
    CudaFunction, CudaSlice, DevicePtr, DevicePtrMut, DeviceRepr, LaunchConfig, PushKernelArg,
};
use spl_core::{Result, bail};

use crate::norm::check_aligned;
use crate::{COMMON_CUH, Cuda, Nn, Ptrs, dim};

const SOURCE: &str = include_str!("../cuda/qkv.cu");

/// Head dimension of every attention kernel in this crate.
pub const HEAD_DIM: usize = 64;
/// Rotation pairs per head (`HEAD_DIM / 2`).
pub const PAIRS: usize = HEAD_DIM / 2;

/// Which channels RoPE rotates together.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeLayout {
    /// VGGT `RotaryPositionEmbedding2D`: channels 0-31 rotate by the row
    /// position and 32-63 by the column, each half rotate-half style
    /// (pairs `(i, i + 16)`).
    Vggt,
    /// DINOv3 `RopePositionEmbedding`: rotate-half over the whole head
    /// (pairs `(i, i + 32)`), pair `i < 16` driven by the row coordinate and
    /// `i >= 16` by the column.
    Dinov3,
}

impl RopeLayout {
    /// Distance between the two channels of a pair.
    pub fn half(self) -> usize {
        match self {
            Self::Vggt => 16,
            Self::Dinov3 => 32,
        }
    }

    /// The pair (row of the angle table) channel `d` belongs to.
    pub fn pair(self, d: usize) -> usize {
        let h = self.half();
        (d / (2 * h)) * h + d % h
    }
}

/// `cos`/`sin` per (token in frame, pair), as the kernel reads them.
#[derive(Debug, Clone)]
pub struct RopeTable {
    pub layout: RopeLayout,
    pub frame_len: usize,
    /// `frame_len × PAIRS` of `[cos, sin]`.
    pub cos_sin: Vec<[f32; 2]>,
}

impl RopeTable {
    /// Builds a table from per-token angles (`angle(token, pair)` in
    /// radians).
    pub fn from_angles(
        layout: RopeLayout,
        frame_len: usize,
        angle: impl Fn(usize, usize) -> f64,
    ) -> Self {
        let mut cos_sin = Vec::with_capacity(frame_len * PAIRS);
        for t in 0..frame_len {
            for p in 0..PAIRS {
                let a = angle(t, p);
                cos_sin.push([a.cos() as f32, a.sin() as f32]);
            }
        }
        Self {
            layout,
            frame_len,
            cos_sin,
        }
    }

    /// VGGT: `special` leading tokens (camera, registers) at position 0
    /// (no rotation), then a `grid_h × grid_w` patch grid at positions
    /// `(y + 1, x + 1)`; frequency `base^(-i/16)` for pair `i` of each half
    /// (VGGT uses base 100).
    pub fn vggt(grid_h: usize, grid_w: usize, special: usize, base: f64) -> Self {
        Self::from_angles(RopeLayout::Vggt, special + grid_h * grid_w, |t, p| {
            if t < special {
                return 0.0;
            }
            let (y, x) = ((t - special) / grid_w, (t - special) % grid_w);
            let pos = if p < 16 { y + 1 } else { x + 1 } as f64;
            pos * base.powf(-((p % 16) as f64) / 16.0)
        })
    }

    /// DINOv3: `special` leading tokens unrotated, patch coordinates
    /// normalised to `[-1, 1]` per axis (`(i + 0.5) / n · 2 - 1`), period
    /// `base^(j/16)` for pair `j` of each axis (DINOv3 uses base 100).
    pub fn dinov3(grid_h: usize, grid_w: usize, special: usize, base: f64) -> Self {
        Self::from_angles(RopeLayout::Dinov3, special + grid_h * grid_w, |t, p| {
            if t < special {
                return 0.0;
            }
            let (y, x) = ((t - special) / grid_w, (t - special) % grid_w);
            let coord = if p < 16 {
                (y as f64 + 0.5) / grid_h as f64 * 2.0 - 1.0
            } else {
                (x as f64 + 0.5) / grid_w as f64 * 2.0 - 1.0
            };
            2.0 * PI * coord / base.powf((p % 16) as f64 / 16.0)
        })
    }

    pub fn upload(&self, nn: &Nn) -> Result<CudaSlice<f32>> {
        let flat: Vec<f32> = self.cos_sin.iter().flatten().copied().collect();
        nn.stream.clone_htod(&flat).cuda("upload RoPE table")
    }
}

/// QK-norm flavour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormKind {
    LayerNorm = 1,
    Rms = 2,
}

/// QK-norm parameters, f32 `[64]` each, shared across heads. Biases are
/// used by LayerNorm only.
pub struct QkNorm<'a> {
    pub kind: NormKind,
    pub q_weight: &'a dyn DevicePtr<f32>,
    pub q_bias: Option<&'a dyn DevicePtr<f32>>,
    pub k_weight: &'a dyn DevicePtr<f32>,
    pub k_bias: Option<&'a dyn DevicePtr<f32>>,
    pub eps: f32,
}

/// RoPE on the device: the uploaded table and its layout.
pub struct Rope<'a> {
    pub table: &'a dyn DevicePtr<f32>,
    pub layout: RopeLayout,
    pub frame_len: usize,
}

/// The fused projection output `[tokens][ld]` (q | k | v, `heads · 64`
/// channels each) and its optional per-token scales, per-channel scales
/// (`[3 · heads · 64]`) and bias.
pub struct QkvIn<'a> {
    pub y: &'a dyn DevicePtr<u16>,
    pub ld: usize,
    pub tok: Option<&'a dyn DevicePtr<f32>>,
    pub chan: Option<&'a dyn DevicePtr<f32>>,
    pub bias: Option<&'a dyn DevicePtr<u16>>,
}

/// Element format of V for attention: bf16, or fp16 for the attention
/// tiles that run P·V on fp16-accumulating tensor cores
/// ([`crate::attention::Tile::pv16`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VFormat {
    Bf16,
    F16,
}

/// Head-major attention inputs, `[heads][tokens][64]` each: Q and K in
/// bf16, V in `v_format`.
pub struct QkvOut<'a> {
    pub q: &'a mut dyn DevicePtrMut<u16>,
    pub k: &'a mut dyn DevicePtrMut<u16>,
    pub v: &'a mut dyn DevicePtrMut<u16>,
    pub v_format: VFormat,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct QkvArgs {
    y: u64,
    ytok: u64,
    ychan: u64,
    bias: u64,
    qn_w: u64,
    qn_b: u64,
    kn_w: u64,
    kn_b: u64,
    rope: u64,
    q: u64,
    k: u64,
    v: u64,
    tokens: i64,
    ld: i32,
    heads: i32,
    frame_len: i32,
    norm: i32,
    rope_half: i32,
    eps: f32,
    v_f16: i32,
    pad: i32,
}

// SAFETY: plain old data matching `struct QkvArgs` in cuda/qkv.cu.
unsafe impl DeviceRepr for QkvArgs {}

pub struct QkvPrologue {
    f: CudaFunction,
}

impl QkvPrologue {
    pub fn new(nn: &Nn) -> Result<Self> {
        let src = [COMMON_CUH, SOURCE].concat();
        let [f] = nn.functions("spl_nn_qkv", &src, ["qkv_prologue"])?;
        Ok(Self { f })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        nn: &Nn,
        input: &QkvIn<'_>,
        tokens: usize,
        heads: usize,
        norm: Option<&QkNorm<'_>>,
        rope: Option<&Rope<'_>>,
        out: QkvOut<'_>,
    ) -> Result<()> {
        let width = 3 * heads * HEAD_DIM;
        if input.ld < width || !input.ld.is_multiple_of(8) {
            bail!(
                "qkv: ld {} must be a multiple of 8 and at least {width}",
                input.ld
            );
        }
        if tokens == 0 || heads == 0 {
            return Ok(());
        }
        let mut ptrs = Ptrs::new(&nn.stream);
        let n = tokens * heads * HEAD_DIM;
        let mut a = QkvArgs {
            y: ptrs.input(input.y, (tokens - 1) * input.ld + width, "qkv y")?,
            ytok: ptrs.opt(input.tok, tokens, "qkv tok scales")?,
            ychan: ptrs.opt(input.chan, width, "qkv channel scales")?,
            bias: ptrs.opt(input.bias, width, "qkv bias")?,
            q: ptrs.output(out.q, n, "qkv q")?,
            k: ptrs.output(out.k, n, "qkv k")?,
            v: ptrs.output(out.v, n, "qkv v")?,
            tokens: tokens as i64,
            ld: dim(input.ld, "ld")?,
            heads: dim(heads, "heads")?,
            frame_len: 1,
            v_f16: (out.v_format == VFormat::F16) as i32,
            ..Default::default()
        };
        if let Some(nm) = norm {
            a.norm = nm.kind as i32;
            a.eps = nm.eps;
            a.qn_w = ptrs.input(nm.q_weight, HEAD_DIM, "q norm weight")?;
            a.kn_w = ptrs.input(nm.k_weight, HEAD_DIM, "k norm weight")?;
            if nm.kind == NormKind::LayerNorm {
                let (Some(qb), Some(kb)) = (nm.q_bias, nm.k_bias) else {
                    bail!("qkv: LayerNorm QK-norm needs q and k biases");
                };
                a.qn_b = ptrs.input(qb, HEAD_DIM, "q norm bias")?;
                a.kn_b = ptrs.input(kb, HEAD_DIM, "k norm bias")?;
            }
        }
        if let Some(r) = rope {
            if r.frame_len == 0 {
                bail!("qkv: RoPE frame length is 0");
            }
            a.rope = ptrs.input(r.table, r.frame_len * PAIRS * 2, "RoPE table")?;
            a.rope_half = r.layout.half() as i32;
            a.frame_len = dim(r.frame_len, "frame_len")?;
        }
        check_aligned(&[a.y, a.ychan, a.bias, a.q, a.k, a.v])?;
        let rows = tokens * 3 * heads;
        let blocks = rows.div_ceil(32).min(nn.multiprocessors() as usize * 64);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = nn.stream.launch_builder(&self.f);
        b.arg(&a);
        // SAFETY: `a` matches `QkvArgs`; sizes and alignment were checked.
        unsafe { b.launch(cfg) }.cuda("qkv_prologue")?;
        drop(ptrs);
        Ok(())
    }
}

/// CPU reference of [`QkvPrologue::run`] in f64 on f32 inputs: returns
/// (q, k, v) as `[heads][tokens][64]`.
#[allow(clippy::too_many_arguments)]
pub fn reference(
    y: &[f32],
    ld: usize,
    tokens: usize,
    heads: usize,
    tok: Option<&[f32]>,
    chan: Option<&[f32]>,
    bias: Option<&[f32]>,
    norm: Option<(NormKind, [&[f32]; 4], f32)>,
    rope: Option<&RopeTable>,
) -> [Vec<f64>; 3] {
    let n = tokens * heads * HEAD_DIM;
    let mut out = [vec![0.0f64; n], vec![0.0f64; n], vec![0.0f64; n]];
    for t in 0..tokens {
        for (part, o) in out.iter_mut().enumerate() {
            for h in 0..heads {
                let ch0 = (part * heads + h) * HEAD_DIM;
                let mut x: Vec<f64> = (0..HEAD_DIM)
                    .map(|d| {
                        let c = ch0 + d;
                        let mut v = y[t * ld + c] as f64;
                        v *= tok.map_or(1.0, |s| s[t] as f64);
                        v *= chan.map_or(1.0, |s| s[c] as f64);
                        v + bias.map_or(0.0, |b| b[c] as f64)
                    })
                    .collect();
                if part < 2 {
                    if let Some((kind, [qw, qb, kw, kb], eps)) = norm {
                        let (w, b) = if part == 0 { (qw, qb) } else { (kw, kb) };
                        let mean = if kind == NormKind::LayerNorm {
                            x.iter().sum::<f64>() / HEAD_DIM as f64
                        } else {
                            0.0
                        };
                        let var =
                            x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / HEAD_DIM as f64;
                        let r = 1.0 / (var + eps as f64).sqrt();
                        for d in 0..HEAD_DIM {
                            let bias = if kind == NormKind::LayerNorm {
                                b[d] as f64
                            } else {
                                0.0
                            };
                            x[d] = (x[d] - mean) * r * w[d] as f64 + bias;
                        }
                    }
                    if let Some(rt) = rope {
                        let half = rt.layout.half();
                        let row = &rt.cos_sin[(t % rt.frame_len) * PAIRS..];
                        let src = x.clone();
                        for d in 0..HEAD_DIM {
                            let [c, s] = row[rt.layout.pair(d)];
                            let p = src[d ^ half];
                            x[d] = if d & half == 0 {
                                src[d] * c as f64 - p * s as f64
                            } else {
                                src[d] * c as f64 + p * s as f64
                            };
                        }
                    }
                }
                o[(h * tokens + t) * HEAD_DIM..(h * tokens + t + 1) * HEAD_DIM].copy_from_slice(&x);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_pair_the_right_channels() {
        let v = RopeLayout::Vggt;
        assert_eq!((v.pair(0), v.pair(16), v.pair(15)), (0, 0, 15));
        assert_eq!((v.pair(32), v.pair(48), v.pair(63)), (16, 16, 31));
        let d = RopeLayout::Dinov3;
        assert_eq!(
            (d.pair(0), d.pair(32), d.pair(31), d.pair(63)),
            (0, 0, 31, 31)
        );
        for l in [v, d] {
            for c in 0..HEAD_DIM {
                assert_eq!(l.pair(c), l.pair(c ^ l.half()), "{l:?} {c}");
            }
        }
    }

    #[test]
    fn vggt_table_matches_its_formula() {
        // 2 special tokens, a 3 × 4 grid.
        let t = RopeTable::vggt(3, 4, 2, 100.0);
        assert_eq!(t.frame_len, 14);
        assert_eq!(
            t.cos_sin[PAIRS + 5],
            [1.0, 0.0],
            "special tokens do not rotate"
        );
        // Token 2 + 1·4 + 3 = patch (y=1, x=3) → positions (2, 4).
        let row = &t.cos_sin[9 * PAIRS..10 * PAIRS];
        let a = 2.0 * 100f64.powf(-3.0 / 16.0);
        assert!((row[3][0] as f64 - a.cos()).abs() < 1e-7);
        assert!((row[3][1] as f64 - a.sin()).abs() < 1e-7);
        let a = 4.0 * 100f64.powf(-5.0 / 16.0);
        assert!((row[16 + 5][1] as f64 - a.sin()).abs() < 1e-7);
    }

    #[test]
    fn dinov3_table_matches_its_formula() {
        let t = RopeTable::dinov3(2, 2, 1, 100.0);
        // Patch (0, 1): coords (-0.5, 0.5).
        let row = &t.cos_sin[2 * PAIRS..3 * PAIRS];
        let a = 2.0 * PI * -0.5;
        assert!((row[0][1] as f64 - a.sin()).abs() < 1e-7);
        let a = 2.0 * PI * 0.5 / 100f64.powf(2.0 / 16.0);
        assert!((row[18][0] as f64 - a.cos()).abs() < 1e-7);
    }

    #[test]
    fn reference_rotation_preserves_pair_norms() {
        let tokens = 3;
        let heads = 2;
        let ld = 3 * heads * HEAD_DIM;
        let y: Vec<f32> = (0..tokens * ld)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
            .collect();
        let rope = RopeTable::vggt(1, 2, 1, 100.0);
        let [q0, _, v0] = reference(&y, ld, tokens, heads, None, None, None, None, None);
        let [q1, _, v1] = reference(&y, ld, tokens, heads, None, None, None, None, Some(&rope));
        assert_eq!(v0, v1, "V is never rotated");
        for i in 0..tokens * heads {
            let a = &q0[i * HEAD_DIM..(i + 1) * HEAD_DIM];
            let b = &q1[i * HEAD_DIM..(i + 1) * HEAD_DIM];
            for d in 0..16 {
                let n0 = a[d].powi(2) + a[d + 16].powi(2);
                let n1 = b[d].powi(2) + b[d + 16].powi(2);
                assert!(
                    (n0 - n1).abs() <= 1e-6 * n0.max(1.0),
                    "f32 cos/sin keep norms"
                );
            }
        }
        // Token 0 is special: unrotated.
        assert_eq!(&q0[..HEAD_DIM], &q1[..HEAD_DIM]);
    }
}
