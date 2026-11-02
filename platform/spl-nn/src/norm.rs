//! LayerNorm fused with FP8 quantisation (`cuda/norm.cu`), and the row
//! epilogue that finishes a GEMM output.
//!
//! [`Norm::layer_norm`] reads the residual stream once (one warp per token,
//! the row held in registers), writes the normalised row as bf16 and/or FP8
//! E4M3, and for FP8 picks one scale per token (`amax / 448`) or a fixed
//! calibrated scale. [`Norm::residual_layer_norm`] first applies the
//! LayerScale residual update `x += ls ⊙ (y · tok · chan + bias)` and writes
//! `x` back, so the residual add and the next block's LayerNorm cost one
//! pass instead of two.
//!
//! [`Norm::row_epilogue`] applies what cuBLASLt cannot on Ada: per-token
//! activation scales (with per-channel weight scales and bias), then GELU,
//! and re-quantises the row for the next FP8 GEMM. The fc1 → fc2 hand-off
//! needs a quantisation pass anyway, so this costs no extra traffic.

use cudarc::driver::{
    CudaFunction, DevicePtr, DevicePtrMut, DeviceRepr, LaunchConfig, PushKernelArg,
};
use spl_core::{Result, bail};

use crate::{Act, COMMON_CUH, Cuda, Nn, Ptrs, dim};

const SOURCE: &str = include_str!("../cuda/norm.cu");

/// Chunk counts with compiled LayerNorm kernels: C = 256 · n.
const LN_CHUNKS: [usize; 6] = [1, 2, 3, 4, 6, 8];
/// Row-epilogue widths: C = 1024 · n.
const ROW_CHUNKS: [usize; 4] = [1, 2, 3, 4];

/// LayerNorm input: the residual stream in f32 or bf16, `[rows][c]`.
pub enum Rows<'a> {
    F32(&'a dyn DevicePtr<f32>),
    Bf16(&'a dyn DevicePtr<u16>),
}

/// The residual stream when it is updated in place.
pub enum RowsMut<'a> {
    F32(&'a mut dyn DevicePtrMut<f32>),
    Bf16(&'a mut dyn DevicePtrMut<u16>),
}

/// LayerNorm affine parameters (f32, `[c]`) and epsilon.
pub struct LnParams<'a> {
    pub gamma: &'a dyn DevicePtr<f32>,
    pub beta: &'a dyn DevicePtr<f32>,
    pub eps: f32,
}

/// FP8 E4M3 output of a normalisation or epilogue.
pub enum Quant<'a> {
    None,
    /// Dynamic scale per row: `scales[r] = amax_r / 448`, `q = fp8(v / scale)`.
    PerToken {
        q: &'a mut dyn DevicePtrMut<u8>,
        scales: &'a mut dyn DevicePtrMut<f32>,
    },
    /// One fixed scale for every row, read from device memory (`[1]`);
    /// values past `448 · scale` saturate.
    Static {
        q: &'a mut dyn DevicePtrMut<u8>,
        scale: &'a dyn DevicePtr<f32>,
    },
}

/// The residual branch added before normalising:
/// `x += layer_scale ⊙ (y · tok[r] · chan + bias)`; every factor but `y`
/// is optional. `tok` and `chan` are the dequantisation scales of an FP8
/// GEMM whose epilogue could not apply them.
pub struct Branch<'a> {
    pub y: &'a dyn DevicePtr<u16>,
    pub tok: Option<&'a dyn DevicePtr<f32>>,
    pub chan: Option<&'a dyn DevicePtr<f32>>,
    pub bias: Option<&'a dyn DevicePtr<u16>>,
    pub layer_scale: Option<&'a dyn DevicePtr<f32>>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct LnArgs {
    x: u64,
    y: u64,
    ytok: u64,
    ychan: u64,
    ybias: u64,
    ls: u64,
    gamma: u64,
    beta: u64,
    out: u64,
    q: u64,
    qscale: u64,
    static_scale: u64,
    rows: i32,
    eps: f32,
}

// SAFETY: plain old data matching `struct LnArgs` in cuda/norm.cu.
unsafe impl DeviceRepr for LnArgs {}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct RowArgs {
    y: u64,
    ytok: u64,
    ychan: u64,
    bias: u64,
    out: u64,
    q: u64,
    qscale: u64,
    static_scale: u64,
    rows: i32,
    ld: i32,
    act: i32,
    pad: i32,
}

// SAFETY: plain old data matching `struct RowArgs` in cuda/norm.cu.
unsafe impl DeviceRepr for RowArgs {}

pub struct Norm {
    ln_f32: [CudaFunction; 6],
    ln_bf16: [CudaFunction; 6],
    row: [CudaFunction; 4],
}

impl Norm {
    pub fn new(nn: &Nn) -> Result<Self> {
        if !nn.has_fp8() {
            bail!("the normalisation kernels emit FP8 E4M3 and need sm_89 or newer");
        }
        let src = [COMMON_CUH, SOURCE].concat();
        let ln_f32 = nn.functions(
            "spl_nn_norm",
            &src,
            [
                "ln_f32_c1",
                "ln_f32_c2",
                "ln_f32_c3",
                "ln_f32_c4",
                "ln_f32_c6",
                "ln_f32_c8",
            ],
        )?;
        let ln_bf16 = nn.functions(
            "spl_nn_norm",
            &src,
            [
                "ln_bf16_c1",
                "ln_bf16_c2",
                "ln_bf16_c3",
                "ln_bf16_c4",
                "ln_bf16_c6",
                "ln_bf16_c8",
            ],
        )?;
        let row = nn.functions(
            "spl_nn_norm",
            &src,
            ["row_c1", "row_c2", "row_c3", "row_c4"],
        )?;
        Ok(Self {
            ln_f32,
            ln_bf16,
            row,
        })
    }

    /// LayerNorm of `rows` rows of `c` channels (`c` = 256, 512, 768, 1024,
    /// 1536 or 2048), written as bf16 (`out`) and/or FP8 (`quant`).
    #[allow(clippy::too_many_arguments)]
    pub fn layer_norm<'a>(
        &self,
        nn: &'a Nn,
        x: Rows<'a>,
        rows: usize,
        c: usize,
        p: &'a LnParams<'a>,
        out: Option<&'a mut dyn DevicePtrMut<u16>>,
        quant: Quant<'a>,
    ) -> Result<()> {
        let mut ptrs = Ptrs::new(&nn.stream);
        let (xp, xf32) = match x {
            Rows::F32(b) => (ptrs.input(b, rows * c, "layer_norm x")?, true),
            Rows::Bf16(b) => (ptrs.input(b, rows * c, "layer_norm x")?, false),
        };
        let a = LnArgs {
            x: xp,
            ..Default::default()
        };
        self.launch_ln(nn, ptrs, a, xf32, rows, c, p, out, quant)
    }

    /// `x += ls ⊙ (y · tok · chan + bias)` in place, then LayerNorm of the
    /// updated rows as in [`Self::layer_norm`]. A bf16 stream is rounded to
    /// bf16 after the add, and the norm sees the rounded values.
    #[allow(clippy::too_many_arguments)]
    pub fn residual_layer_norm<'a>(
        &self,
        nn: &'a Nn,
        x: RowsMut<'a>,
        branch: &'a Branch<'a>,
        rows: usize,
        c: usize,
        p: &'a LnParams<'a>,
        out: Option<&'a mut dyn DevicePtrMut<u16>>,
        quant: Quant<'a>,
    ) -> Result<()> {
        let mut ptrs = Ptrs::new(&nn.stream);
        let n = rows * c;
        let (xp, xf32) = match x {
            RowsMut::F32(b) => (ptrs.output(b, n, "residual x")?, true),
            RowsMut::Bf16(b) => (ptrs.output(b, n, "residual x")?, false),
        };
        let a = LnArgs {
            x: xp,
            y: ptrs.input(branch.y, n, "residual y")?,
            ytok: ptrs.opt(branch.tok, rows, "residual tok scales")?,
            ychan: ptrs.opt(branch.chan, c, "residual channel scales")?,
            ybias: ptrs.opt(branch.bias, c, "residual bias")?,
            ls: ptrs.opt(branch.layer_scale, c, "layer scale")?,
            ..Default::default()
        };
        self.launch_ln(nn, ptrs, a, xf32, rows, c, p, out, quant)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_ln<'a>(
        &self,
        nn: &'a Nn,
        mut ptrs: Ptrs<'a>,
        mut a: LnArgs,
        xf32: bool,
        rows: usize,
        c: usize,
        p: &'a LnParams<'_>,
        out: Option<&'a mut dyn DevicePtrMut<u16>>,
        quant: Quant<'a>,
    ) -> Result<()> {
        let Some(slot) = LN_CHUNKS.iter().position(|&n| n * 256 == c) else {
            bail!("layer_norm: {c} channels unsupported (256, 512, 768, 1024, 1536 or 2048)");
        };
        if rows == 0 {
            return Ok(());
        }
        a.rows = dim(rows, "rows")?;
        a.eps = p.eps;
        a.gamma = ptrs.input(p.gamma, c, "layer_norm gamma")?;
        a.beta = ptrs.input(p.beta, c, "layer_norm beta")?;
        a.out = ptrs.opt_out(out, rows * c, "layer_norm out")?;
        (a.q, a.qscale, a.static_scale) = quant_ptrs(&mut ptrs, quant, rows, c)?;
        if a.out == 0 && a.q == 0 && a.y == 0 {
            bail!("layer_norm: nothing to write");
        }
        check_aligned(&[
            a.x, a.y, a.ybias, a.ychan, a.ls, a.gamma, a.beta, a.out, a.q,
        ])?;
        let f = if xf32 {
            &self.ln_f32[slot]
        } else {
            &self.ln_bf16[slot]
        };
        let cfg = LaunchConfig {
            grid_dim: (rows.div_ceil(4) as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = nn.stream.launch_builder(f);
        b.arg(&a);
        // SAFETY: `a` matches `LnArgs`; every pointer was checked to cover
        // rows × c (or c, or rows) elements and to be 16-byte aligned.
        unsafe { b.launch(cfg) }.cuda("layer_norm")?;
        drop(ptrs);
        Ok(())
    }

    /// Finishes a GEMM output `y` (`[rows][ld]` bf16, `c` = 1024, 2048, 3072
    /// or 4096 channels used): `v = act(y · tok[r] · chan[c] + bias[c])`,
    /// written back into `y` as bf16 when `in_place` and/or quantised.
    #[allow(clippy::too_many_arguments)]
    pub fn row_epilogue<'a>(
        &self,
        nn: &'a Nn,
        y: &'a mut dyn DevicePtrMut<u16>,
        rows: usize,
        c: usize,
        ld: usize,
        tok: Option<&'a dyn DevicePtr<f32>>,
        chan: Option<&'a dyn DevicePtr<f32>>,
        bias: Option<&'a dyn DevicePtr<u16>>,
        act: Act,
        in_place: bool,
        quant: Quant<'a>,
    ) -> Result<()> {
        let Some(slot) = ROW_CHUNKS.iter().position(|&n| n * 1024 == c) else {
            bail!("row_epilogue: {c} channels unsupported (1024, 2048, 3072 or 4096)");
        };
        if in_place && ld != c {
            bail!("row_epilogue: in-place output needs ld == c");
        }
        if ld < c || !ld.is_multiple_of(8) {
            bail!("row_epilogue: ld {ld} must be a multiple of 8 and at least {c}");
        }
        if rows == 0 {
            return Ok(());
        }
        let mut ptrs = Ptrs::new(&nn.stream);
        let yp = ptrs.output(y, (rows - 1) * ld + c, "row_epilogue y")?;
        let mut a = RowArgs {
            y: yp,
            ytok: ptrs.opt(tok, rows, "row_epilogue tok scales")?,
            ychan: ptrs.opt(chan, c, "row_epilogue channel scales")?,
            bias: ptrs.opt(bias, c, "row_epilogue bias")?,
            out: if in_place { yp } else { 0 },
            rows: dim(rows, "rows")?,
            ld: dim(ld, "ld")?,
            act: act as i32,
            ..Default::default()
        };
        (a.q, a.qscale, a.static_scale) = quant_ptrs(&mut ptrs, quant, rows, c)?;
        if a.out == 0 && a.q == 0 {
            bail!("row_epilogue: nothing to write");
        }
        check_aligned(&[a.y, a.ychan, a.bias, a.q])?;
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = nn.stream.launch_builder(&self.row[slot]);
        b.arg(&a);
        // SAFETY: `a` matches `RowArgs`; buffers were checked for size and
        // 16-byte alignment.
        unsafe { b.launch(cfg) }.cuda("row_epilogue")?;
        drop(ptrs);
        Ok(())
    }
}

fn quant_ptrs<'a>(
    ptrs: &mut Ptrs<'a>,
    quant: Quant<'a>,
    rows: usize,
    c: usize,
) -> Result<(u64, u64, u64)> {
    Ok(match quant {
        Quant::None => (0, 0, 0),
        Quant::PerToken { q, scales } => (
            ptrs.output(q, rows * c, "fp8 out")?,
            ptrs.output(scales, rows, "fp8 scales")?,
            0,
        ),
        Quant::Static { q, scale } => (
            ptrs.output(q, rows * c, "fp8 out")?,
            0,
            ptrs.input(scale, 1, "fp8 static scale")?,
        ),
    })
}

/// The kernels move 16 bytes at a time (8 bytes for FP8).
pub(crate) fn check_aligned(ptrs: &[u64]) -> Result<()> {
    if let Some(p) = ptrs.iter().find(|&&p| !p.is_multiple_of(16)) {
        bail!("device pointer {p:#x} is not 16-byte aligned");
    }
    Ok(())
}
