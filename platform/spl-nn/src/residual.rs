//! LayerScale, bias and residual add in one pass (`cuda/residual.cu`):
//! `x[r][c] += ls[c] · (y[r][c] · tok[r] · chan[c] + bias[c])`.
//!
//! `tok` and `chan` are the per-token and per-channel dequantisation scales
//! of an FP8 projection whose epilogue could not apply them, so the FP8
//! path pays nothing extra for per-token activation scales. When the next
//! op is a LayerNorm, [`crate::norm::Norm::residual_layer_norm`] fuses this
//! update into the norm and saves a pass over the stream.

use cudarc::driver::{CudaFunction, DeviceRepr, LaunchConfig, PushKernelArg};
use spl_core::{Result, bail};

use crate::norm::{Branch, RowsMut, check_aligned};
use crate::{COMMON_CUH, Cuda, Nn, Ptrs, dim};

const SOURCE: &str = include_str!("../cuda/residual.cu");

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct ResArgs {
    x: u64,
    y: u64,
    ytok: u64,
    ychan: u64,
    ybias: u64,
    ls: u64,
    chunks: i64,
    c: i32,
    ld: i32,
}

// SAFETY: plain old data matching `struct ResArgs` in cuda/residual.cu.
unsafe impl DeviceRepr for ResArgs {}

pub struct Residual {
    f32: CudaFunction,
    bf16: CudaFunction,
}

impl Residual {
    pub fn new(nn: &Nn) -> Result<Self> {
        let src = [COMMON_CUH, SOURCE].concat();
        let [f32, bf16] =
            nn.functions("spl_nn_residual", &src, ["residual_f32", "residual_bf16"])?;
        Ok(Self { f32, bf16 })
    }

    /// Updates `x` (`[rows][c]`) in place from `branch.y` (`[rows][ld]`).
    pub fn add(
        &self,
        nn: &Nn,
        x: RowsMut<'_>,
        branch: &Branch<'_>,
        rows: usize,
        c: usize,
        ld: usize,
    ) -> Result<()> {
        if !c.is_multiple_of(8) || !ld.is_multiple_of(8) || ld < c {
            bail!("residual: c ({c}) and ld ({ld}) must be multiples of 8 with ld >= c");
        }
        if rows == 0 || c == 0 {
            return Ok(());
        }
        let mut ptrs = Ptrs::new(&nn.stream);
        let n = rows * c;
        let (xp, f) = match x {
            RowsMut::F32(b) => (ptrs.output(b, n, "residual x")?, &self.f32),
            RowsMut::Bf16(b) => (ptrs.output(b, n, "residual x")?, &self.bf16),
        };
        let a = ResArgs {
            x: xp,
            y: ptrs.input(branch.y, (rows - 1) * ld + c, "residual y")?,
            ytok: ptrs.opt(branch.tok, rows, "residual tok scales")?,
            ychan: ptrs.opt(branch.chan, c, "residual channel scales")?,
            ybias: ptrs.opt(branch.bias, c, "residual bias")?,
            ls: ptrs.opt(branch.layer_scale, c, "layer scale")?,
            chunks: (n / 8) as i64,
            c: dim(c, "c")?,
            ld: dim(ld, "ld")?,
        };
        check_aligned(&[a.x, a.y, a.ychan, a.ybias, a.ls])?;
        let blocks = (n / 8)
            .div_ceil(256)
            .min(nn.multiprocessors() as usize * 16);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = nn.stream.launch_builder(f);
        b.arg(&a);
        // SAFETY: `a` matches `ResArgs`; sizes and alignment were checked.
        unsafe { b.launch(cfg) }.cuda("residual")?;
        drop(ptrs);
        Ok(())
    }
}
