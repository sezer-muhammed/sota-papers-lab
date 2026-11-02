//! Image-space ops on NHWC bf16 tensors (`cuda/conv.cu`), the pieces of a
//! DPT-style dense head that are not a Linear layer:
//!
//! - [`Conv::conv2d`]: implicit-GEMM convolution on `mma.sync` (bf16 in,
//!   fp32 accumulate), any kernel size and stride, `cin % 32 == 0`, with an
//!   optional ReLU on the input, bias and up to two residual adds fused.
//!   1×1 convolutions are Linear layers over pixels: use [`crate::gemm`].
//! - [`Conv::depth_to_space`]: the scatter after a per-pixel GEMM that
//!   implements `ConvTranspose2d(kernel = stride)`, and pixel shuffle.
//! - [`Conv::upsample_bilinear`]: bilinear resize, `align_corners=True`.
//! - [`Conv::add_table`]: per-position table add (positional embeddings),
//!   optionally gathering rows out of a token sequence with a prefix.
//!
//! Tensors are `[n][h][w][c]` bf16 (`u16`); weights `[cout][kh][kw][cin]`.

use cudarc::driver::{
    CudaFunction, DevicePtr, DevicePtrMut, DeviceRepr, LaunchConfig, PushKernelArg,
};
use spl_core::{Result, bail};

use crate::{COMMON_CUH, Cuda, Nn, Ptrs, dim};

const SOURCE: &str = include_str!("../cuda/conv.cu");

/// Convolution geometry. Output size follows `torch.nn.Conv2d`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvShape {
    pub n: usize,
    pub h: usize,
    pub w: usize,
    pub cin: usize,
    pub cout: usize,
    pub kh: usize,
    pub kw: usize,
    pub stride: usize,
    pub pad: usize,
}

impl ConvShape {
    /// Square kernel `k`, padding `k / 2`.
    pub fn same(
        n: usize,
        h: usize,
        w: usize,
        cin: usize,
        cout: usize,
        k: usize,
        stride: usize,
    ) -> Self {
        Self {
            n,
            h,
            w,
            cin,
            cout,
            kh: k,
            kw: k,
            stride,
            pad: k / 2,
        }
    }

    pub fn ho(&self) -> usize {
        (self.h + 2 * self.pad - self.kh) / self.stride + 1
    }

    pub fn wo(&self) -> usize {
        (self.w + 2 * self.pad - self.kw) / self.stride + 1
    }

    /// Output pixels (GEMM M).
    pub fn pixels(&self) -> usize {
        self.n * self.ho() * self.wo()
    }

    pub fn flops(&self) -> f64 {
        2.0 * self.pixels() as f64 * self.cout as f64 * (self.kh * self.kw * self.cin) as f64
    }
}

/// What the convolution epilogue adds to `x * w`.
#[derive(Default)]
pub struct ConvEpilogue<'a> {
    /// f32 `[cout]`.
    pub bias: Option<&'a dyn DevicePtr<f32>>,
    /// bf16 `[pixels][cout]`, added after the bias.
    pub residual: [Option<&'a dyn DevicePtr<u16>>; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct ConvArgs {
    x: u64,
    w: u64,
    bias: u64,
    res0: u64,
    res1: u64,
    y: u64,
    n: i32,
    h: i32,
    w_: i32,
    cin: i32,
    ho: i32,
    wo: i32,
    cout: i32,
    kh: i32,
    kw: i32,
    stride: i32,
    pad: i32,
    pre_relu: i32,
}

// SAFETY: plain old data matching `struct ConvArgs` in cuda/conv.cu.
unsafe impl DeviceRepr for ConvArgs {}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct D2sArgs {
    input: u64,
    bias: u64,
    out: u64,
    n: i32,
    h: i32,
    w: i32,
    c: i32,
    s: i32,
    cmajor: i32,
}

// SAFETY: matches `struct D2sArgs`.
unsafe impl DeviceRepr for D2sArgs {}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct UpArgs {
    input: u64,
    out: u64,
    n: i32,
    h: i32,
    w: i32,
    c: i32,
    ho: i32,
    wo: i32,
}

// SAFETY: matches `struct UpArgs`.
unsafe impl DeviceRepr for UpArgs {}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct TableArgs {
    input: u64,
    table: u64,
    out: u64,
    frames: i32,
    rows: i32,
    start: i32,
    p_count: i32,
    c: i32,
    scale: f32,
}

// SAFETY: matches `struct TableArgs`.
unsafe impl DeviceRepr for TableArgs {}

/// Channel order of a [`Conv::depth_to_space`] input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum D2sOrder {
    /// Channel `(i·s + j)·c + o`: the per-pixel GEMM of a
    /// `ConvTranspose2d` whose weight rows were laid out `(ky, kx, o)`.
    TapMajor,
    /// Channel `o·s·s + i·s + j`: `torch.nn.functional.pixel_shuffle`.
    ChannelMajor,
}

pub struct Conv {
    conv: CudaFunction,
    d2s: CudaFunction,
    up: CudaFunction,
    table: CudaFunction,
}

fn grid_for(nn: &Nn, items: usize) -> LaunchConfig {
    let blocks = items
        .div_ceil(256)
        .clamp(1, nn.multiprocessors() as usize * 16);
    LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

impl Conv {
    pub fn new(nn: &Nn) -> Result<Self> {
        let src = [COMMON_CUH, SOURCE].concat();
        let [conv, d2s, up, table] = nn.functions(
            "spl_nn_conv",
            &src,
            [
                "conv2d_bf16",
                "depth_to_space_bf16",
                "upsample_bilinear_ac_bf16",
                "add_table_bf16",
            ],
        )?;
        Ok(Self {
            conv,
            d2s,
            up,
            table,
        })
    }

    /// `y = conv(pre_relu ? relu(x) : x, w) + bias + residual[0] + residual[1]`.
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d(
        &self,
        nn: &Nn,
        x: &dyn DevicePtr<u16>,
        w: &dyn DevicePtr<u16>,
        epi: &ConvEpilogue<'_>,
        pre_relu: bool,
        y: &mut dyn DevicePtrMut<u16>,
        s: ConvShape,
    ) -> Result<()> {
        if !s.cin.is_multiple_of(32) || !s.cout.is_multiple_of(8) {
            bail!(
                "conv2d: cin ({}) must be a multiple of 32 and cout ({}) of 8",
                s.cin,
                s.cout
            );
        }
        if s.h + 2 * s.pad < s.kh || s.w + 2 * s.pad < s.kw || s.stride == 0 {
            bail!("conv2d: kernel larger than the padded input");
        }
        let m = s.pixels();
        if m == 0 {
            return Ok(());
        }
        let k = s.kh * s.kw * s.cin;
        let mut p = Ptrs::new(&nn.stream);
        let a = ConvArgs {
            x: p.input(x, s.n * s.h * s.w * s.cin, "conv2d x")?,
            w: p.input(w, s.cout * k, "conv2d w")?,
            bias: p.opt(epi.bias, s.cout, "conv2d bias")?,
            res0: p.opt(epi.residual[0], m * s.cout, "conv2d residual 0")?,
            res1: p.opt(epi.residual[1], m * s.cout, "conv2d residual 1")?,
            y: p.output(y, m * s.cout, "conv2d y")?,
            n: dim(s.n, "n")?,
            h: dim(s.h, "h")?,
            w_: dim(s.w, "w")?,
            cin: dim(s.cin, "cin")?,
            ho: dim(s.ho(), "ho")?,
            wo: dim(s.wo(), "wo")?,
            cout: dim(s.cout, "cout")?,
            kh: dim(s.kh, "kh")?,
            kw: dim(s.kw, "kw")?,
            stride: dim(s.stride, "stride")?,
            pad: dim(s.pad, "pad")?,
            pre_relu: pre_relu as i32,
        };
        dim(m * s.cout, "conv2d output elements")?;
        let cfg = LaunchConfig {
            grid_dim: (m.div_ceil(128) as u32, s.cout.div_ceil(128) as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = nn.stream.launch_builder(&self.conv);
        b.arg(&a);
        // SAFETY: `a` matches `ConvArgs`; buffer sizes were checked.
        unsafe { b.launch(cfg) }.cuda("conv2d")?;
        drop(p);
        Ok(())
    }

    /// `[n][h][w][s·s·c]` → `[n][h·s][w·s][c]` (+ bias `[c]`).
    #[allow(clippy::too_many_arguments)]
    pub fn depth_to_space(
        &self,
        nn: &Nn,
        x: &dyn DevicePtr<u16>,
        bias: Option<&dyn DevicePtr<f32>>,
        order: D2sOrder,
        y: &mut dyn DevicePtrMut<u16>,
        n: usize,
        h: usize,
        w: usize,
        c: usize,
        s: usize,
    ) -> Result<()> {
        if !c.is_multiple_of(8) {
            bail!("depth_to_space: c ({c}) must be a multiple of 8");
        }
        let len = n * h * w * s * s * c;
        let mut p = Ptrs::new(&nn.stream);
        let a = D2sArgs {
            input: p.input(x, len, "depth_to_space x")?,
            bias: p.opt(bias, c, "depth_to_space bias")?,
            out: p.output(y, len, "depth_to_space y")?,
            n: dim(n, "n")?,
            h: dim(h, "h")?,
            w: dim(w, "w")?,
            c: dim(c, "c")?,
            s: dim(s, "s")?,
            cmajor: (order == D2sOrder::ChannelMajor) as i32,
        };
        let mut b = nn.stream.launch_builder(&self.d2s);
        b.arg(&a);
        // SAFETY: `a` matches `D2sArgs`; sizes checked.
        unsafe { b.launch(grid_for(nn, len / 8)) }.cuda("depth_to_space")?;
        drop(p);
        Ok(())
    }

    /// Bilinear resize `[n][h][w][c]` → `[n][ho][wo][c]`, `align_corners=True`.
    #[allow(clippy::too_many_arguments)]
    pub fn upsample_bilinear(
        &self,
        nn: &Nn,
        x: &dyn DevicePtr<u16>,
        y: &mut dyn DevicePtrMut<u16>,
        n: usize,
        h: usize,
        w: usize,
        c: usize,
        ho: usize,
        wo: usize,
    ) -> Result<()> {
        if !c.is_multiple_of(8) || h == 0 || w == 0 {
            bail!("upsample_bilinear: c ({c}) must be a multiple of 8, input non-empty");
        }
        let mut p = Ptrs::new(&nn.stream);
        let a = UpArgs {
            input: p.input(x, n * h * w * c, "upsample x")?,
            out: p.output(y, n * ho * wo * c, "upsample y")?,
            n: dim(n, "n")?,
            h: dim(h, "h")?,
            w: dim(w, "w")?,
            c: dim(c, "c")?,
            ho: dim(ho, "ho")?,
            wo: dim(wo, "wo")?,
        };
        let mut b = nn.stream.launch_builder(&self.up);
        b.arg(&a);
        // SAFETY: `a` matches `UpArgs`; sizes checked.
        unsafe { b.launch(grid_for(nn, n * ho * wo * c / 8)) }.cuda("upsample_bilinear")?;
        drop(p);
        Ok(())
    }

    /// `y[f][p] = x[f·rows + start + p] + scale · table[p]` for `p <
    /// p_count`, rows of `c` channels; `table` f32 `[p_count][c]` (or none:
    /// a pure gather). `x` and `y` must not overlap unless `rows == p_count`
    /// and `start == 0`; for an in-place add pass the same buffer as both
    /// via [`Self::add_table_inplace`].
    #[allow(clippy::too_many_arguments)]
    pub fn add_table(
        &self,
        nn: &Nn,
        x: &dyn DevicePtr<u16>,
        table: Option<&dyn DevicePtr<f32>>,
        scale: f32,
        y: &mut dyn DevicePtrMut<u16>,
        frames: usize,
        rows: usize,
        start: usize,
        p_count: usize,
        c: usize,
    ) -> Result<()> {
        if !c.is_multiple_of(8) || start + p_count > rows {
            bail!("add_table: c ({c}) must be a multiple of 8 and start + p_count <= rows");
        }
        let mut p = Ptrs::new(&nn.stream);
        let input = p.input(x, frames * rows * c, "add_table x")?;
        let a = TableArgs {
            input,
            table: p.opt(table, p_count * c, "add_table table")?,
            out: p.output(y, frames * p_count * c, "add_table y")?,
            frames: dim(frames, "frames")?,
            rows: dim(rows, "rows")?,
            start: dim(start, "start")?,
            p_count: dim(p_count, "p_count")?,
            c: dim(c, "c")?,
            scale,
        };
        self.launch_table(nn, a, frames * p_count * c / 8)?;
        drop(p);
        Ok(())
    }

    /// `x[f][p] += scale · table[p]` in place, `x` `[frames][p_count][c]`.
    #[allow(clippy::too_many_arguments)]
    pub fn add_table_inplace(
        &self,
        nn: &Nn,
        x: &mut dyn DevicePtrMut<u16>,
        table: &dyn DevicePtr<f32>,
        scale: f32,
        frames: usize,
        p_count: usize,
        c: usize,
    ) -> Result<()> {
        if !c.is_multiple_of(8) {
            bail!("add_table: c ({c}) must be a multiple of 8");
        }
        let mut p = Ptrs::new(&nn.stream);
        let tab = p.input(table, p_count * c, "add_table table")?;
        let xp = p.output(x, frames * p_count * c, "add_table x")?;
        let a = TableArgs {
            input: xp,
            table: tab,
            out: xp,
            frames: dim(frames, "frames")?,
            rows: dim(p_count, "rows")?,
            start: 0,
            p_count: dim(p_count, "p_count")?,
            c: dim(c, "c")?,
            scale,
        };
        self.launch_table(nn, a, frames * p_count * c / 8)?;
        drop(p);
        Ok(())
    }

    fn launch_table(&self, nn: &Nn, a: TableArgs, items: usize) -> Result<()> {
        let mut b = nn.stream.launch_builder(&self.table);
        b.arg(&a);
        // SAFETY: `a` matches `TableArgs`; sizes checked by the callers.
        unsafe { b.launch(grid_for(nn, items)) }.cuda("add_table")?;
        Ok(())
    }
}

/// CPU reference of [`Conv::conv2d`] in f64 (test oracle).
pub fn conv2d_reference(
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    residual: &[&[f32]],
    pre_relu: bool,
    s: ConvShape,
) -> Vec<f64> {
    let (ho, wo) = (s.ho(), s.wo());
    let k = s.kh * s.kw * s.cin;
    let mut y = vec![0f64; s.pixels() * s.cout];
    for n in 0..s.n {
        for oy in 0..ho {
            for ox in 0..wo {
                let m = (n * ho + oy) * wo + ox;
                for co in 0..s.cout {
                    let mut acc = bias.map_or(0.0, |b| b[co] as f64);
                    for ky in 0..s.kh {
                        let iy = (oy * s.stride + ky) as isize - s.pad as isize;
                        if iy < 0 || iy >= s.h as isize {
                            continue;
                        }
                        for kx in 0..s.kw {
                            let ix = (ox * s.stride + kx) as isize - s.pad as isize;
                            if ix < 0 || ix >= s.w as isize {
                                continue;
                            }
                            let xo = ((n * s.h + iy as usize) * s.w + ix as usize) * s.cin;
                            let wo_ = co * k + (ky * s.kw + kx) * s.cin;
                            for ci in 0..s.cin {
                                let mut v = x[xo + ci];
                                if pre_relu {
                                    v = v.max(0.0);
                                }
                                acc += v as f64 * w[wo_ + ci] as f64;
                            }
                        }
                    }
                    for r in residual {
                        acc += r[m * s.cout + co] as f64;
                    }
                    y[m * s.cout + co] = acc;
                }
            }
        }
    }
    y
}
