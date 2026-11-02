//! The Gaussian store: fixed-capacity SoA parameters on the device, their host
//! mirror, gradient buffers, capacity planning, compaction with emission and
//! Morton re-sorting. Owner: densify (E6). Layout frozen by spec-raster 1.1
//! (planar SH rest) and spec-train 1, 5.8-5.10, 13.
//!
//! Device layout, capacity `cap` (a multiple of 1024), live count `n`:
//!
//! | Buffer | Shape | Element of Gaussian p |
//! |---|---|---|
//! | `mean` | f32 `[cap][3]` | `3p + axis` |
//! | `log_scale` | f32 `[cap][3]` | `3p + axis`; s = exp(ŝ) |
//! | `quat` | f32 `[cap][4]` | `4p + (w, x, y, z)`; raw, normalized on use; one 128-bit load |
//! | `opacity` | f32 `[cap]` | `p`; logit |
//! | `sh_dc` | f32 `[cap][3]` | `3p + c` |
//! | `sh_rest` | f32 `[45][cap]` | `(3k + c)·cap + p`, k = basis − 1 (0..14) |
//! | `filter3d` | f32 `[cap]`, optional | `p`; Mip-Splatting 3D filter variance |
//! | `last_visible` | u32 `[cap]` | `p`; iteration that last touched ≥ 1 tile |
//!
//! Gradients ([`Grads`]) use the identical index math; Adam moments are
//! `float2 (m, v)` at the same element index (`kernels::adam::Moments`). The
//! host mirror [`GaussiansHost`] is compact (`n`, not `cap`) and keeps SH rest
//! per Gaussian as `[n][15][3]` (SPZ / Faster-GS order); upload and download
//! transpose it.

use std::marker::PhantomData;
use std::ops::Range;

use anyhow::ensure;
use spl_core::{Result, bail};
use spl_gpu::cudarc::driver::{
    CudaFunction, CudaSlice, CudaStream, DevicePtrMut, DeviceRepr, LaunchConfig, PushKernelArg,
    ValidAsZeroBits,
};

use crate::gpu::{CudaResultExt, Defines, Gpu};
use crate::kernels::adam::Moments;
use crate::kernels::densify::{COMPACT_SOURCE, DensifyStats};
use crate::rng::Philox;

/// Optimizer parameter groups (spec-train 4.2), in `GROUP_*` order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Group {
    Means,
    Scales,
    Rotations,
    Opacities,
    Sh0,
    ShRest,
}

impl Group {
    pub const ALL: [Group; 6] = [
        Group::Means,
        Group::Scales,
        Group::Rotations,
        Group::Opacities,
        Group::Sh0,
        Group::ShRest,
    ];

    /// Floats per Gaussian: 3, 3, 4, 1, 3, 45 (59 in all).
    pub const fn floats(self) -> usize {
        match self {
            Group::Means | Group::Scales | Group::Sh0 => 3,
            Group::Rotations => 4,
            Group::Opacities => 1,
            Group::ShRest => 45,
        }
    }

    /// Index into per-group arrays (`GROUP_*` in adam.cuh).
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// Floats of parameters per Gaussian.
pub const FLOATS_PER_GAUSSIAN: usize = 59;

/// Device parameters (raster spec names; spec-train's `means`, `scales`, ...
/// are the same buffers).
pub struct Gaussians {
    /// Live count. Kernels read it from `FrameParams.n_gaussians`, never `cap`.
    pub n: u32,
    pub cap: u32,
    pub mean: CudaSlice<f32>,
    pub log_scale: CudaSlice<f32>,
    pub quat: CudaSlice<f32>,
    pub opacity: CudaSlice<f32>,
    pub sh_dc: CudaSlice<f32>,
    pub sh_rest: CudaSlice<f32>,
    pub filter3d: Option<CudaSlice<f32>>,
    pub last_visible: CudaSlice<u32>,
}

/// Host mirror for init, snapshots, export and tests: compact (`n` Gaussians),
/// SH rest per Gaussian as `[15][3]` (coefficient-major, channel-minor).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GaussiansHost {
    pub means: Vec<f32>,
    /// Log scales.
    pub scales: Vec<f32>,
    /// Raw quaternions (w, x, y, z).
    pub rotations: Vec<f32>,
    /// Logits.
    pub opacities: Vec<f32>,
    pub sh0: Vec<f32>,
    pub sh_rest: Vec<f32>,
    pub sh_degree: u32,
    pub antialiased: bool,
}

impl GaussiansHost {
    pub fn len(&self) -> usize {
        self.opacities.len()
    }

    pub fn is_empty(&self) -> bool {
        self.opacities.is_empty()
    }

    /// `n` Gaussians at the origin, unit scale, identity rotation, zero colour.
    pub fn zeros(n: usize) -> Self {
        let mut rotations = vec![0.0; n * 4];
        rotations.iter_mut().step_by(4).for_each(|w| *w = 1.0);
        Self {
            means: vec![0.0; n * 3],
            scales: vec![0.0; n * 3],
            rotations,
            opacities: vec![0.0; n],
            sh0: vec![0.0; n * 3],
            sh_rest: vec![0.0; n * 45],
            sh_degree: 3,
            antialiased: false,
        }
    }

    pub fn validate(&self) -> Result<()> {
        let n = self.len();
        for (name, len, per) in [
            ("means", self.means.len(), 3),
            ("scales", self.scales.len(), 3),
            ("rotations", self.rotations.len(), 4),
            ("sh0", self.sh0.len(), 3),
            ("sh_rest", self.sh_rest.len(), 45),
        ] {
            ensure!(
                len == n * per,
                "GaussiansHost.{name} has {len} floats, expected {}",
                n * per
            );
        }
        ensure!(self.sh_degree <= 3, "sh_degree {} > 3", self.sh_degree);
        Ok(())
    }
}

/// Rounds a capacity up to a multiple of 1024 (keeps planar sub-arrays 4 KB aligned).
pub const fn round_cap(n: usize) -> usize {
    n.div_ceil(1024) * 1024
}

impl Gaussians {
    pub fn alloc(gpu: &Gpu, capacity: usize, filter3d: bool) -> Result<Self> {
        let cap = round_cap(capacity.max(1));
        ensure!(cap <= u32::MAX as usize / 45, "capacity {cap} too large");
        let s = &gpu.stream;
        let z = |len: usize, what: &str| s.alloc_zeros::<f32>(len).cuda(what);
        Ok(Self {
            n: 0,
            cap: cap as u32,
            mean: z(cap * 3, "alloc mean")?,
            log_scale: z(cap * 3, "alloc log_scale")?,
            quat: z(cap * 4, "alloc quat")?,
            opacity: z(cap, "alloc opacity")?,
            sh_dc: z(cap * 3, "alloc sh_dc")?,
            sh_rest: z(cap * 45, "alloc sh_rest")?,
            filter3d: if filter3d {
                Some(z(cap, "alloc filter3d")?)
            } else {
                None
            },
            last_visible: s.alloc_zeros::<u32>(cap).cuda("alloc last_visible")?,
        })
    }

    /// Replaces the live Gaussians with `host` (n ≤ cap). `last_visible` is reset to 0.
    pub fn upload(&mut self, gpu: &Gpu, host: &GaussiansHost) -> Result<()> {
        host.validate()?;
        let n = host.len();
        let cap = self.cap as usize;
        if n > cap {
            bail!("{n} Gaussians do not fit capacity {cap}");
        }
        let s = &gpu.stream;
        s.memcpy_htod(&host.means, &mut self.mean.slice_mut(..n * 3))
            .cuda("upload mean")?;
        s.memcpy_htod(&host.scales, &mut self.log_scale.slice_mut(..n * 3))
            .cuda("upload log_scale")?;
        s.memcpy_htod(&host.rotations, &mut self.quat.slice_mut(..n * 4))
            .cuda("upload quat")?;
        s.memcpy_htod(&host.opacities, &mut self.opacity.slice_mut(..n))
            .cuda("upload opacity")?;
        s.memcpy_htod(&host.sh0, &mut self.sh_dc.slice_mut(..n * 3))
            .cuda("upload sh_dc")?;
        let planar = sh_rest_to_planar(&host.sh_rest, n);
        for j in 0..45 {
            s.memcpy_htod(
                &planar[j * n..(j + 1) * n],
                &mut self.sh_rest.slice_mut(j * cap..j * cap + n),
            )
            .cuda("upload sh_rest")?;
        }
        s.memset_zeros(&mut self.last_visible)
            .cuda("reset last_visible")?;
        self.n = n as u32;
        Ok(())
    }

    /// The live Gaussians as a host mirror (synchronizes the stream).
    pub fn download(&self, gpu: &Gpu) -> Result<GaussiansHost> {
        let n = self.n as usize;
        let cap = self.cap as usize;
        let s = &gpu.stream;
        let get = |src: &CudaSlice<f32>, len: usize, what: &str| -> Result<Vec<f32>> {
            let mut out = vec![0f32; len];
            if len > 0 {
                s.memcpy_dtoh(&src.slice(..len), &mut out).cuda(what)?;
            }
            Ok(out)
        };
        let mut planar = vec![0f32; n * 45];
        if n > 0 {
            for j in 0..45 {
                s.memcpy_dtoh(
                    &self.sh_rest.slice(j * cap..j * cap + n),
                    &mut planar[j * n..(j + 1) * n],
                )
                .cuda("download sh_rest")?;
            }
        }
        let out = GaussiansHost {
            means: get(&self.mean, n * 3, "download mean")?,
            scales: get(&self.log_scale, n * 3, "download log_scale")?,
            rotations: get(&self.quat, n * 4, "download quat")?,
            opacities: get(&self.opacity, n, "download opacity")?,
            sh0: get(&self.sh_dc, n * 3, "download sh_dc")?,
            sh_rest: sh_rest_from_planar(&planar, n),
            sh_degree: 3,
            antialiased: self.filter3d.is_some(),
        };
        gpu.sync()?;
        Ok(out)
    }

    pub fn param(&self, g: Group) -> &CudaSlice<f32> {
        match g {
            Group::Means => &self.mean,
            Group::Scales => &self.log_scale,
            Group::Rotations => &self.quat,
            Group::Opacities => &self.opacity,
            Group::Sh0 => &self.sh_dc,
            Group::ShRest => &self.sh_rest,
        }
    }

    pub fn param_mut(&mut self, g: Group) -> &mut CudaSlice<f32> {
        match g {
            Group::Means => &mut self.mean,
            Group::Scales => &mut self.log_scale,
            Group::Rotations => &mut self.quat,
            Group::Opacities => &mut self.opacity,
            Group::Sh0 => &mut self.sh_dc,
            Group::ShRest => &mut self.sh_rest,
        }
    }
}

/// `[n][45]` (per Gaussian) to `[45][n]` (planar, stride n).
pub fn sh_rest_to_planar(per_gaussian: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0f32; n * 45];
    for (p, coeffs) in per_gaussian.as_chunks::<45>().0.iter().enumerate() {
        for (j, &x) in coeffs.iter().enumerate() {
            out[j * n + p] = x;
        }
    }
    out
}

/// `[45][n]` (planar, stride n) to `[n][45]`.
pub fn sh_rest_from_planar(planar: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0f32; n * 45];
    for j in 0..45 {
        for p in 0..n {
            out[p * 45 + j] = planar[j * n + p];
        }
    }
    out
}

/// One f32 gradient buffer per group, same shapes and index math as [`Gaussians`].
pub struct Grads {
    pub cap: u32,
    pub mean: CudaSlice<f32>,
    pub log_scale: CudaSlice<f32>,
    pub quat: CudaSlice<f32>,
    pub opacity: CudaSlice<f32>,
    pub sh_dc: CudaSlice<f32>,
    pub sh_rest: CudaSlice<f32>,
}

impl Grads {
    pub fn alloc(gpu: &Gpu, capacity: usize) -> Result<Self> {
        let cap = round_cap(capacity.max(1));
        let s = &gpu.stream;
        let z = |len: usize, what: &str| s.alloc_zeros::<f32>(len).cuda(what);
        Ok(Self {
            cap: cap as u32,
            mean: z(cap * 3, "alloc grad mean")?,
            log_scale: z(cap * 3, "alloc grad log_scale")?,
            quat: z(cap * 4, "alloc grad quat")?,
            opacity: z(cap, "alloc grad opacity")?,
            sh_dc: z(cap * 3, "alloc grad sh_dc")?,
            sh_rest: z(cap * 45, "alloc grad sh_rest")?,
        })
    }

    pub fn get(&self, g: Group) -> &CudaSlice<f32> {
        match g {
            Group::Means => &self.mean,
            Group::Scales => &self.log_scale,
            Group::Rotations => &self.quat,
            Group::Opacities => &self.opacity,
            Group::Sh0 => &self.sh_dc,
            Group::ShRest => &self.sh_rest,
        }
    }

    pub fn get_mut(&mut self, g: Group) -> &mut CudaSlice<f32> {
        match g {
            Group::Means => &mut self.mean,
            Group::Scales => &mut self.log_scale,
            Group::Rotations => &mut self.quat,
            Group::Opacities => &mut self.opacity,
            Group::Sh0 => &mut self.sh_dc,
            Group::ShRest => &mut self.sh_rest,
        }
    }

    pub fn zero(&mut self, gpu: &Gpu, g: Group) -> Result<()> {
        gpu.stream
            .memset_zeros(self.get_mut(g))
            .cuda("zero gradients")
    }
}

/// Bytes per Gaussian of capacity (spec-train 13): parameters 236, Adam
/// moments 472 (236 bf16), gradients 236, last_visible 4, densify stats 12,
/// VCD scores 8, raster per-Gaussian buffers ~80, compaction slab 180, radix
/// sort pairs 16 and the emission plan 10 (out_count, two flags, priority).
/// MCMC adds 24 of its own scratch (`densify::McmcWork`), not counted here.
pub const fn bytes_per_gaussian(bf16_moments: bool) -> u64 {
    236 + if bf16_moments { 236 } else { 472 } + 236 + 4 + 12 + 8 + 80 + 180 + 16 + 10
}

/// Capacity from free VRAM (spec-train 5.10):
/// `min(preset cap, (free − images − raster reserve − 512 MiB) / bytes_per_gaussian)`,
/// rounded down to a multiple of 1024.
pub fn plan_capacity(
    free_vram: u64,
    images: u64,
    raster_reserve: u64,
    preset_cap: u64,
    bf16_moments: bool,
) -> u64 {
    let budget = free_vram.saturating_sub(images + raster_reserve + (512 << 20));
    let c = (budget / bytes_per_gaussian(bf16_moments)).min(preset_cap);
    c / 1024 * 1024
}

/// One run of `words` u32 per Gaussian that compaction and Morton sorting move
/// as a unit: an interleaved array (`mean`, words 3) or one row of a planar
/// array (`sh_rest` row j, words 1). Mirrors `Column` in compact.cu / mcmc.cu.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Column {
    /// Device address of Gaussian 0's first word.
    pub ptr: u64,
    pub words: u32,
    /// [`COL_COPY`] or [`COL_ZERO`]: what clones, split children and MCMC copies get.
    pub kind: u32,
}

// SAFETY: plain #[repr(C)] data (a device address and two u32).
unsafe impl DeviceRepr for Column {}
// SAFETY: all-zero bits are a valid (null, empty) column.
unsafe impl ValidAsZeroBits for Column {}

/// New entries copy the parent (parameters, `last_visible`).
pub const COL_COPY: u32 = 0;
/// New entries start at zero (moments, gradients, statistics).
pub const COL_ZERO: u32 = 1;

/// u32 words per Gaussian one slab batch may hold (`cap × 180 B`, spec-train 5.8).
pub const SLAB_WORDS: u32 = 45;
/// Gaussians per tile of the scan and radix kernels (`CT_TILE`).
pub const TILE: u32 = 2048;
/// Column descriptors the compactor can hold (params 52, grads 50, moments 50, stats 3).
pub const MAX_COLUMNS: usize = 256;

/// Every buffer indexed by Gaussian besides the parameters (gradients,
/// moments, statistics), as compaction, Morton sorting and MCMC moves them.
/// Holds the `&mut` borrows for as long as the columns are in use.
#[derive(Default)]
pub struct PerGaussianArrays<'a> {
    pub cols: Vec<Column>,
    _borrows: PhantomData<&'a mut ()>,
}

impl<'a> PerGaussianArrays<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    /// `rows` columns of `words` u32 each; row r starts `r · row_stride` words in.
    pub fn push<T: DeviceRepr>(
        &mut self,
        s: &CudaStream,
        buf: &'a mut CudaSlice<T>,
        words: u32,
        rows: u32,
        row_stride: u32,
        kind: u32,
    ) {
        let (ptr, _sync) = buf.device_ptr_mut(s);
        for r in 0..rows as u64 {
            self.cols.push(Column {
                ptr: ptr + r * row_stride as u64 * 4,
                words,
                kind,
            });
        }
    }

    /// Gradient buffers: survivors keep their partial sums, new entries start at 0.
    pub fn add_grads(&mut self, s: &CudaStream, g: &'a mut Grads) {
        self.add_grad_groups(s, g, [true; 6]);
    }

    /// Only the gradient groups in `groups` (`Group::index` order). After the
    /// optimizer step, a group that stepped (or is overwritten by the raster
    /// every iteration) holds nothing a later iteration reads, so emission can
    /// skip it; only accumulating groups between steps carry partial sums.
    pub fn add_grad_groups(&mut self, s: &CudaStream, g: &'a mut Grads, groups: [bool; 6]) {
        let cap = g.cap;
        let Grads {
            mean,
            log_scale,
            quat,
            opacity,
            sh_dc,
            sh_rest,
            ..
        } = g;
        let bufs: [(&'a mut CudaSlice<f32>, u32, u32, u32); 6] = [
            (mean, 3, 1, 0),
            (log_scale, 3, 1, 0),
            (quat, 4, 1, 0),
            (opacity, 1, 1, 0),
            (sh_dc, 3, 1, 0),
            (sh_rest, 1, 45, cap),
        ];
        for ((buf, words, rows, stride), on) in bufs.into_iter().zip(groups) {
            if on {
                self.push(s, buf, words, rows, stride, COL_ZERO);
            }
        }
    }

    /// Adam moments of every group, any precision (words per element are
    /// read off the buffer size: 2 for f32 (m, v), 1 for packed bf16).
    pub fn add_moments(&mut self, s: &CudaStream, m: &'a mut Moments, cap: u32) {
        for (g, buf) in Group::ALL.into_iter().zip(m.groups.iter_mut()) {
            let floats = g.floats() as u32;
            let wpe = (buf.len() / (cap as usize * floats as usize)).max(1) as u32;
            if g == Group::ShRest {
                self.push(s, buf, wpe, 45, cap * wpe, COL_ZERO);
            } else {
                self.push(s, buf, floats * wpe, 1, 0, COL_ZERO);
            }
        }
    }

    pub fn add_stats(&mut self, s: &CudaStream, st: &'a mut DensifyStats) {
        self.push(s, &mut st.vis_count, 1, 1, 0, COL_ZERO);
        self.push(s, &mut st.grad_sum, 1, 1, 0, COL_ZERO);
        if let Some(a) = st.grad_abs_sum.as_mut() {
            self.push(s, a, 1, 1, 0, COL_ZERO);
        }
    }

    /// Total u32 words per Gaussian.
    pub fn words(&self) -> u32 {
        self.cols.iter().map(|c| c.words).sum()
    }
}

/// The parameter columns of `g` (all [`COL_COPY`]), in a fixed order:
/// mean, log_scale, quat, opacity, sh_dc, 45 sh_rest rows, filter3d, last_visible.
pub fn gaussian_columns(s: &CudaStream, g: &mut Gaussians) -> Vec<Column> {
    let cap = g.cap;
    let mut a = PerGaussianArrays::new();
    a.push(s, &mut g.mean, 3, 1, 0, COL_COPY);
    a.push(s, &mut g.log_scale, 3, 1, 0, COL_COPY);
    a.push(s, &mut g.quat, 4, 1, 0, COL_COPY);
    a.push(s, &mut g.opacity, 1, 1, 0, COL_COPY);
    a.push(s, &mut g.sh_dc, 3, 1, 0, COL_COPY);
    a.push(s, &mut g.sh_rest, 1, 45, cap, COL_COPY);
    if let Some(f) = g.filter3d.as_mut() {
        a.push(s, f, 1, 1, 0, COL_COPY);
    }
    a.push(s, &mut g.last_visible, 1, 1, 0, COL_COPY);
    a.cols
}

/// Consecutive runs of columns with at most [`SLAB_WORDS`] words each.
pub fn slab_batches(cols: &[Column]) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let (mut a, mut w) = (0, 0);
    for (i, c) in cols.iter().enumerate() {
        if w + c.words > SLAB_WORDS && i > a {
            out.push(a..i);
            (a, w) = (i, 0);
        }
        w += c.words;
    }
    if a < cols.len() {
        out.push(a..cols.len());
    }
    out
}

/// Prune / clone / split decisions of one densify step, on the pre-step
/// indexing (spec-train 5.8), written by `densify_plan`.
pub struct EmitPlan {
    pub n_before: u32,
    /// out_count[i]: 0 = pruned or split parent, 1 = kept, 2 = FastGS prune
    /// candidate VCP did not draw (kept).
    pub out_count: CudaSlice<u32>,
    pub clone_flag: CudaSlice<u8>,
    pub split_flag: CudaSlice<u8>,
    /// Densify statistic for capping when the tail does not fit (descending).
    pub priority: CudaSlice<f32>,
}

impl EmitPlan {
    pub fn alloc(gpu: &Gpu, capacity: usize) -> Result<Self> {
        let cap = round_cap(capacity.max(1));
        let s = &gpu.stream;
        Ok(Self {
            n_before: 0,
            out_count: s.alloc_zeros::<u32>(cap).cuda("alloc out_count")?,
            clone_flag: s.alloc_zeros::<u8>(cap).cuda("alloc clone_flag")?,
            split_flag: s.alloc_zeros::<u8>(cap).cuda("alloc split_flag")?,
            priority: s.alloc_zeros::<f32>(cap).cuda("alloc priority")?,
        })
    }
}

/// What one emission did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EmitResult {
    pub n_before: u32,
    pub n_after: u32,
    pub survivors: u32,
    pub clones: u32,
    pub splits: u32,
    /// Tail entries dropped because the capacity was full.
    pub capped: u32,
}

pub struct CompactKernels {
    pub scan_u32: CudaFunction,
    pub emit_count: CudaFunction,
    pub emit_dst: CudaFunction,
    pub slab_load: CudaFunction,
    pub slab_scatter: CudaFunction,
    pub slab_gather: CudaFunction,
    pub slab_store: CudaFunction,
    pub split_children: CudaFunction,
    pub aabb_reduce: CudaFunction,
    pub morton_codes: CudaFunction,
    pub radix_hist: CudaFunction,
    pub radix_scan_rows: CudaFunction,
    pub radix_scatter: CudaFunction,
}

impl CompactKernels {
    pub fn load(gpu: &Gpu) -> Result<Self> {
        let k = gpu.kernels(&COMPACT_SOURCE, &Defines::new())?;
        Ok(Self {
            scan_u32: k.get("scan_u32")?,
            emit_count: k.get("emit_count")?,
            emit_dst: k.get("emit_dst")?,
            slab_load: k.get("slab_load")?,
            slab_scatter: k.get("slab_scatter")?,
            slab_gather: k.get("slab_gather")?,
            slab_store: k.get("slab_store")?,
            split_children: k.get("split_children")?,
            aabb_reduce: k.get("aabb_reduce")?,
            morton_codes: k.get("morton_codes")?,
            radix_hist: k.get("radix_hist")?,
            radix_scan_rows: k.get("radix_scan_rows")?,
            radix_scatter: k.get("radix_scatter")?,
        })
    }
}

fn blocks(n: u32, per: u32) -> u32 {
    n.div_ceil(per).max(1)
}

fn cfg(grid: (u32, u32), block: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (grid.0, grid.1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Order-preserving in-place compaction with emission, and the Morton
/// re-sort, through one `cap × 180 B` slab (spec-train 5.8-5.9). Also the
/// stable radix sort FastGS's Gumbel-top-k uses.
pub struct Compactor {
    pub capacity: u32,
    pub slab: CudaSlice<u32>,
    pub kernels: CompactKernels,
    /// Ping-pong (key, value) buffers of the radix sort; the sorted pairs end in
    /// `[0]`. Emission reuses `keys[0]` as dst and `vals[0]` as tail_dst.
    pub keys: [CudaSlice<u32>; 2],
    pub vals: [CudaSlice<u32>; 2],
    /// Per-tile partial sums (3 × tiles) or radix histograms (256 × tiles).
    pub part: CudaSlice<u32>,
    /// survivors, clones, splits, scratch.
    pub totals: CudaSlice<u32>,
    /// Digit totals of the radix passes, `[4][256]`.
    pub digit_tot: CudaSlice<u32>,
    pub aabb: CudaSlice<u32>,
    pub cols: CudaSlice<Column>,
}

impl Compactor {
    pub fn new(gpu: &Gpu, capacity: usize) -> Result<Self> {
        let cap = round_cap(capacity.max(1));
        let s = &gpu.stream;
        let tiles = cap.div_ceil(TILE as usize);
        let z = |len: usize, what: &str| s.alloc_zeros::<u32>(len).cuda(what);
        Ok(Self {
            capacity: cap as u32,
            slab: z(cap * SLAB_WORDS as usize, "alloc compaction slab")?,
            kernels: CompactKernels::load(gpu)?,
            keys: [z(cap, "alloc sort keys")?, z(cap, "alloc sort keys")?],
            vals: [z(cap, "alloc sort values")?, z(cap, "alloc sort values")?],
            part: z(256 * tiles, "alloc tile partials")?,
            totals: z(4, "alloc emission totals")?,
            digit_tot: z(4 * 256, "alloc digit totals")?,
            aabb: z(6, "alloc AABB")?,
            cols: s
                .alloc_zeros::<Column>(MAX_COLUMNS)
                .cuda("alloc column table")?,
        })
    }

    /// Bytes on the device besides the slab's 180 per Gaussian (16 per Gaussian).
    pub fn vram_bytes(&self) -> u64 {
        let c = self.capacity as u64;
        c * (SLAB_WORDS as u64 * 4 + 16) + 256 * 4 * c.div_ceil(TILE as u64)
    }

    /// Exclusive scan of `len` words of `part` starting at `at`; the sum goes to totals[slot].
    fn scan_part(&mut self, gpu: &Gpu, at: usize, len: u32, slot: usize) -> Result<()> {
        let mut data = self.part.slice_mut(at..at + len as usize);
        let mut tot = self.totals.slice_mut(slot..slot + 1);
        let mut l = gpu.stream.launch_builder(&self.kernels.scan_u32);
        l.arg(&mut data).arg(&len).arg(&mut tot);
        // SAFETY: scan_u32(uint32_t* data, uint32_t count, uint32_t* total), one block of 1024.
        unsafe { l.launch(cfg((1, 1), 1024)) }.cuda("scan_u32")?;
        Ok(())
    }

    fn count_and_scan(&mut self, gpu: &Gpu, plan: &EmitPlan) -> Result<[u32; 3]> {
        let n = plan.n_before;
        let tiles = blocks(n, TILE);
        let mut l = gpu.stream.launch_builder(&self.kernels.emit_count);
        l.arg(&plan.out_count)
            .arg(&plan.clone_flag)
            .arg(&plan.split_flag)
            .arg(&n)
            .arg(&mut self.part);
        // SAFETY: emit_count(out_count, clone_flag, split_flag, n, part[3][tiles]), one block per tile.
        unsafe { l.launch(cfg((tiles, 1), 256)) }.cuda("emit_count")?;
        for c in 0..3 {
            self.scan_part(gpu, c * tiles as usize, tiles, c)?;
        }
        let t = gpu
            .stream
            .clone_dtoh(&self.totals.slice(0..3))
            .cuda("read emission totals")?;
        Ok([t[0], t[1], t[2]])
    }

    /// Applies `plan` to the parameters and every array in `extra`; split
    /// children get their Philox offsets (RNG_SPLIT at iteration `t`). Sets
    /// `g.n`. One host sync (the totals); a second pass only when capping.
    pub fn emit(
        &mut self,
        gpu: &Gpu,
        plan: &mut EmitPlan,
        g: &mut Gaussians,
        extra: &PerGaussianArrays<'_>,
        rng: Philox,
        t: u32,
    ) -> Result<EmitResult> {
        let n = plan.n_before;
        let mut r = EmitResult {
            n_before: n,
            ..Default::default()
        };
        if n == 0 {
            return Ok(r);
        }
        ensure!(
            n <= self.capacity && n <= g.cap,
            "emit: n {n} over capacity"
        );
        let mut tot = self.count_and_scan(gpu, plan)?;
        let cap = self.capacity.min(g.cap);
        if tot[0] + tot[1] + 2 * tot[2] > cap {
            r.capped = cap_tail(gpu, plan, cap, tot)?;
            tot = self.count_and_scan(gpu, plan)?;
        }
        (r.survivors, r.clones, r.splits) = (tot[0], tot[1], tot[2]);
        r.n_after = tot[0] + tot[1] + 2 * tot[2];
        let s = gpu.stream.clone();
        let tiles = blocks(n, TILE);
        {
            let [dst, _] = &mut self.keys;
            let [tail, _] = &mut self.vals;
            let mut l = s.launch_builder(&self.kernels.emit_dst);
            l.arg(&plan.out_count)
                .arg(&plan.clone_flag)
                .arg(&plan.split_flag)
                .arg(&n)
                .arg(&self.part)
                .arg(&self.totals)
                .arg(dst)
                .arg(tail);
            // SAFETY: emit_dst(out_count, clone_flag, split_flag, n, part, totals, dst, tail_dst), one block per tile.
            unsafe { l.launch(cfg((tiles, 1), 256)) }.cuda("emit_dst")?;
        }
        let mut cols = gaussian_columns(&s, g);
        let (mean, scale, quat) = (cols[0].ptr, cols[1].ptr, cols[2].ptr);
        cols.extend_from_slice(&extra.cols);
        self.upload_columns(gpu, &cols)?;
        for b in slab_batches(&cols) {
            self.move_batch(gpu, &cols, b, n, BatchOp::Emit(&plan.split_flag))?;
        }
        if r.splits > 0 {
            let mut l = s.launch_builder(&self.kernels.split_children);
            let seed = Uint2::from(rng.seed);
            l.arg(&plan.split_flag)
                .arg(&self.vals[0])
                .arg(&n)
                .arg(&seed)
                .arg(&t)
                .arg(&mean)
                .arg(&scale)
                .arg(&quat);
            // SAFETY: split_children(split_flag, tail_dst, n, uint2 seed, iteration, float* mean,
            // float* log_scale, const float4* quat); the three addresses are g's live buffers.
            unsafe { l.launch(LaunchConfig::for_num_elems(n)) }.cuda("split_children")?;
        }
        g.n = r.n_after;
        Ok(r)
    }

    fn upload_columns(&mut self, gpu: &Gpu, cols: &[Column]) -> Result<()> {
        ensure!(
            cols.len() <= MAX_COLUMNS,
            "{} columns > {MAX_COLUMNS}",
            cols.len()
        );
        gpu.stream
            .memcpy_htod(cols, &mut self.cols.slice_mut(..cols.len()))
            .cuda("upload column table")
    }

    fn move_batch(
        &mut self,
        gpu: &Gpu,
        cols: &[Column],
        b: Range<usize>,
        n: u32,
        op: BatchOp<'_>,
    ) -> Result<()> {
        let s = &gpu.stream;
        let maxw = cols[b.clone()].iter().map(|c| c.words).max().unwrap_or(1);
        let grid = (blocks(n * maxw, 256 * 4).min(1024), b.len() as u32);
        let table = self.cols.slice(b);
        match op {
            BatchOp::Emit(split) => {
                let mut l = s.launch_builder(&self.kernels.slab_load);
                l.arg(&table).arg(&n).arg(&mut self.slab);
                // SAFETY: slab_load(const Column*, n, slab); batch words <= SLAB_WORDS so n * words fits.
                unsafe { l.launch(cfg(grid, 256)) }.cuda("slab_load")?;
                let mut l = s.launch_builder(&self.kernels.slab_scatter);
                l.arg(&table)
                    .arg(&n)
                    .arg(&self.slab)
                    .arg(&self.keys[0])
                    .arg(&self.vals[0])
                    .arg(split);
                // SAFETY: slab_scatter(cols, n, slab, dst, tail_dst, split_flag); destinations < n_after <= cap.
                unsafe { l.launch(cfg(grid, 256)) }.cuda("slab_scatter")?;
            }
            BatchOp::Gather => {
                let mut l = s.launch_builder(&self.kernels.slab_gather);
                l.arg(&table).arg(&n).arg(&self.vals[0]).arg(&mut self.slab);
                // SAFETY: slab_gather(cols, n, perm, slab); perm is a permutation of 0..n.
                unsafe { l.launch(cfg(grid, 256)) }.cuda("slab_gather")?;
                let mut l = s.launch_builder(&self.kernels.slab_store);
                l.arg(&table).arg(&n).arg(&self.slab);
                // SAFETY: slab_store(cols, n, slab): the inverse copy of slab_load.
                unsafe { l.launch(cfg(grid, 256)) }.cuda("slab_store")?;
            }
        }
        Ok(())
    }

    /// Stable LSD radix sort of `(keys[0], vals[0])[..n]` by the low `bits`
    /// bits (8 per pass); the result ends in `keys[0]`, `vals[0]`.
    pub fn sort(&mut self, gpu: &Gpu, n: u32, bits: u32) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let s = gpu.stream.clone();
        let tiles = blocks(n, TILE);
        s.memset_zeros(&mut self.digit_tot)
            .cuda("zero digit totals")?;
        for pass in 0..bits.div_ceil(8) {
            let shift = 8 * pass;
            let at = 256 * pass as usize;
            {
                let mut tot = self.digit_tot.slice_mut(at..at + 256);
                let mut l = s.launch_builder(&self.kernels.radix_hist);
                l.arg(&self.keys[0])
                    .arg(&n)
                    .arg(&shift)
                    .arg(&mut self.part)
                    .arg(&mut tot);
                // SAFETY: radix_hist(keys, n, shift, hist[256][tiles], digit_tot[256]), one block of
                // 256 per tile.
                unsafe { l.launch(cfg((tiles, 1), 256)) }.cuda("radix_hist")?;
            }
            {
                let tot = self.digit_tot.slice(at..at + 256);
                let mut l = s.launch_builder(&self.kernels.radix_scan_rows);
                l.arg(&mut self.part).arg(&tiles).arg(&tot);
                // SAFETY: radix_scan_rows(hist[256][tiles], tiles, digit_tot[256]), one block of 1024
                // per digit.
                unsafe { l.launch(cfg((256, 1), 1024)) }.cuda("radix_scan_rows")?;
            }
            {
                let [k0, k1] = &mut self.keys;
                let [v0, v1] = &mut self.vals;
                let mut l = s.launch_builder(&self.kernels.radix_scatter);
                l.arg(&*k0)
                    .arg(&*v0)
                    .arg(&n)
                    .arg(&shift)
                    .arg(&self.part)
                    .arg(k1)
                    .arg(v1);
                // SAFETY: radix_scatter(keys, vals, n, shift, scanned hist, keys_out, vals_out), the
                // same tiling as radix_hist; outputs are a permutation into [0, n).
                unsafe { l.launch(cfg((tiles, 1), 256)) }.cuda("radix_scatter")?;
            }
            self.keys.swap(0, 1);
            self.vals.swap(0, 1);
        }
        Ok(())
    }

    /// 30-bit Morton codes over the means' AABB, radix sort, gather the
    /// parameters and every array in `extra` into that order.
    pub fn morton_sort(
        &mut self,
        gpu: &Gpu,
        g: &mut Gaussians,
        extra: &PerGaussianArrays<'_>,
    ) -> Result<()> {
        let n = g.n;
        if n < 2 {
            return Ok(());
        }
        let s = gpu.stream.clone();
        s.memcpy_htod(&[u32::MAX, u32::MAX, u32::MAX, 0, 0, 0], &mut self.aabb)
            .cuda("init AABB")?;
        {
            let mut l = s.launch_builder(&self.kernels.aabb_reduce);
            l.arg(&g.mean).arg(&n).arg(&mut self.aabb);
            let grid = blocks(n, 256).min(gpu.multiprocessors() * 8);
            // SAFETY: aabb_reduce(const float* mean, n, uint32_t aabb[6]), grid-stride.
            unsafe { l.launch(cfg((grid, 1), 256)) }.cuda("aabb_reduce")?;
        }
        {
            let [k0, _] = &mut self.keys;
            let [v0, _] = &mut self.vals;
            let mut l = s.launch_builder(&self.kernels.morton_codes);
            l.arg(&g.mean).arg(&self.aabb).arg(&n).arg(k0).arg(v0);
            // SAFETY: morton_codes(mean, aabb, n, codes, idx), one thread per Gaussian.
            unsafe { l.launch(LaunchConfig::for_num_elems(n)) }.cuda("morton_codes")?;
        }
        self.sort(gpu, n, 30)?;
        let mut cols = gaussian_columns(&s, g);
        cols.extend_from_slice(&extra.cols);
        self.upload_columns(gpu, &cols)?;
        for b in slab_batches(&cols) {
            self.move_batch(gpu, &cols, b, n, BatchOp::Gather)?;
        }
        Ok(())
    }
}

enum BatchOp<'a> {
    Emit(&'a CudaSlice<u8>),
    Gather,
}

/// `uint2` kernel argument.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Uint2 {
    pub x: u32,
    pub y: u32,
}

// SAFETY: two u32, the layout of CUDA's uint2.
unsafe impl DeviceRepr for Uint2 {}

impl From<[u32; 2]> for Uint2 {
    fn from(v: [u32; 2]) -> Self {
        Self { x: v[0], y: v[1] }
    }
}

/// Capacity full (spec-train 5.8 step 3): keep every survivor and take tail
/// entries in descending priority (ties by index) until full; a dropped split
/// keeps its parent. Host side (never expected on Mip360 with C from VRAM).
/// Returns the number of dropped tail entries.
fn cap_tail(gpu: &Gpu, plan: &mut EmitPlan, cap: u32, tot: [u32; 3]) -> Result<u32> {
    let n = plan.n_before as usize;
    let s = &gpu.stream;
    let mut oc = s
        .clone_dtoh(&plan.out_count.slice(..n))
        .cuda("read out_count")?;
    let mut cf = s
        .clone_dtoh(&plan.clone_flag.slice(..n))
        .cuda("read clone_flag")?;
    let mut sf = s
        .clone_dtoh(&plan.split_flag.slice(..n))
        .cuda("read split_flag")?;
    let pri = s
        .clone_dtoh(&plan.priority.slice(..n))
        .cuda("read priority")?;
    let dropped = cap_tail_host(&mut oc, &mut cf, &mut sf, &pri, cap, tot);
    s.memcpy_htod(&oc, &mut plan.out_count.slice_mut(..n))
        .cuda("write out_count")?;
    s.memcpy_htod(&cf, &mut plan.clone_flag.slice_mut(..n))
        .cuda("write clone_flag")?;
    s.memcpy_htod(&sf, &mut plan.split_flag.slice_mut(..n))
        .cuda("write split_flag")?;
    Ok(dropped)
}

/// The capping rule on host arrays (shared with the CPU reference).
pub fn cap_tail_host(
    oc: &mut [u32],
    cf: &mut [u8],
    sf: &mut [u8],
    pri: &[f32],
    cap: u32,
    tot: [u32; 3],
) -> u32 {
    let mut cand: Vec<usize> = (0..oc.len())
        .filter(|&i| cf[i] != 0 || sf[i] != 0)
        .collect();
    cand.sort_by(|&a, &b| pri[b].total_cmp(&pri[a]).then(a.cmp(&b)));
    // Start with every tail dropped (split parents restored), then add back.
    let mut free = cap as i64 - (tot[0] + tot[2]) as i64;
    let mut dropped = 0;
    for i in cand {
        if free >= 1 {
            free -= 1;
        } else if sf[i] != 0 {
            sf[i] = 0;
            oc[i] = 1;
            dropped += 2;
        } else {
            cf[i] = 0;
            dropped += 1;
        }
    }
    dropped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_cover_59_floats_in_adam_order() {
        assert_eq!(
            Group::ALL.iter().map(|g| g.floats()).sum::<usize>(),
            FLOATS_PER_GAUSSIAN
        );
        for (i, g) in Group::ALL.iter().enumerate() {
            assert_eq!(g.index(), i);
        }
        assert_eq!(Group::ShRest.index(), crate::kernels::layout::GROUP_SH_REST);
    }

    #[test]
    fn sh_rest_transposes_round_trip() {
        let n = 5;
        let per: Vec<f32> = (0..n * 45).map(|i| i as f32).collect();
        let planar = sh_rest_to_planar(&per, n);
        // Gaussian 2, coefficient k = 1, channel c = 2 -> j = 5.
        assert_eq!(planar[5 * n + 2], per[2 * 45 + 5]);
        assert_eq!(sh_rest_from_planar(&planar, n), per);
    }

    #[test]
    fn host_mirror_validates_lengths() {
        let mut g = GaussiansHost::zeros(3);
        g.validate().unwrap();
        assert_eq!(g.rotations[4], 1.0);
        g.sh_rest.pop();
        assert!(g.validate().is_err());
    }

    #[test]
    fn capacity_follows_the_memory_table() {
        assert_eq!(bytes_per_gaussian(false), 1254);
        assert_eq!(bytes_per_gaussian(true), 1018);
        assert_eq!(round_cap(1), 1024);
        assert_eq!(round_cap(2048), 2048);
        // 12 GB card, 0.8 GB images, 1.5 GB raster: about 7M Gaussians (spec-train 13).
        let c = plan_capacity(11_500 << 20, 800 << 20, 1_500 << 20, u64::MAX, false);
        assert!((7_000_000..8_200_000).contains(&c), "{c}");
        assert_eq!(c % 1024, 0);
        assert_eq!(plan_capacity(1 << 30, 2 << 30, 0, 10, false), 0);
    }
}
