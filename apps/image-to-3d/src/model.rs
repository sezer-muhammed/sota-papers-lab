//! The Gaussian store: fixed-capacity SoA parameters on the device, their host
//! mirror, gradient buffers, capacity planning, and (stub) compaction and
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

use anyhow::ensure;
use spl_core::{Result, bail};
use spl_gpu::cudarc::driver::CudaSlice;

use crate::gpu::{CudaResultExt, Gpu, not_implemented};

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
/// VCD scores 8, raster per-Gaussian buffers ~80, compaction slab 180.
pub const fn bytes_per_gaussian(bf16_moments: bool) -> u64 {
    236 + if bf16_moments { 236 } else { 472 } + 236 + 4 + 12 + 8 + 80 + 180
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

/// Every buffer indexed by Gaussian (params, moments, grads, last_visible,
/// stats), as compaction and Morton sorting move them. STUB (E6).
pub struct PerGaussianArrays<'a> {
    /// (buffer viewed as u32 words, words per Gaussian, planar stride or 0, zero for new entries)
    pub arrays: Vec<(&'a mut CudaSlice<u32>, u32, u32, bool)>,
}

/// Prune / clone / split decisions of one densify step, on the pre-step
/// indexing (spec-train 5.8). STUB (E6).
pub struct EmitPlan {
    pub n_before: u32,
    /// out_count[i] ∈ {0, 1} for old Gaussians (0 = pruned or split parent).
    pub out_count: CudaSlice<u32>,
    pub clone_flag: CudaSlice<u8>,
    pub split_flag: CudaSlice<u8>,
    /// Densify statistic for capping when the tail does not fit (descending).
    pub priority: CudaSlice<f32>,
}

/// Order-preserving in-place chunked compaction with emission, and the
/// Morton re-sort, through one `cap × 180 B` slab. STUB (E6).
pub struct Compactor {
    pub capacity: u32,
    pub slab: CudaSlice<u32>,
}

impl Compactor {
    pub fn new(_gpu: &Gpu, _capacity: usize) -> Result<Self> {
        Err(not_implemented("Compactor::new", "densify (E6)"))
    }

    /// Applies `plan` to every array in `arrays`. Returns the new n.
    pub fn emit(
        &mut self,
        _gpu: &Gpu,
        _plan: &EmitPlan,
        _arrays: &mut PerGaussianArrays<'_>,
    ) -> Result<usize> {
        Err(not_implemented("Compactor::emit", "densify (E6)"))
    }

    /// 30-bit Morton codes over the means' AABB, radix sort, gather every array.
    pub fn morton_sort(&mut self, _gpu: &Gpu, _arrays: &mut PerGaussianArrays<'_>) -> Result<()> {
        Err(not_implemented("Compactor::morton_sort", "densify (E6)"))
    }
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
        assert_eq!(bytes_per_gaussian(false), 1228);
        assert_eq!(bytes_per_gaussian(true), 992);
        assert_eq!(round_cap(1), 1024);
        assert_eq!(round_cap(2048), 2048);
        // 12 GB card, 0.8 GB images, 1.5 GB raster: about 7M Gaussians (spec-train 13).
        let c = plan_capacity(11_500 << 20, 800 << 20, 1_500 << 20, u64::MAX, false);
        assert!((7_000_000..8_200_000).contains(&c), "{c}");
        assert_eq!(c % 1024, 0);
        assert_eq!(plan_capacity(1 << 30, 2 << 30, 0, 10, false), 0);
    }
}
