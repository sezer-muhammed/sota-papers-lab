//! Stage B initialization (Report section 3): per-frame depth (VGGT-Ω, or
//! COLMAP for tests) to an initial Gaussian store, plus the depth-prior loss
//! ([`depth_prior`]). Owner: init-b. Kernels: `cuda/init.cu`.
//!
//! Pipeline, all on the GPU except two tiny host decisions:
//! 1. `init_conf_hist`: log2-confidence histogram of valid-depth pixels and
//!    the mean pixel footprint z / f. The host picks the confidence threshold
//!    at [`InitOptions::conf_percentile`] (low-confidence pixels, sky
//!    included, are dropped) and a first voxel size.
//! 2. `init_voxel_insert` (count only) a few times: the host adjusts the voxel
//!    size until the occupied-voxel count is within [`InitOptions::tolerance`]
//!    of [`InitOptions::target`] (log-log secant inside a bracket).
//! 3. `init_voxel_insert` (accumulate) then `init_finalize`: one Gaussian per
//!    voxel at the centroid, SH DC from the mean colour, log scale from the
//!    mean pixel footprint (kNN over neighbouring voxels as the fallback, or
//!    as the mode), identity rotation, opacity [`InitOptions::opacity`],
//!    written into [`Gaussians`] with the layout of `model.rs`.
//!
//! The output order follows hash slots (not deterministic across runs, nor
//! are float-atomic centroid sums in the last bits); the Morton re-sort of
//! the densify owner orders the store for locality.

pub mod depth_prior;
pub mod sources;

#[cfg(test)]
mod gpu_tests;

use std::time::Instant;

use anyhow::ensure;
use spl_core::{Result, bail};
use spl_gpu::cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use crate::gpu::{COMMON_CUH, CudaFile, CudaResultExt, Defines, Gpu, KernelSource, Kernels};
use crate::model::Gaussians;

pub const SOURCE: KernelSource = KernelSource {
    module: "init",
    files: &[
        COMMON_CUH,
        CudaFile {
            name: "init.cu",
            text: include_str!("../../cuda/init.cu"),
        },
    ],
    kernels: &[
        "init_conf_hist",
        "init_table_clear",
        "init_voxel_insert",
        "init_finalize",
        "ssi_partial",
        "ssi_solve",
        "ssi_backward",
    ],
};

/// Floats per frame in the device frame table (`INIT_FRAME_FLOATS`).
pub const FRAME_FLOATS: usize = 16;
/// Bins of the log2-confidence histogram (`INIT_HIST_BINS`).
pub const HIST_BINS: usize = 4096;
const HIST_LO: f32 = -16.0;
const HIST_PER_OCTAVE: f32 = 128.0;
const SLOT_FLOATS: usize = 8;
const BLOCK: u32 = 256;

/// One frame's camera at the depth-map resolution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InitFrame {
    /// World to camera, row-major.
    pub r_cw: [[f32; 3]; 3],
    pub t_cw: [f32; 3],
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
}

impl InitFrame {
    /// The device record (`INIT_FRAME_FLOATS`).
    pub fn pack(&self) -> [f32; FRAME_FLOATS] {
        let r = &self.r_cw;
        [
            r[0][0],
            r[0][1],
            r[0][2],
            r[1][0],
            r[1][1],
            r[1][2],
            r[2][0],
            r[2][1],
            r[2][2],
            self.t_cw[0],
            self.t_cw[1],
            self.t_cw[2],
            self.fx,
            self.fy,
            self.cx,
            self.cy,
        ]
    }

    /// Camera centre C = −Rᵀ t.
    pub fn center(&self) -> [f32; 3] {
        let (r, t) = (&self.r_cw, &self.t_cw);
        std::array::from_fn(|j| -(r[0][j] * t[0] + r[1][j] * t[1] + r[2][j] * t[2]))
    }

    /// World point of pixel (u, v) (integer indices; centre at +0.5) at z-depth
    /// `z`: the host mirror of `init_voxel_insert`'s unprojection.
    pub fn unproject(&self, u: u32, v: u32, z: f32) -> [f32; 3] {
        let xc = (u as f32 + 0.5 - self.cx) / self.fx * z - self.t_cw[0];
        let yc = (v as f32 + 0.5 - self.cy) / self.fy * z - self.t_cw[1];
        let zc = z - self.t_cw[2];
        let r = &self.r_cw;
        std::array::from_fn(|j| r[0][j] * xc + r[1][j] * yc + r[2][j] * zc)
    }

    /// (u, v, z) of world point `x` (continuous pixel coordinates).
    pub fn project(&self, x: [f32; 3]) -> [f32; 3] {
        let r = &self.r_cw;
        let c: [f32; 3] = std::array::from_fn(|i| {
            r[i][0] * x[0] + r[i][1] * x[1] + r[i][2] * x[2] + self.t_cw[i]
        });
        [
            self.fx * c[0] / c[2] + self.cx,
            self.fy * c[1] / c[2] + self.cy,
            c[2],
        ]
    }
}

/// Host-side input: `n` frames of `width × height` depth, confidence and RGB.
#[derive(Clone, Debug, Default)]
pub struct DepthFramesHost {
    pub width: u32,
    pub height: u32,
    pub frames: Vec<InitFrame>,
    /// `[F][H][W]` z-depth; ≤ 0 or non-finite = none.
    pub depth: Vec<f32>,
    /// `[F][H][W]`, larger = more confident (VGGT's 1 + exp(x) works as is).
    pub conf: Vec<f32>,
    /// `[F][H][W][3]`.
    pub rgb: Vec<u8>,
}

impl DepthFramesHost {
    pub fn validate(&self) -> Result<()> {
        let px = self.frames.len() * self.width as usize * self.height as usize;
        ensure!(!self.frames.is_empty(), "no depth frames");
        ensure!(
            self.depth.len() == px,
            "depth has {} values, expected {px}",
            self.depth.len()
        );
        ensure!(
            self.conf.len() == px,
            "conf has {} values, expected {px}",
            self.conf.len()
        );
        ensure!(
            self.rgb.len() == 3 * px,
            "rgb has {} bytes, expected {}",
            self.rgb.len(),
            3 * px
        );
        Ok(())
    }
}

/// Device copy of [`DepthFramesHost`] (what the engine's dense head writes).
pub struct DepthFrames {
    pub width: u32,
    pub height: u32,
    pub n_frames: u32,
    pub frames: CudaSlice<f32>,
    pub depth: CudaSlice<f32>,
    pub conf: CudaSlice<f32>,
    pub rgb: CudaSlice<u8>,
    /// Camera centres (host), for the voxel origin.
    pub centers: Vec<[f32; 3]>,
}

impl DepthFrames {
    pub fn upload(gpu: &Gpu, host: &DepthFramesHost) -> Result<Self> {
        host.validate()?;
        let s = &gpu.stream;
        let packed: Vec<f32> = host.frames.iter().flat_map(|f| f.pack()).collect();
        Ok(Self {
            width: host.width,
            height: host.height,
            n_frames: host.frames.len() as u32,
            frames: s.clone_htod(&packed).cuda("upload init frames")?,
            depth: s.clone_htod(&host.depth).cuda("upload depth")?,
            conf: s.clone_htod(&host.conf).cuda("upload conf")?,
            rgb: s.clone_htod(&host.rgb).cuda("upload rgb")?,
            centers: host.frames.iter().map(InitFrame::center).collect(),
        })
    }

    pub fn pixels(&self) -> u64 {
        self.n_frames as u64 * self.width as u64 * self.height as u64
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleMode {
    /// Mean pixel footprint z / f of the voxel's pixels × `footprint_mult`,
    /// at least `min_voxel_frac` × voxel; kNN when it is not available.
    Footprint,
    /// sqrt(mean of the 3 smallest squared distances to neighbouring voxel
    /// centroids); the voxel size when fewer than 3 neighbours exist.
    Knn,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InitOptions {
    /// Gaussians wanted (Report: 0.5-1.5M).
    pub target: u32,
    /// Accepted relative deviation from `target`.
    pub tolerance: f32,
    /// Fraction of valid-depth pixels dropped by lowest confidence (0 = keep all).
    pub conf_percentile: f32,
    /// Initial opacity (Report: 0.1-0.3), stored as a logit.
    pub opacity: f32,
    pub scale_mode: ScaleMode,
    pub footprint_mult: f32,
    pub min_voxel_frac: f32,
    /// Voxel-size search passes (count only) before the final one.
    pub max_search: u32,
}

impl Default for InitOptions {
    fn default() -> Self {
        Self {
            target: 1_000_000,
            tolerance: 0.1,
            conf_percentile: 0.2,
            opacity: 0.1,
            scale_mode: ScaleMode::Footprint,
            footprint_mult: 1.0,
            min_voxel_frac: 0.5,
            max_search: 10,
        }
    }
}

impl InitOptions {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.target > 0, "init target must be positive");
        ensure!(
            (0.0..1.0).contains(&self.conf_percentile),
            "conf_percentile must be in [0, 1)"
        );
        ensure!(
            self.opacity > 0.0 && self.opacity < 1.0,
            "init opacity must be in (0, 1)"
        );
        ensure!(self.tolerance > 0.0, "init tolerance must be positive");
        Ok(())
    }
}

/// What the initialization did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InitReport {
    pub n_gaussians: u32,
    /// Occupied voxels (> `n_gaussians` only when the store was too small).
    pub n_voxels: u32,
    pub voxel: f32,
    pub conf_threshold: f32,
    pub valid_pixels: u64,
    pub kept_points: u32,
    pub dropped_points: u32,
    pub mean_footprint: f32,
    /// (voxel, occupied) per count pass.
    pub search: Vec<(f32, u32)>,
    pub ms_hist: f64,
    pub ms_search: f64,
    pub ms_final: f64,
    pub ms_total: f64,
}

/// The loaded kernels of [`SOURCE`].
pub struct InitKernels(Kernels);

impl InitKernels {
    pub fn load(gpu: &Gpu) -> Result<Self> {
        Ok(Self(gpu.kernels(&SOURCE, &Defines::new())?))
    }

    pub(crate) fn get(&self, name: &str) -> Result<spl_gpu::cudarc::driver::CudaFunction> {
        self.0.get(name)
    }
}

/// Confidence at fraction `q` of a histogram over log2 bins (the lower edge of
/// the bin where the cumulative count reaches q · total).
pub fn hist_percentile(hist: &[u32], q: f32) -> f32 {
    let total: u64 = hist.iter().map(|&c| c as u64).sum();
    if q <= 0.0 || total == 0 {
        return f32::NEG_INFINITY;
    }
    let want = (q as f64 * total as f64 * (1.0 - 1e-6)).ceil().max(1.0) as u64;
    let mut acc = 0u64;
    for (b, &c) in hist.iter().enumerate() {
        acc += c as u64;
        if acc >= want {
            // Drop bins below b entirely, plus bin b (its pixels are ≤ its upper edge).
            return 2f32.powf(HIST_LO + (b + 1) as f32 / HIST_PER_OCTAVE);
        }
    }
    f32::INFINITY
}

/// The histogram bin of confidence `c` (`init_hist_bin`); bins at or above it
/// pass a threshold `hist_percentile` returned.
fn hist_bin(c: f32) -> usize {
    if c <= 0.0 {
        return 0;
    }
    ((c.log2() - HIST_LO) * HIST_PER_OCTAVE).clamp(0.0, (HIST_BINS - 1) as f32) as usize
}

/// Next voxel size: secant on log(count) vs log(voxel) with the slope clamped
/// to [−3, −0.5], kept inside the bracket (geometric bisection otherwise).
fn next_voxel(search: &[(f32, u32)], target: u32, lo: f32, hi: f32) -> f32 {
    let &(v, c) = search.last().expect("one pass done");
    let c = c.max(1) as f32;
    let slope = match search.len() {
        n if n >= 2 => {
            let (v0, c0) = search[n - 2];
            let dv = (v / v0).ln();
            if dv.abs() > 1e-6 && c0 > 0 {
                ((c / c0 as f32).ln() / dv).clamp(-3.0, -0.5)
            } else {
                -2.0
            }
        }
        _ => -2.0,
    };
    let mut nv = v * (target as f32 / c).powf(1.0 / slope);
    if lo > 0.0 && hi.is_finite() && !(nv > lo && nv < hi) {
        nv = (lo * hi).sqrt();
    } else if lo > 0.0 && nv <= lo {
        nv = lo * 2.0;
    } else if hi.is_finite() && nv >= hi {
        nv = hi * 0.5;
    }
    nv
}

/// Logit of `p`.
pub fn logit(p: f32) -> f32 {
    (p / (1.0 - p)).ln()
}

struct Table {
    size: u32,
    keys: CudaSlice<u64>,
    vals: CudaSlice<f32>,
    ctr: CudaSlice<u32>,
}

/// Fills `store` (from index 0; `store.n` is set, SH rest zeroed, other
/// buffers above `n` untouched) from `frames`. The store needs capacity for
/// `target · (1 + tolerance)`; extra voxels beyond the capacity are dropped
/// and reported (`n_voxels > n_gaussians`).
pub fn init_from_depth(
    gpu: &Gpu,
    k: &InitKernels,
    frames: &DepthFrames,
    opts: &InitOptions,
    store: &mut Gaussians,
) -> Result<InitReport> {
    opts.validate()?;
    let s = &gpu.stream;
    gpu.sync()?;
    let t0 = Instant::now();
    let mut rep = InitReport::default();
    let grid = |n: u64| {
        (n.div_ceil(BLOCK as u64)
            .min(gpu.multiprocessors() as u64 * 16)) as u32
    };
    let cfg = |blocks: u32| LaunchConfig {
        grid_dim: (blocks.max(1), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let hw = frames.width * frames.height;
    let px = frames.pixels();

    // 1. Confidence histogram and footprint.
    let mut hist = s.alloc_zeros::<u32>(HIST_BINS).cuda("alloc hist")?;
    let mut foot = s.alloc_zeros::<f32>(HIST_BINS).cuda("alloc foot")?;
    {
        let f = k.get("init_conf_hist")?;
        let mut l = s.launch_builder(&f);
        l.arg(&frames.depth)
            .arg(&frames.conf)
            .arg(&frames.frames)
            .arg(&frames.n_frames)
            .arg(&hw)
            .arg(&mut hist)
            .arg(&mut foot);
        // SAFETY: init_conf_hist(depth, conf, frames, n_frames, hw, hist[4096], foot_hist[4096]).
        unsafe { l.launch(cfg(grid(px).min(gpu.multiprocessors() * 4))) }.cuda("init_conf_hist")?;
    }
    let hist_h = s.clone_dtoh(&hist).cuda("download hist")?;
    let foot_h = s.clone_dtoh(&foot).cuda("download foot")?;
    let valid: u64 = hist_h.iter().map(|&c| c as u64).sum();
    rep.valid_pixels = valid;
    if valid == 0 {
        bail!("Stage B init: no pixel has a valid depth");
    }
    rep.conf_threshold = hist_percentile(&hist_h, opts.conf_percentile);
    // Footprint and count over the bins that survive the threshold.
    let first = hist_bin(rep.conf_threshold * (1.0 + 1e-4));
    let kept: u64 = hist_h[first..].iter().map(|&c| c as u64).sum();
    let kept_foot: f64 = foot_h[first..].iter().map(|&f| f as f64).sum();
    rep.mean_footprint = (kept_foot / kept.max(1) as f64) as f32;
    let kept_est = kept.max(1) as f32;
    rep.ms_hist = t0.elapsed().as_secs_f64() * 1e3;

    // Voxel origin: mean camera centre (keeps keys within ±2^20 voxels).
    let n = frames.centers.len() as f32;
    let origin: [f32; 3] =
        std::array::from_fn(|a| frames.centers.iter().map(|c| c[a]).sum::<f32>() / n);

    let target = opts.target.min(store.cap);
    let size = (target as u64 * 4)
        .next_power_of_two()
        .clamp(1 << 16, 1 << 30) as u32;
    let mut table = Table {
        size,
        keys: s
            .alloc_zeros::<u64>(size as usize)
            .cuda("alloc voxel keys")?,
        vals: s
            .alloc_zeros::<f32>(size as usize * SLOT_FLOATS)
            .cuda("alloc voxel vals")?,
        ctr: s.alloc_zeros::<u32>(4).cuda("alloc voxel counters")?,
    };
    let pass = |table: &mut Table, voxel: f32, accumulate: bool| -> Result<[u32; 4]> {
        let clear = u32::from(accumulate);
        let f = k.get("init_table_clear")?;
        let mut l = s.launch_builder(&f);
        l.arg(&mut table.keys)
            .arg(&mut table.vals)
            .arg(&table.size)
            .arg(&mut table.ctr)
            .arg(&clear);
        // SAFETY: init_table_clear(keys[size], vals[size*8], size, ctr[4], clear_vals).
        unsafe { l.launch(cfg(grid(table.size as u64))) }.cuda("init_table_clear")?;
        let f = k.get("init_voxel_insert")?;
        let mut l = s.launch_builder(&f);
        let acc = u32::from(accumulate);
        l.arg(&frames.depth)
            .arg(&frames.conf)
            .arg(&frames.rgb)
            .arg(&frames.frames)
            .arg(&frames.n_frames)
            .arg(&frames.width)
            .arg(&frames.height)
            .arg(&rep.conf_threshold)
            .arg(&origin[0])
            .arg(&origin[1])
            .arg(&origin[2])
            .arg(&voxel)
            .arg(&mut table.keys)
            .arg(&mut table.vals)
            .arg(&table.size)
            .arg(&mut table.ctr)
            .arg(&acc);
        // SAFETY: argument list matches init_voxel_insert in init.cu; buffers sized above.
        unsafe { l.launch(cfg(grid(px))) }.cuda("init_voxel_insert")?;
        let c = s.clone_dtoh(&table.ctr).cuda("download voxel counters")?;
        Ok([c[0], c[1], c[2], c[3]])
    };

    // 2. Voxel-size search.
    let t1 = Instant::now();
    let vmin = (rep.mean_footprint * 0.05).max(f32::MIN_POSITIVE);
    let mut voxel = if kept_est as u64 <= target as u64 {
        vmin
    } else {
        rep.mean_footprint * (kept_est / target as f32).sqrt()
    };
    let (mut lo, mut hi) = (0.0f32, f32::INFINITY);
    let full = size / 2;
    for it in 0..opts.max_search.max(1) {
        let c = pass(&mut table, voxel, false)?;
        // A dropped insert means the table filled: the voxel is far too small.
        let occupied = if c[1] > 0 && c[0] >= full {
            u32::MAX / 2
        } else {
            c[0]
        };
        rep.search.push((voxel, occupied));
        let rel = occupied as f32 / target as f32;
        if (rel - 1.0).abs() <= opts.tolerance || (occupied < target && voxel <= vmin) {
            break;
        }
        if occupied > target {
            lo = lo.max(voxel);
        } else {
            hi = hi.min(voxel);
        }
        if it + 1 == opts.max_search.max(1) {
            break;
        }
        voxel = next_voxel(&rep.search, target, lo, hi).max(vmin);
    }
    // The best pass not above the target (+ tolerance), else the largest voxel.
    let ok = |c: u32| c as f32 <= target as f32 * (1.0 + opts.tolerance);
    voxel = rep
        .search
        .iter()
        .filter(|(_, c)| ok(*c))
        .max_by_key(|(_, c)| *c)
        .or_else(|| rep.search.iter().max_by(|a, b| a.0.total_cmp(&b.0)))
        .map(|&(v, _)| v)
        .expect("one pass done");
    rep.ms_search = t1.elapsed().as_secs_f64() * 1e3;

    // 3. Accumulate and write the store.
    let t2 = Instant::now();
    let c = pass(&mut table, voxel, true)?;
    rep.voxel = voxel;
    rep.n_voxels = c[0];
    rep.dropped_points = c[1];
    rep.kept_points = c[2];
    s.memset_zeros(&mut store.sh_rest).cuda("zero sh_rest")?;
    {
        let f = k.get("init_finalize")?;
        let mut l = s.launch_builder(&f);
        let mode = match opts.scale_mode {
            ScaleMode::Footprint => 0u32,
            ScaleMode::Knn => 1,
        };
        let op = logit(opts.opacity);
        l.arg(&table.keys)
            .arg(&table.vals)
            .arg(&table.size)
            .arg(&mut table.ctr)
            .arg(&store.cap)
            .arg(&voxel)
            .arg(&opts.footprint_mult)
            .arg(&opts.min_voxel_frac)
            .arg(&mode)
            .arg(&op)
            .arg(&mut store.mean)
            .arg(&mut store.log_scale)
            .arg(&mut store.quat)
            .arg(&mut store.opacity)
            .arg(&mut store.sh_dc);
        // SAFETY: argument list matches init_finalize; every store buffer holds cap Gaussians.
        unsafe { l.launch(LaunchConfig::for_num_elems(table.size)) }.cuda("init_finalize")?;
    }
    let c = s.clone_dtoh(&table.ctr).cuda("download counters")?;
    store.n = c[3].min(store.cap);
    s.memset_zeros(&mut store.last_visible)
        .cuda("reset last_visible")?;
    gpu.sync()?;
    rep.n_gaussians = store.n;
    rep.ms_final = t2.elapsed().as_secs_f64() * 1e3;
    rep.ms_total = t0.elapsed().as_secs_f64() * 1e3;
    Ok(rep)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_init_kernel_is_declared() {
        let text = SOURCE.text();
        for k in SOURCE.kernels {
            assert!(text.contains(&format!("__global__ void {k}(")), "{k}");
        }
    }

    #[test]
    fn unproject_inverts_project() {
        let a = 0.3f32;
        let f = InitFrame {
            r_cw: [
                [a.cos(), 0.0, a.sin()],
                [0.0, 1.0, 0.0],
                [-a.sin(), 0.0, a.cos()],
            ],
            t_cw: [0.2, -0.1, 1.5],
            fx: 300.0,
            fy: 310.0,
            cx: 160.0,
            cy: 120.0,
        };
        let x = f.unproject(40, 70, 2.5);
        let [u, v, z] = f.project(x);
        assert!((u - 40.5).abs() < 1e-3 && (v - 70.5).abs() < 1e-3 && (z - 2.5).abs() < 1e-4);
        let c = f.center();
        let [_, _, zc] = f.project([c[0] + 1e-3, c[1], c[2]]);
        assert!(zc.abs() < 1e-2);
    }

    #[test]
    fn percentile_from_histogram() {
        let mut h = vec![0u32; HIST_BINS];
        // conf 1.0 → bin (0 + 16) · 128 = 2048; conf 4.0 → 2304.
        h[2048] = 30;
        h[2304] = 70;
        let thr = hist_percentile(&h, 0.3);
        assert!(thr > 1.0 && thr <= 4.0, "{thr}");
        assert_eq!(hist_percentile(&h, 0.0), f32::NEG_INFINITY);
        let thr = hist_percentile(&h, 0.5);
        assert!(thr > 4.0, "{thr}");
    }

    #[test]
    fn voxel_search_converges_on_a_power_law() {
        // occupied = 1e6 · (0.01 / v)^2
        let count = |v: f32| (1e6 * (0.01 / v).powi(2)) as u32;
        let mut search = vec![(0.003f32, count(0.003))];
        let (mut lo, mut hi) = (0.0f32, f32::INFINITY);
        for _ in 0..6 {
            let &(v, c) = search.last().unwrap();
            if c > 250_000 {
                lo = lo.max(v)
            } else {
                hi = hi.min(v)
            }
            let nv = next_voxel(&search, 250_000, lo, hi);
            search.push((nv, count(nv)));
        }
        let c = search.last().unwrap().1 as f32;
        assert!((c / 250_000.0 - 1.0).abs() < 0.02, "{search:?}");
    }

    #[test]
    fn logit_of_opacity() {
        assert!((1.0 / (1.0 + (-logit(0.1)).exp()) - 0.1).abs() < 1e-6);
    }
}
