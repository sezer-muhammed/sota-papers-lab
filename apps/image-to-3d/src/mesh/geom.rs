//! Geometry rendering (median depth, normals) and the Geometry preset's
//! regularisers: depth distortion and depth-normal consistency (2DGS) with
//! PGSR's ray/plane depth and a flattening term on the smallest scale.
//! Owner: mesh-mode. Kernels: `cuda/mesh_geom.cu` (math and conventions in
//! its header).
//!
//! The kernels walk the raster's sorted tile lists after `raster_fwd`, so
//! they see exactly the Gaussians and blend order of the colour image. The
//! alpha-chain gradients land in `g2d` before `raster_bwd` (which
//! accumulates) and flow through `preprocess_bwd` like the colour loss; the
//! normal and depth terms go straight to the quaternion and mean gradients
//! after it.
//!
//! Multi-view terms (PGSR): each training view's plane depth is cached when it
//! is rendered; a later iteration of a neighbouring view checks its own plane
//! depth against that cache (geometric) and warps 7x7 photo patches through
//! its per-pixel planes into the neighbour photo (photometric NCC). The cache
//! replaces PGSR's second render of the neighbour (stale by up to one epoch).

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use spl_core::Result;
use spl_gpu::cudarc::driver::{CudaFunction, CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg};
use ts_rs::TS;

use crate::gpu::{COMMON_CUH, CudaFile, CudaResultExt, Gpu, KernelSource, SPLAT_COMMON_CUH};
use crate::io::images::ImageCache;
use crate::io::scene::View;
use crate::kernels::raster::{GpuRaster, RASTER_COMMON_CUH};
use crate::kernels::{Capacity, RasterVariant};
use crate::model::{Gaussians, Grads};

pub const SOURCE: KernelSource = KernelSource {
    module: "mesh_geom",
    files: &[
        COMMON_CUH,
        SPLAT_COMMON_CUH,
        RASTER_COMMON_CUH,
        CudaFile {
            name: "mesh_geom.cu",
            text: include_str!("../../cuda/mesh_geom.cu"),
        },
    ],
    kernels: &[
        "geom_prep",
        "geom_fwd",
        "geom_pixel",
        "geom_bwd",
        "geom_post",
        "geom_clip_depth",
        "geom_cache",
        "geom_mv",
    ],
};

/// Output channels of `geom_fwd`, planar `[GEOM_CH][P]`.
pub const GEOM_CH: usize = 9;
/// Channel of the median depth (0 where the transmittance never reaches 0.5).
pub const CH_MEDIAN: usize = 0;
pub const CH_ALPHA: usize = 1;
pub const CH_NORMAL: usize = 4;
pub const CH_EXPECTED: usize = 7;
/// Blended plane offset `sum w (n . mu)`; plane depth = Ds / (N . ray).
pub const CH_DS: usize = 8;
const GPIX_CH: usize = 6;
/// f32 words of one `GeomAux`.
const AUX_WORDS: usize = 8;
/// Depth-cache budget; larger scenes cache every 2nd, 3rd... pixel.
const CACHE_BYTES: usize = 512 << 20;

/// Weights and schedule of the geometry regularisers (Geometry preset).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct GeometryConfig {
    /// Depth distortion weight (2DGS: 1000 on DTU, 0 on unbounded scenes).
    #[serde(default = "d_dist")]
    pub lambda_dist: f32,
    /// Depth-normal consistency weight (2DGS: 0.05).
    #[serde(default = "d_normal")]
    pub lambda_normal: f32,
    /// Flattening weight on each visible Gaussian's smallest scale (PGSR-style).
    #[serde(default = "d_flat")]
    pub lambda_flat: f32,
    /// Distortion depth mapping m = 1 − near / z, near = this × scene extent.
    #[serde(default = "d_near")]
    pub near_rel: f32,
    /// Fractions of the run where each term switches on (2DGS: 3k and 7k of 30k).
    #[serde(default = "d_dist_from")]
    pub dist_from: f32,
    #[serde(default = "d_normal_from")]
    pub normal_from: f32,
    #[serde(default)]
    pub flat_from: f32,
    /// Median depth replaced by the plane depth `Ds / (N . ray)` (PGSR).
    #[serde(default = "yes")]
    pub plane_depth: bool,
    /// Multi-view geometric consistency weight (relative depth residual).
    #[serde(default = "d_mv_geo")]
    pub lambda_mv_geo: f32,
    /// Multi-view photometric weight (1 − NCC of 7x7 patches; PGSR: 0.15).
    #[serde(default = "d_mv_ncc")]
    pub lambda_mv_ncc: f32,
    /// Fraction of the run where the multi-view terms switch on (PGSR: 7k of 30k).
    #[serde(default = "d_normal_from")]
    pub mv_from: f32,
    /// Multi-view terms on every n-th pixel along each axis.
    #[serde(default = "d_mv_stride")]
    pub mv_stride: u32,
    /// Forward-backward reprojection error (pixels) above which a pixel counts as occluded.
    #[serde(default = "d_mv_err")]
    pub mv_max_err: f32,
    /// Gray variance of the reference patch below which NCC is skipped.
    #[serde(default = "d_ncc_var")]
    pub ncc_min_var: f32,
    /// Neighbours per view (closest centres with 3–60° between optical axes).
    #[serde(default = "d_mv_nb")]
    pub mv_neighbours: u32,
}

fn yes() -> bool {
    true
}
fn d_mv_geo() -> f32 {
    0.5
}
fn d_mv_ncc() -> f32 {
    0.15
}
fn d_mv_stride() -> u32 {
    2
}
fn d_mv_err() -> f32 {
    1.0
}
fn d_ncc_var() -> f32 {
    1e-4
}
fn d_mv_nb() -> u32 {
    8
}

fn d_dist() -> f32 {
    100.0
}
fn d_normal() -> f32 {
    0.05
}
fn d_flat() -> f32 {
    1e-4
}
fn d_near() -> f32 {
    0.06
}
fn d_dist_from() -> f32 {
    0.1
}
fn d_normal_from() -> f32 {
    7.0 / 30.0
}

impl Default for GeometryConfig {
    fn default() -> Self {
        Self {
            lambda_dist: d_dist(),
            lambda_normal: d_normal(),
            lambda_flat: d_flat(),
            near_rel: d_near(),
            dist_from: d_dist_from(),
            normal_from: d_normal_from(),
            flat_from: 0.0,
            plane_depth: true,
            lambda_mv_geo: d_mv_geo(),
            lambda_mv_ncc: d_mv_ncc(),
            mv_from: d_normal_from(),
            mv_stride: d_mv_stride(),
            mv_max_err: d_mv_err(),
            ncc_min_var: d_ncc_var(),
            mv_neighbours: d_mv_nb(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GeomParams {
    pub near_m: f32,
    /// Both pixel losses are means over the frame (the kernel divides by P).
    pub lambda_dist: f32,
    pub lambda_normal: f32,
    pub lambda_flat: f32,
    pub dist_from: u32,
    pub normal_from: u32,
    pub flat_from: u32,
    pub v_cap: u32,
    /// 1: channel 0 holds the plane depth instead of the median.
    pub plane_depth: u32,
    pub lambda_mv_geo: f32,
    pub lambda_mv_ncc: f32,
    pub mv_from: u32,
    pub mv_stride: u32,
    pub mv_max_err: f32,
    pub ncc_min_var: f32,
    pub pad: u32,
}

const _: () = assert!(size_of::<GeomParams>() == 64);
// SAFETY: #[repr(C)] f32/u32 only, no padding; matches `GeomParams` in mesh_geom.cu.
unsafe impl DeviceRepr for GeomParams {}

pub struct GeomKernels {
    prep: CudaFunction,
    fwd: CudaFunction,
    pixel: CudaFunction,
    bwd: CudaFunction,
    post: CudaFunction,
    clip: CudaFunction,
    cache: CudaFunction,
    mv: CudaFunction,
    tile_px: u32,
}

impl GeomKernels {
    pub fn load(gpu: &Gpu, v: &RasterVariant) -> Result<Self> {
        let k = gpu.kernels(&SOURCE, &v.defines())?;
        Ok(Self {
            prep: k.get("geom_prep")?,
            fwd: k.get("geom_fwd")?,
            pixel: k.get("geom_pixel")?,
            bwd: k.get("geom_bwd")?,
            post: k.get("geom_post")?,
            clip: k.get("geom_clip_depth")?,
            cache: k.get("geom_cache")?,
            mv: k.get("geom_mv")?,
            tile_px: v.tile_pixels(),
        })
    }
}

/// Device buffers sized for one raster capacity (visible and pixel caps are
/// fixed for a run; a raster regrow only changes instances).
pub struct GeomBuffers {
    pub cap: Capacity,
    /// `GeomAux` per visible Gaussian.
    pub aux: CudaSlice<f32>,
    /// `[GEOM_CH][P]`.
    pub out: CudaSlice<f32>,
    gpix: Option<CudaSlice<f32>>,
    /// Camera-space mean gradient, 3 rows of `visible`.
    gz: Option<CudaSlice<f32>>,
    gn: Option<CudaSlice<f32>>,
    /// Running loss sums (distortion, normal, multi-view geometric,
    /// photometric, multi-view pixels), for logs and tests.
    pub sums: Option<CudaSlice<f32>>,
}

impl GeomBuffers {
    pub fn alloc(gpu: &Gpu, cap: Capacity, train: bool) -> Result<Self> {
        let s = &gpu.stream;
        let vis = cap.visible.max(1) as usize;
        let px = cap.pixels.max(1) as usize;
        let z = |n: usize, what: &str| s.alloc_zeros::<f32>(n).cuda(what);
        Ok(Self {
            cap,
            aux: z(AUX_WORDS * vis, "alloc geom aux")?,
            out: z(GEOM_CH * px, "alloc geom out")?,
            gpix: if train {
                Some(z(GPIX_CH * px, "alloc geom gpix")?)
            } else {
                None
            },
            gz: if train {
                Some(z(3 * vis, "alloc geom gz")?)
            } else {
                None
            },
            gn: if train {
                Some(z(3 * vis, "alloc geom gn")?)
            } else {
                None
            },
            sums: if train {
                Some(z(5, "alloc geom sums")?)
            } else {
                None
            },
        })
    }

    pub fn bytes(cap: Capacity, train: bool) -> u64 {
        let (vis, px) = (cap.visible as u64, cap.pixels as u64);
        4 * (AUX_WORDS as u64 * vis
            + GEOM_CH as u64 * px
            + if train {
                GPIX_CH as u64 * px + 6 * vis + 5
            } else {
                0
            })
    }
}

fn grid_stride(n: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(256).clamp(1, 4096), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// `geom_prep` + `geom_fwd` after a `raster_fwd` of the same frame: fills
/// `b.out` (median depth, alpha, moments, normals, expected depth).
pub fn render(
    gpu: &Gpu,
    k: &GeomKernels,
    raster: &GpuRaster,
    g: &Gaussians,
    b: &mut GeomBuffers,
    gp: GeomParams,
) -> Result<()> {
    let s = &gpu.stream;
    let f = &raster.frame;
    let v_cap = f.cap.visible;
    let null = 0u64;
    {
        let mut l = s.launch_builder(&k.prep);
        l.arg(&f.iter.cam)
            .arg(&f.vis_idx)
            .arg(&g.mean)
            .arg(&g.log_scale)
            .arg(&g.quat)
            .arg(&mut b.aux);
        match (&mut b.gz, &mut b.gn) {
            (Some(gz), Some(gn)) => l.arg(gz).arg(gn),
            _ => l.arg(&null).arg(&null),
        };
        l.arg(&v_cap).arg(&f.iter.status);
        // SAFETY: arguments match `geom_prep`; v < min(status[ST_N_VIS], v_cap),
        // the size of vis_idx, aux (6 words each), gz and gn (3 rows of v_cap).
        unsafe { l.launch(grid_stride(v_cap)) }.cuda("geom_prep")?;
    }
    let mut l = s.launch_builder(&k.fwd);
    l.arg(&f.iter.cam)
        .arg(&f.rec)
        .arg(&b.aux)
        .arg(&f.inst_v[f.tile_sorted])
        .arg(&f.tile_start)
        .arg(&gp)
        .arg(&mut b.out)
        .arg(&f.iter.status);
    let cfg = LaunchConfig {
        grid_dim: (f.cap.tiles.max(1), 1, 1),
        block_dim: (k.tile_px, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: arguments match `geom_fwd`; tiles and the image size come from
    // the device FrameCam (≤ cap.tiles, cap.pixels), records v < v_cap.
    unsafe { l.launch(cfg) }.cuda("geom_fwd")?;
    Ok(())
}

/// Zeroes the median depth of pixels whose point is outside `lo..hi`
/// (after [`render`] of the same frame).
pub fn clip_depth(
    gpu: &Gpu,
    k: &GeomKernels,
    raster: &GpuRaster,
    b: &mut GeomBuffers,
    lo: [f32; 3],
    hi: [f32; 3],
) -> Result<()> {
    let f = &raster.frame;
    let mut l = gpu.stream.launch_builder(&k.clip);
    l.arg(&f.iter.cam).arg(&mut b.out);
    for v in lo.iter().chain(hi.iter()) {
        l.arg(v);
    }
    l.arg(&f.iter.status);
    // SAFETY: arguments match `geom_clip_depth`; pixels < width·height ≤ cap.pixels.
    unsafe { l.launch(grid_stride(f.cap.pixels)) }.cuda("geom_clip_depth")?;
    Ok(())
}

/// The Geometry preset's per-iteration work, captured with the iteration.
pub struct GeomTrain {
    pub cfg: GeometryConfig,
    pub kernels: GeomKernels,
    pub buf: GeomBuffers,
    params: GeomParams,
    mv: Option<MvState>,
    /// Multi-view (geometric, photometric) loss per used pixel and pixels per
    /// frame, from the last [`GeomTrain::take_sums`].
    pub mv_sums: (f64, f64, f64),
}

/// One training view for the multi-view terms; matches `MvView` in mesh_geom.cu.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MvView {
    pub r: [f32; 9],
    pub t: [f32; 3],
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub width: u32,
    pub height: u32,
    pub img_lo: u32,
    pub img_hi: u32,
    pub slot: u32,
    pub n_nb: u32,
    pub nb: [u32; 8],
    pub pad: [u32; 2],
}

const _: () = assert!(size_of::<MvView>() == 128);
// SAFETY: #[repr(C)] f32/u32 only, no padding; matches `MvView` in mesh_geom.cu.
unsafe impl DeviceRepr for MvView {}

struct MvState {
    views: CudaSlice<MvView>,
    cache: CudaSlice<f32>,
    /// Cache stride (pixels) and floats per slot.
    stride: u32,
    slot_px: u32,
}

/// Neighbours of each view in `train`: the closest centres whose optical
/// axes differ by 3–60°, at most `k`.
pub fn neighbours(views: &[View], train: &[usize], k: usize) -> Vec<Vec<usize>> {
    let axis = |v: &View| v.r_cw[2];
    let (lo, hi) = (3f64.to_radians().cos(), 60f64.to_radians().cos());
    (0..views.len())
        .map(|i| {
            if !train.contains(&i) {
                return Vec::new();
            }
            let (ci, ai) = (views[i].center(), axis(&views[i]));
            let mut c: Vec<(f64, usize)> = train
                .iter()
                .filter(|&&j| j != i)
                .filter_map(|&j| {
                    let aj = axis(&views[j]);
                    let cos = ai[0] * aj[0] + ai[1] * aj[1] + ai[2] * aj[2];
                    let cj = views[j].center();
                    let d = (0..3).map(|k| (ci[k] - cj[k]).powi(2)).sum::<f64>();
                    (cos < lo && cos > hi).then_some((d, j))
                })
                .collect();
            c.sort_by(|a, b| a.0.total_cmp(&b.0));
            c.into_iter().take(k).map(|(_, j)| j).collect()
        })
        .collect()
}

impl GeomTrain {
    pub fn new(
        gpu: &Gpu,
        cfg: GeometryConfig,
        variant: &RasterVariant,
        cap: Capacity,
        extent: f32,
        total: u32,
    ) -> Result<Self> {
        let at = |f: f32| ((f.clamp(0.0, 1.0) * total as f32).round() as u32).max(1);
        let mv_on = cfg.lambda_mv_geo > 0.0 || cfg.lambda_mv_ncc > 0.0;
        let params = GeomParams {
            near_m: cfg.near_rel * extent,
            lambda_dist: cfg.lambda_dist,
            lambda_normal: cfg.lambda_normal,
            lambda_flat: cfg.lambda_flat,
            dist_from: if cfg.lambda_dist > 0.0 {
                at(cfg.dist_from)
            } else {
                u32::MAX
            },
            normal_from: if cfg.lambda_normal > 0.0 {
                at(cfg.normal_from)
            } else {
                u32::MAX
            },
            flat_from: if cfg.lambda_flat > 0.0 {
                at(cfg.flat_from)
            } else {
                u32::MAX
            },
            v_cap: cap.visible,
            plane_depth: cfg.plane_depth as u32,
            lambda_mv_geo: cfg.lambda_mv_geo,
            lambda_mv_ncc: cfg.lambda_mv_ncc,
            mv_from: if mv_on { at(cfg.mv_from) } else { u32::MAX },
            mv_stride: cfg.mv_stride.max(1),
            mv_max_err: cfg.mv_max_err,
            ncc_min_var: cfg.ncc_min_var,
            pad: 0,
        };
        Ok(Self {
            kernels: GeomKernels::load(gpu, variant)?,
            buf: GeomBuffers::alloc(gpu, cap, true)?,
            cfg,
            params,
            mv: None,
            mv_sums: (0.0, 0.0, 0.0),
        })
    }

    /// Turns the multi-view terms on (when their weights are positive): the
    /// view table, neighbours and the depth cache. `images` holds the photos
    /// the trainer samples (its views are the cache slots).
    pub fn enable_multiview(
        &mut self,
        gpu: &Gpu,
        views: &[View],
        images: &ImageCache,
    ) -> Result<()> {
        if self.cfg.lambda_mv_geo <= 0.0 && self.cfg.lambda_mv_ncc <= 0.0 {
            return Ok(());
        }
        let train = &images.views;
        let nb = neighbours(views, train, (self.cfg.mv_neighbours as usize).min(8));
        let max_px = train
            .iter()
            .map(|&i| views[i].width as usize * views[i].height as usize)
            .max()
            .unwrap_or(1);
        let mut stride = 1u32;
        while train.len() * max_px.div_ceil((stride * stride) as usize) * 4 > CACHE_BYTES {
            stride += 1;
        }
        let slot_px = train
            .iter()
            .map(|&i| (views[i].width.div_ceil(stride) * views[i].height.div_ceil(stride)) as usize)
            .max()
            .unwrap_or(1);
        let table: Vec<MvView> = views
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let mut m = MvView {
                    fx: v.fx as f32,
                    fy: v.fy as f32,
                    cx: v.cx as f32,
                    cy: v.cy as f32,
                    width: v.width,
                    height: v.height,
                    slot: u32::MAX,
                    ..Default::default()
                };
                for r in 0..3 {
                    for c in 0..3 {
                        m.r[3 * r + c] = v.r_cw[r][c] as f32;
                    }
                }
                m.t = v.t_cw.map(|x| x as f32);
                if let Some(e) = train.iter().position(|&j| j == i) {
                    let ent = images.entries[e];
                    if ent.width == v.width && ent.height == v.height {
                        m.slot = e as u32;
                        m.img_lo = ent.offset as u32;
                        m.img_hi = (ent.offset >> 32) as u32;
                        m.n_nb = nb[i].len() as u32;
                        for (k, &j) in nb[i].iter().enumerate() {
                            m.nb[k] = j as u32;
                        }
                    }
                }
                m
            })
            .collect();
        // Drop neighbours whose photo is not at the view size.
        let table: Vec<MvView> = table
            .iter()
            .map(|m| {
                let mut m = *m;
                let keep: Vec<u32> = m.nb[..m.n_nb as usize]
                    .iter()
                    .copied()
                    .filter(|&j| table[j as usize].slot != u32::MAX)
                    .collect();
                m.n_nb = keep.len() as u32;
                m.nb = [0; 8];
                m.nb[..keep.len()].copy_from_slice(&keep);
                m
            })
            .collect();
        let s = &gpu.stream;
        let used = table.iter().filter(|m| m.n_nb > 0).count();
        tracing::info!(
            "multi-view terms: {used}/{} views with neighbours, depth cache stride {stride}, {} MiB",
            train.len(),
            (train.len() * slot_px * 4) >> 20
        );
        self.mv = Some(MvState {
            views: s.clone_htod(&table).cuda("upload mv views")?,
            cache: s
                .alloc_zeros::<f32>(train.len().max(1) * slot_px)
                .cuda("alloc mv depth cache")?,
            stride,
            slot_px: slot_px as u32,
        });
        Ok(())
    }

    fn check(&self, raster: &GpuRaster) -> Result<()> {
        let c = raster.frame.cap;
        anyhow::ensure!(
            c.visible <= self.buf.cap.visible && c.pixels <= self.buf.cap.pixels,
            "geometry buffers were sized for {:?}, the raster is {:?}",
            self.buf.cap,
            c
        );
        Ok(())
    }

    /// After `raster_fwd(Train)`, before `raster_bwd`: losses and the
    /// alpha-chain gradients into `g2d`.
    pub fn after_forward(
        &mut self,
        gpu: &Gpu,
        raster: &mut GpuRaster,
        g: &Gaussians,
        images: Option<&ImageCache>,
    ) -> Result<()> {
        self.check(raster)?;
        let mut gp = self.params;
        gp.v_cap = raster.frame.cap.visible;
        let k = &self.kernels;
        render(gpu, k, raster, g, &mut self.buf, gp)?;
        let s = &gpu.stream;
        let f = &raster.frame;
        let b = &mut self.buf;
        let (Some(gpix), Some(gz), Some(gn), Some(sums)) =
            (&mut b.gpix, &mut b.gz, &mut b.gn, &mut b.sums)
        else {
            unreachable!("training buffers are allocated")
        };
        {
            let mut l = s.launch_builder(&k.pixel);
            l.arg(&f.iter.params)
                .arg(&f.iter.cam)
                .arg(&b.out)
                .arg(&gp)
                .arg(&mut *gpix)
                .arg(&mut *sums)
                .arg(&f.iter.status);
            // SAFETY: arguments match `geom_pixel`; pixels < width·height ≤ cap.pixels.
            unsafe { l.launch(grid_stride(f.cap.pixels)) }.cuda("geom_pixel")?;
        }
        if let (Some(mv), Some(images)) = (&mut self.mv, images) {
            let mut l = s.launch_builder(&k.mv);
            l.arg(&f.iter.params)
                .arg(&f.iter.cam)
                .arg(&mv.views)
                .arg(&images.pool)
                .arg(&mv.cache)
                .arg(&mv.stride)
                .arg(&mv.slot_px)
                .arg(&b.out)
                .arg(&gp)
                .arg(&mut *gpix)
                .arg(&mut *sums)
                .arg(&f.iter.status);
            // SAFETY: arguments match `geom_mv`; cam_index < views.len() (the table
            // has every scene view), slots < the cache's slot count, image offsets
            // are entries of `images`, pixels < width·height ≤ cap.pixels.
            unsafe { l.launch(grid_stride(f.cap.pixels / 4)) }.cuda("geom_mv")?;
            let mut l = s.launch_builder(&k.cache);
            l.arg(&f.iter.params)
                .arg(&f.iter.cam)
                .arg(&mv.views)
                .arg(&b.out)
                .arg(&mut mv.cache)
                .arg(&mv.stride)
                .arg(&mv.slot_px)
                .arg(&gp)
                .arg(&f.iter.status);
            // SAFETY: arguments match `geom_cache`; writes ≤ slot_px floats of slot < n_slots.
            unsafe { l.launch(grid_stride(f.cap.pixels)) }.cuda("geom_cache")?;
        }
        let mut l = s.launch_builder(&k.bwd);
        l.arg(&f.iter.params)
            .arg(&f.iter.cam)
            .arg(&f.rec)
            .arg(&b.aux)
            .arg(&f.inst_v[f.tile_sorted])
            .arg(&f.tile_start)
            .arg(&gp)
            .arg(&b.out)
            .arg(&*gpix)
            .arg(&mut raster.g2d.g)
            .arg(&mut *gz)
            .arg(&mut *gn)
            .arg(&f.iter.status);
        let cfg = LaunchConfig {
            grid_dim: (f.cap.tiles.max(1), 1, 1),
            block_dim: (k.tile_px, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: arguments match `geom_bwd`; g2d has 9 rows of v_cap, gz and gn
        // 1 and 3 rows of v_cap, v < v_cap.
        unsafe { l.launch(cfg) }.cuda("geom_bwd")?;
        Ok(())
    }

    /// After `raster.backward` (preprocess_bwd wrote the gradients at p),
    /// before the optimizer step.
    pub fn after_backward(
        &mut self,
        gpu: &Gpu,
        raster: &GpuRaster,
        g: &Gaussians,
        grads: &mut Grads,
    ) -> Result<()> {
        let f = &raster.frame;
        let mut gp = self.params;
        gp.v_cap = f.cap.visible;
        // The flattening loss is a mean over the visible set in PGSR; per
        // Gaussian here (Adam normalises the scale anyway).
        let (Some(gz), Some(gn)) = (&self.buf.gz, &self.buf.gn) else {
            unreachable!("training buffers are allocated")
        };
        let mut l = gpu.stream.launch_builder(&self.kernels.post);
        l.arg(&f.iter.params)
            .arg(&f.iter.cam)
            .arg(&f.vis_idx)
            .arg(&g.mean)
            .arg(&g.log_scale)
            .arg(&g.quat)
            .arg(gz)
            .arg(gn)
            .arg(&gp)
            .arg(&mut grads.mean)
            .arg(&mut grads.log_scale)
            .arg(&mut grads.quat)
            .arg(&f.iter.status);
        // SAFETY: arguments match `geom_post`; v < min(status[ST_N_VIS], v_cap),
        // p = vis_idx[v] < n ≤ the store's and the gradients' capacity.
        unsafe { l.launch(grid_stride(f.cap.visible)) }.cuda("geom_post")?;
        Ok(())
    }

    /// Mean (distortion, normal) loss per frame since the last call (host sync).
    pub fn take_sums(&mut self, gpu: &Gpu, frames: u32) -> Result<(f64, f64)> {
        let Some(s) = &mut self.buf.sums else {
            return Ok((0.0, 0.0));
        };
        let v = gpu.stream.clone_dtoh(&*s).cuda("download geom sums")?;
        gpu.stream.memset_zeros(s).cuda("reset geom sums")?;
        let n = frames.max(1) as f64;
        let used = (v[4] as f64).max(1.0);
        self.mv_sums = (v[2] as f64 / used, v[3] as f64 / used, v[4] as f64 / n);
        Ok((v[0] as f64 / n, v[1] as f64 / n))
    }
}

#[cfg(test)]
#[path = "geom_gpu_tests.rs"]
mod gpu_tests;
