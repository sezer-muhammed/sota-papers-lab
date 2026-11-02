//! Splats + cameras → textured GLB. Owner: mesh-mode.
//!
//! 1. Median depth of every view from the rasterizer ([`super::geom`]).
//! 2. GPU TSDF fusion of those depths ([`super::tsdf`]) and marching cubes.
//! 3. Small disconnected pieces dropped (2DGS keeps its 50 largest clusters).
//! 4. Decimation ([`super::simplify`]), UV atlas ([`super::uv`]), texture bake
//!    from the training photos ([`super::bake`]), GLB ([`super::glb`]).

use std::time::Instant;

use anyhow::ensure;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use spl_core::{Result, bail};
use ts_rs::TS;

use super::geom::{self, GeomBuffers, GeomKernels, GeomParams};
use super::glb::{TextureFormat, write_glb};
use super::mc::ExtractOptions;
use super::tsdf::{DepthCamera, TsdfParams, TsdfVolume};
use super::{BakeCamera, Mesh, TriMesh};
use crate::frame::{CameraTable, FrameParams, IterParams};
use crate::gpu::Gpu;
use crate::io::images::load_view_image;
use crate::io::scene::View;
use crate::kernels::layout::{EPS2D_DEFAULT, FLAG_AA};
use crate::kernels::raster::{FrameSource, GpuRaster, SplatRaster};
use crate::kernels::{Capacity, RasterVariant, RenderMode};
use crate::model::{Gaussians, GaussiansHost};

fn d_res() -> u32 {
    1024
}
fn d_trunc() -> f32 {
    4.0
}
fn d_target() -> u32 {
    300_000
}
fn d_atlas() -> u32 {
    4096
}
fn d_bake_views() -> u32 {
    24
}
fn d_chart_max() -> u32 {
    30_000
}
fn d_min_weight() -> f32 {
    3.0
}
fn d_keep() -> u32 {
    50
}
fn yes() -> bool {
    true
}

/// Meshing options (the `mesh` flag of train and the `image_to_3d.mesh` op).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct MeshConfig {
    /// TSDF samples across the longest side of the bounds (2DGS: 1024).
    #[serde(default = "d_res")]
    pub resolution: u32,
    /// Voxel size in scene units; overrides `resolution`.
    #[serde(default)]
    #[ts(optional)]
    pub voxel: Option<f32>,
    /// Truncation in voxels (2DGS DTU: 4).
    #[serde(default = "d_trunc")]
    pub trunc_voxels: f32,
    /// Depths beyond this are not fused (2DGS `depth_trunc`).
    #[serde(default)]
    #[ts(optional)]
    pub depth_max: Option<f32>,
    /// World AABB (min xyz, max xyz); the 1st–99th percentile box of the
    /// opaque Gaussians, padded 5%, when absent.
    #[serde(default)]
    #[ts(optional)]
    pub bounds: Option<[f32; 6]>,
    /// Samples observed by fewer views are not meshed (drops floaters seen once).
    #[serde(default = "d_min_weight")]
    pub min_weight: f32,
    /// Keep the largest this many connected pieces (0 = all).
    #[serde(default = "d_keep")]
    pub keep_components: u32,
    /// Triangle budget after decimation (0 = no decimation).
    #[serde(default = "d_target")]
    pub target_triangles: u32,
    /// Bake a texture atlas from the photos (else vertex-less grey material).
    #[serde(default = "yes")]
    pub texture: bool,
    #[serde(default = "d_atlas")]
    pub atlas_size: u32,
    /// Photos used for the bake, spread evenly over the views.
    #[serde(default = "d_bake_views")]
    pub bake_views: u32,
    /// Chart unwrapping (uvunwrap) up to this many triangles, the
    /// per-triangle grid atlas above (chart unwrapping is superlinear).
    #[serde(default = "d_chart_max")]
    pub chart_uv_max_triangles: u32,
    /// Fuse the plane depth `Ds / (N . ray)` (PGSR) instead of the median depth.
    #[serde(default = "yes")]
    pub plane_depth: bool,
}

impl Default for MeshConfig {
    fn default() -> Self {
        Self {
            resolution: d_res(),
            voxel: None,
            trunc_voxels: d_trunc(),
            depth_max: None,
            bounds: None,
            min_weight: d_min_weight(),
            keep_components: d_keep(),
            target_triangles: d_target(),
            texture: true,
            atlas_size: d_atlas(),
            bake_views: d_bake_views(),
            chart_uv_max_triangles: d_chart_max(),
            plane_depth: true,
        }
    }
}

/// What the mesher did and how long each stage took.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct MeshSummary {
    pub views_fused: u32,
    pub voxel: f32,
    pub blocks: u32,
    pub raw_triangles: u32,
    pub kept_triangles: u32,
    pub triangles: u32,
    pub vertices: u32,
    pub atlas_size: u32,
    pub glb_bytes: u64,
    pub depth_ms: f64,
    pub fuse_ms: f64,
    pub extract_ms: f64,
    pub clean_ms: f64,
    pub simplify_ms: f64,
    pub uv_ms: f64,
    pub bake_ms: f64,
    pub glb_ms: f64,
    pub total_s: f64,
    /// What the mesher changed to fit (clipped bounds, coarser voxel).
    #[serde(default)]
    pub notes: Vec<String>,
}

/// The mesher's results: the fused surface before decimation (for
/// evaluation), and the GLB.
pub struct MeshResult {
    pub raw: Mesh,
    pub glb: Vec<u8>,
    pub summary: MeshSummary,
}

pub fn depth_camera(v: &View) -> DepthCamera {
    DepthCamera {
        r_cw: v.r_cw.map(|r| r.map(|x| x as f32)),
        t_cw: v.t_cw.map(|x| x as f32),
        fx: v.fx as f32,
        fy: v.fy as f32,
        cx: v.cx as f32,
        cy: v.cy as f32,
        width: v.width,
        height: v.height,
    }
}

fn bake_camera(v: &View) -> BakeCamera {
    let d = depth_camera(v);
    BakeCamera {
        r_cw: d.r_cw,
        t_cw: d.t_cw,
        fx: d.fx,
        fy: d.fy,
        cx: d.cx,
        cy: d.cy,
        width: d.width,
        height: d.height,
    }
}

/// The 1st–99th percentile box of the Gaussians with opacity ≥ 0.5, padded 5%.
pub fn auto_bounds(cloud: &GaussiansHost) -> [f32; 6] {
    let mut axes: [Vec<f32>; 3] = Default::default();
    for i in 0..cloud.len() {
        if cloud.opacities[i] >= 0.0 {
            for (k, a) in axes.iter_mut().enumerate() {
                a.push(cloud.means[3 * i + k]);
            }
        }
    }
    let mut b = [0.0; 6];
    for (k, a) in axes.iter_mut().enumerate() {
        if a.is_empty() {
            return [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        }
        a.sort_unstable_by(f32::total_cmp);
        let q = |f: f64| a[((a.len() - 1) as f64 * f).round() as usize];
        let (lo, hi) = (q(0.01), q(0.99));
        let pad = 0.05 * (hi - lo).max(1e-6);
        b[k] = lo - pad;
        b[3 + k] = hi + pad;
    }
    b
}

/// The fusion box: [`auto_bounds`], clipped to the region the cameras cover
/// when the splat reaches far beyond them (unbounded outdoor scenes, where
/// the background Gaussians would otherwise stretch the volume). Object
/// captures (DTU: cameras on a cap around a small object) are unchanged.
pub fn scene_bounds(cloud: &GaussiansHost, views: &[View]) -> ([f32; 6], bool) {
    let b = auto_bounds(cloud);
    if views.len() < 2 {
        return (b, false);
    }
    let mut c = [
        f32::INFINITY,
        f32::INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    ];
    for v in views {
        let p = v.center();
        for k in 0..3 {
            c[k] = c[k].min(p[k] as f32);
            c[3 + k] = c[3 + k].max(p[k] as f32);
        }
    }
    let cam = (0..3).map(|k| c[3 + k] - c[k]).fold(0.0f32, f32::max);
    let size = (0..3).map(|k| b[3 + k] - b[k]).fold(0.0f32, f32::max);
    if !(cam > 0.0 && cam.is_finite()) || size <= CLIP_RATIO * cam {
        return (b, false);
    }
    // Keep the camera box grown by its longest side on every face.
    let mut out = b;
    for k in 0..3 {
        let mid = 0.5 * (c[k] + c[3 + k]);
        out[k] = b[k].max(mid - 1.5 * cam);
        out[3 + k] = b[3 + k].min(mid + 1.5 * cam);
        if out[k] >= out[3 + k] {
            return (b, false);
        }
    }
    (out, true)
}

/// Splat boxes wider than this many camera extents are clipped.
const CLIP_RATIO: f32 = 3.0;
/// Voxel growth for the one retry after a TSDF pool overflow.
const RETRY_COARSEN: f32 = 1.75;

/// glTF requires unit normals: degenerate faces leave zero vectors.
pub fn unit_normals(normals: &mut [[f32; 3]]) {
    for n in normals {
        let l = (n[0] as f64).hypot(n[1] as f64).hypot(n[2] as f64);
        *n = if l > 1e-6 {
            n.map(|x| (x as f64 / l) as f32)
        } else {
            [0.0, 0.0, 1.0]
        };
    }
}

/// Drops all but the `keep` largest edge-connected pieces (by face count).
pub fn keep_largest(mesh: &mut Mesh, keep: usize) {
    if keep == 0 || mesh.faces.is_empty() {
        return;
    }
    let n = mesh.positions.len();
    let mut parent: Vec<u32> = (0..n as u32).collect();
    fn find(p: &mut [u32], mut x: u32) -> u32 {
        while p[x as usize] != x {
            p[x as usize] = p[p[x as usize] as usize];
            x = p[x as usize];
        }
        x
    }
    for f in &mesh.faces {
        let a = find(&mut parent, f[0]);
        for &v in &f[1..] {
            let b = find(&mut parent, v);
            if a != b {
                parent[b as usize] = a;
            }
        }
    }
    let mut count = std::collections::HashMap::<u32, usize>::new();
    let roots: Vec<u32> = mesh.faces.iter().map(|f| find(&mut parent, f[0])).collect();
    for &r in &roots {
        *count.entry(r).or_default() += 1;
    }
    if count.len() <= keep {
        return;
    }
    let mut sizes: Vec<(usize, u32)> = count.iter().map(|(&r, &c)| (c, r)).collect();
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    let kept: std::collections::HashSet<u32> = sizes[..keep].iter().map(|&(_, r)| r).collect();
    let faces = std::mem::take(&mut mesh.faces);
    mesh.faces = faces
        .into_iter()
        .zip(roots)
        .filter(|(_, r)| kept.contains(r))
        .map(|(f, _)| f)
        .collect();
    mesh.remove_unreferenced();
}

/// Renders the median depth of `views` and fuses it; returns the volume.
pub fn fuse(
    gpu: &Gpu,
    cloud: &GaussiansHost,
    views: &[View],
    cfg: &MeshConfig,
    summary: &mut MeshSummary,
) -> Result<TsdfVolume> {
    fuse_with(gpu, cloud, views, cfg, summary, &mut |_, _| Ok(()))
}

/// [`fuse`], calling `on_view(i, buffers)` after each view's geometry render
/// (diagnostics: depth and normal maps).
pub fn fuse_with(
    gpu: &Gpu,
    cloud: &GaussiansHost,
    views: &[View],
    cfg: &MeshConfig,
    summary: &mut MeshSummary,
    on_view: &mut dyn FnMut(usize, &GeomBuffers) -> Result<()>,
) -> Result<TsdfVolume> {
    let n = cloud.len();
    ensure!(n > 0, "the splat is empty");
    ensure!(!views.is_empty(), "no cameras to fuse");
    let cap = n.next_multiple_of(1024);
    let mut g = Gaussians::alloc(gpu, cap, cloud.antialiased)?;
    g.upload(gpu, cloud)?;
    let cams = CameraTable::upload(gpu, views)?;
    let (mw, mh) = views
        .iter()
        .fold((1, 1), |(w, h), v| (w.max(v.width), h.max(v.height)));
    let variant = RasterVariant::default();
    let mut instances = (n as u64 * 16).clamp(1 << 20, 1 << 27) as u32;
    let mut raster = GpuRaster::new(
        gpu,
        Capacity::new(cap as u32, instances, mw, mh, &variant),
        variant,
        false,
    )?;
    let kernels = GeomKernels::load(gpu, &variant)?;
    let mut buf = GeomBuffers::alloc(gpu, raster.frame.cap, false)?;

    let b = match cfg.bounds {
        Some(b) => b,
        None => {
            let (b, clipped) = scene_bounds(cloud, views);
            if clipped {
                summary.notes.push(format!(
                    "bounds clipped to the camera-covered region {b:?} (the splat reaches far beyond the cameras)"
                ));
            }
            b
        }
    };
    let lo = [b[0], b[1], b[2]];
    let size = (0..3).map(|k| b[3 + k] - b[k]).fold(0.0f32, f32::max);
    ensure!(size > 0.0 && size.is_finite(), "empty mesh bounds {b:?}");
    let res = match cfg.voxel {
        Some(v) => {
            ensure!(v > 0.0, "voxel must be positive");
            (size / v).ceil() as u32
        }
        None => cfg.resolution,
    }
    .clamp(16, 8192);
    let mut tp = TsdfParams::for_bounds(lo, size, res);
    if let Some(v) = cfg.voxel {
        tp.voxel = v;
    }
    tp.trunc = cfg.trunc_voxels * tp.voxel;
    if let Some(d) = cfg.depth_max {
        tp.depth_max = d;
    }
    // A third of the free memory for the block pool, 4 KiB + 12 B per block.
    let (free, _) = gpu.mem_info()?;
    tp.max_blocks = ((free / 3) / 4108).clamp(1 << 14, 1 << 21) as u32;
    tp.hash_slots = (2 * tp.max_blocks).next_power_of_two();
    let mut vol = TsdfVolume::new(gpu, tp)?;
    // Fuse only surface points inside the bounds (plus the truncation band).
    let clip_lo = [0, 1, 2].map(|k| b[k] - tp.trunc);
    let clip_hi = [0, 1, 2].map(|k| b[3 + k] + tp.trunc);
    summary.voxel = tp.voxel;

    let gp = GeomParams {
        near_m: 1.0,
        v_cap: raster.frame.cap.visible,
        plane_depth: cfg.plane_depth as u32,
        ..Default::default()
    };
    let started = Instant::now();
    let mut depth_s = 0.0;
    for (i, v) in views.iter().enumerate() {
        let params = IterParams {
            frame: FrameParams {
                iteration: 1,
                cam_index: i as u32,
                n_gaussians: n as u32,
                sh_degree: 0,
                eps2d: EPS2D_DEFAULT,
                flags: if cloud.antialiased { FLAG_AA } else { 0 },
                render_w: v.width,
                render_h: v.height,
                box_beta: 1.0,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut tries = 0;
        loop {
            let t0 = Instant::now();
            raster.forward(
                gpu,
                &mut g,
                &cams,
                FrameSource::Eager(&params),
                RenderMode::Infer,
            )?;
            let st = raster.frame.iter.read_status(gpu)?;
            if st.overflow == 0 {
                geom::render(gpu, &kernels, &raster, &g, &mut buf, gp)?;
                geom::clip_depth(gpu, &kernels, &raster, &mut buf, clip_lo, clip_hi)?;
                gpu.sync()?;
                depth_s += t0.elapsed().as_secs_f64();
                break;
            }
            tries += 1;
            if tries > 4 {
                bail!("mesh: raster buffers kept overflowing");
            }
            instances = (st.n_instances as u64 * 5 / 4)
                .max(instances as u64 * 2)
                .min(1 << 30) as u32;
            raster = GpuRaster::new(
                gpu,
                Capacity::new(cap as u32, instances, mw, mh, &variant),
                variant,
                false,
            )?;
        }
        vol.integrate_device(&buf.out, &depth_camera(v))?;
        on_view(i, &buf)?;
    }
    gpu.sync()?;
    summary.views_fused = views.len() as u32;
    summary.depth_ms = depth_s * 1e3;
    summary.fuse_ms = started.elapsed().as_secs_f64() * 1e3 - summary.depth_ms;
    summary.blocks = vol.blocks();
    Ok(vol)
}

/// The full pipeline: fuse `views`, extract, clean, decimate, unwrap, bake
/// from the photos of `views` (their `image_path`s), GLB.
pub fn build(
    gpu: &Gpu,
    cloud: &GaussiansHost,
    views: &[View],
    cfg: &MeshConfig,
) -> Result<MeshResult> {
    let started = Instant::now();
    let mut summary = MeshSummary::default();
    let vol = match fuse(gpu, cloud, views, cfg, &mut summary) {
        Err(e) if is_overflow(&e) => {
            // Same box, coarser samples: blocks fall by about RETRY_COARSEN².
            let mut retry = cfg.clone();
            retry.voxel = Some(summary.voxel * RETRY_COARSEN);
            let note = format!(
                "{e:#}; retried with voxel {:.4} (was {:.4})",
                summary.voxel * RETRY_COARSEN,
                summary.voxel
            );
            let mut notes = std::mem::take(&mut summary.notes);
            notes.push(note);
            if retry.bounds.is_none() {
                retry.bounds = Some(scene_bounds(cloud, views).0);
            }
            summary = MeshSummary::default();
            let vol = fuse(gpu, cloud, views, &retry, &mut summary)?;
            summary.notes = notes;
            vol
        }
        r => r?,
    };
    let t = Instant::now();
    let mut raw = vol.extract(ExtractOptions {
        min_weight: cfg.min_weight,
        ..Default::default()
    })?;
    drop(vol);
    summary.extract_ms = t.elapsed().as_secs_f64() * 1e3;
    summary.raw_triangles = raw.faces.len() as u32;
    let t = Instant::now();
    keep_largest(&mut raw, cfg.keep_components as usize);
    summary.kept_triangles = raw.faces.len() as u32;
    summary.clean_ms = t.elapsed().as_secs_f64() * 1e3;
    ensure!(!raw.faces.is_empty(), "the fused surface is empty");

    let mut tri = TriMesh::from(raw.clone());
    if cfg.target_triangles > 0 && tri.num_triangles() > cfg.target_triangles as usize {
        let (m, ms) = super::simplify::simplify(&tri, cfg.target_triangles as usize, 1e-2)?;
        tri = m;
        summary.simplify_ms = ms;
    }
    tri.compute_normals();
    let (uv, ms) = if tri.num_triangles() <= cfg.chart_uv_max_triangles as usize {
        super::uv::generate_atlas(&tri, cfg.atlas_size, 2)?
    } else {
        super::fastuv::grid_atlas(&tri, cfg.atlas_size)?
    };
    summary.uv_ms = ms;
    let mut uv = uv;
    unit_normals(&mut uv.normals);
    let atlas = if cfg.texture && !views.is_empty() {
        let k = (cfg.bake_views.max(1) as usize).min(views.len());
        let picks: Vec<usize> = (0..k).map(|i| i * views.len() / k).collect();
        let imgs = picks
            .iter()
            .map(|&i| load_view_image(&views[i]))
            .collect::<Result<Vec<_>>>()?;
        let cams: Vec<BakeCamera> = picks.iter().map(|&i| bake_camera(&views[i])).collect();
        let data: Vec<&[u8]> = imgs.iter().map(|m| m.data.as_slice()).collect();
        let (a, ms) = super::fastbake::bake(&uv, &cams, &data)?;
        summary.bake_ms = ms;
        a
    } else {
        let mut a = super::TextureAtlas::new(4, 4);
        a.pixels.fill(200);
        a
    };
    let (glb, ms) = write_glb(&uv, &atlas, TextureFormat::Jpeg)?;
    summary.glb_ms = ms;
    summary.triangles = (uv.indices.len() / 3) as u32;
    summary.vertices = uv.positions.len() as u32;
    summary.atlas_size = atlas.width;
    summary.glb_bytes = glb.len() as u64;
    summary.total_s = started.elapsed().as_secs_f64();
    Ok(MeshResult { raw, glb, summary })
}

fn is_overflow(e: &spl_core::Error) -> bool {
    format!("{e:#}").contains("TSDF overflow")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keep_largest_drops_small_pieces() {
        let mut m = Mesh {
            positions: (0..9).map(|i| [i as f32, 0.0, 0.0]).collect(),
            normals: Vec::new(),
            faces: vec![[0, 1, 2], [1, 2, 3], [4, 5, 6]],
        };
        keep_largest(&mut m, 1);
        assert_eq!(m.faces.len(), 2);
        assert_eq!(m.positions.len(), 4);
    }

    #[test]
    fn bounds_cover_the_opaque_points() {
        let mut c = GaussiansHost::zeros(1000);
        for i in 0..1000 {
            c.means[3 * i] = i as f32 / 999.0;
            c.opacities[i] = 1.0;
        }
        let b = auto_bounds(&c);
        assert!(b[0] < 0.02 && b[3] > 0.98, "{b:?}");
    }

    fn view_at(x: f64) -> View {
        View {
            name: String::new(),
            camera: 1,
            r_cw: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            t_cw: [-x, 0.0, 0.0],
            fx: 1.0,
            fy: 1.0,
            cx: 0.5,
            cy: 0.5,
            width: 1,
            height: 1,
            image_path: Default::default(),
        }
    }

    #[test]
    fn bounds_clip_far_background_only() {
        // Object at the origin, cameras within ±1: no clipping.
        let mut c = GaussiansHost::zeros(1000);
        for i in 0..1000 {
            c.means[3 * i] = i as f32 / 999.0 - 0.5;
            c.opacities[i] = 1.0;
        }
        let views = [view_at(-1.0), view_at(1.0)];
        let (b, clipped) = scene_bounds(&c, &views);
        assert!(!clipped && b == auto_bounds(&c));
        // A background shell out to ±100: clipped to cameras ± 3.
        for i in 0..1000 {
            c.means[3 * i] = (i as f32 / 999.0 - 0.5) * 200.0;
        }
        let (b, clipped) = scene_bounds(&c, &views);
        assert!(clipped && b[0] == -3.0 && b[3] == 3.0, "{b:?}");
    }

    #[test]
    fn config_defaults_parse() {
        let c: MeshConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(c, MeshConfig::default());
    }
}

#[cfg(test)]
#[path = "pipeline_gpu_tests.rs"]
mod gpu_tests;
