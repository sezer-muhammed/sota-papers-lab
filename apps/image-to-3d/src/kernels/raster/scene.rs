//! Raster test scenes (test-only, owner: raster E3): random 2D splat records,
//! CPU tile binning (rectangles like gsplat, or the exact subtile test),
//! chain offsets, a CPU f64 reference of the blend forward and backward
//! (spec-raster 5.1 with gsplat's decisions), and the upload into the
//! per-frame buffers the kernels read. Preprocess and binning are other
//! engineers' stages, so these scenes are how the raster is tested alone.

use spl_core::Result;
use spl_gpu::cudarc::driver::CudaSlice;

use crate::frame::{FrameCam, IterParams};
use crate::gpu::{CudaResultExt, Gpu};
use crate::kernels::layout::{
    ALPHA_MAX, ALPHA_MIN, BUCKET, KBIN_ABS, KBIN_REL, LOG2E, ONE_MINUS_ALPHA_MIN, ST_N_CHAINS,
    ST_N_INST, ST_N_VIS, STATUS_LEN, T_MIN, TILE_W,
};
use crate::kernels::{
    BlendSemantics, Capacity, FrameBuffers, Grad2d, PixelGrads, RasterVariant, RenderOut, SplatRec,
};
use crate::rng::Philox;

use super::ckpt_words;

const LN2: f64 = std::f64::consts::LN_2;

/// Visible splats of one view as the raster sees them (v = index).
#[derive(Clone, Debug)]
pub struct Scene2d {
    pub width: u32,
    pub height: u32,
    pub recs: Vec<SplatRec>,
    pub bg: [f32; 3],
}

/// How random scenes are drawn.
#[derive(Clone, Copy, Debug)]
pub struct SceneOpts {
    pub n: usize,
    pub width: u32,
    pub height: u32,
    /// Standard deviations are log-uniform in [0.3, max_sigma] px (before the 0.3 px² dilation).
    pub max_sigma: f32,
    /// Opacity uniform in [0.05, max_opacity]; above ALPHA_MAX exercises the clamp.
    pub max_opacity: f32,
    /// Fraction of colour channels marked clamped (colour 0, flag bit set).
    pub clamp_frac: f32,
    /// Means may fall this far outside the image.
    pub margin: f32,
    pub bg: [f32; 3],
}

impl Default for SceneOpts {
    fn default() -> Self {
        Self {
            n: 600,
            width: 67,
            height: 53,
            max_sigma: 12.0,
            max_opacity: 0.98,
            clamp_frac: 0.05,
            margin: 6.0,
            bg: [0.1, 0.2, 0.3],
        }
    }
}

/// A record from the unscaled conic (A, B, C) of sigma = ½(A dx² + C dy²) + B dx dy.
pub fn record(
    mean: [f32; 2],
    conic: [f64; 3],
    o: f32,
    rgb: [f32; 3],
    z: f32,
    flags: u32,
) -> SplatRec {
    let l2e = LOG2E as f64;
    SplatRec {
        mx: mean[0],
        my: mean[1],
        ca: (-0.5 * l2e * conic[0]) as f32,
        cb: (-l2e * conic[1]) as f32,
        cc: (-0.5 * l2e * conic[2]) as f32,
        o,
        r: rgb[0],
        g: rgb[1],
        b: rgb[2],
        pthr: (255.0 * o as f64).log2() as f32,
        z,
        flags,
    }
}

/// The unscaled conic (A, B, C) of a record (f64 from the stored f32 fields).
pub fn conic(r: &SplatRec) -> [f64; 3] {
    [
        -2.0 * LN2 * r.ca as f64,
        -LN2 * r.cb as f64,
        -2.0 * LN2 * r.cc as f64,
    ]
}

pub fn random_scene(seed: u64, o: &SceneOpts) -> Scene2d {
    let p = Philox::new(seed);
    let u = |i: usize, k: u32| p.uniform(i as u32, k, 0);
    let recs = (0..o.n)
        .map(|i| {
            let mx = -o.margin + u(i, 0) * (o.width as f32 + 2.0 * o.margin);
            let my = -o.margin + u(i, 1) * (o.height as f32 + 2.0 * o.margin);
            let ls = |k| {
                (0.3f64.ln() + u(i, k) as f64 * ((o.max_sigma as f64).ln() - 0.3f64.ln())).exp()
            };
            let (s1, s2) = (ls(2), ls(3));
            let th = u(i, 4) as f64 * std::f64::consts::PI;
            let (c, s) = (th.cos(), th.sin());
            let sxx = c * c * s1 * s1 + s * s * s2 * s2 + 0.3;
            let syy = s * s * s1 * s1 + c * c * s2 * s2 + 0.3;
            let sxy = c * s * (s1 * s1 - s2 * s2);
            let det = sxx * syy - sxy * sxy;
            let con = [syy / det, -sxy / det, sxx / det];
            let op = 0.05 + u(i, 5) * (o.max_opacity - 0.05);
            let mut rgb = [u(i, 6), u(i, 7), u(i, 8)];
            let mut flags = 0;
            for (ch, c) in rgb.iter_mut().enumerate() {
                if u(i, 9 + ch as u32) < o.clamp_frac {
                    *c = 0.0;
                    flags |= 1 << ch;
                }
            }
            let z = 1.0 + 9.0 * u(i, 12);
            record([mx, my], con, op, rgb, z, flags)
        })
        .collect();
    Scene2d {
        width: o.width,
        height: o.height,
        recs,
        bg: o.bg,
    }
}

/// Sorted per-tile instance lists.
#[derive(Clone, Debug, Default)]
pub struct TileLists {
    pub tile_h: u32,
    pub grid_w: u32,
    pub grid_h: u32,
    /// [n_tiles + 1].
    pub tile_start: Vec<u32>,
    pub inst_v: Vec<u32>,
}

impl TileLists {
    pub fn n_tiles(&self) -> u32 {
        self.grid_w * self.grid_h
    }

    pub fn instances(&self) -> u32 {
        self.inst_v.len() as u32
    }

    /// `chain_off[t]` = chains before tile t, chains per tile ⌈⌈n_t/32⌉/K⌉,
    /// total at `[n_tiles]` (what bin_scan writes).
    pub fn chain_off(&self, k: u32) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.tile_start.len());
        let mut acc = 0u32;
        for t in 0..self.n_tiles() as usize {
            out.push(acc);
            let n = self.tile_start[t + 1] - self.tile_start[t];
            acc += n.div_ceil(BUCKET).div_ceil(k);
        }
        out.push(acc);
        out
    }

    /// Lists from per-tile membership, depth-sorted (stable on v), like the
    /// two-stage sort of spec-raster 3.
    pub fn build(
        scene: &Scene2d,
        tile_h: u32,
        member: impl Fn(&SplatRec, u32, u32) -> bool,
    ) -> Self {
        let grid_w = scene.width.div_ceil(TILE_W);
        let grid_h = scene.height.div_ceil(tile_h);
        let mut lists: Vec<Vec<u32>> = vec![Vec::new(); (grid_w * grid_h) as usize];
        let mut order: Vec<u32> = (0..scene.recs.len() as u32).collect();
        order.sort_by(|&a, &b| {
            scene.recs[a as usize]
                .z
                .total_cmp(&scene.recs[b as usize].z)
                .then(a.cmp(&b))
        });
        for &v in &order {
            let r = &scene.recs[v as usize];
            let Some((x0, x1, y0, y1)) = bbox(r) else {
                continue;
            };
            let tx0 = (x0.max(0.0) / TILE_W as f64).floor() as i64;
            let tx1 = ((x1.min(scene.width as f64) / TILE_W as f64).floor() as i64)
                .min(grid_w as i64 - 1);
            let ty0 = (y0.max(0.0) / tile_h as f64).floor() as i64;
            let ty1 = ((y1.min(scene.height as f64) / tile_h as f64).floor() as i64)
                .min(grid_h as i64 - 1);
            for ty in ty0.max(0)..=ty1 {
                for tx in tx0.max(0)..=tx1 {
                    if member(r, tx as u32, ty as u32) {
                        lists[(ty as u32 * grid_w + tx as u32) as usize].push(v);
                    }
                }
            }
        }
        let mut tile_start = vec![0u32];
        let mut inst_v = Vec::new();
        for l in &lists {
            inst_v.extend_from_slice(l);
            tile_start.push(inst_v.len() as u32);
        }
        Self {
            tile_h,
            grid_w,
            grid_h,
            tile_start,
            inst_v,
        }
    }

    /// Every tile the record's (margin-inflated) bounding box touches, like
    /// gsplat's rectangle binning: lists longer than needed.
    pub fn rect(scene: &Scene2d, tile_h: u32) -> Self {
        Self::build(scene, tile_h, |_, _, _| true)
    }

    /// Only tiles with a subtile the exact test keeps (spec-raster 2.8 in f64).
    pub fn exact(scene: &Scene2d, tile_h: u32) -> Self {
        Self::build(scene, tile_h, |r, tx, ty| {
            subtile_mask_f64(r, (tx * TILE_W) as f64, (ty * tile_h) as f64, tile_h / 4) != 0
        })
    }
}

/// Pixel-space box of {alpha ≥ 1/255} with the binning margin plus 1 px.
fn bbox(r: &SplatRec) -> Option<(f64, f64, f64, f64)> {
    let (a, b, c) = (-r.ca as f64, -r.cb as f64, -r.cc as f64);
    let p = r.pthr as f64 * (1.0 + KBIN_REL as f64) + KBIN_ABS as f64;
    let d = 4.0 * a * c - b * b;
    if p <= 0.0 || d <= 0.0 {
        return None;
    }
    let xe = (4.0 * c * p / d).sqrt() + 1.0;
    let ye = (4.0 * a * p / d).sqrt() + 1.0;
    let (mx, my) = (r.mx as f64, r.my as f64);
    Some((mx - xe, mx + xe, my - ye, my + ye))
}

/// f64 version of `rs_subtile_mask` (cuda/raster_common.cuh) for `rows`
/// subtile rows: the binning a CPU test can trust.
pub fn subtile_mask_f64(r: &SplatRec, tx0: f64, ty0: f64, rows: u32) -> u32 {
    let (a, b, c) = (-r.ca as f64, -r.cb as f64, -r.cc as f64);
    let p = r.pthr as f64 * (1.0 + KBIN_REL as f64) + KBIN_ABS as f64;
    let d = 4.0 * a * c - b * b;
    if p <= 0.0 {
        return 0;
    }
    if d <= 0.0 || a <= 0.0 {
        return (1 << (rows * 2)) - 1;
    }
    let xe = (4.0 * c * p / d).sqrt();
    let ye = (4.0 * a * p / d).sqrt();
    let dyl = b * xe / (2.0 * c);
    let (mx, my) = (r.mx as f64, r.my as f64);
    let mut mask = 0;
    for row in 0..rows {
        let ya = ty0 + (row * 4) as f64 + 0.5 - my;
        let yb = ya + 3.0;
        if yb < -ye || ya > ye {
            continue;
        }
        let dl = dyl.clamp(ya, yb);
        let dr = (-dyl).clamp(ya, yb);
        let xl = mx + (-b * dl - (4.0 * a * p - d * dl * dl).max(0.0).sqrt()) / (2.0 * a);
        let xr = mx + (-b * dr + (4.0 * a * p - d * dr * dr).max(0.0).sqrt()) / (2.0 * a);
        for col in 0..2 {
            let xa = tx0 + (col * 8) as f64 + 0.5;
            if xl <= xa + 7.0 && xr >= xa {
                mask |= 1 << (row * 2 + col);
            }
        }
    }
    mask
}

/// Loss gradients w.r.t. the outputs (planar like the kernels').
#[derive(Clone, Debug)]
pub struct PixelGradsHost {
    pub d_rgb: Vec<f32>,
    pub d_alpha: Option<Vec<f32>>,
    pub d_extra: Vec<f32>,
}

impl PixelGradsHost {
    pub fn random(seed: u64, pixels: usize, extra_ch: usize, with_alpha: bool) -> Self {
        let p = Philox::new(seed);
        let draw = |k: u32, n: usize| {
            (0..n)
                .map(|i| p.uniform(i as u32, k, 1) - 0.5)
                .collect::<Vec<f32>>()
        };
        Self {
            d_rgb: draw(0, 3 * pixels),
            d_alpha: with_alpha.then(|| draw(1, pixels)),
            d_extra: draw(2, extra_ch * pixels),
        }
    }
}

/// f64 reference forward outputs.
#[derive(Clone, Debug)]
pub struct RefOut {
    pub rgb: Vec<f64>,
    pub trans: Vec<f64>,
    pub nc: Vec<u32>,
    pub extra: Vec<f64>,
    pub tile_max_nc: Vec<u32>,
    /// Per pixel, the blended v in order (for finite differences with frozen decisions).
    pub pairs: Vec<Vec<u32>>,
}

/// f64 reference 2D gradients, SoA like `Grad2d`.
#[derive(Clone, Debug)]
pub struct RefGrads {
    /// [9][V]: mx, my, A, B, C, o, r, g, b.
    pub g: Vec<f64>,
    pub abs: Vec<f64>,
    pub extra: Vec<f64>,
}

struct Contrib {
    v: usize,
    alpha: f64,
    a_raw: f64,
    g: f64,
    t: f64,
    dx: f64,
    dy: f64,
    through: bool,
}

/// The blend of spec-raster 4.2 / 5.1 in f64. Skip and stop decisions are
/// made in f32 exactly like the kernels (same fused exponent, f32 T), so
/// both blend the same (Gaussian, pixel) pairs; values are f64.
pub fn reference(
    scene: &Scene2d,
    lists: &TileLists,
    blend: BlendSemantics,
    extra_ch: usize,
    pg: &PixelGradsHost,
) -> (RefOut, RefGrads) {
    let (w, h) = (scene.width as usize, scene.height as usize);
    let np = w * h;
    let nv = scene.recs.len();
    let nch = 3 + extra_ch;
    let mut out = RefOut {
        rgb: vec![0.0; 3 * np],
        trans: vec![0.0; np],
        nc: vec![0; np],
        extra: vec![0.0; extra_ch * np],
        tile_max_nc: vec![0; lists.n_tiles() as usize],
        pairs: vec![Vec::new(); np],
    };
    let mut gr = RefGrads {
        g: vec![0.0; 9 * nv],
        abs: vec![0.0; 2 * nv],
        extra: vec![0.0; extra_ch * nv],
    };
    let bg = scene.bg.map(f64::from);
    for t in 0..lists.n_tiles() {
        let (tx, ty) = (t % lists.grid_w, t / lists.grid_w);
        let list = &lists.inst_v
            [lists.tile_start[t as usize] as usize..lists.tile_start[t as usize + 1] as usize];
        for ly in 0..lists.tile_h {
            for lx in 0..TILE_W {
                let (gx, gy) = (tx * TILE_W + lx, ty * lists.tile_h + ly);
                if gx >= scene.width || gy >= scene.height {
                    continue;
                }
                let pix = (gy * scene.width + gx) as usize;
                let (pcx32, pcy32) = (gx as f32 + 0.5, gy as f32 + 0.5);
                let mut t32 = 1f32;
                let mut tt = 1f64;
                let mut col = vec![0f64; nch];
                let mut nc = 0u32;
                let mut contribs = Vec::new();
                for (j, &v) in list.iter().enumerate() {
                    let r = &scene.recs[v as usize];
                    let (dx32, dy32) = (r.mx - pcx32, r.my - pcy32);
                    let e32 = (r.ca * dx32)
                        .mul_add(dx32, (r.cb * dx32).mul_add(dy32, (r.cc * dy32) * dy32));
                    let g32 = (e32 as f64).exp2() as f32;
                    let a_raw32 = r.o * g32;
                    let alpha32 = match blend {
                        BlendSemantics::Gsplat => a_raw32.min(ALPHA_MAX),
                        BlendSemantics::FasterGs => a_raw32,
                    };
                    if !(e32 <= 0.0 && alpha32 >= ALPHA_MIN) {
                        continue;
                    }
                    let tn32 = t32 * (1.0 - alpha32);
                    if blend == BlendSemantics::Gsplat && tn32 <= T_MIN {
                        break;
                    }
                    let (dx, dy) = (
                        r.mx as f64 - (gx as f64 + 0.5),
                        r.my as f64 - (gy as f64 + 0.5),
                    );
                    let e = r.ca as f64 * dx * dx + r.cb as f64 * dx * dy + r.cc as f64 * dy * dy;
                    let g = e.exp2();
                    let a_raw = r.o as f64 * g;
                    let alpha = match blend {
                        BlendSemantics::Gsplat => a_raw.min(ALPHA_MAX as f64),
                        BlendSemantics::FasterGs => a_raw,
                    };
                    let wgt = alpha * tt;
                    let c = channel_values(r, extra_ch);
                    for k in 0..nch {
                        col[k] += wgt * c[k];
                    }
                    contribs.push(Contrib {
                        v: v as usize,
                        alpha,
                        a_raw,
                        g,
                        t: tt,
                        dx,
                        dy,
                        through: blend == BlendSemantics::FasterGs || a_raw32 <= ALPHA_MAX,
                    });
                    tt *= 1.0 - alpha;
                    t32 = tn32;
                    nc = j as u32 + 1;
                    if blend == BlendSemantics::FasterGs && t32 < T_MIN {
                        break;
                    }
                }
                for k in 0..3 {
                    out.rgb[k * np + pix] = col[k] + tt * bg[k];
                }
                for k in 0..extra_ch {
                    out.extra[k * np + pix] = col[3 + k];
                }
                out.trans[pix] = tt;
                out.nc[pix] = nc;
                out.pairs[pix] = contribs.iter().map(|c| c.v as u32).collect();
                let tm = &mut out.tile_max_nc[t as usize];
                *tm = (*tm).max(nc);

                // Backward (spec-raster 5.1), front to back.
                let mut gc: Vec<f64> = (0..3).map(|k| pg.d_rgb[k * np + pix] as f64).collect();
                for k in 0..extra_ch {
                    gc.push(pg.d_extra[k * np + pix] as f64);
                }
                let da = pg.d_alpha.as_ref().map_or(0.0, |a| a[pix] as f64);
                let gac = tt * (da - (0..3).map(|k| gc[k] * bg[k]).sum::<f64>());
                let mut acc = col;
                for c in &contribs {
                    let r = &scene.recs[c.v];
                    let cv = channel_values(r, extra_ch);
                    let wgt = c.alpha * c.t;
                    for (k, g) in gc.iter().take(3).enumerate() {
                        if r.flags & (1 << k) == 0 {
                            gr.g[(6 + k) * nv + c.v] += wgt * g;
                        }
                    }
                    for k in 0..extra_ch {
                        gr.extra[k * nv + c.v] += wgt * gc[3 + k];
                    }
                    for k in 0..nch {
                        acc[k] -= cv[k] * wgt;
                    }
                    let ra = 1.0 / (1.0 - c.alpha).max(ONE_MINUS_ALPHA_MIN as f64);
                    let mut d_alpha = gac * ra;
                    for k in 0..nch {
                        d_alpha += gc[k] * (cv[k] * c.t - acc[k] * ra);
                    }
                    if c.through {
                        let [ca, cb, cc] = conic(r);
                        gr.g[5 * nv + c.v] += c.g * d_alpha;
                        let gs = -c.a_raw * d_alpha; // dL/dsigma
                        gr.g[2 * nv + c.v] += 0.5 * gs * c.dx * c.dx;
                        gr.g[3 * nv + c.v] += gs * c.dx * c.dy;
                        gr.g[4 * nv + c.v] += 0.5 * gs * c.dy * c.dy;
                        let gmx = gs * (ca * c.dx + cb * c.dy);
                        let gmy = gs * (cb * c.dx + cc * c.dy);
                        gr.g[c.v] += gmx;
                        gr.g[nv + c.v] += gmy;
                        gr.abs[c.v] += gmx.abs();
                        gr.abs[nv + c.v] += gmy.abs();
                    }
                }
            }
        }
    }
    (out, gr)
}

fn channel_values(r: &SplatRec, extra_ch: usize) -> Vec<f64> {
    let mut c = vec![r.r as f64, r.g as f64, r.b as f64];
    for k in 0..extra_ch {
        c.push(if k == 0 { r.z as f64 } else { 0.0 });
    }
    c
}

/// Everything one raster launch pair reads and writes, uploaded.
pub struct Staged {
    pub f: FrameBuffers,
    pub out: RenderOut,
    pub pg: PixelGrads,
    pub g2d: Grad2d,
    pub n_chains: u32,
}

/// Allocates buffers for `scene` + `lists` under `v` and uploads them, with
/// `flags` (FLAG_*) in the frame parameters.
pub fn stage(
    gpu: &Gpu,
    v: &RasterVariant,
    scene: &Scene2d,
    lists: &TileLists,
    pg: &PixelGradsHost,
    flags: u32,
) -> Result<Staged> {
    let s = &gpu.stream;
    let n = scene.recs.len().max(1) as u32;
    let cap = Capacity {
        gaussians: n.next_multiple_of(1024),
        visible: n,
        instances: lists.instances().max(1),
        tiles: lists.n_tiles(),
        pixels: scene.width * scene.height,
    };
    let mut f = FrameBuffers::alloc(gpu, cap, v)?;
    let need = cap.chains(v) as u64 * v.tile_pixels() as u64 * ckpt_words(v);
    if (f.ckpt.len() as u64) < need {
        f.ckpt = s
            .alloc_zeros::<f32>(need as usize)
            .cuda("alloc checkpoints")?;
    }
    if !scene.recs.is_empty() {
        s.memcpy_htod(&scene.recs, &mut f.rec.slice_mut(0..scene.recs.len()))
            .cuda("upload rec")?;
    }
    if !lists.inst_v.is_empty() {
        s.memcpy_htod(
            &lists.inst_v,
            &mut f.inst_v[0].slice_mut(0..lists.inst_v.len()),
        )
        .cuda("upload inst_v")?;
    }
    f.tile_sorted = 0;
    s.memcpy_htod(&lists.tile_start, &mut f.tile_start)
        .cuda("upload tile_start")?;
    let chain_off = lists.chain_off(v.chain_k as u32);
    let n_chains = *chain_off.last().unwrap_or(&0);
    s.memcpy_htod(&chain_off, &mut f.chain_off)
        .cuda("upload chain_off")?;
    let mut params = IterParams::default();
    params.frame.bg = scene.bg;
    params.frame.flags = flags;
    params.frame.render_w = scene.width;
    params.frame.render_h = scene.height;
    let cam = FrameCam {
        width: scene.width,
        height: scene.height,
        grid_w: lists.grid_w,
        grid_h: lists.grid_h,
        n_tiles: lists.n_tiles(),
        ..Default::default()
    };
    f.iter.upload(gpu, &params, Some(&cam))?;
    let mut status = [0u32; STATUS_LEN];
    status[ST_N_VIS] = scene.recs.len() as u32;
    status[ST_N_INST] = lists.instances();
    status[ST_N_CHAINS] = n_chains;
    s.memcpy_htod(&status, &mut f.iter.status)
        .cuda("upload status")?;
    let out = RenderOut::alloc(gpu, cap, v, true)?;
    let mut grads = PixelGrads::alloc(gpu, cap, v, pg.d_alpha.is_some())?;
    s.memcpy_htod(&pg.d_rgb, &mut grads.d_rgb)
        .cuda("upload d_rgb")?;
    if let (Some(src), Some(dst)) = (&pg.d_alpha, &mut grads.d_alpha) {
        s.memcpy_htod(src, dst).cuda("upload d_alpha")?;
    }
    if let Some(dst) = &mut grads.d_extra {
        s.memcpy_htod(&pg.d_extra, dst).cuda("upload d_extra")?;
    }
    let g2d = Grad2d::alloc(gpu, cap, v, true)?;
    Ok(Staged {
        f,
        out,
        pg: grads,
        g2d,
        n_chains,
    })
}

/// Downloads a device slice (tests).
pub fn host<T: spl_gpu::cudarc::driver::DeviceRepr + Default + Clone>(
    gpu: &Gpu,
    d: &CudaSlice<T>,
) -> Result<Vec<T>> {
    gpu.stream.clone_dtoh(d).cuda("download")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_offsets_follow_buckets_and_k() {
        let lists = TileLists {
            tile_h: 16,
            grid_w: 3,
            grid_h: 1,
            tile_start: vec![0, 0, 33, 33 + 300],
            inst_v: vec![0; 333],
        };
        // Tile sizes 0, 33 (2 buckets), 300 (10 buckets).
        assert_eq!(lists.chain_off(1), [0, 0, 2, 12]);
        assert_eq!(lists.chain_off(4), [0, 0, 1, 4]);
        assert_eq!(lists.chain_off(8), [0, 0, 1, 3]);
    }

    /// The exact lists hold every (tile, Gaussian) with a pixel centre where
    /// alpha ≥ 1/255 (brute force over all pixels), and are no longer than
    /// the rectangle lists.
    #[test]
    fn exact_lists_cover_every_contributing_pixel() {
        for tile_h in [16, 12] {
            let scene = random_scene(
                3,
                &SceneOpts {
                    n: 300,
                    max_sigma: 20.0,
                    ..Default::default()
                },
            );
            let exact = TileLists::exact(&scene, tile_h);
            let rect = TileLists::rect(&scene, tile_h);
            assert!(exact.instances() <= rect.instances());
            for (v, r) in scene.recs.iter().enumerate() {
                for gy in 0..scene.height {
                    for gx in 0..scene.width {
                        let dx = r.mx as f64 - (gx as f64 + 0.5);
                        let dy = r.my as f64 - (gy as f64 + 0.5);
                        let e =
                            r.ca as f64 * dx * dx + r.cb as f64 * dx * dy + r.cc as f64 * dy * dy;
                        if r.o as f64 * e.exp2() < 1.0 / 255.0 {
                            continue;
                        }
                        let t = (gy / tile_h) * exact.grid_w + gx / TILE_W;
                        let l = &exact.inst_v[exact.tile_start[t as usize] as usize
                            ..exact.tile_start[t as usize + 1] as usize];
                        assert!(
                            l.contains(&(v as u32)),
                            "tile {t} misses v {v} at ({gx}, {gy})"
                        );
                    }
                }
            }
        }
    }

    /// The reference conserves colour: with dL/dC = 1 on channel 0 only and no
    /// background, Σ_v dL/dc_v·c_v equals the rendered channel sum.
    #[test]
    fn reference_colour_gradient_reproduces_the_image() {
        let mut scene = random_scene(9, &SceneOpts::default());
        scene.bg = [0.0; 3];
        for r in &mut scene.recs {
            r.flags = 0;
        }
        let lists = TileLists::rect(&scene, 16);
        let np = (scene.width * scene.height) as usize;
        let mut pg = PixelGradsHost::random(1, np, 0, false);
        pg.d_rgb = (0..3 * np)
            .map(|i| if i < np { 1.0 } else { 0.0 })
            .collect();
        let (out, g) = reference(&scene, &lists, BlendSemantics::Gsplat, 0, &pg);
        let nv = scene.recs.len();
        let lhs: f64 = (0..nv)
            .map(|v| g.g[6 * nv + v] * scene.recs[v].r as f64)
            .sum();
        let rhs: f64 = out.rgb[..np].iter().sum();
        assert!(
            (lhs - rhs).abs() < 1e-9 * rhs.abs().max(1.0),
            "{lhs} vs {rhs}"
        );
    }

    /// Loss Σ d_rgb·C + d_alpha·(1 − T) of the blend with the pairs frozen
    /// (no skip or stop decisions), over f64 parameters
    /// [mx, my, A, B, C, o, r, g, b] per v.
    fn frozen_loss(
        scene: &Scene2d,
        params: &[[f64; 9]],
        pairs: &[Vec<u32>],
        pg: &PixelGradsHost,
    ) -> f64 {
        let np = (scene.width * scene.height) as usize;
        let mut l = 0.0;
        for (pix, list) in pairs.iter().enumerate() {
            let (gx, gy) = (pix as u32 % scene.width, pix as u32 / scene.width);
            let (pcx, pcy) = (gx as f64 + 0.5, gy as f64 + 0.5);
            let mut t = 1.0;
            let mut col = [0.0f64; 3];
            for &v in list {
                let p = &params[v as usize];
                let (dx, dy) = (p[0] - pcx, p[1] - pcy);
                let sigma = 0.5 * (p[2] * dx * dx + p[4] * dy * dy) + p[3] * dx * dy;
                let alpha = (p[5] * (-sigma).exp()).min(ALPHA_MAX as f64);
                for (k, c) in col.iter_mut().enumerate() {
                    *c += p[6 + k] * alpha * t;
                }
                t *= 1.0 - alpha;
            }
            for (k, c) in col.iter().enumerate() {
                l += (c + t * scene.bg[k] as f64) * pg.d_rgb[k * np + pix] as f64;
            }
            l += (1.0 - t) * pg.d_alpha.as_ref().unwrap()[pix] as f64;
        }
        l
    }

    /// Central finite differences of the frozen-decision blend agree with the
    /// reference's analytic gradients for all nine 2D quantities.
    #[test]
    fn reference_gradients_match_finite_differences() {
        let o = SceneOpts {
            n: 60,
            width: 40,
            height: 36,
            max_sigma: 8.0,
            max_opacity: 0.95,
            clamp_frac: 0.0,
            margin: 2.0,
            bg: [0.2, 0.1, 0.4],
        };
        let scene = random_scene(21, &o);
        let lists = TileLists::rect(&scene, 16);
        let np = (scene.width * scene.height) as usize;
        let pg = PixelGradsHost::random(5, np, 0, true);
        let (out, g) = reference(&scene, &lists, BlendSemantics::Gsplat, 0, &pg);
        let params: Vec<[f64; 9]> = scene
            .recs
            .iter()
            .map(|r| {
                let c = conic(r);
                [
                    r.mx as f64,
                    r.my as f64,
                    c[0],
                    c[1],
                    c[2],
                    r.o as f64,
                    r.r as f64,
                    r.g as f64,
                    r.b as f64,
                ]
            })
            .collect();
        let nv = scene.recs.len();
        let mut checked = 0;
        let mut worst = 0f64;
        for v in 0..nv {
            for slot in 0..9 {
                // Central differences at h and h/2, Richardson-extrapolated
                // (truncation O(h⁴), round-off ≈ 1e-9 relative).
                let central = |h: f64| {
                    let (mut p, mut m) = (params.clone(), params.clone());
                    p[v][slot] += h;
                    m[v][slot] -= h;
                    (frozen_loss(&scene, &p, &out.pairs, &pg)
                        - frozen_loss(&scene, &m, &out.pairs, &pg))
                        / (2.0 * h)
                };
                let h = 1e-4 * params[v][slot].abs().max(1.0);
                let fd = (4.0 * central(0.5 * h) - central(h)) / 3.0;
                let an = g.g[slot * nv + v];
                let err = (fd - an).abs() / an.abs().max(fd.abs()).max(1e-3);
                worst = worst.max(err);
                assert!(err < 1e-6, "v {v} slot {slot}: fd {fd} vs analytic {an}");
                if an.abs() > 1e-3 {
                    checked += 1;
                }
            }
        }
        assert!(checked > 100, "only {checked} significant gradients");
        eprintln!("{checked} significant gradients, worst relative error {worst:.2e}");
    }
}
