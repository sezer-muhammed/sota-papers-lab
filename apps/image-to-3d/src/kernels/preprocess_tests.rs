//! Tests of the preprocess stage (E1).
//!
//! - CPU: the f32 span mirror against brute-force pixel coverage; the f64
//!   reference below against its own finite differences (sanity of the FD rig).
//! - GPU (`#[ignore]`, run with `scripts/gpu-test.sh <checkout> preprocess`):
//!   forward against an f64 reference written from the spec (independent of
//!   the CUDA code), backward against central finite differences of that
//!   reference (all parameters, pose and focal), named edge cases, compaction
//!   order across many chunks, determinism, the gsplat oracle (staged, see
//!   `benchmarks/image-to-3d/ORACLE.md`), and the garden benchmark.

use std::path::{Path, PathBuf};

use super::span;
use super::{PreprocessKernels, cam_reduce, frame_begin_eager, preprocess_bwd, preprocess_fwd};
use crate::frame::{CameraTable, FrameCam, IterParams, Status};
use crate::gpu::Gpu;
use crate::io::scene::View;
use crate::kernels::densify::DensifyStats;
use crate::kernels::layout::{
    CONIC_SCALE_AC, CONIC_SCALE_B, FLAG_AA, FLAG_AA_GRAD, FLAG_CAM_GRAD, FLAG_DENSIFY_STATS,
    FLAG_FILTER3D, OVF_VISIBLE,
};
use crate::kernels::{
    CameraGrads, Capacity, FrameBuffers, Grad2d, GradSink, RasterVariant, SplatKernels, SplatRec,
};
use crate::model::{Gaussians, GaussiansHost, Grads};
use crate::rng::{Philox, normal2, u01_open};
use crate::testkit::npz::{NpyData, Npz};
use crate::testkit::synthetic::{SyntheticOptions, k_matrix, synthetic_scene, view_matrix};
use crate::testkit::timing::{Stats, bench, gpu_snapshot};
use crate::testkit::tol::{Diff, Tol};

// ================================================================ f64 reference (spec-raster 2)

const ALPHA_MIN64: f64 = 1.0 / 255.0;
const SH_C0_64: f64 = 0.282_094_791_773_878_14;
const SH_C1_64: f64 = 0.488_602_511_902_919_9;
const SH_C2_64: [f64; 5] = [
    1.092_548_430_592_079_2,
    -1.092_548_430_592_079_2,
    0.315_391_565_252_520_05,
    -1.092_548_430_592_079_2,
    0.546_274_215_296_039_6,
];
const SH_C3_64: [f64; 7] = [
    -0.590_043_589_926_643_5,
    2.890_611_442_640_554,
    -0.457_045_799_464_465_8,
    0.373_176_332_590_115_4,
    -0.457_045_799_464_465_8,
    1.445_305_721_320_277,
    -0.590_043_589_926_643_5,
];

type M3 = [[f64; 3]; 3];

fn mul3(a: &M3, b: &M3) -> M3 {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

fn mulv(a: &M3, v: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| a[i][0] * v[0] + a[i][1] * v[1] + a[i][2] * v[2])
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// SE(3) exponential of (ρ, φ), f64.
fn se3_exp64(xi: [f64; 6]) -> (M3, [f64; 3]) {
    let phi = [xi[3], xi[4], xi[5]];
    let th2 = phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2];
    let th = th2.sqrt();
    let (a, b, c) = if th < 1e-4 {
        (1.0 - th2 / 6.0, 0.5 - th2 / 24.0, 1.0 / 6.0 - th2 / 120.0)
    } else {
        (
            th.sin() / th,
            (1.0 - th.cos()) / th2,
            (th - th.sin()) / (th2 * th),
        )
    };
    let k: M3 = [
        [0.0, -phi[2], phi[1]],
        [phi[2], 0.0, -phi[0]],
        [-phi[1], phi[0], 0.0],
    ];
    let k2 = mul3(&k, &k);
    let id = |i: usize, j: usize| if i == j { 1.0 } else { 0.0 };
    let r = std::array::from_fn(|i| std::array::from_fn(|j| id(i, j) + a * k[i][j] + b * k2[i][j]));
    let v: M3 =
        std::array::from_fn(|i| std::array::from_fn(|j| id(i, j) + b * k[i][j] + c * k2[i][j]));
    (r, mulv(&v, [xi[0], xi[1], xi[2]]))
}

/// The composed camera of one frame, f64 (mirror of `frame_begin`).
#[derive(Clone, Copy, Debug)]
struct RefCam {
    r: M3,
    t: [f64; 3],
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    w: u32,
    h: u32,
    /// lim_x0, lim_x1, lim_y0, lim_y1 (held constant under perturbations).
    lim: [f64; 4],
    near: f64,
    far: f64,
}

impl RefCam {
    /// From a view (f64 pose and intrinsics), render size and pose delta.
    fn compose(v: &View, render: (u32, u32), xi: [f64; 6]) -> Self {
        let (rd, td) = se3_exp64(xi);
        let r = mul3(&rd, &v.r_cw);
        let t0 = mulv(&rd, v.t_cw);
        let (sx, sy) = (
            render.0 as f64 / v.width as f64,
            render.1 as f64 / v.height as f64,
        );
        let (fx, fy, cx, cy) = (v.fx * sx, v.fy * sy, v.cx * sx, v.cy * sy);
        let (w, h) = (render.0 as f64, render.1 as f64);
        Self {
            r,
            t: std::array::from_fn(|i| t0[i] + td[i]),
            fx,
            fy,
            cx,
            cy,
            w: render.0,
            h: render.1,
            lim: [
                -(cx + 0.15 * w) / fx,
                (1.15 * w - cx) / fx,
                -(cy + 0.15 * h) / fy,
                (1.15 * h - cy) / fy,
            ],
            near: 0.01,
            far: 1e4,
        }
    }

    /// The GPU's FrameCam, exactly (so forward comparisons isolate preprocess).
    fn from_frame(c: &FrameCam) -> Self {
        let f = |x: f32| x as f64;
        Self {
            r: std::array::from_fn(|i| std::array::from_fn(|j| f(c.r[3 * i + j]))),
            t: c.t.map(f),
            fx: f(c.fx),
            fy: f(c.fy),
            cx: f(c.cx),
            cy: f(c.cy),
            w: c.width,
            h: c.height,
            lim: [f(c.lim_x0), f(c.lim_x1), f(c.lim_y0), f(c.lim_y1)],
            near: f(c.near_z),
            far: f(c.far_z),
        }
    }

    /// Left-perturbed pose exp(ε^)·W and shifted intrinsics; clip limits kept.
    fn perturbed(&self, eps: [f64; 6], d_intr: [f64; 4]) -> Self {
        let (rd, td) = se3_exp64(eps);
        let t0 = mulv(&rd, self.t);
        Self {
            r: mul3(&rd, &self.r),
            t: std::array::from_fn(|i| t0[i] + td[i]),
            fx: self.fx + d_intr[0],
            fy: self.fy + d_intr[1],
            cx: self.cx + d_intr[2],
            cy: self.cy + d_intr[3],
            ..*self
        }
    }

    fn campos(&self) -> [f64; 3] {
        std::array::from_fn(|j| -(0..3).map(|i| self.r[i][j] * self.t[i]).sum::<f64>())
    }
}

/// One Gaussian's raw parameters, f64. Flat order: mean 3, log_scale 3,
/// quat 4, opacity 1, dc 3, rest 45 (k-major, channel-minor) = 59.
#[derive(Clone, Copy, Debug)]
struct GParams {
    mean: [f64; 3],
    log_scale: [f64; 3],
    quat: [f64; 4],
    opacity: f64,
    dc: [f64; 3],
    rest: [f64; 45],
    v3: f64,
}

const N_FLAT: usize = 59;
/// (group name, first flat index, length).
const GROUPS: [(&str, usize, usize); 6] = [
    ("means", 0, 3),
    ("scales", 3, 3),
    ("quats", 6, 4),
    ("opacities", 10, 1),
    ("sh0", 11, 3),
    ("sh_rest", 14, 45),
];

impl GParams {
    fn from_host(g: &GaussiansHost, i: usize, v3: f64) -> Self {
        let f = |x: &f32| *x as f64;
        Self {
            mean: std::array::from_fn(|k| f(&g.means[3 * i + k])),
            log_scale: std::array::from_fn(|k| f(&g.scales[3 * i + k])),
            quat: std::array::from_fn(|k| f(&g.rotations[4 * i + k])),
            opacity: f(&g.opacities[i]),
            dc: std::array::from_fn(|k| f(&g.sh0[3 * i + k])),
            rest: std::array::from_fn(|k| f(&g.sh_rest[45 * i + k])),
            v3,
        }
    }

    fn get(&self, j: usize) -> f64 {
        match j {
            0..3 => self.mean[j],
            3..6 => self.log_scale[j - 3],
            6..10 => self.quat[j - 6],
            10 => self.opacity,
            11..14 => self.dc[j - 11],
            _ => self.rest[j - 14],
        }
    }

    fn set(&mut self, j: usize, x: f64) {
        match j {
            0..3 => self.mean[j] = x,
            3..6 => self.log_scale[j - 3] = x,
            6..10 => self.quat[j - 6] = x,
            10 => self.opacity = x,
            11..14 => self.dc[j - 11] = x,
            _ => self.rest[j - 14] = x,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct RefOpts {
    deg: u32,
    eps2d: f64,
    aa: bool,
    /// Gradient through the AA compensation (gsplat oracle mode); otherwise
    /// finite differences freeze c2 at the unperturbed value.
    aa_grad: bool,
    filter3d: bool,
    box_beta: f64,
}

impl Default for RefOpts {
    fn default() -> Self {
        Self {
            deg: 3,
            eps2d: 0.3,
            aa: false,
            aa_grad: false,
            filter3d: false,
            box_beta: 1.0,
        }
    }
}

/// Everything preprocess emits for one Gaussian, f64.
#[derive(Clone, Copy, Debug)]
struct RefOut {
    mx: f64,
    my: f64,
    /// (A, B, C), scalar-B.
    conic: [f64; 3],
    o: f64,
    oeff: f64,
    c2: f64,
    col: [f64; 3],
    raw: [f64; 3],
    z: f64,
}

impl RefOut {
    /// The quantities g2d holds gradients for, plus depth: mx, my, A, B, C, o, r, g, b, z.
    fn outputs(&self) -> [f64; 10] {
        [
            self.mx,
            self.my,
            self.conic[0],
            self.conic[1],
            self.conic[2],
            self.oeff,
            self.col[0],
            self.col[1],
            self.col[2],
            self.z,
        ]
    }
}

fn sh_basis64(deg: u32, d: [f64; 3]) -> [f64; 15] {
    let [x, y, z] = d;
    let mut b = [0.0; 15];
    if deg >= 1 {
        b[0] = -SH_C1_64 * y;
        b[1] = SH_C1_64 * z;
        b[2] = -SH_C1_64 * x;
    }
    if deg >= 2 {
        let (xx, yy, zz) = (x * x, y * y, z * z);
        let c = SH_C2_64;
        b[3] = c[0] * x * y;
        b[4] = c[1] * y * z;
        b[5] = c[2] * (2.0 * zz - xx - yy);
        b[6] = c[3] * x * z;
        b[7] = c[4] * (xx - yy);
        if deg >= 3 {
            let c = SH_C3_64;
            b[8] = c[0] * y * (3.0 * xx - yy);
            b[9] = c[1] * x * y * z;
            b[10] = c[2] * y * (4.0 * zz - xx - yy);
            b[11] = c[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy);
            b[12] = c[4] * x * (4.0 * zz - xx - yy);
            b[13] = c[5] * z * (xx - yy);
            b[14] = c[6] * x * (xx - 3.0 * yy);
        }
    }
    b
}

/// Preprocess of one Gaussian in f64. `cull` applies the forward's culls
/// (None when culled before the tile span); without it the formulas are
/// evaluated regardless (finite differences). `c2_frozen` replaces the AA
/// compensation (finite differences of the detached-compensation gradient).
fn ref_eval(
    p: &GParams,
    cam: &RefCam,
    o: &RefOpts,
    cull: bool,
    c2_frozen: Option<f64>,
) -> Option<RefOut> {
    let pc = {
        let m = mulv(&cam.r, p.mean);
        [m[0] + cam.t[0], m[1] + cam.t[1], m[2] + cam.t[2]]
    };
    let z = pc[2];
    if cull && !(z >= cam.near && z <= cam.far) {
        return None;
    }
    let op = sigmoid(p.opacity);
    if cull && op < ALPHA_MIN64 {
        return None;
    }
    let n2: f64 = p.quat.iter().map(|q| q * q).sum();
    if cull && n2 < 1e-8 {
        return None;
    }
    let qn = p.quat.map(|q| q / n2.sqrt());
    let [w, x, y, zq] = qn;
    let rq: M3 = [
        [
            1.0 - 2.0 * (y * y + zq * zq),
            2.0 * (x * y - w * zq),
            2.0 * (x * zq + w * y),
        ],
        [
            2.0 * (x * y + w * zq),
            1.0 - 2.0 * (x * x + zq * zq),
            2.0 * (y * zq - w * x),
        ],
        [
            2.0 * (x * zq - w * y),
            2.0 * (y * zq + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ];
    let v3 = if o.filter3d { p.v3 } else { 0.0 };
    let s2 = p.log_scale.map(|s| (2.0 * s).exp());
    let sf2 = s2.map(|s| s + v3);
    let c3 = if v3 > 0.0 {
        ((s2[0] * s2[1] * s2[2]) / (sf2[0] * sf2[1] * sf2[2])).sqrt()
    } else {
        1.0
    };
    let (xr, yr) = (pc[0] / z, pc[1] / z);
    let xc = xr.clamp(cam.lim[0], cam.lim[1]);
    let yc = yr.clamp(cam.lim[2], cam.lim[3]);
    let jm = [
        [cam.fx / z, 0.0, -cam.fx * xc / z],
        [0.0, cam.fy / z, -cam.fy * yc / z],
    ];
    let m: [[f64; 3]; 2] = std::array::from_fn(|r| {
        std::array::from_fn(|j| (0..3).map(|k| jm[r][k] * cam.r[k][j]).sum())
    });
    let t: [[f64; 3]; 2] =
        std::array::from_fn(|r| std::array::from_fn(|k| (0..3).map(|j| m[r][j] * rq[j][k]).sum()));
    let a0: f64 = (0..3).map(|k| t[0][k] * t[0][k] * sf2[k]).sum();
    let b: f64 = (0..3).map(|k| t[0][k] * t[1][k] * sf2[k]).sum();
    let c0: f64 = (0..3).map(|k| t[1][k] * t[1][k] * sf2[k]).sum();
    let (a, c) = (a0 + o.eps2d, c0 + o.eps2d);
    let det = a * c - b * b;
    if cull && det <= 0.0 {
        return None;
    }
    let det0 = a0 * c0 - b * b;
    let c2 = match c2_frozen {
        Some(v) => v,
        None if o.aa => (det0 / det).max(0.005 * 0.005).sqrt(),
        None => 1.0,
    };
    let oeff = op * c2 * c3;
    if cull && oeff < ALPHA_MIN64 {
        return None;
    }
    let campos = cam.campos();
    let dir = [
        p.mean[0] - campos[0],
        p.mean[1] - campos[1],
        p.mean[2] - campos[2],
    ];
    let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
    let basis = sh_basis64(o.deg, dir.map(|v| v / len));
    let raw: [f64; 3] = std::array::from_fn(|ch| {
        0.5 + SH_C0_64 * p.dc[ch] + (0..15).map(|k| basis[k] * p.rest[3 * k + ch]).sum::<f64>()
    });
    Some(RefOut {
        mx: cam.fx * xr + cam.cx,
        my: cam.fy * yr + cam.cy,
        conic: [c / det, -b / det, a / det],
        o: op,
        oeff,
        c2,
        col: raw.map(|v| v.max(0.0)),
        raw,
        z,
    })
}

/// Tiles (tx, ty) with a pixel centre inside {q ≤ k} of the ellipse, brute force.
fn exact_tiles(
    mx: f64,
    my: f64,
    conic: [f64; 3],
    k: f64,
    w: u32,
    h: u32,
    tile_h: u32,
) -> Vec<(u32, u32)> {
    let [a, b, c] = conic;
    let det = a * c - b * b;
    if det <= 0.0 || k <= 0.0 {
        return Vec::new();
    }
    let (xe, ye) = ((k * c / det).sqrt() + 1.0, (k * a / det).sqrt() + 1.0);
    let x0 = ((mx - xe).floor().max(0.0)) as u32;
    let x1 = ((mx + xe).ceil().min(w as f64 - 1.0)).max(-1.0);
    let y0 = ((my - ye).floor().max(0.0)) as u32;
    let y1 = ((my + ye).ceil().min(h as f64 - 1.0)).max(-1.0);
    if x1 < 0.0 || y1 < 0.0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for py in y0..=y1 as u32 {
        for px in x0..=x1 as u32 {
            let (dx, dy) = (px as f64 + 0.5 - mx, py as f64 + 0.5 - my);
            if a * dx * dx + 2.0 * b * dx * dy + c * dy * dy <= k {
                out.push((px / 16, py / tile_h));
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Conics compared relative to their scale max(|A|, |C|) per Gaussian
/// (spec-raster 6.4 "conic rel ≤ 1e-5"; the off-diagonal B alone is often ~0).
const CONIC_SCALED: Tol = Tol {
    max_abs: Some(1e-5),
    ..Tol::NONE
};

/// (ours, reference) conics `[A, B, C]...` divided by the reference's max(|A|, |C|).
fn scaled_conics(ours: &[f32], reference: &[f64]) -> (Vec<f32>, Vec<f64>) {
    let mut a = Vec::with_capacity(ours.len());
    let mut b = Vec::with_capacity(ours.len());
    for (o, r) in ours.chunks(3).zip(reference.chunks(3)) {
        let s = r[0].abs().max(r[2].abs()).max(1e-30);
        a.extend(o.iter().map(|x| (*x as f64 / s) as f32));
        b.extend(r.iter().map(|x| x / s));
    }
    (a, b)
}

/// The record's ellipse in f64 (A, B, C from the stored pre-scaled conic).
fn rec_conic(r: &SplatRec) -> [f64; 3] {
    [
        r.ca as f64 / CONIC_SCALE_AC as f64,
        r.cb as f64 / CONIC_SCALE_B as f64,
        r.cc as f64 / CONIC_SCALE_AC as f64,
    ]
}

// ================================================================ CPU tests

/// The f32 span mirror covers every pixel centre inside the ellipse and adds
/// at most a thin band around it (spec-raster 2.8), for random records
/// including rotated, thin and off-image ones.
#[test]
fn span_mirror_covers_the_ellipse_and_little_more() {
    let rng = Philox::new(11);
    let (w, h) = (157u32, 101u32);
    let (mut exact_total, mut ours_total) = (0usize, 0usize);
    for i in 0..3000u32 {
        let d = rng.draw4(i, 0, 1);
        let e = rng.draw4(i, 1, 1);
        let mx = -20.0 + 200.0 * u01_open(d[0]);
        let my = -20.0 + 140.0 * u01_open(d[1]);
        let s1 = 0.3 + 30.0 * u01_open(d[2]).powi(3);
        let s2 = 0.3 + 30.0 * u01_open(d[3]).powi(3);
        let th = std::f32::consts::PI * u01_open(e[0]);
        let (sn, cs) = th.sin_cos();
        // Σ = R diag(s1², s2²) Rᵀ, conic = Σ⁻¹.
        let (sxx, syy, sxy) = (
            cs * cs * s1 * s1 + sn * sn * s2 * s2,
            sn * sn * s1 * s1 + cs * cs * s2 * s2,
            cs * sn * (s1 * s1 - s2 * s2),
        );
        let det = sxx * syy - sxy * sxy;
        let o = 0.005 + 0.99 * u01_open(e[1]);
        let tile_h = if i % 2 == 0 { 16 } else { 12 };
        let rec = SplatRec {
            mx,
            my,
            ca: CONIC_SCALE_AC * (syy / det),
            cb: CONIC_SCALE_B * (-sxy / det),
            cc: CONIC_SCALE_AC * (sxx / det),
            o,
            pthr: (255.0 * o).log2().max(0.0),
            ..Default::default()
        };
        let ours = span::tiles(&rec, 1.0, w, h, tile_h);
        let k = 2.0 * (255.0 * o as f64).ln();
        let exact = exact_tiles(mx as f64, my as f64, rec_conic(&rec), k, w, h, tile_h);
        for t in &exact {
            assert!(
                ours.binary_search_by(|x| (x.1, x.0).cmp(&(t.1, t.0)))
                    .is_ok(),
                "record {i}: tile {t:?} has a pixel inside but is missing ({rec:?})"
            );
        }
        // Upper bound: every listed tile has a pixel centre inside the ellipse
        // grown by one pixel (continuous band, rounding margin).
        let [a, b, c] = rec_conic(&rec);
        let lmax = 0.5 * (a + c) + (0.25 * (a - c) * (a - c) + b * b).sqrt();
        let grown = exact_tiles(
            mx as f64,
            my as f64,
            [a, b, c],
            (k.sqrt() + 1.5 * lmax.sqrt()).powi(2),
            w,
            h,
            tile_h,
        );
        for t in &ours {
            assert!(
                grown.contains(t),
                "record {i}: tile {t:?} is listed but far from the ellipse"
            );
        }
        exact_total += exact.len();
        ours_total += ours.len();
    }
    println!(
        "span mirror: {ours_total} tiles listed vs {exact_total} with a pixel centre inside ({:.3}x)",
        ours_total as f64 / exact_total as f64
    );
    assert!(ours_total as f64 <= 1.10 * exact_total as f64);
}

/// The f64 reference's finite differences are self-consistent: a linear
/// output (colour in an SH coefficient) differentiates exactly.
#[test]
fn reference_is_linear_in_sh_coefficients() {
    let s = synthetic_scene(&SyntheticOptions {
        n: 4,
        ..Default::default()
    });
    let cam = RefCam::compose(&s.views[0], (64, 48), [0.0; 6]);
    let o = RefOpts::default();
    let p0 = GParams::from_host(&s.gaussians, 0, 0.0);
    let base = ref_eval(&p0, &cam, &o, false, None).unwrap();
    let mut p1 = p0;
    p1.rest[3 * 7 + 1] += 1.0;
    let moved = ref_eval(&p1, &cam, &o, false, None).unwrap();
    let campos = cam.campos();
    let dir: [f64; 3] = std::array::from_fn(|i| p0.mean[i] - campos[i]);
    let len = dir.iter().map(|x| x * x).sum::<f64>().sqrt();
    let y = sh_basis64(3, dir.map(|v| v / len))[7];
    assert!((moved.raw[1] - base.raw[1] - y).abs() < 1e-12);
    // Unit-sphere identities of the Cartesian basis: Σ_m Y_lm² = (2l+1)/(4π).
    let b = sh_basis64(3, [0.6, -0.48, 0.64]);
    let band = |r: std::ops::Range<usize>| b[r].iter().map(|v| v * v).sum::<f64>();
    let four_pi = 4.0 * std::f64::consts::PI;
    assert!((band(0..3) - 3.0 / four_pi).abs() < 1e-12);
    assert!((band(3..8) - 5.0 / four_pi).abs() < 1e-12);
    assert!((band(8..15) - 7.0 / four_pi).abs() < 1e-12);
}

// ================================================================ GPU rig

fn gpu() -> Gpu {
    Gpu::open().expect("open GPU")
}

/// Kernels, store, frame buffers and cameras for one scene.
struct Rig {
    gpu: Gpu,
    k: SplatKernels,
    g: Gaussians,
    f: FrameBuffers,
    g2d: Grad2d,
    cams: CameraTable,
    n: usize,
}

impl Rig {
    fn new(
        gpu: &Gpu,
        host: &GaussiansHost,
        views: &[View],
        v: RasterVariant,
        filter3d: Option<&[f32]>,
        visible_cap: Option<u32>,
    ) -> Self {
        let n = host.len();
        let mut g = Gaussians::alloc(gpu, n, filter3d.is_some()).unwrap();
        g.upload(gpu, host).unwrap();
        if let (Some(src), Some(dst)) = (filter3d, g.filter3d.as_mut()) {
            gpu.stream
                .memcpy_htod(src, &mut dst.slice_mut(..n))
                .unwrap();
        }
        let (w, h) = views
            .iter()
            .fold((1, 1), |(w, h), v| (w.max(v.width), h.max(v.height)));
        let mut cap = Capacity::new(g.cap, 1 << 16, w, h, &v);
        if let Some(vc) = visible_cap {
            cap.visible = vc;
        }
        Self {
            gpu: gpu.clone(),
            k: SplatKernels::load(gpu, v).unwrap(),
            f: FrameBuffers::alloc(gpu, cap, &v).unwrap(),
            g2d: Grad2d::alloc(gpu, cap, &v, true).unwrap(),
            cams: CameraTable::upload(gpu, views).unwrap(),
            g,
            n,
        }
    }

    fn set_pose_delta(&mut self, image: usize, xi: [f64; 6]) {
        let mut all = self.gpu.stream.clone_dtoh(&self.cams.pose_delta).unwrap();
        all[6 * image..6 * image + 6].copy_from_slice(&xi.map(|x| x as f32));
        self.gpu
            .stream
            .memcpy_htod(&all, &mut self.cams.pose_delta)
            .unwrap();
    }

    fn forward(&mut self, params: &IterParams) -> Fwd {
        let s = self.gpu.stream.clone();
        self.f.iter.upload(&self.gpu, params, None).unwrap();
        frame_begin_eager(&self.k, &s, &self.cams, &mut self.f).unwrap();
        preprocess_fwd(&self.k, &s, &mut self.g, &mut self.f, &mut self.g2d).unwrap();
        let status = self.f.iter.read_status(&self.gpu).unwrap();
        let v = status.n_visible.min(self.f.cap.visible) as usize;
        let get = |x: &spl_gpu::cudarc::driver::CudaSlice<u32>, len: usize| -> Vec<u32> {
            if len == 0 {
                Vec::new()
            } else {
                s.clone_dtoh(&x.slice(..len)).unwrap()
            }
        };
        Fwd {
            status,
            cam: s.clone_dtoh(&self.f.iter.cam).unwrap()[0],
            vis_idx: get(&self.f.vis_idx, v),
            rec: if v == 0 {
                Vec::new()
            } else {
                s.clone_dtoh(&self.f.rec.slice(..v)).unwrap()
            },
            n_tiles: get(&self.f.n_tiles, v),
            depth_key: get(&self.f.depth_key[0], v),
            depth_val: get(&self.f.depth_val[0], v),
            vis_mask: s.clone_dtoh(&self.f.vis_mask).unwrap(),
            last_visible: get(&self.g.last_visible, self.n),
        }
    }

    /// Runs preprocess_bwd (Write sink, overwrite mode) on host 2D gradients
    /// `g2d[v]` = (mx, my, A, B, C, o, r, g, b, z) for v < V, then cam_reduce.
    fn backward(&mut self, params: &IterParams, g2d: &[[f32; 10]], stats: bool) -> Bwd {
        let s = self.gpu.stream.clone();
        let vc = self.f.cap.visible as usize;
        let mut rows = vec![0f32; 9 * vc];
        let mut extra = vec![0f32; vc];
        for (v, x) in g2d.iter().enumerate() {
            for j in 0..9 {
                rows[j * vc + v] = x[j];
            }
            extra[v] = x[9];
        }
        s.memcpy_htod(&rows, &mut self.g2d.g).unwrap();
        if let Some(e) = self.g2d.extra.as_mut() {
            s.memcpy_htod(&extra, &mut e.slice_mut(..vc)).unwrap();
        }
        let mut p = *params;
        p.adam.overwrite_mask = 0x3f;
        self.f.iter.upload(&self.gpu, &p, None).unwrap();
        let mut grads = Grads::alloc(&self.gpu, self.g.cap as usize).unwrap();
        let mut cam = CameraGrads::alloc(
            &self.gpu,
            self.cams.n_images,
            self.cams.n_intr,
            PreprocessKernels::cam_partial_blocks(&self.gpu),
        )
        .unwrap();
        let mut st = DensifyStats::alloc(&self.gpu, self.g.cap as usize, true).unwrap();
        preprocess_bwd(
            &self.k,
            &s,
            &mut self.g,
            &self.f,
            &self.g2d,
            GradSink::Write(&mut grads),
            Some(&mut cam),
            if stats { Some(&mut st) } else { None },
        )
        .unwrap();
        cam_reduce(&self.k, &s, &self.f, &mut cam).unwrap();
        let n = self.n;
        let cap = self.g.cap as usize;
        let rd = |x: &spl_gpu::cudarc::driver::CudaSlice<f32>, len: usize| -> Vec<f32> {
            s.clone_dtoh(&x.slice(..len)).unwrap()
        };
        let planar = rd(&grads.sh_rest, 45 * cap);
        let sh_rest = (0..n * 45)
            .map(|i| planar[(i % 45) * cap + i / 45])
            .collect();
        let fc: FrameCam = s.clone_dtoh(&self.f.iter.cam).unwrap()[0];
        let pose = rd(&cam.pose, cam.pose.len());
        let intr = rd(&cam.intr, cam.intr.len());
        let img = fc.image_index as usize;
        let iid = fc.intr_id as usize;
        Bwd {
            means: rd(&grads.mean, 3 * n),
            scales: rd(&grads.log_scale, 3 * n),
            quats: rd(&grads.quat, 4 * n),
            opacities: rd(&grads.opacity, n),
            sh0: rd(&grads.sh_dc, 3 * n),
            sh_rest,
            pose: std::array::from_fn(|i| pose[6 * img + i]),
            intr: std::array::from_fn(|i| intr[4 * iid + i]),
            vis_count: rd(&st.vis_count, n),
            grad_sum: rd(&st.grad_sum, n),
        }
    }
}

struct Fwd {
    status: Status,
    cam: FrameCam,
    vis_idx: Vec<u32>,
    rec: Vec<SplatRec>,
    n_tiles: Vec<u32>,
    depth_key: Vec<u32>,
    depth_val: Vec<u32>,
    vis_mask: Vec<u32>,
    last_visible: Vec<u32>,
}

struct Bwd {
    means: Vec<f32>,
    scales: Vec<f32>,
    quats: Vec<f32>,
    opacities: Vec<f32>,
    sh0: Vec<f32>,
    sh_rest: Vec<f32>,
    pose: [f32; 6],
    intr: [f32; 4],
    vis_count: Vec<f32>,
    grad_sum: Vec<f32>,
}

impl Bwd {
    /// Gradient of flat parameter j (GParams order) of Gaussian p.
    fn flat(&self, p: usize, j: usize) -> f32 {
        match j {
            0..3 => self.means[3 * p + j],
            3..6 => self.scales[3 * p + j - 3],
            6..10 => self.quats[4 * p + j - 6],
            10 => self.opacities[p],
            11..14 => self.sh0[3 * p + j - 11],
            _ => self.sh_rest[45 * p + j - 14],
        }
    }
}

fn iter_params(n: usize, cam_index: u32, deg: u32, flags: u32, eps2d: f32) -> IterParams {
    let mut p = IterParams::default();
    p.frame.iteration = 7;
    p.frame.cam_index = cam_index;
    p.frame.n_gaussians = n as u32;
    p.frame.sh_degree = deg;
    p.frame.eps2d = eps2d;
    p.frame.flags = flags;
    p.frame.box_beta = 1.0;
    p
}

fn ref_opts(params: &IterParams) -> RefOpts {
    let fl = params.frame.flags;
    RefOpts {
        deg: params.frame.sh_degree,
        eps2d: params.frame.eps2d as f64,
        aa: fl & FLAG_AA != 0,
        aa_grad: fl & FLAG_AA != 0 && fl & FLAG_AA_GRAD != 0,
        filter3d: fl & FLAG_FILTER3D != 0,
        box_beta: params.frame.box_beta as f64,
    }
}

/// Visible means at least one pixel centre with α ≥ 1/255 under box_beta = 1;
/// a tiny relative margin separates "clearly visible" from threshold cases.
fn ref_coverage(r: &RefOut, cam: &RefCam, o: &RefOpts, tile_h: u32) -> (usize, usize) {
    let k = o.box_beta * 2.0 * (255.0 * r.oeff).ln();
    let inner = exact_tiles(
        r.mx,
        r.my,
        r.conic,
        k * (1.0 - 1e-5) - 1e-5,
        cam.w,
        cam.h,
        tile_h,
    );
    let outer = exact_tiles(
        r.mx,
        r.my,
        r.conic,
        k * (1.0 + 1e-5) + 1e-5,
        cam.w,
        cam.h,
        tile_h,
    );
    (inner.len(), outer.len())
}

/// Checks one forward against the f64 reference; returns the reference outputs
/// of the GPU-visible Gaussians in v order.
fn check_forward(
    name: &str,
    host: &GaussiansHost,
    v3: Option<&[f32]>,
    params: &IterParams,
    fwd: &Fwd,
    tile_h: u32,
) -> Vec<RefOut> {
    let n = host.len();
    let cam = RefCam::from_frame(&fwd.cam);
    let o = ref_opts(params);
    let v = fwd.vis_idx.len();
    assert_eq!(fwd.status.overflow, 0, "{name}: overflow");
    assert_eq!(fwd.status.n_visible as usize, v, "{name}: N_VIS");
    // Order-preserving compaction, keys, mask and stamps.
    assert!(
        fwd.vis_idx.windows(2).all(|w| w[0] < w[1]),
        "{name}: vis_idx not increasing"
    );
    let mut vis = vec![false; n];
    for (i, &p) in fwd.vis_idx.iter().enumerate() {
        vis[p as usize] = true;
        assert_eq!(fwd.depth_val[i], i as u32, "{name}: depth_val");
        assert_eq!(
            fwd.depth_key[i],
            fwd.rec[i].z.to_bits(),
            "{name}: depth_key"
        );
        assert!(fwd.n_tiles[i] > 0, "{name}: visible with 0 tiles");
    }
    for (p, &vp) in vis.iter().enumerate() {
        let bit = fwd.vis_mask[p / 32] >> (p % 32) & 1 == 1;
        assert_eq!(bit, vp, "{name}: vis_mask bit {p}");
        assert_eq!(
            fwd.last_visible[p] == params.frame.iteration,
            vp,
            "{name}: last_visible {p}"
        );
    }
    // Visible set vs the reference.
    let (mut missing, mut extra, mut borderline) = (0, 0, 0);
    let mut refs = Vec::with_capacity(v);
    let mut by_p = vec![None; n];
    for (p, slot) in by_p.iter_mut().enumerate() {
        let gp = GParams::from_host(host, p, v3.map_or(0.0, |x| x[p] as f64));
        *slot = ref_eval(&gp, &cam, &o, true, None);
        let culled_near_threshold = || {
            let r = ref_eval(&gp, &cam, &o, false, None).unwrap();
            (r.oeff - ALPHA_MIN64).abs() < 1e-5 || (r.o - ALPHA_MIN64).abs() < 1e-5
        };
        match (*slot, vis[p]) {
            (Some(r), false) => {
                let (inner, _) = ref_coverage(&r, &cam, &o, tile_h);
                if inner > 0 {
                    if culled_near_threshold() {
                        borderline += 1;
                    } else {
                        missing += 1;
                        eprintln!("{name}: Gaussian {p} contributes but is culled: {r:?}");
                    }
                }
            }
            (None, true) => {
                if culled_near_threshold() {
                    borderline += 1;
                } else {
                    extra += 1;
                    eprintln!("{name}: Gaussian {p} is visible but the reference culls it");
                }
            }
            (Some(r), true) => {
                let (_, outer) = ref_coverage(&r, &cam, &o, tile_h);
                if outer == 0 {
                    // Listed although no pixel centre reaches α ≥ 1/255: allowed
                    // (the band is continuous in y); counted for the report.
                    borderline += 1;
                }
            }
            (None, false) => {}
        }
    }
    assert_eq!((missing, extra), (0, 0), "{name}: visible set differs");
    // Values of the visible Gaussians.
    let (mut m2, mut z, mut conic, mut oeff, mut col, mut pthr) = (
        (Vec::new(), Vec::new()),
        (Vec::new(), Vec::new()),
        (Vec::new(), Vec::new()),
        (Vec::new(), Vec::new()),
        (Vec::new(), Vec::new()),
        0f64,
    );
    let (mut tiles_ours, mut tiles_exact) = (0usize, 0usize);
    for (i, &p) in fwd.vis_idx.iter().enumerate() {
        let r = fwd.rec[i];
        let Some(rf) = by_p[p as usize] else {
            continue;
        };
        refs.push(rf);
        // Mean error in units of max(1e-4 px, 1e-6 |m|) (spec-raster 6.4).
        for (a, b) in [(r.mx, rf.mx), (r.my, rf.my)] {
            let unit = (1e-6 * b.abs()).max(1e-4);
            m2.0.push((a as f64 / unit) as f32);
            m2.1.push(b / unit);
        }
        z.0.push(r.z);
        z.1.push(rf.z);
        // Conic error relative to the conic's scale (B alone is often ~0).
        let sc = rf.conic[0].abs().max(rf.conic[2].abs());
        conic.0.extend(rec_conic(&r).map(|x| (x / sc) as f32));
        conic.1.extend(rf.conic.map(|x| x / sc));
        oeff.0.push(r.o);
        oeff.1.push(rf.oeff);
        col.0.extend([r.r, r.g, r.b]);
        col.1.extend(rf.col);
        pthr = pthr.max((r.pthr as f64 - (255.0 * rf.oeff).log2().max(0.0)).abs());
        for c in 0..3 {
            assert_eq!(
                r.flags >> c & 1 == 1,
                rf.raw[c] <= 0.0,
                "{name}: clamp flag {c} of {p} (raw {})",
                rf.raw[c]
            );
        }
        // Tile count: bit-exact vs the CPU span mirror, and every tile with a
        // contributing pixel centre (f64 truth) is among the spanned tiles.
        let spanned = span::tiles(&r, params.frame.box_beta, cam.w, cam.h, tile_h);
        assert_eq!(
            spanned.len() as u32,
            fwd.n_tiles[i],
            "{name}: n_tiles of {p} vs the span mirror"
        );
        let k = o.box_beta * 2.0 * (255.0 * rf.oeff).ln();
        let truth = exact_tiles(
            rf.mx,
            rf.my,
            rf.conic,
            k * (1.0 - 1e-5),
            cam.w,
            cam.h,
            tile_h,
        );
        for t in &truth {
            assert!(
                spanned.contains(t),
                "{name}: Gaussian {p} contributes to tile {t:?} which is not spanned"
            );
        }
        tiles_ours += spanned.len();
        tiles_exact += truth.len();
    }
    let report = |d: Diff| println!("  {d}");
    let mean_tol = Tol {
        max_abs: Some(1.0),
        ..Tol::NONE
    };
    for (label, (a, b), tol) in [
        ("means2d/unit", m2, mean_tol),
        (
            "depth",
            z,
            Tol {
                elem_rel: Some((1e-6, 0.0)),
                ..Tol::NONE
            },
        ),
        ("conic/scale", conic, CONIC_SCALED),
        ("opacity_eff", oeff, Tol::CONIC),
        (
            "colour",
            col,
            Tol {
                max_abs: Some(1e-5),
                ..Tol::NONE
            },
        ),
    ] {
        let d = Diff::new(&format!("{name} {label}"), &a, &b, 0.0);
        d.check(&tol).unwrap();
        report(d);
    }
    assert!(pthr < 1e-5, "{name}: pthr error {pthr}");
    println!(
        "  {name}: {v}/{n} visible ({borderline} threshold cases), {tiles_ours} tiles spanned, {tiles_exact} with a contributing pixel ({:.3}x)",
        tiles_ours as f64 / tiles_exact.max(1) as f64
    );
    refs
}

/// Random 2D gradients for v < V (Philox, unit normal; colour rows scaled down).
fn random_g2d(v: usize, seed: u64, depth: bool) -> Vec<[f32; 10]> {
    let rng = Philox::new(seed);
    (0..v as u32)
        .map(|i| {
            let a = rng.draw4(i, 0, 9);
            let b = rng.draw4(i, 1, 9);
            let c = rng.draw4(i, 2, 9);
            let (g0, g1) = normal2(a[0], a[1]);
            let (g2, g3) = normal2(a[2], a[3]);
            let (g4, g5) = normal2(b[0], b[1]);
            let (g6, g7) = normal2(b[2], b[3]);
            let (g8, g9) = normal2(c[0], c[1]);
            [
                g0,
                g1,
                g2,
                g3,
                g4,
                g5,
                g6,
                g7,
                g8,
                if depth { g9 } else { 0.0 },
            ]
        })
        .collect()
}

/// Central differences of Σ_v w_v · outputs_v w.r.t. every parameter of every
/// visible Gaussian, and w.r.t. the camera (left-perturbed pose, normalized
/// intrinsics; clip limits held constant).
struct FdGrads {
    /// [v][59]
    params: Vec<[f64; N_FLAT]>,
    pose: [f64; 6],
    intr: [f64; 4],
}

fn finite_differences(
    host: &GaussiansHost,
    v3: Option<&[f32]>,
    vis_idx: &[u32],
    cam: &RefCam,
    o: &RefOpts,
    w: &[[f32; 10]],
) -> FdGrads {
    let gp: Vec<GParams> = vis_idx
        .iter()
        .map(|&p| GParams::from_host(host, p as usize, v3.map_or(0.0, |x| x[p as usize] as f64)))
        .collect();
    // Detached AA compensation: freeze c2 at the unperturbed value.
    let frozen: Vec<Option<f64>> = gp
        .iter()
        .map(|p| (o.aa && !o.aa_grad).then(|| ref_eval(p, cam, o, false, None).unwrap().c2))
        .collect();
    let loss = |p: &GParams, c: &RefCam, f: Option<f64>, wv: &[f32; 10]| -> f64 {
        let out = ref_eval(p, c, o, false, f).unwrap().outputs();
        (0..10).map(|j| wv[j] as f64 * out[j]).sum()
    };
    let h = 1e-6;
    let params = gp
        .iter()
        .enumerate()
        .map(|(i, p)| {
            std::array::from_fn(|j| {
                let x = p.get(j);
                let step = h * x.abs().max(1.0);
                let (mut a, mut b) = (*p, *p);
                a.set(j, x + step);
                b.set(j, x - step);
                (loss(&a, cam, frozen[i], &w[i]) - loss(&b, cam, frozen[i], &w[i])) / (2.0 * step)
            })
        })
        .collect();
    let total = |c: &RefCam| -> f64 {
        gp.iter()
            .enumerate()
            .map(|(i, p)| loss(p, c, frozen[i], &w[i]))
            .sum()
    };
    let pose = std::array::from_fn(|j| {
        let mut e = [0.0; 6];
        e[j] = h;
        let a = total(&cam.perturbed(e, [0.0; 4]));
        e[j] = -h;
        let b = total(&cam.perturbed(e, [0.0; 4]));
        (a - b) / (2.0 * h)
    });
    let size = [cam.w as f64, cam.h as f64, cam.w as f64, cam.h as f64];
    let intr = std::array::from_fn(|j| {
        let mut d = [0.0; 4];
        d[j] = h * size[j];
        let a = total(&cam.perturbed([0.0; 6], d));
        d[j] = -h * size[j];
        let b = total(&cam.perturbed([0.0; 6], d));
        (a - b) / (2.0 * h)
    });
    FdGrads { params, pose, intr }
}

/// One configuration of the forward/backward reference tests.
struct Case {
    name: &'static str,
    opts: SyntheticOptions,
    deg: u32,
    flags: u32,
    eps2d: f32,
    filter3d: bool,
    render: Option<(u32, u32)>,
    xi: [f64; 6],
    variant: RasterVariant,
}

fn cases() -> Vec<Case> {
    let base = SyntheticOptions {
        n: 400,
        width: 96,
        height: 72,
        ..Default::default()
    };
    let v = RasterVariant::default();
    let depth = RasterVariant { extra_ch: 1, ..v };
    let c = |name, deg, flags, filter3d| Case {
        name,
        opts: base.clone(),
        deg,
        flags,
        eps2d: 0.3,
        filter3d,
        render: None,
        xi: [0.0; 6],
        variant: v,
    };
    vec![
        c("deg3", 3, 0, false),
        c("deg0", 0, 0, false),
        c("deg1", 1, 0, false),
        c("deg2", 2, 0, false),
        c("aa_detached", 3, FLAG_AA, false),
        c("aa_grad", 3, FLAG_AA | FLAG_AA_GRAD, false),
        c("filter3d", 3, FLAG_FILTER3D | FLAG_AA, true),
        Case {
            eps2d: 0.1,
            ..c("eps0.1", 2, FLAG_AA | FLAG_AA_GRAD, false)
        },
        Case {
            variant: depth,
            ..c("depth_channel", 3, 0, false)
        },
        Case {
            render: Some((61, 37)),
            xi: [0.05, -0.03, 0.08, 0.02, -0.04, 0.03],
            ..c("pose_delta_scaled", 3, FLAG_AA, false)
        },
        Case {
            variant: RasterVariant { tile_h: 12, ..v },
            opts: SyntheticOptions {
                scale_range: (0.05, 0.6),
                ..base.clone()
            },
            ..c("tile_h12_large", 3, 0, false)
        },
    ]
}

fn filter_values(n: usize, seed: u64) -> Vec<f32> {
    let rng = Philox::new(seed);
    (0..n as u32)
        .map(|i| 1e-4 + 2e-3 * rng.uniform(i, 0, 3))
        .collect()
}

#[test]
#[ignore]
fn gpu_preprocess_matches_f64_reference_and_finite_differences() {
    let gpu = gpu();
    let mut worst: Vec<String> = Vec::new();
    for case in cases() {
        let scene = synthetic_scene(&case.opts);
        let host = &scene.gaussians;
        let n = host.len();
        let v3 = case.filter3d.then(|| filter_values(n, 5));
        let mut rig = Rig::new(&gpu, host, &scene.views, case.variant, v3.as_deref(), None);
        let view = 1usize;
        rig.set_pose_delta(view, case.xi);
        let mut params = iter_params(
            n,
            view as u32,
            case.deg,
            case.flags | FLAG_CAM_GRAD,
            case.eps2d,
        );
        if let Some((w, h)) = case.render {
            params.frame.render_w = w;
            params.frame.render_h = h;
        }
        println!("case {}:", case.name);
        let fwd = rig.forward(&params);
        // frame_begin: the composed camera vs f64.
        let want = RefCam::compose(
            &scene.views[view],
            case.render
                .unwrap_or((scene.views[view].width, scene.views[view].height)),
            case.xi,
        );
        let got = RefCam::from_frame(&fwd.cam);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (got.r[i][j] - want.r[i][j]).abs() < 1e-6,
                    "{}: R",
                    case.name
                );
            }
            assert!((got.t[i] - want.t[i]).abs() < 1e-5, "{}: t", case.name);
        }
        for (a, b) in [
            (got.fx, want.fx),
            (got.fy, want.fy),
            (got.cx, want.cx),
            (got.cy, want.cy),
        ]
        .into_iter()
        .chain(got.lim.into_iter().zip(want.lim))
        {
            assert!(
                (a - b).abs() <= 1e-6 * b.abs().max(1.0),
                "{}: intrinsics",
                case.name
            );
        }
        let refs = check_forward(
            case.name,
            host,
            v3.as_deref(),
            &params,
            &fwd,
            case.variant.tile_h as u32,
        );
        assert_eq!(
            refs.len(),
            fwd.vis_idx.len(),
            "{}: reference culls a visible Gaussian",
            case.name
        );

        // Backward vs central differences of the f64 reference.
        let w = random_g2d(fwd.vis_idx.len(), 17, case.variant.extra_ch > 0);
        let bwd = rig.backward(&params, &w, false);
        let o = ref_opts(&params);
        let fd = finite_differences(host, v3.as_deref(), &fwd.vis_idx, &got, &o, &w);
        for (gname, first, len) in GROUPS {
            let (mut ours, mut want) = (Vec::new(), Vec::new());
            for (i, &p) in fwd.vis_idx.iter().enumerate() {
                for j in first..first + len {
                    ours.push(bwd.flat(p as usize, j));
                    want.push(fd.params[i][j]);
                }
            }
            let d = Diff::new(&format!("{} d{gname}", case.name), &ours, &want, 1e-3);
            println!("  {d}");
            d.check(&Tol {
                rel_l2: Some(1e-4),
                min_cosine: Some(0.99999),
                ..Tol::NONE
            })
            .unwrap();
            worst.push(format!(
                "{:<18} {gname:<9} rel_l2 {:.2e} cos {:.9}",
                case.name, d.rel_l2, d.cosine
            ));
        }
        // Invisible Gaussians keep a zero gradient (overwrite mode only writes visible ones).
        let mut vis = vec![false; n];
        fwd.vis_idx.iter().for_each(|&p| vis[p as usize] = true);
        for p in (0..n).filter(|&p| !vis[p]) {
            assert!(
                (0..N_FLAT).all(|j| bwd.flat(p, j) == 0.0),
                "{}: invisible {p} has a gradient",
                case.name
            );
        }
        // Pose (ρ, φ) and normalized intrinsics vs finite differences.
        for (label, ours, want) in [
            ("pose", bwd.pose.to_vec(), fd.pose.to_vec()),
            ("intr", bwd.intr.to_vec(), fd.intr.to_vec()),
        ] {
            let norm = want.iter().map(|x| x * x).sum::<f64>().sqrt();
            let err = ours
                .iter()
                .zip(&want)
                .map(|(a, b)| (*a as f64 - b).abs())
                .fold(0.0, f64::max);
            println!(
                "  {} d{label}: max err {err:.3e} of |g| {norm:.3e}\n    ours {ours:?}\n    fd   {want:?}",
                case.name
            );
            assert!(
                err <= 1e-4 * norm,
                "{}: {label} gradient off by {err:e} (|g| {norm:e})",
                case.name
            );
            worst.push(format!(
                "{:<18} {label:<9} rel {:.2e}",
                case.name,
                err / norm
            ));
        }
    }
    println!("summary:\n  {}", worst.join("\n  "));
}

/// Two runs of the same frame give bitwise identical forward outputs and
/// backward gradients (no atomics in either; camera partials in fixed order).
#[test]
#[ignore]
fn gpu_preprocess_is_deterministic() {
    let gpu = gpu();
    let scene = synthetic_scene(&SyntheticOptions {
        n: 50_000,
        width: 320,
        height: 240,
        half_extent: 2.0,
        ..Default::default()
    });
    let n = scene.gaussians.len();
    let mut rig = Rig::new(
        &gpu,
        &scene.gaussians,
        &scene.views,
        RasterVariant::default(),
        None,
        None,
    );
    let params = iter_params(n, 2, 3, FLAG_AA | FLAG_CAM_GRAD | FLAG_DENSIFY_STATS, 0.3);
    let a = rig.forward(&params);
    let b = rig.forward(&params);
    assert_eq!(a.vis_idx, b.vis_idx);
    assert_eq!(a.n_tiles, b.n_tiles);
    assert!(
        a.rec
            .iter()
            .zip(&b.rec)
            .all(|(x, y)| format!("{x:?}") == format!("{y:?}"))
    );
    let w = random_g2d(a.vis_idx.len(), 3, false);
    let ga = rig.backward(&params, &w, true);
    let gb = rig.backward(&params, &w, true);
    assert_eq!(ga.means, gb.means);
    assert_eq!(ga.sh_rest, gb.sh_rest);
    assert_eq!(ga.pose, gb.pose);
    assert_eq!(ga.intr, gb.intr);
    // Densify statistics: one count per visible Gaussian, NDC-scaled norms.
    let mut vis = vec![false; n];
    a.vis_idx.iter().for_each(|&p| vis[p as usize] = true);
    for (i, &p) in a.vis_idx.iter().enumerate() {
        let p = p as usize;
        assert_eq!(ga.vis_count[p], 1.0);
        let (hw, hh) = (0.5 * a.cam.width as f32, 0.5 * a.cam.height as f32);
        let want = ((w[i][0] * hw).powi(2) + (w[i][1] * hh).powi(2)).sqrt();
        assert!((ga.grad_sum[p] - want).abs() <= 1e-5 * want.max(1.0));
    }
    assert!((0..n).filter(|&p| !vis[p]).all(|p| ga.vis_count[p] == 0.0));
    println!(
        "{} visible of {n}: forward and backward bitwise identical",
        a.vis_idx.len()
    );
}

/// Compaction across many chunks (decoupled look-back), against the
/// reference's visible list; then V overflow and zero visible.
#[test]
#[ignore]
fn gpu_preprocess_compacts_in_order_and_flags_overflow() {
    let gpu = gpu();
    let scene = synthetic_scene(&SyntheticOptions {
        n: 300_001,
        width: 400,
        height: 300,
        half_extent: 3.0,
        ..Default::default()
    });
    let host = &scene.gaussians;
    let n = host.len();
    let mut rig = Rig::new(
        &gpu,
        host,
        &scene.views,
        RasterVariant::default(),
        None,
        None,
    );
    let params = iter_params(n, 0, 3, 0, 0.3);
    let fwd = rig.forward(&params);
    let refs = check_forward("300k", host, None, &params, &fwd, 16);
    println!("300k: {} visible, {} chunks", refs.len(), n.div_ceil(256));

    // V overflow: capacity below the visible count sets OVF_VISIBLE and writes nothing past it.
    let cap = (fwd.vis_idx.len() / 2) as u32;
    let mut small = Rig::new(
        &gpu,
        host,
        &scene.views,
        RasterVariant::default(),
        None,
        Some(cap),
    );
    let o = small.forward(&params);
    assert_eq!(o.status.overflow & OVF_VISIBLE, OVF_VISIBLE);
    assert_eq!(o.status.n_visible as usize, fwd.vis_idx.len());
    assert_eq!(&o.vis_idx[..], &fwd.vis_idx[..cap as usize]);
    println!("overflow: V {} > V_cap {cap} flagged", o.status.n_visible);

    // Zero visible: every Gaussian behind the camera.
    let mut behind = host.clone();
    let v = &scene.views[0];
    let c = v.center();
    let fwd_dir = v.r_cw[2];
    for p in 0..n {
        for k in 0..3 {
            behind.means[3 * p + k] = (c[k] - 2.0 * fwd_dir[k]) as f32;
        }
    }
    let mut rig = Rig::new(
        &gpu,
        &behind,
        &scene.views,
        RasterVariant::default(),
        None,
        None,
    );
    let z = rig.forward(&params);
    assert_eq!((z.status.n_visible, z.status.overflow), (0, 0));
    assert!(z.vis_mask.iter().all(|&w| w == 0));
}

/// Named edge cases of spec-raster 6.5 that preprocess owns.
#[test]
#[ignore]
fn gpu_preprocess_edge_cases() {
    let gpu = gpu();
    let (w, h) = (1297u32, 840u32);
    let view = View {
        name: "edge".into(),
        camera: 1,
        r_cw: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        t_cw: [0.0; 3],
        fx: 1000.0,
        fy: 1000.0,
        cx: 648.5,
        cy: 420.0,
        width: w,
        height: h,
        image_path: Default::default(),
    };
    let lim_x1 = (1.15 * w as f64 - 648.5) / 1000.0;
    // (name, mean, log scale, quat, opacity logit, expect visible?)
    type Edge = (&'static str, [f32; 3], f32, [f32; 4], f32, Option<bool>);
    let edges: Vec<Edge> = vec![
        (
            "behind",
            [0.0, 0.0, -1.0],
            -3.0,
            [1.0, 0.0, 0.0, 0.0],
            2.0,
            Some(false),
        ),
        (
            "near_plane",
            [0.0, 0.0, 0.011],
            -6.0,
            [1.0, 0.0, 0.0, 0.0],
            2.0,
            None,
        ),
        (
            "whole_image",
            [0.0, 0.0, 2.0],
            1.6,
            [0.9, 0.1, 0.2, 0.3],
            3.0,
            Some(true),
        ),
        (
            "clip_inside",
            [(lim_x1 - 1e-3) as f32 * 5.0, 0.0, 5.0],
            -1.0,
            [1.0, 0.0, 0.0, 0.0],
            2.0,
            None,
        ),
        (
            "clip_outside",
            [(lim_x1 + 1e-3) as f32 * 5.0, 0.0, 5.0],
            -1.0,
            [1.0, 0.0, 0.0, 0.0],
            2.0,
            None,
        ),
        (
            "degenerate_quat",
            [0.0, 0.0, 3.0],
            -3.0,
            [0.0, 0.0, 0.0, 0.0],
            2.0,
            Some(false),
        ),
        (
            "tiny_scale",
            // On a pixel centre: the dilated footprint reaches no other one.
            [0.099, 0.1005, 3.0],
            -15.0,
            [1.0, 0.0, 0.0, 0.0],
            2.0,
            Some(true),
        ),
        (
            "huge_scale",
            [0.0, 0.0, 3.0],
            15.0,
            [1.0, 0.0, 0.0, 0.0],
            2.0,
            None,
        ),
        (
            "opacity_at_threshold",
            [0.2, 0.0, 3.0],
            -3.0,
            [1.0, 0.0, 0.0, 0.0],
            -5.537_334,
            None,
        ),
        (
            "transparent",
            [0.2, 0.0, 3.0],
            -3.0,
            [1.0, 0.0, 0.0, 0.0],
            -9.0,
            Some(false),
        ),
        (
            "off_screen",
            [10.0, 0.0, 3.0],
            -3.0,
            [1.0, 0.0, 0.0, 0.0],
            2.0,
            Some(false),
        ),
        (
            "far",
            [0.0, 0.0, 2e4],
            2.0,
            [1.0, 0.0, 0.0, 0.0],
            2.0,
            Some(false),
        ),
    ];
    let n_edges = edges.len();
    let mut host = GaussiansHost::zeros(n_edges + 2000);
    for (i, (_, m, s, q, o, _)) in edges.iter().enumerate() {
        host.means[3 * i..3 * i + 3].copy_from_slice(m);
        host.scales[3 * i..3 * i + 3].copy_from_slice(&[*s; 3]);
        host.rotations[4 * i..4 * i + 4].copy_from_slice(q);
        host.opacities[i] = *o;
    }
    // Filler: random Gaussians in front of the camera.
    let rng = Philox::new(3);
    for i in n_edges..host.len() {
        let d = rng.draw4(i as u32, 0, 2);
        host.means[3 * i] = 4.0 * (u01_open(d[0]) - 0.5);
        host.means[3 * i + 1] = 3.0 * (u01_open(d[1]) - 0.5);
        host.means[3 * i + 2] = 2.0 + 4.0 * u01_open(d[2]);
        host.scales[3 * i..3 * i + 3].copy_from_slice(&[-4.0 + u01_open(d[3]); 3]);
        host.opacities[i] = 1.0;
        host.sh0[3 * i] = 0.5;
    }
    for variant in [
        RasterVariant::default(),
        RasterVariant {
            tile_h: 12,
            ..Default::default()
        },
    ] {
        let mut rig = Rig::new(
            &gpu,
            &host,
            std::slice::from_ref(&view),
            variant,
            None,
            None,
        );
        let params = iter_params(host.len(), 0, 3, FLAG_AA, 0.3);
        let fwd = rig.forward(&params);
        let th = variant.tile_h as u32;
        check_forward(&format!("edges/tile_h{th}"), &host, None, &params, &fwd, th);
        for r in &fwd.rec {
            let words = [
                r.mx, r.my, r.ca, r.cb, r.cc, r.o, r.r, r.g, r.b, r.pthr, r.z,
            ];
            assert!(
                words.iter().all(|x| x.is_finite()),
                "non-finite record {r:?}"
            );
        }
        for (i, (name, .., expect)) in edges.iter().enumerate() {
            let at = fwd.vis_idx.iter().position(|&p| p as usize == i);
            if let Some(e) = expect {
                assert_eq!(at.is_some(), *e, "edge case {name}");
            }
            let tiles = at.map_or(0, |v| fwd.n_tiles[v]);
            println!(
                "  tile_h {th} {name:<22} visible {} tiles {tiles}",
                at.is_some()
            );
        }
        // The whole-image splat spans every tile (warp-cooperative row count).
        let whole = fwd.vis_idx.iter().position(|&p| p == 2).unwrap();
        assert_eq!(fwd.n_tiles[whole], variant.tiles(w, h));
    }
    // Odd image sizes, including 1x1.
    for (ww, hh) in [(17u32, 13u32), (1, 1), (33, 1)] {
        let s = synthetic_scene(&SyntheticOptions {
            n: 300,
            width: ww,
            height: hh,
            ..Default::default()
        });
        let mut rig = Rig::new(
            &gpu,
            &s.gaussians,
            &s.views,
            RasterVariant::default(),
            None,
            None,
        );
        let params = iter_params(300, 0, 3, 0, 0.3);
        let fwd = rig.forward(&params);
        check_forward(&format!("{ww}x{hh}"), &s.gaussians, None, &params, &fwd, 16);
    }
}

// ================================================================ gsplat oracle (staged)

/// gsplat `project` and `sh` on the cases below, staged under
/// `/out/image-to-3d/oracle/pre_<case>.<op>.{in,ours}.npz`. After
/// `gs-oracle batch ~/spl-work/out/image-to-3d/oracle` on the host, a second
/// run compares against `*.out.npz` (ORACLE.md, "preprocess").
#[test]
#[ignore]
fn gpu_preprocess_matches_gsplat_oracle() {
    let gpu = gpu();
    let dir = crate::testkit::oracle::stage_dir();
    let mut compared = 0;
    for (name, deg, aa, n) in [
        ("pre_deg3", 3u32, false, 512usize),
        ("pre_deg3_aa", 3, true, 512),
        ("pre_deg1_aa", 1, true, 512),
        ("pre_deg0", 0, false, 256),
    ] {
        let s = synthetic_scene(&SyntheticOptions {
            n,
            width: 96,
            height: 72,
            sh_degree: deg,
            seed: 40 + n as u64 + deg as u64,
            ..Default::default()
        });
        let host = &s.gaussians;
        let view = &s.views[2];
        let variant = RasterVariant {
            extra_ch: 1,
            ..Default::default()
        };
        let mut rig = Rig::new(&gpu, host, &s.views, variant, None, None);
        let flags = FLAG_CAM_GRAD | if aa { FLAG_AA | FLAG_AA_GRAD } else { 0 };
        let params = iter_params(n, 2, deg, flags, 0.3);
        let fwd = rig.forward(&params);
        let v = fwd.vis_idx.len();
        let opac: Vec<f32> = host
            .opacities
            .iter()
            .map(|&o| 1.0 / (1.0 + (-o).exp()))
            .collect();

        // Upstream gradients in p-space (zero for Gaussians we cull).
        let w = random_g2d(v, 77, true);
        let (mut v_m2, mut v_con, mut v_dep, mut v_comp) = (
            vec![0f32; 2 * n],
            vec![0f32; 3 * n],
            vec![0f32; n],
            vec![0f32; n],
        );
        let mut proj_w = w.clone();
        for (i, &p) in fwd.vis_idx.iter().enumerate() {
            let p = p as usize;
            v_m2[2 * p..2 * p + 2].copy_from_slice(&w[i][0..2]);
            v_con[3 * p..3 * p + 3].copy_from_slice(&w[i][2..5]);
            v_dep[p] = w[i][9];
            // gsplat's v_compensations = dL/dc2 = g_o · o (c3 = 1).
            v_comp[p] = if aa { w[i][5] * opac[p] } else { 0.0 };
            if !aa {
                proj_w[i][5] = 0.0;
            }
            proj_w[i][6..9].copy_from_slice(&[0.0; 3]);
        }
        let mut input = Npz::new();
        input
            .f32("means", &[n, 3], host.means.clone())
            .unwrap()
            .f32("quats", &[n, 4], host.rotations.clone())
            .unwrap()
            .f32(
                "scales",
                &[n, 3],
                host.scales.iter().map(|x| x.exp()).collect(),
            )
            .unwrap()
            .f32("opacities", &[n], opac.clone())
            .unwrap()
            .f32("viewmats", &[1, 4, 4], view_matrix(view).to_vec())
            .unwrap()
            .f32("Ks", &[1, 3, 3], k_matrix(view).to_vec())
            .unwrap()
            .i32_scalar("width", view.width as i32)
            .unwrap()
            .i32_scalar("height", view.height as i32)
            .unwrap()
            .f32_scalar("eps2d", 0.3)
            .unwrap()
            .f32_scalar("near_plane", 0.01)
            .unwrap()
            .f32_scalar("far_plane", 1e4)
            .unwrap()
            .i32_scalar("calc_compensations", i32::from(aa))
            .unwrap()
            .f32("v_means2d", &[1, n, 2], v_m2)
            .unwrap()
            .f32("v_conics", &[1, n, 3], v_con)
            .unwrap()
            .f32("v_depths", &[1, n], v_dep)
            .unwrap();
        if aa {
            input.f32("v_compensations", &[1, n], v_comp).unwrap();
        }
        crate::testkit::oracle::stage(&dir, name, "project", &input).unwrap();
        let proj = rig.backward(&params, &proj_w, false);

        // SH: dirs = μ − campos, v_colors zero where our colour is clamped.
        let campos = RefCam::from_frame(&fwd.cam).campos();
        let mut dirs = vec![0f32; 3 * n];
        let mut coeffs = vec![0f32; n * 16 * 3];
        let mut masks = vec![false; n];
        let mut v_col = vec![0f32; 3 * n];
        let mut col_w = vec![[0f32; 10]; v];
        for p in 0..n {
            for k in 0..3 {
                dirs[3 * p + k] = (host.means[3 * p + k] as f64 - campos[k]) as f32;
            }
            coeffs[48 * p..48 * p + 3].copy_from_slice(&host.sh0[3 * p..3 * p + 3]);
            coeffs[48 * p + 3..48 * p + 48].copy_from_slice(&host.sh_rest[45 * p..45 * p + 45]);
        }
        for (i, &p) in fwd.vis_idx.iter().enumerate() {
            let p = p as usize;
            masks[p] = true;
            for c in 0..3 {
                if fwd.rec[i].flags >> c & 1 == 0 {
                    v_col[3 * p + c] = w[i][6 + c];
                    col_w[i][6 + c] = w[i][6 + c];
                }
            }
        }
        let mut sh_in = Npz::new();
        sh_in
            .i32_scalar("degree", deg as i32)
            .unwrap()
            .f32("dirs", &[n, 3], dirs)
            .unwrap()
            .f32("coeffs", &[n, 16, 3], coeffs)
            .unwrap()
            .insert("masks", &[n], NpyData::Bool(masks))
            .unwrap()
            .f32("v_colors", &[n, 3], v_col)
            .unwrap();
        crate::testkit::oracle::stage(&dir, name, "sh", &sh_in).unwrap();
        let shg = rig.backward(&params, &col_w, false);

        // Our side, as gs-oracle keys (zero where culled), for `gs-oracle compare`.
        let mut ours = Npz::new();
        let (mut m2, mut dep, mut con, mut comp) = (
            vec![0f32; 2 * n],
            vec![0f32; n],
            vec![0f32; 3 * n],
            vec![0f32; n],
        );
        for (i, &p) in fwd.vis_idx.iter().enumerate() {
            let p = p as usize;
            let r = fwd.rec[i];
            m2[2 * p] = r.mx;
            m2[2 * p + 1] = r.my;
            dep[p] = r.z;
            let c = rec_conic(&r);
            con[3 * p..3 * p + 3].copy_from_slice(&c.map(|x| x as f32));
            comp[p] = r.o / opac[p];
        }
        ours.f32("means2d", &[1, n, 2], m2.clone())
            .unwrap()
            .f32("depths", &[1, n], dep.clone())
            .unwrap()
            .f32("conics", &[1, n, 3], con.clone())
            .unwrap()
            .f32("v_means", &[n, 3], proj.means.clone())
            .unwrap()
            .f32("v_quats", &[n, 4], proj.quats.clone())
            .unwrap();
        crate::testkit::oracle::stage_ours(&dir, name, "project", &ours).unwrap();

        let out_path = crate::testkit::oracle::case_path(&dir, name, "project", "out");
        let sh_path = crate::testkit::oracle::case_path(&dir, name, "sh", "out");
        if !out_path.is_file() || !sh_path.is_file() {
            println!(
                "{name}: staged in {}; run `gs-oracle batch {}` on the host, then rerun",
                dir.display(),
                host_path(&dir).display()
            );
            continue;
        }
        let gs = Npz::read(&out_path).unwrap();
        let gsh = Npz::read(&sh_path).unwrap();
        compared += 1;
        println!("{name}: {v}/{n} visible");
        let radii = gs.get("radii").unwrap().to_f64();
        let gs_vis: Vec<bool> = radii.chunks(2).map(|r| r[0] > 0.0 && r[1] > 0.0).collect();
        let mut visv = vec![false; n];
        fwd.vis_idx.iter().for_each(|&p| visv[p as usize] = true);
        let not_in_gsplat = (0..n).filter(|&p| visv[p] && !gs_vis[p]).count();
        assert_eq!(
            not_in_gsplat, 0,
            "{name}: visible here but culled by gsplat"
        );
        println!(
            "  culled set: {} here vs {} with gsplat radii > 0 (ours ⊆ gsplat)",
            v,
            gs_vis.iter().filter(|&&x| x).count()
        );
        let pick = |src: &[f64], stride: usize| -> Vec<f64> {
            fwd.vis_idx
                .iter()
                .flat_map(|&p| src[stride * p as usize..stride * (p as usize + 1)].to_vec())
                .collect()
        };
        let pick32 = |src: &[f32], stride: usize| -> Vec<f32> {
            fwd.vis_idx
                .iter()
                .flat_map(|&p| src[stride * p as usize..stride * (p as usize + 1)].to_vec())
                .collect()
        };
        let mut checks: Vec<(String, Vec<f32>, Vec<f64>, Tol)> = vec![
            (
                "means2d".into(),
                pick32(&m2, 2),
                pick(&gs.get("means2d").unwrap().to_f64(), 2),
                Tol {
                    max_abs: Some(1e-4),
                    ..Tol::NONE
                },
            ),
            (
                "depths".into(),
                pick32(&dep, 1),
                pick(&gs.get("depths").unwrap().to_f64(), 1),
                Tol {
                    elem_rel: Some((1e-6, 0.0)),
                    ..Tol::NONE
                },
            ),
            (
                "conics".into(),
                scaled_conics(
                    &pick32(&con, 3),
                    &pick(&gs.get("conics").unwrap().to_f64(), 3),
                )
                .0,
                scaled_conics(
                    &pick32(&con, 3),
                    &pick(&gs.get("conics").unwrap().to_f64(), 3),
                )
                .1,
                CONIC_SCALED,
            ),
        ];
        if aa {
            checks.push((
                "compensations".into(),
                pick32(&comp, 1),
                pick(&gs.get("compensations").unwrap().to_f64(), 1),
                Tol::CONIC,
            ));
        }
        // 3D gradients: means (projection + SH paths), raw quats, log scales.
        let v_means = gs.get("v_means").unwrap().to_f64();
        let v_dirs = gsh.get("v_dirs").unwrap().to_f64();
        let v_scales = gs.get("v_scales").unwrap().to_f64();
        let scales: Vec<f64> = host.scales.iter().map(|x| (*x as f64).exp()).collect();
        let ls: Vec<f64> = v_scales.iter().zip(&scales).map(|(g, s)| g * s).collect();
        checks.push((
            "v_means(proj)".into(),
            pick32(&proj.means, 3),
            pick(&v_means, 3),
            Tol::GRAD_3D,
        ));
        checks.push((
            "v_quats".into(),
            pick32(&proj.quats, 4),
            pick(&gs.get("v_quats").unwrap().to_f64(), 4),
            Tol::GRAD_3D,
        ));
        checks.push((
            "v_log_scales".into(),
            pick32(&proj.scales, 3),
            pick(&ls, 3),
            Tol::GRAD_3D,
        ));
        checks.push((
            "v_means(sh)".into(),
            pick32(&shg.means, 3),
            pick(&v_dirs, 3),
            Tol::GRAD_3D,
        ));
        // SH colours and coefficient gradients.
        let gcol = gsh.get("colors").unwrap().to_f64();
        let want_col: Vec<f64> = gcol.iter().map(|c| (c + 0.5).max(0.0)).collect();
        let our_col: Vec<f32> = fwd.rec.iter().flat_map(|r| [r.r, r.g, r.b]).collect();
        checks.push((
            "colors".into(),
            our_col,
            pick(&want_col, 3),
            Tol {
                max_abs: Some(1e-5),
                ..Tol::NONE
            },
        ));
        let v_coeffs = gsh.get("v_coeffs").unwrap().to_f64();
        let k_used = ((deg + 1) * (deg + 1)) as usize;
        let mut ours_c = Vec::new();
        let mut want_c = Vec::new();
        for &p in &fwd.vis_idx {
            let p = p as usize;
            for k in 0..k_used {
                for c in 0..3 {
                    ours_c.push(if k == 0 {
                        shg.sh0[3 * p + c]
                    } else {
                        shg.sh_rest[45 * p + 3 * (k - 1) + c]
                    });
                    want_c.push(v_coeffs[48 * p + 3 * k + c]);
                }
            }
        }
        checks.push(("v_coeffs".into(), ours_c, want_c, Tol::GRAD_3D));
        for (label, a, b, tol) in checks {
            let d = Diff::new(&format!("{name} {label}"), &a, &b, 1e-3);
            println!("  {d}");
            d.check(&tol).unwrap();
        }
        // Pose: gsplat v_viewmats [G_R | g_t] → (ρ, φ) = (g_t, vee'(G_R Rᵀ) + t × g_t).
        let vm = gs.get("v_viewmats").unwrap().to_f64();
        let cam = RefCam::from_frame(&fwd.cam);
        let gr: M3 = std::array::from_fn(|i| std::array::from_fn(|j| vm[4 * i + j]));
        let gt = [vm[3], vm[7], vm[11]];
        let rt: M3 = std::array::from_fn(|i| std::array::from_fn(|j| cam.r[j][i]));
        let x = mul3(&gr, &rt);
        let t = cam.t;
        let want_pose = [
            gt[0],
            gt[1],
            gt[2],
            x[2][1] - x[1][2] + t[1] * gt[2] - t[2] * gt[1],
            x[0][2] - x[2][0] + t[2] * gt[0] - t[0] * gt[2],
            x[1][0] - x[0][1] + t[0] * gt[1] - t[1] * gt[0],
        ];
        let norm = want_pose.iter().map(|x| x * x).sum::<f64>().sqrt();
        let err = proj
            .pose
            .iter()
            .zip(&want_pose)
            .map(|(a, b)| (*a as f64 - b).abs())
            .fold(0.0, f64::max);
        println!(
            "  {name} pose: max err {err:.3e} of |g| {norm:.3e}\n    ours   {:?}\n    gsplat {want_pose:?}",
            proj.pose
        );
        assert!(
            err <= 1e-3 * norm,
            "{name}: pose gradient vs gsplat v_viewmats"
        );
    }
    println!("{compared} oracle cases compared");
}

/// Where `/out/...` lives on the host (for the instructions printed above).
fn host_path(dir: &Path) -> PathBuf {
    match dir.strip_prefix("/out") {
        Ok(rest) => PathBuf::from("~/spl-work/out").join(rest),
        Err(_) => dir.to_path_buf(),
    }
}

// ================================================================ garden benchmark

/// A 3DGS binary little-endian PLY (Inria / gsplat export layout) as the host
/// mirror; SH rest is stored channel-major there (`f_rest_{c·15 + k}`).
fn read_ply(path: &Path) -> GaussiansHost {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let marker = b"end_header\n";
    let end = bytes
        .windows(marker.len())
        .position(|w| w == marker)
        .expect("PLY header")
        + marker.len();
    let header = std::str::from_utf8(&bytes[..end]).unwrap();
    assert!(
        header.contains("binary_little_endian"),
        "binary PLY expected"
    );
    let mut n = 0usize;
    let mut props = Vec::new();
    for line in header.lines() {
        let w: Vec<&str> = line.split_whitespace().collect();
        match w.as_slice() {
            ["element", "vertex", c] => n = c.parse().unwrap(),
            ["property", "float", name] => props.push(name.to_string()),
            ["property", ..] => panic!("only float properties are supported: {line}"),
            _ => {}
        }
    }
    let col = |name: &str| {
        props
            .iter()
            .position(|p| p == name)
            .unwrap_or_else(|| panic!("PLY lacks {name}"))
    };
    let stride = props.len();
    let data: Vec<f32> = bytes[end..end + 4 * stride * n]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    let mut g = GaussiansHost::zeros(n);
    let (x, dc, op, sc, rot) = (
        col("x"),
        col("f_dc_0"),
        col("opacity"),
        col("scale_0"),
        col("rot_0"),
    );
    let rest = props.iter().position(|p| p == "f_rest_0");
    for p in 0..n {
        let row = &data[p * stride..(p + 1) * stride];
        g.means[3 * p..3 * p + 3].copy_from_slice(&row[x..x + 3]);
        g.sh0[3 * p..3 * p + 3].copy_from_slice(&row[dc..dc + 3]);
        g.opacities[p] = row[op];
        g.scales[3 * p..3 * p + 3].copy_from_slice(&row[sc..sc + 3]);
        g.rotations[4 * p..4 * p + 4].copy_from_slice(&row[rot..rot + 4]);
        if let Some(r) = rest {
            for k in 0..15 {
                for c in 0..3 {
                    g.sh_rest[45 * p + 3 * k + c] = row[r + c * 15 + k];
                }
            }
        }
    }
    g
}

/// 30-bit Morton order of the means (the order the trainer keeps, Faster-GS).
fn morton_order(g: &GaussiansHost) -> Vec<u32> {
    let n = g.len();
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for p in 0..n {
        for k in 0..3 {
            lo[k] = lo[k].min(g.means[3 * p + k]);
            hi[k] = hi[k].max(g.means[3 * p + k]);
        }
    }
    let spread = |mut v: u32| {
        v &= 0x3ff;
        v = (v | (v << 16)) & 0x0300_00ff;
        v = (v | (v << 8)) & 0x0300_f00f;
        v = (v | (v << 4)) & 0x030c_30c3;
        (v | (v << 2)) & 0x0924_9249
    };
    let mut keys: Vec<(u32, u32)> = (0..n)
        .map(|p| {
            let q = |k: usize| {
                let t = (g.means[3 * p + k] - lo[k]) / (hi[k] - lo[k]).max(1e-12);
                (t * 1023.0).clamp(0.0, 1023.0) as u32
            };
            (
                spread(q(0)) | spread(q(1)) << 1 | spread(q(2)) << 2,
                p as u32,
            )
        })
        .collect();
    keys.sort_unstable();
    keys.into_iter().map(|(_, p)| p).collect()
}

/// `src` reordered by `order`, optionally each Gaussian twice (second copy
/// jittered by a fraction of its scale) to reach a larger N with the same
/// spatial statistics.
fn gather(src: &GaussiansHost, order: &[u32], duplicate: bool) -> GaussiansHost {
    let copies = if duplicate { 2 } else { 1 };
    let mut g = GaussiansHost::zeros(order.len() * copies);
    let rng = Philox::new(9);
    for (i, &p) in order.iter().enumerate() {
        let p = p as usize;
        for c in 0..copies {
            let d = copies * i + c;
            g.means[3 * d..3 * d + 3].copy_from_slice(&src.means[3 * p..3 * p + 3]);
            if c == 1 {
                let r = rng.draw4(i as u32, 0, 5);
                for (k, &rk) in r.iter().take(3).enumerate() {
                    g.means[3 * d + k] += 0.5 * src.scales[3 * p + k].exp() * (u01_open(rk) - 0.5);
                }
            }
            g.scales[3 * d..3 * d + 3].copy_from_slice(&src.scales[3 * p..3 * p + 3]);
            g.rotations[4 * d..4 * d + 4].copy_from_slice(&src.rotations[4 * p..4 * p + 4]);
            g.opacities[d] = src.opacities[p];
            g.sh0[3 * d..3 * d + 3].copy_from_slice(&src.sh0[3 * p..3 * p + 3]);
            g.sh_rest[45 * d..45 * d + 45].copy_from_slice(&src.sh_rest[45 * p..45 * p + 45]);
        }
    }
    g
}

#[derive(serde::Serialize)]
struct BenchRow {
    label: String,
    n: usize,
    views: usize,
    visible_mean: f64,
    instances_mean: f64,
    fwd: Stats,
    bwd: Stats,
    bwd_cam_stats: Stats,
    fwd_gbps: f64,
    bwd_gbps: f64,
}

/// Mip-NeRF 360 garden at 1/4 resolution with a trained 3M-Gaussian model
/// (gsplat MCMC, 10k steps; `SPL_GARDEN_PLY`, default
/// /out/image-to-3d/garden_mcmc3m_10k.ply), Morton-ordered, at N = 1M (every
/// third), 3M and 6M (each twice, jittered), over every 8th training view.
/// Times frame_begin + preprocess_fwd and preprocess_bwd (Write sink; with and
/// without camera gradients and densify statistics) with CUDA events.
/// Writes /out/image-to-3d/bench/preprocess.json and garden_views.npz (the
/// cameras and Morton order for the gsplat comparison script).
#[test]
#[ignore]
fn gpu_preprocess_bench_garden() {
    let Some(root) = crate::testkit::mip360_scene("garden") else {
        eprintln!("skipped: garden dataset not found");
        return;
    };
    let ply = std::env::var_os("SPL_GARDEN_PLY")
        .map(PathBuf::from)
        .unwrap_or_else(|| crate::testkit::out_dir("garden_mcmc3m_10k.ply"));
    if !ply.is_file() {
        eprintln!("skipped: {} not found", ply.display());
        return;
    }
    let scene = crate::io::scene::load_colmap_scene(
        &root,
        &crate::io::scene::LoadOptions {
            images_dir: "images_4".into(),
            ..Default::default()
        },
    )
    .unwrap();
    let views: Vec<View> = scene
        .train
        .iter()
        .map(|&i| scene.views[i].clone())
        .collect();
    let bench_views: Vec<usize> = (0..views.len()).step_by(8).collect();
    let t0 = std::time::Instant::now();
    let model = read_ply(&ply);
    let order = morton_order(&model);
    println!(
        "garden: {} Gaussians from {} in {:.1} s, {} train views ({}x{}), {} timed",
        model.len(),
        ply.display(),
        t0.elapsed().as_secs_f64(),
        views.len(),
        views[0].width,
        views[0].height,
        bench_views.len()
    );
    let out = crate::testkit::out_dir("bench");
    std::fs::create_dir_all(&out).unwrap();
    {
        let mut z = Npz::new();
        let nv = bench_views.len();
        z.f32(
            "viewmats",
            &[nv, 4, 4],
            bench_views
                .iter()
                .flat_map(|&i| view_matrix(&views[i]))
                .collect(),
        )
        .unwrap()
        .f32(
            "Ks",
            &[nv, 3, 3],
            bench_views
                .iter()
                .flat_map(|&i| k_matrix(&views[i]))
                .collect(),
        )
        .unwrap()
        .i32_scalar("width", views[0].width as i32)
        .unwrap()
        .i32_scalar("height", views[0].height as i32)
        .unwrap()
        .insert("order", &[order.len()], NpyData::U32(order.clone()))
        .unwrap();
        z.write(&out.join("garden_views.npz")).unwrap();
    }
    let gpu = gpu();
    let before = gpu_snapshot();
    let mut rows = Vec::new();
    let every_third: Vec<u32> = order.iter().copied().step_by(3).collect();
    for (label, host) in [
        ("1M", gather(&model, &every_third, false)),
        ("3M", gather(&model, &order, false)),
        ("6M", gather(&model, &order, true)),
    ] {
        let n = host.len();
        let mut rig = Rig::new(&gpu, &host, &views, RasterVariant::default(), None, None);
        drop(host);
        let s = gpu.stream.clone();
        let vc = rig.f.cap.visible as usize;
        // Random 2D gradients to restore after each forward (which zeroes g2d).
        let rng = Philox::new(4);
        let src: Vec<f32> = (0..9 * vc as u32)
            .map(|i| 1e-3 * normal2(rng.draw4(i, 0, 8)[0], rng.draw4(i, 0, 8)[1]).0)
            .collect();
        let src = s.clone_htod(&src).unwrap();
        let mut grads = Grads::alloc(&gpu, rig.g.cap as usize).unwrap();
        let mut camg = CameraGrads::alloc(
            &gpu,
            rig.cams.n_images,
            rig.cams.n_intr,
            PreprocessKernels::cam_partial_blocks(&gpu),
        )
        .unwrap();
        let mut stats = DensifyStats::alloc(&gpu, rig.g.cap as usize, true).unwrap();
        let (mut fwd_ms, mut bwd_ms, mut bwd2_ms) = (Vec::new(), Vec::new(), Vec::new());
        let (mut vis_sum, mut inst_sum) = (0f64, 0f64);
        let (mut fwd_bytes, mut bwd_bytes) = (0f64, 0f64);
        for &vi in &bench_views {
            let mut params = iter_params(n, vi as u32, 3, 0, 0.3);
            params.adam.overwrite_mask = 0x3f;
            rig.f.iter.upload(&gpu, &params, None).unwrap();
            let k = &rig.k;
            let fstat = bench(&gpu, 3, 10, || {
                frame_begin_eager(k, &s, &rig.cams, &mut rig.f)?;
                preprocess_fwd(k, &s, &mut rig.g, &mut rig.f, &mut rig.g2d)
            })
            .unwrap();
            let status = rig.f.iter.read_status(&gpu).unwrap();
            let v = status.n_visible as usize;
            let tiles: Vec<u32> = s.clone_dtoh(&rig.f.n_tiles.slice(..v.max(1))).unwrap();
            vis_sum += v as f64;
            inst_sum += tiles[..v].iter().map(|&t| t as f64).sum::<f64>();
            s.memcpy_dtod(&src, &mut rig.g2d.g).unwrap();
            let bstat = bench(&gpu, 3, 10, || {
                preprocess_bwd(
                    k,
                    &s,
                    &mut rig.g,
                    &rig.f,
                    &rig.g2d,
                    GradSink::Write(&mut grads),
                    None,
                    None,
                )
            })
            .unwrap();
            let mut p2 = params;
            p2.frame.flags = FLAG_CAM_GRAD | FLAG_DENSIFY_STATS;
            rig.f.iter.upload(&gpu, &p2, None).unwrap();
            let b2 = bench(&gpu, 3, 10, || {
                preprocess_bwd(
                    k,
                    &s,
                    &mut rig.g,
                    &rig.f,
                    &rig.g2d,
                    GradSink::Write(&mut grads),
                    Some(&mut camg),
                    Some(&mut stats),
                )?;
                cam_reduce(k, &s, &rig.f, &mut camg)
            })
            .unwrap();
            // Bytes model: all N read mean + opacity (16 B); visible read the
            // rest of the parameters (16 + 12 + 12 + 180 B) and write 48 B rec,
            // 16 B keys/idx/tiles, 44 B g2d zeroing; vis mask N/8.
            fwd_bytes += 16.0 * n as f64 + n as f64 / 8.0 + (220.0 + 108.0 + 4.0) * v as f64;
            // Backward per visible: vis_idx 4 + g2d 36 + params 236 + gradients 236.
            bwd_bytes += 512.0 * v as f64;
            fwd_ms.push(fstat.median_ms);
            bwd_ms.push(bstat.median_ms);
            bwd2_ms.push(b2.median_ms);
        }
        let nv = bench_views.len() as f64;
        let fs = Stats::from_samples(&fwd_ms);
        let bs = Stats::from_samples(&bwd_ms);
        let row = BenchRow {
            label: label.into(),
            n,
            views: bench_views.len(),
            visible_mean: vis_sum / nv,
            instances_mean: inst_sum / nv,
            fwd_gbps: fwd_bytes / nv / (fs.mean_ms * 1e-3) / 1e9,
            bwd_gbps: bwd_bytes / nv / (bs.mean_ms * 1e-3) / 1e9,
            bwd_cam_stats: Stats::from_samples(&bwd2_ms),
            fwd: fs,
            bwd: bs,
        };
        println!(
            "{label}: N {n}, V {:.0} ({:.1}%), instances {:.0} ({:.2}/visible) | fwd mean {:.3} ms (median {:.3}, p10 {:.3}, p90 {:.3}) ~{:.0} GB/s | bwd mean {:.3} ms (median {:.3}) ~{:.0} GB/s | bwd+cam+stats {:.3} ms",
            row.visible_mean,
            100.0 * row.visible_mean / n as f64,
            row.instances_mean,
            row.instances_mean / row.visible_mean,
            row.fwd.mean_ms,
            row.fwd.median_ms,
            row.fwd.p10_ms,
            row.fwd.p90_ms,
            row.fwd_gbps,
            row.bwd.mean_ms,
            row.bwd.median_ms,
            row.bwd_gbps,
            row.bwd_cam_stats.mean_ms
        );
        rows.push(row);
    }
    let after = gpu_snapshot();
    println!("nvidia-smi before {before:?}\nnvidia-smi after  {after:?}");
    let report = serde_json::json!({
        "ply": ply.display().to_string(),
        "gpu_before": before,
        "gpu_after": after,
        "rows": rows,
    });
    std::fs::write(
        out.join("preprocess.json"),
        serde_json::to_string_pretty(&report).unwrap(),
    )
    .unwrap();
}
