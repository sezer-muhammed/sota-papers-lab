//! Bundle adjustment on depth-guided KLT tracks, before training. The
//! engine's poses are about 0.6-0.9° off COLMAP on Mip-NeRF 360 (median
//! pairwise rotation error) and photometric joint refinement does not remove
//! that (the Gaussians absorb it); a reprojection-error BA on real
//! correspondences does. Owner: mvp-quality-2.
//!
//! 1. Keypoints: per frame, the strongest Shi-Tomasi corner of each grid
//!    cell (half trainer resolution) that has engine depth.
//! 2. Tracks: each keypoint is predicted in the frame's overlapping
//!    neighbours ([`geom::neighbours`]) from the engine depth and cameras,
//!    then refined by pyramidal Lucas-Kanade (translation) and kept when
//!    tracking back lands within `fb_px` of the start.
//! 3. BA: Levenberg-Marquardt over every frame's pose (the first training
//!    frame is the gauge), one shared log-focal and one inverse depth per
//!    keypoint (in its anchor frame), Huber on the pixel residual; the point
//!    block is eliminated (Schur complement) and the reduced camera system
//!    solved densely.
//!
//! CPU, rayon. Conventions as [`Posed`] (world-to-camera, OpenCV axes, pixel
//! centres at +0.5).

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::geom::{
    Cam, conf_threshold, depth_view, grid_samples, mm, mv, neighbours, so3_exp, transpose,
};
use super::scene::Posed;
use crate::io::images::Rgb8Image;
use crate::io::scene::InitPoints;

type M3 = [[f64; 3]; 3];
type V3 = [f64; 3];

/// BA settings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BaConfig {
    /// LM iterations; 0 = off.
    pub iters: u32,
    /// Keypoints per frame (grid cells).
    pub keypoints: u32,
    /// Neighbours each keypoint is tracked into.
    pub neighbours: u32,
    /// Working resolution: trainer size divided by this.
    pub downscale: u32,
    /// KLT window half size and pyramid levels.
    pub half_win: u32,
    pub levels: u32,
    /// Forward-backward tolerance (working pixels).
    pub fb_px: f64,
    /// Huber width (working pixels).
    pub huber_px: f64,
    /// Refine the shared focal.
    pub focal: bool,
    /// Above 0: a second round without observations off by more (working px).
    pub reject_px: f64,
    /// Above 0: σ (degrees) of a prior holding each pose near its start
    /// (translation σ: the same fraction of the camera spread).
    pub prior_deg: f64,
}

impl Default for BaConfig {
    fn default() -> Self {
        Self {
            iters: 0,
            keypoints: 1024,
            neighbours: 6,
            downscale: 2,
            half_win: 6,
            levels: 3,
            fb_px: 0.5,
            huber_px: 1.0,
            focal: true,
            reject_px: 0.0,
            prior_deg: 0.0,
        }
    }
}

/// What the BA did.
#[derive(Clone, Debug, Default, Serialize)]
pub struct BaReport {
    pub frames: usize,
    pub points: usize,
    pub observations: usize,
    /// After the outlier rejection.
    pub observations_kept: usize,
    /// Tracks attempted and kept.
    pub tracks_tried: usize,
    pub tracks_kept: usize,
    /// Robust RMS reprojection error (working pixels), before and after.
    pub rms_before: f64,
    pub rms_after: f64,
    /// Inliers (residual < 3 Huber widths) before and after.
    pub inliers_before: f64,
    pub inliers_after: f64,
    pub focal_scale: f64,
    pub median_rot_change_deg: f64,
    pub iterations: u32,
    pub track_s: f64,
    pub solve_s: f64,
}

/// `want` of `list` (best first): the first half, then evenly spaced from
/// the rest.
fn spread_pick(list: &[usize], want: usize) -> Vec<usize> {
    if list.len() <= want {
        return list.to_vec();
    }
    let head = want.div_ceil(2);
    let mut out: Vec<usize> = list[..head].to_vec();
    let rest = &list[head..];
    let k = want - head;
    for j in 0..k {
        out.push(rest[(j * rest.len() + rest.len() / 2) / k.max(1)]);
    }
    out
}

/// A float gray image.
struct Gray {
    w: usize,
    h: usize,
    p: Vec<f32>,
}

impl Gray {
    /// Luma, area-averaged by `k`.
    fn from_rgb(img: &Rgb8Image, k: usize) -> Self {
        let k = k.max(1);
        let (w, h) = (img.width as usize / k, img.height as usize / k);
        let sw = img.width as usize;
        let mut p = vec![0.0f32; w * h];
        let norm = 1.0 / (k * k) as f32;
        p.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for (x, o) in row.iter_mut().enumerate() {
                let mut s = 0.0f32;
                for dy in 0..k {
                    for dx in 0..k {
                        let q = ((y * k + dy) * sw + x * k + dx) * 3;
                        let d = &img.data[q..q + 3];
                        s += 0.299 * d[0] as f32 + 0.587 * d[1] as f32 + 0.114 * d[2] as f32;
                    }
                }
                *o = s * norm / 255.0;
            }
        });
        Self { w, h, p }
    }

    fn half(&self) -> Self {
        let (w, h) = (self.w / 2, self.h / 2);
        let mut p = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let a = |xx: usize, yy: usize| self.p[yy * self.w + xx];
                p[y * w + x] = 0.25
                    * (a(2 * x, 2 * y)
                        + a(2 * x + 1, 2 * y)
                        + a(2 * x, 2 * y + 1)
                        + a(2 * x + 1, 2 * y + 1));
            }
        }
        Self { w, h, p }
    }

    /// Bilinear at array coordinates (pixel `i` at `i`, not `i + 0.5`).
    #[inline]
    fn at(&self, x: f32, y: f32) -> Option<f32> {
        if !(x >= 0.0 && y >= 0.0) {
            return None;
        }
        let (x0, y0) = (x as usize, y as usize);
        if x0 + 1 >= self.w || y0 + 1 >= self.h {
            return None;
        }
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let i = y0 * self.w + x0;
        let (a, b, c, d) = (
            self.p[i],
            self.p[i + 1],
            self.p[i + self.w],
            self.p[i + self.w + 1],
        );
        Some((a * (1.0 - fx) + b * fx) * (1.0 - fy) + (c * (1.0 - fx) + d * fx) * fy)
    }
}

fn pyramid(g: Gray, levels: usize) -> Vec<Gray> {
    let mut out = vec![g];
    while out.len() < levels.max(1) {
        let next = out.last().unwrap().half();
        if next.w < 16 || next.h < 16 {
            break;
        }
        out.push(next);
    }
    out
}

/// The strongest Shi-Tomasi corner per grid cell (array coordinates).
fn corners(g: &Gray, n: usize) -> Vec<(f32, f32)> {
    let cell = ((g.w * g.h) as f64 / n.max(1) as f64).sqrt().max(4.0) as usize;
    let (cw, ch) = (g.w.div_ceil(cell), g.h.div_ceil(cell));
    let r = 2usize;
    (0..cw * ch)
        .into_par_iter()
        .filter_map(|c| {
            let (cx, cy) = (c % cw, c / cw);
            let mut best = (0.0f32, 0usize, 0usize);
            for y in (cy * cell).max(r + 1)..((cy + 1) * cell).min(g.h - r - 1) {
                for x in (cx * cell).max(r + 1)..((cx + 1) * cell).min(g.w - r - 1) {
                    let (mut xx, mut xy, mut yy) = (0.0f32, 0.0f32, 0.0f32);
                    for dy in 0..=2 * r {
                        for dx in 0..=2 * r {
                            let (px, py) = (x + dx - r, y + dy - r);
                            let i = py * g.w + px;
                            let gx = 0.5 * (g.p[i + 1] - g.p[i - 1]);
                            let gy = 0.5 * (g.p[i + g.w] - g.p[i - g.w]);
                            xx += gx * gx;
                            xy += gx * gy;
                            yy += gy * gy;
                        }
                    }
                    let tr = 0.5 * (xx + yy);
                    let s = tr - (0.25 * (xx - yy) * (xx - yy) + xy * xy).sqrt();
                    if s > best.0 {
                        best = (s, x, y);
                    }
                }
            }
            (best.0 > 1e-4).then_some((best.1 as f32, best.2 as f32))
        })
        .collect()
}

/// Pyramidal translation-only Lucas-Kanade from `a` at `p` to `b`, starting
/// at `guess`. Array coordinates at level 0.
fn klt(a: &[Gray], b: &[Gray], p: (f32, f32), guess: (f32, f32), hw: i32) -> Option<(f32, f32)> {
    let levels = a.len().min(b.len());
    let mut d = (
        (guess.0 - p.0) / (1 << (levels - 1)) as f32,
        (guess.1 - p.1) / (1 << (levels - 1)) as f32,
    );
    let n = ((2 * hw + 1) * (2 * hw + 1)) as usize;
    let mut tpl = vec![0.0f32; n];
    let mut gxs = vec![0.0f32; n];
    let mut gys = vec![0.0f32; n];
    for l in (0..levels).rev() {
        let s = 1.0 / (1 << l) as f32;
        let (px, py) = (p.0 * s, p.1 * s);
        let (ga, gb) = (&a[l], &b[l]);
        let (mut h11, mut h12, mut h22) = (0.0f32, 0.0f32, 0.0f32);
        let mut k = 0;
        for dy in -hw..=hw {
            for dx in -hw..=hw {
                let (x, y) = (px + dx as f32, py + dy as f32);
                let v = ga.at(x, y)?;
                let gx = 0.5 * (ga.at(x + 1.0, y)? - ga.at(x - 1.0, y)?);
                let gy = 0.5 * (ga.at(x, y + 1.0)? - ga.at(x, y - 1.0)?);
                tpl[k] = v;
                gxs[k] = gx;
                gys[k] = gy;
                h11 += gx * gx;
                h12 += gx * gy;
                h22 += gy * gy;
                k += 1;
            }
        }
        let det = h11 * h22 - h12 * h12;
        if det < 1e-8 {
            return None;
        }
        for _ in 0..20 {
            let (mut b1, mut b2) = (0.0f32, 0.0f32);
            let mut k = 0;
            for dy in -hw..=hw {
                for dx in -hw..=hw {
                    let v = gb.at(px + d.0 + dx as f32, py + d.1 + dy as f32)?;
                    let e = v - tpl[k];
                    b1 += gxs[k] * e;
                    b2 += gys[k] * e;
                    k += 1;
                }
            }
            let ux = -(h22 * b1 - h12 * b2) / det;
            let uy = -(h11 * b2 - h12 * b1) / det;
            d.0 += ux;
            d.1 += uy;
            if ux * ux + uy * uy < 1e-4 {
                break;
            }
        }
        if l > 0 {
            d = (2.0 * d.0, 2.0 * d.1);
        }
    }
    let q = (p.0 + d.0, p.1 + d.1);
    let (w, h) = (b[0].w as f32, b[0].h as f32);
    (q.0 >= 0.0 && q.1 >= 0.0 && q.0 < w - 1.0 && q.1 < h - 1.0).then_some(q)
}

/// A keypoint: anchor frame (index into `frames`), its pixel (pixel-centre
/// coordinates at working resolution) and inverse depth.
#[derive(Clone, Copy, Debug)]
struct Point {
    anchor: usize,
    u: f64,
    v: f64,
    rho: f64,
}

#[derive(Clone, Copy, Debug)]
struct Obs {
    point: usize,
    frame: usize,
    u: f64,
    v: f64,
}

/// Shared pinhole at working resolution (log-focal scale `s`).
#[derive(Clone, Copy, Debug)]
struct Intr {
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
}

/// Cameras: world-to-camera rotation and translation.
#[derive(Clone, Copy, Debug)]
struct Pose {
    r: M3,
    t: V3,
}

impl Pose {
    fn update(&self, xi: &[f64]) -> Self {
        let rd = so3_exp([xi[3], xi[4], xi[5]]);
        let t = mv(&rd, self.t);
        Self {
            r: mm(&rd, &self.r),
            t: [t[0] + xi[0], t[1] + xi[1], t[2] + xi[2]],
        }
    }
}

fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// One observation's residual (2) and Jacobians: anchor pose (2×6), frame
/// pose (2×6), inverse depth (2), log focal (2). `None` behind the camera.
#[allow(clippy::type_complexity)]
fn linearize(
    k: &Intr,
    s: f64,
    pa: &Pose,
    pb: &Pose,
    pt: &Point,
    o: &Obs,
) -> Option<([f64; 2], [[f64; 6]; 2], [[f64; 6]; 2], [f64; 2], [f64; 2])> {
    let (fx, fy) = (k.fx * s.exp(), k.fy * s.exp());
    let ray = [(pt.u - k.cx) / fx, (pt.v - k.cy) / fy, 1.0];
    let z = 1.0 / pt.rho;
    let q = [ray[0] * z, ray[1] * z, z];
    // X = Raᵀ (q − ta); p = Rb X + tb.
    let rat = transpose(&pa.r);
    let x = mv(&rat, [q[0] - pa.t[0], q[1] - pa.t[1], q[2] - pa.t[2]]);
    let pc = mv(&pb.r, x);
    let p = [pc[0] + pb.t[0], pc[1] + pb.t[1], pc[2] + pb.t[2]];
    if p[2] <= 1e-9 {
        return None;
    }
    let iz = 1.0 / p[2];
    let r = [fx * p[0] * iz + k.cx - o.u, fy * p[1] * iz + k.cy - o.v];
    // ∂(u, v)/∂p.
    let jp = [
        [fx * iz, 0.0, -fx * p[0] * iz * iz],
        [0.0, fy * iz, -fy * p[1] * iz * iz],
    ];
    let rbra = mm(&pb.r, &rat);
    let apply = |m: &M3, v: V3| mv(m, v);
    // Frame b (left update): ∂p/∂v = I, ∂p/∂ω = −[p]×.
    let mut jb = [[0.0; 6]; 2];
    // Anchor a: ∂X/∂v_a = −Raᵀ, ∂X/∂ω_a = Raᵀ [q]× (as −Raᵀ (e × q) = Raᵀ (q × e)).
    let mut ja = [[0.0; 6]; 2];
    for c in 0..3 {
        let mut e = [0.0; 3];
        e[c] = 1.0;
        let dpb_v = e;
        let dpb_w = cross(e, p);
        let dpa_v = apply(&rbra, [-e[0], -e[1], -e[2]]);
        let dpa_w = apply(&rbra, cross(q, e));
        for row in 0..2 {
            let d = |v: V3| jp[row][0] * v[0] + jp[row][1] * v[1] + jp[row][2] * v[2];
            jb[row][c] = d(dpb_v);
            jb[row][3 + c] = d(dpb_w);
            ja[row][c] = d(dpa_v);
            ja[row][3 + c] = d(dpa_w);
        }
    }
    // Inverse depth: ∂q/∂ρ = −ray / ρ².
    let dq_rho = ray.map(|v| -v * z * z);
    let dp_rho = apply(&rbra, dq_rho);
    // Log focal: ∂q/∂s = (−q_x, −q_y, 0); ∂(u, v)/∂s also directly (fx·x/z).
    let dp_s = apply(&rbra, [-q[0], -q[1], 0.0]);
    let mut jr = [0.0; 2];
    let mut js = [0.0; 2];
    for row in 0..2 {
        let d = |v: V3| jp[row][0] * v[0] + jp[row][1] * v[1] + jp[row][2] * v[2];
        jr[row] = d(dp_rho);
        js[row] = d(dp_s);
    }
    js[0] += fx * p[0] * iz;
    js[1] += fy * p[1] * iz;
    Some((r, ja, jb, jr, js))
}

fn huber_w(r: f64, k: f64) -> f64 {
    if r <= k { 1.0 } else { k / r }
}

/// Dense symmetric positive definite solve (Cholesky); `None` if not SPD.
fn cholesky_solve(a: &mut [f64], b: &mut [f64], n: usize) -> Option<()> {
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= a[j * n + k] * a[j * n + k];
        }
        if d <= 0.0 || !d.is_finite() {
            return None;
        }
        let d = d.sqrt();
        a[j * n + j] = d;
        let (head, tail) = a.split_at_mut((j + 1) * n);
        let rj = &head[j * n..j * n + j];
        tail.par_chunks_mut(n).for_each(|ri| {
            let mut s = ri[j];
            for k in 0..j {
                s -= ri[k] * rj[k];
            }
            ri[j] = s / d;
        });
    }
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= a[i * n + k] * b[k];
        }
        b[i] = s / a[i * n + i];
    }
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in i + 1..n {
            s -= a[k * n + i] * b[k];
        }
        b[i] = s / a[i * n + i];
    }
    Some(())
}

/// One point's Schur pieces: `Hdd`, `bd` and `Hpd` per parameter block.
struct PointLin {
    hdd: f64,
    bd: f64,
    hpd: Vec<(usize, usize, [f64; 6])>,
}

/// The reduced camera system `H dx = g` (points eliminated) and every
/// point's pieces for the back-substitution.
#[allow(clippy::too_many_arguments)]
fn linear_system(
    k: &Intr,
    s: f64,
    poses: &[Pose],
    points: &[Point],
    obs: &[Obs],
    by_point: &[Vec<usize>],
    huber: f64,
    fdim: usize,
) -> (Vec<PointLin>, Vec<f64>, Vec<f64>) {
    let n = poses.len();
    let dim = 6 * (n - 1) + fdim;
    let off = |f: usize| (f > 0).then(|| 6 * (f - 1));
    let chunk = points
        .len()
        .div_ceil(rayon::current_num_threads().max(1) * 2)
        .max(1);
    let parts: Vec<(Vec<PointLin>, Vec<f64>, Vec<f64>)> = (0..points.len())
        .collect::<Vec<_>>()
        .par_chunks(chunk)
        .map(|ids| {
            let mut h = vec![0.0f64; dim * dim];
            let mut g = vec![0.0f64; dim];
            let mut lins = Vec::with_capacity(ids.len());
            for &pi in ids {
                let pt = &points[pi];
                let mut lin = PointLin {
                    hdd: 0.0,
                    bd: 0.0,
                    hpd: Vec::with_capacity(8),
                };
                for &oi in &by_point[pi] {
                    let o = &obs[oi];
                    let Some((r, ja, jb, jr, js)) =
                        linearize(k, s, &poses[pt.anchor], &poses[o.frame], pt, o)
                    else {
                        continue;
                    };
                    let w = huber_w((r[0] * r[0] + r[1] * r[1]).sqrt(), huber);
                    let mut bl: [(usize, usize, [[f64; 6]; 2]); 3] = [(0, 0, [[0.0; 6]; 2]); 3];
                    let mut nb = 0;
                    if let Some(o_a) = off(pt.anchor) {
                        bl[nb] = (o_a, 6, ja);
                        nb += 1;
                    }
                    if let Some(o_b) = off(o.frame) {
                        bl[nb] = (o_b, 6, jb);
                        nb += 1;
                    }
                    if fdim == 1 {
                        let mut j = [[0.0; 6]; 2];
                        j[0][0] = js[0];
                        j[1][0] = js[1];
                        bl[nb] = (dim - 1, 1, j);
                        nb += 1;
                    }
                    for &(oa, la, ref j1) in &bl[..nb] {
                        let e = match lin.hpd.iter().position(|x| x.0 == oa) {
                            Some(e) => e,
                            None => {
                                lin.hpd.push((oa, la, [0.0; 6]));
                                lin.hpd.len() - 1
                            }
                        };
                        for x in 0..la {
                            g[oa + x] -= w * (j1[0][x] * r[0] + j1[1][x] * r[1]);
                            lin.hpd[e].2[x] += w * (j1[0][x] * jr[0] + j1[1][x] * jr[1]);
                        }
                        for &(ob, lb, ref j2) in &bl[..nb] {
                            for x in 0..la {
                                let row = &mut h[(oa + x) * dim + ob..(oa + x) * dim + ob + lb];
                                for (y, v) in row.iter_mut().enumerate() {
                                    *v += w * (j1[0][x] * j2[0][y] + j1[1][x] * j2[1][y]);
                                }
                            }
                        }
                    }
                    lin.hdd += w * (jr[0] * jr[0] + jr[1] * jr[1]);
                    lin.bd -= w * (jr[0] * r[0] + jr[1] * r[1]);
                }
                // Schur: H −= Hpd Hdd⁻¹ Hdp; g −= Hpd Hdd⁻¹ bd.
                let ihdd = 1.0 / (lin.hdd + 1e-12);
                for &(oa, la, ref va) in &lin.hpd {
                    for x in 0..la {
                        g[oa + x] -= va[x] * lin.bd * ihdd;
                        let f = va[x] * ihdd;
                        for &(ob, lb, ref vb) in &lin.hpd {
                            let row = &mut h[(oa + x) * dim + ob..(oa + x) * dim + ob + lb];
                            for (y, v) in row.iter_mut().enumerate() {
                                *v -= f * vb[y];
                            }
                        }
                    }
                }
                lins.push(lin);
            }
            (lins, h, g)
        })
        .collect();
    let mut lins = Vec::with_capacity(points.len());
    let mut h = vec![0.0f64; dim * dim];
    let mut g = vec![0.0f64; dim];
    for (l, hh, gg) in parts {
        lins.extend(l);
        h.par_iter_mut().zip(&hh).for_each(|(a, b)| *a += b);
        g.iter_mut().zip(&gg).for_each(|(a, b)| *a += b);
    }
    (lins, h, g)
}

/// Robust cost, inliers and per-observation residual norms.
fn evaluate(
    k: &Intr,
    s: f64,
    poses: &[Pose],
    points: &[Point],
    obs: &[Obs],
    huber: f64,
) -> (f64, usize, Vec<f64>) {
    let norms: Vec<f64> = obs
        .par_iter()
        .map(|o| {
            let pt = &points[o.point];
            linearize(k, s, &poses[pt.anchor], &poses[o.frame], pt, o)
                .map_or(f64::INFINITY, |(r, ..)| (r[0] * r[0] + r[1] * r[1]).sqrt())
        })
        .collect();
    let mut cost = 0.0;
    let mut inl = 0;
    for &n in &norms {
        if n.is_finite() {
            cost += if n <= huber {
                0.5 * n * n
            } else {
                huber * (n - 0.5 * huber)
            };
            if n < 3.0 * huber {
                inl += 1;
            }
        } else {
            cost += 1e6;
        }
    }
    (cost, inl, norms)
}

/// A weak Gaussian prior that holds every pose near its start (the
/// engine's, globally consistent): tracks between overlapping neighbours
/// fix the local error but leave slow drift along long chains free (T&T
/// train drifted from 0.65° to 1.8° without it).
struct Prior {
    poses: Vec<Pose>,
    /// Inverse variances: rotation (1/rad²) and translation (1/unit²).
    wr: f64,
    wt: f64,
}

/// `log` of a rotation matrix (axis × angle).
fn so3_log(r: &M3) -> V3 {
    let c = ((r[0][0] + r[1][1] + r[2][2] - 1.0) / 2.0).clamp(-1.0, 1.0);
    let th = c.acos();
    let v = [r[2][1] - r[1][2], r[0][2] - r[2][0], r[1][0] - r[0][1]];
    let k = if th < 1e-6 {
        0.5
    } else {
        th / (2.0 * th.sin())
    };
    v.map(|x| x * k)
}

impl Prior {
    /// The left-perturbation offset `[v, ω]` of `cur` from frame `f`'s start.
    fn delta(&self, f: usize, cur: &Pose) -> [f64; 6] {
        let r0 = &self.poses[f];
        let rd = mm(&cur.r, &transpose(&r0.r));
        let w = so3_log(&rd);
        let t = mv(&rd, r0.t);
        [
            cur.t[0] - t[0],
            cur.t[1] - t[1],
            cur.t[2] - t[2],
            w[0],
            w[1],
            w[2],
        ]
    }

    fn cost(&self, poses: &[Pose]) -> f64 {
        poses
            .iter()
            .enumerate()
            .map(|(f, p)| {
                let d = self.delta(f, p);
                0.5 * (self.wt * (d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
                    + self.wr * (d[3] * d[3] + d[4] * d[4] + d[5] * d[5]))
            })
            .sum()
    }
}

/// Levenberg-Marquardt on the reduced camera system; returns the
/// iterations run.
#[allow(clippy::too_many_arguments)]
fn lm(
    k: &Intr,
    poses: &mut Vec<Pose>,
    points: &mut Vec<Point>,
    s: &mut f64,
    obs: &[Obs],
    iters: u32,
    huber: f64,
    fdim: usize,
    prior: Option<&Prior>,
) -> u32 {
    let n = poses.len();
    let pcost = |p: &[Pose]| prior.map_or(0.0, |q| q.cost(p));
    let dim = 6 * (n - 1) + fdim;
    let off = |f: usize| (f > 0).then(|| 6 * (f - 1));
    let mut by_point: Vec<Vec<usize>> = vec![Vec::new(); points.len()];
    for (i, o) in obs.iter().enumerate() {
        by_point[o.point].push(i);
    }
    let (mut cost, ..) = evaluate(k, *s, poses, points, obs, huber);
    cost += pcost(poses);
    let mut iters_done = 0;
    let mut lambda = 1e-4;
    for _ in 0..iters {
        iters_done += 1;
        let (lins, mut h, mut g) = linear_system(k, *s, poses, points, obs, &by_point, huber, fdim);
        if let Some(q) = prior {
            for (f, pose) in poses.iter().enumerate().skip(1) {
                let o = 6 * (f - 1);
                let d = q.delta(f, pose);
                for i in 0..6 {
                    let w = if i < 3 { q.wt } else { q.wr };
                    h[(o + i) * dim + o + i] += w;
                    g[o + i] -= w * d[i];
                }
            }
        }
        let mut improved = false;
        for _ in 0..6 {
            let mut a = h.clone();
            for i in 0..dim {
                a[i * dim + i] += lambda * h[i * dim + i].max(1e-9) + 1e-9;
            }
            let mut dx = g.clone();
            if cholesky_solve(&mut a, &mut dx, dim).is_none() {
                lambda *= 10.0;
                continue;
            }
            let new_poses: Vec<Pose> = (0..n)
                .map(|f| off(f).map_or(poses[f], |o| poses[f].update(&dx[o..o + 6])))
                .collect();
            let new_s = if fdim == 1 { *s + dx[dim - 1] } else { *s };
            // Back-substitute the inverse depths.
            let new_points: Vec<Point> = points
                .par_iter()
                .zip(&lins)
                .map(|(pt, c)| {
                    let mut num = c.bd;
                    for &(o, len, ref v) in &c.hpd {
                        for x in 0..len {
                            num -= v[x] * dx[o + x];
                        }
                    }
                    let d = num / (c.hdd + 1e-12);
                    let rho = pt.rho + d;
                    Point {
                        rho: if rho > 1e-6 { rho } else { pt.rho },
                        ..*pt
                    }
                })
                .collect();
            let (c2, ..) = evaluate(k, new_s, &new_poses, &new_points, obs, huber);
            let c2 = c2 + pcost(&new_poses);
            if c2 < cost {
                cost = c2;
                *poses = new_poses;
                *points = new_points;
                *s = new_s;
                lambda = (lambda * 0.3).max(1e-7);
                improved = true;
                break;
            }
            lambda *= 10.0;
        }
        if !improved {
            break;
        }
    }
    iters_done
}

/// Runs the BA on `frames` (indices into `p`; `frames[0]` is the gauge)
/// with `images[k]` the trainer image of `frames[k]`. Writes the refined
/// extrinsics and the shared focal into `p` and recomputes its world points.
pub fn bundle_adjust(
    p: &mut Posed,
    frames: &[usize],
    images: &[&Rgb8Image],
    cfg: &BaConfig,
) -> (BaReport, InitPoints) {
    let t0 = std::time::Instant::now();
    let n = frames.len();
    let mut rep = BaReport {
        frames: n,
        focal_scale: 1.0,
        ..Default::default()
    };
    if cfg.iters == 0 || n < 2 || images.len() != n {
        return (rep, InitPoints::default());
    }
    let ds = cfg.downscale.max(1) as usize;
    let pyr: Vec<Vec<Gray>> = images
        .par_iter()
        .map(|im| pyramid(Gray::from_rgb(im, ds), cfg.levels as usize))
        .collect();
    let (gw, gh) = (pyr[0][0].w as f64, pyr[0][0].h as f64);
    let (ew, eh) = (p.cams.width as f64, p.cams.height as f64);
    let (sx, sy) = (gw / ew, gh / eh);
    let focal = p.shared_focal(frames, gw as u32, gh as u32);
    let k = Intr {
        fx: focal.0,
        fy: focal.1,
        cx: gw / 2.0,
        cy: gh / 2.0,
    };
    let cams: Vec<Cam> = frames.iter().map(|&i| Cam::of(p, i)).collect();
    let thr = conf_threshold(p, frames, 0.3);
    let samples: Vec<Vec<(f64, f64, f64)>> = frames
        .par_iter()
        .map(|&i| grid_samples(p, i, thr, 2000))
        .collect();
    // Half the best-overlapping frames, half spread over a 4× longer
    // overlap list: the wider baselines close loops on 360° captures (T&T
    // train: 4 best only drifted 0.90° → 1.07°, 10 best reached 0.61°).
    let want = cfg.neighbours as usize;
    let nb: Vec<Vec<usize>> = neighbours(p, frames, &cams, &samples, 4 * want, 0.05)
        .into_iter()
        .map(|list| spread_pick(&list, want))
        .collect();

    // Keypoints with engine depth, then tracks.
    let hw = cfg.half_win as i32;
    let per_frame: Vec<(Vec<Point>, Vec<Obs>, usize)> = (0..n)
        .into_par_iter()
        .map(|a| {
            let dv = depth_view(p, frames[a]);
            let ca = &cams[a];
            let mut pts = Vec::new();
            let mut obs = Vec::new();
            let mut tried = 0;
            for (x, y) in corners(&pyr[a][0], cfg.keypoints as usize) {
                // Working pixel centre → engine pixel.
                let (u, v) = (x as f64 + 0.5, y as f64 + 0.5);
                let Some(z) = dv.sample(u / sx, v / sy) else {
                    continue;
                };
                let xw = ca.world_of(ca.unproject(u / sx, v / sy, z));
                let mut mine = Vec::new();
                for &b in &nb[a] {
                    let cb = &cams[b];
                    let Some((ub, vb)) = cb.project(cb.cam_of(xw)) else {
                        continue;
                    };
                    let guess = ((ub * sx - 0.5) as f32, (vb * sy - 0.5) as f32);
                    tried += 1;
                    let Some(q) = klt(&pyr[a], &pyr[b], (x, y), guess, hw) else {
                        continue;
                    };
                    let Some(back) = klt(&pyr[b], &pyr[a], q, (x, y), hw) else {
                        continue;
                    };
                    let e = ((back.0 - x).powi(2) + (back.1 - y).powi(2)).sqrt() as f64;
                    if e > cfg.fb_px {
                        continue;
                    }
                    mine.push((b, q.0 as f64 + 0.5, q.1 as f64 + 0.5));
                }
                if mine.is_empty() {
                    continue;
                }
                let id = pts.len();
                pts.push(Point {
                    anchor: a,
                    u,
                    v,
                    rho: 1.0 / z,
                });
                for (b, ub, vb) in mine {
                    obs.push(Obs {
                        point: id,
                        frame: b,
                        u: ub,
                        v: vb,
                    });
                }
            }
            (pts, obs, tried)
        })
        .collect();
    let mut points = Vec::new();
    let mut obs = Vec::new();
    for (pts, o, tried) in per_frame {
        let base = points.len();
        points.extend(pts);
        obs.extend(o.into_iter().map(|mut x| {
            x.point += base;
            x
        }));
        rep.tracks_tried += tried;
    }
    rep.tracks_kept = obs.len();
    rep.points = points.len();
    rep.observations = obs.len();
    rep.track_s = t0.elapsed().as_secs_f64();
    if obs.len() < 100 {
        return (rep, InitPoints::default());
    }

    // LM on the reduced camera system.
    let t1 = std::time::Instant::now();
    let mut poses: Vec<Pose> = cams.iter().map(|c| Pose { r: c.r, t: c.t }).collect();
    let start = poses.clone();
    let mut s = 0.0f64;
    let fdim = usize::from(cfg.focal);
    let huber = cfg.huber_px;
    let (_, inl, norms) = evaluate(&k, s, &poses, &points, &obs, huber);
    let finite: Vec<f64> = norms.iter().copied().filter(|v| v.is_finite()).collect();
    rep.rms_before =
        (finite.iter().map(|v| v * v).sum::<f64>() / finite.len().max(1) as f64).sqrt();
    rep.inliers_before = inl as f64 / obs.len() as f64;
    let prior = (cfg.prior_deg > 0.0).then(|| {
        // Translation σ: the same fraction of the camera spread as the
        // rotation σ is of a radian.
        let c: Vec<V3> = cams.iter().map(Cam::center).collect();
        let m: V3 = std::array::from_fn(|j| c.iter().map(|x| x[j]).sum::<f64>() / c.len() as f64);
        let mut d: Vec<f64> = c
            .iter()
            .map(|x| ((0..3).map(|j| (x[j] - m[j]).powi(2)).sum::<f64>()).sqrt())
            .collect();
        d.sort_by(f64::total_cmp);
        let spread = d[d.len() / 2].max(1e-9);
        let sr = cfg.prior_deg.to_radians();
        Prior {
            poses: poses.clone(),
            wr: 1.0 / (sr * sr),
            wt: 1.0 / (sr * spread).powi(2),
        }
    });
    rep.iterations = lm(
        &k,
        &mut poses,
        &mut points,
        &mut s,
        &obs,
        cfg.iters,
        huber,
        fdim,
        prior.as_ref(),
    );
    if cfg.reject_px > 0.0 {
        // Second round without the gross outliers.
        let (_, _, norms) = evaluate(&k, s, &poses, &points, &obs, huber);
        obs = obs
            .into_iter()
            .zip(norms)
            .filter(|(_, r)| *r < cfg.reject_px)
            .map(|(o, _)| o)
            .collect();
        rep.observations_kept = obs.len();
        rep.iterations += lm(
            &k,
            &mut poses,
            &mut points,
            &mut s,
            &obs,
            cfg.iters.div_ceil(2),
            huber,
            fdim,
            prior.as_ref(),
        );
    }
    let (_, inl, norms) = evaluate(&k, s, &poses, &points, &obs, huber);
    let finite: Vec<f64> = norms.iter().copied().filter(|v| v.is_finite()).collect();
    rep.rms_after = (finite.iter().map(|v| v * v).sum::<f64>() / finite.len().max(1) as f64).sqrt();
    rep.inliers_after = inl as f64 / obs.len() as f64;
    rep.focal_scale = s.exp();
    let mut changes: Vec<f64> = start
        .iter()
        .zip(&poses)
        .map(|(a, b)| {
            let d = mm(&b.r, &transpose(&a.r));
            ((d[0][0] + d[1][1] + d[2][2] - 1.0) / 2.0)
                .clamp(-1.0, 1.0)
                .acos()
                .to_degrees()
        })
        .collect();
    changes.sort_by(f64::total_cmp);
    rep.median_rot_change_deg = changes[changes.len() / 2];
    for (&i, q) in frames.iter().zip(&poses) {
        let e = &mut p.cams.extrinsics[i];
        for r in 0..3 {
            e[4 * r] = q.r[r][0] as f32;
            e[4 * r + 1] = q.r[r][1] as f32;
            e[4 * r + 2] = q.r[r][2] as f32;
            e[4 * r + 3] = q.t[r] as f32;
        }
    }
    let f = s.exp() as f32;
    for kk in p.cams.intrinsics.iter_mut() {
        kk[0] *= f;
        kk[4] *= f;
    }
    let all: Vec<usize> = (0..p.frames()).collect();
    p.reproject(&all);
    // Triangulated tracks (seen by at least two other frames) as SfM-style
    // init points, coloured from their anchor image.
    let mut seen = vec![0u32; points.len()];
    obs.iter().for_each(|o| seen[o.point] += 1);
    let mut cloud = InitPoints::default();
    let (fx, fy) = (k.fx * s.exp(), k.fy * s.exp());
    for (pt, &c) in points.iter().zip(&seen) {
        if c < 2 {
            continue;
        }
        let pa = &poses[pt.anchor];
        let z = 1.0 / pt.rho;
        let q = [(pt.u - k.cx) / fx * z, (pt.v - k.cy) / fy * z, z];
        let x = mv(
            &transpose(&pa.r),
            [q[0] - pa.t[0], q[1] - pa.t[1], q[2] - pa.t[2]],
        );
        let img = images[pt.anchor];
        let (px, py) = (
            ((pt.u * ds as f64) as usize).min(img.width as usize - 1),
            ((pt.v * ds as f64) as usize).min(img.height as usize - 1),
        );
        let o = (py * img.width as usize + px) * 3;
        cloud.xyz.push(x.map(|v| v as f32));
        cloud
            .rgb
            .push([img.data[o], img.data[o + 1], img.data[o + 2]]);
    }
    rep.solve_s = t1.elapsed().as_secs_f64();
    (rep, cloud)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    fn numeric(
        k: &Intr,
        s: f64,
        pa: &Pose,
        pb: &Pose,
        pt: &Point,
        o: &Obs,
        which: usize,
        c: usize,
    ) -> [f64; 2] {
        let e = 1e-6;
        let f = |sign: f64| {
            let mut xi = [0.0; 6];
            let (mut pa2, mut pb2, mut pt2, mut s2) = (*pa, *pb, *pt, s);
            match which {
                0 => {
                    xi[c] = sign * e;
                    pa2 = pa.update(&xi);
                }
                1 => {
                    xi[c] = sign * e;
                    pb2 = pb.update(&xi);
                }
                2 => pt2.rho += sign * e,
                _ => s2 += sign * e,
            }
            linearize(k, s2, &pa2, &pb2, &pt2, o).unwrap().0
        };
        let (p, m) = (f(1.0), f(-1.0));
        [(p[0] - m[0]) / (2.0 * e), (p[1] - m[1]) / (2.0 * e)]
    }

    #[test]
    fn jacobians_match_finite_differences() {
        let k = Intr {
            fx: 500.0,
            fy: 480.0,
            cx: 320.0,
            cy: 240.0,
        };
        let pa = Pose {
            r: so3_exp([0.1, -0.2, 0.05]),
            t: [0.3, -0.1, 0.2],
        };
        let pb = Pose {
            r: so3_exp([-0.05, 0.15, 0.1]),
            t: [-0.4, 0.2, 0.1],
        };
        let pt = Point {
            anchor: 0,
            u: 300.0,
            v: 200.0,
            rho: 0.25,
        };
        let o = Obs {
            point: 0,
            frame: 1,
            u: 310.0,
            v: 190.0,
        };
        let s = 0.02;
        let (_, ja, jb, jr, js) = linearize(&k, s, &pa, &pb, &pt, &o).unwrap();
        for c in 0..6 {
            let na = numeric(&k, s, &pa, &pb, &pt, &o, 0, c);
            let nb = numeric(&k, s, &pa, &pb, &pt, &o, 1, c);
            for row in 0..2 {
                assert!(
                    (na[row] - ja[row][c]).abs() < 1e-3 * (1.0 + na[row].abs()),
                    "a {c}"
                );
                assert!(
                    (nb[row] - jb[row][c]).abs() < 1e-3 * (1.0 + nb[row].abs()),
                    "b {c}"
                );
            }
        }
        let nr = numeric(&k, s, &pa, &pb, &pt, &o, 2, 0);
        let ns = numeric(&k, s, &pa, &pb, &pt, &o, 3, 0);
        for row in 0..2 {
            assert!((nr[row] - jr[row]).abs() < 1e-3 * (1.0 + nr[row].abs()));
            assert!((ns[row] - js[row]).abs() < 1e-3 * (1.0 + ns[row].abs()));
        }
    }

    #[test]
    fn lm_recovers_perturbed_poses_and_focal() {
        let k = Intr {
            fx: 400.0,
            fy: 400.0,
            cx: 320.0,
            cy: 240.0,
        };
        // Cameras on an arc looking at the origin region (+z forward).
        let truth: Vec<Pose> = (0..6)
            .map(|i| {
                let a = 0.08 * i as f64;
                let r = so3_exp([0.0, -a, 0.0]);
                let c = [4.0 * a.sin(), 0.0, -4.0 * a.cos()];
                let t = mv(&r, c).map(|v| -v);
                Pose { r, t }
            })
            .collect();
        let mut rng = 12345u64;
        let mut rnd = || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        let mut points = Vec::new();
        let mut obs = Vec::new();
        for a in 0..6 {
            for _ in 0..80 {
                let (u, v) = (320.0 + 500.0 * rnd(), 240.0 + 380.0 * rnd());
                let z = 3.5 + 1.5 * rnd();
                let pt = Point {
                    anchor: a,
                    u,
                    v,
                    rho: 1.0 / z,
                };
                let id = points.len();
                for b in 0..6 {
                    if b == a {
                        continue;
                    }
                    let o = Obs {
                        point: id,
                        frame: b,
                        u: 0.0,
                        v: 0.0,
                    };
                    let (r, ..) = linearize(&k, 0.0, &truth[a], &truth[b], &pt, &o).unwrap();
                    obs.push(Obs {
                        u: r[0],
                        v: r[1],
                        ..o
                    });
                }
                // Depth prior is off: start 10 % away.
                points.push(Point {
                    rho: pt.rho * (1.0 + 0.2 * rnd()),
                    ..pt
                });
            }
        }
        let mut poses: Vec<Pose> = truth
            .iter()
            .enumerate()
            .map(|(i, p)| {
                if i == 0 {
                    *p
                } else {
                    p.update(&[
                        0.02 * rnd(),
                        0.02 * rnd(),
                        0.02 * rnd(),
                        0.01 * rnd(),
                        0.01 * rnd(),
                        0.01 * rnd(),
                    ])
                }
            })
            .collect();
        let mut s = 0.01;
        lm(&k, &mut poses, &mut points, &mut s, &obs, 30, 1.0, 1, None);
        let (cost, ..) = evaluate(&k, s, &poses, &points, &obs, 1.0);
        assert!(cost / (obs.len() as f64) < 1e-6, "cost {cost}");
        assert!(s.abs() < 1e-4, "focal {s}");
        for (p, t) in poses.iter().zip(&truth) {
            let d = mm(&p.r, &transpose(&t.r));
            let ang = ((d[0][0] + d[1][1] + d[2][2] - 1.0) / 2.0)
                .clamp(-1.0, 1.0)
                .acos();
            assert!(ang.to_degrees() < 1e-3, "{}", ang.to_degrees());
        }
    }

    #[test]
    fn so3_log_inverts_exp() {
        let w = [0.1, -0.25, 0.4];
        let l = so3_log(&so3_exp(w));
        for i in 0..3 {
            assert!((l[i] - w[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn spread_pick_keeps_the_best_and_spreads_the_rest() {
        let list: Vec<usize> = (0..16).collect();
        assert_eq!(spread_pick(&list, 4), vec![0, 1, 5, 12]);
        assert_eq!(spread_pick(&list[..3], 4), vec![0, 1, 2]);
    }

    #[test]
    fn cholesky_solves() {
        let mut a = vec![4.0, 2.0, 0.6, 2.0, 5.0, 1.0, 0.6, 1.0, 3.0];
        let x = [1.0, -2.0, 0.5];
        let mut b: Vec<f64> = (0..3)
            .map(|i| (0..3).map(|j| a[i * 3 + j] * x[j]).sum())
            .collect();
        cholesky_solve(&mut a, &mut b, 3).unwrap();
        for i in 0..3 {
            assert!((b[i] - x[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn klt_finds_a_shift() {
        let (w, h) = (64usize, 64usize);
        let f = |x: f32, y: f32| ((x * 0.3).sin() + (y * 0.21).cos() + (x * y * 0.01).sin()) * 0.3;
        let a = Gray {
            w,
            h,
            p: (0..w * h)
                .map(|i| f((i % w) as f32, (i / w) as f32))
                .collect(),
        };
        let b = Gray {
            w,
            h,
            p: (0..w * h)
                .map(|i| f((i % w) as f32 - 2.4, (i / w) as f32 + 1.3))
                .collect(),
        };
        let (pa, pb) = (pyramid(a, 2), pyramid(b, 2));
        let q = klt(&pa, &pb, (30.0, 30.0), (31.0, 29.0), 6).unwrap();
        assert!(
            (q.0 - 32.4).abs() < 0.05 && (q.1 - 28.7).abs() < 0.05,
            "{q:?}"
        );
    }
}
