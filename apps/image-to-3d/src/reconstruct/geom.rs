//! Multi-view geometry on the engine's dense depth, with no feature matcher:
//! which frames overlap, a geometric pose refinement that makes every
//! frame's depth agree with its neighbours' (ICP-style, block Gauss-Newton
//! on the relative depth residual, per-frame depth scale optional), and
//! depth-consistency filtering and fusion of the init points (MVS-style
//! "seen consistently by ≥ k views"). CPU, rayon. Owner: mvp-quality.
//!
//! Conventions as [`Posed`]: world-to-camera `[R | t]`, OpenCV axes, pixel
//! centres at `+0.5`, depth = camera z.

use rayon::prelude::*;
use serde::Serialize;

use super::scene::Posed;
use crate::io::scene::InitPoints;

type M3 = [[f64; 3]; 3];
type V3 = [f64; 3];

pub(super) fn mv(m: &M3, v: V3) -> V3 {
    std::array::from_fn(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2])
}

fn mtv(m: &M3, v: V3) -> V3 {
    std::array::from_fn(|i| m[0][i] * v[0] + m[1][i] * v[1] + m[2][i] * v[2])
}

pub(super) fn mm(a: &M3, b: &M3) -> M3 {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

/// Rodrigues: `exp([φ]×)`.
pub(super) fn so3_exp(p: V3) -> M3 {
    let th2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
    let th = th2.sqrt();
    let (a, b) = if th < 1e-9 {
        (1.0, 0.5)
    } else {
        (th.sin() / th, (1.0 - th.cos()) / th2)
    };
    let k = [[0.0, -p[2], p[1]], [p[2], 0.0, -p[0]], [-p[1], p[0], 0.0]];
    let k2 = mm(&k, &k);
    std::array::from_fn(|i| {
        std::array::from_fn(|j| (i == j) as u8 as f64 + a * k[i][j] + b * k2[i][j])
    })
}

/// One frame's camera at the engine resolution, plus its depth scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cam {
    pub r: M3,
    pub t: V3,
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    /// Multiplies the frame's engine depth.
    pub scale: f64,
}

impl Cam {
    pub fn of(p: &Posed, i: usize) -> Self {
        let (r, t) = p.pose(i);
        let k = &p.cams.intrinsics[i];
        Self {
            r: r.map(|row| row.map(f64::from)),
            t: t.map(f64::from),
            fx: k[0] as f64,
            fy: k[4] as f64,
            cx: k[2] as f64,
            cy: k[5] as f64,
            scale: 1.0,
        }
    }

    pub(super) fn cam_of(&self, x: V3) -> V3 {
        let c = mv(&self.r, x);
        [c[0] + self.t[0], c[1] + self.t[1], c[2] + self.t[2]]
    }

    pub(super) fn world_of(&self, c: V3) -> V3 {
        mtv(
            &self.r,
            [c[0] - self.t[0], c[1] - self.t[1], c[2] - self.t[2]],
        )
    }

    pub(super) fn project(&self, c: V3) -> Option<(f64, f64)> {
        (c[2] > 1e-9).then(|| {
            (
                self.fx * c[0] / c[2] + self.cx,
                self.fy * c[1] / c[2] + self.cy,
            )
        })
    }

    pub(super) fn unproject(&self, u: f64, v: f64, z: f64) -> V3 {
        [(u - self.cx) / self.fx * z, (v - self.cy) / self.fy * z, z]
    }

    /// `exp(ξ) · T` (left update of the world-to-camera transform) and the
    /// scale times `exp(ξ[6])`.
    fn perturbed(&self, xi: &[f64; 7]) -> Self {
        let rd = so3_exp([xi[3], xi[4], xi[5]]);
        let t = mv(&rd, self.t);
        Self {
            r: mm(&rd, &self.r),
            t: [t[0] + xi[0], t[1] + xi[1], t[2] + xi[2]],
            scale: self.scale * xi[6].exp(),
            ..*self
        }
    }

    pub fn center(&self) -> V3 {
        self.world_of([0.0; 3])
    }
}

/// One frame's engine depth, `w × h`.
pub(super) struct DepthView<'a> {
    d: &'a [f32],
    w: usize,
    h: usize,
}

impl DepthView<'_> {
    /// Bilinear depth at pixel coordinates `(u, v)` (centres at +0.5); `None`
    /// off the image, at invalid depth or across a discontinuity (> 5 %).
    pub(super) fn sample(&self, u: f64, v: f64) -> Option<f64> {
        let (x, y) = (u - 0.5, v - 0.5);
        if !(x >= 0.0 && y >= 0.0) {
            return None;
        }
        let (x0, y0) = (x as usize, y as usize);
        if x0 + 1 >= self.w || y0 + 1 >= self.h {
            return None;
        }
        let (fx, fy) = (x - x0 as f64, y - y0 as f64);
        let at = |xx: usize, yy: usize| self.d[yy * self.w + xx] as f64;
        let c = [
            at(x0, y0),
            at(x0 + 1, y0),
            at(x0, y0 + 1),
            at(x0 + 1, y0 + 1),
        ];
        let (mut lo, mut hi) = (f64::INFINITY, 0.0f64);
        for &z in &c {
            if !(z.is_finite() && z > 0.0) {
                return None;
            }
            lo = lo.min(z);
            hi = hi.max(z);
        }
        if hi > lo * 1.05 {
            return None;
        }
        Some((c[0] * (1.0 - fx) + c[1] * fx) * (1.0 - fy) + (c[2] * (1.0 - fx) + c[3] * fx) * fy)
    }
}

/// Confidence threshold at `percentile` (0-1) over `frames`.
pub(super) fn conf_threshold(p: &Posed, frames: &[usize], percentile: f32) -> f32 {
    let px = p.cams.width * p.cams.height;
    let mut c: Vec<f32> = frames
        .iter()
        .flat_map(|&i| p.dense.conf[i * px..(i + 1) * px].iter().copied())
        .filter(|c| c.is_finite())
        .step_by(7)
        .collect();
    if c.is_empty() {
        return f32::INFINITY;
    }
    c.sort_by(f32::total_cmp);
    c[((percentile.clamp(0.0, 1.0) * (c.len() - 1) as f32) as usize).min(c.len() - 1)]
}

/// About `n` confident pixels of frame `i` on a regular grid: (u, v, depth).
pub(super) fn grid_samples(p: &Posed, i: usize, thr: f32, n: usize) -> Vec<(f64, f64, f64)> {
    let (w, h) = (p.cams.width, p.cams.height);
    let px = w * h;
    let step = ((px as f64 / n.max(1) as f64).sqrt().floor() as usize).max(1);
    let mut out = Vec::new();
    for y in (step / 2..h).step_by(step) {
        for x in (step / 2..w).step_by(step) {
            let k = i * px + y * w + x;
            let (z, c) = (p.dense.depth[k], p.dense.conf[k]);
            if z.is_finite() && z > 0.0 && c >= thr {
                out.push((x as f64 + 0.5, y as f64 + 0.5, z as f64));
            }
        }
    }
    out
}

pub(super) fn depth_view(p: &Posed, i: usize) -> DepthView<'_> {
    let (w, h) = (p.cams.width, p.cams.height);
    DepthView {
        d: &p.dense.depth[i * w * h..(i + 1) * w * h],
        w,
        h,
    }
}

/// For every frame of `frames`, up to `k` others (indices into `frames`)
/// that see most of its sampled surface consistently (relative depth
/// within `tau`), best first.
pub(super) fn neighbours(
    p: &Posed,
    frames: &[usize],
    cams: &[Cam],
    samples: &[Vec<(f64, f64, f64)>],
    k: usize,
    tau: f64,
) -> Vec<Vec<usize>> {
    let n = frames.len();
    (0..n)
        .into_par_iter()
        .map(|a| {
            let ca = &cams[a];
            let pts: Vec<V3> = samples[a]
                .iter()
                .step_by(4)
                .map(|&(u, v, z)| ca.world_of(ca.unproject(u, v, z * ca.scale)))
                .collect();
            let mut score: Vec<(usize, usize)> = (0..n)
                .filter(|&b| b != a)
                .map(|b| {
                    let cb = &cams[b];
                    let dv = depth_view(p, frames[b]);
                    let hits = pts
                        .iter()
                        .filter(|&&x| {
                            let c = cb.cam_of(x);
                            cb.project(c)
                                .and_then(|(u, v)| dv.sample(u, v))
                                .is_some_and(|d| (c[2] - d * cb.scale).abs() < tau * d * cb.scale)
                        })
                        .count();
                    (b, hits)
                })
                .filter(|&(_, h)| h * 20 >= pts.len().max(1))
                .collect();
            score.sort_by(|x, y| y.1.cmp(&x.1).then(x.0.cmp(&y.0)));
            score.truncate(k);
            score.into_iter().map(|(b, _)| b).collect()
        })
        .collect()
}

/// Geometric pose refinement settings.
#[derive(Clone, Debug, PartialEq)]
pub struct GeoRefine {
    /// Outer (Jacobi) iterations; 0 = off.
    pub iters: u32,
    /// Sampled pixels per frame.
    pub samples: usize,
    pub neighbours: usize,
    /// Robust (Huber) width of the relative depth residual.
    pub huber: f64,
    /// Residuals above this (relative) are outliers; it shrinks to
    /// `huber × 3` over the run.
    pub reject: f64,
    /// Also refine one depth scale per frame.
    pub scale: bool,
    /// Confidence percentile below which pixels are ignored.
    pub conf_drop: f32,
}

impl Default for GeoRefine {
    fn default() -> Self {
        Self {
            iters: 0,
            samples: 3000,
            neighbours: 8,
            huber: 0.01,
            reject: 0.1,
            scale: true,
            conf_drop: 0.3,
        }
    }
}

/// What the refinement did (for logs and the result file).
#[derive(Clone, Debug, Default, Serialize)]
pub struct GeoReport {
    pub frames: usize,
    pub residuals: usize,
    /// Robust mean |relative depth residual| over inliers, first and last pass.
    pub rms_before: f64,
    pub rms_after: f64,
    pub inliers_before: f64,
    pub inliers_after: f64,
    /// Median absolute rotation change in degrees.
    pub median_rot_change_deg: f64,
    pub scale_min: f64,
    pub scale_max: f64,
    pub seconds: f64,
}

/// Sum of the robust cost of frame `a`'s residuals given its candidate
/// camera `ca` (everything else fixed), and the Gauss-Newton system.
#[allow(clippy::too_many_arguments)]
fn residuals_of(
    p: &Posed,
    frames: &[usize],
    cams: &[Cam],
    samples: &[Vec<(f64, f64, f64)>],
    nb: &[Vec<usize>],
    rev: &[Vec<usize>],
    a: usize,
    ca: &Cam,
    reject: f64,
    out: &mut Vec<f64>,
) {
    out.clear();
    // A: a's samples, seen by its neighbours.
    for &b in &nb[a] {
        let cb = &cams[b];
        let dv = depth_view(p, frames[b]);
        for &(u, v, z) in &samples[a] {
            let x = ca.world_of(ca.unproject(u, v, z * ca.scale));
            let c = cb.cam_of(x);
            out.push(
                cb.project(c)
                    .and_then(|(pu, pv)| dv.sample(pu, pv))
                    .map(|d| (c[2] - d * cb.scale) / (d * cb.scale))
                    .filter(|r| r.abs() < reject)
                    .unwrap_or(f64::NAN),
            );
        }
    }
    // B: the frames that list a as a neighbour, seen by a.
    let dv = depth_view(p, frames[a]);
    for &b in &rev[a] {
        let cb = &cams[b];
        for &(u, v, z) in &samples[b] {
            let x = cb.world_of(cb.unproject(u, v, z * cb.scale));
            let c = ca.cam_of(x);
            out.push(
                ca.project(c)
                    .and_then(|(pu, pv)| dv.sample(pu, pv))
                    .map(|d| (c[2] - d * ca.scale) / (d * ca.scale))
                    .filter(|r| r.abs() < reject)
                    .unwrap_or(f64::NAN),
            );
        }
    }
}

fn huber_w(r: f64, k: f64) -> f64 {
    if r.abs() <= k { 1.0 } else { k / r.abs() }
}

/// Solves `(A + λ diag A) x = b` (7×7, symmetric positive semi-definite).
#[allow(clippy::needless_range_loop)]
fn solve7(a: &[[f64; 7]; 7], b: &[f64; 7], lambda: f64, dims: usize) -> Option<[f64; 7]> {
    let mut m = [[0.0; 8]; 7];
    for i in 0..dims {
        for j in 0..dims {
            m[i][j] = a[i][j];
        }
        m[i][i] += lambda * a[i][i].max(1e-12) + 1e-12;
        m[i][7] = b[i];
    }
    for c in 0..dims {
        let piv = (c..dims).max_by(|&x, &y| m[x][c].abs().total_cmp(&m[y][c].abs()))?;
        m.swap(c, piv);
        if m[c][c].abs() < 1e-30 {
            return None;
        }
        for r in 0..dims {
            if r != c {
                let f = m[r][c] / m[c][c];
                for k in c..8 {
                    m[r][k] -= f * m[c][k];
                }
            }
        }
    }
    let mut x = [0.0; 7];
    for i in 0..dims {
        x[i] = m[i][7] / m[i][i];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

/// Refines the cameras of `frames` (frame `frames[0]` is the gauge: fixed
/// pose and scale) so their engine depth maps agree. Writes the new
/// extrinsics into `p`, multiplies each frame's depth by its scale and
/// recomputes its world points. Returns a report.
pub fn refine_poses(p: &mut Posed, frames: &[usize], cfg: &GeoRefine) -> GeoReport {
    let t0 = std::time::Instant::now();
    let n = frames.len();
    let mut rep = GeoReport {
        frames: n,
        ..Default::default()
    };
    if cfg.iters == 0 || n < 2 {
        return rep;
    }
    let thr = conf_threshold(p, frames, cfg.conf_drop);
    let samples: Vec<Vec<(f64, f64, f64)>> = frames
        .par_iter()
        .map(|&i| grid_samples(p, i, thr, cfg.samples))
        .collect();
    let start: Vec<Cam> = frames.iter().map(|&i| Cam::of(p, i)).collect();
    let mut cams = start.clone();
    let nb = neighbours(p, frames, &cams, &samples, cfg.neighbours, 0.05);
    let mut rev = vec![Vec::new(); n];
    for (a, list) in nb.iter().enumerate() {
        for &b in list {
            rev[b].push(a);
        }
    }
    let dims = if cfg.scale { 7 } else { 6 };
    let iters = cfg.iters as usize;
    for it in 0..iters {
        let f = it as f64 / (iters - 1).max(1) as f64;
        let reject = cfg.reject * (cfg.huber * 3.0 / cfg.reject).powf(f);
        let pass: Vec<(Cam, f64, usize, usize)> = (0..n)
            .into_par_iter()
            .map(|a| {
                let ca = cams[a];
                let mut r0 = Vec::new();
                residuals_of(
                    p, frames, &cams, &samples, &nb, &rev, a, &ca, reject, &mut r0,
                );
                let valid = r0.iter().filter(|r| r.is_finite()).count();
                let cost: f64 = r0
                    .iter()
                    .filter(|r| r.is_finite())
                    .map(|r| r.abs())
                    .sum::<f64>();
                if a == 0 || valid < 50 {
                    return (ca, cost, valid, r0.len());
                }
                // Numeric Jacobian (central differences).
                let eps = [1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4];
                let tscale = ca.t.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-3)
                    + samples[a].first().map_or(1.0, |s| s.2);
                let mut jac = vec![[0.0f64; 7]; r0.len()];
                let mut rp = Vec::new();
                let mut rm = Vec::new();
                for d in 0..dims {
                    let mut xi = [0.0; 7];
                    let e = if d < 3 { eps[d] * tscale } else { eps[d] };
                    xi[d] = e;
                    residuals_of(
                        p,
                        frames,
                        &cams,
                        &samples,
                        &nb,
                        &rev,
                        a,
                        &ca.perturbed(&xi),
                        f64::INFINITY,
                        &mut rp,
                    );
                    xi[d] = -e;
                    residuals_of(
                        p,
                        frames,
                        &cams,
                        &samples,
                        &nb,
                        &rev,
                        a,
                        &ca.perturbed(&xi),
                        f64::INFINITY,
                        &mut rm,
                    );
                    for k in 0..r0.len() {
                        jac[k][d] = (rp[k] - rm[k]) / (2.0 * e);
                    }
                }
                let mut h = [[0.0; 7]; 7];
                let mut g = [0.0; 7];
                for (k, &r) in r0.iter().enumerate() {
                    let j = &jac[k];
                    if !r.is_finite() || !j[..dims].iter().all(|v| v.is_finite()) {
                        continue;
                    }
                    let w = huber_w(r, cfg.huber);
                    for x in 0..dims {
                        g[x] -= w * j[x] * r;
                        for y in 0..dims {
                            h[x][y] += w * j[x] * j[y];
                        }
                    }
                }
                let Some(dx) = solve7(&h, &g, 1e-3, dims) else {
                    return (ca, cost, valid, r0.len());
                };
                // Jacobi update: both ends of every pair move, so half a step.
                let half: [f64; 7] = std::array::from_fn(|k| 0.5 * dx[k]);
                (ca.perturbed(&half), cost, valid, r0.len())
            })
            .collect();
        let total: usize = pass.iter().map(|x| x.3).sum();
        let valid: usize = pass.iter().map(|x| x.2).sum();
        let cost: f64 = pass.iter().map(|x| x.1).sum::<f64>() / valid.max(1) as f64;
        if it == 0 {
            rep.rms_before = cost;
            rep.inliers_before = valid as f64 / total.max(1) as f64;
        }
        rep.rms_after = cost;
        rep.inliers_after = valid as f64 / total.max(1) as f64;
        rep.residuals = total;
        for (a, (c, ..)) in pass.into_iter().enumerate() {
            cams[a] = c;
        }
    }
    let mut changes: Vec<f64> = start
        .iter()
        .zip(&cams)
        .map(|(s, c)| {
            let d = mm(&c.r, &transpose(&s.r));
            ((d[0][0] + d[1][1] + d[2][2] - 1.0) / 2.0)
                .clamp(-1.0, 1.0)
                .acos()
                .to_degrees()
        })
        .collect();
    changes.sort_by(f64::total_cmp);
    rep.median_rot_change_deg = changes[changes.len() / 2];
    rep.scale_min = cams.iter().map(|c| c.scale).fold(f64::INFINITY, f64::min);
    rep.scale_max = cams.iter().map(|c| c.scale).fold(0.0, f64::max);
    apply(p, frames, &cams);
    rep.seconds = t0.elapsed().as_secs_f64();
    rep
}

pub(super) fn transpose(m: &M3) -> M3 {
    std::array::from_fn(|i| std::array::from_fn(|j| m[j][i]))
}

/// Writes `cams` into `p` (extrinsics, depth × scale, world points).
fn apply(p: &mut Posed, frames: &[usize], cams: &[Cam]) {
    let (w, h) = (p.cams.width, p.cams.height);
    let px = w * h;
    for (&i, c) in frames.iter().zip(cams) {
        let e = &mut p.cams.extrinsics[i];
        for r in 0..3 {
            e[4 * r] = c.r[r][0] as f32;
            e[4 * r + 1] = c.r[r][1] as f32;
            e[4 * r + 2] = c.r[r][2] as f32;
            e[4 * r + 3] = c.t[r] as f32;
        }
        let depth = &mut p.dense.depth[i * px..(i + 1) * px];
        let world = &mut p.dense.world[i * px * 3..(i + 1) * px * 3];
        for y in 0..h {
            for x in 0..w {
                let k = y * w + x;
                depth[k] = (depth[k] as f64 * c.scale) as f32;
                let xw = c.world_of(c.unproject(x as f64 + 0.5, y as f64 + 0.5, depth[k] as f64));
                for j in 0..3 {
                    world[k * 3 + j] = xw[j] as f32;
                }
            }
        }
    }
}

/// Depth-consistency filtered and fused init points of `frames`: confident
/// pixels on a grid of about `per_frame` per frame whose 3D point agrees
/// (relative depth within `tau`) with at least `min_views` of the frame's
/// neighbours; each kept point is the mean of its own and the agreeing
/// neighbours' points. Subsampled evenly to at most `max_points`.
pub fn consistent_points(
    p: &Posed,
    frames: &[usize],
    conf_drop: f32,
    per_frame: usize,
    tau: f64,
    min_views: usize,
    max_points: usize,
) -> InitPoints {
    let thr = conf_threshold(p, frames, conf_drop);
    let cams: Vec<Cam> = frames.iter().map(|&i| Cam::of(p, i)).collect();
    let nb_samples: Vec<Vec<(f64, f64, f64)>> = frames
        .par_iter()
        .map(|&i| grid_samples(p, i, thr, 3000))
        .collect();
    let nb = neighbours(p, frames, &cams, &nb_samples, 8, 0.05);
    let (w, h) = (p.cams.width, p.cams.height);
    let per: Vec<Vec<([f32; 3], [u8; 3])>> = (0..frames.len())
        .into_par_iter()
        .map(|a| {
            let ca = &cams[a];
            let i = frames[a];
            let mut out = Vec::new();
            for (u, v, z) in grid_samples(p, i, thr, per_frame) {
                let x = ca.world_of(ca.unproject(u, v, z));
                let mut acc = x;
                let mut agree = 0usize;
                for &b in &nb[a] {
                    let cb = &cams[b];
                    let c = cb.cam_of(x);
                    let Some((pu, pv)) = cb.project(c) else {
                        continue;
                    };
                    let Some(d) = depth_view(p, frames[b]).sample(pu, pv) else {
                        continue;
                    };
                    if (c[2] - d).abs() < tau * d {
                        agree += 1;
                        let y = cb.world_of(cb.unproject(pu, pv, d));
                        for k in 0..3 {
                            acc[k] += y[k];
                        }
                    }
                }
                if agree >= min_views {
                    let m = (agree + 1) as f64;
                    let (px_, py_) = (u as usize, v as usize);
                    let o = (i * w * h + py_ * w + px_) * 3;
                    let c = &p.prepared.pixels[o..o + 3];
                    out.push((acc.map(|s| (s / m) as f32), [c[0], c[1], c[2]]));
                }
            }
            out
        })
        .collect();
    let all: Vec<([f32; 3], [u8; 3])> = per.into_iter().flatten().collect();
    let stride = all.len().div_ceil(max_points.max(1)).max(1);
    let mut out = InitPoints::default();
    for (x, c) in all.into_iter().step_by(stride) {
        if x.iter().all(|v| v.is_finite()) {
            out.xyz.push(x);
            out.rgb.push(c);
        }
    }
    out
}

/// Median pairwise relative rotation error in degrees between two sets of
/// world-to-camera rotations (same frames, same order): the
/// `posecmp.py` metric, gauge-free.
pub fn relative_rotation_error(a: &[M3], b: &[M3]) -> f64 {
    let mut e = Vec::new();
    for i in 0..a.len() {
        for j in i + 1..a.len() {
            let ra = mm(&a[i], &transpose(&a[j]));
            let rb = mm(&b[i], &transpose(&b[j]));
            let d = mm(&ra, &transpose(&rb));
            e.push(
                ((d[0][0] + d[1][1] + d[2][2] - 1.0) / 2.0)
                    .clamp(-1.0, 1.0)
                    .acos()
                    .to_degrees(),
            );
        }
    }
    if e.is_empty() {
        return 0.0;
    }
    e.sort_by(f64::total_cmp);
    e[e.len() / 2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perturbation_is_a_left_update() {
        let c = Cam {
            r: so3_exp([0.1, -0.2, 0.3]),
            t: [0.5, -1.0, 2.0],
            fx: 100.0,
            fy: 100.0,
            cx: 50.0,
            cy: 40.0,
            scale: 1.0,
        };
        let xi = [0.01, 0.02, -0.03, 0.001, -0.002, 0.003, 0.0];
        let q = c.perturbed(&xi);
        let x = [0.3, 0.2, 4.0];
        let rd = so3_exp([xi[3], xi[4], xi[5]]);
        let a = c.cam_of(x);
        let want = mv(&rd, a);
        let got = q.cam_of(x);
        for k in 0..3 {
            assert!((got[k] - (want[k] + xi[k])).abs() < 1e-12);
        }
        let back = c.world_of(c.cam_of(x));
        for k in 0..3 {
            assert!((back[k] - x[k]).abs() < 1e-12);
        }
    }

    #[test]
    fn solve7_solves() {
        let mut a = [[0.0; 7]; 7];
        for (i, row) in a.iter_mut().enumerate() {
            row[i] = 2.0 + i as f64;
            if i > 0 {
                row[i - 1] = 0.5;
            }
            if i < 6 {
                row[i + 1] = 0.5;
            }
        }
        let x = [1.0, -2.0, 0.5, 3.0, 0.0, 1.5, -1.0];
        let b: [f64; 7] = std::array::from_fn(|i| (0..7).map(|j| a[i][j] * x[j]).sum());
        let got = solve7(&a, &b, 0.0, 7).unwrap();
        for i in 0..7 {
            assert!((got[i] - x[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn rotation_error_is_gauge_free() {
        let a = vec![so3_exp([0.1, 0.0, 0.0]), so3_exp([0.0, 0.2, 0.0])];
        let g = so3_exp([0.3, -0.1, 0.2]);
        let b: Vec<M3> = a.iter().map(|r| mm(r, &g)).collect();
        assert!(relative_rotation_error(&a, &b) < 1e-6);
        let c = vec![a[0], mm(&so3_exp([0.0, 0.0, 0.01]), &a[1])];
        let e = relative_rotation_error(&a, &c);
        assert!((e - 0.01f64.to_degrees()).abs() < 1e-6);
    }
}
