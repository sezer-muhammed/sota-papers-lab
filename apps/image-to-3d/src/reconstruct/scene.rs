//! VGGT-Ω outputs → the trainer's scene: views at the trainer resolution
//! with one shared focal per physical camera, Stage B depth frames, and the
//! per-view depth priors. Owner: mvp-pipeline.
//!
//! Units are the engine's (world = first frame's camera, model scale). The
//! engine's principal point is the image centre and its images are resized
//! without a crop when the aspect is in [0.5, 2], so intrinsics scale by
//! `W / w` and `H / h` independently.

use std::path::PathBuf;

use anyhow::ensure;
use spl_core::Result;

use crate::init::{DepthFramesHost, InitFrame};
use crate::io::scene::{InitPoints, Scene, View, extent};
use crate::vggt::{DenseHost, Prepared, VggtCameras};

/// The engine's results on the host, for every posed frame.
pub struct Posed {
    pub names: Vec<String>,
    pub cams: VggtCameras,
    pub prepared: Prepared,
    pub dense: DenseHost,
    /// Init points that replace the engine's (diagnostics: reference points).
    pub seed: Option<InitPoints>,
}

impl Posed {
    /// `n` frames shaped like `like` (same engine resolution), zeroed.
    pub fn empty(n: usize, like: &Posed) -> Self {
        let (w, h) = (like.cams.width, like.cams.height);
        let px = w * h;
        Posed {
            names: vec![String::new(); n],
            cams: VggtCameras {
                pose_enc: vec![[0.0; 9]; n],
                extrinsics: vec![[0.0; 12]; n],
                intrinsics: vec![[0.0; 9]; n],
                width: w,
                height: h,
            },
            prepared: Prepared {
                geom: crate::vggt::Geometry::new(n, h, w),
                pixels: vec![0; n * px * 3],
            },
            dense: DenseHost {
                depth: vec![0.0; n * px],
                conf: vec![0.0; n * px],
                world: vec![0.0; n * px * 3],
            },
            seed: None,
        }
    }

    /// Copies the engine results of one chunk (`part` frame k is frame
    /// `chunk[k]`) into `self`, mapped into the frame of the frames already
    /// present (`done`) by a similarity on their shared camera centres. The
    /// first chunk defines the world.
    pub fn merge_chunk(&mut self, part: &Posed, chunk: &[usize], done: &mut [bool]) -> Result<()> {
        use super::align::{M3, V3, map_point, map_pose, sim3_between};
        ensure!(
            part.cams.width == self.cams.width && part.cams.height == self.cams.height,
            "engine chunks differ in resolution"
        );
        let as64 = |(r, t): ([[f32; 3]; 3], [f32; 3])| -> (M3, V3) {
            (r.map(|row| row.map(f64::from)), t.map(f64::from))
        };
        let shared: Vec<usize> = (0..chunk.len()).filter(|&k| done[chunk[k]]).collect();
        let sim = if shared.is_empty() {
            (
                1.0,
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [0.0; 3],
            )
        } else {
            ensure!(shared.len() >= 3, "a chunk shares fewer than 3 frames");
            let dst: Vec<(M3, V3)> = shared.iter().map(|&k| as64(self.pose(chunk[k]))).collect();
            let src: Vec<(M3, V3)> = shared.iter().map(|&k| as64(part.pose(k))).collect();
            sim3_between(&dst, &src)
        };
        let px = self.cams.width * self.cams.height;
        for (k, &f) in chunk.iter().enumerate() {
            if done[f] {
                continue;
            }
            let (r, t) = map_pose(&sim, &as64(part.pose(k)));
            let e = &mut self.cams.extrinsics[f];
            for i in 0..3 {
                e[4 * i] = r[i][0] as f32;
                e[4 * i + 1] = r[i][1] as f32;
                e[4 * i + 2] = r[i][2] as f32;
                e[4 * i + 3] = t[i] as f32;
            }
            self.cams.intrinsics[f] = part.cams.intrinsics[k];
            self.cams.pose_enc[f] = part.cams.pose_enc[k];
            self.names[f] = part.names[k].clone();
            let s = sim.0 as f32;
            let (dst, src) = (f * px, k * px);
            for p in 0..px {
                self.dense.depth[dst + p] = part.dense.depth[src + p] * s;
                self.dense.conf[dst + p] = part.dense.conf[src + p];
                let x: V3 = std::array::from_fn(|j| part.dense.world[(src + p) * 3 + j] as f64);
                let y = map_point(&sim, x);
                let o = (dst + p) * 3;
                self.dense.world[o..o + 3].copy_from_slice(&y.map(|v| v as f32));
            }
            self.prepared.pixels[dst * 3..(dst + px) * 3]
                .copy_from_slice(&part.prepared.pixels[src * 3..(src + px) * 3]);
            done[f] = true;
        }
        Ok(())
    }

    pub fn frames(&self) -> usize {
        self.cams.extrinsics.len()
    }

    /// World-to-camera rotation and translation of frame `i`.
    pub fn pose(&self, i: usize) -> ([[f32; 3]; 3], [f32; 3]) {
        let e = &self.cams.extrinsics[i];
        (
            [[e[0], e[1], e[2]], [e[4], e[5], e[6]], [e[8], e[9], e[10]]],
            [e[3], e[7], e[11]],
        )
    }

    /// Multiplies world units by `k` (translations, depth, world points).
    pub fn rescale(&mut self, k: f32) {
        for e in &mut self.cams.extrinsics {
            e[3] *= k;
            e[7] *= k;
            e[11] *= k;
        }
        self.dense.depth.iter_mut().for_each(|d| *d *= k);
        self.dense.world.iter_mut().for_each(|w| *w *= k);
    }

    /// Replaces the cameras of `frames` by trainer views (`views[k]` is frame
    /// `frames[k]`, at `width × height`) and recomputes their world points
    /// from the engine depth with the new cameras.
    pub fn update_cameras(&mut self, frames: &[usize], views: &[View], width: u32, height: u32) {
        let (w, h) = (self.cams.width, self.cams.height);
        let (sx, sy) = (w as f64 / width as f64, h as f64 / height as f64);
        for (&i, v) in frames.iter().zip(views) {
            let e = &mut self.cams.extrinsics[i];
            for r in 0..3 {
                e[4 * r] = v.r_cw[r][0] as f32;
                e[4 * r + 1] = v.r_cw[r][1] as f32;
                e[4 * r + 2] = v.r_cw[r][2] as f32;
                e[4 * r + 3] = v.t_cw[r] as f32;
            }
            let k = &mut self.cams.intrinsics[i];
            k[0] = (v.fx * sx) as f32;
            k[4] = (v.fy * sy) as f32;
        }
        self.reproject(frames);
    }

    /// Square pixels at the trainer size `width`×`height`: the focal along
    /// the long side is kept and the other derived from it (the engine's two
    /// fovs disagree by up to 3.6 % on Mip-NeRF 360, the long-side one is
    /// within 0.5 % of COLMAP); world points are recomputed.
    pub fn square_pixels(&mut self, width: u32, height: u32) {
        let (w, h) = (self.cams.width as f32, self.cams.height as f32);
        let (sx, sy) = (w / width as f32, h / height as f32);
        let long_x = width >= height;
        for k in self.cams.intrinsics.iter_mut() {
            if long_x {
                k[4] = k[0] / sx * sy;
            } else {
                k[0] = k[4] / sy * sx;
            }
        }
        let all: Vec<usize> = (0..self.frames()).collect();
        self.reproject(&all);
    }

    /// Recomputes the world points of `frames` from their depth and cameras.
    pub fn reproject(&mut self, frames: &[usize]) {
        let (w, h) = (self.cams.width, self.cams.height);
        for &i in frames {
            let (rot, t) = self.pose(i);
            let k = self.cams.intrinsics[i];
            for py in 0..h {
                for px in 0..w {
                    let p = i * w * h + py * w + px;
                    let z = self.dense.depth[p];
                    let xc = [
                        (px as f32 + 0.5 - k[2]) / k[0] * z - t[0],
                        (py as f32 + 0.5 - k[5]) / k[4] * z - t[1],
                        z - t[2],
                    ];
                    let world: [f32; 3] = std::array::from_fn(|j| {
                        rot[0][j] * xc[0] + rot[1][j] * xc[1] + rot[2][j] * xc[2]
                    });
                    self.dense.world[p * 3..p * 3 + 3].copy_from_slice(&world);
                }
            }
        }
    }

    /// Mean distance of the camera centres of `frames` from their centroid.
    pub fn spread(&self, frames: &[usize]) -> f32 {
        let c: Vec<[f32; 3]> = frames
            .iter()
            .map(|&i| {
                let (r, t) = self.pose(i);
                std::array::from_fn(|j| -(r[0][j] * t[0] + r[1][j] * t[1] + r[2][j] * t[2]))
            })
            .collect();
        let n = c.len().max(1) as f32;
        let m: [f32; 3] = std::array::from_fn(|k| c.iter().map(|p| p[k]).sum::<f32>() / n);
        c.iter()
            .map(|p| ((p[0] - m[0]).powi(2) + (p[1] - m[1]).powi(2) + (p[2] - m[2]).powi(2)).sqrt())
            .sum::<f32>()
            / n
    }

    /// Median focal over `frames` at a `width × height` image, (fx, fy).
    pub fn shared_focal(&self, frames: &[usize], width: u32, height: u32) -> (f64, f64) {
        let (w, h) = (self.cams.width as f64, self.cams.height as f64);
        let med = |mut v: Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v.get(v.len() / 2).copied().unwrap_or(1.0)
        };
        let fx = med(frames
            .iter()
            .map(|&i| self.cams.intrinsics[i][0] as f64 * width as f64 / w)
            .collect());
        let fy = med(frames
            .iter()
            .map(|&i| self.cams.intrinsics[i][4] as f64 * height as f64 / h)
            .collect());
        (fx, fy)
    }

    /// Trainer views of `frames` (all `width × height`), shared focal.
    pub fn views(&self, frames: &[usize], width: u32, height: u32, focal: (f64, f64)) -> Vec<View> {
        frames
            .iter()
            .map(|&i| {
                let (r, t) = self.pose(i);
                View {
                    name: self.names[i].clone(),
                    camera: 0,
                    r_cw: r.map(|row| row.map(f64::from)),
                    t_cw: t.map(f64::from),
                    fx: focal.0,
                    fy: focal.1,
                    cx: width as f64 / 2.0,
                    cy: height as f64 / 2.0,
                    width,
                    height,
                    image_path: PathBuf::from(&self.names[i]),
                }
            })
            .collect()
    }

    /// Stage B input: depth, confidence and colour of `frames` at the
    /// engine's resolution, with the engine's per-frame intrinsics.
    pub fn depth_frames(&self, frames: &[usize]) -> DepthFramesHost {
        let (w, h) = (self.cams.width, self.cams.height);
        let px = w * h;
        let mut out = DepthFramesHost {
            width: w as u32,
            height: h as u32,
            ..Default::default()
        };
        for &i in frames {
            let (r_cw, t_cw) = self.pose(i);
            let k = &self.cams.intrinsics[i];
            out.frames.push(InitFrame {
                r_cw,
                t_cw,
                fx: k[0],
                fy: k[4],
                cx: k[2],
                cy: k[5],
            });
            out.depth
                .extend_from_slice(&self.dense.depth[i * px..(i + 1) * px]);
            out.conf
                .extend_from_slice(&self.dense.conf[i * px..(i + 1) * px]);
            out.rgb
                .extend_from_slice(&self.prepared.pixels[i * px * 3..(i + 1) * px * 3]);
        }
        out
    }

    /// Depth priors of `frames` (`[F][h][w]`) and their weights: 1 where the
    /// confidence is at or above the `percentile` (0-1) of these frames, else 0.
    pub fn depth_priors(&self, frames: &[usize], percentile: f32) -> (Vec<f32>, Vec<f32>) {
        let px = self.cams.width * self.cams.height;
        let mut conf: Vec<f32> = frames
            .iter()
            .flat_map(|&i| self.dense.conf[i * px..(i + 1) * px].iter().copied())
            .filter(|c| c.is_finite())
            .collect();
        conf.sort_by(f32::total_cmp);
        let q = ((percentile.clamp(0.0, 1.0) * conf.len().saturating_sub(1) as f32) as usize)
            .min(conf.len().saturating_sub(1));
        let thr = conf.get(q).copied().unwrap_or(f32::INFINITY);
        let mut depth = Vec::with_capacity(frames.len() * px);
        let mut weight = Vec::with_capacity(frames.len() * px);
        for &i in frames {
            for p in i * px..(i + 1) * px {
                let d = self.dense.depth[p];
                let ok = d.is_finite() && d > 0.0 && self.dense.conf[p] >= thr;
                depth.push(if ok { d } else { 0.0 });
                weight.push(if ok { 1.0 } else { 0.0 });
            }
        }
        (depth, weight)
    }

    /// A sparse colour point cloud of `frames` (every `stride`-th confident
    /// pixel), for the trainer's own bookkeeping and the early `points` slot.
    pub fn points(&self, frames: &[usize], percentile: f32, max_points: usize) -> InitPoints {
        let px = self.cams.width * self.cams.height;
        let mut conf: Vec<f32> = frames
            .iter()
            .flat_map(|&i| self.dense.conf[i * px..(i + 1) * px].iter().copied())
            .filter(|c| c.is_finite())
            .collect();
        conf.sort_by(f32::total_cmp);
        let q = ((percentile.clamp(0.0, 1.0) * conf.len().saturating_sub(1) as f32) as usize)
            .min(conf.len().saturating_sub(1));
        let thr = conf.get(q).copied().unwrap_or(f32::INFINITY);
        let kept = conf.len() - q;
        let stride = kept.div_ceil(max_points.max(1)).max(1);
        let mut out = InitPoints::default();
        let mut seen = 0usize;
        for &i in frames {
            for p in i * px..(i + 1) * px {
                if self.dense.conf[p]
                    .partial_cmp(&thr)
                    .is_none_or(|o| o.is_lt())
                {
                    continue;
                }
                seen += 1;
                if !(seen - 1).is_multiple_of(stride) {
                    continue;
                }
                let xyz = [
                    self.dense.world[p * 3],
                    self.dense.world[p * 3 + 1],
                    self.dense.world[p * 3 + 2],
                ];
                if xyz.iter().all(|v| v.is_finite()) {
                    out.xyz.push(xyz);
                    let c = &self.prepared.pixels[p * 3..p * 3 + 3];
                    out.rgb.push([c[0], c[1], c[2]]);
                }
            }
        }
        out
    }
}

/// The training scene: `views` are the training frames only (scene view
/// index = training index), so held-out frames never reach the optimizer.
/// Merges `points` that share a voxel (mean position and colour). The voxel
/// edge is `voxel × extent` inside the camera sphere (`center`, `extent`) and
/// grows linearly with the distance beyond it, so the far background is not
/// over-represented by every view that sees it.
pub fn voxel_merge(points: &InitPoints, center: [f32; 3], extent: f32, voxel: f32) -> InitPoints {
    use std::collections::HashMap;
    let base = (voxel * extent).max(1e-9);
    let mut cells: HashMap<[i64; 4], ([f64; 3], [u32; 3], u32)> = HashMap::new();
    let mut order = Vec::new();
    for (x, c) in points.xyz.iter().zip(&points.rgb) {
        let d = ((0..3).map(|k| (x[k] - center[k]).powi(2)).sum::<f32>()).sqrt();
        // Shells of constant voxel size, each one voxel doubling.
        let level = (d / extent).max(1.0).log2().floor() as i64;
        let size = base * (1u64 << level.clamp(0, 30)) as f32;
        let key = [
            level,
            (x[0] / size).floor() as i64,
            (x[1] / size).floor() as i64,
            (x[2] / size).floor() as i64,
        ];
        let e = cells.entry(key).or_insert_with(|| {
            order.push(key);
            ([0.0; 3], [0; 3], 0)
        });
        for k in 0..3 {
            e.0[k] += x[k] as f64;
            e.1[k] += c[k] as u32;
        }
        e.2 += 1;
    }
    let mut out = InitPoints::default();
    for key in order {
        let (p, c, n) = cells[&key];
        out.xyz.push(p.map(|v| (v / n as f64) as f32));
        out.rgb.push(c.map(|v| (v / n) as u8));
    }
    out
}

/// Every `stride`-th point so at most `max` remain.
pub fn subsample(points: InitPoints, max: usize) -> InitPoints {
    let stride = points.xyz.len().div_ceil(max.max(1)).max(1);
    if stride == 1 {
        return points;
    }
    InitPoints {
        xyz: points.xyz.into_iter().step_by(stride).collect(),
        rgb: points.rgb.into_iter().step_by(stride).collect(),
    }
}

/// Median distance from `center` and the fraction beyond `extent`.
pub fn spread_stats(points: &InitPoints, center: [f32; 3], extent: f32) -> (f32, f32) {
    let mut d: Vec<f32> = points
        .xyz
        .iter()
        .map(|x| ((0..3).map(|k| (x[k] - center[k]).powi(2)).sum::<f32>()).sqrt() / extent)
        .collect();
    if d.is_empty() {
        return (0.0, 0.0);
    }
    let far = d.iter().filter(|&&v| v > 1.0).count() as f32 / d.len() as f32;
    d.sort_by(f32::total_cmp);
    (d[d.len() / 2], far)
}

pub fn training_scene(views: Vec<View>, points: InitPoints) -> Result<Scene> {
    ensure!(!views.is_empty(), "no training views");
    let (center, r) = extent(views.iter().map(View::center));
    let n = views.len();
    Ok(Scene {
        root: PathBuf::new(),
        views,
        train: (0..n).collect(),
        test: Vec::new(),
        points,
        extent: r.max(1e-3),
        center,
        max_pp_offset: 0.0,
    })
}

/// Mip-NeRF 360 / Inria protocol split of `n` name-sorted frames: every
/// `every`-th (from 0) is held out. Returns (train, test).
pub fn holdout_split(n: usize, every: usize) -> (Vec<usize>, Vec<usize>) {
    if every == 0 {
        return ((0..n).collect(), Vec::new());
    }
    (0..n).partition(|i| i % every != 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vggt::Geometry;

    fn posed(n: usize, w: usize, h: usize) -> Posed {
        let mut cams = VggtCameras {
            width: w,
            height: h,
            ..Default::default()
        };
        for i in 0..n {
            cams.pose_enc.push([0.0; 9]);
            cams.extrinsics
                .push([1., 0., 0., i as f32, 0., 1., 0., 0., 0., 0., 1., 0.]);
            let f = 100.0 + i as f32;
            cams.intrinsics
                .push([f, 0., w as f32 / 2., 0., f, h as f32 / 2., 0., 0., 1.]);
        }
        let px = n * w * h;
        Posed {
            names: (0..n).map(|i| format!("f{i}.jpg")).collect(),
            cams,
            prepared: Prepared {
                geom: Geometry::new(n, h, w),
                pixels: vec![128; px * 3],
            },
            dense: DenseHost {
                depth: (0..px).map(|p| 1.0 + (p % 7) as f32).collect(),
                conf: (0..px).map(|p| (p % 10) as f32).collect(),
                world: vec![0.5; px * 3],
            },
            seed: None,
        }
    }

    #[test]
    fn views_scale_intrinsics_and_share_focal() {
        let p = posed(5, 32, 16);
        let frames = [0, 1, 2, 3, 4];
        let f = p.shared_focal(&frames, 64, 32);
        assert_eq!(f, (204.0, 204.0), "median of 100..104 doubled");
        let v = p.views(&frames, 64, 32, f);
        assert_eq!(v.len(), 5);
        assert!(
            v.iter()
                .all(|v| v.fx == 204.0 && v.cx == 32.0 && v.cy == 16.0)
        );
        assert_eq!(v[3].t_cw, [3.0, 0.0, 0.0]);
        let s = training_scene(v, InitPoints::default()).unwrap();
        assert_eq!(s.train, vec![0, 1, 2, 3, 4]);
        // Centres at x = -0..-4: mean -2, max distance 2, extent 2.2.
        assert!((s.extent - 2.2).abs() < 1e-5);
    }

    #[test]
    fn depth_inputs_follow_the_frame_subset() {
        let p = posed(4, 8, 4);
        let d = p.depth_frames(&[1, 3]);
        d.validate().unwrap();
        assert_eq!(d.frames.len(), 2);
        assert_eq!(d.frames[1].t_cw, [3.0, 0.0, 0.0]);
        assert_eq!(d.depth[..32], p.dense.depth[32..64]);
        let (prior, w) = p.depth_priors(&[1, 3], 0.5);
        assert_eq!(prior.len(), 64);
        let kept = w.iter().filter(|&&x| x == 1.0).count();
        assert!((25..=40).contains(&kept), "about half kept, got {kept}");
        assert!(prior.iter().zip(&w).all(|(d, w)| (*w == 0.0) || *d > 0.0));
        let pts = p.points(&[0, 1], 0.0, 10);
        assert!(!pts.xyz.is_empty() && pts.xyz.len() <= 10);
    }

    #[test]
    fn updated_cameras_reproject_depth() {
        let mut p = posed(2, 4, 2);
        let views = p.views(&[1], 8, 4, (200.0, 200.0));
        let mut v = views[0].clone();
        v.t_cw = [0.0, 0.0, 1.0];
        p.update_cameras(&[1], &[v], 8, 4);
        assert_eq!(p.cams.intrinsics[1][0], 100.0);
        // Pixel (0, 0) of frame 1: depth d, x = (0.5 − 2)/100·d, world z = d − 1.
        let px = 8;
        let d = p.dense.depth[px];
        assert!((p.dense.world[px * 3] - (0.5 - 2.0) / 100.0 * d).abs() < 1e-6);
        assert!((p.dense.world[px * 3 + 2] - (d - 1.0)).abs() < 1e-6);
        assert_eq!(p.dense.world[0], 0.5, "other frames untouched");
    }

    #[test]
    fn merged_chunks_land_in_the_first_chunks_frame() {
        // Chunk 0 = frames 0..4; chunk 1 = frames 0..3 again plus frame 4,
        // seen in a world scaled by 2 and shifted by (1, 0, 0).
        let full = posed(5, 4, 2);
        let mut a = posed(4, 4, 2);
        a.cams.extrinsics.truncate(4);
        let mut b = posed(4, 4, 2);
        // Frames 0, 1, 2 of chunk 1 are frames 0, 1, 2 of the scene; frame 3 is scene frame 4.
        for (k, f) in [0usize, 1, 2, 4].iter().enumerate() {
            let mut e = full.cams.extrinsics[*f];
            // X' = 2X + (1,0,0): t' = 2t − R·(1,0,0) with R = I.
            e[3] = 2.0 * e[3] - 1.0;
            e[7] *= 2.0;
            e[11] *= 2.0;
            b.cams.extrinsics[k] = e;
        }
        let mut all = Posed::empty(5, &a);
        let mut done = vec![false; 5];
        all.merge_chunk(&a, &[0, 1, 2, 3], &mut done).unwrap();
        all.merge_chunk(&b, &[0, 1, 2, 4], &mut done).unwrap();
        assert!(done.iter().all(|&d| d));
        let e = all.cams.extrinsics[4];
        assert!(
            (e[3] - 4.0).abs() < 1e-5,
            "t_x of frame 4 back in chunk-0 units: {e:?}"
        );
        // Depth scaled by 1/2, world points mapped back.
        assert!((all.dense.depth[4 * 8] - b.dense.depth[3 * 8] * 0.5).abs() < 1e-6);
        assert!((all.dense.world[4 * 8 * 3] - (0.5 - 1.0) * 0.5).abs() < 1e-5);
    }

    #[test]
    fn voxel_merge_fuses_close_points_and_coarsens_far_ones() {
        let pts = InitPoints {
            xyz: vec![
                [0.01, 0.01, 0.01],
                [0.02, 0.02, 0.02],
                [0.5, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.3, 0.0, 0.0],
            ],
            rgb: vec![[0, 0, 0], [100, 100, 100], [7, 7, 7], [1, 1, 1], [3, 3, 3]],
        };
        let m = voxel_merge(&pts, [0.0; 3], 1.0, 0.1);
        // Near: 0.1 voxels (the first two fuse); at 10: 0.8 voxels (fuse too).
        assert_eq!(m.xyz.len(), 3);
        assert_eq!(m.rgb[0], [50, 50, 50]);
        assert!((m.xyz[0][0] - 0.015).abs() < 1e-6);
        assert!((m.xyz[2][0] - 10.15).abs() < 1e-4);
        assert_eq!(subsample(m, 2).xyz.len(), 2);
    }

    #[test]
    fn square_pixels_keep_the_long_side_focal() {
        let mut p = posed(2, 8, 6);
        // Trainer 16×10 from engine 8×6: sx = 0.5, sy = 0.6.
        p.square_pixels(16, 10);
        let v = p.views(&[1], 16, 10, p.shared_focal(&[1], 16, 10));
        assert!((v[0].fx - 202.0).abs() < 1e-3 && (v[0].fy - 202.0).abs() < 1e-3);
    }

    #[test]
    fn holdout_matches_the_protocol() {
        let (tr, te) = holdout_split(17, 8);
        assert_eq!(te, vec![0, 8, 16]);
        assert_eq!(tr.len(), 14);
    }
}
