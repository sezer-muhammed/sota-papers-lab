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
}

impl Posed {
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
    fn holdout_matches_the_protocol() {
        let (tr, te) = holdout_split(17, 8);
        assert_eq!(te, vec![0, 8, 16]);
        assert_eq!(tr.len(), 14);
    }
}
