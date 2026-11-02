//! Depth inputs for Stage B besides the engine: COLMAP sparse depth projected
//! into each view (the stand-in for VGGT-Ω depth on benchmark scenes) and a
//! synthetic ray-cast scene with known geometry and a low-confidence sky.
//! Owner: init-b.

use std::collections::HashMap;

use rayon::prelude::*;
use spl_core::Result;

use super::{DepthFramesHost, InitFrame};
use crate::io::colmap::ColmapModel;
use crate::io::images::Rgb8Image;
use crate::io::scene::View;

/// The view's camera as an [`InitFrame`] at `width × height` (intrinsics scaled
/// from the view's size).
pub fn frame_of_view(v: &View, width: u32, height: u32) -> InitFrame {
    let sx = width as f64 / v.width as f64;
    let sy = height as f64 / v.height as f64;
    InitFrame {
        r_cw: v.r_cw.map(|r| r.map(|x| x as f32)),
        t_cw: v.t_cw.map(|x| x as f32),
        fx: (v.fx * sx) as f32,
        fy: (v.fy * sy) as f32,
        cx: (v.cx * sx) as f32,
        cy: (v.cy * sy) as f32,
    }
}

/// Sparse depth for `views` (with their decoded images at view size): every
/// COLMAP point observed by the view is projected; the nearest wins per
/// pixel. Confidence = track length, colour = the image at that pixel.
/// Pixels without a point have depth 0.
pub fn colmap_sparse_depth(
    model: &ColmapModel,
    views: &[&View],
    images: &[Rgb8Image],
) -> DepthFramesHost {
    let w = views[0].width;
    let h = views[0].height;
    let idx: HashMap<u64, usize> = model
        .points
        .iter()
        .enumerate()
        .map(|(i, p)| (p.id, i))
        .collect();
    let by_name: HashMap<&str, _> = model
        .images
        .values()
        .map(|im| (im.name.as_str(), im))
        .collect();
    let hw = (w * h) as usize;
    let per: Vec<FrameData> = views
        .par_iter()
        .zip(images)
        .map(|(v, img)| {
            let f = frame_of_view(v, w, h);
            let mut depth = vec![0f32; hw];
            let mut conf = vec![0f32; hw];
            let mut rgb = vec![0u8; 3 * hw];
            if let Some(im) = by_name.get(v.name.as_str()) {
                for &(_, _, id) in &im.points2d {
                    let Some(&pi) = (id >= 0).then(|| idx.get(&(id as u64))).flatten() else {
                        continue;
                    };
                    let p = &model.points[pi];
                    let [u, vv, z] = f.project(p.xyz.map(|x| x as f32));
                    if z.is_nan()
                        || z <= 0.0
                        || u < 0.0
                        || vv < 0.0
                        || u >= w as f32
                        || vv >= h as f32
                    {
                        continue;
                    }
                    let (x, y) = (u as u32, vv as u32);
                    let k = (y * w + x) as usize;
                    if depth[k] > 0.0 && depth[k] <= z {
                        continue;
                    }
                    depth[k] = z;
                    conf[k] = p.track.len() as f32;
                    let (ix, iy) = (
                        (x as u64 * img.width as u64 / w as u64) as u32,
                        (y as u64 * img.height as u64 / h as u64) as u32,
                    );
                    rgb[3 * k..3 * k + 3].copy_from_slice(&img.pixel(ix, iy));
                }
            }
            (f, depth, conf, rgb)
        })
        .collect();
    gather(w, h, per)
}

/// One frame's camera, depth, confidence and RGB.
type FrameData = (InitFrame, Vec<f32>, Vec<f32>, Vec<u8>);

fn gather(w: u32, h: u32, per: Vec<FrameData>) -> DepthFramesHost {
    let mut out = DepthFramesHost {
        width: w,
        height: h,
        ..Default::default()
    };
    for (f, d, c, r) in per {
        out.frames.push(f);
        out.depth.extend(d);
        out.conf.extend(c);
        out.rgb.extend(r);
    }
    out
}

/// The synthetic scene: a unit sphere at the origin on a 6 × 6 floor at
/// y = +1.2 (COLMAP: y down), sky elsewhere.
pub mod synthetic {
    use super::*;

    pub const FLOOR_Y: f32 = 1.2;
    pub const FLOOR_HALF: f32 = 3.0;
    pub const SKY_DEPTH: f32 = 1000.0;
    pub const SKY_CONF: f32 = 0.5;

    /// Colour of a surface point (smooth, so voxel means stay close).
    pub fn albedo(p: [f32; 3]) -> [u8; 3] {
        let c = |x: f32| (128.0 + 100.0 * x.sin()).round() as u8;
        [c(2.0 * p[0]), c(2.0 * p[1] + 1.0), c(2.0 * p[2] + 2.0)]
    }

    /// Distance of `p` to the scene surface.
    pub fn surface_distance(p: [f32; 3]) -> f32 {
        let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        let sphere = (r - 1.0).abs();
        let dx = (p[0].abs() - FLOOR_HALF).max(0.0);
        let dz = (p[2].abs() - FLOOR_HALF).max(0.0);
        let floor = ((p[1] - FLOOR_Y).powi(2) + dx * dx + dz * dz).sqrt();
        sphere.min(floor)
    }

    /// Camera `i` of `n` on a ring of radius 4, 1 above the centre, looking at it.
    pub fn camera(i: usize, n: usize, w: u32, h: u32) -> InitFrame {
        let a = i as f32 / n as f32 * std::f32::consts::TAU;
        let c = [4.0 * a.cos(), -1.0 - 0.5 * (3.0 * a).sin(), 4.0 * a.sin()];
        let norm = |v: [f32; 3]| {
            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            v.map(|x| x / l)
        };
        let cross = |a: [f32; 3], b: [f32; 3]| {
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        };
        let f = norm(c.map(|x| -x));
        let r = norm(cross([0.0, 1.0, 0.0], f));
        let d = cross(f, r);
        let rot = [r, d, f];
        let t: [f32; 3] =
            std::array::from_fn(|k| -(rot[k][0] * c[0] + rot[k][1] * c[1] + rot[k][2] * c[2]));
        let focal = 0.8 * w as f32;
        InitFrame {
            r_cw: rot,
            t_cw: t,
            fx: focal,
            fy: focal,
            cx: w as f32 / 2.0,
            cy: h as f32 / 2.0,
        }
    }

    /// Nearest hit of the ray o + s·d (s > 0) with the scene.
    fn cast(o: [f32; 3], d: [f32; 3]) -> Option<f32> {
        let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let mut best: Option<f32> = None;
        let a = dot(d, d);
        let b = dot(o, d);
        let c = dot(o, o) - 1.0;
        let disc = b * b - a * c;
        if disc >= 0.0 {
            let s = (-b - disc.sqrt()) / a;
            if s > 1e-4 {
                best = Some(s);
            }
        }
        if d[1].abs() > 1e-6 {
            let s = (FLOOR_Y - o[1]) / d[1];
            let p = [o[0] + s * d[0], o[2] + s * d[2]];
            if s > 1e-4 && p[0].abs() <= FLOOR_HALF && p[1].abs() <= FLOOR_HALF {
                best = Some(best.map_or(s, |b| b.min(s)));
            }
        }
        best
    }

    /// `n` frames of `w × h`: exact z-depth, confidence 2-5 on surfaces (lower
    /// at grazing angles), sky at [`SKY_DEPTH`] with [`SKY_CONF`].
    pub fn frames(n: usize, w: u32, h: u32) -> DepthFramesHost {
        let per = (0..n)
            .into_par_iter()
            .map(|i| {
                let f = camera(i, n, w, h);
                let o = f.center();
                let hw = (w * h) as usize;
                let mut depth = vec![0f32; hw];
                let mut conf = vec![0f32; hw];
                let mut rgb = vec![0u8; 3 * hw];
                for y in 0..h {
                    for x in 0..w {
                        let k = (y * w + x) as usize;
                        // Direction of the pixel centre at z = 1, in world.
                        let p1 = f.unproject(x, y, 1.0);
                        let dz = [p1[0] - o[0], p1[1] - o[1], p1[2] - o[2]];
                        match cast(o, dz) {
                            Some(s) => {
                                depth[k] = s; // |dz| has camera z = 1, so s is z-depth
                                let p = [o[0] + s * dz[0], o[1] + s * dz[1], o[2] + s * dz[2]];
                                conf[k] = 2.0 + 3.0 * (1.0 - (s / 8.0).min(1.0));
                                rgb[3 * k..3 * k + 3].copy_from_slice(&albedo(p));
                            }
                            None => {
                                depth[k] = SKY_DEPTH;
                                conf[k] = SKY_CONF;
                                rgb[3 * k..3 * k + 3].copy_from_slice(&[150, 190, 240]);
                            }
                        }
                    }
                }
                (f, depth, conf, rgb)
            })
            .collect();
        gather(w, h, per)
    }
}

/// Fraction of synthetic sky pixels (for the percentile choice in tests).
pub fn sky_fraction(f: &DepthFramesHost) -> f32 {
    let sky = f
        .depth
        .iter()
        .filter(|&&z| z >= synthetic::SKY_DEPTH)
        .count();
    sky as f32 / f.depth.len() as f32
}

/// Loads `views` of a scene for [`colmap_sparse_depth`].
pub fn load_images(views: &[&View]) -> Result<Vec<Rgb8Image>> {
    views
        .par_iter()
        .map(|v| crate::io::images::load_view_image(v))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::synthetic::*;
    use super::*;

    #[test]
    fn synthetic_depth_lies_on_the_surface() {
        let f = frames(2, 64, 48);
        f.validate().unwrap();
        let mut hits = 0;
        for (i, fr) in f.frames.iter().enumerate() {
            for y in 0..48 {
                for x in 0..64 {
                    let k = i * 64 * 48 + (y * 64 + x) as usize;
                    let z = f.depth[k];
                    if z >= SKY_DEPTH {
                        continue;
                    }
                    hits += 1;
                    let p = fr.unproject(x, y, z);
                    assert!(surface_distance(p) < 1e-3, "{p:?}");
                }
            }
        }
        assert!(hits > 1000 && sky_fraction(&f) > 0.05, "{hits}");
    }
}
