//! Parallel texture bake (rayon): the same method as [`super::bake`]'s CPU
//! path (per-view z-buffer, cosine-weighted average of the views that see a
//! texel, 4 dilation passes), with depth maps per view and texels per
//! triangle in parallel, and the geometric face normal. Owner: mesh-mode.

use std::time::Instant;

use rayon::prelude::*;
use spl_core::Result;

use super::{BakeCamera, TextureAtlas, UvMesh, cross, sub};

fn to_cam(p: [f32; 3], c: &BakeCamera) -> [f32; 3] {
    std::array::from_fn(|i| {
        c.r_cw[i][0] * p[0] + c.r_cw[i][1] * p[1] + c.r_cw[i][2] * p[2] + c.t_cw[i]
    })
}

fn edge(a: [f32; 2], b: [f32; 2], p: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}

/// Camera-space z-buffer of the mesh (triangles with a vertex behind
/// `near` are skipped).
fn depth_map(m: &UvMesh, c: &BakeCamera) -> Vec<f32> {
    let (w, h) = (c.width as usize, c.height as usize);
    let mut d = vec![f32::INFINITY; w * h];
    for t in m.indices.as_chunks::<3>().0 {
        let cc = [0, 1, 2].map(|k| to_cam(m.positions[t[k] as usize], c));
        if cc.iter().any(|p| p[2] <= 1e-3) {
            continue;
        }
        let px = cc.map(|p| [c.fx * p[0] / p[2] + c.cx, c.fy * p[1] / p[2] + c.cy]);
        let area = edge(px[0], px[1], px[2]);
        if area.abs() < 1e-12 {
            continue;
        }
        let x0 = px
            .iter()
            .map(|p| p[0])
            .fold(f32::MAX, f32::min)
            .floor()
            .max(0.0) as usize;
        let x1 = (px.iter().map(|p| p[0]).fold(f32::MIN, f32::max).ceil() as i64).clamp(0, w as i64)
            as usize;
        let y0 = px
            .iter()
            .map(|p| p[1])
            .fold(f32::MAX, f32::min)
            .floor()
            .max(0.0) as usize;
        let y1 = (px.iter().map(|p| p[1]).fold(f32::MIN, f32::max).ceil() as i64).clamp(0, h as i64)
            as usize;
        for y in y0..y1 {
            for x in x0..x1 {
                let p = [x as f32 + 0.5, y as f32 + 0.5];
                let b = [
                    edge(px[1], px[2], p) / area,
                    edge(px[2], px[0], p) / area,
                    edge(px[0], px[1], p) / area,
                ];
                if b.iter().any(|&v| v < 0.0) {
                    continue;
                }
                // Perspective-correct depth: 1/z interpolates linearly.
                let z = 1.0 / (b[0] / cc[0][2] + b[1] / cc[1][2] + b[2] / cc[2][2]);
                let o = &mut d[y * w + x];
                if z < *o {
                    *o = z;
                }
            }
        }
    }
    d
}

/// Bakes `images` (interleaved RGB8, `width·3` stride) onto `mesh`'s atlas.
pub fn bake(mesh: &UvMesh, cams: &[BakeCamera], images: &[&[u8]]) -> Result<(TextureAtlas, f64)> {
    let started = Instant::now();
    let (aw, ah) = (mesh.atlas_width, mesh.atlas_height);
    let depth: Vec<Vec<f32>> = cams.par_iter().map(|c| depth_map(mesh, c)).collect();
    let centres: Vec<[f32; 3]> = cams
        .iter()
        .map(|c| {
            std::array::from_fn(|j| {
                -(c.r_cw[0][j] * c.t_cw[0] + c.r_cw[1][j] * c.t_cw[1] + c.r_cw[2][j] * c.t_cw[2])
            })
        })
        .collect();
    let texels: Vec<(u32, [u8; 3])> = mesh
        .indices
        .par_chunks_exact(3)
        .flat_map_iter(|t| {
            let p = [0, 1, 2].map(|k| mesh.positions[t[k] as usize]);
            let uv = [0, 1, 2].map(|k| {
                let q = mesh.uvs[t[k] as usize];
                [q[0] * aw as f32, q[1] * ah as f32]
            });
            let n = cross(sub(p[1], p[0]), sub(p[2], p[0]));
            let nl = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            let area = edge(uv[0], uv[1], uv[2]);
            let mut out = Vec::new();
            if nl <= 0.0 || area.abs() < 1e-12 {
                return out.into_iter();
            }
            let n = n.map(|x| x / nl);
            let x0 = uv
                .iter()
                .map(|q| q[0])
                .fold(f32::MAX, f32::min)
                .floor()
                .max(0.0) as u32;
            let x1 = (uv.iter().map(|q| q[0]).fold(f32::MIN, f32::max).ceil() as u32).min(aw);
            let y0 = uv
                .iter()
                .map(|q| q[1])
                .fold(f32::MAX, f32::min)
                .floor()
                .max(0.0) as u32;
            let y1 = (uv.iter().map(|q| q[1]).fold(f32::MIN, f32::max).ceil() as u32).min(ah);
            for ty in y0..y1 {
                for tx in x0..x1 {
                    let q = [tx as f32 + 0.5, ty as f32 + 0.5];
                    let b = [
                        edge(uv[1], uv[2], q) / area,
                        edge(uv[2], uv[0], q) / area,
                        edge(uv[0], uv[1], q) / area,
                    ];
                    if b.iter().any(|&v| v < -1e-4) {
                        continue;
                    }
                    let x: [f32; 3] =
                        std::array::from_fn(|k| b[0] * p[0][k] + b[1] * p[1][k] + b[2] * p[2][k]);
                    let mut acc = [0f32; 4];
                    for (v, c) in cams.iter().enumerate() {
                        let pc = to_cam(x, c);
                        if pc[2] <= 1e-3 {
                            continue;
                        }
                        let (u, w) = (c.fx * pc[0] / pc[2] + c.cx, c.fy * pc[1] / pc[2] + c.cy);
                        if u < 0.0 || w < 0.0 || u >= c.width as f32 || w >= c.height as f32 {
                            continue;
                        }
                        let (iu, iw) = (u as usize, w as usize);
                        if pc[2] > depth[v][iw * c.width as usize + iu] * 1.01 {
                            continue;
                        }
                        let d = sub(centres[v], x);
                        let dl = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                        let cosv = (n[0] * d[0] + n[1] * d[1] + n[2] * d[2]) / dl.max(1e-12);
                        if cosv <= 0.0 {
                            continue;
                        }
                        let o = (iw * c.width as usize + iu) * 3;
                        let img = images[v];
                        // Grazing views blur; weight by cos² (sharper than cos).
                        let wgt = cosv * cosv;
                        for k in 0..3 {
                            acc[k] += img[o + k] as f32 * wgt;
                        }
                        acc[3] += wgt;
                    }
                    if acc[3] > 0.0 {
                        out.push((
                            ty * aw + tx,
                            [0, 1, 2].map(|k| (acc[k] / acc[3]).round().clamp(0.0, 255.0) as u8),
                        ));
                    }
                }
            }
            out.into_iter()
        })
        .collect();
    let mut atlas = TextureAtlas::new(aw, ah);
    let mut filled = vec![false; (aw * ah) as usize];
    for (i, c) in texels {
        let o = i as usize * 4;
        atlas.pixels[o..o + 3].copy_from_slice(&c);
        atlas.pixels[o + 3] = 255;
        filled[i as usize] = true;
    }
    dilate(&mut atlas, &mut filled, 4);
    Ok((atlas, started.elapsed().as_secs_f64() * 1e3))
}

/// `passes` rounds of filling empty texels with the mean of their filled
/// 4-neighbours (seam gutters).
fn dilate(a: &mut TextureAtlas, filled: &mut [bool], passes: u32) {
    let (w, h) = (a.width as i64, a.height as i64);
    for _ in 0..passes {
        let prev = a.pixels.clone();
        let pf = filled.to_vec();
        let rows: Vec<(usize, [u8; 3])> = (0..h)
            .into_par_iter()
            .flat_map_iter(|y| {
                let (prev, pf) = (&prev, &pf);
                (0..w).filter_map(move |x| {
                    let i = (y * w + x) as usize;
                    if pf[i] {
                        return None;
                    }
                    let mut s = [0u32; 4];
                    for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                        let (nx, ny) = (x + dx, y + dy);
                        if nx >= 0 && nx < w && ny >= 0 && ny < h {
                            let j = (ny * w + nx) as usize;
                            if pf[j] {
                                for k in 0..3 {
                                    s[k] += prev[j * 4 + k] as u32;
                                }
                                s[3] += 1;
                            }
                        }
                    }
                    (s[3] > 0).then(|| (i, [0, 1, 2].map(|k| (s[k] / s[3]) as u8)))
                })
            })
            .collect();
        for (i, c) in rows {
            a.pixels[i * 4..i * 4 + 3].copy_from_slice(&c);
            a.pixels[i * 4 + 3] = 255;
            filled[i] = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit quad facing a camera, textured from a two-colour photo: the
    /// left half of the atlas region reads red, the right blue.
    #[test]
    fn bakes_a_quad_from_one_view() {
        let mesh = UvMesh {
            positions: vec![
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ],
            normals: vec![[0.0, 0.0, -1.0]; 4],
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            // Counter-clockwise seen from −z (the camera side).
            indices: vec![0, 2, 1, 0, 3, 2],
            atlas_width: 32,
            atlas_height: 32,
        };
        let cam = BakeCamera {
            r_cw: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            t_cw: [0.0, 0.0, 4.0],
            fx: 100.0,
            fy: 100.0,
            cx: 64.0,
            cy: 64.0,
            width: 128,
            height: 128,
        };
        let mut img = vec![0u8; 128 * 128 * 3];
        for y in 0..128 {
            for x in 0..128 {
                let o = (y * 128 + x) * 3;
                if x < 64 {
                    img[o] = 255;
                } else {
                    img[o + 2] = 255;
                }
            }
        }
        let (a, _) = bake(&mesh, &[cam], &[&img]).unwrap();
        let px = |x: usize, y: usize| &a.pixels[(y * 32 + x) * 4..(y * 32 + x) * 4 + 4];
        assert_eq!(px(4, 16), &[255, 0, 0, 255]);
        assert_eq!(px(28, 16), &[0, 0, 255, 255]);
    }
}
