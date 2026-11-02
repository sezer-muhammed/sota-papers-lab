//! Texture baking from training photos with visibility (z-buffer per view) and
//! view weighting. Owner: mesh-tex.
//!
//! CPU path: rasterize depth buffers per camera, then for each UV texel project
//! to cameras, z-test, blend. GPU path: `cuda/mesh_bake.cu` parallelizes the
//! per-texel work.

use std::time::Instant;

use spl_core::Result;

use super::{BakeCamera, TextureAtlas, UvMesh};
use crate::gpu::{COMMON_CUH, CudaFile, Defines, KernelSource};

pub const MESH_BAKE_CU: CudaFile = CudaFile {
    name: "mesh_bake.cu",
    text: include_str!("../../cuda/mesh_bake.cu"),
};

pub const MESH_BAKE_SOURCE: KernelSource = KernelSource {
    module: "mesh_bake",
    files: &[COMMON_CUH, MESH_BAKE_CU],
    kernels: &["bake_texels", "render_depth"],
};

pub fn mesh_bake_defines() -> Defines {
    Defines::new()
}

/// Bake texture from views onto the UV atlas using the CPU path.
///
/// For each texel in the atlas, finds the triangle it belongs to, computes its
/// 3D position, projects into each camera, z-tests against a per-view depth
/// buffer, and blends visible contributions weighted by the cosine of the view
/// angle.
pub fn bake_texture_cpu(
    mesh: &UvMesh,
    cameras: &[BakeCamera],
    images: &[&[u8]],
    image_strides: &[u32],
) -> Result<(TextureAtlas, f64)> {
    let started = Instant::now();
    let aw = mesh.atlas_width;
    let ah = mesh.atlas_height;

    let depth_maps: Vec<Vec<f32>> = cameras
        .iter()
        .map(|cam| render_depth_map(mesh, cam))
        .collect();

    let mut atlas = TextureAtlas::new(aw, ah);
    let mut coverage = vec![false; (aw * ah) as usize];

    for tri_idx in 0..mesh.indices.len() / 3 {
        let i0 = mesh.indices[tri_idx * 3] as usize;
        let i1 = mesh.indices[tri_idx * 3 + 1] as usize;
        let i2 = mesh.indices[tri_idx * 3 + 2] as usize;

        let uv0 = mesh.uvs[i0];
        let uv1 = mesh.uvs[i1];
        let uv2 = mesh.uvs[i2];

        let p0 = mesh.positions[i0];
        let p1 = mesh.positions[i1];
        let p2 = mesh.positions[i2];

        let n0 = mesh.normals[i0];

        let min_u = (uv0[0].min(uv1[0]).min(uv2[0]) * aw as f32).floor() as i32;
        let max_u = (uv0[0].max(uv1[0]).max(uv2[0]) * aw as f32).ceil() as i32;
        let min_v = (uv0[1].min(uv1[1]).min(uv2[1]) * ah as f32).floor() as i32;
        let max_v = (uv0[1].max(uv1[1]).max(uv2[1]) * ah as f32).ceil() as i32;

        let min_u = min_u.max(0) as u32;
        let max_u = (max_u as u32).min(aw);
        let min_v = min_v.max(0) as u32;
        let max_v = (max_v as u32).min(ah);

        for ty in min_v..max_v {
            for tx in min_u..max_u {
                let u = (tx as f32 + 0.5) / aw as f32;
                let v = (ty as f32 + 0.5) / ah as f32;

                let Some((b0, b1, b2)) = barycentric_2d(uv0, uv1, uv2, [u, v]) else {
                    continue;
                };

                if b0 < -1e-4 || b1 < -1e-4 || b2 < -1e-4 {
                    continue;
                }

                let world_pos = [
                    b0 * p0[0] + b1 * p1[0] + b2 * p2[0],
                    b0 * p0[1] + b1 * p1[1] + b2 * p2[1],
                    b0 * p0[2] + b1 * p1[2] + b2 * p2[2],
                ];

                let tex_idx = (ty * aw + tx) as usize;
                let (r, g, b, total_w) =
                    blend_views(world_pos, n0, cameras, images, image_strides, &depth_maps);

                if total_w > 0.0 {
                    let off = tex_idx * 4;
                    atlas.pixels[off] = (r / total_w).clamp(0.0, 255.0) as u8;
                    atlas.pixels[off + 1] = (g / total_w).clamp(0.0, 255.0) as u8;
                    atlas.pixels[off + 2] = (b / total_w).clamp(0.0, 255.0) as u8;
                    atlas.pixels[off + 3] = 255;
                    coverage[tex_idx] = true;
                }
            }
        }
    }

    dilate_atlas(&mut atlas, &coverage);

    let ms = started.elapsed().as_secs_f64() * 1e3;
    Ok((atlas, ms))
}

fn barycentric_2d(a: [f32; 2], b: [f32; 2], c: [f32; 2], p: [f32; 2]) -> Option<(f32, f32, f32)> {
    let v0 = [b[0] - a[0], b[1] - a[1]];
    let v1 = [c[0] - a[0], c[1] - a[1]];
    let v2 = [p[0] - a[0], p[1] - a[1]];

    let d00 = v0[0] * v0[0] + v0[1] * v0[1];
    let d01 = v0[0] * v1[0] + v0[1] * v1[1];
    let d11 = v1[0] * v1[0] + v1[1] * v1[1];
    let d20 = v2[0] * v0[0] + v2[1] * v0[1];
    let d21 = v2[0] * v1[0] + v2[1] * v1[1];

    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1e-12 {
        return None;
    }
    let inv = 1.0 / denom;
    let u = (d11 * d20 - d01 * d21) * inv;
    let v = (d00 * d21 - d01 * d20) * inv;
    Some((1.0 - u - v, u, v))
}

fn render_depth_map(mesh: &UvMesh, cam: &BakeCamera) -> Vec<f32> {
    let w = cam.width as usize;
    let h = cam.height as usize;
    let mut depth = vec![f32::INFINITY; w * h];

    for tri_idx in 0..mesh.indices.len() / 3 {
        let i0 = mesh.indices[tri_idx * 3] as usize;
        let i1 = mesh.indices[tri_idx * 3 + 1] as usize;
        let i2 = mesh.indices[tri_idx * 3 + 2] as usize;

        let c0 = world_to_cam(mesh.positions[i0], cam);
        let c1 = world_to_cam(mesh.positions[i1], cam);
        let c2 = world_to_cam(mesh.positions[i2], cam);

        if c0[2] <= 0.0 && c1[2] <= 0.0 && c2[2] <= 0.0 {
            continue;
        }

        let px0 = cam_to_pixel(c0, cam);
        let px1 = cam_to_pixel(c1, cam);
        let px2 = cam_to_pixel(c2, cam);

        let min_x = px0[0].min(px1[0]).min(px2[0]).floor() as i32;
        let max_x = px0[0].max(px1[0]).max(px2[0]).ceil() as i32;
        let min_y = px0[1].min(px1[1]).min(px2[1]).floor() as i32;
        let max_y = px0[1].max(px1[1]).max(px2[1]).ceil() as i32;

        let min_x = min_x.max(0) as usize;
        let max_x = (max_x as usize).min(w);
        let min_y = min_y.max(0) as usize;
        let max_y = (max_y as usize).min(h);

        for y in min_y..max_y {
            for x in min_x..max_x {
                let px = [x as f32 + 0.5, y as f32 + 0.5];
                let Some((b0, b1, b2)) = barycentric_2d(px0, px1, px2, px) else {
                    continue;
                };
                if b0 < 0.0 || b1 < 0.0 || b2 < 0.0 {
                    continue;
                }
                let z = b0 * c0[2] + b1 * c1[2] + b2 * c2[2];
                if z > 0.0 && z < depth[y * w + x] {
                    depth[y * w + x] = z;
                }
            }
        }
    }
    depth
}

fn world_to_cam(p: [f32; 3], cam: &BakeCamera) -> [f32; 3] {
    let r = &cam.r_cw;
    let t = &cam.t_cw;
    [
        r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2] + t[0],
        r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2] + t[1],
        r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2] + t[2],
    ]
}

fn cam_to_pixel(c: [f32; 3], cam: &BakeCamera) -> [f32; 2] {
    if c[2] <= 0.0 {
        return [f32::NAN, f32::NAN];
    }
    [cam.fx * c[0] / c[2] + cam.cx, cam.fy * c[1] / c[2] + cam.cy]
}

fn blend_views(
    world_pos: [f32; 3],
    normal: [f32; 3],
    cameras: &[BakeCamera],
    images: &[&[u8]],
    strides: &[u32],
    depth_maps: &[Vec<f32>],
) -> (f32, f32, f32, f32) {
    let mut r_acc = 0.0f32;
    let mut g_acc = 0.0f32;
    let mut b_acc = 0.0f32;
    let mut w_acc = 0.0f32;

    for (view_idx, cam) in cameras.iter().enumerate() {
        let cam_pos = world_to_cam(world_pos, cam);
        if cam_pos[2] <= 0.01 {
            continue;
        }

        let px = cam_to_pixel(cam_pos, cam);
        let ix = px[0] as i32;
        let iy = px[1] as i32;

        if ix < 0 || iy < 0 || ix >= cam.width as i32 || iy >= cam.height as i32 {
            continue;
        }

        let depth_idx = iy as usize * cam.width as usize + ix as usize;
        let stored_z = depth_maps[view_idx][depth_idx];
        let bias = 0.01 * cam_pos[2];
        if cam_pos[2] > stored_z + bias {
            continue;
        }

        let view_dir = [
            -(cam.r_cw[0][0] * cam.t_cw[0]
                + cam.r_cw[1][0] * cam.t_cw[1]
                + cam.r_cw[2][0] * cam.t_cw[2])
                - world_pos[0],
            -(cam.r_cw[0][1] * cam.t_cw[0]
                + cam.r_cw[1][1] * cam.t_cw[1]
                + cam.r_cw[2][1] * cam.t_cw[2])
                - world_pos[1],
            -(cam.r_cw[0][2] * cam.t_cw[0]
                + cam.r_cw[1][2] * cam.t_cw[1]
                + cam.r_cw[2][2] * cam.t_cw[2])
                - world_pos[2],
        ];
        let vlen =
            (view_dir[0] * view_dir[0] + view_dir[1] * view_dir[1] + view_dir[2] * view_dir[2])
                .sqrt();
        if vlen < 1e-8 {
            continue;
        }
        let vd = [view_dir[0] / vlen, view_dir[1] / vlen, view_dir[2] / vlen];

        let cos_angle = normal[0] * vd[0] + normal[1] * vd[1] + normal[2] * vd[2];
        if cos_angle <= 0.0 {
            continue;
        }

        let weight = cos_angle;
        let stride = strides[view_idx] as usize;
        let pix_off = iy as usize * stride + ix as usize * 3;
        if pix_off + 2 >= images[view_idx].len() {
            continue;
        }

        r_acc += images[view_idx][pix_off] as f32 * weight;
        g_acc += images[view_idx][pix_off + 1] as f32 * weight;
        b_acc += images[view_idx][pix_off + 2] as f32 * weight;
        w_acc += weight;
    }

    (r_acc, g_acc, b_acc, w_acc)
}

/// Fill uncovered texels by propagating from nearest covered neighbour.
fn dilate_atlas(atlas: &mut TextureAtlas, coverage: &[bool]) {
    let w = atlas.width as i32;
    let h = atlas.height as i32;
    let mut filled = coverage.to_vec();

    for _pass in 0..4 {
        let prev = atlas.pixels.clone();
        let prev_filled = filled.clone();
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                if prev_filled[idx] {
                    continue;
                }
                let mut r = 0u32;
                let mut g = 0u32;
                let mut b = 0u32;
                let mut count = 0u32;
                for &(dx, dy) in &[(-1, 0), (1, 0), (0, -1), (0, 1)] {
                    let nx = x + dx;
                    let ny = y + dy;
                    if nx >= 0 && nx < w && ny >= 0 && ny < h {
                        let ni = (ny * w + nx) as usize;
                        if prev_filled[ni] {
                            r += prev[ni * 4] as u32;
                            g += prev[ni * 4 + 1] as u32;
                            b += prev[ni * 4 + 2] as u32;
                            count += 1;
                        }
                    }
                }
                if let (Some(ra), Some(ga), Some(ba)) = (
                    r.checked_div(count),
                    g.checked_div(count),
                    b.checked_div(count),
                ) {
                    let off = idx * 4;
                    atlas.pixels[off] = ra as u8;
                    atlas.pixels[off + 1] = ga as u8;
                    atlas.pixels[off + 2] = ba as u8;
                    atlas.pixels[off + 3] = 255;
                    filled[idx] = true;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::tests::cube_mesh;
    use crate::mesh::uv::generate_atlas;

    fn look_at_camera(eye: [f32; 3], target: [f32; 3]) -> BakeCamera {
        let fwd = {
            let d = [target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]];
            let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            [d[0] / l, d[1] / l, d[2] / l]
        };
        let world_up = [0.0f32, 1.0, 0.0];
        let right = {
            let c = [
                fwd[1] * world_up[2] - fwd[2] * world_up[1],
                fwd[2] * world_up[0] - fwd[0] * world_up[2],
                fwd[0] * world_up[1] - fwd[1] * world_up[0],
            ];
            let l = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
            if l > 1e-6 {
                [c[0] / l, c[1] / l, c[2] / l]
            } else {
                [1.0, 0.0, 0.0]
            }
        };
        // COLMAP: x=right, y=down, z=forward
        let down = [
            fwd[1] * right[2] - fwd[2] * right[1],
            fwd[2] * right[0] - fwd[0] * right[2],
            fwd[0] * right[1] - fwd[1] * right[0],
        ];
        let r_cw = [right, down, fwd];
        let t_cw = [
            -(r_cw[0][0] * eye[0] + r_cw[0][1] * eye[1] + r_cw[0][2] * eye[2]),
            -(r_cw[1][0] * eye[0] + r_cw[1][1] * eye[1] + r_cw[1][2] * eye[2]),
            -(r_cw[2][0] * eye[0] + r_cw[2][1] * eye[1] + r_cw[2][2] * eye[2]),
        ];
        BakeCamera {
            r_cw,
            t_cw,
            fx: 200.0,
            fy: 200.0,
            cx: 128.0,
            cy: 128.0,
            width: 256,
            height: 256,
        }
    }

    fn test_cameras() -> Vec<BakeCamera> {
        let d = 5.0f32;
        vec![
            look_at_camera([d, 0.0, 0.0], [0.0, 0.0, 0.0]),
            look_at_camera([-d, 0.0, 0.0], [0.0, 0.0, 0.0]),
            look_at_camera([0.0, 0.0, d], [0.0, 0.0, 0.0]),
            look_at_camera([0.0, 0.0, -d], [0.0, 0.0, 0.0]),
        ]
    }

    fn solid_color_images(cameras: &[BakeCamera], color: [u8; 3]) -> Vec<Vec<u8>> {
        cameras
            .iter()
            .map(|cam| {
                let n = cam.width as usize * cam.height as usize;
                let mut img = Vec::with_capacity(n * 3);
                for _ in 0..n {
                    img.extend_from_slice(&color);
                }
                img
            })
            .collect()
    }

    #[test]
    fn bake_produces_nonzero_texture() {
        let mesh = cube_mesh();
        let (uv_mesh, _) = generate_atlas(&mesh, 128, 2).unwrap();
        let cameras = test_cameras();
        let imgs = solid_color_images(&cameras, [200, 100, 50]);
        let img_refs: Vec<&[u8]> = imgs.iter().map(|i| i.as_slice()).collect();
        let strides: Vec<u32> = cameras.iter().map(|c| c.width * 3).collect();

        let (atlas, ms) = bake_texture_cpu(&uv_mesh, &cameras, &img_refs, &strides).unwrap();

        assert_eq!(atlas.width, uv_mesh.atlas_width);
        assert_eq!(atlas.height, uv_mesh.atlas_height);
        assert!(ms >= 0.0);

        let nonzero = atlas
            .pixels
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|px| px[3] > 0)
            .count();
        assert!(nonzero > 0, "at least some texels should be covered");
    }

    #[test]
    fn bake_color_matches_input() {
        let mesh = cube_mesh();
        let (uv_mesh, _) = generate_atlas(&mesh, 64, 1).unwrap();
        let cameras = test_cameras();
        let imgs = solid_color_images(&cameras, [128, 64, 32]);
        let img_refs: Vec<&[u8]> = imgs.iter().map(|i| i.as_slice()).collect();
        let strides: Vec<u32> = cameras.iter().map(|c| c.width * 3).collect();

        let (atlas, _) = bake_texture_cpu(&uv_mesh, &cameras, &img_refs, &strides).unwrap();

        let covered: Vec<_> = atlas
            .pixels
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|px| px[3] > 0)
            .collect();

        if !covered.is_empty() {
            for px in &covered {
                assert!(
                    (px[0] as i32 - 128).unsigned_abs() <= 2
                        && (px[1] as i32 - 64).unsigned_abs() <= 2
                        && (px[2] as i32 - 32).unsigned_abs() <= 2,
                    "baked color {:?} should be close to [128, 64, 32]",
                    &px[..3]
                );
            }
        }
    }

    #[test]
    fn depth_map_front_face_closer() {
        let mesh = cube_mesh();
        let (uv_mesh, _) = generate_atlas(&mesh, 64, 1).unwrap();
        let cam = BakeCamera {
            r_cw: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            t_cw: [0.0, 0.0, 5.0],
            fx: 100.0,
            fy: 100.0,
            cx: 32.0,
            cy: 32.0,
            width: 64,
            height: 64,
        };
        let depth = render_depth_map(&uv_mesh, &cam);
        let center = depth[32 * 64 + 32];
        assert!(center < f32::INFINITY, "center pixel should have depth");
        assert!(center > 0.0, "depth should be positive");
    }
}
