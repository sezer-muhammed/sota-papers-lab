//! Canonicalisation of VGGT-Ω poses for Pixal3D-mv (spec-pixal3d §2).
//!
//! VGGT-Ω gives OpenCV w2c extrinsics + intrinsics + depth. This module
//! transforms them into the Pixal3D training frame: unit-cube object,
//! Blender-convention cameras, view 0 = canonical front pose F, moderate
//! fov, square crops.

use std::path::Path;

use anyhow::ensure;
use spl_core::Result;

/// A single canonicalised view ready for the Pixal3D sidecar.
#[derive(Debug, Clone)]
pub struct CanonView {
    /// Blender-convention c2w 4×4, row-major.
    pub c2w: [[f64; 4]; 4],
    /// Horizontal FOV in radians.
    pub fov_x: f64,
    /// 1024×1024 RGBA PNG bytes.
    pub rgba: Vec<u8>,
}

/// Result of canonicalisation.
#[derive(Debug)]
pub struct CanonResult {
    pub views: Vec<CanonView>,
    pub mesh_scale: f64,
}

/// 3×3 matrix multiply.
fn mat3_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    c
}

/// 3×3 transpose.
fn mat3_t(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

fn vec3_sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn vec3_add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn vec3_scale(v: [f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn vec3_dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn vec3_cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn vec3_norm(v: [f64; 3]) -> f64 {
    vec3_dot(v, v).sqrt()
}

fn vec3_normalise(v: [f64; 3]) -> [f64; 3] {
    let n = vec3_norm(v);
    if n < 1e-15 {
        return [0.0, 0.0, 1.0];
    }
    vec3_scale(v, 1.0 / n)
}

fn mat3_vec(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// w2c extrinsics (OpenCV) → c2w position and rotation columns.
fn w2c_to_c2w(ext: &[f32; 12]) -> ([[f64; 3]; 3], [f64; 3]) {
    let r = [
        [ext[0] as f64, ext[1] as f64, ext[2] as f64],
        [ext[4] as f64, ext[5] as f64, ext[6] as f64],
        [ext[8] as f64, ext[9] as f64, ext[10] as f64],
    ];
    let t = [ext[3] as f64, ext[7] as f64, ext[11] as f64];
    let rt = mat3_t(&r);
    let pos = mat3_vec(&rt, vec3_scale(t, -1.0));
    (rt, pos)
}

/// Re-aim a camera to look at `target` with minimal roll change.
/// Returns the new rotation matrix (c2w columns: right, down→-up, forward).
fn re_aim(r_c2w: &[[f64; 3]; 3], pos: [f64; 3], target: [f64; 3]) -> [[f64; 3]; 3] {
    let forward = vec3_normalise(vec3_sub(target, pos));
    let old_up = [r_c2w[0][1], r_c2w[1][1], r_c2w[2][1]];
    let right = vec3_normalise(vec3_cross(forward, old_up));
    let up = vec3_cross(right, forward);
    // OpenCV c2w: col0=right, col1=-up (down), col2=forward
    [
        [right[0], -up[0], forward[0]],
        [right[1], -up[1], forward[1]],
        [right[2], -up[2], forward[2]],
    ]
}

/// OpenCV c2w → Blender c2w: Blender camera looks -Z, up is +Y.
/// The conversion flips the Y and Z axes of the camera frame.
fn opencv_to_blender_c2w(r: &[[f64; 3]; 3], pos: [f64; 3]) -> [[f64; 4]; 4] {
    // OpenCV: X=right, Y=down, Z=forward
    // Blender: X=right, Y=up, Z=backward
    // So: X_bl = X_cv, Y_bl = -Y_cv, Z_bl = -Z_cv
    [
        [r[0][0], -r[0][1], -r[0][2], pos[0]],
        [r[1][0], -r[1][1], -r[1][2], pos[1]],
        [r[2][0], -r[2][1], -r[2][2], pos[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Canonicalise VGGT-Ω poses for Pixal3D-mv.
///
/// - `extrinsics`: w2c `[R|t]` row-major, OpenCV convention, per view.
/// - `intrinsics`: `[fx, 0, cx, 0, fy, cy, 0, 0, 1]` per view.
/// - `depths`: per-view depth maps, `[H, W]` f32 in VGGT model pixels.
/// - `confs`: per-view confidence maps, same shape.
/// - `masks`: per-view binary masks (true = object), same spatial size as
///   the input images.
/// - `images`: RGBA PNG bytes per view, original resolution.
/// - `main_view`: index of the main view (default: 0).
/// - `model_wh`: (width, height) of the VGGT model output.
/// - `image_wh`: (width, height) of the original images.
#[allow(clippy::too_many_arguments)]
pub fn canonicalise(
    extrinsics: &[[f32; 12]],
    intrinsics: &[[f32; 9]],
    depths: &[Vec<f32>],
    confs: &[Vec<f32>],
    masks: &[Vec<bool>],
    images: &[Vec<u8>],
    main_view: usize,
    model_wh: (usize, usize),
    image_wh: (usize, usize),
) -> Result<CanonResult> {
    let v = extrinsics.len();
    ensure!((1..=4).contains(&v), "need 1-4 views, got {v}");
    ensure!(
        intrinsics.len() == v
            && depths.len() == v
            && confs.len() == v
            && masks.len() == v
            && images.len() == v,
        "all inputs must have the same number of views"
    );
    ensure!(main_view < v, "main_view {main_view} >= {v}");

    let (mw, mh) = model_wh;
    // 1. Object points from masked depth.
    let mut world_pts: Vec<[f64; 3]> = Vec::new();
    for vi in 0..v {
        let (r_c2w, pos) = w2c_to_c2w(&extrinsics[vi]);
        let k = &intrinsics[vi];
        let (fx, fy, cx, cy) = (k[0] as f64, k[4] as f64, k[2] as f64, k[5] as f64);

        let mask = &masks[vi];
        let depth = &depths[vi];
        let conf = &confs[vi];

        // Scale mask to model resolution if needed.
        let (iw, ih) = image_wh;
        let sx = mw as f64 / iw as f64;
        let sy = mh as f64 / ih as f64;

        for py in 0..mh {
            for px in 0..mw {
                let idx = py * mw + px;
                if idx >= depth.len() || idx >= conf.len() {
                    continue;
                }
                // Map model pixel to image pixel for mask lookup.
                let ix = (px as f64 / sx).min((iw - 1) as f64) as usize;
                let iy = (py as f64 / sy).min((ih - 1) as f64) as usize;
                let mi = iy * iw + ix;
                if mi >= mask.len() || !mask[mi] {
                    continue;
                }
                let d = depth[idx] as f64;
                if !d.is_finite() || d <= 0.0 {
                    continue;
                }
                let x_cam = (px as f64 - cx) / fx * d;
                let y_cam = (py as f64 - cy) / fy * d;
                let z_cam = d;
                let p_cam = [x_cam, y_cam, z_cam];
                let p_world = vec3_add(mat3_vec(&r_c2w, p_cam), pos);
                if p_world.iter().all(|v| v.is_finite()) {
                    world_pts.push(p_world);
                }
            }
        }
    }

    ensure!(!world_pts.is_empty(), "no valid object points found");

    // Drop top/bottom 2% per axis.
    let trimmed = trim_percentile(&world_pts, 0.02);
    ensure!(!trimmed.is_empty(), "all object points trimmed");

    // AABB.
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for p in &trimmed {
        for i in 0..3 {
            lo[i] = lo[i].min(p[i]);
            hi[i] = hi[i].max(p[i]);
        }
    }
    let centre = [
        (lo[0] + hi[0]) * 0.5,
        (lo[1] + hi[1]) * 0.5,
        (lo[2] + hi[2]) * 0.5,
    ];
    let extent = (hi[0] - lo[0])
        .max(hi[1] - lo[1])
        .max(hi[2] - lo[2])
        .max(1e-6);

    // 2. Similarity: x' = (x - centre) / extent.
    // 3. World rotation from the main view (after re-aiming).
    let (r_main, pos_main) = w2c_to_c2w(&extrinsics[main_view]);
    let r_main_aimed = re_aim(&r_main, pos_main, centre);

    // World frame: Z = up of main camera, Y = viewing direction, X = Y × Z.
    let view_dir = [r_main_aimed[0][2], r_main_aimed[1][2], r_main_aimed[2][2]];
    let up_dir = [
        -r_main_aimed[0][1],
        -r_main_aimed[1][1],
        -r_main_aimed[2][1],
    ];
    let world_y = vec3_normalise(view_dir);
    let world_z = vec3_normalise(up_dir);
    let world_x = vec3_normalise(vec3_cross(world_y, world_z));
    let world_z = vec3_cross(world_x, world_y);

    // world_rot: transforms old-world coords to new-world coords.
    let world_rot: [[f64; 3]; 3] = [
        [world_x[0], world_x[1], world_x[2]],
        [world_y[0], world_y[1], world_y[2]],
        [world_z[0], world_z[1], world_z[2]],
    ];

    // Build canonical views.
    let mut views = Vec::with_capacity(v);
    for vi in 0..v {
        let (r_c2w, pos) = w2c_to_c2w(&extrinsics[vi]);
        let r_aimed = re_aim(&r_c2w, pos, centre);

        // Transform position into the new coordinate system.
        let pos_new = mat3_vec(&world_rot, vec3_sub(pos, centre));
        let pos_norm = vec3_scale(pos_new, 1.0 / extent);
        let d_v = vec3_norm(pos_norm);

        // Camera rotation in the new world.
        let r_c2w_new = mat3_mul(&world_rot, &r_aimed);

        // FOV: camera_angle_x = 2 * asin((√3/2) / d_v) * 1.05, clamped.
        let bsphere_r = (3.0_f64).sqrt() * 0.5;
        let sin_half = (bsphere_r / d_v.max(bsphere_r + 0.01)).min(1.0);
        let fov_x =
            (2.0 * sin_half.asin() * 1.05).clamp(10.0_f64.to_radians(), 70.0_f64.to_radians());

        // Blender c2w from the new-world OpenCV c2w.
        let c2w_bl = opencv_to_blender_c2w(&r_c2w_new, pos_norm);

        // Generate the 1024×1024 RGBA image.
        // For now, pass through the original RGBA data. The actual warping
        // (re-aim homography + square crop) happens in the sidecar or on the
        // host with image processing. Here we pass the RGBA bytes as-is
        // since the sidecar's run_mv.py handles the resize.
        let rgba = images[vi].clone();

        views.push(CanonView {
            c2w: c2w_bl,
            fov_x,
            rgba,
        });
    }

    // Verify main view ≈ F.
    let f_rot = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]];
    let main_c2w = &views[main_view].c2w;
    let main_rot = [
        [main_c2w[0][0], main_c2w[0][1], main_c2w[0][2]],
        [main_c2w[1][0], main_c2w[1][1], main_c2w[1][2]],
        [main_c2w[2][0], main_c2w[2][1], main_c2w[2][2]],
    ];
    let mut rot_err = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            rot_err += (main_rot[i][j] - f_rot[i][j]).abs();
        }
    }
    if rot_err > 1e-3 {
        tracing::warn!(rot_err, "main view rotation does not match F within 1e-3");
    }

    Ok(CanonResult {
        views,
        mesh_scale: 1.0,
    })
}

/// Write a Pixal3D `transforms.json` and RGBA PNGs to `dir`.
pub fn write_transforms(dir: &Path, result: &CanonResult, names: &[String]) -> Result<()> {
    use serde_json::{json, to_string_pretty};
    use std::fs;

    fs::create_dir_all(dir)?;

    let frames: Vec<serde_json::Value> = result
        .views
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let name = names.get(i).map(|n| n.as_str()).unwrap_or("view");
            let png_name = format!("{i:04}_{name}.png");
            json!({
                "file_path": png_name,
                "transform_matrix": v.c2w,
                "camera_angle_x": v.fov_x,
            })
        })
        .collect();

    let doc = json!({
        "mesh_scale": result.mesh_scale,
        "frames": frames,
    });

    fs::write(dir.join("transforms.json"), to_string_pretty(&doc)?)?;

    for (i, v) in result.views.iter().enumerate() {
        let name = names.get(i).map(|n| n.as_str()).unwrap_or("view");
        let png_name = format!("{i:04}_{name}.png");
        fs::write(dir.join(&png_name), &v.rgba)?;
    }

    Ok(())
}

/// Trim top/bottom `frac` of points per axis.
fn trim_percentile(pts: &[[f64; 3]], frac: f64) -> Vec<[f64; 3]> {
    if pts.is_empty() {
        return Vec::new();
    }
    let n = pts.len();
    let lo_idx = (n as f64 * frac) as usize;
    let hi_idx = n.saturating_sub(lo_idx + 1);

    let mut keep = vec![true; n];
    for axis in 0..3 {
        let mut sorted: Vec<(usize, f64)> =
            pts.iter().enumerate().map(|(i, p)| (i, p[axis])).collect();
        sorted.sort_by(|a, b| a.1.total_cmp(&b.1));
        for &(i, _) in &sorted[..lo_idx] {
            keep[i] = false;
        }
        if hi_idx < n {
            for &(i, _) in &sorted[hi_idx + 1..] {
                keep[i] = false;
            }
        }
    }

    pts.iter()
        .enumerate()
        .filter(|(i, _)| keep[*i])
        .map(|(_, p)| *p)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mat3_ops() {
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert_eq!(mat3_mul(&id, &id), id);
        assert_eq!(mat3_t(&id), id);
    }

    #[test]
    fn opencv_to_blender_identity() {
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let bl = opencv_to_blender_c2w(&id, [0.0, -1.0, 0.0]);
        // Blender: X=right, Y=up(-down), Z=back(-fwd)
        assert!((bl[0][0] - 1.0).abs() < 1e-10);
        assert!((bl[1][1] - (-1.0)).abs() < 1e-10);
        assert!((bl[2][2] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn trim_percentile_keeps_middle() {
        let pts: Vec<[f64; 3]> = (0..100).map(|i| [i as f64, 0.0, 0.0]).collect();
        let trimmed = trim_percentile(&pts, 0.02);
        assert!(trimmed.len() < 100);
        assert!(trimmed.len() > 90);
    }
}
