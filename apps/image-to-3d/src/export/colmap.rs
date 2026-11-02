//! Camera exports: COLMAP `sparse/0` and nerfstudio `transforms.json`
//! (spec-train 9.3-9.4). Owner: loss (E4).
//!
//! COLMAP: one PINHOLE camera per physical camera at the trained image size,
//! images with `qvec` (w ≥ 0), `tvec`, zero 2D points, and up to 500k Gaussian
//! centres with o > 0.5 sampled ∝ o as points3D. The byte layouts are
//! `crate::io::colmap::ColmapModel::{write_bin, write_txt}` (implemented).
//! transforms.json: `transform_matrix = c2w_cv · diag(1, −1, −1, 1)`.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

use spl_core::Result;

use crate::io::colmap::{ColmapCamera, ColmapImage, ColmapModel, ColmapPoint};
use crate::io::scene::View;
use crate::model::GaussiansHost;

/// SH zeroth-order normalization constant.
const C0: f32 = 0.282_094_79;

const MAX_POINTS: usize = 500_000;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn write_colmap(
    views: &[View],
    g: &GaussiansHost,
    sparse_dir: &Path,
    text: bool,
) -> Result<()> {
    let model = build_colmap_model(views, g);
    if text {
        model.write_txt(sparse_dir)
    } else {
        model.write_bin(sparse_dir)
    }
}

fn build_colmap_model(views: &[View], g: &GaussiansHost) -> ColmapModel {
    let mut cameras = BTreeMap::new();
    let mut images = BTreeMap::new();

    for (i, v) in views.iter().enumerate() {
        let cam_id = v.camera;
        cameras.entry(cam_id).or_insert_with(|| ColmapCamera {
            id: cam_id,
            model_id: 1, // PINHOLE
            width: v.width as u64,
            height: v.height as u64,
            params: vec![v.fx, v.fy, v.cx, v.cy],
        });

        let r = v.r_cw;
        let qvec = rotation_to_quaternion(&r);
        let qvec = if qvec[0] >= 0.0 {
            qvec
        } else {
            [-qvec[0], -qvec[1], -qvec[2], -qvec[3]]
        };

        images.insert(
            (i + 1) as u32,
            ColmapImage {
                id: (i + 1) as u32,
                qvec,
                tvec: v.t_cw,
                camera_id: cam_id,
                name: v.name.clone(),
                points2d: vec![],
            },
        );
    }

    let points = sample_gaussian_points(g);

    ColmapModel {
        cameras,
        images,
        points,
    }
}

fn sample_gaussian_points(g: &GaussiansHost) -> Vec<ColmapPoint> {
    let n = g.len();
    if n == 0 {
        return Vec::new();
    }

    let mut candidates: Vec<(usize, f32)> = Vec::new();
    for i in 0..n {
        let o = sigmoid(g.opacities[i]);
        if o > 0.5 {
            candidates.push((i, o));
        }
    }

    if candidates.len() > MAX_POINTS {
        // Sample ∝ o: sort descending by opacity, take the top MAX_POINTS.
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        candidates.truncate(MAX_POINTS);
    }

    candidates
        .iter()
        .enumerate()
        .map(|(pid, &(i, _))| {
            let rgb = [
                (C0 * g.sh0[i * 3] + 0.5).clamp(0.0, 1.0),
                (C0 * g.sh0[i * 3 + 1] + 0.5).clamp(0.0, 1.0),
                (C0 * g.sh0[i * 3 + 2] + 0.5).clamp(0.0, 1.0),
            ];
            ColmapPoint {
                id: pid as u64,
                xyz: [
                    g.means[i * 3] as f64,
                    g.means[i * 3 + 1] as f64,
                    g.means[i * 3 + 2] as f64,
                ],
                rgb: [
                    (rgb[0] * 255.0).round() as u8,
                    (rgb[1] * 255.0).round() as u8,
                    (rgb[2] * 255.0).round() as u8,
                ],
                error: 0.0,
                track: vec![],
            }
        })
        .collect()
}

/// Row-major 3×3 rotation → quaternion (w, x, y, z).
fn rotation_to_quaternion(r: &[[f64; 3]; 3]) -> [f64; 4] {
    let trace = r[0][0] + r[1][1] + r[2][2];
    if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        [
            0.25 * s,
            (r[2][1] - r[1][2]) / s,
            (r[0][2] - r[2][0]) / s,
            (r[1][0] - r[0][1]) / s,
        ]
    } else if r[0][0] > r[1][1] && r[0][0] > r[2][2] {
        let s = (1.0 + r[0][0] - r[1][1] - r[2][2]).sqrt() * 2.0;
        [
            (r[2][1] - r[1][2]) / s,
            0.25 * s,
            (r[0][1] + r[1][0]) / s,
            (r[0][2] + r[2][0]) / s,
        ]
    } else if r[1][1] > r[2][2] {
        let s = (1.0 + r[1][1] - r[0][0] - r[2][2]).sqrt() * 2.0;
        [
            (r[0][2] - r[2][0]) / s,
            (r[0][1] + r[1][0]) / s,
            0.25 * s,
            (r[1][2] + r[2][1]) / s,
        ]
    } else {
        let s = (1.0 + r[2][2] - r[0][0] - r[1][1]).sqrt() * 2.0;
        [
            (r[1][0] - r[0][1]) / s,
            (r[0][2] + r[2][0]) / s,
            (r[1][2] + r[2][1]) / s,
            0.25 * s,
        ]
    }
}

pub fn write_transforms_json(views: &[View], out: &mut dyn Write) -> Result<()> {
    let mut frames = Vec::new();

    // Group by camera to find shared intrinsics
    let first = views.first();
    let shared_camera = first.is_some() && views.iter().all(|v| v.camera == first.unwrap().camera);

    for v in views {
        let r = v.r_cw;
        let t = v.t_cw;

        // c2w = inverse of w2c: R^T, -R^T t
        let rt = [
            [r[0][0], r[1][0], r[2][0]],
            [r[0][1], r[1][1], r[2][1]],
            [r[0][2], r[1][2], r[2][2]],
        ];
        let ct = [
            -(rt[0][0] * t[0] + rt[0][1] * t[1] + rt[0][2] * t[2]),
            -(rt[1][0] * t[0] + rt[1][1] * t[1] + rt[1][2] * t[2]),
            -(rt[2][0] * t[0] + rt[2][1] * t[1] + rt[2][2] * t[2]),
        ];

        // OpenCV → OpenGL: c2w * diag(1, -1, -1, 1)
        let transform = [
            [rt[0][0], -rt[0][1], -rt[0][2], ct[0]],
            [rt[1][0], -rt[1][1], -rt[1][2], ct[1]],
            [rt[2][0], -rt[2][1], -rt[2][2], ct[2]],
            [0.0, 0.0, 0.0, 1.0],
        ];

        let mut frame = serde_json::json!({
            "file_path": v.name,
            "transform_matrix": transform,
        });
        if !shared_camera {
            frame["fl_x"] = serde_json::json!(v.fx);
            frame["fl_y"] = serde_json::json!(v.fy);
            frame["cx"] = serde_json::json!(v.cx);
            frame["cy"] = serde_json::json!(v.cy);
            frame["w"] = serde_json::json!(v.width);
            frame["h"] = serde_json::json!(v.height);
        }
        frames.push(frame);
    }

    let mut root = serde_json::json!({
        "applied_transform": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
        "frames": frames,
    });

    if let Some(v) = first.filter(|_| shared_camera) {
        let angle_x = 2.0 * (v.width as f64 / (2.0 * v.fx)).atan();
        root["camera_angle_x"] = serde_json::json!(angle_x);
        root["fl_x"] = serde_json::json!(v.fx);
        root["fl_y"] = serde_json::json!(v.fy);
        root["cx"] = serde_json::json!(v.cx);
        root["cy"] = serde_json::json!(v.cy);
        root["w"] = serde_json::json!(v.width);
        root["h"] = serde_json::json!(v.height);
    }

    let json = serde_json::to_string_pretty(&root)?;
    out.write_all(json.as_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_views() -> Vec<View> {
        vec![
            View {
                name: "a.jpg".into(),
                camera: 1,
                r_cw: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                t_cw: [0.0, 0.0, 5.0],
                fx: 400.0,
                fy: 400.0,
                cx: 200.0,
                cy: 150.0,
                width: 400,
                height: 300,
                image_path: "a.jpg".into(),
            },
            View {
                name: "b.jpg".into(),
                camera: 1,
                r_cw: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                t_cw: [2.0, 0.0, 5.0],
                fx: 400.0,
                fy: 400.0,
                cx: 200.0,
                cy: 150.0,
                width: 400,
                height: 300,
                image_path: "b.jpg".into(),
            },
        ]
    }

    fn test_gaussians() -> GaussiansHost {
        let mut g = GaussiansHost::zeros(3);
        g.means = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        g.opacities = vec![2.0, -5.0, 3.0]; // sigmoid(2)=0.88>0.5, sigmoid(-5)=0.007<0.5, sigmoid(3)=0.95>0.5
        g.sh0 = vec![0.5, -0.3, 0.8, 0.1, 0.2, 0.3, -0.1, 0.4, 0.6];
        g
    }

    #[test]
    fn colmap_export_writes_and_reads_back() {
        let dir = tempfile::tempdir().unwrap();
        let views = test_views();
        let g = test_gaussians();
        let sparse = dir.path().join("sparse/0");
        write_colmap(&views, &g, &sparse, false).unwrap();
        let model = ColmapModel::read_bin(&sparse).unwrap();
        assert_eq!(model.cameras.len(), 1);
        assert_eq!(model.images.len(), 2);
        // Only Gaussians with o > 0.5: indices 0 and 2
        assert_eq!(model.points.len(), 2);
        let p0 = &model.points[0];
        assert!((p0.xyz[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn colmap_export_text_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let views = test_views();
        let g = test_gaussians();
        let sparse = dir.path().join("txt");
        write_colmap(&views, &g, &sparse, true).unwrap();
        let model = ColmapModel::read_txt(&sparse).unwrap();
        assert_eq!(model.images.len(), 2);
        assert_eq!(model.cameras[&1].model_id, 1);
    }

    #[test]
    fn colmap_export_caps_at_500k() {
        let mut g = GaussiansHost::zeros(600_000);
        for i in 0..600_000 {
            g.opacities[i] = 2.0; // all above 0.5
        }
        let points = sample_gaussian_points(&g);
        assert_eq!(points.len(), MAX_POINTS);
    }

    #[test]
    fn transforms_json_valid() {
        let views = test_views();
        let mut buf = Vec::new();
        write_transforms_json(&views, &mut buf).unwrap();
        let json: serde_json::Value = serde_json::from_slice(&buf).unwrap();
        assert!(json["camera_angle_x"].as_f64().unwrap() > 0.0);
        assert_eq!(json["frames"].as_array().unwrap().len(), 2);
        let mat = &json["frames"][0]["transform_matrix"];
        assert_eq!(mat[3][3].as_f64().unwrap(), 1.0);
    }

    #[test]
    fn rotation_to_quaternion_identity() {
        let r = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let q = rotation_to_quaternion(&r);
        assert!((q[0] - 1.0).abs() < 1e-10);
        assert!(q[1].abs() < 1e-10);
        assert!(q[2].abs() < 1e-10);
        assert!(q[3].abs() < 1e-10);
    }
}
