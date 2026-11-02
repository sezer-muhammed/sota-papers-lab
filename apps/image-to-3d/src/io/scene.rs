//! A COLMAP-posed scene as training sees it: views with intrinsics scaled to
//! the images on disk, the name-sorted every-8th split, scene extent and the
//! initial point cloud. Owner: trainer (E7), scene I/O. Spec: spec-train
//! 7.2, 8.2-8.6.
//!
//! Layout of a dataset directory (Mip-NeRF 360, Inria `tandt_db.zip`):
//! `<root>/sparse/0/{cameras,images,points3D}.{bin,txt}` and
//! `<root>/<images_dir>/<name>` (`images`, `images_2`, `images_4`, ...).

use std::path::{Path, PathBuf};

use anyhow::ensure;
use kiddo::SquaredEuclidean;
use rayon::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use spl_core::{Context, Result, bail};
use ts_rs::TS;

use super::colmap::{ColmapModel, model_name};
use crate::model::GaussiansHost;

/// Where the principal point comes from (spec-train 8.3).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export, export_to = "image-to-3d/")]
pub enum PrincipalPoint {
    /// COLMAP's (cx, cy), scaled with the image (benchmarks).
    #[default]
    Colmap,
    /// The image centre (3DGS's assumption).
    Centered,
}

fn default_images_dir() -> String {
    "images".into()
}

fn default_test_every() -> u32 {
    8
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct LoadOptions {
    /// Image folder under the dataset root: `images`, `images_2`, `images_4`, ...
    /// (Mip-NeRF 360 benchmarks: `images_4` outdoor, `images_2` indoor).
    #[serde(default = "default_images_dir")]
    pub images_dir: String,
    /// Hold out every n-th image of the name-sorted list (index % n == 0) for
    /// testing; 0 trains on everything.
    #[serde(default = "default_test_every")]
    pub test_every: u32,
    #[serde(default)]
    pub principal_point: PrincipalPoint,
    /// Downscale wider images to this width (area filter, intrinsics scaled
    /// alike). None for benchmarks; 1600 is the Inria rule for user data.
    #[serde(default)]
    #[ts(optional)]
    pub max_width: Option<u32>,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            images_dir: default_images_dir(),
            test_every: default_test_every(),
            principal_point: PrincipalPoint::Colmap,
            max_width: None,
        }
    }
}

/// One registered image, with intrinsics at the size it is trained at.
#[derive(Clone, Debug, PartialEq)]
pub struct View {
    pub name: String,
    /// COLMAP camera id (physical camera: focal sharing, PPISP).
    pub camera: u32,
    /// World to camera, row-major.
    pub r_cw: [[f64; 3]; 3],
    pub t_cw: [f64; 3],
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    /// Size the image is trained at (after any `max_width` downscale).
    pub width: u32,
    pub height: u32,
    pub image_path: PathBuf,
}

impl View {
    /// Camera centre C = −Rᵀ t.
    pub fn center(&self) -> [f64; 3] {
        let (r, t) = (&self.r_cw, &self.t_cw);
        std::array::from_fn(|j| -(r[0][j] * t[0] + r[1][j] * t[1] + r[2][j] * t[2]))
    }
}

/// The initial point cloud (COLMAP points3D).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InitPoints {
    pub xyz: Vec<[f32; 3]>,
    pub rgb: Vec<[u8; 3]>,
}

#[derive(Clone, Debug)]
pub struct Scene {
    pub root: PathBuf,
    /// Sorted by name.
    pub views: Vec<View>,
    /// Indices into `views`.
    pub train: Vec<usize>,
    pub test: Vec<usize>,
    pub points: InitPoints,
    /// r = 1.1 · max distance of a training camera centre from their mean.
    pub extent: f32,
    pub center: [f32; 3],
    /// Largest |principal point − image centre| over views, in pixels (logged).
    pub max_pp_offset: f32,
}

/// `<root>/sparse/0`, else `<root>/sparse`, else `<root>` itself.
pub fn find_sparse_dir(root: &Path) -> Result<PathBuf> {
    for dir in [
        root.join("sparse").join("0"),
        root.join("sparse"),
        root.to_path_buf(),
    ] {
        if dir.join("cameras.bin").is_file() || dir.join("cameras.txt").is_file() {
            return Ok(dir);
        }
    }
    bail!(
        "{} has no COLMAP model (expected sparse/0/cameras.bin or cameras.txt)",
        root.display()
    )
}

/// The dataset root inside an extracted upload: the shallowest directory with a
/// COLMAP model, searched at most `depth` levels down.
pub fn find_dataset_root(dir: &Path, depth: u32) -> Option<PathBuf> {
    if find_sparse_dir(dir).is_ok() && dir.join("sparse").is_dir() {
        return Some(dir.to_path_buf());
    }
    if depth == 0 {
        return None;
    }
    let mut subdirs: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    subdirs.sort();
    subdirs
        .into_iter()
        .find_map(|d| find_dataset_root(&d, depth - 1))
}

/// Loads a COLMAP-posed dataset: model, image sizes (headers only), split,
/// extent and points. Fails on non-pinhole cameras and missing images.
pub fn load_colmap_scene(root: &Path, opts: &LoadOptions) -> Result<Scene> {
    let sparse = find_sparse_dir(root)?;
    let model = ColmapModel::read_dir(&sparse)
        .with_context(|| format!("read COLMAP model in {}", sparse.display()))?;
    for c in model.cameras.values() {
        ensure!(
            c.pinhole().is_some(),
            "camera {} is {}; training needs PINHOLE or SIMPLE_PINHOLE: run `colmap image_undistorter` first",
            c.id,
            model_name(c.model_id).unwrap_or("an unknown model")
        );
    }
    ensure!(
        !model.images.is_empty(),
        "the COLMAP model has no registered images"
    );
    let images_dir = root.join(&opts.images_dir);
    ensure!(
        images_dir.is_dir(),
        "image folder {} does not exist",
        images_dir.display()
    );

    let mut images: Vec<_> = model.images.values().collect();
    images.sort_by(|a, b| a.name.cmp(&b.name));
    let views = images
        .par_iter()
        .map(|img| {
            let cam = &model.cameras[&img.camera_id];
            let [fx, fy, cx, cy] = cam.pinhole().expect("checked above");
            let path = images_dir.join(&img.name);
            let (w0, h0) = image::image_dimensions(&path)
                .with_context(|| format!("read the size of {}", path.display()))?;
            let (w, h) = match opts.max_width {
                Some(m) if w0 > m => (
                    m,
                    ((h0 as f64 * m as f64 / w0 as f64).round() as u32).max(1),
                ),
                _ => (w0, h0),
            };
            // Per-axis scaling from the COLMAP camera's size to the trained size (8.3).
            let (sx, sy) = (w as f64 / cam.width as f64, h as f64 / cam.height as f64);
            let (cx, cy) = match opts.principal_point {
                PrincipalPoint::Colmap => (cx * sx, cy * sy),
                PrincipalPoint::Centered => (w as f64 / 2.0, h as f64 / 2.0),
            };
            Ok(View {
                name: img.name.clone(),
                camera: img.camera_id,
                r_cw: img.rotation(),
                t_cw: img.tvec,
                fx: fx * sx,
                fy: fy * sy,
                cx,
                cy,
                width: w,
                height: h,
                image_path: path,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let (train, test) = split(views.len(), opts.test_every);
    ensure!(!train.is_empty(), "no training views left after the split");
    let (center, extent) = extent(train.iter().map(|&i| views[i].center()));
    let max_pp_offset = views
        .iter()
        .map(|v| {
            (v.cx - v.width as f64 / 2.0)
                .abs()
                .max((v.cy - v.height as f64 / 2.0).abs())
        })
        .fold(0.0, f64::max) as f32;
    let points = InitPoints {
        xyz: model
            .points
            .iter()
            .map(|p| p.xyz.map(|x| x as f32))
            .collect(),
        rgb: model.points.iter().map(|p| p.rgb).collect(),
    };
    Ok(Scene {
        root: root.to_path_buf(),
        views,
        train,
        test,
        points,
        extent,
        center,
        max_pp_offset,
    })
}

/// Name-sorted indices: test = `i % every == 0`, train = the rest (spec-train 7.2).
pub fn split(n: usize, every: u32) -> (Vec<usize>, Vec<usize>) {
    if every == 0 {
        return ((0..n).collect(), Vec::new());
    }
    (0..n).partition(|i| i % every as usize != 0)
}

/// Inria `getNerfppNorm`: centre = mean camera centre, r = 1.1 · max distance.
pub fn extent(centers: impl Iterator<Item = [f64; 3]>) -> ([f32; 3], f32) {
    let c: Vec<[f64; 3]> = centers.collect();
    if c.is_empty() {
        return ([0.0; 3], 1.0);
    }
    let n = c.len() as f64;
    let mean: [f64; 3] = std::array::from_fn(|k| c.iter().map(|p| p[k]).sum::<f64>() / n);
    let max = c
        .iter()
        .map(|p| {
            ((p[0] - mean[0]).powi(2) + (p[1] - mean[1]).powi(2) + (p[2] - mean[2]).powi(2)).sqrt()
        })
        .fold(0.0, f64::max);
    (mean.map(|x| x as f32), (1.1 * max) as f32)
}

/// Standard 3DGS init or MCMC's (spec-train 8.6).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum InitKind {
    #[default]
    Standard,
    Mcmc,
}

/// SH zeroth-order normalization constant: `1 / (2·sqrt(π))`.
const C0: f32 = 0.282_094_79;

/// Point cloud → Gaussians: SH0 from colour, log scale from the mean squared
/// distance to the 3 nearest neighbours (exact kNN with `kiddo`), identity
/// rotation, opacity logit(0.1) (MCMC: 0.5 and scales × 0.1).
pub fn init_gaussians(points: &InitPoints, kind: InitKind) -> Result<GaussiansHost> {
    let n = points.xyz.len();
    ensure!(n > 0, "point cloud is empty");

    let tree: kiddo::KdTree<f32, 3> = (&points.xyz).into();

    let dist2: Vec<f32> = (0..n)
        .into_par_iter()
        .map(|i| {
            let q = points.xyz[i];
            let nn = tree.nearest_n::<SquaredEuclidean>(&q, 4);
            let mut sum = 0.0f32;
            let mut count = 0u32;
            for entry in &nn {
                if entry.distance > 0.0 {
                    sum += entry.distance;
                    count += 1;
                    if count == 3 {
                        break;
                    }
                }
            }
            if count > 0 { sum / count as f32 } else { 1e-7 }
        })
        .collect();

    let mut means = Vec::with_capacity(n * 3);
    let mut sh0 = Vec::with_capacity(n * 3);
    let mut scales = Vec::with_capacity(n * 3);
    let mut rotations = Vec::with_capacity(n * 4);
    let mut opacities = Vec::with_capacity(n);

    let logit_01 = (0.1f32 / 0.9).ln();
    let logit_05 = 0.0f32;
    let logit_o = match kind {
        InitKind::Standard => logit_01,
        InitKind::Mcmc => logit_05,
    };
    let mcmc_scale_factor = match kind {
        InitKind::Standard => 0.0,
        InitKind::Mcmc => (0.1f32).ln(),
    };

    for (i, (&xyz, &rgb)) in points.xyz.iter().zip(&points.rgb).enumerate() {
        means.extend_from_slice(&xyz);

        let [r, g, b] = rgb;
        sh0.push((r as f32 / 255.0 - 0.5) / C0);
        sh0.push((g as f32 / 255.0 - 0.5) / C0);
        sh0.push((b as f32 / 255.0 - 0.5) / C0);

        let s = (dist2[i].max(1e-7).sqrt()).ln() + mcmc_scale_factor;
        scales.extend_from_slice(&[s, s, s]);

        rotations.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);

        opacities.push(logit_o);
    }

    let g = GaussiansHost {
        means,
        scales,
        rotations,
        opacities,
        sh0,
        sh_rest: vec![0.0; n * 45],
        sh_degree: 3,
        antialiased: false,
    };
    g.validate()?;
    Ok(g)
}

#[cfg(test)]
mod tests {
    use super::super::colmap::{ColmapCamera, ColmapImage, ColmapPoint};
    use super::*;

    #[test]
    fn split_holds_out_every_eighth_name() {
        let (train, test) = split(17, 8);
        assert_eq!(test, [0, 8, 16]);
        assert_eq!(train.len(), 14);
        assert_eq!(split(3, 0).1, Vec::<usize>::new());
    }

    #[test]
    fn extent_is_inria_nerfpp_norm() {
        let (c, r) = extent([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]].into_iter());
        assert_eq!(c, [1.0, 0.0, 0.0]);
        assert!((r - 1.1).abs() < 1e-6);
    }

    /// A 2-camera dataset on disk with downscaled images (like `images_4`).
    fn write_dataset(dir: &Path, model_id: i32) {
        let mut m = ColmapModel::default();
        let params = if model_id == 1 {
            vec![400.0, 410.0, 200.0, 150.0]
        } else {
            vec![400.0; 8]
        };
        m.cameras.insert(
            1,
            ColmapCamera {
                id: 1,
                model_id,
                width: 400,
                height: 300,
                params,
            },
        );
        for (id, name, tx) in [(1, "b.png", 0.0), (2, "a.png", 2.0), (3, "c.png", 4.0)] {
            m.images.insert(
                id,
                ColmapImage {
                    id,
                    qvec: [1.0, 0.0, 0.0, 0.0],
                    tvec: [tx, 0.0, 0.0],
                    camera_id: 1,
                    name: name.into(),
                    points2d: vec![],
                },
            );
        }
        m.points.push(ColmapPoint {
            id: 1,
            xyz: [0.5, 1.0, 2.0],
            rgb: [10, 20, 30],
            error: 0.1,
            track: vec![],
        });
        m.write_bin(&dir.join("sparse/0")).unwrap();
        std::fs::create_dir_all(dir.join("images_4")).unwrap();
        for name in ["a.png", "b.png", "c.png"] {
            image::RgbImage::new(100, 75)
                .save(dir.join("images_4").join(name))
                .unwrap();
        }
    }

    #[test]
    fn loads_a_downscaled_pinhole_dataset() {
        let dir = tempfile::tempdir().unwrap();
        write_dataset(dir.path(), 1);
        let opts = LoadOptions {
            images_dir: "images_4".into(),
            test_every: 2,
            ..Default::default()
        };
        let s = load_colmap_scene(dir.path(), &opts).unwrap();
        assert_eq!(
            s.views.iter().map(|v| v.name.as_str()).collect::<Vec<_>>(),
            ["a.png", "b.png", "c.png"]
        );
        assert_eq!((s.train.clone(), s.test.clone()), (vec![1], vec![0, 2]));
        let v = &s.views[0];
        assert_eq!((v.width, v.height), (100, 75));
        assert_eq!((v.fx, v.fy, v.cx, v.cy), (100.0, 102.5, 50.0, 37.5));
        // a.png is image 2 with t = (2, 0, 0): C = (-2, 0, 0).
        assert_eq!(v.center(), [-2.0, 0.0, 0.0]);
        assert_eq!(s.points.xyz, vec![[0.5, 1.0, 2.0]]);
        assert_eq!(s.max_pp_offset, 0.0);

        let capped = LoadOptions {
            max_width: Some(50),
            ..opts.clone()
        };
        let s = load_colmap_scene(dir.path(), &capped).unwrap();
        assert_eq!(
            (s.views[0].width, s.views[0].height, s.views[0].fx),
            (50, 38, 50.0)
        );
        assert!(
            load_colmap_scene(dir.path(), &LoadOptions::default()).is_err(),
            "no images/ folder"
        );
        assert_eq!(
            find_dataset_root(dir.path(), 0).as_deref(),
            Some(dir.path())
        );
    }

    #[test]
    fn finds_a_dataset_nested_in_an_upload() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("pack").join("garden");
        write_dataset(&nested, 1);
        std::fs::create_dir_all(dir.path().join("aaa-empty")).unwrap();
        assert_eq!(find_dataset_root(dir.path(), 2), Some(nested));
        assert_eq!(find_dataset_root(dir.path(), 1), None);
    }

    #[test]
    fn rejects_distorted_cameras() {
        let dir = tempfile::tempdir().unwrap();
        write_dataset(dir.path(), 4);
        let opts = LoadOptions {
            images_dir: "images_4".into(),
            ..Default::default()
        };
        let err = load_colmap_scene(dir.path(), &opts).unwrap_err();
        assert!(format!("{err:#}").contains("image_undistorter"), "{err:#}");
    }

    #[test]
    fn init_gaussians_standard() {
        let points = InitPoints {
            xyz: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [2.0, 0.0, 0.0],
            ],
            rgb: vec![
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
                [128, 128, 128],
                [255, 255, 255],
            ],
        };
        let g = init_gaussians(&points, InitKind::Standard).unwrap();
        assert_eq!(g.len(), 5);
        g.validate().unwrap();

        // Means match input positions
        for i in 0..5 {
            for a in 0..3 {
                assert_eq!(g.means[i * 3 + a], points.xyz[i][a]);
            }
        }
        // SH0 from colour: (r/255 - 0.5) / C0
        let expected_sh0_r = (255.0 / 255.0 - 0.5) / C0;
        assert!(
            (g.sh0[0] - expected_sh0_r).abs() < 1e-5,
            "sh0[0]={}",
            g.sh0[0]
        );
        // SH rest is zero
        assert!(g.sh_rest.iter().all(|&x| x == 0.0));
        // Rotation is identity
        for i in 0..5 {
            assert_eq!(g.rotations[i * 4], 1.0);
            assert_eq!(g.rotations[i * 4 + 1], 0.0);
        }
        // Opacity is logit(0.1)
        let logit_01 = (0.1f32 / 0.9).ln();
        for i in 0..5 {
            assert!((g.opacities[i] - logit_01).abs() < 1e-5);
        }
        // Scales are finite and negative (small Gaussians)
        for &s in &g.scales {
            assert!(s.is_finite() && s < 2.0, "scale {s}");
        }
    }

    #[test]
    fn init_gaussians_mcmc_differs() {
        let points = InitPoints {
            xyz: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            rgb: vec![[100; 3]; 3],
        };
        let std = init_gaussians(&points, InitKind::Standard).unwrap();
        let mcmc = init_gaussians(&points, InitKind::Mcmc).unwrap();
        // MCMC: opacity logit(0.5) = 0
        assert!((mcmc.opacities[0]).abs() < 1e-5, "mcmc opacity should be 0");
        // MCMC scales are smaller (× 0.1 → ln(0.1) offset)
        assert!(
            mcmc.scales[0] < std.scales[0],
            "mcmc scale {} should be < std {}",
            mcmc.scales[0],
            std.scales[0]
        );
    }

    #[test]
    fn init_gaussians_single_point() {
        let points = InitPoints {
            xyz: vec![[5.0, 3.0, -1.0]],
            rgb: vec![[0, 0, 0]],
        };
        let g = init_gaussians(&points, InitKind::Standard).unwrap();
        assert_eq!(g.len(), 1);
        // With only one point, kNN returns no neighbours → d² = 1e-7
        let expected_scale = (1e-7f32).sqrt().ln();
        assert!(
            (g.scales[0] - expected_scale).abs() < 1e-3,
            "scale {} expected {}",
            g.scales[0],
            expected_scale
        );
    }
}
