//! Tiny deterministic scenes for kernel tests: Gaussians in a box, cameras on a
//! ring looking at it, all drawn from Philox so CPU references, GPU kernels and
//! the gsplat oracle see the same numbers. [`oracle_render_input`] converts a
//! scene into gs-oracle's `render` op inputs (activated values, gsplat layouts).

use spl_core::Result;

use super::npz::{NpyData, Npz};
use crate::io::scene::View;
use crate::model::GaussiansHost;
use crate::rng::{Philox, normal2, u01_open};

#[derive(Clone, Debug, PartialEq)]
pub struct SyntheticOptions {
    pub n: usize,
    pub width: u32,
    pub height: u32,
    pub views: usize,
    /// Coefficients above this degree are zero.
    pub sh_degree: u32,
    pub seed: u64,
    /// Means uniform in [−half_extent, half_extent]³.
    pub half_extent: f32,
    /// Scales log-uniform in [min, max].
    pub scale_range: (f32, f32),
    /// Camera ring radius and height (cameras look at the origin).
    pub ring_radius: f64,
    pub ring_height: f64,
}

impl Default for SyntheticOptions {
    /// The spec's tiny-scene shape: 256 Gaussians, 64×48, degree 3.
    fn default() -> Self {
        Self {
            n: 256,
            width: 64,
            height: 48,
            views: 4,
            sh_degree: 3,
            seed: 1,
            half_extent: 1.0,
            scale_range: (0.02, 0.2),
            ring_radius: 4.0,
            ring_height: 1.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SyntheticScene {
    pub gaussians: GaussiansHost,
    pub views: Vec<View>,
    pub opts: SyntheticOptions,
}

/// World-to-camera rotation (rows x, y, z) of a camera at `eye` looking at
/// `target`, OpenCV axes (x right, y down, z forward), world up = +z.
pub fn look_at(eye: [f64; 3], target: [f64; 3]) -> [[f64; 3]; 3] {
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let norm = |a: [f64; 3]| {
        let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        a.map(|x| x / l)
    };
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let z = norm(sub(target, eye));
    let up = [0.0, 0.0, 1.0];
    let d = up[0] * z[0] + up[1] * z[1] + up[2] * z[2];
    let y = norm([
        -(up[0] - d * z[0]),
        -(up[1] - d * z[1]),
        -(up[2] - d * z[2]),
    ]);
    let x = cross(y, z);
    [x, y, z]
}

pub fn synthetic_scene(opts: &SyntheticOptions) -> SyntheticScene {
    let rng = Philox::new(opts.seed);
    let n = opts.n;
    let mut g = GaussiansHost::zeros(n);
    g.sh_degree = opts.sh_degree;
    let (lo, hi) = (opts.scale_range.0.ln(), opts.scale_range.1.ln());
    let active = ((opts.sh_degree + 1) * (opts.sh_degree + 1) - 1) as usize;
    for i in 0..n {
        let e = i as u32;
        let a = rng.draw4(e, 0, 100);
        let b = rng.draw4(e, 1, 100);
        let c = rng.draw4(e, 2, 100);
        for k in 0..3 {
            g.means[3 * i + k] = (2.0 * u01_open(a[k]) - 1.0) * opts.half_extent;
            g.scales[3 * i + k] = lo + (hi - lo) * u01_open(b[k]);
        }
        let (q0, q1) = normal2(c[0], c[1]);
        let (q2, q3) = normal2(c[2], c[3]);
        g.rotations[4 * i..4 * i + 4].copy_from_slice(&[q0, q1, q2, q3]);
        let (o, _) = normal2(a[3], b[3]);
        g.opacities[i] = 1.5 * o;
        for k in 0..3 {
            let d = rng.draw4(e, 3 + k as u32, 100);
            g.sh0[3 * i + k] = normal2(d[0], d[1]).0 * 0.8;
        }
        for j in 0..active * 3 {
            let d = rng.draw4(e, 10 + j as u32, 100);
            g.sh_rest[45 * i + j] = normal2(d[0], d[1]).0 * 0.15;
        }
    }
    let views = (0..opts.views)
        .map(|v| {
            let angle = std::f64::consts::TAU * v as f64 / opts.views.max(1) as f64;
            let eye = [
                opts.ring_radius * angle.cos(),
                opts.ring_radius * angle.sin(),
                opts.ring_height,
            ];
            let r = look_at(eye, [0.0; 3]);
            let t =
                std::array::from_fn(|i| -(r[i][0] * eye[0] + r[i][1] * eye[1] + r[i][2] * eye[2]));
            let f = opts.width as f64;
            View {
                name: format!("synthetic_{v:03}.png"),
                camera: 1,
                r_cw: r,
                t_cw: t,
                fx: f,
                fy: f,
                cx: opts.width as f64 / 2.0,
                cy: opts.height as f64 / 2.0,
                width: opts.width,
                height: opts.height,
                image_path: Default::default(),
            }
        })
        .collect();
    SyntheticScene {
        gaussians: g,
        views,
        opts: opts.clone(),
    }
}

/// gsplat `viewmats` entry: world-to-camera 4×4, row-major.
pub fn view_matrix(v: &View) -> [f32; 16] {
    let r = &v.r_cw;
    let t = &v.t_cw;
    [
        r[0][0], r[0][1], r[0][2], t[0], //
        r[1][0], r[1][1], r[1][2], t[1], //
        r[2][0], r[2][1], r[2][2], t[2], //
        0.0, 0.0, 0.0, 1.0,
    ]
    .map(|x| x as f32)
}

/// gsplat `Ks` entry: pixel intrinsics 3×3, row-major.
pub fn k_matrix(v: &View) -> [f32; 9] {
    [v.fx, 0.0, v.cx, 0.0, v.fy, v.cy, 0.0, 0.0, 1.0].map(|x| x as f32)
}

/// gs-oracle `render` inputs for one view (ORACLE.md): activated scales and
/// opacities, raw quaternions, SH as `[N, K, 3]` with the DC term at index 0.
pub fn oracle_render_input(
    scene: &SyntheticScene,
    view: usize,
    eps2d: f32,
    antialiased: bool,
) -> Result<Npz> {
    let g = &scene.gaussians;
    let v = &scene.views[view];
    let n = g.len();
    let k = 16;
    let mut colors = vec![0f32; n * k * 3];
    for i in 0..n {
        colors[i * k * 3..i * k * 3 + 3].copy_from_slice(&g.sh0[3 * i..3 * i + 3]);
        colors[i * k * 3 + 3..(i + 1) * k * 3].copy_from_slice(&g.sh_rest[45 * i..45 * (i + 1)]);
    }
    let mut z = Npz::new();
    z.f32("means", &[n, 3], g.means.clone())?
        .f32("quats", &[n, 4], g.rotations.clone())?
        .f32(
            "scales",
            &[n, 3],
            g.scales.iter().map(|s| s.exp()).collect(),
        )?
        .f32(
            "opacities",
            &[n],
            g.opacities
                .iter()
                .map(|o| 1.0 / (1.0 + (-o).exp()))
                .collect(),
        )?
        .f32("colors", &[n, k, 3], colors)?
        .f32("viewmats", &[1, 4, 4], view_matrix(v).to_vec())?
        .f32("Ks", &[1, 3, 3], k_matrix(v).to_vec())?
        .f32("backgrounds", &[1, 3], vec![0.0; 3])?
        .i32_scalar("sh_degree", g.sh_degree as i32)?
        .i32_scalar("width", v.width as i32)?
        .i32_scalar("height", v.height as i32)?
        .f32_scalar("eps2d", eps2d)?
        .f32_scalar("near_plane", 0.01)?
        .f32_scalar("far_plane", 1e4)?
        .i32_scalar("rasterize_mode", i32::from(antialiased))?;
    z.insert("tile_size", &[], NpyData::I32(vec![16]))?;
    Ok(z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scenes_are_deterministic_and_visible() {
        let o = SyntheticOptions::default();
        let a = synthetic_scene(&o);
        let b = synthetic_scene(&o);
        assert_eq!(a.gaussians, b.gaussians);
        a.gaussians.validate().unwrap();
        assert_ne!(
            a.gaussians,
            synthetic_scene(&SyntheticOptions {
                seed: 2,
                ..o.clone()
            })
            .gaussians
        );
        // Every mean projects in front of every camera, near the image.
        for v in &a.views {
            for p in a.gaussians.means.chunks(3) {
                let pc: [f64; 3] = std::array::from_fn(|i| {
                    v.r_cw[i][0] * p[0] as f64
                        + v.r_cw[i][1] * p[1] as f64
                        + v.r_cw[i][2] * p[2] as f64
                        + v.t_cw[i]
                });
                assert!(pc[2] > 1.5, "depth {}", pc[2]);
                let u = v.fx * pc[0] / pc[2] + v.cx;
                assert!((-0.5 * v.width as f64..1.5 * v.width as f64).contains(&u));
            }
        }
        // Degree 1: only 3 of 15 coefficients are non-zero.
        let d1 = synthetic_scene(&SyntheticOptions {
            sh_degree: 1,
            n: 3,
            ..o
        });
        assert!(d1.gaussians.sh_rest[9..45].iter().all(|&x| x == 0.0));
        assert!(d1.gaussians.sh_rest[..9].iter().any(|&x| x != 0.0));
    }

    #[test]
    fn look_at_is_a_right_handed_opencv_frame() {
        let r = look_at([4.0, 0.0, 1.0], [0.0; 3]);
        let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
            - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
        assert!((det - 1.0).abs() < 1e-12);
        // World up (+z) maps to image up (−y in camera space).
        assert!(r[1][2] < 0.0);
    }

    #[test]
    fn oracle_input_has_gsplat_layouts() {
        let s = synthetic_scene(&SyntheticOptions {
            n: 5,
            ..Default::default()
        });
        let z = oracle_render_input(&s, 1, 0.3, false).unwrap();
        assert_eq!(z.get("colors").unwrap().shape, vec![5, 16, 3]);
        assert_eq!(z.get("viewmats").unwrap().shape, vec![1, 4, 4]);
        let o = z.get("opacities").unwrap().to_f64();
        assert!(o.iter().all(|&x| x > 0.0 && x < 1.0));
        let c = z.get("colors").unwrap().to_f64();
        assert_eq!(c[0] as f32, s.gaussians.sh0[0]);
        assert_eq!(c[3] as f32, s.gaussians.sh_rest[0]);
    }

    /// End to end through gsplat (runs where gs-oracle is installed: natively on
    /// the GPU host): our npz writer, the input conversion and the reader.
    #[test]
    #[ignore]
    fn oracle_renders_a_synthetic_scene() {
        let Some(oracle) = crate::testkit::oracle::Oracle::find() else {
            eprintln!("skipped: gs-oracle not found");
            return;
        };
        let s = synthetic_scene(&SyntheticOptions::default());
        let (w, h) = (s.opts.width as usize, s.opts.height as usize);
        for aa in [false, true] {
            let input = oracle_render_input(&s, 0, 0.3, aa).unwrap();
            let out = oracle.run("render", &input).unwrap();
            let rgb = out.get("render_colors").unwrap();
            assert_eq!(rgb.shape, vec![1, h, w, 3]);
            let alpha = out.get("render_alphas").unwrap().to_f64();
            let mean_alpha = alpha.iter().sum::<f64>() / alpha.len() as f64;
            let radii = out.get("radii").unwrap().to_f64();
            let visible = radii.chunks(2).filter(|r| r[0] > 0.0).count();
            println!(
                "gsplat render (aa {aa}): mean alpha {mean_alpha:.3}, {visible}/{} visible, keys {:?}",
                s.opts.n,
                out.0.keys().collect::<Vec<_>>()
            );
            assert!(alpha.iter().all(|a| (0.0..=1.0).contains(a)));
            assert!(mean_alpha > 0.05 && visible > s.opts.n / 2);
        }
    }
}
