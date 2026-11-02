//! GPU tests of Stage B and the depth prior (`#[ignore]`; run on the GPU host
//! with `apps/image-to-3d/scripts/gpu-test.sh <checkout> init::`).

use std::collections::HashMap;

use super::depth_prior::{DepthPrior, ssi_cpu};
use super::sources::{self, synthetic};
use super::*;
use crate::io::colmap::ColmapModel;
use crate::io::scene::{LoadOptions, find_sparse_dir, load_colmap_scene};
use crate::model::Gaussians;

fn run(
    gpu: &Gpu,
    k: &InitKernels,
    host: &DepthFramesHost,
    opts: &InitOptions,
) -> (InitReport, crate::model::GaussiansHost) {
    let frames = DepthFrames::upload(gpu, host).unwrap();
    let mut store = Gaussians::alloc(
        gpu,
        (opts.target as f32 * (1.0 + opts.tolerance)) as usize + 1024,
        false,
    )
    .unwrap();
    // Warm-up (NVRTC, first-touch), then the timed run.
    init_from_depth(gpu, k, &frames, opts, &mut store).unwrap();
    let rep = init_from_depth(gpu, k, &frames, opts, &mut store).unwrap();
    let g = store.download(gpu).unwrap();
    (rep, g)
}

fn print_report(tag: &str, frames: usize, rep: &InitReport) {
    println!(
        "{tag}: frames {frames} valid_px {} kept {} dropped {} voxels {} gaussians {} voxel {:.5} conf_thr {:.3} \
         passes {} | hist {:.2} ms search {:.2} ms final {:.2} ms total {:.2} ms",
        rep.valid_pixels,
        rep.kept_points,
        rep.dropped_points,
        rep.n_voxels,
        rep.n_gaussians,
        rep.voxel,
        rep.conf_threshold,
        rep.search.len(),
        rep.ms_hist,
        rep.ms_search,
        rep.ms_final,
        rep.ms_total
    );
    println!("  search (voxel, occupied): {:?}", rep.search);
}

fn check_params(g: &crate::model::GaussiansHost, opts: &InitOptions) {
    let n = g.len();
    let lo = logit(opts.opacity);
    for p in 0..n {
        assert_eq!(&g.rotations[4 * p..4 * p + 4], &[1.0, 0.0, 0.0, 0.0]);
        assert!((g.opacities[p] - lo).abs() < 1e-6);
        assert!(g.scales[3 * p..3 * p + 3].iter().all(|s| s.is_finite()));
        assert!(g.means[3 * p..3 * p + 3].iter().all(|s| s.is_finite()));
    }
    assert!(g.sh_rest.iter().all(|&x| x == 0.0));
}

#[test]
#[ignore]
fn gpu_init_synthetic_and_timings() {
    let gpu = Gpu::open().unwrap();
    let k = InitKernels::load(&gpu).unwrap();
    let (w, h) = (518u32, 392u32); // the engine's output size
    for n in [50usize, 200] {
        let host = synthetic::frames(n, w, h);
        let sky = sources::sky_fraction(&host);
        for target in [500_000u32, 1_000_000, 1_500_000] {
            let opts = InitOptions {
                target,
                conf_percentile: sky + 0.02,
                ..Default::default()
            };
            let (rep, g) = run(&gpu, &k, &host, &opts);
            print_report(&format!("synthetic target {target}"), n, &rep);
            check_params(&g, &opts);
            let got = rep.n_gaussians as f32 / target as f32;
            assert!(
                (got - 1.0).abs() <= opts.tolerance + 1e-3,
                "count {} vs {target}",
                rep.n_gaussians
            );
            // Sky removed, points on the surface (within a voxel), colour close to the albedo.
            let mut worst = 0f32;
            let mut col_err = 0f64;
            for p in 0..g.len() {
                let m = [g.means[3 * p], g.means[3 * p + 1], g.means[3 * p + 2]];
                worst = worst.max(synthetic::surface_distance(m));
                let a = synthetic::albedo(m);
                for (c, &ac) in a.iter().enumerate() {
                    let v = g.sh0[3 * p + c] * 0.282_094_8 + 0.5;
                    col_err += (v as f64 - ac as f64 / 255.0).abs();
                }
            }
            col_err /= 3.0 * g.len() as f64;
            let mean_scale: f64 =
                g.scales.iter().map(|&s| (s as f64).exp()).sum::<f64>() / g.scales.len() as f64;
            println!(
                "  max surface distance {worst:.5} (voxel {:.5}), mean |colour error| {col_err:.4}, mean scale {mean_scale:.5}",
                rep.voxel
            );
            assert!(
                worst < 1.0 * rep.voxel,
                "a Gaussian lies {worst} from the surface"
            );
            assert!(col_err < 0.03, "colour error {col_err}");
        }
    }
    // kNN scale mode: within a factor of a few of the voxel size.
    let host = synthetic::frames(50, w, h);
    let opts = InitOptions {
        target: 500_000,
        conf_percentile: sources::sky_fraction(&host) + 0.02,
        scale_mode: ScaleMode::Knn,
        ..Default::default()
    };
    let (rep, g) = run(&gpu, &k, &host, &opts);
    let med = {
        let mut s: Vec<f32> = g.scales.iter().step_by(3).map(|x| x.exp()).collect();
        s.sort_by(f32::total_cmp);
        s[s.len() / 2]
    };
    println!("knn: median scale {med:.5}, voxel {:.5}", rep.voxel);
    assert!(med > 0.5 * rep.voxel && med < 2.5 * rep.voxel);
}

#[test]
#[ignore]
fn gpu_init_mip360_colmap_depth() {
    let Some(root) = crate::testkit::mip360_scene("garden") else {
        println!("Mip-NeRF 360 garden not found; skipped");
        return;
    };
    let gpu = Gpu::open().unwrap();
    let k = InitKernels::load(&gpu).unwrap();
    let opts_scene = LoadOptions {
        images_dir: "images_4".into(),
        max_width: Some(648),
        ..Default::default()
    };
    let scene = load_colmap_scene(&root, &opts_scene).unwrap();
    let model = ColmapModel::read_dir(&find_sparse_dir(&root).unwrap()).unwrap();
    for n in [50usize, scene.train.len().min(200)] {
        let views: Vec<_> = scene
            .train
            .iter()
            .take(n)
            .map(|&i| &scene.views[i])
            .collect();
        let t = std::time::Instant::now();
        let images = sources::load_images(&views).unwrap();
        let host = sources::colmap_sparse_depth(&model, &views, &images);
        println!(
            "garden: {n} views {}x{} prepared in {:.0} ms",
            host.width,
            host.height,
            t.elapsed().as_secs_f64() * 1e3
        );
        // Sparse input: far fewer points than 0.5M, so the voxel stays at its
        // floor and every confident point survives (the dedup check below).
        let opts = InitOptions {
            target: 500_000,
            conf_percentile: 0.1,
            ..Default::default()
        };
        let (rep, g) = run(&gpu, &k, &host, &opts);
        print_report("garden colmap-depth", n, &rep);
        check_params(&g, &opts);
        assert!(rep.n_gaussians > 10_000);
        // Every Gaussian is near a COLMAP point (sub-pixel unprojection error
        // plus the voxel merge).
        let cell = (rep.voxel * 4.0).max(rep.mean_footprint * 4.0);
        let key = |p: [f32; 3]| p.map(|x| (x / cell).floor() as i32);
        let mut grid: HashMap<[i32; 3], Vec<[f32; 3]>> = HashMap::new();
        for p in &model.points {
            let x = p.xyz.map(|v| v as f32);
            grid.entry(key(x)).or_default().push(x);
        }
        let mut far = 0usize;
        let mut sum = 0f64;
        for p in 0..g.len() {
            let m = [g.means[3 * p], g.means[3 * p + 1], g.means[3 * p + 2]];
            let c = key(m);
            let mut best = f32::INFINITY;
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if let Some(v) = grid.get(&[c[0] + dx, c[1] + dy, c[2] + dz]) {
                            for q in v {
                                let d = ((q[0] - m[0]).powi(2)
                                    + (q[1] - m[1]).powi(2)
                                    + (q[2] - m[2]).powi(2))
                                .sqrt();
                                best = best.min(d);
                            }
                        }
                    }
                }
            }
            if best > cell {
                far += 1;
            } else {
                sum += best as f64;
            }
        }
        let frac_far = far as f64 / g.len() as f64;
        println!(
            "  mean distance to the nearest COLMAP point {:.5} (cell {cell:.5}), {:.3}% farther than a cell",
            sum / (g.len() - far) as f64,
            100.0 * frac_far
        );
        assert!(frac_far < 0.01);
    }
}

#[test]
#[ignore]
fn gpu_depth_prior_matches_cpu() {
    let gpu = Gpu::open().unwrap();
    let k = InitKernels::load(&gpu).unwrap();
    let mut dp = DepthPrior::alloc(&gpu).unwrap();
    let s = &gpu.stream;
    for &n in &[1usize, 1000, 518 * 392, 1297 * 840] {
        let d: Vec<f32> = (0..n)
            .map(|i| 1.0 + ((i * 7919) % 1009) as f32 * 0.005)
            .collect();
        let prior: Vec<f32> = d
            .iter()
            .enumerate()
            .map(|(i, &x)| 0.4 * x + 2.0 + ((i * 31) % 17) as f32 * 0.01)
            .collect();
        let w: Vec<f32> = (0..n)
            .map(|i| {
                if i % 9 == 0 {
                    0.0
                } else {
                    1.0 + (i % 4) as f32 * 0.5
                }
            })
            .collect();
        let (fit, g) = ssi_cpu(&d, &prior, &w);
        let (dd, pd, wd) = (
            s.clone_htod(&d).unwrap(),
            s.clone_htod(&prior).unwrap(),
            s.clone_htod(&w).unwrap(),
        );
        let x = super::depth_prior::SsiInputs {
            depth: dd.as_view(),
            prior: pd.as_view(),
            weight: wd.as_view(),
        };
        let mut gd = s.clone_htod(&vec![0.25f32; n]).unwrap();
        dp.forward(&gpu, &k, &x).unwrap();
        let lambda = 0.1;
        dp.backward(&gpu, &k, &x, lambda, &mut gd.as_view_mut())
            .unwrap();
        let gf = dp.read_fit(&gpu).unwrap();
        let got = s.clone_dtoh(&gd).unwrap();
        let rel = |a: f32, b: f32| (a - b).abs() / (1e-6 + b.abs().max(1e-3));
        println!("n {n}: gpu {gf:?} cpu {fit:?}");
        if n > 1 {
            assert!(
                rel(gf.scale, fit.scale) < 1e-3 && rel(gf.shift, fit.shift) < 1e-3,
                "{gf:?} {fit:?}"
            );
            assert!((gf.loss - fit.loss).abs() < 1e-3 * fit.loss.max(1e-6) + 1e-7);
        }
        let max_g = g.iter().fold(0f32, |m, x| m.max(x.abs())) * lambda;
        for i in (0..n).step_by((n / 97).max(1)) {
            let want = 0.25 + lambda * g[i];
            assert!(
                (got[i] - want).abs() <= 1e-3 * max_g + 1e-7,
                "grad {i}: {} vs {want}",
                got[i]
            );
        }
        // Timing on an engine-sized image.
        if n == 518 * 392 {
            gpu.sync().unwrap();
            let t = std::time::Instant::now();
            for _ in 0..200 {
                dp.forward(&gpu, &k, &x).unwrap();
                dp.backward(&gpu, &k, &x, lambda, &mut gd.as_view_mut())
                    .unwrap();
            }
            gpu.sync().unwrap();
            let ms = t.elapsed().as_secs_f64() * 1e3 / 200.0;
            println!("depth prior fwd+bwd at 518x392: {ms:.4} ms");
        }
    }
}
