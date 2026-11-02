//! GPU tests of the geometry renderer, its gradients and the mesher
//! (`#[ignore]`: run on the GPU host). Owner: mesh-mode.

use super::*;
use crate::frame::{CameraTable, FrameParams, IterParams};
use crate::kernels::layout::EPS2D_DEFAULT;
use crate::kernels::raster::{FrameSource, GradTargets, SplatRaster};
use crate::kernels::{GradSink, RenderMode};
use crate::mesh::pipeline::{self, MeshConfig};
use crate::model::GaussiansHost;
use crate::testkit::synthetic::{SyntheticOptions, synthetic_scene};

/// `n` opaque flat discs tiling the unit sphere (Fibonacci lattice), each
/// with its shortest axis along the radius.
fn sphere_cloud(n: usize) -> GaussiansHost {
    let mut g = GaussiansHost::zeros(n);
    let spacing = (4.0 * std::f32::consts::PI / n as f32).sqrt();
    let golden = std::f32::consts::PI * (3.0 - 5f32.sqrt());
    for i in 0..n {
        let z = 1.0 - 2.0 * (i as f32 + 0.5) / n as f32;
        let r = (1.0 - z * z).sqrt();
        let phi = golden * i as f32;
        let p = [r * phi.cos(), r * phi.sin(), z];
        g.means[3 * i..3 * i + 3].copy_from_slice(&p);
        g.scales[3 * i..3 * i + 3].copy_from_slice(&[spacing.ln(), spacing.ln(), (1e-3f32).ln()]);
        // Rotation taking +z to p: q = (1 + p_z, −p_y, p_x, 0), normalised.
        let q = [1.0 + p[2] + 1e-6, -p[1], p[0], 0.0];
        let l = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        for (k, qk) in q.iter().enumerate() {
            g.rotations[4 * i + k] = qk / l;
        }
        g.opacities[i] = 5.0;
        g.sh0[3 * i..3 * i + 3].copy_from_slice(&[0.5, 0.2, -0.3]);
    }
    g.sh_degree = 0;
    g
}

fn eager_params(view: usize, n: u32, w: u32, h: u32) -> IterParams {
    IterParams {
        frame: FrameParams {
            iteration: 1,
            cam_index: view as u32,
            n_gaussians: n,
            sh_degree: 0,
            eps2d: EPS2D_DEFAULT,
            render_w: w,
            render_h: h,
            box_beta: 1.0,
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
#[ignore = "GPU: run on the GPU host"]
fn gpu_geom_sphere_depth_normals_and_mesh() {
    let gpu = Gpu::open().unwrap();
    let opts = SyntheticOptions {
        width: 320,
        height: 240,
        views: 12,
        ..Default::default()
    };
    let views = synthetic_scene(&opts).views;
    let cloud = sphere_cloud(40_000);

    // Median depth and normal at the centre pixel of view 0: the ray hits the
    // unit sphere at distance |eye| − 1 (the camera looks at the origin).
    let n = cloud.len();
    let cap = n.next_multiple_of(1024);
    let mut g = Gaussians::alloc(&gpu, cap, false).unwrap();
    g.upload(&gpu, &cloud).unwrap();
    let cams = CameraTable::upload(&gpu, &views).unwrap();
    let v = RasterVariant::default();
    let mut raster = GpuRaster::new(
        &gpu,
        Capacity::new(cap as u32, 1 << 22, opts.width, opts.height, &v),
        v,
        false,
    )
    .unwrap();
    let k = GeomKernels::load(&gpu, &v).unwrap();
    let mut b = GeomBuffers::alloc(&gpu, raster.frame.cap, false).unwrap();
    let params = eager_params(0, n as u32, opts.width, opts.height);
    raster
        .forward(
            &gpu,
            &mut g,
            &cams,
            FrameSource::Eager(&params),
            RenderMode::Infer,
        )
        .unwrap();
    assert_eq!(raster.frame.iter.read_status(&gpu).unwrap().overflow, 0);
    let gp = GeomParams {
        near_m: 1.0,
        v_cap: raster.frame.cap.visible,
        ..Default::default()
    };
    render(&gpu, &k, &raster, &g, &mut b, gp).unwrap();
    let out = gpu.stream.clone_dtoh(&b.out).unwrap();
    let p = (opts.width * opts.height) as usize;
    let c = (opts.height as usize / 2) * opts.width as usize + opts.width as usize / 2;
    let eye = (opts.ring_radius.powi(2) + opts.ring_height.powi(2)).sqrt() as f32;
    let (zmed, alpha) = (out[CH_MEDIAN * p + c], out[CH_ALPHA * p + c]);
    let nrm = [4, 5, 6].map(|ch| out[ch * p + c] / alpha.max(1e-6));
    let zexp = out[CH_EXPECTED * p + c] / alpha.max(1e-6);
    eprintln!(
        "centre: median {zmed:.5} expected {zexp:.5} (truth {:.5}) alpha {alpha:.4} normal {nrm:?}",
        eye - 1.0
    );
    assert!((zmed - (eye - 1.0)).abs() < 5e-3, "median depth {zmed}");
    assert!(alpha > 0.99);
    // The surface faces the camera: camera-space normal ≈ (0, 0, −1) at the centre.
    assert!(nrm[2] < -0.98, "normal {nrm:?}");
    let valid = out[..p].iter().filter(|&&z| z > 0.0).count();
    eprintln!("median depth valid on {valid}/{p} pixels");

    // Fusion + marching cubes: every vertex within a voxel of the sphere.
    let cfg = MeshConfig {
        bounds: Some([-1.2, -1.2, -1.2, 1.2, 1.2, 1.2]),
        resolution: 256,
        keep_components: 1,
        target_triangles: 20_000,
        texture: false,
        atlas_size: 1024,
        ..Default::default()
    };
    let res = pipeline::build(&gpu, &cloud, &views, &cfg).unwrap();
    let voxel = res.summary.voxel;
    let pts = res.raw.sample(20_000, 7);
    let err: Vec<f32> = pts
        .iter()
        .map(|q| (q.iter().map(|x| x * x).sum::<f32>().sqrt() - 1.0).abs())
        .collect();
    let mean = err.iter().sum::<f32>() / err.len() as f32;
    let max = err.iter().cloned().fold(0.0, f32::max);
    eprintln!(
        "sphere mesh: {:?}; |r − 1| mean {mean:.5} max {max:.5} (voxel {voxel:.5}), volume {:.4} (4π/3 = 4.18879)",
        res.summary,
        res.raw.signed_volume()
    );
    assert!(mean < 0.25 * voxel, "mean radial error {mean}");
    assert!(max < 2.0 * voxel, "max radial error {max}");
    // The ring never sees the poles, so the surface is open there.
    assert!(res.summary.kept_triangles as f32 > 0.95 * res.summary.raw_triangles as f32);
    let info = crate::mesh::glb::validate_glb(&res.glb).unwrap();
    assert_eq!(info.mesh_count, 1);
    if let Ok(dir) = std::env::var("SPL_GLB_OUT") {
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(std::path::Path::new(&dir).join("sphere.glb"), &res.glb).unwrap();
    }
}

/// The distortion gradient of the opacities against central differences of
/// the loss (the alpha chain is exact; the depth chain drops the plane terms,
/// so only opacities are compared).
#[test]
#[ignore = "GPU: run on the GPU host"]
fn gpu_geom_distortion_gradient_matches_finite_differences() {
    let gpu = Gpu::open().unwrap();
    let opts = SyntheticOptions {
        n: 256,
        width: 64,
        height: 48,
        views: 2,
        ..Default::default()
    };
    let scene = synthetic_scene(&opts);
    let mut host = scene.gaussians.clone();
    host.sh_degree = 0;
    // Semi-transparent, so no pixel reaches the T_MIN stop and no alpha the
    // 0.99 clamp (both are discontinuities central differences straddle).
    for o in &mut host.opacities {
        *o = 0.5 * *o - 1.5;
    }
    let n = host.len();
    let cap = n.next_multiple_of(1024);
    let mut g = Gaussians::alloc(&gpu, cap, false).unwrap();
    let cams = CameraTable::upload(&gpu, &scene.views).unwrap();
    let v = RasterVariant::default();
    let rcap = Capacity::new(cap as u32, 1 << 18, opts.width, opts.height, &v);
    let mut raster = GpuRaster::new(&gpu, rcap, v, false).unwrap();
    let cfg = GeometryConfig {
        lambda_dist: 1.0,
        lambda_normal: 0.0,
        lambda_flat: 0.0,
        near_rel: 1.0,
        dist_from: 0.0,
        normal_from: 0.0,
        flat_from: 0.0,
        plane_depth: false,
        lambda_mv_geo: 0.0,
        lambda_mv_ncc: 0.0,
        ..Default::default()
    };
    let mut geo = GeomTrain::new(&gpu, cfg, &v, rcap, 1.0, 100).unwrap();
    let params = eager_params(0, n as u32, opts.width, opts.height);
    let p = (opts.width * opts.height) as f64;

    let loss = |gpu: &Gpu, g: &mut Gaussians, raster: &mut GpuRaster, geo: &mut GeomTrain| {
        raster
            .forward(
                gpu,
                g,
                &cams,
                FrameSource::Eager(&params),
                RenderMode::Train,
            )
            .unwrap();
        geo.after_forward(gpu, raster, g, None).unwrap();
        let (d, _) = geo.take_sums(gpu, 1).unwrap();
        d / p
    };
    g.upload(&gpu, &host).unwrap();
    let l0 = loss(&gpu, &mut g, &mut raster, &mut geo);
    // Analytic: the colour gradient is zero, so g2d holds only the geometry terms.
    let mut grads = crate::model::Grads::alloc(&gpu, cap).unwrap();
    raster.pixel_grads.d_rgb = gpu
        .stream
        .alloc_zeros(raster.pixel_grads.d_rgb.len())
        .unwrap();
    raster
        .backward(
            &gpu,
            &mut g,
            GradTargets {
                sink: GradSink::Write(&mut grads),
                camera: None,
                stats: None,
            },
        )
        .unwrap();
    geo.after_backward(&gpu, &raster, &g, &mut grads).unwrap();
    let ga = gpu.stream.clone_dtoh(&grads.opacity).unwrap();
    eprintln!("distortion loss {l0:.6e}");
    assert!(l0 > 0.0);
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| ga[b].abs().total_cmp(&ga[a].abs()));
    // The loss is a difference of f32 moments (A·M2 − M1²): a large step keeps
    // its rounding below the change.
    let eps = 1e-1f32;
    let mut checked = 0;
    for &i in idx.iter().take(8) {
        let mut h = host.clone();
        h.opacities[i] += eps;
        g.upload(&gpu, &h).unwrap();
        let lp = loss(&gpu, &mut g, &mut raster, &mut geo);
        h.opacities[i] -= 2.0 * eps;
        g.upload(&gpu, &h).unwrap();
        let lm = loss(&gpu, &mut g, &mut raster, &mut geo);
        let fd = (lp - lm) / (2.0 * eps as f64);
        let an = ga[i] as f64;
        eprintln!("opacity[{i}]: analytic {an:.4e} fd {fd:.4e}");
        // Pixels crossing the 1/255 alpha cutoff are a jump the analytic
        // gradient (like the colour backward's) does not see.
        let rel = (an - fd).abs() / an.abs().max(fd.abs());
        assert!(rel < 0.2, "opacity[{i}]: analytic {an} vs fd {fd}");
        if rel < 0.1 {
            checked += 1;
        }
    }
    assert!(checked >= 6, "{checked}/8 opacities within 10%");
}

/// The multi-view gradients (plane depth → N, Ds → means) against central
/// differences: one term at a time, occlusion weighting off (`mv_max_err` 0),
/// view 1's depth cached first, then the loss of view 0 against it.
#[test]
#[ignore = "GPU: run on the GPU host"]
fn gpu_geom_multiview_gradients_match_finite_differences() {
    use crate::io::images::{ImageCache, Rgb8Image};
    let gpu = Gpu::open().unwrap();
    let opts = SyntheticOptions {
        width: 160,
        height: 120,
        views: 24,
        ..Default::default()
    };
    let views = synthetic_scene(&opts).views;
    let host = sphere_cloud(6000);
    let n = host.len();
    let cap = n.next_multiple_of(1024);
    let mut g = Gaussians::alloc(&gpu, cap, false).unwrap();
    let cams = CameraTable::upload(&gpu, &views).unwrap();
    // Smooth texture, different per view (NCC stays differentiable).
    let images: Vec<Rgb8Image> = (0..2)
        .map(|k| {
            let mut im = Rgb8Image::new(opts.width, opts.height);
            for y in 0..opts.height {
                for x in 0..opts.width {
                    let (xf, yf) = (x as f32 + 7.0 * k as f32, y as f32);
                    let v = 128.0 + 90.0 * (0.21 * xf).sin() * (0.17 * yf + 0.3 * k as f32).cos();
                    let i = 3 * (y * opts.width + x) as usize;
                    im.data[i..i + 3].fill(v as u8);
                }
            }
            im
        })
        .collect();
    let cache = ImageCache::from_images(&gpu, vec![0, 1], &images, 0.0).unwrap();
    let v = RasterVariant::default();
    let rcap = Capacity::new(cap as u32, 1 << 20, opts.width, opts.height, &v);
    let mut raster = GpuRaster::new(&gpu, rcap, v, false).unwrap();
    for (name, lg, lp) in [("geometric", 1.0f32, 0.0f32), ("photometric", 0.0, 1.0)] {
        let cfg = GeometryConfig {
            lambda_dist: 0.0,
            lambda_normal: 0.0,
            lambda_flat: 0.0,
            lambda_mv_geo: lg,
            lambda_mv_ncc: lp,
            mv_from: 0.0,
            mv_stride: 1,
            mv_max_err: 0.0,
            ncc_min_var: 0.0,
            ..Default::default()
        };
        let mut geo = GeomTrain::new(&gpu, cfg, &v, rcap, 1.0, 100).unwrap();
        geo.enable_multiview(&gpu, &views[..2], &cache).unwrap();
        let run = |gpu: &Gpu,
                   g: &mut Gaussians,
                   raster: &mut GpuRaster,
                   geo: &mut GeomTrain,
                   view: usize| {
            let params = eager_params(view, n as u32, opts.width, opts.height);
            raster
                .forward(
                    gpu,
                    g,
                    &cams,
                    FrameSource::Eager(&params),
                    RenderMode::Train,
                )
                .unwrap();
            geo.after_forward(gpu, raster, g, Some(&cache)).unwrap();
            let _ = geo.take_sums(gpu, 1).unwrap();
            let (lgeo, lpho, used) = geo.mv_sums;
            let scale = 1.0 / (opts.width * opts.height) as f64;
            scale * used * (lg as f64 * lgeo + lp as f64 * lpho)
        };
        // A slightly different cloud for the neighbour's cached depth.
        let mut other = host.clone();
        for m in other.means.iter_mut() {
            *m *= 1.003;
        }
        g.upload(&gpu, &other).unwrap();
        run(&gpu, &mut g, &mut raster, &mut geo, 1);
        g.upload(&gpu, &host).unwrap();
        let l0 = run(&gpu, &mut g, &mut raster, &mut geo, 0);
        let mut grads = crate::model::Grads::alloc(&gpu, cap).unwrap();
        raster.pixel_grads.d_rgb = gpu
            .stream
            .alloc_zeros(raster.pixel_grads.d_rgb.len())
            .unwrap();
        raster
            .backward(
                &gpu,
                &mut g,
                GradTargets {
                    sink: GradSink::Write(&mut grads),
                    camera: None,
                    stats: None,
                },
            )
            .unwrap();
        geo.after_backward(&gpu, &raster, &g, &mut grads).unwrap();
        let ga = gpu.stream.clone_dtoh(&grads.mean).unwrap();
        eprintln!("{name}: loss {l0:.6e}, mv pixels {:.0}", geo.mv_sums.2);
        assert!(l0 > 0.0);
        let mut idx: Vec<usize> = (0..3 * n).collect();
        idx.sort_by(|&a, &b| ga[b].abs().total_cmp(&ga[a].abs()));
        let eps = 2e-4f32;
        let mut close = 0;
        for &i in idx.iter().take(8) {
            let mut h = host.clone();
            h.means[i] += eps;
            g.upload(&gpu, &h).unwrap();
            let lp_ = run(&gpu, &mut g, &mut raster, &mut geo, 0);
            h.means[i] -= 2.0 * eps;
            g.upload(&gpu, &h).unwrap();
            let lm = run(&gpu, &mut g, &mut raster, &mut geo, 0);
            let fd = (lp_ - lm) / (2.0 * eps as f64);
            let an = ga[i] as f64;
            let rel = (an - fd).abs() / an.abs().max(fd.abs());
            eprintln!("{name} mean[{i}]: analytic {an:.4e} fd {fd:.4e} rel {rel:.3}");
            if rel < 0.2 {
                close += 1;
            }
        }
        assert!(close >= 6, "{name}: {close}/8 mean gradients within 20%");
    }
}
