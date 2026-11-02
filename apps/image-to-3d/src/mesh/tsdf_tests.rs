//! TSDF + marching cubes tests. CPU: the projection conventions. GPU
//! (`#[ignore]`, run with `scripts/gpu-test.sh <checkout> mesh`): synthetic
//! shapes ray-cast to depth maps, fused at 512³ effective resolution and
//! compared with the analytic surface (Chamfer); a Mip-NeRF 360 scene fused
//! from COLMAP points splatted into depth maps (scale and timings).

use std::time::Instant;

use rayon::prelude::*;

use super::*;
use crate::mesh::{Mesh, chamfer, mean_nn};
use crate::testkit::mip360_scene;

/// Exact signed distance functions.
#[derive(Clone, Copy, Debug)]
enum Shape {
    Sphere(f32),
    Torus(f32, f32),
    Box([f32; 3]),
}

impl Shape {
    fn sdf(self, p: [f32; 3]) -> f32 {
        match self {
            Shape::Sphere(r) => (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() - r,
            Shape::Torus(big, r) => {
                let q = (p[0] * p[0] + p[1] * p[1]).sqrt() - big;
                (q * q + p[2] * p[2]).sqrt() - r
            }
            Shape::Box(h) => {
                let q: [f32; 3] = std::array::from_fn(|k| p[k].abs() - h[k]);
                let o = q.map(|v| v.max(0.0));
                (o[0] * o[0] + o[1] * o[1] + o[2] * o[2]).sqrt() + q[0].max(q[1]).max(q[2]).min(0.0)
            }
        }
    }

    /// `n` points uniform on the surface.
    fn sample(self, n: usize) -> Vec<[f32; 3]> {
        let mut s = 0x2545_f491_4f6c_dd1du64;
        let mut u = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32
        };
        let tau = std::f32::consts::TAU;
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            match self {
                Shape::Sphere(r) => {
                    let z = 2.0 * u() - 1.0;
                    let a = tau * u();
                    let q = (1.0 - z * z).sqrt();
                    out.push([r * q * a.cos(), r * q * a.sin(), r * z]);
                }
                Shape::Torus(big, r) => {
                    let (a, b) = (tau * u(), tau * u());
                    // Accept by the area element (R + r cos b).
                    if u() * (big + r) <= big + r * b.cos() {
                        let q = big + r * b.cos();
                        out.push([q * a.cos(), q * a.sin(), r * b.sin()]);
                    }
                }
                Shape::Box(h) => {
                    let areas = [h[1] * h[2], h[0] * h[2], h[0] * h[1]];
                    let t = u() * (areas[0] + areas[1] + areas[2]);
                    let ax = if t < areas[0] {
                        0
                    } else if t < areas[0] + areas[1] {
                        1
                    } else {
                        2
                    };
                    let mut p: [f32; 3] = std::array::from_fn(|k| (2.0 * u() - 1.0) * h[k]);
                    p[ax] = if u() < 0.5 { -h[ax] } else { h[ax] };
                    out.push(p);
                }
            }
        }
        out
    }
}

/// `n` cameras on a Fibonacci sphere of radius `dist` looking at the origin.
fn orbit(n: usize, dist: f32, size: u32, fov_deg: f32) -> Vec<DepthCamera> {
    let f = 0.5 * size as f32 / (0.5 * fov_deg.to_radians()).tan();
    (0..n)
        .map(|i| {
            let z = 1.0 - 2.0 * (i as f32 + 0.5) / n as f32;
            let a = i as f32 * std::f32::consts::PI * (3.0 - 5f32.sqrt());
            let q = (1.0 - z * z).sqrt();
            let c = [dist * q * a.cos(), dist * q * a.sin(), dist * z];
            let fwd = c.map(|v| -v / dist);
            let up = if fwd[2].abs() > 0.9 {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 0.0, 1.0]
            };
            let norm = |v: [f32; 3]| {
                let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                v.map(|x| x / l)
            };
            let cr = |a: [f32; 3], b: [f32; 3]| {
                [
                    a[1] * b[2] - a[2] * b[1],
                    a[2] * b[0] - a[0] * b[2],
                    a[0] * b[1] - a[1] * b[0],
                ]
            };
            let right = norm(cr(fwd, up));
            let down = cr(fwd, right);
            let r = [right, down, fwd];
            let t = std::array::from_fn(|k| -(r[k][0] * c[0] + r[k][1] * c[1] + r[k][2] * c[2]));
            DepthCamera {
                r_cw: r,
                t_cw: t,
                fx: f,
                fy: f,
                cx: 0.5 * size as f32,
                cy: 0.5 * size as f32,
                width: size,
                height: size,
            }
        })
        .collect()
}

/// Sphere-traced z-depth (0 = miss).
fn render_depth(shape: Shape, cam: &DepthCamera) -> Vec<f32> {
    let r = cam.r_cw;
    let c: [f32; 3] = std::array::from_fn(|j| -(0..3).map(|k| r[k][j] * cam.t_cw[k]).sum::<f32>());
    (0..cam.width * cam.height)
        .into_par_iter()
        .map(|i| {
            let (u, v) = ((i % cam.width) as f32 + 0.5, (i / cam.width) as f32 + 0.5);
            let dc = [(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, 1.0];
            let d: [f32; 3] = std::array::from_fn(|j| (0..3).map(|k| r[k][j] * dc[k]).sum());
            let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let mut t = 0.0f32;
            for _ in 0..256 {
                let p = std::array::from_fn(|k| c[k] + t * d[k] / len);
                let s = shape.sdf(p);
                if s < 1e-6 {
                    return t / len;
                }
                t += s;
                if t > 10.0 {
                    break;
                }
            }
            0.0
        })
        .collect()
}

#[test]
fn camera_struct_matches_cuda_field_count() {
    let src = include_str!("../../cuda/mesh_tsdf.cu");
    let body = &src[src.find("struct TsdfCam {").unwrap()..];
    let body = &body[..body.find("};").unwrap()];
    // r[9] + t[3] + 4 intrinsics + 2 ints + 2 depth limits.
    assert!(body.contains("float r[9]") && body.contains("float t[3]"));
    assert_eq!(size_of::<TsdfCam>(), (9 + 3 + 4 + 2 + 2) * 4);
}

#[test]
fn orbit_cameras_see_the_origin_at_the_centre() {
    let shape = Shape::Sphere(0.3);
    for cam in orbit(7, 1.5, 64, 50.0) {
        let z = cam.t_cw[2];
        assert!((z - 1.5).abs() < 1e-5, "origin depth {z}");
        let d = render_depth(shape, &cam);
        assert!((d[32 * 64 + 32] - 1.2).abs() < 2e-3, "{}", d[32 * 64 + 32]);
    }
}

fn gpu() -> Gpu {
    Gpu::open().expect("open GPU")
}

struct Report {
    name: &'static str,
    views: usize,
    blocks: u32,
    integrate_ms_per_view: f64,
    extract_ms: f64,
    mesh: Mesh,
    chamfer_vox: f64,
    acc_vox: f64,
    comp_vox: f64,
    sdf_mean_vox: f64,
}

fn run_shape(gpu: &Gpu, name: &'static str, shape: Shape, views: usize) -> Report {
    let res = 512;
    let p = TsdfParams::for_bounds([-0.5; 3], 1.0, res);
    let cams = orbit(views, 1.6, 512, 45.0);
    let depths: Vec<Vec<f32>> = cams.iter().map(|c| render_depth(shape, c)).collect();
    let dev: Vec<_> = depths
        .iter()
        .map(|d| gpu.stream.clone_htod(d).unwrap())
        .collect();
    let mut vol = TsdfVolume::new(gpu, p).unwrap();
    gpu.sync().unwrap();
    let t0 = Instant::now();
    for (d, c) in dev.iter().zip(&cams) {
        vol.integrate_device(d, c).unwrap();
    }
    gpu.sync().unwrap();
    let integrate_ms_per_view = t0.elapsed().as_secs_f64() * 1e3 / views as f64;
    let opts = ExtractOptions::default();
    let mut mesh = vol.extract(opts).unwrap();
    let mut extract_ms = f64::MAX;
    for _ in 0..3 {
        let t = Instant::now();
        mesh = vol.extract(opts).unwrap();
        extract_ms = extract_ms.min(t.elapsed().as_secs_f64() * 1e3);
    }
    let v = p.voxel as f64;
    let n = 1_000_000;
    let ms = mesh.sample(n, 7);
    let gt = shape.sample(n);
    let (ch, acc, comp) = chamfer(&ms, &gt);
    let sdf_mean = ms.iter().map(|q| shape.sdf(*q).abs() as f64).sum::<f64>() / ms.len() as f64;
    Report {
        name,
        views,
        blocks: vol.blocks(),
        integrate_ms_per_view,
        extract_ms,
        mesh,
        chamfer_vox: ch / v,
        acc_vox: acc / v,
        comp_vox: comp / v,
        sdf_mean_vox: sdf_mean / v,
    }
}

#[test]
#[ignore]
fn gpu_mesh_synthetic_shapes_512() {
    let gpu = gpu();
    let shapes = [
        ("sphere", Shape::Sphere(0.3)),
        ("torus", Shape::Torus(0.28, 0.1)),
        ("box", Shape::Box([0.3, 0.2, 0.15])),
    ];
    for (name, shape) in shapes {
        let r = run_shape(&gpu, name, shape, 96);
        let m = &r.mesh;
        let open = m.boundary_or_nonmanifold_edges();
        println!(
            "{{\"shape\":\"{}\",\"res\":512,\"views\":{},\"blocks\":{},\"vertices\":{},\"faces\":{},\"open_edges\":{},\"integrate_ms_per_view\":{:.3},\"extract_ms\":{:.2},\"chamfer_vox\":{:.4},\"acc_vox\":{:.4},\"comp_vox\":{:.4},\"sdf_abs_mean_vox\":{:.4}}}",
            r.name,
            r.views,
            r.blocks,
            m.positions.len(),
            m.faces.len(),
            open,
            r.integrate_ms_per_view,
            r.extract_ms,
            r.chamfer_vox,
            r.acc_vox,
            r.comp_vox,
            r.sdf_mean_vox
        );
        assert!(
            r.chamfer_vox < 0.5,
            "{name}: Chamfer {} voxels",
            r.chamfer_vox
        );
        assert_eq!(open, 0, "{name}: open or non-manifold edges");
        assert!(m.signed_volume() > 0.0);
        // Normals agree with the face winding.
        let agree = m
            .faces
            .iter()
            .filter(|f| {
                let [a, b, c] = f.map(|i| m.positions[i as usize]);
                let n = crate::mesh::cross(crate::mesh::sub(b, a), crate::mesh::sub(c, a));
                let vn = m.normals[f[0] as usize];
                n[0] * vn[0] + n[1] * vn[1] + n[2] * vn[2] > 0.0
            })
            .count();
        assert!(
            agree as f64 > 0.99 * m.faces.len() as f64,
            "{name}: normals vs winding {agree}"
        );
    }
}

#[test]
#[ignore]
fn gpu_mesh_matches_cpu_reference() {
    // Same fused volume, extracted on the GPU and densely on the CPU.
    let gpu = gpu();
    let res = 128;
    let p = TsdfParams::for_bounds([-0.5; 3], 1.0, res);
    let shape = Shape::Torus(0.28, 0.1);
    let mut vol = TsdfVolume::new(&gpu, p).unwrap();
    for c in orbit(48, 1.6, 256, 45.0) {
        vol.integrate(&render_depth(shape, &c), &c).unwrap();
    }
    let mut g = vol.extract(ExtractOptions::default()).unwrap();
    // Densify the sparse volume on the host.
    let (tsdf, weight, coords) = vol.voxels();
    let tsdf = gpu.stream.clone_dtoh(tsdf).unwrap();
    let weight = gpu.stream.clone_dtoh(weight).unwrap();
    let coords = gpu.stream.clone_dtoh(coords).unwrap();
    let n = res as usize + 16;
    let off = 8i32;
    let mut dv = vec![0.0f32; n * n * n];
    let mut dw = vec![0.0f32; n * n * n];
    for b in 0..vol.blocks() as usize {
        for t in 0..512 {
            let g = [t & 7, t >> 3 & 7, t >> 6].map(|v| v as i32);
            let q: [i32; 3] = std::array::from_fn(|k| coords[3 * b + k] * 8 + g[k] + off);
            if q.iter().all(|&v| v >= 0 && (v as usize) < n) {
                let i = (q[2] as usize * n + q[1] as usize) * n + q[0] as usize;
                dv[i] = tsdf[b * 512 + t];
                dw[i] = weight[b * 512 + t];
            }
        }
    }
    let o = -0.5 - off as f32 * p.voxel;
    let mut c = mc::extract_dense(&dv, &dw, [n; 3], [o; 3], p.voxel, 0.99);
    c.remove_unreferenced();
    g.canonicalize();
    c.canonicalize();
    assert_eq!(g.faces.len(), c.faces.len());
    assert_eq!(g.positions.len(), c.positions.len());
    let d = mean_nn(&g.positions, &c.positions);
    assert!(d < 1e-5, "GPU vs CPU vertex distance {d}");
    println!(
        "gpu == cpu: {} faces, {} vertices",
        g.faces.len(),
        g.positions.len()
    );
}

#[test]
#[ignore]
fn gpu_mesh_mip360_colmap_depth() {
    use crate::io::scene::{LoadOptions, load_colmap_scene};
    let Some(root) = mip360_scene("garden") else {
        eprintln!("skipped: no mipnerf360/garden dataset");
        return;
    };
    let scene = load_colmap_scene(
        &root,
        &LoadOptions {
            images_dir: "images_4".into(),
            ..Default::default()
        },
    )
    .unwrap();
    // Depth: COLMAP points splatted (3×3, nearest wins) at 1/4 of the image
    // size — sparse, so the mesh is fragments; this measures scale and speed.
    let cams: Vec<DepthCamera> = scene
        .views
        .iter()
        .map(|v| {
            let s = 0.25;
            DepthCamera {
                r_cw: v.r_cw.map(|r| r.map(|x| x as f32)),
                t_cw: v.t_cw.map(|x| x as f32),
                fx: (v.fx * s) as f32,
                fy: (v.fy * s) as f32,
                cx: (v.cx * s) as f32,
                cy: (v.cy * s) as f32,
                width: (v.width as f64 * s) as u32,
                height: (v.height as f64 * s) as u32,
            }
        })
        .collect();
    let pts = &scene.points.xyz;
    let depths: Vec<Vec<f32>> = cams
        .par_iter()
        .map(|c| {
            let mut d = vec![f32::MAX; (c.width * c.height) as usize];
            for p in pts {
                let q: [f32; 3] = std::array::from_fn(|k| {
                    c.r_cw[k][0] * p[0] + c.r_cw[k][1] * p[1] + c.r_cw[k][2] * p[2] + c.t_cw[k]
                });
                if q[2] <= 1e-3 {
                    continue;
                }
                let u = (c.fx * q[0] / q[2] + c.cx) as i32;
                let v = (c.fy * q[1] / q[2] + c.cy) as i32;
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let (x, y) = (u + dx, v + dy);
                        if x >= 0 && y >= 0 && (x as u32) < c.width && (y as u32) < c.height {
                            let i = (y as u32 * c.width + x as u32) as usize;
                            d[i] = d[i].min(q[2]);
                        }
                    }
                }
            }
            d.iter()
                .map(|&z| if z == f32::MAX { 0.0 } else { z })
                .collect()
        })
        .collect();
    let size = 2.0 * scene.extent;
    let lo = scene.center.map(|c| c - scene.extent);
    let mut p = TsdfParams::for_bounds(lo, size, 512);
    p.depth_max = 2.0 * scene.extent;
    p.max_blocks = 1 << 19;
    p.hash_slots = 1 << 21;
    let gpu = gpu();
    let mut vol = TsdfVolume::new(&gpu, p).unwrap();
    let dev: Vec<_> = depths
        .iter()
        .map(|d| gpu.stream.clone_htod(d).unwrap())
        .collect();
    gpu.sync().unwrap();
    let t0 = Instant::now();
    for (d, c) in dev.iter().zip(&cams) {
        vol.integrate_device(d, c).unwrap();
    }
    gpu.sync().unwrap();
    let integrate_ms = t0.elapsed().as_secs_f64() * 1e3;
    let opts = ExtractOptions::default();
    let _ = vol.extract(opts).unwrap();
    let t = Instant::now();
    let m = vol.extract(opts).unwrap();
    let extract_ms = t.elapsed().as_secs_f64() * 1e3;
    println!(
        "{{\"scene\":\"garden\",\"res\":512,\"voxel\":{:.5},\"views\":{},\"depth_size\":[{},{}],\"points\":{},\"blocks\":{},\"integrate_ms_total\":{:.1},\"extract_ms\":{:.1},\"vertices\":{},\"faces\":{}}}",
        p.voxel,
        cams.len(),
        cams[0].width,
        cams[0].height,
        pts.len(),
        vol.blocks(),
        integrate_ms,
        extract_ms,
        m.positions.len(),
        m.faces.len()
    );
    assert!(!m.is_empty());
}
