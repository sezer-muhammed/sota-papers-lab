//! Train + mesh on a dataset (GPU host only, `#[ignore]`). Owner: mesh-mode.
//! Driven by `SPL_TRAIN_*` variables (the ones `scripts/gpu-test.sh` forwards):
//!
//! | Variable | Default | Meaning |
//! |---|---|---|
//! | `SPL_TRAIN_SCENE` | `dtu/scan24` | dataset under `$SPL_DATASETS` |
//! | `SPL_TRAIN_IMAGES` | `image_hd` | image folder |
//! | `SPL_TRAIN_MAX_WIDTH` | none | downscale (2DGS DTU: `-r 2` = 800) |
//! | `SPL_TRAIN_TEST_EVERY` | 0 | hold-out cadence (0 = train on all, as 2DGS on DTU) |
//! | `SPL_TRAIN_PRESET` | `geometry` | preset |
//! | `SPL_TRAIN_ITERS` | preset | iteration override |
//! | `SPL_TRAIN_GEOM` | default | `GeometryConfig` JSON override |
//! | `SPL_TRAIN_OVERRIDES` | `{}` | `PresetOverrides` JSON |
//! | `SPL_TRAIN_MESH` | 2DGS DTU settings | `MeshConfig` JSON |
//! | `SPL_TRAIN_SPLAT` | none | mesh this PLY instead of training |
//! | `SPL_TRAIN_TAG` | `<scene>-<preset>` | output folder |
//!
//! Writes `/out/image-to-3d/mesh/<tag>/`: `raw.ply` (fused surface before
//! decimation, in the training frame), `mesh.glb`, `splat.ply`, `result.json`.

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use spl_core::CancellationToken;

use super::*;
use crate::gpu::Gpu;
use crate::io::scene::{LoadOptions, load_colmap_scene};
use crate::presets::PresetName;
use crate::trainer::{TrainConfig, Trainer};

fn env(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

/// Binary little-endian PLY of positions and faces.
pub fn write_ply(m: &Mesh, path: &std::path::Path) -> std::io::Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    write!(
        f,
        "ply\nformat binary_little_endian 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nelement face {}\nproperty list uchar int vertex_indices\nend_header\n",
        m.positions.len(),
        m.faces.len()
    )?;
    for p in &m.positions {
        for x in p {
            f.write_all(&x.to_le_bytes())?;
        }
    }
    for t in &m.faces {
        f.write_all(&[3u8])?;
        for i in t {
            f.write_all(&(*i as i32).to_le_bytes())?;
        }
    }
    f.flush()
}

#[test]
#[ignore = "GPU and dataset: run on the GPU host"]
fn gpu_mesh_dataset() {
    let scene_name = env("SPL_TRAIN_SCENE", "dtu/scan24");
    let preset_s = env("SPL_TRAIN_PRESET", "geometry");
    let preset: PresetName =
        serde_json::from_value(serde_json::Value::String(preset_s.clone())).expect("preset");
    let short = scene_name.rsplit('/').next().unwrap().to_string();
    let tag = env("SPL_TRAIN_TAG", &format!("{short}-{preset_s}"));
    let out = PathBuf::from("/out/image-to-3d/mesh").join(&tag);
    std::fs::create_dir_all(&out).unwrap();
    let root = PathBuf::from(env("SPL_DATASETS", "/datasets")).join(&scene_name);
    let load = LoadOptions {
        images_dir: env("SPL_TRAIN_IMAGES", "image_hd"),
        test_every: env("SPL_TRAIN_TEST_EVERY", "0").parse().unwrap(),
        max_width: std::env::var("SPL_TRAIN_MAX_WIDTH")
            .ok()
            .map(|v| v.parse().unwrap()),
        ..Default::default()
    };
    let scene = load_colmap_scene(&root, &load).unwrap();
    let views: Vec<_> = scene
        .train
        .iter()
        .map(|&i| scene.views[i].clone())
        .collect();
    // 2DGS DTU: voxel 0.004, sdf_trunc 0.016 (normalised frame). Its
    // depth_trunc 3 cuts scan 63's whole object (cameras ~3.2 away): the
    // bounds crop instead.
    let mesh_cfg: MeshConfig = serde_json::from_str(&env(
        "SPL_TRAIN_MESH",
        r#"{"voxel": 0.004, "trunc_voxels": 4,
            "bounds": [-1.1, -1.1, -1.1, 1.1, 1.1, 1.1]}"#,
    ))
    .unwrap();
    let geometry = std::env::var("SPL_TRAIN_GEOM")
        .ok()
        .map(|s| serde_json::from_str(&s).expect("SPL_TRAIN_GEOM"));
    let cfg = TrainConfig {
        preset,
        iterations: std::env::var("SPL_TRAIN_ITERS")
            .ok()
            .map(|v| v.parse().unwrap()),
        geometry,
        overrides: serde_json::from_str(&env("SPL_TRAIN_OVERRIDES", "{}"))
            .expect("SPL_TRAIN_OVERRIDES"),
        ..Default::default()
    };
    eprintln!(
        "{tag}: {} views at {}x{}, extent {:.3}, geometry {:?}, mesh {mesh_cfg:?}",
        views.len(),
        views[0].width,
        views[0].height,
        scene.extent,
        cfg.geometry()
    );
    let gpu = Gpu::open().unwrap();
    let (cloud, train_s, iterations) = match std::env::var("SPL_TRAIN_SPLAT") {
        Ok(path) => {
            let bytes = std::fs::read(&path).unwrap();
            (crate::export::ply::read_inria_ply(&bytes).unwrap(), 0.0, 0)
        }
        Err(_) => {
            let mut trainer = Trainer::new(&gpu, scene, cfg).unwrap();
            let mut last = 0;
            let summary = trainer
                .run(&CancellationToken::new(), &mut |p| {
                    if p.t >= last + 2000 || p.t == p.total {
                        last = p.t;
                        eprintln!(
                            "t {:>6}/{} n {:>8} {:>6.1} it/s loss {:?}",
                            p.t, p.total, p.n_gaussians, p.iterations_per_s, p.loss
                        );
                    }
                })
                .unwrap();
            if let Some(geo) = &mut trainer.geometry {
                let (d, n) = geo.take_sums(&gpu, summary.iterations).unwrap();
                eprintln!(
                    "geometry losses (mean per frame, pixel sums): distortion {d:.4e} normal {n:.4e}; multi-view per pixel: geometric {:.4e} photometric {:.4e}, {:.0} pixels per frame",
                    geo.mv_sums.0, geo.mv_sums.1, geo.mv_sums.2
                );
            }
            let (cloud, _) = trainer.finish().unwrap();
            let mut ply = Vec::new();
            crate::export::ply::write_inria_ply(&cloud, &mut ply).unwrap();
            std::fs::write(out.join("splat.ply"), &ply).unwrap();
            (cloud, summary.train_s, summary.iterations)
        }
    };
    eprintln!("train {train_s:.1}s, {} Gaussians", cloud.len());
    // Normal (camera space, RGB = (1 − n)/2) and median-depth maps of 3 views.
    let mut dump = |i: usize, b: &super::super::geom::GeomBuffers| -> spl_core::Result<()> {
        if !i.is_multiple_of(16) {
            return Ok(());
        }
        let v = &views[i];
        let p = (v.width * v.height) as usize;
        let out_h = gpu.stream.clone_dtoh(&b.out.slice(0..8 * p)).unwrap();
        let mut nimg = crate::io::images::Rgb8Image::new(v.width, v.height);
        let mut dimg = crate::io::images::Rgb8Image::new(v.width, v.height);
        let zs: Vec<f32> = out_h[..p].iter().cloned().filter(|&z| z > 0.0).collect();
        let (zlo, zhi) = zs
            .iter()
            .fold((f32::MAX, f32::MIN), |(a, b), &z| (a.min(z), b.max(z)));
        for k in 0..p {
            let a = out_h[p + k].max(1e-6);
            for c in 0..3 {
                nimg.data[3 * k + c] =
                    ((1.0 - out_h[(4 + c) * p + k] / a) * 127.5).clamp(0.0, 255.0) as u8;
            }
            let z = out_h[k];
            let g = if z > 0.0 {
                255.0 * (1.0 - (z - zlo) / (zhi - zlo).max(1e-6))
            } else {
                0.0
            };
            dimg.data[3 * k..3 * k + 3].fill(g as u8);
        }
        nimg.save_png(&out.join(format!("normal_{i:03}.png")))?;
        dimg.save_png(&out.join(format!("depth_{i:03}.png")))?;
        Ok(())
    };
    let mut sum = MeshSummary::default();
    // Diagnostics only: `build` below retries an overflow, this pass does not.
    if let Err(e) = fuse_with(&gpu, &cloud, &views, &mesh_cfg, &mut sum, &mut dump) {
        eprintln!("diagnostic fuse failed: {e:#}");
    }
    let t = Instant::now();
    let res = build(&gpu, &cloud, &views, &mesh_cfg).unwrap();
    let mesh_s = t.elapsed().as_secs_f64();
    eprintln!("mesh {mesh_s:.1}s: {:?}", res.summary);
    write_ply(&res.raw, &out.join("raw.ply")).unwrap();
    std::fs::write(out.join("mesh.glb"), &res.glb).unwrap();
    super::super::glb::validate_glb(&res.glb).unwrap();
    let result = serde_json::json!({
        "tag": tag,
        "scene": scene_name,
        "preset": preset_s,
        "iterations": iterations,
        "train_s": train_s,
        "final_gaussians": cloud.len(),
        "mesh_s": mesh_s,
        "mesh": res.summary,
    });
    std::fs::write(
        out.join("result.json"),
        serde_json::to_string_pretty(&result).unwrap(),
    )
    .unwrap();
}
