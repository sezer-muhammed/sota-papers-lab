//! Tests on the benchmark datasets (`~/spl-work/datasets` on the GPU host,
//! `/datasets` in the GPU container). All `#[ignore]`; they skip with a note
//! when the datasets are absent. CPU ones also run natively on the host.

use std::time::Instant;

use super::colmap::ColmapModel;
use super::images::{ImageCache, decode_bytes_image_crate, decode_file, load_views};
use super::scene::{InitKind, LoadOptions, find_sparse_dir, init_gaussians, load_colmap_scene};
use crate::export::{colmap as colmap_export, ply, spz};
use crate::gpu::Gpu;
use crate::testkit::npz::{NpyData, Npz};
use crate::testkit::{datasets_dir, mip360_scene, out_dir};

const OUTDOOR: [&str; 5] = ["bicycle", "flowers", "garden", "stump", "treehill"];
const INDOOR: [&str; 4] = ["room", "counter", "kitchen", "bonsai"];

fn opts(images_dir: &str) -> LoadOptions {
    LoadOptions {
        images_dir: images_dir.into(),
        ..Default::default()
    }
}

#[test]
#[ignore]
fn mip360_garden_parses() {
    let Some(root) = mip360_scene("garden") else {
        eprintln!("skipped: no mipnerf360/garden dataset");
        return;
    };
    let sparse = find_sparse_dir(&root).unwrap();
    let read = |n: &str| std::fs::read(sparse.join(n)).unwrap();
    let (c, i, p) = (
        read("cameras.bin"),
        read("images.bin"),
        read("points3D.bin"),
    );
    let started = Instant::now();
    let model = ColmapModel::parse_bin(&c, &i, &p).unwrap();
    let parse_ms = started.elapsed().as_secs_f64() * 1e3;
    // Our writer reproduces COLMAP's bytes when its order is by id.
    let [c2, i2, p2] = model.to_bin();
    println!(
        "garden: {} cameras, {} images, {} points, {} 2D points; parse {parse_ms:.1} ms; byte-exact rewrite: cameras {} images {} points {}",
        model.cameras.len(),
        model.images.len(),
        model.points.len(),
        model
            .images
            .values()
            .map(|im| im.points2d.len())
            .sum::<usize>(),
        c2 == c,
        i2 == i,
        p2 == p
    );
    let [ct, it, pt] = model.to_txt();
    assert_eq!(
        ColmapModel::parse_txt(&ct, &it, &pt).unwrap(),
        model,
        "bin -> txt -> model"
    );

    let started = Instant::now();
    let s = load_colmap_scene(&root, &opts("images_4")).unwrap();
    let v = &s.views[0];
    println!(
        "garden images_4: {} views ({} train / {} test), {}x{}, f = ({:.2}, {:.2}), c = ({:.2}, {:.2}), pp offset {:.3} px, extent {:.4}, centre {:?}, load {:.0} ms",
        s.views.len(),
        s.train.len(),
        s.test.len(),
        v.width,
        v.height,
        v.fx,
        v.fy,
        v.cx,
        v.cy,
        s.max_pp_offset,
        s.extent,
        s.center,
        started.elapsed().as_secs_f64() * 1e3
    );
    assert_eq!((s.views.len(), s.train.len(), s.test.len()), (185, 161, 24));
    assert_eq!(s.test, (0..185).step_by(8).collect::<Vec<_>>());
    assert!(s.views.windows(2).all(|w| w[0].name < w[1].name));
    assert_eq!(s.points.xyz.len(), model.points.len());
    assert!(s.extent > 0.0);
}

/// Every benchmark scene on the host loads with its protocol's image folder.
#[test]
#[ignore]
fn all_benchmark_scenes_parse() {
    let Some(base) = datasets_dir() else {
        eprintln!("skipped: no datasets");
        return;
    };
    let mut scenes = Vec::new();
    for s in OUTDOOR {
        scenes.push((format!("mipnerf360/{s}"), "images_4"));
    }
    for s in INDOOR {
        scenes.push((format!("mipnerf360/{s}"), "images_2"));
    }
    for s in ["tandt/truck", "tandt/train", "db/drjohnson", "db/playroom"] {
        scenes.push((s.to_string(), "images"));
    }
    println!(
        "{:<22} {:>5} {:>5} {:>4} {:>11} {:>8} {:>8} {:>7} {:>6}",
        "scene", "views", "train", "test", "size", "points", "extent", "pp px", "ms"
    );
    let mut loaded = 0;
    for (name, dir) in scenes {
        let root = base.join(&name);
        if !root.is_dir() {
            println!("{name:<22} missing");
            continue;
        }
        let dir = if root.join(dir).is_dir() {
            dir
        } else {
            "images"
        };
        let started = Instant::now();
        let s = load_colmap_scene(&root, &opts(dir)).unwrap_or_else(|e| panic!("{name}: {e:#}"));
        let v = &s.views[0];
        println!(
            "{name:<22} {:>5} {:>5} {:>4} {:>11} {:>8} {:>8.3} {:>7.3} {:>6.0}",
            s.views.len(),
            s.train.len(),
            s.test.len(),
            format!("{}x{}", v.width, v.height),
            s.points.xyz.len(),
            s.extent,
            s.max_pp_offset,
            started.elapsed().as_secs_f64() * 1e3
        );
        loaded += 1;
    }
    assert!(loaded > 0);
}

/// Decodes the first 10 garden images for the PIL parity check (spec-train
/// 8.4): writes `/out/image-to-3d/decode_parity/garden.npz` (our decoder,
/// libjpeg) and `garden_zune.npz` (the `image` crate) with one `[H, W, 3] u8`
/// array per image name, to compare with PIL in the PyTorch container
/// (`scripts/decode_parity.py`).
#[test]
#[ignore]
fn garden_decode_dump_for_pil_parity() {
    let Some(root) = mip360_scene("garden") else {
        eprintln!("skipped: no mipnerf360/garden dataset");
        return;
    };
    let s = load_colmap_scene(&root, &opts("images_4")).unwrap();
    let (mut ours, mut zune) = (Npz::new(), Npz::new());
    for v in s.views.iter().take(10) {
        let bytes = std::fs::read(&v.image_path).unwrap();
        for (z, img) in [
            (&mut ours, decode_file(&v.image_path).unwrap()),
            (&mut zune, decode_bytes_image_crate(&bytes).unwrap()),
        ] {
            let shape = [img.height as usize, img.width as usize, 3];
            z.insert(&v.name, &shape, NpyData::U8(img.data)).unwrap();
        }
    }
    for (name, z) in [("garden.npz", ours), ("garden_zune.npz", zune)] {
        let p = out_dir("decode_parity").join(name);
        z.write(&p).unwrap();
        println!("wrote {}", p.display());
    }
}

/// Decodes garden's training views and uploads them into the u8 pool.
#[test]
#[ignore]
fn gpu_garden_image_cache() {
    let Some(root) = mip360_scene("garden") else {
        eprintln!("skipped: no mipnerf360/garden dataset");
        return;
    };
    let gpu = Gpu::open().unwrap();
    let s = load_colmap_scene(&root, &opts("images_4")).unwrap();
    let threads = rayon::current_num_threads();
    let started = Instant::now();
    let images = load_views(&s.views, &s.train).unwrap();
    let decode_s = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let cache = ImageCache::from_images(&gpu, s.train.clone(), &images, decode_s).unwrap();
    gpu.sync().unwrap();
    let upload_s = started.elapsed().as_secs_f64();
    let mb = cache.bytes() as f64 / 1e6;
    println!(
        "garden: {} train images, {mb:.0} MB u8; decode {decode_s:.3} s on {threads} threads ({:.1} ms/image/thread); upload {:.3} s ({:.1} GB/s)",
        images.len(),
        decode_s * 1e3 * threads as f64 / images.len() as f64,
        upload_s,
        mb / 1e3 / upload_s
    );
    for (k, &view) in s.train.iter().enumerate().step_by(40) {
        assert_eq!(
            cache.download(&gpu, view).unwrap(),
            images[k],
            "view {view}"
        );
    }
    let e = cache.entry(s.train[1]).unwrap();
    assert_eq!(e.offset, images[0].data.len() as u64);
}

/// Initializes Gaussians from garden's COLMAP points and exports PLY, SPZ, COLMAP.
#[test]
#[ignore]
fn garden_init_and_export() {
    let Some(root) = mip360_scene("garden") else {
        eprintln!("skipped: no mipnerf360/garden dataset");
        return;
    };
    let s = load_colmap_scene(&root, &opts("images_4")).unwrap();
    let started = Instant::now();
    let g = init_gaussians(&s.points, InitKind::Standard).unwrap();
    let init_ms = started.elapsed().as_secs_f64() * 1e3;
    println!(
        "garden init: {} Gaussians from {} points in {init_ms:.1} ms; scale range [{:.3}, {:.3}]",
        g.len(),
        s.points.xyz.len(),
        g.scales.iter().cloned().fold(f32::INFINITY, f32::min),
        g.scales.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
    );
    g.validate().unwrap();
    assert_eq!(g.len(), s.points.xyz.len());

    let out = out_dir("exports");
    std::fs::create_dir_all(&out).unwrap();

    // PLY
    let ply_path = out.join("garden.ply");
    {
        let mut f = std::fs::File::create(&ply_path).unwrap();
        ply::write_inria_ply(&g, &mut f).unwrap();
    }
    let ply_bytes = std::fs::read(&ply_path).unwrap();
    let back = ply::read_inria_ply(&ply_bytes).unwrap();
    assert_eq!(back.len(), g.len());
    println!("  PLY: {} bytes, round-trip OK", ply_bytes.len());

    // SPZ v3
    let spz_path = out.join("garden.spz");
    let spz_bytes = spz::write_spz(&g, &spz::SpzOptions::default()).unwrap();
    std::fs::write(&spz_path, &spz_bytes).unwrap();
    let spz_back = spz::read_spz(&spz_bytes).unwrap();
    assert_eq!(spz_back.len(), g.len());
    println!("  SPZ v3: {} bytes, round-trip OK", spz_bytes.len());

    // SPZ v4
    let spz4_bytes = spz::write_spz(
        &g,
        &spz::SpzOptions {
            version: 4,
            ..Default::default()
        },
    )
    .unwrap();
    let spz4_path = out.join("garden_v4.spz");
    std::fs::write(&spz4_path, &spz4_bytes).unwrap();
    let spz4_back = spz::read_spz(&spz4_bytes).unwrap();
    assert_eq!(spz4_back.len(), g.len());
    println!("  SPZ v4: {} bytes, round-trip OK", spz4_bytes.len());

    // COLMAP export
    let colmap_dir = out.join("colmap_bin");
    colmap_export::write_colmap(&s.views, &g, &colmap_dir, false).unwrap();
    let model = ColmapModel::read_bin(&colmap_dir).unwrap();
    assert!(!model.cameras.is_empty());
    assert_eq!(model.images.len(), s.views.len());
    println!(
        "  COLMAP bin: {} cameras, {} images, {} points",
        model.cameras.len(),
        model.images.len(),
        model.points.len()
    );

    // transforms.json
    let json_path = out.join("transforms.json");
    {
        let mut f = std::fs::File::create(&json_path).unwrap();
        colmap_export::write_transforms_json(&s.views, &mut f).unwrap();
    }
    let json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&json_path).unwrap()).unwrap();
    assert_eq!(json["frames"].as_array().unwrap().len(), s.views.len());
    println!("  transforms.json: {} frames", s.views.len());
}
