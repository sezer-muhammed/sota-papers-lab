//! Tests on the benchmark datasets (`~/spl-work/datasets` on the GPU host,
//! `/datasets` in the GPU container). All `#[ignore]`; they skip with a note
//! when the datasets are absent. CPU ones also run natively on the host.

use std::time::Instant;

use super::colmap::ColmapModel;
use super::images::{ImageCache, decode_bytes_image_crate, decode_file, load_views};
use super::scene::{LoadOptions, find_sparse_dir, load_colmap_scene};
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
