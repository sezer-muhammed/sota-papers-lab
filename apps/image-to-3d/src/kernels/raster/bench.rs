//! Raster kernel benchmark on trained garden models (owner: raster E3).
//!
//! Input: per-view dumps written by `benchmarks/image-to-3d/py/raster_scene.py`
//! (gsplat 1.5.3's projection, SH colours and tile lists of a trained
//! checkpoint, plus gsplat's own `rasterize_to_pixels` forward/backward kernel
//! times on the same data) under `$SPL_RASTER_BENCH` or
//! `/out/image-to-3d/raster-bench`. For every view the benchmark times our
//! `raster_fwd` (Train) and `raster_bwd` with CUDA events on gsplat's lists
//! and on our exact lists (spec-raster 2.8), for several variants, and
//! records V, I, chains, the early-termination fraction, checkpoint bytes and
//! nvidia-smi before and after. Run it under the GPU lock:
//! `SPL_GPU_LOCK=1 apps/image-to-3d/scripts/gpu-test.sh <checkout> raster::bench`.

use std::path::PathBuf;

use serde::Serialize;

use super::scene::{PixelGradsHost, Scene2d, TileLists, record, stage};
use super::*;
use crate::kernels::layout::TILE_W;
use crate::testkit::npz::Npz;
use crate::testkit::timing::{Stats, bench, gpu_snapshot};

fn bench_dir() -> PathBuf {
    std::env::var_os("SPL_RASTER_BENCH")
        .map(PathBuf::from)
        .unwrap_or_else(|| crate::testkit::out_dir("raster-bench"))
}

struct View {
    name: String,
    scene: Scene2d,
    gsplat: TileLists,
    gs_fwd_ms: f64,
    gs_bwd_ms: f64,
}

fn load(path: &std::path::Path) -> View {
    let z = Npz::read(path).unwrap();
    let f = |k: &str| -> Vec<f64> { z.get(k).unwrap().to_f64() };
    let (w, h) = (f("width")[0] as u32, f("height")[0] as u32);
    let (m, c, col, o, d) = (
        f("means2d"),
        f("conics"),
        f("colors"),
        f("opacities"),
        f("depths"),
    );
    let recs = (0..o.len())
        .map(|i| {
            record(
                [m[2 * i] as f32, m[2 * i + 1] as f32],
                [c[3 * i], c[3 * i + 1], c[3 * i + 2]],
                o[i] as f32,
                [
                    col[3 * i] as f32,
                    col[3 * i + 1] as f32,
                    col[3 * i + 2] as f32,
                ],
                d[i] as f32,
                0,
            )
        })
        .collect();
    let mut tile_start: Vec<u32> = f("isect_offsets").iter().map(|&x| x as u32).collect();
    let ids: Vec<u32> = f("flatten_ids").iter().map(|&x| x as u32).collect();
    tile_start.push(ids.len() as u32);
    View {
        name: path.file_stem().unwrap().to_string_lossy().into_owned(),
        scene: Scene2d {
            width: w,
            height: h,
            recs,
            bg: [0.0; 3],
        },
        gsplat: TileLists {
            tile_h: 16,
            grid_w: w.div_ceil(TILE_W),
            grid_h: h.div_ceil(16),
            tile_start,
            inst_v: ids,
        },
        gs_fwd_ms: f("gs_fwd_ms")[0],
        gs_bwd_ms: f("gs_bwd_ms")[0],
    }
}

#[derive(Serialize)]
struct Row {
    view: String,
    lists: &'static str,
    variant: String,
    visible: usize,
    instances: u32,
    chains: u32,
    /// Σ n_contrib / Σ (list length over the pixels' tiles): the share of
    /// the lists the forward actually walks before terminating.
    walked_frac: f64,
    ckpt_mb: f64,
    fwd: Stats,
    bwd: Stats,
    gsplat_fwd_ms: f64,
    gsplat_bwd_ms: f64,
}

#[test]
#[ignore]
fn gpu_raster_bench() {
    let dir = bench_dir();
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map(|d| d.filter_map(|e| e.ok().map(|e| e.path())).collect())
        .unwrap_or_default();
    files.retain(|p| p.extension().is_some_and(|e| e == "npz"));
    files.sort();
    if files.is_empty() {
        eprintln!(
            "skipped: no scene dumps in {} (write them with benchmarks/image-to-3d/py/raster_scene.py)",
            dir.display()
        );
        return;
    }
    let only: Option<String> = std::env::var("SPL_RASTER_BENCH_ONLY").ok();
    let gpu = Gpu::open().unwrap();
    let before = gpu_snapshot();
    println!("nvidia-smi before: {before:?}");
    let mut rows = Vec::new();
    for path in &files {
        if only
            .as_ref()
            .is_some_and(|o| !path.to_string_lossy().contains(o.as_str()))
        {
            continue;
        }
        let view = load(path);
        let np = (view.scene.width * view.scene.height) as usize;
        let mut pg = PixelGradsHost::random(1, np, 0, false);
        for x in &mut pg.d_rgb {
            *x *= 2e-6;
        }
        let started = std::time::Instant::now();
        let exact16 = TileLists::exact(&view.scene, 16);
        let exact12 = TileLists::exact(&view.scene, 12);
        println!(
            "{}: {}x{}, V {}, gsplat I {}, exact I {} (16x16) {} (16x12), lists in {:.1} s; gsplat fwd {:.3} ms bwd {:.3} ms",
            view.name,
            view.scene.width,
            view.scene.height,
            view.scene.recs.len(),
            view.gsplat.instances(),
            exact16.instances(),
            exact12.instances(),
            started.elapsed().as_secs_f64(),
            view.gs_fwd_ms,
            view.gs_bwd_ms
        );
        let d = RasterVariant::default();
        let mut cases: Vec<(&'static str, &TileLists, RasterVariant)> = vec![
            ("gsplat", &view.gsplat, d),
            ("exact", &exact16, RasterVariant { chain_k: 1, ..d }),
            ("exact", &exact16, RasterVariant { chain_k: 2, ..d }),
            ("exact", &exact16, d),
            ("exact", &exact16, RasterVariant { chain_k: 8, ..d }),
            (
                "exact",
                &exact16,
                RasterVariant {
                    ckpt_f16: true,
                    ..d
                },
            ),
            ("exact", &exact12, RasterVariant { tile_h: 12, ..d }),
        ];
        if std::env::var_os("SPL_RASTER_BENCH_QUICK").is_some() {
            cases.truncate(4);
        }
        for (lname, lists, v) in cases {
            let k = SplatKernels::load(&gpu, v).unwrap();
            let mut st = stage(&gpu, &v, &view.scene, lists, &pg, 0).unwrap();
            let s = gpu.stream.clone();
            let fwd = bench(&gpu, 3, 20, || {
                raster_fwd(&k, &s, &mut st.f, RenderMode::Train, &mut st.out)
            })
            .unwrap();
            let bwd = bench(&gpu, 3, 20, || {
                raster_bwd(&k, &s, &mut st.f, &st.out, &st.pg, &mut st.g2d)
            })
            .unwrap();
            let nc = super::scene::host(&gpu, &st.out.n_contrib).unwrap();
            let (mut walked, mut listed) = (0u64, 0u64);
            for (pix, &n) in nc[..np].iter().enumerate() {
                let (x, y) = (pix as u32 % view.scene.width, pix as u32 / view.scene.width);
                let t = (y / lists.tile_h * lists.grid_w + x / TILE_W) as usize;
                walked += n as u64;
                listed += (lists.tile_start[t + 1] - lists.tile_start[t]) as u64;
            }
            let row = Row {
                view: view.name.clone(),
                lists: lname,
                variant: format!(
                    "K{} {}x{}{}",
                    v.chain_k,
                    TILE_W,
                    v.tile_h,
                    if v.ckpt_f16 { " f16" } else { "" }
                ),
                visible: view.scene.recs.len(),
                instances: lists.instances(),
                chains: st.n_chains,
                walked_frac: walked as f64 / listed.max(1) as f64,
                ckpt_mb: st.n_chains as f64 * v.tile_pixels() as f64 * ckpt_words(&v) as f64 * 4.0
                    / 1e6,
                fwd,
                bwd,
                gsplat_fwd_ms: view.gs_fwd_ms,
                gsplat_bwd_ms: view.gs_bwd_ms,
            };
            println!(
                "  {:<7} {:<14} I {:>9} chains {:>7} walked {:>5.1}%  fwd {:>7.3} ms (p10 {:.3} p90 {:.3})  bwd {:>7.3} ms (p10 {:.3} p90 {:.3})",
                row.lists,
                row.variant,
                row.instances,
                row.chains,
                100.0 * row.walked_frac,
                row.fwd.median_ms,
                row.fwd.p10_ms,
                row.fwd.p90_ms,
                row.bwd.median_ms,
                row.bwd.p10_ms,
                row.bwd.p90_ms
            );
            rows.push(row);
        }
    }
    let after = gpu_snapshot();
    println!("nvidia-smi after: {after:?}");
    let json = serde_json::json!({
        "gpu_before": before,
        "gpu_after": after,
        "rows": rows,
    });
    let out = dir.join("results.json");
    std::fs::write(&out, serde_json::to_string_pretty(&json).unwrap()).unwrap();
    println!("wrote {}", out.display());
}
