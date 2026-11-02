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
use crate::testkit::timing::{GpuTimer, Stats, bench, gpu_snapshot};

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

/// One forward and one backward per variant on the first dump (or the one
/// `SPL_RASTER_BENCH_ONLY` names), for Nsight Compute: it replays each launch
/// itself, so no timing loop. Variants: `SPL_RASTER_PROFILE_K` (default 4),
/// `SPL_RASTER_PROFILE_TILE_H` (16).
#[test]
#[ignore]
fn gpu_raster_profile() {
    let dir = bench_dir();
    let only = std::env::var("SPL_RASTER_BENCH_ONLY").unwrap_or_default();
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map(|d| d.filter_map(|e| e.ok().map(|e| e.path())).collect())
        .unwrap_or_default();
    files.retain(|p| {
        p.extension().is_some_and(|e| e == "npz") && p.to_string_lossy().contains(only.as_str())
    });
    files.sort();
    let Some(path) = files.first() else {
        eprintln!("skipped: no scene dumps in {}", dir.display());
        return;
    };
    let env = |k: &str, d: u8| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(d)
    };
    let v = RasterVariant {
        chain_k: env("SPL_RASTER_PROFILE_K", 4),
        tile_h: env("SPL_RASTER_PROFILE_TILE_H", 16),
        ..Default::default()
    };
    let view = load(path);
    let lists = TileLists::exact(&view.scene, v.tile_h as u32);
    let np = (view.scene.width * view.scene.height) as usize;
    let mut pg = PixelGradsHost::random(1, np, 0, false);
    for x in &mut pg.d_rgb {
        *x *= 2e-6;
    }
    let gpu = Gpu::open().unwrap();
    let k = SplatKernels::load(&gpu, v).unwrap();
    let mut st = stage(&gpu, &v, &view.scene, &lists, &pg, 0).unwrap();
    raster_fwd(&k, &gpu.stream, &mut st.f, RenderMode::Train, &mut st.out).unwrap();
    raster_bwd(&k, &gpu.stream, &mut st.f, &st.out, &st.pg, &mut st.g2d).unwrap();
    gpu.sync().unwrap();
    println!(
        "profiled {} with {v:?}: V {}, I {}, chains {}, bwd grid {} blocks, regs fwd {} bwd {}",
        view.name,
        view.scene.recs.len(),
        lists.instances(),
        st.n_chains,
        k.raster.bwd_blocks,
        k.raster.fwd.num_regs().unwrap(),
        k.raster.bwd.num_regs().unwrap()
    );
}

/// What the backward has to do, from the forward's n_contrib.
#[derive(Clone, Copy, Debug, Serialize)]
struct BwdWork {
    /// Buckets with at least one active pixel.
    buckets: u64,
    /// Σ over those buckets of (active pixels + 31): diagonal-pipeline steps.
    steps: u64,
    /// Share of the steps spent filling and draining the pipeline.
    ramp_frac: f64,
    /// (lane, step) pairs whose Gaussian is before the pixel's last
    /// contributor, over 32 × steps (upper bound on useful lanes).
    lane_util: f64,
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
    bwd_work: BwdWork,
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
            // raster_bwd consumes the checkpoints: an untimed forward before each.
            let timer = GpuTimer::new(&gpu).unwrap();
            let mut samples = Vec::new();
            for rep in 0..23 {
                raster_fwd(&k, &s, &mut st.f, RenderMode::Train, &mut st.out).unwrap();
                timer.start(&s).unwrap();
                raster_bwd(&k, &s, &mut st.f, &st.out, &st.pg, &mut st.g2d).unwrap();
                let ms = timer.stop_ms(&s).unwrap() as f64;
                if rep >= 3 {
                    samples.push(ms);
                }
            }
            let bwd = Stats::from_samples(&samples);
            let nc = super::scene::host(&gpu, &st.out.n_contrib).unwrap();
            let (mut walked, mut listed) = (0u64, 0u64);
            let mut tile_nc: Vec<Vec<u32>> = vec![Vec::new(); lists.n_tiles() as usize];
            for (pix, &n) in nc[..np].iter().enumerate() {
                let (x, y) = (pix as u32 % view.scene.width, pix as u32 / view.scene.width);
                let t = (y / lists.tile_h * lists.grid_w + x / TILE_W) as usize;
                walked += n as u64;
                listed += (lists.tile_start[t + 1] - lists.tile_start[t]) as u64;
                tile_nc[t].push(n);
            }
            // Backward work: buckets with an active pixel, diagonal-pipeline
            // steps Σ (n_active + 31), and the share of steps in the ramps.
            let (mut buckets, mut steps, mut pairs) = (0u64, 0u64, 0u64);
            for ncs in &tile_nc {
                let max = ncs.iter().copied().max().unwrap_or(0);
                for b in 0..max.div_ceil(32) {
                    let active = ncs.iter().filter(|&&n| n > 32 * b).count() as u64;
                    buckets += 1;
                    steps += active + 31;
                    pairs += ncs
                        .iter()
                        .map(|&n| n.saturating_sub(32 * b).min(32) as u64)
                        .sum::<u64>();
                }
            }
            let bwd_work = BwdWork {
                buckets,
                steps,
                ramp_frac: (31 * buckets) as f64 / steps.max(1) as f64,
                lane_util: pairs as f64 / (32 * steps).max(1) as f64,
            };
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
                bwd_work,
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
            println!(
                "          bwd buckets {} steps {} ramp {:.1}% lanes busy ≤ {:.1}%  ckpt {:.0} MB",
                bwd_work.buckets,
                bwd_work.steps,
                100.0 * bwd_work.ramp_frac,
                100.0 * bwd_work.lane_util,
                row.ckpt_mb
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
