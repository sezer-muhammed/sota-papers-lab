//! Realistic sort and binning workloads, timed per kernel with CUDA events.
//! Owner: sort (E2). GPU tests, `#[ignore]`; numbers that get reported run
//! with `SPL_GPU_LOCK=1 apps/image-to-3d/scripts/gpu-test.sh <checkout> sort::bench`.
//!
//! Workload: Mip-NeRF 360 garden at 1/4 resolution (images_4, 1297×840,
//! 4346 tiles of 16×16) seen from real training cameras. N Gaussians are
//! spawned around the COLMAP points (surface-like: one flat axis, log-normal
//! scales ∝ extent/√N, random rotations, a spread of opacities) and projected
//! with spec-raster 2's EWA rule; the records then go through `count_tiles`,
//! `sort_depth` and `bin_tiles` exactly as in training. N, the scale factor
//! and the views are chosen so V and I span late-training sizes (V 0.5-4M,
//! I 10-60M).
//!
//! `SPL_SORT_DUMP=1` also writes each workload's sort inputs and our results
//! under `/out/image-to-3d/sort-bench/<name>/` for the CUB / Faster-GS /
//! gsplat-style reference (`benchmarks/image-to-3d/sort/cub_ref.cu`), which
//! sorts the same data and checks it gets our results.

use std::path::Path;

use rayon::prelude::*;
use serde::Serialize;

use super::tests::{Rng, cpu_tiles, frame_cam, gpu, rec_from_cov, upload_frame};
use super::*;
use crate::io::scene::{LoadOptions, Scene, View, load_colmap_scene};
use crate::kernels::layout::{OVF_INSTANCES, ST_N_CHAINS, ST_N_TILES_HIT, ST_OVERFLOW};
use crate::kernels::{Capacity, RasterVariant, SplatRec};
use crate::testkit::timing::{GpuTimer, Stats, gpu_snapshot};
use crate::testkit::{mip360_scene, out_dir};

/// One projected frame ready for binning.
pub(super) struct Workload {
    pub name: String,
    pub w: u32,
    pub h: u32,
    pub n_gaussians: usize,
    pub recs: Vec<SplatRec>,
    pub instances: u64,
}

fn garden() -> Option<Scene> {
    let root = mip360_scene("garden")?;
    let opts = LoadOptions {
        images_dir: "images_4".into(),
        ..Default::default()
    };
    load_colmap_scene(&root, &opts).ok()
}

/// spec-raster 2 (steps 1-6, no AA compensation, ε = 0.3) in f64.
fn project(v: &View, m: [f64; 3], s: [f64; 3], q: [f64; 4], o: f64) -> Option<SplatRec> {
    let (r, t) = (&v.r_cw, &v.t_cw);
    let pc: [f64; 3] =
        std::array::from_fn(|i| r[i][0] * m[0] + r[i][1] * m[1] + r[i][2] * m[2] + t[i]);
    let z = pc[2];
    if !(0.01..1e4).contains(&z) || o < 1.0 / 255.0 {
        return None;
    }
    let n = (q.iter().map(|x| x * x).sum::<f64>()).sqrt();
    let [w, x, y, zq] = q.map(|c| c / n);
    let rq = [
        [
            1.0 - 2.0 * (y * y + zq * zq),
            2.0 * (x * y - w * zq),
            2.0 * (x * zq + w * y),
        ],
        [
            2.0 * (x * y + w * zq),
            1.0 - 2.0 * (x * x + zq * zq),
            2.0 * (y * zq - w * x),
        ],
        [
            2.0 * (x * zq - w * y),
            2.0 * (y * zq + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ];
    // Σ = Rq diag(s²) Rqᵀ
    let sig: [[f64; 3]; 3] = std::array::from_fn(|i| {
        std::array::from_fn(|j| (0..3).map(|k| rq[i][k] * rq[j][k] * s[k] * s[k]).sum())
    });
    let (fx, fy, cx, cy) = (v.fx, v.fy, v.cx, v.cy);
    let (wf, hf) = (v.width as f64, v.height as f64);
    let (xn, yn) = (pc[0] / z, pc[1] / z);
    let xt = xn.clamp(-1.3 * cx / fx, 1.3 * (wf - cx) / fx);
    let yt = yn.clamp(-1.3 * cy / fy, 1.3 * (hf - cy) / fy);
    let j = [[fx / z, 0.0, -fx * xt / z], [0.0, fy / z, -fy * yt / z]];
    // M = J R (2×3), Σ2 = M Σ Mᵀ
    let mm: [[f64; 3]; 2] =
        std::array::from_fn(|i| std::array::from_fn(|c| (0..3).map(|k| j[i][k] * r[k][c]).sum()));
    let quad = |a: &[f64; 3], b: &[f64; 3]| -> f64 {
        (0..3)
            .map(|i| (0..3).map(|k| a[i] * sig[i][k] * b[k]).sum::<f64>())
            .sum()
    };
    let (sxx, sxy, syy) = (
        quad(&mm[0], &mm[0]) + 0.3,
        quad(&mm[0], &mm[1]),
        quad(&mm[1], &mm[1]) + 0.3,
    );
    let (mx, my) = (fx * xn + cx, fy * yn + cy);
    let (ex, ey) = (3.5 * sxx.sqrt(), 3.5 * syy.sqrt());
    if mx + ex < 0.0 || mx - ex > wf || my + ey < 0.0 || my - ey > hf {
        return None;
    }
    Some(rec_from_cov(
        mx as f32,
        my as f32,
        [sxx, sxy, syy],
        o as f32,
        z as f32,
    ))
}

/// N Gaussians around the COLMAP points of `scene`, seen from training view
/// `view`. `scale` multiplies the base scale extent/√N.
pub(super) fn garden_workload(
    scene: &Scene,
    n: usize,
    scale: f64,
    view: usize,
    seed: u64,
) -> Workload {
    let pts = &scene.points.xyz;
    let v = &scene.views[scene.train[view % scene.train.len()]];
    let base = scale * scene.extent as f64 / (n as f64).sqrt();
    let recs: Vec<SplatRec> = (0..n)
        .into_par_iter()
        .filter_map(|i| {
            let mut rng = Rng::new(seed.wrapping_mul(0x9E37_79B9).wrapping_add(i as u64));
            let p = pts[rng.below(pts.len() as u32) as usize];
            let mut gauss = || {
                let (a, b) = (rng.f32() as f64, rng.f32() as f64);
                (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
            };
            let m = [
                p[0] as f64 + 3.0 * base * gauss(),
                p[1] as f64 + 3.0 * base * gauss(),
                p[2] as f64 + 3.0 * base * gauss(),
            ];
            let mut s = [0.0; 3];
            for (k, sk) in s.iter_mut().enumerate() {
                *sk = base * (0.7 * gauss()).exp() * if k == 2 { 0.25 } else { 1.0 };
            }
            let q = [gauss(), gauss(), gauss(), gauss()];
            let o = if rng.below(2) == 0 {
                rng.range(0.004, 0.2) as f64
            } else {
                rng.range(0.2, 1.0) as f64
            };
            project(v, m, s, q, o)
        })
        .collect();
    let (w, h) = (v.width, v.height);
    let instances = recs
        .par_iter()
        .map(|r| cpu_tiles(r, w, h, 16).len() as u64)
        .sum();
    Workload {
        name: format!("garden_n{}m_s{scale}_v{view}", n / 1_000_000),
        w,
        h,
        n_gaussians: n,
        recs,
        instances,
    }
}

/// CPU only: V and I of the garden workload over a grid of N and scale (to
/// choose `workloads`).
#[test]
#[ignore]
fn garden_workload_calibration() {
    let Some(scene) = garden() else {
        eprintln!("skipped: no mipnerf360/garden");
        return;
    };
    println!(
        "garden extent {:.3}, {} points",
        scene.extent,
        scene.points.xyz.len()
    );
    for n in [1_000_000, 3_000_000, 6_000_000] {
        for scale in [2.0, 3.0, 4.0, 6.0, 8.0, 12.0] {
            let wl = garden_workload(&scene, n, scale, 0, 1);
            println!(
                "N {n:>8} scale {scale:>4}: V {:>8} I {:>9} ({:.2} per visible)",
                wl.recs.len(),
                wl.instances,
                wl.instances as f64 / wl.recs.len().max(1) as f64
            );
        }
    }
}

/// The workloads of the report (scales from `garden_workload_calibration`):
/// early training (N 1M, ~4 tiles per visible), mid (3M, ~5), late training at
/// the spec's model (6M, ~4.6: V 3.3M, I ~15M), the same from another view,
/// and two large-splat cases that push I to ~30M and ~60M.
pub(super) fn workloads(scene: &Scene) -> Vec<Workload> {
    [
        (1_000_000, 4.0, 0),
        (3_000_000, 8.0, 0),
        (6_000_000, 10.0, 0),
        (6_000_000, 10.0, 40),
        (6_000_000, 18.0, 0),
        (6_000_000, 30.0, 0),
    ]
    .into_iter()
    .map(|(n, s, v)| garden_workload(scene, n, s, v, 1))
    .collect()
}

#[derive(Debug, Default, Serialize)]
struct Row {
    name: String,
    n_gaussians: usize,
    visible: u32,
    instances: u32,
    chains: u32,
    tiles_hit: u32,
    kernels: Vec<(String, Stats)>,
    pipeline: Stats,
    depth_sort: Stats,
    tile_sort: Stats,
    count_tiles: Stats,
}

/// FrameBuffers for a workload with the checkpoint buffer dropped (the sort
/// never touches it; it would be gigabytes at 60M instances).
pub(super) fn frame_for(gpu: &Gpu, wl: &Workload) -> FrameBuffers {
    let v = RasterVariant::default();
    let cap = Capacity::new(
        wl.recs.len().max(1) as u32,
        (wl.instances as u32).max(1),
        wl.w,
        wl.h,
        &v,
    );
    let mut f = FrameBuffers::alloc(gpu, cap, &v).unwrap();
    f.ckpt = gpu.stream.alloc_zeros(1).unwrap();
    f
}

fn time(gpu: &Gpu, timer: &GpuTimer, f: impl FnOnce()) -> f64 {
    timer.start(&gpu.stream).unwrap();
    f();
    timer.stop_ms(&gpu.stream).unwrap() as f64
}

fn write_raw<T: Copy>(path: &Path, v: &[T]) {
    // SAFETY: plain-old-data slices (u16/u32) viewed as bytes.
    let bytes =
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) };
    std::fs::write(path, bytes).unwrap();
}

/// Per-kernel and whole-pipeline CUDA-event timings of sort_depth + bin_tiles
/// on the garden workloads (median of 15 after 3 warm-ups).
#[test]
#[ignore]
fn gpu_bench_sort_and_binning() {
    let Some(scene) = garden() else {
        eprintln!("skipped: no mipnerf360/garden");
        return;
    };
    let gpu = gpu();
    let dump = std::env::var("SPL_SORT_DUMP").is_ok_and(|v| v == "1");
    let before = gpu_snapshot();
    let k = SplatKernels::load(&gpu, RasterVariant::default()).unwrap();
    let wls = workloads(&scene);
    let max_items = wls
        .iter()
        .map(|w| (w.instances as u32).max(w.recs.len() as u32))
        .max()
        .unwrap();
    let mut sorter = RadixSorter::new(&gpu, max_items).unwrap();
    let timer = GpuTimer::new(&gpu).unwrap();
    let (warm, reps) = (3, 15);
    let mut rows = Vec::new();
    for wl in &wls {
        let s = gpu.stream.clone();
        let mut f = frame_for(&gpu, wl);
        upload_frame(&gpu, &mut f, &frame_cam(wl.w, wl.h, 16), &wl.recs);
        let pk = s.clone_dtod(&f.depth_key[0]).unwrap();
        let pv = s.clone_dtod(&f.depth_val[0]).unwrap();
        let mut count = Vec::new();
        for i in 0..warm + reps {
            let t = time(&gpu, &timer, || count_tiles(&k, &s, &mut f).unwrap());
            if i >= warm {
                count.push(t);
            }
        }
        let names = [
            "radix_hist32",
            "onesweep_u32 p0",
            "onesweep_u32 p1",
            "onesweep_u32 p2",
            "onesweep_u32 p3",
            "scan_tiles",
            "emit",
            "tile_hist",
            "bin_scan",
            "onesweep_u16 p0",
            "onesweep_u16 p1",
        ];
        let mut per: Vec<Vec<f64>> = vec![Vec::new(); names.len()];
        let mut whole = Vec::new();
        let (mut ds, mut ts) = (Vec::new(), Vec::new());
        for i in 0..warm + reps {
            let keep = i >= warm;
            // Kernel by kernel (each between its own events).
            s.memcpy_dtod(&pk, &mut f.depth_key[0]).unwrap();
            s.memcpy_dtod(&pv, &mut f.depth_val[0]).unwrap();
            let mut t = Vec::with_capacity(names.len());
            {
                let FrameBuffers {
                    iter,
                    depth_key: [k0, k1],
                    depth_val: [v0, v1],
                    ..
                } = &mut f;
                let n = iter.status.slice(ST_N_VIS..ST_N_VIS + 1);
                t.push(time(&gpu, &timer, || {
                    sorter.hist_u32(&s, k0, &n, 32, None).unwrap()
                }));
                for p in 0..4 {
                    t.push(time(&gpu, &timer, || {
                        if p % 2 == 0 {
                            sorter
                                .pass_u32(&s, [&*k0, &*v0], [&mut *k1, &mut *v1], &n, p, 32, None)
                                .unwrap();
                        } else {
                            sorter
                                .pass_u32(&s, [&*k1, &*v1], [&mut *k0, &mut *v0], &n, p, 32, None)
                                .unwrap();
                        }
                    }));
                }
            }
            f.depth_sorted = 0;
            t.push(time(&gpu, &timer, || {
                scan_tiles(&k, &s, &mut sorter, &mut f).unwrap()
            }));
            t.push(time(&gpu, &timer, || emit(&k, &s, &mut f).unwrap()));
            t.push(time(&gpu, &timer, || {
                tile_hist(&k, &s, &sorter, &mut f).unwrap()
            }));
            t.push(time(&gpu, &timer, || bin_scan(&k, &s, &mut f).unwrap()));
            {
                let FrameBuffers {
                    iter,
                    tile_key: [k0, k1],
                    inst_v: [v0, v1],
                    tile_digit_offsets,
                    ..
                } = &mut f;
                let n = iter.status.slice(ST_N_INST..ST_N_INST + 1);
                let lo = tile_digit_offsets.slice(0..RADIX_BINS as usize);
                let hi = tile_digit_offsets.slice(RADIX_BINS as usize..2 * RADIX_BINS as usize);
                let passes = u16_passes(13, TILE_LO_BITS);
                t.push(time(&gpu, &timer, || {
                    sorter
                        .pass_u16(
                            &s,
                            (&*k0, &*v0),
                            (&mut *k1, &mut *v1),
                            &n,
                            passes[0].0,
                            passes[0].1,
                            &lo,
                            None,
                        )
                        .unwrap()
                }));
                t.push(time(&gpu, &timer, || {
                    sorter
                        .pass_u16(
                            &s,
                            (&*k1, &*v1),
                            (&mut *k0, &mut *v0),
                            &n,
                            passes[1].0,
                            passes[1].1,
                            &hi,
                            None,
                        )
                        .unwrap()
                }));
            }
            f.tile_sorted = 0;
            // Whole sequence, as one graph would run it.
            s.memcpy_dtod(&pk, &mut f.depth_key[0]).unwrap();
            s.memcpy_dtod(&pv, &mut f.depth_val[0]).unwrap();
            let whole_ms = time(&gpu, &timer, || {
                sort_depth(&k, &s, &mut sorter, &mut f).unwrap();
                bin_tiles(&k, &s, &mut sorter, &mut f).unwrap();
            });
            s.memcpy_dtod(&pk, &mut f.depth_key[0]).unwrap();
            s.memcpy_dtod(&pv, &mut f.depth_val[0]).unwrap();
            let d_ms = time(&gpu, &timer, || {
                sort_depth(&k, &s, &mut sorter, &mut f).unwrap()
            });
            scan_tiles(&k, &s, &mut sorter, &mut f).unwrap();
            emit(&k, &s, &mut f).unwrap();
            tile_hist(&k, &s, &sorter, &mut f).unwrap();
            bin_scan(&k, &s, &mut f).unwrap();
            let t_ms = time(&gpu, &timer, || {
                sort_tiles(&k, &s, &mut sorter, &mut f).unwrap()
            });
            if keep {
                for (p, x) in per.iter_mut().zip(&t) {
                    p.push(*x);
                }
                whole.push(whole_ms);
                ds.push(d_ms);
                ts.push(t_ms);
            }
        }
        let st = s.clone_dtoh(&f.iter.status).unwrap();
        assert_eq!(st[ST_OVERFLOW] & OVF_INSTANCES, 0);
        assert_eq!(
            st[ST_N_INST] as u64, wl.instances,
            "{}: GPU and CPU instance counts",
            wl.name
        );
        let row = Row {
            name: wl.name.clone(),
            n_gaussians: wl.n_gaussians,
            visible: st[ST_N_VIS],
            instances: st[ST_N_INST],
            chains: st[ST_N_CHAINS],
            tiles_hit: st[ST_N_TILES_HIT],
            kernels: names
                .iter()
                .zip(&per)
                .map(|(n, x)| (n.to_string(), Stats::from_samples(x)))
                .collect(),
            pipeline: Stats::from_samples(&whole),
            depth_sort: Stats::from_samples(&ds),
            tile_sort: Stats::from_samples(&ts),
            count_tiles: Stats::from_samples(&count),
        };
        println!(
            "\n{}: N {} V {} I {} ({:.2}/visible) chains {} tiles hit {}",
            row.name,
            row.n_gaussians,
            row.visible,
            row.instances,
            row.instances as f64 / row.visible.max(1) as f64,
            row.chains,
            row.tiles_hit
        );
        for (name, st) in &row.kernels {
            println!(
                "  {name:<16} {:>8.3} ms (p10 {:.3}, p90 {:.3})",
                st.median_ms, st.p10_ms, st.p90_ms
            );
        }
        let per_sum: f64 = row.kernels.iter().map(|(_, s)| s.median_ms).sum();
        println!(
            "  depth sort {:.3} ms, tile sort {:.3} ms, sort_depth + bin_tiles {:.3} ms (sum of kernels {per_sum:.3}); count_tiles {:.3} ms",
            row.depth_sort.median_ms,
            row.tile_sort.median_ms,
            row.pipeline.median_ms,
            row.count_tiles.median_ms
        );
        if dump {
            let dir = out_dir("sort-bench").join(&wl.name);
            std::fs::create_dir_all(&dir).unwrap();
            let nv = row.visible as usize;
            let ni = row.instances as usize;
            let nt = (wl.w.div_ceil(16) * wl.h.div_ceil(16)) as usize;
            write_raw(
                &dir.join("depth_keys.u32"),
                &s.clone_dtoh(&pk.slice(0..nv)).unwrap(),
            );
            write_raw(
                &dir.join("n_tiles.u32"),
                &s.clone_dtoh(&f.n_tiles.slice(0..nv)).unwrap(),
            );
            // Emitted (unsorted) instances: re-run emit into buffer 0.
            emit(&k, &s, &mut f).unwrap();
            write_raw(
                &dir.join("emit_keys.u16"),
                &s.clone_dtoh(&f.tile_key[0].slice(0..ni)).unwrap(),
            );
            write_raw(
                &dir.join("emit_vals.u32"),
                &s.clone_dtoh(&f.inst_v[0].slice(0..ni)).unwrap(),
            );
            tile_hist(&k, &s, &sorter, &mut f).unwrap();
            bin_scan(&k, &s, &mut f).unwrap();
            sort_tiles(&k, &s, &mut sorter, &mut f).unwrap();
            write_raw(
                &dir.join("ours_sorted_vals.u32"),
                &s.clone_dtoh(&f.inst_v[f.tile_sorted].slice(0..ni)).unwrap(),
            );
            write_raw(
                &dir.join("ours_tile_start.u32"),
                &s.clone_dtoh(&f.tile_start.slice(0..nt + 1)).unwrap(),
            );
            write_raw(
                &dir.join("ours_depth_order.u32"),
                &s.clone_dtoh(&f.depth_val[f.depth_sorted].slice(0..nv))
                    .unwrap(),
            );
            std::fs::write(
                dir.join("meta.json"),
                serde_json::json!({ "visible": nv, "instances": ni, "tiles": nt, "tile_bits": 13, "width": wl.w, "height": wl.h }).to_string(),
            )
            .unwrap();
            println!("  dumped to {}", dir.display());
        }
        rows.push(row);
    }
    let after = gpu_snapshot();
    println!("\nnvidia-smi before {before:?}\nnvidia-smi after  {after:?}");
    let out = out_dir("sort-bench");
    std::fs::create_dir_all(&out).unwrap();
    std::fs::write(
        out.join("results.json"),
        serde_json::to_string_pretty(&serde_json::json!({ "rows": rows, "before": before, "after": after, "config": format!("{:?}", sorter.config) })).unwrap(),
    )
    .unwrap();
}

/// The onesweep tuning sweep: depth keys and emitted tile keys of one garden
/// workload sorted under each `RadixConfig` (median of 15).
#[test]
#[ignore]
fn gpu_bench_radix_configs() {
    let Some(scene) = garden() else {
        eprintln!("skipped: no mipnerf360/garden");
        return;
    };
    let gpu = gpu();
    let s = gpu.stream.clone();
    let k = SplatKernels::load(&gpu, RasterVariant::default()).unwrap();
    let wl = garden_workload(&scene, 6_000_000, 10.0, 0, 1);
    let mut f = frame_for(&gpu, &wl);
    upload_frame(&gpu, &mut f, &frame_cam(wl.w, wl.h, 16), &wl.recs);
    let mut base = RadixSorter::new(&gpu, (wl.instances as u32).max(wl.recs.len() as u32)).unwrap();
    count_tiles(&k, &s, &mut f).unwrap();
    let pk = s.clone_dtod(&f.depth_key[0]).unwrap();
    let pv = s.clone_dtod(&f.depth_val[0]).unwrap();
    sort_depth(&k, &s, &mut base, &mut f).unwrap();
    scan_tiles(&k, &s, &mut base, &mut f).unwrap();
    emit(&k, &s, &mut f).unwrap();
    tile_hist(&k, &s, &base, &mut f).unwrap();
    bin_scan(&k, &s, &mut f).unwrap();
    let tk = s.clone_dtod(&f.tile_key[0]).unwrap();
    let tv = s.clone_dtod(&f.inst_v[0]).unwrap();
    let st = s.clone_dtoh(&f.iter.status).unwrap();
    println!("{}: V {} I {}", wl.name, st[ST_N_VIS], st[ST_N_INST]);
    let timer = GpuTimer::new(&gpu).unwrap();
    let mut configs = Vec::new();
    for threads in [256u32, 384, 512] {
        for ipt in [8u32, 12, 16, 20, 24] {
            for ballot in [false, true] {
                configs.push(RadixConfig {
                    threads,
                    ipt32: ipt,
                    ipt16: ipt,
                    min_blocks: 1,
                    rank_ballot: ballot,
                    ..Default::default()
                });
            }
        }
    }
    let mut results = Vec::new();
    for cfg in configs {
        if cfg.validate().is_err() {
            continue;
        }
        let mut sorter = RadixSorter::with_config(&gpu, base.max_items, cfg).unwrap();
        let (mut d, mut t) = (Vec::new(), Vec::new());
        for i in 0..18 {
            s.memcpy_dtod(&pk, &mut f.depth_key[0]).unwrap();
            s.memcpy_dtod(&pv, &mut f.depth_val[0]).unwrap();
            let dm = time(&gpu, &timer, || {
                sort_depth(&k, &s, &mut sorter, &mut f).unwrap()
            });
            s.memcpy_dtod(&tk, &mut f.tile_key[0]).unwrap();
            s.memcpy_dtod(&tv, &mut f.inst_v[0]).unwrap();
            let tm = time(&gpu, &timer, || {
                sort_tiles(&k, &s, &mut sorter, &mut f).unwrap()
            });
            if i >= 3 {
                d.push(dm);
                t.push(tm);
            }
        }
        let (d, t) = (Stats::from_samples(&d), Stats::from_samples(&t));
        println!(
            "threads {:>3} ipt {:>2} {:<6}: depth sort {:.3} ms, tile sort {:.3} ms",
            cfg.threads,
            cfg.ipt32,
            if cfg.rank_ballot { "ballot" } else { "match" },
            d.median_ms,
            t.median_ms
        );
        results.push(
            serde_json::json!({ "config": format!("{cfg:?}"), "depth_sort": d, "tile_sort": t }),
        );
    }
    let out = out_dir("sort-bench");
    std::fs::create_dir_all(&out).unwrap();
    std::fs::write(
        out.join("configs.json"),
        serde_json::to_string_pretty(&results).unwrap(),
    )
    .unwrap();
}

/// One pass of the late-training workload (N 6M, scale 10, view 0) through
/// count_tiles, sort_depth and bin_tiles, for Nsight Compute
/// (`SPL_SORT_PROFILE_SCALE` overrides the scale).
#[test]
#[ignore]
fn gpu_profile_pipeline_once() {
    let Some(scene) = garden() else {
        eprintln!("skipped: no mipnerf360/garden");
        return;
    };
    let scale = std::env::var("SPL_SORT_PROFILE_SCALE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10.0);
    let gpu = gpu();
    let s = gpu.stream.clone();
    let k = SplatKernels::load(&gpu, RasterVariant::default()).unwrap();
    let wl = garden_workload(&scene, 6_000_000, scale, 0, 1);
    let mut f = frame_for(&gpu, &wl);
    let items = (wl.instances as u32).max(wl.recs.len() as u32);
    let mut sorter = RadixSorter::new(&gpu, items).unwrap();
    upload_frame(&gpu, &mut f, &frame_cam(wl.w, wl.h, 16), &wl.recs);
    count_tiles(&k, &s, &mut f).unwrap();
    sort_depth(&k, &s, &mut sorter, &mut f).unwrap();
    bin_tiles(&k, &s, &mut sorter, &mut f).unwrap();
    gpu.sync().unwrap();
    let st = s.clone_dtoh(&f.iter.status).unwrap();
    println!("{}: V {} I {}", wl.name, st[ST_N_VIS], st[ST_N_INST]);
}
