//! Raster kernel tests (owner: raster E3). CPU: variant and buffer checks.
//! GPU (`#[ignore]`; on the host `apps/image-to-3d/scripts/gpu-test.sh
//! <checkout> raster::`): forward and backward against the CPU f64 reference
//! of `scene` for every implemented variant and the spec-raster 6.5 edge
//! cases, determinism, the three modes, the overflow no-op, and gsplat
//! 1.5.3's `rasterize_to_pixels` through gs-oracle (benchmarks/image-to-3d/ORACLE.md).

use std::path::Path;

use super::scene::{
    PixelGradsHost, Scene2d, SceneOpts, Staged, TileLists, host, random_scene, record, reference,
    stage,
};
use super::*;
use crate::kernels::layout::{FLAG_ABS_GRAD, ST_OVERFLOW, TILE_W};
use crate::kernels::{BlendSemantics, SplatRec};
use crate::testkit::npz::{NpyData, Npz};
use crate::testkit::oracle::{case_path, stage_dir, stage_ours};
use crate::testkit::tol::{Diff, Tol, check_close};

#[test]
fn unimplemented_variants_are_refused() {
    let d = RasterVariant::default();
    assert!(check_variant(&d).is_ok());
    assert!(check_variant(&RasterVariant { rec32: true, ..d }).is_err());
    assert!(
        check_variant(&RasterVariant {
            deterministic: true,
            ..d
        })
        .is_err()
    );
    assert_eq!(ckpt_words(&d), 4);
    assert_eq!(
        ckpt_words(&RasterVariant {
            ckpt_f16: true,
            extra_ch: 1,
            ..d
        }),
        3
    );
}

#[test]
fn raster_units_include_the_shared_header() {
    for src in [&FWD_SOURCE, &BWD_SOURCE] {
        let text = src.text();
        let common = text.find("#line 1 \"raster_common.cuh\"").unwrap();
        let own = text
            .find(&format!("#line 1 \"{}.cu\"", src.module))
            .unwrap();
        assert!(common < own, "{}", src.module);
    }
}

fn gpu() -> Gpu {
    Gpu::open().expect("open GPU")
}

/// Everything the two kernels produce, downloaded.
struct Run {
    rgb: Vec<f32>,
    trans: Vec<f32>,
    nc: Vec<u32>,
    extra: Vec<f32>,
    tile_max_nc: Vec<u32>,
    g: Vec<f32>,
    abs: Vec<f32>,
    g_extra: Vec<f32>,
}

fn download(gpu: &Gpu, st: &Staged, n_tiles: usize) -> Run {
    Run {
        rgb: host(gpu, &st.out.rgb).unwrap(),
        trans: host(gpu, &st.out.trans).unwrap(),
        nc: host(gpu, &st.out.n_contrib).unwrap(),
        extra: st
            .out
            .extra
            .as_ref()
            .map_or(Vec::new(), |x| host(gpu, x).unwrap()),
        tile_max_nc: host(gpu, &st.f.tile_max_nc).unwrap()[..n_tiles].to_vec(),
        g: host(gpu, &st.g2d.g).unwrap(),
        abs: st
            .g2d
            .abs
            .as_ref()
            .map_or(Vec::new(), |x| host(gpu, x).unwrap()),
        g_extra: st
            .g2d
            .extra
            .as_ref()
            .map_or(Vec::new(), |x| host(gpu, x).unwrap()),
    }
}

/// Forward (train) + backward on `scene` with `lists`.
fn run(
    gpu: &Gpu,
    v: RasterVariant,
    scene: &Scene2d,
    lists: &TileLists,
    pg: &PixelGradsHost,
    flags: u32,
) -> Run {
    let k = SplatKernels::load(gpu, v).unwrap();
    let mut st = stage(gpu, &v, scene, lists, pg, flags).unwrap();
    raster_fwd(&k, &gpu.stream, &mut st.f, RenderMode::Train, &mut st.out).unwrap();
    raster_bwd(&k, &gpu.stream, &mut st.f, &st.out, &st.pg, &mut st.g2d).unwrap();
    gpu.sync().unwrap();
    download(gpu, &st, lists.n_tiles() as usize)
}

/// Gradient tolerances by variant: spec-raster 6.4 for f32 checkpoints;
/// fp16 checkpoints perturb A and T of deep Gaussians by ≈2⁻¹¹ (an ablation).
fn grad_tol(v: &RasterVariant) -> Tol {
    if v.ckpt_f16 {
        Tol {
            rel_l2: Some(2e-2),
            min_cosine: Some(0.999),
            ..Tol::NONE
        }
    } else {
        Tol::GRAD_2D
    }
}

const GROUPS: [&str; 9] = ["mx", "my", "A", "B", "C", "o", "r", "g", "b"];

/// Checks a run against the f64 reference; returns one summary line.
fn check_against_reference(
    name: &str,
    v: &RasterVariant,
    scene: &Scene2d,
    lists: &TileLists,
    pg: &PixelGradsHost,
    got: &Run,
    abs_on: bool,
) -> String {
    let e = v.extra_ch as usize;
    let (want, wg) = reference(scene, lists, v.blend, e, pg);
    let np = (scene.width * scene.height) as usize;
    let nv = scene.recs.len();
    let img = check_close(
        &format!("{name} rgb"),
        &got.rgb[..3 * np],
        &want.rgb,
        &Tol::IMAGE,
    )
    .unwrap_or_else(|err| panic!("{err:#}"));
    check_close(
        &format!("{name} T"),
        &got.trans[..np],
        &want.trans,
        &Tol::TRANSMITTANCE,
    )
    .unwrap_or_else(|err| panic!("{err:#}"));
    let bad_nc = (0..np).filter(|&i| got.nc[i] != want.nc[i]).count();
    assert_eq!(
        bad_nc, 0,
        "{name}: {bad_nc} pixels with a different n_contrib"
    );
    assert_eq!(got.tile_max_nc, want.tile_max_nc, "{name}: tile_max_nc");
    if e > 0 {
        check_close(
            &format!("{name} extra"),
            &got.extra[..e * np],
            &want.extra,
            &Tol::IMAGE,
        )
        .unwrap_or_else(|err| panic!("{err:#}"));
    }
    let tol = grad_tol(v);
    let mut worst = (String::new(), 1.0f64, 0.0f64);
    for (k, gname) in GROUPS.iter().enumerate() {
        let d = Diff::new(
            &format!("{name} d{gname}"),
            &got.g[k * nv..(k + 1) * nv],
            &wg.g[k * nv..(k + 1) * nv],
            tol.elem_rel.map_or(0.0, |x| x.1),
        );
        d.check(&tol).unwrap_or_else(|err| panic!("{err:#}"));
        if d.rel_l2 > worst.2 {
            worst = (gname.to_string(), d.cosine, d.rel_l2);
        }
    }
    if abs_on {
        check_close(&format!("{name} abs"), &got.abs[..2 * nv], &wg.abs, &tol)
            .unwrap_or_else(|err| panic!("{err:#}"));
    }
    if e > 0 {
        check_close(
            &format!("{name} dextra"),
            &got.g_extra[..e * nv],
            &wg.extra,
            &tol,
        )
        .unwrap_or_else(|err| panic!("{err:#}"));
    }
    format!(
        "{name:<26} I {:>7}  img max {:.1e} mean {:.1e}  worst grad d{} cos {:.8} relL2 {:.1e}",
        lists.instances(),
        img.max_abs,
        img.mean_abs,
        worst.0,
        worst.1,
        worst.2
    )
}

fn variant(f: impl FnOnce(&mut RasterVariant)) -> RasterVariant {
    let mut v = RasterVariant::default();
    f(&mut v);
    v
}

/// Every implemented variant, on loose (rectangle) and tight (exact) lists,
/// against the f64 reference.
#[test]
#[ignore]
fn gpu_raster_matches_reference_across_variants() {
    let gpu = gpu();
    let base = SceneOpts::default();
    let dense = SceneOpts {
        n: 4000,
        width: 83,
        height: 61,
        max_sigma: 14.0,
        max_opacity: 1.0,
        ..base
    };
    let cases: Vec<(&str, RasterVariant, SceneOpts, bool)> = vec![
        ("default rect", RasterVariant::default(), base, false),
        ("default exact", RasterVariant::default(), base, true),
        ("K=1", variant(|v| v.chain_k = 1), base, false),
        ("K=2", variant(|v| v.chain_k = 2), base, false),
        ("K=8", variant(|v| v.chain_k = 8), base, false),
        ("tile 16x12", variant(|v| v.tile_h = 12), base, false),
        (
            "fastergs",
            variant(|v| v.blend = BlendSemantics::FasterGs),
            base,
            false,
        ),
        ("depth channel", variant(|v| v.extra_ch = 1), base, false),
        ("4 extra channels", variant(|v| v.extra_ch = 4), base, false),
        ("dense K=4", RasterVariant::default(), dense, true),
        ("dense K=1", variant(|v| v.chain_k = 1), dense, true),
        (
            "dense K=8 16x12",
            variant(|v| {
                v.chain_k = 8;
                v.tile_h = 12
            }),
            dense,
            true,
        ),
        (
            "dense fastergs",
            variant(|v| v.blend = BlendSemantics::FasterGs),
            dense,
            true,
        ),
        (
            "dense fp16 ckpt",
            variant(|v| v.ckpt_f16 = true),
            dense,
            true,
        ),
    ];
    for (i, (name, v, o, exact)) in cases.into_iter().enumerate() {
        let scene = random_scene(100 + i as u64, &o);
        let lists = if exact {
            TileLists::exact(&scene, v.tile_h as u32)
        } else {
            TileLists::rect(&scene, v.tile_h as u32)
        };
        let np = (scene.width * scene.height) as usize;
        let pg = PixelGradsHost::random(7 + i as u64, np, v.extra_ch as usize, true);
        let got = run(&gpu, v, &scene, &lists, &pg, FLAG_ABS_GRAD);
        println!(
            "{}",
            check_against_reference(name, &v, &scene, &lists, &pg, &got, true)
        );
    }
}

/// Clamped alpha (o·G > 0.99), pixels that terminate, no d_alpha, abs off.
#[test]
#[ignore]
fn gpu_raster_clamp_termination_and_optional_inputs() {
    let gpu = gpu();
    let v = RasterVariant::default();
    let o = SceneOpts {
        n: 1500,
        max_opacity: 1.0,
        max_sigma: 10.0,
        ..Default::default()
    };
    let scene = random_scene(55, &o);
    let lists = TileLists::exact(&scene, 16);
    let np = (scene.width * scene.height) as usize;
    let pg = PixelGradsHost::random(3, np, 0, false);
    let got = run(&gpu, v, &scene, &lists, &pg, 0);
    let (want, _) = reference(&scene, &lists, v.blend, 0, &pg);
    let terminated = want.trans.iter().filter(|&&t| t < 2e-4).count();
    let clamped = scene.recs.iter().filter(|r| r.o > 0.99).count();
    println!(
        "{}  ({terminated} of {np} pixels terminate, {clamped} records can clamp)",
        check_against_reference("clamp+stop", &v, &scene, &lists, &pg, &got, false)
    );
    assert!(terminated > np / 10 && clamped > 0);
    assert!(
        got.abs.iter().all(|&x| x == 0.0),
        "abs gradients without FLAG_ABS_GRAD"
    );
}

/// Edge cases of spec-raster 6.5 that reach the raster.
#[test]
#[ignore]
fn gpu_raster_edge_cases() {
    let gpu = gpu();
    let v = RasterVariant::default();
    let mk = |w: u32, h: u32, recs: Vec<SplatRec>| Scene2d {
        width: w,
        height: h,
        recs,
        bg: [0.25, 0.5, 0.75],
    };
    let big = |mx: f32, my: f32, s: f64, o: f32| {
        record(
            [mx, my],
            [1.0 / (s * s + 0.3), 0.0, 1.0 / (s * s + 0.3)],
            o,
            [0.9, 0.4, 0.1],
            2.0,
            0,
        )
    };
    let mut cases: Vec<(&str, Scene2d)> = vec![
        ("no Gaussians", mk(40, 30, Vec::new())),
        (
            "1x1 image",
            mk(1, 1, vec![big(0.3, 0.7, 0.8, 0.7), big(0.5, 0.5, 2.0, 0.5)]),
        ),
        ("whole-image splat", {
            let mut s = random_scene(8, &SceneOpts::default());
            s.recs.push(big(30.0, 25.0, 400.0, 0.6));
            s
        }),
        (
            "17x33 image, tile 16",
            random_scene(
                9,
                &SceneOpts {
                    width: 17,
                    height: 33,
                    ..Default::default()
                },
            ),
        ),
    ];
    // One tile with more than 65,535 listed Gaussians (u16-free index math,
    // many batches and chains): faint splats that rarely terminate a pixel.
    let mut crowd = random_scene(
        10,
        &SceneOpts {
            n: 70_000,
            width: 16,
            height: 16,
            max_sigma: 3.0,
            max_opacity: 0.02,
            margin: 0.0,
            ..Default::default()
        },
    );
    crowd.bg = [0.0; 3];
    cases.push((">65535 in one tile", crowd));
    for (i, (name, scene)) in cases.iter().enumerate() {
        let lists = TileLists::rect(scene, 16);
        let np = (scene.width * scene.height) as usize;
        let pg = PixelGradsHost::random(30 + i as u64, np, 0, true);
        let got = run(&gpu, v, scene, &lists, &pg, FLAG_ABS_GRAD);
        println!(
            "{}",
            check_against_reference(name, &v, scene, &lists, &pg, &got, true)
        );
        if scene.recs.is_empty() {
            assert!(got.rgb[..np].iter().all(|&x| x == 0.25));
            assert!(got.trans[..np].iter().all(|&x| x == 1.0));
        }
    }
}

/// Forward outputs are bitwise identical run to run; Infer renders the same
/// image as Train without touching n_contrib; Count mode counts exactly the
/// masked (pixel, Gaussian) pairs the reference blends; an overflowed
/// iteration is a no-op for both kernels.
#[test]
#[ignore]
fn gpu_raster_modes_determinism_and_overflow() {
    let gpu = gpu();
    let v = RasterVariant::default();
    let k = SplatKernels::load(&gpu, v).unwrap();
    let s = &gpu.stream;
    let scene = random_scene(
        77,
        &SceneOpts {
            n: 2000,
            width: 97,
            height: 71,
            ..Default::default()
        },
    );
    let lists = TileLists::exact(&scene, 16);
    let np = (scene.width * scene.height) as usize;
    let pg = PixelGradsHost::random(4, np, 0, true);
    let mut st = stage(&gpu, &v, &scene, &lists, &pg, FLAG_ABS_GRAD).unwrap();

    raster_fwd(&k, s, &mut st.f, RenderMode::Train, &mut st.out).unwrap();
    let first = download(&gpu, &st, lists.n_tiles() as usize);
    raster_fwd(&k, s, &mut st.f, RenderMode::Train, &mut st.out).unwrap();
    let second = download(&gpu, &st, lists.n_tiles() as usize);
    assert!(
        first
            .rgb
            .iter()
            .zip(&second.rgb)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    assert_eq!(first.nc, second.nc);

    let sentinel = vec![u32::MAX; np];
    s.memcpy_htod(&sentinel, &mut st.out.n_contrib).unwrap();
    raster_fwd(&k, s, &mut st.f, RenderMode::Infer, &mut st.out).unwrap();
    let infer = download(&gpu, &st, lists.n_tiles() as usize);
    assert!(
        first
            .rgb
            .iter()
            .zip(&infer.rgb)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    assert!(
        infer.nc[..np].iter().all(|&x| x == u32::MAX),
        "Infer wrote n_contrib"
    );

    // Count mode: a checkerboard of 8x8 blocks.
    let mask: Vec<u8> = (0..np)
        .map(|i| {
            let (x, y) = (i % scene.width as usize, i / scene.width as usize);
            u8::from((x / 8 + y / 8) % 2 == 0)
        })
        .collect();
    let mask_d = s.clone_htod(&mask).unwrap();
    raster_fwd(
        &k,
        s,
        &mut st.f,
        RenderMode::Count { mask: &mask_d },
        &mut st.out,
    )
    .unwrap();
    let counts = host(&gpu, st.out.count.as_ref().unwrap()).unwrap();
    let (want, _) = reference(&scene, &lists, v.blend, 0, &pg);
    let mut want_counts = vec![0u32; scene.recs.len()];
    for (pix, pairs) in want.pairs.iter().enumerate() {
        if mask[pix] != 0 {
            for &v in pairs {
                want_counts[v as usize] += 1;
            }
        }
    }
    assert_eq!(&counts[..scene.recs.len()], &want_counts[..]);
    println!(
        "count mode: {} masked (pixel, Gaussian) pairs, all equal",
        want_counts.iter().sum::<u32>()
    );

    // Overflow: neither kernel touches anything.
    raster_fwd(&k, s, &mut st.f, RenderMode::Train, &mut st.out).unwrap();
    raster_bwd(&k, s, &mut st.f, &st.out, &st.pg, &mut st.g2d).unwrap();
    let before = download(&gpu, &st, lists.n_tiles() as usize);
    let mut status = host(&gpu, &st.f.iter.status).unwrap();
    status[ST_OVERFLOW] = 2;
    s.memcpy_htod(&status, &mut st.f.iter.status).unwrap();
    s.memset_zeros(&mut st.out.rgb).unwrap();
    raster_fwd(&k, s, &mut st.f, RenderMode::Train, &mut st.out).unwrap();
    raster_bwd(&k, s, &mut st.f, &st.out, &st.pg, &mut st.g2d).unwrap();
    let after = download(&gpu, &st, lists.n_tiles() as usize);
    assert!(
        after.rgb.iter().all(|&x| x == 0.0),
        "forward ran on overflow"
    );
    assert_eq!(before.g, after.g, "backward ran on overflow");
}

/// K = 1, 2, 4, 8 give the same gradients up to the order of float atomics.
#[test]
#[ignore]
fn gpu_raster_chain_lengths_agree() {
    let gpu = gpu();
    let scene = random_scene(
        5,
        &SceneOpts {
            n: 6000,
            width: 128,
            height: 96,
            max_sigma: 16.0,
            ..Default::default()
        },
    );
    let lists = TileLists::exact(&scene, 16);
    let np = (scene.width * scene.height) as usize;
    let pg = PixelGradsHost::random(6, np, 0, true);
    let base = run(&gpu, variant(|v| v.chain_k = 1), &scene, &lists, &pg, 0);
    for k in [2u8, 4, 8] {
        let got = run(&gpu, variant(|v| v.chain_k = k), &scene, &lists, &pg, 0);
        assert!(
            base.rgb
                .iter()
                .zip(&got.rgb)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        // Not bitwise even without atomics: a chain start re-derives the
        // remaining colour A = Cf − C_ckpt, while inside a chain A is carried
        // by subtraction, so the f32 rounding of A differs (T is identical).
        // Both sides match the f64 reference to ~2e-6 (the variant test).
        let d = Diff::new(&format!("K={k} vs K=1"), &got.g, &base.g, 0.0);
        println!("{d}");
        assert!(d.rel_l2 < 1e-5 && d.cosine > 0.999_999_9, "{d}");
    }
}

// ---------------------------------------------------------------- gsplat oracle

const ORACLE_CASES: [(&str, u64, SceneOpts); 3] = [
    (
        "raster_small",
        11,
        SceneOpts {
            n: 300,
            width: 64,
            height: 48,
            max_sigma: 10.0,
            max_opacity: 0.98,
            clamp_frac: 0.0,
            margin: 6.0,
            bg: [0.1, 0.2, 0.3],
        },
    ),
    (
        "raster_odd",
        12,
        SceneOpts {
            n: 1200,
            width: 67,
            height: 53,
            max_sigma: 16.0,
            max_opacity: 0.98,
            clamp_frac: 0.0,
            margin: 6.0,
            bg: [0.0, 0.0, 0.0],
        },
    ),
    (
        "raster_dense",
        13,
        SceneOpts {
            n: 20_000,
            width: 160,
            height: 120,
            max_sigma: 12.0,
            max_opacity: 0.98,
            clamp_frac: 0.0,
            margin: 6.0,
            bg: [0.3, 0.3, 0.3],
        },
    ),
];

/// gsplat `rasterize` inputs for a scene (layout in ORACLE.md): our records'
/// unscaled conics, opacity-aware radii (+1 px) so gsplat's rectangle
/// binning holds every contributing pixel, and our pixel gradients in HWC.
fn oracle_input(scene: &Scene2d, pg: &PixelGradsHost) -> Npz {
    let n = scene.recs.len();
    let (w, h) = (scene.width as usize, scene.height as usize);
    let np = w * h;
    let mut z = Npz::new();
    let mut means = Vec::with_capacity(2 * n);
    let mut conics = Vec::with_capacity(3 * n);
    let mut colors = Vec::with_capacity(3 * n);
    let mut opac = Vec::with_capacity(n);
    let mut radii = Vec::with_capacity(2 * n);
    let mut depths = Vec::with_capacity(n);
    for r in &scene.recs {
        let c = super::scene::conic(r);
        means.extend([r.mx, r.my]);
        conics.extend(c.map(|x| x as f32));
        colors.extend([r.r, r.g, r.b]);
        opac.push(r.o);
        let k = 2.0 * (255.0 * r.o as f64).ln();
        let det = c[0] * c[2] - c[1] * c[1];
        let rx = (k * c[2] / det).sqrt().ceil() as i32 + 1;
        let ry = (k * c[0] / det).sqrt().ceil() as i32 + 1;
        radii.extend([rx, ry]);
        depths.push(r.z);
    }
    z.f32("means2d", &[1, n, 2], means).unwrap();
    z.f32("conics", &[1, n, 3], conics).unwrap();
    z.f32("colors", &[1, n, 3], colors).unwrap();
    z.f32("opacities", &[1, n], opac).unwrap();
    z.insert("radii", &[1, n, 2], NpyData::I32(radii)).unwrap();
    z.f32("depths", &[1, n], depths).unwrap();
    z.i32_scalar("width", w as i32).unwrap();
    z.i32_scalar("height", h as i32).unwrap();
    z.i32_scalar("tile_size", TILE_W as i32).unwrap();
    z.f32("backgrounds", &[1, 3], scene.bg.to_vec()).unwrap();
    z.i32_scalar("absgrad", 1).unwrap();
    let hwc: Vec<f32> = (0..np)
        .flat_map(|p| (0..3).map(move |c| (c, p)))
        .map(|(c, p)| pg.d_rgb[c * np + p])
        .collect();
    z.f32("v_render_colors", &[1, h, w, 3], hwc).unwrap();
    z.f32(
        "v_render_alphas",
        &[1, h, w, 1],
        pg.d_alpha.clone().unwrap(),
    )
    .unwrap();
    z
}

fn f32s(z: &Npz, key: &str) -> Vec<f32> {
    z.get(key)
        .unwrap()
        .to_f64()
        .into_iter()
        .map(|x| x as f32)
        .collect()
}

/// Our raster vs gsplat 1.5.3 `rasterize_to_pixels` (gs-oracle `rasterize`)
/// on gsplat's own tile lists and on our exact lists. Two passes on the GPU
/// host: the first run stages `<oracle dir>/raster_*.rasterize.in.npz` and
/// fails; `gs-oracle batch ~/spl-work/out/image-to-3d/oracle` then writes the
/// outputs and the second run compares (ORACLE.md, "Raster kernels").
#[test]
#[ignore]
fn gpu_raster_matches_gsplat_oracle() {
    let dir = stage_dir();
    let mut missing = Vec::new();
    for (name, seed, o) in ORACLE_CASES {
        let scene = random_scene(seed, &o);
        let np = (o.width * o.height) as usize;
        let pg = PixelGradsHost::random(seed + 1000, np, 0, true);
        let input = oracle_input(&scene, &pg);
        let inp = case_path(&dir, name, "rasterize", "in");
        let stale = Npz::read(&inp).map_or(true, |z| z != input);
        if stale {
            input.write(&inp).unwrap();
        }
        let outp = case_path(&dir, name, "rasterize", "out");
        if stale || !outp.is_file() {
            missing.push(outp);
        }
    }
    assert!(
        missing.is_empty(),
        "staged gsplat cases; run `gs-oracle batch {}` on the host, then rerun ({missing:?})",
        host_path(&dir).display()
    );
    let gpu = gpu();
    let v = RasterVariant::default();
    for (name, seed, o) in ORACLE_CASES {
        let scene = random_scene(seed, &o);
        let (w, h) = (o.width as usize, o.height as usize);
        let np = w * h;
        let nv = scene.recs.len();
        let pg = PixelGradsHost::random(seed + 1000, np, 0, true);
        let gs = Npz::read(&case_path(&dir, name, "rasterize", "out")).unwrap();
        // gsplat's lists: isect_offsets [1, TH, TW] and flatten_ids [M].
        let offsets: Vec<u32> = gs
            .get("isect_offsets")
            .unwrap()
            .to_f64()
            .iter()
            .map(|&x| x as u32)
            .collect();
        let ids: Vec<u32> = gs
            .get("flatten_ids")
            .unwrap()
            .to_f64()
            .iter()
            .map(|&x| x as u32)
            .collect();
        let mut tile_start = offsets;
        tile_start.push(ids.len() as u32);
        let gs_lists = TileLists {
            tile_h: 16,
            grid_w: o.width.div_ceil(16),
            grid_h: o.height.div_ceil(16),
            tile_start,
            inst_v: ids,
        };
        let colors = f32s(&gs, "render_colors"); // HWC
        let want_rgb: Vec<f32> = (0..3 * np).map(|i| colors[(i % np) * 3 + i / np]).collect();
        let want_t: Vec<f32> = f32s(&gs, "render_alphas").iter().map(|a| 1.0 - a).collect();
        let vm = f32s(&gs, "v_means2d");
        let vc = f32s(&gs, "v_conics");
        let vcol = f32s(&gs, "v_colors");
        let vo = f32s(&gs, "v_opacities");
        let vabs = f32s(&gs, "v_means2d_abs");
        let soa = |src: &[f32], stride: usize, k: usize| -> Vec<f32> {
            (0..nv).map(|i| src[i * stride + k]).collect()
        };
        let want_g: Vec<Vec<f32>> = vec![
            soa(&vm, 2, 0),
            soa(&vm, 2, 1),
            soa(&vc, 3, 0),
            soa(&vc, 3, 1),
            soa(&vc, 3, 2),
            vo.clone(),
            soa(&vcol, 3, 0),
            soa(&vcol, 3, 1),
            soa(&vcol, 3, 2),
        ];
        let exact = TileLists::exact(&scene, 16);
        for (lname, lists) in [("gsplat lists", &gs_lists), ("exact lists", &exact)] {
            let got = run(&gpu, v, &scene, lists, &pg, FLAG_ABS_GRAD);
            let tag = format!("{name} {lname}");
            let img = check_close(
                &format!("{tag} rgb"),
                &got.rgb[..3 * np],
                &want_rgb,
                &Tol::IMAGE,
            )
            .unwrap_or_else(|e| panic!("{e:#}"));
            let tr = check_close(
                &format!("{tag} T"),
                &got.trans[..np],
                &want_t,
                &Tol::TRANSMITTANCE,
            )
            .unwrap_or_else(|e| panic!("{e:#}"));
            let mut line = format!(
                "{tag:<28} I {:>7}  rgb max {:.1e} mean {:.1e}  T max {:.1e}",
                lists.instances(),
                img.max_abs,
                img.mean_abs,
                tr.max_abs
            );
            // gsplat's backward recovers T back to front by division, so on
            // deep pixels its own error against float64 reaches 1-3e-4
            // (`gs-oracle check`, ORACLE.md); against gsplat we allow that,
            // against the f64 reference we hold spec-raster 6.4.
            let vs_gsplat = Tol {
                rel_l2: Some(1e-3),
                min_cosine: Some(0.9999),
                ..Tol::NONE
            };
            let (_, wr) = reference(&scene, lists, v.blend, 0, &pg);
            let (mut worst_gs, mut worst_ref) = (0f64, 0f64);
            for (k, gname) in GROUPS.iter().enumerate() {
                let ours = &got.g[k * nv..(k + 1) * nv];
                let d = check_close(&format!("{tag} d{gname}"), ours, &want_g[k], &vs_gsplat)
                    .unwrap_or_else(|e| panic!("{e:#}"));
                let r = check_close(
                    &format!("{tag} d{gname} vs f64"),
                    ours,
                    &wr.g[k * nv..(k + 1) * nv],
                    &Tol::GRAD_2D,
                )
                .unwrap_or_else(|e| panic!("{e:#}"));
                worst_gs = worst_gs.max(d.rel_l2);
                worst_ref = worst_ref.max(r.rel_l2);
            }
            let mut want_abs = soa(&vabs, 2, 0);
            want_abs.extend(soa(&vabs, 2, 1));
            let d = check_close(
                &format!("{tag} abs"),
                &got.abs[..2 * nv],
                &want_abs,
                &vs_gsplat,
            )
            .unwrap_or_else(|e| panic!("{e:#}"));
            line += &format!(
                "  grads relL2 vs gsplat ≤ {worst_gs:.1e} (abs {:.1e}), vs f64 ≤ {worst_ref:.1e}",
                d.rel_l2
            );
            println!("{line}");
            if lname == "gsplat lists" {
                let ours = oracle_output(&scene, &got, &gs, &pg);
                stage_ours(&dir, name, "rasterize", &ours).unwrap();
            }
        }
    }
    println!(
        "our outputs: {}/<case>.rasterize.ours.npz; on the host `gs-oracle check rasterize <in> <ours>` grades them against float64",
        host_path(&dir).display()
    );
}

/// Our results in gs-oracle's `rasterize` output layout, for
/// `gs-oracle check` (which rebuilds the float64 reference on the
/// candidate's lists: we ran on gsplat's, so those are copied over).
/// `v_backgrounds` = Σ_px T·dL/dC is not a raster output; it is summed here
/// in f64 from our T so the file is complete.
fn oracle_output(scene: &Scene2d, got: &Run, gs: &Npz, pg: &PixelGradsHost) -> Npz {
    let (w, h) = (scene.width as usize, scene.height as usize);
    let (np, nv) = (w * h, scene.recs.len());
    let mut z = Npz::new();
    for key in ["isect_ids", "flatten_ids", "isect_offsets"] {
        z.0.insert(key.to_string(), gs.get(key).unwrap().clone());
    }
    let v_bg: Vec<f32> = (0..3)
        .map(|c| {
            (0..np)
                .map(|p| got.trans[p] as f64 * pg.d_rgb[c * np + p] as f64)
                .sum::<f64>() as f32
        })
        .collect();
    z.f32("v_backgrounds", &[1, 3], v_bg).unwrap();
    let aos = |planes: &[usize]| -> Vec<f32> {
        (0..nv)
            .flat_map(|i| planes.iter().map(move |&k| (k, i)))
            .map(|(k, i)| got.g[k * nv + i])
            .collect()
    };
    let hwc: Vec<f32> = (0..np)
        .flat_map(|p| (0..3).map(move |c| got.rgb[c * np + p]))
        .collect();
    z.f32("render_colors", &[1, h, w, 3], hwc).unwrap();
    z.f32(
        "render_alphas",
        &[1, h, w, 1],
        got.trans[..np].iter().map(|t| 1.0 - t).collect(),
    )
    .unwrap();
    z.f32("v_means2d", &[1, nv, 2], aos(&[0, 1])).unwrap();
    z.f32("v_conics", &[1, nv, 3], aos(&[2, 3, 4])).unwrap();
    z.f32("v_opacities", &[1, nv], aos(&[5])).unwrap();
    z.f32("v_colors", &[1, nv, 3], aos(&[6, 7, 8])).unwrap();
    let abs: Vec<f32> = (0..nv)
        .flat_map(|i| [got.abs[i], got.abs[nv + i]])
        .collect();
    z.f32("v_means2d_abs", &[1, nv, 2], abs).unwrap();
    z
}

/// `/out/...` inside the container is `~/spl-work/out/...` on the host.
fn host_path(p: &Path) -> std::path::PathBuf {
    match p.strip_prefix("/out") {
        Ok(rest) => Path::new("~/spl-work/out").join(rest),
        Err(_) => p.to_path_buf(),
    }
}
