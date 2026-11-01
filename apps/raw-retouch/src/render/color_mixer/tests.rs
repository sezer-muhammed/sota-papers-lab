//! Color Mixer and Calibration tests. The CPU ones describe the sliders;
//! the GPU ones (`#[ignore]`, run on the GPU host with
//! `gpu-test <checkout> color_mixer --ignored`) check the kernel against
//! [`apply`], render the samples, and write strips under /out/render-color/.

use std::path::PathBuf;
use std::sync::LazyLock;
use std::time::Instant;

use spl_gpu::cudarc::driver::{LaunchConfig, PushKernelArg};

use super::super::layout::{CAND_STRIDE, MIXER_LEN};
use super::super::testkit;
use super::super::{CudaContextExt, RenderEngine, SOURCE, curves};
use super::*;
use crate::types::{Gpu, PARAM_SPECS, ParamGroup, Rgb8Image};

fn neutral() -> Params {
    Params::neutral(5500.0, 0.0)
}

fn packed(p: &Params) -> Vec<f32> {
    let mut m = vec![f32::NAN; MIXER_LEN];
    pack(p, &mut m);
    m
}

fn lch(lab: [f32; 3]) -> (f32, f32, f32) {
    (
        lab[0],
        lab[1].hypot(lab[2]),
        lab[2].atan2(lab[1]).to_degrees().rem_euclid(360.0),
    )
}

fn from_lch(l: f32, c: f32, h: f32) -> [f32; 3] {
    let (s, co) = h.to_radians().sin_cos();
    [l, c * co, c * s]
}

/// Signed hue difference b - a in degrees, in (-180, 180].
fn dhue(a: f32, b: f32) -> f32 {
    let d = (b - a).rem_euclid(360.0);
    if d > 180.0 { d - 360.0 } else { d }
}

/// A small deterministic generator in -1..1.
fn rng(seed: u64) -> impl FnMut() -> f32 {
    let mut state = seed;
    move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f64 / (1u64 << 31) as f64) as f32 * 2.0 - 1.0
    }
}

/// Every Color Mixer slider and Calibration slider drawn at random (half
/// of them pinned to +-100), shadows tint included unless `tint` is false.
fn random_mixer(seed: u64, base: Params, tint: bool) -> Params {
    let mut next = rng(seed);
    let mut a = base.to_array();
    for (v, spec) in a.iter_mut().zip(&PARAM_SPECS) {
        if matches!(spec.group, ParamGroup::ColorMixer | ParamGroup::Calibration) {
            let r = next();
            *v = if next() > 0.0 {
                100.0 * r.signum()
            } else {
                100.0 * r
            };
        }
    }
    let mut p = Params::from_array(a);
    if !tint {
        p.shadow_tint = 0.0;
    }
    p
}

#[test]
fn defines_match_cuda() {
    let src = include_str!("../../../cuda/color_mixer.cu");
    let define = |name: &str| -> f32 {
        let line = src
            .lines()
            .find_map(|l| l.strip_prefix(&format!("#define {name} ")))
            .unwrap_or_else(|| panic!("#define {name}"));
        line.trim()
            .trim_start_matches('(')
            .trim_end_matches(')')
            .trim_end_matches('f')
            .parse()
            .unwrap_or_else(|_| panic!("{name}: {line}"))
    };
    assert_eq!(define("MIX_ON"), MIX_ON as f32);
    assert_eq!(define("MIX_TINT"), MIX_TINT as f32);
    assert_eq!(define("MIX_LUT"), MIX_LUT as f32);
    assert_eq!(define("MIX_N"), MIX_N as f32);
    assert_eq!(define("MIX_CN"), MIX_CN);
    assert_eq!(define("MIX_CL"), MIX_CL);
    assert_eq!(define("MIX_TINT_L1"), MIX_TINT_L1);
    assert!(src.lines().any(|l| l == "#define COLOR_MIXER_HOOK"));
    const { assert!(MIX_ON >= 31 && MIX_TINT > MIX_ON && MIX_LUT >= MIX_TINT + 2) };
    const { assert!(MIX_LUT + 3 * (MIX_N + 1) <= MIXER_LEN) };
}

#[test]
fn neutral_is_the_identity() {
    let n = neutral();
    let m = packed(&n);
    assert!(m.iter().all(|v| v.is_finite()));
    assert_eq!(m[MIX_ON], 0.0);
    assert_eq!((m[MIX_TINT], m[MIX_TINT + 1]), (0.0, 0.0));
    let mut next = rng(7);
    for _ in 0..2000 {
        let lab = [next() * 0.6 + 0.5, next() * 0.4, next() * 0.4];
        assert_eq!(apply(&m, lab), lab);
    }
    let cam = [[1.7, -0.5, -0.2], [-0.2, 1.4, -0.2], [0.0, -0.4, 1.4]];
    let mut c = cam;
    calibrate(&n, &mut c);
    assert_eq!(c, cam);
    // Calibration alone leaves the Color Mixer off.
    let calib = Params {
        red_primary_hue: 40.0,
        blue_primary_saturation: 30.0,
        ..n
    };
    assert_eq!(packed(&calib)[MIX_ON], 0.0);
}

#[test]
fn greys_stay_grey() {
    for seed in 0..50 {
        let p = random_mixer(seed, neutral(), false);
        let m = packed(&p);
        assert!(m.iter().all(|v| v.is_finite()));
        for l in [0.0, 0.05, 0.3, 0.6, 0.9, 1.0] {
            assert_eq!(apply(&m, [l, 0.0, 0.0]), [l, 0.0, 0.0]);
            // Near neutral (noise, a faint cast) barely moves and gains no
            // colour; cuts may take what little it has.
            for h in (0..360).step_by(15) {
                let lab = from_lch(l, 0.003, h as f32);
                let out = apply(&m, lab);
                let (_, c, _) = lch(out);
                assert!(
                    (out[0] - lab[0]).abs() < 0.001,
                    "seed {seed}: {lab:?} -> {out:?}"
                );
                assert!(c < 0.003 * 1.1, "seed {seed}: {lab:?} -> {out:?}");
                let turn =
                    (lab[1] * out[2] - lab[2] * out[1]).atan2(lab[1] * out[1] + lab[2] * out[2]);
                assert!(
                    c < 1e-4 || turn.abs().to_degrees() < 3.0,
                    "seed {seed}: {lab:?} -> {out:?}"
                );
            }
        }
    }
}

#[test]
fn every_band_moves_its_colours_the_right_way() {
    let names = [
        "red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta",
    ];
    for (i, &centre) in BAND_HUES.iter().enumerate() {
        let set = |which: usize, v: f32| {
            let mut a = neutral().to_array();
            let at = |name: &str| Params::NAMES.iter().position(|&n| n == name).unwrap();
            let field = ["hue", "saturation", "luminance"][which];
            a[at(&format!("{field}_{}", names[i]))] = v;
            packed(&Params::from_array(a))
        };
        let lab = from_lch(0.6, 0.12, centre);
        let (_, c0, h0) = lch(lab);
        // Hue: toward the next band (+), the previous one (-), never past it.
        let next_gap = gap(i);
        let prev_gap = gap((i + 7) % 8);
        let up = dhue(h0, lch(apply(&set(0, 100.0), lab)).2);
        let down = dhue(h0, lch(apply(&set(0, -100.0), lab)).2);
        assert!(
            up > 0.4 * next_gap.min(HUE_MAX / HUE_REACH) && up < 0.75 * next_gap,
            "{} hue +100 moves {up:.1} deg (gap {next_gap})",
            names[i]
        );
        assert!(
            -down > 0.4 * prev_gap.min(HUE_MAX / HUE_REACH) && -down < 0.75 * prev_gap,
            "{} hue -100 moves {down:.1} deg (gap {prev_gap})",
            names[i]
        );
        // Saturation: grey at -100, about double at +100.
        let c_lo = lch(apply(&set(1, -100.0), lab)).1;
        let c_hi = lch(apply(&set(1, 100.0), lab)).1;
        assert!(c_lo < 0.03 * c0, "{} sat -100 leaves {c_lo}", names[i]);
        assert!(
            (c_hi / c0 - 2.0).abs() < 0.05,
            "{} sat +100 x{}",
            names[i],
            c_hi / c0
        );
        // Luminance: darker at -100, lighter at +100, hue and chroma kept.
        let dark = apply(&set(2, -100.0), lab);
        let light = apply(&set(2, 100.0), lab);
        assert!(
            dark[0] < 0.48 && light[0] > 0.7,
            "{} {dark:?} {light:?}",
            names[i]
        );
        assert!((lch(dark).1 - c0).abs() < 1e-4 && dhue(h0, lch(dark).2).abs() < 0.01);
        // The opposite side of the circle does not notice.
        let far = from_lch(0.6, 0.12, centre + 180.0);
        for which in 0..3 {
            let out = apply(&set(which, 100.0), far);
            assert!(
                (0..3).all(|k| (out[k] - far[k]).abs() < 1e-5),
                "{}",
                names[i]
            );
        }
    }
}

#[test]
fn bands_blend_and_all_alike_act_everywhere() {
    let all = |v: f32| {
        let mut p = neutral();
        p.saturation_red = v;
        p.saturation_orange = v;
        p.saturation_yellow = v;
        p.saturation_green = v;
        p.saturation_aqua = v;
        p.saturation_blue = v;
        p.saturation_purple = v;
        p.saturation_magenta = v;
        p
    };
    for v in [-100.0, -40.0, 50.0] {
        let t = tables(&all(v));
        let want = 1.0 + v / 100.0;
        assert!(t.iter().all(|r| (r[1] - want).abs() < 1e-5 && r[0] == 0.0));
    }
    // Skin (Oklab hue ~65) answers mostly to orange, some to yellow; a blue
    // sky (~240) mostly to blue, some to aqua.
    let gain_at = |p: &Params, h: f32| {
        let lab = from_lch(0.7, 0.1, h);
        lch(apply(&packed(p), lab)).1 / 0.1
    };
    let only = |f: fn(&mut Params)| {
        let mut p = neutral();
        f(&mut p);
        p
    };
    let skin_o = 1.0 - gain_at(&only(|p| p.saturation_orange = -100.0), 65.0);
    let skin_y = 1.0 - gain_at(&only(|p| p.saturation_yellow = -100.0), 65.0);
    assert!(skin_o > 0.85 && skin_y < 0.15 && skin_o + skin_y > 0.99);
    let sky_b = 1.0 - gain_at(&only(|p| p.saturation_blue = -100.0), 240.0);
    let sky_a = 1.0 - gain_at(&only(|p| p.saturation_aqua = -100.0), 240.0);
    assert!(sky_b > 0.6 && sky_a > 0.1 && (sky_b + sky_a - 1.0).abs() < 0.01);
}

#[test]
fn hue_map_never_folds_over() {
    // Hue sliders only: a band at Saturation -100 has no hue left to read.
    let hue_only = |seed: u64| {
        let p = random_mixer(seed, neutral(), false);
        Params {
            hue_red: p.hue_red,
            hue_orange: p.hue_orange,
            hue_yellow: p.hue_yellow,
            hue_green: p.hue_green,
            hue_aqua: p.hue_aqua,
            hue_blue: p.hue_blue,
            hue_purple: p.hue_purple,
            hue_magenta: p.hue_magenta,
            ..neutral()
        }
    };
    let mut cases: Vec<Params> = (0..200).map(hue_only).collect();
    // Neighbours pushed hard into each other, every pair.
    for i in 0..8 {
        let mut a = neutral().to_array();
        let at = |k: usize| {
            let names = [
                "red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta",
            ];
            Params::NAMES
                .iter()
                .position(|&n| n == format!("hue_{}", names[k % 8]))
                .unwrap()
        };
        a[at(i)] = 100.0;
        a[at(i + 1)] = -100.0;
        cases.push(Params::from_array(a));
    }
    for p in &cases {
        let m = packed(p);
        let mut prev: Option<f32> = None;
        let mut total = 0.0;
        for k in 0..=3600 {
            let h = k as f32 * 0.1;
            let out = lch(apply(&m, from_lch(0.6, 0.15, h))).2;
            if let Some(prev) = prev {
                let step = dhue(prev, out);
                assert!(step > 0.0, "hue map folds at {h} deg: step {step}");
                assert!(step < 0.35, "hue map jumps at {h} deg: step {step}");
                total += step;
            }
            prev = Some(out);
        }
        assert!((total - 360.0).abs() < 0.2, "hue map wraps {total}");
    }
}

#[test]
fn luminance_and_saturation_are_smooth_in_hue_and_chroma() {
    for seed in 0..40 {
        let m = packed(&random_mixer(seed, neutral(), false));
        let mut prev: Option<[f32; 3]> = None;
        for k in 0..=3600 {
            let out = apply(&m, from_lch(0.6, 0.15, k as f32 * 0.1));
            if let Some(p) = prev {
                assert!(
                    (out[0] - p[0]).abs() < 0.004,
                    "L jumps at {}",
                    k as f32 * 0.1
                );
                let dc = (lch(out).1 - lch(p).1).abs();
                assert!(dc < 0.004, "C jumps at {}", k as f32 * 0.1);
            }
            prev = Some(out);
        }
        // Along chroma at a fixed hue: lightness and chroma move gradually.
        // Boosts keep the order of chroma; a cut fades in with chroma, so
        // near neutrals keep a little of theirs (at most 0.6 MIX_CN).
        let min_gain = tables(&random_mixer(seed, neutral(), false))
            .iter()
            .map(|r| r[1])
            .fold(f32::INFINITY, f32::min);
        for h in [30.0, 140.0, 250.0] {
            let mut prev: Option<[f32; 3]> = None;
            for k in 0..=400 {
                let c = k as f32 * 0.0005;
                let out = apply(&m, from_lch(0.6, c, h));
                if let Some(p) = prev {
                    assert!((out[0] - p[0]).abs() < 0.003, "L jumps at C {c}");
                    let dc = lch(out).1 - lch(p).1;
                    assert!(dc > -0.0003 && dc < 0.0015, "C jumps at C {c}: {dc}");
                    if min_gain >= 1.0 {
                        assert!(dc >= -1e-6, "chroma order reverses at C {c}");
                    }
                }
                assert!(lch(out).1 <= c.max(0.6 * MIX_CN) * 2.01);
                prev = Some(out);
            }
        }
    }
}

#[test]
fn lightness_stays_in_range_and_in_order() {
    for seed in 0..40 {
        let m = packed(&random_mixer(seed, neutral(), true));
        for h in (0..360).step_by(10) {
            let mut prev = -1.0f32;
            for k in 0..=100 {
                let l = k as f32 * 0.01;
                let out = apply(&m, from_lch(l, 0.1, h as f32))[0];
                assert!((0.0..=1.0).contains(&out), "L {l} -> {out}");
                assert!(out >= prev, "lightness reverses at {l}, hue {h}");
                prev = out;
            }
        }
    }
}

#[test]
fn shadows_tint_is_green_or_magenta_in_the_shadows() {
    let tint = |v: f32, l: f32| {
        let m = packed(&Params {
            shadow_tint: v,
            ..neutral()
        });
        apply(&m, [l, 0.0, 0.0])
    };
    let magenta = tint(100.0, 0.25);
    assert!(magenta[1] > 0.008 && magenta[2] < -0.004, "{magenta:?}");
    let green = tint(-100.0, 0.25);
    assert!(green[1] < -0.008 && green[2] > 0.004, "{green:?}");
    assert_eq!(tint(100.0, 0.0), [0.0, 0.0, 0.0]);
    assert_eq!(tint(100.0, 0.8), [0.8, 0.0, 0.0]);
    let mid = tint(100.0, 0.55)[1];
    assert!(mid > 0.0 && mid < 0.6 * magenta[1]);
}

/// Oklab of linear Rec.2020, in f64.
fn oklab_2020(c: [f64; 3]) -> [f64; 3] {
    let lms = mul_v(&LMS_FROM_2020, c).map(f64::cbrt);
    mul_v(&LAB_FROM_LMS, lms)
}

#[test]
fn calibration_turns_primaries_and_keeps_white() {
    // Adobe's ILCE-7M3 D65 matrix through the test colour model.
    let cam = testkit::TestColor::new([
        [0.7374, -0.2389, -0.0551],
        [-0.5435, 1.3162, 0.2519],
        [-0.1006, 0.1795, 0.6552],
    ]);
    let m0 = crate::render::CameraColor::rec2020_from_cam(&cam, 5500.0);
    let run = |p: &Params, rgb: [f64; 3]| {
        let mut m = m0;
        calibrate(p, &mut m);
        let m = m.map(|r| r.map(f64::from));
        oklab_2020(mul_v(&m, rgb))
    };
    let hue_of = |lab: [f64; 3]| lab[2].atan2(lab[1]).to_degrees().rem_euclid(360.0) as f32;
    let chroma = |lab: [f64; 3]| lab[1].hypot(lab[2]);
    let n = neutral();
    for seed in 0..30 {
        let p = random_mixer(seed, n, false);
        let mut m = m0;
        calibrate(&p, &mut m);
        for (row, old) in m.iter().zip(&m0) {
            let (s, s0) = (row.iter().sum::<f32>(), old.iter().sum::<f32>());
            assert!((s - s0).abs() < 1e-5, "row sum {s} vs {s0}");
        }
        let grey = run(&p, [0.4, 0.4, 0.4]);
        assert!(chroma(grey) < 1e-6, "grey picks up colour {grey:?}");
    }
    // A camera red, green and blue (white balanced) each turn and saturate
    // with their own primary.
    let colours = [
        ("red", [0.8, 0.25, 0.15]),
        ("green", [0.25, 0.6, 0.2]),
        ("blue", [0.15, 0.3, 0.8]),
    ];
    for (k, (name, rgb)) in colours.iter().enumerate() {
        let with = |hue: f32, sat: f32| {
            let mut p = n;
            match k {
                0 => (p.red_primary_hue, p.red_primary_saturation) = (hue, sat),
                1 => (p.green_primary_hue, p.green_primary_saturation) = (hue, sat),
                _ => (p.blue_primary_hue, p.blue_primary_saturation) = (hue, sat),
            }
            run(&p, *rgb)
        };
        let base = with(0.0, 0.0);
        let (h0, c0) = (hue_of(base), chroma(base));
        let up = dhue(h0, hue_of(with(100.0, 0.0)));
        let down = dhue(h0, hue_of(with(-100.0, 0.0)));
        let more = chroma(with(0.0, 100.0)) / c0;
        let less = chroma(with(0.0, -100.0)) / c0;
        println!(
            "{name}: hue {h0:.0} C {c0:.3}; primary hue +-100 -> {up:+.1} / {down:+.1} deg, saturation +-100 -> x{more:.2} / x{less:.2}"
        );
        assert!(
            up > 4.0 && up < 40.0,
            "{name} primary hue +100 turns {up:.1}"
        );
        assert!(
            down < -4.0 && down > -40.0,
            "{name} primary hue -100 turns {down:.1}"
        );
        assert!(
            more > 1.1 && less < 0.85,
            "{name} primary saturation x{more:.2} / x{less:.2}"
        );
    }
}

// ---------------------------------------------------------------- GPU

/// The renderer's source plus a kernel that runs `color_mixer_apply` alone.
static PROBE_SOURCE: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}\n{}",
        *SOURCE,
        r#"
extern "C" __global__ void mixer_probe_kernel(const float* __restrict__ M, const float* __restrict__ in,
                                              float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Lab o;
    o.L = in[3 * i];
    o.a = in[3 * i + 1];
    o.b = in[3 * i + 2];
    color_mixer_apply(M, o);
    out[3 * i] = o.L;
    out[3 * i + 1] = o.a;
    out[3 * i + 2] = o.b;
}
"#
    )
});

fn probe(gpu: &Gpu, m: &[f32], input: &[[f32; 3]]) -> Vec<[f32; 3]> {
    let module = gpu
        .module("color_mixer_probe", PROBE_SOURCE.as_str())
        .expect("compile probe");
    let f = module.load_function("mixer_probe_kernel").expect("probe");
    let s = &gpu.stream;
    let flat: Vec<f32> = input.iter().flatten().copied().collect();
    let dm = s.clone_htod(m).unwrap();
    let din = s.clone_htod(&flat).unwrap();
    let mut dout = s.alloc_zeros::<f32>(flat.len()).unwrap();
    let n = input.len() as i32;
    let mut b = s.launch_builder(&f);
    b.arg(&dm).arg(&din).arg(&mut dout).arg(&n);
    let cfg = LaunchConfig {
        grid_dim: (input.len().div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: arguments match `mixer_probe_kernel`; buffers hold n Lab triples
    // and M holds MIXER_LEN floats.
    unsafe { b.launch(cfg) }.cuda("probe").unwrap();
    let out = s.clone_dtoh(&dout).unwrap();
    out.as_chunks::<3>().0.to_vec()
}

/// The kernel matches [`apply`]: identity bit for bit at neutral, finite
/// everywhere, greys untouched, and within float rounding otherwise.
#[test]
#[ignore]
fn gpu_kernel_matches_cpu() {
    let gpu = Gpu::open().expect("gpu");
    let mut next = rng(11);
    let mut input: Vec<[f32; 3]> = (0..20000)
        .map(|_| {
            let c = 0.35 * (next() * 0.5 + 0.5).powi(2);
            from_lch(next() * 0.55 + 0.5, c, 180.0 * (next() + 1.0))
        })
        .collect();
    for l in [0.0, 0.1, 0.5, 0.9, 1.0] {
        input.push([l, 0.0, 0.0]);
        input.push([l, 1e-9, -1e-9]);
    }
    input.extend([[1.2, 0.3, -0.3], [-0.01, 0.05, 0.05], [0.5, 0.5, 0.5]]);
    let n = neutral();
    let out = probe(&gpu, &packed(&n), &input);
    assert_eq!(out, input, "neutral changed pixels");
    let mut worst = 0.0f32;
    for seed in 0..12 {
        let m = packed(&random_mixer(seed, n, seed % 2 == 0));
        let out = probe(&gpu, &m, &input);
        for (lab, (g, c)) in input
            .iter()
            .zip(out.iter().zip(input.iter().map(|&x| apply(&m, x))))
        {
            assert!(g.iter().all(|v| v.is_finite()), "{lab:?} -> {g:?}");
            let d = (0..3).map(|k| (g[k] - c[k]).abs()).fold(0.0, f32::max);
            worst = worst.max(d);
            assert!(d < 2e-4, "seed {seed}: {lab:?} gpu {g:?} cpu {c:?}");
            if seed % 2 == 1 && lab[1] == 0.0 && lab[2] == 0.0 {
                assert_eq!(g, lab, "grey changed");
            }
        }
    }
    println!("kernel vs CPU: worst difference {worst:.2e}");
}

const CHECKER: &str = "ILCE-7RM3__Sony_-_ILCE-7RM3_-_12bit_uncompressed__3_2_.ARW";
const SKY: &str = "ILCE-6700__DSC00001.ARW";
const FOLIAGE: &str = "ILCE-6000__DSC01542.ARW";

fn out_dir() -> PathBuf {
    let dir = std::env::var_os("SPL_OUT")
        .map_or_else(|| PathBuf::from("/out/render-color"), PathBuf::from);
    std::fs::create_dir_all(&dir).expect("create output dir");
    dir
}

fn write_jpeg(img: &Rgb8Image, name: &str) -> PathBuf {
    let path = out_dir().join(name);
    let mut enc = jpeg_encoder::Encoder::new_file(&path, 92).expect("create jpeg");
    enc.set_sampling_factor(jpeg_encoder::SamplingFactor::R_4_4_4);
    enc.encode(
        &img.data,
        img.width as u16,
        img.height as u16,
        jpeg_encoder::ColorType::Rgb,
    )
    .expect("encode jpeg");
    path
}

fn engine(gpu: &Gpu, s: &testkit::Sample) -> RenderEngine {
    RenderEngine::with_color(gpu, &s.scene, s.color.clone()).expect("engine")
}

fn neutral_of(s: &testkit::Sample) -> Params {
    Params::neutral(s.scene.info.as_shot_temperature, s.scene.info.as_shot_tint)
}

/// Oklab of an 8-bit sRGB pixel.
fn oklab8(p: &[u8]) -> [f64; 3] {
    let lin = |v: u8| {
        let e = v as f64 / 255.0;
        if e <= 0.04045 {
            e / 12.92
        } else {
            ((e + 0.055) / 1.055).powf(2.4)
        }
    };
    let (r, g, b) = (lin(p[0]), lin(p[1]), lin(p[2]));
    let lms = [
        0.412_221_470_8 * r + 0.536_332_536_3 * g + 0.051_445_992_9 * b,
        0.211_903_498_2 * r + 0.680_699_545_1 * g + 0.107_396_956_6 * b,
        0.088_302_461_9 * r + 0.281_718_837_6 * g + 0.629_978_700_5 * b,
    ]
    .map(f64::cbrt);
    mul_v(&LAB_FROM_LMS, lms)
}

/// Mean RGB of a box centred at fractions (fx, fy) of the image.
fn patch(img: &Rgb8Image, fx: f64, fy: f64, half: usize) -> [f64; 3] {
    let (cx, cy) = (
        (fx * img.width as f64) as usize,
        (fy * img.height as f64) as usize,
    );
    let mut s = [0f64; 3];
    let mut n = 0.0;
    for y in cy - half..=cy + half {
        for x in cx - half..=cx + half {
            let p = &img.data[3 * (y * img.width + x)..];
            for c in 0..3 {
                s[c] += p[c] as f64;
            }
            n += 1.0;
        }
    }
    s.map(|v| v / n)
}

/// The classic chart's neutral column in the A7R III ColorChecker shot
/// (display image, unrotated): white to black.
const GREY_PATCHES: [(f64, f64); 6] = [
    (0.732, 0.74),
    (0.732, 0.65),
    (0.732, 0.56),
    (0.732, 0.47),
    (0.732, 0.38),
    (0.732, 0.29),
];

/// On the ColorChecker, strong Color Mixer edits leave the grey patches,
/// the backdrop and the chart's black frame alone; Calibration keeps the
/// grey patches; no setting makes NaN (black) pixels.
#[test]
#[ignore]
fn gpu_checker_greys_hold() {
    let gpu = Gpu::open().expect("gpu");
    let s = testkit::load(CHECKER);
    let mut e = engine(&gpu, &s);
    let n = neutral_of(&s);
    let base = e.display(&n).unwrap();
    let blacks = |img: &Rgb8Image| {
        img.data
            .as_chunks::<3>()
            .0
            .iter()
            .filter(|p| p == &&[0, 0, 0])
            .count()
    };
    let near_neutral: Vec<usize> = base
        .data
        .as_chunks::<3>()
        .0
        .iter()
        .enumerate()
        .filter(|(_, p)| {
            let lab = oklab8(p.as_slice());
            lab[1].hypot(lab[2]) < 0.006 && lab[0] > 0.1 && lab[0] < 0.97
        })
        .map(|(i, _)| i)
        .collect();
    println!(
        "{}: {:.0}% of pixels near neutral; grey patches (Oklab C): {}",
        s.name,
        100.0 * near_neutral.len() as f64 / (base.width * base.height) as f64,
        GREY_PATCHES
            .iter()
            .map(|&(x, y)| {
                let p = patch(&base, x, y, 12).map(|v| v.round() as u8);
                let lab = oklab8(&p);
                format!("{:.4}", lab[1].hypot(lab[2]))
            })
            .collect::<Vec<_>>()
            .join(" ")
    );
    let check = |label: &str, img: &Rgb8Image, patch_tol: f64, px_tol: f64| {
        let worst_patch = GREY_PATCHES
            .iter()
            .map(|&(x, y)| {
                let (a, b) = (patch(&base, x, y, 12), patch(img, x, y, 12));
                (0..3).map(|c| (a[c] - b[c]).abs()).fold(0.0, f64::max)
            })
            .fold(0.0, f64::max);
        let mean_px = near_neutral
            .iter()
            .map(|&i| {
                (0..3)
                    .map(|c| (img.data[3 * i + c] as f64 - base.data[3 * i + c] as f64).abs())
                    .sum::<f64>()
                    / 3.0
            })
            .sum::<f64>()
            / near_neutral.len() as f64;
        println!(
            "  {label:>22}: grey patches move <= {worst_patch:.2} levels, near-neutral pixels {mean_px:.3} on average"
        );
        assert!(
            worst_patch < patch_tol,
            "{label}: grey patch moved {worst_patch:.2}"
        );
        assert!(
            mean_px < px_tol,
            "{label}: near-neutral pixels moved {mean_px:.3}"
        );
        assert!(
            blacks(img) <= blacks(&base) + 10,
            "{label}: black (NaN) pixels"
        );
    };
    for seed in 0..6 {
        let mut p = random_mixer(seed, n, false);
        // Color Mixer only.
        p.red_primary_hue = 0.0;
        p.red_primary_saturation = 0.0;
        p.green_primary_hue = 0.0;
        p.green_primary_saturation = 0.0;
        p.blue_primary_hue = 0.0;
        p.blue_primary_saturation = 0.0;
        check(
            &format!("color mixer #{seed}"),
            &e.display(&p).unwrap(),
            1.0,
            0.6,
        );
    }
    // Calibration keeps exact neutrals; these patches carry a faint cast
    // (C up to 0.015), which the primaries move a little: under a level at
    // everyday strengths (+-30), a few at every slider pinned at random.
    for (seed, scale, tol) in [(0, 0.3, 1.0), (1, 0.3, 1.0), (2, 1.0, 5.0), (3, 1.0, 5.0)] {
        let p = random_mixer(100 + seed, n, false);
        let q = Params {
            red_primary_hue: scale * p.red_primary_hue,
            red_primary_saturation: scale * p.red_primary_saturation,
            green_primary_hue: scale * p.green_primary_hue,
            green_primary_saturation: scale * p.green_primary_saturation,
            blue_primary_hue: scale * p.blue_primary_hue,
            blue_primary_saturation: scale * p.blue_primary_saturation,
            ..n
        };
        check(
            &format!("calibration x{scale} #{seed}"),
            &e.display(&q).unwrap(),
            tol,
            tol,
        );
    }
    // Everything at once, with Basic edits too: finite statistics.
    let all: Vec<Params> = (0..64)
        .map(|seed| {
            let mut p = random_mixer(200 + seed, n, true);
            p.saturation = 60.0;
            p.vibrance = 80.0;
            p.exposure = 0.5;
            p
        })
        .collect();
    for st in e.stats(&all).unwrap() {
        assert!(
            [
                st.luma_mean,
                st.chroma_mean,
                st.chroma_p95,
                st.colorfulness,
                st.cast_a,
                st.cast_b,
                st.structure,
                st.entropy
            ]
            .iter()
            .all(|v| v.is_finite())
        );
    }
}

type Setter = fn(&mut Params, f32);

const ROWS: [(&str, Setter); 16] = [
    ("blue_hue", |p, v| p.hue_blue = v),
    ("blue_sat", |p, v| p.saturation_blue = v),
    ("blue_lum", |p, v| p.luminance_blue = v),
    ("aqua_sat", |p, v| p.saturation_aqua = v),
    ("green_hue", |p, v| p.hue_green = v),
    ("green_sat", |p, v| p.saturation_green = v),
    ("green_lum", |p, v| p.luminance_green = v),
    ("orange_hue", |p, v| p.hue_orange = v),
    ("orange_sat", |p, v| p.saturation_orange = v),
    ("orange_lum", |p, v| p.luminance_orange = v),
    ("red_primary_hue", |p, v| p.red_primary_hue = v),
    ("red_primary_sat", |p, v| p.red_primary_saturation = v),
    ("green_primary_hue", |p, v| p.green_primary_hue = v),
    ("green_primary_sat", |p, v| p.green_primary_saturation = v),
    ("blue_primary_hue", |p, v| p.blue_primary_hue = v),
    ("blue_primary_sat", |p, v| p.blue_primary_saturation = v),
];

/// Strips (-100, 0, +100) and a sheet (-100..100) per slider for a sky, a
/// foliage shot and the ColorChecker, plus a few Lightroom-style looks at
/// display size, for eyeballing.
#[test]
#[ignore]
fn gpu_strips() {
    let gpu = Gpu::open().expect("gpu");
    for file in [SKY, FOLIAGE, CHECKER] {
        let s = testkit::load(file);
        let mut e = engine(&gpu, &s);
        let n = neutral_of(&s);
        let mut rows = Vec::new();
        let tint: (&str, Setter) = ("shadow_tint", |p, v| p.shadow_tint = v);
        for (name, set) in ROWS.iter().chain([&tint]) {
            let imgs: Vec<Rgb8Image> = [-100.0, -50.0, 0.0, 50.0, 100.0]
                .iter()
                .map(|&v| {
                    let mut p = n;
                    set(&mut p, v);
                    e.search(&p).unwrap()
                })
                .collect();
            write_jpeg(
                &testkit::sheet(
                    &[vec![imgs[0].clone(), imgs[2].clone(), imgs[4].clone()]],
                    4,
                ),
                &format!("strip_{}_{name}.jpg", s.name),
            );
            rows.push(
                imgs.iter()
                    .map(|i| testkit::shrink(i, 3))
                    .collect::<Vec<_>>(),
            );
        }
        let path = write_jpeg(&testkit::sheet(&rows, 4), &format!("sheet_{}.jpg", s.name));
        println!("{file} -> {}", path.display());
        let looks = [
            ("neutral", n),
            (
                "deep_sky",
                Params {
                    saturation_blue: 30.0,
                    luminance_blue: -40.0,
                    hue_blue: -10.0,
                    saturation_aqua: 20.0,
                    luminance_aqua: -20.0,
                    ..n
                },
            ),
            (
                "muted_greens",
                Params {
                    hue_green: 30.0,
                    saturation_green: -30.0,
                    saturation_yellow: -20.0,
                    hue_yellow: -20.0,
                    luminance_orange: 20.0,
                    ..n
                },
            ),
            (
                "calibration_pop",
                Params {
                    blue_primary_saturation: 50.0,
                    red_primary_hue: 15.0,
                    shadow_tint: -10.0,
                    ..n
                },
            ),
        ];
        for (label, p) in looks {
            write_jpeg(
                &e.display(&p).unwrap(),
                &format!("look_{}_{label}.jpg", s.name),
            );
        }
    }
}

/// Render cost: stats throughput and packing with Basic edits only and
/// with every Color Mixer and Calibration slider set too.
#[test]
#[ignore]
fn gpu_mixer_cost() {
    let gpu = Gpu::open().expect("gpu");
    let s = testkit::load("ILCE-7M3___DSC0009.ARW");
    let mut e = engine(&gpu, &s);
    let n = neutral_of(&s);
    let mut next = rng(3);
    let basic: Vec<Params> = (0..1536)
        .map(|_| Params {
            exposure: next(),
            contrast: 100.0 * next(),
            highlights: 100.0 * next(),
            shadows: 100.0 * next(),
            vibrance: 100.0 * next(),
            ..n
        })
        .collect();
    let mixed: Vec<Params> = basic
        .iter()
        .enumerate()
        .map(|(i, p)| random_mixer(i as u64, *p, true))
        .collect();
    e.stats(&basic[..64]).unwrap();
    for (label, set) in [("basic only", &basic), ("+ mixer/calibration", &mixed)] {
        let mut buf = vec![0f32; CAND_STRIDE];
        let t = Instant::now();
        for p in set.iter() {
            curves::pack(p, &*e.color, &e.reference, &e.anchors, &mut buf);
        }
        let pack_us = t.elapsed().as_secs_f64() * 1e6 / set.len() as f64;
        let rate = |e: &mut RenderEngine, batch: usize, rounds: usize| {
            let t = Instant::now();
            for r in 0..rounds {
                e.stats(&set[(r * batch) % set.len()..][..batch]).unwrap();
            }
            (batch * rounds) as f64 / t.elapsed().as_secs_f64()
        };
        let big = rate(&mut e, 1536, 4);
        let small = rate(&mut e, 24, 64);
        let t = Instant::now();
        for p in &set[..10] {
            e.display(p).unwrap();
        }
        let display_ms = t.elapsed().as_secs_f64() * 1e2;
        println!(
            "{label:>20}: {big:.0} candidates/s (batch 1536), {small:.0} (batch 24), packing {pack_us:.1} us, display() {display_ms:.1} ms"
        );
    }
}
