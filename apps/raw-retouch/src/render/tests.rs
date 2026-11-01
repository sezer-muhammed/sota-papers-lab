//! Renderer tests. The GPU and sample-file ones are `#[ignore]`; run them on
//! the GPU host with `gpu-test <checkout> render:: --ignored`. They write
//! contact sheets and detail renders under /out/render/.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use super::testkit::{self, Sample, TestColor};
use super::*;
use crate::types::{DISPLAY_EDGE, RawInfo, Rgb16Image};

/// Three test shots: A7 III 14-bit, A7R III 12-bit (like the user's
/// camera), and an APS-C body.
const SAMPLES: [&str; 3] = [
    "ILCE-7M3___DSC0009.ARW",
    "ILCE-7RM3__Sony_-_ILCE-7RM3_-_12bit_uncompressed__3_2_.ARW",
    "ILCE-6400__DSC00087.ARW",
];
const SAMPLE_61MP: &str = "ILCE-7RM5__7RM5-LosslessCompressedLarge.ARW";

/// Adobe's D65 ColorMatrix for the ILCE-7M3 (XYZ -> camera).
const A7M3_XYZ_TO_CAM: [[f32; 3]; 3] = [
    [0.7374, -0.2389, -0.0551],
    [-0.5435, 1.3162, 0.2519],
    [-0.1006, 0.1795, 0.6552],
];

#[test]
fn engine_is_send() {
    fn send<T: Send>() {}
    send::<RenderEngine>();
}

#[test]
fn test_color_round_trips_and_warms_with_temperature() {
    let c = TestColor::new(A7M3_XYZ_TO_CAM);
    for (t, tint) in [(5500.0, 0.0), (3200.0, 10.0), (7500.0, -12.0)] {
        let wb = c.wb_for(t, tint);
        let (t2, tint2) = c.temp_tint_for(wb);
        assert!((t2 - t).abs() / t < 0.01, "{t} -> {t2}");
        assert!((tint2 - tint).abs() < 1.0, "{tint} -> {tint2}");
    }
    let (cool, warm) = (c.wb_for(4000.0, 0.0), c.wb_for(7000.0, 0.0));
    assert!(
        warm[0] > cool[0] && warm[2] < cool[2],
        "higher kelvin must warm the image"
    );
    let (green, magenta) = (c.wb_for(5500.0, -30.0), c.wb_for(5500.0, 30.0));
    assert!(
        magenta[0] > green[0] && magenta[2] > green[2],
        "positive tint must add magenta"
    );
    let m = c.rec2020_from_cam(5500.0);
    for row in m {
        assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-4);
    }
}

fn neutral(s: &Scene) -> Params {
    Params::neutral(s.info.as_shot_temperature, s.info.as_shot_tint)
}

fn engine(gpu: &Gpu, s: &Sample) -> RenderEngine {
    RenderEngine::with_color(gpu, &s.scene, s.color.clone()).expect("engine")
}

fn luma8(img: &Rgb8Image) -> f64 {
    let sum: f64 = img
        .data
        .as_chunks::<3>()
        .0
        .iter()
        .map(|p| 0.2126 * p[0] as f64 + 0.7152 * p[1] as f64 + 0.0722 * p[2] as f64)
        .sum();
    sum / (255.0 * (img.width * img.height) as f64)
}

/// A grey ramp from 2^-12 to past clip, left to right, on an identity camera.
struct Identity;

impl CameraColor for Identity {
    fn rec2020_from_cam(&self, _temperature: f32) -> [[f32; 3]; 3] {
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    }
    fn wb_for(&self, temperature: f32, tint: f32) -> [f32; 3] {
        [
            temperature / 5000.0,
            1.0 - tint / 1000.0,
            5000.0 / temperature,
        ]
    }
}

fn ramp(w: usize, h: usize) -> LinearImage {
    let mut data = Vec::with_capacity(3 * w * h);
    for _ in 0..h {
        for x in 0..w {
            let v = (-12.0 + 12.5 * x as f32 / (w - 1) as f32).exp2().min(1.0);
            data.extend([v, v, v]);
        }
    }
    LinearImage {
        width: w,
        height: h,
        data,
    }
}

fn ramp_scene() -> Scene {
    Scene {
        info: RawInfo {
            make: "test".into(),
            model: "ramp".into(),
            lens: None,
            width: 1536,
            height: 256,
            orientation: 1,
            iso: None,
            exposure_time_s: None,
            f_number: None,
            focal_length_mm: None,
            captured_at: None,
            as_shot_temperature: 5000.0,
            as_shot_tint: 0.0,
            raw_clipped_fraction: 0.0,
        },
        color: crate::color::ColorModel::new(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            None,
        ),
        wb_as_shot: [1.0; 3],
        search: ramp(768, 128),
        display: ramp(1536, 256),
    }
}

fn column_luma(img: &Rgb8Image) -> Vec<f64> {
    (0..img.width)
        .map(|x| {
            let s: f64 = (0..img.height)
                .map(|y| {
                    let p = &img.data[3 * (y * img.width + x)..];
                    0.2126 * p[0] as f64 + 0.7152 * p[1] as f64 + 0.0722 * p[2] as f64
                })
                .sum();
            s / img.height as f64
        })
        .collect()
}

/// Every slider, at every setting, keeps a grey ramp monotonic (no tone
/// reversals), every slider at 0 is byte-identical to neutral, and middle
/// grey renders at about 0.46 sRGB.
#[test]
#[ignore]
fn gpu_ramp_monotonic_and_neutral() {
    let gpu = Gpu::open().expect("gpu");
    let scene = ramp_scene();
    let mut e = RenderEngine::with_color(&gpu, &scene, Arc::new(Identity)).expect("engine");
    let n = neutral(&scene);
    let base = e.display(&n).unwrap();
    let cols = column_luma(&base);
    // Grey: scene 0.18 at exposure 0 is raw 0.18 / 2^BASELINE_EV.
    let grey_x = ((0.18f32 / BASELINE_EV.exp2()).log2() + 12.0) / 12.5 * 1535.0;
    let grey = cols[grey_x.round() as usize];
    println!("middle grey renders at {grey:.1}/255 (target ~117)");
    assert!((grey - 117.3).abs() < 4.0);
    let set = |f: &dyn Fn(&mut Params, f32), v: f32| {
        let mut p = n;
        f(&mut p, v);
        p
    };
    type Setter<'a> = &'a dyn Fn(&mut Params, f32);
    let sliders: [(&str, Setter); 10] = [
        ("exposure", &|p, v| p.exposure = v / 50.0),
        ("contrast", &|p, v| p.contrast = v),
        ("highlights", &|p, v| p.highlights = v),
        ("shadows", &|p, v| p.shadows = v),
        ("whites", &|p, v| p.whites = v),
        ("blacks", &|p, v| p.blacks = v),
        ("vibrance", &|p, v| p.vibrance = v),
        ("saturation", &|p, v| p.saturation = v),
        ("temperature", &|p, v| p.temperature = 5000.0 + 15.0 * v),
        ("tint", &|p, v| p.tint = v / 3.0),
    ];
    for (name, f) in sliders {
        let zero = e.display(&set(f, 0.0)).unwrap();
        assert_eq!(zero.data, base.data, "{name} at 0 changed the render");
        let mut prev_mean = -1.0;
        for v in [-100.0, -50.0, 0.0, 50.0, 100.0] {
            let img = e.display(&set(f, v)).unwrap();
            let c = column_luma(&img);
            // Dithering adds +-0.5 level noise per pixel; columns average 256 rows.
            let worst = c
                .windows(2)
                .map(|w| w[1] - w[0])
                .fold(f64::INFINITY, f64::min);
            assert!(
                worst > -0.25,
                "{name} {v}: ramp reverses by {worst:.2} levels"
            );
            let mean = c.iter().sum::<f64>() / c.len() as f64;
            if ["exposure", "highlights", "shadows", "whites", "blacks"].contains(&name) {
                assert!(
                    mean > prev_mean,
                    "{name} {v}: mean {mean:.2} not above {prev_mean:.2}"
                );
            }
            prev_mean = mean;
        }
    }
}

/// (a) Contact sheets of slider sweeps for three samples, plus display-size
/// detail renders of the extreme settings, for eyeballing.
#[test]
#[ignore]
fn gpu_contact_sheets() {
    let gpu = Gpu::open().expect("gpu");
    for file in SAMPLES {
        let t = Instant::now();
        let s = testkit::load(file);
        let load_ms = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let mut e = engine(&gpu, &s);
        let engine_ms = t.elapsed().as_secs_f64() * 1e3;
        let n = neutral(&s.scene);
        let sweep = [-100.0f32, -50.0, 0.0, 50.0, 100.0];
        let rows: Vec<(&str, Vec<Params>)> = vec![
            (
                "exposure -1..+1",
                [-1.0, -0.5, 0.0, 0.5, 1.0]
                    .map(|v| Params { exposure: v, ..n })
                    .to_vec(),
            ),
            (
                "contrast",
                sweep.map(|v| Params { contrast: v, ..n }).to_vec(),
            ),
            (
                "highlights",
                sweep.map(|v| Params { highlights: v, ..n }).to_vec(),
            ),
            (
                "shadows",
                sweep.map(|v| Params { shadows: v, ..n }).to_vec(),
            ),
            ("whites", sweep.map(|v| Params { whites: v, ..n }).to_vec()),
            ("blacks", sweep.map(|v| Params { blacks: v, ..n }).to_vec()),
            (
                "vibrance",
                sweep.map(|v| Params { vibrance: v, ..n }).to_vec(),
            ),
            (
                "saturation",
                sweep.map(|v| Params { saturation: v, ..n }).to_vec(),
            ),
            (
                "temperature -1500..+1500 K",
                [-1500.0, -750.0, 0.0, 750.0, 1500.0]
                    .map(|d| Params {
                        temperature: n.temperature + d,
                        ..n
                    })
                    .to_vec(),
            ),
            (
                "tint -30..+30",
                [-30.0, -15.0, 0.0, 15.0, 30.0]
                    .map(|d| Params {
                        tint: n.tint + d,
                        ..n
                    })
                    .to_vec(),
            ),
        ];
        let mut sheet_rows = Vec::new();
        for (name, params) in &rows {
            let full: Vec<Rgb8Image> = params.iter().map(|p| e.search(p).unwrap()).collect();
            // The extremes and neutral side by side at search size.
            let strip = [full[0].clone(), full[2].clone(), full[4].clone()];
            let slider = name.split(' ').next().unwrap_or(name);
            testkit::write_jpeg(
                &testkit::sheet(&[strip.to_vec()], 4),
                &format!("strip_{}_{slider}.jpg", s.name),
            );
            sheet_rows.push(
                full.iter()
                    .map(|i| testkit::shrink(i, 2))
                    .collect::<Vec<_>>(),
            );
        }
        let path = testkit::write_jpeg(
            &testkit::sheet(&sheet_rows, 6),
            &format!("sheet_{}.jpg", s.name),
        );
        println!(
            "{file}: as shot {:.0} K tint {:+.1}, load {load_ms:.0} ms, engine {engine_ms:.0} ms -> {}",
            n.temperature,
            n.tint,
            path.display()
        );
        println!(
            "  rows: {}",
            rows.iter().map(|r| r.0).collect::<Vec<_>>().join(" | ")
        );
        let details = [
            ("neutral", n),
            (
                "recover",
                Params {
                    highlights: -100.0,
                    shadows: 100.0,
                    ..n
                },
            ),
            (
                "punchy",
                Params {
                    contrast: 60.0,
                    whites: 30.0,
                    blacks: -30.0,
                    vibrance: 40.0,
                    ..n
                },
            ),
            (
                "bright",
                Params {
                    exposure: 1.0,
                    highlights: -80.0,
                    ..n
                },
            ),
            (
                "flat",
                Params {
                    contrast: -100.0,
                    highlights: -50.0,
                    shadows: 50.0,
                    ..n
                },
            ),
        ];
        for (label, p) in details {
            testkit::write_jpeg(
                &e.display(&p).unwrap(),
                &format!("detail_{}_{label}.jpg", s.name),
            );
        }
    }
}

fn check_stats(st: &Stats) {
    let hist: f32 = st.luma_hist.iter().sum();
    assert!((hist - 1.0).abs() < 1e-4, "hist sums to {hist}");
    assert!(st.luma_p01 <= st.luma_p05 && st.luma_p05 <= st.luma_p50);
    assert!(st.luma_p50 <= st.luma_p95 && st.luma_p95 <= st.luma_p99);
    assert!((0.0..=1.0).contains(&st.structure));
    assert!(st.entropy > 0.0 && st.entropy <= 6.0);
    assert!(st.clip_hi_new <= st.clip_hi);
    assert!(st.chroma_p95 >= 0.0 && st.chroma_mean >= 0.0);
}

fn print_stats(label: &str, st: &Stats) {
    println!(
        "{label:>16}: mean {:.3} p01 {:.3} p50 {:.3} p99 {:.3} hi {:.4} new {:.4} lo {:.4} col {:5.1} C {:.3}/{:.3} oog {:.4} cast {:+.4} {:+.4} struct {:.3} lc {:.4} H {:.2}",
        st.luma_mean,
        st.luma_p01,
        st.luma_p50,
        st.luma_p99,
        st.clip_hi,
        st.clip_hi_new,
        st.clip_lo,
        st.colorfulness,
        st.chroma_mean,
        st.chroma_p95,
        st.out_of_gamut,
        st.cast_a,
        st.cast_b,
        st.structure,
        st.local_contrast,
        st.entropy
    );
}

/// (b) Statistics move the way their names say.
#[test]
#[ignore]
fn gpu_stats_sanity() {
    let gpu = Gpu::open().expect("gpu");
    for file in SAMPLES {
        let s = testkit::load(file);
        let mut e = engine(&gpu, &s);
        let n = neutral(&s.scene);
        println!("{file}");
        let exposures: Vec<Params> = (-8..=8)
            .map(|i| Params {
                exposure: i as f32 * 0.25,
                ..n
            })
            .collect();
        let st = e.stats(&exposures).unwrap();
        for (p, x) in exposures.iter().zip(&st) {
            check_stats(x);
            if p.exposure.fract() == 0.0 {
                print_stats(&format!("exposure {:+.0}", p.exposure), x);
            }
        }
        for w in st.windows(2) {
            assert!(w[1].luma_mean > w[0].luma_mean);
            assert!(w[1].luma_p50 >= w[0].luma_p50);
            assert!(w[1].clip_hi >= w[0].clip_hi);
            assert!(w[1].clip_hi_new >= w[0].clip_hi_new);
            assert!(w[1].clip_lo <= w[0].clip_lo);
        }
        assert!(st[16].clip_hi > st[8].clip_hi, "+2 EV must clip more");

        // The stats kernel and the 8-bit renderer agree.
        let img = e.search(&n).unwrap();
        let cpu = luma8(&img);
        assert!(
            (cpu - st[8].luma_mean as f64).abs() < 0.003,
            "{cpu} vs {}",
            st[8].luma_mean
        );

        let probe = |e: &mut RenderEngine, label: &str, p: Params| {
            let x = e.stats(&[p]).unwrap().remove(0);
            check_stats(&x);
            print_stats(label, &x);
            x
        };
        let base = probe(&mut e, "neutral", n);
        let sat_lo = probe(
            &mut e,
            "saturation -100",
            Params {
                saturation: -100.0,
                ..n
            },
        );
        let sat_hi = probe(
            &mut e,
            "saturation +100",
            Params {
                saturation: 100.0,
                ..n
            },
        );
        assert!(sat_lo.chroma_mean < 0.002 && sat_lo.colorfulness < 1.0);
        assert!(sat_hi.chroma_mean > base.chroma_mean && sat_hi.colorfulness > base.colorfulness);
        assert!(sat_hi.out_of_gamut >= base.out_of_gamut);
        let vib = probe(
            &mut e,
            "vibrance +100",
            Params {
                vibrance: 100.0,
                ..n
            },
        );
        assert!(vib.chroma_mean > base.chroma_mean);
        let con = probe(
            &mut e,
            "contrast +100",
            Params {
                contrast: 100.0,
                ..n
            },
        );
        assert!(con.local_contrast > base.local_contrast);
        assert!(con.luma_p99 - con.luma_p01 > base.luma_p99 - base.luma_p01);
        let hl = probe(
            &mut e,
            "highlights -100",
            Params {
                highlights: -100.0,
                ..n
            },
        );
        assert!(hl.luma_p99 < base.luma_p99 && hl.luma_p95 < base.luma_p95);
        let sh = probe(
            &mut e,
            "shadows +100",
            Params {
                shadows: 100.0,
                ..n
            },
        );
        assert!(sh.luma_p05 > base.luma_p05);
        let wh = probe(&mut e, "whites +100", Params { whites: 100.0, ..n });
        assert!(wh.luma_p99 >= base.luma_p99 && wh.clip_hi >= base.clip_hi);
        let bl = probe(
            &mut e,
            "blacks -100",
            Params {
                blacks: -100.0,
                ..n
            },
        );
        assert!(bl.luma_p01 <= base.luma_p01 && bl.clip_lo >= base.clip_lo);
        let warm = probe(
            &mut e,
            "+1500 K",
            Params {
                temperature: n.temperature + 1500.0,
                ..n
            },
        );
        let cool = probe(
            &mut e,
            "-1500 K",
            Params {
                temperature: n.temperature - 1500.0,
                ..n
            },
        );
        assert!(
            warm.cast_b > base.cast_b && cool.cast_b < base.cast_b,
            "temperature moves Oklab b"
        );
        let mag = probe(
            &mut e,
            "tint +30",
            Params {
                tint: n.tint + 30.0,
                ..n
            },
        );
        assert!(
            mag.cast_a > base.cast_a,
            "positive tint moves Oklab a toward magenta"
        );

        // Deterministic, and the same alone or inside a batch.
        let again = e.stats(&[n]).unwrap().remove(0);
        assert_eq!(again.luma_hist, base.luma_hist);
        assert_eq!(again.structure, base.structure);
        let batch = e.stats(&vec![n; 300]).unwrap();
        assert_eq!(batch[123].luma_hist, base.luma_hist);
        assert!((batch[123].structure - base.structure).abs() < 1e-4);
        assert!((batch[123].colorfulness - base.colorfulness).abs() < 1e-2);
    }
}

/// A small deterministic generator for parameter sets inside Lightroom's ranges.
fn random_params(n: usize, around: Params, seed: u64) -> Vec<Params> {
    let mut state = seed;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f64 / (1u64 << 31) as f64) as f32 * 2.0 - 1.0
    };
    (0..n)
        .map(|_| Params {
            temperature: around.temperature + 1500.0 * next(),
            tint: around.tint + 20.0 * next(),
            exposure: 1.5 * next(),
            contrast: 100.0 * next(),
            highlights: 100.0 * next(),
            shadows: 100.0 * next(),
            whites: 100.0 * next(),
            blacks: 100.0 * next(),
            vibrance: 100.0 * next(),
            saturation: 100.0 * next(),
            ..around
        })
        .collect()
}

/// (c) stats() throughput on the search image, large batches and CMA-ES-sized ones.
#[test]
#[ignore]
fn gpu_stats_throughput() {
    let gpu = Gpu::open().expect("gpu");
    let s = testkit::load(SAMPLES[0]);
    let mut e = engine(&gpu, &s);
    let n = neutral(&s.scene);
    let big = random_params(1536, n, 1);
    e.stats(&big[..64]).unwrap();
    let t = Instant::now();
    let packs = 20;
    let mut buf = vec![0f32; layout::CAND_STRIDE];
    for _ in 0..packs {
        for p in &big {
            curves::pack(p, &*e.color, &e.reference, &e.anchors, &mut buf);
        }
    }
    let pack_us = t.elapsed().as_secs_f64() * 1e6 / (packs * big.len()) as f64;
    let mut rate = |batch: usize, rounds: usize| {
        let t = Instant::now();
        let mut done = 0;
        for r in 0..rounds {
            let ps = &big[(r * batch) % big.len()..][..batch];
            let st = e.stats(ps).unwrap();
            assert_eq!(st.len(), batch);
            done += batch;
        }
        done as f64 / t.elapsed().as_secs_f64()
    };
    let r1536 = rate(1536, 4);
    let r256 = rate(256, 12);
    let r24 = rate(24, 64);
    let r1 = rate(1, 64);
    println!(
        "search image {}x{}: {r1536:.0} candidates/s (batch 1536), {r256:.0} (batch 256), {r24:.0} (batch 24), {r1:.0} (batch 1); packing {pack_us:.1} us per candidate",
        e.search.width, e.search.height
    );
    assert!(r1536 > 1500.0 && r24 > 1500.0, "below 1500 candidates/s");
    let t = Instant::now();
    for _ in 0..10 {
        e.display(&n).unwrap();
    }
    println!(
        "display() {:.1} ms per render",
        t.elapsed().as_secs_f64() * 1e2
    );
}

/// Area-downscales a 16-bit render to `w`x`h`, averaging in linear light or
/// in sRGB-encoded values; returns sRGB-encoded floats (0-255).
fn shrink16(img: &Rgb16Image, edge: usize, linear: bool) -> (usize, usize, Vec<f32>) {
    let data = img
        .data
        .iter()
        .map(|&v| {
            let e = v as f64 / 65535.0;
            if linear {
                (if e <= 0.04045 {
                    e / 12.92
                } else {
                    ((e + 0.055) / 1.055).powf(2.4)
                }) as f32
            } else {
                e as f32
            }
        })
        .collect();
    let small = testkit::downscale(
        &LinearImage {
            width: img.width,
            height: img.height,
            data,
        },
        edge,
    );
    let out = small
        .data
        .iter()
        .map(|&v| {
            255.0
                * if linear {
                    curves::oetf(v as f64) as f32
                } else {
                    v
                }
        })
        .collect();
    (small.width, small.height, out)
}

/// 3x3 box blur of interleaved RGB floats.
fn blur3(w: usize, h: usize, src: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; src.len()];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let (mut s, mut n) = (0f32, 0f32);
                for yy in y.saturating_sub(1)..(y + 2).min(h) {
                    for xx in x.saturating_sub(1)..(x + 2).min(w) {
                        s += src[3 * (yy * w + xx) + c];
                        n += 1.0;
                    }
                }
                out[3 * (y * w + x) + c] = s / n;
            }
        }
    }
    out
}

/// (d) render_full, downscaled to display size, matches display(). Beside
/// the raw mean absolute difference, reports it after a 3x3 blur (tone and
/// colour, without per-pixel texture and noise) and the mean signed
/// difference (any systematic shift). Rendering before or after resampling
/// differs by nature on sub-pixel detail (Jensen's inequality on the tone
/// curve), and the export also has its chroma noise removed; the raw bound
/// is 2/255 for neutral renders and 2.5/255 for a strong edit.
#[test]
#[ignore]
fn gpu_full_matches_display() {
    let gpu = Gpu::open().expect("gpu");
    for file in SAMPLES {
        let s = testkit::load(file);
        let mut e = engine(&gpu, &s);
        let n = neutral(&s.scene);
        for (label, p) in [
            ("neutral", n),
            (
                "edit",
                Params {
                    exposure: 0.4,
                    contrast: 30.0,
                    highlights: -70.0,
                    shadows: 60.0,
                    vibrance: 40.0,
                    ..n
                },
            ),
        ] {
            let disp = e.display(&p).unwrap();
            let (dw, dh) = (disp.width, disp.height);
            let shown: Vec<f32> = disp.data.iter().map(|&v| v as f32).collect();
            let mad = |a: &[f32], b: &[f32]| {
                a.iter()
                    .zip(b)
                    .map(|(x, y)| (x - y).abs() as f64)
                    .sum::<f64>()
                    / a.len() as f64
            };
            let t = Instant::now();
            let full = full::render(&gpu, &s.scene, &s.half, &p, &*s.color).unwrap();
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let (w, h, small) = shrink16(&full, DISPLAY_EDGE, true);
            assert_eq!((w, h), (dw, dh));
            let raw = mad(&small, &shown);
            let look = mad(&blur3(dw, dh, &small), &blur3(dw, dh, &shown));
            let signed: Vec<f64> = (0..3)
                .map(|c| {
                    small
                        .iter()
                        .zip(&shown)
                        .skip(c)
                        .step_by(3)
                        .map(|(a, b)| (a - b) as f64)
                        .sum::<f64>()
                        / (dw * dh) as f64
                })
                .collect();
            let bare = full::Finish {
                sharpen: 0.0,
                chroma_nr: false,
            };
            let plain = full::render_with(&gpu, &s.scene, &s.half, &p, &*s.color, bare).unwrap();
            let raw_bare = mad(&shrink16(&plain, DISPLAY_EDGE, true).2, &shown);
            println!(
                "{file} {label}: full {}x{} in {ms:.0} ms; downscaled vs display(): MAD {raw:.2}/255, \
                 {look:.2} after a 3x3 blur, mean signed {:+.2} {:+.2} {:+.2} (without sharpening and \
                 chroma NR: MAD {raw_bare:.2})",
                full.width, full.height, signed[0], signed[1], signed[2],
            );
            let bound = if label == "neutral" { 2.0 } else { 2.5 };
            assert!(raw < bound, "{file} {label}: MAD {raw}");
            assert!(look < 1.5, "{file} {label}: blurred MAD {look}");
            assert!(
                signed.iter().all(|d| d.abs() < 0.75),
                "{file} {label}: shift {signed:?}"
            );
            if label == "edit" {
                let crop = crop16(&full, 900, 600);
                testkit::write_jpeg(&crop, &format!("full_crop_{}.jpg", s.name));
            }
        }
    }
}

fn crop16(img: &Rgb16Image, w: usize, h: usize) -> Rgb8Image {
    let (x0, y0) = ((img.width - w) / 2, (img.height - h) / 2);
    let mut data = Vec::with_capacity(3 * w * h);
    for y in y0..y0 + h {
        for v in &img.data[3 * (y * img.width + x0)..3 * (y * img.width + x0 + w)] {
            data.push(((*v as u32 + 128) / 257) as u8);
        }
    }
    Rgb8Image {
        width: w,
        height: h,
        data,
    }
}

/// (e) 61 MP render_full: time and GPU memory.
#[test]
#[ignore]
fn gpu_full_61mp() {
    let gpu = Gpu::open().expect("gpu");
    let t = Instant::now();
    let (s, full) = testkit::load_full(SAMPLE_61MP);
    println!(
        "{SAMPLE_61MP}: {}x{} ({:.1} MP) demosaiced on the CPU in {:.1} s",
        full.width,
        full.height,
        (full.width * full.height) as f64 / 1e6,
        t.elapsed().as_secs_f64()
    );
    let p = Params {
        exposure: 0.3,
        highlights: -60.0,
        shadows: 40.0,
        vibrance: 30.0,
        ..neutral(&s.scene)
    };
    // Compile the module outside the timing.
    full::render(&gpu, &s.scene, &s.half, &p, &*s.color).unwrap();
    let (free_before, total) = gpu.ctx.mem_get_info().unwrap();
    let done = AtomicBool::new(false);
    let mut min_free = free_before;
    let mut times = Vec::new();
    let mut out = None;
    std::thread::scope(|scope| {
        let poll = scope.spawn(|| {
            let ctx = gpu.ctx.clone();
            let mut lowest = usize::MAX;
            while !done.load(Ordering::Relaxed) {
                if let Ok((free, _)) = ctx.mem_get_info() {
                    lowest = lowest.min(free);
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            lowest
        });
        for _ in 0..3 {
            let t = Instant::now();
            out = Some(full::render(&gpu, &s.scene, &full, &p, &*s.color).unwrap());
            times.push(t.elapsed().as_secs_f64() * 1e3);
        }
        done.store(true, Ordering::Relaxed);
        min_free = poll.join().unwrap();
    });
    let out = out.unwrap();
    println!(
        "render_full {}x{}: {:.0} / {:.0} / {:.0} ms; GPU memory in use during the render ~{:.0} MB (of {:.0} GB; device-wide, includes other processes)",
        out.width,
        out.height,
        times[0],
        times[1],
        times[2],
        free_before.saturating_sub(min_free) as f64 / 1e6,
        total as f64 / 1e9
    );
    testkit::write_jpeg(&crop16(&out, 1200, 800), "full61_crop.jpg");
    let lin = LinearImage {
        width: out.width,
        height: out.height,
        data: out.data.iter().map(|&v| v as f32 / 65535.0).collect(),
    };
    let small = testkit::downscale(&lin, 1600);
    let preview = Rgb8Image {
        width: small.width,
        height: small.height,
        data: small
            .data
            .iter()
            .map(|&v| (v * 255.0).round() as u8)
            .collect(),
    };
    testkit::write_jpeg(&preview, "full61_preview.jpg");
}
