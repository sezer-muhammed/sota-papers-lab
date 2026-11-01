//! The parametric tone curve. CPU tests describe its shape; the GPU ones
//! (`#[ignore]`, run on the GPU host with
//! `gpu-test <checkout> render::tone_curve --ignored`) check it on a grey ramp
//! and in the statistics, and write -100/-50/0/+50/+100 strips for two samples
//! under /out/render-tone/.

use std::sync::Arc;

use super::super::RenderEngine;
use super::super::presence::tests::{
    Identity, SAMPLES, SWEEP, column_luma, engine, neutral, save, scene, stats_finite,
};
use super::super::testkit;
use super::*;
use crate::types::{Gpu, Rgb8Image};

type Setter = fn(&mut Params, f32);

const REGIONS: [(&str, Setter, f64); 4] = [
    ("shadows", |p, v| p.curve_shadows = v, 0.125),
    ("darks", |p, v| p.curve_darks = v, 0.375),
    ("lights", |p, v| p.curve_lights = v, 0.625),
    ("highlights", |p, v| p.curve_highlights = v, 0.875),
];

fn with(f: impl Fn(&mut Params)) -> Params {
    let mut p = Params::neutral(5000.0, 0.0);
    f(&mut p);
    p
}

#[test]
fn neutral_is_off() {
    let n = Params::neutral(5000.0, 0.0);
    assert!(ParametricCurve::new(&n).is_none());
    let mut out = vec![7.0; CURVE_LEN];
    pack(&n, &mut out);
    assert!(out.iter().all(|&v| v == 0.0));
    assert!(ParametricCurve::new(&with(|p| p.curve_darks = f32::NAN)).is_none());
}

/// Every combination of region settings keeps the curve rising (no tone
/// reversal), black at black and white at white.
#[test]
fn never_reverses_tones() {
    for s in SWEEP {
        for d in SWEEP {
            for l in SWEEP {
                for h in SWEEP {
                    let p = with(|p| {
                        p.curve_shadows = s;
                        p.curve_darks = d;
                        p.curve_lights = l;
                        p.curve_highlights = h;
                    });
                    let Some(c) = ParametricCurve::new(&p) else {
                        continue;
                    };
                    assert_eq!(c.eval(0.0), 0.0);
                    assert!((c.eval(1.0) - 1.0).abs() < 1e-12);
                    let mut prev = 0.0;
                    for i in 1..=1000 {
                        let y = c.eval(i as f64 / 1000.0);
                        assert!(y > prev, "{s} {d} {l} {h} at {i}");
                        prev = y;
                    }
                }
            }
        }
    }
}

/// Each slider moves its own region's centre by its amplitude, half that at
/// its splits, and nothing two regions away; the move grows with the slider.
#[test]
fn regions_move_their_own_tones() {
    for (i, (name, f, centre)) in REGIONS.iter().enumerate() {
        let mut prev = -1.0;
        for v in SWEEP {
            let p = with(|p| f(p, v));
            let y = ParametricCurve::new(&p).map_or(*centre, |c| c.eval(*centre));
            assert!(y > prev, "{name} {v}");
            prev = y;
        }
        let c = ParametricCurve::new(&with(|p| f(p, 100.0))).unwrap();
        assert!(
            (c.eval(*centre) - centre - AMPLITUDE[i]).abs() < 1e-12,
            "{name}"
        );
        for (j, (_, _, other)) in REGIONS.iter().enumerate() {
            if i.abs_diff(j) >= 2 {
                assert!(
                    (c.eval(*other) - other).abs() < 1e-12,
                    "{name} moves {other}"
                );
            }
        }
        for split in [0.25, 0.5, 0.75] {
            if (split - centre).abs() < 0.2 {
                let moved = c.eval(split) - split;
                assert!(
                    (moved - 0.5 * AMPLITUDE[i]).abs() < 1e-12,
                    "{name} at {split}"
                );
            }
        }
        // Lowering is the mirror of raising at the centre.
        let down = ParametricCurve::new(&with(|p| f(p, -100.0))).unwrap();
        assert!((down.eval(*centre) - centre + AMPLITUDE[i]).abs() < 1e-12);
    }
    // Equal settings everywhere lift the whole middle evenly.
    let all = ParametricCurve::new(&with(|p| {
        p.curve_shadows = 50.0;
        p.curve_darks = 50.0;
        p.curve_lights = 50.0;
        p.curve_highlights = 50.0;
    }))
    .unwrap();
    for x in [0.375, 0.5, 0.625] {
        assert!((all.eval(x) - x - 0.05).abs() < 1e-3, "{x}");
    }
}

/// The baked table is the curve, rising, from 0 to 1, and its interpolation
/// (as the kernel does it) stays within a fraction of an 8-bit level.
#[test]
fn table_matches_the_curve() {
    let p = with(|p| {
        p.curve_shadows = 70.0;
        p.curve_darks = -40.0;
        p.curve_lights = 25.0;
        p.curve_highlights = -90.0;
    });
    let c = ParametricCurve::new(&p).unwrap();
    let mut out = vec![0.0; CURVE_LEN];
    pack(&p, &mut out);
    assert_eq!(out[0], 1.0);
    let t = &out[1..1 + CURVE_N];
    assert_eq!(t[0], 0.0);
    assert!((t[CURVE_N - 1] - 1.0).abs() < 1e-6);
    assert!(t.windows(2).all(|w| w[1] > w[0]));
    assert!(out[1 + CURVE_N..].iter().all(|&v| v == 0.0));
    let kernel = |v: f64| {
        let f = v.clamp(0.0, 1.0).sqrt() * (CURVE_N - 1) as f64;
        let i = (f as usize).min(CURVE_N - 2);
        let a = f - i as f64;
        let w = t[i] as f64 + a * (t[i + 1] as f64 - t[i] as f64);
        w * w
    };
    let mut worst: f64 = 0.0;
    for i in 0..=4000 {
        let e = i as f64 / 4000.0;
        let v = super::eotf(e);
        let err = (super::oetf(kernel(v)) - c.eval(e)).abs() * 255.0;
        worst = worst.max(err);
    }
    assert!(worst < 0.25, "table off by {worst} levels");
}

#[test]
fn table_size_matches_cuda() {
    let line = include_str!("../../../cuda/tone_curve.cu")
        .lines()
        .find_map(|l| l.strip_prefix("#define CURVE_N "))
        .expect("CURVE_N in cuda/tone_curve.cu");
    assert_eq!(line.trim().parse::<usize>().unwrap(), CURVE_N);
}

/// On a grey ramp: zero is byte-identical to neutral, every setting keeps
/// the ramp rising, and each slider moves its region's tones monotonically.
#[test]
#[ignore]
fn gpu_tone_curve_ramp() {
    let gpu = Gpu::open().expect("gpu");
    // log2 scene value from -12 to +0.5 across the frame.
    let s = scene(1536, 256, |x, _| [(-12.0 + 12.5 * x).exp2().min(1.0); 3]);
    let mut e = RenderEngine::with_color(&gpu, &s, Arc::new(Identity)).unwrap();
    let n = neutral(&s);
    let base = e.display(&n).unwrap();
    let base_cols = column_luma(&base);
    for (name, f, centre) in REGIONS {
        let zero = e.display(&with_n(n, f, 0.0)).unwrap();
        assert_eq!(zero.data, base.data, "{name} at 0 changed the render");
        // The column whose neutral output sits at the region's centre.
        let col = base_cols
            .iter()
            .position(|&c| c / 255.0 >= centre)
            .expect("ramp reaches the region");
        let mut prev = -1.0;
        for v in SWEEP {
            let p = with_n(n, f, v);
            let img = e.display(&p).unwrap();
            let c = column_luma(&img);
            let worst = c
                .windows(2)
                .map(|w| w[1] - w[0])
                .fold(f64::INFINITY, f64::min);
            assert!(
                worst > -0.25,
                "{name} {v}: ramp reverses by {worst:.2} levels"
            );
            println!(
                "{name:>10} {v:+4}: centre column {:.1} (neutral {:.1})",
                c[col], base_cols[col]
            );
            assert!(c[col] > prev, "{name} {v}: centre not rising");
            prev = c[col];
            let st = e.stats(&[p]).unwrap().remove(0);
            assert!(stats_finite(&st));
        }
        // At +-100 the centre moves by about the amplitude (in 8-bit levels).
        let up = column_luma(&e.display(&with_n(n, f, 100.0)).unwrap())[col] - base_cols[col];
        let i = REGIONS.iter().position(|r| r.0 == name).unwrap();
        assert!(
            (up - 255.0 * AMPLITUDE[i]).abs() < 4.0,
            "{name}: +100 moves the centre {up:.1} levels"
        );
    }
}

fn with_n(n: Params, f: Setter, v: f32) -> Params {
    let mut p = n;
    f(&mut p, v);
    p
}

/// Statistics follow the curve: Darks and Lights move the median, Shadows
/// the dark end, Highlights the bright end.
#[test]
#[ignore]
fn gpu_tone_curve_stats() {
    let gpu = Gpu::open().expect("gpu");
    for file in SAMPLES {
        let s = testkit::load(file);
        let mut e = engine(&gpu, &s);
        let n = neutral(&s.scene);
        for (name, f, _) in REGIONS {
            let st = e.stats(&SWEEP.map(|v| with_n(n, f, v))).unwrap();
            let pick = |x: &crate::types::Stats| match name {
                "shadows" => x.luma_p05,
                "highlights" => x.luma_p95,
                _ => x.luma_p50,
            };
            println!(
                "{file} {name}: {:?}",
                st.iter()
                    .map(|x| (pick(x) * 1000.0).round() / 1000.0)
                    .collect::<Vec<_>>()
            );
            for w in st.windows(2) {
                assert!(pick(&w[1]) > pick(&w[0]), "{name}");
            }
        }
    }
}

/// -100/-50/0/+50/+100 strips for each region slider (half display size).
#[test]
#[ignore]
fn gpu_tone_curve_strips() {
    let gpu = Gpu::open().expect("gpu");
    for file in SAMPLES {
        let s = testkit::load(file);
        let mut e = engine(&gpu, &s);
        let n = neutral(&s.scene);
        let mut rows = Vec::new();
        for (name, f, _) in REGIONS {
            let imgs: Vec<Rgb8Image> = SWEEP
                .iter()
                .map(|&v| testkit::shrink(&e.display(&with_n(n, f, v)).unwrap(), 2))
                .collect();
            save(
                &testkit::sheet(std::slice::from_ref(&imgs), 4),
                &format!("curve_{}_{name}.jpg", s.name),
            );
            rows.push(imgs.into_iter().map(|i| testkit::shrink(&i, 2)).collect());
        }
        let path = save(&testkit::sheet(&rows, 4), &format!("curve_{}.jpg", s.name));
        println!(
            "{file}: rows shadows | darks | lights | highlights -> {}",
            path.display()
        );
    }
}

/// Packing cost per parameter set with the curve on (in release:
/// `cargo test --release -p spl-app-raw-retouch pack_cost -- --ignored --nocapture`).
#[test]
#[ignore]
fn pack_cost() {
    let mut out = vec![0.0; CURVE_LEN];
    let n = 2000;
    let t = std::time::Instant::now();
    for i in 0..n {
        let p = with(|p| {
            p.curve_darks = (i % 200) as f32 - 100.0;
            p.curve_highlights = 30.0;
        });
        pack(&p, &mut out);
    }
    let us = t.elapsed().as_secs_f64() * 1e6 / n as f64;
    println!("tone curve pack: {us:.1} us per parameter set");
}
