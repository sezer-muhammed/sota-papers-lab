//! GPU tests for sharpening and noise reduction. Run them on the CUDA host:
//! `gpu-test <checkout> detail:: --ignored`. Sample crops go to
//! `$SPL_LENS_OUT` (default `/out/lens-detail`): `_before`/`_after` 1:1
//! centre crops of the neutral render, with the auto settings, for each
//! sample at its own ISO and with simulated ISO 6400 noise.

use std::path::PathBuf;
use std::time::Instant;

use super::*;
use crate::decode::decode;
use crate::develop::{develop_full, develop_scene};
use crate::encode::{jpeg, to_rgb8};
use crate::render::render_full;
use crate::types::Params;

fn out_dir() -> PathBuf {
    let dir = std::env::var_os("SPL_LENS_OUT")
        .map_or_else(|| PathBuf::from("/out/lens-detail"), PathBuf::from);
    std::fs::create_dir_all(&dir).expect("create output directory");
    dir
}

fn samples_dir() -> PathBuf {
    std::env::var_os("SPL_RAW_SAMPLES").map_or_else(|| PathBuf::from("/samples"), PathBuf::from)
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

/// Deterministic standard normal samples.
struct Gauss(u64);

impl Gauss {
    fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64).max(1e-12)
    }

    fn next(&mut self) -> f32 {
        let (a, b) = (self.uniform(), self.uniform());
        ((-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()) as f32
    }
}

fn image(w: usize, h: usize, f: impl Fn(usize, usize, usize) -> f32) -> Rgb16Image {
    let mut data = Vec::with_capacity(3 * w * h);
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                data.push((f(x, y, c).clamp(0.0, 1.0) * 65535.0).round() as u16);
            }
        }
    }
    Rgb16Image {
        width: w,
        height: h,
        data,
    }
}

fn at(img: &Rgb16Image, x: usize, y: usize, c: usize) -> f32 {
    f32::from(img.data[3 * (y * img.width + x) + c]) / 65535.0
}

/// Standard deviation of luma and of the colour differences over a window.
fn spread(img: &Rgb16Image, x0: usize, y0: usize, w: usize, h: usize) -> (f32, f32) {
    let mut l = Vec::new();
    let mut c = Vec::new();
    for y in y0..y0 + h {
        for x in x0..x0 + w {
            let [r, g, b] = [0, 1, 2].map(|k| at(img, x, y, k));
            let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            l.push(luma);
            c.push(r - luma);
            c.push(b - luma);
        }
    }
    let sd = |v: &[f32]| {
        let m = v.iter().sum::<f32>() / v.len() as f32;
        (v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / v.len() as f32).sqrt()
    };
    (sd(&l), sd(&c))
}

#[test]
#[ignore = "needs the CUDA host"]
fn gpu_detail_synthetic() {
    let gpu = Gpu::open().expect("GPU");
    let (w, h) = (512, 384);
    let mut rng = Gauss(11);
    // Grey with independent per-channel noise (sigma 0.03): luma and chroma noise.
    let noise: Vec<f32> = (0..3 * w * h).map(|_| 0.03 * rng.next()).collect();
    let flat = image(w, h, |x, y, c| 0.45 + noise[3 * (y * w + x) + c]);

    assert_eq!(
        apply(&gpu, &flat, &DetailSettings::OFF, 1.0).unwrap().data,
        flat.data,
        "off is the identity"
    );
    let (l0, c0) = spread(&flat, 64, 64, 384, 256);
    let nr = DetailSettings {
        luminance_nr: 50.0,
        color_nr: 50.0,
        ..DetailSettings::OFF
    };
    let out = apply(&gpu, &flat, &nr, 1.0).unwrap();
    let (l1, c1) = spread(&out, 64, 64, 384, 256);
    println!("flat: luma sd {l0:.4} -> {l1:.4}, chroma sd {c0:.4} -> {c1:.4}");
    assert!(l1 < l0 / 2.5, "luminance NR: {l0} -> {l1}");
    assert!(c1 < c0 / 5.0, "colour NR: {c0} -> {c1}");
    let only_color = DetailSettings {
        color_nr: 50.0,
        ..DetailSettings::OFF
    };
    let out = apply(&gpu, &flat, &only_color, 1.0).unwrap();
    let (l2, c2) = spread(&out, 64, 64, 384, 256);
    assert!(
        (l2 / l0 - 1.0).abs() < 0.05,
        "colour NR keeps luma: {l0} -> {l2}"
    );
    assert!(c2 < c0 / 5.0);

    // A colour edge (red | green at equal luma-ish) with noise: colour NR
    // must not bleed it.
    let edge = image(w, h, |x, y, c| {
        let base = if x < w / 2 {
            [0.6, 0.25, 0.2]
        } else {
            [0.2, 0.42, 0.25]
        };
        base[c] + noise[3 * (y * w + x) + c]
    });
    let out = apply(&gpu, &edge, &nr, 1.0).unwrap();
    // Rows are noisy; the mean over a column is what bleeding would move.
    let col_mean = |x: usize| {
        (64..h - 64)
            .map(|y| at(&out, x, y, 0) - at(&out, x, y, 2))
            .sum::<f32>()
            / (h - 128) as f32
    };
    let (left, right) = (col_mean(w / 2 - 3), col_mean(w / 2 + 2));
    println!("colour edge: R-B 3 px left {left:.3} (0.40), 2 px right {right:.3} (-0.05)");
    assert!((left - 0.4).abs() < 0.04 && (right + 0.05).abs() < 0.04);

    // A slightly soft step (0.25 -> 0.65), no noise: sharpening steepens
    // it; at Detail 0 the result stays inside the plateau values (no
    // halo), at Detail 100 it overshoots like a plain unsharp mask.
    let step = image(w, h, |x, _, _| {
        let t = (x as f32 - w as f32 / 2.0) / 0.8;
        0.25 + 0.4 / (1.0 + (-t).exp())
    });
    let slope = |img: &Rgb16Image| {
        (w / 2 - 4..w / 2 + 4)
            .map(|x| at(img, x + 1, h / 2, 1) - at(img, x, h / 2, 1))
            .fold(0.0f32, f32::max)
    };
    let row_range = |img: &Rgb16Image| {
        (0..w).fold((1.0f32, 0.0f32), |(lo, hi), x| {
            let v = at(img, x, h / 2, 1);
            (lo.min(v), hi.max(v))
        })
    };
    let sharp = |amount: f32, detail: f32| DetailSettings {
        sharpen_amount: amount,
        sharpen_radius: 1.0,
        sharpen_detail: detail,
        ..DetailSettings::OFF
    };
    let s0 = apply(&gpu, &step, &sharp(100.0, 0.0), 1.0).unwrap();
    let s25 = apply(&gpu, &step, &sharp(40.0, 25.0), 1.0).unwrap();
    let s100 = apply(&gpu, &step, &sharp(100.0, 100.0), 1.0).unwrap();
    let (r0, r100) = (row_range(&s0), row_range(&s100));
    println!(
        "step: slope {:.4} -> {:.4} (100/detail 0), {:.4} (Lightroom default 40/25), {:.4} (100/detail 100); range {r0:?} / {r100:?}",
        slope(&step),
        slope(&s0),
        slope(&s25),
        slope(&s100)
    );
    assert!(
        slope(&s25) > 1.1 * slope(&step),
        "the default sharpens visibly"
    );
    assert!(slope(&s0) > slope(&s25), "more amount, more sharpening");
    assert!(
        r0.0 > 0.25 - 0.003 && r0.1 < 0.65 + 0.003,
        "no halo at detail 0"
    );
    assert!(
        r100.0 < 0.25 - 0.002 && r100.1 > 0.65 + 0.002,
        "detail 100 overshoots like a plain USM"
    );
    // Masking 100 leaves flat noise alone.
    let masked = DetailSettings {
        sharpen_masking: 100.0,
        ..sharp(100.0, 100.0)
    };
    let (lm, _) = spread(&apply(&gpu, &flat, &masked, 1.0).unwrap(), 64, 64, 384, 256);
    let (lu, _) = spread(
        &apply(&gpu, &flat, &sharp(100.0, 100.0), 1.0).unwrap(),
        64,
        64,
        384,
        256,
    );
    println!("flat noise sharpened: luma sd {l0:.4} -> {lu:.4} unmasked, {lm:.4} masked");
    assert!(lm < lu && (lm / l0 - 1.0).abs() < 0.1);
}

/// Strips are invisible: rows near a strip boundary match a run where
/// they sit in the middle of a single strip.
#[test]
#[ignore = "needs the CUDA host"]
fn gpu_detail_strips_are_seamless() {
    let gpu = Gpu::open().expect("GPU");
    let (w, h) = (640, 2 * STRIP_ROWS + 300);
    let mut rng = Gauss(5);
    let noise: Vec<f32> = (0..3 * w * h).map(|_| 0.02 * rng.next()).collect();
    let img = image(w, h, |x, y, c| {
        let base = 0.3 + 0.3 * ((x as f32 / 37.0).sin() * (y as f32 / 23.0).cos());
        base + 0.1 * c as f32 + noise[3 * (y * w + x) + c]
    });
    let settings = DetailSettings {
        sharpen_amount: 150.0,
        sharpen_radius: 3.0,
        sharpen_detail: 60.0,
        sharpen_masking: 30.0,
        luminance_nr: 100.0,
        color_nr: 100.0,
    };
    let profile = noise_profile(&img);
    let whole = apply_with(&gpu, &img, &settings, 1.0, &profile).unwrap();
    // Rows [STRIP_ROWS - 512, STRIP_ROWS + 512) as an image of their own: one strip.
    let y0 = STRIP_ROWS - 512;
    let part = Rgb16Image {
        width: w,
        height: STRIP_ROWS,
        data: img.data[3 * w * y0..3 * w * (y0 + STRIP_ROWS)].to_vec(),
    };
    let alone = apply_with(&gpu, &part, &settings, 1.0, &profile).unwrap();
    let mut worst = 0u16;
    for y in HALO..STRIP_ROWS - HALO {
        for i in 0..3 * w {
            let a = whole.data[3 * w * (y0 + y) + i];
            let b = alone.data[3 * w * y + i];
            worst = worst.max(a.abs_diff(b));
        }
    }
    println!("strip boundary: worst difference {worst} / 65535");
    assert!(worst <= 1, "{worst}");
}

/// Adds Poisson-Gaussian noise to the mosaic as a camera at `iso` would
/// record it after the same exposure (full well 50 000 e- at ISO 100, read
/// noise 2 e-), so demosaicing turns it into realistic grain and blotches.
fn simulate_iso(frame: &mut crate::decode::RawFrame, iso: f32, seed: u64) {
    let electrons = 50_000.0 * 100.0 / iso;
    let read = 2.0 / electrons;
    let mut rng = Gauss(seed);
    let black = frame.black[0];
    let span = frame.white - black;
    for v in &mut frame.data {
        let x = ((f32::from(*v) - black) / span).max(0.0);
        let sigma = (x / electrons + read * read).sqrt();
        *v = (f32::from(*v) + sigma * span * rng.next()).clamp(0.0, 65535.0) as u16;
    }
}

fn crop8(img: &Rgb16Image, x0: usize, y0: usize, w: usize, h: usize) -> crate::types::Rgb8Image {
    let small = to_rgb8(img);
    let mut data = Vec::with_capacity(3 * w * h);
    for y in y0..y0 + h {
        data.extend_from_slice(&small.data[3 * (y * img.width + x0)..3 * (y * img.width + x0 + w)]);
    }
    crate::types::Rgb8Image {
        width: w,
        height: h,
        data,
    }
}

/// Mean absolute 4-neighbour Laplacian over a window, of luma and of the
/// colour differences (R - Y, B - Y): how much fine detail and grain, and
/// how much colour speckle, there is.
fn high_pass(img: &Rgb16Image, x0: usize, y0: usize, w: usize, h: usize) -> [f32; 2] {
    let ycc = |x: usize, y: usize| {
        let [r, g, b] = [0, 1, 2].map(|c| at(img, x, y, c));
        let l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        [l, r - l, b - l]
    };
    let mut sum = [0.0f64; 2];
    for y in y0 + 1..y0 + h - 1 {
        for x in x0 + 1..x0 + w - 1 {
            let [c, l, r, u, d] =
                [(x, y), (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)].map(|(x, y)| ycc(x, y));
            let lap = |k: usize| (4.0 * c[k] - l[k] - r[k] - u[k] - d[k]).abs();
            sum[0] += f64::from(lap(0));
            sum[1] += f64::from(lap(1) + lap(2));
        }
    }
    sum.map(|v| (v / ((w - 2) * (h - 2)) as f64) as f32)
}

/// Real renders at 24 and 61 MP: auto settings at the sample's ISO and at a
/// simulated ISO 6400; crops and timings.
#[test]
#[ignore = "needs the CUDA host and the sample RAWs"]
fn gpu_detail_samples() {
    let gpu = Gpu::open().expect("GPU");
    for (file, iso_sim) in [
        ("ILCE-7M3___DSC0009.ARW", None),
        ("ILCE-7M3___DSC0009.ARW", Some(6400.0)),
        ("ILCE-7RM5__7RM5-LosslessCompressedLarge.ARW", None),
        ("ILCE-6400__DSC00087.ARW", Some(12_800.0)),
    ] {
        let bytes = std::fs::read(samples_dir().join(file)).expect("read sample");
        let mut frame = decode(&bytes).expect("decode");
        let mut name = file.trim_end_matches(".ARW").to_string();
        let mut iso = frame.info.iso;
        if let Some(sim) = iso_sim {
            simulate_iso(&mut frame, sim, 3);
            name = format!("{name}_iso{sim}");
            iso = Some(sim as u32);
        }
        let scene = develop_scene(&gpu, &frame).expect("develop_scene");
        let full = develop_full(&gpu, &frame).expect("develop_full");
        let p = Params::neutral(scene.info.as_shot_temperature, scene.info.as_shot_tint);
        let rendered = render_full(&gpu, &scene, &full, &p).expect("render_full");
        let mp = (rendered.width * rendered.height) as f32 / 1e6;
        let settings = auto_detail(iso, mp);

        let t = Instant::now();
        let noise = noise_profile(&rendered);
        let profile_ms = ms(t);
        let t = Instant::now();
        let out = apply(&gpu, &rendered, &settings, 1.0).expect("apply");
        let cold = ms(t);
        let mut warm = Vec::new();
        for _ in 0..2 {
            let t = Instant::now();
            apply(&gpu, &rendered, &settings, 1.0).expect("apply");
            warm.push(ms(t));
        }
        let (free, total) = gpu.ctx.mem_get_info().unwrap_or((0, 0));
        println!(
            "== {name} {}x{} ({mp:.1} MP), ISO {iso:?}: {settings:?}",
            rendered.width, rendered.height
        );
        println!(
            "   noise profile {profile_ms:.0} ms (luma sigma at mid grey {:.4}, chroma {:.4}) | apply {cold:.0} ms cold, {:.0} / {:.0} ms warm | GPU free {} of {} MB",
            noise.luma[NBINS / 2],
            noise.chroma[NBINS / 2],
            warm[0],
            warm[1],
            free >> 20,
            total >> 20
        );
        let (cw, ch) = (900, 600);
        let (x0, y0) = ((rendered.width - cw) / 2, (rendered.height - ch) / 2);
        for (img, tag) in [(&rendered, "before"), (&out, "after")] {
            let bytes = jpeg(&crop8(img, x0, y0, cw, ch), 92).expect("jpeg");
            std::fs::write(out_dir().join(format!("{name}_detail_{tag}.jpg")), bytes)
                .expect("write");
        }
        let (hf_before, hf_after) = (
            high_pass(&rendered, x0, y0, cw, ch),
            high_pass(&out, x0, y0, cw, ch),
        );
        println!(
            "   centre crop high-pass: luma {:.5} -> {:.5} (x{:.2}), colour {:.5} -> {:.5} (x{:.2})",
            hf_before[0],
            hf_after[0],
            hf_after[0] / hf_before[0],
            hf_before[1],
            hf_after[1],
            hf_after[1] / hf_before[1]
        );
        // Preview path: an 8-bit downscale with its scale factor.
        let small = crate::encode::resize(&to_rgb8(&rendered), 1536);
        let scale = small.width as f32 / rendered.width as f32;
        let t = Instant::now();
        let preview = apply8(&gpu, &small, &settings, scale).expect("apply8");
        println!("   1536 px preview (scale {scale:.3}): {:.1} ms", ms(t));
        assert_eq!(preview.data.len(), small.data.len());
    }
}
