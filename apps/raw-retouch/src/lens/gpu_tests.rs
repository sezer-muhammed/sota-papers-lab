//! Sample-file and GPU tests for lens correction. Run them on the CUDA host:
//! `gpu-test <checkout> lens:: --ignored`. Images go to `$SPL_LENS_OUT`
//! (default `/out/lens-detail`): per sample the neutral display render
//! before and after (`_display_before/after`: vignetting, straight lines)
//! and the top-left corner of the full-resolution render at 2x
//! (`_corner_before/after`: lateral colour fringes).

use std::path::PathBuf;
use std::time::Instant;

use super::*;
use crate::decode::decode;
use crate::develop::{develop_full, develop_scene};
use crate::encode::{jpeg, to_rgb8};
use crate::render::{RenderEngine, render_full};
use crate::types::{Params, Rgb8Image, Rgb16Image, Scene};

fn samples_dir() -> PathBuf {
    std::env::var_os("SPL_RAW_SAMPLES").map_or_else(|| PathBuf::from("/samples"), PathBuf::from)
}

pub(crate) fn out_dir() -> PathBuf {
    let dir = std::env::var_os("SPL_LENS_OUT")
        .map_or_else(|| PathBuf::from("/out/lens-detail"), PathBuf::from);
    std::fs::create_dir_all(&dir).expect("create output directory");
    dir
}

fn samples() -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(samples_dir())
        .expect("sample directory")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("arw")))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no RAW samples found");
    files
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn write(img: &Rgb8Image, name: &str) {
    std::fs::write(out_dir().join(name), jpeg(img, 92).expect("jpeg")).expect("write");
}

/// `w` x `h` at (x0, y0) of a 16-bit render, enlarged `k` times (nearest).
fn crop_zoom(img: &Rgb16Image, x0: usize, y0: usize, w: usize, h: usize, k: usize) -> Rgb8Image {
    let small = to_rgb8(img);
    let mut data = Vec::with_capacity(3 * w * h * k * k);
    for y in 0..h * k {
        for x in 0..w * k {
            let i = 3 * ((y0 + y / k) * img.width + x0 + x / k);
            data.extend_from_slice(&small.data[i..i + 3]);
        }
    }
    Rgb8Image {
        width: w * k,
        height: h * k,
        data,
    }
}

/// Mean luma of a `frac`-sized patch of the 8-bit render at a corner and
/// at the centre, as EV (corner relative to centre), averaged over corners.
fn corner_falloff_ev(img: &Rgb8Image, frac: f32) -> f32 {
    let (w, h) = (img.width, img.height);
    let (pw, ph) = (
        ((w as f32 * frac) as usize).max(1),
        ((h as f32 * frac) as usize).max(1),
    );
    let lin = |v: u8| {
        let s = f32::from(v) / 255.0;
        if s <= 0.04045 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    };
    let patch = |x0: usize, y0: usize| {
        let mut sum = 0.0f64;
        for y in y0..y0 + ph {
            for x in x0..x0 + pw {
                let i = 3 * (y * w + x);
                let px = &img.data[i..i + 3];
                sum += f64::from(0.2126 * lin(px[0]) + 0.7152 * lin(px[1]) + 0.0722 * lin(px[2]));
            }
        }
        (sum / (pw * ph) as f64) as f32
    };
    let centre = patch((w - pw) / 2, (h - ph) / 2).max(1e-6);
    let corners = [(0, 0), (w - pw, 0), (0, h - ph), (w - pw, h - ph)];
    corners
        .iter()
        .map(|&(x, y)| (patch(x, y).max(1e-6) / centre).log2())
        .sum::<f32>()
        / 4.0
}

/// Lateral chromatic aberration as measured in the image: the radial scale
/// `k` at which channel `c` read at `centre + (p - centre) * (1 + k)` best
/// matches green at `p`, over high-contrast pixels in the four corner
/// windows (where it is largest). Compares local contrast (each channel
/// minus its 3x3 mean, over that mean), so colour itself does not matter.
fn radial_misalignment(img: &LinearImage, c: usize) -> f32 {
    let (w, h) = (img.width, img.height);
    let (cx, cy) = ((w - 1) as f32 / 2.0, (h - 1) as f32 / 2.0);
    let px = |x: usize, y: usize, ch: usize| img.data[3 * (y * w + x) + ch].max(1e-4);
    let contrast = |x: usize, y: usize, ch: usize| {
        let mut m = 0.0;
        for yy in y - 1..=y + 1 {
            for xx in x - 1..=x + 1 {
                m += px(xx, yy, ch);
            }
        }
        m /= 9.0;
        (px(x, y, ch) - m) / m
    };
    let s = 700.min(w / 4).min(h / 4);
    // Far enough from the frame edge for the largest scale tried below.
    let margin = (0.0031 * cx.hypot(cy)).ceil() as usize + 3;
    let mut points = Vec::new();
    for (x0, y0) in [(0, 0), (w - s, 0), (0, h - s), (w - s, h - s)] {
        for y in (y0 + margin..y0 + s - margin).step_by(2) {
            for x in (x0 + margin..x0 + s - margin).step_by(2) {
                let g = contrast(x, y, 1);
                if g.abs() > 0.04 {
                    points.push((x, y, g));
                }
            }
        }
    }
    // Bilinear contrast of channel c at a fractional position.
    let sample = |x: f32, y: f32| {
        let (x0, y0) = (x.floor() as usize, y.floor() as usize);
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let v = |xx, yy| contrast(xx, yy, c);
        (1.0 - fy) * ((1.0 - fx) * v(x0, y0) + fx * v(x0 + 1, y0))
            + fy * ((1.0 - fx) * v(x0, y0 + 1) + fx * v(x0 + 1, y0 + 1))
    };
    let error = |k: f32| -> f64 {
        points
            .iter()
            .map(|&(x, y, g)| {
                let sx = cx + (x as f32 - cx) * (1.0 + k);
                let sy = cy + (y as f32 - cy) * (1.0 + k);
                f64::from((sample(sx, sy) - g).powi(2))
            })
            .sum()
    };
    // Coarse grid, then a parabola through the best three.
    let ks: Vec<f32> = (-30..=30).map(|i| i as f32 * 0.0001).collect();
    let errs: Vec<f64> = ks.iter().map(|&k| error(k)).collect();
    let i = (1..ks.len() - 1)
        .min_by(|&a, &b| errs[a].total_cmp(&errs[b]))
        .unwrap_or(1);
    let (a, b, d) = (errs[i - 1], errs[i], errs[i + 1]);
    let denom = a - 2.0 * b + d;
    let offset = if denom > 0.0 {
        0.5 * (a - d) / denom
    } else {
        0.0
    };
    ks[i] + offset as f32 * 0.0001
}

/// Which samples carry a profile, and what it does. Every Sony sample has one.
#[test]
#[ignore = "needs the sample RAWs"]
fn lens_profile_of_every_sample() {
    let mut missing = Vec::new();
    for path in samples() {
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let bytes = std::fs::read(&path).expect("read sample");
        let t = Instant::now();
        let profile = lens_profile(&bytes);
        let read_ms = ms(t);
        let frame = decode(&bytes).expect("decode");
        match profile {
            Some(c) => println!(
                "{name}: {} | zoom {:.4} ({}x{}) | read {read_ms:.1} ms | {:?} {:?} mm",
                c.summary(),
                c.zoom(frame.width, frame.height, LensSettings::AUTO),
                frame.width,
                frame.height,
                frame.info.lens,
                frame.info.focal_length_mm,
            ),
            None => {
                println!("{name}: no lens correction data");
                missing.push(name);
            }
        }
    }
    assert!(missing.is_empty(), "no profile: {missing:?}");
}

const VISUAL: [&str; 6] = [
    "DSC-RX100M3__DSC00734.ARW",
    "ILCE-6700__DSC00001.ARW",
    "ILCE-7RM5__7RM5-LosslessCompressedLarge.ARW",
    "ILCE-6000__DSC01542.ARW",
    "ILCE-6400__DSC00087.ARW",
    "ILCE-7M3___DSC0009.ARW",
];

/// Before/after renders, corner falloff and measured CA, timings at 24
/// and 61 MP.
#[test]
#[ignore = "needs the CUDA host and the sample RAWs"]
fn gpu_lens_before_after() {
    let gpu = Gpu::open().expect("GPU");
    let mut failures = Vec::new();
    for file in VISUAL {
        let name = file.trim_end_matches(".ARW");
        let bytes = std::fs::read(samples_dir().join(file)).expect("read sample");
        let mut frame = decode(&bytes).expect("decode");
        let scene = develop_scene(&gpu, &frame).expect("develop_scene");
        let full = develop_full(&gpu, &frame).expect("develop_full");
        let corr = lens_profile(&bytes).expect("profile");
        // The production order: CA on the RAW samples, then the profile on
        // the demosaiced image.
        let t = Instant::now();
        correct_mosaic(&gpu, &mut frame, &corr).expect("correct_mosaic");
        let mosaic_ms = ms(t);
        let fixed_full = develop_full(&gpu, &frame).expect("develop_full");
        let fixed_scene0 = develop_scene(&gpu, &frame).expect("develop_scene");
        let settings = LensSettings {
            profile: true,
            lateral_ca: false,
        };

        let t = Instant::now();
        let fixed = apply(&gpu, &fixed_full, &corr, settings).expect("apply");
        let cold = ms(t);
        let t = Instant::now();
        let fixed = apply(&gpu, &fixed_full, &corr, settings)
            .map(|_| fixed)
            .expect("apply");
        let warm = ms(t);
        // Kernel alone, the image already on the GPU.
        let dev = gpu.stream.clone_htod(&full.data).expect("upload");
        gpu.stream.synchronize().expect("sync");
        let t = Instant::now();
        let out = apply_device(&gpu, &dev, full.width, full.height, &corr, settings)
            .expect("apply_device");
        gpu.stream.synchronize().expect("sync");
        let kernel = ms(t);
        drop((dev, out));

        let fixed_scene = Scene {
            info: scene.info.clone(),
            color: scene.color.clone(),
            wb_as_shot: scene.wb_as_shot,
            search: apply(&gpu, &fixed_scene0.search, &corr, settings).expect("search"),
            display: apply(&gpu, &fixed_scene0.display, &corr, settings).expect("display"),
        };
        let p = Params::neutral(scene.info.as_shot_temperature, scene.info.as_shot_tint);
        let before = RenderEngine::new(&gpu, &scene)
            .and_then(|mut e| e.display(&p))
            .expect("render before");
        let after = RenderEngine::new(&gpu, &fixed_scene)
            .and_then(|mut e| e.display(&p))
            .expect("render after");
        write(&before, &format!("{name}_display_before.jpg"));
        write(&after, &format!("{name}_display_after.jpg"));
        let (fall_before, fall_after) = (
            corner_falloff_ev(&before, 0.06),
            corner_falloff_ev(&after, 0.06),
        );

        let full_before = render_full(&gpu, &scene, &full, &p).expect("render_full");
        let full_after = render_full(&gpu, &fixed_scene, &fixed, &p).expect("render_full");
        let size = 400;
        write(
            &crop_zoom(&full_before, 0, 0, size, size * 2 / 3, 2),
            &format!("{name}_corner_before.jpg"),
        );
        write(
            &crop_zoom(&full_after, 0, 0, size, size * 2 / 3, 2),
            &format!("{name}_corner_after.jpg"),
        );
        let ca = |img: &LinearImage| [radial_misalignment(img, 0), radial_misalignment(img, 2)];
        let (ca_before, ca_after) = (ca(&full), ca(&fixed));

        println!(
            "== {name} {}x{} ({:.1} MP): {}",
            full.width,
            full.height,
            (full.width * full.height) as f64 / 1e6,
            corr.summary()
        );
        println!(
            "   zoom {:.4} | correct_mosaic {mosaic_ms:.0} ms | apply {cold:.0} ms cold, {warm:.0} ms warm (host round trip), kernel {kernel:.1} ms",
            corr.zoom(full.width, full.height, settings)
        );
        println!(
            "   corner falloff {fall_before:+.2} EV -> {fall_after:+.2} EV | measured CA red {:+.4}% -> {:+.4}%, blue {:+.4}% -> {:+.4}%",
            ca_before[0] * 100.0,
            ca_after[0] * 100.0,
            ca_before[1] * 100.0,
            ca_after[1] * 100.0
        );
        if corr.vignetting.is_some() && fall_after <= fall_before {
            failures.push(format!("{name}: corners did not brighten"));
        }
        if !fixed.data.iter().all(|v| v.is_finite() && *v >= 0.0) {
            failures.push(format!("{name}: non-finite or negative output"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// No-op settings return the input untouched; vignetting alone keeps every
/// pixel in place (no resampling blur).
#[test]
#[ignore = "needs the CUDA host"]
fn gpu_lens_identity_and_vignetting_only() {
    let gpu = Gpu::open().expect("GPU");
    let (w, h) = (300, 200);
    let data: Vec<f32> = (0..w * h * 3)
        .map(|i| ((i * 7919) % 1000) as f32 / 1000.0)
        .collect();
    let img = LinearImage {
        width: w,
        height: h,
        data,
    };
    let vig = [
        11, 0, 320, 768, 1376, 2080, 2816, 3744, 4992, 6528, 8384, 10592,
    ];
    let corr = LensCorrection::from_sony_tags(None, None, Some(&vig));
    let off = LensSettings {
        profile: false,
        lateral_ca: true,
    };
    assert_eq!(apply(&gpu, &img, &corr, off).unwrap().data, img.data);
    let out = apply(&gpu, &img, &corr, LensSettings::AUTO).unwrap();
    let ihd = 1.0 / (w as f32 / 2.0).hypot(h as f32 / 2.0);
    let curve = corr.vignetting.as_ref().unwrap();
    for y in 0..h {
        for x in 0..w {
            let r = (x as f32 - (w - 1) as f32 / 2.0).hypot(y as f32 - (h - 1) as f32 / 2.0) * ihd;
            for c in 0..3 {
                let i = 3 * (y * w + x) + c;
                let want = img.data[i] * curve.at(r);
                assert!(
                    (out.data[i] - want).abs() < 2e-3 * want.max(0.01),
                    "({x}, {y}, {c}): {} vs {want}",
                    out.data[i]
                );
            }
        }
    }
}

/// Lateral CA alone, measured as the radial scale that best aligns a colour
/// with green in the corners. On the RAW sites (no interpolation) red shows
/// the profile's sign and size, which pins the decoding (2^-21 units, knot
/// layout, direction). In the RCD result only about half of it is visible
/// (colour-difference demosaicing drags red and blue onto green's edges), so
/// warping the demosaiced channels by the profile over-corrects, while
/// correcting the RAW samples first ([`correct_mosaic`]) leaves the least.
/// The A6400 shot's corners are out of focus; its numbers are printed only.
#[test]
#[ignore = "needs the CUDA host and the sample RAWs"]
fn gpu_lens_ca_direction() {
    let gpu = Gpu::open().expect("GPU");
    let only_ca = LensSettings {
        profile: false,
        lateral_ca: true,
    };
    let mut failures = Vec::new();
    for file in [
        "DSC-RX100M3__DSC00734.ARW",
        "ILCE-7M3___DSC0009.ARW",
        "ILCE-7M3__UNCOMPRESSED_14bit.ARW",
        "ILCE-6000__DSC01542.ARW",
        "ILCE-6400__DSC00087.ARW",
    ] {
        let sharp_corners = !file.starts_with("ILCE-64");
        let bytes = std::fs::read(samples_dir().join(file)).expect("read sample");
        let mut frame = decode(&bytes).expect("decode");
        let full = develop_full(&gpu, &frame).expect("develop_full");
        let corr = lens_profile(&bytes).expect("profile");
        let warped = apply(&gpu, &full, &corr, only_ca).expect("apply");
        let raw_before = superpixel(&frame);
        let t = Instant::now();
        assert!(correct_mosaic(&gpu, &mut frame, &corr).expect("correct_mosaic"));
        let mosaic_ms = ms(t);
        let fixed = develop_full(&gpu, &frame).expect("develop_full");
        let raw_after = superpixel(&frame);
        let [red, blue] = corr.ca.as_ref().expect("CA data");
        // The windows sit around 0.9 of the half-diagonal.
        let profile = [red.at(0.9) - 1.0, blue.at(0.9) - 1.0];
        for (k, c) in [(0, 0usize), (1, 2)] {
            let [before, after_warp, after_mosaic, raw_b, raw_a] =
                [&full, &warped, &fixed, &raw_before, &raw_after]
                    .map(|img| radial_misalignment(img, c));
            println!(
                "{file} {}: profile {:+.4}% | RAW sites {:+.4}% -> {:+.4}% | demosaiced {:+.4}%, after warp {:+.4}%, after mosaic correction {:+.4}% ({mosaic_ms:.0} ms)",
                ["red", "blue"][k],
                profile[k] * 100.0,
                raw_b * 100.0,
                raw_a * 100.0,
                before * 100.0,
                after_warp * 100.0,
                after_mosaic * 100.0
            );
            // Red is the reliable channel for this estimator (blue is dim
            // and noisy): the RAW sites show the profile's CA, and the
            // demosaiced result is better aligned after the mosaic
            // correction than after the warp.
            let valid = before.abs() < 0.002;
            if c == 0 && sharp_corners && (raw_b * profile[0] <= 0.0 || raw_b / profile[0] > 2.0) {
                failures.push(format!(
                    "{file}: RAW-site CA {raw_b} vs profile {}",
                    profile[0]
                ));
            }
            if c == 0 && valid && after_mosaic.abs() > after_warp.abs() {
                failures.push(format!("{file}: mosaic correction worse than the warp"));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}

/// Half-size image straight from the RAW sites (R, mean of the two G, B per
/// 2x2 block), black subtracted: colour planes with no interpolation between
/// them, so their misalignment is the lens's.
fn superpixel(frame: &crate::decode::RawFrame) -> LinearImage {
    assert_eq!(frame.channels, 1);
    let (w, h) = (frame.width / 2, frame.height / 2);
    let mut data = Vec::with_capacity(3 * w * h);
    for y in 0..h {
        for x in 0..w {
            let mut px = [0.0f32; 3];
            for (slot, (dy, dx)) in [(0, 0), (0, 1), (1, 0), (1, 1)].into_iter().enumerate() {
                let v = f32::from(frame.data[(2 * y + dy) * frame.width + 2 * x + dx]);
                let c = usize::from(frame.cfa[slot]);
                let share = if c == 1 { 0.5 } else { 1.0 };
                px[c] += share * (v - frame.black[slot]).max(0.0) / frame.white;
            }
            data.extend_from_slice(&px);
        }
    }
    LinearImage {
        width: w,
        height: h,
        data,
    }
}
