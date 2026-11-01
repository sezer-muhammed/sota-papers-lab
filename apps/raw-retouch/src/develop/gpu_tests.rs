//! GPU and sample-file tests for decode, colour and develop. They need the
//! CUDA host (`~/spl-work/bin/gpu-test <checkout> develop --ignored`); plain
//! `cargo test` skips them. Outputs go to `$SPL_DEVELOP_OUT` (default
//! `/out/develop`), per sample and through a simple view transform: the
//! display render (`_display`), 600x400 1:1 crops of the full-resolution
//! demosaic at the centre (`_crop`) and at the most detailed spot
//! (`_detail`), and the middle of the latter at 4x (`_zoom`).

use std::path::{Path, PathBuf};
use std::time::Instant;

use jpeg_encoder::{ColorType, Encoder, SamplingFactor};

use super::*;
use crate::color::ColorModel;
use crate::decode::decode;
use crate::types::RawInfo;

fn samples_dir() -> PathBuf {
    std::env::var_os("SPL_RAW_SAMPLES")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/samples"))
}

fn out_dir() -> PathBuf {
    let dir = std::env::var_os("SPL_DEVELOP_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/out/develop"));
    std::fs::create_dir_all(&dir).expect("create output directory");
    dir
}

fn samples() -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(samples_dir())
        .expect("sample directory")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("arw") || e.eq_ignore_ascii_case("dng"))
        })
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no RAW samples found");
    files
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

/// Rec.2020 → Rec.709/sRGB primaries, linear (ITU-R BT.2087).
const REC2020_TO_SRGB: [[f32; 3]; 3] = [
    [1.660_491, -0.587_641, -0.072_85],
    [-0.124_55, 1.132_9, -0.008_349],
    [-0.018_151, -0.100_579, 1.118_73],
];

/// Test-only look: camera → Rec.2020 → sRGB, an exposure gain, the
/// Narkowicz ACES fit as a filmic curve, sRGB gamma.
struct View {
    cam_to_srgb: [[f32; 3]; 3],
    gain: f32,
}

impl View {
    /// Exposure so the median luma lands on 0.18.
    fn new(color: &ColorModel, temperature: f32, img: &LinearImage) -> Self {
        let m = color.rec2020_from_cam(temperature);
        let mut cam_to_srgb = [[0.0; 3]; 3];
        for (r, row) in cam_to_srgb.iter_mut().enumerate() {
            for (c, v) in row.iter_mut().enumerate() {
                *v = (0..3).map(|k| REC2020_TO_SRGB[r][k] * m[k][c]).sum();
            }
        }
        let mut view = Self {
            cam_to_srgb,
            gain: 1.0,
        };
        let mut luma: Vec<f32> = img
            .data
            .as_chunks::<3>()
            .0
            .iter()
            .step_by(7)
            .map(|px| {
                let [r, g, b] = view.linear(*px);
                0.2126 * r + 0.7152 * g + 0.0722 * b
            })
            .collect();
        luma.sort_by(f32::total_cmp);
        let median = luma[luma.len() / 2].max(1e-4);
        view.gain = (0.18 / median).clamp(0.25, 64.0);
        view
    }

    fn linear(&self, px: [f32; 3]) -> [f32; 3] {
        self.cam_to_srgb
            .map(|row| (row[0] * px[0] + row[1] * px[1] + row[2] * px[2]).max(0.0))
    }

    fn encode(&self, px: [f32; 3]) -> [u8; 3] {
        self.linear(px).map(|v| {
            let x = v * self.gain;
            let t = ((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)).clamp(0.0, 1.0);
            let s = if t <= 0.003_130_8 {
                12.92 * t
            } else {
                1.055 * t.powf(1.0 / 2.4) - 0.055
            };
            (s * 255.0 + 0.5).clamp(0.0, 255.0) as u8
        })
    }

    fn jpeg(&self, img: &LinearImage, path: &Path) {
        let rgb: Vec<u8> = img
            .data
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|px| self.encode(*px))
            .collect();
        let mut bytes = Vec::new();
        let mut encoder = Encoder::new(&mut bytes, 92);
        encoder.set_sampling_factor(SamplingFactor::R_4_4_4);
        encoder
            .encode(&rgb, img.width as u16, img.height as u16, ColorType::Rgb)
            .expect("encode JPEG");
        std::fs::write(path, bytes).expect("write JPEG");
    }
}

fn crop(img: &LinearImage, x0: usize, y0: usize, w: usize, h: usize) -> LinearImage {
    let (w, h) = (w.min(img.width - x0), h.min(img.height - y0));
    let mut data = Vec::with_capacity(w * h * 3);
    for y in y0..y0 + h {
        let start = 3 * (y * img.width + x0);
        data.extend_from_slice(&img.data[start..start + 3 * w]);
    }
    LinearImage {
        width: w,
        height: h,
        data,
    }
}

fn zoom(img: &LinearImage, factor: usize) -> LinearImage {
    let (w, h) = (img.width * factor, img.height * factor);
    let mut data = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let i = 3 * ((y / factor) * img.width + x / factor);
            data.extend_from_slice(&img.data[i..i + 3]);
        }
    }
    LinearImage {
        width: w,
        height: h,
        data,
    }
}

/// Top-left corner (in full-resolution pixels) of the `w` x `h` window with
/// the most fine detail, judged on the display image: where demosaic
/// artifacts would show.
fn detailed_window(
    display: &LinearImage,
    full_w: usize,
    full_h: usize,
    w: usize,
    h: usize,
) -> (usize, usize) {
    let s = display.width as f64 / full_w as f64;
    let (dw, dh) = ((w as f64 * s) as usize, (h as f64 * s) as usize);
    let luma = |x: usize, y: usize| {
        let i = 3 * (y * display.width + x);
        display.data[i] + 2.0 * display.data[i + 1] + display.data[i + 2]
    };
    let mut best = (0.0f32, 0, 0);
    for wy in (0..display.height.saturating_sub(dh + 1)).step_by(16) {
        for wx in (0..display.width.saturating_sub(dw + 1)).step_by(16) {
            let mut energy = 0.0;
            for y in (wy..wy + dh).step_by(2) {
                for x in (wx..wx + dw).step_by(2) {
                    let c = luma(x, y);
                    // Detail, not exposure: contrast relative to brightness.
                    energy +=
                        ((luma(x + 1, y) - c).abs() + (luma(x, y + 1) - c).abs()) / (c + 0.05);
                }
            }
            if energy > best.0 {
                best = (energy, wx, wy);
            }
        }
    }
    let x0 = ((best.1 as f64 / s) as usize).min(full_w.saturating_sub(w));
    let y0 = ((best.2 as f64 / s) as usize).min(full_h.saturating_sub(h));
    (x0, y0)
}

/// CPU area-average of `full` to `(w, h)`, the same box filter as the GPU.
fn box_down(full: &LinearImage, w: usize, h: usize) -> LinearImage {
    let sx = full.width as f64 / w as f64;
    let sy = full.height as f64 / h as f64;
    let mut data = vec![0.0f32; w * h * 3];
    for oy in 0..h {
        let (y0, y1) = (
            oy as f64 * sy,
            ((oy + 1) as f64 * sy).min(full.height as f64),
        );
        for ox in 0..w {
            let (x0, x1) = (
                ox as f64 * sx,
                ((ox + 1) as f64 * sx).min(full.width as f64),
            );
            let mut acc = [0.0f64; 3];
            let mut wsum = 0.0;
            for iy in y0.floor() as usize..(y1.ceil() as usize).min(full.height) {
                let wy = (iy as f64 + 1.0).min(y1) - (iy as f64).max(y0);
                for ix in x0.floor() as usize..(x1.ceil() as usize).min(full.width) {
                    let wx = (ix as f64 + 1.0).min(x1) - (ix as f64).max(x0);
                    let wgt = wx * wy;
                    if wgt <= 0.0 {
                        continue;
                    }
                    let i = 3 * (iy * full.width + ix);
                    for (a, &v) in acc.iter_mut().zip(&full.data[i..i + 3]) {
                        *a += wgt * f64::from(v);
                    }
                    wsum += wgt;
                }
            }
            for (o, a) in data[3 * (oy * w + ox)..].iter_mut().zip(acc) {
                *o = (a / wsum) as f32;
            }
        }
    }
    LinearImage {
        width: w,
        height: h,
        data,
    }
}

fn describe(info: &RawInfo) -> String {
    format!(
        "{} {} | {}x{} orient {} | ISO {:?} | {:?} s f/{:?} {:?} mm | {} | {}",
        info.make,
        info.model,
        info.width,
        info.height,
        info.orientation,
        info.iso,
        info.exposure_time_s,
        info.f_number,
        info.focal_length_mm,
        info.lens.as_deref().unwrap_or("-"),
        info.captured_at.as_deref().unwrap_or("-"),
    )
}

/// Decodes and develops every sample: prints metadata, white balance and
/// timings, writes the display render and a 1:1 crop of the demosaic, and
/// checks that the full-resolution image averages down to the display one.
#[test]
#[ignore = "needs the CUDA host and the sample RAWs"]
fn develop_every_sample() {
    let gpu = Gpu::open().expect("GPU");
    let started = Instant::now();
    Kernels::load(&gpu).expect("compile develop.cu");
    println!("NVRTC compile + load: {:.0} ms", ms(started));
    let out = out_dir();
    let mut failures = Vec::new();

    for path in samples() {
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let bytes = std::fs::read(&path).expect("read sample");
        let t = Instant::now();
        let frame = decode(&bytes).unwrap_or_else(|e| panic!("{name}: {e:#}"));
        let decode_ms = ms(t);
        let mp = (frame.width * frame.height) as f64 / 1e6;

        let t = Instant::now();
        let scene = develop_scene(&gpu, &frame).expect("develop_scene");
        let scene_cold_ms = ms(t);
        let t = Instant::now();
        let scene = develop_scene(&gpu, &frame)
            .map(|_| scene)
            .expect("develop_scene");
        let scene_ms = ms(t);
        let free_before = gpu.ctx.mem_get_info().map(|(free, _)| free).unwrap_or(0);
        let t = Instant::now();
        let full = develop_full(&gpu, &frame).expect("develop_full");
        let full_ms = ms(t);
        let t = Instant::now();
        let full = develop_full(&gpu, &frame)
            .map(|_| full)
            .expect("develop_full");
        let full_warm_ms = ms(t);
        gpu.stream.synchronize().expect("sync");
        let free_after = gpu.ctx.mem_get_info().map(|(free, _)| free).unwrap_or(0);

        println!("== {name}");
        println!("   {}", describe(&scene.info));
        println!(
            "   channels {} cfa {:?} black {:?} white {} wb {:.4?} -> {:.0} K tint {:+.1} | raw clipped {:.3}%",
            frame.channels,
            frame.cfa,
            frame.black,
            frame.white,
            frame.wb_as_shot,
            scene.info.as_shot_temperature,
            scene.info.as_shot_tint,
            scene.info.raw_clipped_fraction * 100.0,
        );
        println!(
            "   {mp:.1} MP: decode {decode_ms:.0} ms | develop_scene {scene_cold_ms:.0} ms cold, {scene_ms:.0} ms warm | develop_full {full_ms:.0} ms cold, {full_warm_ms:.0} ms warm | GPU free {} -> {} MB",
            free_before >> 20,
            free_after >> 20
        );

        let view = View::new(&scene.color, scene.info.as_shot_temperature, &scene.display);
        view.jpeg(&scene.display, &out.join(format!("{name}_display.jpg")));
        view.jpeg(
            &crop(
                &full,
                full.width.saturating_sub(600) / 2,
                full.height.saturating_sub(400) / 2,
                600,
                400,
            ),
            &out.join(format!("{name}_crop.jpg")),
        );
        let (cx, cy) = detailed_window(&scene.display, full.width, full.height, 600, 400);
        view.jpeg(
            &crop(&full, cx, cy, 600, 400),
            &out.join(format!("{name}_detail.jpg")),
        );
        // The middle of that crop at 4x, nearest neighbour: demosaic
        // artifacts (zipper, maze, false colour) are visible pixel by pixel.
        view.jpeg(
            &zoom(&crop(&full, cx + 225, cy + 150, 150, 100), 4),
            &out.join(format!("{name}_zoom.jpg")),
        );

        if scene.display.width.max(scene.display.height) != DISPLAY_EDGE
            || scene.search.width.max(scene.search.height) != SEARCH_EDGE
            || (full.width, full.height) != (frame.width, frame.height)
        {
            failures.push(format!("{name}: wrong output size"));
        }
        for img in [&scene.display, &scene.search, &full] {
            if !img.data.iter().all(|v| (0.0..=1.0).contains(v)) {
                failures.push(format!("{name}: values outside 0..1"));
            }
        }

        // The demosaic and the superpixel path must agree once both are at
        // display size: same normalization, white balance and clip.
        let down = box_down(&full, scene.display.width, scene.display.height);
        let mut bias = [0.0f64; 3];
        let mut abs = [[0.0f64; 3]; 2];
        let mut count = [0usize; 2];
        for (a, b) in down
            .data
            .as_chunks::<3>()
            .0
            .iter()
            .zip(scene.display.data.as_chunks::<3>().0)
        {
            let region = usize::from(b.iter().any(|&v| v >= 0.98));
            count[region] += 1;
            for c in 0..3 {
                let d = f64::from(a[c] - b[c]);
                bias[c] += d;
                abs[region][c] += d.abs();
            }
        }
        let n = (down.width * down.height) as f64;
        let per = |sums: [f64; 3], k: usize| sums.map(|v| v / k.max(1) as f64);
        let (open, near_clip) = (per(abs[0], count[0]), per(abs[1], count[1]));
        let bias = bias.map(|v| v / n);
        println!(
            "   full vs display: bias {bias:+.5?} | mean |diff| unclipped {open:.5?} ({:.1}%), near clip {near_clip:.5?}",
            count[0] as f64 * 100.0 / n
        );
        // A bias would mean the paths normalize, balance or clip differently;
        // per-pixel differences are expected where the superpixel and RCD
        // resolve fine texture differently (bare branches against sky: ~0.01).
        if bias.iter().any(|b| b.abs() > 0.002) || open.iter().any(|d| *d > 0.02) {
            failures.push(format!("{name}: full and display disagree"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Every sample's as-shot white balance read as Lightroom temperature/tint,
/// and the round trip back to multipliers.
#[test]
#[ignore = "needs the sample RAWs"]
fn as_shot_temperature_of_every_sample() {
    for path in samples() {
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let frame = decode(&std::fs::read(&path).unwrap()).unwrap();
        let (t, tint) = (frame.info.as_shot_temperature, frame.info.as_shot_tint);
        let back = frame.color.wb_for(t, tint);
        println!(
            "{name:70} wb {:.4?} -> {t:6.0} K {tint:+6.1} -> wb {:.4?} (A matrix: {})",
            frame.wb_as_shot,
            back,
            frame.color.xyz_to_cam_a.is_some()
        );
        assert!((2500.0..9000.0).contains(&t), "{name}: {t} K");
        assert!(tint.abs() < 50.0, "{name}: tint {tint}");
        for (a, b) in back.iter().zip(&frame.wb_as_shot) {
            assert!(
                (a - b).abs() / b < 1e-3,
                "{name}: {back:?} vs {:?}",
                frame.wb_as_shot
            );
        }
    }
}

/// A synthetic frame: `value(x, y, colour)` in white-balanced units (1 =
/// clip), turned into raw samples through the frame's own levels.
fn synthetic(
    width: usize,
    height: usize,
    channels: usize,
    value: impl Fn(usize, usize, usize) -> f32,
) -> RawFrame {
    let wb = [2.2, 1.0, 1.6];
    let (black, white) = (512.0f32, 15360.0f32);
    let cfa = [0u8, 1, 1, 2];
    let mut data = Vec::with_capacity(width * height * channels);
    for y in 0..height {
        for x in 0..width {
            for c in 0..channels {
                let color = if channels == 1 {
                    usize::from(cfa[(y & 1) * 2 + (x & 1)])
                } else {
                    c
                };
                let v = black + value(x, y, color) / wb[color] * (white - black);
                data.push(v.round().clamp(0.0, 16383.0) as u16);
            }
        }
    }
    let color = ColorModel::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], None);
    RawFrame {
        info: RawInfo {
            make: "Test".into(),
            model: "Synthetic".into(),
            lens: None,
            width: width as u32,
            height: height as u32,
            orientation: 1,
            iso: None,
            exposure_time_s: None,
            f_number: None,
            focal_length_mm: None,
            captured_at: None,
            gps: None,
            place: None,
            as_shot_temperature: 5000.0,
            as_shot_tint: 0.0,
            raw_clipped_fraction: 0.0,
        },
        color,
        wb_as_shot: wb,
        width,
        height,
        channels,
        cfa,
        data,
        black: [black; 4],
        white,
    }
}

/// Mean and largest max-minus-min over the channels of each pixel.
fn channel_spread(img: &LinearImage) -> (f32, f32) {
    let spreads: Vec<f32> = img
        .data
        .as_chunks::<3>()
        .0
        .iter()
        .map(|[r, g, b]| r.max(*g).max(*b) - r.min(*g).min(*b))
        .collect();
    let mean = spreads.iter().sum::<f32>() / spreads.len() as f32;
    (mean, spreads.iter().copied().fold(0.0, f32::max))
}

/// Grey stays grey, clipping is exactly 1.0 and neutral, edges of a neutral
/// scene get no colour fringes, and the clipped-pixel count is right.
#[test]
#[ignore = "needs the CUDA host"]
fn synthetic_frames_develop_exactly() {
    let gpu = Gpu::open().expect("GPU");

    let flat = synthetic(400, 300, 1, |_, _, _| 0.25);
    let scene = develop_scene(&gpu, &flat).unwrap();
    let full = develop_full(&gpu, &flat).unwrap();
    for img in [&scene.display, &scene.search, &full] {
        assert!(
            img.data.iter().all(|v| (v - 0.25).abs() < 2e-4),
            "flat grey drifted"
        );
    }
    assert_eq!(scene.info.raw_clipped_fraction, 0.0);

    // Overexposed in every channel by different amounts: exactly white.
    let blown = synthetic(400, 300, 1, |_, _, c| 1.2 + 0.3 * c as f32);
    let scene = develop_scene(&gpu, &blown).unwrap();
    let full = develop_full(&gpu, &blown).unwrap();
    for img in [&scene.display, &scene.search, &full] {
        assert!(
            img.data.iter().all(|&v| v == 1.0),
            "clipped pixels must be exactly 1.0"
        );
    }
    assert_eq!(scene.info.raw_clipped_fraction, 1.0);

    // Only red clipped in the sensor (a bright red light): red is 1, the
    // rest keeps its value.
    let red = synthetic(64, 64, 1, |_, _, c| if c == 0 { 3.0 } else { 0.3 });
    let full = develop_full(&gpu, &red).unwrap();
    assert!(
        full.data
            .as_chunks::<3>()
            .0
            .iter()
            .all(|px| (px[0] - 1.0).abs() < 1e-5 && (px[1] - 0.3).abs() < 1e-3),
        "a clipped red channel must stay at 1.0 without touching green"
    );

    // Neutral edges as a lens delivers them (about two pixels wide), at
    // vertical, diagonal and fine-line frequencies: RCD must not invent colour.
    let soft = |t: f32| {
        let t = t.clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    let edges = synthetic(256, 256, 1, |x, y, _| {
        let (x, y) = (x as f32, y as f32);
        let step = 0.05 + 0.55 * soft((x - 126.0) / 4.0);
        let diagonal = 0.2 * soft((x + y - 254.0) / 4.0);
        let band = soft((y - 60.0) / 4.0) * soft((100.0 - y) / 4.0);
        let lines = 0.1 * band * (1.0 + (x * std::f32::consts::TAU / 6.0).cos());
        step + diagonal + lines
    });
    let full = develop_full(&gpu, &edges).unwrap();
    let (mean, max) = channel_spread(&full);
    println!("channel spread on soft neutral edges: mean {mean:.5}, max {max:.4}");
    assert!(
        mean < 0.001 && max < 0.06,
        "colour fringes on a neutral scene: {mean} {max}"
    );

    // Linear (three-channel) frames take the same transform.
    let linear = synthetic(300, 200, 3, |x, _, _| if x < 150 { 0.4 } else { 1.5 });
    let scene = develop_scene(&gpu, &linear).unwrap();
    let full = develop_full(&gpu, &linear).unwrap();
    assert!(full.data[..3].iter().all(|v| (v - 0.4).abs() < 1e-3));
    assert!(full.data[full.data.len() - 3..].iter().all(|&v| v == 1.0));
    assert!((scene.info.raw_clipped_fraction - 0.5).abs() < 0.01);
}
