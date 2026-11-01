//! Tuning harness: every sample RAW through the whole pipeline on the GPU
//! host (decode, develop, renderer, auto and the four built-in looks at the
//! production budget), one contact sheet per photo next to the camera's own
//! JPEG. Plain `cargo test` skips it; run it with
//! `gpu <checkout> env SPL_TUNE_ROUND=<round> <test binary> tune_every_sample --ignored --nocapture`.
//!
//! Writes `/out/tune/<round>/<photo>.jpg` (camera JPEG, neutral, auto and
//! each look's best, labelled with the sliders and the score) and
//! `<photo>.txt` (every score term and statistic), and prints per-photo
//! timings. It ends with the brightness calibration behind `BASELINE_EV`:
//! the exposure at which the neutral render's median luma matches the
//! camera JPEG's, per photo and as the median over all of them.
//!
//! `SPL_TUNE_ONLY` (comma-separated substrings of file names) limits the
//! photos, `SPL_TUNE_EVALS` changes the evaluations per look, and
//! `SPL_RAW_SAMPLES` / `SPL_TUNE_OUT` move the inputs and outputs.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rawler::decoders::RawDecodeParams;
use rawler::rawsource::RawSource;

use super::{Candidate, Hues, SearchConfig, evaluate, hues, priors, run, warm_shift};
use crate::auto::auto_params;
use crate::decode::decode;
use crate::develop::develop_scene;
use crate::encode;
use crate::objective::{LookSpec, Score};
use crate::render::RenderEngine;
use crate::types::{DISPLAY_EDGE, Gpu, Params, Rgb8Image, Stats};

/// Long edge of every image on a sheet: a third of the display render.
const CELL: usize = DISPLAY_EDGE / 3;
const COLUMNS: usize = 4;
const GAP: usize = 8;
/// 5x7 glyphs drawn at this scale with one pixel of spacing.
const TEXT_SCALE: usize = 2;
const LINE_H: usize = 9 * TEXT_SCALE;
const LABEL_LINES: usize = 5;
const BG: u8 = 24;
const TEXT: [u8; 3] = [225, 225, 225];
const TITLE: [u8; 3] = [255, 205, 80];
const ALERT: [u8; 3] = [255, 95, 95];

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn samples() -> Vec<PathBuf> {
    let dir = PathBuf::from(env_or("SPL_RAW_SAMPLES", "/samples"));
    let only = env_or("SPL_TUNE_ONLY", "");
    let wanted: Vec<&str> = only.split(',').filter(|s| !s.is_empty()).collect();
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("sample directory")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("arw") || e.eq_ignore_ascii_case("dng"))
        })
        .filter(|p| {
            let name = p.to_string_lossy();
            wanted.is_empty() || wanted.iter().any(|w| name.contains(w))
        })
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no RAW samples in {}", dir.display());
    files
}

/// The JPEG the camera embedded in the RAW (full size when the format has
/// one, else the preview), decoded.
fn camera_jpeg(bytes: &[u8]) -> Option<Rgb8Image> {
    let src = RawSource::new_from_slice(bytes);
    let decoder = rawler::get_decoder(&src).ok()?;
    let params = RawDecodeParams::default();
    let img = decoder
        .full_image(&src, &params)
        .ok()
        .flatten()
        .or_else(|| decoder.preview_image(&src, &params).ok().flatten())?;
    let rgb = img.to_rgb8();
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);
    Some(Rgb8Image {
        width,
        height,
        data: rgb.into_raw(),
    })
}

/// What the stats kernel measures, measured the same way on any 8-bit sRGB
/// image (the camera JPEG): Rec.709 luma percentiles and mean (0 to 1),
/// Hasler and Süsstrunk colourfulness, and the share of pixels with a
/// channel at 254 or more.
#[derive(Debug, Clone, Copy)]
struct Measured {
    p01: f32,
    p05: f32,
    p50: f32,
    p95: f32,
    p99: f32,
    mean: f32,
    colorfulness: f32,
    clip: f32,
}

impl Measured {
    fn of(img: &Rgb8Image) -> Self {
        let mut hist = vec![0u64; 1024];
        let (mut sum, mut n, mut clip) = (0.0f64, 0u64, 0u64);
        let (mut rg, mut yb, mut rg2, mut yb2) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for px in img.data.as_chunks::<3>().0 {
            let [r, g, b] = px.map(f32::from);
            let y = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0;
            hist[((y * 1024.0) as usize).min(1023)] += 1;
            sum += f64::from(y);
            n += 1;
            clip += u64::from(px.iter().any(|&v| v >= 254));
            let (a, c) = (f64::from(r - g), f64::from(0.5 * (r + g) - b));
            rg += a;
            yb += c;
            rg2 += a * a;
            yb2 += c * c;
        }
        let n_f = n.max(1) as f64;
        let quantile = |q: f64| {
            let target = q * n_f;
            let mut below = 0.0;
            for (i, &c) in hist.iter().enumerate() {
                let c = c as f64;
                if c > 0.0 && below + c >= target {
                    return ((i as f64 + ((target - below) / c).clamp(0.0, 1.0)) / 1024.0) as f32;
                }
                below += c;
            }
            1.0
        };
        let (mrg, myb) = (rg / n_f, yb / n_f);
        let sigma = ((rg2 / n_f - mrg * mrg).max(0.0) + (yb2 / n_f - myb * myb).max(0.0)).sqrt();
        let mu = (mrg * mrg + myb * myb).sqrt();
        Self {
            p01: quantile(0.01),
            p05: quantile(0.05),
            p50: quantile(0.5),
            p95: quantile(0.95),
            p99: quantile(0.99),
            mean: (sum / n_f) as f32,
            colorfulness: (sigma + 0.3 * mu) as f32,
            clip: (clip as f64 / n_f) as f32,
        }
    }

    fn line(&self) -> String {
        format!(
            "p01 {:.3} p05 {:.3} p50 {:.3} p95 {:.3} p99 {:.3} mean {:.3} | col {:.1} | clip {:.4}",
            self.p01,
            self.p05,
            self.p50,
            self.p95,
            self.p99,
            self.mean,
            self.colorfulness,
            self.clip
        )
    }
}

/// Mean absolute difference of two same-size 8-bit images, in 8-bit levels:
/// how visibly two renders differ.
fn difference(a: &Rgb8Image, b: &Rgb8Image) -> f32 {
    if a.data.len() != b.data.len() || a.data.is_empty() {
        return f32::NAN;
    }
    let sum: u64 = a
        .data
        .iter()
        .zip(&b.data)
        .map(|(x, y)| u64::from(x.abs_diff(*y)))
        .sum();
    sum as f32 / a.data.len() as f32
}

/// Exposure (EV on top of the current `BASELINE_EV`) at which the neutral
/// render's median luma equals `target`, from a sweep of the stats kernel.
fn match_exposure(engine: &mut RenderEngine, neutral: &Params, target: f32) -> f32 {
    let sweep: Vec<Params> = (-120..=120)
        .map(|i| Params {
            exposure: i as f32 * 0.025,
            ..*neutral
        })
        .collect();
    let stats = engine.stats(&sweep).expect("exposure sweep");
    let p50: Vec<f32> = stats.iter().map(|s| s.luma_p50).collect();
    match p50.iter().position(|&v| v >= target) {
        Some(0) => sweep[0].exposure,
        Some(i) => {
            let (a, b) = (p50[i - 1], p50[i]);
            let t = if b > a { (target - a) / (b - a) } else { 0.5 };
            sweep[i - 1].exposure + t * (sweep[i].exposure - sweep[i - 1].exposure)
        }
        None => sweep[sweep.len() - 1].exposure,
    }
}

/// One photo's numbers for the closing summary.
struct Row {
    name: String,
    /// Camera median luma, neutral median luma, EV that matches them.
    calibration: Option<(f32, f32, f32)>,
    camera: Option<Measured>,
    neutral: Stats,
    search_ms: f64,
    evaluations: usize,
    looks: Vec<LookRow>,
}

/// One look's result on one photo.
struct LookRow {
    id: String,
    total: f32,
    feasible: bool,
    stats: Stats,
    /// Mean difference from the natural render, in 8-bit levels.
    from_natural: f32,
}

#[test]
#[ignore = "needs the CUDA host and the sample RAWs"]
fn tune_every_sample() {
    let round = env_or("SPL_TUNE_ROUND", "latest");
    let out = PathBuf::from(env_or("SPL_TUNE_OUT", "/out/tune")).join(&round);
    std::fs::create_dir_all(&out).expect("create output directory");
    let evals: usize = env_or("SPL_TUNE_EVALS", "2000")
        .parse()
        .expect("SPL_TUNE_EVALS");
    let cfg = SearchConfig {
        evals_per_look: evals,
        ..SearchConfig::default()
    };
    let gpu = Gpu::open().expect("GPU");
    println!(
        "round {round}: BASELINE_EV {}, {evals} evals per look -> {}",
        crate::render::BASELINE_EV,
        out.display()
    );
    let rows: Vec<Row> = samples()
        .iter()
        .map(|path| tune_one(&gpu, path, &cfg, &out))
        .collect();
    summary(&rows);
}

fn tune_one(gpu: &Gpu, path: &Path, cfg: &SearchConfig, out: &Path) -> Row {
    let name = path.file_stem().unwrap().to_string_lossy().to_string();
    let bytes = std::fs::read(path).expect("read sample");
    let t = Instant::now();
    let frame = decode(&bytes).unwrap_or_else(|e| panic!("{name}: {e:#}"));
    let decode_ms = ms(t);
    let t = Instant::now();
    let scene = develop_scene(gpu, &frame).unwrap_or_else(|e| panic!("{name}: {e:#}"));
    let develop_ms = ms(t);
    drop(frame);
    let t = Instant::now();
    let mut engine = RenderEngine::new(gpu, &scene).expect("render engine");
    let engine_ms = ms(t);
    let info = scene.info.clone();
    let neutral = Params::neutral(info.as_shot_temperature, info.as_shot_tint);
    let (neutral_stats, _) = evaluate(&mut engine, &scene, &neutral).expect("evaluate neutral");
    let auto = auto_params(&neutral_stats, &scene);
    let t = Instant::now();
    let result = run(
        &mut engine,
        &scene,
        &LookSpec::builtin(),
        cfg,
        &mut |_, _| {},
    )
    .expect("search");
    let search_ms = ms(t);
    let rate = result.evaluations as f64 / (search_ms / 1e3);
    // The search measures neutral at the quantized as-shot white balance,
    // so its auto may differ from this one by a rounding step.
    if result.auto.params != auto.quantized() {
        println!(
            "   note: search auto {:?} vs auto_params {:?}",
            result.auto.params,
            auto.quantized()
        );
    }

    let camera = camera_jpeg(&bytes);
    let measured = camera.as_ref().map(Measured::of);
    let calibration = measured.map(|m| {
        let ev = match_exposure(&mut engine, &neutral, m.p50);
        (m.p50, neutral_stats.luma_p50, ev)
    });

    println!(
        "== {name}: {} {} ISO {:?} {:?} s f/{:?} | as shot {:.0} K {:+.0} | raw clipped {:.2}%",
        info.make,
        info.model,
        info.iso,
        info.exposure_time_s,
        info.f_number,
        info.as_shot_temperature,
        info.as_shot_tint,
        100.0 * info.raw_clipped_fraction
    );
    println!(
        "   decode {decode_ms:.0} ms, develop {develop_ms:.0} ms, engine {engine_ms:.0} ms | search {search_ms:.0} ms, {} evaluations, {rate:.0} evals/s",
        result.evaluations
    );
    println!("   colours {}", colours(&Hues::of(&scene)));
    if let (Some((_, _, ev)), Some(m)) = (calibration, measured) {
        println!("   camera  {} | neutral matches at {ev:+.2} EV", m.line());
    } else {
        println!("   no camera JPEG");
    }

    let as_shot = (info.as_shot_temperature, info.as_shot_tint);
    let mut cells = Vec::new();
    let mut txt = String::new();
    if let Some(m) = measured {
        let _ = writeln!(txt, "camera JPEG: {}\n", m.line());
    }
    let upright = |img: &Rgb8Image| encode::resize(&encode::orient(img, info.orientation), CELL);
    if let (Some(cam), Some(m)) = (&camera, measured) {
        // The preview is stored like the sensor data; rotate it the same way
        // unless it is already upright.
        let stored_landscape = info.width >= info.height;
        let img = if (cam.width >= cam.height) == stored_landscape {
            upright(cam)
        } else {
            encode::resize(cam, CELL)
        };
        let ev = calibration.map_or(f32::NAN, |c| c.2);
        cells.push(Cell {
            img: Some(img),
            lines: vec![
                ("CAMERA JPEG".into(), TITLE),
                (format!("P50 {:.3} MEAN {:.3}", m.p50, m.mean), TEXT),
                (format!("NEUTRAL MATCHES AT {ev:+.2} EV"), TEXT),
                (format!("P01 {:.3} P99 {:.3}", m.p01, m.p99), TEXT),
                (
                    format!("COL {:.0} CLIP {:.2}%", m.colorfulness, 100.0 * m.clip),
                    TEXT,
                ),
            ],
        });
    }
    let candidates: Vec<&Candidate> = [&result.neutral, &result.auto]
        .into_iter()
        .chain(&result.looks)
        .collect();
    let mut renders = Vec::new();
    for c in &candidates {
        let img = upright(&engine.display(&c.params).expect("display render"));
        cells.push(Cell {
            img: Some(img.clone()),
            lines: label(&c.label, &c.params, &c.stats, &c.score, as_shot),
        });
        renders.push(img);
        describe(&mut txt, c, as_shot);
        println!("   {}", one_line(c, as_shot));
        println!("   {:14}   + {}", "", settings(&c.params));
    }
    // Renders: neutral, auto, then the looks in order (natural first).
    let natural_at = 2;
    let distances: Vec<f32> = (natural_at..renders.len())
        .map(|i| difference(&renders[i], &renders[natural_at]))
        .collect();
    println!(
        "   difference from natural (8-bit levels): auto {:.1} | {}",
        difference(&renders[1], &renders[natural_at]),
        result
            .looks
            .iter()
            .zip(&distances)
            .map(|(c, d)| format!("{} {d:.1}", c.id))
            .collect::<Vec<_>>()
            .join(" ")
    );
    let mut info_lines = vec![
        (name.chars().take(40).collect::<String>(), TITLE),
        (format!("{} {}", info.make, info.model), TEXT),
        (
            format!(
                "ISO {} F{:.1} {}",
                info.iso.unwrap_or(0),
                info.f_number.unwrap_or(0.0),
                info.exposure_time_s.map_or("-".into(), |s| if s < 1.0 {
                    format!("1/{:.0}S", 1.0 / s)
                } else {
                    format!("{s:.1}S")
                })
            ),
            TEXT,
        ),
        (
            format!("AS SHOT {:.0}K TINT {:+.0}", as_shot.0, as_shot.1),
            TEXT,
        ),
        (
            format!("RAW CLIPPED {:.2}%", 100.0 * info.raw_clipped_fraction),
            TEXT,
        ),
        (
            format!("SEARCH {:.0} MS {} EVALS", search_ms, result.evaluations),
            TEXT,
        ),
        (format!("{rate:.0} EVALS/S"), TEXT),
        (
            format!("BASELINE EV {:.2}", crate::render::BASELINE_EV),
            TEXT,
        ),
    ];
    if let Some((cam, neu, ev)) = calibration {
        info_lines.push((format!("P50 CAM {cam:.3} NEUTRAL {neu:.3}"), TEXT));
        info_lines.push((format!("MATCH {ev:+.2} EV"), TEXT));
    }
    cells.push(Cell {
        img: None,
        lines: info_lines,
    });
    let sheet = compose(&cells);
    let jpeg = encode::jpeg(&sheet, 90).expect("encode sheet");
    std::fs::write(out.join(format!("{name}.jpg")), jpeg).expect("write sheet");
    let header = format!(
        "{name}\n{} {} ISO {:?} | as shot {:.0} K {:+.1} | raw clipped {:.3}%\nsearch {search_ms:.0} ms, {} evaluations, {rate:.0}/s\n\n",
        info.make,
        info.model,
        info.iso,
        as_shot.0,
        as_shot.1,
        100.0 * info.raw_clipped_fraction,
        result.evaluations
    );
    std::fs::write(out.join(format!("{name}.txt")), header + &txt).expect("write text");

    Row {
        name,
        calibration,
        camera: measured,
        neutral: neutral_stats,
        search_ms,
        evaluations: result.evaluations,
        looks: result
            .looks
            .iter()
            .zip(&distances)
            .map(|(c, d)| LookRow {
                id: c.id.clone(),
                total: c.score.total,
                feasible: c.score.feasible,
                stats: c.stats.clone(),
                from_natural: *d,
            })
            .collect(),
    }
}

fn median(mut v: Vec<f32>) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    v.sort_by(f32::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn summary(rows: &[Row]) {
    println!("\n== summary over {} photos", rows.len());
    println!(
        "{:58} {:>7} {:>7} {:>7} {:>8} {:>7} {:>6}",
        "photo", "cam p50", "neu p50", "ratio", "match EV", "search", "ev/s"
    );
    for r in rows {
        let (cam, neu, ev) = r.calibration.unwrap_or((f32::NAN, f32::NAN, f32::NAN));
        println!(
            "{:58} {cam:7.3} {neu:7.3} {:7.3} {ev:+8.2} {:5.0}ms {:6.0}",
            r.name.chars().take(58).collect::<String>(),
            neu / cam,
            r.search_ms,
            r.evaluations as f64 / (r.search_ms / 1e3)
        );
    }
    let calibrated: Vec<&(f32, f32, f32)> =
        rows.iter().filter_map(|r| r.calibration.as_ref()).collect();
    let p50_ratio = median(calibrated.iter().map(|c| c.1 / c.0).collect());
    let mean_ratio = median(
        rows.iter()
            .filter_map(|r| r.camera.map(|c| r.neutral.luma_mean / c.mean))
            .collect(),
    );
    let ev = median(calibrated.iter().map(|c| c.2).collect());
    println!(
        "median luma ratio neutral/camera: p50 {p50_ratio:.3}, mean {mean_ratio:.3}; median matching exposure {ev:+.3} EV -> BASELINE_EV {:.3} would match",
        crate::render::BASELINE_EV + ev
    );

    // Each look against the camera JPEG and against natural, per photo.
    println!(
        "\n{:40} {:>23} | {:>23} | {:>23} | {:>23} | {:>23}",
        "p50 / p99 / colourfulness", "camera", "natural", "vivid", "bright", "moody"
    );
    for r in rows {
        let cam = r.camera.map_or(String::from("-"), |m| {
            format!("{:.2} {:.2} {:5.1}", m.p50, m.p99, m.colorfulness)
        });
        let looks: Vec<String> = r
            .looks
            .iter()
            .map(|l| {
                format!(
                    "{:.2} {:.2} {:5.1} d{:4.1}{}",
                    l.stats.luma_p50,
                    l.stats.luma_p99,
                    l.stats.colorfulness,
                    l.from_natural,
                    if l.feasible { " " } else { "X" }
                )
            })
            .collect();
        println!(
            "{:40} {cam:>23} | {}",
            r.name.chars().take(40).collect::<String>(),
            looks.join(" | ")
        );
    }
    let look_median = |id: &str, f: &dyn Fn(&LookRow) -> f32| {
        median(
            rows.iter()
                .flat_map(|r| r.looks.iter().filter(|l| l.id == id).map(f))
                .collect(),
        )
    };
    for id in ["look:natural", "look:vivid", "look:bright", "look:moody"] {
        println!(
            "{id:13} median p50 {:.3} p01 {:.3} p99 {:.3} colourfulness {:5.1} | difference from natural {:4.1} levels",
            look_median(id, &|l| l.stats.luma_p50),
            look_median(id, &|l| l.stats.luma_p01),
            look_median(id, &|l| l.stats.luma_p99),
            look_median(id, &|l| l.stats.colorfulness),
            look_median(id, &|l| l.from_natural)
        );
    }
    let cams: Vec<Measured> = rows.iter().filter_map(|r| r.camera).collect();
    println!(
        "camera JPEG   median p50 {:.3} p01 {:.3} p99 {:.3} colourfulness {:5.1}",
        median(cams.iter().map(|m| m.p50).collect()),
        median(cams.iter().map(|m| m.p01).collect()),
        median(cams.iter().map(|m| m.p99).collect()),
        median(cams.iter().map(|m| m.colorfulness).collect())
    );
    let times: Vec<f32> = rows.iter().map(|r| r.search_ms as f32).collect();
    let rates: Vec<f32> = rows
        .iter()
        .map(|r| (r.evaluations as f64 / (r.search_ms / 1e3)) as f32)
        .collect();
    println!(
        "search per photo: median {:.0} ms, max {:.0} ms; evaluations/s median {:.0}, min {:.0}",
        median(times.clone()),
        times.iter().copied().fold(0.0, f32::max),
        median(rates.clone()),
        rates.iter().copied().fold(f32::INFINITY, f32::min)
    );
    let infeasible: Vec<String> = rows
        .iter()
        .flat_map(|r| {
            r.looks
                .iter()
                .filter(|l| !l.feasible)
                .map(move |l| format!("{} {}", r.name, l.id))
        })
        .collect();
    println!("infeasible looks: {infeasible:?}");
    let mean_score = |id: &str| {
        let v: Vec<f32> = rows
            .iter()
            .flat_map(|r| r.looks.iter().filter(|l| l.id == id).map(|l| l.total))
            .collect();
        v.iter().sum::<f32>() / v.len().max(1) as f32
    };
    println!(
        "mean look scores: natural {:.2} vivid {:.2} bright {:.2} moody {:.2}",
        mean_score("look:natural"),
        mean_score("look:vivid"),
        mean_score("look:bright"),
        mean_score("look:moody")
    );
}

/// Sliders relative to as-shot white balance, compactly.
fn sliders(p: &Params, as_shot: (f32, f32)) -> String {
    format!(
        "wb {:+.0} mired tint {:+.0} | exp {:+.2} con {:+.0} hl {:+.0} sh {:+.0} wh {:+.0} bl {:+.0} vib {:+.0} sat {:+.0}",
        warm_shift(as_shot.0, p.temperature),
        p.tint - as_shot.1,
        p.exposure,
        p.contrast,
        p.highlights,
        p.shadows,
        p.whites,
        p.blacks,
        p.vibrance,
        p.saturation
    )
}

/// The settings beyond the Basic panel that are not zero, with short names.
fn settings(p: &Params) -> String {
    let a = p.to_array();
    let short = |name: &str| {
        name.replace("curve_", "curve.")
            .replace("saturation_", "sat.")
            .replace("luminance_", "lum.")
            .replace("hue_", "hue.")
            .replace("_primary_saturation", "P.sat")
            .replace("_primary_hue", "P.hue")
    };
    let parts: Vec<String> = (Params::BASIC..priors::END)
        .filter(|&i| a[i] != 0.0)
        .map(|i| format!("{} {:+.0}", short(Params::NAMES[i]), a[i]))
        .collect();
    if parts.is_empty() {
        "no other settings".into()
    } else {
        parts.join(" ")
    }
}

/// The colour profile: each band's share (and colourfulness) and skin.
fn colours(h: &Hues) -> String {
    const NAMES: [&str; hues::BANDS] = [
        "red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta",
    ];
    let bands: Vec<String> = (0..hues::BANDS)
        .filter(|&b| h.mass[b] >= 0.005)
        .map(|b| {
            let searched = if h.searched(b) { "" } else { " (fixed)" };
            format!(
                "{} {:.1}% C/L {:.2}{searched}",
                NAMES[b],
                100.0 * h.mass[b],
                h.chroma[b]
            )
        })
        .collect();
    format!("{} | skin {:.1}%", bands.join(", "), 100.0 * h.skin)
}

fn stats_line(s: &Stats) -> String {
    format!(
        "p01 {:.3} p05 {:.3} p50 {:.3} p95 {:.3} p99 {:.3} mean {:.3} | hi {:.4} new {:.4} lo {:.4} | col {:.1} C95 {:.3} oog {:.4} cast {:+.4} {:+.4} | struct {:.3} lc {:.4}",
        s.luma_p01,
        s.luma_p05,
        s.luma_p50,
        s.luma_p95,
        s.luma_p99,
        s.luma_mean,
        s.clip_hi,
        s.clip_hi_new,
        s.clip_lo,
        s.colorfulness,
        s.chroma_p95,
        s.out_of_gamut,
        s.cast_a,
        s.cast_b,
        s.structure,
        s.local_contrast
    )
}

fn one_line(c: &Candidate, as_shot: (f32, f32)) -> String {
    format!(
        "{:14} {:+7.2}{} | {} | p01 {:.3} p50 {:.3} p99 {:.3} new {:.4} lo {:.4} col {:.1} cast {:+.3} {:+.3}",
        c.id,
        c.score.total,
        if c.score.feasible { " " } else { "X" },
        sliders(&c.params, as_shot),
        c.stats.luma_p01,
        c.stats.luma_p50,
        c.stats.luma_p99,
        c.stats.clip_hi_new,
        c.stats.clip_lo,
        c.stats.colorfulness,
        c.stats.cast_a,
        c.stats.cast_b
    )
}

fn describe(txt: &mut String, c: &Candidate, as_shot: (f32, f32)) {
    let _ = writeln!(
        txt,
        "{} ({}, {}): total {:+.3}, feasible {}",
        c.id,
        c.label,
        c.source.as_str(),
        c.score.total,
        c.score.feasible
    );
    let _ = writeln!(txt, "  params {:?}", c.params);
    let _ = writeln!(txt, "  {}", sliders(&c.params, as_shot));
    let _ = writeln!(txt, "  {}", settings(&c.params));
    let _ = writeln!(txt, "  {}", stats_line(&c.stats));
    let terms: Vec<String> = c
        .score
        .terms
        .iter()
        .filter(|(_, v)| *v != 0.0)
        .map(|(n, v)| format!("{n} {v:+.3}"))
        .collect();
    let _ = writeln!(txt, "  terms: {}\n", terms.join(", "));
}

fn label(
    title: &str,
    p: &Params,
    s: &Stats,
    score: &Score,
    as_shot: (f32, f32),
) -> Vec<(String, [u8; 3])> {
    let flag = if score.feasible { "" } else { " INFEASIBLE" };
    vec![
        (
            format!("{} {:+.2}{flag}", title.to_uppercase(), score.total),
            if score.feasible { TITLE } else { ALERT },
        ),
        (
            format!(
                "EV{:+.2} CON{:+.0} HL{:+.0} SH{:+.0}",
                p.exposure, p.contrast, p.highlights, p.shadows
            ),
            TEXT,
        ),
        (
            format!(
                "WH{:+.0} BL{:+.0} VIB{:+.0} SAT{:+.0}",
                p.whites, p.blacks, p.vibrance, p.saturation
            ),
            TEXT,
        ),
        (
            format!(
                "WB{:+.0}M TN{:+.0} P50 {:.2} P99 {:.2}",
                warm_shift(as_shot.0, p.temperature),
                p.tint - as_shot.1,
                s.luma_p50,
                s.luma_p99
            ),
            TEXT,
        ),
        (
            format!(
                "P01 {:.3} NEW {:.2}% COL {:.0}",
                s.luma_p01,
                100.0 * s.clip_hi_new,
                s.colorfulness
            ),
            TEXT,
        ),
    ]
}

struct Cell {
    img: Option<Rgb8Image>,
    lines: Vec<(String, [u8; 3])>,
}

/// Cells in rows of [`COLUMNS`], each image with its label lines under it.
fn compose(cells: &[Cell]) -> Rgb8Image {
    let images = || cells.iter().filter_map(|c| c.img.as_ref());
    let cw = images().map(|i| i.width).max().unwrap_or(CELL).max(CELL);
    let ih = images().map(|i| i.height).max().unwrap_or(CELL);
    let ch = ih + LABEL_LINES * LINE_H + 6;
    let rows = cells.len().div_ceil(COLUMNS);
    let (w, h) = (COLUMNS * (cw + GAP) + GAP, rows * (ch + GAP) + GAP);
    let mut sheet = Rgb8Image {
        width: w,
        height: h,
        data: vec![BG; 3 * w * h],
    };
    for (k, cell) in cells.iter().enumerate() {
        let (ox, oy) = (
            GAP + (k % COLUMNS) * (cw + GAP),
            GAP + (k / COLUMNS) * (ch + GAP),
        );
        let text_y = match &cell.img {
            Some(img) => {
                for y in 0..img.height {
                    let src = &img.data[3 * y * img.width..3 * (y + 1) * img.width];
                    let d = 3 * ((oy + y) * w + ox);
                    sheet.data[d..d + src.len()].copy_from_slice(src);
                }
                oy + ih + 4
            }
            None => oy,
        };
        for (i, (text, color)) in cell.lines.iter().enumerate() {
            draw_text(&mut sheet, ox, text_y + i * LINE_H, text, *color, cw);
        }
    }
    sheet
}

fn draw_text(img: &mut Rgb8Image, x0: usize, y0: usize, text: &str, color: [u8; 3], max_w: usize) {
    for (k, ch) in text.chars().enumerate() {
        let x = x0 + k * 6 * TEXT_SCALE;
        if x + 5 * TEXT_SCALE > x0 + max_w {
            break;
        }
        for (row, bits) in glyph(ch).iter().enumerate() {
            for col in 0..5 {
                if (bits >> (4 - col)) & 1 == 0 {
                    continue;
                }
                for dy in 0..TEXT_SCALE {
                    for dx in 0..TEXT_SCALE {
                        let (px, py) = (x + col * TEXT_SCALE + dx, y0 + row * TEXT_SCALE + dy);
                        if px < img.width && py < img.height {
                            let i = 3 * (py * img.width + px);
                            img.data[i..i + 3].copy_from_slice(&color);
                        }
                    }
                }
            }
        }
    }
}

/// A classic 5x7 font: rows top to bottom, bit 4 is the leftmost column.
fn glyph(ch: char) -> [u8; 7] {
    match ch.to_ascii_uppercase() {
        '0' => [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
        '1' => [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
        '2' => [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F],
        '3' => [0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E],
        '4' => [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
        '5' => [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
        '6' => [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
        '7' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        '8' => [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
        '9' => [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
        'A' => [0x0E, 0x11, 0x11, 0x11, 0x1F, 0x11, 0x11],
        'B' => [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E],
        'C' => [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
        'D' => [0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C],
        'E' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
        'F' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
        'G' => [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F],
        'H' => [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'I' => [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
        'J' => [0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C],
        'K' => [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11],
        'L' => [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
        'M' => [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11],
        'N' => [0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11],
        'O' => [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'P' => [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
        'Q' => [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D],
        'R' => [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
        'S' => [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E],
        'T' => [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        'U' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'V' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04],
        'W' => [0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A],
        'X' => [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11],
        'Y' => [0x11, 0x11, 0x11, 0x0A, 0x04, 0x04, 0x04],
        'Z' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F],
        '+' => [0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00],
        '-' => [0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00],
        '.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C],
        ',' => [0x00, 0x00, 0x00, 0x00, 0x0C, 0x04, 0x08],
        ':' => [0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00],
        '/' => [0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x00],
        '%' => [0x18, 0x19, 0x02, 0x04, 0x08, 0x13, 0x03],
        '(' => [0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02],
        ')' => [0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08],
        '=' => [0x00, 0x00, 0x1F, 0x00, 0x1F, 0x00, 0x00],
        '_' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F],
        '<' => [0x02, 0x04, 0x08, 0x10, 0x08, 0x04, 0x02],
        '>' => [0x08, 0x04, 0x02, 0x01, 0x02, 0x04, 0x08],
        ' ' => [0; 7],
        _ => [0x0E, 0x11, 0x01, 0x02, 0x04, 0x00, 0x04],
    }
}
