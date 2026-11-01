//! Texture, Clarity and Dehaze. The CPU tests check the packing and the
//! scales; the GPU ones (`#[ignore]`, run on the GPU host with
//! `gpu-test <checkout> render::presence --ignored`) check each slider's
//! effect, identity at zero, finiteness, halos on a step edge, preview vs
//! export, and timing, and write -100/-50/0/+50/+100 strips for two samples
//! under /out/render-tone/ (or `$SPL_TONE_OUT`).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use super::super::testkit::{self, Sample};
use super::super::{CameraColor, RenderEngine, full};
use super::*;
use crate::types::{DISPLAY_EDGE, LinearImage, RawInfo, Rgb8Image, Rgb16Image, Scene, Stats};

/// Outdoor trees under an overcast sky (A7 III) and a frog in front of
/// bokeh (A6400).
pub(in crate::render) const SAMPLES: [&str; 2] =
    ["ILCE-7M3___DSC0009.ARW", "ILCE-6400__DSC00087.ARW"];
/// Where the detail crops sit in each sample (centre, as fractions).
const CROP_AT: [(f32, f32); 2] = [(0.5, 0.4), (0.22, 0.62)];
pub(in crate::render) const SWEEP: [f32; 5] = [-100.0, -50.0, 0.0, 50.0, 100.0];

pub(in crate::render) fn out_path(name: &str) -> PathBuf {
    let dir = std::env::var_os("SPL_TONE_OUT")
        .map_or_else(|| PathBuf::from("/out/render-tone"), PathBuf::from);
    std::fs::create_dir_all(&dir).expect("create output dir");
    dir.join(name)
}

pub(in crate::render) fn save(img: &Rgb8Image, name: &str) -> PathBuf {
    let path = out_path(name);
    let mut enc = jpeg_encoder::Encoder::new_file(&path, 90).expect("create jpeg");
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

pub(in crate::render) fn crop(
    img: &Rgb8Image,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
) -> Rgb8Image {
    let mut data = Vec::with_capacity(3 * w * h);
    for y in y0..y0 + h {
        data.extend_from_slice(&img.data[3 * (y * img.width + x0)..3 * (y * img.width + x0 + w)]);
    }
    Rgb8Image {
        width: w,
        height: h,
        data,
    }
}

/// An identity camera (Rec.2020 in, as-shot white balance = 1).
pub(in crate::render) struct Identity;

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

pub(in crate::render) fn image(
    w: usize,
    h: usize,
    f: impl Fn(usize, usize) -> [f32; 3],
) -> LinearImage {
    let mut data = Vec::with_capacity(3 * w * h);
    for y in 0..h {
        for x in 0..w {
            data.extend(f(x, y));
        }
    }
    LinearImage {
        width: w,
        height: h,
        data,
    }
}

/// A synthetic scene on the identity camera, `f` over normalized (x, y).
pub(in crate::render) fn scene(w: usize, h: usize, f: impl Fn(f32, f32) -> [f32; 3]) -> Scene {
    let at = |sw: usize, sh: usize| {
        image(sw, sh, |x, y| {
            f((x as f32 + 0.5) / sw as f32, (y as f32 + 0.5) / sh as f32)
        })
    };
    Scene {
        info: RawInfo {
            make: "test".into(),
            model: "synthetic".into(),
            lens: None,
            width: w as u32,
            height: h as u32,
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
        color: crate::color::ColorModel::new(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            None,
        ),
        wb_as_shot: [1.0; 3],
        search: at(w / 2, h / 2),
        display: at(w, h),
    }
}

pub(in crate::render) fn column_luma(img: &Rgb8Image) -> Vec<f64> {
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

pub(in crate::render) fn neutral(s: &Scene) -> Params {
    Params::neutral(s.info.as_shot_temperature, s.info.as_shot_tint)
}

pub(in crate::render) fn engine(gpu: &Gpu, s: &Sample) -> RenderEngine {
    RenderEngine::with_color(gpu, &s.scene, s.color.clone()).expect("engine")
}

pub(in crate::render) fn stats_finite(st: &Stats) -> bool {
    let v = serde_json::to_value(st).unwrap();
    v.as_object().unwrap().values().all(|x| match x {
        serde_json::Value::Number(n) => n.as_f64().is_some_and(f64::is_finite),
        serde_json::Value::Array(a) => a.iter().all(|n| n.as_f64().is_some_and(f64::is_finite)),
        _ => false,
    })
}

type Setter = fn(&mut Params, f32);

const SLIDERS: [(&str, Setter); 3] = [
    ("texture", |p, v| p.texture = v),
    ("clarity", |p, v| p.clarity = v),
    ("dehaze", |p, v| p.dehaze = v),
];

fn with(n: Params, f: Setter, v: f32) -> Params {
    let mut p = n;
    f(&mut p, v);
    p
}

#[test]
fn neutral_packs_off_and_sliders_scale() {
    let mut out = [9.0f32; P_PRES_N];
    let n = Params::neutral(5000.0, 0.0);
    pack(&n, &mut out);
    assert!(out.iter().all(|&v| v == 0.0));
    for (name, f) in SLIDERS {
        for v in SWEEP {
            pack(&with(n, f, v), &mut out);
            assert!(out.iter().all(|x| x.is_finite()), "{name} {v}");
            assert_eq!(out[0] != 0.0, v != 0.0, "{name} {v}: on flag");
        }
    }
    let at = |f: Setter, v: f32, k: usize| {
        let mut o = [0.0; P_PRES_N];
        pack(&with(n, f, v), &mut o);
        o[k]
    };
    assert_eq!(at(SLIDERS[0].1, 100.0, 1), TEXTURE_UP);
    assert_eq!(at(SLIDERS[0].1, -100.0, 1), -TEXTURE_DOWN);
    assert_eq!(at(SLIDERS[1].1, 50.0, 2), 0.5 * CLARITY_UP);
    assert_eq!(at(SLIDERS[2].1, 100.0, 3), DEHAZE_UNVEIL);
    assert_eq!(at(SLIDERS[2].1, -100.0, 3), 0.0);
    assert_eq!(at(SLIDERS[2].1, -100.0, 4), DEHAZE_FOG);
    // Garbage stays legal.
    pack(&with(n, SLIDERS[0].1, f32::NAN), &mut out);
    assert!(out.iter().all(|&v| v == 0.0));
}

#[test]
fn scales_follow_the_long_edge() {
    let (search, display, full) = (Scales::new(768), Scales::new(1536), Scales::new(6000));
    assert!(display.tex_rad >= 2 * search.tex_rad - 1 && display.tex_rad <= 2 * search.tex_rad + 1);
    assert_eq!(full.tex_rad, 24);
    assert_eq!(display.patch_rad, 15);
    assert!((full.fine_sigma - 2.4).abs() < 1e-5);
    // The export's strip halo covers Texture's support at 61 MP, and is
    // not grown when no presence slider is set.
    let n = Params::neutral(5000.0, 0.0);
    let textured = Params { texture: 1.0, ..n };
    let support = (3.0f32 * 0.0004 * 9504.0).ceil() as usize + 38;
    assert!(halo_rows(9504, 6336, &textured) >= support);
    assert_eq!(halo_rows(9504, 6336, &n), 0);
}

#[test]
fn kernel_constants_match() {
    let run = include_str!("../../../cuda/presence.cu")
        .lines()
        .find_map(|l| l.strip_prefix("#define PRES_RUN "))
        .expect("PRES_RUN in cuda/presence.cu");
    assert_eq!(run.trim().parse::<usize>().unwrap(), BOX_RUN);
}

#[test]
fn airlight_is_a_high_quantile() {
    let mut v: Vec<f32> = (0..10_000).map(|i| i as f32 / 10_000.0).collect();
    assert!((airlight(&v) - 0.999).abs() < 2e-3);
    v[5] = f32::NAN;
    assert!(airlight(&v).is_finite());
    assert_eq!(airlight(&[0.0; 16]), 1e-4);
}

/// Every slider at zero renders byte-identically to neutral, every setting
/// stays finite (statistics, display, export), and each slider moves the
/// statistic it should, monotonically: Texture and Clarity local contrast,
/// Dehaze the darkest tones down and colour up.
#[test]
#[ignore]
fn gpu_presence_effects() {
    let gpu = Gpu::open().expect("gpu");
    for file in SAMPLES {
        let s = testkit::load(file);
        let mut e = engine(&gpu, &s);
        let n = neutral(&s.scene);
        let base = e.display(&n).unwrap();
        println!("{file}");
        for (name, f) in SLIDERS {
            assert_eq!(
                e.display(&with(n, f, 0.0)).unwrap().data,
                base.data,
                "{name} 0"
            );
            let ps: Vec<Params> = SWEEP.iter().map(|&v| with(n, f, v)).collect();
            let st = e.stats(&ps).unwrap();
            for (v, x) in SWEEP.iter().zip(&st) {
                assert!(stats_finite(x), "{name} {v}: {x:?}");
                println!(
                    "  {name:>8} {v:+4}: lc {:.4} struct {:.3} p05 {:.3} p50 {:.3} p95 {:.3} chroma {:.4} col {:.1}",
                    x.local_contrast,
                    x.structure,
                    x.luma_p05,
                    x.luma_p50,
                    x.luma_p95,
                    x.chroma_mean,
                    x.colorfulness
                );
            }
            for w in st.windows(2) {
                match name {
                    "texture" | "clarity" => assert!(
                        w[1].local_contrast > w[0].local_contrast,
                        "{name}: local contrast must rise"
                    ),
                    _ => {
                        assert!(
                            w[1].luma_p05 < w[0].luma_p05,
                            "dehaze: darkest tones must fall"
                        );
                        assert!(
                            w[1].chroma_mean > w[0].chroma_mean,
                            "dehaze: colour must rise"
                        );
                    }
                }
            }
        }
        let all = Params {
            texture: 100.0,
            clarity: 100.0,
            dehaze: 100.0,
            ..n
        };
        let full = full::render(&gpu, &s.scene, &s.half, &all, &*s.color).unwrap();
        assert_eq!(full.data.len(), 3 * full.width * full.height);
        assert!(e.display(&all).is_ok());
        // The statistics measure the final image: the stats kernel and the
        // 8-bit search render agree with every new stage on.
        for p in [
            all,
            Params {
                texture: -60.0,
                clarity: 40.0,
                dehaze: -30.0,
                curve_shadows: 30.0,
                curve_lights: -40.0,
                curve_highlights: 60.0,
                ..n
            },
        ] {
            let st = e.stats(&[p]).unwrap().remove(0);
            let img = e.search(&p).unwrap();
            let mean = img
                .data
                .as_chunks::<3>()
                .0
                .iter()
                .map(|c| 0.2126 * c[0] as f64 + 0.7152 * c[1] as f64 + 0.0722 * c[2] as f64)
                .sum::<f64>()
                / (255.0 * (img.width * img.height) as f64);
            assert!(
                (mean - st.luma_mean as f64).abs() < 0.003,
                "stats {} vs render {mean}",
                st.luma_mean
            );
        }
    }
}

/// A 5 EV step edge: with each slider at +-100 the plateaus stay flat
/// beyond a few pixels of the edge (no halo), at display size and in the
/// export.
#[test]
#[ignore]
fn gpu_presence_no_halos() {
    let gpu = Gpu::open().expect("gpu");
    let (w, h) = (1536, 512);
    let dark = 0.18 / 2f32.powf(crate::render::BASELINE_EV) / 8.0;
    let s = scene(
        w,
        h,
        |x, _| if x < 0.5 { [dark; 3] } else { [dark * 32.0; 3] },
    );
    let full_img = image(2 * w, 2 * h, |x, _| {
        if x < w { [dark; 3] } else { [dark * 32.0; 3] }
    });
    let mut e = RenderEngine::with_color(&gpu, &s, Arc::new(Identity)).unwrap();
    let n = neutral(&s);
    let edge = w / 2;
    // Largest deviation of a column from its plateau (the column 300 px off
    // the edge), beyond `gap` pixels of the edge, in 8-bit levels.
    let halo = |cols: &[f64], edge: usize, gap: usize, far: usize| {
        let (left, right) = (cols[edge - far], cols[edge + far]);
        let l = (0..edge - gap).map(|x| (cols[x] - left).abs());
        let r = (edge + gap..cols.len()).map(|x| (cols[x] - right).abs());
        l.chain(r).fold(0.0f64, f64::max)
    };
    for (name, f) in SLIDERS {
        for v in [-100.0, 100.0] {
            let p = with(n, f, v);
            let cols = column_luma(&e.display(&p).unwrap());
            let d = halo(&cols, edge, 4, 300);
            let big = full::render_with(
                &gpu,
                &s,
                &full_img,
                &p,
                &Identity,
                full::Finish {
                    sharpen: 0.0,
                    chroma_nr: false,
                },
            )
            .unwrap();
            let cols16 = column_luma(&to8(&big));
            let d_full = halo(&cols16, 2 * edge, 8, 600);
            println!(
                "{name} {v:+}: plateaus {:.1} / {:.1}, halo {d:.2} levels (display), {d_full:.2} (export)",
                cols[edge - 300],
                cols[edge + 300]
            );
            assert!(d < 2.5 && d_full < 2.5, "{name} {v}: halo {d} / {d_full}");
        }
    }
}

pub(in crate::render) fn to8(img: &Rgb16Image) -> Rgb8Image {
    Rgb8Image {
        width: img.width,
        height: img.height,
        data: img
            .data
            .iter()
            .map(|&v| ((v as u32 + 128) / 257) as u8)
            .collect(),
    }
}

/// The export, downscaled to display size, matches display() with every
/// presence slider set (Texture and Clarity act at matching scales).
#[test]
#[ignore]
fn gpu_presence_full_matches_display() {
    let gpu = Gpu::open().expect("gpu");
    for file in SAMPLES {
        let s = testkit::load(file);
        let mut e = engine(&gpu, &s);
        // Detail off: display() would sharpen, the bare export below not.
        let n = crate::finish::Detail::from(crate::detail::DetailSettings::OFF)
            .apply(&neutral(&s.scene));
        for (label, p) in [
            (
                "texture+clarity",
                Params {
                    texture: 60.0,
                    clarity: 50.0,
                    ..n
                },
            ),
            ("dehaze", Params { dehaze: 60.0, ..n }),
            (
                "soft",
                Params {
                    texture: -60.0,
                    clarity: -50.0,
                    dehaze: -40.0,
                    ..n
                },
            ),
        ] {
            let disp = e.display(&p).unwrap();
            let bare = full::Finish {
                sharpen: 0.0,
                chroma_nr: false,
            };
            let big = full::render_with(&gpu, &s.scene, &s.half, &p, &*s.color, bare).unwrap();
            let small = downscale8(&big, DISPLAY_EDGE);
            assert_eq!((small.width, small.height), (disp.width, disp.height));
            let mad = small
                .data
                .iter()
                .zip(&disp.data)
                .map(|(&a, &b)| (a as f64 - b as f64).abs())
                .sum::<f64>()
                / disp.data.len() as f64;
            let signed = small
                .data
                .iter()
                .zip(&disp.data)
                .map(|(&a, &b)| a as f64 - b as f64)
                .sum::<f64>()
                / disp.data.len() as f64;
            println!("{file} {label}: export vs display MAD {mad:.2}/255, mean {signed:+.2}");
            assert!(
                mad < 3.0 && signed.abs() < 1.0,
                "{label}: MAD {mad}, shift {signed}"
            );
        }
    }
}

/// Area downscale of a 16-bit render in linear light, to 8-bit sRGB.
pub(in crate::render) fn downscale8(img: &Rgb16Image, edge: usize) -> Rgb8Image {
    let lin = |v: u16| {
        let e = v as f64 / 65535.0;
        (if e <= 0.04045 {
            e / 12.92
        } else {
            ((e + 0.055) / 1.055).powf(2.4)
        }) as f32
    };
    let small = testkit::downscale(
        &LinearImage {
            width: img.width,
            height: img.height,
            data: img.data.iter().map(|&v| lin(v)).collect(),
        },
        edge,
    );
    Rgb8Image {
        width: small.width,
        height: small.height,
        data: small
            .data
            .iter()
            .map(|&v| (super::super::curves::oetf(v as f64) * 255.0).round() as u8)
            .collect(),
    }
}

/// -100/-50/0/+50/+100 strips for each slider: whole frames at a quarter
/// of display size (`presence_<photo>_<slider>.jpg`), and 1:1 crops at
/// -100, 0 and +100 of the display image (`..._crop.jpg`) and of the
/// half-resolution export (`..._export.jpg`, about a 100% view).
#[test]
#[ignore]
fn gpu_presence_strips() {
    let gpu = Gpu::open().expect("gpu");
    for (k, file) in SAMPLES.into_iter().enumerate() {
        let s = testkit::load(file);
        let mut e = engine(&gpu, &s);
        let n = neutral(&s.scene);
        for (name, f) in SLIDERS {
            let imgs: Vec<Rgb8Image> = SWEEP
                .iter()
                .map(|&v| e.display(&with(n, f, v)).unwrap())
                .collect();
            let whole: Vec<Rgb8Image> = imgs.iter().map(|i| testkit::shrink(i, 4)).collect();
            save(
                &testkit::sheet(&[whole], 4),
                &format!("presence_{}_{name}.jpg", s.name),
            );
            let (cw, ch) = (640, 420);
            let at = |img: &Rgb8Image| {
                let (fx, fy) = CROP_AT[k];
                let x0 = ((img.width as f32 * fx) as usize).saturating_sub(cw / 2);
                let y0 = ((img.height as f32 * fy) as usize).saturating_sub(ch / 2);
                crop(img, x0.min(img.width - cw), y0.min(img.height - ch), cw, ch)
            };
            let crops: Vec<Rgb8Image> = [0, 2, 4].iter().map(|&i| at(&imgs[i])).collect();
            save(
                &testkit::sheet(&[crops], 4),
                &format!("presence_{}_{name}_crop.jpg", s.name),
            );
            let exports: Vec<Rgb8Image> = [-100.0, 0.0, 100.0]
                .iter()
                .map(|&v| {
                    let big = full::render(&gpu, &s.scene, &s.half, &with(n, f, v), &*s.color);
                    at(&to8(&big.unwrap()))
                })
                .collect();
            let path = save(
                &testkit::sheet(&[exports], 4),
                &format!("presence_{}_{name}_export.jpg", s.name),
            );
            println!("{name} -> {}", path.display());
        }
    }
}

/// Median milliseconds of `runs` calls of `f`.
fn median_ms(runs: usize, mut f: impl FnMut()) -> f64 {
    let mut t: Vec<f64> = (0..runs)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    t.sort_by(f64::total_cmp);
    t[runs / 2]
}

/// Cost: engine build (with its planes), display() and stats() with the
/// new sliders off and on, and the export at 24 and 61 MP (medians).
#[test]
#[ignore]
fn gpu_presence_timing() {
    let gpu = Gpu::open().expect("gpu");
    let s = testkit::load(SAMPLES[0]);
    let mut e = engine(&gpu, &s);
    let build = median_ms(5, || drop(engine(&gpu, &s)));
    let n = neutral(&s.scene);
    let on = Params {
        texture: 40.0,
        clarity: 30.0,
        dehaze: 20.0,
        curve_darks: 20.0,
        curve_lights: -10.0,
        ..n
    };
    e.display(&on).unwrap();
    let d_off = median_ms(21, || drop(e.display(&n).unwrap()));
    let d_on = median_ms(21, || drop(e.display(&on).unwrap()));
    let batch = |p: Params| -> Vec<Params> {
        (0..1536)
            .map(|i| Params {
                exposure: (i % 17) as f32 * 0.05 - 0.4,
                ..p
            })
            .collect()
    };
    let (b_off, b_on) = (batch(n), batch(on));
    e.stats(&b_on[..64]).unwrap();
    let r_off = 1536.0 / median_ms(5, || drop(e.stats(&b_off).unwrap())) * 1e3;
    let r_on = 1536.0 / median_ms(5, || drop(e.stats(&b_on).unwrap())) * 1e3;
    println!(
        "TIMING engine build {build:.0} ms; display() {d_off:.2} ms (new sliders off), {d_on:.2} ms \
         (presence + curve on); stats {r_off:.0} / {r_on:.0} candidates/s (batch 1536)"
    );
    for file in [SAMPLES[0], "ILCE-7RM5__7RM5-LosslessCompressedLarge.ARW"] {
        let (s, big) = testkit::load_full(file);
        let n = neutral(&s.scene);
        let on = Params {
            texture: 40.0,
            clarity: 30.0,
            dehaze: 20.0,
            curve_darks: 20.0,
            ..n
        };
        full::render(&gpu, &s.scene, &big, &on, &*s.color).unwrap();
        let t_off = median_ms(3, || {
            drop(full::render(&gpu, &s.scene, &big, &n, &*s.color))
        });
        let t_on = median_ms(3, || {
            drop(full::render(&gpu, &s.scene, &big, &on, &*s.color))
        });
        println!(
            "TIMING export {}x{} ({:.0} MP): {t_off:.0} ms new sliders off, {t_on:.0} ms with presence + curve",
            big.width,
            big.height,
            (big.width * big.height) as f64 / 1e6,
        );
    }
}
