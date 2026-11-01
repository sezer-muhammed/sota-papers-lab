//! Black & White on the GPU (`#[ignore]`; run on the GPU host with
//! `gpu-test <checkout> render::bw --ignored`): the kernel against
//! [`apply`], grey and finite renders at every size, the gray mixer on real
//! colours, and contact sheets of the black-and-white look under /out/bw/.

use std::path::PathBuf;
use std::sync::LazyLock;
use std::time::Instant;

use spl_gpu::cudarc::driver::{LaunchConfig, PushKernelArg};

use super::super::testkit::{self, Sample};
use super::super::{CudaContextExt, RenderEngine, SOURCE, full};
use super::*;
use crate::finish::Detail;
use crate::objective::LookSpec;
use crate::search::{self, SearchConfig};
use crate::types::{Gpu, Rgb8Image, Rgb16Image};

/// A sky, foliage, a colour chart and two everyday scenes.
const SAMPLES: [&str; 3] = [
    "ILCE-6700__DSC00001.ARW",
    "ILCE-6000__DSC01542.ARW",
    "ILCE-7M3___DSC0009.ARW",
];
const SKY: &str = "ILCE-6700__DSC00001.ARW";
const CHECKER: &str = "ILCE-7RM3__Sony_-_ILCE-7RM3_-_12bit_uncompressed__3_2_.ARW";

fn out_dir() -> PathBuf {
    let dir = std::env::var_os("SPL_OUT").map_or_else(|| PathBuf::from("/out/bw"), PathBuf::from);
    std::fs::create_dir_all(&dir).expect("create output dir");
    dir
}

fn write_jpeg(img: &Rgb8Image, name: &str) -> PathBuf {
    let path = out_dir().join(name);
    let enc = jpeg_encoder::Encoder::new_file(&path, 90).expect("create jpeg");
    enc.encode(
        &img.data,
        img.width as u16,
        img.height as u16,
        jpeg_encoder::ColorType::Rgb,
    )
    .expect("encode jpeg");
    path
}

/// The renderer's source plus a kernel that runs the stage alone.
static PROBE_SOURCE: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}\n{}",
        *SOURCE,
        r#"
extern "C" __global__ void bw_probe_kernel(const float* __restrict__ B, float gain, const float* __restrict__ in,
                                           float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = in[3 * i], y = in[3 * i + 1], z = in[3 * i + 2];
    bw_apply(B, gain, x, y, z);
    out[3 * i] = x;
    out[3 * i + 1] = y;
    out[3 * i + 2] = z;
}
"#
    )
});

fn probe(gpu: &Gpu, b: &[f32], gain: f32, input: &[[f32; 3]]) -> Vec<[f32; 3]> {
    let module = gpu
        .module("bw_probe", PROBE_SOURCE.as_str())
        .expect("compile probe");
    let f = module.load_function("bw_probe_kernel").expect("probe");
    let s = &gpu.stream;
    let flat: Vec<f32> = input.iter().flatten().copied().collect();
    let db = s.clone_htod(b).unwrap();
    let din = s.clone_htod(&flat).unwrap();
    let mut dout = s.alloc_zeros::<f32>(flat.len()).unwrap();
    let n = input.len() as i32;
    let mut l = s.launch_builder(&f);
    l.arg(&db).arg(&gain).arg(&din).arg(&mut dout).arg(&n);
    let cfg = LaunchConfig {
        grid_dim: (input.len().div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: arguments match `bw_probe_kernel`; buffers hold n RGB triples
    // and B holds BW_LEN floats.
    unsafe { l.launch(cfg) }.cuda("probe").unwrap();
    s.clone_dtoh(&dout).unwrap().as_chunks::<3>().0.to_vec()
}

fn rng(seed: u64) -> impl FnMut() -> f32 {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn bw_of(p: &Params) -> Params {
    Params { bw: 1.0, ..*p }
}

/// The kernel matches [`apply`]: plain luminance with a neutral mixer,
/// greys untouched by any mixer, equal channels, finite everywhere.
#[test]
#[ignore]
fn gpu_kernel_matches_cpu() {
    let gpu = Gpu::open().expect("gpu");
    let mut next = rng(5);
    let mut input: Vec<[f32; 3]> = (0..20000)
        .map(|_| {
            let k = next().powi(3) * 1.2;
            [next() * k, next() * k, next() * k]
        })
        .collect();
    for v in [0.0, 1e-6, 0.002, 0.18, 1.0, 4.0] {
        input.push([v; 3]);
    }
    input.extend([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0; 3]]);
    let base = Params::neutral(5500.0, 0.0);
    let plain = probe(&gpu, &packed_of(&bw_of(&base)), 1.0, &input);
    let mut worst = 0.0f32;
    for seed in 0..8u64 {
        let mut r = rng(seed + 100);
        let mut a = bw_of(&base).to_array();
        for v in &mut a[Params::COUNT - 8..] {
            *v = ((r() * 2.0 - 1.0) * 100.0).round();
        }
        let p = Params::from_array(a);
        let b = packed_of(&p);
        for gain in [0.5f32, 1.0, 4.0] {
            let out = probe(&gpu, &b, gain, &input);
            for ((rgb, g), pl) in input.iter().zip(&out).zip(&plain) {
                assert!(g.iter().all(|v| v.is_finite()), "{rgb:?} -> {g:?}");
                assert!(g[0] == g[1] && g[1] == g[2], "{rgb:?} -> {g:?}");
                let c = apply(&b, gain, *rgb);
                let d = (g[0] - c[0]).abs() / c[0].abs().max(1e-3);
                worst = worst.max(d);
                assert!(d < 1e-4, "seed {seed}: {rgb:?} gpu {g:?} cpu {c:?}");
                if rgb[0] == rgb[1] && rgb[1] == rgb[2] {
                    assert_eq!(g, pl, "grey changed");
                }
            }
        }
    }
    // With a neutral mixer the grey is the luminance.
    for (rgb, g) in input.iter().zip(&plain) {
        let y = 0.262_700_2 * rgb[0] + 0.677_998_07 * rgb[1] + 0.059_301_717 * rgb[2];
        assert!((g[0] - y).abs() <= 1e-6 * y.max(1e-3), "{rgb:?} -> {g:?}");
    }
    // In colour the stage does nothing.
    let colour = probe(&gpu, &packed_of(&base), 1.0, &input);
    assert_eq!(colour, input);
    println!("kernel vs CPU: worst relative difference {worst:.2e}");
}

fn packed_of(p: &Params) -> Vec<f32> {
    let mut out = vec![0.0; BW_LEN];
    pack(p, &mut out);
    out
}

fn assert_grey8(img: &Rgb8Image, what: &str) {
    let bad = img
        .data
        .as_chunks::<3>()
        .0
        .iter()
        .filter(|p| p[0] != p[1] || p[1] != p[2])
        .count();
    assert_eq!(bad, 0, "{what}: {bad} coloured pixels");
}

fn assert_grey16(img: &Rgb16Image, what: &str) {
    let bad = img
        .data
        .as_chunks::<3>()
        .0
        .iter()
        .filter(|p| p[0] != p[1] || p[1] != p[2])
        .count();
    assert_eq!(bad, 0, "{what}: {bad} coloured pixels");
}

fn engine(gpu: &Gpu, s: &Sample) -> RenderEngine {
    RenderEngine::with_color(gpu, &s.scene, s.color.clone()).expect("engine")
}

fn neutral(s: &Sample) -> Params {
    crate::finish::with_auto(
        &Params::neutral(s.scene.info.as_shot_temperature, s.scene.info.as_shot_tint),
        &s.scene.info,
    )
}

/// Oklab (a, b) of an 8-bit sRGB pixel, roughly: enough to find colours.
fn ab8(p: &[u8]) -> (f32, f32) {
    let lin = |v: u8| {
        let e = v as f32 / 255.0;
        if e <= 0.04045 {
            e / 12.92
        } else {
            ((e + 0.055) / 1.055).powf(2.4)
        }
    };
    let (r, g, b) = (lin(p[0]), lin(p[1]), lin(p[2]));
    let l = (0.412_221_47 * r + 0.536_332_55 * g + 0.051_445_995 * b).cbrt();
    let m = (0.211_903_5 * r + 0.680_699_5 * g + 0.107_396_96 * b).cbrt();
    let s = (0.088_302_46 * r + 0.281_718_85 * g + 0.629_978_7 * b).cbrt();
    (
        1.977_998_5 * l - 2.428_592_2 * m + 0.450_593_7 * s,
        0.025_904_037 * l + 0.782_771_77 * m - 0.808_675_77 * s,
    )
}

/// Every render of black and white is grey (display with Detail, search,
/// full resolution) and finite, its stats see no colour, and the blue
/// slider moves a real sky monotonically while greys stay put.
#[test]
#[ignore]
fn gpu_renders_are_grey_and_bands_act_monotonically() {
    let gpu = Gpu::open().expect("gpu");
    for file in [SKY, CHECKER] {
        let (s, full_img) = testkit::load_full(file);
        let mut e = engine(&gpu, &s);
        let n = neutral(&s);
        let colour = e.display(&n).unwrap();
        // Blue (sky) pixels and near-grey pixels of the colour render.
        let (mut sky, mut greys) = (Vec::new(), Vec::new());
        for (i, p) in colour.data.as_chunks::<3>().0.iter().enumerate() {
            let (a, b) = ab8(p);
            let c = a.hypot(b);
            let h = b.atan2(a).to_degrees().rem_euclid(360.0);
            let mid = (40..230).contains(&p[1]);
            if c > 0.06 && (245.0..285.0).contains(&h) && mid {
                sky.push(i);
            } else if c < 0.008 && mid {
                greys.push(i);
            }
        }
        let mean = |img: &Rgb8Image, px: &[usize]| {
            px.iter().map(|&i| f64::from(img.data[3 * i])).sum::<f64>() / px.len().max(1) as f64
        };
        let mut last = -1.0;
        let mut grey_means = Vec::new();
        for v in [-80.0, -40.0, 0.0, 40.0, 80.0] {
            let p = Params {
                gray_blue: v,
                ..bw_of(&n)
            };
            let d = e.display(&p).unwrap();
            assert_grey8(&d, "display");
            assert_grey8(&e.search(&p).unwrap(), "search");
            let st = e.stats(&[p]).unwrap().remove(0);
            for v in [st.luma_p50, st.luma_mean, st.local_contrast, st.structure] {
                assert!(v.is_finite());
            }
            // chroma_p95 reads the first bin of the chroma histogram (0.4 / 256).
            assert!(st.colorfulness < 0.5 && st.chroma_mean < 1e-5, "{st:?}");
            assert!(st.chroma_p95 < 0.4 / 256.0, "{st:?}");
            assert_eq!(st.out_of_gamut, 0.0);
            let m = mean(&d, &sky);
            println!(
                "{} blue {v:+}: sky {m:.1} over {} px, greys {:.2}",
                s.name,
                sky.len(),
                mean(&d, &greys)
            );
            if sky.len() > 500 {
                assert!(m > last + 1.0, "sky {m} after {last}");
            }
            last = m;
            grey_means.push(mean(&d, &greys));
        }
        let spread = grey_means.iter().fold(f64::MIN, |a, &b| a.max(b))
            - grey_means.iter().fold(f64::MAX, |a, &b| a.min(b));
        assert!(spread < 2.0, "greys moved by {spread:.2}");
        // The export: grey at full resolution, with Detail and a mixer.
        let p = Params {
            gray_blue: -40.0,
            gray_orange: 20.0,
            ..bw_of(&n)
        };
        let mut f = full::render(&gpu, &s.scene, &full_img, &p, &*s.color).unwrap();
        crate::finish::finish_full(
            &gpu,
            &mut f,
            &Detail::of(&p),
            &crate::finish::LensCorrection::new(&p, &s.scene.info),
        )
        .unwrap();
        grey16(&mut f);
        assert_grey16(&f, "full");
        let raw = full::render(&gpu, &s.scene, &full_img, &p, &*s.color).unwrap();
        assert_grey16(&raw, "full before Detail");
    }
}

/// Contact sheets of the black-and-white look on three samples: colour
/// auto, a plain desaturation (auto, treatment on, neutral mixer), the
/// searched look, then gray mixer sweeps. Prints the search's timings.
#[test]
#[ignore]
fn gpu_bw_contact_sheets() {
    let gpu = Gpu::open().expect("gpu");
    for file in SAMPLES {
        let s = testkit::load(file);
        let mut e = engine(&gpu, &s);
        let cfg = SearchConfig::default();
        let t = Instant::now();
        let r = search::run(
            &mut e,
            &s.scene,
            &[LookSpec::natural(), LookSpec::black_and_white()],
            &cfg,
            &mut |_, _| {},
        )
        .unwrap();
        let both_ms = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let only = search::run(
            &mut e,
            &s.scene,
            &[LookSpec::black_and_white()],
            &cfg,
            &mut |_, _| {},
        )
        .unwrap();
        let bw_ms = t.elapsed().as_secs_f64() * 1e3;
        let look = r.looks.iter().find(|c| c.id == "look:bw").unwrap();
        let p = look.params;
        assert!(p.is_bw() && look.score.feasible, "{:?}", look.score);
        assert!(only.looks[0].params.is_bw());
        println!(
            "{}: natural+bw {both_ms:.0} ms ({} evals), bw alone {bw_ms:.0} ms ({} evals); \
             bw score {:.2}: exp {:+.2} con {:+} hi {:+} sh {:+} wh {:+} bl {:+} \
             clar {:+} tex {:+} | mixer R{:+} O{:+} Y{:+} G{:+} A{:+} B{:+} P{:+} M{:+}",
            s.name,
            r.evaluations,
            only.evaluations,
            look.score.total,
            p.exposure,
            p.contrast,
            p.highlights,
            p.shadows,
            p.whites,
            p.blacks,
            p.clarity,
            p.texture,
            p.gray_red,
            p.gray_orange,
            p.gray_yellow,
            p.gray_green,
            p.gray_aqua,
            p.gray_blue,
            p.gray_purple,
            p.gray_magenta,
        );
        for (name, v) in &look.score.terms {
            if v.abs() > 0.01 {
                print!("{name} {v:.2}  ");
            }
        }
        println!();
        let auto = r.auto.params;
        let desat = Params { bw: 1.0, ..auto };
        let shrink = |img: Rgb8Image| testkit::shrink(&img, 3);
        let mut rows = vec![vec![
            shrink(e.display(&auto).unwrap()),
            shrink(e.display(&desat).unwrap()),
            shrink(e.display(&p).unwrap()),
        ]];
        for (key, idx) in [("blue", 5), ("orange", 1), ("green", 3)] {
            let mut row = Vec::new();
            for v in [-60.0, 0.0, 60.0] {
                let mut a = p.to_array();
                a[Params::COUNT - 8 + idx] = v;
                let img = e.display(&Params::from_array(a)).unwrap();
                assert_grey8(&img, key);
                row.push(shrink(img));
            }
            rows.push(row);
        }
        let path = write_jpeg(&testkit::sheet(&rows, 8), &format!("{}_bw.jpg", s.name));
        write_jpeg(&e.display(&p).unwrap(), &format!("{}_bw_look.jpg", s.name));
        println!("sheet {}", path.display());
    }
}
