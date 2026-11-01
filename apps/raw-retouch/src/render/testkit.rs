//! Test-only stand-ins for the develop stage, which is built in parallel: a
//! CPU loader that turns a RAW into the `LinearImage`s the renderer expects
//! (rawler decode, black/white levels, as-shot white balance, clip, 2x2
//! superpixel, area downscale) and a dcraw-style colour model (one D65
//! matrix, Planckian temperature/tint). Production gets both from develop.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use rawler::decoders::RawDecodeParams;
use rawler::imgop::xyz::Illuminant;
use rawler::rawimage::RawPhotometricInterpretation;
use rawler::rawsource::RawSource;
use rawler::{CFA, RawImageData};

use super::CameraColor;
use crate::color::ColorModel;
use crate::types::{DISPLAY_EDGE, LinearImage, RawInfo, Rgb8Image, SEARCH_EDGE, Scene};

pub(crate) fn samples_dir() -> PathBuf {
    std::env::var_os("SPL_SAMPLES").map_or_else(|| PathBuf::from("/samples"), PathBuf::from)
}

pub(crate) fn out_dir() -> PathBuf {
    let dir =
        std::env::var_os("SPL_OUT").map_or_else(|| PathBuf::from("/out/render"), PathBuf::from);
    std::fs::create_dir_all(&dir).expect("create output dir");
    dir
}

type M3 = [[f64; 3]; 3];

fn mul(a: &M3, b: &M3) -> M3 {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

fn mulv(a: &M3, v: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| (0..3).map(|k| a[i][k] * v[k]).sum())
}

fn inv(m: &M3) -> M3 {
    let [[a, b, c], [d, e, f], [g, h, i]] = *m;
    let (ca, cb, cc) = (e * i - f * h, f * g - d * i, d * h - e * g);
    let det = a * ca + b * cb + c * cc;
    [
        [ca / det, (c * h - b * i) / det, (b * f - c * e) / det],
        [cb / det, (a * i - c * g) / det, (c * d - a * f) / det],
        [cc / det, (b * g - a * h) / det, (a * e - b * d) / det],
    ]
}

const XYZ_FROM_REC2020: M3 = [
    [0.636_958_048_3, 0.144_616_903_6, 0.168_880_975_2],
    [0.262_700_212_0, 0.677_998_071_5, 0.059_301_716_5],
    [0.0, 0.028_072_693_0, 1.060_985_057_7],
];

/// Planckian locus in CIE 1960 uv (Kim et al. 2002 fit).
fn planck_uv(t: f64) -> (f64, f64) {
    let t = t.clamp(1667.0, 25000.0);
    let (t2, t3) = (t * t, t * t * t);
    let x = if t <= 4000.0 {
        -0.266_123_9e9 / t3 - 0.234_358_9e6 / t2 + 0.877_695_6e3 / t + 0.179_910
    } else {
        -3.025_846_9e9 / t3 + 2.107_037_9e6 / t2 + 0.222_634_7e3 / t + 0.240_390
    };
    let (x2, x3) = (x * x, x * x * x);
    let y = if t <= 2222.0 {
        -1.106_381_4 * x3 - 1.348_110_20 * x2 + 2.185_558_32 * x - 0.202_196_83
    } else if t <= 4000.0 {
        -0.954_947_6 * x3 - 1.374_185_93 * x2 + 2.091_370_15 * x - 0.167_488_67
    } else {
        3.081_758_0 * x3 - 5.873_386_70 * x2 + 3.751_129_97 * x - 0.370_014_83
    };
    let d = -2.0 * x + 12.0 * y + 3.0;
    (4.0 * x / d, 6.0 * y / d)
}

/// Unit normal of the locus at `t`, pointing to the green side (higher v).
fn planck_normal(t: f64) -> (f64, f64) {
    let (u0, v0) = planck_uv(t);
    let (u1, v1) = planck_uv(t * 1.01);
    let (du, dv) = (u1 - u0, v1 - v0);
    let len = du.hypot(dv);
    let (nu, nv) = (-dv / len, du / len);
    if nv < 0.0 { (-nu, -nv) } else { (nu, nv) }
}

/// Lightroom's tint scale: tint = 3000 * Duv, positive (magenta in the
/// image) meaning a greener light was neutralized.
const TINT_SCALE: f64 = 3000.0;

fn illuminant_xyz(t: f64, tint: f64) -> [f64; 3] {
    let (u, v) = planck_uv(t);
    let (nu, nv) = planck_normal(t);
    let d = tint / TINT_SCALE;
    let (u, v) = (u + d * nu, v + d * nv);
    let den = 2.0 * u - 8.0 * v + 4.0;
    let (x, y) = (3.0 * u / den, 2.0 * v / den);
    [x / y, 1.0, (1.0 - x - y) / y]
}

/// dcraw-style camera colour from the RAW's D65 ColorMatrix (XYZ -> camera).
pub(crate) struct TestColor {
    cam_from_xyz: M3,
    xyz_from_cam: M3,
    rec2020_from_cam: [[f32; 3]; 3],
}

impl TestColor {
    pub(crate) fn new(xyz_to_cam: [[f32; 3]; 3]) -> Self {
        let cam_from_xyz: M3 = xyz_to_cam.map(|r| r.map(f64::from));
        // Rows normalized so Rec.2020 white maps to camera (1, 1, 1), the
        // white-balanced neutral; the inverse then maps neutral to white.
        let mut cam_from_2020 = mul(&cam_from_xyz, &XYZ_FROM_REC2020);
        for row in &mut cam_from_2020 {
            let s: f64 = row.iter().sum();
            row.iter_mut().for_each(|v| *v /= s);
        }
        Self {
            cam_from_xyz,
            xyz_from_cam: inv(&cam_from_xyz),
            rec2020_from_cam: inv(&cam_from_2020).map(|r| r.map(|v| v as f32)),
        }
    }

    pub(crate) fn temp_tint_for(&self, wb: [f32; 3]) -> (f32, f32) {
        let cam = [1.0 / wb[0] as f64, 1.0, 1.0 / wb[2] as f64];
        let [x, y, z] = mulv(&self.xyz_from_cam, cam);
        let den = x + 15.0 * y + 3.0 * z;
        let (u, v) = (4.0 * x / den, 6.0 * y / den);
        let dist = |t: f64| {
            let (uc, vc) = planck_uv(t);
            (u - uc).hypot(v - vc)
        };
        let mut best = 1700.0;
        let mut t = 1700.0;
        while t < 25000.0 {
            if dist(t) < dist(best) {
                best = t;
            }
            t *= 1.002;
        }
        let (mut lo, mut hi) = (best / 1.002, best * 1.002);
        for _ in 0..60 {
            let (a, b) = (lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0);
            if dist(a) < dist(b) { hi = b } else { lo = a }
        }
        let t = 0.5 * (lo + hi);
        let (uc, vc) = planck_uv(t);
        let (nu, nv) = planck_normal(t);
        let duv = (u - uc) * nu + (v - vc) * nv;
        (t as f32, (duv * TINT_SCALE) as f32)
    }
}

impl CameraColor for TestColor {
    fn rec2020_from_cam(&self, _temperature: f32) -> [[f32; 3]; 3] {
        self.rec2020_from_cam
    }

    fn wb_for(&self, temperature: f32, tint: f32) -> [f32; 3] {
        let cam = mulv(
            &self.cam_from_xyz,
            illuminant_xyz(temperature as f64, tint as f64),
        );
        [(cam[1] / cam[0]) as f32, 1.0, (cam[1] / cam[2]) as f32]
    }
}

/// A decoded RAW, black/white normalized (1 = white level), cropped.
struct Raw {
    width: usize,
    height: usize,
    cpp: usize,
    data: Vec<f32>,
    /// CFA colour at (row, col) of the cropped image.
    cfa: Option<CFA>,
    crop: (usize, usize),
    wb: [f32; 3],
    xyz_to_cam: [[f32; 3]; 3],
    make: String,
    model: String,
}

impl Raw {
    fn color(&self, row: usize, col: usize) -> usize {
        self.cfa
            .as_ref()
            .map_or(0, |c| c.color_at(row + self.crop.1, col + self.crop.0))
    }
}

fn decode(path: &Path) -> Raw {
    let src = RawSource::new(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let img = rawler::decode(&src, &RawDecodeParams::default())
        .unwrap_or_else(|e| panic!("decode {}: {e}", path.display()));
    let samples: Vec<f32> = match &img.data {
        RawImageData::Integer(v) => v.iter().map(|&x| x as f32).collect(),
        RawImageData::Float(v) => v.clone(),
    };
    let levels = img.blacklevel.as_vec();
    let white = img.whitelevel.0.first().copied().unwrap_or(65535) as f32;
    let (cx, cy, cw, ch) = match img.crop_area.or(img.active_area) {
        Some(r) => (r.p.x, r.p.y, r.d.w, r.d.h),
        None => (0, 0, img.width, img.height),
    };
    let cpp = img.cpp;
    let mut data = Vec::with_capacity(cw * ch * cpp);
    for y in cy..cy + ch {
        for x in cx..cx + cw {
            for c in 0..cpp {
                let black = if cpp == 1 && levels.len() == 4 {
                    levels[(y % 2) * 2 + x % 2]
                } else {
                    levels[c.min(levels.len() - 1)]
                };
                let v = samples[(y * img.width + x) * cpp + c];
                data.push(((v - black) / (white - black)).max(0.0));
            }
        }
    }
    let g = img.wb_coeffs[1];
    let wb = [img.wb_coeffs[0] / g, 1.0, img.wb_coeffs[2] / g];
    let wb = if wb.iter().all(|v| v.is_finite() && *v > 0.0) {
        wb
    } else {
        [2.0, 1.0, 1.5]
    };
    let m = img
        .color_matrix
        .get(&Illuminant::D65)
        .or_else(|| img.color_matrix.values().next())
        .expect("colour matrix");
    let xyz_to_cam = [[m[0], m[1], m[2]], [m[3], m[4], m[5]], [m[6], m[7], m[8]]];
    let cfa = match &img.photometric {
        RawPhotometricInterpretation::Cfa(c) => Some(c.cfa.clone()),
        _ => None,
    };
    Raw {
        width: cw,
        height: ch,
        cpp,
        data,
        cfa,
        crop: (cx, cy),
        wb,
        xyz_to_cam,
        make: img.clean_make.clone(),
        model: img.clean_model.clone(),
    }
}

/// White balance, then clip: fully clipped pixels become exactly (1, 1, 1).
fn balance(v: [f32; 3], wb: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|c| (v[c] * wb[c]).min(1.0))
}

/// Half-size RGB: one pixel per 2x2 CFA block (or the linear RAW as is).
fn superpixel(raw: &Raw) -> LinearImage {
    if raw.cpp == 3 {
        let data = raw
            .data
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|&c| balance(c, raw.wb))
            .collect();
        return LinearImage {
            width: raw.width,
            height: raw.height,
            data,
        };
    }
    let (w, h) = (raw.width / 2, raw.height / 2);
    let mut data = Vec::with_capacity(3 * w * h);
    for y in 0..h {
        for x in 0..w {
            let mut sum = [0.0f32; 3];
            let mut n = [0.0f32; 3];
            for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                let (r, c) = (2 * y + dy, 2 * x + dx);
                let col = raw.color(r, c).min(2);
                sum[col] += raw.data[r * raw.width + c];
                n[col] += 1.0;
            }
            data.extend(balance(
                std::array::from_fn(|c| sum[c] / n[c].max(1.0)),
                raw.wb,
            ));
        }
    }
    LinearImage {
        width: w,
        height: h,
        data,
    }
}

/// Full-size RGB by averaging each colour over the 3x3 neighbourhood
/// (bilinear-quality demosaic; fine for timing and matching tests).
fn demosaic(raw: &Raw) -> LinearImage {
    if raw.cpp == 3 {
        return superpixel(raw);
    }
    let (w, h) = (raw.width, raw.height);
    let mut data = vec![0.0f32; 3 * w * h];
    data.chunks_mut(3 * w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let mut sum = [0.0f32; 3];
            let mut n = [0.0f32; 3];
            for yy in y.saturating_sub(1)..(y + 2).min(h) {
                for xx in x.saturating_sub(1)..(x + 2).min(w) {
                    let col = raw.color(yy, xx).min(2);
                    sum[col] += raw.data[yy * w + xx];
                    n[col] += 1.0;
                }
            }
            row[3 * x..3 * x + 3].copy_from_slice(&balance(
                std::array::from_fn(|c| sum[c] / n[c].max(1.0)),
                raw.wb,
            ));
        }
    });
    LinearImage {
        width: w,
        height: h,
        data,
    }
}

/// Area-average downscale so the long edge is `edge` (never upscales).
pub(crate) fn downscale(img: &LinearImage, edge: usize) -> LinearImage {
    let long = img.width.max(img.height);
    if long <= edge {
        return img.clone();
    }
    let scale = edge as f64 / long as f64;
    let w = ((img.width as f64 * scale).round() as usize).max(1);
    let h = ((img.height as f64 * scale).round() as usize).max(1);
    let horizontal = resample_axis(&img.data, img.width, img.height, w, true);
    let data = resample_axis(&horizontal, w, img.height, h, false);
    LinearImage {
        width: w,
        height: h,
        data,
    }
}

/// Box-filter one axis from `n_in` to `n_out` samples (exact area weights).
fn resample_axis(src: &[f32], w: usize, h: usize, n_out: usize, along_x: bool) -> Vec<f32> {
    let n_in = if along_x { w } else { h };
    let (ow, oh) = if along_x { (n_out, h) } else { (w, n_out) };
    let mut out = vec![0.0f32; 3 * ow * oh];
    let ratio = n_in as f64 / n_out as f64;
    for o in 0..n_out {
        let (a, b) = (o as f64 * ratio, (o + 1) as f64 * ratio);
        let mut taps = Vec::new();
        let mut i = a.floor() as usize;
        while (i as f64) < b && i < n_in {
            let wgt = (b.min(i as f64 + 1.0) - a.max(i as f64)) / ratio;
            taps.push((i, wgt as f32));
            i += 1;
        }
        let lines = if along_x { h } else { w };
        for line in 0..lines {
            let mut acc = [0.0f32; 3];
            for &(i, wgt) in &taps {
                let s = if along_x { line * w + i } else { i * w + line };
                for c in 0..3 {
                    acc[c] += wgt * src[3 * s + c];
                }
            }
            let d = if along_x {
                line * ow + o
            } else {
                o * ow + line
            };
            out[3 * d..3 * d + 3].copy_from_slice(&acc);
        }
    }
    out
}

/// "ILCE-7M3___DSC0009.ARW" -> "ILCE-7M3_DSC0009": camera and file, for output names.
fn short_name(file: &str) -> String {
    let stem = file.trim_end_matches(".ARW");
    let camera = stem.split("__").next().unwrap_or(stem);
    let last = stem.rsplit("__").next().unwrap_or(stem).trim_matches('_');
    format!("{camera}_{last}")
}

pub(crate) struct Sample {
    pub(crate) name: String,
    pub(crate) scene: Scene,
    pub(crate) color: Arc<TestColor>,
    /// Half-resolution (superpixel) image: a stand-in for develop_full.
    pub(crate) half: LinearImage,
}

fn scene_from(raw: &Raw, half: &LinearImage, color: &TestColor) -> Scene {
    let (t0, tint0) = color.temp_tint_for(raw.wb);
    let info = RawInfo {
        make: raw.make.clone(),
        model: raw.model.clone(),
        lens: None,
        width: raw.width as u32,
        height: raw.height as u32,
        orientation: 1,
        iso: None,
        exposure_time_s: None,
        f_number: None,
        focal_length_mm: None,
        captured_at: None,
        as_shot_temperature: t0,
        as_shot_tint: tint0,
        raw_clipped_fraction: 0.0,
    };
    Scene {
        info,
        color: ColorModel::new(raw.xyz_to_cam, None),
        wb_as_shot: raw.wb,
        search: downscale(half, SEARCH_EDGE),
        display: downscale(half, DISPLAY_EDGE),
    }
}

pub(crate) fn load(file: &str) -> Sample {
    let raw = decode(&samples_dir().join(file));
    let half = superpixel(&raw);
    let color = TestColor::new(raw.xyz_to_cam);
    let scene = scene_from(&raw, &half, &color);
    Sample {
        name: short_name(file),
        scene,
        color: Arc::new(color),
        half,
    }
}

/// A sample plus its full-resolution image (demosaiced on the CPU).
pub(crate) fn load_full(file: &str) -> (Sample, LinearImage) {
    let raw = decode(&samples_dir().join(file));
    let half = superpixel(&raw);
    let full = demosaic(&raw);
    let color = TestColor::new(raw.xyz_to_cam);
    let scene = scene_from(&raw, &half, &color);
    let sample = Sample {
        name: short_name(file),
        scene,
        color: Arc::new(color),
        half,
    };
    (sample, full)
}

/// Box downscale of an 8-bit image by an integer factor.
pub(crate) fn shrink(img: &Rgb8Image, f: usize) -> Rgb8Image {
    let (w, h) = (img.width / f, img.height / f);
    let mut data = Vec::with_capacity(3 * w * h);
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let mut s = 0u32;
                for dy in 0..f {
                    for dx in 0..f {
                        s += img.data[3 * ((y * f + dy) * img.width + x * f + dx) + c] as u32;
                    }
                }
                data.push(((s + (f * f / 2) as u32) / (f * f) as u32) as u8);
            }
        }
    }
    Rgb8Image {
        width: w,
        height: h,
        data,
    }
}

/// Tiles images in rows on a dark background.
pub(crate) fn sheet(rows: &[Vec<Rgb8Image>], gap: usize) -> Rgb8Image {
    let cell_w = rows.iter().flatten().map(|i| i.width).max().unwrap_or(0);
    let cell_h = rows.iter().flatten().map(|i| i.height).max().unwrap_or(0);
    let cols = rows.iter().map(Vec::len).max().unwrap_or(0);
    let (w, h) = (
        cols * (cell_w + gap) + gap,
        rows.len() * (cell_h + gap) + gap,
    );
    let mut data = vec![24u8; 3 * w * h];
    for (r, row) in rows.iter().enumerate() {
        for (c, img) in row.iter().enumerate() {
            let (ox, oy) = (gap + c * (cell_w + gap), gap + r * (cell_h + gap));
            for y in 0..img.height {
                let src = &img.data[3 * y * img.width..3 * (y + 1) * img.width];
                let d = 3 * ((oy + y) * w + ox);
                data[d..d + src.len()].copy_from_slice(src);
            }
        }
    }
    Rgb8Image {
        width: w,
        height: h,
        data,
    }
}

pub(crate) fn write_jpeg(img: &Rgb8Image, name: &str) -> PathBuf {
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
