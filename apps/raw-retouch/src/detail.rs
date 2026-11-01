//! Lightroom's Detail panel on the GPU (`cuda/detail.cu`): sharpening and
//! noise reduction. Owner: detail.
//!
//! It works on the renderer's output, display-referred 16-bit sRGB
//! ([`Rgb16Image`]), after every tone and colour slider, because that is
//! where noise and sharpness are seen and where a fixed threshold means the
//! same thing in shadows and highlights. The image is split into
//! gamma-encoded luma Y' (BT.709 weights) and colour differences
//! Cb = B' - Y', Cr = R' - Y':
//!
//! - **Luminance noise reduction**: a bilateral filter of Y' whose range
//!   sigma is a multiple of the image's own noise at that brightness (a
//!   16-bin noise profile measured from the image), so the slider means a
//!   similar visual strength at any ISO, as Lightroom's calibrated sliders do.
//! - **Colour noise reduction**: chroma is box-averaged onto a grid 4x
//!   coarser (blotches become pixels), cross-bilateral filtered there twice
//!   (luma and chroma both gate the weights, so colour does not bleed across
//!   edges), upsampled, and each pixel's own chroma detail above the noise
//!   level added back (so colour edges stay exact).
//! - **Sharpening**: an unsharp mask on the denoised Y' with Lightroom's four
//!   controls: Amount (gain), Radius (Gaussian sigma in pixels), Detail (low
//!   values sharpen edges and suppress halos by clamping to the local range
//!   and softly thresholding small detail; 100 is a plain unsharp mask) and
//!   Masking (an edge mask from the blurred luma's gradient).
//!
//! The image is processed in horizontal strips with halo rows, so GPU memory
//! stays bounded (about 400 MB at 61 MP) at any resolution. [`auto_detail`]
//! picks settings from the photo's ISO and size.
//!
//! Wiring: `render_full` still applies its own light capture sharpening and
//! chroma noise reduction (`render/full.rs`, `Finish`); switch those off
//! where [`apply`] runs, or the export gets both.

use serde::{Deserialize, Serialize};
use spl_core::{Result, anyhow};
use spl_gpu::cudarc::driver::{CudaFunction, CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg};

use crate::types::{Gpu, Rgb8Image, Rgb16Image};

const SOURCE: &str = include_str!("../cuda/detail.cu");

/// Lightroom's Detail panel, in Lightroom's units and ranges. XMP names:
/// `crs:Sharpness`, `SharpenRadius`, `SharpenDetail`, `SharpenEdgeMasking`,
/// `LuminanceSmoothing`, `ColorNoiseReduction`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DetailSettings {
    /// Sharpening amount, 0 to 150.
    pub sharpen_amount: f32,
    /// Sharpening radius in pixels, 0.5 to 3.0.
    pub sharpen_radius: f32,
    /// 0 to 100: low values sharpen edges and suppress halos, high values
    /// bring out fine texture too.
    pub sharpen_detail: f32,
    /// 0 to 100: 0 sharpens everywhere, 100 only the strongest edges.
    pub sharpen_masking: f32,
    /// Luminance noise reduction, 0 to 100.
    pub luminance_nr: f32,
    /// Colour noise reduction, 0 to 100.
    pub color_nr: f32,
}

impl DetailSettings {
    /// Everything off: [`apply`] returns the image unchanged.
    pub const OFF: Self = Self {
        sharpen_amount: 0.0,
        sharpen_radius: 1.0,
        sharpen_detail: 25.0,
        sharpen_masking: 0.0,
        luminance_nr: 0.0,
        color_nr: 0.0,
    };

    /// Clamped to Lightroom's ranges and rounded to its steps (whole units,
    /// radius in tenths), so an exported XMP loses nothing.
    pub fn quantized(&self) -> Self {
        let whole = |v: f32, hi: f32| v.clamp(0.0, hi).round();
        Self {
            sharpen_amount: whole(self.sharpen_amount, 150.0),
            sharpen_radius: (self.sharpen_radius.clamp(0.5, 3.0) * 10.0).round() / 10.0,
            sharpen_detail: whole(self.sharpen_detail, 100.0),
            sharpen_masking: whole(self.sharpen_masking, 100.0),
            luminance_nr: whole(self.luminance_nr, 100.0),
            color_nr: whole(self.color_nr, 100.0),
        }
    }
}

impl Default for DetailSettings {
    /// Lightroom's defaults for a RAW file.
    fn default() -> Self {
        Self {
            sharpen_amount: 40.0,
            sharpen_radius: 1.0,
            sharpen_detail: 25.0,
            sharpen_masking: 0.0,
            luminance_nr: 0.0,
            color_nr: 25.0,
        }
    }
}

/// Settings for a photo, deterministically from its ISO and size: modest
/// sharpening for every photo (Lightroom's RAW defaults, a touch finer on
/// high-resolution bodies and gentler on noise at high ISO), luminance noise
/// reduction from ISO 400 up, colour noise reduction 25 and more above ISO
/// 1600. Unknown ISO counts as 400.
pub fn auto_detail(iso: Option<u32>, megapixels: f32) -> DetailSettings {
    let iso = iso.filter(|&v| v > 0).unwrap_or(400).clamp(50, 409_600) as f32;
    // Stops above ISO 100.
    let stops = (iso / 100.0).log2();
    let high_iso = (stops - 3.0).max(0.0);
    let mp = if megapixels.is_finite() {
        megapixels
    } else {
        24.0
    };
    DetailSettings {
        sharpen_amount: 40.0 - 2.0 * high_iso,
        sharpen_radius: 1.0 - 0.1 * ((mp - 30.0) / 20.0).clamp(0.0, 1.0),
        sharpen_detail: 25.0 - 3.0 * high_iso,
        sharpen_masking: 6.0 * (stops - 2.0).max(0.0),
        luminance_nr: 8.0 * (stops - 2.0).max(0.0),
        color_nr: 25.0 + 10.0 * (stops - 4.0).max(0.0),
    }
    .quantized()
}

/// Output rows per strip and halo rows above and below it: enough for the
/// widest chain of filters (coarse chroma window and upsampling, 4x; the
/// luma filter, blur and halo window), and a multiple of every grid factor.
const STRIP_ROWS: usize = 1024;
const HALO: usize = 64;
const NBINS: usize = 16;

/// Mirrors `struct Detail` in cuda/detail.cu.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct Detail {
    w: i32,
    rows: i32,
    lw: i32,
    lrows: i32,
    f: i32,
    row0: i32,
    out_rows: i32,
    l_rad: i32,
    l_inv2ss: f32,
    l_k: f32,
    l_mix: f32,
    c_rad: i32,
    c_inv2ss: f32,
    c_floor: f32,
    c_k: f32,
    cy_floor: f32,
    cy_k: f32,
    c_mix: f32,
    cu_k: f32,
    g_rad: i32,
    g_sigma: f32,
    s_amount: f32,
    s_tex: f32,
    s_noise: f32,
    s_halo: f32,
    s_mask: f32,
    s_win: i32,
    luma_sigma: [f32; NBINS],
    chroma_sigma: [f32; NBINS],
}

// SAFETY: plain #[repr(C)] data of 4-byte fields, the same layout as the CUDA struct.
unsafe impl DeviceRepr for Detail {}

/// Noise standard deviation of Y' and of Cb/Cr per Y' bin, in 0..1 units.
#[derive(Debug, Clone, Copy, PartialEq)]
struct NoiseProfile {
    luma: [f32; NBINS],
    chroma: [f32; NBINS],
}

impl Detail {
    /// Kernel parameters for `settings` on an image whose pixels are
    /// `scale` times the full-resolution ones.
    fn new(settings: &DetailSettings, scale: f32, noise: &NoiseProfile) -> Self {
        let s = settings.quantized();
        let scale = if scale.is_finite() {
            scale.clamp(0.05, 1.0)
        } else {
            1.0
        };
        let mut p = Self {
            luma_sigma: noise.luma,
            chroma_sigma: noise.chroma,
            f: grid_factor(scale) as i32,
            ..Self::default()
        };

        let l = s.luminance_nr / 100.0;
        if l > 0.0 {
            let sigma = ((0.8 + 1.4 * l) * scale).max(0.6);
            p.l_rad = ((2.0 * sigma).ceil() as i32).clamp(1, 4);
            p.l_inv2ss = 0.5 / (sigma * sigma);
            p.l_k = 0.8 + 3.2 * l;
            p.l_mix = (8.0 * l).min(1.0);
        }

        let c = s.color_nr / 100.0;
        if c > 0.0 {
            p.c_rad = 2 + (4.0 * c).round() as i32;
            let sigma = p.c_rad as f32 / 2.0;
            p.c_inv2ss = 0.5 / (sigma * sigma);
            p.c_floor = 0.002 + 0.02 * c;
            p.c_k = 1.0 + 4.0 * c;
            p.cy_floor = 0.06;
            p.cy_k = 2.0;
            p.cu_k = 1.5 + 6.0 * c;
            p.c_mix = (10.0 * c).min(1.0);
        }

        if s.sharpen_amount > 0.0 {
            let radius = s.sharpen_radius * scale;
            // Below half a pixel the blur stops resolving: shrink the gain instead.
            p.g_sigma = radius.max(0.5);
            p.g_rad = ((3.0 * p.g_sigma).ceil() as i32).clamp(1, 9);
            // Lightroom's default 40 is a gain of 0.6 on the high-pass.
            p.s_amount = 0.015 * s.sharpen_amount * (radius / 0.5).min(1.0);
            let detail = s.sharpen_detail / 100.0;
            p.s_tex = 0.006 * (1.0 - detail).powf(1.5);
            p.s_noise = 0.6 * (1.0 - 0.5 * detail) * (1.0 - 0.6 * p.l_mix);
            p.s_halo = 0.1 + 0.9 * detail;
            let masking = s.sharpen_masking / 100.0;
            p.s_mask = 0.12 * masking * masking;
            p.s_win = ((1.5 * p.g_sigma).round() as i32).clamp(1, 4);
        }
        p
    }

    fn does_anything(&self) -> bool {
        self.l_mix > 0.0 || self.c_mix > 0.0 || self.s_amount > 0.0
    }
}

/// Coarse chroma grid factor: 4 at full resolution, finer for previews.
fn grid_factor(scale: f32) -> usize {
    if scale >= 0.6 {
        4
    } else if scale >= 0.3 {
        2
    } else {
        1
    }
}

/// Per-bin noise of Y' and chroma from robust high-pass statistics. Luma:
/// the 3x3 Laplacian `[1 -2 1; -2 4 -2; 1 -2 1]` of white noise with sigma s
/// has sigma 6s. Chroma: the residual from the 5x5 mean, which (unlike a
/// Laplacian) also sees the blotchy, demosaic-correlated part of colour
/// noise. Either way the 10th percentile of the magnitude is 0.1257 sigma,
/// and the low percentile keeps edges and most texture (which only add large
/// values) out of the estimate. Rows are sampled across the image, on all
/// cores.
fn noise_profile(img: &Rgb16Image) -> NoiseProfile {
    const HB: usize = 2048;
    const HMAX: f32 = 0.032;
    const MIN_COUNT: u32 = 200;
    const DEFAULT: f32 = 0.004;
    const FLOOR: f32 = 0.0005;
    let (w, h) = (img.width, img.height);
    if w < 5 || h < 5 || img.data.len() < 3 * w * h {
        return NoiseProfile {
            luma: [DEFAULT; NBINS],
            chroma: [DEFAULT; NBINS],
        };
    }
    let rows: Vec<usize> = {
        let n = (h - 4).min(384);
        (0..n).map(|i| 2 + i * (h - 4) / n).collect()
    };
    let x_step = (w / 3072).max(1);
    type Hist = Vec<[u32; HB]>;
    let scan = |rows: &[usize]| -> (Hist, Hist) {
        let mut hl = vec![[0u32; HB]; NBINS];
        let mut hc = vec![[0u32; HB]; NBINS];
        let mut planes = vec![[0f32; 3]; 5 * w];
        let q = |v: f32| ((v.abs() / HMAX * HB as f32) as usize).min(HB - 1);
        for &y in rows {
            for (k, plane) in planes.chunks_mut(w).enumerate() {
                let row = &img.data[3 * (y + k - 2) * w..3 * (y + k - 1) * w];
                for (px, out) in row.as_chunks::<3>().0.iter().zip(plane.iter_mut()) {
                    let [r, g, b] = px.map(|v| f32::from(v) / 65535.0);
                    let l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                    *out = [l, b - l, r - l];
                }
            }
            let at = |dy: usize, x: usize| planes[dy * w + x];
            for x in (2..w - 2).step_by(x_step) {
                let mut lap = 0.0;
                let mut mean = 0.0;
                for (dy, wr) in [(1, 1.0), (2, -2.0), (3, 1.0)] {
                    for (xx, wc) in [(x - 1, 1.0), (x, -2.0), (x + 1, 1.0)] {
                        let v = at(dy, xx)[0];
                        mean += v;
                        lap += wr * wc * v;
                    }
                }
                let mut box5 = [0.0f32; 2];
                for dy in 0..5 {
                    for xx in x - 2..=x + 2 {
                        let v = at(dy, xx);
                        box5[0] += v[1];
                        box5[1] += v[2];
                    }
                }
                let centre = at(2, x);
                let bin = ((mean / 9.0 * NBINS as f32) as usize).min(NBINS - 1);
                hl[bin][q(lap / 6.0)] += 1;
                // The residual of white noise from a 5x5 mean has sigma 0.98 s.
                hc[bin][q((centre[1] - box5[0] / 25.0) / 0.98)] += 1;
                hc[bin][q((centre[2] - box5[1] / 25.0) / 0.98)] += 1;
            }
        }
        (hl, hc)
    };
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get().min(8));
    let per = rows.len().div_ceil(threads).max(1);
    let parts: Vec<(Hist, Hist)> = std::thread::scope(|scope| {
        let handles: Vec<_> = rows
            .chunks(per)
            .map(|chunk| scope.spawn(move || scan(chunk)))
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("noise scan thread"))
            .collect()
    });
    let merge = |pick: fn(&(Hist, Hist)) -> &Hist| {
        let mut total = vec![[0u32; HB]; NBINS];
        for part in &parts {
            for (t, p) in total.iter_mut().zip(pick(part)) {
                for (a, b) in t.iter_mut().zip(p) {
                    *a += b;
                }
            }
        }
        total
    };
    let sigmas = |hist: Hist| -> [f32; NBINS] {
        let raw: Vec<Option<f32>> = hist
            .iter()
            .map(|bins| {
                let count: u32 = bins.iter().sum();
                if count < MIN_COUNT {
                    return None;
                }
                let target = count / 10;
                let mut seen = 0;
                let idx = bins
                    .iter()
                    .position(|&n| {
                        seen += n;
                        seen > target
                    })
                    .unwrap_or(HB - 1);
                let q10 = (idx as f32 + 0.5) * HMAX / HB as f32;
                Some((q10 / 0.1257).max(FLOOR))
            })
            .collect();
        fill_and_smooth(&raw, DEFAULT)
    };
    NoiseProfile {
        luma: sigmas(merge(|p| &p.0)),
        chroma: sigmas(merge(|p| &p.1)),
    }
}

/// Empty bins take the nearest measured neighbour (or `default` when none
/// is measured); then a 1-2-1 smoothing.
fn fill_and_smooth(raw: &[Option<f32>], default: f32) -> [f32; NBINS] {
    let mut filled = [default; NBINS];
    for (i, f) in filled.iter_mut().enumerate() {
        let nearest = (0..NBINS)
            .filter_map(|j| raw[j].map(|v| (i.abs_diff(j), v)))
            .min_by_key(|&(d, _)| d);
        if let Some((_, v)) = nearest {
            *f = v;
        }
    }
    std::array::from_fn(|i| {
        let a = filled[i.saturating_sub(1)];
        let c = filled[(i + 1).min(NBINS - 1)];
        0.25 * a + 0.5 * filled[i] + 0.25 * c
    })
}

struct Kernels {
    ycc: CudaFunction,
    luma_nr: CudaFunction,
    chroma_down: CudaFunction,
    chroma_nr: CudaFunction,
    blur_h: CudaFunction,
    blur_v: CudaFunction,
    finish: CudaFunction,
}

impl Kernels {
    fn load(gpu: &Gpu) -> Result<Self> {
        let module = gpu.module("detail", SOURCE)?;
        let f = |name: &str| {
            module
                .load_function(name)
                .map_err(|e| anyhow!("load kernel {name}: {e:?}"))
        };
        Ok(Self {
            ycc: f("ycc_kernel")?,
            luma_nr: f("luma_nr_kernel")?,
            chroma_down: f("chroma_down_kernel")?,
            chroma_nr: f("chroma_nr_kernel")?,
            blur_h: f("blur_h_kernel")?,
            blur_v: f("blur_v_kernel")?,
            finish: f("finish_kernel")?,
        })
    }
}

/// Sharpening and noise reduction of a display-referred sRGB render.
/// `scale` is the image's pixel size relative to the full-resolution photo
/// (1 for the export, about 0.25 for a 1536 px preview of a 24 MP photo):
/// radii shrink with it, so a preview shows roughly what the export will.
pub fn apply(
    gpu: &Gpu,
    img: &Rgb16Image,
    settings: &DetailSettings,
    scale: f32,
) -> Result<Rgb16Image> {
    let (w, h) = (img.width, img.height);
    if img.data.len() != 3 * w * h {
        return Err(anyhow!(
            "detail: image is malformed: {w}x{h} with {} samples",
            img.data.len()
        ));
    }
    if w == 0 || h == 0 || !Detail::new(settings, scale, &NoiseProfile::flat()).does_anything() {
        return Ok(img.clone());
    }
    apply_with(gpu, img, settings, scale, &noise_profile(img))
}

/// [`apply`] with a given noise profile.
fn apply_with(
    gpu: &Gpu,
    img: &Rgb16Image,
    settings: &DetailSettings,
    scale: f32,
    noise: &NoiseProfile,
) -> Result<Rgb16Image> {
    let (w, h) = (img.width, img.height);
    if i32::try_from(3 * w * (STRIP_ROWS + 2 * HALO)).is_err() {
        return Err(anyhow!("detail: image too wide: {w} px"));
    }
    let mut p = Detail::new(settings, scale, noise);
    let k = Kernels::load(gpu)?;
    let s = &gpu.stream;
    let f = p.f as usize;
    let rows_max = (STRIP_ROWS + 2 * HALO).min(h);
    let plane = w * rows_max;
    let (lw, lrows_max) = (w.div_ceil(f), rows_max.div_ceil(f));
    let alloc = |n: usize| -> Result<CudaSlice<f32>> {
        s.alloc_zeros::<f32>(n.max(1))
            .map_err(|e| anyhow!("detail: allocate {} MB on the GPU: {e:?}", (4 * n) >> 20))
    };
    let mut rgb = s
        .alloc_zeros::<u16>(3 * plane)
        .map_err(|e| anyhow!("detail: allocate strip: {e:?}"))?;
    let mut out16 = s
        .alloc_zeros::<u16>(3 * w * STRIP_ROWS.min(h))
        .map_err(|e| anyhow!("detail: allocate strip: {e:?}"))?;
    let (mut y, mut cb, mut cr) = (alloc(plane)?, alloc(plane)?, alloc(plane)?);
    let mut yd = if p.l_mix > 0.0 {
        alloc(plane)?
    } else {
        alloc(1)?
    };
    let sharpen = p.s_amount > 0.0;
    let (mut tmp, mut blur) = if sharpen {
        (alloc(plane)?, alloc(plane)?)
    } else {
        (alloc(1)?, alloc(1)?)
    };
    let coarse = if p.c_mix > 0.0 { lw * lrows_max } else { 1 };
    let (mut yl, mut cbl, mut crl) = (alloc(coarse)?, alloc(coarse)?, alloc(coarse)?);
    let (mut cbo, mut cro) = (alloc(coarse)?, alloc(coarse)?);
    let (mut cbf, mut crf) = (alloc(coarse)?, alloc(coarse)?);
    let mut data = vec![0u16; 3 * w * h];

    let grid = |gw: usize, gh: usize| LaunchConfig {
        grid_dim: (gw.div_ceil(32) as u32, gh.div_ceil(8) as u32, 1),
        block_dim: (32, 8, 1),
        shared_mem_bytes: 0,
    };
    for y_out in (0..h).step_by(STRIP_ROWS) {
        let out_rows = STRIP_ROWS.min(h - y_out);
        let y_in = y_out.saturating_sub(HALO);
        let rows = (y_out + out_rows + HALO).min(h) - y_in;
        let n = w * rows;
        p.rows = rows as i32;
        p.w = w as i32;
        p.lw = lw as i32;
        p.lrows = rows.div_ceil(f) as i32;
        p.row0 = (y_out - y_in) as i32;
        p.out_rows = out_rows as i32;
        s.memcpy_htod(
            &img.data[3 * w * y_in..3 * w * (y_in + rows)],
            &mut rgb.slice_mut(..3 * n),
        )
        .map_err(|e| anyhow!("detail: upload strip: {e:?}"))?;
        let n_i = n as i32;
        // SAFETY (every launch below): arguments match the kernel of the same
        // name in cuda/detail.cu; full-resolution planes hold `rows` x `w`
        // values, coarse ones `lrows` x `lw`, and `out16` `out_rows` x `w`
        // RGB pixels. Buffers that a disabled stage would use are never read.
        {
            let mut b = s.launch_builder(&k.ycc);
            b.arg(&rgb).arg(&n_i).arg(&mut y).arg(&mut cb).arg(&mut cr);
            let cfg = LaunchConfig::for_num_elems(n as u32);
            unsafe { b.launch(cfg) }.map_err(gpu_err("ycc_kernel"))?;
        }
        if p.l_mix > 0.0 {
            let mut b = s.launch_builder(&k.luma_nr);
            b.arg(&p).arg(&y).arg(&mut yd);
            unsafe { b.launch(grid(w, rows)) }.map_err(gpu_err("luma_nr_kernel"))?;
        }
        let luma = if p.l_mix > 0.0 { &yd } else { &y };
        if p.c_mix > 0.0 {
            let lrows = rows.div_ceil(f);
            let mut b = s.launch_builder(&k.chroma_down);
            b.arg(&p)
                .arg(&y)
                .arg(&cb)
                .arg(&cr)
                .arg(&mut yl)
                .arg(&mut cbl)
                .arg(&mut crl);
            unsafe { b.launch(grid(lw, lrows)) }.map_err(gpu_err("chroma_down_kernel"))?;
            // Two passes, the second on the first's output: twice the
            // reach for blotches, at a sixteenth of the pixels.
            let mut b = s.launch_builder(&k.chroma_nr);
            b.arg(&p)
                .arg(&yl)
                .arg(&cbl)
                .arg(&crl)
                .arg(&mut cbo)
                .arg(&mut cro);
            unsafe { b.launch(grid(lw, lrows)) }.map_err(gpu_err("chroma_nr_kernel"))?;
            let mut b = s.launch_builder(&k.chroma_nr);
            b.arg(&p)
                .arg(&yl)
                .arg(&cbo)
                .arg(&cro)
                .arg(&mut cbf)
                .arg(&mut crf);
            unsafe { b.launch(grid(lw, lrows)) }.map_err(gpu_err("chroma_nr_kernel"))?;
        }
        if sharpen {
            let mut b = s.launch_builder(&k.blur_h);
            b.arg(&p).arg(luma).arg(&mut tmp);
            unsafe { b.launch(grid(w, rows)) }.map_err(gpu_err("blur_h_kernel"))?;
            let mut b = s.launch_builder(&k.blur_v);
            b.arg(&p).arg(&tmp).arg(&mut blur);
            unsafe { b.launch(grid(w, rows)) }.map_err(gpu_err("blur_v_kernel"))?;
        }
        {
            let mut b = s.launch_builder(&k.finish);
            b.arg(&p)
                .arg(&y)
                .arg(luma)
                .arg(&blur)
                .arg(&cb)
                .arg(&cr)
                .arg(&cbl)
                .arg(&crl)
                .arg(&cbf)
                .arg(&crf)
                .arg(&mut out16);
            unsafe { b.launch(grid(w, out_rows)) }.map_err(gpu_err("finish_kernel"))?;
        }
        let n_out = 3 * w * out_rows;
        s.memcpy_dtoh(
            &out16.slice(..n_out),
            &mut data[3 * w * y_out..3 * w * y_out + n_out],
        )
        .map_err(|e| anyhow!("detail: download strip: {e:?}"))?;
    }
    s.synchronize().map_err(gpu_err("synchronize"))?;
    Ok(Rgb16Image {
        width: w,
        height: h,
        data,
    })
}

/// [`apply`] for an 8-bit image (previews).
pub fn apply8(
    gpu: &Gpu,
    img: &Rgb8Image,
    settings: &DetailSettings,
    scale: f32,
) -> Result<Rgb8Image> {
    let wide = Rgb16Image {
        width: img.width,
        height: img.height,
        data: img.data.iter().map(|&v| u16::from(v) * 257).collect(),
    };
    let out = apply(gpu, &wide, settings, scale)?;
    Ok(Rgb8Image {
        width: out.width,
        height: out.height,
        data: out
            .data
            .iter()
            .map(|&v| ((u32::from(v) + 128) / 257) as u8)
            .collect(),
    })
}

impl NoiseProfile {
    /// Stand-in for deciding whether anything will run.
    fn flat() -> Self {
        Self {
            luma: [0.004; NBINS],
            chroma: [0.004; NBINS],
        }
    }
}

fn gpu_err<E: std::fmt::Debug>(what: &'static str) -> impl Fn(E) -> spl_core::Error {
    move |e| anyhow!("GPU detail {what}: {e:?}")
}

#[cfg(test)]
mod gpu_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_detail_follows_iso() {
        let low = auto_detail(Some(100), 24.0);
        assert_eq!(low.luminance_nr, 0.0);
        assert_eq!(low.color_nr, 25.0);
        assert_eq!(low.sharpen_amount, 40.0);
        assert_eq!(low.sharpen_radius, 1.0);
        assert_eq!(low.sharpen_masking, 0.0);
        let mut last = low;
        for iso in [200, 400, 800, 1600, 3200, 6400, 12_800, 25_600, 102_400] {
            let s = auto_detail(Some(iso), 24.0);
            assert!(s.luminance_nr >= last.luminance_nr, "{iso}: {s:?}");
            assert!(s.color_nr >= last.color_nr, "{iso}: {s:?}");
            assert!(s.sharpen_amount > 0.0 && s.sharpen_amount <= last.sharpen_amount);
            assert!(s.sharpen_masking >= last.sharpen_masking);
            assert_eq!(s, s.quantized(), "already in Lightroom steps");
            last = s;
        }
        let iso6400 = auto_detail(Some(6400), 24.0);
        assert!((25.0..=40.0).contains(&iso6400.luminance_nr), "{iso6400:?}");
        assert!(iso6400.color_nr > 25.0);
        assert!(last.luminance_nr <= 100.0 && last.color_nr <= 100.0);
        assert!(auto_detail(Some(100), 61.0).sharpen_radius < 1.0);
        assert_eq!(auto_detail(None, 24.0), auto_detail(Some(400), 24.0));
        assert_eq!(auto_detail(Some(0), f32::NAN), auto_detail(Some(400), 24.0));
    }

    #[test]
    fn quantized_clamps_to_lightroom_ranges() {
        let q = DetailSettings {
            sharpen_amount: 170.4,
            sharpen_radius: 0.27,
            sharpen_detail: 33.6,
            sharpen_masking: -3.0,
            luminance_nr: 101.0,
            color_nr: 24.5,
        }
        .quantized();
        assert_eq!(
            q,
            DetailSettings {
                sharpen_amount: 150.0,
                sharpen_radius: 0.5,
                sharpen_detail: 34.0,
                sharpen_masking: 0.0,
                luminance_nr: 100.0,
                color_nr: 25.0,
            }
        );
    }

    #[test]
    fn kernel_params_follow_settings() {
        let noise = NoiseProfile::flat();
        assert!(!Detail::new(&DetailSettings::OFF, 1.0, &noise).does_anything());
        let p = Detail::new(&DetailSettings::default(), 1.0, &noise);
        assert!(p.s_amount > 0.0 && p.c_mix > 0.0 && p.l_mix == 0.0);
        assert_eq!(p.f, 4);
        // Whatever the settings, the filter chain fits in the halo rows.
        let max = DetailSettings {
            sharpen_amount: 150.0,
            sharpen_radius: 3.0,
            sharpen_detail: 100.0,
            sharpen_masking: 100.0,
            luminance_nr: 100.0,
            color_nr: 100.0,
        };
        let p = Detail::new(&max, 1.0, &noise);
        let luma_reach = p.l_rad + p.g_rad.max(p.s_win) + 1;
        let chroma_reach = (2 * p.c_rad + 3) * p.f;
        assert!(luma_reach.max(chroma_reach) < HALO as i32, "{p:?}");
        assert_eq!(STRIP_ROWS % 4, 0);
        assert_eq!(HALO % 4, 0);
        // A small preview: finer chroma grid, gentler sharpening.
        let small = Detail::new(&DetailSettings::default(), 0.25, &noise);
        assert_eq!(small.f, 1);
        assert!(small.s_amount < Detail::new(&DetailSettings::default(), 1.0, &noise).s_amount);
    }

    fn noisy(w: usize, h: usize, level: f32, sigma: f32, seed: u64) -> Rgb16Image {
        let mut state = seed;
        let mut gauss = || {
            // Box-Muller on a 64-bit LCG.
            let mut u = || {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 11) as f64 / (1u64 << 53) as f64).max(1e-12)
            };
            let (a, b) = (u(), u());
            ((-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()) as f32
        };
        let data = (0..3 * w * h)
            .map(|_| ((level + sigma * gauss()).clamp(0.0, 1.0) * 65535.0).round() as u16)
            .collect();
        Rgb16Image {
            width: w,
            height: h,
            data,
        }
    }

    #[test]
    fn noise_profile_measures_white_noise() {
        let sigma = 0.02;
        let img = noisy(800, 600, 0.5, sigma, 7);
        let n = noise_profile(&img);
        // Independent channel noise: luma sigma = sigma * |(0.2126, 0.7152, 0.0722)|.
        let luma = sigma * (0.2126f32.powi(2) + 0.7152f32.powi(2) + 0.0722f32.powi(2)).sqrt();
        let bin = n.luma[NBINS / 2];
        assert!((bin / luma - 1.0).abs() < 0.12, "luma {bin} vs {luma}");
        let chroma = n.chroma[NBINS / 2];
        assert!(
            chroma > luma,
            "chroma differences carry more noise: {chroma}"
        );
        let clean = noise_profile(&noisy(800, 600, 0.5, 0.0, 7));
        assert!(clean.luma.iter().all(|&v| v <= 0.001), "{clean:?}");
        let tiny = noise_profile(&noisy(2, 2, 0.5, 0.0, 7));
        assert_eq!(tiny.luma, [0.004; NBINS]);
    }

    #[test]
    fn fill_and_smooth_uses_nearest_bins() {
        let mut raw = [None; NBINS];
        raw[4] = Some(0.01);
        raw[10] = Some(0.03);
        let f = fill_and_smooth(&raw, 0.004);
        assert!((f[0] - 0.01).abs() < 1e-6 && (f[15] - 0.03).abs() < 1e-6);
        assert!(f[7] >= 0.01 && f[7] <= 0.03);
        assert_eq!(fill_and_smooth(&[None; NBINS], 0.004), [0.004; NBINS]);
    }
}
