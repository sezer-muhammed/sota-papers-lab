//! Texture, Clarity and Dehaze: the multi-pass part of the renderer, with
//! `cuda/presence.cu` (which describes the planes and the per-pixel
//! application in detail). Owner: presence.
//!
//! These need blurred copies of the image, so they come in two parts:
//! - per-image *planes* (a float4 per pixel, independent of the sliders),
//!   made once by [`preview_planes`] for the search and display previews
//!   when a `RenderEngine` is built (and for the display image of every
//!   full-resolution export), and by [`strip_planes`] for each strip of a
//!   full-resolution export: Texture's band, Clarity's band, the haze
//!   transmission and the airlight. The kernels (`render8`, `stats`,
//!   `full`) hand each pixel's planes to `presence_apply`, so the search's
//!   thousands of candidates reuse them and its statistics see the effect;
//! - the sliders, which [`pack`] turns into strengths at `P_PRES`.
//!
//! What each slider does, at +100 / -100:
//! - Texture multiplies the fine band (0.04% to 0.4% of the long edge,
//!   edge-aware) by 1 + [`TEXTURE_UP`] / 1 - [`TEXTURE_DOWN`] in log
//!   luminance: crisper or smoother surfaces, strong edges untouched.
//! - Clarity does the same, by [`CLARITY_UP`] / [`CLARITY_DOWN`], for
//!   everything between that band and the base layer (about 2.5%,
//!   edge-aware) plus half the fine band, weighted toward midtones.
//! - Dehaze removes [`DEHAZE_UNVEIL`] of the dark-channel haze estimate
//!   (veil subtraction and rescaling by the transmission), with a little
//!   extra local contrast and saturation; negative values lay a neutral fog
//!   of the airlight's brightness over the image, thicker where it is hazy.
//!
//! Scales are fractions of the long edge ([`Scales`]), so the 768 px search
//! image, the 1536 px display image and the export look alike at their
//! sizes. The export filters its strips at Texture's small scale (the
//! strip halo in `render/full.rs` covers [`halo_rows`]); the base layer and
//! the transmission come from the display image.

use spl_core::{Result, anyhow};
use spl_gpu::cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};

use super::layout::P_PRES_N;
use super::{CudaContextExt, SOURCE, blocks};
use crate::types::{Gpu, Params};

/// Texture at +100 / -100: share of the fine band added / removed.
pub(crate) const TEXTURE_UP: f32 = 1.4;
pub(crate) const TEXTURE_DOWN: f32 = 0.8;
/// Clarity at +100 / -100: share of the mid band added / removed (midtones).
pub(crate) const CLARITY_UP: f32 = 1.1;
pub(crate) const CLARITY_DOWN: f32 = 0.7;
/// Dehaze +100: share of the estimated haze removed, the clarity it adds,
/// and its saturation gain.
pub(crate) const DEHAZE_UNVEIL: f32 = 0.9;
const DEHAZE_CLARITY: f32 = 0.25;
const DEHAZE_SATURATION: f32 = 0.2;
/// Dehaze -100: fog blended in where the haze estimate is thickest.
pub(crate) const DEHAZE_FOG: f32 = 0.5;

/// Texture's fine Gaussian sigma and its edge-aware smoothing radius, as
/// fractions of the long edge, and the filter's epsilon: local contrast
/// well above sqrt(eps) EV counts as an edge, not texture.
const TEX_FINE: f32 = 0.0004;
const TEX_COARSE: f32 = 0.004;
const TEX_EPS: f32 = 0.1;
/// Dehaze: dark channel patch radius and the transmission's guided filter
/// radius (fractions of the long edge), its epsilon, the lowest
/// transmission believed, and the dark-channel quantile taken as airlight.
const HAZE_PATCH: f32 = 0.01;
const HAZE_GUIDE: f32 = 0.03;
const HAZE_EPS: f32 = 1e-3;
const T_MIN: f32 = 0.3;
const AIRLIGHT_QUANTILE: f64 = 0.999;
/// Outputs per thread of the sliding box filter (`PRES_RUN` in the kernel).
const BOX_RUN: usize = 32;

/// Writes `p`'s Presence strengths into `out` (`P_PRES_N` floats at `P_PRES`,
/// in the order of `cuda/presence.cu`'s `PR_*` slots). All zero (the stage
/// off) when Texture, Clarity and Dehaze all are.
pub(crate) fn pack(p: &Params, out: &mut [f32]) {
    let out = &mut out[..P_PRES_N];
    out.fill(0.0);
    if !active(p) {
        return;
    }
    let (t, c, d) = (unit(p.texture), unit(p.clarity), unit(p.dehaze));
    let up = d.max(0.0);
    out[..7].copy_from_slice(&[
        1.0,
        t * if t >= 0.0 { TEXTURE_UP } else { TEXTURE_DOWN },
        c * if c >= 0.0 { CLARITY_UP } else { CLARITY_DOWN },
        DEHAZE_UNVEIL * up,
        DEHAZE_FOG * (-d).max(0.0),
        DEHAZE_CLARITY * up,
        1.0 + DEHAZE_SATURATION * up,
    ]);
}

fn unit(v: f32) -> f32 {
    if v.is_finite() {
        v.clamp(-100.0, 100.0) / 100.0
    } else {
        0.0
    }
}

/// Whether `p` uses Texture, Clarity or Dehaze at all.
pub(crate) fn active(p: &Params) -> bool {
    [p.texture, p.clarity, p.dehaze]
        .iter()
        .any(|&v| unit(v) != 0.0)
}

/// One image's presence planes on the GPU: four floats per pixel, or none.
pub(crate) struct Planes {
    /// A float4 per pixel when `on`, otherwise a single placeholder float4.
    pub(crate) data: CudaSlice<f32>,
    pub(crate) on: bool,
}

impl Planes {
    pub(crate) fn none(gpu: &Gpu) -> Result<Self> {
        Ok(Self {
            data: gpu
                .stream
                .alloc_zeros::<f32>(4)
                .cuda("alloc presence planes")?,
            on: false,
        })
    }

    /// The kernels' `pres_on` argument.
    pub(crate) fn flag(&self) -> i32 {
        i32::from(self.on)
    }
}

/// Filter sizes in pixels for an image whose long edge is `long`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Scales {
    fine_sigma: f32,
    fine_rad: i32,
    tex_rad: i32,
    patch_rad: i32,
    guide_rad: i32,
}

impl Scales {
    pub(crate) fn new(long: usize) -> Self {
        let px = |f: f32| (f * long as f32).round().max(1.0) as i32;
        let fine_sigma = (TEX_FINE * long as f32).max(0.2);
        Self {
            fine_sigma,
            fine_rad: (3.0 * fine_sigma).ceil() as i32,
            tex_rad: px(TEX_COARSE),
            patch_rad: px(HAZE_PATCH),
            guide_rad: px(HAZE_GUIDE),
        }
    }

    /// Rows above and below a strip that its Texture band depends on.
    fn support(&self) -> usize {
        (self.fine_rad + self.tex_rad) as usize
    }
}

/// Halo rows a full-resolution strip of a `width` x `height` export of `p`
/// needs for [`strip_planes`] (none when `p` does not use the planes).
pub(crate) fn halo_rows(width: usize, height: usize, p: &Params) -> usize {
    if active(p) {
        Scales::new(width.max(height)).support()
    } else {
        0
    }
}

struct Kernels {
    lum: CudaFunction,
    gauss: CudaFunction,
    min: CudaFunction,
    boxf: CudaFunction,
    self_prep: CudaFunction,
    self_coef: CudaFunction,
    haze_prep: CudaFunction,
    coef: CudaFunction,
    planes: CudaFunction,
    strip: CudaFunction,
}

impl Kernels {
    fn load(gpu: &Gpu) -> Result<Self> {
        let module = gpu.module("render", &SOURCE)?;
        let f = |name: &str| module.load_function(name).cuda(name);
        Ok(Self {
            lum: f("pres_lum_kernel")?,
            gauss: f("pres_gauss_kernel")?,
            min: f("pres_min_kernel")?,
            boxf: f("pres_box_kernel")?,
            self_prep: f("pres_self_prep_kernel")?,
            self_coef: f("pres_self_coef_kernel")?,
            haze_prep: f("pres_haze_prep_kernel")?,
            coef: f("pres_coef_kernel")?,
            planes: f("pres_planes_kernel")?,
            strip: f("pres_strip_kernel")?,
        })
    }
}

fn cfg(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (blocks(n, 256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Launches on one image (a preview or a strip) of `w` x `h` pixels.
struct Pass<'a> {
    gpu: &'a Gpu,
    k: &'a Kernels,
    w: i32,
    h: i32,
    n: usize,
}

impl<'a> Pass<'a> {
    fn new(gpu: &'a Gpu, k: &'a Kernels, w: usize, h: usize) -> Self {
        Self {
            gpu,
            k,
            w: w as i32,
            h: h as i32,
            n: w * h,
        }
    }

    fn alloc(&self, floats_per_px: usize) -> Result<CudaSlice<f32>> {
        self.gpu
            .stream
            .alloc_zeros::<f32>(floats_per_px * self.n)
            .cuda("alloc presence scratch")
    }

    /// log2 luminance into `l`, and the darkest channel into `dark` if given.
    /// `src` holds `stride` floats per pixel, camera RGB first.
    fn lum(
        &self,
        src: &CudaSlice<f32>,
        stride: i32,
        y_row: [f32; 3],
        l: &mut CudaSlice<f32>,
        dark: Option<&mut CudaSlice<f32>>,
    ) -> Result<()> {
        let n = self.n as i32;
        let want = i32::from(dark.is_some());
        let mut placeholder = None;
        let dark = match dark {
            Some(d) => d,
            None => placeholder.insert(self.gpu.stream.alloc_zeros::<f32>(1).cuda("alloc")?),
        };
        let mut b = self.gpu.stream.launch_builder(&self.k.lum);
        b.arg(src).arg(&stride).arg(&n);
        b.arg(&y_row[0]).arg(&y_row[1]).arg(&y_row[2]);
        b.arg(l).arg(dark).arg(&want);
        // SAFETY: arguments match `pres_lum_kernel`; `src` holds n pixels of
        // `stride` floats and `l` n floats; `dark` is written only when it
        // holds n floats.
        unsafe { b.launch(cfg(self.n)) }.cuda("pres_lum_kernel")?;
        Ok(())
    }

    /// One separable pass of a one-float-per-pixel filter (`pres_gauss_kernel`
    /// with `sigma`, or `pres_min_kernel`).
    fn pass1(
        &self,
        f: &CudaFunction,
        sigma: Option<f32>,
        rad: i32,
        vertical: bool,
        src: &CudaSlice<f32>,
        dst: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let v = i32::from(vertical);
        let mut b = self.gpu.stream.launch_builder(f);
        b.arg(src).arg(&self.w).arg(&self.h);
        if let Some(sigma) = &sigma {
            b.arg(sigma);
        }
        b.arg(&rad).arg(&v).arg(dst);
        // SAFETY: arguments match `pres_gauss_kernel` (with sigma) or
        // `pres_min_kernel` (without); both buffers hold n floats.
        unsafe { b.launch(cfg(self.n)) }.cuda("presence filter")?;
        Ok(())
    }

    /// Separable filter of `src` into `dst` through `tmp`.
    fn separable(
        &self,
        f: &CudaFunction,
        sigma: Option<f32>,
        rad: i32,
        src: &CudaSlice<f32>,
        tmp: &mut CudaSlice<f32>,
        dst: &mut CudaSlice<f32>,
    ) -> Result<()> {
        self.pass1(f, sigma, rad, false, src, tmp)?;
        self.pass1(f, sigma, rad, true, tmp, dst)
    }

    /// Box mean of `comps` interleaved floats per pixel, in place in `buf`
    /// through `tmp`.
    fn box_mean(
        &self,
        comps: i32,
        rad: i32,
        buf: &mut CudaSlice<f32>,
        tmp: &mut CudaSlice<f32>,
    ) -> Result<()> {
        for vertical in [0i32, 1] {
            let (src, dst) = if vertical == 0 {
                (&*buf, &mut *tmp)
            } else {
                (&*tmp, &mut *buf)
            };
            let mut b = self.gpu.stream.launch_builder(&self.k.boxf);
            b.arg(src).arg(&self.w).arg(&self.h).arg(&comps);
            b.arg(&rad).arg(&vertical).arg(dst);
            let (len, lines) = if vertical == 0 {
                (self.w, self.h)
            } else {
                (self.h, self.w)
            };
            let threads = (len as usize).div_ceil(BOX_RUN) * lines as usize * comps as usize;
            // SAFETY: arguments match `pres_box_kernel` (one thread per run of
            // BOX_RUN outputs); both hold n * comps floats.
            unsafe { b.launch(cfg(threads)) }.cuda("pres_box_kernel")?;
        }
        Ok(())
    }

    /// A per-pixel kernel taking `inputs`, then `scalars`, then n, then `out`.
    fn map(
        &self,
        f: &CudaFunction,
        inputs: &[&CudaSlice<f32>],
        scalars: &[f32],
        out: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let n = self.n as i32;
        let mut b = self.gpu.stream.launch_builder(f);
        for i in inputs {
            b.arg(*i);
        }
        for s in scalars {
            b.arg(s);
        }
        b.arg(&n).arg(out);
        // SAFETY: each caller passes exactly the kernel's arguments in order;
        // every buffer holds n pixels of the kernel's layout.
        unsafe { b.launch(cfg(self.n)) }.cuda("presence map")?;
        Ok(())
    }

    /// Texture's lightly blurred log luminance `l` (into `ga`) and the
    /// coefficients (a, b) of its self-guided smoothing a ga + b (returned,
    /// two floats per pixel; not box-averaged a second time, which would
    /// double the strips' halo for no visible gain).
    fn texture(
        &self,
        sc: &Scales,
        l: &CudaSlice<f32>,
        ga: &mut CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>> {
        {
            let mut scratch = self.alloc(1)?;
            self.separable(
                &self.k.gauss,
                Some(sc.fine_sigma),
                sc.fine_rad,
                l,
                &mut scratch,
                ga,
            )?;
        }
        let mut m = self.alloc(2)?;
        let mut ab = self.alloc(2)?;
        self.map(&self.k.self_prep, &[ga], &[], &mut m)?;
        self.box_mean(2, sc.tex_rad, &mut m, &mut ab)?;
        self.map(&self.k.self_coef, &[&m], &[TEX_EPS], &mut ab)?;
        Ok(ab)
    }
}

/// The airlight: a high quantile of the dark channel (camera RGB at exposure
/// 0, 1 = raw clip), from every fourth sample of a large image (the dark
/// channel is smooth).
fn airlight(dark: &[f32]) -> f32 {
    let step = if dark.len() > 1 << 18 { 4 } else { 1 };
    let mut v: Vec<f32> = dark
        .iter()
        .step_by(step)
        .copied()
        .filter(|x| x.is_finite())
        .collect();
    if v.is_empty() {
        return 1.0;
    }
    let k = ((v.len() - 1) as f64 * AIRLIGHT_QUANTILE).round() as usize;
    let (_, a, _) = v.select_nth_unstable_by(k, f32::total_cmp);
    a.clamp(1e-4, 1.0)
}

/// Planes for a preview image. `px` is the prepared image (`prep_kernel`: a
/// float4 per pixel of camera RGB, as-shot white balance, and the base
/// layer), `width` x `height`; `y_row` maps its RGB to luminance as the base
/// layer does.
pub(crate) fn preview_planes(
    gpu: &Gpu,
    px: &CudaSlice<f32>,
    width: usize,
    height: usize,
    y_row: [f32; 3],
) -> Result<Planes> {
    if width == 0 || height == 0 || px.len() < 4 * width * height {
        return Err(anyhow!("presence planes: bad image {width}x{height}"));
    }
    let k = Kernels::load(gpu)?;
    let pass = Pass::new(gpu, &k, width, height);
    let sc = Scales::new(width.max(height));
    let mut l = pass.alloc(1)?;
    let mut dark = pass.alloc(1)?;
    pass.lum(px, 4, y_row, &mut l, Some(&mut dark))?;
    let mut ga = pass.alloc(1)?;
    let tex_ab = pass.texture(&sc, &l, &mut ga)?;

    // Dark channel, airlight, raw transmission refined by a guided filter.
    let mut dc = pass.alloc(1)?;
    {
        let mut tmp = pass.alloc(1)?;
        pass.separable(&k.min, None, sc.patch_rad, &dark, &mut tmp, &mut dc)?;
    }
    drop(dark);
    let a = airlight(&gpu.stream.clone_dtoh(&dc).cuda("download dark channel")?);
    let inv_a = 1.0 / a;
    let mut haze_ab = pass.alloc(2)?;
    {
        let mut m4 = pass.alloc(4)?;
        let mut t4 = pass.alloc(4)?;
        pass.map(&k.haze_prep, &[&l, &dc], &[inv_a], &mut m4)?;
        pass.box_mean(4, sc.guide_rad, &mut m4, &mut t4)?;
        pass.map(&k.coef, &[&m4], &[HAZE_EPS], &mut haze_ab)?;
        pass.box_mean(2, sc.guide_rad, &mut haze_ab, &mut t4)?;
    }
    drop(dc);

    let mut out = pass.alloc(4)?;
    let n = pass.n as i32;
    let mut b = gpu.stream.launch_builder(&k.planes);
    b.arg(&l).arg(&ga).arg(&tex_ab).arg(&haze_ab);
    b.arg(&inv_a).arg(&a).arg(&T_MIN).arg(&n).arg(&mut out);
    // SAFETY: arguments match `pres_planes_kernel`; out holds n float4s, l
    // and ga n floats, the coefficients two floats per pixel.
    unsafe { b.launch(cfg(pass.n)) }.cuda("pres_planes_kernel")?;
    Ok(Planes {
        data: out,
        on: true,
    })
}

/// One horizontal strip of a full-resolution export, as `render/full.rs`
/// renders it.
pub(crate) struct Strip<'a> {
    /// Camera RGB (3 floats per pixel, as-shot white balance), `rows` rows
    /// of `width` starting at full-image row `y0`, halo rows included.
    pub(crate) rgb: &'a CudaSlice<f32>,
    pub(crate) width: usize,
    pub(crate) rows: usize,
    pub(crate) y0: usize,
    pub(crate) full_height: usize,
    /// Camera RGB -> luminance, as the base layer sees it.
    pub(crate) y_row: [f32; 3],
    /// The display image's planes, and display pixels per full-resolution
    /// pixel.
    pub(crate) display_planes: &'a Planes,
    pub(crate) display_width: usize,
    pub(crate) display_height: usize,
    pub(crate) sx: f32,
    pub(crate) sy: f32,
    /// Whether the export's parameters use the planes at all ([`active`]).
    pub(crate) active: bool,
}

/// Planes for one full-resolution strip (`rows` x `width`, the strip's own
/// pixel order). Texture's band is right only beyond [`halo_rows`] of a
/// strip end that is not the image's.
pub(crate) fn strip_planes(gpu: &Gpu, strip: &Strip) -> Result<Planes> {
    if !strip.active || !strip.display_planes.on {
        return Planes::none(gpu);
    }
    let k = Kernels::load(gpu)?;
    let pass = Pass::new(gpu, &k, strip.width, strip.rows);
    let sc = Scales::new(strip.width.max(strip.full_height));
    let mut l = pass.alloc(1)?;
    pass.lum(strip.rgb, 3, strip.y_row, &mut l, None)?;
    let mut ga = pass.alloc(1)?;
    let tex_ab = pass.texture(&sc, &l, &mut ga)?;
    drop(l);
    let mut out = pass.alloc(4)?;
    let (w, rows, y0) = (strip.width as i32, strip.rows as i32, strip.y0 as i32);
    let (dw, dh) = (strip.display_width as i32, strip.display_height as i32);
    let mut b = gpu.stream.launch_builder(&k.strip);
    b.arg(&w).arg(&rows).arg(&y0).arg(&ga).arg(&tex_ab);
    b.arg(&strip.display_planes.data).arg(&dw).arg(&dh);
    b.arg(&strip.sx).arg(&strip.sy).arg(&mut out);
    // SAFETY: arguments match `pres_strip_kernel`; the strip buffers hold
    // rows x width pixels, the display ones dw x dh.
    unsafe { b.launch(cfg(pass.n)) }.cuda("pres_strip_kernel")?;
    Ok(Planes {
        data: out,
        on: true,
    })
}

#[cfg(test)]
pub(super) mod tests;
