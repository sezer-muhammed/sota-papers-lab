//! Full-resolution export render, in horizontal strips so GPU memory stays
//! bounded (a few hundred MB) at any resolution. Each strip carries a halo of
//! rows for the vertical filter passes. The base layer is the display
//! image's, upsampled edge-aware, so the result matches
//! [`super::RenderEngine::display`].

use spl_core::Result;
use spl_gpu::cudarc::driver::{LaunchConfig, PushKernelArg};

use super::curves::{Anchors, Reference, pack};
use super::grid::BaseGrid;
use super::layout::CAND_STRIDE;
use super::presence::{self, Strip};
use super::{Aux, CameraColor, CudaContextExt, Kernels, Preview, blocks, check_image};
use crate::types::{Gpu, LinearImage, Params, Rgb16Image, Scene};

/// Output rows per strip.
const STRIP_ROWS: usize = 512;
/// Capture sharpening: Gaussian sigma in pixels at a 6000 px long edge
/// (24 MP), scaled with resolution; amount; and the soft threshold in Oklab L
/// below which detail (mostly noise) is left alone.
const SHARP_SIGMA_24MP: f32 = 1.0;
const SHARP_AMOUNT: f32 = 0.5;
const SHARP_THRESHOLD: f32 = 0.004;
/// Chroma noise reduction: spatial sigma in pixels, and the Oklab L and a/b
/// differences at which the blur stops crossing an edge.
const NR_SIGMA: f32 = 1.5;
const NR_L: f32 = 0.05;
const NR_AB: f32 = 0.02;

/// The render's own output finishing: a fixed capture sharpening and
/// chroma noise reduction. Off in production, where Lightroom's Detail panel
/// ([`crate::finish::finish_full`]) does that work from the photo's
/// settings; tests switch it on to compare.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Finish {
    pub(crate) sharpen: f32,
    pub(crate) chroma_nr: bool,
}

impl Finish {
    pub(crate) const OFF: Self = Self {
        sharpen: 0.0,
        chroma_nr: false,
    };
}

impl Default for Finish {
    /// The fixed finishing the export had before the Detail panel.
    fn default() -> Self {
        Self {
            sharpen: SHARP_AMOUNT,
            chroma_nr: true,
        }
    }
}

/// The export render without finishing: Detail follows in
/// [`crate::finish::finish_full`].
pub(crate) fn render(
    gpu: &Gpu,
    scene: &Scene,
    full: &LinearImage,
    params: &Params,
    color: &dyn CameraColor,
) -> Result<Rgb16Image> {
    render_with(gpu, scene, full, params, color, Finish::OFF)
}

pub(crate) fn render_with(
    gpu: &Gpu,
    scene: &Scene,
    full: &LinearImage,
    params: &Params,
    color: &dyn CameraColor,
    finish: Finish,
) -> Result<Rgb16Image> {
    check_image(full, "full-resolution")?;
    check_image(&scene.display, "display")?;
    let reference = Reference::new(
        color,
        scene.info.as_shot_temperature,
        scene.info.as_shot_tint,
        scene.wb_as_shot,
    )?;
    let k = Kernels::load(gpu)?;
    let y_row = reference.luminance_row();
    let grid = BaseGrid::build(&scene.display, y_row);
    let mut cand = vec![0f32; CAND_STRIDE];
    pack(
        params,
        color,
        &reference,
        &Anchors::from_log2(grid.p01, grid.p99),
        &mut cand,
    );

    let s = &gpu.stream;
    let (w, h) = (full.width, full.height);
    let sharp_sigma = (SHARP_SIGMA_24MP * w.max(h) as f32 / 6000.0).clamp(0.5, 3.0);
    let sharp_rad = (3.0 * sharp_sigma).ceil() as i32;
    let nr_rad = if finish.chroma_nr {
        (3.0 * NR_SIGMA).ceil() as i32
    } else {
        0
    };
    let amount = finish.sharpen;
    let halo = sharp_rad.max(nr_rad) as usize;
    let rows_in_max = (STRIP_ROWS + 2 * halo).min(h);
    let plane = w * rows_in_max;

    let dev_grid = s.clone_htod(&grid.data).cuda("upload grid")?;
    let display = Preview::upload(gpu, &k, &scene.display, &grid, &dev_grid, y_row, Aux::Guide)?;
    let (dw, dh) = (display.width as i32, display.height as i32);
    let (sx, sy) = (
        display.width as f32 / w as f32,
        display.height as f32 / h as f32,
    );
    let dev_cand = s.clone_htod(&cand).cuda("upload candidate")?;
    let mut rgb = s.alloc_zeros::<f32>(3 * plane).cuda("alloc strip")?;
    let mut planes = Vec::with_capacity(6);
    for _ in 0..6 {
        planes.push(s.alloc_zeros::<f32>(plane).cuda("alloc strip planes")?);
    }
    let mut out16 = s
        .alloc_zeros::<u16>(3 * w * STRIP_ROWS.min(h))
        .cuda("alloc strip output")?;
    let mut data = vec![0u16; 3 * w * h];

    let wi = w as i32;
    for y_out in (0..h).step_by(STRIP_ROWS) {
        let out_rows = STRIP_ROWS.min(h - y_out);
        let y_in = y_out.saturating_sub(halo);
        let rows_in = (y_out + out_rows + halo).min(h) - y_in;
        let n_in = w * rows_in;
        s.memcpy_htod(
            &full.data[3 * w * y_in..3 * w * (y_in + rows_in)],
            &mut rgb.slice_mut(..3 * n_in),
        )
        .cuda("upload strip")?;
        let cfg = |n: usize| LaunchConfig {
            grid_dim: (blocks(n, 256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (rows_i, y_in_i) = (rows_in as i32, y_in as i32);
        let [lp, ap, bp, lh, ah, bh] = &mut planes[..] else {
            unreachable!("six planes")
        };
        let pres = presence::strip_planes(
            gpu,
            &Strip {
                rgb: &rgb,
                width: w,
                rows: rows_in,
                y0: y_in,
                full_height: h,
                display_px: &display.px,
                display_planes: &display.pres,
                display_width: display.width,
                display_height: display.height,
                sx,
                sy,
            },
        )?;
        let pres_on = pres.flag();
        {
            let mut b = s.launch_builder(&k.full);
            b.arg(&rgb).arg(&wi).arg(&rows_i).arg(&y_in_i);
            b.arg(&display.px)
                .arg(&display.aux)
                .arg(&dw)
                .arg(&dh)
                .arg(&sx)
                .arg(&sy);
            b.arg(&y_row[0])
                .arg(&y_row[1])
                .arg(&y_row[2])
                .arg(&dev_cand);
            b.arg(&pres.data).arg(&pres_on);
            b.arg(&mut *lp).arg(&mut *ap).arg(&mut *bp);
            // SAFETY: arguments match `full_kernel`; every buffer holds n_in
            // pixels (`pres` a float4 each when `pres_on`).
            unsafe { b.launch(cfg(n_in)) }.cuda("full_kernel")?;
        }
        {
            let mut b = s.launch_builder(&k.hpass);
            b.arg(&*lp).arg(&*ap).arg(&*bp).arg(&wi).arg(&rows_i);
            b.arg(&sharp_sigma)
                .arg(&sharp_rad)
                .arg(&NR_SIGMA)
                .arg(&nr_rad)
                .arg(&NR_L)
                .arg(&NR_AB);
            b.arg(&mut *lh).arg(&mut *ah).arg(&mut *bh);
            // SAFETY: arguments match `hpass_kernel`.
            unsafe { b.launch(cfg(n_in)) }.cuda("hpass_kernel")?;
        }
        {
            let (row0, out_rows_i) = ((y_out - y_in) as i32, out_rows as i32);
            let mut b = s.launch_builder(&k.finish16);
            b.arg(&*lp)
                .arg(&*lh)
                .arg(&*ah)
                .arg(&*bh)
                .arg(&wi)
                .arg(&rows_i)
                .arg(&row0)
                .arg(&out_rows_i);
            b.arg(&sharp_sigma)
                .arg(&sharp_rad)
                .arg(&amount)
                .arg(&SHARP_THRESHOLD);
            b.arg(&NR_SIGMA)
                .arg(&nr_rad)
                .arg(&NR_L)
                .arg(&NR_AB)
                .arg(&mut out16);
            // SAFETY: arguments match `finish16_kernel`; out16 holds out_rows rows.
            unsafe { b.launch(cfg(w * out_rows)) }.cuda("finish16_kernel")?;
        }
        let n_out = 3 * w * out_rows;
        s.memcpy_dtoh(
            &out16.slice(..n_out),
            &mut data[3 * w * y_out..3 * w * y_out + n_out],
        )
        .cuda("download strip")?;
    }
    s.synchronize().cuda("full render")?;
    Ok(Rgb16Image {
        width: w,
        height: h,
        data,
    })
}
