//! [`RawFrame`] → linear camera RGB on the GPU (`cuda/develop.cu`). Owner: develop.

use std::sync::Arc;

use spl_core::{Result, anyhow, bail};
use spl_gpu::cudarc::driver::{
    CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg,
};

use crate::decode::RawFrame;
use crate::types::{DISPLAY_EDGE, Gpu, LinearImage, SEARCH_EDGE, Scene};

const SOURCE: &str = include_str!("../cuda/develop.cu");

/// Smallest frame the kernels accept: the demosaic mirrors up to 6 pixels
/// past the border.
const MIN_EDGE: usize = 16;

/// Normalize black/white levels, apply as-shot white balance, clip
/// highlights to neutral, half-size superpixel, then area-downscale to the
/// search and display sizes.
pub fn develop_scene(gpu: &Gpu, frame: &RawFrame) -> Result<Scene> {
    let mosaic = Mosaic::new(frame)?;
    let k = Kernels::load(gpu)?;
    let stream = &gpu.stream;
    let raw = stream
        .clone_htod(&frame.data)
        .map_err(gpu_err("upload RAW"))?;
    let mut clipped = stream.alloc_zeros::<u32>(1).map_err(gpu_err("alloc"))?;

    // Half-size RGB for Bayer data (one pixel per 2x2 block); linear data is
    // already RGB at full size.
    let (base_w, base_h) = if frame.channels == 1 {
        (frame.width / 2, frame.height / 2)
    } else {
        (frame.width, frame.height)
    };
    let mut base = alloc::<f32>(stream, base_w * base_h * 3)?;
    if frame.channels == 1 {
        let (w, h) = (base_w as i32, base_h as i32);
        let mut launch = stream.launch_builder(&k.superpixel);
        launch
            .arg(&raw)
            .arg(&mosaic)
            .arg(&mut base)
            .arg(&w)
            .arg(&h)
            .arg(&mut clipped);
        // SAFETY: argument types and order match `superpixel` in cuda/develop.cu;
        // `base` holds w * h RGB pixels and `raw` the 2w x 2h (or larger) mosaic.
        unsafe { launch.launch(grid(base_w, base_h)) }.map_err(gpu_err("superpixel"))?;
    } else {
        let mut launch = stream.launch_builder(&k.linear_prep);
        launch
            .arg(&raw)
            .arg(&mosaic)
            .arg(&mut base)
            .arg(&mut clipped);
        // SAFETY: argument types and order match `linear_prep`; both buffers
        // hold width * height RGB pixels.
        unsafe { launch.launch(grid(frame.width, frame.height)) }
            .map_err(gpu_err("linear_prep"))?;
    }
    drop(raw);

    let display = downscale(&k, stream, &base, base_w, base_h, DISPLAY_EDGE)?;
    let search = downscale(&k, stream, &base, base_w, base_h, SEARCH_EDGE)?;
    drop(base);
    let clipped = stream.clone_dtoh(&clipped).map_err(gpu_err("download"))?[0];

    let mut info = frame.info.clone();
    info.raw_clipped_fraction = clipped as f32 / (base_w * base_h).max(1) as f32;
    Ok(Scene {
        info,
        color: frame.color.clone(),
        wb_as_shot: frame.wb_as_shot,
        search,
        display,
    })
}

/// Full-resolution demosaic with the same normalization as
/// [`develop_scene`]. Cropped and unrotated.
///
/// RCD (ratio corrected demosaicing) for Bayer data. Peak GPU memory is 24
/// bytes per pixel plus the upload: about 1.6 GB for 61 MP.
pub fn develop_full(gpu: &Gpu, frame: &RawFrame) -> Result<LinearImage> {
    let mosaic = Mosaic::new(frame)?;
    let k = Kernels::load(gpu)?;
    let stream = &gpu.stream;
    let (w, h) = (frame.width, frame.height);
    let pixels = w * h;
    let mut out = alloc::<f32>(stream, pixels * 3)?;
    let cfg = grid(w, h);

    if frame.channels == 3 {
        let raw = stream
            .clone_htod(&frame.data)
            .map_err(gpu_err("upload RAW"))?;
        let mut clipped = stream.alloc_zeros::<u32>(1).map_err(gpu_err("alloc"))?;
        let mut launch = stream.launch_builder(&k.linear_prep);
        launch
            .arg(&raw)
            .arg(&mosaic)
            .arg(&mut out)
            .arg(&mut clipped);
        // SAFETY: matches `linear_prep`; both buffers hold w * h RGB pixels.
        unsafe { launch.launch(cfg) }.map_err(gpu_err("linear_prep"))?;
    } else {
        let mut cfa = alloc::<f32>(stream, pixels)?;
        {
            let raw = stream
                .clone_htod(&frame.data)
                .map_err(gpu_err("upload RAW"))?;
            let mut launch = stream.launch_builder(&k.mosaic_prep);
            launch.arg(&raw).arg(&mosaic).arg(&mut cfa);
            // SAFETY: matches `mosaic_prep`; `raw` and `cfa` hold w * h samples.
            unsafe { launch.launch(cfg) }.map_err(gpu_err("mosaic_prep"))?;
        }
        let mut vh = alloc::<f32>(stream, pixels)?;
        // Low-pass values first, then (once green is done with them) the
        // diagonal directions.
        let mut aux = alloc::<f32>(stream, pixels)?;
        // SAFETY (all launches below): argument types and order match the
        // kernels of the same name; every plane holds w * h values and `out`
        // w * h RGB pixels.
        let mut launch = stream.launch_builder(&k.rcd_directions);
        launch.arg(&cfa).arg(&mosaic).arg(&mut vh).arg(&mut aux);
        unsafe { launch.launch(cfg) }.map_err(gpu_err("rcd_directions"))?;
        let mut launch = stream.launch_builder(&k.rcd_green);
        launch
            .arg(&cfa)
            .arg(&vh)
            .arg(&aux)
            .arg(&mosaic)
            .arg(&mut out);
        unsafe { launch.launch(cfg) }.map_err(gpu_err("rcd_green"))?;
        let mut launch = stream.launch_builder(&k.rcd_diagonals);
        launch.arg(&cfa).arg(&mosaic).arg(&mut aux);
        unsafe { launch.launch(cfg) }.map_err(gpu_err("rcd_diagonals"))?;
        let mut launch = stream.launch_builder(&k.rcd_rb_at_rb);
        launch.arg(&cfa).arg(&aux).arg(&mosaic).arg(&mut out);
        unsafe { launch.launch(cfg) }.map_err(gpu_err("rcd_rb_at_rb"))?;
        drop((cfa, aux));
        let mut launch = stream.launch_builder(&k.rcd_rb_at_g);
        launch.arg(&vh).arg(&mosaic).arg(&mut out);
        unsafe { launch.launch(cfg) }.map_err(gpu_err("rcd_rb_at_g"))?;
    }

    let n = (pixels * 3) as u32;
    let mut launch = stream.launch_builder(&k.clamp_unit);
    launch.arg(&mut out).arg(&n);
    // SAFETY: matches `clamp_unit`; `out` holds n floats.
    unsafe { launch.launch(LaunchConfig::for_num_elems(n)) }.map_err(gpu_err("clamp_unit"))?;
    let data = stream.clone_dtoh(&out).map_err(gpu_err("download"))?;
    Ok(LinearImage {
        width: w,
        height: h,
        data,
    })
}

/// Mirrors `struct Mosaic` in cuda/develop.cu.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Mosaic {
    width: i32,
    height: i32,
    cfa: [i32; 4],
    black: [f32; 4],
    scale: [f32; 4],
    white: f32,
}

// SAFETY: plain #[repr(C)] data with the same layout as the CUDA struct.
unsafe impl DeviceRepr for Mosaic {}

impl Mosaic {
    /// Folds black, white, white balance and the neutral highlight clip into
    /// one scale per CFA position (or channel), so a sample becomes
    /// `min((v - black) * scale, 1)`.
    fn new(frame: &RawFrame) -> Result<Self> {
        if frame.channels != 1 && frame.channels != 3 {
            bail!("unsupported RAW layout: {} channels", frame.channels);
        }
        if frame.width < MIN_EDGE || frame.height < MIN_EDGE {
            bail!("RAW image too small: {}x{}", frame.width, frame.height);
        }
        if frame.data.len() != frame.width * frame.height * frame.channels {
            bail!(
                "RAW data has {} samples, expected {}x{}x{}",
                frame.data.len(),
                frame.width,
                frame.height,
                frame.channels
            );
        }
        if i32::try_from(frame.data.len()).is_err() {
            bail!("RAW image too large: {}x{}", frame.width, frame.height);
        }
        if frame.wb_as_shot.iter().any(|&m| !m.is_finite() || m <= 0.0) {
            bail!("invalid white balance {:?}", frame.wb_as_shot);
        }
        let clip = frame.wb_as_shot.iter().copied().fold(f32::MAX, f32::min);
        let color = |slot: usize| {
            if frame.channels == 1 {
                usize::from(frame.cfa[slot]).min(2)
            } else {
                slot.min(2)
            }
        };
        let scale = [0, 1, 2, 3].map(|slot| {
            frame.wb_as_shot[color(slot)] / ((frame.white - frame.black[slot]).max(1.0) * clip)
        });
        Ok(Self {
            width: frame.width as i32,
            height: frame.height as i32,
            cfa: frame.cfa.map(i32::from),
            black: frame.black,
            scale,
            white: frame.white,
        })
    }
}

struct Kernels {
    superpixel: CudaFunction,
    linear_prep: CudaFunction,
    area_downscale: CudaFunction,
    mosaic_prep: CudaFunction,
    rcd_directions: CudaFunction,
    rcd_green: CudaFunction,
    rcd_diagonals: CudaFunction,
    rcd_rb_at_rb: CudaFunction,
    rcd_rb_at_g: CudaFunction,
    clamp_unit: CudaFunction,
}

impl Kernels {
    fn load(gpu: &Gpu) -> Result<Self> {
        let module: Arc<CudaModule> = gpu.module("develop", SOURCE)?;
        let f = |name: &str| {
            module
                .load_function(name)
                .map_err(|e| anyhow!("load kernel {name}: {e:?}"))
        };
        Ok(Self {
            superpixel: f("superpixel")?,
            linear_prep: f("linear_prep")?,
            area_downscale: f("area_downscale")?,
            mosaic_prep: f("mosaic_prep")?,
            rcd_directions: f("rcd_directions")?,
            rcd_green: f("rcd_green")?,
            rcd_diagonals: f("rcd_diagonals")?,
            rcd_rb_at_rb: f("rcd_rb_at_rb")?,
            rcd_rb_at_g: f("rcd_rb_at_g")?,
            clamp_unit: f("clamp_unit")?,
        })
    }
}

/// Output size with the long edge at `edge` and the aspect ratio kept; never
/// upscales.
fn fit(width: usize, height: usize, edge: usize) -> (usize, usize) {
    let long = width.max(height);
    if long <= edge {
        return (width, height);
    }
    let short = |s: usize| ((s as f64 * edge as f64 / long as f64).round() as usize).max(1);
    if width >= height {
        (edge, short(height))
    } else {
        (short(width), edge)
    }
}

fn downscale(
    k: &Kernels,
    stream: &Arc<CudaStream>,
    base: &CudaSlice<f32>,
    base_w: usize,
    base_h: usize,
    edge: usize,
) -> Result<LinearImage> {
    let (w, h) = fit(base_w, base_h, edge);
    let mut out = alloc::<f32>(stream, w * h * 3)?;
    let (in_w, in_h, out_w, out_h) = (base_w as i32, base_h as i32, w as i32, h as i32);
    let mut launch = stream.launch_builder(&k.area_downscale);
    launch
        .arg(base)
        .arg(&in_w)
        .arg(&in_h)
        .arg(&mut out)
        .arg(&out_w)
        .arg(&out_h);
    // SAFETY: matches `area_downscale`; `base` holds in_w * in_h RGB pixels and
    // `out` out_w * out_h.
    unsafe { launch.launch(grid(w, h)) }.map_err(gpu_err("area_downscale"))?;
    let data = stream.clone_dtoh(&out).map_err(gpu_err("download"))?;
    Ok(LinearImage {
        width: w,
        height: h,
        data,
    })
}

fn alloc<T: DeviceRepr>(stream: &Arc<CudaStream>, len: usize) -> Result<CudaSlice<T>> {
    // SAFETY: every caller's kernels write each element before anything reads it.
    unsafe { stream.alloc::<T>(len) }.map_err(|e| {
        anyhow!(
            "allocate {} MB on the GPU: {e:?}",
            (len * std::mem::size_of::<T>()) >> 20
        )
    })
}

/// One thread per pixel, 32 x 8 blocks (a warp per row segment).
fn grid(width: usize, height: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (width.div_ceil(32) as u32, height.div_ceil(8) as u32, 1),
        block_dim: (32, 8, 1),
        shared_mem_bytes: 0,
    }
}

fn gpu_err<E: std::fmt::Debug>(what: &'static str) -> impl Fn(E) -> spl_core::Error {
    move |e| anyhow!("GPU {what}: {e:?}")
}

#[cfg(test)]
mod gpu_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_keeps_aspect_and_never_upscales() {
        assert_eq!(fit(3000, 2000, 1536), (1536, 1024));
        assert_eq!(fit(2000, 3000, 768), (512, 768));
        assert_eq!(fit(3096, 2064, 1536), (1536, 1024));
        assert_eq!(fit(4608, 3072, 768), (768, 512));
        assert_eq!(fit(1000, 600, 1536), (1000, 600));
        assert_eq!(fit(10000, 3, 768), (768, 1));
    }
}
