//! [`Params`] → pixels and [`Stats`] on the GPU (`cuda/render.cu`). Owner: render.
//!
//! Per pixel: the white-balance delta and camera → linear Rec.2020 (fully
//! clipped pixels stay neutral), exposure (plus [`BASELINE_EV`]), Highlights
//! and Shadows as a gain driven by an edge-aware base layer (bilateral grid of
//! log2 luminance, `render/grid.rs`), Whites and Blacks as a gain by the
//! pixel's luminance, anchored at the photo's own dark and bright ends, then
//! a film-like tone curve bent by Contrast and applied hue-preservingly to the
//! largest and smallest channel (colours roll off to white instead of
//! clipping), Vibrance and Saturation on Oklab chroma, and chroma-only gamut
//! mapping into sRGB. What each slider does is defined in `render/curves.rs`
//! on the CPU; the kernels apply the matrices and lookup tables it packs.
//!
//! Statistics come from the same kernel code on the search image, one launch
//! per batch of parameter sets; [`render_full`] adds capture sharpening and
//! chroma noise reduction at full resolution and upsamples the display
//! image's base layer, so the export matches [`RenderEngine::display`].
//!
//! Settings beyond the Basic panel plug in through hooks with owners of
//! their own: the parametric tone curve (`render/tone_curve.rs`,
//! `cuda/tone_curve.cu`), the Color Mixer and Calibration
//! (`render/color_mixer.rs`, `cuda/color_mixer.cu`), Texture, Clarity and
//! Dehaze (`render/presence.rs`, `cuda/presence.cu`), Black & White and
//! its gray mixer (`render/bw.rs`, `cuda/bw.cu`), and Detail and Lens
//! Corrections after rendering ([`crate::finish`]). The `.cu` hook files are
//! spliced into `cuda/render.cu` to make one NVRTC source, [`SOURCE`].

mod bw;
mod color_mixer;
mod curves;
#[cfg(test)]
mod fingerprint;
mod full;
mod grid;
mod layout;
mod presence;
mod stats;
#[cfg(test)]
mod testkit;
#[cfg(test)]
mod tests;
mod tone_curve;

use std::sync::{Arc, LazyLock};

use spl_core::{Result, anyhow};
use spl_gpu::cudarc::driver::sys::CUdevice_attribute;
use spl_gpu::cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};

use crate::color::ColorModel;
use crate::finish::{self, Detail, LensCorrection};
use crate::types::{Gpu, LinearImage, Params, RawInfo, Rgb8Image, Rgb16Image, Scene, Stats};
use curves::{Anchors, Reference, pack};
use grid::BaseGrid;
use layout::{CAND_STRIDE, N_SUMS, PART_U32, STATS_THREADS};
use presence::Planes;

/// The line of `cuda/render.cu` after which the hook files go.
const HOOKS: &str = "// HOOKS:";

/// The renderer's NVRTC source: `cuda/render.cu` with its hook files
/// inserted after its `HOOKS` line, ahead of `render_core`.
pub(crate) static SOURCE: LazyLock<String> = LazyLock::new(|| {
    let render = include_str!("../cuda/render.cu");
    let at = render.find(HOOKS).expect("cuda/render.cu has a HOOKS line");
    let at = at + render[at..].find('\n').map_or(render.len() - at, |n| n + 1);
    [
        &render[..at],
        include_str!("../cuda/tone_curve.cu"),
        "\n",
        include_str!("../cuda/color_mixer.cu"),
        "\n",
        include_str!("../cuda/presence.cu"),
        "\n",
        include_str!("../cuda/bw.cu"),
        "\n",
        &render[at..],
    ]
    .concat()
});

/// EV added at Exposure 0 so a normally exposed RAW renders like the
/// camera's JPEG (middle grey near 0.46 sRGB). The integrator calibrates it
/// against camera JPEGs; the tone curve's white point follows it, so raw
/// clip at Exposure 0 always lands just below white.
pub const BASELINE_EV: f32 = 0.95;

/// Candidates per stats launch: bounds the accumulator memory (about 5 KB each).
const MAX_BATCH: usize = 4096;
/// Batches at least this large pack their parameter sets on several threads.
const PARALLEL_PACK: usize = 128;

/// Camera colour as the renderer needs it: the scene's [`ColorModel`] in
/// production, a stand-in in tests.
pub(crate) trait CameraColor: Send + Sync {
    fn rec2020_from_cam(&self, temperature: f32) -> [[f32; 3]; 3];
    fn wb_for(&self, temperature: f32, tint: f32) -> [f32; 3];
}

impl CameraColor for ColorModel {
    fn rec2020_from_cam(&self, temperature: f32) -> [[f32; 3]; 3] {
        ColorModel::rec2020_from_cam(self, temperature)
    }

    fn wb_for(&self, temperature: f32, tint: f32) -> [f32; 3] {
        ColorModel::wb_for(self, temperature, tint)
    }
}

trait CudaContextExt<T> {
    fn cuda(self, what: &str) -> Result<T>;
}

impl<T, E: std::fmt::Debug> CudaContextExt<T> for std::result::Result<T, E> {
    fn cuda(self, what: &str) -> Result<T> {
        self.map_err(|e| anyhow!("{what}: {e:?}"))
    }
}

struct Kernels {
    prep: CudaFunction,
    stats: CudaFunction,
    render8: CudaFunction,
    full: CudaFunction,
    hpass: CudaFunction,
    finish16: CudaFunction,
}

impl Kernels {
    fn load(gpu: &Gpu) -> Result<Self> {
        let module = gpu.module("render", &SOURCE)?;
        let f = |name: &str| module.load_function(name).cuda(name);
        Ok(Self {
            prep: f("prep_kernel")?,
            stats: f("stats_kernel")?,
            render8: f("render8_kernel")?,
            full: f("full_kernel")?,
            hpass: f("hpass_kernel")?,
            finish16: f("finish16_kernel")?,
        })
    }
}

fn check_image(img: &LinearImage, what: &str) -> Result<()> {
    if img.width == 0 || img.height == 0 || img.data.len() != 3 * img.width * img.height {
        return Err(anyhow!(
            "{what} image is malformed: {}x{} with {} samples",
            img.width,
            img.height,
            img.data.len()
        ));
    }
    Ok(())
}

fn blocks(n: usize, threads: usize) -> u32 {
    n.div_ceil(threads) as u32
}

/// What `prep_kernel` writes next to each preview pixel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Aux {
    None = 0,
    /// Scene log2 luminance for the structure statistic (search image).
    StructureLuma = 1,
    /// Grid log2 luminance guiding the full-resolution base upsampling.
    Guide = 2,
}

/// A preview image resident on the GPU, ready to render.
struct Preview {
    width: usize,
    height: usize,
    /// RGB and base layer per pixel (a float4).
    px: CudaSlice<f32>,
    /// Per-pixel extra plane chosen by [`Aux`].
    aux: CudaSlice<f32>,
    /// Texture/Clarity/Dehaze planes (`render/presence.rs`).
    pres: Planes,
    out8: Option<CudaSlice<u8>>,
}

impl Preview {
    fn upload(
        gpu: &Gpu,
        k: &Kernels,
        img: &LinearImage,
        grid: &BaseGrid,
        dev_grid: &CudaSlice<f32>,
        y_row: [f32; 3],
        aux: Aux,
    ) -> Result<Self> {
        let s = &gpu.stream;
        let n = img.width * img.height;
        let rgb = s.clone_htod(&img.data).cuda("upload preview")?;
        let mut px = s.alloc_zeros::<f32>(4 * n).cuda("alloc preview")?;
        let mut aux_plane = s
            .alloc_zeros::<f32>(if aux == Aux::None { 1 } else { n })
            .cuda("alloc preview plane")?;
        let (w, h) = (img.width as i32, img.height as i32);
        let (gnx, gny, gnz) = (grid.nx as i32, grid.ny as i32, grid.nz as i32);
        let cells = BaseGrid::cells_per_px(img.width, img.height);
        let mode = aux as i32;
        let mut b = s.launch_builder(&k.prep);
        b.arg(&rgb)
            .arg(&w)
            .arg(&h)
            .arg(dev_grid)
            .arg(&gnx)
            .arg(&gny)
            .arg(&gnz)
            .arg(&cells);
        b.arg(&y_row[0])
            .arg(&y_row[1])
            .arg(&y_row[2])
            .arg(&mut px)
            .arg(&mut aux_plane)
            .arg(&mode);
        let cfg = LaunchConfig {
            grid_dim: (blocks(n, 256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: arguments match `prep_kernel` in cuda/render.cu; buffers hold
        // n pixels (the aux plane only when written).
        unsafe { b.launch(cfg) }.cuda("prep_kernel")?;
        let pres = presence::preview_planes(gpu, &px, img.width, img.height, y_row)?;
        Ok(Self {
            width: img.width,
            height: img.height,
            px,
            aux: aux_plane,
            pres,
            out8: None,
        })
    }

    fn render8(&mut self, gpu: &Gpu, k: &Kernels, cand: &CudaSlice<f32>) -> Result<Rgb8Image> {
        let s = &gpu.stream;
        let n = self.width * self.height;
        let (w, h) = (self.width as i32, self.height as i32);
        let out8 = match &mut self.out8 {
            Some(buf) => buf,
            empty => empty.insert(s.alloc_zeros::<u8>(3 * n).cuda("alloc output")?),
        };
        let pres_on = self.pres.flag();
        let mut b = s.launch_builder(&k.render8);
        b.arg(&self.px).arg(&self.pres.data).arg(&pres_on);
        b.arg(&w).arg(&h).arg(cand).arg(&mut *out8);
        let cfg = LaunchConfig {
            grid_dim: (blocks(n, 256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: arguments match `render8_kernel`; `cand` holds one packed
        // candidate; `pres` holds a float4 per pixel when `pres_on`.
        unsafe { b.launch(cfg) }.cuda("render8_kernel")?;
        let data = s.clone_dtoh(&*out8).cuda("download render")?;
        Ok(Rgb8Image {
            width: self.width,
            height: self.height,
            data,
        })
    }
}

/// Keeps one photo's search and display images (and their base layers) on
/// the GPU so a candidate costs about a millisecond.
pub struct RenderEngine {
    gpu: Gpu,
    info: RawInfo,
    kernels: Kernels,
    color: Arc<dyn CameraColor>,
    reference: Reference,
    anchors: Anchors,
    search: Preview,
    display: Preview,
    sms: usize,
    host_cands: Vec<f32>,
    dev_cands: CudaSlice<f32>,
    dev_u32: CudaSlice<u32>,
    dev_f64: CudaSlice<f64>,
}

impl RenderEngine {
    pub fn new(gpu: &Gpu, scene: &Scene) -> Result<Self> {
        Self::with_color(gpu, scene, Arc::new(scene.color.clone()))
    }

    pub(crate) fn with_color(
        gpu: &Gpu,
        scene: &Scene,
        color: Arc<dyn CameraColor>,
    ) -> Result<Self> {
        check_image(&scene.search, "search")?;
        check_image(&scene.display, "display")?;
        let reference = Reference::new(
            &*color,
            scene.info.as_shot_temperature,
            scene.info.as_shot_tint,
            scene.wb_as_shot,
        )?;
        let kernels = Kernels::load(gpu)?;
        let y_row = reference.luminance_row();
        let grid = BaseGrid::build(&scene.display, y_row);
        let s = &gpu.stream;
        let dev_grid = s.clone_htod(&grid.data).cuda("upload grid")?;
        let search = Preview::upload(
            gpu,
            &kernels,
            &scene.search,
            &grid,
            &dev_grid,
            y_row,
            Aux::StructureLuma,
        )?;
        let display = Preview::upload(
            gpu,
            &kernels,
            &scene.display,
            &grid,
            &dev_grid,
            y_row,
            Aux::None,
        )?;
        let sms = gpu
            .ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .cuda("SM count")?
            .max(1) as usize;
        let dev_cands = s.alloc_zeros::<f32>(CAND_STRIDE).cuda("alloc candidates")?;
        let dev_u32 = s.alloc_zeros::<u32>(PART_U32).cuda("alloc stats")?;
        let dev_f64 = s.alloc_zeros::<f64>(N_SUMS).cuda("alloc stats")?;
        s.synchronize().cuda("engine setup")?;
        Ok(Self {
            gpu: gpu.clone(),
            info: scene.info.clone(),
            kernels,
            color,
            reference,
            anchors: Anchors::from_log2(grid.p01, grid.p99),
            search,
            display,
            sms,
            host_cands: Vec::new(),
            dev_cands,
            dev_u32,
            dev_f64,
        })
    }

    /// Renders every parameter set on the search image and measures it, batched.
    pub fn stats(&mut self, params: &[Params]) -> Result<Vec<Stats>> {
        let mut out = Vec::with_capacity(params.len());
        for batch in params.chunks(MAX_BATCH) {
            self.stats_batch(batch, &mut out)?;
        }
        Ok(out)
    }

    fn stats_batch(&mut self, params: &[Params], out: &mut Vec<Stats>) -> Result<()> {
        let n = params.len();
        if n == 0 {
            return Ok(());
        }
        let tiles = self.search.width.div_ceil(16) * self.search.height.div_ceil(16);
        // Enough blocks to fill the GPU even for a small batch.
        let chunks = (self.sms * 8).div_ceil(n).clamp(1, tiles.min(64));
        self.upload_candidates(params)?;
        let parts = n * chunks;
        let s = self.gpu.stream.clone();
        if self.dev_u32.len() < parts * PART_U32 {
            self.dev_u32 = s.alloc_zeros::<u32>(parts * PART_U32).cuda("alloc stats")?;
            self.dev_f64 = s.alloc_zeros::<f64>(parts * N_SUMS).cuda("alloc stats")?;
        }
        let (w, h, c) = (
            self.search.width as i32,
            self.search.height as i32,
            chunks as i32,
        );
        let pres_on = self.search.pres.flag();
        let mut b = s.launch_builder(&self.kernels.stats);
        b.arg(&self.search.px).arg(&self.search.aux);
        b.arg(&self.search.pres.data).arg(&pres_on);
        b.arg(&w).arg(&h);
        b.arg(&self.dev_cands)
            .arg(&c)
            .arg(&mut self.dev_u32)
            .arg(&mut self.dev_f64);
        let cfg = LaunchConfig {
            grid_dim: (chunks as u32, n as u32, 1),
            block_dim: (STATS_THREADS as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: arguments match `stats_kernel`; `dev_cands` holds n packed
        // candidates and the outputs hold n * chunks partial records.
        unsafe { b.launch(cfg) }.cuda("stats_kernel")?;
        let u = s
            .clone_dtoh(&self.dev_u32.slice(..parts * PART_U32))
            .cuda("download stats")?;
        let f = s
            .clone_dtoh(&self.dev_f64.slice(..parts * N_SUMS))
            .cuda("download stats")?;
        let mut counts = vec![0u32; PART_U32];
        let mut sums = [0f64; N_SUMS];
        for i in 0..n {
            counts.fill(0);
            sums.fill(0.0);
            for part in i * chunks..(i + 1) * chunks {
                for (acc, v) in counts
                    .iter_mut()
                    .zip(&u[part * PART_U32..(part + 1) * PART_U32])
                {
                    *acc += v;
                }
                for (acc, v) in sums.iter_mut().zip(&f[part * N_SUMS..(part + 1) * N_SUMS]) {
                    *acc += v;
                }
            }
            out.push(stats::finalize(&counts, &sums));
        }
        Ok(())
    }

    fn upload_candidates(&mut self, params: &[Params]) -> Result<()> {
        let len = params.len() * CAND_STRIDE;
        self.host_cands.resize(len, 0.0);
        let (color, reference, anchors) = (&*self.color, &self.reference, &self.anchors);
        let packed = self.host_cands.as_chunks_mut::<CAND_STRIDE>().0;
        let pack_all = |ps: &[Params], out: &mut [[f32; CAND_STRIDE]]| {
            for (p, dst) in ps.iter().zip(out) {
                pack(p, color, reference, anchors, dst);
            }
        };
        // Packing costs tens of microseconds per candidate (the tone tables),
        // as much as rendering it, so big batches pack on several cores.
        if params.len() >= PARALLEL_PACK {
            let threads = std::thread::available_parallelism().map_or(4, |n| n.get().min(8));
            let per = params.len().div_ceil(threads);
            std::thread::scope(|scope| {
                for (ps, out) in params.chunks(per).zip(packed.chunks_mut(per)) {
                    scope.spawn(move || pack_all(ps, out));
                }
            });
        } else {
            pack_all(params, packed);
        }
        let s = &self.gpu.stream;
        if self.dev_cands.len() < len {
            self.dev_cands = s.alloc_zeros::<f32>(len).cuda("alloc candidates")?;
        }
        s.memcpy_htod(&self.host_cands, &mut self.dev_cands.slice_mut(..len))
            .cuda("upload candidates")
    }

    /// Display-size sRGB render, unrotated, with Detail and Lens
    /// Corrections ([`finish::finish_preview`]).
    pub fn display(&mut self, params: &Params) -> Result<Rgb8Image> {
        self.upload_candidates(std::slice::from_ref(params))?;
        let mut img = self
            .display
            .render8(&self.gpu, &self.kernels, &self.dev_cands)?;
        let scale = img.width as f32 / self.info.width.max(1) as f32;
        finish::finish_preview(
            &self.gpu,
            &mut img,
            &Detail::of(params),
            &LensCorrection::new(params, &self.info),
            scale,
        )?;
        if params.is_bw() {
            bw::grey8(&mut img);
        }
        Ok(img)
    }

    /// Search-size sRGB render, unrotated.
    pub fn search(&mut self, params: &Params) -> Result<Rgb8Image> {
        self.upload_candidates(std::slice::from_ref(params))?;
        self.search
            .render8(&self.gpu, &self.kernels, &self.dev_cands)
    }
}

/// Full-resolution render with output sharpening and chroma noise
/// reduction, then Detail and Lens Corrections ([`finish::finish_full`]).
/// Looks the same as [`RenderEngine::display`] for the same params.
pub fn render_full(
    gpu: &Gpu,
    scene: &Scene,
    full: &LinearImage,
    params: &Params,
) -> Result<Rgb16Image> {
    let mut img = full::render(gpu, scene, full, params, &scene.color)?;
    finish::finish_full(
        gpu,
        &mut img,
        &Detail::of(params),
        &LensCorrection::new(params, &scene.info),
    )?;
    if params.is_bw() {
        bw::grey16(&mut img);
    }
    Ok(img)
}
