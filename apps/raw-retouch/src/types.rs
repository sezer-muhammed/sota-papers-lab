//! The app's shared vocabulary. Every module speaks these types, and the
//! serializable ones are also the web UI's TypeScript bindings
//! (`web/bindings/raw-retouch/`), so change them together with every user.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use spl_core::{Result, anyhow};
use spl_gpu::cudarc::driver::{CudaContext, CudaModule, CudaStream};
use ts_rs::TS;

/// Long edge of the image the search scores thousands of times per photo.
pub const SEARCH_EDGE: usize = 768;
/// Long edge of the image people and Claude look at.
pub const DISPLAY_EDGE: usize = 1536;

/// Lightroom Classic's Basic panel, in Lightroom's own units and ranges.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "raw-retouch/")]
pub struct Params {
    /// White balance temperature in kelvin (2000 to 50000).
    pub temperature: f32,
    /// White balance tint, -150 (green) to +150 (magenta).
    pub tint: f32,
    /// Exposure in EV, -5 to +5.
    pub exposure: f32,
    /// -100 to +100, like every slider below.
    pub contrast: f32,
    pub highlights: f32,
    pub shadows: f32,
    pub whites: f32,
    pub blacks: f32,
    pub vibrance: f32,
    pub saturation: f32,
}

impl Params {
    pub const COUNT: usize = 10;
    pub const NAMES: [&'static str; Self::COUNT] = [
        "temperature",
        "tint",
        "exposure",
        "contrast",
        "highlights",
        "shadows",
        "whites",
        "blacks",
        "vibrance",
        "saturation",
    ];

    /// As-shot white balance and every other slider at zero.
    pub fn neutral(temperature: f32, tint: f32) -> Self {
        Self {
            temperature,
            tint,
            exposure: 0.0,
            contrast: 0.0,
            highlights: 0.0,
            shadows: 0.0,
            whites: 0.0,
            blacks: 0.0,
            vibrance: 0.0,
            saturation: 0.0,
        }
    }

    pub fn to_array(&self) -> [f32; Self::COUNT] {
        [
            self.temperature,
            self.tint,
            self.exposure,
            self.contrast,
            self.highlights,
            self.shadows,
            self.whites,
            self.blacks,
            self.vibrance,
            self.saturation,
        ]
    }

    pub fn from_array(a: [f32; Self::COUNT]) -> Self {
        Self {
            temperature: a[0],
            tint: a[1],
            exposure: a[2],
            contrast: a[3],
            highlights: a[4],
            shadows: a[5],
            whites: a[6],
            blacks: a[7],
            vibrance: a[8],
            saturation: a[9],
        }
    }

    /// Clamped to Lightroom's legal ranges and rounded to its steps (whole
    /// kelvin, 0.01 EV, whole slider units), so an exported XMP loses nothing.
    pub fn quantized(&self) -> Self {
        let whole = |v: f32| v.clamp(-100.0, 100.0).round();
        Self {
            temperature: self.temperature.clamp(2000.0, 50000.0).round(),
            tint: self.tint.clamp(-150.0, 150.0).round(),
            exposure: (self.exposure.clamp(-5.0, 5.0) * 100.0).round() / 100.0,
            contrast: whole(self.contrast),
            highlights: whole(self.highlights),
            shadows: whole(self.shadows),
            whites: whole(self.whites),
            blacks: whole(self.blacks),
            vibrance: whole(self.vibrance),
            saturation: whole(self.saturation),
        }
    }
}

/// What the camera recorded, plus what decoding learned about the file.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "raw-retouch/")]
pub struct RawInfo {
    pub make: String,
    pub model: String,
    pub lens: Option<String>,
    /// Developed size after the camera's default crop, before orientation.
    pub width: u32,
    pub height: u32,
    /// EXIF orientation 1 to 8 (1 = upright as stored).
    pub orientation: u8,
    pub iso: Option<u32>,
    pub exposure_time_s: Option<f32>,
    pub f_number: Option<f32>,
    pub focal_length_mm: Option<f32>,
    /// EXIF DateTimeOriginal as the camera wrote it.
    pub captured_at: Option<String>,
    pub as_shot_temperature: f32,
    pub as_shot_tint: f32,
    /// Share of pixels clipped in at least one channel in the RAW (0 to 1).
    pub raw_clipped_fraction: f32,
}

/// Linear camera RGB, interleaved, row-major, unrotated. As-shot white
/// balance is applied, so a neutral surface reads (k, k, k). 1.0 is the clip
/// point: clipped highlights are exactly 1.0 in every channel.
#[derive(Debug, Clone, Default)]
pub struct LinearImage {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

/// sRGB, 8 bits, interleaved RGB, row-major.
#[derive(Debug, Clone, Default)]
pub struct Rgb8Image {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

/// sRGB, 16 bits (0 to 65535), interleaved RGB, row-major.
#[derive(Debug, Clone, Default)]
pub struct Rgb16Image {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u16>,
}

/// Everything the renderer needs about one photo, at preview sizes.
pub struct Scene {
    pub info: RawInfo,
    pub color: crate::color::ColorModel,
    /// As-shot white balance multipliers (green = 1), already applied to the images.
    pub wb_as_shot: [f32; 3],
    /// Long edge [`SEARCH_EDGE`].
    pub search: LinearImage,
    /// Long edge [`DISPLAY_EDGE`].
    pub display: LinearImage,
}

/// What the objective sees of one render: measured on the GPU on the search
/// image, in display-referred sRGB.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "raw-retouch/")]
pub struct Stats {
    /// 64-bin histogram of Rec.709 luma of the sRGB-encoded output, summing to 1.
    pub luma_hist: Vec<f32>,
    pub luma_mean: f32,
    pub luma_p01: f32,
    pub luma_p05: f32,
    pub luma_p50: f32,
    pub luma_p95: f32,
    pub luma_p99: f32,
    /// Share of pixels with any channel at or above 254/255.
    pub clip_hi: f32,
    /// `clip_hi`, counting only pixels that were not already clipped in the RAW.
    pub clip_hi_new: f32,
    /// Share of pixels with luma at or below 2/255.
    pub clip_lo: f32,
    /// Hasler and Süsstrunk colorfulness on 0 to 255 sRGB.
    pub colorfulness: f32,
    /// Oklab chroma: mean and 95th percentile.
    pub chroma_mean: f32,
    pub chroma_p95: f32,
    /// Share of pixels outside the sRGB gamut before gamut mapping.
    pub out_of_gamut: f32,
    /// Mean Oklab a and b of near-neutral pixels: a colour-cast estimate.
    pub cast_a: f32,
    pub cast_b: f32,
    /// TMQI-style structural fidelity to the linear RAW, 0 to 1 (1 = all local structure kept).
    pub structure: f32,
    /// Mean local standard deviation of luma at a mid scale.
    pub local_contrast: f32,
    /// Shannon entropy of the 64-bin luma histogram, in bits: 0 (one
    /// tone) to 6 (every bin equally full).
    pub entropy: f32,
}

/// Device 0's CUDA context, shared by every run in the process, with a
/// stream of this run's own and a per-process cache of compiled modules.
#[derive(Clone)]
pub struct Gpu {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
}

static CONTEXT: Mutex<Option<Arc<CudaContext>>> = Mutex::new(None);
static MODULES: Mutex<Option<HashMap<&'static str, Arc<CudaModule>>>> = Mutex::new(None);

impl Gpu {
    pub fn open() -> Result<Self> {
        let ctx = {
            let mut slot = CONTEXT
                .lock()
                .map_err(|_| anyhow!("GPU context lock poisoned"))?;
            match &*slot {
                Some(ctx) => ctx.clone(),
                None => {
                    let ctx = spl_gpu::context(0)?;
                    *slot = Some(ctx.clone());
                    ctx
                }
            }
        };
        let stream = ctx
            .new_stream()
            .map_err(|e| anyhow!("create CUDA stream: {e:?}"))?;
        Ok(Self { ctx, stream })
    }

    /// Compiles `source` with NVRTC on first use and keeps it for the process.
    /// `name` must be unique per source file.
    pub fn module(&self, name: &'static str, source: &'static str) -> Result<Arc<CudaModule>> {
        let mut cache = MODULES
            .lock()
            .map_err(|_| anyhow!("GPU module lock poisoned"))?;
        let cache = cache.get_or_insert_with(HashMap::new);
        if let Some(module) = cache.get(name) {
            return Ok(module.clone());
        }
        let module = spl_gpu::load_cuda_source(&self.ctx, source)?;
        cache.insert(name, module.clone());
        Ok(module)
    }
}
