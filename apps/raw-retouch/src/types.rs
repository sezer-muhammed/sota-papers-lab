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

/// Lightroom Classic's develop settings, in Lightroom's own units and
/// ranges: the Basic panel, the parametric Tone Curve, Texture/Clarity/
/// Dehaze, the Color Mixer, Calibration, Detail, Lens Corrections and the
/// Black & White treatment with its gray mixer.
/// [`PARAM_SPECS`] describes every field, in this order (range, step,
/// default, XMP name). Fields after the Basic panel take Lightroom's
/// defaults when absent, so older clients and stored runs still parse.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "raw-retouch/")]
pub struct Params {
    // Basic.
    /// White balance temperature in kelvin (2000 to 50000).
    pub temperature: f32,
    /// White balance tint, -150 (green) to +150 (magenta).
    pub tint: f32,
    /// Exposure in EV, -5 to +5.
    pub exposure: f32,
    /// -100 to +100, like every slider below unless it says otherwise.
    pub contrast: f32,
    pub highlights: f32,
    pub shadows: f32,
    pub whites: f32,
    pub blacks: f32,
    pub vibrance: f32,
    pub saturation: f32,

    // Tone Curve, parametric: four regions split at 25, 50 and 75 (fixed at
    // Lightroom's defaults, see [`TONE_CURVE_SPLITS`]).
    /// Tone curve Highlights region (75 to 100).
    #[serde(default)]
    pub curve_highlights: f32,
    /// Tone curve Lights region (50 to 75).
    #[serde(default)]
    pub curve_lights: f32,
    /// Tone curve Darks region (25 to 50).
    #[serde(default)]
    pub curve_darks: f32,
    /// Tone curve Shadows region (0 to 25).
    #[serde(default)]
    pub curve_shadows: f32,

    // Presence (Lightroom shows these above Vibrance and Saturation).
    /// Texture: fine local contrast.
    #[serde(default)]
    pub texture: f32,
    /// Clarity: mid-scale local contrast.
    #[serde(default)]
    pub clarity: f32,
    /// Dehaze: removes (negative: adds) atmospheric haze.
    #[serde(default)]
    pub dehaze: f32,

    // Color Mixer (HSL), per hue band.
    /// Color Mixer hue shift of the red band; the bands below likewise.
    #[serde(default)]
    pub hue_red: f32,
    #[serde(default)]
    pub hue_orange: f32,
    #[serde(default)]
    pub hue_yellow: f32,
    #[serde(default)]
    pub hue_green: f32,
    #[serde(default)]
    pub hue_aqua: f32,
    #[serde(default)]
    pub hue_blue: f32,
    #[serde(default)]
    pub hue_purple: f32,
    #[serde(default)]
    pub hue_magenta: f32,
    /// Color Mixer saturation of the red band; the bands below likewise.
    #[serde(default)]
    pub saturation_red: f32,
    #[serde(default)]
    pub saturation_orange: f32,
    #[serde(default)]
    pub saturation_yellow: f32,
    #[serde(default)]
    pub saturation_green: f32,
    #[serde(default)]
    pub saturation_aqua: f32,
    #[serde(default)]
    pub saturation_blue: f32,
    #[serde(default)]
    pub saturation_purple: f32,
    #[serde(default)]
    pub saturation_magenta: f32,
    /// Color Mixer luminance of the red band; the bands below likewise.
    #[serde(default)]
    pub luminance_red: f32,
    #[serde(default)]
    pub luminance_orange: f32,
    #[serde(default)]
    pub luminance_yellow: f32,
    #[serde(default)]
    pub luminance_green: f32,
    #[serde(default)]
    pub luminance_aqua: f32,
    #[serde(default)]
    pub luminance_blue: f32,
    #[serde(default)]
    pub luminance_purple: f32,
    #[serde(default)]
    pub luminance_magenta: f32,

    // Calibration.
    /// Calibration Shadows tint, -100 (green) to +100 (magenta).
    #[serde(default)]
    pub shadow_tint: f32,
    /// Calibration Red Primary hue; the primaries below likewise.
    #[serde(default)]
    pub red_primary_hue: f32,
    #[serde(default)]
    pub red_primary_saturation: f32,
    #[serde(default)]
    pub green_primary_hue: f32,
    #[serde(default)]
    pub green_primary_saturation: f32,
    #[serde(default)]
    pub blue_primary_hue: f32,
    #[serde(default)]
    pub blue_primary_saturation: f32,

    // Detail: set per photo from its metadata (`finish::auto_detail`), not searched.
    /// Sharpening Amount, 0 to 150 (Lightroom's default 40).
    #[serde(default = "default_sharpening")]
    pub sharpening: f32,
    /// Sharpening Radius in pixels, 0.5 to 3.0 in steps of 0.1 (default 1.0).
    #[serde(default = "default_sharpen_radius")]
    pub sharpen_radius: f32,
    /// Sharpening Detail, 0 to 100 (default 25).
    #[serde(default = "default_sharpen_detail")]
    pub sharpen_detail: f32,
    /// Sharpening Masking, 0 to 100 (default 0).
    #[serde(default)]
    pub sharpen_masking: f32,
    /// Luminance noise reduction, 0 to 100 (default 0).
    #[serde(default)]
    pub luminance_nr: f32,
    /// Color noise reduction, 0 to 100 (default 25).
    #[serde(default = "default_color_nr")]
    pub color_nr: f32,

    // Lens Corrections: switches, 0 (off) or 1 (on) as Lightroom writes
    // them, set per photo from its metadata (`finish::auto_lens`).
    /// Enable Profile Corrections (distortion and vignetting).
    #[serde(default)]
    pub lens_profile: f32,
    /// Remove Chromatic Aberration.
    #[serde(default)]
    pub lens_ca: f32,

    // Black & White: Lightroom's Treatment switch and its B&W (gray) mixer.
    /// Treatment: 0 (colour) or 1 (black and white), a switch like the
    /// Lens Corrections ones. In black and white every pixel's grey is its
    /// luminance, brightened or darkened by the gray mixer per hue band,
    /// and the output has no colour (R = G = B).
    #[serde(default)]
    pub bw: f32,
    /// Gray mixer, -100 to +100: how much darker (-) or brighter (+) the
    /// red band's colours turn grey; the bands below likewise. Greys keep
    /// their tone; the stronger a colour, the more it moves.
    #[serde(default)]
    pub gray_red: f32,
    #[serde(default)]
    pub gray_orange: f32,
    #[serde(default)]
    pub gray_yellow: f32,
    #[serde(default)]
    pub gray_green: f32,
    #[serde(default)]
    pub gray_aqua: f32,
    #[serde(default)]
    pub gray_blue: f32,
    #[serde(default)]
    pub gray_purple: f32,
    #[serde(default)]
    pub gray_magenta: f32,
}

fn default_sharpening() -> f32 {
    40.0
}

fn default_sharpen_radius() -> f32 {
    1.0
}

fn default_sharpen_detail() -> f32 {
    25.0
}

fn default_color_nr() -> f32 {
    25.0
}

/// The parametric tone curve's split points, fixed at Lightroom's defaults
/// and written to the XMP next to the four region sliders.
pub const TONE_CURVE_SPLITS: [(&str, f32); 3] = [
    ("ParametricShadowSplit", 25.0),
    ("ParametricMidtoneSplit", 50.0),
    ("ParametricHighlightSplit", 75.0),
];

/// The Lightroom panel a setting belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, TS)]
#[ts(export, export_to = "raw-retouch/")]
pub enum ParamGroup {
    Basic,
    ToneCurve,
    Presence,
    ColorMixer,
    Calibration,
    Detail,
    Lens,
    BlackWhite,
}

/// One field of [`Params`]: what the UI, the XMP sidecar and the search
/// need to treat every setting alike. The table is [`PARAM_SPECS`]; the web
/// UI gets it as `PARAM_SPECS` from `web/bindings/raw-retouch/paramSpecs.ts`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, TS)]
#[ts(export, export_to = "raw-retouch/")]
pub struct ParamSpec {
    /// The field's name in [`Params`] (and its JSON key).
    pub key: &'static str,
    /// Lightroom's label for the slider.
    pub label: &'static str,
    /// Lightroom's XMP attribute, without the `crs:` prefix.
    pub crs: &'static str,
    pub group: ParamGroup,
    /// Lightroom's legal range and step.
    pub min: f32,
    pub max: f32,
    pub step: f32,
    /// Lightroom's default; none for white balance, whose default is as shot.
    pub neutral: Option<f32>,
    /// Whether the per-photo search varies it. Otherwise every candidate
    /// keeps the value its photo's search box fixes (`search::bounds`).
    pub searched: bool,
}

impl ParamSpec {
    /// A -100 to +100 slider in whole units, default 0, not searched.
    const fn slider(
        key: &'static str,
        label: &'static str,
        crs: &'static str,
        group: ParamGroup,
    ) -> Self {
        Self {
            key,
            label,
            crs,
            group,
            min: -100.0,
            max: 100.0,
            step: 1.0,
            neutral: Some(0.0),
            searched: false,
        }
    }

    const fn range(self, min: f32, max: f32, step: f32) -> Self {
        Self {
            min,
            max,
            step,
            ..self
        }
    }

    const fn default(self, neutral: Option<f32>) -> Self {
        Self { neutral, ..self }
    }

    const fn searched(self) -> Self {
        Self {
            searched: true,
            ..self
        }
    }

    /// `v` clamped to the legal range and rounded to the step, the way
    /// Lightroom stores it. NaN stays NaN.
    pub fn quantize(&self, v: f32) -> f32 {
        let v = v.clamp(self.min, self.max);
        if self.step >= 1.0 {
            v.round()
        } else {
            let per = (1.0 / self.step).round();
            (v * per).round() / per
        }
    }

    /// Indices into [`Params::to_array`] of every field in `groups`.
    pub fn indices_of(groups: &[ParamGroup]) -> Vec<usize> {
        (0..Params::COUNT)
            .filter(|&i| groups.contains(&PARAM_SPECS[i].group))
            .collect()
    }
}

use ParamGroup::{Basic, BlackWhite, Calibration, ColorMixer, Detail, Lens, Presence, ToneCurve};

/// Every field of [`Params`], in its order.
pub const PARAM_SPECS: [ParamSpec; Params::COUNT] = [
    ParamSpec::slider("temperature", "Temp", "Temperature", Basic)
        .range(2000.0, 50000.0, 1.0)
        .default(None)
        .searched(),
    ParamSpec::slider("tint", "Tint", "Tint", Basic)
        .range(-150.0, 150.0, 1.0)
        .default(None)
        .searched(),
    ParamSpec::slider("exposure", "Exposure", "Exposure2012", Basic)
        .range(-5.0, 5.0, 0.01)
        .searched(),
    ParamSpec::slider("contrast", "Contrast", "Contrast2012", Basic).searched(),
    ParamSpec::slider("highlights", "Highlights", "Highlights2012", Basic).searched(),
    ParamSpec::slider("shadows", "Shadows", "Shadows2012", Basic).searched(),
    ParamSpec::slider("whites", "Whites", "Whites2012", Basic).searched(),
    ParamSpec::slider("blacks", "Blacks", "Blacks2012", Basic).searched(),
    ParamSpec::slider("vibrance", "Vibrance", "Vibrance", Basic).searched(),
    ParamSpec::slider("saturation", "Saturation", "Saturation", Basic).searched(),
    ParamSpec::slider(
        "curve_highlights",
        "Highlights",
        "ParametricHighlights",
        ToneCurve,
    )
    .searched(),
    ParamSpec::slider("curve_lights", "Lights", "ParametricLights", ToneCurve).searched(),
    ParamSpec::slider("curve_darks", "Darks", "ParametricDarks", ToneCurve).searched(),
    ParamSpec::slider("curve_shadows", "Shadows", "ParametricShadows", ToneCurve).searched(),
    ParamSpec::slider("texture", "Texture", "Texture", Presence).searched(),
    ParamSpec::slider("clarity", "Clarity", "Clarity2012", Presence).searched(),
    ParamSpec::slider("dehaze", "Dehaze", "Dehaze", Presence).searched(),
    ParamSpec::slider("hue_red", "Red Hue", "HueAdjustmentRed", ColorMixer),
    ParamSpec::slider(
        "hue_orange",
        "Orange Hue",
        "HueAdjustmentOrange",
        ColorMixer,
    ),
    ParamSpec::slider(
        "hue_yellow",
        "Yellow Hue",
        "HueAdjustmentYellow",
        ColorMixer,
    ),
    ParamSpec::slider("hue_green", "Green Hue", "HueAdjustmentGreen", ColorMixer),
    ParamSpec::slider("hue_aqua", "Aqua Hue", "HueAdjustmentAqua", ColorMixer),
    ParamSpec::slider("hue_blue", "Blue Hue", "HueAdjustmentBlue", ColorMixer),
    ParamSpec::slider(
        "hue_purple",
        "Purple Hue",
        "HueAdjustmentPurple",
        ColorMixer,
    ),
    ParamSpec::slider(
        "hue_magenta",
        "Magenta Hue",
        "HueAdjustmentMagenta",
        ColorMixer,
    ),
    ParamSpec::slider(
        "saturation_red",
        "Red Saturation",
        "SaturationAdjustmentRed",
        ColorMixer,
    )
    .searched(),
    ParamSpec::slider(
        "saturation_orange",
        "Orange Saturation",
        "SaturationAdjustmentOrange",
        ColorMixer,
    )
    .searched(),
    ParamSpec::slider(
        "saturation_yellow",
        "Yellow Saturation",
        "SaturationAdjustmentYellow",
        ColorMixer,
    )
    .searched(),
    ParamSpec::slider(
        "saturation_green",
        "Green Saturation",
        "SaturationAdjustmentGreen",
        ColorMixer,
    )
    .searched(),
    ParamSpec::slider(
        "saturation_aqua",
        "Aqua Saturation",
        "SaturationAdjustmentAqua",
        ColorMixer,
    )
    .searched(),
    ParamSpec::slider(
        "saturation_blue",
        "Blue Saturation",
        "SaturationAdjustmentBlue",
        ColorMixer,
    )
    .searched(),
    ParamSpec::slider(
        "saturation_purple",
        "Purple Saturation",
        "SaturationAdjustmentPurple",
        ColorMixer,
    )
    .searched(),
    ParamSpec::slider(
        "saturation_magenta",
        "Magenta Saturation",
        "SaturationAdjustmentMagenta",
        ColorMixer,
    )
    .searched(),
    ParamSpec::slider(
        "luminance_red",
        "Red Luminance",
        "LuminanceAdjustmentRed",
        ColorMixer,
    )
    .searched(),
    ParamSpec::slider(
        "luminance_orange",
        "Orange Luminance",
        "LuminanceAdjustmentOrange",
        ColorMixer,
    )
    .searched(),
    ParamSpec::slider(
        "luminance_yellow",
        "Yellow Luminance",
        "LuminanceAdjustmentYellow",
        ColorMixer,
    )
    .searched(),
    ParamSpec::slider(
        "luminance_green",
        "Green Luminance",
        "LuminanceAdjustmentGreen",
        ColorMixer,
    )
    .searched(),
    ParamSpec::slider(
        "luminance_aqua",
        "Aqua Luminance",
        "LuminanceAdjustmentAqua",
        ColorMixer,
    )
    .searched(),
    ParamSpec::slider(
        "luminance_blue",
        "Blue Luminance",
        "LuminanceAdjustmentBlue",
        ColorMixer,
    )
    .searched(),
    ParamSpec::slider(
        "luminance_purple",
        "Purple Luminance",
        "LuminanceAdjustmentPurple",
        ColorMixer,
    )
    .searched(),
    ParamSpec::slider(
        "luminance_magenta",
        "Magenta Luminance",
        "LuminanceAdjustmentMagenta",
        ColorMixer,
    )
    .searched(),
    ParamSpec::slider("shadow_tint", "Shadows Tint", "ShadowTint", Calibration),
    ParamSpec::slider("red_primary_hue", "Red Primary Hue", "RedHue", Calibration),
    ParamSpec::slider(
        "red_primary_saturation",
        "Red Primary Saturation",
        "RedSaturation",
        Calibration,
    )
    .searched(),
    ParamSpec::slider(
        "green_primary_hue",
        "Green Primary Hue",
        "GreenHue",
        Calibration,
    ),
    ParamSpec::slider(
        "green_primary_saturation",
        "Green Primary Saturation",
        "GreenSaturation",
        Calibration,
    )
    .searched(),
    ParamSpec::slider(
        "blue_primary_hue",
        "Blue Primary Hue",
        "BlueHue",
        Calibration,
    ),
    ParamSpec::slider(
        "blue_primary_saturation",
        "Blue Primary Saturation",
        "BlueSaturation",
        Calibration,
    )
    .searched(),
    ParamSpec::slider("sharpening", "Sharpening", "Sharpness", Detail)
        .range(0.0, 150.0, 1.0)
        .default(Some(40.0)),
    ParamSpec::slider("sharpen_radius", "Radius", "SharpenRadius", Detail)
        .range(0.5, 3.0, 0.1)
        .default(Some(1.0)),
    ParamSpec::slider("sharpen_detail", "Detail", "SharpenDetail", Detail)
        .range(0.0, 100.0, 1.0)
        .default(Some(25.0)),
    ParamSpec::slider("sharpen_masking", "Masking", "SharpenEdgeMasking", Detail)
        .range(0.0, 100.0, 1.0),
    ParamSpec::slider("luminance_nr", "Luminance", "LuminanceSmoothing", Detail)
        .range(0.0, 100.0, 1.0),
    ParamSpec::slider("color_nr", "Color", "ColorNoiseReduction", Detail)
        .range(0.0, 100.0, 1.0)
        .default(Some(25.0)),
    ParamSpec::slider(
        "lens_profile",
        "Enable Profile Corrections",
        "LensProfileEnable",
        Lens,
    )
    .range(0.0, 1.0, 1.0),
    ParamSpec::slider(
        "lens_ca",
        "Remove Chromatic Aberration",
        "AutoLateralCA",
        Lens,
    )
    .range(0.0, 1.0, 1.0),
    ParamSpec::slider("bw", "Black & White", "ConvertToGrayscale", BlackWhite).range(0.0, 1.0, 1.0),
    ParamSpec::slider("gray_red", "B&W Red", "GrayMixerRed", BlackWhite).searched(),
    ParamSpec::slider("gray_orange", "B&W Orange", "GrayMixerOrange", BlackWhite).searched(),
    ParamSpec::slider("gray_yellow", "B&W Yellow", "GrayMixerYellow", BlackWhite).searched(),
    ParamSpec::slider("gray_green", "B&W Green", "GrayMixerGreen", BlackWhite).searched(),
    ParamSpec::slider("gray_aqua", "B&W Aqua", "GrayMixerAqua", BlackWhite).searched(),
    ParamSpec::slider("gray_blue", "B&W Blue", "GrayMixerBlue", BlackWhite).searched(),
    ParamSpec::slider("gray_purple", "B&W Purple", "GrayMixerPurple", BlackWhite).searched(),
    ParamSpec::slider(
        "gray_magenta",
        "B&W Magenta",
        "GrayMixerMagenta",
        BlackWhite,
    )
    .searched(),
];

/// [`Params::to_array`] and [`Params::from_array`] from the field list, in
/// [`PARAM_SPECS`] order (a test checks the order).
macro_rules! param_arrays {
    ($($field:ident),* $(,)?) => {
        impl Params {
            /// Every field, in [`PARAM_SPECS`] order.
            pub fn to_array(&self) -> [f32; Self::COUNT] {
                [$(self.$field),*]
            }

            pub fn from_array(a: [f32; Self::COUNT]) -> Self {
                let [$($field),*] = a;
                Self { $($field),* }
            }
        }
    };
}

param_arrays!(
    temperature,
    tint,
    exposure,
    contrast,
    highlights,
    shadows,
    whites,
    blacks,
    vibrance,
    saturation,
    curve_highlights,
    curve_lights,
    curve_darks,
    curve_shadows,
    texture,
    clarity,
    dehaze,
    hue_red,
    hue_orange,
    hue_yellow,
    hue_green,
    hue_aqua,
    hue_blue,
    hue_purple,
    hue_magenta,
    saturation_red,
    saturation_orange,
    saturation_yellow,
    saturation_green,
    saturation_aqua,
    saturation_blue,
    saturation_purple,
    saturation_magenta,
    luminance_red,
    luminance_orange,
    luminance_yellow,
    luminance_green,
    luminance_aqua,
    luminance_blue,
    luminance_purple,
    luminance_magenta,
    shadow_tint,
    red_primary_hue,
    red_primary_saturation,
    green_primary_hue,
    green_primary_saturation,
    blue_primary_hue,
    blue_primary_saturation,
    sharpening,
    sharpen_radius,
    sharpen_detail,
    sharpen_masking,
    luminance_nr,
    color_nr,
    lens_profile,
    lens_ca,
    bw,
    gray_red,
    gray_orange,
    gray_yellow,
    gray_green,
    gray_aqua,
    gray_blue,
    gray_purple,
    gray_magenta,
);

impl Params {
    /// Every field.
    pub const COUNT: usize = 65;
    /// The Basic panel: the first `BASIC` fields, the ones Claude plans and
    /// adjusts. The search varies these and every other field
    /// [`PARAM_SPECS`] marks `searched`: the Tone Curve, Presence, the Color
    /// Mixer's saturation and luminance and the primaries' saturation (and
    /// the gray mixer, in the black-and-white look only).
    pub const BASIC: usize = 10;
    /// Field names in [`PARAM_SPECS`] order; the first [`Params::BASIC`]
    /// are the Basic panel.
    pub const NAMES: [&'static str; Self::COUNT] = {
        let mut names = [""; Self::COUNT];
        let mut i = 0;
        while i < Self::COUNT {
            names[i] = PARAM_SPECS[i].key;
            i += 1;
        }
        names
    };

    /// As-shot white balance and every other setting at Lightroom's default
    /// (zero, except Detail's sharpening and colour noise reduction).
    pub fn neutral(temperature: f32, tint: f32) -> Self {
        let mut a = PARAM_SPECS.map(|s| s.neutral.unwrap_or(0.0));
        a[0] = temperature;
        a[1] = tint;
        Self::from_array(a)
    }

    /// The Basic panel's sliders, in [`Params::NAMES`] order.
    pub fn basic(&self) -> [f32; Self::BASIC] {
        let a = self.to_array();
        std::array::from_fn(|i| a[i])
    }

    /// `self` with the Basic panel set to `basic` and every other field kept.
    pub fn with_basic(&self, basic: [f32; Self::BASIC]) -> Self {
        let mut a = self.to_array();
        a[..Self::BASIC].copy_from_slice(&basic);
        Self::from_array(a)
    }

    /// Clamped to Lightroom's legal ranges, not rounded.
    pub fn clamped(&self) -> Self {
        let mut a = self.to_array();
        for (v, s) in a.iter_mut().zip(&PARAM_SPECS) {
            *v = v.clamp(s.min, s.max);
        }
        Self::from_array(a)
    }

    /// Whether the Black & White treatment is on.
    pub fn is_bw(&self) -> bool {
        self.bw >= 0.5
    }

    /// Clamped to Lightroom's legal ranges and rounded to its steps (whole
    /// kelvin, 0.01 EV, whole slider units, 0.1 px of sharpening radius),
    /// so an exported XMP loses nothing.
    pub fn quantized(&self) -> Self {
        let mut a = self.to_array();
        for (v, s) in a.iter_mut().zip(&PARAM_SPECS) {
            *v = s.quantize(*v);
        }
        Self::from_array(a)
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
    /// When the photo was taken, ISO 8601 from EXIF DateTimeOriginal:
    /// "2025-10-12T14:33:05", with the UTC offset ("+03:00") when the file
    /// has OffsetTimeOriginal, else the camera's local time.
    pub captured_at: Option<String>,
    /// Where it was taken, when the file carries a GPS position.
    #[serde(default)]
    pub gps: Option<Gps>,
    /// The nearest city to `gps`, e.g. "Bornova, İzmir, Türkiye", from a
    /// bundled offline table (`crate::place`; the position never leaves
    /// the host).
    #[serde(default)]
    pub place: Option<String>,
    pub as_shot_temperature: f32,
    pub as_shot_tint: f32,
    /// Share of pixels clipped in at least one channel in the RAW (0 to 1).
    pub raw_clipped_fraction: f32,
}

/// A GPS position from the file's EXIF.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "raw-retouch/")]
pub struct Gps {
    /// Degrees, north positive.
    pub lat: f64,
    /// Degrees, east positive.
    pub lon: f64,
    /// Metres above sea level (negative below).
    pub alt_m: Option<f64>,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_order_matches_the_spec_table() {
        let json = serde_json::to_value(Params::neutral(5000.0, 0.0)).unwrap();
        assert_eq!(json.as_object().unwrap().len(), Params::COUNT);
        for i in 0..Params::COUNT {
            let mut a = [0.0; Params::COUNT];
            a[i] = 1.0;
            let json = serde_json::to_value(Params::from_array(a)).unwrap();
            assert_eq!(json[PARAM_SPECS[i].key], 1.0, "{}", PARAM_SPECS[i].key);
            assert_eq!(Params::from_array(a).to_array(), a);
        }
        assert!(
            PARAM_SPECS[..Params::BASIC]
                .iter()
                .all(|s| s.group == ParamGroup::Basic && s.searched)
        );
        assert!(
            PARAM_SPECS[Params::BASIC..]
                .iter()
                .all(|s| s.group != ParamGroup::Basic)
        );
        let mut crs: Vec<_> = PARAM_SPECS.iter().map(|s| s.crs).collect();
        crs.sort_unstable();
        crs.dedup();
        assert_eq!(crs.len(), Params::COUNT, "XMP names are unique");
    }

    #[test]
    fn old_basic_only_json_parses_with_lightroom_defaults() {
        let old = r#"{"temperature": 5200, "tint": 3, "exposure": 0.35, "contrast": 10,
            "highlights": -40, "shadows": 30, "whites": 5, "blacks": -5,
            "vibrance": 20, "saturation": 0}"#;
        let p: Params = serde_json::from_str(old).unwrap();
        let want = Params {
            exposure: 0.35,
            contrast: 10.0,
            highlights: -40.0,
            shadows: 30.0,
            whites: 5.0,
            blacks: -5.0,
            vibrance: 20.0,
            ..Params::neutral(5200.0, 3.0)
        };
        assert_eq!(p, want);
        assert_eq!(
            (p.sharpening, p.sharpen_radius, p.sharpen_detail),
            (40.0, 1.0, 25.0)
        );
        assert_eq!((p.color_nr, p.lens_profile, p.lens_ca), (25.0, 0.0, 0.0));
        let back: Params = serde_json::from_value(serde_json::to_value(p).unwrap()).unwrap();
        assert_eq!(back, p);
    }

    #[test]
    fn neutral_is_legal_and_on_lightroom_steps() {
        let n = Params::neutral(5200.0, 3.0);
        assert_eq!(n.quantized(), n);
        assert_eq!(n.clamped(), n);
        for (s, v) in PARAM_SPECS.iter().zip(n.to_array()).skip(2) {
            assert_eq!(Some(v), s.neutral, "{}", s.key);
        }
        let q = Params {
            exposure: 0.346,
            sharpen_radius: 1.26,
            lens_profile: 0.7,
            hue_red: -140.0,
            ..n
        }
        .quantized();
        assert_eq!((q.exposure, q.sharpen_radius), (0.35, 1.3));
        assert_eq!((q.lens_profile, q.hue_red), (1.0, -100.0));
        assert_eq!(
            ParamSpec::indices_of(&[ParamGroup::Basic]),
            (0..Params::BASIC).collect::<Vec<_>>()
        );
    }

    /// The spec table as a TypeScript constant next to the ts-rs bindings,
    /// regenerated by `cargo test` like them.
    #[test]
    fn export_param_specs_to_typescript() {
        let dir = std::env::var("TS_RS_EXPORT_DIR").map_or_else(
            |_| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../web/bindings"),
            std::path::PathBuf::from,
        );
        let rows: Vec<String> = PARAM_SPECS
            .iter()
            .map(|s| format!("  {},", serde_json::to_string(s).unwrap()))
            .collect();
        let ts = format!(
            "// Generated by `cargo test` from PARAM_SPECS in apps/raw-retouch/src/types.rs. \
             Do not edit this file manually.\n\
             import type {{ ParamSpec }} from \"./ParamSpec\";\n\n\
             /** Every field of Params, in its order. */\n\
             export const PARAM_SPECS: ParamSpec[] = [\n{}\n];\n",
            rows.join("\n")
        );
        let dir = dir.join("raw-retouch");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("paramSpecs.ts"), ts).unwrap();
    }
}
