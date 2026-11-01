//! The app's ops and their wire types (also the UI's TypeScript bindings).
//! Owner: integration.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use spl_core::{ArtifactRef, Op, Result, RunCx};
use ts_rs::TS;

use crate::types::{Params, RawInfo};

fn yes() -> bool {
    true
}

fn default_model() -> String {
    "opus".into()
}

fn default_evals() -> u32 {
    1500
}

fn default_quality() -> u8 {
    95
}

#[derive(Debug, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "raw-retouch/")]
pub struct InspectInput {
    /// Artifact id of the RAW file (upload it with `POST /v1/artifacts`).
    pub raw: String,
}

#[derive(Debug, Serialize, JsonSchema, TS)]
#[ts(export, export_to = "raw-retouch/")]
pub struct InspectOutput {
    pub info: RawInfo,
    /// As-shot white balance, every other slider at zero.
    pub neutral: Params,
    /// Display-size JPEG at neutral settings, upright.
    pub preview: ArtifactRef,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, TS)]
#[ts(export, export_to = "raw-retouch/")]
pub struct ExportOptions {
    /// Full-resolution JPEG.
    #[serde(default = "yes")]
    pub jpeg: bool,
    #[serde(default = "default_quality")]
    pub jpeg_quality: u8,
    /// Full-resolution 16-bit TIFF (large).
    #[serde(default)]
    pub tiff16: bool,
    /// Lightroom / Camera Raw XMP sidecar with the sliders.
    #[serde(default = "yes")]
    pub xmp: bool,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            jpeg: true,
            jpeg_quality: default_quality(),
            tiff16: false,
            xmp: true,
        }
    }
}

#[derive(Debug, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "raw-retouch/")]
pub struct AutoInput {
    /// Artifact id of the RAW file.
    pub raw: String,
    /// Original file name; names the downloads (`<stem>.jpg`, `<stem>.xmp`).
    #[serde(default)]
    #[ts(optional)]
    pub filename: Option<String>,
    /// Let Claude plan looks, pick the winner and refine it. Off: the objective picks.
    #[serde(default = "yes")]
    pub use_claude: bool,
    /// `claude --model` alias: "opus", "sonnet", ...
    #[serde(default = "default_model")]
    pub model: String,
    /// Claude proposes up to three looks before the search.
    #[serde(default = "yes")]
    pub plan: bool,
    /// Claude suggests one refinement of the winner, kept only if it wins a blind A/B.
    #[serde(default = "yes")]
    pub refine: bool,
    /// Parameter-search evaluations per look.
    #[serde(default = "default_evals")]
    pub evals_per_look: u32,
    #[serde(default)]
    pub export: ExportOptions,
}

#[derive(Debug, Serialize, JsonSchema, TS)]
#[ts(export, export_to = "raw-retouch/")]
pub struct CandidateSummary {
    pub id: String,
    pub label: String,
    /// neutral, auto, look, claude or refined.
    pub source: String,
    pub params: Params,
    /// Objective score, higher is better.
    pub score: f32,
    /// 1 = Claude's favourite; null when Claude did not rank it.
    pub rank: Option<u32>,
    /// Display-size JPEG, upright.
    pub preview: ArtifactRef,
    pub note: Option<String>,
}

#[derive(Debug, Serialize, JsonSchema, TS)]
#[ts(export, export_to = "raw-retouch/")]
pub struct JudgeSummary {
    pub model: String,
    /// Claude's read of the scene.
    pub scene: Option<String>,
    /// Why the winner won.
    pub reasons: String,
    /// Share of the order-swapped rankings that agreed on the winner (0 to 1).
    pub agreement: Option<f32>,
    /// Whether Claude's refinement won its blind A/B and became the result.
    pub refined: bool,
    pub refine_note: Option<String>,
    pub calls: u32,
    pub cache_hits: u32,
    pub cost_usd: f64,
    /// Set when Claude failed and the objective chose instead.
    pub error: Option<String>,
}

#[derive(Debug, Default, Serialize, JsonSchema, TS)]
#[ts(export, export_to = "raw-retouch/")]
pub struct Timings {
    pub decode_ms: f64,
    pub develop_ms: f64,
    pub search_ms: f64,
    pub judge_ms: f64,
    pub export_ms: f64,
    pub total_ms: f64,
}

#[derive(Debug, Serialize, JsonSchema, TS)]
#[ts(export, export_to = "raw-retouch/")]
pub struct AutoOutput {
    pub info: RawInfo,
    /// The chosen edit, in Lightroom units.
    pub params: Params,
    /// Id of the chosen candidate.
    pub chosen: String,
    pub candidates: Vec<CandidateSummary>,
    pub judge: Option<JudgeSummary>,
    /// Display-size JPEGs, upright: neutral settings and the chosen edit.
    pub before: ArtifactRef,
    pub after: ArtifactRef,
    pub jpeg: Option<ArtifactRef>,
    pub tiff: Option<ArtifactRef>,
    pub xmp: Option<ArtifactRef>,
    /// Download name stem: the RAW's file name without extension.
    pub stem: String,
    pub timings: Timings,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export, export_to = "raw-retouch/")]
pub enum RenderSize {
    /// Display size, fast: for live slider edits.
    #[default]
    Preview,
    /// Full resolution, plus the export files.
    Full,
}

#[derive(Debug, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "raw-retouch/")]
pub struct RenderInput {
    pub raw: String,
    pub params: Params,
    #[serde(default)]
    pub size: RenderSize,
    #[serde(default)]
    #[ts(optional)]
    pub filename: Option<String>,
    /// Used when `size` is "full".
    #[serde(default)]
    pub export: ExportOptions,
}

#[derive(Debug, Serialize, JsonSchema, TS)]
#[ts(export, export_to = "raw-retouch/")]
pub struct RenderOutput {
    /// Display-size JPEG, upright.
    pub preview: ArtifactRef,
    pub jpeg: Option<ArtifactRef>,
    pub tiff: Option<ArtifactRef>,
    pub xmp: Option<ArtifactRef>,
    pub stem: String,
    /// Upright size of `jpeg` (or of `preview` when there is no jpeg).
    pub width: u32,
    pub height: u32,
}

pub struct Inspect;

impl Op for Inspect {
    const ID: &'static str = "raw_retouch.inspect";
    const SUMMARY: &'static str = "Read a RAW file's metadata and render it at neutral settings";

    type Input = InspectInput;
    type Output = InspectOutput;

    async fn run(&self, cx: &RunCx, input: InspectInput) -> Result<InspectOutput> {
        let _ = (cx, input);
        unimplemented!("raw_retouch.inspect")
    }
}

pub struct Auto;

impl Op for Auto {
    const ID: &'static str = "raw_retouch.auto";
    const SUMMARY: &'static str =
        "Find the most pleasing Lightroom-style edit of a RAW photo and export it";

    type Input = AutoInput;
    type Output = AutoOutput;

    async fn run(&self, cx: &RunCx, input: AutoInput) -> Result<AutoOutput> {
        let _ = (cx, input);
        unimplemented!("raw_retouch.auto")
    }
}

pub struct Render;

impl Op for Render {
    const ID: &'static str = "raw_retouch.render";
    const SUMMARY: &'static str = "Render a RAW photo with given Lightroom-style sliders";

    type Input = RenderInput;
    type Output = RenderOutput;

    async fn run(&self, cx: &RunCx, input: RenderInput) -> Result<RenderOutput> {
        let _ = (cx, input);
        unimplemented!("raw_retouch.render")
    }
}
