//! Claude as analyst, look planner and pairwise judge, through the
//! subscription-backed `claude -p` CLI. Never the numeric optimizer.
//! Owner: judge.

use std::path::PathBuf;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use spl_core::Result;
use ts_rs::TS;

use crate::types::Params;

#[derive(Debug, Clone)]
pub struct JudgeOptions {
    /// `claude --model` alias or full name ("opus", "sonnet", ...).
    pub model: String,
    /// `claude --effort` (low, medium, high, xhigh, max); None = CLI default.
    pub effort: Option<String>,
    /// Per call, including retries.
    pub timeout_s: u64,
    /// Verdicts are cached here by a hash of model, prompt, schema and images.
    pub cache_dir: Option<PathBuf>,
}

/// One image shown to Claude. `label` is how prompts refer to it ("A", "B", ...).
pub struct JudgeImage {
    pub label: String,
    pub jpeg: Vec<u8>,
}

/// What Claude is told about the photo besides the pixels.
#[derive(Debug, Clone, Serialize)]
pub struct PhotoBrief {
    /// "Sony ILCE-7M3, FE 16-35mm F2.8 GM"
    pub camera: String,
    /// "1/250 s, f/4, ISO 400, 35 mm"
    pub exposure: String,
    /// Histogram percentiles and clipping of the neutral render.
    pub stats: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "raw-retouch/")]
pub struct PlannedLook {
    pub name: String,
    pub rationale: String,
    /// Absolute Lightroom values, clamped to the bounds given to `plan`.
    pub params: Params,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanVerdict {
    /// One or two sentences on the scene and what it needs.
    pub scene: String,
    pub looks: Vec<PlannedLook>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankVerdict {
    /// Candidate indices, best first (Borda count over the order-swapped calls).
    pub order: Vec<usize>,
    pub winner: usize,
    /// Share of calls whose first choice was the winner (0 to 1).
    pub agreement: f32,
    pub reasons: String,
    pub scene: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefineVerdict {
    /// Absolute Lightroom values, clamped to the bounds given to `refine`.
    pub params: Params,
    pub rationale: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompareVerdict {
    /// True only when B won in both orders.
    pub prefer_b: bool,
    pub votes_for_b: u32,
    pub votes: u32,
    pub reasons: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct JudgeUsage {
    pub calls: u32,
    pub cache_hits: u32,
    /// What the CLI reports as `total_cost_usd` (subscription usage, not a bill).
    pub cost_usd: f64,
    pub seconds: f64,
}

pub struct Claude {}

impl Claude {
    /// Finds the CLI ($SPL_CLAUDE_BIN, then PATH, then ~/.local/bin/claude).
    pub fn locate(opts: JudgeOptions) -> Result<Self> {
        let _ = opts;
        unimplemented!("Claude::locate")
    }

    /// Reads the neutral and auto renders plus stats; proposes up to three
    /// visibly different looks inside `lo`..`hi`.
    pub async fn plan(
        &self,
        photo: &PhotoBrief,
        neutral: &JudgeImage,
        auto: &JudgeImage,
        auto_params: &Params,
        lo: &Params,
        hi: &Params,
    ) -> Result<PlanVerdict> {
        let _ = (photo, neutral, auto, auto_params, lo, hi);
        unimplemented!("Claude::plan")
    }

    /// Ranks 2 to 8 candidates: two calls with the order reversed, Borda count.
    pub async fn rank(&self, photo: &PhotoBrief, candidates: &[JudgeImage]) -> Result<RankVerdict> {
        let _ = (photo, candidates);
        unimplemented!("Claude::rank")
    }

    /// One targeted improvement of the current best, inside `lo`..`hi`.
    pub async fn refine(
        &self,
        photo: &PhotoBrief,
        current: &JudgeImage,
        params: &Params,
        lo: &Params,
        hi: &Params,
    ) -> Result<RefineVerdict> {
        let _ = (photo, current, params, lo, hi);
        unimplemented!("Claude::refine")
    }

    /// Blind A/B in both orders.
    pub async fn compare(
        &self,
        photo: &PhotoBrief,
        a: &JudgeImage,
        b: &JudgeImage,
    ) -> Result<CompareVerdict> {
        let _ = (photo, a, b);
        unimplemented!("Claude::compare")
    }

    pub fn usage(&self) -> JudgeUsage {
        unimplemented!("Claude::usage")
    }
}
