//! Per-photo parameter search: seeds, CMA-ES per look, diverse finalists.
//! Owner: search.

use serde::{Deserialize, Serialize};
use spl_core::Result;

use crate::objective::{LookSpec, Score};
use crate::render::RenderEngine;
use crate::types::{Params, Scene, Stats};

#[derive(Debug, Clone)]
pub struct SearchConfig {
    pub evals_per_look: usize,
    pub max_finalists: usize,
    pub seed: u64,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            evals_per_look: 1500,
            max_finalists: 6,
            seed: 7,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateSource {
    Neutral,
    Auto,
    Look,
    Claude,
    Refined,
}

impl CandidateSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Neutral => "neutral",
            Self::Auto => "auto",
            Self::Look => "look",
            Self::Claude => "claude",
            Self::Refined => "refined",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    pub id: String,
    pub label: String,
    pub source: CandidateSource,
    pub params: Params,
    pub score: Score,
    pub stats: Stats,
}

pub struct SearchResult {
    pub neutral: Candidate,
    pub auto: Candidate,
    /// One optimized candidate per look, in the order of the looks given.
    pub looks: Vec<Candidate>,
    /// Visibly different candidates for judging, best first; always includes auto.
    pub finalists: Vec<Candidate>,
    pub evaluations: usize,
}

/// Legal search box for this photo in Lightroom units: expert ranges
/// (PPR10K) around the as-shot white balance. Returns (low, high).
pub fn bounds(scene: &Scene) -> (Params, Params) {
    let _ = scene;
    unimplemented!("bounds")
}

pub fn run(
    engine: &mut RenderEngine,
    scene: &Scene,
    looks: &[LookSpec],
    cfg: &SearchConfig,
    progress: &mut dyn FnMut(f32, &str),
) -> Result<SearchResult> {
    let _ = (engine, scene, looks, cfg, progress);
    unimplemented!("search::run")
}

/// Short local search from a hand-picked parameter set (Claude's
/// refinement): fixes clipping and gamut without moving far.
pub fn polish(
    engine: &mut RenderEngine,
    scene: &Scene,
    start: &Params,
    cfg: &SearchConfig,
) -> Result<Candidate> {
    let _ = (engine, scene, start, cfg);
    unimplemented!("search::polish")
}

/// Stats of one parameter set, scored against the natural look.
pub fn evaluate(
    engine: &mut RenderEngine,
    scene: &Scene,
    params: &Params,
) -> Result<(Stats, Score)> {
    let _ = (engine, scene, params);
    unimplemented!("search::evaluate")
}
