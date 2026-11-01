//! How good one render is: constraints, fidelity, expert prior and look
//! targets. Higher is better. Owner: search.

use serde::{Deserialize, Serialize};

use crate::types::{Params, Stats};

/// A look the search optimizes for.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LookSpec {
    pub id: String,
    pub label: String,
    /// A parameter set to stay near (Claude's proposal), in Lightroom units.
    pub anchor: Option<Params>,
}

impl LookSpec {
    /// The looks every photo gets: natural, vivid, bright and airy, moody...
    pub fn builtin() -> Vec<LookSpec> {
        unimplemented!("LookSpec::builtin")
    }

    /// A look centred on a proposed parameter set.
    pub fn proposal(id: &str, label: &str, params: Params) -> LookSpec {
        let _ = (id, label, params);
        unimplemented!("LookSpec::proposal")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Score {
    pub total: f32,
    /// False when a hard constraint (clipping, crushed shadows, gamut) fails.
    pub feasible: bool,
    pub terms: Vec<(String, f32)>,
}

/// Per-photo facts the objective needs besides one render's stats.
#[derive(Debug, Clone)]
pub struct SceneContext {
    pub as_shot_temperature: f32,
    pub as_shot_tint: f32,
    pub neutral: Stats,
    pub raw_clipped_fraction: f32,
    pub iso: Option<u32>,
}

pub fn score(stats: &Stats, params: &Params, look: &LookSpec, cx: &SceneContext) -> Score {
    let _ = (stats, params, look, cx);
    unimplemented!("score")
}
