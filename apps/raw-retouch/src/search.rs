//! Per-photo parameter search: seeds, CMA-ES per look, diverse finalists.
//! Owner: search.
//!
//! Every render goes into one archive, and a look's result is the best
//! archived render under that look's objective, so a point found while
//! searching one look can win another. Looks run one after another; each
//! starts from the best of its seeds (auto, auto moved the look's way, the
//! anchor, the PPR10K expert means and scrambled Halton points over the
//! whole box) and runs CMA-ES with one batched render call per generation.
//! Every parameter set is quantized to Lightroom's steps before it is
//! rendered, so the archive holds only exportable edits: an optimum that
//! sits on a constraint boundary cannot be rounded across it on export, and
//! a candidate's stats and score describe exactly the sliders it exports.

mod halton;
mod space;
#[cfg(test)]
mod synthetic;
#[cfg(test)]
mod tune;

use std::collections::HashSet;
use std::ops::Range;

use serde::{Deserialize, Serialize};
use spl_core::{Context, Result, bail};

use crate::cmaes::{self, SplitMix64};
use crate::objective::{
    LookSpec, Objective, PPR10K, SceneContext, Score, look_offset, offset_params,
};
use crate::render::RenderEngine;
use crate::types::{Params, Scene, Stats};
use space::{LinearKey, Space};

// The search box: PPR10K's union of the three experts' 1st to 99th
// percentiles, in Lightroom units.
const EXPOSURE: (f32, f32) = (-1.2, 2.5);
const CONTRAST: (f32, f32) = (-32.0, 45.0);
const HIGHLIGHTS: (f32, f32) = (-87.0, 40.0);
const SHADOWS: (f32, f32) = (-28.0, 68.0);
const WHITES: (f32, f32) = (-55.0, 50.0);
const BLACKS: (f32, f32) = (-65.0, 38.0);
const VIBRANCE: (f32, f32) = (-13.0, 45.0);
const SATURATION: (f32, f32) = (-32.0, 18.0);
/// White balance as a warm shift from as-shot in mireds (+ = warmer).
const WARM_SHIFT: (f32, f32) = (-50.0, 70.0);
const TINT_SHIFT: f32 = 17.0;

// PPR10K is portraits; these widen the box for scenes it does not cover.
const WIDE_EXPOSURE: (f32, f32) = (-2.0, 3.0);
const WIDE_HIGHLIGHTS_LO: f32 = -100.0;
const WIDE_SHADOWS_HI: f32 = 85.0;
/// Median of the linear image (1.0 = RAW clip) on an ordinary exposure:
/// middle grey sits about 3.5 stops under clipping, scene medians nearby.
const TYPICAL_MEDIAN: f32 = 0.09;
/// Stops under the typical median over which exposure's upper bound (and
/// the shadow lift) widen: night and badly underexposed frames.
const DARK_STOPS: (f32, f32) = (1.5, 2.5);
/// Stops over it over which exposure's lower bound widens.
const BRIGHT_STOPS: (f32, f32) = (0.7, 1.7);
/// Linear dynamic range (p99.5 over p5, stops) over which highlight and
/// shadow recovery widen: backlight, bright sky over land.
const HDR_STOPS: (f32, f32) = (7.5, 9.5);
/// RAW-clipped share that alone gets highlights the full widening.
const RAW_CLIPPED_WIDE: f32 = 0.01;

const POPULATION: usize = 20;
const MAX_POPULATION: usize = 48;
const LOOK_SIGMA0: f64 = 0.25;
/// Anchor looks start at Claude's proposal and only need to repair it.
const ANCHOR_SIGMA0: f64 = 0.2;
const RESTARTS: usize = 2;
/// Steps below this (box units: 0.004 EV, 0.1 slider points) change nothing visible.
const TOL_X: f64 = 1e-3;
const TOL_FUN: f64 = 1e-5;
const HALTON_SEEDS: usize = 32;
/// Natural's seeds bracket auto's exposure by this much.
const NATURAL_BRACKET_EV: f32 = 0.3;
/// Finalists closer than this (Euclidean, unit box) look alike to a judge.
const DISTINCT: f32 = 0.08;
const POLISH_SIGMA0: f64 = 0.05;
const POLISH_POPULATION: usize = 12;
const POLISH_EVALS: usize = 300;

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

/// `score` is under the look the candidate was optimized for: natural for
/// neutral and auto, the look itself for look candidates, the starting
/// point (as an anchor) for a polished one.
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

/// Renders parameter sets on the search image and measures them, batched:
/// the GPU [`RenderEngine`], or an analytic stand-in in tests.
pub trait Evaluator {
    fn stats(&mut self, params: &[Params]) -> Result<Vec<Stats>>;
}

impl Evaluator for RenderEngine {
    fn stats(&mut self, params: &[Params]) -> Result<Vec<Stats>> {
        RenderEngine::stats(self, params)
    }
}

/// Lightroom's temperature range, with NaN mapped to daylight.
pub(crate) fn sane_temperature(kelvin: f32) -> f32 {
    if kelvin.is_finite() {
        kelvin.clamp(2000.0, 50000.0)
    } else {
        5500.0
    }
}

/// The Lightroom temperature that renders `shift` mireds warmer than
/// `kelvin`. Lightroom's slider names the light being corrected for, so a
/// higher value assumes bluer light and warms the picture: warmer means
/// more kelvin and fewer mireds (1e6 / K). Clamped to 2000..50000 K.
pub fn warmed(kelvin: f32, shift: f32) -> f32 {
    let kelvin = sane_temperature(kelvin);
    if shift == 0.0 || !shift.is_finite() {
        return kelvin;
    }
    let mired = 1.0e6 / f64::from(kelvin) - f64::from(shift);
    ((1.0e6 / mired.max(1.0e6 / 50000.0)) as f32).clamp(2000.0, 50000.0)
}

/// How much warmer `kelvin` renders than `from`, in mireds: the inverse of
/// [`warmed`] (`warm_shift(t, warmed(t, s)) == s` inside the legal range).
pub fn warm_shift(from: f32, kelvin: f32) -> f32 {
    (1.0e6 / f64::from(sane_temperature(from)) - 1.0e6 / f64::from(sane_temperature(kelvin))) as f32
}

/// `p` with every slider clamped into `lo..=hi` (temperature in kelvin);
/// NaN becomes the middle of the range.
pub fn clamp_params(p: &Params, lo: &Params, hi: &Params) -> Params {
    let (p, lo, hi) = (p.to_array(), lo.to_array(), hi.to_array());
    let mut out = [0.0f32; Params::COUNT];
    for (((o, v), l), h) in out.iter_mut().zip(p).zip(lo).zip(hi) {
        *o = if v.is_finite() {
            v.max(l).min(h)
        } else {
            0.5 * (l + h)
        };
    }
    Params::from_array(out)
}

/// Legal search box for this photo in Lightroom units: expert ranges
/// (PPR10K) around the as-shot white balance. Returns (low, high).
///
/// White balance is searched as a warm shift of -50 to +70 mireds from
/// as-shot, so `low.temperature` is the coolest (fewest kelvin) and
/// `high.temperature` the warmest, and tint within 17 of as-shot. The box
/// widens where PPR10K's portraits say nothing: exposure toward -2 or +3 EV
/// when the linear image is far brighter or darker than an ordinary
/// exposure, highlights to -100 and shadows to +85 for high dynamic range
/// or clipped RAWs. Every bound sits on Lightroom's step grid, so
/// quantizing an in-box value keeps it in the box.
pub fn bounds(scene: &Scene) -> (Params, Params) {
    bounds_for(
        scene.info.as_shot_temperature,
        scene.info.as_shot_tint,
        scene.info.raw_clipped_fraction,
        LinearKey::measure(&scene.search),
    )
}

fn bounds_for(
    as_shot_temperature: f32,
    as_shot_tint: f32,
    raw_clipped_fraction: f32,
    key: Option<LinearKey>,
) -> (Params, Params) {
    let ramp = |x: f32, (from, to): (f32, f32)| ((x - from) / (to - from)).clamp(0.0, 1.0);
    let lerp = |(a, b): (f32, f32), t: f32| a + (b - a) * t;
    let (mut dark, mut bright, mut hdr) = (0.0, 0.0, 0.0);
    if let Some(key) = key {
        let stops = (key.median.max(1e-6) / TYPICAL_MEDIAN).log2();
        dark = ramp(-stops, DARK_STOPS);
        bright = ramp(stops, BRIGHT_STOPS);
        hdr = ramp((key.p995.max(1e-6) / key.p05.max(1e-6)).log2(), HDR_STOPS);
    }
    let raw_clipped = if raw_clipped_fraction.is_finite() {
        raw_clipped_fraction
    } else {
        0.0
    };
    hdr = f32::max(hdr, ramp(raw_clipped, (0.0, RAW_CLIPPED_WIDE)));

    let temperature = sane_temperature(as_shot_temperature);
    let tint = if as_shot_tint.is_finite() {
        as_shot_tint.clamp(-150.0, 150.0)
    } else {
        0.0
    };
    let cents_up = |v: f32| (v * 100.0).ceil() / 100.0;
    let cents_down = |v: f32| (v * 100.0).floor() / 100.0;
    let lo = Params {
        temperature: warmed(temperature, WARM_SHIFT.0).ceil(),
        tint: (tint - TINT_SHIFT).max(-150.0).ceil(),
        exposure: cents_up(lerp((EXPOSURE.0, WIDE_EXPOSURE.0), bright)),
        contrast: CONTRAST.0,
        highlights: lerp((HIGHLIGHTS.0, WIDE_HIGHLIGHTS_LO), hdr).ceil(),
        shadows: SHADOWS.0,
        whites: WHITES.0,
        blacks: BLACKS.0,
        vibrance: VIBRANCE.0,
        saturation: SATURATION.0,
    };
    let hi = Params {
        temperature: warmed(temperature, WARM_SHIFT.1).floor(),
        tint: (tint + TINT_SHIFT).min(150.0).floor(),
        exposure: cents_down(lerp((EXPOSURE.1, WIDE_EXPOSURE.1), dark)),
        contrast: CONTRAST.1,
        highlights: HIGHLIGHTS.1,
        shadows: lerp((SHADOWS.1, WIDE_SHADOWS_HI), f32::max(dark, hdr)).floor(),
        whites: WHITES.1,
        blacks: BLACKS.1,
        vibrance: VIBRANCE.1,
        saturation: SATURATION.1,
    };
    (lo, hi)
}

pub fn run(
    engine: &mut RenderEngine,
    scene: &Scene,
    looks: &[LookSpec],
    cfg: &SearchConfig,
    progress: &mut dyn FnMut(f32, &str),
) -> Result<SearchResult> {
    run_with(engine, scene, looks, cfg, progress)
}

/// [`run`] against any [`Evaluator`].
pub fn run_with<E: Evaluator + ?Sized>(
    eval: &mut E,
    scene: &Scene,
    looks: &[LookSpec],
    cfg: &SearchConfig,
    progress: &mut dyn FnMut(f32, &str),
) -> Result<SearchResult> {
    let (lo, hi) = bounds(scene);
    let space = Space::new(lo, hi);
    let mut book = Book::new(eval);

    progress(0.0, "Rendering the neutral and auto edits");
    let (neutral_at, cx) = book.neutral(scene, &space)?;
    let natural = Objective::new(&LookSpec::natural(), &cx);
    let auto_params = space
        .clamp(&crate::auto::auto_params(&cx.neutral, scene))
        .quantized();
    let auto_at = book.measure(&[auto_params])?.start;
    let neutral = book.candidate(
        neutral_at,
        "neutral",
        "As shot",
        CandidateSource::Neutral,
        &natural,
    );
    let auto = book.candidate(auto_at, "auto", "Auto", CandidateSource::Auto, &natural);

    let objectives: Vec<Objective> = looks.iter().map(|l| Objective::new(l, &cx)).collect();
    let share = 0.95 / looks.len().max(1) as f32;
    progress(0.04, "Searching looks");
    for (k, (look, objective)) in looks.iter().zip(&objectives).enumerate() {
        search_look(&mut book, &space, look, objective, &auto_params, cfg, k)?;
        let best = &book.entries[book.best(objective)];
        let (total, feasible) = objective.total(&best.stats, &best.params);
        let note = if feasible {
            ""
        } else {
            ", constraints not met"
        };
        progress(
            0.04 + share * (k + 1) as f32,
            &format!(
                "{} ({}/{}): score {total:.2}{note}",
                look.label,
                k + 1,
                looks.len()
            ),
        );
    }

    let looks_out: Vec<Candidate> = looks
        .iter()
        .zip(&objectives)
        .zip(look_ids(looks))
        .map(|((look, objective), id)| {
            let source = if look.anchor.is_some() {
                CandidateSource::Claude
            } else {
                CandidateSource::Look
            };
            book.candidate(book.best(objective), &id, &look.label, source, objective)
        })
        .collect();
    let finalists = finalists(&auto, &looks_out, &space, cfg.max_finalists);
    progress(1.0, "Search done");
    Ok(SearchResult {
        neutral,
        auto,
        looks: looks_out,
        finalists,
        evaluations: book.entries.len(),
    })
}

/// Short local search from a hand-picked parameter set (Claude's
/// refinement): fixes clipping and gamut without moving far.
pub fn polish(
    engine: &mut RenderEngine,
    scene: &Scene,
    start: &Params,
    cfg: &SearchConfig,
) -> Result<Candidate> {
    polish_with(engine, scene, start, cfg)
}

/// [`polish`] against any [`Evaluator`].
pub fn polish_with<E: Evaluator + ?Sized>(
    eval: &mut E,
    scene: &Scene,
    start: &Params,
    cfg: &SearchConfig,
) -> Result<Candidate> {
    let (lo, hi) = bounds(scene);
    let space = Space::new(lo, hi);
    let mut book = Book::new(eval);
    let (_, cx) = book.neutral(scene, &space)?;
    let start = space.clamp(start);
    let look = LookSpec::proposal("refined", "Refined", start);
    let objective = Objective::new(&look, &cx);
    book.measure(&[start.quantized()])?;
    let opts = cmaes::Options {
        sigma0: POLISH_SIGMA0,
        lambda: POLISH_POPULATION,
        max_lambda: POLISH_POPULATION,
        max_evals: POLISH_EVALS.min(cfg.evals_per_look),
        restarts: 0,
        restart_from_best: true,
        tol_x: TOL_X,
        tol_fun: TOL_FUN,
        seed: derived_seed(cfg.seed, u64::MAX),
    };
    minimize_into(&mut book, &space, &objective, &space.unit_of(&start), &opts)?;
    let at = book.best(&objective);
    Ok(book.candidate(
        at,
        "refined",
        "Refined",
        CandidateSource::Refined,
        &objective,
    ))
}

/// Stats of one parameter set, scored against the natural look.
pub fn evaluate(
    engine: &mut RenderEngine,
    scene: &Scene,
    params: &Params,
) -> Result<(Stats, Score)> {
    evaluate_with(engine, scene, params)
}

/// [`evaluate`] against any [`Evaluator`].
pub fn evaluate_with<E: Evaluator + ?Sized>(
    eval: &mut E,
    scene: &Scene,
    params: &Params,
) -> Result<(Stats, Score)> {
    let (lo, hi) = bounds(scene);
    let space = Space::new(lo, hi);
    let mut book = Book::new(eval);
    let (_, cx) = book.neutral(scene, &space)?;
    let at = book.measure(&[*params])?.start;
    let stats = book.entries[at].stats.clone();
    let score = Objective::new(&LookSpec::natural(), &cx).score(&stats, params);
    Ok((stats, score))
}

/// Every render one search made, in order. Its length is the evaluation count.
struct Book<'a, E: Evaluator + ?Sized> {
    eval: &'a mut E,
    entries: Vec<Entry>,
}

struct Entry {
    params: Params,
    stats: Stats,
}

impl<'a, E: Evaluator + ?Sized> Book<'a, E> {
    fn new(eval: &'a mut E) -> Self {
        Self {
            eval,
            entries: Vec::new(),
        }
    }

    /// Renders and measures `params` in one call and archives the results.
    fn measure(&mut self, params: &[Params]) -> Result<Range<usize>> {
        let start = self.entries.len();
        if params.is_empty() {
            return Ok(start..start);
        }
        let stats = self
            .eval
            .stats(params)
            .with_context(|| format!("render and measure {} candidates", params.len()))?;
        if stats.len() != params.len() {
            bail!(
                "the renderer measured {} of {} parameter sets",
                stats.len(),
                params.len()
            );
        }
        self.entries
            .extend(params.iter().zip(stats).map(|(p, s)| Entry {
                params: *p,
                stats: s,
            }));
        Ok(start..self.entries.len())
    }

    /// The as-shot render with every slider at zero, and the context it gives the objective.
    fn neutral(&mut self, scene: &Scene, space: &Space) -> Result<(usize, SceneContext)> {
        let info = &scene.info;
        let params = space
            .clamp(&Params::neutral(
                info.as_shot_temperature,
                info.as_shot_tint,
            ))
            .quantized();
        let at = self.measure(&[params])?.start;
        Ok((at, SceneContext::new(scene, self.entries[at].stats.clone())))
    }

    /// Index of the best entry under `objective`: feasible first, then by
    /// total, the earliest on ties.
    fn best(&self, objective: &Objective) -> usize {
        let mut best = (0, false, f32::NEG_INFINITY);
        for (i, e) in self.entries.iter().enumerate() {
            let (total, feasible) = objective.total(&e.stats, &e.params);
            if (feasible, total) > (best.1, best.2) {
                best = (i, feasible, total);
            }
        }
        best.0
    }

    fn candidate(
        &self,
        at: usize,
        id: &str,
        label: &str,
        source: CandidateSource,
        objective: &Objective,
    ) -> Candidate {
        let e = &self.entries[at];
        Candidate {
            id: id.into(),
            label: label.into(),
            source,
            params: e.params,
            score: objective.score(&e.stats, &e.params),
            stats: e.stats.clone(),
        }
    }
}

fn derived_seed(seed: u64, stream: u64) -> u64 {
    SplitMix64(seed ^ stream.wrapping_mul(0xD1B5_4A32_D192_ED03)).next_u64()
}

/// One look: seeds in one batch, then CMA-ES from the best archived point.
fn search_look<E: Evaluator + ?Sized>(
    book: &mut Book<E>,
    space: &Space,
    look: &LookSpec,
    objective: &Objective,
    auto: &Params,
    cfg: &SearchConfig,
    index: usize,
) -> Result<()> {
    let seed = derived_seed(cfg.seed, index as u64);
    let mut seeds = seeds(look, auto, space, seed);
    seeds.truncate(cfg.evals_per_look);
    book.measure(&seeds)?;
    let remaining = cfg.evals_per_look - seeds.len();
    if remaining == 0 {
        return Ok(());
    }
    let start = book.entries[book.best(objective)].params;
    let opts = cmaes::Options {
        sigma0: if look.anchor.is_some() {
            ANCHOR_SIGMA0
        } else {
            LOOK_SIGMA0
        },
        lambda: POPULATION,
        max_lambda: MAX_POPULATION,
        max_evals: remaining,
        restarts: RESTARTS,
        restart_from_best: true,
        tol_x: TOL_X,
        tol_fun: TOL_FUN,
        seed: derived_seed(seed, 1),
    };
    minimize_into(book, space, objective, &space.unit_of(&start), &opts)
}

/// CMA-ES over the box on Lightroom's step grid, archiving every render.
fn minimize_into<E: Evaluator + ?Sized>(
    book: &mut Book<E>,
    space: &Space,
    objective: &Objective,
    start: &[f64],
    opts: &cmaes::Options,
) -> Result<()> {
    cmaes::minimize(start, opts, |points| -> Result<Vec<f64>> {
        let params: Vec<Params> = points
            .iter()
            .map(|u| space.params_at(u).quantized())
            .collect();
        let range = book.measure(&params)?;
        Ok(book.entries[range]
            .iter()
            .map(|e| -f64::from(objective.total(&e.stats, &e.params).0))
            .collect())
    })?;
    Ok(())
}

/// Where a look's search may start, most informed first: auto, auto moved
/// the look's way, the anchor, the PPR10K expert means, then scrambled
/// Halton points over the whole box. All inside the box, on Lightroom's grid.
fn seeds(look: &LookSpec, auto: &Params, space: &Space, seed: u64) -> Vec<Params> {
    let mut out = vec![*auto];
    let offset = look_offset(look);
    if let Some(anchor) = look.anchor {
        out.push(anchor);
    } else if offset.iter().any(|d| *d != 0.0) {
        out.extend([0.6, 1.4].map(|f| offset_params(auto, &offset, f)));
    } else {
        out.extend([NATURAL_BRACKET_EV, -NATURAL_BRACKET_EV].map(|ev| Params {
            exposure: auto.exposure + ev,
            ..*auto
        }));
    }
    out.extend(expert_means(auto));
    out.extend(
        halton::scrambled(HALTON_SEEDS, Params::COUNT, derived_seed(seed, 2))
            .iter()
            .map(|u| space.params_at(u)),
    );
    out.iter().map(|p| space.clamp(p).quantized()).collect()
}

/// The three PPR10K experts' mean edits, around `base`'s white balance.
fn expert_means(base: &Params) -> Vec<Params> {
    (0..3)
        .map(|e| {
            let m = |slider: usize| PPR10K[slider][e].0;
            Params {
                temperature: warmed(base.temperature, m(8)),
                tint: base.tint + m(9),
                exposure: m(0),
                contrast: m(1),
                highlights: m(2),
                shadows: m(3),
                whites: m(4),
                blacks: m(5),
                vibrance: m(6),
                saturation: m(7),
            }
        })
        .collect()
}

/// "look:<id>", made unique when two looks share an id.
fn look_ids(looks: &[LookSpec]) -> Vec<String> {
    let mut seen = HashSet::new();
    looks
        .iter()
        .map(|l| {
            let base = format!("look:{}", l.id);
            let mut id = base.clone();
            let mut k = 2;
            while !seen.insert(id.clone()) {
                id = format!("{base}-{k}");
                k += 1;
            }
            id
        })
        .collect()
}

/// Auto plus each look's best, best first, dropping any candidate within
/// [`DISTINCT`] of a better one. Auto always stays: it is the baseline a
/// judge must see, so a better candidate close to it is kept as well, but
/// one that renders exactly like auto is dropped.
fn finalists(auto: &Candidate, looks: &[Candidate], space: &Space, max: usize) -> Vec<Candidate> {
    let mut pool: Vec<&Candidate> = looks.iter().chain([auto]).collect();
    pool.sort_by(|a, b| {
        b.score
            .feasible
            .cmp(&a.score.feasible)
            .then(b.score.total.total_cmp(&a.score.total))
    });
    let mut kept: Vec<&Candidate> = Vec::new();
    for c in pool {
        let is_auto = c.source == CandidateSource::Auto;
        let same_as_auto = !is_auto && c.params == auto.params;
        let distinct = kept
            .iter()
            .all(|k| space.distance(&k.params, &c.params) >= DISTINCT);
        if is_auto || (distinct && !same_as_auto) {
            kept.push(c);
        }
    }
    while kept.len() > max.max(1) {
        match kept.iter().rposition(|c| c.source != CandidateSource::Auto) {
            Some(i) => {
                kept.remove(i);
            }
            None => break,
        }
    }
    kept.into_iter().cloned().collect()
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::synthetic::Synthetic;
    use super::*;

    fn quiet() -> impl FnMut(f32, &str) {
        |_, _| {}
    }

    fn look<'a>(r: &'a SearchResult, id: &str) -> &'a Candidate {
        r.looks
            .iter()
            .find(|c| c.id == format!("look:{id}"))
            .unwrap()
    }

    fn inside(p: &Params, lo: &Params, hi: &Params) -> bool {
        let (p, lo, hi) = (p.to_array(), lo.to_array(), hi.to_array());
        (0..Params::COUNT).all(|i| lo[i] <= p[i] && p[i] <= hi[i])
    }

    fn run_synthetic(syn: &mut Synthetic, looks: &[LookSpec], cfg: &SearchConfig) -> SearchResult {
        let scene = syn.scene();
        run_with(syn, &scene, looks, cfg, &mut quiet()).unwrap()
    }

    #[test]
    fn warm_shift_mapping_is_explicit() {
        // Warmer picture = higher Lightroom temperature = fewer mireds.
        let t = warmed(5000.0, 20.0);
        assert!((t - 1.0e6 / 180.0).abs() < 0.01, "{t}");
        assert!(warmed(5000.0, -20.0) < 5000.0);
        assert_eq!(warmed(5000.0, 0.0), 5000.0);
        for s in [-50.0, -7.5, 33.0, 70.0] {
            assert!((warm_shift(5000.0, warmed(5000.0, s)) - s).abs() < 1e-3);
        }
        assert_eq!(warmed(14000.0, 70.0), 50000.0);
        assert_eq!(warmed(2100.0, -50.0), 2000.0);
    }

    #[test]
    fn bounds_follow_ppr10k_around_as_shot() {
        let (lo, hi) = bounds_for(5200.0, 8.0, 0.0, None);
        assert_eq!(lo.temperature, warmed(5200.0, -50.0).ceil());
        assert_eq!(hi.temperature, warmed(5200.0, 70.0).floor());
        assert!(lo.temperature < 5200.0 && 5200.0 < hi.temperature);
        assert!((warm_shift(5200.0, lo.temperature) + 50.0).abs() < 0.1);
        assert!((warm_shift(5200.0, hi.temperature) - 70.0).abs() < 0.1);
        assert_eq!((lo.tint, hi.tint), (-9.0, 25.0));
        assert_eq!((lo.exposure, hi.exposure), (-1.2, 2.5));
        assert_eq!((lo.contrast, hi.contrast), (-32.0, 45.0));
        assert_eq!((lo.highlights, hi.highlights), (-87.0, 40.0));
        assert_eq!((lo.shadows, hi.shadows), (-28.0, 68.0));
        assert_eq!((lo.whites, hi.whites), (-55.0, 50.0));
        assert_eq!((lo.blacks, hi.blacks), (-65.0, 38.0));
        assert_eq!((lo.vibrance, hi.vibrance), (-13.0, 45.0));
        assert_eq!((lo.saturation, hi.saturation), (-32.0, 18.0));
        // On Lightroom's grid: quantizing keeps every bound where it is.
        assert_eq!(lo.quantized(), lo);
        assert_eq!(hi.quantized(), hi);
        // The neutral render is inside.
        let neutral = Params::neutral(5200.0, 8.0);
        assert!(inside(&neutral, &lo, &hi));
        // Extreme as-shot values stay legal and ordered.
        for (t, tint) in [(2000.0, -150.0), (50000.0, 150.0), (f32::NAN, f32::NAN)] {
            let (lo, hi) = bounds_for(t, tint, f32::NAN, None);
            assert!(
                lo.to_array()
                    .iter()
                    .zip(hi.to_array())
                    .all(|(l, h)| *l <= h)
            );
            assert!(lo.temperature >= 2000.0 && hi.temperature <= 50000.0);
        }
    }

    #[test]
    fn bounds_widen_for_dark_bright_and_high_range_scenes() {
        let key = |median: f32, p05: f32, p995: f32| Some(LinearKey { median, p05, p995 });
        let (lo, hi) = bounds_for(5200.0, 8.0, 0.0, key(0.09, 0.01, 0.8));
        assert_eq!(
            (lo.exposure, hi.exposure, lo.highlights, hi.shadows),
            (-1.2, 2.5, -87.0, 68.0)
        );
        let (_, night) = bounds_for(5200.0, 8.0, 0.0, key(0.004, 0.0004, 0.05));
        assert_eq!((night.exposure, night.shadows), (3.0, 85.0));
        let (bright, _) = bounds_for(5200.0, 8.0, 0.0, key(0.4, 0.1, 1.0));
        assert_eq!(bright.exposure, -2.0);
        let (backlit, backlit_hi) = bounds_for(5200.0, 8.0, 0.0, key(0.05, 0.001, 1.0));
        assert_eq!((backlit.highlights, backlit_hi.shadows), (-100.0, 85.0));
        let (clipped, _) = bounds_for(5200.0, 8.0, 0.02, key(0.09, 0.01, 0.8));
        assert_eq!(clipped.highlights, -100.0);
        // The scene-level entry point reads the linear search image.
        let syn = Synthetic::new(0.003, 0.9, 0.05, (0.0, 0.0));
        let (_, hi) = bounds(&syn.scene());
        assert_eq!(hi.exposure, 3.0);
    }

    #[test]
    fn space_round_trips_and_spaces_temperature_in_mireds() {
        let (lo, hi) = bounds_for(5200.0, 8.0, 0.0, None);
        let space = Space::new(lo, hi);
        let p = Params {
            temperature: 6100.0,
            tint: 3.0,
            exposure: 0.7,
            contrast: 12.0,
            highlights: -40.0,
            shadows: 25.0,
            whites: 5.0,
            blacks: -10.0,
            vibrance: 20.0,
            saturation: -4.0,
        };
        let back = space.params_at(&space.unit_of(&p));
        for (a, b) in p.to_array().iter().zip(back.to_array()) {
            assert!((a - b).abs() < 1e-3 * a.abs().max(1.0), "{p:?} vs {back:?}");
        }
        assert_eq!(space.params_at(&[0.0; 10]), lo);
        assert_eq!(space.params_at(&[1.0; 10]), hi);
        // Halfway in the box is halfway in mireds, not in kelvin.
        let mid = space.params_at(&[0.5; 10]).temperature;
        assert!((warm_shift(5200.0, mid) - 10.0).abs() < 0.05, "{mid}");
        assert!(mid < 0.5 * (lo.temperature + hi.temperature));
        assert!((space.distance(&lo, &hi) - (10.0f32).sqrt()).abs() < 1e-4);
    }

    #[test]
    fn search_improves_over_auto() {
        let mut syn = Synthetic::daylight();
        let r = run_synthetic(&mut syn, &LookSpec::builtin(), &SearchConfig::default());
        let natural = look(&r, "natural");
        assert!(natural.score.feasible, "{:?}", natural.score);
        assert!(
            natural.score.total > r.auto.score.total + 0.05,
            "natural {} vs auto {}",
            natural.score.total,
            r.auto.score.total
        );
        assert!(natural.score.total > r.neutral.score.total);
        // It looks like a natural edit, not a stunt.
        assert!(
            (0.40..=0.52).contains(&natural.stats.luma_p50),
            "{:?}",
            natural.stats
        );
        for c in &r.looks {
            assert!(c.score.feasible, "{}: {:?}", c.id, c.score);
        }
        let p50 = |id: &str| look(&r, id).stats.luma_p50;
        assert!(p50("bright") > p50("natural") && p50("natural") > p50("moody"));
        let color = |id: &str| look(&r, id).stats.colorfulness;
        assert!(color("vivid") > color("natural") && color("natural") > color("moody"));
    }

    #[test]
    fn search_respects_bounds_and_lightroom_steps() {
        let mut syn = Synthetic::new(0.02, 1.1, 0.07, (-0.01, 0.03));
        let scene = syn.scene();
        let (lo, hi) = bounds(&scene);
        let cfg = SearchConfig {
            evals_per_look: 600,
            ..SearchConfig::default()
        };
        let r = run_synthetic(&mut syn, &LookSpec::builtin(), &cfg);
        let all = [&r.neutral, &r.auto]
            .into_iter()
            .chain(&r.looks)
            .chain(&r.finalists);
        for c in all {
            assert!(inside(&c.params, &lo, &hi), "{}: {:?}", c.id, c.params);
            assert_eq!(c.params.quantized(), c.params, "{}", c.id);
        }
    }

    #[test]
    fn search_is_deterministic() {
        let cfg = SearchConfig {
            evals_per_look: 500,
            ..SearchConfig::default()
        };
        let a = run_synthetic(&mut Synthetic::daylight(), &LookSpec::builtin(), &cfg);
        let b = run_synthetic(&mut Synthetic::daylight(), &LookSpec::builtin(), &cfg);
        let key = |r: &SearchResult| {
            r.finalists
                .iter()
                .map(|c| (c.id.clone(), c.params, c.score.total.to_bits()))
                .collect::<Vec<_>>()
        };
        assert_eq!(key(&a), key(&b));
        assert_eq!(a.evaluations, b.evaluations);
        let other = SearchConfig { seed: 8, ..cfg };
        let c = run_synthetic(&mut Synthetic::daylight(), &LookSpec::builtin(), &other);
        let params = |r: &SearchResult| r.looks.iter().map(|c| c.params).collect::<Vec<_>>();
        assert_ne!(params(&a), params(&c));
    }

    #[test]
    fn finalists_are_distinct_sorted_and_include_auto() {
        let mut syn = Synthetic::daylight();
        let scene = syn.scene();
        let space = Space::new(bounds(&scene).0, bounds(&scene).1);
        let mut looks = LookSpec::builtin();
        // A duplicate look must not produce a duplicate finalist.
        looks.push(LookSpec {
            id: "natural".into(),
            label: "Natural again".into(),
            anchor: None,
        });
        let r = run_synthetic(&mut syn, &looks, &SearchConfig::default());
        assert_eq!(r.looks.len(), 5);
        assert_eq!(r.looks[4].id, "look:natural-2");
        let f = &r.finalists;
        assert!(f.iter().any(|c| c.id == "auto"));
        assert!(f.len() <= 6 && f.len() >= 3, "{}", f.len());
        assert!(
            f.windows(2).all(|w| (w[0].score.feasible, w[0].score.total)
                >= (w[1].score.feasible, w[1].score.total))
        );
        for (i, a) in f.iter().enumerate() {
            for b in &f[i + 1..] {
                let close = space.distance(&a.params, &b.params) < DISTINCT;
                assert!(!close || b.id == "auto", "{} and {} are alike", a.id, b.id);
            }
        }
        let ids: HashSet<&str> = f.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids.len(), f.len());
        assert!(!(ids.contains("look:natural") && ids.contains("look:natural-2")));

        // A tight cap keeps auto.
        let capped = finalists(&r.auto, &r.looks, &space, 2);
        assert_eq!(capped.len(), 2);
        assert!(capped.iter().any(|c| c.id == "auto"));
        assert_eq!(finalists(&r.auto, &r.looks, &space, 0).len(), 1);
    }

    #[test]
    fn four_looks_of_1500_evals_take_under_a_second() {
        let mut syn = Synthetic::daylight();
        let scene = syn.scene();
        let mut messages = Vec::new();
        let start = Instant::now();
        let r = run_with(
            &mut syn,
            &scene,
            &LookSpec::builtin(),
            &SearchConfig::default(),
            &mut |f, m| messages.push((f, m.to_string())),
        )
        .unwrap();
        let elapsed = start.elapsed();
        eprintln!(
            "4 looks x 1500 evals: {} evaluations, {} renderer calls, {:.0} ms",
            r.evaluations,
            syn.calls,
            elapsed.as_secs_f64() * 1000.0
        );
        // Unoptimized test builds are several times slower than release.
        let limit = if cfg!(debug_assertions) { 3.0 } else { 1.0 };
        assert!(elapsed.as_secs_f64() < limit, "{elapsed:?}");
        assert!(r.evaluations > 4 * 1000 && r.evaluations <= 4 * 1500 + 2);
        // One batched call per generation: about 1500 / 20 per look.
        assert!(syn.calls < 4 * 90, "{} calls", syn.calls);
        // Progress after each look, rising to 1.
        assert!(messages.len() >= 6);
        assert!(messages.windows(2).all(|w| w[0].0 <= w[1].0));
        assert_eq!(messages.last().unwrap().0, 1.0);
        assert!(messages.iter().any(|(_, m)| m.starts_with("Moody (4/4)")));
    }

    #[test]
    fn anchor_look_fixes_clipping_without_wandering() {
        let mut syn = Synthetic::daylight();
        let scene = syn.scene();
        // Claude asks for a bright look that blows the clouds.
        let anchor = Params {
            exposure: 1.4,
            contrast: 10.0,
            highlights: 0.0,
            shadows: 20.0,
            whites: 20.0,
            blacks: 0.0,
            vibrance: 20.0,
            saturation: 0.0,
            ..Params::neutral(5200.0, 8.0)
        };
        let (lo, hi) = bounds(&scene);
        let space = Space::new(lo, hi);
        let looks = [LookSpec::proposal("claude-1", "Bright clouds", anchor)];
        let r = run_synthetic(&mut syn, &looks, &SearchConfig::default());
        let c = &r.looks[0];
        assert_eq!(
            (c.id.as_str(), c.source),
            ("look:claude-1", CandidateSource::Claude)
        );
        assert_eq!(c.label, "Bright clouds");
        let at_anchor = syn.render(&anchor);
        assert!(at_anchor.clip_hi_new > 0.02, "the anchor itself clips");
        assert!(c.score.feasible, "{:?}", c.score);
        let moved = space.distance(&c.params, &anchor);
        assert!(moved < 0.35, "wandered {moved}: {:?}", c.params);
        assert!(c.params.exposure > 0.7, "{:?}", c.params);
    }

    #[test]
    fn polish_stays_local_and_feasible() {
        let mut syn = Synthetic::daylight();
        let scene = syn.scene();
        let start = Params {
            exposure: 1.2,
            highlights: -10.0,
            shadows: 30.0,
            vibrance: 25.0,
            ..Params::neutral(5200.0, 8.0)
        };
        assert!(syn.render(&start).clip_hi_new > 0.01);
        let space = Space::new(bounds(&scene).0, bounds(&scene).1);
        let r = polish_with(&mut syn, &scene, &start, &SearchConfig::default()).unwrap();
        assert_eq!(
            (r.id.as_str(), r.source),
            ("refined", CandidateSource::Refined)
        );
        assert!(r.score.feasible, "{:?}", r.score);
        assert!(space.distance(&r.params, &start) < 0.3);
        assert!(syn.calls < 40, "{} calls", syn.calls);
    }

    #[test]
    fn evaluate_matches_the_search_scores() {
        let mut syn = Synthetic::daylight();
        let scene = syn.scene();
        let cfg = SearchConfig {
            evals_per_look: 300,
            ..SearchConfig::default()
        };
        let r = run_with(&mut syn, &scene, &[LookSpec::natural()], &cfg, &mut quiet()).unwrap();
        let natural = &r.looks[0];
        let (stats, score) = evaluate_with(&mut syn, &scene, &natural.params).unwrap();
        assert_eq!(score.total, natural.score.total);
        assert_eq!(stats.luma_p50, natural.stats.luma_p50);
        let (_, auto) = evaluate_with(&mut syn, &scene, &r.auto.params).unwrap();
        assert_eq!(auto.total, r.auto.score.total);
    }

    #[test]
    fn renderer_failures_surface_as_errors() {
        let mut syn = Synthetic::daylight();
        syn.fail_after = Some(5);
        let scene = syn.scene();
        let err = run_with(
            &mut syn,
            &scene,
            &LookSpec::builtin(),
            &SearchConfig::default(),
            &mut quiet(),
        )
        .err()
        .unwrap();
        assert!(
            format!("{err:#}").contains("CUDA_ERROR_LAUNCH_FAILED"),
            "{err:#}"
        );

        struct Short;
        impl Evaluator for Short {
            fn stats(&mut self, params: &[Params]) -> Result<Vec<Stats>> {
                Ok(vec![Stats::default(); params.len().saturating_sub(1)])
            }
        }
        let err = evaluate_with(&mut Short, &scene, &Params::neutral(5200.0, 8.0))
            .err()
            .unwrap();
        assert!(err.to_string().contains("measured 0 of 1"), "{err}");
    }

    #[test]
    fn nan_stats_do_not_break_the_search() {
        let mut syn = Synthetic::daylight();
        syn.nan_above = Some(0.6);
        let cfg = SearchConfig {
            evals_per_look: 400,
            ..SearchConfig::default()
        };
        let r = run_synthetic(&mut syn, &LookSpec::builtin(), &cfg);
        for c in r.looks.iter().chain(&r.finalists) {
            assert!(
                c.score.total.is_finite() && c.params.exposure <= 0.6,
                "{}: {:?}",
                c.id,
                c.params
            );
        }
    }

    #[test]
    fn tiny_budgets_and_no_looks_still_work() {
        let none = run_synthetic(&mut Synthetic::daylight(), &[], &SearchConfig::default());
        assert!(none.looks.is_empty());
        assert_eq!(none.finalists.len(), 1);
        assert_eq!(none.finalists[0].id, "auto");
        assert_eq!(none.evaluations, 2);

        let cfg = SearchConfig {
            evals_per_look: 0,
            max_finalists: 6,
            seed: 1,
        };
        let r = run_synthetic(&mut Synthetic::daylight(), &LookSpec::builtin(), &cfg);
        assert_eq!(r.looks.len(), 4);
        assert_eq!(r.evaluations, 2);
        assert!(
            r.looks
                .iter()
                .all(|c| c.params == r.auto.params || c.params == r.neutral.params)
        );
        let cfg = SearchConfig {
            evals_per_look: 5,
            ..cfg
        };
        let r = run_synthetic(&mut Synthetic::daylight(), &LookSpec::builtin(), &cfg);
        assert_eq!(r.evaluations, 2 + 4 * 5);
    }
}
