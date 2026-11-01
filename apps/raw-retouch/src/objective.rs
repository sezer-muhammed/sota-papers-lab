//! How good one render is: constraints, fidelity, expert prior and look
//! targets. Higher is better. Owner: search.
//!
//! The order is the report's (`reports/Automatic RAW photo retouching.md`):
//! hard constraints reject what no photographer accepts, fidelity keeps the
//! RAW's detail and honest colour, the PPR10K expert prior keeps every
//! slider where retouchers put theirs, and only then do look targets
//! (histogram percentiles, contrast, colourfulness) express taste. No
//! learned aesthetic score takes part: none ranks edits of one photo
//! reliably, and an optimizer would chase its biases.
//!
//! `Score::total` is the sum of `Score::terms`; penalties are negative.
//! Weights are in units where 1.0 is a clearly visible flaw.

use serde::{Deserialize, Serialize};

use crate::search::{warm_shift, warmed};
use crate::types::{Params, Scene, Stats};

// Hard constraints. Any violation costs a flat amount, so a feasible render
// outranks nearly every infeasible one, plus a part that grows with the
// violation, so the search knows which way is out.

/// New highlight clipping a viewer accepts: specular glints, not skies.
const CLIP_HI_LIMIT: f32 = 0.004;
/// Extra new clipping allowed per unit of RAW-clipped area: the rolloff
/// around a blown sun or lamp clips as well.
const CLIP_HI_RAW_ALLOWANCE: f32 = 0.5;
const CLIP_HI_LIMIT_MAX: f32 = 0.02;
/// Crushed shadows (luma at or below 2/255) on a normally lit scene.
const CLIP_LO_LIMIT: f32 = 0.01;
/// Night scenes have true blacks: this much more may crush at full darkness.
const CLIP_LO_DARK_EXTRA: f32 = 0.06;
/// Never stricter than what the neutral render itself crushes, plus this.
const CLIP_LO_NEUTRAL_MARGIN: f32 = 0.005;
const GAMUT_LIMIT: f32 = 0.02;
/// Saturated subjects start out of gamut; allow the neutral's share plus this.
const GAMUT_NEUTRAL_MARGIN: f32 = 0.01;
const W_HARD_BASE: f32 = 4.0;
/// Per relative unit of violation, and again per its square.
const W_HARD: f32 = 8.0;

// Fidelity to the RAW.

/// Reward for TMQI-style structure relative to the neutral render. Gains
/// are capped: unlimited reward for "more visible detail" drives the search
/// to the flat, over-tonemapped HDR look.
const W_STRUCTURE: f32 = 4.0;
const STRUCTURE_GAIN_CAP: f32 = 0.02;
/// Colour cast of near-neutral pixels (Oklab a/b). Warm casts get more room
/// than cool or green/magenta ones: warm light is usually wanted.
const W_CAST: f32 = 0.25;
const CAST_TOL_A: f32 = 0.006;
const CAST_TOL_COOL: f32 = 0.008;
const CAST_TOL_WARM: f32 = 0.025;
const CAST_SCALE: f32 = 0.01;
/// Half of the neutral render's own excess cast is tolerated too: the
/// search may soften a strong as-shot cast but should not erase the mood.
const CAST_KEEP: f32 = 0.5;
/// Oklab chroma p95 above this looks like a filter, not a photo.
const W_CHROMA: f32 = 0.6;
const CHROMA_P95_CEILING: f32 = 0.20;
const CHROMA_NEUTRAL_MARGIN: f32 = 0.02;
const CHROMA_SCALE: f32 = 0.02;
/// Saturation far above the neutral render's. Saturation here is
/// colourfulness per unit of mean luma: Hasler and Süsstrunk's measure
/// grows with brightness, and lifting a dark photo must not read as
/// oversaturating it.
const W_COLORFUL: f32 = 0.6;
const COLORFUL_CEILING_RATIO: f32 = 1.45;
/// Extra headroom in colourfulness units, for nearly grey scenes.
const COLORFUL_CEILING_MARGIN: f32 = 4.0;
const COLORFUL_SCALE: f32 = 0.1;
/// Mean luma below which saturation stops dividing by it.
const SATURATION_LUMA_FLOOR: f32 = 0.05;

// Expert prior and editing restraint.

/// PPR10K per-slider prior: a soft minimum over the three experts, each
/// cheap inside its p5..p95 range and quadratic beyond it.
const W_PRIOR: f32 = 0.3;
const PRIOR_INSIDE: f32 = 0.1;
const PRIOR_OUTSIDE: f32 = 1.0;
/// Soft-minimum temperature: a value only one expert would use costs at
/// most `PRIOR_SOFTMIN * ln 3` more than one all three would.
const PRIOR_SOFTMIN: f32 = 0.2;
/// Squared normalized distance from the look's baseline (auto plus the
/// look's typical offset): among similar renders, the plainer edit wins.
const W_OVEREDIT: f32 = 1.0;
/// White balance away from the baseline's. Render stats barely see a
/// white-balance shift that stays inside the cast tolerance, yet people
/// do, so without this term it drifts at random.
const W_WB_DRIFT: f32 = 0.25;
const WB_DRIFT_MIRED: f32 = 30.0;
const WB_DRIFT_TINT: f32 = 10.0;
/// Lifting a high-ISO photo's shadows lifts its noise.
const W_NOISE: f32 = 0.2;
const NOISE_ISO_FROM: f32 = 800.0;

// Look targets: a free band around each target, quadratic for the first
// scale unit outside it and linear beyond, so an unreachable target (a
// foggy scene's black point) pulls steadily instead of overruling the
// prior with an exploding square.

const W_P50: f32 = 1.0;
const P50_SCALE: f32 = 0.03;
const W_P99: f32 = 0.5;
const P99_SCALE: f32 = 0.02;
const W_P01: f32 = 0.4;
const P01_SCALE: f32 = 0.015;
const W_P05: f32 = 0.4;
const P05_SCALE: f32 = 0.03;
/// Global contrast as the standard deviation of output luma.
const W_SPREAD: f32 = 0.3;
const SPREAD_SCALE: f32 = 0.03;
/// Saturation (see `W_COLORFUL`) relative to the neutral render.
const W_COLOR: f32 = 0.5;
const COLOR_SCALE: f32 = 0.08;
const W_LOCAL: f32 = 0.3;
const LOCAL_SCALE: f32 = 0.08;
/// Stylized built-in looks start this far behind natural, which is the
/// safest pick when nothing else (Claude, the user) decides.
const STYLE_COST: f32 = 0.3;
/// Raw-clipped share above which p99 is white whatever the sliders do.
const RAW_CLIPPED_P99: f32 = 0.004;
/// How much of a very dark or very bright scene's key the looks keep
/// (0 = pull every photo to the look's midtone, 1 = keep the neutral's).
const KEY_KEEP_DARK: f32 = 0.5;
const KEY_KEEP_BRIGHT: f32 = 0.5;

// Anchor looks (Claude's proposals).

/// Squared normalized distance to the anchor: moving one slider by a tenth
/// of its range costs 0.25, far less than any hard violation.
const W_ANCHOR: f32 = 25.0;
/// Fidelity terms still break ties near the anchor, at this share.
const ANCHOR_FIDELITY_SHARE: f32 = 0.25;

/// What unusable stats (NaN, infinity) score: finite, so rankings work.
const INVALID_TOTAL: f32 = -1.0e6;

/// Per-slider scale of a "full" edit, the width of the PPR10K search
/// bounds, in [`Params`] order; temperature in mireds. Distances in these
/// units make one exposure stop and about 20 contrast points comparable.
const SLIDER_SCALE: [f32; Params::COUNT] = [
    120.0, 34.0, 3.7, 77.0, 127.0, 96.0, 105.0, 103.0, 58.0, 50.0,
];

/// PPR10K expert statistics (33,483 Camera Raw XMPs, PV2012), experts a, b
/// and c as (mean, p5, p95), for exposure, contrast, highlights, shadows,
/// whites, blacks, vibrance, saturation, white-balance warm shift (mireds,
/// positive is warmer) and tint shift from as-shot. The data gives tint
/// only as staying within about 17 of as-shot; ±8 stands in for its p5..p95.
pub(crate) const PPR10K: [[(f32, f32, f32); 3]; 10] = [
    [
        (0.60, -0.40, 1.80),
        (0.11, -0.80, 1.15),
        (0.34, -0.60, 1.45),
    ],
    [(12.4, 0.0, 32.0), (-5.5, -29.0, 22.0), (6.0, -13.0, 28.0)],
    [
        (-46.8, -63.0, 0.0),
        (-31.5, -77.0, 18.0),
        (-56.2, -81.0, -5.0),
    ],
    [(13.7, 0.0, 39.0), (28.4, 0.0, 56.0), (28.9, -1.0, 58.0)],
    [(-27.1, -41.0, 0.0), (2.6, -11.0, 25.0), (5.6, -29.0, 33.0)],
    [(-13.5, -48.0, 4.0), (0.7, -21.0, 26.0), (-16.3, -32.0, 0.0)],
    [(0.5, 0.0, 1.0), (3.6, 0.0, 14.0), (17.8, 0.0, 37.0)],
    [(-4.9, -10.0, 0.0), (-0.9, -8.0, 0.0), (-6.8, -24.0, 3.0)],
    [
        (17.9, -24.0, 69.0),
        (-9.3, -49.0, 27.0),
        (-0.5, -42.0, 43.0),
    ],
    [(0.0, -8.0, 8.0), (0.0, -8.0, 8.0), (0.0, -8.0, 8.0)],
];

/// Narrowest half-range an expert gets, so expert a's near-constant
/// vibrance (0 to 1) does not make every other value look absurd.
const PRIOR_MIN_HALF: [f32; 10] = [0.25, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 8.0, 4.0];

const BUILTIN: [(&str, &str); 4] = [
    ("natural", "Natural"),
    ("vivid", "Vivid"),
    ("bright", "Bright and airy"),
    ("moody", "Moody"),
];

/// A look the search optimizes for.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LookSpec {
    pub id: String,
    pub label: String,
    /// A parameter set to stay near (Claude's proposal), in Lightroom units.
    pub anchor: Option<Params>,
}

impl LookSpec {
    /// The looks every photo gets: natural, vivid, bright and airy, moody.
    /// Targets are keyed by id, so a hand-built look with one of these ids
    /// and no anchor behaves the same; any other id without an anchor gets
    /// natural's targets.
    pub fn builtin() -> Vec<LookSpec> {
        BUILTIN
            .iter()
            .map(|(id, label)| LookSpec {
                id: (*id).into(),
                label: (*label).into(),
                anchor: None,
            })
            .collect()
    }

    /// The reference look: neutral, auto and `search::evaluate` are scored
    /// against it.
    pub fn natural() -> LookSpec {
        let (id, label) = BUILTIN[0];
        LookSpec {
            id: id.into(),
            label: label.into(),
            anchor: None,
        }
    }

    /// A look centred on a proposed parameter set: closeness to it is the
    /// main term, so the search fixes clipping without wandering off.
    pub fn proposal(id: &str, label: &str, params: Params) -> LookSpec {
        LookSpec {
            id: id.into(),
            label: label.into(),
            anchor: Some(params),
        }
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

impl SceneContext {
    /// `neutral` is the stats of the as-shot render with every slider at zero.
    pub fn new(scene: &Scene, neutral: Stats) -> Self {
        Self {
            as_shot_temperature: scene.info.as_shot_temperature,
            as_shot_tint: scene.info.as_shot_tint,
            neutral,
            raw_clipped_fraction: scene.info.raw_clipped_fraction,
            iso: scene.info.iso,
        }
    }
}

pub fn score(stats: &Stats, params: &Params, look: &LookSpec, cx: &SceneContext) -> Score {
    Objective::new(look, cx).score(stats, params)
}

/// sRGB transfer function, decode.
fn srgb_to_linear(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// sRGB transfer function, encode.
fn linear_to_srgb(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// Hermite step from 0 at `e0` to 1 at `e1` (either order).
pub(crate) fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// How dark the neutral render reads, 0 for an ordinary exposure to 1 for
/// night: dark scenes keep part of their key and may hold true blacks. A
/// frame one or two stops under (median luma 0.2 and up) is a mistake to
/// correct, not a mood, so the ramp starts below that.
pub(crate) fn darkness(neutral_p50: f32) -> f32 {
    smoothstep(0.24, 0.06, neutral_p50)
}

/// The same for high-key scenes (snow, bright studio backgrounds).
fn brightness(neutral_p50: f32) -> f32 {
    smoothstep(0.55, 0.75, neutral_p50)
}

/// The median luma a look should aim for on this photo: the look's own
/// midtone, pulled toward the neutral's key on very dark or very bright
/// scenes. The blend is geometric in linear light, so it reads like a
/// partial exposure correction.
pub(crate) fn adapted_p50(look_p50: f32, neutral_p50: f32) -> f32 {
    let neutral_p50 = neutral_p50.clamp(0.01, 0.99);
    let keep = KEY_KEEP_DARK * darkness(neutral_p50) + KEY_KEEP_BRIGHT * brightness(neutral_p50);
    let target =
        srgb_to_linear(look_p50).ln() * (1.0 - keep) + srgb_to_linear(neutral_p50).ln() * keep;
    linear_to_srgb(target.exp())
}

/// 0 up to ISO 800, 1 from ISO 6400: how much a lift costs in noise.
pub(crate) fn noise_factor(iso: Option<u32>) -> f32 {
    iso.map_or(0.0, |iso| {
        ((iso.max(1) as f32 / NOISE_ISO_FROM).log2() / 3.0).clamp(0.0, 1.0)
    })
}

/// Squared distance between two parameter sets in [`SLIDER_SCALE`] units.
fn distance_sq(a: &Params, b: &Params) -> f32 {
    let (a, b) = (a.to_array(), b.to_array());
    let mired = |k: f32| 1.0e6 / k.max(1.0);
    let dt = (mired(a[0]) - mired(b[0])) / SLIDER_SCALE[0];
    dt * dt
        + (1..Params::COUNT)
            .map(|i| ((a[i] - b[i]) / SLIDER_SCALE[i]).powi(2))
            .sum::<f32>()
}

/// A look's typical edit relative to auto, in [`Params`] order with
/// temperature as a warm shift in mireds. Seeds the search and centres the
/// over-editing penalty, so a moody look is not punished for being moody.
pub(crate) fn look_offset(look: &LookSpec) -> [f32; Params::COUNT] {
    if look.anchor.is_some() {
        return [0.0; Params::COUNT];
    }
    match look.id.as_str() {
        "vivid" => [0.0, 0.0, 0.0, 15.0, -5.0, 0.0, 0.0, -8.0, 20.0, 6.0],
        "bright" => [0.0, 0.0, 0.45, -12.0, -15.0, 25.0, 5.0, 15.0, -5.0, -3.0],
        "moody" => [
            0.0, 0.0, -0.45, 12.0, -25.0, -5.0, -10.0, -20.0, -8.0, -12.0,
        ],
        _ => [0.0; Params::COUNT],
    }
}

/// `base` moved by `factor` times `offset` (see [`look_offset`]).
pub(crate) fn offset_params(base: &Params, offset: &[f32; Params::COUNT], factor: f32) -> Params {
    let mut a = base.to_array();
    a[0] = warmed(base.temperature, factor * offset[0]);
    for (v, d) in a.iter_mut().zip(offset).skip(1) {
        *v += factor * d;
    }
    Params::from_array(a)
}

#[derive(Debug, Clone, Copy)]
struct Band {
    lo: f32,
    hi: f32,
}

impl Band {
    const fn new(lo: f32, hi: f32) -> Self {
        Self { lo, hi }
    }

    /// Distance outside the band in units of `scale`, through [`huber`].
    fn cost(self, x: f32, scale: f32) -> f32 {
        huber(((self.lo - x).max(0.0) + (x - self.hi).max(0.0)) / scale)
    }
}

/// Square up to 1, then linear with matching slope.
fn huber(d: f32) -> f32 {
    if d <= 1.0 { d * d } else { 2.0 * d - 1.0 }
}

/// Colourfulness per unit of mean luma (see `W_COLORFUL`).
fn saturation(s: &Stats) -> f32 {
    s.colorfulness.max(0.0) / s.luma_mean.max(SATURATION_LUMA_FLOOR)
}

/// What a built-in look aims for, before adapting to the photo.
#[derive(Debug, Clone, Copy)]
struct Targets {
    p50: f32,
    p50_half: f32,
    p99: Band,
    p01: Band,
    /// Lower limit for p05 (lifted shadows); 0 = none.
    p05_min: f32,
    /// Standard deviation of output luma.
    spread: Band,
    /// Saturation relative to the neutral render.
    color: Band,
    /// Local contrast relative to the neutral render; 0 = none.
    local_min: f32,
    style_cost: f32,
}

const NATURAL: Targets = Targets {
    p50: 0.46,
    p50_half: 0.04,
    p99: Band::new(0.94, 0.98),
    p01: Band::new(0.02, 0.04),
    p05_min: 0.0,
    spread: Band::new(0.17, 0.26),
    color: Band::new(1.0, 1.12),
    local_min: 0.0,
    style_cost: 0.0,
};

/// Natural's colourfulness plus 15 to 25 %, more contrast, deeper blacks.
const VIVID: Targets = Targets {
    p50: 0.44,
    p99: Band::new(0.95, 0.985),
    p01: Band::new(0.012, 0.03),
    spread: Band::new(0.21, 0.30),
    color: Band::new(1.28, 1.45),
    local_min: 1.08,
    style_cost: STYLE_COST,
    ..NATURAL
};

const BRIGHT: Targets = Targets {
    p50: 0.55,
    p50_half: 0.03,
    p99: Band::new(0.95, 0.985),
    p01: Band::new(0.04, 0.08),
    p05_min: 0.14,
    spread: Band::new(0.13, 0.22),
    color: Band::new(0.95, 1.12),
    style_cost: STYLE_COST,
    ..NATURAL
};

/// Darker midtones and blacks, a little less colour, highlights kept.
const MOODY: Targets = Targets {
    p50: 0.36,
    p99: Band::new(0.86, 0.95),
    p01: Band::new(0.008, 0.025),
    spread: Band::new(0.18, 0.28),
    color: Band::new(0.82, 0.98),
    style_cost: STYLE_COST,
    ..NATURAL
};

#[derive(Debug, Clone, Copy)]
enum Mode {
    Look(Targets),
    Anchor(Params),
}

/// The neutral render's numbers the objective compares against, cleaned of
/// NaN so one bad measurement cannot poison every score.
#[derive(Debug, Clone, Copy)]
struct Reference {
    structure: f32,
    saturation: f32,
    local_contrast: f32,
}

/// One look's objective with everything that depends only on the photo
/// worked out once, so scoring thousands of renders stays cheap.
#[derive(Debug, Clone)]
pub(crate) struct Objective {
    mode: Mode,
    /// Where over-editing is measured from: auto plus the look's offset.
    baseline: Params,
    reference: Reference,
    clip_hi_limit: f32,
    clip_lo_limit: f32,
    gamut_limit: f32,
    chroma_ceiling: f32,
    /// Highest saturation ratio to the neutral before the fidelity term bites.
    saturation_ceiling: f32,
    cast_a: f32,
    cast_warm: f32,
    cast_cool: f32,
    as_shot_temperature: f32,
    as_shot_tint: f32,
    noise: f32,
}

fn finite_or(x: f32, fallback: f32) -> f32 {
    if x.is_finite() { x } else { fallback }
}

impl Objective {
    pub(crate) fn new(look: &LookSpec, cx: &SceneContext) -> Self {
        let n = &cx.neutral;
        let p50 = finite_or(n.luma_p50, 0.42).clamp(0.0, 1.0);
        let colorfulness = finite_or(n.colorfulness, 0.0).max(1.0);
        let luma_mean = finite_or(n.luma_mean, p50).max(SATURATION_LUMA_FLOOR);
        let reference = Reference {
            structure: finite_or(n.structure, 0.0),
            saturation: colorfulness / luma_mean,
            local_contrast: finite_or(n.local_contrast, 0.0).max(0.0),
        };
        let dark = darkness(p50);
        let raw_clipped = finite_or(cx.raw_clipped_fraction, 0.0).clamp(0.0, 1.0);
        let cast = |value: f32, tol: f32| tol + CAST_KEEP * (finite_or(value, 0.0) - tol).max(0.0);
        let as_shot_temperature = crate::search::sane_temperature(cx.as_shot_temperature);
        let as_shot_tint = finite_or(cx.as_shot_tint, 0.0).clamp(-150.0, 150.0);

        let auto = crate::auto::auto_from_stats(n, as_shot_temperature, as_shot_tint, cx.iso);
        let mode = match look.anchor {
            Some(anchor) => Mode::Anchor(fill_non_finite(anchor, &auto)),
            None => {
                let base = match look.id.as_str() {
                    "vivid" => VIVID,
                    "bright" => BRIGHT,
                    "moody" => MOODY,
                    _ => NATURAL,
                };
                let mut t = base;
                t.p50 = adapted_p50(base.p50, p50);
                // Night scenes may keep true blacks and dim highlights.
                t.p01.lo *= 1.0 - dark;
                t.p05_min *= 1.0 - dark;
                t.p99.lo -= 0.15 * dark;
                if raw_clipped > RAW_CLIPPED_P99 {
                    t.p99.hi = 1.0;
                }
                Mode::Look(t)
            }
        };
        let baseline = match mode {
            Mode::Anchor(anchor) => anchor,
            Mode::Look(_) => offset_params(&auto, &look_offset(look), 1.0),
        };

        Self {
            mode,
            baseline,
            reference,
            clip_hi_limit: (CLIP_HI_LIMIT + CLIP_HI_RAW_ALLOWANCE * raw_clipped)
                .min(CLIP_HI_LIMIT_MAX),
            clip_lo_limit: (CLIP_LO_LIMIT + CLIP_LO_DARK_EXTRA * dark)
                .max(finite_or(n.clip_lo, 0.0).clamp(0.0, 1.0) + CLIP_LO_NEUTRAL_MARGIN),
            gamut_limit: GAMUT_LIMIT
                .max(finite_or(n.out_of_gamut, 0.0).clamp(0.0, 1.0) + GAMUT_NEUTRAL_MARGIN),
            chroma_ceiling: CHROMA_P95_CEILING
                .max(finite_or(n.chroma_p95, 0.0) + CHROMA_NEUTRAL_MARGIN),
            saturation_ceiling: COLORFUL_CEILING_RATIO + COLORFUL_CEILING_MARGIN / colorfulness,
            cast_a: cast(n.cast_a.abs(), CAST_TOL_A),
            cast_warm: cast(n.cast_b, CAST_TOL_WARM),
            cast_cool: cast(-n.cast_b, CAST_TOL_COOL),
            as_shot_temperature,
            as_shot_tint,
            noise: noise_factor(cx.iso),
        }
    }

    pub(crate) fn score(&self, stats: &Stats, params: &Params) -> Score {
        let mut terms = Vec::with_capacity(20);
        let (total, feasible) = self.eval(stats, params, |name, value| {
            terms.push((name.to_string(), value))
        });
        Score {
            total,
            feasible,
            terms,
        }
    }

    /// Total and feasibility without the term list: the search's hot path.
    pub(crate) fn total(&self, stats: &Stats, params: &Params) -> (f32, bool) {
        self.eval(stats, params, |_, _| {})
    }

    fn eval(&self, s: &Stats, p: &Params, mut term: impl FnMut(&'static str, f32)) -> (f32, bool) {
        if !stats_valid(s) || !p.to_array().iter().all(|v| v.is_finite()) {
            term("invalid", INVALID_TOTAL);
            return (INVALID_TOTAL, false);
        }
        let mut total = 0.0;
        let mut add = |name: &'static str, value: f32| {
            total += value;
            term(name, value);
        };

        let clip_hi = violation(s.clip_hi_new, self.clip_hi_limit);
        let clip_lo = violation(s.clip_lo, self.clip_lo_limit);
        let gamut = violation(s.out_of_gamut, self.gamut_limit);
        add("hard.clip_hi", -hard_cost(clip_hi));
        add("hard.clip_lo", -hard_cost(clip_lo));
        add("hard.gamut", -hard_cost(gamut));
        let feasible = clip_hi <= 0.0 && clip_lo <= 0.0 && gamut <= 0.0;

        let share = match self.mode {
            Mode::Look(_) => 1.0,
            Mode::Anchor(_) => ANCHOR_FIDELITY_SHARE,
        };
        let structure_gain = (s.structure - self.reference.structure).min(STRUCTURE_GAIN_CAP);
        add("fidelity.structure", share * W_STRUCTURE * structure_gain);
        let excess = |x: f32, tol: f32| ((x - tol).max(0.0) / CAST_SCALE).powi(2);
        let cast = excess(s.cast_a.abs(), self.cast_a)
            + excess(s.cast_b, self.cast_warm)
            + excess(-s.cast_b, self.cast_cool);
        add("fidelity.cast", -share * W_CAST * cast);
        let chroma = ((s.chroma_p95 - self.chroma_ceiling) / CHROMA_SCALE).max(0.0);
        add("fidelity.chroma", -share * W_CHROMA * chroma * chroma);
        let saturation = saturation(s) / self.reference.saturation;
        let colorful = ((saturation - self.saturation_ceiling) / COLORFUL_SCALE).max(0.0);
        add(
            "fidelity.colorfulness",
            -share * W_COLORFUL * colorful * colorful,
        );
        let lift = p.exposure.max(0.0) / 1.5 + p.shadows.max(0.0) / 60.0 + p.blacks.max(0.0) / 60.0;
        add("prior.noise", -share * W_NOISE * self.noise * lift * lift);

        match self.mode {
            Mode::Anchor(anchor) => add("anchor.distance", -W_ANCHOR * distance_sq(p, &anchor)),
            Mode::Look(t) => {
                let prior = expert_prior(p, self.as_shot_temperature, self.as_shot_tint);
                add("prior.experts", -W_PRIOR * prior);
                let overedit = distance_sq(p, &self.baseline);
                add("prior.overedit", -W_OVEREDIT * overedit);
                let drift = (warm_shift(self.baseline.temperature, p.temperature) / WB_DRIFT_MIRED)
                    .powi(2)
                    + ((p.tint - self.baseline.tint) / WB_DRIFT_TINT).powi(2);
                add("prior.white_balance", -W_WB_DRIFT * drift);
                let p50 = Band::new(t.p50 - t.p50_half, t.p50 + t.p50_half);
                add("look.p50", -W_P50 * p50.cost(s.luma_p50, P50_SCALE));
                add("look.p99", -W_P99 * t.p99.cost(s.luma_p99, P99_SCALE));
                add("look.p01", -W_P01 * t.p01.cost(s.luma_p01, P01_SCALE));
                let p05 = Band::new(t.p05_min, 1.0);
                add("look.p05", -W_P05 * p05.cost(s.luma_p05, P05_SCALE));
                add(
                    "look.contrast",
                    -W_SPREAD * t.spread.cost(luma_std(s), SPREAD_SCALE),
                );
                add(
                    "look.color",
                    -W_COLOR * t.color.cost(saturation, COLOR_SCALE),
                );
                let local = if t.local_min > 0.0 && self.reference.local_contrast > 0.0 {
                    let ratio = s.local_contrast / self.reference.local_contrast;
                    Band::new(t.local_min, f32::INFINITY).cost(ratio, LOCAL_SCALE)
                } else {
                    0.0
                };
                add("look.local_contrast", -W_LOCAL * local);
                add("look.style", -t.style_cost);
            }
        }
        if total.is_finite() {
            (total, feasible)
        } else {
            (INVALID_TOTAL, false)
        }
    }
}

fn stats_valid(s: &Stats) -> bool {
    [
        s.luma_mean,
        s.luma_p01,
        s.luma_p05,
        s.luma_p50,
        s.luma_p95,
        s.luma_p99,
        s.clip_hi_new,
        s.clip_lo,
        s.colorfulness,
        s.chroma_p95,
        s.out_of_gamut,
        s.cast_a,
        s.cast_b,
        s.structure,
        s.local_contrast,
    ]
    .iter()
    .all(|v| v.is_finite())
}

/// `p` with any NaN or infinite slider taken from `fallback`.
fn fill_non_finite(p: Params, fallback: &Params) -> Params {
    let mut a = p.to_array();
    for (v, f) in a.iter_mut().zip(fallback.to_array()) {
        if !v.is_finite() {
            *v = f;
        }
    }
    Params::from_array(a)
}

/// Relative excess over a limit; positive means violated.
fn violation(x: f32, limit: f32) -> f32 {
    (x - limit) / limit.max(1e-6)
}

fn hard_cost(v: f32) -> f32 {
    if v > 0.0 {
        W_HARD_BASE + W_HARD * (v + v * v)
    } else {
        0.0
    }
}

/// Standard deviation of output luma from the histogram, or from the
/// percentile spread when the histogram is missing.
fn luma_std(s: &Stats) -> f32 {
    let sum: f32 = s.luma_hist.iter().sum();
    if s.luma_hist.len() >= 2 && sum.is_finite() && sum > 0.5 {
        let bins = s.luma_hist.len() as f32;
        let centre = |i: usize| (i as f32 + 0.5) / bins;
        let mean = s
            .luma_hist
            .iter()
            .enumerate()
            .map(|(i, h)| h * centre(i))
            .sum::<f32>()
            / sum;
        let var = s
            .luma_hist
            .iter()
            .enumerate()
            .map(|(i, h)| h * (centre(i) - mean).powi(2))
            .sum::<f32>()
            / sum;
        var.max(0.0).sqrt()
    } else {
        ((s.luma_p95 - s.luma_p05) / 3.29).max(0.0)
    }
}

/// Output luma at quantile `q` from the luma histogram (linear inside a
/// bin), or None without a usable histogram.
pub(crate) fn luma_quantile(s: &Stats, q: f32) -> Option<f32> {
    let total: f32 = s.luma_hist.iter().map(|h| h.max(0.0)).sum();
    if s.luma_hist.len() < 2 || !total.is_finite() || total <= 0.5 {
        return None;
    }
    let bins = s.luma_hist.len() as f32;
    let target = q.clamp(0.0, 1.0) * total;
    let mut below = 0.0;
    for (i, h) in s.luma_hist.iter().enumerate() {
        let h = h.max(0.0);
        if h > 0.0 && below + h >= target {
            return Some((i as f32 + ((target - below) / h).clamp(0.0, 1.0)) / bins);
        }
        below += h;
    }
    Some(1.0)
}

/// Sum over sliders of a soft minimum over the three PPR10K experts.
fn expert_prior(p: &Params, as_shot_temperature: f32, as_shot_tint: f32) -> f32 {
    let values = [
        p.exposure,
        p.contrast,
        p.highlights,
        p.shadows,
        p.whites,
        p.blacks,
        p.vibrance,
        p.saturation,
        warm_shift(as_shot_temperature, p.temperature),
        p.tint - as_shot_tint,
    ];
    values
        .iter()
        .zip(&PPR10K)
        .zip(&PRIOR_MIN_HALF)
        .map(|((x, experts), min_half)| {
            let q = experts.map(|(mean, p5, p95)| {
                let half = ((p95 - p5) / 2.0).max(*min_half);
                let inside = (x - mean) / half;
                let outside = (p5 - x).max(x - p95).max(0.0) / half;
                PRIOR_INSIDE * inside * inside + PRIOR_OUTSIDE * outside * outside
            });
            let best = q.iter().copied().fold(f32::INFINITY, f32::min);
            let mean_weight = q
                .iter()
                .map(|qe| (-(qe - best) / PRIOR_SOFTMIN).exp())
                .sum::<f32>()
                / 3.0;
            best - PRIOR_SOFTMIN * mean_weight.ln()
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A plausible neutral render of an ordinary daylight scene.
    fn neutral_stats() -> Stats {
        let mut hist = vec![0.0; 64];
        for (i, h) in hist.iter_mut().enumerate() {
            let x = (i as f32 + 0.5) / 64.0;
            *h = (-((x - 0.42) / 0.28).powi(2)).exp();
        }
        let sum: f32 = hist.iter().sum();
        hist.iter_mut().for_each(|h| *h /= sum);
        Stats {
            luma_hist: hist,
            luma_mean: 0.43,
            luma_p01: 0.03,
            luma_p05: 0.08,
            luma_p50: 0.42,
            luma_p95: 0.86,
            luma_p99: 0.95,
            clip_hi: 0.001,
            clip_hi_new: 0.001,
            clip_lo: 0.002,
            colorfulness: 30.0,
            chroma_mean: 0.06,
            chroma_p95: 0.14,
            out_of_gamut: 0.002,
            cast_a: 0.001,
            cast_b: 0.004,
            structure: 0.85,
            local_contrast: 0.05,
            entropy: 5.5,
        }
    }

    fn cx() -> SceneContext {
        SceneContext {
            as_shot_temperature: 5200.0,
            as_shot_tint: 8.0,
            neutral: neutral_stats(),
            raw_clipped_fraction: 0.0,
            iso: Some(200),
        }
    }

    fn look(id: &str) -> LookSpec {
        LookSpec::builtin()
            .into_iter()
            .find(|l| l.id == id)
            .unwrap()
    }

    fn term(score: &Score, name: &str) -> f32 {
        score
            .terms
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| *v)
            .unwrap()
    }

    fn natural_params() -> Params {
        Params {
            exposure: 0.2,
            contrast: 8.0,
            highlights: -35.0,
            shadows: 20.0,
            whites: 5.0,
            blacks: -5.0,
            vibrance: 18.0,
            saturation: 2.0,
            ..Params::neutral(5200.0, 8.0)
        }
    }

    #[test]
    fn builtin_looks_have_stable_ids_and_labels() {
        let looks = LookSpec::builtin();
        let ids: Vec<&str> = looks.iter().map(|l| l.id.as_str()).collect();
        assert_eq!(ids, ["natural", "vivid", "bright", "moody"]);
        assert!(
            looks
                .iter()
                .all(|l| l.anchor.is_none() && !l.label.is_empty())
        );
        assert_eq!(looks[2].label, "Bright and airy");
        let p = LookSpec::proposal("c1", "Warm evening", natural_params());
        assert_eq!((p.id.as_str(), p.label.as_str()), ("c1", "Warm evening"));
        assert_eq!(p.anchor, Some(natural_params()));
    }

    #[test]
    fn total_is_the_sum_of_terms() {
        for l in
            LookSpec::builtin()
                .into_iter()
                .chain([LookSpec::proposal("p", "P", natural_params())])
        {
            let s = score(&neutral_stats(), &natural_params(), &l, &cx());
            let sum: f32 = s.terms.iter().map(|(_, v)| v).sum();
            assert!(
                (s.total - sum).abs() < 1e-4,
                "{}: {} vs {sum}",
                l.id,
                s.total
            );
            assert!(s.terms.len() >= 8, "{}: {:?}", l.id, s.terms);
        }
    }

    #[test]
    fn a_good_render_is_feasible_and_scores_near_zero() {
        let s = score(&neutral_stats(), &natural_params(), &look("natural"), &cx());
        assert!(s.feasible, "{s:?}");
        assert!(s.total > -0.5 && s.total < 0.5, "{s:?}");
    }

    #[test]
    fn new_highlight_clipping_is_infeasible_and_grows() {
        let mut a = neutral_stats();
        a.clip_hi_new = 0.006;
        let mut b = a.clone();
        b.clip_hi_new = 0.02;
        let (sa, sb) = (
            score(&a, &natural_params(), &look("natural"), &cx()),
            score(&b, &natural_params(), &look("natural"), &cx()),
        );
        assert!(!sa.feasible && !sb.feasible);
        assert!(sb.total < sa.total);
        assert!(term(&sa, "hard.clip_hi") <= -W_HARD_BASE);

        // Clipped in the RAW already: the rolloff may clip a little more.
        let mut clipped = cx();
        clipped.raw_clipped_fraction = 0.01;
        assert!(score(&a, &natural_params(), &look("natural"), &clipped).feasible);
    }

    #[test]
    fn crushed_shadows_are_relaxed_for_night_scenes() {
        let mut crushed = neutral_stats();
        crushed.clip_lo = 0.03;
        assert!(!score(&crushed, &natural_params(), &look("natural"), &cx()).feasible);

        let mut night = cx();
        night.neutral.luma_p50 = 0.08;
        night.neutral.luma_p01 = 0.0;
        assert!(score(&crushed, &natural_params(), &look("natural"), &night).feasible);
    }

    #[test]
    fn gamut_limit_follows_saturated_subjects() {
        let mut s = neutral_stats();
        s.out_of_gamut = 0.03;
        assert!(!score(&s, &natural_params(), &look("natural"), &cx()).feasible);
        let mut neon = cx();
        neon.neutral.out_of_gamut = 0.025;
        assert!(score(&s, &natural_params(), &look("natural"), &neon).feasible);
    }

    #[test]
    fn expert_prior_is_small_inside_and_quadratic_outside() {
        let inside = expert_prior(&natural_params(), 5200.0, 8.0);
        let mut far = natural_params();
        far.highlights = 40.0;
        far.vibrance = 60.0;
        let outside = expert_prior(&far, 5200.0, 8.0);
        let mut farther = far;
        farther.vibrance = 80.0;
        assert!(inside < 0.6, "inside {inside}");
        assert!(outside > inside + 1.0, "outside {outside} inside {inside}");
        let step1 = outside - inside;
        let step2 = expert_prior(&farther, 5200.0, 8.0) - outside;
        assert!(step2 > step1, "quadratic growth: {step1} then {step2}");
        // Vibrance +30 is inside expert c's range: barely penalized.
        let mut vivid = natural_params();
        vivid.vibrance = 30.0;
        assert!(expert_prior(&vivid, 5200.0, 8.0) - inside < 0.25);
    }

    #[test]
    fn white_balance_prior_uses_warm_shift_from_as_shot() {
        let base = natural_params();
        let mut warm = base;
        warm.temperature = warmed(5200.0, 40.0);
        let mut very_cool = base;
        very_cool.temperature = warmed(5200.0, -120.0);
        let p0 = expert_prior(&base, 5200.0, 8.0);
        assert!(expert_prior(&warm, 5200.0, 8.0) - p0 < 0.2);
        assert!(expert_prior(&very_cool, 5200.0, 8.0) - p0 > 2.0);
    }

    #[test]
    fn natural_wants_midtones_in_its_band() {
        let p = natural_params();
        let mut dark = neutral_stats();
        dark.luma_p50 = 0.30;
        let mut good = neutral_stats();
        good.luma_p50 = 0.46;
        let sd = score(&dark, &p, &look("natural"), &cx());
        let sg = score(&good, &p, &look("natural"), &cx());
        assert!(term(&sd, "look.p50") < -1.0);
        assert_eq!(term(&sg, "look.p50"), 0.0);
        assert!(sg.total > sd.total);
    }

    #[test]
    fn dark_scenes_keep_part_of_their_key() {
        let normal = adapted_p50(0.46, 0.40);
        let night = adapted_p50(0.46, 0.08);
        let snow = adapted_p50(0.46, 0.72);
        assert!((normal - 0.46).abs() < 1e-3, "{normal}");
        assert!(night > 0.15 && night < 0.30, "{night}");
        assert!(snow > 0.50 && snow < 0.62, "{snow}");
    }

    #[test]
    fn looks_prefer_their_own_renders() {
        let p = natural_params();
        let natural = neutral_stats();
        let mut vivid = natural.clone();
        vivid.colorfulness = 40.0;
        vivid.local_contrast = 0.06;
        vivid.luma_p01 = 0.02;
        let mut bright = natural.clone();
        bright.luma_p50 = 0.55;
        bright.luma_p05 = 0.16;
        bright.luma_p01 = 0.05;
        bright.luma_p99 = 0.97;
        let mut moody = natural.clone();
        moody.luma_p50 = 0.36;
        moody.luma_p01 = 0.015;
        moody.luma_p99 = 0.92;
        moody.colorfulness = 27.0;

        let total = |s: &Stats, id: &str| score(s, &p, &look(id), &cx()).total;
        for (id, own) in [("vivid", &vivid), ("bright", &bright), ("moody", &moody)] {
            assert!(
                total(own, id) > total(&natural, id),
                "{id} prefers its render"
            );
        }
        assert!(total(&natural, "natural") > total(&bright, "natural"));
        assert!(total(&natural, "natural") > total(&moody, "natural"));
    }

    #[test]
    fn anchor_closeness_dominates_but_clipping_still_matters() {
        let anchor = natural_params();
        let l = LookSpec::proposal("c", "Claude", anchor);
        let s = neutral_stats();
        let at = score(&s, &anchor, &l, &cx());
        let mut near = anchor;
        near.exposure += 0.1;
        let mut far = anchor;
        far.exposure += 0.8;
        far.contrast += 20.0;
        assert!(at.total > score(&s, &near, &l, &cx()).total);
        assert!(score(&s, &near, &l, &cx()).total > score(&s, &far, &l, &cx()).total);
        // Look targets do not apply: a dark render at the anchor is fine...
        let mut dark = s.clone();
        dark.luma_p50 = 0.25;
        assert!((score(&dark, &anchor, &l, &cx()).total - at.total).abs() < 1e-4);
        // ...but clipping at the anchor costs more than a small move away.
        let mut clipping = s.clone();
        clipping.clip_hi_new = 0.01;
        let moved = score(&s, &near, &l, &cx()).total;
        assert!(score(&clipping, &anchor, &l, &cx()).total < moved - 1.0);
    }

    #[test]
    fn over_editing_and_noisy_lifts_cost_something() {
        let n = cx();
        let auto = crate::auto::auto_from_stats(&n.neutral, 5200.0, 8.0, n.iso);
        let o = Objective::new(&look("natural"), &n);
        let s = neutral_stats();
        let mut edited = auto;
        edited.contrast += 30.0;
        edited.whites -= 30.0;
        let near = o.score(&s, &auto);
        let far = o.score(&s, &edited);
        assert!(term(&far, "prior.overedit") < term(&near, "prior.overedit") - 0.1);

        let mut noisy = cx();
        noisy.iso = Some(12800);
        let mut lifted = auto;
        lifted.exposure = 1.5;
        lifted.shadows = 60.0;
        let quiet = score(&s, &lifted, &look("natural"), &cx());
        let loud = score(&s, &lifted, &look("natural"), &noisy);
        assert!(term(&loud, "prior.noise") < term(&quiet, "prior.noise") - 0.3);
    }

    #[test]
    fn colour_cast_beyond_tolerance_costs_and_warmth_is_tolerated() {
        let p = natural_params();
        let base = score(&neutral_stats(), &p, &look("natural"), &cx());
        let mut green = neutral_stats();
        green.cast_a = -0.03;
        let mut warm = neutral_stats();
        warm.cast_b = 0.02;
        let mut blue = neutral_stats();
        blue.cast_b = -0.02;
        assert!(term(&score(&green, &p, &look("natural"), &cx()), "fidelity.cast") < -0.5);
        assert_eq!(
            term(&score(&warm, &p, &look("natural"), &cx()), "fidelity.cast"),
            0.0
        );
        assert!(term(&score(&blue, &p, &look("natural"), &cx()), "fidelity.cast") < -0.2);
        assert_eq!(term(&base, "fidelity.cast"), 0.0);
    }

    #[test]
    fn oversaturation_costs() {
        let p = natural_params();
        let mut loud = neutral_stats();
        loud.chroma_p95 = 0.26;
        loud.colorfulness = 60.0;
        let s = score(&loud, &p, &look("natural"), &cx());
        assert!(term(&s, "fidelity.chroma") < -1.0);
        assert!(term(&s, "fidelity.colorfulness") < -1.0);
    }

    #[test]
    fn structure_loss_costs_more_than_gain_pays() {
        let p = natural_params();
        let mut lost = neutral_stats();
        lost.structure -= 0.1;
        let mut gained = neutral_stats();
        gained.structure += 0.1;
        let l = term(
            &score(&lost, &p, &look("natural"), &cx()),
            "fidelity.structure",
        );
        let g = term(
            &score(&gained, &p, &look("natural"), &cx()),
            "fidelity.structure",
        );
        assert!((l + 0.4).abs() < 1e-4, "{l}");
        assert!((g - W_STRUCTURE * STRUCTURE_GAIN_CAP).abs() < 1e-4, "{g}");
    }

    #[test]
    fn broken_stats_score_low_but_finite() {
        let mut s = neutral_stats();
        s.luma_p50 = f32::NAN;
        let sc = score(&s, &natural_params(), &look("natural"), &cx());
        assert!(!sc.feasible);
        assert_eq!(sc.total, INVALID_TOTAL);
        let mut bad_cx = cx();
        bad_cx.neutral.structure = f32::NAN;
        bad_cx.as_shot_temperature = f32::NAN;
        let ok = score(
            &neutral_stats(),
            &natural_params(),
            &look("natural"),
            &bad_cx,
        );
        assert!(ok.total.is_finite());
    }

    #[test]
    fn saturation_does_not_count_brightness_as_colour() {
        // Lifting a dark scene raises colourfulness with luma: no colour cost.
        let mut dark = cx();
        dark.neutral.luma_mean = 0.15;
        dark.neutral.luma_p50 = 0.14;
        dark.neutral.colorfulness = 12.0;
        let mut lifted = neutral_stats();
        lifted.luma_mean = 0.40;
        lifted.colorfulness = 32.0;
        let s = score(&lifted, &natural_params(), &look("natural"), &dark);
        assert_eq!(term(&s, "look.color"), 0.0);
        assert_eq!(term(&s, "fidelity.colorfulness"), 0.0);
        // The same colourfulness at the neutral's brightness is oversaturated.
        let mut loud = lifted.clone();
        loud.luma_mean = 0.15;
        let s = score(&loud, &natural_params(), &look("natural"), &dark);
        assert!(term(&s, "fidelity.colorfulness") < -1.0, "{s:?}");
    }

    #[test]
    fn far_misses_grow_linearly_not_quadratically() {
        let p = natural_params();
        let at = |p01: f32| {
            let mut s = neutral_stats();
            s.luma_p01 = p01;
            term(&score(&s, &p, &look("natural"), &cx()), "look.p01")
        };
        assert_eq!(at(0.03), 0.0);
        assert!((at(0.04 + P01_SCALE) + W_P01).abs() < 1e-4);
        let (far, farther) = (at(0.04 + 5.0 * P01_SCALE), at(0.04 + 10.0 * P01_SCALE));
        assert!((far - 5.0 * farther / 10.0).abs() < 0.5, "{far} {farther}");
        assert!((farther - far + 10.0 * W_P01).abs() < 1e-3);
    }

    #[test]
    fn white_balance_does_not_drift_for_free() {
        let s = neutral_stats();
        let base = score(&s, &natural_params(), &look("natural"), &cx());
        assert_eq!(term(&base, "prior.white_balance"), 0.0);
        let mut warm = natural_params();
        warm.temperature = warmed(5200.0, WB_DRIFT_MIRED);
        let drifted = score(&s, &warm, &look("natural"), &cx());
        assert!((term(&drifted, "prior.white_balance") + W_WB_DRIFT).abs() < 1e-3);
        let mut green = natural_params();
        green.tint -= WB_DRIFT_TINT;
        let tinted = score(&s, &green, &look("natural"), &cx());
        assert!((term(&tinted, "prior.white_balance") + W_WB_DRIFT).abs() < 1e-3);
    }

    #[test]
    fn anchors_with_holes_are_filled_from_auto() {
        let mut anchor = natural_params();
        anchor.exposure = f32::NAN;
        let l = LookSpec::proposal("c", "C", anchor);
        let s = score(&neutral_stats(), &natural_params(), &l, &cx());
        assert!(s.total.is_finite() && s.feasible, "{s:?}");
    }

    #[test]
    fn luma_std_from_histogram_or_percentiles() {
        let s = neutral_stats();
        let std = luma_std(&s);
        assert!(std > 0.1 && std < 0.25, "{std}");
        let mut no_hist = s.clone();
        no_hist.luma_hist.clear();
        assert!((luma_std(&no_hist) - (0.86 - 0.08) / 3.29).abs() < 1e-6);
    }

    #[test]
    fn distance_uses_mireds_for_temperature() {
        let a = Params::neutral(5000.0, 0.0);
        let mut b = a;
        b.temperature = 1.0e6 / (200.0 - 12.0);
        assert!((distance_sq(&a, &b) - 0.01).abs() < 1e-4);
        let mut c = a;
        c.exposure = 0.37;
        assert!((distance_sq(&a, &c) - 0.01).abs() < 1e-5);
    }
}
