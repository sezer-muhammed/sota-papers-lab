//! `raw_retouch.auto`: measure the neutral render, let Claude plan looks,
//! search every look, let Claude rank the finalists and refine the winner,
//! then export the chosen edit at full resolution.
//!
//! Claude is optional at every step: without it (switched off, or any
//! failure while ranking) the objective's best finalist is the result. A
//! failed plan only costs the proposed looks, and a failed refinement keeps
//! Claude's ranked winner.

use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;
use std::time::Instant;

use spl_core::{ArtifactRef, DataDir, Result, RunCx};

use super::cache::{self, Photo};
use super::export::{self, display, put_jpeg};
use super::work::{
    Cancellable, Reporter, Span, Stage, Timeline, blocking, effort, ensure_live, ms, open_gpu,
    stem, unless_cancelled,
};
use super::{AutoInput, AutoOutput, CandidateSummary, JudgeSummary, Timings};
use crate::auto::auto_params;
use crate::finish;
use crate::judge::{
    Claude, CompareVerdict, JudgeImage, JudgeOptions, PhotoBrief, PlannedLook, RankVerdict,
};
use crate::objective::LookSpec;
use crate::render::RenderEngine;
use crate::search::{self, Candidate, SearchConfig};
use crate::types::{Gpu, PARAM_SPECS, Params, Scene, Stats};

/// Images Claude ranks at once: the finalists and the neutral render.
const MAX_SHOWN: usize = 7;
/// Per Claude call, its retry included.
const CLAUDE_TIMEOUT_S: u64 = 300;

/// Usual stage durations in seconds, for the progress bar.
const LOAD_S: f64 = 3.0;
const PLAN_S: f64 = 30.0;
const SEARCH_S: f64 = 12.0;
const RANK_S: f64 = 40.0;
const REFINE_S: f64 = 45.0;
const EXPORT_S: f64 = 6.0;

/// What a run keeps on the GPU between its stages.
struct Bench {
    gpu: Gpu,
    engine: RenderEngine,
    photo: Arc<Photo>,
    /// The photo's images with the run's lens corrections (every candidate
    /// has the automatic ones).
    scene: Arc<Scene>,
}

/// The neutral render and what follows from it, before the search.
struct First {
    stats: Stats,
    auto: Params,
    before: Vec<u8>,
    /// Neutral and auto as Claude sees them, when it plans.
    planner: Option<(JudgeImage, JudgeImage)>,
}

/// Claude's refinement after the polish, and how its A/B went.
struct Refined {
    candidate: Candidate,
    preview: ArtifactRef,
    rationale: String,
    won: bool,
}

pub(super) async fn run(cx: &RunCx, input: AutoInput) -> Result<AutoOutput> {
    let started = Instant::now();
    let effort = effort(&input.effort)?;
    let stem = stem(input.filename.as_deref());
    let plan = input.use_claude && input.plan;
    let refine = input.use_claude && input.refine;
    input.export.check()?;
    let full_res = input.export.full_res();
    let timeline = Timeline::new(&[
        (Stage::Load, LOAD_S),
        (Stage::Plan, if plan { PLAN_S } else { 0.0 }),
        (Stage::Search, SEARCH_S),
        (Stage::Rank, if input.use_claude { RANK_S } else { 0.5 }),
        (Stage::Refine, if refine { REFINE_S } else { 0.0 }),
        (Stage::Export, if full_res { EXPORT_S } else { 0.2 }),
    ]);
    let mut timings = Timings::default();
    let mut claude_failures = 0u32;

    // Decode and develop (or reuse), then measure the neutral render.
    let gpu = open_gpu(cx).await?;
    let load = timeline.span(Stage::Load);
    let loaded = unless_cancelled(
        cx,
        cache::developed(cx, &input.raw, &gpu, load.part(0.0, 0.8)),
    )
    .await?;
    timings.decode_ms = loaded.decode_ms;
    timings.develop_ms = loaded.develop_ms;
    ensure_live(cx)?;
    let info = loaded.photo.scene.info.clone();
    let orientation = info.orientation;
    // As shot, with this photo's automatic Detail and Lens Corrections like
    // every candidate, so before and after differ only in the edit.
    let neutral = finish::with_auto(
        &Params::neutral(info.as_shot_temperature, info.as_shot_tint),
        &info,
    )
    .quantized();

    let measuring = Instant::now();
    let photo = loaded.photo.clone();
    let (mut bench, first) = blocking(cx, load.part(0.8, 1.0), move |r| {
        r.progress(0.0, "Measuring the neutral render");
        let scene = photo.scene_for(&gpu, &neutral)?;
        let mut engine = RenderEngine::new(&gpu, &scene)?;
        let (stats, _) = search::evaluate(&mut engine, &scene, &neutral)?;
        let auto = auto_params(&stats, &scene).quantized();
        let before = display(&mut engine, &neutral, orientation)?;
        let planner = if plan {
            let auto_shown = display(&mut engine, &auto, orientation)?;
            Some((
                JudgeImage::from_rgb("neutral", &before.upright)?,
                JudgeImage::from_rgb("auto", &auto_shown.upright)?,
            ))
        } else {
            None
        };
        let first = First {
            stats,
            auto,
            before: before.jpeg,
            planner,
        };
        let bench = Bench {
            gpu,
            engine,
            photo,
            scene,
        };
        Ok((bench, first))
    })
    .await?;
    let mut search_ms = ms(measuring);
    let before = put_jpeg(cx, first.before).await?;
    let brief = PhotoBrief::new(&info, &first.stats);
    let (lo, hi) = search::bounds(&bench.scene);
    cx.log(format!("Auto: {}", describe(&first.auto)));

    // Claude, when asked for and installed.
    let mut failure: Option<String> = None;
    let claude = if input.use_claude {
        let opts = JudgeOptions {
            model: input.model.trim().to_owned(),
            effort: Some(effort.clone()),
            timeout_s: CLAUDE_TIMEOUT_S,
            cache_dir: cx
                .resources
                .get::<DataDir>()
                .map(|dir| dir.0.join("judge-cache")),
        };
        match Claude::locate(opts) {
            Ok(claude) => Some(claude),
            Err(e) => {
                cx.log(format!(
                    "Claude is not available, so the objective chooses: {e:#}"
                ));
                failure = Some(format!("{e:#}"));
                None
            }
        }
    } else {
        None
    };
    let mut judge_ms = 0.0;

    // Plan: up to three looks of Claude's for the search to start from.
    let mut proposals: Vec<PlannedLook> = Vec::new();
    let mut scene: Option<String> = None;
    if let (Some(claude), Some((neutral_image, auto_image))) = (&claude, &first.planner) {
        cx.progress(
            timeline.span(Stage::Plan).at(0.0),
            "Claude is planning looks",
        );
        let step = claude.plan(&brief, neutral_image, auto_image, &first.auto, &lo, &hi);
        let (planned, wall_ms) = ask(cx, claude, "plan", step).await;
        judge_ms += wall_ms;
        match planned {
            Ok(planned) => {
                if !planned.scene.is_empty() {
                    cx.log(format!("Claude's read of the scene: {}", planned.scene));
                    scene = Some(planned.scene);
                }
                for look in &planned.looks {
                    cx.log(format!(
                        "Claude proposes \"{}\": {} ({})",
                        look.name,
                        look.rationale,
                        describe(&look.params)
                    ));
                }
                proposals = planned.looks;
            }
            Err(e) => {
                ensure_live(cx)?;
                claude_failures += 1;
                cx.log(format!(
                    "Claude could not plan looks, so the search runs the built-in ones: {e:#}"
                ));
            }
        }
    }

    // Search every look over the same archive of renders.
    ensure_live(cx)?;
    let mut looks = LookSpec::builtin();
    let mut rationales: HashMap<String, String> = HashMap::new();
    for (i, look) in proposals.iter().enumerate() {
        let id = format!("claude-{}", i + 1);
        rationales.insert(format!("look:{id}"), look.rationale.clone());
        looks.push(LookSpec::proposal(&id, &look.name, look.params));
    }
    let cfg = SearchConfig {
        evals_per_look: input.evals_per_look.max(1) as usize,
        ..SearchConfig::default()
    };
    let searching = Instant::now();
    let look_count = looks.len();
    let labels: Vec<String> = looks.iter().map(|l| l.label.clone()).collect();
    let search_cfg = cfg.clone();
    let (b, result) = blocking(cx, timeline.span(Stage::Search), move |r| {
        let mut bench = bench;
        let Bench { engine, scene, .. } = &mut bench;
        let mut eval = Cancellable {
            engine,
            cancel: r.cancel(),
        };
        let mut report =
            |fraction: f32, message: &str| search_progress(r, &labels, fraction, message);
        let result = search::run_with(&mut eval, scene, &looks, &search_cfg, &mut report)?;
        Ok((bench, result))
    })
    .await?;
    bench = b;
    search_ms += ms(searching);
    let evaluations = result.evaluations;

    // The finalists, best first, and the neutral render to fall back on.
    let mut pool: Vec<Candidate> = result
        .finalists
        .iter()
        .take(MAX_SHOWN - 1)
        .cloned()
        .collect();
    if !pool.iter().any(|c| c.params == result.neutral.params) {
        pool.push(result.neutral.clone());
    }
    cx.log(format!(
        "Finalists: {}",
        pool.iter()
            .map(|c| format!("{} ({:.2})", c.label, c.score.total))
            .collect::<Vec<_>>()
            .join(", ")
    ));
    ensure_live(cx)?;
    let rank_span = timeline.span(Stage::Rank);
    let judging = claude.is_some();
    let shown: Vec<(String, Params)> = pool.iter().map(|c| (c.id.clone(), c.params)).collect();
    let (b, renders) = blocking(cx, rank_span.part(0.0, 0.05), move |r| {
        let mut bench = bench;
        r.progress(0.0, format!("Rendering {} candidates", shown.len()));
        let mut renders = Vec::with_capacity(shown.len());
        for (id, params) in &shown {
            r.check()?;
            let rendered = display(&mut bench.engine, params, orientation)?;
            let image = if judging {
                Some(JudgeImage::from_rgb(id.clone(), &rendered.upright)?)
            } else {
                None
            };
            renders.push((rendered.jpeg, image));
        }
        Ok((bench, renders))
    })
    .await?;
    bench = b;
    let mut previews = Vec::with_capacity(renders.len());
    let mut images = Vec::with_capacity(renders.len());
    for (jpeg, image) in renders {
        previews.push(put_jpeg(cx, jpeg).await?);
        images.extend(image);
    }

    // Rank: two order-swapped calls, run at once.
    let mut verdict: Option<RankVerdict> = None;
    if let Some(claude) = &claude {
        cx.progress(
            rank_span.at(0.05),
            format!("Claude is ranking {} candidates", images.len()),
        );
        let (ranked, wall_ms) = ask(cx, claude, "rank", claude.rank(&brief, &images)).await;
        judge_ms += wall_ms;
        match ranked {
            Ok(ranked) => {
                let order: Vec<&str> = ranked
                    .order
                    .iter()
                    .filter_map(|&i| pool.get(i).map(|c| c.label.as_str()))
                    .collect();
                cx.log(format!(
                    "Claude ranks {} (agreement {:.0}%)",
                    order.join(" > "),
                    ranked.agreement * 100.0
                ));
                if let Some(best) = pool.get(ranked.winner) {
                    cx.log(format!(
                        "Claude's winner: {}. {}",
                        best.label, ranked.reasons
                    ));
                }
                verdict = Some(ranked);
            }
            Err(e) => {
                ensure_live(cx)?;
                claude_failures += 1;
                cx.log(format!(
                    "Claude could not rank the finalists, so the objective chooses: {e:#}"
                ));
                failure = Some(format!("{e:#}"));
            }
        }
    }
    let winner = verdict.as_ref().map_or(0, |v| v.winner.min(pool.len() - 1));

    // Refine: one tweak of the winner, polished by the search and kept
    // only if it wins a blind A/B in both orders.
    let mut refined: Option<Refined> = None;
    let mut refine_note: Option<String> = None;
    if let (true, Some(claude), Some(_)) = (refine, &claude, &verdict) {
        ensure_live(cx)?;
        let span = timeline.span(Stage::Refine);
        let best = &pool[winner];
        cx.progress(span.at(0.0), "Claude is refining the winner");
        let step = claude.refine(&brief, &images[winner], &best.params, &lo, &hi);
        let (proposal, wall_ms) = ask(cx, claude, "refine", step).await;
        judge_ms += wall_ms;
        match proposal {
            Err(e) => {
                ensure_live(cx)?;
                claude_failures += 1;
                cx.log(format!("Claude could not refine the winner: {e:#}"));
                refine_note = Some(format!("The refinement failed: {e:#}"));
            }
            Ok(proposal) if proposal.params.quantized() == best.params => {
                cx.log(format!(
                    "Claude keeps the winner as it is: {}",
                    proposal.rationale
                ));
                refine_note = Some(join([
                    proposal.rationale.as_str(),
                    "No change was proposed.",
                ]));
            }
            Ok(proposal) => {
                let target = proposal.params.quantized();
                cx.log(format!(
                    "Claude's refinement: {} ({})",
                    proposal.rationale,
                    describe(&target)
                ));
                cx.progress(span.at(0.3), "Polishing the refinement");
                let polishing = Instant::now();
                let polish_cfg = cfg.clone();
                let (b, (polished, jpeg, image)) = blocking(cx, span.part(0.3, 0.4), move |r| {
                    let mut bench = bench;
                    let Bench { engine, scene, .. } = &mut bench;
                    let mut eval = Cancellable {
                        engine,
                        cancel: r.cancel(),
                    };
                    let polished = search::polish_with(&mut eval, scene, &target, &polish_cfg)?;
                    let rendered = display(engine, &polished.params, orientation)?;
                    let image = JudgeImage::from_rgb("refined", &rendered.upright)?;
                    Ok((bench, (polished, rendered.jpeg, image)))
                })
                .await?;
                bench = b;
                search_ms += ms(polishing);
                if polished.params == best.params {
                    cx.log("The polished refinement came back to the winner");
                    refine_note = Some(join([
                        proposal.rationale.as_str(),
                        "Fixing its clipping led back to the winner's settings.",
                    ]));
                } else {
                    let preview = put_jpeg(cx, jpeg).await?;
                    cx.progress(
                        span.at(0.45),
                        "Claude is comparing the refinement with the winner",
                    );
                    let step = claude.compare(&brief, &images[winner], &image);
                    let (compared, wall_ms) = ask(cx, claude, "compare", step).await;
                    judge_ms += wall_ms;
                    let won = match compared {
                        Ok(cmp) => {
                            cx.log(format!(
                                "The refinement won {} of {} blind comparisons: {}",
                                cmp.votes_for_b,
                                cmp.votes,
                                if cmp.prefer_b {
                                    "it becomes the result"
                                } else {
                                    "the winner stays"
                                }
                            ));
                            refine_note = Some(compare_note(&proposal.rationale, &cmp));
                            cmp.prefer_b
                        }
                        Err(e) => {
                            ensure_live(cx)?;
                            claude_failures += 1;
                            cx.log(format!("Claude could not compare the refinement: {e:#}"));
                            refine_note = Some(join([
                                proposal.rationale.as_str(),
                                &format!("Not kept: the blind comparison failed ({e:#})."),
                            ]));
                            false
                        }
                    };
                    refined = Some(Refined {
                        candidate: polished,
                        preview,
                        rationale: proposal.rationale,
                        won,
                    });
                }
            }
        }
    }

    // The chosen edit at full resolution.
    ensure_live(cx)?;
    let (chosen, after) = match &refined {
        Some(r) if r.won => (&r.candidate, &r.preview),
        _ => (&pool[winner], &previews[winner]),
    };
    cx.log(format!(
        "Chosen: {} ({}), score {:.2}",
        chosen.label,
        describe(&chosen.params),
        chosen.score.total
    ));
    let Bench {
        gpu, engine, photo, ..
    } = bench;
    // The export needs the engine's GPU memory more than the engine does.
    blocking(cx, Span::NONE, move |_| {
        drop(engine);
        Ok(())
    })
    .await?;
    let exporting = Instant::now();
    let files = export::export(
        cx,
        &gpu,
        &photo,
        &chosen.params,
        &input.export,
        timeline.span(Stage::Export),
    )
    .await?;
    timings.export_ms = ms(exporting);

    // What the page shows: every candidate Claude saw, plus the refinement.
    let rank_of = |i: usize| {
        let v = verdict.as_ref()?;
        let at = v.order.iter().position(|&c| c == i)?;
        Some(at as u32 + 1)
    };
    let note_of = |i: usize, id: &str| {
        verdict
            .as_ref()
            .and_then(|v| v.notes.get(i).cloned().flatten())
            .filter(|n| !n.is_empty())
            .or_else(|| rationales.get(id).cloned())
    };
    let mut candidates: Vec<CandidateSummary> = pool
        .iter()
        .zip(&previews)
        .enumerate()
        .map(|(i, (c, preview))| summary(c, preview, rank_of(i), note_of(i, &c.id)))
        .collect();
    if let Some(r) = &refined {
        candidates.push(summary(
            &r.candidate,
            &r.preview,
            None,
            Some(r.rationale.clone()).filter(|n| !n.is_empty()),
        ));
    }

    let judge = input.use_claude.then(|| {
        let usage = claude.as_ref().map(Claude::usage).unwrap_or_default();
        JudgeSummary {
            model: input.model.trim().to_owned(),
            scene: scene.or_else(|| verdict.as_ref().and_then(|v| v.scene.clone())),
            reasons: verdict
                .as_ref()
                .map(|v| v.reasons.clone())
                .unwrap_or_default(),
            agreement: verdict.as_ref().map(|v| v.agreement),
            refined: refined.as_ref().is_some_and(|r| r.won),
            refine_note,
            calls: usage.calls,
            cache_hits: usage.cache_hits,
            cost_usd: usage.cost_usd,
            error: failure,
        }
    });

    timings.search_ms = search_ms;
    timings.judge_ms = judge_ms;
    timings.total_ms = ms(started);
    for (key, value) in [
        ("decode_ms", timings.decode_ms),
        ("develop_ms", timings.develop_ms),
        ("search_ms", timings.search_ms),
        ("judge_ms", timings.judge_ms),
        ("export_ms", timings.export_ms),
        ("total_ms", timings.total_ms),
    ] {
        cx.metric(key, value, "ms");
    }
    cx.metric("evaluations", evaluations as f64, "");
    cx.metric("looks", look_count as f64, "");
    cx.metric("candidates", candidates.len() as f64, "");
    cx.metric("chosen_score", f64::from(chosen.score.total), "");
    if let Some(judge) = &judge {
        cx.metric("claude_calls", f64::from(judge.calls), "");
        cx.metric("claude_cache_hits", f64::from(judge.cache_hits), "");
        cx.metric("claude_cost", judge.cost_usd, "USD");
        cx.metric("claude_failures", f64::from(claude_failures), "");
        cx.metric("refined", f64::from(u8::from(judge.refined)), "");
        if let Some(agreement) = judge.agreement {
            cx.metric("rank_agreement", f64::from(agreement), "");
        }
    }
    cx.progress(1.0, "Done");

    Ok(AutoOutput {
        info,
        params: chosen.params,
        chosen: chosen.id.clone(),
        candidates,
        judge,
        before,
        after: after.clone(),
        jpeg: files.jpeg,
        tiff: files.tiff,
        png: files.png,
        xmp: files.xmp,
        stem,
        timings,
    })
}

/// One Claude step: dropped (its CLI processes killed) when the run is
/// cancelled, timed, and recorded per kind as metrics and a log line.
/// Returns the step's result and its wall time in milliseconds.
async fn ask<T>(
    cx: &RunCx,
    claude: &Claude,
    kind: &str,
    step: impl Future<Output = Result<T>>,
) -> (Result<T>, f64) {
    let before = claude.usage();
    let started = Instant::now();
    let result = unless_cancelled(cx, step).await;
    let wall_s = started.elapsed().as_secs_f64();
    let after = claude.usage();
    let calls = after.calls.saturating_sub(before.calls);
    let cached = after.cache_hits.saturating_sub(before.cache_hits);
    cx.metric(format!("claude_{kind}_s"), wall_s, "s");
    cx.metric(format!("claude_{kind}_calls"), f64::from(calls), "");
    if calls > 0 {
        let per_call = (after.seconds - before.seconds) / f64::from(calls);
        cx.metric(format!("claude_{kind}_s_per_call"), per_call, "s");
    }
    let mut line = format!(
        "Claude {kind}: {} in {wall_s:.1} s",
        plural(calls, "call", "calls")
    );
    if cached > 0 {
        line.push_str(&format!(
            ", {} from the cache",
            plural(cached, "answer", "answers")
        ));
    }
    if result.is_ok() {
        let cost = after.cost_usd - before.cost_usd;
        if cost > 0.0 {
            line.push_str(&format!(", ${cost:.3}"));
        }
    }
    cx.log(line);
    (result, wall_s * 1000.0)
}

/// What the bar says while the search runs: the look being searched. The
/// search reports "<label> (k/n): score ..." as it finishes look k.
fn search_progress(r: &Reporter, labels: &[String], fraction: f32, message: &str) {
    let n = labels.len();
    let text = match finished_looks(message) {
        Some(done) => {
            r.log(format!("Search: {message}"));
            match labels.get(done) {
                Some(next) => format!("Searching {next} ({}/{n})", done + 1),
                None => "Choosing finalists".to_owned(),
            }
        }
        None if fraction > 0.0 && fraction < 1.0 && n > 0 => {
            format!("Searching {} (1/{n})", labels[0])
        }
        None => message.to_owned(),
    };
    r.progress(f64::from(fraction), text);
}

/// k of a "... (k/n) ..." progress message: how many looks are done.
fn finished_looks(message: &str) -> Option<usize> {
    let open = message.rfind('(')?;
    let (inside, _) = message[open + 1..].split_once(')')?;
    let (done, total) = inside.split_once('/')?;
    total.trim().parse::<usize>().ok()?;
    done.trim().parse().ok()
}

fn summary(
    c: &Candidate,
    preview: &ArtifactRef,
    rank: Option<u32>,
    note: Option<String>,
) -> CandidateSummary {
    CandidateSummary {
        id: c.id.clone(),
        label: c.label.clone(),
        source: c.source.as_str().to_owned(),
        params: c.params,
        score: c.score.total,
        rank,
        preview: preview.clone(),
        note,
    }
}

/// The refinement's note: what Claude changed, then how the A/B went.
fn compare_note(rationale: &str, cmp: &CompareVerdict) -> String {
    let verdict = format!(
        "{}: it won {} of {} blind comparisons with the winner.",
        if cmp.prefer_b { "Kept" } else { "Not kept" },
        cmp.votes_for_b,
        cmp.votes
    );
    // The judge leads its reasons with the tally in its own words.
    let tally = format!("b won {} of {} orders.", cmp.votes_for_b, cmp.votes);
    let detail = cmp
        .reasons
        .strip_prefix(&tally)
        .unwrap_or(&cmp.reasons)
        .trim();
    join([rationale, &verdict, detail])
}

fn join<'a>(parts: impl IntoIterator<Item = &'a str>) -> String {
    parts
        .into_iter()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn plural(n: u32, one: &str, many: &str) -> String {
    format!("{n} {}", if n == 1 { one } else { many })
}

/// "5450 K, tint +3, exposure +0.35, contrast +10, highlights -30" for the
/// log: white balance and exposure always, other settings when not at
/// Lightroom's default.
fn describe(p: &Params) -> String {
    let mut parts = vec![
        format!("{:.0} K", p.temperature),
        format!("tint {:+.0}", p.tint),
        format!("exposure {:+.2}", p.exposure),
    ];
    for (spec, value) in PARAM_SPECS.iter().zip(p.to_array()).skip(3) {
        if Some(value) != spec.neutral {
            parts.push(format!("{} {value:+.0}", spec.key));
        }
    }
    parts.join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_how_many_looks_are_done() {
        assert_eq!(finished_looks("Natural (1/5): score 0.83"), Some(1));
        assert_eq!(
            finished_looks("Warm (golden) light (4/7): score -1.20, constraints not met"),
            Some(4)
        );
        assert_eq!(finished_looks("Searching looks"), None);
        assert_eq!(finished_looks("Rendering the neutral and auto edits"), None);
        assert_eq!(finished_looks("Warm (golden)"), None);
    }

    #[test]
    fn compare_notes_say_what_happened() {
        let cmp = CompareVerdict {
            prefer_b: true,
            votes_for_b: 2,
            votes: 2,
            reasons: "b won 2 of 2 orders. The brighter version keeps the sky.".into(),
        };
        assert_eq!(
            compare_note("Lift the shadows a little.", &cmp),
            "Lift the shadows a little. Kept: it won 2 of 2 blind comparisons with the winner. \
             The brighter version keeps the sky."
        );
        let lost = CompareVerdict {
            prefer_b: false,
            votes_for_b: 1,
            votes: 2,
            reasons: "Something else.".into(),
        };
        assert_eq!(
            compare_note("", &lost),
            "Not kept: it won 1 of 2 blind comparisons with the winner. Something else."
        );
    }

    #[test]
    fn describes_sliders_briefly() {
        let mut p = Params::neutral(5450.0, 3.0);
        p.exposure = 0.35;
        p.highlights = -30.0;
        assert_eq!(
            describe(&p),
            "5450 K, tint +3, exposure +0.35, highlights -30"
        );
    }
}
