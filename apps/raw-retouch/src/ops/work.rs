//! What every op shares: blocking GPU work that reports live progress,
//! cancellation, the progress bar's layout and download names.

use std::future::Future;

use spl_core::{CancellationToken, Result, RunCx, anyhow, bail};
use tokio::sync::mpsc::{UnboundedSender, unbounded_channel};

use crate::render::RenderEngine;
use crate::search::Evaluator;
use crate::types::{Gpu, Params, Stats};

/// The error every stage stops with once the run is cancelled.
const CANCELLED: &str = "cancelled";

/// Stops the run between stages once someone cancelled it.
pub(super) fn ensure_live(cx: &RunCx) -> Result<()> {
    if cx.is_cancelled() {
        bail!(CANCELLED);
    }
    Ok(())
}

/// `future`, dropped as soon as the run is cancelled. Dropping a Claude
/// request kills its CLI processes (they are spawned with `kill_on_drop`).
pub(super) async fn unless_cancelled<T>(
    cx: &RunCx,
    future: impl Future<Output = Result<T>>,
) -> Result<T> {
    tokio::select! {
        biased;
        () = cx.cancel.cancelled() => bail!(CANCELLED),
        out = future => out,
    }
}

/// This run's handle on device 0 (a stream of its own; compiled kernels
/// are shared by the process).
pub(super) async fn open_gpu(cx: &RunCx) -> Result<Gpu> {
    blocking(cx, Span::NONE, |_| Gpu::open()).await
}

/// A stretch of the run's progress bar, 0 to 1.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Span {
    pub start: f64,
    pub len: f64,
}

impl Span {
    pub const NONE: Span = Span {
        start: 0.0,
        len: 0.0,
    };

    /// The bar's position `fraction` of the way through this span.
    pub fn at(&self, fraction: f64) -> f64 {
        let f = if fraction.is_finite() {
            fraction.clamp(0.0, 1.0)
        } else {
            0.0
        };
        self.start + self.len * f
    }

    /// The part of this span from `from` to `to` (fractions of it).
    pub fn part(&self, from: f64, to: f64) -> Span {
        let start = self.at(from);
        Span {
            start,
            len: (self.at(to) - start).max(0.0),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Stage {
    /// Read, decode and develop the RAW (or take it from the cache).
    Load,
    /// Display-size renders.
    Render,
    Plan,
    Search,
    Rank,
    Refine,
    Export,
}

/// The progress bar's layout: each stage gets a share by its usual
/// duration, so the bar moves at a steady pace.
pub(super) struct Timeline(Vec<(Stage, Span)>);

impl Timeline {
    /// Stages in order with their usual durations in seconds; 0 for a
    /// stage that is off.
    pub fn new(stages: &[(Stage, f64)]) -> Self {
        let weight = |w: f64| if w.is_finite() { w.max(0.0) } else { 0.0 };
        let total = stages.iter().map(|&(_, w)| weight(w)).sum::<f64>();
        let mut start = 0.0;
        let spans = stages
            .iter()
            .map(|&(stage, w)| {
                let len = if total > 0.0 { weight(w) / total } else { 0.0 };
                let span = Span { start, len };
                start += len;
                (stage, span)
            })
            .collect();
        Self(spans)
    }

    /// Where `stage` sits on the bar; a stage the timeline lacks takes no room at the end.
    pub fn span(&self, stage: Stage) -> Span {
        self.0.iter().find(|(s, _)| *s == stage).map_or(
            Span {
                start: 1.0,
                len: 0.0,
            },
            |&(_, span)| span,
        )
    }
}

enum Event {
    Progress(f64, String),
    Log(String),
}

/// What blocking work can tell its run while it goes.
pub(super) struct Reporter {
    events: UnboundedSender<Event>,
    cancel: CancellationToken,
    span: Span,
}

impl Reporter {
    /// `fraction` is of this piece of work; the run's bar shows it within its span.
    pub fn progress(&self, fraction: f64, message: impl Into<String>) {
        // The run may have stopped listening (it was dropped); nothing to do then.
        let _ = self
            .events
            .send(Event::Progress(self.span.at(fraction), message.into()));
    }

    pub fn log(&self, message: impl Into<String>) {
        let _ = self.events.send(Event::Log(message.into()));
    }

    /// Stops the work once the run is cancelled.
    pub fn check(&self) -> Result<()> {
        if self.cancel.is_cancelled() {
            bail!(CANCELLED);
        }
        Ok(())
    }

    pub fn cancel(&self) -> &CancellationToken {
        &self.cancel
    }
}

/// Runs `work` on the blocking pool (CUDA and heavy CPU work), forwarding
/// its progress and log lines to the run as they come, and waits for it:
/// work checks [`Reporter::check`] between steps, so a cancelled run
/// returns soon and never leaves GPU work running behind it.
pub(super) async fn blocking<T, F>(cx: &RunCx, span: Span, work: F) -> Result<T>
where
    T: Send + 'static,
    F: FnOnce(&Reporter) -> Result<T> + Send + 'static,
{
    let (events, mut rx) = unbounded_channel();
    let reporter = Reporter {
        events,
        cancel: cx.cancel.clone(),
        span,
    };
    let mut task = tokio::task::spawn_blocking(move || work(&reporter));
    let joined = loop {
        tokio::select! {
            Some(event) = rx.recv() => forward(cx, event),
            joined = &mut task => break joined,
        }
    };
    while let Ok(event) = rx.try_recv() {
        forward(cx, event);
    }
    match joined {
        Ok(result) => result,
        Err(e) if e.is_panic() => {
            let payload = e.into_panic();
            let message = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_owned())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "no message".into());
            Err(anyhow!("worker panicked: {message}"))
        }
        Err(e) => Err(anyhow!("worker stopped: {e}")),
    }
}

fn forward(cx: &RunCx, event: Event) {
    match event {
        Event::Progress(fraction, message) => cx.progress(fraction, message),
        Event::Log(message) => cx.log(message),
    }
}

/// The render engine for the search, failing its next batch once the run
/// is cancelled: a search stops within one generation (tens of renders).
pub(super) struct Cancellable<'a> {
    pub engine: &'a mut RenderEngine,
    pub cancel: &'a CancellationToken,
}

impl Evaluator for Cancellable<'_> {
    fn stats(&mut self, params: &[Params]) -> Result<Vec<Stats>> {
        if self.cancel.is_cancelled() {
            bail!(CANCELLED);
        }
        self.engine.stats(params)
    }
}

/// Download name stem: the file name without directories and extension,
/// or "photo" when there is none.
pub(super) fn stem(filename: Option<&str>) -> String {
    let name = filename.unwrap_or_default().trim();
    let base = name.rsplit(['/', '\\']).next().unwrap_or_default();
    let stem = match base.rfind('.') {
        Some(dot) if dot > 0 => &base[..dot],
        _ => base,
    };
    let stem = stem.trim();
    if stem.is_empty() {
        "photo".into()
    } else {
        stem.into()
    }
}

/// `claude --effort` as the auto op accepts it.
pub(super) fn effort(value: &str) -> Result<String> {
    let effort = value.trim().to_ascii_lowercase();
    match effort.as_str() {
        "low" | "medium" | "high" => Ok(effort),
        _ => bail!("effort must be \"low\", \"medium\" or \"high\", not {value:?}"),
    }
}

/// Milliseconds since `start`.
pub(super) fn ms(start: std::time::Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stems_drop_directories_and_extension() {
        assert_eq!(stem(Some("DSC01234.ARW")), "DSC01234");
        assert_eq!(stem(Some("shoot/day 1/DSC01234.arw")), "DSC01234");
        assert_eq!(stem(Some("C:\\photos\\A7R.V.ARW")), "A7R.V");
        assert_eq!(stem(Some(".hidden")), ".hidden");
        assert_eq!(stem(Some("noext")), "noext");
        assert_eq!(stem(Some("  ")), "photo");
        assert_eq!(stem(Some("dir/")), "photo");
        assert_eq!(stem(None), "photo");
    }

    #[test]
    fn effort_takes_three_levels() {
        assert_eq!(effort("medium").unwrap(), "medium");
        assert_eq!(effort(" High ").unwrap(), "high");
        assert_eq!(effort("low").unwrap(), "low");
        assert!(effort("max").is_err());
        assert!(effort("").is_err());
    }

    #[test]
    fn timeline_shares_the_bar_by_duration() {
        let t = Timeline::new(&[(Stage::Load, 1.0), (Stage::Plan, 0.0), (Stage::Search, 3.0)]);
        assert_eq!(
            t.span(Stage::Load),
            Span {
                start: 0.0,
                len: 0.25
            }
        );
        assert_eq!(
            t.span(Stage::Plan),
            Span {
                start: 0.25,
                len: 0.0
            }
        );
        assert_eq!(
            t.span(Stage::Search),
            Span {
                start: 0.25,
                len: 0.75
            }
        );
        assert_eq!(
            t.span(Stage::Export),
            Span {
                start: 1.0,
                len: 0.0
            }
        );
        let s = t.span(Stage::Search);
        assert_eq!(s.at(0.5), 0.625);
        assert_eq!(s.at(2.0), 1.0);
        assert_eq!(s.at(f64::NAN), 0.25);
        assert_eq!(
            s.part(0.0, 0.5),
            Span {
                start: 0.25,
                len: 0.375
            }
        );
        let empty = Timeline::new(&[(Stage::Load, 0.0)]);
        assert_eq!(
            empty.span(Stage::Load),
            Span {
                start: 0.0,
                len: 0.0
            }
        );
    }
}
