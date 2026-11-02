use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::{ArtifactRef, ArtifactStore, EventSink, RunEvent};

/// Shared, host-owned services (GPU pool, DB handle, caches) looked up by type.
/// Keeps this crate free of CUDA and database dependencies.
#[derive(Default, Clone)]
pub struct Resources {
    items: HashMap<TypeId, Arc<dyn Any + Send + Sync>>,
}

impl Resources {
    pub fn insert<T: Send + Sync + 'static>(&mut self, value: Arc<T>) {
        self.items.insert(TypeId::of::<T>(), value);
    }

    pub fn get<T: Send + Sync + 'static>(&self) -> Option<Arc<T>> {
        self.items
            .get(&TypeId::of::<T>())
            .and_then(|item| item.clone().downcast::<T>().ok())
    }
}

/// The host's data directory (database, artifacts, caches). Hosts insert it
/// into [`Resources`]; ops keep their own caches under it.
#[derive(Debug, Clone)]
pub struct DataDir(pub std::path::PathBuf);

/// Everything an op may touch while it runs.
pub struct RunCx {
    pub run_id: String,
    pub artifacts: Arc<dyn ArtifactStore>,
    pub resources: Resources,
    /// Fires when someone asks the host to stop this run. Long loops poll
    /// [`RunCx::is_cancelled`]; blocking work takes a clone of the token.
    pub cancel: CancellationToken,
    events: EventSink,
}

impl RunCx {
    pub fn new(
        run_id: String,
        artifacts: Arc<dyn ArtifactStore>,
        resources: Resources,
        events: EventSink,
    ) -> Self {
        Self::with_cancel(
            run_id,
            artifacts,
            resources,
            events,
            CancellationToken::new(),
        )
    }

    /// Like [`RunCx::new`], with a token the host keeps, so it can cancel
    /// the run from outside (`POST /v1/runs/{id}/cancel`).
    pub fn with_cancel(
        run_id: String,
        artifacts: Arc<dyn ArtifactStore>,
        resources: Resources,
        events: EventSink,
        cancel: CancellationToken,
    ) -> Self {
        Self {
            run_id,
            artifacts,
            resources,
            cancel,
            events,
        }
    }

    pub fn progress(&self, fraction: f64, message: impl Into<String>) {
        self.events.emit(RunEvent::Progress {
            fraction: fraction.clamp(0.0, 1.0),
            message: message.into(),
            data: None,
        });
    }

    /// [`progress`](Self::progress) with structured state a custom page
    /// reads, such as `{"stage": "refine", "iteration": 7000, "gaussians":
    /// 1.2e6}`. Like the message, only the latest is kept and only while the
    /// run goes.
    pub fn progress_with(
        &self,
        fraction: f64,
        message: impl Into<String>,
        data: serde_json::Value,
    ) {
        self.events.emit(RunEvent::Progress {
            fraction: fraction.clamp(0.0, 1.0),
            message: message.into(),
            data: Some(data).filter(|d| !d.is_null()),
        });
    }

    /// Shows an artifact this run stored as its current `slot` (`"points"`,
    /// `"cameras"`, `"splat"`) before the run ends: `GET /v1/runs/{id}`
    /// lists the latest artifact per slot with a revision that grows on
    /// every announcement, while the run goes and after it ends. `info` is
    /// a small JSON value shown with it (`Value::Null` for none). The
    /// artifact is kept; see [`announce_snapshot`](Self::announce_snapshot)
    /// for previews that replace each other.
    pub fn announce(
        &self,
        slot: impl Into<String>,
        artifact: &ArtifactRef,
        info: serde_json::Value,
    ) {
        self.emit_announce(slot.into(), artifact, false, info);
    }

    /// Like [`announce`](Self::announce), for snapshots that replace each
    /// other (a splat every few seconds of training): when the run ends,
    /// the host deletes the snapshots a later one of the same slot
    /// replaced, unless the run's output names them or something else uses
    /// the same bytes. The latest of each slot is kept.
    pub fn announce_snapshot(
        &self,
        slot: impl Into<String>,
        artifact: &ArtifactRef,
        info: serde_json::Value,
    ) {
        self.emit_announce(slot.into(), artifact, true, info);
    }

    fn emit_announce(
        &self,
        slot: String,
        artifact: &ArtifactRef,
        snapshot: bool,
        info: serde_json::Value,
    ) {
        self.events.emit(RunEvent::Announce {
            slot,
            artifact: artifact.clone(),
            snapshot,
            info: Some(info).filter(|i| !i.is_null()),
        });
    }

    pub fn log(&self, message: impl Into<String>) {
        self.events.emit(RunEvent::Log {
            message: message.into(),
        });
    }

    /// Recorded in the database and shown in benchmark views.
    pub fn metric(&self, key: impl Into<String>, value: f64, unit: impl Into<String>) {
        self.events.emit(RunEvent::Metric {
            key: key.into(),
            value,
            unit: unit.into(),
        });
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }
}
