use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::{ArtifactStore, EventSink, RunEvent};

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
        Self {
            run_id,
            artifacts,
            resources,
            cancel: CancellationToken::new(),
            events,
        }
    }

    pub fn progress(&self, fraction: f64, message: impl Into<String>) {
        self.events.emit(RunEvent::Progress {
            fraction: fraction.clamp(0.0, 1.0),
            message: message.into(),
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
