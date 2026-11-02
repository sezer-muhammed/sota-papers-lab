use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use ts_rs::TS;

use crate::ArtifactRef;

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(tag = "type", rename_all = "snake_case")]
#[ts(export)]
pub enum RunEvent {
    Progress {
        fraction: f64,
        message: String,
        /// Structured state for a custom page (stage, iteration, counts).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[ts(optional, type = "unknown")]
        data: Option<serde_json::Value>,
    },
    Log {
        message: String,
    },
    Metric {
        key: String,
        value: f64,
        unit: String,
    },
    /// `artifact` is now the run's current `slot` (see
    /// [`RunCx::announce`](crate::RunCx::announce)).
    Announce {
        slot: String,
        artifact: ArtifactRef,
        /// Superseded snapshots of the slot are deleted when the run ends.
        snapshot: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[ts(optional, type = "unknown")]
        info: Option<serde_json::Value>,
    },
}

/// Fire-and-forget: an op never fails because nobody is listening.
#[derive(Clone, Default)]
pub struct EventSink {
    tx: Option<mpsc::UnboundedSender<RunEvent>>,
}

impl EventSink {
    pub fn channel() -> (Self, mpsc::UnboundedReceiver<RunEvent>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (Self { tx: Some(tx) }, rx)
    }

    pub fn emit(&self, event: RunEvent) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(event);
        }
    }
}
