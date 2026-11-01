use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use ts_rs::TS;

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(tag = "type", rename_all = "snake_case")]
#[ts(export)]
pub enum RunEvent {
    Progress {
        fraction: f64,
        message: String,
    },
    Log {
        message: String,
    },
    Metric {
        key: String,
        value: f64,
        unit: String,
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
