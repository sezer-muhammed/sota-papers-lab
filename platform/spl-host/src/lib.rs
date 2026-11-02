//! The runtime behind every front door: it runs any registered op, records
//! the run, its metrics and its environment, and serves the universal API.

mod api;
mod config;
mod host;
mod queue;
mod recording;

pub use api::{Health, RunDetail, RunProgress, RunRequest, router};
pub use config::Config;
pub use host::{Host, LiveRun, StagedFile};
pub use queue::WaitReason;
