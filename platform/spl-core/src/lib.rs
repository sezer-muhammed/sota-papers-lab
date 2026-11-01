//! The contract every SOTA Papers Lab app implements.
//!
//! An app is a crate that exposes one or more [`Op`]s and a `register`
//! function. Hosts (the server, the CLI, a benchmark, a pipeline) only ever
//! talk to ops through this crate, so apps never depend on each other.

mod artifact;
mod context;
mod event;
mod op;
mod registry;

pub use artifact::{ArtifactKind, ArtifactRef, ArtifactStore};
pub use context::{DataDir, Resources, RunCx};
pub use event::{EventSink, RunEvent};
pub use op::{BoxFuture, DynOp, Op, OpDescriptor};
pub use registry::Registry;

pub use anyhow::{Context, Error, Result, anyhow, bail};
pub use schemars::JsonSchema;
/// The type of [`RunCx::cancel`], re-exported so hosts and ops need no
/// direct `tokio-util` dependency.
pub use tokio_util::sync::CancellationToken;
