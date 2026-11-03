//! Persistent records: runs, artifacts, lineage and metrics.
//!
//! Everything above this crate talks to [`Store`], never to SQL. Each backend
//! (SQLite today; PostgreSQL and Oracle Autonomous Database next) implements
//! the trait with its own dialect, and all backends share one logical schema
//! (`migrations/<backend>/`, kept in lockstep by a test below).

mod migrations;
mod model;
mod sqlite;

use std::sync::Arc;

use async_trait::async_trait;
use spl_core::{Result, bail};

pub use migrations::{Migration, ORACLE, POSTGRES, SQLITE};
pub use model::{
    Announcement, ArtifactRecord, MetricRecord, NewRun, RunArtifact, RunRecord, RunStatus,
};
pub use sqlite::SqliteStore;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Sqlite,
    Postgres,
    Oracle,
}

#[async_trait]
pub trait Store: Send + Sync {
    fn backend(&self) -> Backend;
    /// Applies pending migrations; safe to call on every start.
    async fn migrate(&self) -> Result<()>;

    async fn create_run(&self, run: &NewRun) -> Result<()>;
    async fn start_run(&self, id: &str, started_at_ms: i64) -> Result<()>;
    async fn finish_run(
        &self,
        id: &str,
        status: RunStatus,
        output: Option<&serde_json::Value>,
        error_message: Option<&str>,
        finished_at_ms: i64,
    ) -> Result<()>;
    async fn get_run(&self, id: &str) -> Result<Option<RunRecord>>;
    /// Newest first.
    async fn list_runs(&self, op_id: Option<&str>, limit: u32) -> Result<Vec<RunRecord>>;
    /// Marks every `queued` or `running` run as `failed` with `error_message`
    /// and returns how many changed. A server calls it once at startup: runs
    /// a previous process left unfinished will never finish.
    async fn fail_unfinished_runs(&self, error_message: &str, finished_at_ms: i64) -> Result<u64>;

    /// Idempotent: artifacts are content-addressed.
    async fn put_artifact(&self, artifact: &ArtifactRecord) -> Result<()>;
    async fn get_artifact(&self, id: &str) -> Result<Option<ArtifactRecord>>;
    async fn link_artifact(
        &self,
        run_id: &str,
        artifact_id: &str,
        direction: &str,
        slot_name: &str,
    ) -> Result<()>;
    /// Ordered by direction, then slot name (as text).
    async fn list_run_artifacts(&self, run_id: &str) -> Result<Vec<RunArtifact>>;
    /// Deletes the record of an artifact `run_id` links to and nothing else
    /// references (no other run's link, no announcement), together with
    /// that run's links to it. False, and nothing changed, otherwise. The
    /// caller then deletes the bytes.
    async fn forget_artifact(&self, run_id: &str, artifact_id: &str) -> Result<bool>;
    /// Deletes a run with its links, announcements and metrics, and the
    /// records of the artifacts it linked or announced that nothing else
    /// references any more; returns those artifact ids (the caller then
    /// deletes the bytes). Empty when there is no such run.
    async fn delete_run(&self, run_id: &str) -> Result<Vec<String>>;

    /// Makes `announcement` the run's current artifact for its slot,
    /// replacing the previous one. The artifact must be recorded.
    async fn announce(&self, run_id: &str, announcement: &Announcement) -> Result<()>;
    /// The current announcement of every slot, by slot name.
    async fn list_announcements(&self, run_id: &str) -> Result<Vec<Announcement>>;

    async fn add_metrics(&self, metrics: &[MetricRecord]) -> Result<()>;
    async fn list_metrics(&self, run_id: &str) -> Result<Vec<MetricRecord>>;
}

/// Opens the backend named by the URL scheme and runs migrations.
pub async fn connect(url: &str) -> Result<Arc<dyn Store>> {
    let store: Arc<dyn Store> = match url.split_once(':').map(|(scheme, _)| scheme) {
        Some("sqlite") => Arc::new(SqliteStore::connect(url).await?),
        Some("postgres" | "postgresql") => {
            bail!("the PostgreSQL backend is not built yet; see docs/DATABASE.md")
        }
        Some("oracle") => bail!("the Oracle backend is not built yet; see docs/DATABASE.md"),
        _ => bail!("unsupported database URL `{url}`"),
    };
    store.migrate().await?;
    Ok(store)
}
