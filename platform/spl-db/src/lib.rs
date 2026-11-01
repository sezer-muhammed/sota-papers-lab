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
pub use model::{ArtifactRecord, MetricRecord, NewRun, RunRecord, RunStatus};
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
