use std::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use spl_core::{Context, Result};
use sqlx::Row;
use sqlx::sqlite::{
    SqliteConnectOptions, SqliteJournalMode, SqlitePool, SqlitePoolOptions, SqliteRow,
};

use crate::{
    ArtifactRecord, Backend, MetricRecord, NewRun, RunArtifact, RunRecord, RunStatus, SQLITE, Store,
};

/// Single-file database for one machine (local Linux box or one cloud VM).
pub struct SqliteStore {
    pool: SqlitePool,
}

impl SqliteStore {
    pub async fn connect(url: &str) -> Result<Self> {
        let options = SqliteConnectOptions::from_str(url)
            .with_context(|| format!("bad SQLite URL `{url}`"))?
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal)
            .foreign_keys(true);
        let pool = SqlitePoolOptions::new()
            .max_connections(8)
            .connect_with(options)
            .await
            .with_context(|| format!("open {url}"))?;
        Ok(Self { pool })
    }
}

fn json_text(value: Option<&serde_json::Value>) -> Option<String> {
    value.map(|v| v.to_string())
}

fn parse_json(text: Option<String>) -> Result<Option<serde_json::Value>> {
    text.map(|t| serde_json::from_str(&t))
        .transpose()
        .context("corrupt JSON column")
}

fn run_from_row(row: &SqliteRow) -> Result<RunRecord> {
    Ok(RunRecord {
        id: row.try_get("id")?,
        op_id: row.try_get("op_id")?,
        status: RunStatus::parse(row.try_get("status")?)?,
        input: parse_json(row.try_get("input_json")?)?.unwrap_or_default(),
        output: parse_json(row.try_get("output_json")?)?,
        error_message: row.try_get("error_message")?,
        host_name: row.try_get("host_name")?,
        git_sha: row.try_get("git_sha")?,
        env: parse_json(row.try_get("env_json")?)?,
        created_at_ms: row.try_get("created_at_ms")?,
        started_at_ms: row.try_get("started_at_ms")?,
        finished_at_ms: row.try_get("finished_at_ms")?,
    })
}

const RUN_COLUMNS: &str = "id, op_id, status, input_json, output_json, error_message, host_name, git_sha, env_json, created_at_ms, started_at_ms, finished_at_ms";

#[async_trait]
impl Store for SqliteStore {
    fn backend(&self) -> Backend {
        Backend::Sqlite
    }

    async fn migrate(&self) -> Result<()> {
        sqlx::raw_sql(
            "CREATE TABLE IF NOT EXISTS spl_schema_migrations (
                version       INTEGER PRIMARY KEY,
                name          TEXT NOT NULL,
                applied_at_ms INTEGER NOT NULL
            )",
        )
        .execute(&self.pool)
        .await?;

        let applied: Vec<i64> = sqlx::query("SELECT version FROM spl_schema_migrations")
            .fetch_all(&self.pool)
            .await?
            .iter()
            .map(|row| row.try_get("version"))
            .collect::<Result<_, _>>()?;

        for migration in SQLITE.iter().filter(|m| !applied.contains(&m.version)) {
            let mut tx = self.pool.begin().await?;
            sqlx::raw_sql(migration.sql)
                .execute(&mut *tx)
                .await
                .with_context(|| format!("migration {} {}", migration.version, migration.name))?;
            sqlx::query(
                "INSERT INTO spl_schema_migrations (version, name, applied_at_ms) VALUES (?, ?, ?)",
            )
            .bind(migration.version)
            .bind(migration.name)
            .bind(now_ms())
            .execute(&mut *tx)
            .await?;
            tx.commit().await?;
        }
        Ok(())
    }

    async fn create_run(&self, run: &NewRun) -> Result<()> {
        sqlx::query(
            "INSERT INTO spl_runs (id, op_id, status, input_json, host_name, git_sha, env_json, created_at_ms)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&run.id)
        .bind(&run.op_id)
        .bind(RunStatus::Queued.as_str())
        .bind(run.input.to_string())
        .bind(&run.host_name)
        .bind(&run.git_sha)
        .bind(json_text(run.env.as_ref()))
        .bind(run.created_at_ms)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn start_run(&self, id: &str, started_at_ms: i64) -> Result<()> {
        sqlx::query("UPDATE spl_runs SET status = ?, started_at_ms = ? WHERE id = ?")
            .bind(RunStatus::Running.as_str())
            .bind(started_at_ms)
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn finish_run(
        &self,
        id: &str,
        status: RunStatus,
        output: Option<&serde_json::Value>,
        error_message: Option<&str>,
        finished_at_ms: i64,
    ) -> Result<()> {
        sqlx::query(
            "UPDATE spl_runs SET status = ?, output_json = ?, error_message = ?, finished_at_ms = ? WHERE id = ?",
        )
        .bind(status.as_str())
        .bind(json_text(output))
        .bind(error_message)
        .bind(finished_at_ms)
        .bind(id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn get_run(&self, id: &str) -> Result<Option<RunRecord>> {
        let sql = format!("SELECT {RUN_COLUMNS} FROM spl_runs WHERE id = ?");
        sqlx::query(sqlx::AssertSqlSafe(sql))
            .bind(id)
            .fetch_optional(&self.pool)
            .await?
            .as_ref()
            .map(run_from_row)
            .transpose()
    }

    async fn list_runs(&self, op_id: Option<&str>, limit: u32) -> Result<Vec<RunRecord>> {
        let sql = format!(
            "SELECT {RUN_COLUMNS} FROM spl_runs
             WHERE (?1 IS NULL OR op_id = ?1)
             ORDER BY created_at_ms DESC, id DESC LIMIT ?2"
        );
        sqlx::query(sqlx::AssertSqlSafe(sql))
            .bind(op_id)
            .bind(i64::from(limit))
            .fetch_all(&self.pool)
            .await?
            .iter()
            .map(run_from_row)
            .collect()
    }

    async fn fail_unfinished_runs(&self, error_message: &str, finished_at_ms: i64) -> Result<u64> {
        let done = sqlx::query(
            "UPDATE spl_runs SET status = ?, error_message = ?, finished_at_ms = ? WHERE status IN (?, ?)",
        )
        .bind(RunStatus::Failed.as_str())
        .bind(error_message)
        .bind(finished_at_ms)
        .bind(RunStatus::Queued.as_str())
        .bind(RunStatus::Running.as_str())
        .execute(&self.pool)
        .await?;
        Ok(done.rows_affected())
    }

    async fn put_artifact(&self, a: &ArtifactRecord) -> Result<()> {
        sqlx::query(
            "INSERT INTO spl_artifacts (id, kind, media_type, size_bytes, uri, meta_json, created_at_ms)
             VALUES (?, ?, ?, ?, ?, ?, ?)
             ON CONFLICT (id) DO NOTHING",
        )
        .bind(&a.id)
        .bind(&a.kind)
        .bind(&a.media_type)
        .bind(a.size_bytes)
        .bind(&a.uri)
        .bind(json_text(a.meta.as_ref()))
        .bind(a.created_at_ms)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn get_artifact(&self, id: &str) -> Result<Option<ArtifactRecord>> {
        let row = sqlx::query(
            "SELECT id, kind, media_type, size_bytes, uri, meta_json, created_at_ms
             FROM spl_artifacts WHERE id = ?",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;
        row.map(|row| -> Result<ArtifactRecord> {
            Ok(ArtifactRecord {
                id: row.try_get("id")?,
                kind: row.try_get("kind")?,
                media_type: row.try_get("media_type")?,
                size_bytes: row.try_get("size_bytes")?,
                uri: row.try_get("uri")?,
                meta: parse_json(row.try_get("meta_json")?)?,
                created_at_ms: row.try_get("created_at_ms")?,
            })
        })
        .transpose()
    }

    async fn link_artifact(
        &self,
        run_id: &str,
        artifact_id: &str,
        direction: &str,
        slot_name: &str,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO spl_run_artifacts (run_id, artifact_id, direction, slot_name) VALUES (?, ?, ?, ?)",
        )
        .bind(run_id)
        .bind(artifact_id)
        .bind(direction)
        .bind(slot_name)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn list_run_artifacts(&self, run_id: &str) -> Result<Vec<RunArtifact>> {
        sqlx::query(
            "SELECT run_id, artifact_id, direction, slot_name FROM spl_run_artifacts
             WHERE run_id = ? ORDER BY direction, slot_name",
        )
        .bind(run_id)
        .fetch_all(&self.pool)
        .await?
        .iter()
        .map(|row| -> Result<RunArtifact> {
            Ok(RunArtifact {
                run_id: row.try_get("run_id")?,
                artifact_id: row.try_get("artifact_id")?,
                direction: row.try_get("direction")?,
                slot_name: row.try_get("slot_name")?,
            })
        })
        .collect()
    }

    async fn add_metrics(&self, metrics: &[MetricRecord]) -> Result<()> {
        let mut tx = self.pool.begin().await?;
        for m in metrics {
            sqlx::query(
                "INSERT INTO spl_metrics (run_id, seq, metric_key, metric_value, unit) VALUES (?, ?, ?, ?, ?)",
            )
            .bind(&m.run_id)
            .bind(m.seq)
            .bind(&m.key)
            .bind(m.value)
            .bind(m.unit.as_deref().filter(|u| !u.is_empty()))
            .execute(&mut *tx)
            .await?;
        }
        tx.commit().await?;
        Ok(())
    }

    async fn list_metrics(&self, run_id: &str) -> Result<Vec<MetricRecord>> {
        sqlx::query(
            "SELECT run_id, seq, metric_key, metric_value, unit FROM spl_metrics WHERE run_id = ? ORDER BY seq",
        )
        .bind(run_id)
        .fetch_all(&self.pool)
        .await?
        .iter()
        .map(|row| -> Result<MetricRecord> {
            Ok(MetricRecord {
                run_id: row.try_get("run_id")?,
                seq: row.try_get("seq")?,
                key: row.try_get("metric_key")?,
                value: row.try_get("metric_value")?,
                unit: row.try_get("unit")?,
            })
        })
        .collect()
    }
}

pub(crate) fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn run_lifecycle_round_trips() -> Result<()> {
        let store = crate::connect("sqlite::memory:").await?;
        store.migrate().await?; // second call must be a no-op

        let input = serde_json::json!({ "quality": 0.4 });
        store
            .create_run(&NewRun {
                id: "r1".into(),
                op_id: "demo.echo".into(),
                input: input.clone(),
                host_name: None,
                git_sha: None,
                env: None,
                created_at_ms: 1,
            })
            .await?;
        store.start_run("r1", 2).await?;
        let output = serde_json::json!({ "ok": true });
        store
            .finish_run("r1", RunStatus::Succeeded, Some(&output), None, 3)
            .await?;
        store
            .add_metrics(&[MetricRecord {
                run_id: "r1".into(),
                seq: 0,
                key: "psnr".into(),
                value: 31.5,
                unit: Some("dB".into()),
            }])
            .await?;

        let run = store.get_run("r1").await?.expect("run exists");
        assert_eq!(run.status, RunStatus::Succeeded);
        assert_eq!(run.input, input);
        assert_eq!(run.output, Some(output));
        assert_eq!(store.list_runs(Some("demo.echo"), 10).await?.len(), 1);
        assert_eq!(store.list_runs(Some("other.op"), 10).await?.len(), 0);
        assert_eq!(store.list_metrics("r1").await?[0].value, 31.5);
        Ok(())
    }

    fn new_run(id: &str) -> NewRun {
        NewRun {
            id: id.into(),
            op_id: "demo.echo".into(),
            input: serde_json::json!({}),
            host_name: None,
            git_sha: None,
            env: None,
            created_at_ms: 1,
        }
    }

    #[tokio::test]
    async fn unfinished_runs_fail_and_finished_ones_stay() -> Result<()> {
        let store = crate::connect("sqlite::memory:").await?;
        for id in ["queued", "running", "done"] {
            store.create_run(&new_run(id)).await?;
        }
        store.start_run("running", 2).await?;
        store.start_run("done", 2).await?;
        store
            .finish_run("done", RunStatus::Succeeded, None, None, 3)
            .await?;

        assert_eq!(store.fail_unfinished_runs("host restarted", 9).await?, 2);
        for id in ["queued", "running"] {
            let run = store.get_run(id).await?.expect("run exists");
            assert_eq!(run.status, RunStatus::Failed);
            assert_eq!(run.error_message.as_deref(), Some("host restarted"));
            assert_eq!(run.finished_at_ms, Some(9));
        }
        let done = store.get_run("done").await?.expect("run exists");
        assert_eq!(done.status, RunStatus::Succeeded);
        assert_eq!(done.finished_at_ms, Some(3));
        assert_eq!(store.fail_unfinished_runs("again", 10).await?, 0);
        Ok(())
    }

    #[tokio::test]
    async fn run_artifact_links_round_trip() -> Result<()> {
        let store = crate::connect("sqlite::memory:").await?;
        store.create_run(&new_run("r1")).await?;
        for id in ["a1", "a2"] {
            store
                .put_artifact(&ArtifactRecord {
                    id: id.into(),
                    kind: "blob".into(),
                    media_type: "text/plain".into(),
                    size_bytes: 1,
                    uri: format!("file:///{id}"),
                    meta: Some(serde_json::json!({ "filename": "a.txt" })),
                    created_at_ms: 1,
                })
                .await?;
        }
        store.link_artifact("r1", "a2", "output", "out-1").await?;
        store.link_artifact("r1", "a1", "output", "out-0").await?;
        store.link_artifact("r1", "a1", "input", "raw").await?;

        let links = store.list_run_artifacts("r1").await?;
        let slots: Vec<_> = links
            .iter()
            .map(|l| {
                (
                    l.direction.as_str(),
                    l.slot_name.as_str(),
                    l.artifact_id.as_str(),
                )
            })
            .collect();
        assert_eq!(
            slots,
            [
                ("input", "raw", "a1"),
                ("output", "out-0", "a1"),
                ("output", "out-1", "a2")
            ]
        );
        assert!(store.list_run_artifacts("other").await?.is_empty());
        let meta = store.get_artifact("a1").await?.and_then(|a| a.meta);
        assert_eq!(meta, Some(serde_json::json!({ "filename": "a.txt" })));
        Ok(())
    }
}
