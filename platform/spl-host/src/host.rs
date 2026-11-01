use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use spl_core::{
    ArtifactStore, DataDir, EventSink, Registry, Resources, Result, RunCx, RunEvent, bail,
};
use spl_data::LocalStore;
use spl_db::{MetricRecord, NewRun, RunRecord, RunStatus, Store};
use spl_gpu::GpuReport;

use crate::Config;

pub struct Host {
    registry: Registry,
    store: Arc<dyn Store>,
    artifacts: Arc<dyn ArtifactStore>,
    resources: Resources,
    /// Captured once at startup and stored with every run for reproducibility.
    env: serde_json::Value,
    host_name: Option<String>,
}

impl Host {
    pub async fn open(config: &Config, registry: Registry) -> Result<Arc<Self>> {
        std::fs::create_dir_all(&config.data_dir)?;
        let store = spl_db::connect(&config.database_url).await?;
        let artifacts: Arc<dyn ArtifactStore> = Arc::new(LocalStore::new(config.artifact_dir()));

        let gpu = Arc::new(spl_gpu::probe());
        let env = serde_json::json!({
            "spl_version": env!("CARGO_PKG_VERSION"),
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "gpu": &*gpu,
        });
        let mut resources = Resources::default();
        resources.insert::<GpuReport>(gpu);
        resources.insert(Arc::new(DataDir(config.data_dir.clone())));

        Ok(Arc::new(Self {
            registry,
            store,
            artifacts,
            resources,
            env,
            host_name: std::env::var("HOSTNAME").ok(),
        }))
    }

    pub fn registry(&self) -> &Registry {
        &self.registry
    }

    pub fn store(&self) -> &Arc<dyn Store> {
        &self.store
    }

    pub fn artifacts(&self) -> &Arc<dyn ArtifactStore> {
        &self.artifacts
    }

    pub fn env(&self) -> &serde_json::Value {
        &self.env
    }

    /// Runs one op to completion and returns its record. An op error is a
    /// failed run, not an `Err`; `Err` means the host itself broke.
    pub async fn run(&self, op_id: &str, input: serde_json::Value) -> Result<RunRecord> {
        let Some(op) = self.registry.get(op_id) else {
            bail!("unknown op `{op_id}`");
        };
        let run_id = uuid::Uuid::now_v7().to_string();
        self.store
            .create_run(&NewRun {
                id: run_id.clone(),
                op_id: op_id.to_owned(),
                input: input.clone(),
                host_name: self.host_name.clone(),
                git_sha: option_env!("SPL_GIT_SHA").map(str::to_owned),
                env: Some(self.env.clone()),
                created_at_ms: now_ms(),
            })
            .await?;
        self.store.start_run(&run_id, now_ms()).await?;

        let (events, mut rx) = EventSink::channel();
        let cx = RunCx::new(
            run_id.clone(),
            self.artifacts.clone(),
            self.resources.clone(),
            events,
        );
        let result = op.run_json(&cx, input).await;
        drop(cx);

        let mut metrics = Vec::new();
        while let Ok(event) = rx.try_recv() {
            match event {
                RunEvent::Metric { key, value, unit } => metrics.push(MetricRecord {
                    run_id: run_id.clone(),
                    seq: metrics.len() as i64,
                    key,
                    value,
                    unit: Some(unit).filter(|u| !u.is_empty()),
                }),
                RunEvent::Progress { .. } | RunEvent::Log { .. } => {
                    // Streamed live once the WebSocket channel lands.
                }
            }
        }
        self.store.add_metrics(&metrics).await?;

        match &result {
            Ok(output) => {
                self.store
                    .finish_run(&run_id, RunStatus::Succeeded, Some(output), None, now_ms())
                    .await?
            }
            Err(e) => {
                let message = format!("{e:#}");
                self.store
                    .finish_run(&run_id, RunStatus::Failed, None, Some(&message), now_ms())
                    .await?
            }
        }
        match self.store.get_run(&run_id).await? {
            Some(run) => Ok(run),
            None => bail!("run {run_id} vanished"),
        }
    }
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or_default()
}
