use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use spl_core::{
    ArtifactKind, ArtifactRef, ArtifactStore, CancellationToken, Context, DataDir, DynOp,
    EventSink, Registry, Resources, Result, RunCx, RunEvent, anyhow, bail,
};
use spl_data::LocalStore;
use spl_db::{MetricRecord, NewRun, RunRecord, RunStatus, Store};
use spl_gpu::GpuReport;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinHandle;

use crate::Config;
use crate::api::RunProgress;
use crate::recording::{RecordingStore, artifact_record};

/// Log lines a live run keeps for `GET /v1/runs/{id}`.
const LOG_LINES: usize = 50;

pub struct Host {
    registry: Registry,
    store: Arc<dyn Store>,
    artifacts: Arc<dyn ArtifactStore>,
    resources: Resources,
    /// Captured once at startup and stored with every run for reproducibility.
    env: serde_json::Value,
    host_name: Option<String>,
    /// Runs this process has queued or is running, by run id.
    live: Mutex<HashMap<String, Arc<LiveEntry>>>,
    /// One permit per op allowed to execute at once. Tokio's semaphore is
    /// fair, so queued runs start in the order they were submitted.
    slots: Arc<Semaphore>,
    /// Lets `&self` methods hand a run to a task that outlives the call.
    this: Weak<Host>,
}

/// What `GET /v1/runs/{id}` shows of a run that is still going.
#[derive(Debug, Clone)]
pub struct LiveRun {
    pub progress: Option<RunProgress>,
    /// The last 50 lines, oldest first.
    pub logs: Vec<String>,
    /// Reported so far; saved to the database when the run ends.
    pub metrics: Vec<MetricRecord>,
}

#[derive(Default)]
struct LiveEntry {
    cancel: CancellationToken,
    state: Mutex<LiveState>,
}

#[derive(Default)]
struct LiveState {
    progress: Option<RunProgress>,
    logs: VecDeque<String>,
    metrics: Vec<MetricRecord>,
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

        Ok(Arc::new_cyclic(|this| Self {
            registry,
            store,
            artifacts,
            resources,
            env,
            host_name: std::env::var("HOSTNAME").ok(),
            live: Mutex::default(),
            slots: Arc::new(Semaphore::new(config.max_concurrent_runs.max(1))),
            this: this.clone(),
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
    /// failed run, not an `Err`; `Err` means the host itself broke. The run
    /// executes in a task of its own, so it still finishes and is recorded
    /// when the caller stops waiting (an HTTP client that hangs up).
    pub async fn run(&self, op_id: &str, input: serde_json::Value) -> Result<RunRecord> {
        let (_, task) = self.submit(op_id, input).await?;
        task.await.map_err(|e| anyhow!("run task died: {e}"))?
    }

    /// Starts one op and returns at once with its record: `running` when a
    /// slot was free, `queued` while `max_concurrent_runs` others execute.
    pub async fn start(&self, op_id: &str, input: serde_json::Value) -> Result<RunRecord> {
        let (record, _detached) = self.submit(op_id, input).await?;
        Ok(record)
    }

    /// Asks a queued or running run of this process to stop. A queued run
    /// ends as `cancelled` without starting; a running op sees
    /// `cx.is_cancelled()` and its run ends as `cancelled` once it returns.
    /// False when this process is not running `run_id`.
    pub fn cancel(&self, run_id: &str) -> bool {
        match lock(&self.live).get(run_id) {
            Some(entry) => {
                entry.cancel.cancel();
                true
            }
            None => false,
        }
    }

    /// Progress, recent logs and metrics of a run this process has queued or
    /// is running.
    pub fn live(&self, run_id: &str) -> Option<LiveRun> {
        let entry = lock(&self.live).get(run_id).cloned()?;
        let state = lock(&entry.state);
        Some(LiveRun {
            progress: state.progress.clone(),
            logs: state.logs.iter().cloned().collect(),
            metrics: state.metrics.clone(),
        })
    }

    /// Marks runs that an earlier process left `queued` or `running` as
    /// failed, since nothing will ever finish them. Call it once when a
    /// server starts, before it accepts runs, and never from a short-lived
    /// command (`spl run`, `spl put`): it would fail the runs of a server
    /// that shares the database.
    pub async fn fail_interrupted_runs(&self) -> Result<u64> {
        self.store
            .fail_unfinished_runs("host restarted", now_ms())
            .await
    }

    /// Stores bytes that come from outside any run (uploads, `spl put`) and
    /// records them, with `meta` such as `{"filename": "DSC0009.ARW"}`.
    /// Artifacts are content-addressed: the first record of the bytes wins.
    pub async fn put_artifact(
        &self,
        bytes: Vec<u8>,
        kind: ArtifactKind,
        media_type: &str,
        meta: Option<serde_json::Value>,
    ) -> Result<ArtifactRef> {
        let artifact = self.artifacts.put(bytes, kind, media_type).await?;
        let record = artifact_record(&artifact, self.artifacts.uri(&artifact.id), meta);
        self.store.put_artifact(&record).await?;
        Ok(artifact)
    }

    /// Records a new run and hands it to a task of its own. The run takes a
    /// free slot right away when there is one, so the returned record already
    /// says `running`.
    async fn submit(
        &self,
        op_id: &str,
        input: serde_json::Value,
    ) -> Result<(RunRecord, JoinHandle<Result<RunRecord>>)> {
        let Some(op) = self.registry.get(op_id) else {
            bail!("unknown op `{op_id}`");
        };
        let host = self.this.upgrade().context("host is shutting down")?;
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

        let entry = Arc::new(LiveEntry::default());
        lock(&self.live).insert(run_id.clone(), entry.clone());
        let guard = LiveGuard {
            host: host.clone(),
            run_id: run_id.clone(),
        };
        let permit = self.slots.clone().try_acquire_owned().ok();
        if permit.is_some() {
            self.store.start_run(&run_id, now_ms()).await?;
        }
        let record = self.record(&run_id).await?;

        let run = Execution {
            run_id,
            op_id: op_id.to_owned(),
            op,
            input,
            entry,
        };
        let task = tokio::spawn(async move {
            let run_id = run.run_id.clone();
            let result = host.execute(run, permit, guard).await;
            if let Err(e) = &result {
                tracing::error!(run = %run_id, "could not record the run: {e:#}");
            }
            result
        });
        Ok((record, task))
    }

    async fn execute(
        &self,
        run: Execution,
        permit: Option<OwnedSemaphorePermit>,
        _guard: LiveGuard,
    ) -> Result<RunRecord> {
        let Execution {
            run_id,
            op_id,
            op,
            input,
            entry,
        } = run;
        let permit = match permit {
            Some(permit) => permit,
            None => tokio::select! {
                biased;
                () = entry.cancel.cancelled() => {
                    self.store
                        .finish_run(&run_id, RunStatus::Cancelled, None, Some("cancelled while queued"), now_ms())
                        .await?;
                    tracing::info!(run = %run_id, op = %op_id, "run cancelled while queued");
                    return self.record(&run_id).await;
                }
                permit = self.slots.clone().acquire_owned() => {
                    let permit = permit.context("run queue closed")?;
                    self.store.start_run(&run_id, now_ms()).await?;
                    permit
                }
            },
        };
        tracing::info!(run = %run_id, op = %op_id, "run started");
        let started = Instant::now();

        let (events, rx) = EventSink::channel();
        let consumer = tokio::spawn(consume(rx, entry.clone(), run_id.clone(), op_id.clone()));
        let artifacts: Arc<dyn ArtifactStore> = Arc::new(RecordingStore::new(
            self.artifacts.clone(),
            self.store.clone(),
            run_id.clone(),
        ));
        let cx = RunCx::with_cancel(
            run_id.clone(),
            artifacts,
            self.resources.clone(),
            events,
            entry.cancel.clone(),
        );
        // A task of its own turns a panicking op into a failed run instead
        // of a run left `running` forever.
        let outcome = tokio::spawn(async move { op.run_json(&cx, input).await }).await;
        drop(permit);
        // The op task owned `cx`, the only event sender, so the channel is
        // closed and the consumer has seen every event.
        if let Err(e) = consumer.await {
            tracing::error!(run = %run_id, "event consumer died: {e}");
        }

        let result = match outcome {
            Ok(Ok(output)) => Ok(output),
            Ok(Err(e)) => Err(format!("{e:#}")),
            Err(e) => Err(panic_message(e)),
        };
        let (status, output, error) = match (result, entry.cancel.is_cancelled()) {
            (Ok(output), false) => (RunStatus::Succeeded, Some(output), None),
            (Ok(output), true) => (RunStatus::Cancelled, Some(output), Some("cancelled".into())),
            (Err(message), true) => (RunStatus::Cancelled, None, Some(message)),
            (Err(message), false) => (RunStatus::Failed, None, Some(message)),
        };

        let metrics = lock(&entry.state).metrics.clone();
        if let Err(e) = self.store.add_metrics(&metrics).await {
            tracing::error!(run = %run_id, "could not save metrics: {e:#}");
        }
        self.store
            .finish_run(&run_id, status, output.as_ref(), error.as_deref(), now_ms())
            .await?;
        tracing::info!(
            run = %run_id,
            op = %op_id,
            status = status.as_str(),
            seconds = started.elapsed().as_secs_f64(),
            error = error.as_deref().unwrap_or(""),
            "run finished"
        );
        self.record(&run_id).await
    }

    async fn record(&self, run_id: &str) -> Result<RunRecord> {
        self.store
            .get_run(run_id)
            .await?
            .with_context(|| format!("run {run_id} vanished"))
    }
}

/// Everything a run's task needs besides the host.
struct Execution {
    run_id: String,
    op_id: String,
    op: Arc<dyn DynOp>,
    input: serde_json::Value,
    entry: Arc<LiveEntry>,
}

/// Forgets a run's live state when its task ends, however it ends.
struct LiveGuard {
    host: Arc<Host>,
    run_id: String,
}

impl Drop for LiveGuard {
    fn drop(&mut self) {
        lock(&self.host.live).remove(&self.run_id);
    }
}

/// Keeps a run's latest progress, recent logs and metrics while it runs.
async fn consume(
    mut events: UnboundedReceiver<RunEvent>,
    entry: Arc<LiveEntry>,
    run_id: String,
    op_id: String,
) {
    while let Some(event) = events.recv().await {
        match event {
            RunEvent::Progress { fraction, message } => {
                let fraction = if fraction.is_finite() {
                    fraction.clamp(0.0, 1.0)
                } else {
                    0.0
                };
                lock(&entry.state).progress = Some(RunProgress { fraction, message });
            }
            RunEvent::Log { message } => {
                tracing::info!(target: "spl::op", run = %run_id, op = %op_id, "{message}");
                let mut state = lock(&entry.state);
                if state.logs.len() == LOG_LINES {
                    state.logs.pop_front();
                }
                state.logs.push_back(message);
            }
            RunEvent::Metric { key, value, unit } => {
                // SQL stores NaN as NULL, which would make the whole batch fail.
                if !value.is_finite() {
                    tracing::warn!(run = %run_id, op = %op_id, "dropped metric {key} = {value}");
                    continue;
                }
                let mut state = lock(&entry.state);
                let seq = state.metrics.len() as i64;
                state.metrics.push(MetricRecord {
                    run_id: run_id.clone(),
                    seq,
                    key,
                    value,
                    unit: Some(unit).filter(|u| !u.is_empty()),
                });
            }
        }
    }
}

fn panic_message(error: tokio::task::JoinError) -> String {
    if !error.is_panic() {
        return "op task was aborted".into();
    }
    let payload = error.into_panic();
    let message = payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_owned())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "no message".into());
    format!("op panicked: {message}")
}

/// Every critical section here is a few assignments, so a poisoned lock
/// still holds consistent data.
pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

pub(crate) fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or_default()
}
