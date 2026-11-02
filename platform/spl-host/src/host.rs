use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use spl_core::{
    ArtifactKind, ArtifactRef, ArtifactStore, CancellationToken, Context, DataDir, DynOp,
    EventSink, Registry, Resources, Result, RunCx, RunEvent, anyhow, bail,
};
use spl_data::LocalStore;
use spl_db::{Announcement, MetricRecord, NewRun, RunRecord, RunStatus, Store};
use spl_gpu::GpuReport;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::task::JoinHandle;

use crate::Config;
use crate::api::RunProgress;
use crate::queue::{Admission, Pending, RunQueue, WaitReason};
use crate::recording::{RecordingStore, artifact_record};

/// Log lines a live run keeps for `GET /v1/runs/{id}`.
const LOG_LINES: usize = 50;
/// Longest slot name an announcement may use (the schema's column width).
const SLOT_MAX: usize = 200;
/// Largest `info` an announcement keeps, serialized; it is polled often.
const INFO_MAX_BYTES: usize = 16 * 1024;
/// Staged files untouched this long belong to a process that died.
const STALE_STAGING: Duration = Duration::from_secs(60 * 60);

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
    /// Run slots and the GPU lock; queued runs start in submission order.
    queue: RunQueue,
    /// Where uploads land before they move into the artifact store.
    staging: PathBuf,
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
    /// While queued: what the run waits for.
    pub waiting_for: Option<WaitReason>,
}

/// A file in the host's staging directory, on the artifact store's
/// filesystem, so storing it is a rename. Deleted on drop unless it was
/// stored.
pub struct StagedFile {
    path: PathBuf,
}

impl StagedFile {
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for StagedFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
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
    waiting_for: Option<WaitReason>,
    /// The current announcement of each slot.
    slots: HashMap<String, Announced>,
    /// Snapshots a later announcement of their slot replaced, oldest first.
    superseded: Vec<String>,
    /// Artifacts ever announced to be kept (not as snapshots).
    kept: HashSet<String>,
}

struct Announced {
    revision: u32,
    artifact_id: String,
    snapshot: bool,
}

impl Host {
    pub async fn open(config: &Config, registry: Registry) -> Result<Arc<Self>> {
        std::fs::create_dir_all(&config.data_dir)?;
        let store = spl_db::connect(&config.database_url).await?;
        let artifacts: Arc<dyn ArtifactStore> = Arc::new(LocalStore::new(config.artifact_dir()));
        let staging = config.staging_dir();
        std::fs::create_dir_all(&staging)
            .with_context(|| format!("create {}", staging.display()))?;
        remove_stale_files(&staging);

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
            queue: RunQueue::new(config.max_concurrent_runs),
            staging,
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

    /// Starts one op and returns at once with its record: `running` when it
    /// could start, `queued` while it waits for a slot (`max_concurrent_runs`
    /// others execute) or for the GPU (an exclusive op holds it).
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
            waiting_for: state.waiting_for,
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
        self.record_artifact(&artifact, meta).await?;
        Ok(artifact)
    }

    /// [`put_artifact`](Self::put_artifact) for a file, usually a
    /// [`StagedFile`]: it moves into the store without being read into
    /// memory, so uploads of phone videos cost no RAM.
    pub async fn put_artifact_file(
        &self,
        path: &Path,
        kind: ArtifactKind,
        media_type: &str,
        meta: Option<serde_json::Value>,
    ) -> Result<ArtifactRef> {
        let artifact = self.artifacts.put_file(path, kind, media_type).await?;
        self.record_artifact(&artifact, meta).await?;
        Ok(artifact)
    }

    /// A fresh, empty path to write an upload to before
    /// [`put_artifact_file`](Self::put_artifact_file). The file is removed
    /// when the guard drops, unless it was stored.
    pub fn staging_file(&self) -> StagedFile {
        StagedFile {
            path: self
                .staging
                .join(format!("{}.part", uuid::Uuid::now_v7().simple())),
        }
    }

    async fn record_artifact(
        &self,
        artifact: &ArtifactRef,
        meta: Option<serde_json::Value>,
    ) -> Result<()> {
        let record = artifact_record(artifact, self.artifacts.uri(&artifact.id), meta);
        self.store.put_artifact(&record).await
    }

    /// Records a new run and hands it to a task of its own. The run is
    /// admitted right away when it can be, so the returned record already
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
        let admission = self.queue.try_admit(op.gpu());
        match &admission {
            Ok(_) => self.store.start_run(&run_id, now_ms()).await?,
            Err(pending) => lock(&entry.state).waiting_for = Some(pending.waiting_for()),
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
            let result = host.execute(run, admission, guard).await;
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
        admission: Result<Admission, Pending>,
        _guard: LiveGuard,
    ) -> Result<RunRecord> {
        let Execution {
            run_id,
            op_id,
            op,
            input,
            entry,
        } = run;
        let admission = match admission {
            Ok(admission) => admission,
            Err(pending) => {
                let note = |reason: WaitReason| lock(&entry.state).waiting_for = Some(reason);
                tokio::select! {
                    biased;
                    () = entry.cancel.cancelled() => {
                        self.store
                            .finish_run(&run_id, RunStatus::Cancelled, None, Some("cancelled while queued"), now_ms())
                            .await?;
                        tracing::info!(run = %run_id, op = %op_id, "run cancelled while queued");
                        return self.record(&run_id).await;
                    }
                    admission = self.queue.admit(pending, note) => {
                        let admission = admission?;
                        lock(&entry.state).waiting_for = None;
                        self.store.start_run(&run_id, now_ms()).await?;
                        admission
                    }
                }
            }
        };
        tracing::info!(run = %run_id, op = %op_id, gpu = ?op.gpu(), "run started");
        let started = Instant::now();

        let (events, rx) = EventSink::channel();
        let consumer = tokio::spawn(consume(
            rx,
            entry.clone(),
            self.store.clone(),
            run_id.clone(),
            op_id.clone(),
        ));
        let recording = Arc::new(RecordingStore::new(
            self.artifacts.clone(),
            self.store.clone(),
            run_id.clone(),
        ));
        let cx = RunCx::with_cancel(
            run_id.clone(),
            recording.clone(),
            self.resources.clone(),
            events,
            entry.cancel.clone(),
        );
        // A task of its own turns a panicking op into a failed run instead
        // of a run left `running` forever.
        let outcome = tokio::spawn(async move { op.run_json(&cx, input).await }).await;
        drop(admission);
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
        self.prune_snapshots(&run_id, &entry, &recording.created(), output.as_ref())
            .await;
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

    /// Deletes the snapshots a later announcement of their slot replaced:
    /// only bytes this run created, that its output does not name, that no
    /// slot shows now, and that nothing else references. A failure only
    /// leaves a file behind.
    async fn prune_snapshots(
        &self,
        run_id: &str,
        entry: &LiveEntry,
        created: &HashSet<String>,
        output: Option<&serde_json::Value>,
    ) {
        let (superseded, mut keep): (Vec<String>, HashSet<String>) = {
            let state = lock(&entry.state);
            let current = state.slots.values().map(|a| a.artifact_id.clone());
            (
                state.superseded.clone(),
                current.chain(state.kept.iter().cloned()).collect(),
            )
        };
        if superseded.is_empty() {
            return;
        }
        let output = output.map(|o| o.to_string()).unwrap_or_default();
        let (mut count, mut bytes) = (0u32, 0i64);
        for id in superseded {
            // Inserting also skips an id superseded twice.
            if !keep.insert(id.clone()) || !created.contains(&id) || output.contains(&id) {
                continue;
            }
            let size = match self.store.get_artifact(&id).await {
                Ok(record) => record.map_or(0, |r| r.size_bytes),
                Err(_) => 0,
            };
            match self.store.forget_artifact(run_id, &id).await {
                Ok(true) => match self.artifacts.delete(&id).await {
                    Ok(()) => {
                        count += 1;
                        bytes += size;
                    }
                    Err(e) => {
                        tracing::warn!(run = %run_id, "could not delete snapshot {id}: {e:#}")
                    }
                },
                Ok(false) => {}
                Err(e) => tracing::warn!(run = %run_id, "could not forget snapshot {id}: {e:#}"),
            }
        }
        if count > 0 {
            tracing::info!(
                run = %run_id,
                snapshots = count,
                mib = bytes as f64 / (1024.0 * 1024.0),
                "deleted superseded snapshots"
            );
        }
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

/// Keeps a run's latest progress, recent logs and metrics while it runs,
/// and records its announcements as they come.
async fn consume(
    mut events: UnboundedReceiver<RunEvent>,
    entry: Arc<LiveEntry>,
    store: Arc<dyn Store>,
    run_id: String,
    op_id: String,
) {
    while let Some(event) = events.recv().await {
        match event {
            RunEvent::Progress {
                fraction,
                message,
                data,
            } => {
                let fraction = if fraction.is_finite() {
                    fraction.clamp(0.0, 1.0)
                } else {
                    0.0
                };
                lock(&entry.state).progress = Some(RunProgress {
                    fraction,
                    message,
                    data,
                });
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
            RunEvent::Announce {
                slot,
                artifact,
                snapshot,
                info,
            } => {
                let slot = slot.trim().to_owned();
                if slot.is_empty() || slot.len() > SLOT_MAX {
                    tracing::warn!(run = %run_id, op = %op_id, "dropped an announcement with slot `{slot}`");
                    continue;
                }
                let info = info.filter(|i| {
                    let fits = i.to_string().len() <= INFO_MAX_BYTES;
                    if !fits {
                        tracing::warn!(run = %run_id, op = %op_id, "dropped the info of `{slot}`: over {INFO_MAX_BYTES} bytes");
                    }
                    fits
                });
                let revision = lock(&entry.state)
                    .slots
                    .get(&slot)
                    .map_or(1, |a| a.revision.saturating_add(1));
                let announcement = Announcement {
                    slot: slot.clone(),
                    artifact,
                    revision,
                    info,
                    announced_at_ms: now_ms(),
                };
                if let Err(e) = store.announce(&run_id, &announcement).await {
                    tracing::warn!(run = %run_id, op = %op_id, "could not announce `{slot}`: {e:#}");
                    continue;
                }
                let current = Announced {
                    revision,
                    artifact_id: announcement.artifact.id,
                    snapshot,
                };
                let mut state = lock(&entry.state);
                if !snapshot {
                    state.kept.insert(current.artifact_id.clone());
                }
                if let Some(previous) = state.slots.insert(slot, current)
                    && previous.snapshot
                {
                    state.superseded.push(previous.artifact_id);
                }
            }
        }
    }
}

/// Removes files a crashed process left in the staging directory. A live
/// upload writes to its file all the time, so it never looks stale.
fn remove_stale_files(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let now = SystemTime::now();
    for entry in entries.flatten() {
        let stale = entry
            .metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|modified| now.duration_since(modified).ok())
            .is_some_and(|age| age > STALE_STAGING);
        if stale {
            let _ = std::fs::remove_file(entry.path());
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
