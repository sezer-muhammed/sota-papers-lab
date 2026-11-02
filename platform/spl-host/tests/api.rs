//! The HTTP API end to end, through `tower::ServiceExt::oneshot`: async runs,
//! progress, cancellation, the run queue and the GPU lock, announcements,
//! uploads, downloads and restarts.

use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use axum::Router;
use axum::body::{Body, Bytes};
use axum::http::{HeaderMap, Request, StatusCode, header};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use spl_core::{ArtifactKind, ArtifactRef, GpuUse, Op, Registry, Result, RunCx, bail, media};
use spl_db::{NewRun, RunStatus};
use spl_host::{Config, Host, router};
use tempfile::TempDir;
use tower::ServiceExt;

/// Counts the test ops running now, to catch an exclusive op that shares
/// the GPU.
#[derive(Default)]
struct Tracker {
    running: AtomicUsize,
    peak: AtomicUsize,
    gpu: AtomicUsize,
    exclusive: AtomicUsize,
    /// Times an exclusive op ran next to another GPU op.
    overlaps: AtomicUsize,
}

/// One registered variant of the test op.
trait Flavor: Send + Sync + 'static {
    const ID: &'static str;
    const GPU: GpuUse;
}

struct SharedGpu;
struct ExclusiveGpu;
struct CpuOnly;

impl Flavor for SharedGpu {
    const ID: &'static str = "test.work";
    const GPU: GpuUse = GpuUse::Shared;
}

impl Flavor for ExclusiveGpu {
    const ID: &'static str = "test.exclusive";
    const GPU: GpuUse = GpuUse::Exclusive;
}

impl Flavor for CpuOnly {
    const ID: &'static str = "test.cpu";
    const GPU: GpuUse = GpuUse::CpuOnly;
}

/// Works in steps, reporting progress, a log line and a metric at each.
struct Work<F> {
    tracker: Arc<Tracker>,
    flavor: PhantomData<F>,
}

impl<F> Work<F> {
    fn new(tracker: &Arc<Tracker>) -> Self {
        Self {
            tracker: tracker.clone(),
            flavor: PhantomData,
        }
    }
}

#[derive(Deserialize, JsonSchema)]
struct WorkInput {
    #[serde(default)]
    steps: u32,
    #[serde(default)]
    step_ms: u64,
    /// Log this many lines before the steps.
    #[serde(default)]
    logs: u32,
    /// After the steps, wait until cancelled.
    #[serde(default)]
    hold: bool,
    #[serde(default)]
    fail: bool,
    #[serde(default)]
    panic: bool,
    /// Stored as text artifacts, in order.
    #[serde(default)]
    store: Vec<String>,
    /// Stored as splats and announced, in order, before the steps.
    #[serde(default)]
    announce: Vec<Shown>,
    /// Steps report progress with structured data.
    #[serde(default)]
    progress_data: bool,
}

#[derive(Deserialize, JsonSchema)]
struct Shown {
    slot: String,
    text: String,
    #[serde(default)]
    snapshot: bool,
    /// Name the artifact in the run's output.
    #[serde(default)]
    output: bool,
    #[serde(default)]
    info: Value,
    /// Announce an id that was never stored.
    #[serde(default)]
    unrecorded: bool,
}

impl<F: Flavor> Op for Work<F> {
    const ID: &'static str = F::ID;
    const SUMMARY: &'static str = "Sleeps in steps and reports as it goes";
    const GPU: GpuUse = F::GPU;

    type Input = WorkInput;
    type Output = Value;

    async fn run(&self, cx: &RunCx, input: WorkInput) -> Result<Value> {
        let t = &self.tracker;
        let now = t.running.fetch_add(1, Ordering::SeqCst) + 1;
        t.peak.fetch_max(now, Ordering::SeqCst);
        match F::GPU {
            GpuUse::Exclusive => {
                t.exclusive.fetch_add(1, Ordering::SeqCst);
                if t.gpu.fetch_add(1, Ordering::SeqCst) > 0 {
                    t.overlaps.fetch_add(1, Ordering::SeqCst);
                }
            }
            GpuUse::Shared => {
                t.gpu.fetch_add(1, Ordering::SeqCst);
                if t.exclusive.load(Ordering::SeqCst) > 0 {
                    t.overlaps.fetch_add(1, Ordering::SeqCst);
                }
            }
            GpuUse::CpuOnly => {}
        }
        let result = work(cx, input).await;
        if F::GPU != GpuUse::CpuOnly {
            t.gpu.fetch_sub(1, Ordering::SeqCst);
        }
        if F::GPU == GpuUse::Exclusive {
            t.exclusive.fetch_sub(1, Ordering::SeqCst);
        }
        t.running.fetch_sub(1, Ordering::SeqCst);
        result
    }
}

async fn work(cx: &RunCx, input: WorkInput) -> Result<Value> {
    let mut ids = Vec::new();
    for text in input.store {
        let artifact = cx
            .artifacts
            .put(text.into_bytes(), ArtifactKind::Blob, "text/plain")
            .await?;
        ids.push(artifact.id);
    }
    let mut named = Vec::new();
    for shown in input.announce {
        let artifact = if shown.unrecorded {
            ArtifactRef {
                id: "f".repeat(64),
                kind: ArtifactKind::Splat,
                media_type: media::SPZ.into(),
                size_bytes: 1,
            }
        } else {
            cx.artifacts
                .put(shown.text.into_bytes(), ArtifactKind::Splat, media::SPZ)
                .await?
        };
        if shown.snapshot {
            cx.announce_snapshot(shown.slot, &artifact, shown.info);
        } else {
            cx.announce(shown.slot, &artifact, shown.info);
        }
        if shown.output {
            named.push(artifact.id);
        }
    }
    for line in 0..input.logs {
        cx.log(format!("line {line}"));
    }
    cx.metric("steps", f64::from(input.steps), "");
    cx.metric("not_a_number", f64::NAN, "");
    for step in 0..input.steps {
        if cx.is_cancelled() {
            bail!("cancelled at step {step}");
        }
        let fraction = f64::from(step) / f64::from(input.steps);
        if input.progress_data {
            cx.progress_with(fraction, format!("step {step}"), json!({ "step": step }));
        } else {
            cx.progress(fraction, format!("step {step}"));
        }
        cx.log(format!("step {step}"));
        cx.metric("step_ms", input.step_ms as f64, "ms");
        tokio::time::sleep(Duration::from_millis(input.step_ms)).await;
    }
    if input.hold {
        cx.progress(0.5, "holding");
        cx.cancel.cancelled().await;
        bail!("stopped holding: cancelled");
    }
    if input.fail {
        bail!("asked to fail");
    }
    if input.panic {
        panic!("asked to panic");
    }
    Ok(json!({ "steps": input.steps, "artifacts": ids, "named": named }))
}

struct Lab {
    app: Router,
    host: Arc<Host>,
    tracker: Arc<Tracker>,
    dir: TempDir,
}

async fn lab(configure: impl FnOnce(&mut Config)) -> Lab {
    lab_in(tempfile::tempdir().unwrap(), configure).await
}

async fn lab_in(dir: TempDir, configure: impl FnOnce(&mut Config)) -> Lab {
    let mut config = Config {
        data_dir: dir.path().to_owned(),
        database_url: database_url(&dir),
        ..Config::default()
    };
    configure(&mut config);
    let tracker = Arc::new(Tracker::default());
    let mut registry = Registry::new();
    registry
        .op(Work::<SharedGpu>::new(&tracker))
        .op(Work::<ExclusiveGpu>::new(&tracker))
        .op(Work::<CpuOnly>::new(&tracker));
    let host = Host::open(&config, registry).await.unwrap();
    Lab {
        app: router(host.clone(), &config),
        host,
        tracker,
        dir,
    }
}

fn database_url(dir: &TempDir) -> String {
    format!("sqlite://{}", dir.path().join("spl.db").display())
}

impl Lab {
    async fn send(&self, request: Request<Body>) -> (StatusCode, HeaderMap, Bytes) {
        let response = self.app.clone().oneshot(request).await.unwrap();
        let status = response.status();
        let headers = response.headers().clone();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        (status, headers, body)
    }

    async fn json(&self, method: &str, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
        let request = Request::builder().method(method).uri(uri);
        let request = match body {
            Some(body) => request
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body.to_string())),
            None => request.body(Body::empty()),
        };
        let (status, _, bytes) = self.send(request.unwrap()).await;
        let value = if bytes.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&bytes).unwrap()
        };
        (status, value)
    }

    /// `POST /v1/runs` of `test.work` with `wait: false`.
    async fn start(&self, input: Value) -> Value {
        self.start_op("test.work", input).await
    }

    async fn start_op(&self, op: &str, input: Value) -> Value {
        let request = json!({ "op": op, "input": input, "wait": false });
        let (status, run) = self.json("POST", "/v1/runs", Some(request)).await;
        assert_eq!(status, StatusCode::ACCEPTED, "{run}");
        run
    }

    /// `POST /v1/runs`, waiting for the result.
    async fn run(&self, input: Value) -> Value {
        let request = json!({ "op": "test.work", "input": input });
        let (status, run) = self.json("POST", "/v1/runs", Some(request)).await;
        assert_eq!(status, StatusCode::OK, "{run}");
        run
    }

    async fn cancel(&self, id: &str) -> (StatusCode, Value) {
        self.json("POST", &format!("/v1/runs/{id}/cancel"), None)
            .await
    }

    async fn detail(&self, id: &str) -> Value {
        let (status, detail) = self.json("GET", &format!("/v1/runs/{id}"), None).await;
        assert_eq!(status, StatusCode::OK, "{detail}");
        detail
    }

    async fn wait_until(&self, id: &str, done: impl Fn(&Value) -> bool) -> Value {
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            let detail = self.detail(id).await;
            if done(&detail) {
                return detail;
            }
            assert!(Instant::now() < deadline, "timed out on run {id}: {detail}");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    async fn wait_finished(&self, id: &str) -> Value {
        self.wait_until(id, |d| !matches!(status(d), "queued" | "running"))
            .await
    }

    async fn get(&self, uri: &str, if_none_match: Option<&str>) -> (StatusCode, HeaderMap, Bytes) {
        let mut request = Request::get(uri);
        if let Some(tag) = if_none_match {
            request = request.header(header::IF_NONE_MATCH, tag);
        }
        self.send(request.body(Body::empty()).unwrap()).await
    }

    async fn get_with(
        &self,
        uri: &str,
        headers: &[(&str, &str)],
    ) -> (StatusCode, HeaderMap, Bytes) {
        let mut request = Request::get(uri);
        for &(name, value) in headers {
            request = request.header(name, value);
        }
        self.send(request.body(Body::empty()).unwrap()).await
    }

    /// Files left in the staging directory: none once uploads settle.
    fn staged(&self) -> usize {
        std::fs::read_dir(self.dir.path().join("staging"))
            .unwrap()
            .count()
    }
}

fn status(detail: &Value) -> &str {
    detail["run"]["status"].as_str().unwrap()
}

fn id(run: &Value) -> String {
    run["id"].as_str().unwrap().to_owned()
}

fn upload(
    bytes: impl Into<Body>,
    media_type: Option<&str>,
    filename: Option<&str>,
) -> Request<Body> {
    let mut request = Request::post("/v1/artifacts");
    if let Some(media_type) = media_type {
        request = request.header(header::CONTENT_TYPE, media_type);
    }
    if let Some(filename) = filename {
        request = request.header("x-spl-filename", filename);
    }
    request.body(bytes.into()).unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn async_runs_show_progress_then_save_their_metrics() {
    let lab = lab(|_| {}).await;
    let run = lab.start(json!({ "steps": 10, "step_ms": 30 })).await;
    assert_eq!(run["status"], "running");
    assert!(run["started_at_ms"].is_number());
    let id = id(&run);

    let live = lab
        .wait_until(&id, |d| d["logs"].as_array().is_some_and(|l| l.len() >= 2))
        .await;
    assert_eq!(status(&live), "running");
    let fraction = live["progress"]["fraction"].as_f64().unwrap();
    assert!((0.0..1.0).contains(&fraction), "{live}");
    assert!(
        live["progress"]["message"]
            .as_str()
            .unwrap()
            .starts_with("step ")
    );
    assert!(live["metrics"].as_array().unwrap().len() >= 2, "{live}");

    let done = lab.wait_finished(&id).await;
    assert_eq!(status(&done), "succeeded", "{done}");
    assert_eq!(done["run"]["output"]["steps"], 10);
    assert!(done["progress"].is_null());
    assert_eq!(done["logs"], json!([]));
    // One `steps` metric and one per step; the NaN one is dropped, not fatal.
    let metrics = done["metrics"].as_array().unwrap();
    assert_eq!(metrics.len(), 11, "{done}");
    assert_eq!(metrics[0]["key"], "steps");
    assert!(metrics[0]["unit"].is_null());
    assert_eq!(metrics[1]["unit"], "ms");
    let seqs: Vec<i64> = metrics.iter().map(|m| m["seq"].as_i64().unwrap()).collect();
    assert_eq!(seqs, (0..11).collect::<Vec<_>>());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn waiting_stays_the_default() {
    let lab = lab(|_| {}).await;
    let run = lab.run(json!({ "steps": 2 })).await;
    assert_eq!(run["status"], "succeeded");
    assert_eq!(run["output"]["steps"], 2);

    let request = json!({ "op": "test.work", "input": { "steps": 1 }, "wait": true });
    let (code, run) = lab.json("POST", "/v1/runs", Some(request)).await;
    assert_eq!(code, StatusCode::OK);
    assert_eq!(run["status"], "succeeded");

    let (code, _) = lab
        .json("POST", "/v1/runs", Some(json!({ "op": "no.such" })))
        .await;
    assert_eq!(code, StatusCode::NOT_FOUND);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn failed_and_panicking_ops_are_recorded_with_their_metrics() {
    let lab = lab(|_| {}).await;
    let failed = lab.run(json!({ "steps": 1, "fail": true })).await;
    assert_eq!(failed["status"], "failed");
    assert_eq!(failed["error_message"], "asked to fail");
    let detail = lab.detail(&id(&failed)).await;
    assert_eq!(detail["metrics"].as_array().unwrap().len(), 2, "{detail}");

    let panicked = lab.run(json!({ "panic": true })).await;
    assert_eq!(panicked["status"], "failed");
    assert_eq!(panicked["error_message"], "op panicked: asked to panic");
    assert!(panicked["finished_at_ms"].is_number());

    let after = lab.run(json!({ "steps": 1 })).await;
    assert_eq!(after["status"], "succeeded");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancel_stops_a_running_op_that_kept_its_last_50_log_lines() {
    let lab = lab(|_| {}).await;
    let run = lab.start(json!({ "logs": 120, "hold": true })).await;
    let id = id(&run);
    let live = lab
        .wait_until(&id, |d| d["progress"]["message"] == "holding")
        .await;
    let logs = live["logs"].as_array().unwrap();
    assert_eq!(logs.len(), 50);
    assert_eq!(logs[0], "line 70");
    assert_eq!(logs[49], "line 119");

    let (code, accepted) = lab.cancel(&id).await;
    assert_eq!(code, StatusCode::ACCEPTED, "{accepted}");
    assert_eq!(accepted["id"], id.as_str());
    let done = lab.wait_finished(&id).await;
    assert_eq!(status(&done), "cancelled", "{done}");
    assert_eq!(done["run"]["error_message"], "stopped holding: cancelled");
    assert!(done["run"]["finished_at_ms"].is_number());
    assert!(done["progress"].is_null());
    assert_eq!(done["logs"], json!([]));

    let (code, body) = lab.cancel(&id).await;
    assert_eq!(code, StatusCode::CONFLICT, "{body}");
    assert!(body["error"].as_str().unwrap().contains("cancelled"));
    let (code, _) = lab.cancel("no-such-run").await;
    assert_eq!(code, StatusCode::NOT_FOUND);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn runs_beyond_the_limit_queue_and_start_in_order() {
    let lab = lab(|c| c.max_concurrent_runs = 2).await;
    let mut ids = Vec::new();
    for i in 0..5 {
        let run = lab.start(json!({ "steps": 4, "step_ms": 40 })).await;
        if i < 2 {
            assert_eq!(run["status"], "running", "run {i}: {run}");
        } else {
            assert_eq!(run["status"], "queued", "run {i}: {run}");
            assert!(run["started_at_ms"].is_null());
        }
        ids.push(id(&run));
    }
    assert_eq!(status(&lab.detail(&ids[4]).await), "queued");

    let mut started = Vec::new();
    for id in &ids {
        let done = lab.wait_finished(id).await;
        assert_eq!(status(&done), "succeeded", "{done}");
        started.push(done["run"]["started_at_ms"].as_i64().unwrap());
    }
    assert_eq!(lab.tracker.peak.load(Ordering::SeqCst), 2);
    assert!(
        started[2] <= started[3] && started[3] <= started[4],
        "{started:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_queued_run_can_be_cancelled_before_it_starts() {
    let lab = lab(|c| c.max_concurrent_runs = 1).await;
    let holder = id(&lab.start(json!({ "hold": true })).await);
    let queued = lab.start(json!({ "steps": 1 })).await;
    assert_eq!(queued["status"], "queued");
    let queued = id(&queued);

    let (code, _) = lab.cancel(&queued).await;
    assert_eq!(code, StatusCode::ACCEPTED);
    let done = lab.wait_finished(&queued).await;
    assert_eq!(status(&done), "cancelled");
    assert_eq!(done["run"]["error_message"], "cancelled while queued");
    assert!(done["run"]["started_at_ms"].is_null());
    assert_eq!(status(&lab.detail(&holder).await), "running");

    let (code, _) = lab.cancel(&holder).await;
    assert_eq!(code, StatusCode::ACCEPTED);
    assert_eq!(status(&lab.wait_finished(&holder).await), "cancelled");
    let next = lab.start(json!({ "steps": 1 })).await;
    assert_eq!(
        next["status"], "running",
        "the freed slot is reused: {next}"
    );
    assert_eq!(status(&lab.wait_finished(&id(&next)).await), "succeeded");
    assert_eq!(lab.tracker.peak.load(Ordering::SeqCst), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn artifacts_an_op_stores_are_recorded_as_its_outputs() {
    let lab = lab(|_| {}).await;
    let run = lab
        .run(json!({ "store": ["first", "second", "first"] }))
        .await;
    assert_eq!(run["status"], "succeeded", "{run}");
    let ids: Vec<String> = serde_json::from_value(run["output"]["artifacts"].clone()).unwrap();
    assert_eq!(ids[0], ids[2], "content-addressed");

    let links = lab
        .host
        .store()
        .list_run_artifacts(&id(&run))
        .await
        .unwrap();
    let links: Vec<_> = links
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
        links,
        [
            ("output", "out-0", ids[0].as_str()),
            ("output", "out-1", ids[1].as_str())
        ]
    );
    for artifact in &ids[..2] {
        let record = lab
            .host
            .store()
            .get_artifact(artifact)
            .await
            .unwrap()
            .expect("recorded");
        assert_eq!(record.kind, "blob");
        assert_eq!(record.media_type, "text/plain");
        assert!(record.uri.contains(artifact.as_str()));
        let (code, headers, _) = lab.get(&format!("/v1/artifacts/{artifact}"), None).await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(headers[header::CONTENT_TYPE], "text/plain");
    }
}

#[tokio::test]
async fn uploads_record_their_name_and_downloads_cache_forever() {
    let lab = lab(|_| {}).await;
    let bytes = b"not really a raw file".to_vec();
    let request = upload(
        bytes.clone(),
        Some("image/x-sony-arw"),
        Some("%C3%87ay%20bah%C3%A7esi.ARW"),
    );
    let (code, headers, body) = lab.send(request).await;
    assert_eq!(code, StatusCode::CREATED);
    let artifact: Value = serde_json::from_slice(&body).unwrap();
    let id = artifact["id"].as_str().unwrap();
    assert_eq!(id.len(), 64);
    assert_eq!(artifact["kind"], "blob");
    assert_eq!(artifact["media_type"], "image/x-sony-arw");
    assert_eq!(artifact["size_bytes"], bytes.len());
    assert_eq!(
        headers[header::LOCATION],
        format!("/v1/artifacts/{id}").as_str()
    );
    let record = lab.host.store().get_artifact(id).await.unwrap().unwrap();
    assert_eq!(record.meta, Some(json!({ "filename": "Çay bahçesi.ARW" })));
    assert_eq!(record.size_bytes, bytes.len() as i64);

    let uri = format!("/v1/artifacts/{id}");
    let (code, headers, body) = lab.get(&uri, None).await;
    assert_eq!(code, StatusCode::OK);
    assert_eq!(body, bytes);
    assert_eq!(headers[header::CONTENT_TYPE], "image/x-sony-arw");
    assert_eq!(
        headers[header::CONTENT_LENGTH],
        bytes.len().to_string().as_str()
    );
    assert_eq!(headers[header::ETAG], format!("\"{id}\"").as_str());
    assert_eq!(
        headers[header::CACHE_CONTROL],
        "public, max-age=31536000, immutable"
    );
    assert_eq!(headers[header::X_CONTENT_TYPE_OPTIONS], "nosniff");
    assert!(headers.get(header::CONTENT_DISPOSITION).is_none());

    let downloads = [
        (
            "photo%20edit.jpg",
            "attachment; filename=\"photo edit.jpg\"",
        ),
        (
            "%C3%87ay.jpg",
            "attachment; filename=\"_ay.jpg\"; filename*=UTF-8''%C3%87ay.jpg",
        ),
        // No name given: the recorded one.
        (
            "",
            "attachment; filename=\"_ay bah_esi.ARW\"; filename*=UTF-8''%C3%87ay%20bah%C3%A7esi.ARW",
        ),
    ];
    for (name, disposition) in downloads {
        let (code, headers, body) = lab.get(&format!("{uri}?download={name}"), None).await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(body, bytes);
        assert_eq!(headers[header::CONTENT_DISPOSITION], disposition);
    }

    for tag in [
        format!("\"{id}\""),
        format!("W/\"x\", \"{id}\""),
        "*".into(),
    ] {
        let (code, headers, body) = lab.get(&uri, Some(&tag)).await;
        assert_eq!(code, StatusCode::NOT_MODIFIED, "{tag}");
        assert!(body.is_empty());
        assert_eq!(headers[header::ETAG], format!("\"{id}\"").as_str());
    }
    let (code, _, _) = lab.get(&uri, Some("\"other\"")).await;
    assert_eq!(code, StatusCode::OK);

    let missing = "0".repeat(64);
    for uri in [
        format!("/v1/artifacts/{missing}"),
        "/v1/artifacts/not-hex".into(),
    ] {
        let (code, _, body) = lab.get(&uri, None).await;
        assert_eq!(code, StatusCode::NOT_FOUND, "{uri}");
        let error: Value = serde_json::from_slice(&body).unwrap();
        assert!(error["error"].is_string());
    }
    let (code, _, _) = lab
        .get(&format!("/v1/artifacts/{missing}"), Some("*"))
        .await;
    assert_eq!(
        code,
        StatusCode::NOT_FOUND,
        "no 304 for what was never stored"
    );

    let (code, _, _) = lab.send(upload(Vec::new(), None, None)).await;
    assert_eq!(code, StatusCode::BAD_REQUEST);
    let (code, _, body) = lab.send(upload(b"plain".to_vec(), None, None)).await;
    assert_eq!(code, StatusCode::CREATED);
    let artifact: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(artifact["media_type"], "application/octet-stream");
    let id = artifact["id"].as_str().unwrap();
    let record = lab.host.store().get_artifact(id).await.unwrap().unwrap();
    assert!(record.meta.is_none());
}

#[tokio::test]
async fn uploads_over_the_limit_are_refused() {
    let lab = lab(|c| c.max_upload_mb = 1).await;
    let (code, _, body) = lab
        .send(upload(vec![7u8; 1024 * 1024 + 1], None, None))
        .await;
    assert_eq!(code, StatusCode::PAYLOAD_TOO_LARGE);
    let error: Value = serde_json::from_slice(&body).unwrap();
    assert!(
        error["error"].as_str().unwrap().contains("1 MiB"),
        "{error}"
    );
    let (code, _, _) = lab.send(upload(vec![7u8; 1024 * 1024], None, None)).await;
    assert_eq!(code, StatusCode::CREATED);
}

#[tokio::test]
async fn uploads_and_cancels_need_the_token() {
    let lab = lab(|c| c.auth_token = Some("s3cret".into())).await;
    let (code, _, _) = lab.send(upload(b"x".to_vec(), None, None)).await;
    assert_eq!(code, StatusCode::UNAUTHORIZED);
    let mut request = upload(b"x".to_vec(), None, None);
    request
        .headers_mut()
        .insert(header::AUTHORIZATION, "Bearer s3cret".parse().unwrap());
    let (code, _, _) = lab.send(request).await;
    assert_eq!(code, StatusCode::CREATED);
    let (code, _) = lab.cancel("anything").await;
    assert_eq!(code, StatusCode::UNAUTHORIZED);
    let (code, _) = lab.json("GET", "/v1/health", None).await;
    assert_eq!(code, StatusCode::OK);
}

#[tokio::test]
async fn a_restarted_server_fails_the_runs_a_previous_one_left() {
    let dir = tempfile::tempdir().unwrap();
    {
        let store = spl_db::connect(&database_url(&dir)).await.unwrap();
        for id in ["left-queued", "left-running", "finished"] {
            store
                .create_run(&NewRun {
                    id: id.into(),
                    op_id: "test.work".into(),
                    input: json!({}),
                    host_name: None,
                    git_sha: None,
                    env: None,
                    created_at_ms: 1,
                })
                .await
                .unwrap();
        }
        store.start_run("left-running", 2).await.unwrap();
        store.start_run("finished", 2).await.unwrap();
        store
            .finish_run("finished", RunStatus::Succeeded, Some(&json!({})), None, 3)
            .await
            .unwrap();
    }
    let lab = lab_in(dir, |_| {}).await;
    // Opening a host, as `spl run` and `spl put` do, leaves other runs alone.
    assert_eq!(status(&lab.detail("left-running").await), "running");
    let (code, body) = lab.cancel("left-running").await;
    assert_eq!(code, StatusCode::CONFLICT, "not this process's run: {body}");

    assert_eq!(lab.host.fail_interrupted_runs().await.unwrap(), 2);
    for id in ["left-queued", "left-running"] {
        let detail = lab.detail(id).await;
        assert_eq!(status(&detail), "failed");
        assert_eq!(detail["run"]["error_message"], "host restarted");
        assert!(detail["run"]["finished_at_ms"].is_number());
    }
    let finished = lab.detail("finished").await;
    assert_eq!(status(&finished), "succeeded");
    assert_eq!(finished["run"]["finished_at_ms"], 3);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_spa_html_is_revalidated_and_its_assets_are_not() {
    let web = tempfile::tempdir().unwrap();
    std::fs::write(web.path().join("index.html"), "<!doctype html><p>app</p>").unwrap();
    std::fs::create_dir(web.path().join("assets")).unwrap();
    std::fs::write(web.path().join("assets/app-1a2b.js"), "console.log(1)").unwrap();
    let dir = web.path().to_owned();
    let lab = lab(|c| c.web_dir = Some(dir)).await;
    let get = |uri: &str| Request::builder().uri(uri).body(Body::empty()).unwrap();
    for uri in ["/", "/index.html", "/retouch/deep/link"] {
        let (status, headers, body) = lab.send(get(uri)).await;
        assert_eq!(status, StatusCode::OK, "{uri}");
        assert!(body.starts_with(b"<!doctype html>"), "{uri}");
        assert_eq!(headers[header::CACHE_CONTROL], "no-cache", "{uri}");
    }
    let (status, headers, _) = lab.send(get("/assets/app-1a2b.js")).await;
    assert_eq!(status, StatusCode::OK);
    assert!(!headers.contains_key(header::CACHE_CONTROL));
    // The API keeps its own headers.
    let (status, headers, _) = lab.send(get("/v1/health")).await;
    assert_eq!(status, StatusCode::OK);
    assert!(!headers.contains_key(header::CACHE_CONTROL));
}

fn started(detail: &Value) -> i64 {
    detail["run"]["started_at_ms"].as_i64().unwrap()
}

fn finished(detail: &Value) -> i64 {
    detail["run"]["finished_at_ms"].as_i64().unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_exclusive_run_waits_for_the_gpu_and_then_has_it_alone() {
    let lab = lab(|c| c.max_concurrent_runs = 3).await;
    let quick = json!({ "steps": 4, "step_ms": 40 });
    let first = lab.start(quick.clone()).await;
    let second = lab.start(quick.clone()).await;
    assert_eq!(first["status"], "running");
    assert_eq!(second["status"], "running");

    let exclusive = lab.start_op("test.exclusive", quick.clone()).await;
    assert_eq!(exclusive["status"], "queued", "{exclusive}");
    let exclusive = id(&exclusive);
    assert_eq!(lab.detail(&exclusive).await["waiting_for"], "gpu");
    // A slot is free, but the exclusive run is ahead in line for the GPU.
    let later = lab.start(quick.clone()).await;
    assert_eq!(later["status"], "queued", "{later}");
    let later = id(&later);
    assert_eq!(lab.detail(&later).await["waiting_for"], "gpu");
    // CPU-only work does not wait for the GPU.
    let cpu = lab.start_op("test.cpu", quick).await;
    assert_eq!(cpu["status"], "running", "{cpu}");

    let mut done = Vec::new();
    for run in [id(&first), id(&second), exclusive, later, id(&cpu)] {
        let detail = lab.wait_finished(&run).await;
        assert_eq!(status(&detail), "succeeded", "{detail}");
        assert!(detail["waiting_for"].is_null());
        done.push(detail);
    }
    let [first, second, exclusive, later, _cpu] = &done[..] else {
        unreachable!()
    };
    // Submission order holds; the tracker proves no GPU op ran beside the
    // exclusive one (its lock is held exactly while the op runs).
    assert!(started(exclusive) >= started(first).max(started(second)));
    assert!(started(later) >= started(exclusive));
    assert!(finished(later) >= finished(exclusive));
    assert_eq!(lab.tracker.overlaps.load(Ordering::SeqCst), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cpu_only_runs_go_on_while_an_exclusive_run_holds_the_gpu() {
    let lab = lab(|c| c.max_concurrent_runs = 2).await;
    let exclusive = lab
        .start_op("test.exclusive", json!({ "hold": true }))
        .await;
    assert_eq!(exclusive["status"], "running");
    let exclusive = id(&exclusive);
    let shared = lab.start(json!({ "steps": 1 })).await;
    assert_eq!(shared["status"], "queued");
    let shared = id(&shared);

    for _ in 0..2 {
        let cpu = lab
            .start_op("test.cpu", json!({ "steps": 2, "step_ms": 5 }))
            .await;
        assert_eq!(cpu["status"], "running", "{cpu}");
        assert_eq!(status(&lab.wait_finished(&id(&cpu)).await), "succeeded");
    }
    let waiting = lab.detail(&shared).await;
    assert_eq!(status(&waiting), "queued");
    assert_eq!(waiting["waiting_for"], "gpu");

    let (code, _) = lab.cancel(&exclusive).await;
    assert_eq!(code, StatusCode::ACCEPTED);
    assert_eq!(status(&lab.wait_finished(&exclusive).await), "cancelled");
    assert_eq!(status(&lab.wait_finished(&shared).await), "succeeded");
    assert_eq!(lab.tracker.overlaps.load(Ordering::SeqCst), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancelling_a_queued_exclusive_run_lets_the_runs_behind_it_start() {
    let lab = lab(|c| c.max_concurrent_runs = 3).await;
    let holder = id(&lab.start(json!({ "hold": true })).await);
    let exclusive = lab.start_op("test.exclusive", json!({ "steps": 1 })).await;
    assert_eq!(exclusive["status"], "queued");
    let exclusive = id(&exclusive);
    let behind = lab.start(json!({ "steps": 1 })).await;
    assert_eq!(behind["status"], "queued");
    let behind = id(&behind);

    let (code, _) = lab.cancel(&exclusive).await;
    assert_eq!(code, StatusCode::ACCEPTED);
    let cancelled = lab.wait_finished(&exclusive).await;
    assert_eq!(status(&cancelled), "cancelled");
    assert_eq!(cancelled["run"]["error_message"], "cancelled while queued");
    // The run behind it shares the GPU with the holder at once.
    assert_eq!(status(&lab.wait_finished(&behind).await), "succeeded");
    assert_eq!(status(&lab.detail(&holder).await), "running");

    lab.cancel(&holder).await;
    assert_eq!(status(&lab.wait_finished(&holder).await), "cancelled");
    let alone = lab.start_op("test.exclusive", json!({ "steps": 1 })).await;
    assert_eq!(alone["status"], "running", "{alone}");
    assert_eq!(status(&lab.wait_finished(&id(&alone)).await), "succeeded");
    assert_eq!(lab.tracker.overlaps.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn ops_say_how_they_use_the_gpu() {
    let lab = lab(|_| {}).await;
    let (code, ops) = lab.json("GET", "/v1/ops", None).await;
    assert_eq!(code, StatusCode::OK);
    let gpu: Vec<(&str, &str)> = ops
        .as_array()
        .unwrap()
        .iter()
        .map(|op| (op["id"].as_str().unwrap(), op["gpu"].as_str().unwrap()))
        .collect();
    assert_eq!(
        gpu,
        [
            ("test.cpu", "cpu_only"),
            ("test.exclusive", "exclusive"),
            ("test.work", "shared")
        ]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn announced_artifacts_show_live_and_superseded_snapshots_are_pruned() {
    let lab = lab(|_| {}).await;
    // Bytes that exist before the run are never pruned.
    let (code, _, _) = lab
        .send(upload(b"uploaded".to_vec(), Some("text/plain"), None))
        .await;
    assert_eq!(code, StatusCode::CREATED);

    let shown = |slot: &str, text: &str, snapshot: bool| json!({ "slot": slot, "text": text, "snapshot": snapshot });
    let announce = json!([
        shown("points", "pts", false),
        shown("splat", "s1", true),
        shown("splat", "uploaded", true),
        { "slot": "splat", "text": "s2", "snapshot": true, "output": true },
        shown("splat", "pts", true),
        { "slot": "splat", "text": "s3", "snapshot": true, "info": { "iteration": 3000 } },
        shown("preview", "p1", true),
        shown("preview", "p2", true),
        shown("", "no slot", false),
        { "slot": "ghost", "text": "", "unrecorded": true },
    ]);
    let input = json!({ "announce": announce, "steps": 40, "step_ms": 25, "progress_data": true });
    let run = id(&lab.start(input).await);
    let slots = |detail: &Value| -> Vec<(String, u64)> {
        detail["announced"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| {
                let slot = a["slot"].as_str().unwrap().to_owned();
                (slot, a["revision"].as_u64().unwrap())
            })
            .collect()
    };
    let expected = [
        ("points".to_owned(), 1),
        ("preview".to_owned(), 2),
        ("splat".to_owned(), 5),
    ];
    let live = lab
        .wait_until(&run, |d| {
            slots(d) == expected && d["progress"]["data"]["step"].is_number()
        })
        .await;
    assert_eq!(status(&live), "running", "visible before the run ends");
    let splat = &live["announced"][2];
    assert_eq!(splat["artifact"]["kind"], "splat");
    assert_eq!(splat["artifact"]["media_type"], "application/x-spz");
    assert_eq!(splat["artifact"]["size_bytes"], 2);
    assert_eq!(splat["info"], json!({ "iteration": 3000 }));
    assert!(splat["announced_at_ms"].is_number());
    assert!(live["announced"][0]["info"].is_null());

    let done = lab.wait_finished(&run).await;
    assert_eq!(status(&done), "succeeded", "{done}");
    assert_eq!(slots(&done), expected, "kept after the run");
    let hash = |text: &str| blake3::hash(text.as_bytes()).to_hex().to_string();
    assert_eq!(done["announced"][2]["artifact"]["id"], hash("s3"));
    assert!(done["progress"].is_null());

    let store = lab.host.store();
    for (text, kept) in [
        ("s1", false),
        ("p1", false),
        ("uploaded", true),
        ("s2", true),
        ("pts", true),
        ("s3", true),
        ("p2", true),
    ] {
        let record = store.get_artifact(&hash(text)).await.unwrap();
        assert_eq!(record.is_some(), kept, "{text}");
        let (code, _, _) = lab
            .get(&format!("/v1/artifacts/{}", hash(text)), None)
            .await;
        let expected = if kept {
            StatusCode::OK
        } else {
            StatusCode::NOT_FOUND
        };
        assert_eq!(code, expected, "{text}");
    }
    let linked: Vec<String> = store
        .list_run_artifacts(&run)
        .await
        .unwrap()
        .into_iter()
        .map(|link| link.artifact_id)
        .collect();
    // pts, s1, uploaded, s2, s3, p1, p2 and the one with no slot, less two.
    assert_eq!(linked.len(), 6, "pruned snapshots lose their links");
    assert!(!linked.contains(&hash("s1")) && !linked.contains(&hash("p1")));

    // Downloads of op outputs get a name with the right extension.
    let uri = format!("/v1/artifacts/{}?download=", hash("s3"));
    let (code, headers, _) = lab.get(&uri, None).await;
    assert_eq!(code, StatusCode::OK);
    assert_eq!(headers[header::CONTENT_TYPE], "application/x-spz");
    let name = format!("attachment; filename=\"splat-{}.spz\"", &hash("s3")[..12]);
    assert_eq!(headers[header::CONTENT_DISPOSITION], name.as_str());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_cancelled_run_keeps_only_its_latest_snapshot() {
    let lab = lab(|_| {}).await;
    let shown = |text: &str| json!({ "slot": "splat", "text": text, "snapshot": true });
    let input = json!({ "announce": [shown("a"), shown("b"), shown("c")], "hold": true });
    let run = id(&lab.start(input).await);
    lab.wait_until(&run, |d| d["announced"][0]["revision"] == 3)
        .await;
    lab.cancel(&run).await;
    let done = lab.wait_finished(&run).await;
    assert_eq!(status(&done), "cancelled");
    let hash = |text: &str| blake3::hash(text.as_bytes()).to_hex().to_string();
    assert_eq!(done["announced"][0]["artifact"]["id"], hash("c"));
    let store = lab.host.store();
    for (text, kept) in [("a", false), ("b", false), ("c", true)] {
        let record = store.get_artifact(&hash(text)).await.unwrap();
        assert_eq!(record.is_some(), kept, "{text}");
    }
}

#[tokio::test]
async fn uploads_stream_to_disk_and_may_name_their_kind() {
    let lab = lab(|c| c.max_upload_mb = 1).await;
    let chunk = |i: u8| Ok::<_, std::io::Error>(Bytes::from(vec![i; 64 * 1024]));
    let streamed =
        |chunks: u8| Body::from_stream(futures_util::stream::iter((0..chunks).map(chunk)));
    let expected: Vec<u8> = (0..16u8).flat_map(|i| vec![i; 64 * 1024]).collect();

    let mut request = upload(streamed(16), Some("application/x-spz"), Some("scene.spz"));
    request
        .headers_mut()
        .insert("x-spl-kind", "splat".parse().unwrap());
    let (code, _, body) = lab.send(request).await;
    assert_eq!(code, StatusCode::CREATED, "exactly the limit");
    let artifact: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(artifact["id"], blake3::hash(&expected).to_hex().as_str());
    assert_eq!(artifact["kind"], "splat");
    assert_eq!(artifact["size_bytes"], expected.len());
    let id = artifact["id"].as_str().unwrap();
    let record = lab.host.store().get_artifact(id).await.unwrap().unwrap();
    assert_eq!(record.kind, "splat");
    assert_eq!(record.meta, Some(json!({ "filename": "scene.spz" })));
    assert_eq!(lab.host.artifacts().get(id).await.unwrap(), expected);
    assert_eq!(lab.staged(), 0);

    // Over the limit while streaming, with no length announced.
    let (code, _, body) = lab.send(upload(streamed(17), None, None)).await;
    assert_eq!(code, StatusCode::PAYLOAD_TOO_LARGE);
    assert!(String::from_utf8_lossy(&body).contains("1 MiB"));
    assert_eq!(lab.staged(), 0, "the partial upload is gone");

    // A declared length over the limit is refused before the body is read.
    let unreadable = futures_util::stream::once(async {
        Err::<Bytes, _>(std::io::Error::other("must not be read"))
    });
    let mut request = upload(Body::from_stream(unreadable), None, None);
    request
        .headers_mut()
        .insert(header::CONTENT_LENGTH, "1048577".parse().unwrap());
    let (code, _, _) = lab.send(request).await;
    assert_eq!(code, StatusCode::PAYLOAD_TOO_LARGE);

    let mut request = upload(b"x".to_vec(), None, None);
    request
        .headers_mut()
        .insert("x-spl-kind", "hologram".parse().unwrap());
    let (code, _, body) = lab.send(request).await;
    assert_eq!(code, StatusCode::BAD_REQUEST);
    assert!(String::from_utf8_lossy(&body).contains("splat"));
    assert_eq!(lab.staged(), 0);

    let (code, health) = lab.json("GET", "/v1/health", None).await;
    assert_eq!(code, StatusCode::OK);
    assert_eq!(health["max_upload_bytes"], 1024 * 1024);
}

#[test]
fn the_default_upload_limit_fits_a_phone_video() {
    assert_eq!(Config::default().max_upload_mb, 2048);
    assert_eq!(Config::default().max_upload_bytes(), 2 << 30);
}

#[tokio::test]
async fn downloads_answer_byte_ranges() {
    let lab = lab(|_| {}).await;
    let bytes: Vec<u8> = (0..1000u32).map(|i| (i % 251) as u8).collect();
    let (code, _, body) = lab
        .send(upload(bytes.clone(), Some("video/quicktime"), None))
        .await;
    assert_eq!(code, StatusCode::CREATED);
    let artifact: Value = serde_json::from_slice(&body).unwrap();
    let id = artifact["id"].as_str().unwrap();
    let uri = format!("/v1/artifacts/{id}");
    let etag = format!("\"{id}\"");

    let (code, headers, body) = lab.get(&uri, None).await;
    assert_eq!(code, StatusCode::OK);
    assert_eq!(body, bytes);
    assert_eq!(headers[header::ACCEPT_RANGES], "bytes");
    assert_eq!(headers[header::CONTENT_LENGTH], "1000");

    let parts = [
        ("bytes=100-199", 100..200),
        ("bytes=990-", 990..1000),
        ("bytes=-10", 990..1000),
        ("bytes=0-0", 0..1),
        ("bytes=500-5000", 500..1000),
    ];
    for (range, expected) in parts {
        let (code, headers, body) = lab.get_with(&uri, &[("range", range)]).await;
        assert_eq!(code, StatusCode::PARTIAL_CONTENT, "{range}");
        assert_eq!(body, bytes[expected.clone()], "{range}");
        let content_range = format!("bytes {}-{}/1000", expected.start, expected.end - 1);
        assert_eq!(headers[header::CONTENT_RANGE], content_range.as_str());
        let length = expected.len().to_string();
        assert_eq!(headers[header::CONTENT_LENGTH], length.as_str());
        assert_eq!(headers[header::CONTENT_TYPE], "video/quicktime");
        assert_eq!(headers[header::ETAG], etag.as_str());
    }

    let (code, headers, _) = lab.get_with(&uri, &[("range", "bytes=1000-")]).await;
    assert_eq!(code, StatusCode::RANGE_NOT_SATISFIABLE);
    assert_eq!(headers[header::CONTENT_RANGE], "bytes */1000");

    // Only for the copy the client holds; otherwise the whole new body.
    let (code, _, body) = lab
        .get_with(&uri, &[("range", "bytes=0-9"), ("if-range", "\"other\"")])
        .await;
    assert_eq!(code, StatusCode::OK);
    assert_eq!(body.len(), 1000);
    let (code, _, body) = lab
        .get_with(&uri, &[("range", "bytes=0-9"), ("if-range", &etag)])
        .await;
    assert_eq!(code, StatusCode::PARTIAL_CONTENT);
    assert_eq!(body.len(), 10);
    // Several ranges are answered with the whole body.
    let (code, _, body) = lab.get_with(&uri, &[("range", "bytes=0-1,5-6")]).await;
    assert_eq!(code, StatusCode::OK);
    assert_eq!(body.len(), 1000);

    let head = Request::head(&uri).body(Body::empty()).unwrap();
    let (code, headers, body) = lab.send(head).await;
    assert_eq!(code, StatusCode::OK);
    assert_eq!(headers[header::CONTENT_LENGTH], "1000");
    assert!(body.is_empty());

    // Uploads without a recorded name download as `<kind>-<id>.<ext>`.
    let (_, headers, _) = lab.get(&format!("{uri}?download="), None).await;
    let name = format!("attachment; filename=\"blob-{}.mov\"", &id[..12]);
    assert_eq!(headers[header::CONTENT_DISPOSITION], name.as_str());
}
