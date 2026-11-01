//! The HTTP API end to end, through `tower::ServiceExt::oneshot`: async runs,
//! progress, cancellation, the run queue, uploads, downloads and restarts.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use axum::Router;
use axum::body::{Body, Bytes};
use axum::http::{HeaderMap, Request, StatusCode, header};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use spl_core::{ArtifactKind, Op, Registry, Result, RunCx, bail};
use spl_db::{NewRun, RunStatus};
use spl_host::{Config, Host, router};
use tempfile::TempDir;
use tower::ServiceExt;

/// Works in steps, reporting progress, a log line and a metric at each, and
/// counts how many of its runs execute at once.
#[derive(Default)]
struct Work {
    running: Arc<AtomicUsize>,
    peak: Arc<AtomicUsize>,
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
}

impl Op for Work {
    const ID: &'static str = "test.work";
    const SUMMARY: &'static str = "Sleeps in steps and reports as it goes";

    type Input = WorkInput;
    type Output = Value;

    async fn run(&self, cx: &RunCx, input: WorkInput) -> Result<Value> {
        let now = self.running.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(now, Ordering::SeqCst);
        let result = work(cx, input).await;
        self.running.fetch_sub(1, Ordering::SeqCst);
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
    for line in 0..input.logs {
        cx.log(format!("line {line}"));
    }
    cx.metric("steps", f64::from(input.steps), "");
    cx.metric("not_a_number", f64::NAN, "");
    for step in 0..input.steps {
        if cx.is_cancelled() {
            bail!("cancelled at step {step}");
        }
        cx.progress(
            f64::from(step) / f64::from(input.steps),
            format!("step {step}"),
        );
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
    Ok(json!({ "steps": input.steps, "artifacts": ids }))
}

struct Lab {
    app: Router,
    host: Arc<Host>,
    peak: Arc<AtomicUsize>,
    _dir: TempDir,
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
    let work = Work::default();
    let peak = work.peak.clone();
    let mut registry = Registry::new();
    registry.op(work);
    let host = Host::open(&config, registry).await.unwrap();
    Lab {
        app: router(host.clone(), &config),
        host,
        peak,
        _dir: dir,
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

    /// `POST /v1/runs` with `wait: false`.
    async fn start(&self, input: Value) -> Value {
        let request = json!({ "op": "test.work", "input": input, "wait": false });
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
    assert_eq!(lab.peak.load(Ordering::SeqCst), 2);
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
    assert_eq!(lab.peak.load(Ordering::SeqCst), 1);
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
