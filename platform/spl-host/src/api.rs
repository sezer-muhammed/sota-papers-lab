//! The universal API. Every op of every linked app is reachable through these
//! routes with no per-app code; `web/sdk` is its typed client.

use std::io::SeekFrom;
use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::extract::{Path, Query, Request, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use spl_core::{ArtifactKind, ArtifactRef, ArtifactStore, Context, OpDescriptor, media};
use spl_db::{Announcement, MetricRecord, RunRecord, RunStatus};
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt, BufWriter};
use tokio_util::io::ReaderStream;
use tower_http::services::{ServeDir, ServeFile};
use tower_http::trace::TraceLayer;
use ts_rs::TS;

use crate::queue::WaitReason;
use crate::{Config, Host};

#[derive(Debug, Deserialize, Serialize, TS)]
#[ts(export)]
pub struct RunRequest {
    pub op: String,
    #[serde(default)]
    #[ts(type = "unknown")]
    pub input: serde_json::Value,
    /// `false`: answer at once (202) with the run `running`, or `queued`
    /// while it waits for a slot or the GPU, and keep working; poll
    /// `GET /v1/runs/{id}` for progress. Default `true`.
    #[serde(default)]
    #[ts(optional)]
    pub wait: Option<bool>,
}

/// The latest progress event of a run that is still going.
#[derive(Debug, Clone, Serialize, TS)]
#[ts(export)]
pub struct RunProgress {
    /// 0 to 1.
    pub fraction: f64,
    pub message: String,
    /// Structured state the op attached (`cx.progress_with`): stage,
    /// iteration, counts.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional, type = "unknown")]
    pub data: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, TS)]
#[ts(export)]
pub struct RunDetail {
    pub run: RunRecord,
    pub metrics: Vec<MetricRecord>,
    /// Only while the run is going.
    pub progress: Option<RunProgress>,
    /// Recent log lines of a run that is still going (oldest first).
    pub logs: Vec<String>,
    /// The latest artifact per slot the op announced (`cx.announce`), by
    /// slot name: intermediate results while the run goes, kept after it
    /// ends. Fetch an artifact again only when its `revision` grew.
    pub announced: Vec<Announcement>,
    /// Only while the run is queued: what it waits for.
    pub waiting_for: Option<WaitReason>,
}

#[derive(Debug, Serialize, TS)]
#[ts(export)]
pub struct Health {
    pub status: String,
    pub version: String,
    pub ops: u32,
    /// Largest upload `POST /v1/artifacts` takes (`SPL_MAX_UPLOAD_MB`), so
    /// a page can refuse a bigger file before sending it.
    #[ts(type = "number")]
    pub max_upload_bytes: u64,
    #[ts(type = "unknown")]
    pub env: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct RunsQuery {
    op: Option<String>,
    limit: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct ArtifactQuery {
    /// Serve as an attachment under this file name.
    download: Option<String>,
}

#[derive(Clone)]
struct AppState {
    host: Arc<Host>,
    auth_token: Option<Arc<str>>,
    max_upload_mb: u64,
    max_upload_bytes: u64,
}

/// Artifacts are content-addressed, so a URL's bytes never change.
const IMMUTABLE: &str = "public, max-age=31536000, immutable";
/// Original file name of an upload, percent-encoded (`encodeURIComponent`).
const FILENAME_HEADER: &str = "x-spl-filename";
/// The [`ArtifactKind`] to record an upload as (`splat`, `mesh`, ...);
/// `blob` when absent.
const KIND_HEADER: &str = "x-spl-kind";
/// Longest media type the database schema holds.
const MEDIA_TYPE_MAX: usize = 255;
/// Uploads reach the disk in writes of this size.
const UPLOAD_BUFFER: usize = 1 << 20;
/// Downloads leave the disk in reads of this size.
const DOWNLOAD_CHUNK: usize = 256 * 1024;

pub fn router(host: Arc<Host>, config: &Config) -> Router {
    let state = AppState {
        host,
        auth_token: config.auth_token.as_deref().map(Arc::from),
        max_upload_mb: config.max_upload_mb,
        max_upload_bytes: config.max_upload_bytes(),
    };
    // Uploads stream to disk and enforce `max_upload_mb` themselves: the
    // `Body` extractor they use has no size limit of its own.
    let protected = Router::new()
        .route("/ops", get(list_ops))
        .route("/runs", get(list_runs).post(create_run))
        .route("/runs/{id}", get(get_run))
        .route("/runs/{id}/cancel", post(cancel_run))
        .route("/artifacts", post(upload_artifact))
        .route("/artifacts/{id}", get(get_artifact))
        .route_layer(middleware::from_fn_with_state(state.clone(), require_token));
    let api = Router::new()
        .route("/health", get(health))
        .merge(protected)
        .with_state(state);

    let mut app = Router::new().nest("/v1", api);
    if let Some(dir) = &config.web_dir {
        // Single-page app: unknown paths fall back to index.html.
        let spa = ServeDir::new(dir).fallback(ServeFile::new(dir.join("index.html")));
        let spa = Router::new()
            .fallback_service(spa)
            .layer(middleware::map_response(revalidate_html));
        app = app.fallback_service(spa);
    }
    app.layer(TraceLayer::new_for_http())
}

/// The SPA's HTML (index.html, at any route) is revalidated on every load,
/// so a reload after a deploy picks up the new build; the hashed assets it
/// names keep their ordinary caching.
async fn revalidate_html(mut response: Response) -> Response {
    let html = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("text/html"));
    if html {
        response
            .headers_mut()
            .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    }
    response
}

async fn require_token(State(state): State<AppState>, request: Request, next: Next) -> Response {
    if let Some(expected) = &state.auth_token {
        let given = request
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "));
        if given != Some(&**expected) {
            return ApiError(
                StatusCode::UNAUTHORIZED,
                "missing or wrong bearer token".into(),
            )
            .into_response();
        }
    }
    next.run(request).await
}

async fn health(State(state): State<AppState>) -> Json<Health> {
    Json(Health {
        status: "ok".into(),
        version: env!("CARGO_PKG_VERSION").into(),
        ops: state.host.registry().len() as u32,
        max_upload_bytes: state.max_upload_bytes,
        env: state.host.env().clone(),
    })
}

async fn list_ops(State(state): State<AppState>) -> Json<Vec<OpDescriptor>> {
    Json(state.host.registry().describe())
}

/// `wait` absent or true: 200 with the finished run. `wait: false`: 202 with
/// the run `running` or `queued`.
async fn create_run(
    State(state): State<AppState>,
    Json(request): Json<RunRequest>,
) -> Result<(StatusCode, Json<RunRecord>), ApiError> {
    if state.host.registry().get(&request.op).is_none() {
        return Err(ApiError(
            StatusCode::NOT_FOUND,
            format!("unknown op `{}`", request.op),
        ));
    }
    let input = if request.input.is_null() {
        serde_json::json!({})
    } else {
        request.input
    };
    if request.wait == Some(false) {
        let run = state.host.start(&request.op, input).await?;
        Ok((StatusCode::ACCEPTED, Json(run)))
    } else {
        let run = state.host.run(&request.op, input).await?;
        Ok((StatusCode::OK, Json(run)))
    }
}

async fn list_runs(
    State(state): State<AppState>,
    Query(query): Query<RunsQuery>,
) -> Result<Json<Vec<RunRecord>>, ApiError> {
    let limit = query.limit.unwrap_or(50).min(500);
    Ok(Json(
        state
            .host
            .store()
            .list_runs(query.op.as_deref(), limit)
            .await?,
    ))
}

async fn get_run(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<RunDetail>, ApiError> {
    let store = state.host.store();
    let run = store
        .get_run(&id)
        .await?
        .ok_or_else(|| ApiError(StatusCode::NOT_FOUND, format!("no run `{id}`")))?;
    // Read after the record: a run that has just ended is either still live
    // here or already has its metrics saved.
    let live = is_going(run.status).then(|| state.host.live(&id)).flatten();
    let announced = store.list_announcements(&id).await?;
    let detail = match live {
        Some(live) => RunDetail {
            run,
            metrics: live.metrics,
            progress: live.progress,
            logs: live.logs,
            announced,
            waiting_for: live.waiting_for,
        },
        None => RunDetail {
            run,
            metrics: store.list_metrics(&id).await?,
            progress: None,
            logs: Vec::new(),
            announced,
            waiting_for: None,
        },
    };
    Ok(Json(detail))
}

/// 202 with the run as it is now; it ends as `cancelled` shortly after.
async fn cancel_run(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<(StatusCode, Json<RunRecord>), ApiError> {
    let store = state.host.store();
    let not_running = |run: &RunRecord| {
        ApiError(
            StatusCode::CONFLICT,
            format!("run `{id}` is {}, not running", run.status.as_str()),
        )
    };
    let run = store
        .get_run(&id)
        .await?
        .ok_or_else(|| ApiError(StatusCode::NOT_FOUND, format!("no run `{id}`")))?;
    if !is_going(run.status) {
        return Err(not_running(&run));
    }
    if !state.host.cancel(&id) {
        // Finished in the meantime, or started by another process (`spl run`).
        let now = store.get_run(&id).await?.unwrap_or(run);
        return Err(if is_going(now.status) {
            ApiError(
                StatusCode::CONFLICT,
                format!("run `{id}` is not running in this host"),
            )
        } else {
            not_running(&now)
        });
    }
    let run = store.get_run(&id).await?.unwrap_or(run);
    Ok((StatusCode::ACCEPTED, Json(run)))
}

fn is_going(status: RunStatus) -> bool {
    matches!(status, RunStatus::Queued | RunStatus::Running)
}

/// The raw body is the file, streamed to disk: memory use does not grow
/// with the upload. `content-type` becomes the media type, `x-spl-filename`
/// the recorded file name, `x-spl-kind` the artifact kind. 201 with the
/// `ArtifactRef`; 413 past `max_upload_mb`, before reading anything when
/// `content-length` says so.
async fn upload_artifact(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Body,
) -> Result<Response, ApiError> {
    let too_large = || {
        ApiError(
            StatusCode::PAYLOAD_TOO_LARGE,
            format!(
                "upload is larger than {} MiB (SPL_MAX_UPLOAD_MB)",
                state.max_upload_mb
            ),
        )
    };
    let media_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .unwrap_or(media::OCTET_STREAM);
    if media_type.len() > MEDIA_TYPE_MAX {
        return Err(ApiError(
            StatusCode::BAD_REQUEST,
            format!("content-type is longer than {MEDIA_TYPE_MAX} bytes"),
        ));
    }
    let kind = match headers.get(KIND_HEADER) {
        None => ArtifactKind::Blob,
        Some(value) => value
            .to_str()
            .ok()
            .and_then(|v| ArtifactKind::parse(v.trim()))
            .ok_or_else(|| {
                let kinds: Vec<_> = ArtifactKind::ALL.iter().map(|k| k.as_str()).collect();
                ApiError(
                    StatusCode::BAD_REQUEST,
                    format!("{KIND_HEADER} is not one of {}", kinds.join(", ")),
                )
            })?,
    };
    let meta = headers
        .get(FILENAME_HEADER)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| clean_filename(&percent_decode(v)))
        .map(|name| serde_json::json!({ "filename": name }));
    let declared = headers
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok());
    if declared.is_some_and(|length| length > state.max_upload_bytes) {
        return Err(too_large());
    }

    // Removed on every early return; moved into the store on success.
    let staged = state.host.staging_file();
    let size = match receive(body, staged.path(), state.max_upload_bytes).await {
        Ok(size) => size,
        Err(Received::TooLarge) => return Err(too_large()),
        Err(Received::Broken(e)) => {
            return Err(ApiError(
                StatusCode::BAD_REQUEST,
                format!("upload interrupted: {e}"),
            ));
        }
        Err(Received::Disk(e)) => {
            return Err(ApiError(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("could not store the upload: {e}"),
            ));
        }
    };
    if size == 0 {
        return Err(ApiError(StatusCode::BAD_REQUEST, "empty upload".into()));
    }
    let artifact: ArtifactRef = state
        .host
        .put_artifact_file(staged.path(), kind, media_type, meta)
        .await?;
    let location = format!("/v1/artifacts/{}", artifact.id);
    Ok((
        StatusCode::CREATED,
        [(header::LOCATION, location)],
        Json(artifact),
    )
        .into_response())
}

/// Why an upload stopped.
enum Received {
    TooLarge,
    /// The client's body failed (it hung up).
    Broken(axum::Error),
    Disk(std::io::Error),
}

/// Streams `body` to a new file at `path`, at most `limit` bytes, and
/// returns how many it wrote.
async fn receive(body: Body, path: &std::path::Path, limit: u64) -> Result<u64, Received> {
    let file = tokio::fs::File::create(path)
        .await
        .map_err(Received::Disk)?;
    let mut file = BufWriter::with_capacity(UPLOAD_BUFFER, file);
    let mut stream = body.into_data_stream();
    let mut size = 0u64;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(Received::Broken)?;
        size += chunk.len() as u64;
        if size > limit {
            return Err(Received::TooLarge);
        }
        file.write_all(&chunk).await.map_err(Received::Disk)?;
    }
    // Waits for the last write, so the file is complete when it moves.
    file.flush().await.map_err(Received::Disk)?;
    Ok(size)
}

/// The bytes, streamed from disk. A single byte range (`Range: bytes=…`)
/// answers 206, so video players seek and viewers fetch parts of big
/// splats; `If-Range` and `If-None-Match` use the ETag (the id).
async fn get_artifact(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(query): Query<ArtifactQuery>,
    headers: HeaderMap,
) -> Result<Response, ApiError> {
    let missing = || ApiError(StatusCode::NOT_FOUND, format!("no artifact `{id}`"));
    let etag = HeaderValue::from_str(&format!("\"{id}\"")).map_err(|_| missing())?;
    let record = state.host.store().get_artifact(&id).await?;
    // Same id, same bytes: a client holding this ETag has the artifact, so
    // a recorded artifact needs no disk read to answer it.
    let fresh = if_none_match(&headers, &etag);
    if fresh && record.is_some() {
        return Ok(not_modified(etag));
    }
    let content = Content::open(state.host.artifacts().as_ref(), &id)
        .await
        .map_err(|e| ApiError(StatusCode::NOT_FOUND, format!("{e:#}")))?;
    if fresh {
        return Ok(not_modified(etag));
    }

    let length = content.len();
    let range = if if_range_matches(&headers, &etag) {
        let requested = headers.get(header::RANGE).and_then(|v| v.to_str().ok());
        byte_range(requested, length)
    } else {
        ByteRange::Full
    };
    let (start, count) = match range {
        ByteRange::Full => (0, length),
        ByteRange::Part { start, end } => (start, end - start + 1),
        ByteRange::Unsatisfiable => {
            let mut response = ApiError(
                StatusCode::RANGE_NOT_SATISFIABLE,
                format!("artifact `{id}` has {length} bytes"),
            )
            .into_response();
            let out = response.headers_mut();
            out.insert(header::ACCEPT_RANGES, HeaderValue::from_static("bytes"));
            out.insert(
                header::CONTENT_RANGE,
                header_value(format!("bytes */{length}")),
            );
            return Ok(response);
        }
    };
    let body = content.body(start, count).await.map_err(|e| {
        ApiError(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("read artifact `{id}`: {e}"),
        )
    })?;

    let media_type = record
        .as_ref()
        .and_then(|r| HeaderValue::from_str(&r.media_type).ok())
        .unwrap_or_else(|| HeaderValue::from_static(media::OCTET_STREAM));
    let mut response = Response::new(body);
    if let ByteRange::Part { start, end } = range {
        *response.status_mut() = StatusCode::PARTIAL_CONTENT;
        response.headers_mut().insert(
            header::CONTENT_RANGE,
            header_value(format!("bytes {start}-{end}/{length}")),
        );
    }
    let out = response.headers_mut();
    out.insert(header::CONTENT_TYPE, media_type);
    out.insert(header::CONTENT_LENGTH, HeaderValue::from(count));
    out.insert(header::ACCEPT_RANGES, HeaderValue::from_static("bytes"));
    out.insert(header::ETAG, etag);
    out.insert(header::CACHE_CONTROL, HeaderValue::from_static(IMMUTABLE));
    // Uploads choose their own media type: never let one run as a page on
    // this origin, where the bearer token lives.
    out.insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    out.insert(
        header::CONTENT_SECURITY_POLICY,
        HeaderValue::from_static("sandbox"),
    );
    if let Some(name) = query.download {
        // No name given: the recorded file name, else one from the kind
        // and media type (`splat-1a2b3c4d5e6f.spz`).
        let fallback = || match &record {
            Some(r) => r
                .meta
                .as_ref()
                .and_then(|m| m.get("filename"))
                .and_then(|v| v.as_str())
                .map(str::to_owned)
                .unwrap_or_else(|| download_name(&id, &r.kind, &r.media_type)),
            None => download_name(&id, "artifact", media::OCTET_STREAM),
        };
        let name = clean_filename(&name).unwrap_or_else(fallback);
        out.insert(header::CONTENT_DISPOSITION, attachment(&name));
    }
    Ok(response)
}

/// An artifact's bytes, ready to stream: a local file, or the bytes a store
/// without local files returned.
enum Content {
    File(tokio::fs::File, u64),
    Bytes(Bytes),
}

impl Content {
    async fn open(store: &dyn ArtifactStore, id: &str) -> spl_core::Result<Self> {
        if let Some(path) = store.local_path(id) {
            let file = tokio::fs::File::open(&path)
                .await
                .with_context(|| format!("artifact {id} not found"))?;
            let length = file.metadata().await?.len();
            return Ok(Self::File(file, length));
        }
        Ok(Self::Bytes(Bytes::from(store.get(id).await?)))
    }

    fn len(&self) -> u64 {
        match self {
            Self::File(_, length) => *length,
            Self::Bytes(bytes) => bytes.len() as u64,
        }
    }

    /// `count` bytes from `start`; [`byte_range`] keeps both in bounds.
    async fn body(self, start: u64, count: u64) -> std::io::Result<Body> {
        Ok(match self {
            Self::File(mut file, _) => {
                if start > 0 {
                    file.seek(SeekFrom::Start(start)).await?;
                }
                Body::from_stream(ReaderStream::with_capacity(
                    file.take(count),
                    DOWNLOAD_CHUNK,
                ))
            }
            Self::Bytes(bytes) => {
                let start = usize::try_from(start)
                    .unwrap_or(usize::MAX)
                    .min(bytes.len());
                let count = usize::try_from(count).unwrap_or(usize::MAX);
                let end = start.saturating_add(count).min(bytes.len());
                Body::from(bytes.slice(start..end))
            }
        })
    }
}

/// What a `Range` header asks of a body of `length` bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ByteRange {
    /// No header, another unit, several ranges or nonsense: the whole body
    /// (a server may always ignore `Range`).
    Full,
    /// Inclusive, within the body.
    Part { start: u64, end: u64 },
    /// Starts past the end, or asks for an empty suffix: 416.
    Unsatisfiable,
}

fn byte_range(header: Option<&str>, length: u64) -> ByteRange {
    let Some(spec) = header.and_then(|h| h.trim().strip_prefix("bytes=")) else {
        return ByteRange::Full;
    };
    let Some((first, last)) = spec.split_once('-').filter(|_| !spec.contains(',')) else {
        return ByteRange::Full;
    };
    let (first, last) = (first.trim(), last.trim());
    if first.is_empty() {
        // `bytes=-n`: the last n bytes.
        return match last.parse::<u64>() {
            Ok(0) => ByteRange::Unsatisfiable,
            Ok(_) if length == 0 => ByteRange::Unsatisfiable,
            Ok(n) => ByteRange::Part {
                start: length - n.min(length),
                end: length - 1,
            },
            Err(_) => ByteRange::Full,
        };
    }
    let Ok(start) = first.parse::<u64>() else {
        return ByteRange::Full;
    };
    let end = match last {
        "" => u64::MAX,
        last => match last.parse::<u64>() {
            Ok(end) if end >= start => end,
            _ => return ByteRange::Full,
        },
    };
    if start >= length {
        return ByteRange::Unsatisfiable;
    }
    ByteRange::Part {
        start,
        end: end.min(length - 1),
    }
}

/// `If-Range`: honour `Range` only while the client's copy is this one. The
/// ETag is the only validator (strong; no `Last-Modified`), so a date or a
/// weak tag never matches and the whole body goes out.
fn if_range_matches(headers: &HeaderMap, etag: &HeaderValue) -> bool {
    headers
        .get(header::IF_RANGE)
        .is_none_or(|tag| tag.as_bytes().trim_ascii() == etag.as_bytes())
}

/// What a download of an artifact without a recorded file name is called.
fn download_name(id: &str, kind: &str, media_type: &str) -> String {
    let stem = format!("{kind}-{}", id.get(..12).unwrap_or(id));
    match media::extension(media_type) {
        Some(extension) => format!("{stem}.{extension}"),
        None => stem,
    }
}

fn header_value(text: String) -> HeaderValue {
    HeaderValue::try_from(text).unwrap_or_else(|_| HeaderValue::from_static(""))
}

fn if_none_match(headers: &HeaderMap, etag: &HeaderValue) -> bool {
    headers
        .get_all(header::IF_NONE_MATCH)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(','))
        .map(|tag| tag.trim().trim_start_matches("W/"))
        .any(|tag| tag == "*" || tag.as_bytes() == etag.as_bytes())
}

fn not_modified(etag: HeaderValue) -> Response {
    (
        StatusCode::NOT_MODIFIED,
        [
            (header::ETAG, etag),
            (header::CACHE_CONTROL, HeaderValue::from_static(IMMUTABLE)),
        ],
    )
        .into_response()
}

/// `attachment` with a plain ASCII `filename` for old clients and, when the
/// name needs it, the exact UTF-8 name as RFC 5987 `filename*`, which every
/// current browser prefers.
fn attachment(name: &str) -> HeaderValue {
    let ascii: String = name
        .chars()
        .map(|c| match c {
            '"' | '\\' | '%' => '_',
            c if c == ' ' || c.is_ascii_graphic() => c,
            _ => '_',
        })
        .collect();
    let mut value = format!("attachment; filename=\"{ascii}\"");
    if ascii != name {
        value.push_str("; filename*=UTF-8''");
        for byte in name.bytes() {
            if byte.is_ascii_alphanumeric() || b"!#$&+-.^_`|~".contains(&byte) {
                value.push(char::from(byte));
            } else {
                value.push_str(&format!("%{byte:02X}"));
            }
        }
    }
    HeaderValue::from_str(&value).unwrap_or_else(|_| HeaderValue::from_static("attachment"))
}

/// The last path component, without control characters, at most 255
/// characters; `None` when nothing usable is left.
fn clean_filename(raw: &str) -> Option<String> {
    let base = raw.rsplit(['/', '\\']).next().unwrap_or(raw);
    let name: String = base.chars().filter(|c| !c.is_control()).take(255).collect();
    let name = name.trim();
    (!name.is_empty() && name != "." && name != "..").then(|| name.to_owned())
}

/// Undoes `encodeURIComponent`. Malformed escapes stay as they are.
fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let hex = |b: u8| char::from(b).to_digit(16);
        if bytes[i] == b'%'
            && let Some((hi, lo)) = bytes
                .get(i + 1)
                .zip(bytes.get(i + 2))
                .and_then(|(&h, &l)| Some((hex(h)?, hex(l)?)))
        {
            out.push((hi * 16 + lo) as u8);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

struct ApiError(StatusCode, String);

impl From<spl_core::Error> for ApiError {
    fn from(e: spl_core::Error) -> Self {
        Self(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}"))
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(serde_json::json!({ "error": self.1 }))).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_names_need_no_extended_parameter() {
        assert_eq!(
            attachment("DSC0009 edit.jpg"),
            "attachment; filename=\"DSC0009 edit.jpg\""
        );
    }

    #[test]
    fn other_names_also_carry_the_exact_utf8_name() {
        assert_eq!(
            attachment("Çay bahçesi.jpg"),
            "attachment; filename=\"_ay bah_esi.jpg\"; filename*=UTF-8''%C3%87ay%20bah%C3%A7esi.jpg"
        );
        assert_eq!(
            attachment("a\"b%.jpg"),
            "attachment; filename=\"a_b_.jpg\"; filename*=UTF-8''a%22b%25.jpg"
        );
    }

    #[test]
    fn filenames_lose_paths_and_control_characters() {
        assert_eq!(
            clean_filename("../../etc/passwd").as_deref(),
            Some("passwd")
        );
        assert_eq!(clean_filename("C:\\x\\a.ARW").as_deref(), Some("a.ARW"));
        assert_eq!(clean_filename("a\r\nb.jpg").as_deref(), Some("ab.jpg"));
        assert_eq!(clean_filename(" .. "), None);
        assert_eq!(clean_filename(""), None);
        assert_eq!(clean_filename(&"x".repeat(300)).map(|n| n.len()), Some(255));
    }

    #[test]
    fn single_byte_ranges_are_served_and_others_ignored() {
        use ByteRange::{Full, Part, Unsatisfiable};
        let part = |start, end| Part { start, end };
        let cases = [
            (None, 100, Full),
            (Some("bytes=0-9"), 100, part(0, 9)),
            (Some(" bytes=10-"), 100, part(10, 99)),
            (Some("bytes=90-500"), 100, part(90, 99)),
            (Some("bytes=-5"), 100, part(95, 99)),
            (Some("bytes=-500"), 100, part(0, 99)),
            (Some("bytes=99-99"), 100, part(99, 99)),
            (Some("bytes=100-"), 100, Unsatisfiable),
            (Some("bytes=-0"), 100, Unsatisfiable),
            (Some("bytes=-1"), 0, Unsatisfiable),
            (Some("bytes=0-"), 0, Unsatisfiable),
            (Some("bytes=5-1"), 100, Full),
            (Some("bytes=0-1,5-6"), 100, Full),
            (Some("items=0-1"), 100, Full),
            (Some("bytes=x-1"), 100, Full),
            (Some("bytes=1"), 100, Full),
        ];
        for (header, length, expected) in cases {
            assert_eq!(
                byte_range(header, length),
                expected,
                "{header:?} of {length}"
            );
        }
    }

    #[test]
    fn downloads_without_a_name_get_one_with_the_right_extension() {
        let id = "1a2b3c4d5e6f7a8b9c0d";
        let cases = [
            ("splat", "application/x-spz", "splat-1a2b3c4d5e6f.spz"),
            ("splat", "application/ply", "splat-1a2b3c4d5e6f.ply"),
            ("mesh", "model/gltf-binary", "mesh-1a2b3c4d5e6f.glb"),
            ("cameras", "application/json", "cameras-1a2b3c4d5e6f.json"),
            ("blob", "x/unknown", "blob-1a2b3c4d5e6f"),
        ];
        for (kind, media_type, name) in cases {
            assert_eq!(download_name(id, kind, media_type), name);
        }
        assert_eq!(download_name("short", "blob", "x/unknown"), "blob-short");
    }

    #[test]
    fn percent_decoding_matches_encode_uri_component() {
        assert_eq!(
            percent_decode("%C3%87ay%20bah%C3%A7esi.ARW"),
            "Çay bahçesi.ARW"
        );
        assert_eq!(percent_decode("a+b%2Bc"), "a+b+c");
        assert_eq!(percent_decode("100%"), "100%");
        assert_eq!(percent_decode("%zz%4"), "%zz%4");
    }
}
