//! The universal API. Every op of every linked app is reachable through these
//! routes with no per-app code; `web/sdk` is its typed client.

use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::extract::rejection::BytesRejection;
use axum::extract::{DefaultBodyLimit, Path, Query, Request, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use spl_core::{ArtifactKind, ArtifactRef, OpDescriptor};
use spl_db::{MetricRecord, RunRecord, RunStatus};
use tower_http::services::{ServeDir, ServeFile};
use tower_http::trace::TraceLayer;
use ts_rs::TS;

use crate::{Config, Host};

#[derive(Debug, Deserialize, Serialize, TS)]
#[ts(export)]
pub struct RunRequest {
    pub op: String,
    #[serde(default)]
    #[ts(type = "unknown")]
    pub input: serde_json::Value,
    /// `false`: answer at once (202) with the run `running`, or `queued`
    /// while other runs hold every slot, and keep working; poll
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
}

#[derive(Debug, Serialize, TS)]
#[ts(export)]
pub struct Health {
    pub status: String,
    pub version: String,
    pub ops: u32,
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
}

/// Artifacts are content-addressed, so a URL's bytes never change.
const IMMUTABLE: &str = "public, max-age=31536000, immutable";
/// Original file name of an upload, percent-encoded (`encodeURIComponent`).
const FILENAME_HEADER: &str = "x-spl-filename";
/// Longest media type the database schema holds.
const MEDIA_TYPE_MAX: usize = 255;

pub fn router(host: Arc<Host>, config: &Config) -> Router {
    let state = AppState {
        host,
        auth_token: config.auth_token.as_deref().map(Arc::from),
        max_upload_mb: config.max_upload_mb,
    };
    let upload = post(upload_artifact).layer(DefaultBodyLimit::max(config.max_upload_bytes()));
    let protected = Router::new()
        .route("/ops", get(list_ops))
        .route("/runs", get(list_runs).post(create_run))
        .route("/runs/{id}", get(get_run))
        .route("/runs/{id}/cancel", post(cancel_run))
        .route("/artifacts", upload)
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
        app = app.fallback_service(spa);
    }
    app.layer(TraceLayer::new_for_http())
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
    let detail = match live {
        Some(live) => RunDetail {
            run,
            metrics: live.metrics,
            progress: live.progress,
            logs: live.logs,
        },
        None => RunDetail {
            run,
            metrics: store.list_metrics(&id).await?,
            progress: None,
            logs: Vec::new(),
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

/// The raw body is the file. `content-type` becomes the media type and
/// `x-spl-filename` the recorded file name. 201 with the `ArtifactRef`.
async fn upload_artifact(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Result<Response, ApiError> {
    let body = body.map_err(|rejection| {
        if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE {
            ApiError(
                StatusCode::PAYLOAD_TOO_LARGE,
                format!(
                    "upload is larger than {} MiB (SPL_MAX_UPLOAD_MB)",
                    state.max_upload_mb
                ),
            )
        } else {
            ApiError(rejection.status(), rejection.body_text())
        }
    })?;
    if body.is_empty() {
        return Err(ApiError(StatusCode::BAD_REQUEST, "empty upload".into()));
    }
    let media_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .unwrap_or("application/octet-stream");
    if media_type.len() > MEDIA_TYPE_MAX {
        return Err(ApiError(
            StatusCode::BAD_REQUEST,
            format!("content-type is longer than {MEDIA_TYPE_MAX} bytes"),
        ));
    }
    let meta = headers
        .get(FILENAME_HEADER)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| clean_filename(&percent_decode(v)))
        .map(|name| serde_json::json!({ "filename": name }));

    let artifact: ArtifactRef = state
        .host
        .put_artifact(Vec::from(body), ArtifactKind::Blob, media_type, meta)
        .await?;
    let location = format!("/v1/artifacts/{}", artifact.id);
    Ok((
        StatusCode::CREATED,
        [(header::LOCATION, location)],
        Json(artifact),
    )
        .into_response())
}

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
    let bytes = state
        .host
        .artifacts()
        .get(&id)
        .await
        .map_err(|e| ApiError(StatusCode::NOT_FOUND, format!("{e:#}")))?;
    if fresh {
        return Ok(not_modified(etag));
    }

    let media_type = record
        .as_ref()
        .and_then(|r| HeaderValue::from_str(&r.media_type).ok())
        .unwrap_or_else(|| HeaderValue::from_static("application/octet-stream"));
    let length = HeaderValue::from(bytes.len());
    let mut response = Response::new(Body::from(bytes));
    let out = response.headers_mut();
    out.insert(header::CONTENT_TYPE, media_type);
    out.insert(header::CONTENT_LENGTH, length);
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
        let fallback = || {
            let recorded = record
                .as_ref()
                .and_then(|r| r.meta.as_ref())
                .and_then(|m| m.get("filename"))
                .and_then(|v| v.as_str());
            recorded.unwrap_or(&id).to_owned()
        };
        let name = clean_filename(&name).unwrap_or_else(fallback);
        out.insert(header::CONTENT_DISPOSITION, attachment(&name));
    }
    Ok(response)
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
