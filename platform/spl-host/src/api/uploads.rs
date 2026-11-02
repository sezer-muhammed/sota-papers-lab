//! Resumable chunked uploads: a big file (a phone video, a folder of
//! photos) arrives as parts of a few MiB each, so no single request passes
//! a proxy's body limit (Cloudflare stops at 100 MB), and a dropped
//! connection costs one part, not the file.
//!
//! `POST /v1/uploads` opens a session and preallocates its staging file;
//! `PUT /v1/uploads/{id}/parts/{n}` streams part `n` to its offset, in any
//! order and in parallel; `GET /v1/uploads/{id}` lists the parts the host
//! has, so a client resumes after a reload; `POST .../complete` moves the
//! file into the artifact store, like a one-shot `POST /v1/artifacts`.
//! Sessions live in memory; one idle for an hour is dropped with its file.

use std::collections::HashMap;
use std::io::SeekFrom;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::{Json, Router};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use spl_core::{ArtifactKind, media};
use tokio::io::{AsyncSeekExt, AsyncWriteExt, BufWriter};
use ts_rs::TS;

use super::{ApiError, AppState, MEDIA_TYPE_MAX, UPLOAD_BUFFER, clean_filename};
use crate::StagedFile;
use crate::host::lock;

/// Part size when the client names none.
const DEFAULT_PART: u64 = 16 << 20;
/// Smallest part a client may ask for (the last part may be shorter).
const MIN_PART: u64 = 256 << 10;
/// Largest part: well under a 100 MB proxy limit.
const MAX_PART: u64 = 64 << 20;
/// Sessions open at once, so a client cannot reserve the disk away.
const MAX_SESSIONS: usize = 32;
/// A session untouched this long is dropped with its file.
const IDLE: Duration = Duration::from_secs(60 * 60);

#[derive(Debug, Deserialize, Serialize, TS)]
#[ts(export)]
pub struct UploadRequest {
    /// Total bytes of the file.
    #[ts(type = "number")]
    pub size: u64,
    #[ts(optional)]
    pub filename: Option<String>,
    /// Recorded media type; `application/octet-stream` when absent.
    #[ts(optional)]
    pub media_type: Option<String>,
    /// Recorded artifact kind; `blob` when absent.
    #[ts(optional)]
    pub kind: Option<ArtifactKind>,
    /// Bytes per part (256 KiB to 64 MiB); 16 MiB when absent.
    #[ts(optional, type = "number")]
    pub part_bytes: Option<u64>,
}

/// An upload in progress.
#[derive(Debug, Serialize, TS)]
#[ts(export)]
pub struct UploadSession {
    pub id: String,
    #[ts(type = "number")]
    pub size: u64,
    #[ts(type = "number")]
    pub part_bytes: u64,
    /// Number of parts; part `n` covers bytes `n * part_bytes` onwards.
    pub parts: u32,
    /// Parts the host holds completely, ascending.
    pub received: Vec<u32>,
}

#[derive(Clone, Copy, PartialEq)]
enum Part {
    Missing,
    Writing,
    Done,
}

struct Session {
    file: StagedFile,
    size: u64,
    part_bytes: u64,
    kind: ArtifactKind,
    media_type: String,
    filename: Option<String>,
    state: Mutex<SessionState>,
}

struct SessionState {
    parts: Vec<Part>,
    touched: Instant,
    /// `complete` took it: no more parts.
    closed: bool,
}

impl Session {
    fn parts(&self) -> u32 {
        self.size.div_ceil(self.part_bytes) as u32
    }

    /// Byte range of part `n`.
    fn range(&self, n: u32) -> (u64, u64) {
        let start = u64::from(n) * self.part_bytes;
        (start, (start + self.part_bytes).min(self.size) - start)
    }

    fn view(&self, id: &str) -> UploadSession {
        let state = lock(&self.state);
        UploadSession {
            id: id.to_owned(),
            size: self.size,
            part_bytes: self.part_bytes,
            parts: self.parts(),
            received: (0..self.parts())
                .filter(|&n| state.parts[n as usize] == Part::Done)
                .collect(),
        }
    }
}

/// The open sessions, by id.
#[derive(Default)]
pub(super) struct Uploads {
    sessions: Mutex<HashMap<String, Arc<Session>>>,
}

impl Uploads {
    fn get(&self, id: &str) -> Result<Arc<Session>, ApiError> {
        lock(&self.sessions)
            .get(id)
            .cloned()
            .ok_or_else(|| ApiError(StatusCode::NOT_FOUND, format!("no upload `{id}`")))
    }

    fn prune(&self) {
        lock(&self.sessions).retain(|_, s| {
            let state = lock(&s.state);
            state.touched.elapsed() < IDLE || state.parts.contains(&Part::Writing)
        });
    }
}

pub(super) fn routes() -> Router<AppState> {
    use axum::routing::{post, put};
    Router::new()
        .route("/uploads", post(create))
        .route("/uploads/{id}", axum::routing::get(status).delete(abort))
        .route("/uploads/{id}/parts/{n}", put(put_part))
        .route("/uploads/{id}/complete", post(complete))
}

fn bad(message: impl Into<String>) -> ApiError {
    ApiError(StatusCode::BAD_REQUEST, message.into())
}

/// 201 with the session. 413 past `max_upload_mb` for the whole file.
async fn create(
    State(state): State<AppState>,
    Json(request): Json<UploadRequest>,
) -> Result<Response, ApiError> {
    if request.size == 0 {
        return Err(bad("empty upload"));
    }
    if request.size > state.max_upload_bytes {
        return Err(ApiError(
            StatusCode::PAYLOAD_TOO_LARGE,
            format!(
                "upload is larger than {} MiB (SPL_MAX_UPLOAD_MB)",
                state.max_upload_mb
            ),
        ));
    }
    let part_bytes = request.part_bytes.unwrap_or(DEFAULT_PART);
    if !(MIN_PART..=MAX_PART).contains(&part_bytes) {
        return Err(bad(format!(
            "part_bytes must be between {MIN_PART} and {MAX_PART}"
        )));
    }
    let media_type = request
        .media_type
        .as_deref()
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .unwrap_or(media::OCTET_STREAM)
        .to_owned();
    if media_type.len() > MEDIA_TYPE_MAX {
        return Err(bad(format!(
            "media_type is longer than {MEDIA_TYPE_MAX} bytes"
        )));
    }
    state.uploads.prune();
    if lock(&state.uploads.sessions).len() >= MAX_SESSIONS {
        return Err(ApiError(
            StatusCode::TOO_MANY_REQUESTS,
            format!("{MAX_SESSIONS} uploads are open; finish or abort one"),
        ));
    }

    let file = state.host.staging_file();
    let created = async {
        let f = tokio::fs::File::create(file.path()).await?;
        // Sparse on the usual filesystems: parts fill it in any order.
        f.set_len(request.size).await
    };
    created.await.map_err(|e| {
        ApiError(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("could not stage the upload: {e}"),
        )
    })?;
    let parts = request.size.div_ceil(part_bytes);
    let session = Arc::new(Session {
        file,
        size: request.size,
        part_bytes,
        kind: request.kind.unwrap_or(ArtifactKind::Blob),
        media_type,
        filename: request.filename.as_deref().and_then(clean_filename),
        state: Mutex::new(SessionState {
            parts: vec![Part::Missing; parts as usize],
            touched: Instant::now(),
            closed: false,
        }),
    });
    let id = uuid::Uuid::now_v7().simple().to_string();
    let view = session.view(&id);
    lock(&state.uploads.sessions).insert(id.clone(), session);
    let location = format!("/v1/uploads/{id}");
    Ok((
        StatusCode::CREATED,
        [(header::LOCATION, location)],
        Json(view),
    )
        .into_response())
}

async fn status(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<UploadSession>, ApiError> {
    Ok(Json(state.uploads.get(&id)?.view(&id)))
}

/// 204; the staged file goes with the session.
async fn abort(State(state): State<AppState>, Path(id): Path<String>) -> StatusCode {
    match lock(&state.uploads.sessions).remove(&id) {
        Some(_) => StatusCode::NO_CONTENT,
        None => StatusCode::NOT_FOUND,
    }
}

/// The raw body is part `n`, exactly its length. 200 with the session; a
/// part sent again replaces itself; 409 while the same part is being
/// written.
async fn put_part(
    State(state): State<AppState>,
    Path((id, n)): Path<(String, u32)>,
    body: Body,
) -> Result<Json<UploadSession>, ApiError> {
    let session = state.uploads.get(&id)?;
    if n >= session.parts() {
        return Err(bad(format!(
            "part {n} is past the last part ({})",
            session.parts() - 1
        )));
    }
    {
        let mut s = lock(&session.state);
        if s.closed {
            return Err(ApiError(
                StatusCode::CONFLICT,
                "upload is completing".into(),
            ));
        }
        if s.parts[n as usize] == Part::Writing {
            return Err(ApiError(
                StatusCode::CONFLICT,
                format!("part {n} is being written"),
            ));
        }
        s.parts[n as usize] = Part::Writing;
        s.touched = Instant::now();
    }
    let (offset, length) = session.range(n);
    let written = write_part(&session, body, offset, length).await;
    let mut s = lock(&session.state);
    s.touched = Instant::now();
    s.parts[n as usize] = if written.is_ok() {
        Part::Done
    } else {
        Part::Missing
    };
    drop(s);
    written?;
    Ok(Json(session.view(&id)))
}

async fn write_part(
    session: &Session,
    body: Body,
    offset: u64,
    length: u64,
) -> Result<(), ApiError> {
    let disk = |e: std::io::Error| {
        ApiError(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("could not store the part: {e}"),
        )
    };
    let mut file = tokio::fs::OpenOptions::new()
        .write(true)
        .open(session.file.path())
        .await
        .map_err(disk)?;
    file.seek(SeekFrom::Start(offset)).await.map_err(disk)?;
    let mut file = BufWriter::with_capacity(UPLOAD_BUFFER, file);
    let mut stream = body.into_data_stream();
    let mut size = 0u64;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| bad(format!("part interrupted: {e}")))?;
        size += chunk.len() as u64;
        if size > length {
            return Err(bad(format!("part is longer than {length} bytes")));
        }
        file.write_all(&chunk).await.map_err(disk)?;
    }
    if size != length {
        return Err(bad(format!("part has {size} bytes, expected {length}")));
    }
    file.flush().await.map_err(disk)?;
    Ok(())
}

/// 201 with the `ArtifactRef`; 409 listing the parts still missing.
async fn complete(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    let session = state.uploads.get(&id)?;
    {
        let mut s = lock(&session.state);
        let missing: Vec<String> = (0..session.parts())
            .filter(|&n| s.parts[n as usize] != Part::Done)
            .map(|n| n.to_string())
            .collect();
        if !missing.is_empty() {
            let shown = missing.iter().take(20).cloned().collect::<Vec<_>>();
            return Err(ApiError(
                StatusCode::CONFLICT,
                format!("parts missing: {}", shown.join(", ")),
            ));
        }
        if s.closed {
            return Err(ApiError(
                StatusCode::CONFLICT,
                "upload is completing".into(),
            ));
        }
        s.closed = true;
    }
    let meta = session
        .filename
        .as_ref()
        .map(|name| serde_json::json!({ "filename": name }));
    let stored = state
        .host
        .put_artifact_file(session.file.path(), session.kind, &session.media_type, meta)
        .await;
    match stored {
        Ok(artifact) => {
            lock(&state.uploads.sessions).remove(&id);
            let location = format!("/v1/artifacts/{}", artifact.id);
            Ok((
                StatusCode::CREATED,
                [(header::LOCATION, location)],
                Json(artifact),
            )
                .into_response())
        }
        Err(e) => {
            lock(&session.state).closed = false;
            Err(e.into())
        }
    }
}
