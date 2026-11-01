//! The universal API. Every op of every linked app is reachable through these
//! routes with no per-app code; `web/sdk` is its typed client.

use std::sync::Arc;

use axum::extract::{Path, Query, Request, State};
use axum::http::{StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use spl_core::OpDescriptor;
use spl_db::{MetricRecord, RunRecord};
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
}

#[derive(Debug, Serialize, TS)]
#[ts(export)]
pub struct RunDetail {
    pub run: RunRecord,
    pub metrics: Vec<MetricRecord>,
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

#[derive(Clone)]
struct AppState {
    host: Arc<Host>,
    auth_token: Option<Arc<str>>,
}

pub fn router(host: Arc<Host>, config: &Config) -> Router {
    let state = AppState {
        host,
        auth_token: config.auth_token.as_deref().map(Arc::from),
    };
    let protected = Router::new()
        .route("/ops", get(list_ops))
        .route("/runs", get(list_runs).post(create_run))
        .route("/runs/{id}", get(get_run))
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

async fn create_run(
    State(state): State<AppState>,
    Json(request): Json<RunRequest>,
) -> Result<Json<RunRecord>, ApiError> {
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
    Ok(Json(state.host.run(&request.op, input).await?))
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
    let metrics = store.list_metrics(&id).await?;
    Ok(Json(RunDetail { run, metrics }))
}

async fn get_artifact(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    let media_type = state
        .host
        .store()
        .get_artifact(&id)
        .await?
        .map(|a| a.media_type)
        .unwrap_or_else(|| "application/octet-stream".into());
    let bytes = state
        .host
        .artifacts()
        .get(&id)
        .await
        .map_err(|e| ApiError(StatusCode::NOT_FOUND, format!("{e:#}")))?;
    Ok(([(header::CONTENT_TYPE, media_type)], bytes).into_response())
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
