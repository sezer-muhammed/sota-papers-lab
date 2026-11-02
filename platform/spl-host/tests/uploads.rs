//! Resumable chunked uploads (`/v1/uploads`) end to end, through
//! `tower::ServiceExt::oneshot`.

use std::sync::Arc;

use axum::Router;
use axum::body::{Body, Bytes};
use axum::http::{Request, StatusCode, header};
use serde_json::{Value, json};
use spl_core::Registry;
use spl_host::{Config, Host, router};
use tempfile::TempDir;
use tower::ServiceExt;

const PART: usize = 256 << 10;

struct Lab {
    app: Router,
    host: Arc<Host>,
    dir: TempDir,
}

async fn lab(configure: impl FnOnce(&mut Config)) -> Lab {
    let dir = tempfile::tempdir().unwrap();
    let mut config = Config {
        data_dir: dir.path().to_owned(),
        database_url: format!("sqlite://{}", dir.path().join("spl.db").display()),
        ..Config::default()
    };
    configure(&mut config);
    let host = Host::open(&config, Registry::new()).await.unwrap();
    Lab {
        app: router(host.clone(), &config),
        host,
        dir,
    }
}

impl Lab {
    async fn send(&self, request: Request<Body>) -> (StatusCode, Bytes) {
        let response = self.app.clone().oneshot(request).await.unwrap();
        let status = response.status();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        (status, body)
    }

    async fn json(&self, method: &str, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
        let request = Request::builder().method(method).uri(uri);
        let request = match body {
            Some(body) => request
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body.to_string())),
            None => request.body(Body::empty()),
        };
        let (status, bytes) = self.send(request.unwrap()).await;
        let value = if bytes.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&bytes).unwrap()
        };
        (status, value)
    }

    async fn open(&self, size: usize) -> Value {
        let request = json!({
            "size": size,
            "filename": "walk around.mov",
            "media_type": "video/quicktime",
            "part_bytes": PART,
        });
        let (status, session) = self.json("POST", "/v1/uploads", Some(request)).await;
        assert_eq!(status, StatusCode::CREATED, "{session}");
        session
    }

    async fn put(&self, id: &str, n: usize, bytes: &[u8]) -> (StatusCode, Value) {
        let request = Request::put(format!("/v1/uploads/{id}/parts/{n}"))
            .body(Body::from(bytes.to_vec()))
            .unwrap();
        let (status, body) = self.send(request).await;
        (status, serde_json::from_slice(&body).unwrap_or(Value::Null))
    }

    fn staged(&self) -> usize {
        std::fs::read_dir(self.dir.path().join("staging"))
            .unwrap()
            .count()
    }
}

fn file(size: usize) -> Vec<u8> {
    (0..size).map(|i| (i * 31 % 251) as u8).collect()
}

fn part(bytes: &[u8], n: usize) -> &[u8] {
    &bytes[n * PART..((n + 1) * PART).min(bytes.len())]
}

#[tokio::test]
async fn parts_in_any_order_assemble_to_one_artifact() {
    let lab = lab(|_| {}).await;
    let bytes = file(2 * PART + 1234);
    let session = lab.open(bytes.len()).await;
    let id = session["id"].as_str().unwrap().to_owned();
    assert_eq!(session["parts"], 3);
    assert_eq!(session["part_bytes"], PART);
    assert_eq!(session["received"], json!([]));

    for n in [2, 0] {
        let (status, s) = lab.put(&id, n, part(&bytes, n)).await;
        assert_eq!(status, StatusCode::OK, "{s}");
    }
    let (status, s) = lab.json("GET", &format!("/v1/uploads/{id}"), None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(s["received"], json!([0, 2]));

    let (status, error) = lab
        .json("POST", &format!("/v1/uploads/{id}/complete"), None)
        .await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert!(error["error"].as_str().unwrap().contains('1'), "{error}");

    let (status, _) = lab.put(&id, 1, part(&bytes, 1)).await;
    assert_eq!(status, StatusCode::OK);
    let (status, artifact) = lab
        .json("POST", &format!("/v1/uploads/{id}/complete"), None)
        .await;
    assert_eq!(status, StatusCode::CREATED, "{artifact}");
    assert_eq!(artifact["size_bytes"], bytes.len());
    assert_eq!(artifact["media_type"], "video/quicktime");
    assert_eq!(artifact["id"], blake3::hash(&bytes).to_hex().as_str());
    let aid = artifact["id"].as_str().unwrap();
    let record = lab.host.store().get_artifact(aid).await.unwrap().unwrap();
    assert_eq!(record.meta, Some(json!({ "filename": "walk around.mov" })));

    let (status, body) = lab
        .send(
            Request::get(format!("/v1/artifacts/{aid}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body.as_ref(), bytes.as_slice());
    // The session is gone and nothing stays staged.
    let (status, _) = lab.json("GET", &format!("/v1/uploads/{id}"), None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(lab.staged(), 0);
}

#[tokio::test]
async fn a_bad_part_is_refused_and_can_be_sent_again() {
    let lab = lab(|_| {}).await;
    let bytes = file(PART + 10);
    let id = lab.open(bytes.len()).await["id"]
        .as_str()
        .unwrap()
        .to_owned();
    // Short, long, past the end.
    let (status, _) = lab.put(&id, 0, &bytes[..PART - 1]).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, _) = lab.put(&id, 1, &bytes[..11]).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, _) = lab.put(&id, 2, &bytes[..10]).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (_, s) = lab.json("GET", &format!("/v1/uploads/{id}"), None).await;
    assert_eq!(s["received"], json!([]));

    // A resend replaces a part.
    let (status, _) = lab.put(&id, 0, &vec![0u8; PART]).await;
    assert_eq!(status, StatusCode::OK);
    for n in 0..2 {
        let (status, _) = lab.put(&id, n, part(&bytes, n)).await;
        assert_eq!(status, StatusCode::OK);
    }
    let (status, artifact) = lab
        .json("POST", &format!("/v1/uploads/{id}/complete"), None)
        .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(artifact["id"], blake3::hash(&bytes).to_hex().as_str());
}

#[tokio::test]
async fn sessions_are_checked_and_can_be_aborted() {
    let lab = lab(|c| c.max_upload_mb = 1).await;
    let too_big = json!({ "size": (1 << 20) + 1 });
    let (status, _) = lab.json("POST", "/v1/uploads", Some(too_big)).await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    let (status, _) = lab
        .json("POST", "/v1/uploads", Some(json!({ "size": 0 })))
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let tiny = json!({ "size": 10, "part_bytes": 1024 });
    let (status, _) = lab.json("POST", "/v1/uploads", Some(tiny)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let huge = json!({ "size": 10, "part_bytes": 1u64 << 30 });
    let (status, _) = lab.json("POST", "/v1/uploads", Some(huge)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // The default part size; one part for a small file.
    let (status, s) = lab
        .json("POST", "/v1/uploads", Some(json!({ "size": 10 })))
        .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(s["part_bytes"], 16 << 20);
    assert_eq!(s["parts"], 1);
    assert_eq!(lab.staged(), 1);
    let id = s["id"].as_str().unwrap();
    let (status, _) = lab.json("DELETE", &format!("/v1/uploads/{id}"), None).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    assert_eq!(lab.staged(), 0);
    let (status, _) = lab.put(id, 0, b"0123456789").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn uploads_need_the_token() {
    let lab = lab(|c| c.auth_token = Some("s3cret".into())).await;
    let (status, _) = lab
        .json("POST", "/v1/uploads", Some(json!({ "size": 10 })))
        .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    let request = Request::post("/v1/uploads")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, "Bearer s3cret")
        .body(Body::from(json!({ "size": 10 }).to_string()))
        .unwrap();
    let (status, _) = lab.send(request).await;
    assert_eq!(status, StatusCode::CREATED);
}
