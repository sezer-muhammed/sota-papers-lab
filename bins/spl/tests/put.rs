//! `spl put` end to end: the binary stores a file in a fresh data directory
//! and records the file's name with it.

use std::path::Path;
use std::process::{Command, Output};

use serde_json::{Value, json};
use spl_core::Registry;
use spl_host::{Config, Host};

fn config(dir: &Path) -> Config {
    Config {
        data_dir: dir.to_owned(),
        database_url: format!("sqlite://{}", dir.join("spl.db").display()),
        ..Config::default()
    }
}

fn spl_put(dir: &Path, args: &[&str]) -> Output {
    let config = config(dir);
    Command::new(env!("CARGO_BIN_EXE_spl"))
        .arg("put")
        .args(args)
        .env_remove("SPL_CONFIG")
        .env("SPL_DATA_DIR", &config.data_dir)
        .env("SPL_DATABASE_URL", &config.database_url)
        .env("RUST_LOG", "warn")
        .output()
        .unwrap()
}

fn artifact(output: &Output) -> Value {
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "spl put failed: {stderr}");
    serde_json::from_slice(&output.stdout).unwrap()
}

#[tokio::test]
async fn put_stores_a_file_and_records_its_name() {
    let dir = tempfile::tempdir().unwrap();
    let raw = dir.path().join("DSC 0009.ARW");
    std::fs::write(&raw, b"fake raw bytes").unwrap();
    let notes = dir.path().join("notes.bin");
    std::fs::write(&notes, b"plain").unwrap();

    let stored = artifact(&spl_put(dir.path(), &[raw.to_str().unwrap()]));
    assert_eq!(stored["kind"], "blob");
    assert_eq!(stored["media_type"], "image/x-sony-arw");
    assert_eq!(stored["size_bytes"], 14);
    assert!(raw.exists(), "the original stays");
    let overridden = artifact(&spl_put(
        dir.path(),
        &[notes.to_str().unwrap(), "--media-type", "text/plain"],
    ));
    assert_eq!(overridden["media_type"], "text/plain");
    let scene = dir.path().join("scene.SPZ");
    std::fs::write(&scene, b"splats").unwrap();
    let splat = artifact(&spl_put(
        dir.path(),
        &[scene.to_str().unwrap(), "--kind", "splat"],
    ));
    assert_eq!(splat["kind"], "splat");
    assert_eq!(splat["media_type"], "application/x-spz");
    let wrong = spl_put(dir.path(), &[scene.to_str().unwrap(), "--kind", "nope"]);
    assert!(!wrong.status.success());
    assert!(String::from_utf8_lossy(&wrong.stderr).contains("splat"));
    let staging = std::fs::read_dir(dir.path().join("staging")).unwrap();
    assert_eq!(staging.count(), 0, "no copies left behind");

    let host = Host::open(&config(dir.path()), Registry::new())
        .await
        .unwrap();
    let id = stored["id"].as_str().unwrap();
    assert_eq!(host.artifacts().get(id).await.unwrap(), b"fake raw bytes");
    let record = host.store().get_artifact(id).await.unwrap().unwrap();
    assert_eq!(record.meta, Some(json!({ "filename": "DSC 0009.ARW" })));
    assert_eq!(record.media_type, "image/x-sony-arw");

    let missing = spl_put(dir.path(), &["/no/such/photo.ARW"]);
    assert!(!missing.status.success());
    let stderr = String::from_utf8_lossy(&missing.stderr);
    assert!(stderr.contains("read /no/such/photo.ARW"), "{stderr}");
}
