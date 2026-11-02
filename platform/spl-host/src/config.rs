use std::path::{Path, PathBuf};

use serde::Deserialize;
use spl_core::{Context, Result, bail};

/// Loaded from TOML (`deploy/config/*.toml`), then overridden by `SPL_*`
/// environment variables so one image serves every deployment.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub bind: String,
    pub data_dir: PathBuf,
    /// `sqlite://...` today; `postgres://...` and `oracle://...` once those backends land.
    pub database_url: String,
    /// Built web shell (`web/shell/dist`). Not served when unset.
    pub web_dir: Option<PathBuf>,
    /// Bearer token for `/v1/*`. Set it on anything reachable from the internet.
    pub auth_token: Option<String>,
    /// Ops executing at once; further runs wait, in order, as `queued`.
    /// One GPU is shared by every run, so keep this small.
    pub max_concurrent_runs: usize,
    /// Largest `POST /v1/artifacts` body in MiB. Uploads stream to disk
    /// (`<data_dir>/staging`), never into memory, so this bounds disk use
    /// per upload; the default fits a few minutes of 4K phone video.
    pub max_upload_mb: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            bind: "127.0.0.1:7070".into(),
            data_dir: "./data".into(),
            database_url: "sqlite://./data/spl.db".into(),
            web_dir: None,
            auth_token: None,
            max_concurrent_runs: 3,
            max_upload_mb: 2048,
        }
    }
}

impl Config {
    pub fn load(path: Option<&Path>) -> Result<Self> {
        let mut config = match path {
            Some(path) => {
                let text = std::fs::read_to_string(path)
                    .with_context(|| format!("read config {}", path.display()))?;
                toml::from_str(&text).with_context(|| format!("parse config {}", path.display()))?
            }
            None => Self::default(),
        };
        let env = |key: &str| std::env::var(key).ok().filter(|v| !v.is_empty());
        if let Some(v) = env("SPL_BIND") {
            config.bind = v;
        }
        if let Some(v) = env("SPL_DATA_DIR") {
            config.data_dir = v.into();
        }
        if let Some(v) = env("SPL_DATABASE_URL") {
            config.database_url = v;
        }
        if let Some(v) = env("SPL_WEB_DIR") {
            config.web_dir = Some(v.into());
        }
        if let Some(v) = env("SPL_AUTH_TOKEN") {
            config.auth_token = Some(v);
        }
        if let Some(v) = env("SPL_MAX_CONCURRENT_RUNS") {
            config.max_concurrent_runs = v
                .parse()
                .with_context(|| format!("SPL_MAX_CONCURRENT_RUNS `{v}` is not a count"))?;
        }
        if let Some(v) = env("SPL_MAX_UPLOAD_MB") {
            config.max_upload_mb = v
                .parse()
                .with_context(|| format!("SPL_MAX_UPLOAD_MB `{v}` is not a size in MiB"))?;
        }
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        if self.max_concurrent_runs == 0 {
            bail!("max_concurrent_runs must be at least 1");
        }
        if self.max_upload_mb == 0 {
            bail!("max_upload_mb must be at least 1");
        }
        Ok(())
    }

    pub fn artifact_dir(&self) -> PathBuf {
        self.data_dir.join("artifacts")
    }

    /// Uploads and `spl put` files wait here until they move into the
    /// artifact store; on the same filesystem, so that move is a rename.
    pub fn staging_dir(&self) -> PathBuf {
        self.data_dir.join("staging")
    }

    pub fn max_upload_bytes(&self) -> u64 {
        self.max_upload_mb.saturating_mul(1024 * 1024)
    }
}
