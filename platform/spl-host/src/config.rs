use std::path::{Path, PathBuf};

use serde::Deserialize;
use spl_core::{Context, Result};

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
}

impl Default for Config {
    fn default() -> Self {
        Self {
            bind: "127.0.0.1:7070".into(),
            data_dir: "./data".into(),
            database_url: "sqlite://./data/spl.db".into(),
            web_dir: None,
            auth_token: None,
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
        Ok(config)
    }

    pub fn artifact_dir(&self) -> PathBuf {
        self.data_dir.join("artifacts")
    }
}
