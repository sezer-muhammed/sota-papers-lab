use serde::{Deserialize, Serialize};
use spl_core::{Result, bail};
use ts_rs::TS;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum RunStatus {
    Queued,
    Running,
    Succeeded,
    Failed,
    Cancelled,
}

impl RunStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "queued" => Self::Queued,
            "running" => Self::Running,
            "succeeded" => Self::Succeeded,
            "failed" => Self::Failed,
            "cancelled" => Self::Cancelled,
            other => bail!("unknown run status `{other}`"),
        })
    }
}

#[derive(Debug, Clone)]
pub struct NewRun {
    pub id: String,
    pub op_id: String,
    pub input: serde_json::Value,
    pub host_name: Option<String>,
    pub git_sha: Option<String>,
    pub env: Option<serde_json::Value>,
    pub created_at_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export)]
pub struct RunRecord {
    pub id: String,
    pub op_id: String,
    pub status: RunStatus,
    #[ts(type = "unknown")]
    pub input: serde_json::Value,
    #[ts(type = "unknown")]
    pub output: Option<serde_json::Value>,
    pub error_message: Option<String>,
    pub host_name: Option<String>,
    pub git_sha: Option<String>,
    #[ts(type = "unknown")]
    pub env: Option<serde_json::Value>,
    #[ts(type = "number")]
    pub created_at_ms: i64,
    #[ts(type = "number | null")]
    pub started_at_ms: Option<i64>,
    #[ts(type = "number | null")]
    pub finished_at_ms: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export)]
pub struct ArtifactRecord {
    pub id: String,
    pub kind: String,
    pub media_type: String,
    #[ts(type = "number")]
    pub size_bytes: i64,
    pub uri: String,
    #[ts(type = "unknown")]
    pub meta: Option<serde_json::Value>,
    #[ts(type = "number")]
    pub created_at_ms: i64,
}

/// One lineage edge: a run read (`input`) or wrote (`output`) an artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunArtifact {
    pub run_id: String,
    pub artifact_id: String,
    pub direction: String,
    /// The op's name for the artifact; hosts number unnamed outputs `out-0`, `out-1`, ...
    pub slot_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export)]
pub struct MetricRecord {
    pub run_id: String,
    #[ts(type = "number")]
    pub seq: i64,
    pub key: String,
    pub value: f64,
    /// `None` when unitless. Never store an empty string: Oracle reads it back as NULL.
    pub unit: Option<String>,
}
