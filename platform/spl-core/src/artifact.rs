use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use ts_rs::TS;

use crate::Result;

/// The shared data vocabulary between apps. Add kinds here, not in apps, so
/// any app's output can feed any other app's input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum ArtifactKind {
    Image,
    Tensor,
    Bitstream,
    Table,
    Report,
    Blob,
}

impl ArtifactKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Tensor => "tensor",
            Self::Bitstream => "bitstream",
            Self::Table => "table",
            Self::Report => "report",
            Self::Blob => "blob",
        }
    }
}

/// A content-addressed handle (blake3 hex). Ops pass these, never raw bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema, TS)]
#[ts(export)]
pub struct ArtifactRef {
    pub id: String,
    pub kind: ArtifactKind,
    pub media_type: String,
    #[ts(type = "number")]
    pub size_bytes: u64,
}

/// Where artifact bytes live: local disk now, S3-compatible storage in the cloud.
#[async_trait]
pub trait ArtifactStore: Send + Sync {
    async fn put(
        &self,
        bytes: Vec<u8>,
        kind: ArtifactKind,
        media_type: &str,
    ) -> Result<ArtifactRef>;
    async fn get(&self, id: &str) -> Result<Vec<u8>>;
    fn uri(&self, id: &str) -> String;
}
