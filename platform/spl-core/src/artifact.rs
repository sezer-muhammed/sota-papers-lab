use std::path::{Path, PathBuf};

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use ts_rs::TS;

use crate::{Context, Result, bail};

/// The shared data vocabulary between apps. Add kinds here, not in apps, so
/// any app's output can feed any other app's input. [`crate::media`] names
/// the media types that go with them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum ArtifactKind {
    Image,
    Tensor,
    Bitstream,
    Table,
    Report,
    Blob,
    /// 3D Gaussian splats: `application/x-spz` (Niantic SPZ) or
    /// `application/ply` (the INRIA 3DGS layout).
    Splat,
    /// A surface mesh: `model/gltf-binary` (`.glb`) or `application/ply`.
    Mesh,
    /// Camera poses and intrinsics: `application/json`, or a COLMAP model
    /// (`application/zip`).
    Cameras,
}

impl ArtifactKind {
    pub const ALL: [Self; 9] = [
        Self::Image,
        Self::Tensor,
        Self::Bitstream,
        Self::Table,
        Self::Report,
        Self::Blob,
        Self::Splat,
        Self::Mesh,
        Self::Cameras,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Tensor => "tensor",
            Self::Bitstream => "bitstream",
            Self::Table => "table",
            Self::Report => "report",
            Self::Blob => "blob",
            Self::Splat => "splat",
            Self::Mesh => "mesh",
            Self::Cameras => "cameras",
        }
    }

    /// The kind [`as_str`](Self::as_str) names; `None` for anything else.
    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.as_str() == name)
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

    /// Stores the file at `path` and removes it, without holding it in
    /// memory where the store can (the local store moves it into place).
    /// Use it for large outputs written to disk: videos, splat and mesh
    /// exports. On error the file may still be there; the caller cleans up.
    async fn put_file(
        &self,
        path: &Path,
        kind: ArtifactKind,
        media_type: &str,
    ) -> Result<ArtifactRef> {
        let bytes = tokio::fs::read(path)
            .await
            .with_context(|| format!("read {}", path.display()))?;
        let artifact = self.put(bytes, kind, media_type).await?;
        let _ = tokio::fs::remove_file(path).await;
        Ok(artifact)
    }

    async fn get(&self, id: &str) -> Result<Vec<u8>>;

    /// The local file that holds the artifact's bytes, when this store
    /// keeps them in one (whether or not `id` exists). Stream or map large
    /// artifacts (a phone video) from it, or hand it to a child process,
    /// instead of loading them with [`get`](Self::get). Never write to it.
    fn local_path(&self, id: &str) -> Option<PathBuf> {
        let _ = id;
        None
    }

    /// Deletes the bytes; deleting a missing artifact succeeds. Only the
    /// host calls this, after it has dropped the artifact's record (pruned
    /// snapshots). Ops never delete.
    async fn delete(&self, id: &str) -> Result<()> {
        bail!("this artifact store cannot delete {id}")
    }

    fn uri(&self, id: &str) -> String;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds_round_trip_through_their_names() {
        for kind in ArtifactKind::ALL {
            assert_eq!(ArtifactKind::parse(kind.as_str()), Some(kind));
            let json = serde_json::to_value(kind).unwrap();
            assert_eq!(json, kind.as_str());
        }
        assert_eq!(ArtifactKind::parse("Splat"), None);
        assert_eq!(ArtifactKind::parse(""), None);
    }
}
