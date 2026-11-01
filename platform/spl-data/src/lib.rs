//! [`ArtifactStore`] implementations.
//!
//! Artifacts are immutable and named by the blake3 hash of their bytes, so
//! writing the same output twice is free and every run's lineage is exact.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use spl_core::{ArtifactKind, ArtifactRef, ArtifactStore, Context, Result, bail};

/// Files under `<root>/ab/cd/<hash>`; the two-level fan-out keeps directories small.
pub struct LocalStore {
    root: PathBuf,
}

impl LocalStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn path_for(&self, id: &str) -> Result<PathBuf> {
        if id.len() != 64 || !id.bytes().all(|b| b.is_ascii_hexdigit()) {
            bail!("invalid artifact id `{id}`");
        }
        Ok(self.root.join(&id[0..2]).join(&id[2..4]).join(id))
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
}

#[async_trait]
impl ArtifactStore for LocalStore {
    async fn put(
        &self,
        bytes: Vec<u8>,
        kind: ArtifactKind,
        media_type: &str,
    ) -> Result<ArtifactRef> {
        let id = blake3::hash(&bytes).to_hex().to_string();
        let path = self.path_for(&id)?;
        if !tokio::fs::try_exists(&path).await.unwrap_or(false) {
            let dir = path.parent().context("artifact path has no parent")?;
            tokio::fs::create_dir_all(dir).await?;
            // Write then rename so readers never see a partial file.
            let tmp = path.with_extension("partial");
            tokio::fs::write(&tmp, &bytes).await?;
            tokio::fs::rename(&tmp, &path).await?;
        }
        Ok(ArtifactRef {
            id,
            kind,
            media_type: media_type.to_owned(),
            size_bytes: bytes.len() as u64,
        })
    }

    async fn get(&self, id: &str) -> Result<Vec<u8>> {
        let path = self.path_for(id)?;
        tokio::fs::read(&path)
            .await
            .with_context(|| format!("artifact {id} not found"))
    }

    fn uri(&self, id: &str) -> String {
        format!("file://{}", self.root.join(id).display())
    }
}
