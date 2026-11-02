use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use spl_core::{ArtifactKind, ArtifactRef, ArtifactStore, Context, Result};
use spl_db::{ArtifactRecord, Store};

use crate::host::{lock, now_ms};

/// The artifact store one run's op sees. Every artifact the op stores also
/// gets its database row and an `output` link to the run (`out-0`, `out-1`,
/// ... in the order first stored), so lineage needs no help from the op.
/// Ops cannot delete through it.
pub(crate) struct RecordingStore {
    inner: Arc<dyn ArtifactStore>,
    store: Arc<dyn Store>,
    run_id: String,
    /// Ids already linked, in slot order: storing the same bytes twice links once.
    linked: Mutex<Vec<String>>,
    /// Ids this run recorded first: only these may be pruned as snapshots.
    created: Mutex<HashSet<String>>,
}

impl RecordingStore {
    pub(crate) fn new(
        inner: Arc<dyn ArtifactStore>,
        store: Arc<dyn Store>,
        run_id: String,
    ) -> Self {
        Self {
            inner,
            store,
            run_id,
            linked: Mutex::default(),
            created: Mutex::default(),
        }
    }

    /// The artifacts this run stored that no record existed for before.
    pub(crate) fn created(&self) -> HashSet<String> {
        lock(&self.created).clone()
    }

    async fn record(&self, artifact: &ArtifactRef) -> Result<()> {
        let known = self
            .store
            .get_artifact(&artifact.id)
            .await
            .context("look up artifact")?
            .is_some();
        if !known {
            lock(&self.created).insert(artifact.id.clone());
        }
        let record = artifact_record(artifact, self.inner.uri(&artifact.id), None);
        self.store
            .put_artifact(&record)
            .await
            .context("record artifact")?;
        // Reserve the slot before awaiting, so concurrent puts never share one.
        let slot = {
            let mut linked = lock(&self.linked);
            if linked.contains(&artifact.id) {
                None
            } else {
                linked.push(artifact.id.clone());
                Some(linked.len() - 1)
            }
        };
        if let Some(n) = slot {
            self.store
                .link_artifact(&self.run_id, &artifact.id, "output", &format!("out-{n}"))
                .await
                .context("link artifact to its run")?;
        }
        Ok(())
    }
}

#[async_trait]
impl ArtifactStore for RecordingStore {
    async fn put(
        &self,
        bytes: Vec<u8>,
        kind: ArtifactKind,
        media_type: &str,
    ) -> Result<ArtifactRef> {
        let artifact = self.inner.put(bytes, kind, media_type).await?;
        self.record(&artifact).await?;
        Ok(artifact)
    }

    async fn put_file(
        &self,
        path: &Path,
        kind: ArtifactKind,
        media_type: &str,
    ) -> Result<ArtifactRef> {
        let artifact = self.inner.put_file(path, kind, media_type).await?;
        self.record(&artifact).await?;
        Ok(artifact)
    }

    async fn get(&self, id: &str) -> Result<Vec<u8>> {
        self.inner.get(id).await
    }

    fn local_path(&self, id: &str) -> Option<PathBuf> {
        self.inner.local_path(id)
    }

    fn uri(&self, id: &str) -> String {
        self.inner.uri(id)
    }
}

pub(crate) fn artifact_record(
    artifact: &ArtifactRef,
    uri: String,
    meta: Option<serde_json::Value>,
) -> ArtifactRecord {
    ArtifactRecord {
        id: artifact.id.clone(),
        kind: artifact.kind.as_str().to_owned(),
        media_type: artifact.media_type.clone(),
        size_bytes: i64::try_from(artifact.size_bytes).unwrap_or(i64::MAX),
        uri,
        meta,
        created_at_ms: now_ms(),
    }
}
