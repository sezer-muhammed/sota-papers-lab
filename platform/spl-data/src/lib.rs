//! [`ArtifactStore`] implementations.
//!
//! Artifacts are immutable and named by the blake3 hash of their bytes, so
//! writing the same output twice is free and every run's lineage is exact.

use std::io::{ErrorKind, Read};
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use spl_core::{ArtifactKind, ArtifactRef, ArtifactStore, Context, Result, anyhow, bail};

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

    /// Hashes the file where it is and renames it into place, so a 2 GB
    /// video is never copied or held in memory. A file on another
    /// filesystem is copied instead.
    async fn put_file(
        &self,
        source: &Path,
        kind: ArtifactKind,
        media_type: &str,
    ) -> Result<ArtifactRef> {
        let file = source.to_owned();
        let (id, size_bytes) = tokio::task::spawn_blocking(move || hash_file(&file))
            .await
            .map_err(|e| anyhow!("hashing task died: {e}"))??;

        let path = self.path_for(&id)?;
        if tokio::fs::try_exists(&path).await.unwrap_or(false) {
            tokio::fs::remove_file(source).await?;
        } else {
            let dir = path.parent().context("artifact path has no parent")?;
            tokio::fs::create_dir_all(dir).await?;
            match tokio::fs::rename(source, &path).await {
                Ok(()) => {}
                Err(e) if e.kind() == ErrorKind::CrossesDevices => {
                    let tmp = path.with_extension("partial");
                    tokio::fs::copy(source, &tmp).await?;
                    tokio::fs::rename(&tmp, &path).await?;
                    tokio::fs::remove_file(source).await?;
                }
                Err(e) => {
                    return Err(e)
                        .with_context(|| format!("move {} into the store", source.display()));
                }
            }
        }
        Ok(ArtifactRef {
            id,
            kind,
            media_type: media_type.to_owned(),
            size_bytes,
        })
    }

    async fn get(&self, id: &str) -> Result<Vec<u8>> {
        let path = self.path_for(id)?;
        tokio::fs::read(&path)
            .await
            .with_context(|| format!("artifact {id} not found"))
    }

    fn local_path(&self, id: &str) -> Option<PathBuf> {
        self.path_for(id).ok()
    }

    async fn delete(&self, id: &str) -> Result<()> {
        let path = self.path_for(id)?;
        match tokio::fs::remove_file(&path).await {
            Err(e) if e.kind() != ErrorKind::NotFound => {
                Err(e).with_context(|| format!("delete artifact {id}"))
            }
            _ => Ok(()),
        }
    }

    fn uri(&self, id: &str) -> String {
        format!("file://{}", self.root.join(id).display())
    }
}

/// Bytes [`hash_file`] reads at a time; each chunk is hashed on every core.
const HASH_CHUNK: usize = 16 << 20;

/// The blake3 id and size of a file, read in chunks: memory stays at one
/// chunk, and the file is not mapped, so a big upload never shows up as
/// resident memory of the host.
fn hash_file(path: &Path) -> Result<(String, u64)> {
    let mut file = std::fs::File::open(path).with_context(|| format!("read {}", path.display()))?;
    let mut hasher = blake3::Hasher::new();
    let mut chunk = vec![0u8; HASH_CHUNK];
    let mut size = 0u64;
    loop {
        let mut filled = 0;
        while filled < chunk.len() {
            match file.read(&mut chunk[filled..]) {
                Ok(0) => break,
                Ok(n) => filled += n,
                Err(e) if e.kind() == ErrorKind::Interrupted => {}
                Err(e) => return Err(e).with_context(|| format!("read {}", path.display())),
            }
        }
        if filled == 0 {
            break;
        }
        hasher.update_rayon(&chunk[..filled]);
        size += filled as u64;
    }
    Ok((hasher.finalize().to_hex().to_string(), size))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunked_hashing_matches_hashing_in_one_go() -> Result<()> {
        let dir = tempfile::tempdir()?;
        for length in [0, 1, HASH_CHUNK - 1, HASH_CHUNK, HASH_CHUNK + 4097] {
            let bytes: Vec<u8> = (0..length).map(|i| (i % 253) as u8).collect();
            let path = dir.path().join(format!("{length}.bin"));
            std::fs::write(&path, &bytes)?;
            let (id, size) = hash_file(&path)?;
            assert_eq!(id, blake3::hash(&bytes).to_hex().as_str(), "{length}");
            assert_eq!(size, length as u64);
        }
        Ok(())
    }

    #[tokio::test]
    async fn files_are_moved_in_under_the_hash_of_their_bytes() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let store = LocalStore::new(dir.path().join("artifacts"));
        // Large enough for blake3 to map it and hash on several threads.
        let bytes: Vec<u8> = (0..3_000_000u32).map(|i| (i * 7 % 251) as u8).collect();
        let from_memory = store
            .put(bytes.clone(), ArtifactKind::Blob, "video/mp4")
            .await?;

        let source = dir.path().join("upload.part");
        std::fs::write(&source, &bytes)?;
        let moved = store
            .put_file(&source, ArtifactKind::Splat, "application/x-spz")
            .await?;
        assert_eq!(moved.id, from_memory.id);
        assert_eq!(moved.size_bytes, bytes.len() as u64);
        assert_eq!(moved.kind, ArtifactKind::Splat);
        assert!(!source.exists(), "the source is consumed");
        assert_eq!(store.get(&moved.id).await?, bytes);

        let fresh = dir.path().join("fresh.part");
        std::fs::write(&fresh, b"new bytes")?;
        let stored = store
            .put_file(&fresh, ArtifactKind::Blob, "text/plain")
            .await?;
        assert_eq!(stored.id, blake3::hash(b"new bytes").to_hex().as_str());
        let path = store.local_path(&stored.id).expect("a local store");
        assert_eq!(std::fs::read(&path)?, b"new bytes");
        assert!(!fresh.exists());
        Ok(())
    }

    #[tokio::test]
    async fn deleting_removes_the_file_once() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let store = LocalStore::new(dir.path());
        let artifact = store
            .put(b"x".to_vec(), ArtifactKind::Blob, "text/plain")
            .await?;
        store.delete(&artifact.id).await?;
        assert!(store.get(&artifact.id).await.is_err());
        store.delete(&artifact.id).await?;
        assert!(store.delete("../../etc/passwd").await.is_err());
        assert_eq!(store.local_path("not-an-id"), None);
        Ok(())
    }
}
