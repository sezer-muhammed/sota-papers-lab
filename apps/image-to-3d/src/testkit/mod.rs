//! Test harness shared by every kernel owner. Owner: trainer (E7); add
//! helpers through the owner, keep stage-specific test code in your module.
//!
//! - [`npz`]: NumPy `.npz` read/write (the gs-oracle exchange format).
//! - [`oracle`]: calling `gs-oracle` directly or staging cases for it.
//! - [`tol`]: comparison statistics and the spec-raster 6.4 tolerances.
//! - [`synthetic`]: deterministic tiny scenes and their oracle inputs.
//! - [`timing`]: CUDA-event timers, benchmarks, nvidia-smi snapshots.
//! - The NVRTC compile cache is `crate::gpu` (per process, per variant).
//!
//! GPU tests are `#[ignore]` and run on the GPU host with
//! `apps/image-to-3d/scripts/gpu-test.sh <checkout> [filter]`, which mounts
//! the datasets read-only at `/datasets` and `~/spl-work/out` at `/out`.

pub mod npz;
pub mod oracle;
pub mod synthetic;
pub mod timing;
pub mod tol;

use std::path::PathBuf;

/// Where tests write artifacts: `/out/image-to-3d/<sub>` in the GPU container
/// (= `~/spl-work/out/image-to-3d/<sub>` on the host), else
/// `$SPL_TEST_OUT/<sub>`, else the system temp dir.
pub fn out_dir(sub: &str) -> PathBuf {
    let base = if let Some(p) = std::env::var_os("SPL_TEST_OUT") {
        PathBuf::from(p)
    } else if std::path::Path::new("/out").is_dir() {
        PathBuf::from("/out/image-to-3d")
    } else {
        std::env::temp_dir().join("spl-image-to-3d")
    };
    base.join(sub)
}

/// The benchmark datasets root (`mipnerf360/`, `tandt/`, `db/`):
/// `$SPL_DATASETS`, else `/datasets` (GPU container), else `~/spl-work/datasets`.
pub fn datasets_dir() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(p) = std::env::var_os("SPL_DATASETS") {
        candidates.push(PathBuf::from(p));
    }
    candidates.push(PathBuf::from("/datasets"));
    if let Some(home) = std::env::var_os("HOME") {
        candidates.push(PathBuf::from(home).join("spl-work/datasets"));
    }
    candidates.into_iter().find(|p| p.is_dir())
}

/// A Mip-NeRF 360 scene directory (e.g. "garden"), when the datasets are present.
pub fn mip360_scene(name: &str) -> Option<PathBuf> {
    datasets_dir()
        .map(|d| d.join("mipnerf360").join(name))
        .filter(|p| p.join("sparse").is_dir())
}
