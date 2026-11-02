//! The gsplat oracle (`gs-oracle`, built by bench-infra; schemas in
//! benchmarks/image-to-3d/ORACLE.md): run gsplat's own CUDA ops and fused-ssim
//! on inputs our tests save as `.npz`.
//!
//! `gs-oracle` starts a Docker container, so it runs on the GPU host, not
//! inside the CUDA test container. Two ways to use it:
//!
//! - **Direct** (a test running natively on the host): [`Oracle::find`] then
//!   [`Oracle::run`] (one op) or [`Oracle::run_batch`].
//! - **Staged** (a GPU test inside the container): [`stage`] writes
//!   `<dir>/<name>.<op>.in.npz` and [`stage_ours`] our result next to it under
//!   `/out/image-to-3d/oracle/`; afterwards on the host
//!   `gs-oracle batch ~/spl-work/out/image-to-3d/oracle` writes the
//!   `*.out.npz` files, and `gs-oracle compare` (or a test calling
//!   [`load_out`]) checks them.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::ensure;
use spl_core::{Context, Result};

use super::npz::Npz;

/// A located `gs-oracle` executable.
#[derive(Clone, Debug)]
pub struct Oracle {
    pub bin: PathBuf,
}

impl Oracle {
    /// `$SPL_GS_ORACLE`, `~/spl-work/bin/gs-oracle`, or `gs-oracle` on `PATH`.
    pub fn find() -> Option<Self> {
        let mut candidates = Vec::new();
        if let Some(p) = std::env::var_os("SPL_GS_ORACLE") {
            candidates.push(PathBuf::from(p));
        }
        if let Some(home) = std::env::var_os("HOME") {
            candidates.push(Path::new(&home).join("spl-work/bin/gs-oracle"));
        }
        if let Some(path) = std::env::var_os("PATH") {
            candidates.extend(std::env::split_paths(&path).map(|d| d.join("gs-oracle")));
        }
        candidates
            .into_iter()
            .find(|p| p.is_file())
            .map(|bin| Self { bin })
    }

    fn call(&self, args: &[&std::ffi::OsStr]) -> Result<String> {
        let out = Command::new(&self.bin)
            .args(args)
            .output()
            .with_context(|| format!("run {}", self.bin.display()))?;
        ensure!(
            out.status.success(),
            "gs-oracle {:?} failed ({}): {}",
            args,
            out.status,
            String::from_utf8_lossy(&out.stderr)
        );
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    }

    /// Runs one op on `input` and returns its outputs (about 3 s of start-up).
    pub fn run(&self, op: &str, input: &Npz) -> Result<Npz> {
        let dir = tempfile::tempdir().context("temp dir for gs-oracle")?;
        let (inp, out) = (dir.path().join("in.npz"), dir.path().join("out.npz"));
        input.write(&inp)?;
        self.call(&[op.as_ref(), inp.as_os_str(), out.as_os_str()])?;
        Npz::read(&out)
    }

    /// `gs-oracle batch dir`: every `<name>.<op>.in.npz` → `<name>.<op>.out.npz`.
    pub fn run_batch(&self, dir: &Path) -> Result<()> {
        self.call(&["batch".as_ref(), dir.as_os_str()]).map(drop)
    }

    /// `gs-oracle compare a b`: the per-key error table.
    pub fn compare(&self, ours: &Path, reference: &Path) -> Result<String> {
        self.call(&["compare".as_ref(), ours.as_os_str(), reference.as_os_str()])
    }
}

/// Where staged oracle cases go (`/out/image-to-3d/oracle` on the GPU host).
pub fn stage_dir() -> PathBuf {
    super::out_dir("oracle")
}

pub fn case_path(dir: &Path, name: &str, op: &str, kind: &str) -> PathBuf {
    dir.join(format!("{name}.{op}.{kind}.npz"))
}

/// Writes `<dir>/<name>.<op>.in.npz` for a later `gs-oracle batch`.
pub fn stage(dir: &Path, name: &str, op: &str, input: &Npz) -> Result<PathBuf> {
    let p = case_path(dir, name, op, "in");
    input.write(&p)?;
    Ok(p)
}

/// Writes our own result as `<dir>/<name>.<op>.ours.npz` (for `gs-oracle compare`).
pub fn stage_ours(dir: &Path, name: &str, op: &str, ours: &Npz) -> Result<PathBuf> {
    let p = case_path(dir, name, op, "ours");
    ours.write(&p)?;
    Ok(p)
}

/// Reads `<dir>/<name>.<op>.out.npz` written by `gs-oracle batch`.
pub fn load_out(dir: &Path, name: &str, op: &str) -> Result<Npz> {
    Npz::read(&case_path(dir, name, op, "out"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staged_cases_follow_the_batch_naming() {
        let dir = tempfile::tempdir().unwrap();
        let mut z = Npz::new();
        z.i32_scalar("degree", 2).unwrap();
        let p = stage(dir.path(), "tiny", "sh", &z).unwrap();
        assert!(p.ends_with("tiny.sh.in.npz"));
        assert_eq!(Npz::read(&p).unwrap(), z);
        assert!(
            load_out(dir.path(), "tiny", "sh").is_err(),
            "no oracle output yet"
        );
    }

    /// Runs only where gs-oracle is installed (the GPU host, natively).
    #[test]
    #[ignore]
    fn oracle_ssim_matches_its_own_float64_reference() {
        let Some(oracle) = Oracle::find() else {
            eprintln!("skipped: gs-oracle not found");
            return;
        };
        let (h, w) = (37usize, 29usize);
        let p = crate::rng::Philox::new(5);
        let img = |k: u32| {
            (0..3 * h * w)
                .map(|i| p.uniform(i as u32, k, 0))
                .collect::<Vec<f32>>()
        };
        let mut z = Npz::new();
        z.f32("img1", &[1, 3, h, w], img(1)).unwrap();
        z.f32("img2", &[1, 3, h, w], img(2)).unwrap();
        let out = oracle.run("ssim", &z).unwrap();
        let ssim = out.get("ssim").unwrap().to_f64()[0];
        let reference = out.get("ssim_ref").unwrap().to_f64()[0];
        assert!((ssim - reference).abs() < 1e-5, "{ssim} vs {reference}");
    }
}
