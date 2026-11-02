//! Docker sidecar for Pixal3D-mv: runs `run_mv.py` inside the
//! `spl-pixal3d` container with GPU access.

use std::io::{BufRead, BufReader, Read as IoRead};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, ensure};
use serde::Deserialize;
use spl_core::{CancellationToken, Result};

const IMAGE: &str = "spl-pixal3d";

/// Progress report from the sidecar's stdout JSON lines.
#[derive(Debug, Deserialize)]
pub struct SidecarProgress {
    pub stage: String,
    pub fraction: f32,
    #[serde(default)]
    pub tokens: Option<u32>,
    #[serde(default)]
    pub vram_mb: Option<f32>,
}

/// Final result from the sidecar.
#[derive(Debug, Deserialize)]
pub struct SidecarResult {
    pub glb_path: String,
    pub stages: Vec<StageTime>,
    pub peak_vram_mb: f32,
    pub total_s: f64,
}

#[derive(Debug, Deserialize)]
pub struct StageTime {
    pub name: String,
    pub seconds: f64,
}

/// Run the Pixal3D-mv sidecar container.
///
/// - `work_dir`: host directory with `transforms.json` and RGBA PNGs.
/// - `weights_dir`: host directory with the bf16 weights.
/// - `output_dir`: host directory where the GLB is written.
/// - `resolution`: 512 or 1024.
/// - `max_tokens`: sparse-attention token budget (lower = less VRAM).
/// - `cancel`: cancellation flag.
/// - `on_progress`: callback for each progress line.
pub fn run_sidecar(
    work_dir: &Path,
    weights_dir: &Path,
    output_dir: &Path,
    resolution: u32,
    max_tokens: u32,
    cancel: &CancellationToken,
    mut on_progress: impl FnMut(SidecarProgress),
) -> Result<SidecarResult> {
    ensure!(
        work_dir.join("transforms.json").is_file(),
        "transforms.json not found in {}",
        work_dir.display()
    );

    std::fs::create_dir_all(output_dir)?;

    let mut cmd = Command::new("docker");
    cmd.args(["run", "--rm", "--gpus", "all"])
        .args(["--shm-size", "4g"])
        .arg("-v")
        .arg(format!("{}:/input:ro", work_dir.display()))
        .arg("-v")
        .arg(format!("{}:/weights:ro", weights_dir.display()))
        .arg("-v")
        .arg(format!("{}:/output", output_dir.display()))
        .arg(IMAGE)
        .args([
            "python",
            "/opt/spl-p3d/run_mv.py",
            "--input_dir",
            "/input",
            "--weights_dir",
            "/weights",
            "--output_dir",
            "/output",
            "--resolution",
            &resolution.to_string(),
            "--low_vram",
            "--max_tokens",
            &max_tokens.to_string(),
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd.spawn().context("spawn docker (is Docker running?)")?;
    let stdout = child.stdout.take().unwrap();
    let reader = BufReader::new(stdout);

    let mut result_json: Option<String> = None;

    for line in reader.lines() {
        if cancel.is_cancelled() {
            let _ = child.kill();
            anyhow::bail!("cancelled");
        }
        let line = line.context("read sidecar stdout")?;
        if line.starts_with("{\"stage\"") {
            if let Ok(p) = serde_json::from_str::<SidecarProgress>(&line) {
                on_progress(p);
            }
        } else if line.starts_with("{\"glb_path\"") {
            result_json = Some(line);
        }
    }

    let status = child.wait().context("wait for sidecar")?;
    let stderr = child
        .stderr
        .take()
        .map(|mut s| {
            let mut buf = String::new();
            IoRead::read_to_string(&mut s, &mut buf).ok();
            buf
        })
        .unwrap_or_default();

    ensure!(status.success(), "sidecar exited with {status}:\n{stderr}");

    let json_str = result_json.context("sidecar produced no result JSON")?;
    let result: SidecarResult = serde_json::from_str(&json_str).context("parse sidecar result")?;

    ensure!(
        Path::new(&result.glb_path).is_file() || output_dir.join("output.glb").is_file(),
        "sidecar did not produce a GLB file"
    );

    Ok(result)
}

/// Check whether the spl-pixal3d image exists locally.
pub fn image_exists() -> bool {
    Command::new("docker")
        .args(["image", "inspect", IMAGE])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Whether a sidecar failure looks like an OOM.
pub fn is_oom(err: &anyhow::Error) -> bool {
    let msg = format!("{err:#}");
    msg.contains("CUDA out of memory")
        || msg.contains("OutOfMemoryError")
        || msg.contains("torch.OutOfMemoryError")
}

/// Path where weights are expected on the host.
pub fn default_weights_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
    PathBuf::from(home)
        .join("spl-work")
        .join("models")
        .join("pixal3d")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_oom_detects_torch_messages() {
        let e = anyhow::anyhow!(
            "sidecar exited with 1:\nRuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB"
        );
        assert!(is_oom(&e));

        let e = anyhow::anyhow!("torch.OutOfMemoryError: ...");
        assert!(is_oom(&e));

        let e = anyhow::anyhow!("OutOfMemoryError: ...");
        assert!(is_oom(&e));
    }

    #[test]
    fn is_oom_rejects_other_errors() {
        let e = anyhow::anyhow!("FileNotFoundError: weights not found");
        assert!(!is_oom(&e));

        let e = anyhow::anyhow!("sidecar exited with 137 (killed)");
        assert!(!is_oom(&e));
    }

    #[test]
    fn default_weights_dir_under_spl_work() {
        let d = default_weights_dir();
        assert!(d.to_str().unwrap().contains("spl-work"));
        assert!(d.to_str().unwrap().ends_with("pixal3d"));
    }
}
