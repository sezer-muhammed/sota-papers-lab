//! Benchmark harness: run a suite of scenes, record metrics, import baselines
//! and paper claims. Owner: bench-op. `image_to_3d.bench`.
//!
//! LPIPS: computed by the spl-gs-ref sidecar (`gs-metrics`) on saved PNGs.
//! The harness calls it as a subprocess; will move to ONNX Runtime with
//! the VGG-LPIPS export once the ort integration lands.

use std::io::BufRead;
use std::path::{Path, PathBuf};

use anyhow::Context;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use spl_core::{GpuUse, Op, Result, RunCx, bail};
use ts_rs::TS;

use crate::eval::evaluate;
use crate::gpu::Gpu;
use crate::io::scene::{LoadOptions, load_colmap_scene};
use crate::presets::PresetName;
use crate::trainer::{TrainConfig, Trainer};

// ---------------------------------------------------------------------------
// Wire types (exported to TS)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct BenchInput {
    /// Suite TOML path relative to the repo root,
    /// e.g. "benchmarks/image-to-3d-nvs-colmap.toml".
    #[serde(default)]
    #[ts(optional)]
    pub suite: Option<String>,
    /// Run only these scenes (by name). Empty = all scenes in the suite.
    #[serde(default)]
    pub scenes: Vec<String>,
    /// Training preset override (e.g. "quality", "fast").
    #[serde(default)]
    #[ts(optional)]
    pub preset: Option<PresetName>,
    /// Path to baseline results JSONL.
    #[serde(default)]
    #[ts(optional)]
    pub baselines_jsonl: Option<String>,
    /// Path to paper claims TOML.
    #[serde(default)]
    #[ts(optional)]
    pub paper_toml: Option<String>,
}

#[derive(Debug, Serialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct BenchOutput {
    pub results: Vec<SceneResult>,
    pub baselines: Vec<BaselineRow>,
    pub paper_claims: Vec<PaperClaim>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct SceneResult {
    pub scene: String,
    pub dataset: String,
    pub preset: String,
    pub psnr: f64,
    pub ssim: f64,
    #[ts(optional)]
    pub lpips_vgg: Option<f64>,
    pub train_time_s: f64,
    pub peak_vram_mib: f64,
    pub num_gaussians: u64,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct BaselineRow {
    pub tool: String,
    pub scene: String,
    pub dataset: String,
    #[ts(optional)]
    pub tag: Option<String>,
    pub status: String,
    #[ts(optional)]
    pub psnr: Option<f64>,
    #[ts(optional)]
    pub ssim: Option<f64>,
    #[ts(optional)]
    pub lpips: Option<f64>,
    #[ts(optional)]
    pub train_s: Option<f64>,
    #[ts(optional)]
    pub wall_s: Option<f64>,
    #[ts(optional)]
    pub peak_vram_mib: Option<f64>,
    #[ts(optional)]
    pub num_gaussians: Option<u64>,
    #[ts(optional)]
    pub gpu: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct PaperClaim {
    pub method: String,
    pub dataset: String,
    pub metric: String,
    pub value: f64,
    pub unit: String,
    #[ts(optional)]
    pub gpu: Option<String>,
}

// ---------------------------------------------------------------------------
// Suite TOML (internal)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct SuiteToml {
    #[allow(dead_code)]
    suite: SuiteMeta,
    protocol: Protocol,
    scenes: Vec<SceneEntry>,
    #[serde(default)]
    baselines: Option<BaselinesConfig>,
}

#[derive(Debug, Deserialize)]
struct SuiteMeta {
    #[allow(dead_code)]
    name: String,
    #[allow(dead_code)]
    description: String,
}

#[derive(Debug, Deserialize)]
struct Protocol {
    test_every: u32,
    #[allow(dead_code)]
    metrics: Vec<String>,
    #[serde(default)]
    preset: Option<PresetName>,
    #[serde(default)]
    datasets_toml: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SceneEntry {
    name: String,
    dataset: String,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    images_dir: Option<String>,
}

#[derive(Debug, Deserialize)]
struct BaselinesConfig {
    #[serde(default)]
    results_jsonl: Option<String>,
    #[serde(default)]
    paper_toml: Option<String>,
}

// ---------------------------------------------------------------------------
// Paper claims TOML (matches apps/image-to-3d/paper.toml)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct PaperToml {
    #[allow(dead_code)]
    paper: PaperMeta,
    #[serde(default)]
    claimed: Vec<ClaimedEntry>,
}

#[derive(Debug, Deserialize)]
struct PaperMeta {
    #[allow(dead_code)]
    title: String,
}

#[derive(Debug, Deserialize)]
struct ClaimedEntry {
    dataset: String,
    metric: String,
    value: f64,
    #[serde(default)]
    unit: String,
    at: ClaimedAt,
}

#[derive(Debug, Deserialize)]
struct ClaimedAt {
    method: String,
    #[serde(default)]
    gpu: Option<String>,
}

// ---------------------------------------------------------------------------
// datasets.toml scene manifest (for images_dir lookup)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct DatasetsToml {
    #[serde(default)]
    scenes: std::collections::HashMap<String, DatasetScene>,
}

#[derive(Debug, Deserialize)]
struct DatasetScene {
    #[serde(default)]
    images: Option<String>,
}

fn load_datasets_manifest(
    repo_root: &Path,
    path: Option<&str>,
) -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();
    let Some(p) = path else { return map };
    let abs = if Path::new(p).is_absolute() {
        PathBuf::from(p)
    } else {
        repo_root.join(p)
    };
    let Ok(text) = std::fs::read_to_string(&abs) else {
        return map;
    };
    if let Ok(doc) = toml::from_str::<DatasetsToml>(&text) {
        for (name, scene) in doc.scenes {
            if let Some(images) = scene.images {
                map.insert(name, images);
            }
        }
    }
    map
}

// ---------------------------------------------------------------------------
// Op
// ---------------------------------------------------------------------------

pub struct Bench;

impl Op for Bench {
    const ID: &'static str = "image_to_3d.bench";
    const SUMMARY: &'static str =
        "Run a benchmark suite: train each scene, record metrics, import baselines";
    const GPU: GpuUse = GpuUse::Exclusive;

    type Input = BenchInput;
    type Output = BenchOutput;

    async fn run(&self, cx: &RunCx, input: BenchInput) -> Result<BenchOutput> {
        let datasets_root = datasets_root();
        let repo_root = repo_root()?;

        // --- resolve suite scenes ---
        let (scenes, preset, test_every, baselines_cfg, datasets_manifest) =
            if let Some(ref suite_path) = input.suite {
                let abs = repo_root.join(suite_path);
                let text = std::fs::read_to_string(&abs)
                    .with_context(|| format!("read suite {}", abs.display()))?;
                let suite: SuiteToml = toml::from_str(&text)
                    .with_context(|| format!("parse suite {}", abs.display()))?;
                let preset = input.preset.or(suite.protocol.preset).unwrap_or_default();
                let manifest =
                    load_datasets_manifest(&repo_root, suite.protocol.datasets_toml.as_deref());
                (
                    suite.scenes,
                    preset,
                    suite.protocol.test_every,
                    suite.baselines,
                    manifest,
                )
            } else {
                bail!("suite is required (no inline scene list yet)");
            };

        let filtered: Vec<&SceneEntry> = if input.scenes.is_empty() {
            scenes.iter().collect()
        } else {
            scenes
                .iter()
                .filter(|s| input.scenes.iter().any(|f| f == &s.name))
                .collect()
        };

        if filtered.is_empty() {
            bail!("no scenes matched the filter");
        }

        let total = filtered.len();
        let mut results = Vec::with_capacity(total);
        let cancel = cx.cancel.clone();

        for (i, entry) in filtered.iter().enumerate() {
            cx.progress(
                i as f64 / total as f64,
                format!("[{}/{}] training {}", i + 1, total, entry.name),
            );
            cx.log(format!(
                "scene {}: dataset={}, preset={:?}",
                entry.name, entry.dataset, preset
            ));

            let scene_path = entry
                .path
                .as_deref()
                .map(|p| datasets_root.join(p))
                .unwrap_or_else(|| datasets_root.join(&entry.dataset).join(&entry.name));
            let scene_name = entry.name.clone();
            let dataset_name = entry.dataset.clone();
            let images_dir = entry
                .images_dir
                .clone()
                .or_else(|| datasets_manifest.get(&entry.name).cloned())
                .unwrap_or_else(|| "images".into());
            let load = LoadOptions {
                test_every,
                images_dir,
                ..Default::default()
            };
            let cfg = TrainConfig {
                preset,
                overrides: crate::presets::PresetOverrides {
                    scene: Some(entry.name.clone()),
                    ..Default::default()
                },
                ..Default::default()
            };
            let cancel2 = cancel.clone();

            let result = tokio::task::spawn_blocking(move || -> Result<SceneResult> {
                let scene = load_colmap_scene(&scene_path, &load)
                    .with_context(|| format!("load scene {scene_name}"))?;
                let gpu = Gpu::open()?;
                let mut trainer = Trainer::new(&gpu, scene, cfg)?;
                let summary = trainer.run(&cancel2, &mut |_| {})?;
                let eval_dir = tempfile::tempdir().context("eval temp dir")?;
                let eval = evaluate(&mut trainer, Some(eval_dir.path()))?;
                let (_, mut final_summary) = trainer.finish()?;
                final_summary.train_s = summary.train_s;

                Ok(SceneResult {
                    scene: scene_name,
                    dataset: dataset_name,
                    preset: format!("{preset:?}"),
                    psnr: eval.mean.psnr,
                    ssim: eval.mean.ssim,
                    lpips_vgg: eval.mean.lpips_vgg,
                    train_time_s: final_summary.train_s,
                    peak_vram_mib: final_summary.peak_vram_bytes as f64 / (1u64 << 20) as f64,
                    num_gaussians: final_summary.final_gaussians as u64,
                    status: "ok".into(),
                })
            })
            .await?;

            match result {
                Ok(r) => {
                    cx.metric(format!("{}.psnr", r.scene), r.psnr, "dB");
                    cx.metric(format!("{}.ssim", r.scene), r.ssim, "");
                    cx.metric(format!("{}.train_time", r.scene), r.train_time_s, "s");
                    cx.metric(format!("{}.peak_vram", r.scene), r.peak_vram_mib, "MiB");
                    cx.metric(
                        format!("{}.num_gaussians", r.scene),
                        r.num_gaussians as f64,
                        "",
                    );
                    if let Some(lpips) = r.lpips_vgg {
                        cx.metric(format!("{}.lpips_vgg", r.scene), lpips, "");
                    }
                    results.push(r);
                }
                Err(e) => {
                    cx.log(format!("scene {} failed: {e:#}", entry.name));
                    results.push(SceneResult {
                        scene: entry.name.clone(),
                        dataset: entry.dataset.clone(),
                        preset: format!("{preset:?}"),
                        psnr: 0.0,
                        ssim: 0.0,
                        lpips_vgg: None,
                        train_time_s: 0.0,
                        peak_vram_mib: 0.0,
                        num_gaussians: 0,
                        status: format!("{e:#}"),
                    });
                }
            }
        }

        cx.progress(1.0, "loading baselines and paper claims");

        // --- import baselines ---
        let baselines_path = input
            .baselines_jsonl
            .as_deref()
            .or_else(|| baselines_cfg.as_ref()?.results_jsonl.as_deref())
            .map(expand_home);
        let baselines = match baselines_path {
            Some(path) => load_baselines(&path).unwrap_or_else(|e| {
                cx.log(format!("baselines: {e:#}"));
                Vec::new()
            }),
            None => Vec::new(),
        };

        // --- import paper claims ---
        let paper_path = input
            .paper_toml
            .as_deref()
            .or_else(|| baselines_cfg.as_ref()?.paper_toml.as_deref());
        let paper_path = paper_path.map(|p| {
            if Path::new(p).is_absolute() {
                PathBuf::from(p)
            } else {
                repo_root.join(p)
            }
        });
        let paper_claims = match paper_path {
            Some(path) => load_paper_claims(&path).unwrap_or_else(|e| {
                cx.log(format!("paper claims: {e:#}"));
                Vec::new()
            }),
            None => Vec::new(),
        };

        Ok(BenchOutput {
            results,
            baselines,
            paper_claims,
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn datasets_root() -> PathBuf {
    std::env::var_os("SPL_DATASETS")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
            PathBuf::from(home).join("spl-work/datasets")
        })
}

fn repo_root() -> Result<PathBuf> {
    if let Ok(out) = std::process::Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()
    {
        let path = String::from_utf8(out.stdout)
            .unwrap_or_default()
            .trim()
            .to_owned();
        if !path.is_empty() {
            return Ok(PathBuf::from(path));
        }
    }
    std::env::current_dir().context("current directory as repo root fallback")
}

fn expand_home(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/") {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
        PathBuf::from(home).join(rest)
    } else {
        PathBuf::from(path)
    }
}

fn load_baselines(path: &Path) -> Result<Vec<BaselineRow>> {
    let file =
        std::fs::File::open(path).with_context(|| format!("open baselines {}", path.display()))?;
    let reader = std::io::BufReader::new(file);
    let mut rows = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match serde_json::from_str::<BaselineRow>(line) {
            Ok(row) => rows.push(row),
            Err(e) => tracing::warn!("skip baseline row: {e}"),
        }
    }
    Ok(rows)
}

fn load_paper_claims(path: &Path) -> Result<Vec<PaperClaim>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let doc: PaperToml =
        toml::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
    Ok(doc
        .claimed
        .into_iter()
        .map(|c| PaperClaim {
            method: c.at.method,
            dataset: c.dataset,
            metric: c.metric,
            value: c.value,
            unit: c.unit,
            gpu: c.at.gpu,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_paper_toml() {
        let toml_str = r#"
[paper]
title = "Test"

[[claimed]]
dataset = "mipnerf360"
metric = "psnr"
value = 27.56
unit = "dB"
[claimed.at]
method = "faster-gs"
gpu = "RTX 4090"
"#;
        let doc: PaperToml = toml::from_str(toml_str).unwrap();
        assert_eq!(doc.claimed.len(), 1);
        assert_eq!(doc.claimed[0].at.method, "faster-gs");
    }

    #[test]
    fn parse_suite_toml() {
        let toml_str = r#"
[suite]
name = "nvs-colmap"
description = "NVS on COLMAP-posed scenes"

[protocol]
test_every = 8
metrics = ["psnr", "ssim", "lpips_vgg", "train_time", "peak_vram", "num_gaussians"]

[[scenes]]
name = "garden"
dataset = "mipnerf360"

[[scenes]]
name = "bicycle"
dataset = "mipnerf360"

[baselines]
results_jsonl = "~/spl-work/bench/results.jsonl"
paper_toml = "apps/image-to-3d/paper.toml"
"#;
        let suite: SuiteToml = toml::from_str(toml_str).unwrap();
        assert_eq!(suite.suite.name, "nvs-colmap");
        assert_eq!(suite.scenes.len(), 2);
        assert_eq!(suite.protocol.test_every, 8);
        assert!(suite.baselines.is_some());
    }

    #[test]
    fn parse_baseline_jsonl_row() {
        let line = r#"{"tool":"fastgs","scene":"garden","dataset":"mipnerf360","tag":"quality","status":"ok","psnr":27.30,"ssim":0.854,"lpips":0.128,"train_s":265.0,"wall_s":280.0,"peak_vram_mib":8200,"num_gaussians":1200000,"gpu":"RTX 4070 SUPER"}"#;
        let row: BaselineRow = serde_json::from_str(line).unwrap();
        assert_eq!(row.tool, "fastgs");
        assert_eq!(row.psnr, Some(27.30));
    }

    #[test]
    fn expand_home_works() {
        let p = expand_home("~/foo/bar");
        assert!(p.to_str().unwrap().ends_with("/foo/bar"));
        assert!(!p.to_str().unwrap().starts_with("~"));

        let p2 = expand_home("/absolute/path");
        assert_eq!(p2, PathBuf::from("/absolute/path"));
    }
}
