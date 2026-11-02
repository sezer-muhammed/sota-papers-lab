//! `image_to_3d.pose`: photos or video frames → cameras and a coloured
//! point cloud with the native VGGT-Ω engine ([`crate::vggt`]). Owner:
//! runtime.
//!
//! Outputs: `cameras.json` (w2c OpenCV extrinsics, intrinsics, pose
//! encodings), a COLMAP text model (zip: `cameras.txt`, `images.txt`,
//! `points3D.txt`) and the point cloud as an INRIA-layout PLY of small
//! isotropic Gaussians (degree-0 colour), which splat viewers display; the
//! point cloud is announced in slot `points` as soon as it exists, the
//! cameras in slot `cameras`. GPU-exclusive: the engine takes most of the
//! card.
//!
//! The engine is loaded per run and dropped when it ends (its memory goes
//! back to the driver); frames per pass follow the free memory, and the run
//! waits when other processes leave too little ([`crate::vram`]).
//! Weights: `$SPL_VGGT_WEIGHTS`, else
//! `<data dir>/models/vggt-omega/vggt_omega_1b_512.safetensors` (converted
//! by `scripts/convert_vggt.py`). The VGGT-Ω weights are under the FAIR
//! non-commercial research licence: research use only.

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::ensure;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use spl_core::media;
use spl_core::{ArtifactKind, ArtifactRef, Context, DataDir, GpuUse, Op, Result, RunCx, anyhow};
use ts_rs::TS;

use crate::io::images::Rgb8Image;
use crate::reconstruct::pipeline::{EngineOpts, Event, Frames, pose_frames};
use crate::vggt::{DenseHost, Prepared, ResizeMode, VggtCameras, VggtEngine};

fn default_max_frames() -> u32 {
    64
}

fn default_conf_percentile() -> f32 {
    50.0
}

fn default_max_points() -> u32 {
    2_000_000
}

#[derive(Debug, Clone, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct PoseInput {
    /// Artifact ids of photos (JPEG, PNG, WebP), in capture order; the
    /// first is the reference frame (world = its camera).
    #[serde(default)]
    pub images: Vec<String>,
    /// Artifact id of a zip of photos (sorted by name).
    #[serde(default)]
    #[ts(optional)]
    pub zip: Option<String>,
    /// Artifact id of a video (MP4, MOV, WebM); frames are sampled evenly
    /// (needs `ffmpeg` on the host).
    #[serde(default)]
    #[ts(optional)]
    pub video: Option<String>,
    /// A folder of photos under the host's dataset root (`$SPL_DATASETS`),
    /// e.g. "tandt/truck/images".
    #[serde(default)]
    #[ts(optional)]
    pub dataset: Option<String>,
    /// At most this many frames, sampled evenly.
    #[serde(default = "default_max_frames")]
    pub max_frames: u32,
    #[serde(default)]
    pub resize: ResizeMode,
    /// Keep points whose depth confidence is above this percentile.
    #[serde(default = "default_conf_percentile")]
    pub conf_percentile: f32,
    #[serde(default = "default_max_points")]
    pub max_points: u32,
}

#[derive(Debug, Clone, Serialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct PoseTimes {
    pub decode_s: f64,
    pub load_s: f64,
    /// GPU time of the forward pass (CUDA events).
    pub forward_ms: f32,
    pub trunk_ms: f32,
    pub heads_ms: f32,
    pub wall_s: f64,
}

#[derive(Debug, Serialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct PoseOutput {
    pub frames: u32,
    /// Model resolution (pixels of the preprocessed frames).
    pub width: u32,
    pub height: u32,
    pub cameras: ArtifactRef,
    pub colmap: ArtifactRef,
    pub points: ArtifactRef,
    pub point_count: u32,
    pub times: PoseTimes,
    /// What the run gave up because other processes held GPU memory
    /// (fewer frames per engine pass); empty on the quality path.
    pub gpu_memory: Vec<String>,
    /// Seconds spent waiting for GPU memory.
    pub gpu_wait_s: f64,
}

pub struct Pose;

impl Op for Pose {
    const ID: &'static str = "image_to_3d.pose";
    const SUMMARY: &'static str =
        "Cameras and a coloured point cloud from photos or a video (native VGGT-Omega)";
    const GPU: GpuUse = GpuUse::Exclusive;

    type Input = PoseInput;
    type Output = PoseOutput;

    async fn run(&self, cx: &RunCx, input: PoseInput) -> Result<PoseOutput> {
        let started = std::time::Instant::now();
        ensure!(input.max_frames >= 1, "max_frames must be at least 1");
        cx.progress(0.0, "reading frames");
        let (names, frames) = load_frames(cx, &input).await?;
        let decode_s = started.elapsed().as_secs_f64();
        cx.log(format!(
            "{} frames decoded in {decode_s:.2} s",
            frames.len()
        ));

        let weights = weights_path(cx)?;
        let cancel = cx.cancel.clone();
        let mut opts = EngineOpts::new(false, frames.len());
        opts.resize = input.resize;
        opts.span = (0.05, 0.9);
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
        let job = tokio::task::spawn_blocking(move || -> Result<_> {
            let frames = Frames {
                names: names.clone(),
                engine: frames,
                trainer: Vec::new(),
            };
            let mut send = |e: Event| {
                let _ = tx.send(e);
            };
            let r = pose_frames(&weights, &frames, &opts, &cancel, &mut send)?;
            Ok((names, r))
        });
        let forward = async {
            while let Some(e) = rx.recv().await {
                match e {
                    Event::Progress(f, msg) => cx.progress(f, msg),
                    Event::Waiting(f, msg, data) => cx.progress_with(f, msg, data),
                    Event::Log(msg) => cx.log(msg),
                    _ => {}
                }
            }
        };
        let (res, ()) = tokio::join!(job, forward);
        let (names, r) = res??;
        let (prepared, cams, dense, times, load_s) = (
            r.posed.prepared,
            r.posed.cams,
            r.posed.dense,
            r.times,
            r.load_s,
        );
        let coexist = r.mem;
        cx.metric("frames", cams.pose_enc.len() as f64, "");
        cx.metric("forward_time", times.total_ms as f64, "ms");
        cx.metric("trunk_time", times.trunk_ms as f64, "ms");

        cx.progress(0.9, "writing outputs");
        let cameras_json = serde_json::to_vec_pretty(&cameras_doc(&names, &cams))?;
        let cameras = cx
            .artifacts
            .put(cameras_json, ArtifactKind::Cameras, media::JSON)
            .await?;
        cx.announce(
            "cameras",
            &cameras,
            serde_json::json!({ "frames": names.len() }),
        );

        let cp = input.conf_percentile;
        let mp = input.max_points as usize;
        let frames_out = cams.pose_enc.len() as u32;
        let (width, height) = (cams.width as u32, cams.height as u32);
        let (ply, colmap, count) = tokio::task::spawn_blocking(move || -> Result<_> {
            let pts = point_cloud(&prepared, &dense, cp, mp);
            let mut ply = Vec::new();
            write_points_ply(&pts, &cams, &mut ply)?;
            let colmap = colmap_text_zip(&names, &cams, &pts)?;
            Ok((ply, colmap, pts.len()))
        })
        .await??;
        let points = cx
            .artifacts
            .put(ply, ArtifactKind::Splat, media::PLY)
            .await?;
        cx.announce("points", &points, serde_json::json!({ "points": count }));
        let colmap = cx
            .artifacts
            .put(colmap, ArtifactKind::Cameras, media::ZIP)
            .await?;
        cx.progress(1.0, "done");
        Ok(PoseOutput {
            frames: frames_out,
            width,
            height,
            cameras,
            colmap,
            points,
            point_count: count as u32,
            times: PoseTimes {
                decode_s,
                load_s,
                forward_ms: times.total_ms,
                trunk_ms: times.trunk_ms,
                heads_ms: times.camera_ms + times.dense_ms,
                wall_s: started.elapsed().as_secs_f64(),
            },
            gpu_memory: coexist.notes,
            gpu_wait_s: coexist.waited_s,
        })
    }
}

fn weights_path(cx: &RunCx) -> Result<PathBuf> {
    let p = match std::env::var_os("SPL_VGGT_WEIGHTS") {
        Some(p) => PathBuf::from(p),
        None => VggtEngine::default_weights(
            &cx.resources
                .get::<DataDir>()
                .map(|d| d.0.clone())
                .ok_or_else(|| anyhow!("this host has no data directory"))?
                .join("models")
                .join("vggt-omega"),
        ),
    };
    ensure!(
        p.is_file(),
        "VGGT-Omega weights not found at {} (set SPL_VGGT_WEIGHTS; convert with apps/image-to-3d/scripts/convert_vggt.py)",
        p.display()
    );
    Ok(p)
}

fn is_artifact_id(id: &str) -> bool {
    !id.is_empty() && id.chars().all(|c| c.is_ascii_alphanumeric())
}

/// Evenly spaced indices: `n` of `0..len`, first and last included.
pub fn sample_indices(len: usize, n: usize) -> Vec<usize> {
    if len <= n {
        return (0..len).collect();
    }
    if n == 1 {
        return vec![0];
    }
    (0..n)
        .map(|i| (i * (len - 1) + (n - 1) / 2) / (n - 1))
        .collect()
}

fn is_image_name(name: &str) -> bool {
    let l = name.to_ascii_lowercase();
    [".jpg", ".jpeg", ".png", ".webp"]
        .iter()
        .any(|e| l.ends_with(e))
}

async fn load_frames(cx: &RunCx, input: &PoseInput) -> Result<(Vec<String>, Vec<Rgb8Image>)> {
    let given = [
        !input.images.is_empty(),
        input.zip.is_some(),
        input.video.is_some(),
        input.dataset.is_some(),
    ];
    ensure!(
        given.iter().filter(|&&g| g).count() == 1,
        "give exactly one of `images`, `zip`, `video` and `dataset`"
    );
    let max = input.max_frames as usize;
    let mut named: Vec<(String, Vec<u8>)> = Vec::new();
    if !input.images.is_empty() {
        for i in sample_indices(input.images.len(), max) {
            let id = &input.images[i];
            ensure!(is_artifact_id(id), "images must be artifact ids");
            named.push((id.clone(), cx.artifacts.get(id).await?));
        }
    } else if let Some(id) = &input.zip {
        ensure!(is_artifact_id(id), "zip must be an artifact id");
        let bytes = cx.artifacts.get(id).await?;
        named = tokio::task::spawn_blocking(move || -> Result<_> {
            let mut z = zip::ZipArchive::new(std::io::Cursor::new(bytes))
                .context("the upload is not a zip file")?;
            let mut names: Vec<String> = z
                .file_names()
                .filter_map(|n| n.ok().map(|n| n.into_owned()))
                .filter(|n| is_image_name(n) && !n.starts_with("__MACOSX"))
                .collect();
            names.sort();
            let mut out = Vec::new();
            for i in sample_indices(names.len(), max) {
                let mut f = z.by_name(&names[i])?;
                let mut b = Vec::with_capacity(f.size() as usize);
                std::io::Read::read_to_end(&mut f, &mut b)?;
                out.push((names[i].clone(), b));
            }
            Ok(out)
        })
        .await??;
    } else if let Some(name) = &input.dataset {
        let base = match std::env::var_os("SPL_DATASETS") {
            Some(p) => PathBuf::from(p),
            None => cx
                .resources
                .get::<DataDir>()
                .map(|d| d.0.join("datasets"))
                .ok_or_else(|| anyhow!("this host has no data directory"))?,
        };
        ensure!(
            Path::new(name)
                .components()
                .all(|c| matches!(c, std::path::Component::Normal(_))),
            "dataset must be a relative path"
        );
        let dir = base.join(name);
        let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
            .with_context(|| format!("read {}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(is_image_name)
            })
            .collect();
        files.sort();
        for i in sample_indices(files.len(), max) {
            let n = files[i]
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string();
            named.push((n, std::fs::read(&files[i])?));
        }
    } else if let Some(id) = &input.video {
        ensure!(is_artifact_id(id), "video must be an artifact id");
        let path = match cx.artifacts.local_path(id).filter(|p| p.is_file()) {
            Some(p) => (p, None),
            None => {
                let tmp = tempfile::NamedTempFile::new()?;
                std::fs::write(tmp.path(), cx.artifacts.get(id).await?)?;
                (tmp.path().to_path_buf(), Some(tmp))
            }
        };
        named = tokio::task::spawn_blocking(move || {
            let r = video_frames(&path.0, max);
            drop(path.1);
            r
        })
        .await??;
    }
    ensure!(!named.is_empty(), "no frames found");
    let (names, bytes): (Vec<_>, Vec<_>) = named.into_iter().unzip();
    let frames = tokio::task::spawn_blocking(move || -> Result<Vec<Rgb8Image>> {
        use rayon::prelude::*;
        bytes
            .par_iter()
            .map(|b| crate::vggt::preprocess::decode(b))
            .collect()
    })
    .await??;
    Ok((names, frames))
}

/// Up to `n` evenly spaced frames of a video, as PNG bytes, via ffmpeg.
fn video_frames(path: &Path, n: usize) -> Result<Vec<(String, Vec<u8>)>> {
    let probe = std::process::Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", "-count_packets"])
        .args(["-show_entries", "stream=nb_read_packets", "-of", "csv=p=0"])
        .arg(path)
        .output()
        .context("run ffprobe (is ffmpeg installed on the host?)")?;
    ensure!(
        probe.status.success(),
        "ffprobe failed: {}",
        String::from_utf8_lossy(&probe.stderr)
    );
    let total: usize = String::from_utf8_lossy(&probe.stdout)
        .trim()
        .trim_end_matches(',')
        .parse()
        .context("ffprobe frame count")?;
    let keep = sample_indices(total, n);
    let expr = keep
        .iter()
        .map(|i| format!("eq(n\\,{i})"))
        .collect::<Vec<_>>()
        .join("+");
    let dir = tempfile::tempdir()?;
    let status = std::process::Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-vf", &format!("select={expr}"), "-vsync", "vfr"])
        .arg(dir.path().join("f%05d.png"))
        .status()
        .context("run ffmpeg")?;
    ensure!(status.success(), "ffmpeg failed");
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir.path())?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();
    files.sort();
    files
        .iter()
        .map(|p| {
            Ok((
                p.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("")
                    .to_string(),
                std::fs::read(p)?,
            ))
        })
        .collect()
}

#[derive(Debug, Serialize)]
struct CameraDoc<'a> {
    convention: &'static str,
    width: usize,
    height: usize,
    frames: Vec<FrameDoc<'a>>,
}

#[derive(Debug, Serialize)]
struct FrameDoc<'a> {
    name: &'a str,
    /// World-to-camera 3×4, row-major.
    w2c: [[f32; 4]; 3],
    /// 3×3 intrinsics, preprocessed pixels.
    k: [[f32; 3]; 3],
    fov_y: f32,
    fov_x: f32,
    pose_enc: [f32; 9],
}

fn cameras_doc<'a>(names: &'a [String], c: &VggtCameras) -> CameraDoc<'a> {
    CameraDoc {
        convention: "opencv w2c (x right, y down, z forward); world = frame 0; model-scale units",
        width: c.width,
        height: c.height,
        frames: names
            .iter()
            .zip(c.extrinsics.iter().zip(&c.intrinsics).zip(&c.pose_enc))
            .map(|(name, ((e, k), pe))| FrameDoc {
                name,
                w2c: [
                    [e[0], e[1], e[2], e[3]],
                    [e[4], e[5], e[6], e[7]],
                    [e[8], e[9], e[10], e[11]],
                ],
                k: [[k[0], k[1], k[2]], [k[3], k[4], k[5]], [k[6], k[7], k[8]]],
                fov_y: pe[7],
                fov_x: pe[8],
                pose_enc: *pe,
            })
            .collect(),
    }
}

/// One coloured point; `size` is the footprint of one pixel at its depth.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub xyz: [f32; 3],
    pub rgb: [u8; 3],
    pub size: f32,
}

/// World points of every pixel whose confidence is above the
/// `percentile`-th percentile, thinned by a regular stride to at most
/// `max_points`.
pub fn point_cloud(p: &Prepared, d: &DenseHost, percentile: f32, max_points: usize) -> Vec<Point> {
    let g = p.geom;
    let n = g.frames * g.pixels();
    if n == 0 || d.conf.len() != n || d.world.len() != n * 3 {
        return Vec::new();
    }
    let mut sorted: Vec<f32> = d.conf.iter().copied().filter(|c| c.is_finite()).collect();
    sorted.sort_by(f32::total_cmp);
    let q = (percentile.clamp(0.0, 100.0) / 100.0 * (sorted.len().max(1) - 1) as f32) as usize;
    let thr = sorted.get(q).copied().unwrap_or(f32::INFINITY);
    let kept = d.conf.iter().filter(|&&c| c >= thr).count();
    let stride = kept.div_ceil(max_points.max(1)).max(1);
    let mut out = Vec::with_capacity(kept / stride + 1);
    let mut seen = 0usize;
    for i in 0..n {
        if d.conf[i] < thr {
            continue;
        }
        seen += 1;
        if !(seen - 1).is_multiple_of(stride) {
            continue;
        }
        let xyz = [d.world[i * 3], d.world[i * 3 + 1], d.world[i * 3 + 2]];
        if !xyz.iter().all(|v| v.is_finite()) {
            continue;
        }
        let depth = d.depth.get(i).copied().unwrap_or(1.0);
        out.push(Point {
            xyz,
            rgb: [p.pixels[i * 3], p.pixels[i * 3 + 1], p.pixels[i * 3 + 2]],
            // One pixel at the image centre of a ~60° view: depth · 2·tan(30°) / W.
            size: depth * 1.1547 / g.width as f32 * (stride as f32).sqrt(),
        });
    }
    out
}

/// INRIA 3DGS PLY with degree-0 colour: one small isotropic Gaussian per
/// point.
pub fn write_points_ply(pts: &[Point], _c: &VggtCameras, out: &mut dyn Write) -> Result<()> {
    const SH_C0: f32 = 0.282_094_8;
    let props = [
        "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0",
        "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    ];
    let mut h = format!(
        "ply\nformat binary_little_endian 1.0\nelement vertex {}\n",
        pts.len()
    );
    for p in props {
        h.push_str(&format!("property float {p}\n"));
    }
    h.push_str("end_header\n");
    out.write_all(h.as_bytes())?;
    let mut buf = Vec::with_capacity(pts.len() * props.len() * 4);
    for p in pts {
        let s = p.size.max(1e-6).ln();
        let dc = p.rgb.map(|v| (v as f32 / 255.0 - 0.5) / SH_C0);
        let vals = [
            p.xyz[0], p.xyz[1], p.xyz[2], 0.0, 0.0, 0.0, dc[0], dc[1], dc[2], 4.0, s, s, s, 1.0,
            0.0, 0.0, 0.0,
        ];
        for v in vals {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    out.write_all(&buf)?;
    Ok(())
}

/// Rotation matrix (row-major) → quaternion `[w, x, y, z]`.
pub fn mat_to_quat(r: [[f32; 3]; 3]) -> [f64; 4] {
    let m = r.map(|row| row.map(|v| v as f64));
    let t = m[0][0] + m[1][1] + m[2][2];
    let q = if t > 0.0 {
        let s = (t + 1.0).sqrt() * 2.0;
        [
            0.25 * s,
            (m[2][1] - m[1][2]) / s,
            (m[0][2] - m[2][0]) / s,
            (m[1][0] - m[0][1]) / s,
        ]
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
        [
            (m[2][1] - m[1][2]) / s,
            0.25 * s,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s,
        ]
    } else if m[1][1] > m[2][2] {
        let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
        [
            (m[0][2] - m[2][0]) / s,
            (m[0][1] + m[1][0]) / s,
            0.25 * s,
            (m[1][2] + m[2][1]) / s,
        ]
    } else {
        let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
        [
            (m[1][0] - m[0][1]) / s,
            (m[0][2] + m[2][0]) / s,
            (m[1][2] + m[2][1]) / s,
            0.25 * s,
        ]
    };
    let n = q.iter().map(|v| v * v).sum::<f64>().sqrt();
    let q = q.map(|v| v / n);
    if q[0] < 0.0 { q.map(|v| -v) } else { q }
}

/// COLMAP text model: one PINHOLE camera per frame, poses, points without
/// tracks.
pub fn colmap_text(names: &[String], c: &VggtCameras, pts: &[Point]) -> (String, String, String) {
    let mut cameras = String::from(
        "# Camera list with one line of data per camera:\n#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n",
    );
    let mut images = String::from(
        "# Image list with two lines of data per image:\n#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n#   POINTS2D[] as (X, Y, POINT3D_ID)\n",
    );
    for (i, ((e, k), name)) in c
        .extrinsics
        .iter()
        .zip(&c.intrinsics)
        .zip(names)
        .enumerate()
    {
        let id = i + 1;
        cameras.push_str(&format!(
            "{id} PINHOLE {} {} {} {} {} {}\n",
            c.width, c.height, k[0], k[4], k[2], k[5]
        ));
        let q = mat_to_quat([[e[0], e[1], e[2]], [e[4], e[5], e[6]], [e[8], e[9], e[10]]]);
        let name = name.replace(char::is_whitespace, "_");
        images.push_str(&format!(
            "{id} {} {} {} {} {} {} {} {id} {name}\n\n",
            q[0], q[1], q[2], q[3], e[3], e[7], e[11]
        ));
    }
    let mut points = String::with_capacity(pts.len() * 48);
    points.push_str("# 3D point list with one line of data per point:\n#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n");
    for (i, p) in pts.iter().enumerate() {
        points.push_str(&format!(
            "{} {} {} {} {} {} {} 0\n",
            i + 1,
            p.xyz[0],
            p.xyz[1],
            p.xyz[2],
            p.rgb[0],
            p.rgb[1],
            p.rgb[2]
        ));
    }
    (cameras, images, points)
}

fn colmap_text_zip(names: &[String], c: &VggtCameras, pts: &[Point]) -> Result<Vec<u8>> {
    let (cams, imgs, points) = colmap_text(names, c, pts);
    let mut z = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    let opts = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    for (n, s) in [
        ("sparse/0/cameras.txt", cams),
        ("sparse/0/images.txt", imgs),
        ("sparse/0/points3D.txt", points),
    ] {
        z.start_file(n, opts)?;
        z.write_all(s.as_bytes())?;
    }
    Ok(z.finish()?.into_inner())
}

/// Throws away a resident engine (tests, or to free the GPU).
pub fn unload_engine() {
    crate::reconstruct::pipeline::unload_engine();
}
