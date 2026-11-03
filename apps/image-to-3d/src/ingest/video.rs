//! Video ingest: ffprobe, ffmpeg frame extraction as a cancellable child
//! process, GPU sharpness scoring and keyframe binning. Owner: ingest (A0).
//!
//! Pipeline: the async side reads packed RGB frames from ffmpeg's stdout and
//! hands them over a bounded channel to a blocking GPU worker, which scores
//! each frame and resizes the ones that become their bin's best. Frame
//! buffers flow back to the reader for reuse. Cancel (the token, or dropping
//! the future) kills ffmpeg: the child is spawned with `kill_on_drop`.

use std::path::Path;
use std::process::Stdio;
use std::sync::mpsc as std_mpsc;
use std::time::Instant;

use spl_core::{CancellationToken, Context, Result, anyhow, bail};
use tokio::io::{AsyncReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::mpsc;

use super::capture::CaptureStats;
use super::decode::resize_both;
use super::gpu::{IngestGpu, Resident};
use super::motion::{LumaThumb, Scored, Selection, frame_motion, select_keyframes};
use super::{
    IngestConfig, IngestFrame, IngestResult, IngestStats, KeyframeBins, KeyframeMode, Rgb8,
    SharpnessSample,
};

/// What ffprobe reports about the first video stream.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VideoInfo {
    /// Display size: coded size with the rotation metadata applied (ffmpeg
    /// auto-rotates the decoded frames the same way).
    pub width: u32,
    pub height: u32,
    pub duration_s: f64,
    /// Average frame rate.
    pub fps: f64,
}

/// Parses `ffprobe -of json -show_streams -show_format` output.
pub fn parse_probe(json: &str) -> Result<VideoInfo> {
    let v: serde_json::Value = serde_json::from_str(json).context("ffprobe JSON")?;
    let s = v["streams"]
        .as_array()
        .and_then(|a| a.first())
        .ok_or_else(|| anyhow!("no video stream"))?;
    let num = |x: &serde_json::Value| -> Option<f64> {
        x.as_f64()
            .or_else(|| x.as_str().and_then(|s| s.parse().ok()))
    };
    let (w, h) = (
        num(&s["width"]).ok_or_else(|| anyhow!("no width"))? as u32,
        num(&s["height"]).ok_or_else(|| anyhow!("no height"))? as u32,
    );
    let rotation = s["side_data_list"]
        .as_array()
        .into_iter()
        .flatten()
        .find_map(|d| num(&d["rotation"]))
        .or_else(|| num(&s["tags"]["rotate"]))
        .unwrap_or(0.0);
    let quarter = ((rotation / 90.0).round() as i64).rem_euclid(2) == 1;
    let rate = |r: &serde_json::Value| -> Option<f64> {
        let (n, d) = r.as_str()?.split_once('/')?;
        let (n, d): (f64, f64) = (n.parse().ok()?, d.parse().ok()?);
        (d > 0.0 && n > 0.0).then_some(n / d)
    };
    let fps = rate(&s["avg_frame_rate"])
        .or_else(|| rate(&s["r_frame_rate"]))
        .ok_or_else(|| anyhow!("no frame rate"))?;
    let duration_s = num(&v["format"]["duration"])
        .or_else(|| num(&s["duration"]))
        .filter(|d| *d > 0.0)
        .ok_or_else(|| anyhow!("no duration"))?;
    let (width, height) = if quarter { (h, w) } else { (w, h) };
    Ok(VideoInfo {
        width,
        height,
        duration_s,
        fps,
    })
}

/// Runs ffprobe on `path`.
pub async fn probe_video(path: &Path) -> Result<VideoInfo> {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", "-show_streams"])
        .args(["-show_format", "-of", "json"])
        .arg(path)
        .kill_on_drop(true)
        .stdin(Stdio::null())
        .output()
        .await
        .context("run ffprobe (is ffmpeg installed?)")?;
    if !out.status.success() {
        bail!(
            "ffprobe {} failed: {}",
            path.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    parse_probe(&String::from_utf8_lossy(&out.stdout))
        .with_context(|| format!("probe {}", path.display()))
}

/// The ffmpeg arguments that decode `path` to packed RGB on stdout.
pub fn ffmpeg_args(path: &Path, cfg: &IngestConfig) -> Vec<String> {
    let mut a: Vec<String> = ["-hide_banner", "-loglevel", "error", "-nostdin"]
        .map(String::from)
        .to_vec();
    a.extend(cfg.ffmpeg_input_args.iter().cloned());
    a.push("-i".into());
    a.push(path.to_string_lossy().into_owned());
    a.extend(["-map", "0:v:0", "-an", "-sn", "-dn"].map(String::from));
    if let Some(f) = cfg.sample_fps {
        a.push("-vf".into());
        a.push(format!("fps={f}"));
    } else {
        a.extend(["-fps_mode", "passthrough"].map(String::from));
    }
    a.extend(["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"].map(String::from));
    a
}

struct Decoded {
    index: u32,
    t: f64,
    data: Vec<u8>,
}

/// One decoded frame handed to a pass's step; `None` after the last one.
type Step<S> = dyn FnMut(&mut S, &mut IngestGpu, Option<(u32, f64, &[u8])>) -> Result<()> + Send;

/// One ffmpeg decode of `path`: every frame goes through `step` on a
/// blocking GPU worker. Returns the state, frames decoded and GPU seconds.
async fn decode_pass<S: Send + 'static>(
    path: &Path,
    cfg: &IngestConfig,
    info: VideoInfo,
    cancel: &CancellationToken,
    mut state: S,
    mut step: Box<Step<S>>,
) -> Result<(S, u32, f64)> {
    let rate = cfg.sample_fps.unwrap_or(info.fps);
    let frame_len = info.width as usize * info.height as usize * 3;
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let (tx, mut rx) = mpsc::channel::<Decoded>(3);
    let (free_tx, free_rx) = std_mpsc::channel::<Vec<u8>>();
    let worker_cancel = cancel.clone();
    let worker = tokio::task::spawn_blocking(move || -> Result<_> {
        let mut g = IngestGpu::open()?;
        while let Some(d) = rx.blocking_recv() {
            if worker_cancel.is_cancelled() {
                bail!("cancelled");
            }
            step(&mut state, &mut g, Some((d.index, d.t, &d.data)))?;
            let _ = free_tx.send(d.data);
        }
        step(&mut state, &mut g, None)?;
        Ok((state, g.seconds))
    });

    let mut child = Command::new("ffmpeg")
        .args(ffmpeg_args(path, cfg))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .context("spawn ffmpeg (is it installed?)")?;
    let mut stdout = BufReader::with_capacity(1 << 20, child.stdout.take().expect("piped"));
    let mut stderr = child.stderr.take().expect("piped");
    let stderr_task = tokio::spawn(async move {
        let mut s = String::new();
        let _ = stderr.read_to_string(&mut s).await;
        s
    });

    let mut index = 0u32;
    let read_result: Result<()> = async {
        loop {
            let mut buf = free_rx.try_recv().unwrap_or_default();
            buf.resize(frame_len, 0);
            let n = tokio::select! {
                _ = cancel.cancelled() => bail!("cancelled"),
                r = read_full(&mut stdout, &mut buf) => r.context("read ffmpeg output")?,
            };
            if n == 0 {
                return Ok(());
            }
            if n < frame_len {
                bail!("ffmpeg produced a partial frame ({n} of {frame_len} bytes)");
            }
            let d = Decoded {
                index,
                t: index as f64 / rate,
                data: buf,
            };
            index += 1;
            tokio::select! {
                _ = cancel.cancelled() => bail!("cancelled"),
                r = tx.send(d) => if r.is_err() { return Ok(()) },
            }
        }
    }
    .await;
    drop(tx);
    if read_result.is_err() {
        let _ = child.kill().await;
    }
    let worker_result = worker.await.map_err(|e| anyhow!("ingest worker: {e}"))?;
    read_result?;
    let (state, gpu_seconds) = worker_result?;
    let status = child.wait().await.context("wait for ffmpeg")?;
    let err = stderr_task.await.unwrap_or_default();
    if !status.success() {
        bail!("ffmpeg failed on {name}: {}", err.trim());
    }
    if index == 0 {
        bail!("ffmpeg decoded no frames from {name}");
    }
    Ok((state, index, gpu_seconds))
}

/// Per-frame scores every pass collects: sharpness (GPU) and the global
/// motion from the previous frame (CPU, on a 160 px luma thumbnail).
#[derive(Default)]
struct Scan {
    samples: Vec<SharpnessSample>,
    scored: Vec<Scored>,
    prev: Option<LumaThumb>,
}

impl Scan {
    fn score(
        &mut self,
        g: &mut IngestGpu,
        info: VideoInfo,
        index: u32,
        t: f64,
        data: &[u8],
    ) -> Result<(Resident, f64)> {
        let r = g.upload(info.width, info.height, data)?;
        let s = g.sharpness(r)?;
        let thumb = LumaThumb::from_rgb(info.width, info.height, data);
        let motion = self
            .prev
            .as_ref()
            .map_or(0.0, |p| frame_motion(p, &thumb).score());
        self.prev = Some(thumb);
        self.samples.push(SharpnessSample {
            index,
            timestamp_s: Some(t),
            sharpness: s,
        });
        self.scored.push(Scored {
            index,
            t,
            sharpness: s,
            motion,
        });
        Ok((r, s))
    }
}

fn blank_frame(index: u32, t: f64, s: f64, info: VideoInfo) -> IngestFrame {
    let empty = Rgb8 {
        width: 0,
        height: 0,
        data: Vec::new(),
    };
    IngestFrame {
        source: String::new(),
        index,
        timestamp_s: Some(t),
        sharpness: s,
        source_size: (info.width, info.height),
        focal_prior: None,
        engine: empty.clone(),
        trainer: empty,
    }
}

/// Time slices, one pass: the best frame of the open slice stays on the
/// device and is resized when a later slice opens (frames arrive in time
/// order), so at most `max_frames` frames are resized and downloaded.
async fn time_keyframes(
    path: &Path,
    cfg: &IngestConfig,
    info: VideoInfo,
    cancel: &CancellationToken,
) -> Result<(Vec<IngestFrame>, Scan, u32, f64)> {
    struct St {
        scan: Scan,
        bins: KeyframeBins,
        slots: Vec<Option<IngestFrame>>,
        open: Option<(usize, IngestFrame)>,
    }
    let st = St {
        scan: Scan::default(),
        bins: KeyframeBins::new(info.duration_s, cfg.max_frames),
        slots: vec![None; cfg.max_frames.max(1) as usize],
        open: None,
    };
    let c = cfg.clone();
    let finish = move |st: &mut St, g: &mut IngestGpu, c: &IngestConfig| -> Result<()> {
        if let Some((b, mut f)) = st.open.take() {
            let r = Resident {
                width: f.source_size.0,
                height: f.source_size.1,
            };
            (f.engine, f.trainer) = resize_both(g, c, r, true)?;
            st.slots[b] = Some(f);
        }
        Ok(())
    };
    let step: Box<Step<St>> = Box::new(move |st, g, d| {
        let Some((index, t, data)) = d else {
            return finish(st, g, &c);
        };
        let (_, s) = st.scan.score(g, info, index, t, data)?;
        if let Some(b) = st.bins.offer(t, s, index) {
            if st.open.as_ref().is_some_and(|(ob, _)| *ob != b) {
                finish(st, g, &c)?;
            }
            g.keep();
            st.open = Some((b, blank_frame(index, t, s, info)));
        }
        Ok(())
    });
    let (st, decoded, gpu_s) = decode_pass(path, cfg, info, cancel, st, step).await?;
    Ok((
        st.slots.into_iter().flatten().collect(),
        st.scan,
        decoded,
        gpu_s,
    ))
}

/// Motion slices, two passes: score every frame, pick along the camera's
/// path ([`select_keyframes`]), then decode again and resize the picks.
async fn motion_keyframes(
    path: &Path,
    cfg: &IngestConfig,
    info: VideoInfo,
    cancel: &CancellationToken,
) -> Result<(Vec<IngestFrame>, Scan, Selection, u32, f64)> {
    let step: Box<Step<Scan>> = Box::new(move |scan, g, d| {
        if let Some((index, t, data)) = d {
            scan.score(g, info, index, t, data)?;
        }
        Ok(())
    });
    let (scan, decoded, gpu_a) =
        decode_pass(path, cfg, info, cancel, Scan::default(), step).await?;
    let sel = select_keyframes(&scan.scored, cfg.max_frames as usize);
    let by_index: std::collections::HashMap<u32, f64> =
        scan.scored.iter().map(|f| (f.index, f.sharpness)).collect();
    let keep = sel.kept.clone();
    let c = cfg.clone();
    let step: Box<Step<Vec<IngestFrame>>> = Box::new(move |out, g, d| {
        let Some((index, t, data)) = d else {
            return Ok(());
        };
        if keep.binary_search(&index).is_ok() {
            let r = g.upload(info.width, info.height, data)?;
            let mut f = blank_frame(index, t, by_index.get(&index).copied().unwrap_or(0.0), info);
            (f.engine, f.trainer) = resize_both(g, &c, r, false)?;
            out.push(f);
        }
        Ok(())
    });
    let (frames, _, gpu_b) = decode_pass(path, cfg, info, cancel, Vec::new(), step).await?;
    Ok((frames, scan, sel, decoded, gpu_a + gpu_b))
}

/// Video ingest: up to `cfg.max_frames` sharp keyframes, spread along the
/// camera's path ([`KeyframeMode::Motion`], two decodes) or evenly in time
/// ([`KeyframeMode::Time`], one decode).
pub async fn ingest_video(
    path: &Path,
    cfg: &IngestConfig,
    cancel: &CancellationToken,
) -> Result<IngestResult> {
    let started = Instant::now();
    let info = probe_video(path).await?;
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let (mut frames, scan, sel, decoded, gpu_seconds) = match cfg.keyframes {
        KeyframeMode::Time => {
            let (frames, scan, decoded, gpu_s) = time_keyframes(path, cfg, info, cancel).await?;
            // The same statistics as the motion path, for the capture report.
            let mut sel = select_keyframes(&scan.scored, cfg.max_frames as usize);
            sel.kept = frames.iter().map(|f| f.index).collect();
            (frames, scan, sel, decoded, gpu_s)
        }
        KeyframeMode::Motion => motion_keyframes(path, cfg, info, cancel).await?,
    };
    for f in &mut frames {
        f.source = format!("{name}#{}", f.index);
    }
    let capture = CaptureStats::from_video(&scan.scored, &sel, &frames, cfg.keyframes);
    let seconds = started.elapsed().as_secs_f64();
    Ok(IngestResult {
        stats: IngestStats {
            decoded,
            kept: frames.len() as u32,
            seconds,
            decode_fps: decoded as f64 / seconds.max(1e-9),
            gpu_seconds,
        },
        frames,
        sharpness: scan.samples,
        focal_prior: None,
        capture,
    })
}

/// Reads until `buf` is full or EOF; returns the bytes read.
async fn read_full<R: AsyncReadExt + Unpin>(r: &mut R, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut n = 0;
    while n < buf.len() {
        let k = r.read(&mut buf[n..]).await?;
        if k == 0 {
            break;
        }
        n += k;
    }
    Ok(n)
}
