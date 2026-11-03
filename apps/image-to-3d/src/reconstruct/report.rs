//! Capture and result reports of `image_to_3d.reconstruct`, written for
//! people who are not 3D experts. Owner: web-results.
//!
//! - [`capture_report`]: after ingest and the engine's poses — blur,
//!   standing still and exposure clipping from ingest
//!   ([`crate::ingest::CaptureStats`]), plus how much the camera actually
//!   moved from the poses: the widest baseline over the median camera to
//!   scene distance, the arc the cameras cover around the scene centre (the
//!   median engine point) and the spread of viewing directions. A verdict
//!   (`good`, `ok`, `poor`) and up to four tips in plain words.
//! - [`evaluate`]: after training, on the held-out frames — test-time pose
//!   refinement of the held-out views (coarse to fine, as the benchmark
//!   protocol), PSNR and SSIM with a plain rating, and the three worst views
//!   as small render-next-to-photo thumbnails.
//! - [`floaters`]: drops Gaussians seen by fewer than `k` training cameras,
//!   nearly transparent ones and huge ones; the held-out PSNR is measured
//!   with them hidden, and the cleaned splat is exported as its own `.spz`.
//!
//! Rating thresholds (held-out PSNR / SSIM after test-time refinement) come
//! from our COLMAP-free runs on this GPU (`benchmarks/image-to-3d/
//! RECONSTRUCT_PHASE4.md`): Quality on Mip-NeRF 360 garden and room lands
//! at 25-32 dB, the hardest scenes (bicycle Fast, T&T train) at 20-21.5 dB.

use rayon::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use spl_core::{ArtifactKind, Context, Result, RunCx};
use ts_rs::TS;

use super::pipeline::Reconstruction;
use crate::export::spz::{SpzOptions, write_spz};
use crate::ingest::CaptureStats;
use crate::io::images::Rgb8Image;
use crate::io::scene::View;
use crate::model::GaussiansHost;

/// Held-out PSNR at or above this (with SSIM ≥ [`GOOD_SSIM`]) rates `good`.
pub const GOOD_PSNR: f64 = 25.0;
pub const GOOD_SSIM: f64 = 0.75;
/// Below this PSNR (or SSIM under [`WEAK_SSIM`]) rates `weak`.
pub const WEAK_PSNR: f64 = 21.0;
pub const WEAK_SSIM: f64 = 0.55;
/// Hold out every n-th frame for the result report.
pub const HOLDOUT_EVERY: usize = 8;
/// Fewest frames for a held-out evaluation. Smaller captures train on every
/// frame (each one matters more than the score), so they get no rating.
pub const HOLDOUT_MIN_FRAMES: usize = 80;

/// How the capture went, before training.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct CaptureReport {
    /// `good`, `ok` or `poor` (poor: the result will likely disappoint).
    pub verdict: String,
    pub frames: u32,
    pub stats: CaptureStats,
    /// Widest distance between two cameras over the median camera to
    /// scene-centre distance (turning in place ≈ 0, a full orbit ≈ 2).
    #[ts(optional)]
    pub baseline: Option<f64>,
    /// Widest angle between two cameras as seen from the scene centre, degrees.
    #[ts(optional)]
    pub orbit_deg: Option<f64>,
    /// Widest angle between two viewing directions, degrees.
    #[ts(optional)]
    pub turn_deg: Option<f64>,
    /// What is clearly wrong, in plain words.
    pub problems: Vec<String>,
    /// What to do differently next time (at most four).
    pub tips: Vec<String>,
}

type V3 = [f64; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn norm(a: V3) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

fn angle_deg(a: V3, b: V3) -> f64 {
    let (na, nb) = (norm(a), norm(b));
    if na < 1e-12 || nb < 1e-12 {
        return 0.0;
    }
    let c = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (na * nb);
    c.clamp(-1.0, 1.0).acos().to_degrees()
}

/// Camera centres and viewing directions from `cameras.json` (OpenCV w2c).
pub fn parse_cameras(json: &[u8]) -> Result<Vec<(V3, V3)>> {
    let v: serde_json::Value = serde_json::from_slice(json).context("cameras.json")?;
    let mut out = Vec::new();
    for f in v["frames"].as_array().into_iter().flatten() {
        let m: Vec<Vec<f64>> = serde_json::from_value(f["w2c"].clone()).context("w2c")?;
        if m.len() < 3 || m.iter().any(|r| r.len() < 4) {
            continue;
        }
        // c = −Rᵀt; forward = Rᵀ(0, 0, 1) = third row of R.
        let c = [0, 1, 2].map(|j| -(0..3).map(|i| m[i][j] * m[i][3]).sum::<f64>());
        out.push((c, [m[2][0], m[2][1], m[2][2]]));
    }
    Ok(out)
}

/// Positions from the engine's binary little-endian float PLY (`x y z` first).
pub fn parse_points(ply: &[u8]) -> Vec<V3> {
    let Some(end) = ply.windows(11).position(|w| w == b"end_header\n") else {
        return Vec::new();
    };
    let header = String::from_utf8_lossy(&ply[..end]);
    let mut count = 0usize;
    let mut props = 0usize;
    for l in header.lines() {
        if let Some(n) = l.strip_prefix("element vertex ") {
            count = n.trim().parse().unwrap_or(0);
        } else if l.starts_with("property float ") {
            props += 1;
        }
    }
    if !header.contains("binary_little_endian") || props < 3 {
        return Vec::new();
    }
    let body = &ply[end + 11..];
    let stride = props * 4;
    let count = count.min(body.len() / stride);
    let step = (count / 50_000).max(1);
    let f = |o: usize| f32::from_le_bytes([body[o], body[o + 1], body[o + 2], body[o + 3]]) as f64;
    (0..count)
        .step_by(step)
        .map(|i| [f(i * stride), f(i * stride + 4), f(i * stride + 8)])
        .filter(|p| p.iter().all(|v| v.is_finite()))
        .collect()
}

fn median(mut v: Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    Some(v[v.len() / 2])
}

/// Baseline, orbit and turn ([`CaptureReport`]) of `cams` around `points`.
pub fn camera_spread(cams: &[(V3, V3)], points: &[V3]) -> Option<(f64, f64, f64)> {
    if cams.len() < 2 || points.is_empty() {
        return None;
    }
    let centre = [0, 1, 2].map(|k| median(points.iter().map(|p| p[k]).collect()).unwrap_or(0.0));
    let dist = median(cams.iter().map(|(c, _)| norm(sub(*c, centre))).collect())?;
    if dist < 1e-9 {
        return None;
    }
    let step = (cams.len() / 300).max(1);
    let cs: Vec<&(V3, V3)> = cams.iter().step_by(step).collect();
    let (mut base, mut orbit, mut turn) = (0f64, 0f64, 0f64);
    for (i, a) in cs.iter().enumerate() {
        for b in &cs[i + 1..] {
            base = base.max(norm(sub(a.0, b.0)));
            orbit = orbit.max(angle_deg(sub(a.0, centre), sub(b.0, centre)));
            turn = turn.max(angle_deg(a.1, b.1));
        }
    }
    Some((base / dist, orbit, turn))
}

/// The capture report from ingest's statistics and the engine's cameras
/// (`cameras.json`) and points (PLY).
pub fn capture_report(stats: &CaptureStats, cameras: &[u8], points: &[u8]) -> CaptureReport {
    let cams = parse_cameras(cameras).unwrap_or_default();
    let spread = camera_spread(&cams, &parse_points(points));
    let mut r = CaptureReport {
        frames: if cams.is_empty() {
            stats.kept
        } else {
            cams.len() as u32
        },
        stats: stats.clone(),
        baseline: spread.map(|s| s.0),
        orbit_deg: spread.map(|s| s.1),
        turn_deg: spread.map(|s| s.2),
        ..Default::default()
    };
    let video = stats.keyframes != "photos";
    let pct = |x: f64| format!("{:.0} %", 100.0 * x);
    if r.frames < 8 {
        r.problems.push(format!("Only {} usable frames.", r.frames));
    }
    if r.frames < 20 {
        r.tips.push(if video {
            "Record a longer clip (30-60 s) while walking around the subject.".into()
        } else {
            "Take more photos: 30 to 200 overlapping shots work best.".into()
        });
    }
    if let Some((base, orbit, turn)) = spread {
        if base < 0.2 && turn > 30.0 {
            r.problems
                .push("The camera turned in place instead of moving around the scene.".into());
            r.tips
                .push("Move around the object instead of turning in place.".into());
        } else if base < 0.05 {
            r.problems
                .push("The camera barely moved, so depth cannot be seen.".into());
            r.tips
                .push("Walk around the subject while filming; do not stand in one spot.".into());
        } else if orbit < 20.0 && turn < 20.0 {
            r.tips.push(
                "Cover more angles: walk an arc around the subject, keeping it in view.".into(),
            );
        }
    }
    if stats.kept_blurry_share > 0.3 {
        r.problems.push(format!(
            "{} of the kept frames are blurry.",
            pct(stats.kept_blurry_share)
        ));
    }
    if stats.blurry_share > 0.25 {
        r.tips.push(format!(
            "{} of the frames were blurry: move slower, hold the phone steady, or add light.",
            pct(stats.blurry_share)
        ));
    }
    if stats.bright_clip_share > 0.05 {
        r.tips.push(format!(
            "{} of the image is blown out: avoid bright windows and sky, or lock exposure on the subject.",
            pct(stats.bright_clip_share)
        ));
    }
    if stats.dark_clip_share > 0.1 {
        r.tips.push(format!(
            "{} of the image is pure black: add light or film in brighter conditions.",
            pct(stats.dark_clip_share)
        ));
    }
    if stats.still_share.is_some_and(|s| s > 0.4) {
        r.tips.push(
            "Keep moving at a steady pace; standing still adds no new views (those parts were skipped)."
                .into(),
        );
    }
    r.tips.truncate(4);
    r.verdict = if !r.problems.is_empty() {
        "poor"
    } else if !r.tips.is_empty() {
        "ok"
    } else {
        "good"
    }
    .into();
    r
}

/// Plain rating of held-out metrics.
pub fn rating(psnr: f64, ssim: f64) -> &'static str {
    if psnr < WEAK_PSNR || ssim < WEAK_SSIM {
        "weak"
    } else if psnr >= GOOD_PSNR && ssim >= GOOD_SSIM {
        "good"
    } else {
        "ok"
    }
}

/// One held-out view in the result report.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct HeldOutView {
    pub name: String,
    pub psnr: f64,
    pub ssim: f64,
    /// JPEG: the render (left) next to the photo (right); artifact id.
    #[ts(optional)]
    pub thumb: Option<String>,
}

/// Floater removal: what went, and what it did to the held-out views.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct FloaterReport {
    pub before: u32,
    pub after: u32,
    /// Seen by fewer than `min_views` training cameras.
    pub unseen: u32,
    pub transparent: u32,
    pub huge: u32,
    pub min_views: u32,
    #[ts(optional)]
    pub psnr_before: Option<f64>,
    #[ts(optional)]
    pub psnr_after: Option<f64>,
    /// The cleaned `.spz`; artifact id.
    #[ts(optional)]
    pub spz: Option<String>,
}

/// Quality of the trained splat on frames it never saw.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct ResultReport {
    /// `good`, `ok` or `weak`; absent without held-out frames.
    #[ts(optional)]
    pub rating: Option<String>,
    #[ts(optional)]
    pub psnr: Option<f64>,
    #[ts(optional)]
    pub ssim: Option<f64>,
    pub held_out: u32,
    /// Worst held-out views first (at most three).
    pub worst: Vec<HeldOutView>,
    #[ts(optional)]
    pub floaters: Option<FloaterReport>,
    /// The preset this run used (`fast`, `quality`, `big`).
    pub preset: String,
    /// Seconds spent on this report (pose refinement, renders, cleaning).
    pub seconds: f64,
}

/// [`evaluate`]'s output before its files are stored.
pub struct Evaluated {
    pub report: ResultReport,
    /// JPEG thumbnails, parallel to `report.worst`.
    pub thumbs: Vec<Vec<u8>>,
    /// The cleaned splat, when anything was removed.
    pub clean_spz: Option<Vec<u8>>,
}

/// Floater criteria.
#[derive(Debug, Clone, Copy)]
pub struct FloaterRule {
    pub min_views: u32,
    /// Below this opacity (after the sigmoid).
    pub min_opacity: f32,
    /// Largest axis above this fraction of the camera extent.
    pub max_scale: f32,
}

impl FloaterRule {
    /// Conservative on purpose: on garden video runs (48 keyframes, Fast)
    /// dropping splats seen by < 3 cameras or wider than 0.1 × the camera
    /// extent removed real edges and background and cost 1.3-1.5 dB
    /// held-out PSNR; only splats no training camera sees, nearly
    /// transparent ones and ones wider than the whole camera rig go.
    pub fn for_views(_n: usize) -> Self {
        Self {
            min_views: 1,
            min_opacity: 0.01,
            max_scale: 1.0,
        }
    }
}

/// Why each Gaussian goes (0 = stays; 1 unseen, 2 transparent, 3 huge).
pub fn floaters(g: &GaussiansHost, views: &[View], rule: FloaterRule) -> Vec<u8> {
    // Camera extent: the 3DGS `cameras_extent` (1.1 × the radius of the
    // camera centres around their mean).
    let centres: Vec<V3> = views
        .iter()
        .map(|v| [0, 1, 2].map(|j| -(0..3).map(|i| v.r_cw[i][j] * v.t_cw[i]).sum::<f64>()))
        .collect();
    let n = centres.len().max(1) as f64;
    let mean = [0, 1, 2].map(|k| centres.iter().map(|c| c[k]).sum::<f64>() / n);
    let extent = 1.1
        * centres
            .iter()
            .map(|c| norm(sub(*c, mean)))
            .fold(0.0, f64::max)
            .max(1e-6);
    let max_scale = (rule.max_scale as f64 * extent) as f32;
    (0..g.len())
        .into_par_iter()
        .map(|i| {
            let op = 1.0 / (1.0 + (-g.opacities[i]).exp());
            if op < rule.min_opacity {
                return 2;
            }
            let s = &g.scales[3 * i..3 * i + 3];
            if s.iter().fold(f32::MIN, |a, &b| a.max(b)).exp() > max_scale {
                return 3;
            }
            let p = [
                g.means[3 * i] as f64,
                g.means[3 * i + 1] as f64,
                g.means[3 * i + 2] as f64,
            ];
            let mut seen = 0;
            for v in views {
                let r = &v.r_cw;
                let x = r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2] + v.t_cw[0];
                let y = r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2] + v.t_cw[1];
                let z = r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2] + v.t_cw[2];
                if z <= 1e-3 {
                    continue;
                }
                let (u, w) = (v.fx * x / z + v.cx, v.fy * y / z + v.cy);
                if u >= 0.0 && w >= 0.0 && u < v.width as f64 && w < v.height as f64 {
                    seen += 1;
                    if seen >= rule.min_views {
                        return 0;
                    }
                }
            }
            1
        })
        .collect()
}

/// The Gaussians whose `why` is 0.
pub fn keep_only(g: &GaussiansHost, why: &[u8]) -> GaussiansHost {
    let keep: Vec<usize> = (0..g.len()).filter(|&i| why[i] == 0).collect();
    let pick = |src: &[f32], per: usize| -> Vec<f32> {
        keep.iter()
            .flat_map(|&i| src[i * per..(i + 1) * per].iter().copied())
            .collect()
    };
    let rest = if !g.is_empty() {
        g.sh_rest.len() / g.len()
    } else {
        0
    };
    GaussiansHost {
        means: pick(&g.means, 3),
        scales: pick(&g.scales, 3),
        rotations: pick(&g.rotations, 4),
        opacities: pick(&g.opacities, 1),
        sh0: pick(&g.sh0, 3),
        sh_rest: pick(&g.sh_rest, rest),
        sh_degree: g.sh_degree,
        antialiased: g.antialiased,
    }
}

/// Render (left) next to photo (right), each `w` px wide, as JPEG.
pub fn side_by_side(render: &Rgb8Image, photo: &Rgb8Image, w: u32) -> Result<Vec<u8>> {
    let w = w.min(render.width).max(1);
    let h = ((render.height as f64 * w as f64 / render.width.max(1) as f64).round() as u32).max(1);
    let small = |img: &Rgb8Image| {
        crate::ingest::resize_area_cpu(
            &crate::ingest::Rgb8 {
                width: img.width,
                height: img.height,
                data: img.data.clone(),
            },
            w,
            h,
        )
    };
    let (a, b) = (small(render), small(photo));
    let mut out = vec![0u8; (2 * w * h * 3) as usize];
    let row = (w * 3) as usize;
    for y in 0..h as usize {
        out[y * 2 * row..y * 2 * row + row].copy_from_slice(&a.data[y * row..(y + 1) * row]);
        out[y * 2 * row + row..(y + 1) * 2 * row].copy_from_slice(&b.data[y * row..(y + 1) * row]);
    }
    let mut jpg = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpg, 85)
        .encode(&out, 2 * w, h, image::ExtendedColorType::Rgb8)
        .context("encode thumbnail")?;
    Ok(jpg)
}

/// Mean PSNR of the held-out views at their current poses.
fn held_out_psnr(r: &mut Reconstruction) -> Result<f64> {
    Ok(r.evaluate_test(None)?.0.psnr)
}

/// The result report (module docs). Needs `r.joint`; with held-out views it
/// refines their poses, scores them and renders the worst three; then it
/// measures and exports the floater-free splat.
pub fn evaluate(r: &mut Reconstruction, preset: &str) -> Result<Evaluated> {
    let started = std::time::Instant::now();
    let mut report = ResultReport {
        preset: preset.into(),
        ..Default::default()
    };
    let mut thumbs = Vec::new();
    if r.test.is_some() {
        // The benchmark protocol's coarse-to-fine test-time refinement,
        // shortened (most of the gain comes in the coarse stages).
        for (scale, iters) in [(4, 100), (2, 100), (1, 100)] {
            r.refine_test_at(scale, iters, 1e-3, 1e-3)?;
        }
        let (mean, per) = r.evaluate_test(None)?;
        report.psnr = Some(mean.psnr);
        report.ssim = Some(mean.ssim);
        report.rating = Some(rating(mean.psnr, mean.ssim).into());
        report.held_out = per.len() as u32;
        let mut order: Vec<usize> = (0..per.len()).collect();
        order.sort_by(|&a, &b| per[a].psnr.total_cmp(&per[b].psnr));
        order.truncate(3);
        let (joint, tv) = (
            r.joint.as_mut().expect("evaluated"),
            r.test.as_ref().expect("test"),
        );
        for i in order {
            let v = &tv.views[i];
            let img = joint.render(&tv.cams, i, v.width, v.height)?;
            thumbs.push(side_by_side(&img, &r.test_images[i], 240)?);
            report.worst.push(HeldOutView {
                name: v.name.clone(),
                psnr: per[i].psnr,
                ssim: per[i].ssim,
                thumb: None,
            });
        }
    }
    let rule = FloaterRule::for_views(r.views.len());
    let why = floaters(&r.cloud, &r.views, rule);
    let count = |k: u8| why.iter().filter(|&&w| w == k).count() as u32;
    let mut fl = FloaterReport {
        before: r.cloud.len() as u32,
        unseen: count(1),
        transparent: count(2),
        huge: count(3),
        min_views: rule.min_views,
        psnr_before: report.psnr,
        ..Default::default()
    };
    fl.after = fl.before - fl.unseen - fl.transparent - fl.huge;
    let mut clean_spz = None;
    if fl.after < fl.before {
        let clean = keep_only(&r.cloud, &why);
        if r.test.is_some() {
            // Hide the same Gaussians on the device (same count and order,
            // so per-Gaussian device state stays aligned), score, restore.
            let joint = r.joint.as_mut().expect("evaluated");
            let tr = &mut joint.tr;
            let full = tr.g.download(&tr.gpu)?;
            let why_full = floaters(&full, &r.views, rule);
            let mut hidden = full.clone();
            for (o, &w) in hidden.opacities.iter_mut().zip(&why_full) {
                if w != 0 {
                    *o = -30.0;
                }
            }
            tr.g.upload(&tr.gpu, &hidden)?;
            let after = held_out_psnr(r);
            let joint = r.joint.as_mut().expect("evaluated");
            joint.tr.g.upload(&joint.tr.gpu, &full)?;
            fl.psnr_after = Some(after?);
        }
        clean_spz = Some(write_spz(&clean, &SpzOptions::default())?);
    }
    report.floaters = Some(fl);
    report.seconds = started.elapsed().as_secs_f64();
    Ok(Evaluated {
        report,
        thumbs,
        clean_spz,
    })
}

/// Keeps what the capture report needs while the op relays events: the
/// engine's cameras arrive first, then its points, then the report goes out
/// as the `capture` announcement (before training starts).
pub struct CaptureRelay {
    pub stats: CaptureStats,
    pub cameras: Option<Vec<u8>>,
    pub report: Option<CaptureReport>,
}

impl CaptureRelay {
    pub fn new(stats: CaptureStats) -> Self {
        Self {
            stats,
            cameras: None,
            report: None,
        }
    }

    /// Builds the report from the stored cameras and `points`, stores it as
    /// JSON and announces it; logs problems so they also reach the run log.
    pub async fn announce(&mut self, cx: &RunCx, points: &[u8]) -> Result<()> {
        let Some(cameras) = self.cameras.as_deref() else {
            return Ok(());
        };
        let r = capture_report(&self.stats, cameras, points);
        for p in &r.problems {
            cx.log(format!("capture: {p}"));
        }
        let info = serde_json::to_value(&r)?;
        let a = cx
            .artifacts
            .put(
                serde_json::to_vec_pretty(&r)?,
                ArtifactKind::Report,
                spl_core::media::JSON,
            )
            .await?;
        cx.announce("capture", &a, info);
        self.report = Some(r);
        Ok(())
    }
}

/// Stores the thumbnails and the cleaned splat; returns the report with
/// their artifact ids.
pub async fn store(cx: &RunCx, e: Evaluated) -> Result<ResultReport> {
    let mut r = e.report;
    for (v, jpg) in r.worst.iter_mut().zip(e.thumbs) {
        let a = cx
            .artifacts
            .put(jpg, ArtifactKind::Image, spl_core::media::JPEG)
            .await?;
        v.thumb = Some(a.id);
    }
    if let (Some(f), Some(spz)) = (r.floaters.as_mut(), e.clean_spz) {
        let a = cx
            .artifacts
            .put(spz, ArtifactKind::Splat, crate::export::MEDIA_SPZ)
            .await?;
        f.spz = Some(a.id);
    }
    if let (Some(p), Some(s)) = (r.psnr, r.ssim) {
        cx.metric("held_out_psnr", p, "dB");
        cx.metric("held_out_ssim", s, "");
    }
    Ok(r)
}

#[cfg(test)]
#[path = "report_gpu_tests.rs"]
mod gpu_tests;

#[cfg(test)]
mod tests {
    use super::*;

    fn cams_json(cams: &[(V3, V3)]) -> Vec<u8> {
        // Look-at rotations: forward = d, x ⟂ world up (y).
        let frames: Vec<_> = cams
            .iter()
            .map(|(c, d)| {
                let n = norm(*d);
                let f = d.map(|v| v / n);
                let up = [0.0, 1.0, 0.0];
                let x = [
                    up[1] * f[2] - up[2] * f[1],
                    up[2] * f[0] - up[0] * f[2],
                    up[0] * f[1] - up[1] * f[0],
                ];
                let nx = norm(x);
                let x = x.map(|v| v / nx);
                let y = [
                    f[1] * x[2] - f[2] * x[1],
                    f[2] * x[0] - f[0] * x[2],
                    f[0] * x[1] - f[1] * x[0],
                ];
                let r = [x, y, f];
                let t = [0, 1, 2].map(|i| -(r[i][0] * c[0] + r[i][1] * c[1] + r[i][2] * c[2]));
                let w2c = [0, 1, 2].map(|i| [r[i][0], r[i][1], r[i][2], t[i]]);
                serde_json::json!({ "w2c": w2c })
            })
            .collect();
        serde_json::to_vec(&serde_json::json!({ "frames": frames })).unwrap()
    }

    fn points_ply(pts: &[V3]) -> Vec<u8> {
        let mut b = format!(
            "ply\nformat binary_little_endian 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nend_header\n",
            pts.len()
        )
        .into_bytes();
        for p in pts {
            for v in p {
                b.extend((*v as f32).to_le_bytes());
            }
        }
        b
    }

    fn stats() -> CaptureStats {
        CaptureStats {
            keyframes: "motion".into(),
            decoded: 300,
            kept: 60,
            ..Default::default()
        }
    }

    #[test]
    fn parses_centres_and_points() {
        let cams = parse_cameras(&cams_json(&[([1.0, 2.0, 3.0], [0.0, 0.0, 1.0])])).unwrap();
        for k in 0..3 {
            assert!((cams[0].0[k] - [1.0, 2.0, 3.0][k]).abs() < 1e-9);
            assert!((cams[0].1[k] - [0.0, 0.0, 1.0][k]).abs() < 1e-9);
        }
        let pts = parse_points(&points_ply(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
        assert_eq!(pts, vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    }

    #[test]
    fn an_orbit_is_good_and_turning_in_place_is_poor() {
        let pts = points_ply(&[[0.0, 0.0, 0.0]; 10]);
        let orbit: Vec<(V3, V3)> = (0..40)
            .map(|i| {
                let a = i as f64 / 40.0 * std::f64::consts::TAU;
                let c = [3.0 * a.cos(), 0.0, 3.0 * a.sin()];
                (c, c.map(|v| -v))
            })
            .collect();
        let r = capture_report(&stats(), &cams_json(&orbit), &pts);
        assert_eq!(r.verdict, "good", "{r:?}");
        assert!(r.baseline.unwrap() > 1.9 && r.orbit_deg.unwrap() > 170.0);

        let spin: Vec<(V3, V3)> = (0..40)
            .map(|i| {
                let a = i as f64 / 40.0 * std::f64::consts::PI;
                ([3.0 + 0.05 * a.cos(), 0.0, 0.0], [a.cos(), 0.0, a.sin()])
            })
            .collect();
        let r = capture_report(&stats(), &cams_json(&spin), &pts);
        assert_eq!(r.verdict, "poor", "{r:?}");
        assert!(
            r.tips
                .iter()
                .any(|t| t.contains("instead of turning in place"))
        );
    }

    #[test]
    fn blur_and_clipping_become_tips() {
        let s = CaptureStats {
            blurry_share: 0.4,
            kept_blurry_share: 0.35,
            bright_clip_share: 0.1,
            still_share: Some(0.5),
            ..stats()
        };
        let r = capture_report(&s, b"{}", b"");
        assert_eq!(r.verdict, "poor");
        assert_eq!(r.tips.len(), 3, "{:?}", r.tips);
    }

    #[test]
    fn ratings_follow_the_thresholds() {
        assert_eq!(rating(27.0, 0.85), "good");
        assert_eq!(rating(27.0, 0.7), "ok");
        assert_eq!(rating(23.0, 0.8), "ok");
        assert_eq!(rating(20.0, 0.8), "weak");
        assert_eq!(rating(24.0, 0.5), "weak");
    }

    fn view(c: V3) -> View {
        // Looking down +z from `c` (identity rotation).
        View {
            name: String::new(),
            camera: 0,
            r_cw: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            t_cw: c.map(|v| -v),
            fx: 100.0,
            fy: 100.0,
            cx: 50.0,
            cy: 50.0,
            width: 100,
            height: 100,
            image_path: Default::default(),
        }
    }

    #[test]
    fn floaters_drop_unseen_transparent_and_huge() {
        let views: Vec<View> = (0..4).map(|i| view([i as f64 * 0.1, 0.0, 0.0])).collect();
        let mut g = GaussiansHost::zeros(4);
        g.means = vec![0.1, 0.0, 5.0, 0.1, 0.0, -5.0, 0.1, 0.0, 5.0, 0.1, 0.0, 5.0];
        g.scales = vec![-5.0; 12];
        g.scales[9] = 2.0;
        g.opacities = vec![2.0, 2.0, -8.0, 2.0];
        let rule = FloaterRule {
            min_views: 2,
            min_opacity: 0.01,
            max_scale: 0.1,
        };
        let why = floaters(&g, &views, rule);
        assert_eq!(why, vec![0, 1, 2, 3]);
        let kept = keep_only(&g, &why);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept.sh_rest.len(), g.sh_rest.len() / 4);
    }

    #[test]
    fn thumbnails_are_jpeg_of_twice_the_width() {
        let img = Rgb8Image {
            width: 64,
            height: 32,
            data: vec![90; 64 * 32 * 3],
        };
        let jpg = side_by_side(&img, &img, 32).unwrap();
        let d = image::load_from_memory(&jpg).unwrap();
        assert_eq!((d.width(), d.height()), (64, 16));
    }
}
