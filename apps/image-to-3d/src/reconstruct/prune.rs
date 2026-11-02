//! Prune-first pass after Stage B (Report section 3, Stage B, last bullet):
//! FastGS's VCP scoring on the fresh dense initialization. K evenly spaced
//! training views are rendered; each view's normalized error map above `tau`
//! is the mask, and every Gaussian gets Σ_views E_v · (masked pixels it
//! blends into), E_v = 0.8·L1 + 0.2·(1 − SSIM) of the view (the same score
//! as `kernels/densify.rs` FastGs::score). The top `fraction` by score, the
//! Gaussians that most consistently cover badly reconstructed pixels, are
//! removed (deterministically, instead of FastGS's Gumbel draw). Owner:
//! mvp-pipeline.

use std::time::Instant;

use spl_core::Result;

use crate::gpu::CudaResultExt;
use crate::kernels::RenderMode;
use crate::kernels::adam::AdamIterPlan;
use crate::kernels::densify::{DensifyKernels, ViewScores, vcd_mask};
use crate::kernels::layout::{FLAG_ABS_GRAD, FLAG_DENSIFY_STATS};
use crate::kernels::loss::{ErrorMap, resample_target};
use crate::kernels::raster::{FrameSource, SplatRaster};
use crate::trainer::{Trainer, select};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PruneReport {
    pub before: u32,
    pub after: u32,
    pub views: u32,
    pub ms: f64,
}

/// `k` evenly spaced indices of `0..n`.
pub fn even_views(n: usize, k: usize) -> Vec<usize> {
    if n == 0 || k == 0 {
        return Vec::new();
    }
    let k = k.min(n);
    (0..k).map(|i| i * n / k).collect()
}

/// Indices of the Gaussians to keep: all but the `fraction` with the
/// highest score (ties broken by index; zero scores are always kept).
pub fn keep_after_prune(score: &[f32], fraction: f32) -> Vec<usize> {
    let n = score.len();
    let drop_n = ((n as f64) * fraction.clamp(0.0, 1.0) as f64) as usize;
    let mut order: Vec<usize> = (0..n).filter(|&i| score[i] > 0.0).collect();
    order.sort_by(|&a, &b| score[b].total_cmp(&score[a]).then(a.cmp(&b)));
    let mut dropped = vec![false; n];
    for &i in order.iter().take(drop_n) {
        dropped[i] = true;
    }
    (0..n).filter(|&i| !dropped[i]).collect()
}

/// Runs the pass on the trainer's current Gaussians (before the first step).
pub fn prune_first(tr: &mut Trainer, k: usize, tau: f32, fraction: f32) -> Result<PruneReport> {
    let started = Instant::now();
    let gpu = tr.gpu.clone();
    let n = tr.g.n;
    let mut rep = PruneReport {
        before: n,
        after: n,
        ..Default::default()
    };
    if fraction <= 0.0 || n == 0 || k == 0 {
        return Ok(rep);
    }
    let s = &gpu.stream;
    let max_px = tr
        .scene
        .views
        .iter()
        .map(|v| v.width as usize * v.height as usize)
        .max()
        .unwrap_or(1);
    let dk = DensifyKernels::load(&gpu)?;
    let mut scores = ViewScores::alloc(&gpu, tr.g.cap as usize)?;
    let mut target = s
        .alloc_zeros::<f32>(3 * max_px)
        .cuda("alloc prune target")?;
    let mut err = ErrorMap {
        err: s.alloc_zeros::<f32>(max_px).cuda("alloc prune error")?,
        minmax: s.alloc_zeros::<f32>(2).cuda("alloc prune minmax")?,
        sums: s.alloc_zeros::<f64>(2).cuda("alloc prune sums")?,
    };
    let mut mask = s.alloc_zeros::<u8>(max_px).cuda("alloc prune mask")?;
    let train = tr.scene.train.clone();
    for j in even_views(train.len(), k) {
        let view = train[j];
        let mut p = tr.iter_params(1, view, 0, 1, AdamIterPlan::default());
        p.frame.flags &= !(FLAG_DENSIFY_STATS | FLAG_ABS_GRAD);
        let px = p.frame.render_w * p.frame.render_h;
        let raster = tr.raster.as_mut().expect("raster present");
        raster.forward(
            &gpu,
            &mut tr.g,
            &tr.scene_gpu.cams,
            FrameSource::Eager(&p),
            RenderMode::Infer,
        )?;
        resample_target(
            &gpu,
            &tr.loss.kernels,
            raster.params(),
            &tr.scene_gpu.images,
            &mut target,
        )?;
        tr.loss
            .error_map(&gpu, raster.params(), raster.image(), &target, &mut err)?;
        let sums = s
            .clone_dtoh(&err.sums.slice(0..2))
            .cuda("read prune view loss")?;
        let e = (0.8 * sums[0] + 0.2 * (1.0 - sums[1])) as f32;
        vcd_mask(&gpu, &dk, &err, tau, px, &mut mask)?;
        raster.count_masked(
            &gpu,
            &mut tr.g,
            &tr.scene_gpu.cams,
            FrameSource::Eager(&p),
            &mask,
            e,
            &mut scores,
        )?;
        let st = raster.frame.iter.read_status(&gpu)?;
        if st.overflow != 0 {
            // Scores of an overflowed view are incomplete; skip pruning.
            tracing::warn!("prune-first: raster overflow {:#x}; skipped", st.overflow);
            tr.status.clear();
            return Ok(rep);
        }
        rep.views += 1;
    }
    let score = s
        .clone_dtoh(&scores.score.slice(0..n as usize))
        .cuda("download prune scores")?;
    let keep = keep_after_prune(&score, fraction);
    let host = tr.g.download(&gpu)?;
    let kept = select(&host, &keep);
    tr.g.upload(&gpu, &kept)?;
    gpu.sync()?;
    rep.after = tr.g.n;
    rep.ms = started.elapsed().as_secs_f64() * 1e3;
    Ok(rep)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn even_views_spread_and_clamp() {
        assert_eq!(even_views(100, 4), vec![0, 25, 50, 75]);
        assert_eq!(even_views(3, 10), vec![0, 1, 2]);
        assert!(even_views(0, 3).is_empty());
    }

    #[test]
    fn prune_drops_the_highest_scores_only() {
        let s = [0.0, 5.0, 1.0, 5.0, 0.0, 3.0, 0.5, 0.0, 2.0, 4.0];
        assert_eq!(keep_after_prune(&s, 0.2), vec![0, 2, 4, 5, 6, 7, 8, 9]);
        assert_eq!(keep_after_prune(&s, 0.0).len(), 10);
        // More than the positive scores: zeros survive.
        assert_eq!(keep_after_prune(&s, 0.9), vec![0, 4, 7]);
    }
}
