//! Motion-aware keyframes: a cheap global motion estimate between
//! consecutive decoded frames, then keyframes spread evenly along the
//! camera's path (not the clock). Owner: ingest (A0).
//!
//! - [`LumaThumb`]: a small luma image (160 px wide) sampled on the CPU from
//!   the decoded RGB, normalised to zero mean and unit variance so slow
//!   exposure drift does not count as motion.
//! - [`frame_motion`]: global translation by coarse-to-fine block matching
//!   (three levels, ±8 px at 40 px wide, then ±2 px), plus the residual mean
//!   absolute difference after alignment, which catches what a translation
//!   cannot explain (parallax, zoom, rotation). Score = shift / width +
//!   [`RESIDUAL_WEIGHT`] × (residual above the noise floor).
//! - [`select_keyframes`]: drops very blurry frames (sharpness under
//!   [`BLUR_RATIO`] of the local 75th percentile), cuts the cumulative motion
//!   into `max` equal slices keeping the sharpest frame of each, splits the
//!   widest remaining gaps when slices came out empty, and drops near
//!   duplicates. A clip with almost no motion falls back to time slices.

/// Thumbnail width; height follows the aspect.
pub const THUMB_W: usize = 160;
/// Weight of the post-alignment residual in the motion score.
pub const RESIDUAL_WEIGHT: f64 = 0.1;
/// Residual (normalised luma MAD) of two captures of a still scene: codec
/// noise and sensor grain.
pub const RESIDUAL_FLOOR: f64 = 0.04;
/// A frame is blurry below this fraction of the local sharpness reference.
pub const BLUR_RATIO: f64 = 0.4;
/// Motion score under which two consecutive samples count as standing still.
pub const STILL_MOTION: f64 = 0.002;
/// Total motion under which a clip is treated as static (time slices).
pub const STATIC_TOTAL: f64 = 0.05;

/// Zero-mean, unit-variance luma, `w`×`h`, row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct LumaThumb {
    pub w: usize,
    pub h: usize,
    pub px: Vec<f32>,
}

impl LumaThumb {
    /// Samples packed RGB (`width`×`height`) on a 4×4 grid per output pixel.
    pub fn from_rgb(width: u32, height: u32, rgb: &[u8]) -> Self {
        let (sw, sh) = (width as usize, height as usize);
        let w = THUMB_W.min(sw).max(1);
        let h = ((sh * w) as f64 / sw.max(1) as f64).round().max(1.0) as usize;
        let mut px = vec![0f32; w * h];
        for (oy, row) in px.chunks_mut(w).enumerate() {
            for (ox, out) in row.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for sy in 0..4 {
                    let y = ((oy * 4 + sy) * sh / (h * 4)).min(sh - 1);
                    for sx in 0..4 {
                        let x = ((ox * 4 + sx) * sw / (w * 4)).min(sw - 1);
                        let p = &rgb[(y * sw + x) * 3..];
                        acc += 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32;
                    }
                }
                *out = acc / (16.0 * 255.0);
            }
        }
        let n = px.len() as f32;
        let mean = px.iter().sum::<f32>() / n;
        let var = px.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        let inv = 1.0 / var.sqrt().max(1e-3);
        for v in &mut px {
            *v = (*v - mean) * inv;
        }
        Self { w, h, px }
    }

    fn half(&self) -> Self {
        let (w, h) = ((self.w / 2).max(1), (self.h / 2).max(1));
        let mut px = vec![0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let at = |dx: usize, dy: usize| {
                    self.px[((2 * y + dy).min(self.h - 1)) * self.w + (2 * x + dx).min(self.w - 1)]
                };
                px[y * w + x] = 0.25 * (at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1));
            }
        }
        Self { w, h, px }
    }
}

/// Mean absolute difference of `b` shifted by (`dx`, `dy`) against `a` over
/// the overlap; `None` when the overlap is under a quarter of the image.
fn mad(a: &LumaThumb, b: &LumaThumb, dx: i32, dy: i32) -> Option<f64> {
    let (w, h) = (a.w as i32, a.h as i32);
    let (x0, x1) = (dx.max(0), (w + dx).min(w));
    let (y0, y1) = (dy.max(0), (h + dy).min(h));
    if (x1 - x0) * (y1 - y0) * 4 < w * h {
        return None;
    }
    let mut s = 0.0f64;
    for y in y0..y1 {
        let ra = &a.px[(y * w) as usize..];
        let rb = &b.px[((y - dy) * w) as usize..];
        for x in x0..x1 {
            s += (ra[x as usize] - rb[(x - dx) as usize]).abs() as f64;
        }
    }
    Some(s / ((x1 - x0) * (y1 - y0)) as f64)
}

fn best_shift(a: &LumaThumb, b: &LumaThumb, cx: i32, cy: i32, r: i32) -> (i32, i32, f64) {
    let mut best = (cx, cy, f64::INFINITY);
    for dy in cy - r..=cy + r {
        for dx in cx - r..=cx + r {
            if let Some(m) = mad(a, b, dx, dy)
                && m < best.2
            {
                best = (dx, dy, m);
            }
        }
    }
    best
}

/// Global motion between consecutive frames.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Motion {
    /// Content shift from `prev` to `next` in fractions of the width.
    pub dx: f64,
    pub dy: f64,
    /// Mean absolute normalised-luma difference after alignment.
    pub residual: f64,
}

impl Motion {
    pub fn score(&self) -> f64 {
        self.dx.hypot(self.dy) + RESIDUAL_WEIGHT * (self.residual - RESIDUAL_FLOOR).max(0.0)
    }
}

/// Coarse-to-fine global translation between two thumbnails of one size.
pub fn frame_motion(prev: &LumaThumb, next: &LumaThumb) -> Motion {
    if prev.w != next.w || prev.h != next.h || prev.w < 8 || prev.h < 8 {
        return Motion::default();
    }
    let l1 = (prev.half(), next.half());
    let l2 = (l1.0.half(), l1.1.half());
    let (x, y, _) = best_shift(&l2.0, &l2.1, 0, 0, 8);
    let (x, y, _) = best_shift(&l1.0, &l1.1, 2 * x, 2 * y, 2);
    let (x, y, m) = best_shift(prev, next, 2 * x, 2 * y, 2);
    Motion {
        // `b` matched `a` shifted back by (x, y): the content moved by −(x, y).
        dx: -x as f64 / prev.w as f64,
        dy: -y as f64 / prev.w as f64,
        residual: if m.is_finite() { m } else { 2.0 },
    }
}

/// One decoded frame as the selector sees it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scored {
    pub index: u32,
    pub t: f64,
    pub sharpness: f64,
    /// Motion score from the previous decoded frame (0 for the first).
    pub motion: f64,
}

/// Blurry flags: sharpness under [`BLUR_RATIO`] × the 75th percentile of the
/// frames within ±`half` positions.
pub fn blurry_flags(s: &[Scored], half: usize) -> Vec<bool> {
    (0..s.len())
        .map(|i| {
            let lo = i.saturating_sub(half);
            let hi = (i + half + 1).min(s.len());
            let mut w: Vec<f64> = s[lo..hi].iter().map(|f| f.sharpness).collect();
            w.sort_by(f64::total_cmp);
            let p75 = w[(w.len() * 3 / 4).min(w.len() - 1)];
            s[i].sharpness < BLUR_RATIO * p75
        })
        .collect()
}

/// What the selector did, for the capture report.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Selection {
    /// Kept decode indices, ascending.
    pub kept: Vec<u32>,
    pub blurry: usize,
    /// Consecutive pairs with almost no motion (standing still).
    pub still: usize,
    /// Cumulative motion score along the clip.
    pub total_motion: f64,
    /// Candidates dropped as near duplicates of a kept neighbour.
    pub duplicates: usize,
    /// `true` when the clip barely moved and time slices were used.
    pub static_fallback: bool,
}

/// The sharp frame nearest the middle of the widest gap between kept frames
/// (and the clip's ends) at least `margin` from both sides.
fn widest_gap_pick(kept: &[usize], pos: &[f64], blur: &[bool], margin: f64) -> Option<usize> {
    let n = pos.len();
    let first = kept.first().copied().unwrap_or(0);
    let last = kept.last().copied().unwrap_or(n - 1);
    let mut gaps = vec![(0.0, pos[first], 0, first)];
    gaps.extend(kept.windows(2).map(|w| (pos[w[0]], pos[w[1]], w[0], w[1])));
    gaps.push((pos[last], 1.0, last, n - 1));
    let mut pick: Option<(f64, usize)> = None;
    for (lo, hi, a, b) in gaps {
        let gap = hi - lo;
        if pick.is_some_and(|(g, _)| g >= gap) {
            continue;
        }
        let mid = 0.5 * (lo + hi);
        let cand = (a..=b)
            .filter(|&j| !blur[j] && kept.binary_search(&j).is_err())
            .filter(|&j| pos[j] - lo >= margin && hi - pos[j] >= margin)
            .min_by(|&x, &y| (pos[x] - mid).abs().total_cmp(&(pos[y] - mid).abs()));
        if let Some(j) = cand {
            pick = Some((gap, j));
        }
    }
    pick.map(|p| p.1)
}

/// Picks at most `max` keyframes along the camera's path (module docs).
pub fn select_keyframes(s: &[Scored], max: usize) -> Selection {
    let n = s.len();
    let max = max.max(1);
    let mut out = Selection::default();
    if n == 0 {
        return out;
    }
    let blur = blurry_flags(s, 15);
    out.blurry = blur.iter().filter(|&&b| b).count();
    out.still = s.iter().skip(1).filter(|f| f.motion < STILL_MOTION).count();
    let mut c = Vec::with_capacity(n);
    let mut acc = 0.0;
    for (i, f) in s.iter().enumerate() {
        if i > 0 {
            acc += f.motion.max(0.0);
        }
        c.push(acc);
    }
    out.total_motion = acc;
    // Static clip: the clock is the only axis left.
    let static_clip = acc < STATIC_TOTAL;
    out.static_fallback = static_clip;
    let (t0, t1) = (s[0].t, s[n - 1].t);
    let pos: Vec<f64> = if static_clip {
        s.iter().map(|f| (f.t - t0) / (t1 - t0).max(1e-9)).collect()
    } else {
        c.iter().map(|v| v / acc).collect()
    };
    let better = |a: usize, b: usize| {
        // Sharp frames first, then sharper.
        (!blur[a], s[a].sharpness) > (!blur[b], s[b].sharpness)
    };
    let mut best: Vec<Option<usize>> = vec![None; max];
    for (i, &p) in pos.iter().enumerate() {
        let b = ((p * max as f64).floor() as usize).min(max - 1);
        if best[b].is_none_or(|j| better(i, j)) {
            best[b] = Some(i);
        }
    }
    let mut kept: Vec<usize> = best.into_iter().flatten().collect();
    // Near duplicates: a pick within a fifth of a slice of the previous one.
    let dup = 0.2 / max as f64;
    let mut dedup: Vec<usize> = Vec::with_capacity(kept.len());
    for i in kept.drain(..) {
        match dedup.last() {
            Some(&p) if pos[i] - pos[p] < dup => {
                out.duplicates += 1;
                if better(i, p) {
                    *dedup.last_mut().unwrap() = i;
                }
            }
            _ => dedup.push(i),
        }
    }
    kept = dedup;
    // Fill up: split the widest gap that still has a sharp frame well inside;
    // first a fifth of a slice away from both ends, then anything that moved.
    let moved = if static_clip {
        0.0
    } else {
        0.5 * STILL_MOTION / acc
    };
    for margin in [dup, moved] {
        while kept.len() < max.min(n) {
            let Some(j) = widest_gap_pick(&kept, &pos, &blur, margin) else {
                break;
            };
            let at = kept.partition_point(|&k| k < j);
            kept.insert(at, j);
        }
    }
    out.kept = kept.into_iter().map(|i| s[i].index).collect();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A textured image shifted by (`ox`, `oy`) pixels.
    fn scene(w: u32, h: u32, ox: i64, oy: i64) -> Vec<u8> {
        let mut d = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h as i64 {
            for x in 0..w as i64 {
                let (u, v) = ((x - ox) as f64, (y - oy) as f64);
                let l = 128.0
                    + 60.0 * (u * 0.05).sin() * (v * 0.07).cos()
                    + 40.0 * (u * 0.013 + v * 0.021).sin();
                d.extend([l as u8; 3]);
            }
        }
        d
    }

    #[test]
    fn thumb_is_normalised_and_exposure_invariant() {
        let a = scene(640, 360, 0, 0);
        let b: Vec<u8> = a.iter().map(|&v| (v as f32 * 0.8) as u8).collect();
        let (ta, tb) = (
            LumaThumb::from_rgb(640, 360, &a),
            LumaThumb::from_rgb(640, 360, &b),
        );
        assert_eq!((ta.w, ta.h), (160, 90));
        let mean: f32 = ta.px.iter().sum::<f32>() / ta.px.len() as f32;
        assert!(mean.abs() < 1e-4);
        let m = frame_motion(&ta, &tb);
        assert!(m.score() < STILL_MOTION, "{m:?}");
    }

    #[test]
    fn frame_motion_recovers_translation() {
        let a = LumaThumb::from_rgb(640, 360, &scene(640, 360, 0, 0));
        // 40 px at 640 = 10 px at 160 wide: found at the coarsest level.
        let b = LumaThumb::from_rgb(640, 360, &scene(640, 360, 40, -16));
        let m = frame_motion(&a, &b);
        assert!((m.dx - 10.0 / 160.0).abs() < 1.5 / 160.0, "{m:?}");
        assert!((m.dy + 4.0 / 160.0).abs() < 1.5 / 160.0, "{m:?}");
        assert!(m.residual < 0.2, "{m:?}");
    }

    fn clip(motions: &[f64], sharp: impl Fn(usize) -> f64) -> Vec<Scored> {
        motions
            .iter()
            .enumerate()
            .map(|(i, &m)| Scored {
                index: i as u32,
                t: i as f64 * 0.1,
                sharpness: sharp(i),
                motion: if i == 0 { 0.0 } else { m },
            })
            .collect()
    }

    #[test]
    fn standing_still_does_not_take_keyframes() {
        // 30 still samples, then 30 moving ones.
        let m: Vec<f64> = (0..60).map(|i| if i < 30 { 0.0 } else { 0.05 }).collect();
        let s = clip(&m, |_| 1.0);
        let sel = select_keyframes(&s, 10);
        assert_eq!(sel.kept.len(), 10);
        let still = sel.kept.iter().filter(|&&i| i < 30).count();
        assert!(still <= 1, "{:?}", sel.kept);
        assert_eq!(sel.still, 29);
        // Time slices would put half of them in the still part.
    }

    #[test]
    fn blurry_frames_lose_and_static_clips_use_time() {
        let m = vec![0.01; 40];
        let s = clip(&m, |i| if i % 2 == 0 { 0.1 } else { 1.0 });
        let sel = select_keyframes(&s, 8);
        assert!(sel.kept.iter().all(|i| i % 2 == 1), "{:?}", sel.kept);
        assert_eq!(sel.blurry, 20);
        let still = clip(&[0.0; 40], |_| 1.0);
        let sel = select_keyframes(&still, 4);
        assert!(sel.static_fallback);
        assert_eq!(sel.kept, vec![0, 10, 20, 30]);
    }

    #[test]
    fn empty_slices_are_filled_from_the_widest_gaps() {
        // One big jump: most slices land on it and come out empty.
        let mut m = vec![0.002; 50];
        m[25] = 1.0;
        let s = clip(&m, |_| 1.0);
        let sel = select_keyframes(&s, 12);
        assert_eq!(sel.kept.len(), 12, "{:?}", sel.kept);
        assert!(sel.kept.windows(2).all(|w| w[0] < w[1]));
        assert!(sel.kept.iter().any(|&i| i < 25) && sel.kept.iter().any(|&i| i >= 25));
    }
}
