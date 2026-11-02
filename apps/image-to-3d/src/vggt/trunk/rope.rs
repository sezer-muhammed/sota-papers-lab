//! The two RoPE tables of the trunk (spec §3.2): DINOv3
//! `RopePositionEmbedding` with `normalize_coords="max"` and the periods
//! stored in the checkpoint (the DINO embedder's are bf16-rounded, the
//! aggregator's fp32), special-token prefix unrotated.

use std::f64::consts::PI;

use spl_nn::qkv::{PAIRS, RopeLayout, RopeTable};

/// Table for a `grid_h × grid_w` patch grid behind `prefix` special tokens.
/// Pair `j < 16` turns by the row coordinate with `periods[j]`, pair
/// `j >= 16` by the column with `periods[j - 16]`; both coordinates are
/// `2 (i + 0.5) / max(h, w) - 1`.
pub fn table(grid_h: usize, grid_w: usize, prefix: usize, periods: &[f32]) -> RopeTable {
    assert_eq!(periods.len(), PAIRS / 2, "RoPE needs 16 periods");
    let m = grid_h.max(grid_w) as f32;
    RopeTable::from_angles(RopeLayout::Dinov3, prefix + grid_h * grid_w, |t, p| {
        if t < prefix {
            return 0.0;
        }
        let (r, c) = ((t - prefix) / grid_w, (t - prefix) % grid_w);
        let i = if p < 16 { r } else { c };
        // fp32 like the official module: arange(0.5, n) / max, then 2x - 1.
        let coord = 2.0 * ((i as f32 + 0.5) / m) - 1.0;
        let angle = 2.0 * PI as f32 * coord / periods[p % 16];
        angle as f64
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_is_unrotated_and_axes_use_max_normalisation() {
        let periods: Vec<f32> = (0..16).map(|j| 100f32.powf(j as f32 / 16.0)).collect();
        let t = table(2, 4, 17, &periods);
        assert_eq!(t.frame_len, 17 + 8);
        assert!(
            t.cos_sin[..17 * PAIRS]
                .iter()
                .all(|&[c, s]| c == 1.0 && s == 0.0)
        );
        // Patch (row 1, col 3): y = 2·1.5/4 − 1 = −0.25, x = 2·3.5/4 − 1 = 0.75.
        let row = &t.cos_sin[(17 + 7) * PAIRS..(17 + 8) * PAIRS];
        let want_y = (2.0 * std::f64::consts::PI * -0.25f64).sin();
        let want_x = (2.0 * std::f64::consts::PI * 0.75 / periods[1] as f64).sin();
        assert!((row[0][1] as f64 - want_y).abs() < 1e-6);
        assert!((row[17][1] as f64 - want_x).abs() < 1e-5);
    }
}
