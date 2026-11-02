//! Sinusoidal 2D position embedding of the dense head (owner: heads;
//! `heads/utils.py::create_uv_grid` + `position_grid_to_embed`, spec §5.1).

/// `PE(c, h, w)` for an image of aspect `width / height`, laid out
/// `[h·w][c]` (NHWC rows) and multiplied by `ratio` (0.1 in the model).
/// Channel blocks of `c/4`: `sin(u ω), cos(u ω), sin(v ω), cos(v ω)` with
/// `u` along the width, `ω_j = 100^(−j/(c/4))`. The grid coordinates are
/// rounded to f32 like torch's `linspace(dtype=float32)`; the products and
/// sin/cos run in f64 and round to f32 once.
pub fn table(c: usize, h: usize, w: usize, aspect: f64, ratio: f32) -> Vec<f32> {
    let q = c / 4;
    let diag = (aspect * aspect + 1.0).sqrt();
    let (sx, sy) = (aspect / diag, 1.0 / diag);
    let lin = |span: f64, n: usize| -> Vec<f64> {
        let lo = -span * (n as f64 - 1.0) / n as f64;
        let hi = -lo;
        (0..n)
            .map(|i| {
                let t = if n > 1 {
                    lo + (hi - lo) * i as f64 / (n as f64 - 1.0)
                } else {
                    lo
                };
                t as f32 as f64
            })
            .collect()
    };
    let (u, v) = (lin(sx, w), lin(sy, h));
    let omega: Vec<f64> = (0..q)
        .map(|j| 1.0 / 100f64.powf(j as f64 / q as f64))
        .collect();
    let mut out = vec![0f32; h * w * c];
    for r in 0..h {
        for col in 0..w {
            let row = &mut out[(r * w + col) * c..(r * w + col + 1) * c];
            for j in 0..q {
                let a = u[col] * omega[j];
                let b = v[r] * omega[j];
                row[j] = a.sin() as f32 * ratio;
                row[q + j] = a.cos() as f32 * ratio;
                row[2 * q + j] = b.sin() as f32 * ratio;
                row[3 * q + j] = b.cos() as f32 * ratio;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corners_and_blocks() {
        // 1:1 image, 2×2 grid, c = 8: u = v = ±(1/√2)/2.
        let t = table(8, 2, 2, 1.0, 1.0);
        let s = (0.5f64.sqrt() * 0.5) as f32 as f64;
        // pixel (0, 1): u = +s, v = −s; j = 0 has ω = 1.
        let p = &t[8..16];
        assert!((p[0] as f64 - s.sin()).abs() < 1e-6);
        assert!((p[2] as f64 - s.cos()).abs() < 1e-6);
        assert!((p[4] as f64 + s.sin()).abs() < 1e-6);
        // j = 1: ω = 100^(−1/2) = 0.1.
        assert!((p[1] as f64 - (0.1 * s).sin()).abs() < 1e-6);
    }
}
