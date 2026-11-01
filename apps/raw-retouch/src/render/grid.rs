//! The base layer that Highlights and Shadows act on: log2 luminance
//! smoothed by a bilateral grid (Chen, Paris & Durand 2007). Smoothing
//! averages only pixels of similar luminance, so the base follows strong
//! edges (no halos) while texture stays in the detail. The grid lives in
//! normalized image coordinates, so one grid built from the display image
//! slices identically at search, display and full resolution. It is built
//! on the CPU, deterministically, and sliced on the GPU (`slice_base`).

use super::layout::{GRID_DZ, GRID_PAD, GRID_Z0, GRID_Z1};
use crate::types::LinearImage;

/// Grid cells across the long edge: the spatial scale of the base layer
/// (about 3% of the long edge after the grid blur).
pub(crate) const CELLS_LONG: f32 = 40.0;

pub(crate) struct BaseGrid {
    pub(crate) nx: usize,
    pub(crate) ny: usize,
    pub(crate) nz: usize,
    /// (weighted log2 luminance, weight) per node, x fastest, then y, then z.
    pub(crate) data: Vec<f32>,
    /// 1st and 99th percentiles of the image's log2 luminance: its dark and bright ends.
    pub(crate) p01: f32,
    pub(crate) p99: f32,
}

/// Bins per EV of the luminance histogram behind `p01`/`p99`.
const HIST_PER_EV: f32 = 16.0;

impl BaseGrid {
    /// Grid cells per pixel for an image of this size.
    pub(crate) fn cells_per_px(width: usize, height: usize) -> f32 {
        CELLS_LONG / width.max(height).max(1) as f32
    }

    /// `y_row` maps camera RGB to luminance (Rec.2020 Y at as-shot white balance).
    pub(crate) fn build(img: &LinearImage, y_row: [f32; 3]) -> Self {
        let s = Self::cells_per_px(img.width, img.height);
        let long = img.width.max(img.height).max(1) as f64;
        let cells =
            |n: usize| (n as f64 * CELLS_LONG as f64 / long - 1e-6).ceil().max(0.0) as usize;
        let nx = cells(img.width) + 1 + 2 * GRID_PAD;
        let ny = cells(img.height) + 1 + 2 * GRID_PAD;
        let nz = ((GRID_Z1 - GRID_Z0) / GRID_DZ).ceil() as usize + 1 + 2 * GRID_PAD;
        let mut data = vec![0.0f32; 2 * nx * ny * nz];
        let mut hist = vec![0u32; ((GRID_Z1 - GRID_Z0) * HIST_PER_EV) as usize + 1];
        // Each pixel weighs one cell's share, so a full cell sums to about 1
        // at any resolution and GRID_PRIOR means the same everywhere.
        let px_weight = s * s;
        let pad = GRID_PAD as f32;
        for y in 0..img.height {
            let gy = (y as f32 + 0.5) * s + pad;
            let iy = gy as usize;
            let fy = gy - iy as f32;
            let row = &img.data[3 * y * img.width..3 * (y + 1) * img.width];
            for (x, c) in row.as_chunks::<3>().0.iter().enumerate() {
                let lum = y_row[0] * c[0] + y_row[1] * c[1] + y_row[2] * c[2];
                let l = log2_lum(lum);
                hist[((l - GRID_Z0) * HIST_PER_EV) as usize] += 1;
                let gx = (x as f32 + 0.5) * s + pad;
                let gz = (l - GRID_Z0) / GRID_DZ + pad;
                let (ix, iz) = (gx as usize, gz as usize);
                let (fx, fz) = (gx - ix as f32, gz - iz as f32);
                for (dz, wz) in [(0, 1.0 - fz), (1, fz)] {
                    for (dy, wy) in [(0, 1.0 - fy), (1, fy)] {
                        let base = ((iz + dz) * ny + iy + dy) * nx + ix;
                        let w0 = px_weight * wz * wy * (1.0 - fx);
                        let w1 = px_weight * wz * wy * fx;
                        data[2 * base] += w0 * l;
                        data[2 * base + 1] += w0;
                        data[2 * base + 2] += w1 * l;
                        data[2 * base + 3] += w1;
                    }
                }
            }
        }
        let total = (img.width * img.height) as f64;
        let quantile = |q: f64| {
            let mut below = 0.0;
            for (i, &n) in hist.iter().enumerate() {
                let n = n as f64;
                if n > 0.0 && below + n >= q * total {
                    let within = ((q * total - below) / n).clamp(0.0, 1.0) as f32;
                    return GRID_Z0 + (i as f32 + within) / HIST_PER_EV;
                }
                below += n;
            }
            GRID_Z1
        };
        let (p01, p99) = (quantile(0.01), quantile(0.99));
        let mut grid = Self {
            nx,
            ny,
            nz,
            data,
            p01,
            p99,
        };
        grid.blur(1);
        grid.blur(nx);
        grid.blur(nx * ny);
        grid
    }

    /// [1 4 6 4 1] / 16 along one axis (`stride` in nodes), zero outside.
    fn blur(&mut self, stride: usize) {
        const K: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];
        let n = self.nx * self.ny * self.nz;
        let len = match stride {
            1 => self.nx,
            s if s == self.nx => self.ny,
            _ => self.nz,
        };
        let src = self.data.clone();
        for node in 0..n {
            let pos = (node / stride) % len;
            let (mut s, mut w) = (0.0, 0.0);
            for (k, kw) in K.iter().enumerate() {
                let p = pos as isize + k as isize - 2;
                if p < 0 || p >= len as isize {
                    continue;
                }
                let j = (node as isize + (k as isize - 2) * stride as isize) as usize;
                s += kw * src[2 * j];
                w += kw * src[2 * j + 1];
            }
            self.data[2 * node] = s;
            self.data[2 * node + 1] = w;
        }
    }

    /// CPU mirror of the kernels' `slice_base`.
    #[cfg(test)]
    pub(crate) fn slice(&self, gx: f32, gy: f32, l: f32) -> f32 {
        use super::layout::GRID_PRIOR;
        let pad = GRID_PAD as f32;
        let (gx, gy, gz) = (gx + pad, gy + pad, (l - GRID_Z0) / GRID_DZ + pad);
        let ix = (gx.floor() as usize).min(self.nx - 2);
        let iy = (gy.floor() as usize).min(self.ny - 2);
        let iz = (gz.floor().max(0.0) as usize).min(self.nz - 2);
        let (fx, fy, fz) = (gx - ix as f32, gy - iy as f32, gz - iz as f32);
        let (mut s, mut w) = (0.0, 0.0);
        for (dz, wz) in [(0, 1.0 - fz), (1, fz)] {
            for (dy, wy) in [(0, 1.0 - fy), (1, fy)] {
                for (dx, wx) in [(0, 1.0 - fx), (1, fx)] {
                    let n = ((iz + dz) * self.ny + iy + dy) * self.nx + ix + dx;
                    s += wz * wy * wx * self.data[2 * n];
                    w += wz * wy * wx * self.data[2 * n + 1];
                }
            }
        }
        (s + GRID_PRIOR * l) / (w + GRID_PRIOR)
    }
}

/// log2 luminance clamped to the grid's range (as the kernels do).
pub(crate) fn log2_lum(lum: f32) -> f32 {
    lum.max(1e-30).log2().clamp(GRID_Z0, GRID_Z1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(w: usize, h: usize, f: impl Fn(usize, usize) -> f32) -> LinearImage {
        let mut data = Vec::with_capacity(3 * w * h);
        for y in 0..h {
            for x in 0..w {
                let v = f(x, y);
                data.extend([v, v, v]);
            }
        }
        LinearImage {
            width: w,
            height: h,
            data,
        }
    }

    const Y: [f32; 3] = [0.25, 0.5, 0.25];

    #[test]
    fn flat_image_has_flat_base() {
        let img = image(300, 200, |_, _| 0.1);
        let g = BaseGrid::build(&img, Y);
        assert!((g.p01 - 0.1f32.log2()).abs() < 0.07 && (g.p99 - 0.1f32.log2()).abs() < 0.07);
        let s = BaseGrid::cells_per_px(300, 200);
        for (x, y) in [(0, 0), (150, 100), (299, 199), (10, 190)] {
            let b = g.slice((x as f32 + 0.5) * s, (y as f32 + 0.5) * s, 0.1f32.log2());
            assert!((b - 0.1f32.log2()).abs() < 1e-4, "{x},{y}: {b}");
        }
    }

    /// A 5 EV step: on each side, right up to the edge, the base stays on
    /// that side's level instead of blending across (no halo).
    #[test]
    fn strong_edges_stay_sharp() {
        let (w, h) = (400, 200);
        let img = image(w, h, |x, _| if x < 200 { 0.01 } else { 0.32 });
        let g = BaseGrid::build(&img, Y);
        assert!((g.p01 - 0.01f32.log2()).abs() < 0.07 && (g.p99 - 0.32f32.log2()).abs() < 0.07);
        let s = BaseGrid::cells_per_px(w, h);
        for x in [150usize, 195, 199, 200, 204, 250] {
            let v = if x < 200 { 0.01f32 } else { 0.32 };
            let b = g.slice((x as f32 + 0.5) * s, 100.5 * s, v.log2());
            assert!(
                (b - v.log2()).abs() < 0.05,
                "x={x}: base {b} vs {}",
                v.log2()
            );
        }
    }

    /// Texture smaller than the range kernel goes mostly to the detail layer.
    #[test]
    fn fine_texture_is_detail() {
        let (w, h) = (400, 300);
        let img = image(w, h, |x, y| {
            if (x / 2 + y / 2) % 2 == 0 {
                0.1
            } else {
                0.1 * 1.4
            }
        });
        let g = BaseGrid::build(&img, Y);
        let s = BaseGrid::cells_per_px(w, h);
        let lo = g.slice(200.5 * s, 150.5 * s, 0.1f32.log2());
        let hi = g.slice(202.5 * s, 150.5 * s, (0.14f32).log2());
        // The texture is 0.49 EV peak to peak; the base moves far less.
        assert!((hi - lo).abs() < 0.15, "base follows texture: {lo} {hi}");
    }

    #[test]
    fn same_geometry_at_every_resolution() {
        let small = image(150, 100, |x, _| if x < 75 { 0.02 } else { 0.2 });
        let big = image(600, 400, |x, _| if x < 300 { 0.02 } else { 0.2 });
        let (gs, gb) = (BaseGrid::build(&small, Y), BaseGrid::build(&big, Y));
        assert_eq!((gs.nx, gs.ny, gs.nz), (gb.nx, gb.ny, gb.nz));
        let (ss, sb) = (
            BaseGrid::cells_per_px(150, 100),
            BaseGrid::cells_per_px(600, 400),
        );
        for (xs, xb) in [(10.0f32, 40.0f32), (70.0, 280.0), (80.0, 320.0)] {
            let v = if xs < 75.0 { 0.02f32 } else { 0.2 };
            let a = gs.slice((xs + 0.5) * ss, 50.5 * ss, v.log2());
            let b = gb.slice((xb + 1.5) * sb, 201.5 * sb, v.log2());
            assert!((a - b).abs() < 0.05, "{xs}: {a} vs {b}");
        }
    }
}
