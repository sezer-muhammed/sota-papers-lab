//! Per-triangle atlas: triangles in pairs, each pair one square cell (one
//! triangle per half), a texel of gutter around each. Linear time, used when
//! the chart unwrapper ([`super::uv`]) would be too slow for the triangle
//! count. Every triangle gets the same texel area. Owner: mesh-mode.

use std::time::Instant;

use anyhow::ensure;
use spl_core::Result;

use super::{TriMesh, UvMesh};

/// Unwelded UV mesh (three vertices per triangle) on an `atlas`² page.
pub fn grid_atlas(mesh: &TriMesh, atlas: u32) -> Result<(UvMesh, f64)> {
    let started = Instant::now();
    let tris = mesh.num_triangles();
    ensure!(tris > 0, "empty mesh");
    let cells = tris.div_ceil(2);
    let g = (cells as f64).sqrt().ceil() as u32;
    let c = atlas / g;
    ensure!(
        c >= 4,
        "{tris} triangles need more than a {atlas}² atlas ({c} texel cells); raise atlas_size or lower target_triangles"
    );
    let a = atlas as f32;
    let mut uvm = UvMesh {
        positions: Vec::with_capacity(3 * tris),
        normals: Vec::with_capacity(3 * tris),
        uvs: Vec::with_capacity(3 * tris),
        indices: Vec::with_capacity(3 * tris),
        atlas_width: atlas,
        atlas_height: atlas,
    };
    let (m, d) = (1.0f32, 2.0f32);
    for t in 0..tris {
        let cell = (t / 2) as u32;
        let (x0, y0) = ((cell % g * c) as f32, (cell / g * c) as f32);
        let (x1, y1) = (x0 + c as f32, y0 + c as f32);
        // Lower-left half for even triangles, upper-right for odd; the
        // diagonal gap keeps the two halves' texels apart.
        let uv = if t % 2 == 0 {
            [[x0 + m, y0 + m], [x1 - d - m, y0 + m], [x0 + m, y1 - d - m]]
        } else {
            [[x1 - m, y1 - m], [x0 + d + m, y1 - m], [x1 - m, y0 + d + m]]
        };
        for (k, uvk) in uv.iter().enumerate() {
            let v = mesh.indices[3 * t + k] as usize;
            uvm.positions.push(mesh.positions[v]);
            uvm.normals
                .push(mesh.normals.get(v).copied().unwrap_or([0.0, 0.0, 1.0]));
            uvm.uvs.push([uvk[0] / a, uvk[1] / a]);
            uvm.indices.push((3 * t + k) as u32);
        }
    }
    Ok((uvm, started.elapsed().as_secs_f64() * 1e3))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::tests::cube_mesh;

    #[test]
    fn cells_are_disjoint_and_inside_the_page() {
        let (u, _) = grid_atlas(&cube_mesh(), 64).unwrap();
        assert_eq!(u.indices.len(), 36);
        assert!(
            u.uvs
                .iter()
                .all(|p| p.iter().all(|&x| (0.0..=1.0).contains(&x)))
        );
        // Texel centres covered by two different triangles: none.
        let mut owner = vec![usize::MAX; 64 * 64];
        for t in 0..12 {
            let [a, b, c] = [0, 1, 2].map(|k| u.uvs[3 * t + k].map(|x| x * 64.0));
            for y in 0..64 {
                for x in 0..64 {
                    let p = [x as f32 + 0.5, y as f32 + 0.5];
                    let e = |p0: [f32; 2], p1: [f32; 2]| {
                        (p1[0] - p0[0]) * (p[1] - p0[1]) - (p1[1] - p0[1]) * (p[0] - p0[0])
                    };
                    let (s0, s1, s2) = (e(a, b), e(b, c), e(c, a));
                    let inside = (s0 >= 0.0 && s1 >= 0.0 && s2 >= 0.0)
                        || (s0 <= 0.0 && s1 <= 0.0 && s2 <= 0.0);
                    if inside {
                        assert_eq!(owner[y * 64 + x], usize::MAX, "texel shared");
                        owner[y * 64 + x] = t;
                    }
                }
            }
        }
    }
}
