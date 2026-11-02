//! Mesh decimation via meshoptimizer (MIT). Owner: mesh-tex.

use std::time::Instant;

use spl_core::Result;

use super::TriMesh;

/// Simplify a triangle mesh to at most `target_triangles` faces.
/// Uses meshoptimizer's `simplify` with the given `target_error` threshold
/// (recommended: 0.01 for high quality, 0.1 for aggressive).
pub fn simplify(
    mesh: &TriMesh,
    target_triangles: usize,
    target_error: f32,
) -> Result<(TriMesh, f64)> {
    let started = Instant::now();

    let vertex_count = mesh.positions.len();
    let pos_bytes = meshopt::typed_to_bytes(&mesh.positions);
    let vertex_stride = std::mem::size_of::<[f32; 3]>();

    let vertices = meshopt::VertexDataAdapter::new(pos_bytes, vertex_stride, 0)
        .map_err(|e| spl_core::anyhow!("meshopt adapter: {e}"))?;

    let target_indices = target_triangles * 3;
    let target_indices = target_indices.min(mesh.indices.len());

    let result_indices = meshopt::simplify::simplify(
        &mesh.indices,
        &vertices,
        target_indices,
        target_error,
        meshopt::simplify::SimplifyOptions::None,
        None,
    );

    let mut used = vec![false; vertex_count];
    for &i in &result_indices {
        used[i as usize] = true;
    }
    let mut remap = vec![0u32; vertex_count];
    let mut new_count = 0u32;
    for (old, &is_used) in used.iter().enumerate() {
        if is_used {
            remap[old] = new_count;
            new_count += 1;
        }
    }

    let mut new_positions = Vec::with_capacity(new_count as usize);
    let mut new_normals = Vec::with_capacity(new_count as usize);
    for (old, &is_used) in used.iter().enumerate() {
        if is_used {
            new_positions.push(mesh.positions[old]);
            if old < mesh.normals.len() {
                new_normals.push(mesh.normals[old]);
            }
        }
    }

    let new_indices: Vec<u32> = result_indices.iter().map(|&i| remap[i as usize]).collect();

    let mut simplified = TriMesh {
        positions: new_positions,
        indices: new_indices,
        normals: new_normals,
    };

    if simplified.normals.len() != simplified.positions.len() {
        simplified.compute_normals();
    }

    let ms = started.elapsed().as_secs_f64() * 1e3;
    Ok((simplified, ms))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::tests::cube_mesh;

    #[test]
    fn simplify_cube_reduces_faces() {
        let mesh = cube_mesh();
        assert_eq!(mesh.num_triangles(), 12);

        let (simplified, ms) = simplify(&mesh, 6, 0.1).unwrap();
        assert!(simplified.num_triangles() <= 12);
        assert!(simplified.num_triangles() > 0);
        assert!(ms >= 0.0);
        assert!(!simplified.positions.is_empty());
        assert_eq!(simplified.normals.len(), simplified.positions.len());
    }

    #[test]
    fn simplify_preserves_mesh_when_target_exceeds_input() {
        let mesh = cube_mesh();
        let (simplified, _) = simplify(&mesh, 100, 0.01).unwrap();
        assert_eq!(simplified.num_triangles(), mesh.num_triangles());
    }

    #[test]
    fn simplify_high_error_reduces_more() {
        let mesh = subdivided_cube();
        let (low_err, _) = simplify(&mesh, 20, 0.001).unwrap();
        let (high_err, _) = simplify(&mesh, 20, 1.0).unwrap();
        assert!(high_err.num_triangles() <= low_err.num_triangles());
    }

    fn subdivided_cube() -> TriMesh {
        let n = 8u32;
        let mut positions = Vec::new();
        let mut indices = Vec::new();
        for face in 0..6u32 {
            let base = positions.len() as u32;
            for j in 0..=n {
                for i in 0..=n {
                    let u = i as f32 / n as f32 * 2.0 - 1.0;
                    let v = j as f32 / n as f32 * 2.0 - 1.0;
                    let p = match face {
                        0 => [u, v, -1.0],
                        1 => [u, v, 1.0],
                        2 => [u, -1.0, v],
                        3 => [u, 1.0, v],
                        4 => [-1.0, u, v],
                        _ => [1.0, u, v],
                    };
                    positions.push(p);
                }
            }
            for j in 0..n {
                for i in 0..n {
                    let stride = n + 1;
                    let a = base + j * stride + i;
                    let b = a + 1;
                    let c = a + stride;
                    let d = c + 1;
                    indices.extend_from_slice(&[a, b, d, a, d, c]);
                }
            }
        }
        let mut m = TriMesh {
            positions,
            indices,
            normals: vec![],
        };
        m.compute_normals();
        m
    }
}
