//! UV atlas generation via uvunwrap (MIT, pure Rust). Owner: mesh-tex.

use std::time::Instant;

use spl_core::Result;

use super::{TriMesh, UvMesh};

/// Generate a UV atlas for the given triangle mesh.
///
/// `atlas_size` is the target page size in texels (e.g. 1024, 2048).
/// `padding` is the texel gap between charts (default 2).
pub fn generate_atlas(mesh: &TriMesh, atlas_size: u32, padding: u32) -> Result<(UvMesh, f64)> {
    let started = Instant::now();

    let input = uvunwrap::MeshInput::new(&mesh.positions, &mesh.indices);
    let options = uvunwrap::Options {
        density: uvunwrap::Density::Auto {
            page_size: atlas_size,
        },
        padding,
        max_chart_angle: 70.0,
        max_area_distortion: 4.0,
        packing: uvunwrap::Packing::Shape,
        ..uvunwrap::Options::default()
    };

    let atlas = uvunwrap::generate(&[input], &options)
        .map_err(|e| spl_core::anyhow!("UV unwrap failed: {e}"))?;

    let out = &atlas.meshes[0];

    let mut positions = Vec::with_capacity(out.vertices.len());
    let mut normals = Vec::with_capacity(out.vertices.len());
    let mut uvs = Vec::with_capacity(out.vertices.len());

    for v in &out.vertices {
        let xref = v.xref as usize;
        positions.push(mesh.positions[xref]);
        if xref < mesh.normals.len() {
            normals.push(mesh.normals[xref]);
        } else {
            normals.push([0.0, 0.0, 1.0]);
        }
        uvs.push(v.uv);
    }

    let uv_mesh = UvMesh {
        positions,
        normals,
        uvs,
        indices: out.indices.clone(),
        atlas_width: atlas.width,
        atlas_height: atlas.height,
    };

    let ms = started.elapsed().as_secs_f64() * 1e3;
    Ok((uv_mesh, ms))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::tests::cube_mesh;

    #[test]
    fn atlas_generates_valid_uvs() {
        let mesh = cube_mesh();
        let (uv_mesh, ms) = generate_atlas(&mesh, 256, 2).unwrap();

        assert!(!uv_mesh.positions.is_empty());
        assert!(!uv_mesh.uvs.is_empty());
        assert_eq!(uv_mesh.positions.len(), uv_mesh.uvs.len());
        assert_eq!(uv_mesh.positions.len(), uv_mesh.normals.len());
        assert_eq!(uv_mesh.indices.len() % 3, 0);
        assert!(uv_mesh.atlas_width > 0);
        assert!(uv_mesh.atlas_height > 0);
        assert!(ms >= 0.0);

        for uv in &uv_mesh.uvs {
            assert!(
                uv[0] >= 0.0 && uv[0] <= 1.0 && uv[1] >= 0.0 && uv[1] <= 1.0,
                "UV out of [0,1]: {:?}",
                uv
            );
        }
    }

    #[test]
    fn atlas_triangle_count_matches() {
        let mesh = cube_mesh();
        let (uv_mesh, _) = generate_atlas(&mesh, 256, 2).unwrap();
        assert_eq!(
            uv_mesh.indices.len() / 3,
            mesh.indices.len() / 3,
            "triangle count must be preserved"
        );
    }
}
