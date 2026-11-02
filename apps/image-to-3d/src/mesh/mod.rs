//! Stage D mesh pipeline: simplify, UV unwrap, texture bake, GLB export.
//! Owner: mesh-tex agent. Files: `mesh/{mod,simplify,uv,bake,glb}.rs`,
//! `cuda/mesh_bake*.cu`.

pub mod bake;
pub mod glb;
pub mod simplify;
pub mod uv;

/// Triangle mesh: indexed positions with optional per-vertex normals.
#[derive(Clone, Debug)]
pub struct TriMesh {
    pub positions: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
    pub normals: Vec<[f32; 3]>,
}

impl TriMesh {
    pub fn num_vertices(&self) -> usize {
        self.positions.len()
    }

    pub fn num_triangles(&self) -> usize {
        self.indices.len() / 3
    }

    /// Compute per-vertex normals from face normals (area-weighted).
    pub fn compute_normals(&mut self) {
        let n = self.positions.len();
        self.normals = vec![[0.0; 3]; n];
        for tri in self.indices.as_chunks::<3>().0 {
            let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            let v0 = self.positions[i0];
            let v1 = self.positions[i1];
            let v2 = self.positions[i2];
            let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
            let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
            let nx = e1[1] * e2[2] - e1[2] * e2[1];
            let ny = e1[2] * e2[0] - e1[0] * e2[2];
            let nz = e1[0] * e2[1] - e1[1] * e2[0];
            for &idx in &[i0, i1, i2] {
                self.normals[idx][0] += nx;
                self.normals[idx][1] += ny;
                self.normals[idx][2] += nz;
            }
        }
        for n in &mut self.normals {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if len > 1e-12 {
                n[0] /= len;
                n[1] /= len;
                n[2] /= len;
            }
        }
    }
}

/// UV-mapped mesh ready for texture baking.
#[derive(Clone, Debug)]
pub struct UvMesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
    pub atlas_width: u32,
    pub atlas_height: u32,
}

/// Camera for texture baking: world-to-camera pose and pinhole intrinsics.
#[derive(Clone, Debug)]
pub struct BakeCamera {
    pub r_cw: [[f32; 3]; 3],
    pub t_cw: [f32; 3],
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub width: u32,
    pub height: u32,
}

/// Baked texture atlas (RGBA u8, row-major, top-left origin).
#[derive(Clone, Debug)]
pub struct TextureAtlas {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

impl TextureAtlas {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            pixels: vec![0; (width * height * 4) as usize],
        }
    }
}

/// Full mesh export pipeline result.
#[derive(Debug)]
pub struct MeshExport {
    pub glb_bytes: Vec<u8>,
    pub stats: MeshStats,
}

#[derive(Clone, Debug, Default)]
pub struct MeshStats {
    pub input_triangles: usize,
    pub simplified_triangles: usize,
    pub atlas_width: u32,
    pub atlas_height: u32,
    pub bake_views_used: usize,
    pub simplify_ms: f64,
    pub uv_ms: f64,
    pub bake_ms: f64,
    pub glb_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    pub fn cube_mesh() -> TriMesh {
        #[rustfmt::skip]
        let positions: Vec<[f32; 3]> = vec![
            [-1.0, -1.0, -1.0], [ 1.0, -1.0, -1.0], [ 1.0,  1.0, -1.0], [-1.0,  1.0, -1.0],
            [-1.0, -1.0,  1.0], [ 1.0, -1.0,  1.0], [ 1.0,  1.0,  1.0], [-1.0,  1.0,  1.0],
        ];
        #[rustfmt::skip]
        let indices: Vec<u32> = vec![
            0, 1, 2,  0, 2, 3,  // front
            4, 6, 5,  4, 7, 6,  // back
            0, 4, 5,  0, 5, 1,  // bottom
            2, 6, 7,  2, 7, 3,  // top
            0, 3, 7,  0, 7, 4,  // left
            1, 5, 6,  1, 6, 2,  // right
        ];
        let mut m = TriMesh {
            positions,
            indices,
            normals: vec![],
        };
        m.compute_normals();
        m
    }

    #[test]
    fn cube_normals_are_unit_length() {
        let m = cube_mesh();
        for n in &m.normals {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-6, "normal length {len}");
        }
    }

    #[test]
    fn cube_triangle_count() {
        let m = cube_mesh();
        assert_eq!(m.num_triangles(), 12);
        assert_eq!(m.num_vertices(), 8);
    }
}
