//! Stage D geometry: GPU TSDF fusion of depth maps into a sparse voxel hash
//! ([`tsdf`]) and GPU marching cubes with vertex welding ([`mc`]), producing
//! a [`Mesh`]. Owner: mesh-tsdf.
//!
//! Conventions follow the 2DGS mesher (Open3D `ScalableTSDFVolume`, MIT):
//! projective signed distance `d(pixel) − z_cam`, truncation `trunc`
//! (2DGS uses 5 voxels), voxels further than `trunc` behind the observed
//! surface are not updated, every observation has weight 1 and the running
//! average is capped at `max_weight`. The depth comes from the splats' median
//! depth (GGGS stochastic-solid T = 0.5 crossing) once geometry training
//! lands; tests use synthetic and COLMAP-derived depth.

pub mod mc;
pub mod tsdf;

/// Indexed triangle mesh. Faces wind counter-clockwise seen from outside
/// (the side the TSDF is positive); normals are unit TSDF gradients.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub faces: Vec<[u32; 3]>,
}

pub(crate) fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub(crate) fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

impl Mesh {
    pub fn is_empty(&self) -> bool {
        self.faces.is_empty()
    }

    /// Drops vertices no face references and renumbers the faces.
    pub fn remove_unreferenced(&mut self) {
        let mut map = vec![u32::MAX; self.positions.len()];
        let mut n = 0u32;
        for f in &mut self.faces {
            for v in f.iter_mut() {
                let m = &mut map[*v as usize];
                if *m == u32::MAX {
                    *m = n;
                    n += 1;
                }
                *v = *m;
            }
        }
        let mut pos = vec![[0.0; 3]; n as usize];
        let mut nrm = vec![
            [0.0; 3];
            if self.normals.is_empty() {
                0
            } else {
                n as usize
            }
        ];
        for (old, &new) in map.iter().enumerate() {
            if new != u32::MAX {
                pos[new as usize] = self.positions[old];
                if !nrm.is_empty() {
                    nrm[new as usize] = self.normals[old];
                }
            }
        }
        self.positions = pos;
        self.normals = nrm;
    }

    /// Sorts the faces (rotated so the smallest index leads) and renumbers
    /// vertices in first-use order: a canonical form, so GPU output
    /// (atomic, unordered) compares equal across runs.
    pub fn canonicalize(&mut self) {
        for f in &mut self.faces {
            let k = (0..3).min_by_key(|&i| f[i]).unwrap_or(0);
            f.rotate_left(k);
        }
        let key = |f: &[u32; 3], p: &[[f32; 3]]| {
            let c = p[f[0] as usize];
            (c[0].to_bits(), c[1].to_bits(), c[2].to_bits())
        };
        let p = self.positions.clone();
        self.faces.sort_by_key(|f| key(f, &p));
        self.positions.clear();
        let nrm = std::mem::take(&mut self.normals);
        let mut map = vec![u32::MAX; p.len()];
        for f in &mut self.faces {
            for v in f.iter_mut() {
                if map[*v as usize] == u32::MAX {
                    map[*v as usize] = self.positions.len() as u32;
                    self.positions.push(p[*v as usize]);
                    if !nrm.is_empty() {
                        self.normals.push(nrm[*v as usize]);
                    }
                }
                *v = map[*v as usize];
            }
        }
    }

    /// Signed volume (positive for a closed outward-wound surface).
    pub fn signed_volume(&self) -> f64 {
        self.faces
            .iter()
            .map(|f| {
                let [a, b, c] = f.map(|i| self.positions[i as usize].map(f64::from));
                (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                    + a[2] * (b[0] * c[1] - b[1] * c[0]))
                    / 6.0
            })
            .sum()
    }

    pub fn area(&self) -> f64 {
        self.faces.iter().map(|f| self.face_area(f) as f64).sum()
    }

    fn face_area(&self, f: &[u32; 3]) -> f32 {
        let [a, b, c] = f.map(|i| self.positions[i as usize]);
        let n = cross(sub(b, a), sub(c, a));
        0.5 * (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt()
    }

    /// Undirected edges used by other than exactly two faces in opposite
    /// directions (0 for a closed, consistently oriented 2-manifold).
    pub fn boundary_or_nonmanifold_edges(&self) -> usize {
        use std::collections::HashMap;
        let mut e: HashMap<(u32, u32), i32> = HashMap::new();
        for f in &self.faces {
            for k in 0..3 {
                let (a, b) = (f[k], f[(k + 1) % 3]);
                let (key, d) = if a < b { ((a, b), 1) } else { ((b, a), 1 << 8) };
                *e.entry(key).or_default() += d;
            }
        }
        e.values().filter(|&&v| v != 1 + (1 << 8)).count()
    }

    /// `n` points uniformly by area (deterministic for a given `seed`).
    pub fn sample(&self, n: usize, seed: u64) -> Vec<[f32; 3]> {
        let mut cdf = Vec::with_capacity(self.faces.len());
        let mut acc = 0.0f64;
        for f in &self.faces {
            acc += self.face_area(f) as f64;
            cdf.push(acc);
        }
        if acc <= 0.0 {
            return Vec::new();
        }
        let mut rng = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..n)
            .map(|_| {
                let t = next() * acc;
                let i = cdf.partition_point(|&c| c < t).min(cdf.len() - 1);
                let [a, b, c] = self.faces[i].map(|v| self.positions[v as usize]);
                let (mut u, mut v) = (next() as f32, next() as f32);
                if u + v > 1.0 {
                    (u, v) = (1.0 - u, 1.0 - v);
                }
                std::array::from_fn(|k| a[k] + u * (b[k] - a[k]) + v * (c[k] - a[k]))
            })
            .collect()
    }
}

/// Symmetric Chamfer distance between point sets (mean of the two mean
/// nearest-neighbour distances), plus each direction: `(chamfer, a→b, b→a)`.
pub fn chamfer(a: &[[f32; 3]], b: &[[f32; 3]]) -> (f64, f64, f64) {
    let ab = mean_nn(a, b);
    let ba = mean_nn(b, a);
    (0.5 * (ab + ba), ab, ba)
}

/// Mean distance from each point of `q` to its nearest point of `p`
/// (uniform-grid search).
pub fn mean_nn(q: &[[f32; 3]], p: &[[f32; 3]]) -> f64 {
    use rayon::prelude::*;
    use std::collections::HashMap;
    if q.is_empty() || p.is_empty() {
        return f64::INFINITY;
    }
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for x in p {
        for k in 0..3 {
            lo[k] = lo[k].min(x[k]);
            hi[k] = hi[k].max(x[k]);
        }
    }
    let vol: f32 = (0..3).map(|k| (hi[k] - lo[k]).max(1e-6)).product();
    let cell = (vol / p.len() as f32).cbrt().max(1e-6) * 2.0;
    let key = |x: &[f32; 3]| -> [i32; 3] {
        std::array::from_fn(|k| ((x[k] - lo[k]) / cell).floor() as i32)
    };
    let mut grid: HashMap<[i32; 3], Vec<u32>> = HashMap::new();
    for (i, x) in p.iter().enumerate() {
        grid.entry(key(x)).or_default().push(i as u32);
    }
    let total: f64 = q
        .par_iter()
        .map(|x| {
            let c = key(x);
            let mut best = f32::MAX;
            let mut r = 0i32;
            loop {
                for dx in -r..=r {
                    for dy in -r..=r {
                        for dz in -r..=r {
                            if dx.abs().max(dy.abs()).max(dz.abs()) != r {
                                continue;
                            }
                            if let Some(v) = grid.get(&[c[0] + dx, c[1] + dy, c[2] + dz]) {
                                for &i in v {
                                    let d = sub(*x, p[i as usize]);
                                    best = best.min(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
                                }
                            }
                        }
                    }
                }
                // Every point outside the searched shell is at least r·cell away.
                if best.sqrt() <= r as f32 * cell || r > 4096 {
                    break;
                }
                r += 1;
            }
            best.sqrt() as f64
        })
        .sum();
    total / q.len() as f64
}
