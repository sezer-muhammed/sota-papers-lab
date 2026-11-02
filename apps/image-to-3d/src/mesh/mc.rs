//! Marching cubes: the case tables (generated, not copied), a CPU reference
//! over a dense grid, and the GPU extractor over the sparse TSDF hash.
//! Owner: mesh-tsdf.
//!
//! The tables are built at start-up by tracing, per corner configuration, the
//! iso-segments on each cube face and linking them into loops. On a face with
//! four sign changes (the ambiguous case) the segments always cut off the
//! *outside* corners, a rule that depends on the face's four values only, so
//! the two cubes sharing the face agree and the surface is watertight (no
//! cracks, the classic Lorensen–Cline table's defect). Loops are fan
//! triangulated. Each vertex lives on a grid edge owned by its lower corner,
//! so neighbouring cubes reference the same vertex (welding by construction).

use std::collections::HashMap;
use std::sync::OnceLock;

use spl_core::{Result, bail};
use spl_gpu::cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use super::Mesh;
use super::tsdf::TsdfVolume;
use crate::gpu::CudaResultExt;

/// Room for the longest loop set: ≤ 12 crossings give ≤ 10 triangles.
pub const MAX_TRI_INDICES: usize = 32;

/// Corner `i` sits at `(i & 1, i >> 1 & 1, i >> 2 & 1)`; inside means value < 0.
pub struct McTables {
    /// Edge `e` = (owner corner offset dx, dy, dz, axis).
    pub edges: [[i32; 4]; 12],
    /// Edge endpoints as corner ids.
    pub edge_corners: [[usize; 2]; 12],
    /// Per configuration: edge ids, three per triangle, −1 terminated.
    pub tris: [[i8; MAX_TRI_INDICES]; 256],
    pub tri_count: [u8; 256],
}

fn corner(i: usize) -> [i32; 3] {
    [(i & 1) as i32, (i >> 1 & 1) as i32, (i >> 2 & 1) as i32]
}

fn build_tables() -> McTables {
    let mut edges = [[0i32; 4]; 12];
    let mut edge_corners = [[0usize; 2]; 12];
    let mut n = 0;
    for axis in 0..3 {
        for a in 0..8 {
            if a >> axis & 1 == 0 {
                let c = corner(a);
                edges[n] = [c[0], c[1], c[2], axis];
                edge_corners[n] = [a, a | 1 << axis];
                n += 1;
            }
        }
    }
    let edge_of = |a: usize, b: usize| -> usize {
        let (a, b) = (a.min(b), a.max(b));
        edge_corners
            .iter()
            .position(|e| *e == [a, b])
            .expect("cube edge")
    };
    // Faces: corners in counter-clockwise order seen from outside.
    let mut faces = Vec::new();
    for axis in 0..3usize {
        for side in 0..2usize {
            let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
            let mut cyc: Vec<usize> = [(0, 0), (1, 0), (1, 1), (0, 1)]
                .iter()
                .map(|&(a, b)| side << axis | a << u | b << v)
                .collect();
            let p: Vec<[i32; 3]> = cyc.iter().map(|&c| corner(c)).collect();
            let d1: [i32; 3] = std::array::from_fn(|k| p[1][k] - p[0][k]);
            let d2: [i32; 3] = std::array::from_fn(|k| p[2][k] - p[1][k]);
            let nrm = [
                d1[1] * d2[2] - d1[2] * d2[1],
                d1[2] * d2[0] - d1[0] * d2[2],
                d1[0] * d2[1] - d1[1] * d2[0],
            ];
            if nrm[axis] * (2 * side as i32 - 1) < 0 {
                cyc.reverse();
            }
            faces.push(cyc);
        }
    }
    // Two edges share a face iff some face holds both: a diagonal between
    // them would lie in that face, where the neighbouring cube may draw the
    // same chord (an edge used four times).
    let on_face = |e: usize, f: &Vec<usize>| {
        f.contains(&edge_corners[e][0]) && f.contains(&edge_corners[e][1])
    };
    let share_face = |a: usize, b: usize| faces.iter().any(|f| on_face(a, f) && on_face(b, f));
    let mut tris = [[-1i8; MAX_TRI_INDICES]; 256];
    let mut tri_count = [0u8; 256];
    for cfg in 0..256usize {
        let inside = |c: usize| cfg >> c & 1 == 1;
        let mut next = [usize::MAX; 12];
        for cyc in &faces {
            for j in 0..4 {
                let (a, b) = (cyc[j], cyc[(j + 1) % 4]);
                if !(inside(a) && !inside(b)) {
                    continue;
                }
                // Walk on through the outside run to where it re-enters.
                let mut k = (j + 1) % 4;
                while !(!inside(cyc[k]) && inside(cyc[(k + 1) % 4])) {
                    k = (k + 1) % 4;
                }
                next[edge_of(a, b)] = edge_of(cyc[k], cyc[(k + 1) % 4]);
            }
        }
        let mut seen = [false; 12];
        let mut out = Vec::new();
        for s in 0..12 {
            if next[s] == usize::MAX || seen[s] {
                continue;
            }
            let mut lp = Vec::new();
            let mut e = s;
            while !seen[e] {
                seen[e] = true;
                lp.push(e);
                e = next[e];
            }
            // The face-tracing direction winds the loop clockwise seen from
            // outside; reverse so faces are counter-clockwise from outside.
            lp.reverse();
            let ok = triangulate(&lp, &share_face, &mut out);
            assert!(ok, "configuration {cfg}: no face-safe triangulation");
        }
        tri_count[cfg] = (out.len() / 3) as u8;
        for (i, e) in out.into_iter().enumerate() {
            tris[cfg][i] = e as i8;
        }
    }
    McTables {
        edges,
        edge_corners,
        tris,
        tri_count,
    }
}

/// Triangulates the loop `lp` so that no diagonal (a chord between
/// non-adjacent loop vertices) joins two edges of a common cube face.
fn triangulate(
    lp: &[usize],
    share_face: &dyn Fn(usize, usize) -> bool,
    out: &mut Vec<usize>,
) -> bool {
    let k = lp.len();
    if k < 3 {
        return true;
    }
    if k == 3 {
        out.extend(lp);
        return true;
    }
    // The triangle on chord (lp[0], lp[k-1]) has its apex at some lp[j].
    let n = out.len();
    for j in 1..k - 1 {
        let left_ok = j == 1 || !share_face(lp[0], lp[j]);
        let right_ok = j == k - 2 || !share_face(lp[j], lp[k - 1]);
        if !(left_ok && right_ok) {
            continue;
        }
        out.extend([lp[0], lp[j], lp[k - 1]]);
        if triangulate(&lp[..=j], share_face, out) && triangulate(&lp[j..], share_face, out) {
            return true;
        }
        out.truncate(n);
    }
    false
}

pub fn tables() -> &'static McTables {
    static T: OnceLock<McTables> = OnceLock::new();
    T.get_or_init(build_tables)
}

/// CPU marching cubes over a dense `dims` grid of samples at
/// `origin + voxel · (i, j, k)` (x fastest). Samples with `weight == 0` and
/// cubes with a corner of |value| ≥ `max_abs` are skipped, exactly like the
/// GPU path. The reference for tests; not fast.
pub fn extract_dense(
    values: &[f32],
    weights: &[f32],
    dims: [usize; 3],
    origin: [f32; 3],
    voxel: f32,
    max_abs: f32,
) -> Mesh {
    let t = tables();
    let idx = |x: usize, y: usize, z: usize| (z * dims[1] + y) * dims[0] + x;
    let ok = |i: usize| weights[i] > 0.0 && values[i].abs() < max_abs;
    let mut mesh = Mesh::default();
    let mut vid: HashMap<(usize, usize), u32> = HashMap::new();
    let mut vertex = |x: usize, y: usize, z: usize, axis: usize, mesh: &mut Mesh| -> u32 {
        let a = idx(x, y, z);
        *vid.entry((a, axis)).or_insert_with(|| {
            let mut q = [x, y, z];
            q[axis] += 1;
            let b = idx(q[0], q[1], q[2]);
            let (va, vb) = (values[a], values[b]);
            let s = va / (va - vb);
            let mut p = [x as f32, y as f32, z as f32];
            p[axis] += s;
            mesh.positions
                .push(std::array::from_fn(|k| origin[k] + voxel * p[k]));
            mesh.positions.len() as u32 - 1
        })
    };
    for z in 0..dims[2].saturating_sub(1) {
        for y in 0..dims[1].saturating_sub(1) {
            for x in 0..dims[0].saturating_sub(1) {
                let mut cfg = 0usize;
                let mut valid = true;
                for c in 0..8 {
                    let o = corner(c);
                    let i = idx(x + o[0] as usize, y + o[1] as usize, z + o[2] as usize);
                    valid &= ok(i);
                    if values[i] < 0.0 {
                        cfg |= 1 << c;
                    }
                }
                if !valid {
                    continue;
                }
                let tri = &t.tris[cfg];
                for k in 0..t.tri_count[cfg] as usize {
                    let f: [u32; 3] = std::array::from_fn(|j| {
                        let e = t.edges[tri[3 * k + j] as usize];
                        vertex(
                            x + e[0] as usize,
                            y + e[1] as usize,
                            z + e[2] as usize,
                            e[3] as usize,
                            &mut mesh,
                        )
                    });
                    mesh.faces.push(f);
                }
            }
        }
    }
    mesh
}

/// Device copies of the tables.
pub(super) struct DeviceTables {
    pub tris: CudaSlice<i8>,
    pub tri_count: CudaSlice<u8>,
    pub edges: CudaSlice<i32>,
}

impl DeviceTables {
    pub fn upload(vol: &TsdfVolume) -> Result<Self> {
        let t = tables();
        let s = &vol.gpu().stream;
        let tris: Vec<i8> = t.tris.iter().flatten().copied().collect();
        let edges: Vec<i32> = t.edges.iter().flatten().copied().collect();
        Ok(Self {
            tris: s.clone_htod(&tris).cuda("upload MC triangles")?,
            tri_count: s.clone_htod(&t.tri_count).cuda("upload MC counts")?,
            edges: s.clone_htod(&edges).cuda("upload MC edges")?,
        })
    }
}

/// Options of the GPU extractor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExtractOptions {
    /// Cubes with any corner |tsdf| ≥ this are skipped: suppresses the
    /// spurious sheet where a truncated front (+1) meets a back (−1) at
    /// occlusion boundaries.
    pub max_abs: f32,
    /// Minimum accumulated weight of a sample.
    pub min_weight: f32,
}

impl Default for ExtractOptions {
    fn default() -> Self {
        Self {
            max_abs: 0.99,
            min_weight: 1.0,
        }
    }
}

/// GPU marching cubes over every allocated block: count, emit welded
/// vertices (one per crossing grid edge, position by linear interpolation,
/// normal from the interpolated central-difference gradient), emit faces.
pub(super) fn extract(vol: &TsdfVolume, opts: ExtractOptions) -> Result<Mesh> {
    let gpu = vol.gpu();
    let s = &gpu.stream;
    let k = vol.kernels();
    let nblocks = vol.blocks();
    if nblocks == 0 {
        return Ok(Mesh::default());
    }
    let tab = vol.device_tables()?;
    let mut counters = s.alloc_zeros::<u32>(4).cuda("alloc MC counters")?;
    let (keys, vals, mask) = vol.hash();
    let (tsdf, weight, coords) = vol.voxels();
    let cfg = LaunchConfig {
        grid_dim: (nblocks, 1, 1),
        block_dim: (512, 1, 1),
        shared_mem_bytes: 0,
    };
    let (max_abs, min_w) = (opts.max_abs, opts.min_weight);

    let f = k.get("mc_count")?;
    let mut l = s.launch_builder(&f);
    l.arg(keys)
        .arg(vals)
        .arg(&mask)
        .arg(coords)
        .arg(tsdf)
        .arg(weight);
    l.arg(&max_abs)
        .arg(&min_w)
        .arg(&tab.tri_count)
        .arg(&mut counters);
    // SAFETY: mc_count(keys, vals, mask, coords, tsdf, weight, max_abs,
    // min_w, tri_count[256], counters[4]); one CUDA block per voxel block.
    unsafe { l.launch(cfg) }.cuda("mc_count")?;
    let c = s.clone_dtoh(&counters).cuda("read MC counts")?;
    let (nv, nf) = (c[0] as usize, c[1] as usize);
    if nf == 0 {
        return Ok(Mesh::default());
    }
    s.memset_zeros(&mut counters).cuda("reset MC counters")?;
    let mut edge_vid = s
        .alloc_zeros::<i32>(nblocks as usize * 512 * 3)
        .cuda("alloc edge vertex ids")?;
    let mut pos = s.alloc_zeros::<f32>(nv * 3).cuda("alloc MC positions")?;
    let mut nrm = s.alloc_zeros::<f32>(nv * 3).cuda("alloc MC normals")?;
    let mut faces = s.alloc_zeros::<u32>(nf * 3).cuda("alloc MC faces")?;
    let (voxel, origin) = (vol.params().voxel, vol.params().origin);
    let (nv32, nf32) = (nv as u32, nf as u32);

    let f = k.get("mc_vertices")?;
    let mut l = s.launch_builder(&f);
    l.arg(keys)
        .arg(vals)
        .arg(&mask)
        .arg(coords)
        .arg(tsdf)
        .arg(weight);
    l.arg(&max_abs)
        .arg(&min_w)
        .arg(&voxel)
        .arg(&origin[0])
        .arg(&origin[1])
        .arg(&origin[2]);
    l.arg(&mut edge_vid)
        .arg(&mut pos)
        .arg(&mut nrm)
        .arg(&nv32)
        .arg(&mut counters);
    // SAFETY: mc_vertices writes edge_vid[nblocks·512·3] and at most nv
    // vertices (the count pass's exact total) into pos/nrm[nv·3].
    unsafe { l.launch(cfg) }.cuda("mc_vertices")?;

    let f = k.get("mc_faces")?;
    let mut l = s.launch_builder(&f);
    l.arg(keys)
        .arg(vals)
        .arg(&mask)
        .arg(coords)
        .arg(tsdf)
        .arg(weight);
    l.arg(&max_abs)
        .arg(&min_w)
        .arg(&tab.tris)
        .arg(&tab.tri_count)
        .arg(&tab.edges);
    l.arg(&edge_vid)
        .arg(&mut faces)
        .arg(&nf32)
        .arg(&mut counters);
    // SAFETY: mc_faces reads edge_vid written above and writes at most nf
    // faces into faces[nf·3].
    unsafe { l.launch(cfg) }.cuda("mc_faces")?;

    let c = s.clone_dtoh(&counters).cuda("read MC totals")?;
    if c[0] as usize != nv || c[1] as usize != nf || c[2] != 0 {
        bail!(
            "marching cubes: counted {nv} vertices / {nf} faces, emitted {} / {} ({} unresolved edges)",
            c[0],
            c[1],
            c[2]
        );
    }
    let p = s.clone_dtoh(&pos).cuda("download positions")?;
    let n = s.clone_dtoh(&nrm).cuda("download normals")?;
    let fc = s.clone_dtoh(&faces).cuda("download faces")?;
    let mut mesh = Mesh {
        positions: p.as_chunks::<3>().0.to_vec(),
        normals: n.as_chunks::<3>().0.to_vec(),
        faces: fc.as_chunks::<3>().0.to_vec(),
    };
    // Crossing edges no valid cube uses (a corner failed max_abs/weight).
    mesh.remove_unreferenced();
    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(n: usize, f: impl Fn([f32; 3]) -> f32) -> (Vec<f32>, Vec<f32>, f32, [f32; 3]) {
        let voxel = 1.0 / (n - 1) as f32;
        let origin = [-0.5; 3];
        let mut v = Vec::with_capacity(n * n * n);
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let p = [x, y, z].map(|i| -0.5 + voxel * i as f32);
                    v.push(f(p));
                }
            }
        }
        let w = vec![1.0; v.len()];
        (v, w, voxel, origin)
    }

    #[test]
    fn tables_cover_every_crossing_edge_once_per_side() {
        let t = tables();
        assert_eq!(t.tri_count[0], 0);
        assert_eq!(t.tri_count[255], 0);
        let max = t.tri_count.iter().copied().max().unwrap();
        assert!(max as usize * 3 < MAX_TRI_INDICES);
        for cfg in 0..256usize {
            // Each crossing edge appears in at least one triangle; no other edge does.
            for (e, [a, b]) in t.edge_corners.iter().enumerate() {
                let crosses = (cfg >> a & 1) != (cfg >> b & 1);
                let used = t.tris[cfg][..3 * t.tri_count[cfg] as usize].contains(&(e as i8));
                assert_eq!(crosses, used, "cfg {cfg} edge {e}");
            }
            // Complement symmetry of the classic table does not hold by design;
            // single-corner cases are one triangle.
            if cfg.count_ones() == 1 {
                assert_eq!(t.tri_count[cfg], 1);
            }
        }
    }

    #[test]
    fn sphere_is_closed_outward_and_accurate() {
        let r = 0.3f32;
        let (v, w, voxel, origin) =
            grid(65, |p| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() - r);
        let m = extract_dense(&v, &w, [65; 3], origin, voxel, f32::INFINITY);
        assert_eq!(m.boundary_or_nonmanifold_edges(), 0);
        let vol = m.signed_volume();
        let want = 4.0 / 3.0 * std::f64::consts::PI * (r as f64).powi(3);
        assert!((vol / want - 1.0).abs() < 0.01, "volume {vol} vs {want}");
        let worst = m
            .positions
            .iter()
            .map(|p| ((p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() - r).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 0.1 * voxel, "vertex off surface by {worst}");
    }

    #[test]
    fn random_fields_give_closed_oriented_surfaces() {
        // Noise forces every configuration, ambiguous faces included; a
        // positive border closes the surface.
        let n = 24;
        let mut s = 0x1234_5678_9abc_def0u64;
        let mut seen = [false; 256];
        for _ in 0..20 {
            let mut v = Vec::new();
            for z in 0..n {
                for y in 0..n {
                    for x in 0..n {
                        s ^= s << 13;
                        s ^= s >> 7;
                        s ^= s << 17;
                        let border = [x, y, z].iter().any(|&i| i == 0 || i == n - 1);
                        v.push(if border {
                            1.0
                        } else {
                            (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
                        });
                    }
                }
            }
            for z in 0..n - 1 {
                for y in 0..n - 1 {
                    for x in 0..n - 1 {
                        let mut cfg = 0;
                        for c in 0..8 {
                            let o = corner(c);
                            let i = ((z + o[2] as usize) * n + y + o[1] as usize) * n
                                + x
                                + o[0] as usize;
                            if v[i] < 0.0 {
                                cfg |= 1 << c;
                            }
                        }
                        seen[cfg] = true;
                    }
                }
            }
            let w = vec![1.0; v.len()];
            let m = extract_dense(&v, &w, [n; 3], [0.0; 3], 1.0, f32::INFINITY);
            assert!(!m.is_empty());
            assert_eq!(m.boundary_or_nonmanifold_edges(), 0);
            assert!(m.signed_volume() > 0.0);
        }
        assert!(seen.iter().filter(|&&b| b).count() > 250);
    }

    #[test]
    fn single_inside_corner_faces_outward() {
        // One negative sample in a positive field: a closed blob whose
        // outward winding gives positive volume.
        let n = 5;
        let mut v = vec![1.0f32; n * n * n];
        v[(2 * n + 2) * n + 2] = -1.0;
        let w = vec![1.0; v.len()];
        let m = extract_dense(&v, &w, [n; 3], [0.0; 3], 1.0, f32::INFINITY);
        assert_eq!(m.boundary_or_nonmanifold_edges(), 0);
        assert!(m.signed_volume() > 0.0, "{}", m.signed_volume());
    }
}
