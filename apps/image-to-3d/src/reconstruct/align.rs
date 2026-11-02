//! Similarity alignment of camera trajectories (Horn's quaternion method),
//! for diagnostics against reference poses. Owner: mvp-pipeline. Same
//! method as the engine's accuracy test (`vggt/integration.rs`).

pub type V3 = [f64; 3];
pub type M3 = [[f64; 3]; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn mv(m: &M3, v: V3) -> V3 {
    std::array::from_fn(|i| dot(m[i], v))
}

fn mean(p: &[V3]) -> V3 {
    let n = p.len().max(1) as f64;
    std::array::from_fn(|k| p.iter().map(|x| x[k]).sum::<f64>() / n)
}

/// Eigenvector of the largest eigenvalue of a symmetric 4×4 (Jacobi).
#[allow(clippy::needless_range_loop)]
pub fn top_eigvec4(mut a: [[f64; 4]; 4]) -> [f64; 4] {
    let mut v = [[0.0; 4]; 4];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _ in 0..64 {
        let off: f64 = (0..4)
            .flat_map(|i| (0..4).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[i][j] * a[i][j])
            .sum();
        if off < 1e-30 {
            break;
        }
        for p in 0..4 {
            for q in p + 1..4 {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..4 {
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..4 {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for row in v.iter_mut() {
                    let (vkp, vkq) = (row[p], row[q]);
                    row[p] = c * vkp - s * vkq;
                    row[q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let best = (0..4).max_by(|&i, &j| a[i][i].total_cmp(&a[j][j])).unwrap();
    std::array::from_fn(|k| v[k][best])
}

/// Similarity `b ≈ s R a + t` minimising the squared error (Horn's
/// quaternion method; Umeyama without reflections).
pub fn sim3(a: &[V3], b: &[V3]) -> (f64, M3, V3) {
    let (ma, mb) = (mean(a), mean(b));
    let mut s = [[0.0f64; 3]; 3];
    let mut va = 0.0;
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (sub(*x, ma), sub(*y, mb));
        va += dot(x, x);
        for i in 0..3 {
            for j in 0..3 {
                s[i][j] += x[i] * y[j];
            }
        }
    }
    let [[sxx, sxy, sxz], [syx, syy, syz], [szx, szy, szz]] = s;
    let n = [
        [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
        [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
        [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
        [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
    ];
    let [w, x, y, z] = top_eigvec4(n);
    let r = [
        [
            w * w + x * x - y * y - z * z,
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            w * w - x * x + y * y - z * z,
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            w * w - x * x - y * y + z * z,
        ],
    ];
    let num: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| dot(sub(*y, mb), mv(&r, sub(*x, ma))))
        .sum();
    let scale = num / va;
    let rm = mv(&r, ma);
    let t = std::array::from_fn(|k| mb[k] - scale * rm[k]);
    (scale, r, t)
}

/// Similarity taking the world of `src` cameras to the world of `dst`
/// cameras (`X_dst ≈ s R X_src + t`), from their centres.
pub fn sim3_between(dst: &[(M3, V3)], src: &[(M3, V3)]) -> (f64, M3, V3) {
    let center = |(r, t): &(M3, V3)| -> V3 {
        std::array::from_fn(|j| -(0..3).map(|k| r[k][j] * t[k]).sum::<f64>())
    };
    let a: Vec<V3> = src.iter().map(center).collect();
    let b: Vec<V3> = dst.iter().map(center).collect();
    sim3(&a, &b)
}

/// A `src`-world camera in the `dst` world of `sim = sim3_between(dst, src)`;
/// translations come out in `dst` units.
pub fn map_pose((s, r, t): &(f64, M3, V3), (rg, tg): &(M3, V3)) -> (M3, V3) {
    // x_c = Rg X_src + tg = Rg Rᵀ (X − t) / s + tg  →  scale by s.
    let rn: M3 =
        std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| rg[i][k] * r[j][k]).sum()));
    let rt = mv(&rn, *t);
    let tn: V3 = std::array::from_fn(|i| s * tg[i] - rt[i]);
    (rn, tn)
}

/// A `src`-world point in the `dst` world.
pub fn map_point((s, r, t): &(f64, M3, V3), x: V3) -> V3 {
    let rx = mv(r, x);
    std::array::from_fn(|i| s * rx[i] + t[i])
}

/// `gt` world-to-camera poses expressed in the frame of `ours` (camera
/// centres aligned by a similarity; translations rescaled to `ours` units).
pub fn poses_into_frame(ours: &[(M3, V3)], gt: &[(M3, V3)]) -> Vec<(M3, V3)> {
    let sim = sim3_between(ours, gt);
    gt.iter().map(|p| map_pose(&sim, p)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poses_into_frame_inverts_a_similarity() {
        let rz = |d: f64| -> M3 {
            let (s, c) = d.to_radians().sin_cos();
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
        };
        // gt cameras; ours = the same scene under X' = 2 Rz(30) X + (1, 2, 3).
        let gt: Vec<(M3, V3)> = (0..6)
            .map(|i| {
                (
                    rz(10.0 * i as f64),
                    [i as f64, 0.5 * i as f64, 1.0 + (i % 2) as f64],
                )
            })
            .collect();
        let (s, ra, ta) = (2.0, rz(30.0), [1.0, 2.0, 3.0]);
        let ours: Vec<(M3, V3)> = gt
            .iter()
            .map(|(r, t)| {
                // X = Raᵀ (X' − ta) / s: R' = R Raᵀ, t' = s t − R' ta.
                let rn: M3 = std::array::from_fn(|i| {
                    std::array::from_fn(|j| (0..3).map(|k| r[i][k] * ra[j][k]).sum())
                });
                let rt = mv(&rn, ta);
                (rn, std::array::from_fn(|i| s * t[i] - rt[i]))
            })
            .collect();
        let back = poses_into_frame(&ours, &gt);
        for ((r1, t1), (r2, t2)) in back.iter().zip(&ours) {
            for i in 0..3 {
                assert!((t1[i] - t2[i]).abs() < 1e-9);
                for j in 0..3 {
                    assert!((r1[i][j] - r2[i][j]).abs() < 1e-9);
                }
            }
        }
    }
}
