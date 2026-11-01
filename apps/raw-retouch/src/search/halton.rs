//! Scrambled Halton points: well-spread seeds over the whole search box.

use crate::cmaes::Rng;

const PRIMES: [usize; 16] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];

/// `count` points in [0, 1)^dim, dim at most 16. Every base's digits pass
/// through a random permutation per digit position, drawn from `seed`. On a
/// short run the plain sequence lines up its high bases (the first 23
/// points in bases 23 and 29 fall on one line); the scramble breaks that
/// and makes different seeds explore different points.
pub(crate) fn scrambled(count: usize, dim: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = Rng::new(seed);
    let axes: Vec<(usize, Vec<Vec<usize>>)> = PRIMES
        .iter()
        .take(dim)
        .map(|&base| {
            // Enough digits for about 40 bits of resolution.
            let digits = (40.0 / (base as f64).log2()).ceil() as usize;
            let perms = (0..digits).map(|_| permutation(base, &mut rng)).collect();
            (base, perms)
        })
        .collect();
    (0..count)
        .map(|i| {
            axes.iter()
                .map(|(base, perms)| radical_inverse(i, *base, perms))
                .collect()
        })
        .collect()
}

fn permutation(n: usize, rng: &mut Rng) -> Vec<usize> {
    let mut p: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        p.swap(i, rng.below(i + 1));
    }
    p
}

fn radical_inverse(mut i: usize, base: usize, perms: &[Vec<usize>]) -> f64 {
    let b = base as f64;
    let mut x = 0.0;
    let mut scale = 1.0 / b;
    for perm in perms {
        x += perm[i % base] as f64 * scale;
        i /= base;
        scale /= b;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn points_are_in_the_box_spread_and_seeded() {
        let a = scrambled(32, 10, 1);
        let b = scrambled(32, 10, 1);
        let c = scrambled(32, 10, 2);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(a.len(), 32);
        assert!(
            a.iter()
                .all(|p| p.len() == 10 && p.iter().all(|x| (0.0..1.0).contains(x)))
        );
        // Each coordinate's 32 values cover every eighth of [0, 1).
        for d in 0..10 {
            let mut hit = [false; 8];
            for p in &a {
                hit[(p[d] * 8.0) as usize] = true;
            }
            assert!(hit.iter().all(|h| *h), "axis {d}");
        }
        // Bases 23 and 29 no longer fall on a line.
        let r = correlation(&a.iter().map(|p| (p[8], p[9])).collect::<Vec<_>>());
        assert!(r.abs() < 0.6, "correlation {r}");
    }

    fn correlation(xy: &[(f64, f64)]) -> f64 {
        let n = xy.len() as f64;
        let (mx, my) = (
            xy.iter().map(|p| p.0).sum::<f64>() / n,
            xy.iter().map(|p| p.1).sum::<f64>() / n,
        );
        let cov: f64 = xy.iter().map(|(x, y)| (x - mx) * (y - my)).sum();
        let vx: f64 = xy.iter().map(|(x, _)| (x - mx).powi(2)).sum();
        let vy: f64 = xy.iter().map(|(_, y)| (y - my).powi(2)).sum();
        cov / (vx * vy).sqrt()
    }
}
