//! CMA-ES for small bounded problems. Owner: search.
//!
//! A (mu/mu_w, lambda)-CMA-ES with the default strategy parameters of
//! Hansen's tutorial (arXiv:1604.00772): positive recombination weights,
//! cumulative step-size adaptation, rank-one plus rank-mu covariance
//! updates. It is asked and told a whole generation at a time so the
//! renderer can measure every candidate in one batched GPU call.
//!
//! The search box is the unit cube. Samples are reflected into it, so every
//! evaluated point is legal and an optimum on a bound is reached by the
//! distribution folding onto it instead of being pushed back by a penalty.
//! [`minimize`] adds IPOP restarts (population doubled after each
//! convergence). All randomness comes from a seeded xoshiro256**, so equal
//! seeds give equal runs, bit for bit.

use std::collections::VecDeque;

/// splitmix64 (Steele, Lea and Flood): expands one seed into well-mixed
/// words, here the state of [`Rng`] and the seeds of derived streams.
#[derive(Debug, Clone)]
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// xoshiro256** (Blackman and Vigna) with Box–Muller normals.
#[derive(Debug, Clone)]
pub struct Rng {
    s: [u64; 4],
    spare: Option<f64>,
}

impl Rng {
    /// splitmix64 is a bijection of its counter, so four consecutive outputs
    /// are never all zero, xoshiro's one forbidden state.
    pub fn new(seed: u64) -> Self {
        let mut mix = SplitMix64(seed);
        let s = [
            mix.next_u64(),
            mix.next_u64(),
            mix.next_u64(),
            mix.next_u64(),
        ];
        Self { s, spare: None }
    }

    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform in [0, 1), 53 random bits.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in 0..n by multiply-shift; its bias (at most n / 2^64) is
    /// far below anything a search could notice.
    pub fn below(&mut self, n: usize) -> usize {
        ((u128::from(self.next_u64()) * n as u128) >> 64) as usize
    }

    /// Standard normal. Box–Muller makes two per pair of uniforms; the
    /// second is kept for the next call.
    pub fn normal(&mut self) -> f64 {
        if let Some(z) = self.spare.take() {
            return z;
        }
        let u1 = loop {
            let u = self.uniform();
            if u > 0.0 {
                break u;
            }
        };
        let u2 = self.uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let (sin, cos) = (std::f64::consts::TAU * u2).sin_cos();
        self.spare = Some(r * sin);
        r * cos
    }
}

/// Folds any real into [0, 1] by mirroring at the bounds (a triangle wave),
/// so a sample that overshoots a bound lands as far inside it.
pub fn reflect(x: f64) -> f64 {
    if !x.is_finite() {
        return 0.5;
    }
    let t = x.rem_euclid(2.0);
    if t <= 1.0 { t } else { 2.0 - t }
}

/// Hansen's default population size, 4 + floor(3 ln n).
pub fn default_lambda(n: usize) -> usize {
    4 + (3.0 * (n.max(1) as f64).ln()).floor() as usize
}

/// Why a run converged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
    /// The search distribution shrank below `tol_x` in every coordinate.
    TolX,
    /// Recent best values (and the last generation) span less than `tol_fun`.
    TolFun,
    /// The covariance matrix became too ill-conditioned to sample from.
    Condition,
}

/// Largest step size worth having: the box is 1 wide.
const SIGMA_MAX: f64 = 1.0;
const CONDITION_MAX: f64 = 1e14;

/// One (mu/mu_w, lambda)-CMA-ES run on [0, 1]^n, minimizing.
#[derive(Debug, Clone)]
pub struct Cmaes {
    n: usize,
    lambda: usize,
    weights: Vec<f64>,
    mueff: f64,
    cs: f64,
    damps: f64,
    cc: f64,
    c1: f64,
    cmu: f64,
    chi_n: f64,
    tol_x: f64,
    tol_fun: f64,
    /// Distribution mean, unreflected (it may sit past a bound).
    mean: Vec<f64>,
    sigma: f64,
    ps: Vec<f64>,
    pc: Vec<f64>,
    /// Covariance, n × n row-major.
    cov: Vec<f64>,
    /// Eigenvectors of `cov` as columns, n × n row-major.
    basis: Vec<f64>,
    /// Square roots of the eigenvalues.
    scales: Vec<f64>,
    generation: usize,
    /// This generation's samples, unreflected, in the order `ask` returned them.
    asked: Vec<Vec<f64>>,
    best_history: VecDeque<f64>,
    stop: Option<Stop>,
    rng: Rng,
}

impl Cmaes {
    /// `x0` is the initial mean in [0, 1]^n, `sigma0` the initial step in
    /// units of the box, `lambda` the population (0 = Hansen's default).
    pub fn new(x0: &[f64], sigma0: f64, lambda: usize, seed: u64) -> Self {
        let n = x0.len().max(1);
        let lambda = if lambda == 0 {
            default_lambda(n)
        } else {
            lambda.max(2)
        };
        let mu = lambda / 2;
        let raw: Vec<f64> = (1..=mu)
            .map(|i| ((lambda as f64 + 1.0) / 2.0).ln() - (i as f64).ln())
            .collect();
        let sum: f64 = raw.iter().sum();
        let weights: Vec<f64> = raw.iter().map(|w| w / sum).collect();
        let mueff = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();
        let nf = n as f64;
        let cs = (mueff + 2.0) / (nf + mueff + 5.0);
        let damps = 1.0 + 2.0 * (((mueff - 1.0) / (nf + 1.0)).sqrt() - 1.0).max(0.0) + cs;
        let cc = (4.0 + mueff / nf) / (nf + 4.0 + 2.0 * mueff / nf);
        let c1 = 2.0 / ((nf + 1.3).powi(2) + mueff);
        let cmu = (1.0 - c1).min(2.0 * (mueff - 2.0 + 1.0 / mueff) / ((nf + 2.0).powi(2) + mueff));
        let chi_n = nf.sqrt() * (1.0 - 1.0 / (4.0 * nf) + 1.0 / (21.0 * nf * nf));
        let mut mean: Vec<f64> = x0.iter().map(|x| reflect(*x)).collect();
        mean.resize(n, 0.5);
        Self {
            n,
            lambda,
            weights,
            mueff,
            cs,
            damps,
            cc,
            c1,
            cmu,
            chi_n,
            tol_x: 1e-11,
            tol_fun: 1e-12,
            mean,
            sigma: sigma0.clamp(1e-12, SIGMA_MAX),
            ps: vec![0.0; n],
            pc: vec![0.0; n],
            cov: identity(n),
            basis: identity(n),
            scales: vec![1.0; n],
            generation: 0,
            asked: Vec::new(),
            best_history: VecDeque::new(),
            stop: None,
            rng: Rng::new(seed),
        }
    }

    /// Stop criteria: `tol_x` on the step in box units, `tol_fun` on
    /// objective values.
    pub fn with_tolerances(mut self, tol_x: f64, tol_fun: f64) -> Self {
        self.tol_x = tol_x;
        self.tol_fun = tol_fun;
        self
    }

    pub fn dim(&self) -> usize {
        self.n
    }

    pub fn lambda(&self) -> usize {
        self.lambda
    }

    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    pub fn generation(&self) -> usize {
        self.generation
    }

    /// The distribution mean, folded into the box.
    pub fn mean(&self) -> Vec<f64> {
        self.mean.iter().map(|x| reflect(*x)).collect()
    }

    pub fn stop(&self) -> Option<Stop> {
        self.stop
    }

    /// The next generation: `lambda` points in [0, 1]^n. Answer with
    /// [`Cmaes::tell`] in the same order.
    pub fn ask(&mut self) -> Vec<Vec<f64>> {
        let n = self.n;
        self.asked.clear();
        let mut points = Vec::with_capacity(self.lambda);
        let mut z = vec![0.0; n];
        for _ in 0..self.lambda {
            for (zj, scale) in z.iter_mut().zip(&self.scales) {
                *zj = scale * self.rng.normal();
            }
            let x: Vec<f64> = (0..n)
                .map(|i| {
                    let row = &self.basis[i * n..(i + 1) * n];
                    let y: f64 = row.iter().zip(&z).map(|(b, zj)| b * zj).sum();
                    self.mean[i] + self.sigma * y
                })
                .collect();
            points.push(x.iter().map(|v| reflect(*v)).collect());
            self.asked.push(x);
        }
        points
    }

    /// Updates the distribution from the objective values (lower is better)
    /// of the points the last [`Cmaes::ask`] returned, in that order. NaN
    /// counts as worst. A wrong count is ignored.
    pub fn tell(&mut self, fitness: &[f64]) {
        let n = self.n;
        if fitness.len() != self.asked.len() || self.asked.len() != self.lambda {
            debug_assert!(false, "tell needs one value per asked point");
            return;
        }
        let key = |f: f64| if f.is_nan() { f64::INFINITY } else { f };
        let mut order: Vec<usize> = (0..self.lambda).collect();
        order.sort_by(|a, b| key(fitness[*a]).total_cmp(&key(fitness[*b])));

        // Steps of the mu best samples in units of sigma.
        let steps: Vec<Vec<f64>> = order[..self.weights.len()]
            .iter()
            .map(|k| {
                self.asked[*k]
                    .iter()
                    .zip(&self.mean)
                    .map(|(x, m)| (x - m) / self.sigma)
                    .collect()
            })
            .collect();
        let mut yw = vec![0.0; n];
        for (w, y) in self.weights.iter().zip(&steps) {
            for (acc, yi) in yw.iter_mut().zip(y) {
                *acc += w * yi;
            }
        }
        for (m, y) in self.mean.iter_mut().zip(&yw) {
            *m += self.sigma * y;
        }

        // Step-size path through C^-1/2 = B D^-1 B^T.
        let mut t = vec![0.0; n];
        for (j, tj) in t.iter_mut().enumerate() {
            let dot: f64 = (0..n).map(|i| self.basis[i * n + j] * yw[i]).sum();
            *tj = dot / self.scales[j];
        }
        let cs_norm = (self.cs * (2.0 - self.cs) * self.mueff).sqrt();
        for (i, p) in self.ps.iter_mut().enumerate() {
            let whitened: f64 = (0..n).map(|j| self.basis[i * n + j] * t[j]).sum();
            *p = (1.0 - self.cs) * *p + cs_norm * whitened;
        }
        let ps_norm = self.ps.iter().map(|p| p * p).sum::<f64>().sqrt();
        let decay = 1.0 - (1.0 - self.cs).powi(2 * (self.generation as i32 + 1));
        let hsig = ps_norm / decay.max(1e-300).sqrt() / self.chi_n < 1.4 + 2.0 / (n as f64 + 1.0);

        let cc_norm = (self.cc * (2.0 - self.cc) * self.mueff).sqrt();
        let h = if hsig { 1.0 } else { 0.0 };
        for (p, y) in self.pc.iter_mut().zip(&yw) {
            *p = (1.0 - self.cc) * *p + h * cc_norm * y;
        }

        // Rank-one plus rank-mu covariance update.
        let delta_h = (1.0 - h) * self.cc * (2.0 - self.cc);
        let keep = 1.0 - self.c1 - self.cmu + self.c1 * delta_h;
        for i in 0..n {
            for j in 0..=i {
                let rank_mu: f64 = self
                    .weights
                    .iter()
                    .zip(&steps)
                    .map(|(w, y)| w * y[i] * y[j])
                    .sum();
                let v = keep * self.cov[i * n + j]
                    + self.c1 * self.pc[i] * self.pc[j]
                    + self.cmu * rank_mu;
                self.cov[i * n + j] = v;
                self.cov[j * n + i] = v;
            }
        }

        self.sigma *= ((self.cs / self.damps) * (ps_norm / self.chi_n - 1.0)).exp();
        self.sigma = self.sigma.min(SIGMA_MAX);
        self.generation += 1;

        let (values, basis) = symmetric_eigen(&self.cov, n);
        let finite = values.iter().chain(&basis).all(|v| v.is_finite());
        let (lo, hi) = values.iter().fold((f64::INFINITY, 0.0f64), |(lo, hi), v| {
            (lo.min(*v), hi.max(*v))
        });
        if !finite || !self.sigma.is_finite() || lo <= 0.0 || hi / lo > CONDITION_MAX {
            self.stop = Some(Stop::Condition);
            return;
        }
        self.scales = values.iter().map(|v| v.sqrt()).collect();
        self.basis = basis;

        let best = key(fitness[order[0]]);
        let worst = key(fitness[order[self.lambda - 1]]);
        let horizon = 10 + (30 * n).div_ceil(self.lambda);
        self.best_history.push_back(best);
        while self.best_history.len() > horizon {
            self.best_history.pop_front();
        }
        self.stop = self.check_stop(worst - best, horizon);
    }

    fn check_stop(&self, generation_range: f64, horizon: usize) -> Option<Stop> {
        let n = self.n;
        let tol_x = (0..n).all(|i| {
            let spread = self.cov[i * n + i].sqrt().max(self.pc[i].abs());
            self.sigma * spread < self.tol_x
        });
        if tol_x {
            return Some(Stop::TolX);
        }
        if self.best_history.len() >= horizon {
            let (lo, hi) = self
                .best_history
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
                    (lo.min(*v), hi.max(*v))
                });
            if hi - lo < self.tol_fun && generation_range < self.tol_fun {
                return Some(Stop::TolFun);
            }
        }
        None
    }
}

fn identity(n: usize) -> Vec<f64> {
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

/// Eigen-decomposition of a symmetric n × n matrix (row-major) by cyclic
/// Jacobi rotations, which is exact to rounding and plenty fast for n ≤ 16.
/// Returns the eigenvalues and the eigenvectors as columns (row-major).
fn symmetric_eigen(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut m = a.to_vec();
    let mut v = identity(n);
    for _sweep in 0..64 {
        let mut off = 0.0;
        let mut diag = 0.0;
        for p in 0..n {
            diag += m[p * n + p] * m[p * n + p];
            for q in p + 1..n {
                off += m[p * n + q] * m[p * n + q];
            }
        }
        if off.is_nan() || off <= 1e-30 * diag {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = m[p * n + q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (m[q * n + q] - m[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (kp, kq) = (m[k * n + p], m[k * n + q]);
                    m[k * n + p] = c * kp - s * kq;
                    m[k * n + q] = s * kp + c * kq;
                }
                for k in 0..n {
                    let (pk, qk) = (m[p * n + k], m[q * n + k]);
                    m[p * n + k] = c * pk - s * qk;
                    m[q * n + k] = s * pk + c * qk;
                }
                for k in 0..n {
                    let (kp, kq) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * kp - s * kq;
                    v[k * n + q] = s * kp + c * kq;
                }
            }
        }
    }
    ((0..n).map(|i| m[i * n + i]).collect(), v)
}

/// Settings for [`minimize`].
#[derive(Debug, Clone)]
pub struct Options {
    /// Initial step size in units of the box.
    pub sigma0: f64,
    /// Population size; 0 = Hansen's default 4 + 3 ln n.
    pub lambda: usize,
    /// IPOP never grows the population beyond this.
    pub max_lambda: usize,
    /// Evaluation budget: every point handed to the batch function counts.
    pub max_evals: usize,
    /// IPOP restarts after convergence, each with twice the population.
    pub restarts: usize,
    /// Restart at the best point so far rather than at a uniform random
    /// one (classic IPOP): a broader look around a known good basin.
    pub restart_from_best: bool,
    pub tol_x: f64,
    pub tol_fun: f64,
    pub seed: u64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            sigma0: 0.3,
            lambda: 0,
            max_lambda: 512,
            max_evals: 10_000,
            restarts: 0,
            restart_from_best: false,
            tol_x: 1e-11,
            tol_fun: 1e-12,
            seed: 0,
        }
    }
}

/// The best point [`minimize`] saw.
#[derive(Debug, Clone)]
pub struct Outcome {
    pub x: Vec<f64>,
    pub f: f64,
    pub evals: usize,
    pub generations: usize,
    pub restarts: usize,
    /// Why the last run converged; None when the budget ran out first.
    pub stop: Option<Stop>,
}

/// Minimizes over [0, 1]^n from `x0`. `batch` scores a whole generation
/// (lower is better) in one call; its error ends the search.
pub fn minimize<E>(
    x0: &[f64],
    opts: &Options,
    mut batch: impl FnMut(&[Vec<f64>]) -> Result<Vec<f64>, E>,
) -> Result<Outcome, E> {
    let mut seeds = SplitMix64(opts.seed);
    let mut restart_rng = Rng::new(seeds.next_u64());
    let mut lambda = if opts.lambda == 0 {
        default_lambda(x0.len())
    } else {
        opts.lambda
    };
    let mut es = Cmaes::new(x0, opts.sigma0, lambda, seeds.next_u64())
        .with_tolerances(opts.tol_x, opts.tol_fun);
    let mut out = Outcome {
        x: es.mean(),
        f: f64::INFINITY,
        evals: 0,
        generations: 0,
        restarts: 0,
        stop: None,
    };
    while out.evals < opts.max_evals {
        let remaining = opts.max_evals - out.evals;
        let mut points = es.ask();
        let partial = points.len() > remaining;
        points.truncate(remaining);
        let fitness = batch(&points)?;
        out.evals += points.len();
        for (x, f) in points.iter().zip(&fitness) {
            if *f < out.f {
                out.f = *f;
                out.x.clone_from(x);
            }
        }
        if partial || fitness.len() != points.len() {
            break;
        }
        es.tell(&fitness);
        out.generations += 1;
        let Some(stop) = es.stop() else { continue };
        out.stop = Some(stop);
        let next_lambda = (lambda * 2).min(opts.max_lambda.max(lambda));
        let remaining = opts.max_evals - out.evals;
        if out.restarts >= opts.restarts || remaining < 4 * next_lambda {
            break;
        }
        lambda = next_lambda;
        let start: Vec<f64> = if opts.restart_from_best && out.f.is_finite() {
            out.x.clone()
        } else {
            (0..es.dim()).map(|_| restart_rng.uniform()).collect()
        };
        es = Cmaes::new(&start, opts.sigma0, lambda, seeds.next_u64())
            .with_tolerances(opts.tol_x, opts.tol_fun);
        out.restarts += 1;
        out.stop = None;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(f: impl Fn(&[f64]) -> f64, n: usize, opts: &Options) -> Outcome {
        let x0 = vec![0.5; n];
        let out = minimize(&x0, opts, |xs| {
            Ok::<_, ()>(xs.iter().map(|x| f(x)).collect::<Vec<_>>())
        })
        .unwrap();
        eprintln!(
            "n = {n}: f = {:.1e} after {} evals, {} generations, stop {:?}",
            out.f, out.evals, out.generations, out.stop
        );
        out
    }

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|v| (v - 0.3).powi(2)).sum()
    }

    fn ellipsoid(x: &[f64]) -> f64 {
        let n = x.len() as f64;
        x.iter()
            .enumerate()
            .map(|(i, v)| 10f64.powf(6.0 * i as f64 / (n - 1.0)) * (v - 0.3).powi(2))
            .sum()
    }

    /// Rosenbrock on [-2, 2]^n squeezed into the box; optimum at x = 0.75.
    fn rosenbrock(x: &[f64]) -> f64 {
        let y: Vec<f64> = x.iter().map(|v| 4.0 * v - 2.0).collect();
        y.windows(2)
            .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
            .sum()
    }

    fn opts(max_evals: usize, seed: u64) -> Options {
        Options {
            sigma0: 0.3,
            max_evals,
            seed,
            ..Options::default()
        }
    }

    #[test]
    fn rng_is_deterministic_and_well_spread() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        let xs: Vec<u64> = (0..8).map(|_| a.next_u64()).collect();
        let ys: Vec<u64> = (0..8).map(|_| b.next_u64()).collect();
        assert_eq!(xs, ys);
        assert_ne!(Rng::new(43).next_u64(), xs[0]);

        let mut rng = Rng::new(7);
        let n = 200_000;
        let normals: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
        let mean = normals.iter().sum::<f64>() / n as f64;
        let var = normals.iter().map(|z| (z - mean).powi(2)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.01, "mean {mean}");
        assert!((var - 1.0).abs() < 0.01, "variance {var}");
        let uniforms: Vec<f64> = (0..n).map(|_| rng.uniform()).collect();
        assert!(uniforms.iter().all(|u| (0.0..1.0).contains(u)));
        let mean = uniforms.iter().sum::<f64>() / n as f64;
        assert!((mean - 0.5).abs() < 0.005, "uniform mean {mean}");
        assert!((0..1000).all(|_| rng.below(7) < 7));
    }

    #[test]
    fn reflection_folds_into_the_box() {
        assert_eq!(reflect(0.25), 0.25);
        assert!((reflect(-0.25) - 0.25).abs() < 1e-12);
        assert!((reflect(1.25) - 0.75).abs() < 1e-12);
        assert!((reflect(2.25) - 0.25).abs() < 1e-12);
        assert!((reflect(-1.75) - 0.25).abs() < 1e-12);
        assert_eq!(reflect(f64::NAN), 0.5);
    }

    #[test]
    fn jacobi_reconstructs_a_symmetric_matrix() {
        let n = 6;
        let mut rng = Rng::new(3);
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                let v = rng.normal();
                a[i * n + j] = v;
                a[j * n + i] = v;
            }
        }
        let (values, v) = symmetric_eigen(&a, n);
        for i in 0..n {
            for j in 0..n {
                let r: f64 = (0..n)
                    .map(|k| v[i * n + k] * values[k] * v[j * n + k])
                    .sum();
                assert!(
                    (r - a[i * n + j]).abs() < 1e-10,
                    "({i},{j}): {r} vs {}",
                    a[i * n + j]
                );
                let dot: f64 = (0..n).map(|k| v[k * n + i] * v[k * n + j]).sum();
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((dot - want).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn default_parameters_match_hansen() {
        let es = Cmaes::new(&[0.5; 10], 0.3, 0, 1);
        assert_eq!(es.lambda(), 10);
        assert_eq!(es.weights.len(), 5);
        assert!((es.weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(es.weights.windows(2).all(|w| w[0] > w[1]));
        // Tutorial values for n = 10, lambda = 10.
        assert!((es.mueff - 3.16730).abs() < 1e-4, "mueff {}", es.mueff);
        assert!((es.cs - 0.284430).abs() < 1e-5, "cs {}", es.cs);
        assert!((es.c1 - 0.0152839).abs() < 1e-6, "c1 {}", es.c1);
    }

    #[test]
    fn sphere_converges() {
        let out = run(sphere, 10, &opts(4000, 1));
        assert!(
            out.f < 1e-10,
            "sphere f = {:e} after {} evals",
            out.f,
            out.evals
        );
        assert!(out.x.iter().all(|v| (v - 0.3).abs() < 1e-4));
    }

    #[test]
    fn ellipsoid_converges() {
        let out = run(ellipsoid, 10, &opts(20_000, 2));
        assert!(
            out.f < 1e-8,
            "ellipsoid f = {:e} after {} evals",
            out.f,
            out.evals
        );
    }

    #[test]
    fn rosenbrock_converges() {
        let out = run(rosenbrock, 10, &opts(40_000, 3));
        assert!(
            out.f < 1e-6,
            "rosenbrock f = {:e} after {} evals",
            out.f,
            out.evals
        );
        assert!(out.x.iter().all(|v| (v - 0.75).abs() < 1e-3), "{:?}", out.x);
    }

    #[test]
    fn bounds_are_respected_and_a_boundary_optimum_is_reached() {
        // Optimum at x = -0.5 per coordinate, outside the box: the best legal
        // point is the lower corner.
        let mut seen = 0usize;
        let out = minimize(&[0.6; 4], &opts(3000, 4), |xs| {
            seen += xs.len();
            assert!(xs.iter().flatten().all(|v| (0.0..=1.0).contains(v)));
            Ok::<_, ()>(
                xs.iter()
                    .map(|x| x.iter().map(|v| (v + 0.5).powi(2)).sum())
                    .collect(),
            )
        })
        .unwrap();
        assert_eq!(seen, out.evals);
        assert!(out.evals <= 3000);
        assert!(out.x.iter().all(|v| *v < 1e-4), "{:?}", out.x);
    }

    #[test]
    fn same_seed_same_result() {
        let a = run(rosenbrock, 10, &opts(3000, 9));
        let b = run(rosenbrock, 10, &opts(3000, 9));
        let c = run(rosenbrock, 10, &opts(3000, 10));
        assert_eq!(a.x, b.x);
        assert_eq!(a.f.to_bits(), b.f.to_bits());
        assert_eq!(a.evals, b.evals);
        assert_ne!(a.x, c.x);
    }

    #[test]
    fn budget_is_exact_even_mid_generation() {
        let out = run(sphere, 10, &opts(105, 5));
        assert_eq!(out.evals, 105);
    }

    #[test]
    fn ipop_restarts_with_larger_populations() {
        // Rastrigin is multimodal: runs converge early, so restarts happen.
        let rastrigin = |x: &[f64]| {
            x.iter()
                .map(|v| {
                    let y = 10.24 * v - 5.12;
                    y * y - 10.0 * (std::f64::consts::TAU * y).cos() + 10.0
                })
                .sum::<f64>()
        };
        let mut sizes = Vec::new();
        let options = Options {
            sigma0: 0.2,
            max_evals: 20_000,
            restarts: 3,
            tol_fun: 1e-9,
            tol_x: 1e-9,
            seed: 11,
            ..Options::default()
        };
        let out = minimize(&[0.9; 4], &options, |xs| {
            if sizes.last() != Some(&xs.len()) {
                sizes.push(xs.len());
            }
            Ok::<_, ()>(xs.iter().map(|x| rastrigin(x)).collect())
        })
        .unwrap();
        assert!(out.restarts >= 1, "no restart in {} evals", out.evals);
        assert!(sizes.windows(2).any(|w| w[1] == 2 * w[0]), "{sizes:?}");
        assert!(out.f < 10.0, "rastrigin f = {}", out.f);
    }

    #[test]
    fn batch_errors_end_the_search() {
        let mut calls = 0;
        let err = minimize(&[0.5; 3], &opts(1000, 1), |_xs| {
            calls += 1;
            if calls == 3 {
                Err("gpu gone")
            } else {
                Ok(vec![1.0; 7])
            }
        });
        assert_eq!(err.unwrap_err(), "gpu gone");
    }
}
