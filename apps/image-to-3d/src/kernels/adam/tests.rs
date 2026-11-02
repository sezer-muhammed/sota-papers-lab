//! Adam tests. CPU: schedules, step plans, the host plan (write modes, zero-after),
//! the device plan helpers mirrored on the host. GPU (`#[ignore]`; on the host with
//! `scripts/gpu-test.sh <checkout> kernels::adam`): every sparsity mode against an f64
//! reference with raster-style gradient writes, LazyCatchUp against Dense, bf16
//! stochastic rounding, the torch Adam oracle (staged), and timings.

use super::*;

/// FastGS plan for SH rest: every 16 to 15k, every 32 to 20k, every 64 after.
#[test]
fn step_plan_is_piecewise() {
    let p = StepPlan(vec![(0, 16), (15_001, 32), (20_001, 64)]);
    assert!(p.due(16) && !p.due(17) && p.due(14_992) && !p.due(15_000));
    assert!(p.due(15_040) && !p.due(15_024));
    assert!(p.due(20_032) && !p.due(20_064) && p.due(20_096));
    assert!(StepPlan::every_iteration().due(7));
    assert!(StepPlan(vec![]).due(3));
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-6 * b.abs()
}

#[test]
fn means_lr_decays_exponentially_from_init_to_final() {
    let s = LrSchedule::ExpDecay {
        init: 1.6e-4,
        final_: 1.6e-6,
        max_steps: 30_000,
        hold_until: 0,
        scale_by_extent: true,
    };
    assert!(close(s.at(0, 2.0), 3.2e-4));
    assert!(close(s.at(15_000, 1.0), 1.6e-5));
    assert!(close(s.at(40_000, 1.0), 1.6e-6));
    let held = LrSchedule::ExpDecay {
        init: 1.0,
        final_: 0.01,
        max_steps: 100,
        hold_until: 50,
        scale_by_extent: false,
    };
    assert_eq!(held.at(30, 1.0), 1.0);
    assert!(close(held.at(75, 1.0), 0.1));
}

#[test]
fn warmup_then_decay() {
    let s = LrSchedule::WarmupExp {
        base: 2e-3,
        warmup_steps: 500,
        start_factor: 0.01,
        decay_steps: 29_500,
        final_factor: 0.01,
    };
    assert!(close(s.at(0, 1.0), 2e-5));
    assert!(close(s.at(500, 1.0), 2e-3));
    assert!(close(s.at(30_000, 1.0), 2e-5));
}

fn plans(main: StepPlan, sh: StepPlan) -> [GroupConfig; 6] {
    let lrs = [1.6e-4f32, 5e-3, 1e-3, 2.5e-2, 2.5e-3, 1.25e-4];
    std::array::from_fn(|g| GroupConfig {
        lr: LrSchedule::Constant(lrs[g]),
        plan: if g == 5 { sh.clone() } else { main.clone() },
    })
}

/// The host plan: Overwrite only after a step at t − 1, zero-after before a
/// non-due iteration, bias corrections per group step count.
#[test]
fn host_plan_write_modes_follow_the_step_plans() {
    let groups = plans(StepPlan(vec![(0, 1), (10, 4)]), StepPlan(vec![(0, 16)]));
    let mut a = HostPlan::new(groups, AdamConfig::default());
    let p1 = a.plan(1);
    assert_eq!(p1.due_mask, 0b11111);
    assert_eq!(p1.overwrite_mask & 0xff, 0b11111);
    assert_eq!(p1.overwrite_mask >> ZERO_AFTER_SHIFT, 0);
    for t in 2..9 {
        a.plan(t);
    }
    // t = 9: every 1 until 10, then every 4 (due at 12): zero after the step at 9.
    let p9 = a.plan(9);
    assert_eq!((p9.overwrite_mask >> ZERO_AFTER_SHIFT) & 0b11111, 0b11111);
    let p10 = a.plan(10);
    assert_eq!(p10.due_mask & 1, 0);
    a.plan(11);
    let p12 = a.plan(12);
    assert_eq!(p12.due_mask & 1, 1);
    assert_eq!(p12.overwrite_mask & 1, 0, "accumulated over 10..12");
    assert_eq!((p12.overwrite_mask >> ZERO_AFTER_SHIFT) & 1, 1);
    for t in 13..16 {
        a.plan(t);
    }
    let p16 = a.plan(16);
    assert_eq!(p16.due_mask, 0b111111);
    assert_eq!(p16.overwrite_mask & (1 << 5), 0, "sh_rest accumulates");
    assert_eq!(a.steps[0], 9 + 2, "t = 1..9, 12, 16");
    let k = a.steps[0] as i32;
    let want = 1.6e-4 / (1.0 - 0.9f64.powi(k));
    assert!((p16.step_size[0] as f64 - want).abs() < 1e-6 * want);
    assert_eq!(p16.elem_prefix, [0, 3, 6, 10, 11, 14, 59]);
}

/// The device's closed-form plan helpers (adam.cuh), mirrored, agree with
/// brute force over the FastGS plans.
#[test]
fn closed_form_plan_counts_match_brute_force() {
    let (main, sh) = (
        StepPlan(vec![(0, 1), (15_001, 32), (20_001, 64)]),
        StepPlan(vec![(0, 16), (15_001, 32), (20_001, 64)]),
    );
    let (d, n_plans) = StepPlansDev::new(&plans(main.clone(), sh.clone())).unwrap();
    assert_eq!(n_plans, 2);
    assert_eq!(d.plan_id, [0, 0, 0, 0, 0, 1]);
    for (g, plan) in [(0usize, &main), (5, &sh)] {
        for &(a, b) in &[
            (0u32, 100u32),
            (14_990, 15_100),
            (19_900, 20_300),
            (0, 30_000),
            (7, 7),
        ] {
            assert_eq!(
                dev_due_count(&d, g, a, b),
                plan.due_count(a, b),
                "g {g} ({a}, {b}]"
            );
        }
        for t in [
            1u32, 2, 17, 15_000, 15_001, 15_033, 20_001, 20_064, 20_065, 29_999,
        ] {
            assert_eq!(dev_prev_due(&d, g, t), plan.prev_due(t), "g {g} t {t}");
        }
    }
}

/// Mirror of adam.cuh `plan_due_count`.
fn dev_due_count(pl: &StepPlansDev, g: usize, a: u32, b: u32) -> u32 {
    if b <= a {
        return 0;
    }
    let ns = pl.n_seg[g] as usize;
    if ns == 0 {
        return b - a;
    }
    let mut count = 0;
    let first = pl.from[g][0];
    if first > 0 && a + 1 < first {
        let hi = b.min(first - 1);
        if hi > a {
            count += hi - a;
        }
    }
    for i in 0..ns {
        let s0 = pl.from[g][i];
        let s1 = if i + 1 < ns {
            pl.from[g][i + 1]
        } else {
            u32::MAX
        };
        let l = a.max(s0.saturating_sub(1));
        let h = b.min(s1 - 1);
        if h > l {
            let e = pl.every[g][i].max(1);
            count += h / e - l / e;
        }
    }
    count
}

/// Mirror of adam.cuh `plan_prev_due`.
fn dev_prev_due(pl: &StepPlansDev, g: usize, t: u32) -> u32 {
    let mut x = t - 1;
    while x > 0 {
        let (mut e, mut s0) = (1, 0);
        for i in 0..pl.n_seg[g] as usize {
            if pl.from[g][i] <= x {
                e = pl.every[g][i].max(1);
                s0 = pl.from[g][i];
            }
        }
        let c = (x / e) * e;
        if c >= s0 || s0 == 0 {
            return c;
        }
        x = s0 - 1;
    }
    0
}

/// `Adam::plan` without a GPU (same code path through a plain struct).
struct HostPlan {
    adam_groups: [GroupConfig; 6],
    cfg: AdamConfig,
    steps: [u32; 6],
}

impl HostPlan {
    fn new(adam_groups: [GroupConfig; 6], cfg: AdamConfig) -> Self {
        Self {
            adam_groups,
            cfg,
            steps: [0; 6],
        }
    }

    fn plan(&mut self, t: u32) -> AdamIterPlan {
        plan_for(&self.adam_groups, &self.cfg, 1.0, &mut self.steps, t)
    }
}

// ---------------------------------------------------------------- GPU

use crate::frame::{IterParams, IterParamsDev};
use crate::model::GaussiansHost;
use crate::rng::Philox;
use crate::testkit::npz::Npz;
use crate::testkit::oracle;
use crate::testkit::timing::{bench, gpu_snapshot};

/// Device state per group: parameters and (m, v).
type State = (Vec<Vec<f32>>, Vec<Vec<(f32, f32)>>);

fn gpu() -> Gpu {
    Gpu::open().expect("open GPU")
}

/// Element e of group g belongs to Gaussian p (device layouts).
fn gaussian_of(g: usize, e: usize, cap: usize) -> usize {
    if g == 5 {
        e % cap
    } else {
        e / Group::ALL[g].floats()
    }
}

fn live(g: usize, e: usize, n: usize, cap: usize) -> bool {
    if g == 5 {
        e % cap < n
    } else {
        e < n * Group::ALL[g].floats()
    }
}

/// Reference state: parameters and moments (f64 arithmetic, rounded to f32
/// after each step, so the comparison measures the kernel's arithmetic, not
/// f32 storage drift), the semantic gradient sum since each group's last step.
struct Reference {
    p: Vec<Vec<f64>>,
    m: Vec<Vec<f64>>,
    v: Vec<Vec<f64>>,
    acc: Vec<Vec<f64>>,
}

struct Case {
    n: usize,
    cap: usize,
    sparsity: Sparsity,
    groups: [GroupConfig; 6],
}

/// Runs `iters` iterations of raster-style gradient writes + Adam on the GPU and
/// the f64 reference; returns (device params, device moments (m, v), reference).
/// `per_step`: restart the reference from the device state before every step
/// and keep the worst per-step error in `step_err` (params rel. to max(|p|, 1),
/// moments rel. to the group max).
fn run_case(
    gpu: &Gpu,
    c: &Case,
    iters: u32,
    flush_at_end: bool,
    per_step: bool,
    step_err: &mut [(f64, f64, f64); 6],
) -> (State, Reference) {
    let s = &gpu.stream;
    let (n, cap) = (c.n, c.cap);
    let cfg = AdamConfig {
        sparsity: c.sparsity,
        ..Default::default()
    };
    let mut adam = Adam::new(gpu, cap, cfg.clone(), c.groups.clone(), 1.0).unwrap();
    let mut g = Gaussians::alloc(gpu, cap, false).unwrap();
    let mut grads = Grads::alloc(gpu, cap).unwrap();
    let rng = Philox::new(42);
    let mut host = GaussiansHost::zeros(n);
    for (i, v) in host.means.iter_mut().enumerate() {
        *v = rng.uniform(i as u32, 0, 1) * 2.0 - 1.0;
    }
    for (i, v) in host.scales.iter_mut().enumerate() {
        *v = rng.uniform(i as u32, 0, 2) - 3.0;
    }
    for (i, v) in host.rotations.iter_mut().enumerate() {
        *v = rng.uniform(i as u32, 0, 3);
    }
    for (i, v) in host.opacities.iter_mut().enumerate() {
        *v = rng.uniform(i as u32, 0, 4) * 4.0 - 2.0;
    }
    for (i, v) in host.sh0.iter_mut().enumerate() {
        *v = rng.uniform(i as u32, 0, 5) - 0.5;
    }
    for (i, v) in host.sh_rest.iter_mut().enumerate() {
        *v = (rng.uniform(i as u32, 0, 6) - 0.5) * 0.1;
    }
    host.sh_degree = 3;
    g.upload(gpu, &host).unwrap();
    let mut r = Reference {
        p: Group::ALL
            .iter()
            .map(|&grp| {
                s.clone_dtoh(g.param(grp))
                    .unwrap()
                    .iter()
                    .map(|&x| x as f64)
                    .collect()
            })
            .collect(),
        m: Group::ALL
            .iter()
            .map(|&grp| vec![0.0; cap * grp.floats()])
            .collect(),
        v: Group::ALL
            .iter()
            .map(|&grp| vec![0.0; cap * grp.floats()])
            .collect(),
        acc: Group::ALL
            .iter()
            .map(|&grp| vec![0.0; cap * grp.floats()])
            .collect(),
    };
    let mut lv = vec![0u32; cap];
    let mut pd = IterParamsDev::alloc(gpu).unwrap();
    let (b1, b2, eps) = (0.9f64, 0.999f64, 1e-15f64);
    for t in 1..=iters {
        // Visibility: 60% of the Gaussians, plus a block that stays hidden for 40 iterations.
        for (p, l) in lv.iter_mut().enumerate().take(n) {
            let hidden = (100..400).contains(&p) && (20..60).contains(&t);
            if !hidden && rng.uniform(p as u32, t, 7) < 0.6 {
                *l = t;
            }
        }
        s.memcpy_htod(&lv, &mut g.last_visible).unwrap();
        let plan = adam.plan(t);
        // Raster: per group, Overwrite sets visible entries, Accumulate adds to them.
        for (gi, grp) in Group::ALL.iter().enumerate() {
            let mut buf = s.clone_dtoh(grads.get(*grp)).unwrap();
            let ow = plan.overwrite_mask >> gi & 1 == 1;
            for (e, slot) in buf.iter_mut().enumerate() {
                let p = gaussian_of(gi, e, cap);
                if !live(gi, e, n, cap) || lv[p] != t {
                    continue;
                }
                let x = (rng.uniform(e as u32, t, 10 + gi as u32) - 0.5) * 1e-2;
                *slot = if ow { x } else { *slot + x };
                r.acc[gi][e] += x as f64;
            }
            s.memcpy_htod(&buf, grads.get_mut(*grp)).unwrap();
        }
        let mut ip = IterParams::default();
        ip.frame.iteration = t;
        ip.frame.n_gaussians = n as u32;
        ip.adam = plan;
        pd.upload(gpu, &ip, None).unwrap();
        if per_step {
            let (p, mv) = download(gpu, &g, &adam);
            for gi in 0..6 {
                r.p[gi] = p[gi].iter().map(|&x| x as f64).collect();
                r.m[gi] = mv[gi].iter().map(|x| x.0 as f64).collect();
                r.v[gi] = mv[gi].iter().map(|x| x.1 as f64).collect();
            }
        }
        adam.step(gpu, &mut g, &mut grads, &pd).unwrap();
        // Reference step.
        for gi in 0..6 {
            if plan.due_mask >> gi & 1 == 0 {
                continue;
            }
            let k = adam.steps[gi] as i32;
            let lr = match c.groups[gi].lr {
                LrSchedule::Constant(l) => l as f64,
                _ => unreachable!(),
            };
            let (st, bc2) = (lr / (1.0 - b1.powi(k)), 1.0 / (1.0 - b2.powi(k)).sqrt());
            let prev = c.groups[gi].plan.prev_due(t);
            for e in 0..r.p[gi].len() {
                if !live(gi, e, n, cap) {
                    continue;
                }
                let p = gaussian_of(gi, e, cap);
                let skip = c.sparsity != Sparsity::Dense && lv[p] <= prev;
                if skip {
                    continue;
                }
                // f64 arithmetic, f32 storage (what any f32 optimizer keeps between steps).
                let gr = r.acc[gi][e] as f32 as f64;
                r.m[gi][e] = (b1 * r.m[gi][e] + (1.0 - b1) * gr) as f32 as f64;
                r.v[gi][e] = (b2 * r.v[gi][e] + (1.0 - b2) * gr * gr) as f32 as f64;
                r.p[gi][e] =
                    (r.p[gi][e] - st * r.m[gi][e] / (r.v[gi][e].sqrt() * bc2 + eps)) as f32 as f64;
            }
            r.acc[gi].iter_mut().for_each(|a| *a = 0.0);
        }
        if per_step {
            let (p, mv) = download(gpu, &g, &adam);
            for (gi, e) in errors(c, &p, &mv, &r).iter().enumerate() {
                step_err[gi] = (
                    step_err[gi].0.max(e.0),
                    step_err[gi].1.max(e.1),
                    step_err[gi].2.max(e.2),
                );
            }
        }
    }
    if flush_at_end {
        adam.flush(gpu, &mut g, iters).unwrap();
    }
    let (params, moments) = download(gpu, &g, &adam);
    ((params, moments), r)
}

/// Device parameters and (m, v) per group.
fn download(gpu: &Gpu, g: &Gaussians, adam: &Adam) -> State {
    let s = &gpu.stream;
    let params = Group::ALL
        .iter()
        .map(|&grp| s.clone_dtoh(g.param(grp)).unwrap())
        .collect();
    let moments = (0..6)
        .map(|gi| {
            let w = s.clone_dtoh(&adam.moments.groups[gi]).unwrap();
            match adam.cfg.moments {
                MomentPrecision::Bf16Stochastic => w
                    .iter()
                    .map(|x| {
                        let b = x.to_bits();
                        (f32::from_bits(b << 16), f32::from_bits(b & 0xffff_0000))
                    })
                    .collect(),
                _ => w.chunks(2).map(|c| (c[0], c[1])).collect(),
            }
        })
        .collect();
    (params, moments)
}

/// Max error over live elements: params relative to max(|p|, 1), moments
/// relative to the group's max |m| (max v).
fn errors(c: &Case, p: &[Vec<f32>], mv: &[Vec<(f32, f32)>], r: &Reference) -> [(f64, f64, f64); 6] {
    std::array::from_fn(|gi| {
        let (mut ep, mut em, mut ev) = (0f64, 0f64, 0f64);
        let mmax = r.m[gi].iter().fold(1e-30f64, |a, x| a.max(x.abs()));
        let vmax = r.v[gi].iter().fold(1e-30f64, |a, x| a.max(x.abs()));
        for e in 0..r.p[gi].len() {
            if !live(gi, e, c.n, c.cap) {
                continue;
            }
            ep = ep.max((p[gi][e] as f64 - r.p[gi][e]).abs() / r.p[gi][e].abs().max(1.0));
            em = em.max((mv[gi][e].0 as f64 - r.m[gi][e]).abs() / mmax);
            ev = ev.max((mv[gi][e].1 as f64 - r.v[gi][e]).abs() / vmax);
        }
        (ep, em, ev)
    })
}

/// Every sparsity mode against the f64 reference over 100 iterations with
/// raster-style gradient writes: main groups every 1 then every 4 from t = 60
/// (Overwrite → Accumulate with zero-after), SH rest every 16 (Accumulate),
/// n not a multiple of 4 (partial quads). Per step (reference restarted from
/// the device state): params within 1e-6 relative to max(|p|, 1), moments
/// within 1e-6 of the group's max. After 100 accumulated steps the f32 state
/// drifts by rounding (a few ulp per element): params within 1e-5, moments 1e-5.
#[test]
#[ignore]
fn gpu_adam_matches_f64_reference() {
    let gpu = gpu();
    for sparsity in [Sparsity::Dense, Sparsity::VisibleOnly] {
        let c = Case {
            n: 3001,
            cap: 4096,
            sparsity,
            groups: plans(StepPlan(vec![(0, 1), (60, 4)]), StepPlan(vec![(0, 16)])),
        };
        let mut step = [(0.0, 0.0, 0.0); 6];
        run_case(&gpu, &c, 100, false, true, &mut step);
        let ((p, mv), r) = run_case(&gpu, &c, 100, false, false, &mut [(0.0, 0.0, 0.0); 6]);
        for (gi, (ep, em, ev)) in errors(&c, &p, &mv, &r).iter().enumerate() {
            let (sp, sm, sv) = step[gi];
            println!(
                "{sparsity:?} group {gi}: per step param {sp:.2e}, m {sm:.2e}, v {sv:.2e}; \
                 after 100 steps param {ep:.2e}, m {em:.2e}, v {ev:.2e}"
            );
            assert!(
                sp < 1e-6 && sm < 1e-6 && sv < 1e-6,
                "{sparsity:?} group {gi} per step"
            );
            assert!(
                *ep < 1e-5 && *em < 1e-5 && *ev < 1e-5,
                "{sparsity:?} group {gi} accumulated"
            );
        }
    }
}

/// LazyCatchUp after a flush equals Dense (reference = Dense f64) within 1e-5.
#[test]
#[ignore]
fn gpu_lazy_catchup_equals_dense() {
    let gpu = gpu();
    let c = Case {
        n: 3001,
        cap: 4096,
        sparsity: Sparsity::LazyCatchUp,
        groups: plans(StepPlan(vec![(0, 1), (60, 4)]), StepPlan(vec![(0, 16)])),
    };
    let ((p, mv), _) = run_case(&gpu, &c, 100, true, false, &mut [(0.0, 0.0, 0.0); 6]);
    let dense = Case {
        sparsity: Sparsity::Dense,
        ..c
    };
    let (_, r) = run_case(&gpu, &dense, 100, false, false, &mut [(0.0, 0.0, 0.0); 6]);
    for (gi, (ep, em, ev)) in errors(&dense, &p, &mv, &r).iter().enumerate() {
        println!("LazyCatchUp vs Dense group {gi}: param {ep:.2e}, m {em:.2e}, v {ev:.2e}");
        assert!(*ep < 1e-5 && *em < 1e-5 && *ev < 1e-5, "group {gi}");
    }
    // The test discriminates: VisibleOnly (no catch-up) is far from Dense.
    let vis = Case {
        sparsity: Sparsity::VisibleOnly,
        ..dense
    };
    let ((pv, mvv), _) = run_case(&gpu, &vis, 100, false, false, &mut [(0.0, 0.0, 0.0); 6]);
    let worst = errors(&vis, &pv, &mvv, &r)
        .iter()
        .fold(0f64, |a, e| a.max(e.0).max(e.1));
    println!("VisibleOnly vs Dense: worst {worst:.2e}");
    assert!(worst > 1e-3, "VisibleOnly should differ from Dense");
}

/// bf16 moments with stochastic rounding: over 10k steps of a constant
/// gradient the mean of v stays within 1% of f32 (round-to-nearest freezes v).
#[test]
#[ignore]
fn gpu_bf16_stochastic_rounding_tracks_f32() {
    let gpu = gpu();
    let s = &gpu.stream;
    let n = 4096usize;
    let mut means = Vec::new();
    for moments in [
        MomentPrecision::F32,
        MomentPrecision::Bf16Stochastic,
        MomentPrecision::Bf16MomentF32Var,
    ] {
        let groups = plans(StepPlan::every_iteration(), StepPlan::every_iteration());
        let cfg = AdamConfig {
            moments,
            ..Default::default()
        };
        let mut adam = Adam::new(&gpu, n, cfg, groups, 1.0).unwrap();
        let mut g = Gaussians::alloc(&gpu, n, false).unwrap();
        g.n = n as u32;
        let mut grads = Grads::alloc(&gpu, n).unwrap();
        s.memcpy_htod(&vec![1e-3f32; n], &mut grads.opacity)
            .unwrap();
        let mut pd = IterParamsDev::alloc(&gpu).unwrap();
        for t in 1..=10_000u32 {
            let mut plan = adam.plan(t);
            // Opacity only, Accumulate without zeroing: a constant gradient every step.
            plan.due_mask = 1 << 3;
            plan.overwrite_mask = 0;
            let mut ip = IterParams::default();
            ip.frame.iteration = t;
            ip.frame.n_gaussians = n as u32;
            ip.seed = [1, 2];
            ip.adam = plan;
            pd.upload(&gpu, &ip, None).unwrap();
            adam.step(&gpu, &mut g, &mut grads, &pd).unwrap();
        }
        let w = s.clone_dtoh(&adam.moments.groups[3]).unwrap();
        let v: Vec<f64> = match moments {
            MomentPrecision::Bf16Stochastic => w
                .iter()
                .map(|x| f32::from_bits(x.to_bits() & 0xffff_0000) as f64)
                .collect(),
            _ => w.chunks(2).map(|c| c[1] as f64).collect(),
        };
        let mean = v.iter().sum::<f64>() / n as f64;
        println!(
            "{moments:?}: mean v {mean:.6e} (exact {:.6e})",
            1e-6 * (1.0 - 0.999f64.powi(10_000))
        );
        means.push(mean);
    }
    for m in &means[1..] {
        assert!(
            (m - means[0]).abs() < 0.01 * means[0],
            "{m} vs {}",
            means[0]
        );
    }
}

/// Stages one torch.optim.Adam step (gs-oracle `adam`) and compares when the
/// oracle output exists: max |Δ| < 1e-6 · max |ref| on param, exp_avg,
/// exp_avg_sq (torch updates m with lerp, we with fma: last-bit differences).
#[test]
#[ignore]
fn gpu_adam_matches_torch_oracle() {
    let gpu = gpu();
    let s = &gpu.stream;
    let n = 1000usize;
    let dir = oracle::stage_dir();
    std::fs::create_dir_all(&dir).unwrap();
    let rng = Philox::new(3);
    let f = |k: u32, scale: f32, off: f32| -> Vec<f32> {
        (0..3 * n)
            .map(|i| (rng.uniform(i as u32, k, 0) - off) * scale)
            .collect()
    };
    let (param, grad, m, v) = (
        f(1, 2.0, 0.5),
        f(2, 1e-2, 0.5),
        f(3, 1e-3, 0.5),
        f(4, 1e-5, 0.0),
    );
    let step = 7;
    let mut z = Npz::new();
    for (k, x) in [
        ("param", &param),
        ("grad", &grad),
        ("exp_avg", &m),
        ("exp_avg_sq", &v),
    ] {
        z.f32(k, &[n, 3], x.clone()).unwrap();
    }
    z.i32_scalar("step", step).unwrap();
    z.f32_scalar("lr", 1e-3).unwrap();
    oracle::stage(&dir, "adam_means", "adam", &z).unwrap();

    let mut groups = plans(StepPlan::every_iteration(), StepPlan::every_iteration());
    groups[0].lr = LrSchedule::Constant(1e-3);
    let mut adam = Adam::new(&gpu, n, AdamConfig::default(), groups, 1.0).unwrap();
    adam.steps = [step as u32; 6];
    let mut g = Gaussians::alloc(&gpu, n, false).unwrap();
    let mut grads = Grads::alloc(&gpu, n).unwrap();
    let cap = g.cap as usize;
    let pad = |x: &[f32]| {
        let mut b = x.to_vec();
        b.resize(3 * cap, 0.0);
        b
    };
    s.memcpy_htod(&pad(&param), &mut g.mean).unwrap();
    s.memcpy_htod(&pad(&grad), &mut grads.mean).unwrap();
    let mut mv: Vec<f32> = m.iter().zip(&v).flat_map(|(a, b)| [*a, *b]).collect();
    mv.resize(6 * cap, 0.0);
    s.memcpy_htod(&mv, &mut adam.moments.groups[0]).unwrap();
    let mut plan = adam.plan(1);
    plan.due_mask = 1;
    plan.overwrite_mask = 0;
    let mut ip = IterParams::default();
    ip.frame.iteration = 1;
    ip.frame.n_gaussians = n as u32;
    ip.adam = plan;
    let mut pd = IterParamsDev::alloc(&gpu).unwrap();
    pd.upload(&gpu, &ip, None).unwrap();
    adam.step(&gpu, &mut g, &mut grads, &pd).unwrap();
    let p_out = s.clone_dtoh(&g.mean).unwrap()[..3 * n].to_vec();
    let mv_out = s.clone_dtoh(&adam.moments.groups[0]).unwrap();
    let (m_out, v_out): (Vec<f32>, Vec<f32>) = (0..3 * n)
        .map(|i| (mv_out[2 * i], mv_out[2 * i + 1]))
        .unzip();
    let mut ours = Npz::new();
    ours.f32("param_out", &[n, 3], p_out.clone()).unwrap();
    ours.f32("exp_avg_out", &[n, 3], m_out.clone()).unwrap();
    ours.f32("exp_avg_sq_out", &[n, 3], v_out.clone()).unwrap();
    oracle::stage_ours(&dir, "adam_means", "adam", &ours).unwrap();
    let Ok(out) = oracle::load_out(&dir, "adam_means", "adam") else {
        eprintln!(
            "oracle output missing: run `gs-oracle batch {}` on the host",
            dir.display()
        );
        return;
    };
    for (key, mine) in [
        ("param_out", &p_out),
        ("exp_avg_out", &m_out),
        ("exp_avg_sq_out", &v_out),
    ] {
        let want = out.get(key).unwrap().to_f64();
        let scale = want.iter().fold(1e-30f64, |a, x| a.max(x.abs()));
        let d = mine
            .iter()
            .zip(&want)
            .fold(0f64, |a, (x, y)| a.max((*x as f64 - y).abs()));
        println!(
            "adam vs torch {key}: max |d| {d:.2e} = {:.2e} of max |ref| {scale:.2e}",
            d / scale
        );
        assert!(d < 1e-6 * scale, "{key}");
    }
}

/// Timing (CUDA events, median of 50 after 5 warm-ups) at 1M, 3M and 6M
/// Gaussians: all 59 floats dense (Faster-GS parity), main groups only (SH
/// rest not due), main + SH rest (accumulated), VisibleOnly with 50% visible,
/// and bf16 moments. Effective bandwidth from the 28 B (20 B bf16) per element
/// model. `SPL_GPU_LOCK=1` for reported numbers.
#[test]
#[ignore]
fn gpu_adam_bench() {
    let gpu = gpu();
    let s = &gpu.stream;
    println!("gpu before: {:?}", gpu_snapshot());
    let sizes: Vec<usize> = std::env::var("SPL_ADAM_BENCH_N")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_else(|| vec![1_000_000, 3_000_000, 6_000_000]);
    for n in sizes {
        let configs: [(&str, Sparsity, MomentPrecision, u32, u32); 6] = [
            (
                "all 59 dense",
                Sparsity::Dense,
                MomentPrecision::F32,
                0b111111,
                0b111111,
            ),
            (
                "main 14 dense",
                Sparsity::Dense,
                MomentPrecision::F32,
                0b011111,
                0b011111,
            ),
            (
                "main + sh_rest acc",
                Sparsity::Dense,
                MomentPrecision::F32,
                0b111111,
                0b011111,
            ),
            (
                "all 59 visible-only 50%",
                Sparsity::VisibleOnly,
                MomentPrecision::F32,
                0b111111,
                0b111111,
            ),
            (
                "all 59 dense bf16",
                Sparsity::Dense,
                MomentPrecision::Bf16Stochastic,
                0b111111,
                0b111111,
            ),
            (
                "main 14 dense bf16",
                Sparsity::Dense,
                MomentPrecision::Bf16Stochastic,
                0b011111,
                0b011111,
            ),
        ];
        let Ok(mut g) = Gaussians::alloc(&gpu, n, false) else {
            println!("n {n}: skipped (allocation failed)");
            continue;
        };
        let Ok(mut grads) = Grads::alloc(&gpu, n) else {
            println!("n {n}: skipped (allocation failed)");
            continue;
        };
        g.n = n as u32;
        // Half the Gaussians visible at t = 2 (alternating blocks of 64).
        let lv: Vec<u32> = (0..g.cap as usize)
            .map(|p| if (p / 64) % 2 == 0 { 2 } else { 0 })
            .collect();
        s.memcpy_htod(&lv, &mut g.last_visible).unwrap();
        for (name, sparsity, moments, due, ow) in configs {
            let cfg = AdamConfig {
                sparsity,
                moments,
                ..Default::default()
            };
            let groups = plans(StepPlan::every_iteration(), StepPlan::every_iteration());
            let Ok(mut adam) = Adam::new(&gpu, n, cfg, groups, 1.0) else {
                println!("n {n} {name}: skipped (allocation failed)");
                continue;
            };
            let mut plan = adam.plan(2);
            plan.due_mask = due;
            plan.overwrite_mask = ow;
            let mut ip = IterParams::default();
            ip.frame.iteration = 2;
            ip.frame.n_gaussians = n as u32;
            ip.adam = plan;
            let mut pd = IterParamsDev::alloc(&gpu).unwrap();
            pd.upload(&gpu, &ip, None).unwrap();
            let st = bench(&gpu, 5, 50, || adam.step(&gpu, &mut g, &mut grads, &pd)).unwrap();
            let floats: usize = (0..6)
                .filter(|g| due >> g & 1 == 1)
                .map(|g| Group::ALL[g].floats())
                .sum();
            let per = if moments == MomentPrecision::F32 {
                28.0
            } else {
                20.0
            };
            let frac = if sparsity == Sparsity::Dense {
                1.0
            } else {
                0.5
            };
            let gbs = per * frac * (floats * n) as f64 / (st.median_ms * 1e-3) / 1e9;
            println!(
                "adam n {n} {name}: median {:.3} ms (p10 {:.3}, p90 {:.3}), {:.0} GB/s effective",
                st.median_ms, st.p10_ms, st.p90_ms, gbs
            );
        }
    }
    println!("gpu after: {:?}", gpu_snapshot());
}

#[test]
fn one_minus_beta_rounds_from_the_decimal() {
    assert_eq!(one_minus(0.9), 0.1);
    assert_eq!(one_minus(0.999), 0.001);
}
