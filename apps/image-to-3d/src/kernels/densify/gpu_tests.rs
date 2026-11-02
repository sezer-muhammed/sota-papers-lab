//! GPU tests of densification, compaction, Morton sorting and MCMC against
//! the CPU mirrors in `densify.rs` / `model.rs` and gsplat (staged oracle
//! ops `relocation`, `multinomial`), plus the kernel benchmark. All
//! `#[ignore]`: run on the GPU host with
//! `apps/image-to-3d/scripts/gpu-test.sh <checkout> densify`.

#![allow(clippy::needless_range_loop)]

use std::collections::BTreeMap;

use spl_gpu::cudarc::driver::{LaunchConfig, PushKernelArg};

use super::*;
use crate::frame::IterParamsDev;
use crate::kernels::adam::{MomentPrecision, Moments};
use crate::model::{
    COL_COPY, CompactKernels, Compactor, EmitPlan, Gaussians, GaussiansHost, Grads,
    PerGaussianArrays, cap_tail_host, round_cap,
};
use crate::rng::{Philox, normal2};
use crate::testkit::npz::{NpyData, Npz};
use crate::testkit::oracle::{case_path, load_out, stage};
use crate::testkit::timing::{GpuTimer, Stats, bench, gpu_snapshot};

fn gpu() -> Gpu {
    Gpu::open().expect("open GPU")
}

/// SplitMix64: test data only.
struct Rng(u64);

impl Rng {
    fn u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn u32(&mut self) -> u32 {
        (self.u64() >> 32) as u32
    }
    fn f(&mut self) -> f32 {
        (self.u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.f()
    }
    fn chance(&mut self, p: f32) -> bool {
        self.f() < p
    }
}

const EXTENT: f32 = 4.0;

/// Random Gaussians spanning every threshold of the rules at extent 4.
fn random_host(n: usize, r: &mut Rng) -> GaussiansHost {
    let mut h = GaussiansHost::zeros(n);
    for i in 0..n {
        for a in 0..3 {
            h.means[3 * i + a] = r.range(-3.0, 3.0);
            h.scales[3 * i + a] = r.range(0.0004f32.ln(), 0.8f32.ln());
            h.sh0[3 * i + a] = r.range(-1.0, 1.0);
        }
        for a in 0..4 {
            h.rotations[4 * i + a] = r.range(-1.0, 1.0);
        }
        if r.chance(0.01) {
            h.rotations[4 * i..4 * i + 4].copy_from_slice(&[1e-5, 0.0, 0.0, 0.0]);
        }
        h.opacities[i] = logit(r.range(0.001, 0.99));
        for j in 0..45 {
            h.sh_rest[45 * i + j] = r.range(-0.5, 0.5);
        }
    }
    h
}

struct Stats3 {
    vis: Vec<f32>,
    grad: Vec<f32>,
    abs: Vec<f32>,
    count: Vec<u32>,
    score: Vec<f32>,
}

fn random_stats(n: usize, r: &mut Rng) -> Stats3 {
    let mut s = Stats3 {
        vis: vec![0.0; n],
        grad: vec![0.0; n],
        abs: vec![0.0; n],
        count: vec![0; n],
        score: vec![0.0; n],
    };
    for i in 0..n {
        let c = (r.u32() % 20) as f32;
        s.vis[i] = c;
        s.grad[i] = c * r.range(0.0, 4e-4);
        s.abs[i] = c * r.range(0.0, 2.4e-3);
        s.count[i] = r.u32() % 150;
        s.score[i] = s.count[i] as f32 * r.range(0.05, 0.4);
    }
    s
}

fn upload_stats(gpu: &Gpu, cap: usize, st: &Stats3) -> (DensifyStats, ViewScores) {
    let s = &gpu.stream;
    let n = st.vis.len();
    let mut ds = DensifyStats::alloc(gpu, cap, true).unwrap();
    s.memcpy_htod(&st.vis, &mut ds.vis_count.slice_mut(..n))
        .unwrap();
    s.memcpy_htod(&st.grad, &mut ds.grad_sum.slice_mut(..n))
        .unwrap();
    s.memcpy_htod(
        &st.abs,
        &mut ds.grad_abs_sum.as_mut().unwrap().slice_mut(..n),
    )
    .unwrap();
    let mut vs = ViewScores::alloc(gpu, cap).unwrap();
    s.memcpy_htod(&st.count, &mut vs.count.slice_mut(..n))
        .unwrap();
    s.memcpy_htod(&st.score, &mut vs.score.slice_mut(..n))
        .unwrap();
    (ds, vs)
}

fn read_plan(gpu: &Gpu, p: &EmitPlan) -> PlanOut {
    let n = p.n_before as usize;
    let s = &gpu.stream;
    PlanOut {
        out_count: s.clone_dtoh(&p.out_count.slice(..n)).unwrap(),
        clone_flag: s.clone_dtoh(&p.clone_flag.slice(..n)).unwrap(),
        split_flag: s.clone_dtoh(&p.split_flag.slice(..n)).unwrap(),
        priority: s.clone_dtoh(&p.priority.slice(..n)).unwrap(),
        prune_candidates: 0,
    }
}

#[test]
fn ln_1_6_matches_the_cuda_literal() {
    assert_eq!(LN_1_6, 1.6f64.ln() as f32);
    assert!(include_str!("../../../cuda/compact.cu").contains("#define LN_1_6 0.47000363f"));
}

#[test]
fn emission_order_and_capping_on_the_host() {
    // Survivors in order, then clones, then split children.
    let oc = [1, 0, 1, 0, 1];
    let cf = [1, 0, 0, 0, 1];
    let sf = [0, 1, 0, 1, 0];
    let (m, n) = emission_cpu(&oc, &cf, &sf);
    assert_eq!(n, 3 + 2 + 4);
    assert!(m.contains(&(4, 4, true, 0)) && m.contains(&(1, 5, true, 1)));
    assert!(m.contains(&(3, 8, true, 2)) && m.contains(&(2, 1, false, 0)));
    // Capping: room for one tail entry beyond the restored split parents.
    let (mut oc2, mut cf2, mut sf2) = (oc.to_vec(), cf.to_vec(), sf.to_vec());
    let pri = [0.5, 0.9, 0.0, 0.1, 0.7];
    let dropped = cap_tail_host(&mut oc2, &mut cf2, &mut sf2, &pri, 3 + 2 + 1, [3, 2, 2]);
    // Highest priority (split of 1) stays; clone 4, split 3 and clone 0 go.
    assert_eq!(
        (&sf2, &cf2, &oc2),
        (&vec![0, 1, 0, 0, 0], &vec![0; 5], &vec![1, 0, 1, 1, 1])
    );
    assert_eq!(dropped, 1 + 2 + 1);
    assert_eq!(emission_cpu(&oc2, &cf2, &sf2).1, 6);
}

#[test]
fn view_sampling_is_distinct_and_uniform() {
    let rng = Philox::new(7);
    let mut hits = [0u32; 20];
    for t in 0..20_000 {
        let v = sample_views(&rng, 20, 10, t);
        let mut s = v.clone();
        s.sort();
        s.dedup();
        assert_eq!(s.len(), 10);
        v.iter().for_each(|&i| hits[i] += 1);
    }
    // Each view is in half the rounds: 10000 ± 5 sigma (sigma = 71).
    assert!(
        hits.iter().all(|&h| (9_640..10_360).contains(&h)),
        "{hits:?}"
    );
}

#[test]
fn relocation_reference_keeps_the_total_opacity_footprint() {
    // N copies of (o', s') reproduce o: 1 - (1 - o')^N = o.
    let b = binomial_table(MCMC_N_MAX as usize);
    for &o in &[0.05, 0.3, 0.9] {
        for draws in [0, 1, 4] {
            let (on, _) = relocation_cpu(o, draws, &b, MCMC_N_MAX as usize, 0.0);
            let n = (draws + 1) as i32;
            assert!((1.0 - (1.0 - on).powi(n) - o).abs() < 1e-12);
        }
    }
    assert_eq!(b[5 * 50 + 2], 10.0);
}

// ---------------------------------------------------------------- plan

#[test]
#[ignore]
fn gpu_densify_plan_matches_cpu_in_every_mode() {
    let gpu = gpu();
    let n = 40_000;
    let mut r = Rng(1);
    let host = random_host(n, &mut r);
    let st = random_stats(n, &mut r);
    let mut g = Gaussians::alloc(&gpu, n, false).unwrap();
    g.upload(&gpu, &host).unwrap();
    let (ds, vs) = upload_stats(&gpu, n, &st);
    let mut w = Work::new(&gpu, n).unwrap();
    w.score_minmax(&gpu, &vs.score, n as u32).unwrap();
    let x = PlanInputs {
        log_scale: host.scales.clone(),
        quat: host.rotations.clone(),
        opacity: host.opacities.clone(),
        vis_count: st.vis.clone(),
        grad_sum: st.grad.clone(),
        grad_abs_sum: Some(st.abs.clone()),
        count: st.count.clone(),
        score: st.score.clone(),
    };
    let fc = FastGsConfig {
        grad_abs_thresh: 6e-4,
        ..FastGsConfig::default()
    };
    let cases = [
        (
            "adc t=1000",
            DensifyRules::adc(&AdcConfig::default(), EXTENT, 1000),
        ),
        (
            "adc t=5000",
            DensifyRules::adc(&AdcConfig::default(), EXTENT, 5000),
        ),
        ("fastgs t=1000", DensifyRules::fastgs(&fc, EXTENT, 1000, 1)),
        (
            "fastgs t=6000 s=2",
            DensifyRules::fastgs(&fc, EXTENT, 6000, 2),
        ),
        ("final", DensifyRules::fastgs_final(&fc)),
    ];
    for (name, rules) in cases {
        let cand = w.plan(&gpu, &g, &ds, Some(&vs), &rules).unwrap();
        let got = read_plan(&gpu, &w.plan);
        let want = plan_cpu(&x, &rules);
        let diff = (0..n)
            .filter(|&i| {
                got.out_count[i] != want.out_count[i]
                    || got.clone_flag[i] != want.clone_flag[i]
                    || got.split_flag[i] != want.split_flag[i]
                    || got.priority[i].to_bits() != want.priority[i].to_bits()
            })
            .count();
        let count = |v: &[u8]| v.iter().filter(|&&f| f != 0).count();
        println!(
            "{name}: keep {} clone {} split {} prune-cand {} | mismatches {diff}",
            got.out_count.iter().filter(|&&c| c == 1).count(),
            count(&got.clone_flag),
            count(&got.split_flag),
            cand,
        );
        assert_eq!(diff, 0, "{name}");
        assert_eq!(cand, want.prune_candidates, "{name}");
        assert!(
            count(&got.clone_flag) > 0 || rules.mode == PLAN_FINAL,
            "{name}: no clones"
        );
    }
}

// ---------------------------------------------------------------- emission

/// Host copy of every per-Gaussian array of the test store.
#[derive(Clone, Debug, PartialEq)]
struct Store {
    p: GaussiansHost,
    last_visible: Vec<u32>,
    grads: Vec<Vec<f32>>,
    moments: Vec<Vec<f32>>,
    stats: Vec<Vec<f32>>,
}

/// Per Gaussian floats of each group (grads) and moment words (f32: 2 per float).
const FLOATS: [usize; 6] = [3, 3, 4, 1, 3, 45];

struct DevStore {
    g: Gaussians,
    grads: Grads,
    moments: Moments,
    stats: DensifyStats,
}

/// Group `k` of a per-group host vector, element `(i, f)` (planar for SH rest).
fn at(cap: usize, k: usize, i: usize, f: usize, words: usize) -> usize {
    if k == 5 {
        (f * cap + i) * words
    } else {
        (i * FLOATS[k] + f) * words
    }
}

fn random_store(gpu: &Gpu, n: usize, cap: usize, r: &mut Rng) -> (DevStore, Store) {
    let s = &gpu.stream;
    let p = random_host(n, r);
    let mut g = Gaussians::alloc(gpu, cap, false).unwrap();
    g.upload(gpu, &p).unwrap();
    let cap = g.cap as usize;
    let last_visible: Vec<u32> = (0..n).map(|_| r.u32() % 1000).collect();
    s.memcpy_htod(&last_visible, &mut g.last_visible.slice_mut(..n))
        .unwrap();
    let mut grads = Grads::alloc(gpu, cap).unwrap();
    let mut moments = Moments {
        precision: MomentPrecision::F32,
        groups: std::array::from_fn(|k| s.alloc_zeros::<f32>(cap * FLOATS[k] * 2).unwrap()),
    };
    let mut hg = Vec::new();
    let mut hm = Vec::new();
    for (k, grp) in Group::ALL.into_iter().enumerate() {
        let v: Vec<f32> = (0..cap * FLOATS[k]).map(|_| r.range(-1.0, 1.0)).collect();
        s.memcpy_htod(&v, grads.get_mut(grp)).unwrap();
        hg.push(v);
        let m: Vec<f32> = (0..cap * FLOATS[k] * 2)
            .map(|_| r.range(0.0, 1.0))
            .collect();
        s.memcpy_htod(&m, &mut moments.groups[k]).unwrap();
        hm.push(m);
    }
    let mut stats = DensifyStats::alloc(gpu, cap, true).unwrap();
    let mut hs = Vec::new();
    for buf in [
        &mut stats.vis_count,
        &mut stats.grad_sum,
        stats.grad_abs_sum.as_mut().unwrap(),
    ] {
        let v: Vec<f32> = (0..cap).map(|_| r.range(0.0, 5.0)).collect();
        s.memcpy_htod(&v, buf).unwrap();
        hs.push(v);
    }
    let host = Store {
        p,
        last_visible,
        grads: hg,
        moments: hm,
        stats: hs,
    };
    (
        DevStore {
            g,
            grads,
            moments,
            stats,
        },
        host,
    )
}

fn download_store(gpu: &Gpu, d: &DevStore) -> Store {
    let s = &gpu.stream;
    let n = d.g.n as usize;
    Store {
        p: d.g.download(gpu).unwrap(),
        last_visible: s.clone_dtoh(&d.g.last_visible.slice(..n)).unwrap(),
        grads: Group::ALL
            .iter()
            .map(|&k| s.clone_dtoh(d.grads.get(k)).unwrap())
            .collect(),
        moments: d
            .moments
            .groups
            .iter()
            .map(|m| s.clone_dtoh(m).unwrap())
            .collect(),
        stats: [
            &d.stats.vis_count,
            &d.stats.grad_sum,
            d.stats.grad_abs_sum.as_ref().unwrap(),
        ]
        .iter()
        .map(|b| s.clone_dtoh(*b).unwrap())
        .collect(),
    }
}

fn rotmat(q: &[f32]) -> [f32; 9] {
    let inv = 1.0 / (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let (w, x, y, z) = (q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv);
    [
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - w * z),
        2.0 * (x * z + w * y),
        2.0 * (x * y + w * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - w * x),
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
        1.0 - 2.0 * (x * x + y * y),
    ]
}

/// The emission applied to the host store (split means in f64-free f32, so
/// they are compared with a tolerance; everything else bit for bit).
#[allow(clippy::too_many_arguments)]
fn emit_host(
    before: &Store,
    cap: usize,
    oc: &[u32],
    cf: &[u8],
    sf: &[u8],
    rng: Philox,
    t: u32,
) -> Store {
    let (moves, n_new) = emission_cpu(oc, cf, sf);
    let mut out = before.clone();
    out.p = GaussiansHost::zeros(n_new);
    out.p.antialiased = before.p.antialiased;
    out.last_visible = vec![0; n_new];
    for (src, dst, is_new, child) in moves {
        let b = &before.p;
        let o = &mut out.p;
        o.means[3 * dst..3 * dst + 3].copy_from_slice(&b.means[3 * src..3 * src + 3]);
        o.scales[3 * dst..3 * dst + 3].copy_from_slice(&b.scales[3 * src..3 * src + 3]);
        o.rotations[4 * dst..4 * dst + 4].copy_from_slice(&b.rotations[4 * src..4 * src + 4]);
        o.opacities[dst] = b.opacities[src];
        o.sh0[3 * dst..3 * dst + 3].copy_from_slice(&b.sh0[3 * src..3 * src + 3]);
        o.sh_rest[45 * dst..45 * dst + 45].copy_from_slice(&b.sh_rest[45 * src..45 * src + 45]);
        out.last_visible[dst] = before.last_visible[src];
        if child > 0 {
            let rm = rotmat(&b.rotations[4 * src..4 * src + 4]);
            let c = child as u32 - 1;
            let d = rng.draw4(2 * src as u32 + c, t, crate::kernels::layout::RNG_SPLIT);
            let (e0, e1) = normal2(d[0], d[1]);
            let (e2, _) = normal2(d[2], d[3]);
            let s: Vec<f32> = (0..3).map(|a| b.scales[3 * src + a].exp()).collect();
            let v = [s[0] * e0, s[1] * e1, s[2] * e2];
            for a in 0..3 {
                o.means[3 * dst + a] +=
                    rm[3 * a] * v[0] + rm[3 * a + 1] * v[1] + rm[3 * a + 2] * v[2];
                o.scales[3 * dst + a] -= LN_1_6;
            }
        }
        for k in 0..6 {
            for f in 0..FLOATS[k] {
                let (gs, gd) = (at(cap, k, src, f, 1), at(cap, k, dst, f, 1));
                out.grads[k][gd] = if is_new { 0.0 } else { before.grads[k][gs] };
                let (ms, md) = (at(cap, k, src, f, 2), at(cap, k, dst, f, 2));
                for w in 0..2 {
                    out.moments[k][md + w] = if is_new {
                        0.0
                    } else {
                        before.moments[k][ms + w]
                    };
                }
            }
        }
        for a in 0..3 {
            out.stats[a][dst] = if is_new { 0.0 } else { before.stats[a][src] };
        }
    }
    out
}

/// Compares the live part of two stores: exact except split-child means.
fn assert_store_eq(got: &Store, want: &Store, cap: usize) {
    let n = want.p.len();
    assert_eq!(got.p.len(), n, "n");
    let mut worst = 0f32;
    for (a, b) in got.p.means.iter().zip(&want.p.means) {
        worst = worst.max((a - b).abs() / (1.0 + b.abs()));
    }
    assert!(worst < 2e-5, "means rel err {worst}");
    assert_eq!(got.p.scales, want.p.scales, "log scales");
    assert_eq!(got.p.rotations, want.p.rotations, "quats");
    assert_eq!(got.p.opacities, want.p.opacities, "opacities");
    assert_eq!(got.p.sh0, want.p.sh0, "sh0");
    assert_eq!(got.p.sh_rest, want.p.sh_rest, "sh_rest");
    assert_eq!(got.last_visible, want.last_visible, "last_visible");
    for k in 0..6 {
        for i in 0..n {
            for f in 0..FLOATS[k] {
                let j = at(cap, k, i, f, 1);
                assert_eq!(
                    got.grads[k][j], want.grads[k][j],
                    "grad group {k} at {i}.{f}"
                );
                let m = at(cap, k, i, f, 2);
                assert_eq!(
                    got.moments[k][m..m + 2],
                    want.moments[k][m..m + 2],
                    "moments {k} at {i}.{f}"
                );
            }
        }
    }
    for a in 0..3 {
        assert_eq!(got.stats[a][..n], want.stats[a][..n], "stats {a}");
    }
    println!("  store matches (split means max rel err {worst:.2e})");
}

fn random_plan(n: usize, r: &mut Rng) -> (Vec<u32>, Vec<u8>, Vec<u8>, Vec<f32>) {
    let mut oc = vec![1u32; n];
    let mut cf = vec![0u8; n];
    let mut sf = vec![0u8; n];
    let mut pri = vec![0f32; n];
    for i in 0..n {
        if r.chance(0.06) {
            sf[i] = 1;
            oc[i] = 0;
        } else {
            if r.chance(0.12) {
                oc[i] = 0;
            } else if r.chance(0.05) {
                oc[i] = 2;
            }
            cf[i] = u8::from(r.chance(0.07));
        }
        pri[i] = r.f();
    }
    (oc, cf, sf, pri)
}

fn upload_plan(gpu: &Gpu, plan: &mut EmitPlan, oc: &[u32], cf: &[u8], sf: &[u8], pri: &[f32]) {
    let s = &gpu.stream;
    let n = oc.len();
    plan.n_before = n as u32;
    s.memcpy_htod(oc, &mut plan.out_count.slice_mut(..n))
        .unwrap();
    s.memcpy_htod(cf, &mut plan.clone_flag.slice_mut(..n))
        .unwrap();
    s.memcpy_htod(sf, &mut plan.split_flag.slice_mut(..n))
        .unwrap();
    s.memcpy_htod(pri, &mut plan.priority.slice_mut(..n))
        .unwrap();
}

fn emit_dev(
    gpu: &Gpu,
    c: &mut Compactor,
    plan: &mut EmitPlan,
    d: &mut DevStore,
    rng: Philox,
    t: u32,
) -> crate::model::EmitResult {
    let s = gpu.stream.clone();
    let cap = d.g.cap;
    let mut extra = PerGaussianArrays::new();
    extra.add_grads(&s, &mut d.grads);
    extra.add_moments(&s, &mut d.moments, cap);
    extra.add_stats(&s, &mut d.stats);
    c.emit(gpu, plan, &mut d.g, &extra, rng, t).unwrap()
}

#[test]
#[ignore]
fn gpu_emission_matches_the_host_model_for_every_array() {
    let gpu = gpu();
    let rng = Philox::new(0xDE5);
    for (n, cap, seed) in [(5_000usize, 8_192usize, 2u64), (30_011, 40_960, 3)] {
        let mut r = Rng(seed);
        let (mut d, before) = random_store(&gpu, n, cap, &mut r);
        let cap = d.g.cap as usize;
        let (oc, cf, sf, pri) = random_plan(n, &mut r);
        let mut plan = EmitPlan::alloc(&gpu, cap).unwrap();
        upload_plan(&gpu, &mut plan, &oc, &cf, &sf, &pri);
        let mut c = Compactor::new(&gpu, cap).unwrap();
        let res = emit_dev(&gpu, &mut c, &mut plan, &mut d, rng, 1234);
        println!("n {n} cap {cap}: {res:?}");
        assert_eq!(res.capped, 0);
        let want = emit_host(&before, cap, &oc, &cf, &sf, rng, 1234);
        assert_store_eq(&download_store(&gpu, &d), &want, cap);
    }
}

#[test]
#[ignore]
fn gpu_emission_caps_by_priority_when_full() {
    let gpu = gpu();
    let rng = Philox::new(5);
    let n = 9_200;
    let mut r = Rng(9);
    let (mut d, before) = random_store(&gpu, n, n, &mut r);
    let cap = d.g.cap as usize; // 9216: survivors ~7.6k, tail ~1.7k wanted
    let (oc, cf, sf, pri) = random_plan(n, &mut r);
    let mut plan = EmitPlan::alloc(&gpu, cap).unwrap();
    upload_plan(&gpu, &mut plan, &oc, &cf, &sf, &pri);
    let mut c = Compactor::new(&gpu, cap).unwrap();
    let res = emit_dev(&gpu, &mut c, &mut plan, &mut d, rng, 77);
    let (mut oc2, mut cf2, mut sf2) = (oc.clone(), cf.clone(), sf.clone());
    let tot = [
        oc.iter().filter(|&&x| x != 0).count() as u32,
        cf.iter().filter(|&&x| x != 0).count() as u32,
        sf.iter().filter(|&&x| x != 0).count() as u32,
    ];
    let dropped = cap_tail_host(&mut oc2, &mut cf2, &mut sf2, &pri, cap as u32, tot);
    println!("capped: {res:?}, host dropped {dropped}");
    assert!(res.capped > 0 && res.capped == dropped);
    assert!(res.n_after <= cap as u32 && res.n_after + 1 >= cap as u32);
    let want = emit_host(&before, cap, &oc2, &cf2, &sf2, rng, 77);
    assert_store_eq(&download_store(&gpu, &d), &want, cap);
}

// ---------------------------------------------------------------- sort, Morton

#[test]
#[ignore]
fn gpu_radix_sort_is_stable_and_exact() {
    let gpu = gpu();
    let s = &gpu.stream;
    for (n, bits, mask) in [
        (1_000_003usize, 32u32, u32::MAX),
        (300_000, 30, 0x3fff_ffff),
        (77, 32, 0xff),
    ] {
        let mut r = Rng(n as u64);
        let keys: Vec<u32> = (0..n)
            .map(|_| r.u32() & mask & if n == 77 { 0x7 } else { 0xffff_f00f })
            .collect();
        let mut c = Compactor::new(&gpu, n).unwrap();
        s.memcpy_htod(&keys, &mut c.keys[0].slice_mut(..n)).unwrap();
        let idx: Vec<u32> = (0..n as u32).collect();
        s.memcpy_htod(&idx, &mut c.vals[0].slice_mut(..n)).unwrap();
        c.sort(&gpu, n as u32, bits).unwrap();
        let gk = s.clone_dtoh(&c.keys[0].slice(..n)).unwrap();
        let gv = s.clone_dtoh(&c.vals[0].slice(..n)).unwrap();
        let mut want: Vec<(u32, u32)> = keys.iter().copied().zip(idx).collect();
        want.sort_by_key(|&(k, _)| k);
        assert!(gk.iter().zip(&want).all(|(a, b)| *a == b.0), "keys n={n}");
        assert!(
            gv.iter().zip(&want).all(|(a, b)| *a == b.1),
            "stability n={n}"
        );
        println!("radix n={n} bits={bits}: exact and stable");
    }
}

fn morton_cpu(m: &[f32], lo: [f32; 3], inv: f32) -> u32 {
    let spread = |v: u32| (0..10).fold(0u32, |acc, b| acc | (((v >> b) & 1) << (3 * b)));
    (0..3).fold(0, |c, a| {
        let u = ((m[a] - lo[a]) * inv).clamp(0.0, 1.0);
        c | (spread((u * 1023.0) as u32) << a)
    })
}

#[test]
#[ignore]
fn gpu_morton_sort_permutes_every_array_consistently() {
    let gpu = gpu();
    let n = 200_000;
    let mut r = Rng(11);
    let (mut d, before) = random_store(&gpu, n, n, &mut r);
    let cap = d.g.cap as usize;
    let mut c = Compactor::new(&gpu, cap).unwrap();
    let s = gpu.stream.clone();
    {
        let mut extra = PerGaussianArrays::new();
        extra.add_grads(&s, &mut d.grads);
        extra.add_moments(&s, &mut d.moments, cap as u32);
        extra.add_stats(&s, &mut d.stats);
        c.morton_sort(&gpu, &mut d.g, &extra).unwrap();
    }
    let codes = s.clone_dtoh(&c.keys[0].slice(..n)).unwrap();
    let perm = s.clone_dtoh(&c.vals[0].slice(..n)).unwrap();
    assert!(codes.windows(2).all(|w| w[0] <= w[1]), "codes sorted");
    let mut seen = vec![false; n];
    perm.iter().for_each(|&p| seen[p as usize] = true);
    assert!(seen.iter().all(|&x| x), "bijection");
    // CPU codes in the permuted order agree with the GPU's (fast-math rcp may flip a boundary).
    let m = &before.p.means;
    let lo: [f32; 3] =
        std::array::from_fn(|a| (0..n).map(|i| m[3 * i + a]).fold(f32::INFINITY, f32::min));
    let ext = (0..3)
        .map(|a| {
            (0..n)
                .map(|i| m[3 * i + a])
                .fold(f32::NEG_INFINITY, f32::max)
                - lo[a]
        })
        .fold(0f32, f32::max);
    let off = perm
        .iter()
        .zip(&codes)
        .filter(|&(&p, &k)| morton_cpu(&m[3 * p as usize..], lo, 1.0 / ext) != k)
        .count();
    assert_eq!(off, 0, "{off} codes differ from the CPU");
    let oc = vec![1u32; n];
    let after = download_store(&gpu, &d);
    // Identity emission as the base, then the gather perm[i] -> i on top.
    let mut scratch = emit_host(
        &before,
        cap,
        &oc,
        &vec![0; n],
        &vec![0; n],
        Philox::new(0),
        0,
    );
    for (i, &p) in perm.iter().enumerate() {
        let p = p as usize;
        let (b, o) = (&before.p, &mut scratch.p);
        o.means[3 * i..3 * i + 3].copy_from_slice(&b.means[3 * p..3 * p + 3]);
        o.scales[3 * i..3 * i + 3].copy_from_slice(&b.scales[3 * p..3 * p + 3]);
        o.rotations[4 * i..4 * i + 4].copy_from_slice(&b.rotations[4 * p..4 * p + 4]);
        o.opacities[i] = b.opacities[p];
        o.sh0[3 * i..3 * i + 3].copy_from_slice(&b.sh0[3 * p..3 * p + 3]);
        o.sh_rest[45 * i..45 * i + 45].copy_from_slice(&b.sh_rest[45 * p..45 * p + 45]);
        scratch.last_visible[i] = before.last_visible[p];
        for k in 0..6 {
            for f in 0..FLOATS[k] {
                scratch.grads[k][at(cap, k, i, f, 1)] = before.grads[k][at(cap, k, p, f, 1)];
                let (md, ms) = (at(cap, k, i, f, 2), at(cap, k, p, f, 2));
                scratch.moments[k][md] = before.moments[k][ms];
                scratch.moments[k][md + 1] = before.moments[k][ms + 1];
            }
        }
        for a in 0..3 {
            scratch.stats[a][i] = before.stats[a][p];
        }
    }
    assert_store_eq(&after, &scratch, cap);
    println!("morton n={n}: {off} boundary codes differ from the CPU, every array permuted");
}

// ---------------------------------------------------------------- VCP sampling

/// Chi-square critical value, df 19, p = 0.01.
const CHI2_19_P01: f64 = 36.19;

#[test]
#[ignore]
fn gpu_gumbel_top_k_matches_multinomial_without_replacement() {
    let gpu = gpu();
    let s = &gpu.stream;
    let (n, k, trials) = (20usize, 5usize, 200_000usize);
    let score: Vec<f32> = (0..n)
        .map(|i| (i as f32 * 0.37).sin().abs() * 3.0 + 0.1 * i as f32)
        .collect();
    let mut w = Work::new(&gpu, n * trials).unwrap();
    let ds = s.clone_htod(&score).unwrap();
    w.score_minmax(&gpu, &ds, n as u32).unwrap();
    // One launch over trials x n elements: element t*n + j has score_j.
    let tiled: Vec<f32> = (0..trials).flat_map(|_| score.iter().copied()).collect();
    let dt = s.clone_htod(&tiled).unwrap();
    let total = (n * trials) as u32;
    let seed = Uint2::from(Philox::new(99).seed);
    let it = 4500u32;
    {
        let [k0, _] = &mut w.compactor.keys;
        let [v0, _] = &mut w.compactor.vals;
        let mut l = s.launch_builder(&w.k.gumbel_keys);
        l.arg(&dt)
            .arg(&w.minmax)
            .arg(&seed)
            .arg(&it)
            .arg(&total)
            .arg(k0)
            .arg(v0);
        // SAFETY: gumbel_keys(score, minmax, seed, iteration, n, keys, idx), n = trials * 20.
        unsafe { l.launch(LaunchConfig::for_num_elems(total)) }.unwrap();
    }
    let keys = s
        .clone_dtoh(&w.compactor.keys[0].slice(..n * trials))
        .unwrap();
    let mut got = vec![0u64; n];
    for t in 0..trials {
        let mut row: Vec<(u32, usize)> = (0..n).map(|j| (keys[t * n + j], j)).collect();
        row.sort();
        row[..k].iter().for_each(|&(_, j)| got[j] += 1);
    }
    // Reference: sequential draws without replacement, p ∝ w (torch.multinomial's definition).
    let (lo, hi) = score_range(&score);
    let wts: Vec<f64> = score
        .iter()
        .map(|&x| 1.0 / (1e-6 + 1.0 - prune_score(x, (lo, hi)) as f64))
        .collect();
    let mut r = Rng(1);
    let ref_trials = 1_000_000;
    let mut want = vec![0u64; n];
    for _ in 0..ref_trials {
        let mut left = wts.clone();
        for _ in 0..k {
            let tot: f64 = left.iter().sum();
            let mut u = r.f() as f64 * tot;
            let mut pick = n - 1;
            for (j, &x) in left.iter().enumerate() {
                if u < x {
                    pick = j;
                    break;
                }
                u -= x;
            }
            want[pick] += 1;
            left[pick] = 0.0;
        }
    }
    let chi2 = |obs: &[u64], p: &[f64], t: usize| -> f64 {
        obs.iter()
            .zip(p)
            .map(|(&o, &p)| (o as f64 - p * t as f64).powi(2) / (p * t as f64))
            .sum()
    };
    let p_ref: Vec<f64> = want.iter().map(|&c| c as f64 / ref_trials as f64).collect();
    let x = chi2(&got, &p_ref, trials);
    println!(
        "Gumbel top-{k} of {n} over {trials} trials vs sequential reference: chi2 {x:.2} (crit {CHI2_19_P01})"
    );
    assert!(x < CHI2_19_P01, "chi2 {x}");
    // gsplat / torch.multinomial, staged (run `gs-oracle batch` on the host, re-run the test).
    let dir = crate::testkit::out_dir("oracle-densify");
    let mut inp = Npz::new();
    inp.f32("weights", &[n], wts.iter().map(|&x| x as f32).collect())
        .unwrap();
    inp.i32_scalar("k", k as i32).unwrap();
    inp.i32_scalar("trials", 1_000_000).unwrap();
    stage(&dir, "densify_vcp", "multinomial", &inp).unwrap();
    if case_path(&dir, "densify_vcp", "multinomial", "out").is_file() {
        let o = load_out(&dir, "densify_vcp", "multinomial").unwrap();
        let c = o.get("counts").unwrap().to_f64();
        let p_t: Vec<f64> = c.iter().map(|&x| x / 1e6).collect();
        let xt = chi2(&got, &p_t, trials);
        println!("  vs torch.multinomial(replacement=False), 1e6 trials: chi2 {xt:.2}");
        assert!(xt < CHI2_19_P01, "chi2 vs torch {xt}");
    } else {
        println!("  torch.multinomial reference staged in {}", dir.display());
    }
}

// ---------------------------------------------------------------- MCMC

#[test]
#[ignore]
fn gpu_mcmc_relocation_formula_matches_gsplat_and_f64() {
    let gpu = gpu();
    let s = &gpu.stream;
    let n_max = MCMC_N_MAX;
    let ops = [0.006f32, 0.05, 0.2, 0.5, 0.8, 0.97];
    let rows: Vec<(f32, u32)> = ops
        .iter()
        .flat_map(|&o| (0..n_max).map(move |d| (o, d)))
        .collect();
    let n = rows.len();
    let mut host = GaussiansHost::zeros(n);
    for (i, &(o, _)) in rows.iter().enumerate() {
        host.opacities[i] = logit(o);
        host.scales[3 * i..3 * i + 3].copy_from_slice(&[-2.0, -3.0, -1.5]);
    }
    let mut g = Gaussians::alloc(&gpu, n, false).unwrap();
    g.upload(&gpu, &host).unwrap();
    let mut mw = McmcWork::new(&gpu, n).unwrap();
    let draws: Vec<u32> = rows.iter().map(|r| r.1).collect();
    s.memcpy_htod(&draws, &mut mw.draws.slice_mut(..n)).unwrap();
    let (nn, mo) = (n as u32, 0.005f32);
    let mut l = s.launch_builder(&mw.k.adjust);
    l.arg(&mw.draws)
        .arg(&mw.binoms)
        .arg(&mw.n_max)
        .arg(&nn)
        .arg(&mo)
        .arg(&mut g.opacity)
        .arg(&mut g.log_scale);
    // SAFETY: mcmc_adjust(draws, binoms, n_max, n, min_opacity, opacity, log_scale), one per row.
    unsafe { l.launch(LaunchConfig::for_num_elems(nn)) }.unwrap();
    let out = g.download(&gpu).unwrap();
    let b = binomial_table(n_max as usize);
    let mut worst_by_n = BTreeMap::new();
    for (i, &(o, d)) in rows.iter().enumerate() {
        if d == 0 {
            assert_eq!(out.opacities[i], host.opacities[i]);
            continue;
        }
        let (on, lc) = relocation_cpu(o as f64, d, &b, n_max as usize, mo as f64);
        let go = 1.0 / (1.0 + (-out.opacities[i] as f64).exp());
        let gs = (out.scales[3 * i] - host.scales[3 * i]) as f64;
        let e = ((go - on) / on).abs().max((gs - lc).abs());
        let slot = worst_by_n.entry((d + 1).min(n_max)).or_insert(0f64);
        *slot = slot.max(e);
    }
    let worst_small = worst_by_n.range(..=2).map(|(_, &e)| e).fold(0.0, f64::max);
    let worst_all = worst_by_n.values().fold(0.0, |a: f64, &b| a.max(b));
    println!(
        "relocation vs f64, worst rel err by N: {:?}",
        worst_by_n
            .iter()
            .filter(|(k, _)| [2, 4, 8, 16, 32, 50].contains(*k))
            .collect::<Vec<_>>()
    );
    // f32 with fast-math powf: the alternating sum loses digits as N grows.
    assert!(worst_small < 1e-5, "N <= 2: {worst_small}");
    assert!(worst_all < 1e-3, "all N: {worst_all}");
    // gsplat's kernel: same float sum, so it should agree for every N.
    let dir = crate::testkit::out_dir("oracle-densify");
    let mut inp = Npz::new();
    let o_in: Vec<f32> = rows
        .iter()
        .map(|r| 1.0 / (1.0 + (-logit(r.0)).exp()))
        .collect();
    inp.f32("opacities", &[n], o_in).unwrap();
    inp.f32(
        "scales",
        &[n, 3],
        (0..n)
            .flat_map(|_| [(-2f32).exp(), (-3f32).exp(), (-1.5f32).exp()])
            .collect(),
    )
    .unwrap();
    inp.insert(
        "ratios",
        &[n],
        NpyData::I32(draws.iter().map(|&d| (d + 1).min(n_max) as i32).collect()),
    )
    .unwrap();
    inp.i32_scalar("n_max", n_max as i32).unwrap();
    stage(&dir, "densify_reloc", "relocation", &inp).unwrap();
    if case_path(&dir, "densify_reloc", "relocation", "out").is_file() {
        let o = load_out(&dir, "densify_reloc", "relocation").unwrap();
        let no = o.get("new_opacities_clamped").unwrap().to_f64();
        let ns = o.get("new_scales").unwrap().to_f64();
        let mut worst = BTreeMap::new();
        for (i, &(_, d)) in rows.iter().enumerate() {
            if d == 0 {
                continue;
            }
            let go = 1.0 / (1.0 + (-out.opacities[i] as f64).exp());
            let gs = (out.scales[3 * i] as f64).exp();
            let e = ((go - no[i]) / no[i])
                .abs()
                .max(((gs - ns[3 * i]) / ns[3 * i]).abs());
            let slot = worst.entry((d + 1).min(n_max)).or_insert(0f64);
            *slot = slot.max(e);
        }
        let all = worst.values().fold(0.0, |a: f64, &b| a.max(b));
        println!(
            "  vs gsplat compute_relocation, worst rel err {all:.2e} (by N: {:?})",
            worst
                .iter()
                .filter(|(k, _)| [2, 8, 32, 50].contains(*k))
                .collect::<Vec<_>>()
        );
        assert!(all < 1e-4, "vs gsplat {all}");
    } else {
        println!("  gsplat reference staged in {}", dir.display());
    }
}

#[test]
#[ignore]
fn gpu_mcmc_round_relocates_and_grows() {
    let gpu = gpu();
    let s = gpu.stream.clone();
    let n = 20_000;
    let mut r = Rng(21);
    let (mut d, _) = random_store(&gpu, n, 24_000, &mut r);
    // 10% dead.
    let mut host = d.g.download(&gpu).unwrap();
    for i in 0..n {
        if i % 10 == 3 {
            host.opacities[i] = logit(0.002);
        }
    }
    d.g.upload(&gpu, &host).unwrap();
    let cap = d.g.cap;
    let cfg = McmcConfig {
        grow: 1.0,
        cap_max: cap,
        ..McmcConfig::default()
    };
    let mut mw = McmcWork::new(&gpu, cap as usize).unwrap();
    let before = download_store(&gpu, &d);
    let (relocated, added) = {
        let mut extra = PerGaussianArrays::new();
        extra.add_grads(&s, &mut d.grads);
        extra.add_moments(&s, &mut d.moments, cap);
        mcmc_round(&gpu, &mut mw, &cfg, &mut d.g, &extra, Philox::new(3), 700).unwrap()
    };
    let dl = logit(0.005);
    let want_dead: Vec<u32> = (0..n)
        .filter(|&i| {
            let q = &host.rotations[4 * i..4 * i + 4];
            host.opacities[i] <= dl || q.iter().map(|x| x * x).sum::<f32>() < 1e-8
        })
        .map(|i| i as u32)
        .collect();
    assert_eq!((relocated, added), (want_dead.len() as u32, 0));
    let dead = s
        .clone_dtoh(&mw.dead_list.slice(..relocated as usize))
        .unwrap();
    let src = s.clone_dtoh(&mw.src.slice(..relocated as usize)).unwrap();
    assert_eq!(dead, want_dead);
    let after = download_store(&gpu, &d);
    let mut hist = BTreeMap::<u32, u32>::new();
    src.iter().for_each(|&x| *hist.entry(x).or_default() += 1);
    let b = binomial_table(MCMC_N_MAX as usize);
    let mut worst = 0f64;
    for (&dd, &ss) in dead.iter().zip(&src) {
        let (dd, ss) = (dd as usize, ss as usize);
        assert!(
            want_dead.binary_search(&(ss as u32)).is_err(),
            "source {ss} is dead"
        );
        // Destination is a copy of the adjusted source.
        assert_eq!(after.p.opacities[dd], after.p.opacities[ss]);
        assert_eq!(
            after.p.means[3 * dd..3 * dd + 3],
            before.p.means[3 * ss..3 * ss + 3]
        );
        assert_eq!(
            after.p.sh_rest[45 * dd..45 * dd + 45],
            before.p.sh_rest[45 * ss..45 * ss + 45]
        );
        let o = 1.0 / (1.0 + (-before.p.opacities[ss] as f64).exp());
        let (on, lc) = relocation_cpu(o, hist[&(ss as u32)], &b, MCMC_N_MAX as usize, 0.005);
        let go = 1.0 / (1.0 + (-after.p.opacities[ss] as f64).exp());
        worst = worst
            .max(((go - on) / on).abs())
            .max(((after.p.scales[3 * ss] - before.p.scales[3 * ss]) as f64 - lc).abs());
        for k in 0..6 {
            for f in 0..FLOATS[k] {
                for i in [dd, ss] {
                    let m = at(cap as usize, k, i, f, 2);
                    assert_eq!(
                        after.moments[k][m..m + 2],
                        [0.0, 0.0],
                        "moments zero at {i}"
                    );
                }
            }
        }
    }
    println!(
        "relocated {relocated} into {} distinct sources; (o', s') rel err {worst:.1e}",
        hist.len()
    );
    assert!(worst < 1e-3, "{worst}");
    // Growth: +5%, appended rows are copies of sources.
    let cfg = McmcConfig {
        grow: 1.05,
        cap_max: cap,
        ..McmcConfig::default()
    };
    let before = download_store(&gpu, &d);
    let (_, added) = {
        let mut extra = PerGaussianArrays::new();
        extra.add_grads(&s, &mut d.grads);
        extra.add_moments(&s, &mut d.moments, cap);
        mcmc_round(&gpu, &mut mw, &cfg, &mut d.g, &extra, Philox::new(3), 800).unwrap()
    };
    assert_eq!(added, (n as f64 * 1.05) as u32 - n as u32);
    assert_eq!(d.g.n, n as u32 + added);
    let src = s.clone_dtoh(&mw.src.slice(..added as usize)).unwrap();
    let after = download_store(&gpu, &d);
    for (j, &ss) in src.iter().enumerate() {
        let (dd, ss) = (n + j, ss as usize);
        assert_eq!(after.p.opacities[dd], after.p.opacities[ss]);
        assert_eq!(
            after.p.scales[3 * dd..3 * dd + 3],
            after.p.scales[3 * ss..3 * ss + 3]
        );
        assert_eq!(
            after.p.rotations[4 * dd..4 * dd + 4],
            before.p.rotations[4 * ss..4 * ss + 4]
        );
        assert_eq!(after.grads[0][3 * dd], 0.0);
    }
    println!("grew by {added} to {}", d.g.n);
}

#[test]
#[ignore]
fn gpu_mcmc_noise_has_the_covariance_squared() {
    let gpu = gpu();
    let s = &gpu.stream;
    let n = 1usize << 20;
    let mut host = GaussiansHost::zeros(n);
    let q = [0.9f32, 0.3, -0.2, 0.25];
    let ls = [0.1f32.ln(), 0.2f32.ln(), 0.3f32.ln()];
    let o = 0.004f32;
    for i in 0..n {
        host.rotations[4 * i..4 * i + 4].copy_from_slice(&q);
        host.scales[3 * i..3 * i + 3].copy_from_slice(&ls);
        host.opacities[i] = logit(o);
    }
    let mut g = Gaussians::alloc(&gpu, n, false).unwrap();
    g.upload(&gpu, &host).unwrap();
    let mut pd = IterParamsDev::alloc(&gpu).unwrap();
    let mut p = IterParams::default();
    p.frame.n_gaussians = n as u32;
    p.frame.iteration = 1234;
    p.means_lr = 1.6e-4;
    p.noise_scale = 5e5;
    p.seed = Philox::new(42).seed;
    pd.upload(&gpu, &p, None).unwrap();
    let k = McmcKernels::load(&gpu).unwrap();
    mcmc_noise(&gpu, &k, &pd, &mut g).unwrap();
    let m = s.clone_dtoh(&g.mean.slice(..3 * n)).unwrap();
    let mut cov = [[0f64; 3]; 3];
    for i in 0..n {
        for a in 0..3 {
            for b in 0..3 {
                cov[a][b] += m[3 * i + a] as f64 * m[3 * i + b] as f64 / n as f64;
            }
        }
    }
    // Expected: (k Σ)², Σ = R diag(s²) Rᵀ, k = lr · 5e5 · σ(−100 (o − 0.005)).
    let r = rotmat(&q);
    let f = 1.0 / (1.0 + (100.0 * (o as f64 - 0.005)).exp());
    let kk = p.means_lr as f64 * p.noise_scale as f64 * f;
    let sig: Vec<f64> = ls.iter().map(|&l| (2.0 * l as f64).exp()).collect();
    let cs = |a: usize, b: usize| {
        (0..3)
            .map(|c| r[3 * a + c] as f64 * sig[c] * r[3 * b + c] as f64)
            .sum::<f64>()
            * kk
    };
    let mut worst = 0f64;
    let scale = (0..3)
        .map(|a| (0..3).map(|c| cs(a, c) * cs(c, a)).sum::<f64>())
        .fold(0.0, f64::max);
    for a in 0..3 {
        for b in 0..3 {
            let want: f64 = (0..3).map(|c| cs(a, c) * cs(c, b)).sum();
            worst = worst.max((cov[a][b] - want).abs() / scale);
        }
    }
    println!("noise covariance over {n} draws: max err {worst:.4} of the largest variance");
    assert!(worst < 0.01, "{worst}");
}

#[test]
#[ignore]
fn gpu_small_kernels_match_cpu() {
    let gpu = gpu();
    let s = &gpu.stream;
    let k = DensifyKernels::load(&gpu).unwrap();
    // vcd_mask.
    let px = 64 * 48;
    let mut r = Rng(4);
    let err: Vec<f32> = (0..px).map(|_| r.range(0.0, 0.5)).collect();
    let (lo, hi) = score_range(&err);
    let em = ErrorMap {
        err: s.clone_htod(&err).unwrap(),
        minmax: s.clone_htod(&[lo, hi]).unwrap(),
        sums: s.alloc_zeros::<f32>(2).unwrap(),
    };
    let mut mask = s.alloc_zeros::<u8>(px).unwrap();
    vcd_mask(&gpu, &k, &em, 0.1, px as u32, &mut mask).unwrap();
    let got = s.clone_dtoh(&mask).unwrap();
    let want: Vec<u8> = err
        .iter()
        .map(|&e| u8::from((e - lo) / (hi - lo) > 0.1))
        .collect();
    assert_eq!(got, want);
    // scores_add.
    let nv = 500usize;
    let cnt: Vec<u32> = (0..nv).map(|_| r.u32() % 7).collect();
    let vis: Vec<u32> = (0..nv as u32).map(|v| (v * 7919) % 1024).collect();
    let mut status = vec![0u32; crate::kernels::layout::STATUS_LEN];
    status[crate::kernels::layout::ST_N_VIS] = nv as u32;
    let mut dc = s.clone_htod(&cnt).unwrap();
    let dv = s.clone_htod(&vis).unwrap();
    let ds = s.clone_htod(&status).unwrap();
    let mut acc = ViewScores::alloc(&gpu, 1024).unwrap();
    scores_add(&gpu, &k, &mut dc, &dv, &ds, 0.25, &mut acc).unwrap();
    scores_add(&gpu, &k, &mut dc, &dv, &ds, 0.25, &mut acc).unwrap();
    let (gc, gs) = (
        s.clone_dtoh(&acc.count).unwrap(),
        s.clone_dtoh(&acc.score).unwrap(),
    );
    for v in 0..nv {
        let p = vis[v] as usize;
        assert_eq!((gc[p], gs[p]), (cnt[v], 0.25 * cnt[v] as f32), "v {v}");
    }
    assert!(s.clone_dtoh(&dc).unwrap().iter().all(|&c| c == 0));
    // Overflow guard: a poisoned iteration adds nothing.
    status[crate::kernels::layout::ST_OVERFLOW] = 1;
    let dso = s.clone_htod(&status).unwrap();
    let mut dc2 = s.clone_htod(&cnt).unwrap();
    scores_add(&gpu, &k, &mut dc2, &dv, &dso, 1.0, &mut acc).unwrap();
    assert_eq!(s.clone_dtoh(&acc.count).unwrap(), gc);
    println!("vcd_mask, scores_add (and its overflow guard) exact");
}

// ---------------------------------------------------------------- benchmark

fn timed(gpu: &Gpu, reps: usize, mut f: impl FnMut() -> Result<()>) -> Stats {
    bench(gpu, 1, reps, &mut f).unwrap()
}

/// Kernel timings at realistic sizes (spec-train 5.8-5.9, 5.6). Writes
/// `/out/image-to-3d/densify/bench.json`. Use `SPL_GPU_LOCK=1`.
#[test]
#[ignore]
fn gpu_bench_densify() {
    let gpu = gpu();
    let s = gpu.stream.clone();
    let sizes: Vec<usize> = std::env::var("SPL_DENSIFY_BENCH_N")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_else(|| vec![1_000_000, 3_000_000, 6_000_000]);
    let mut report = Vec::new();
    for n in sizes {
        let cap = round_cap(n + n / 8);
        // params 236 + grads 236 + moments 472 + stats 12 + compactor 196 + plan 10 B.
        let need = cap as u64 * 1162 + (256 << 20);
        let (free, _) = gpu.mem_info().unwrap();
        if need > free {
            println!(
                "n {n}: skipped, needs {} MiB, {} MiB free",
                need >> 20,
                free >> 20
            );
            continue;
        }
        let snap = gpu_snapshot();
        let mut r = Rng(n as u64);
        // Fill on the device-sized buffers from a small host pattern (upload cost is not timed).
        let mut host = random_host(n.min(200_000), &mut r);
        while host.len() < n {
            let m = (n - host.len()).min(host.len());
            let h2 = host.clone();
            host.means.extend_from_slice(&h2.means[..3 * m]);
            host.scales.extend_from_slice(&h2.scales[..3 * m]);
            host.rotations.extend_from_slice(&h2.rotations[..4 * m]);
            host.opacities.extend_from_slice(&h2.opacities[..m]);
            host.sh0.extend_from_slice(&h2.sh0[..3 * m]);
            host.sh_rest.extend_from_slice(&h2.sh_rest[..45 * m]);
        }
        for i in 0..n {
            host.means[3 * i] += (i as f32 * 1e-6).sin();
        }
        let mut g = Gaussians::alloc(&gpu, cap, false).unwrap();
        g.upload(&gpu, &host).unwrap();
        drop(host);
        let mut grads = Grads::alloc(&gpu, cap).unwrap();
        let mut moments = Moments {
            precision: MomentPrecision::F32,
            groups: std::array::from_fn(|k| s.alloc_zeros::<f32>(cap * FLOATS[k] * 2).unwrap()),
        };
        let mut stats = DensifyStats::alloc(&gpu, cap, true).unwrap();
        let st = random_stats(n, &mut r);
        s.memcpy_htod(&st.vis, &mut stats.vis_count.slice_mut(..n))
            .unwrap();
        s.memcpy_htod(&st.grad, &mut stats.grad_sum.slice_mut(..n))
            .unwrap();
        let mut vs = ViewScores::alloc(&gpu, cap).unwrap();
        s.memcpy_htod(&st.count, &mut vs.count.slice_mut(..n))
            .unwrap();
        s.memcpy_htod(&st.score, &mut vs.score.slice_mut(..n))
            .unwrap();
        let mut w = Work::new(&gpu, cap).unwrap();
        let reps = 10;
        let nn = n as u32;
        let mut row = BTreeMap::<String, serde_json::Value>::new();
        row.insert("n".into(), n.into());
        row.insert("gpu_before".into(), serde_json::to_value(&snap).unwrap());
        let mut put = |name: &str, st: Stats, bytes: f64| {
            let gbs = bytes / (st.median_ms * 1e-3) / 1e9;
            println!(
                "n {n:>8} {name:<28} median {:>8.3} ms  p10 {:>8.3}  p90 {:>8.3}  {gbs:>6.0} GB/s",
                st.median_ms, st.p10_ms, st.p90_ms
            );
            row.insert(
                name.into(),
                serde_json::json!({"stats": st, "gb_per_s": gbs}),
            );
        };
        // Plan (ADC and FastGS rules).
        let rules = DensifyRules::adc(&AdcConfig::default(), EXTENT, 5000);
        let t = timed(&gpu, reps, || {
            w.plan(&gpu, &g, &stats, Some(&vs), &rules).map(drop)
        });
        put(
            "plan_adc (incl. 1 sync)",
            t,
            n as f64 * (12.0 + 16.0 + 4.0 + 8.0 + 10.0 + 4.0),
        );
        w.score_minmax(&gpu, &vs.score, nn).unwrap();
        let fr = DensifyRules::fastgs(&FastGsConfig::default(), EXTENT, 5000, 1);
        let t = timed(&gpu, reps, || {
            w.plan(&gpu, &g, &stats, Some(&vs), &fr).map(drop)
        });
        put("plan_fastgs (incl. 1 sync)", t, n as f64 * 66.0);
        // Radix sort 32-bit, Gumbel-top-k (keys + sort + mark).
        let t = timed(&gpu, reps, || w.compactor.sort(&gpu, nn, 32));
        put("radix_sort_32 (k,v)", t, n as f64 * 4.0 * (8.0 * 2.0 + 4.0));
        let cand = w.plan(&gpu, &g, &stats, Some(&vs), &fr).unwrap();
        let t = timed(&gpu, reps, || {
            w.vcp(&gpu, &vs.score, cand / 2, Philox::new(1), 5000)
        });
        put("vcp gumbel_top_k", t, n as f64 * 4.0 * (8.0 * 2.0 + 4.0));
        // Emission: 5% clone, 5% split, 10% pruned, every array (241 words per Gaussian).
        let (oc, cf, sf, pri) = {
            let mut oc = vec![1u32; n];
            let mut cf = vec![0u8; n];
            let mut sf = vec![0u8; n];
            for i in 0..n {
                match r.u32() % 20 {
                    0 => cf[i] = 1,
                    1 => {
                        sf[i] = 1;
                        oc[i] = 0
                    }
                    2 | 3 => oc[i] = 0,
                    _ => {}
                }
            }
            (oc, cf, sf, vec![0f32; n])
        };
        upload_plan(&gpu, &mut w.plan, &oc, &cf, &sf, &pri);
        let words;
        {
            let mut extra = PerGaussianArrays::new();
            extra.add_grads(&s, &mut grads);
            extra.add_moments(&s, &mut moments, cap as u32);
            extra.add_stats(&s, &mut stats);
            words = extra.words() + 59 + 1;
            let timer = GpuTimer::new(&gpu).unwrap();
            let mut samples = Vec::new();
            for rep in 0..=reps {
                g.n = nn;
                w.plan.n_before = nn;
                timer.start(&s).unwrap();
                let res = w
                    .compactor
                    .emit(&gpu, &mut w.plan, &mut g, &extra, Philox::new(1), 600)
                    .unwrap();
                let ms = timer.stop_ms(&s).unwrap() as f64;
                if rep > 0 {
                    samples.push(ms);
                } else {
                    println!("emission: {res:?}");
                }
            }
            put(
                &format!("emit ({words} words/G)"),
                Stats::from_samples(&samples),
                n as f64 * words as f64 * 4.0 * 4.0,
            );
            g.n = nn;
            let t = timed(&gpu, reps, || w.compactor.morton_sort(&gpu, &mut g, &extra));
            put(
                "morton_sort (all arrays)",
                t,
                n as f64 * (words as f64 * 4.0 * 4.0 + 12.0 + 40.0),
            );
        }
        // FastGS bookkeeping: opacity clamp with moments.
        let mut adam_like = moments;
        let t = timed(&gpu, reps, || {
            let buf = &mut adam_like.groups[Group::Opacities.index()];
            let words = 2u32;
            let lm = logit(0.8);
            let mut l = s.launch_builder(&w.k.opacity_clamp);
            l.arg(&mut g.opacity).arg(buf).arg(&words).arg(&lm).arg(&nn);
            // SAFETY: opacity_clamp(opacity, moments (2 words/G), words, logit_max, n).
            unsafe { l.launch(LaunchConfig::for_num_elems(nn)) }
                .cuda("opacity_clamp")
                .map(drop)
        });
        put("opacity_clamp", t, n as f64 * 16.0);
        // MCMC round: 2% dead relocated, +5% added (cap_max large).
        drop(w);
        let mut mw = McmcWork::new(&gpu, cap).unwrap();
        let mut hh = g.download(&gpu).unwrap();
        let cfg = McmcConfig {
            cap_max: u32::MAX,
            ..McmcConfig::default()
        };
        let timer = GpuTimer::new(&gpu).unwrap();
        let mut samples = Vec::new();
        hh.opacities.iter_mut().enumerate().for_each(|(i, o)| {
            if i % 50 == 7 {
                *o = logit(0.001)
            }
        });
        for rep in 0..=reps.min(5) {
            g.upload(&gpu, &hh).unwrap();
            let mut extra = PerGaussianArrays::new();
            extra.add_grads(&s, &mut grads);
            extra.add_moments(&s, &mut adam_like, cap as u32);
            gpu.sync().unwrap();
            timer.start(&s).unwrap();
            let res = mcmc_round(
                &gpu,
                &mut mw,
                &cfg,
                &mut g,
                &extra,
                Philox::new(rep as u64),
                1000,
            )
            .unwrap();
            let ms = timer.stop_ms(&s).unwrap() as f64;
            if rep > 0 {
                samples.push(ms);
            } else {
                println!("mcmc round: relocated {}, added {}", res.0, res.1);
            }
        }
        put(
            "mcmc_round (2% dead, +5%)",
            Stats::from_samples(&samples),
            n as f64 * 0.07 * 236.0 * 3.0 + n as f64 * 40.0,
        );
        g.n = nn;
        let mut pd = IterParamsDev::alloc(&gpu).unwrap();
        let mut p = IterParams::default();
        p.frame.n_gaussians = nn;
        p.means_lr = 1e-5;
        p.noise_scale = 5e5;
        pd.upload(&gpu, &p, None).unwrap();
        let t = timed(&gpu, reps, || mcmc_noise(&gpu, &mw.k, &pd, &mut g));
        put("mcmc_noise", t, n as f64 * 44.0);
        row.insert(
            "gpu_after".into(),
            serde_json::to_value(gpu_snapshot()).unwrap(),
        );
        report.push(row);
    }
    let dir = crate::testkit::out_dir("densify");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("bench.json"),
        serde_json::to_string_pretty(&report).unwrap(),
    )
    .unwrap();
    let _ = (COL_COPY, CompactKernels::load);
}
