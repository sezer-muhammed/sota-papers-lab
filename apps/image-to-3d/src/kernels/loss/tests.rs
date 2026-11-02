//! Loss tests. CPU: the window, the f64 reference against finite differences,
//! the resampling weights. GPU (`#[ignore]`, run on the GPU host with
//! `scripts/gpu-test.sh <checkout> kernels::loss`): both SSIM kernels against
//! the f64 reference, v2 against v1, the error map, target resampling, the
//! fused-ssim oracle (staged), and timings.

use super::*;
use crate::frame::IterParams;
use crate::rng::Philox;

/// The window is the normalized σ = 1.5 Gaussian.
#[test]
fn ssim_window_is_the_normalized_gaussian() {
    let raw: Vec<f64> = (-5..=5)
        .map(|i| (-(i * i) as f64 / (2.0 * 1.5 * 1.5)).exp())
        .collect();
    let sum: f64 = raw.iter().sum();
    for (w, r) in SSIM_WINDOW.iter().zip(&raw) {
        assert!((*w as f64 - r / sum).abs() < 1e-8, "{w} vs {}", r / sum);
    }
}

/// A render/target pair: `kind` 0 independent uniform noise, 1 a smooth target
/// and a render close to it (S near 1, the regime of training).
fn images(w: usize, h: usize, seed: u64, kind: u32) -> (Vec<f32>, Vec<f32>) {
    let p = Philox::new(seed);
    let n = 3 * w * h;
    if kind == 0 {
        let x = (0..n).map(|i| p.uniform(i as u32, 1, 0)).collect();
        let y = (0..n).map(|i| p.uniform(i as u32, 2, 0)).collect();
        return (x, y);
    }
    let mut y = vec![0f32; n];
    let mut x = vec![0f32; n];
    for c in 0..3 {
        for r in 0..h {
            for q in 0..w {
                let i = c * w * h + r * w + q;
                let v = 0.5
                    + 0.3 * ((q as f32 * 0.21 + c as f32).sin() * (r as f32 * 0.13).cos())
                    + 0.05 * (p.uniform(i as u32, 3, 0) - 0.5);
                y[i] = v.clamp(0.0, 1.0);
                x[i] = (v + 0.04 * (p.uniform(i as u32, 4, 0) - 0.5)).clamp(0.0, 1.0);
            }
        }
    }
    (x, y)
}

/// Central differences of the f64 reference loss confirm its analytic gradient.
#[test]
fn reference_gradient_matches_finite_differences() {
    let (w, h) = (13, 11);
    for kind in [0, 1] {
        let (x, y) = images(w, h, 7, kind);
        let (_, _, _, g) = reference_loss(&x, &y, w, h, 0.8, 0.2);
        let p = Philox::new(9);
        let eps = 1e-3f32;
        for k in 0..20 {
            let i = (p.draw4(k, 0, 0)[0] as usize) % x.len();
            if (x[i] - y[i]).abs() < 2.0 * eps {
                continue; // the L1 kink
            }
            let mut xp = x.clone();
            xp[i] += eps;
            let mut xm = x.clone();
            xm[i] -= eps;
            let fd = (reference_loss(&xp, &y, w, h, 0.8, 0.2).0
                - reference_loss(&xm, &y, w, h, 0.8, 0.2).0)
                / (2.0 * eps as f64);
            let rel = (fd - g[i]).abs() / g[i].abs().max(1e-6);
            assert!(rel < 1e-2, "kind {kind} pixel {i}: fd {fd} vs {}", g[i]);
        }
    }
}

/// f64 mirror of `aa_weights` (PyTorch's antialiased bilinear) for one axis.
fn aa_weights_ref(i: usize, inp: usize, out: usize) -> (usize, Vec<f64>) {
    let scale = inp as f64 / out as f64;
    let support = scale.max(1.0);
    let inv = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    let center = scale * (i as f64 + 0.5);
    let xmin = ((center - support + 0.5) as isize).max(0) as usize;
    let xmax = ((center + support + 0.5) as usize).min(inp);
    let mut w: Vec<f64> = (xmin..xmax)
        .map(|j| {
            let x = ((j as f64 - center + 0.5) * inv).abs();
            if x < 1.0 { 1.0 - x } else { 0.0 }
        })
        .collect();
    let t: f64 = w.iter().sum();
    w.iter_mut().for_each(|v| *v /= t);
    (xmin, w)
}

/// Downscaling by an integer factor s with the tent filter averages a
/// symmetric 2s-wide neighbourhood; weights are normalized.
#[test]
fn antialiased_weights_are_symmetric_tents() {
    let (x0, w) = aa_weights_ref(3, 40, 10);
    assert_eq!(x0, 10);
    assert_eq!(w.len(), 8);
    assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    for k in 0..4 {
        assert!((w[k] - w[7 - k]).abs() < 1e-12);
    }
    // Scale 1 is the identity (the second tap has weight 0).
    let (x0, w) = aa_weights_ref(5, 10, 10);
    assert_eq!((x0, w[0]), (5, 1.0));
    assert!(w[1..].iter().all(|&v| v == 0.0));
}

// ---------------------------------------------------------------- GPU

use crate::testkit::npz::Npz;
use crate::testkit::oracle;
use crate::testkit::timing::{bench, gpu_snapshot};

fn gpu() -> Gpu {
    Gpu::open().expect("open GPU")
}

fn params_for(w: u32, h: u32) -> IterParams {
    let mut p = IterParams::default();
    p.frame.iteration = 1;
    p.frame.render_w = w;
    p.frame.render_h = h;
    p
}

struct Run {
    sums: LossValue,
    grad: Vec<f32>,
}

fn run_loss(gpu: &Gpu, kernel: SsimKernel, x: &[f32], y: &[f32], w: u32, h: u32) -> Run {
    let s = &gpu.stream;
    let cfg = LossConfig {
        kernel,
        ..Default::default()
    };
    let mut loss = Loss::new(gpu, w, h, cfg).unwrap();
    let mut pd = IterParamsDev::alloc(gpu).unwrap();
    pd.upload(gpu, &params_for(w, h), None).unwrap();
    let dx = s.clone_htod(x).unwrap();
    let dy = s.clone_htod(y).unwrap();
    let mut dg = s.alloc_zeros::<f32>(x.len()).unwrap();
    loss.forward_backward(gpu, &pd, &dx, &dy, &mut dg).unwrap();
    let sums = loss.take_sums(gpu, 1).unwrap();
    Run {
        sums,
        grad: s.clone_dtoh(&dg).unwrap(),
    }
}

fn max_abs_diff(a: &[f32], b: &[f64]) -> (f64, f64) {
    let scale = b.iter().fold(0f64, |m, v| m.max(v.abs()));
    let d = a
        .iter()
        .zip(b)
        .fold(0f64, |m, (p, q)| m.max((*p as f64 - q).abs()));
    (d, scale)
}

/// Both kernels against the f64 reference on odd sizes (borders), noise and
/// smooth pairs: |ΔS̄| / S̄ < 1e-5 and max |Δg| < 1e-4 · max |g|; v2 vs v1
/// max |Δg| < 1e-6 · max |g|.
#[test]
#[ignore]
fn gpu_ssim_matches_f64_reference() {
    let gpu = gpu();
    for (w, h) in [(37usize, 29usize), (131, 77), (64, 48), (300, 17)] {
        for kind in [0, 1] {
            let (x, y) = images(w, h, (w * h) as u64, kind);
            let (l_ref, l1_ref, s_ref, g_ref) = reference_loss(&x, &y, w, h, 0.8, 0.2);
            let v1 = run_loss(&gpu, SsimKernel::TwoPass, &x, &y, w as u32, h as u32);
            let v2 = run_loss(&gpu, SsimKernel::Fused, &x, &y, w as u32, h as u32);
            for (name, r) in [("v1", &v1), ("v2", &v2)] {
                let ds = (r.sums.ssim - s_ref).abs() / s_ref.abs().max(1e-3);
                let dl1 = (r.sums.l1 - l1_ref).abs() / l1_ref;
                let (dg, gs) = max_abs_diff(&r.grad, &g_ref);
                println!(
                    "{w}x{h} kind {kind} {name}: S {:.9} ref {s_ref:.9} rel {ds:.2e}; L1 rel {dl1:.2e}; \
                     loss {:.9} ref {l_ref:.9}; grad max |d| {dg:.2e} (max |g| {gs:.2e}, rel {:.2e})",
                    r.sums.ssim,
                    r.sums.loss,
                    dg / gs
                );
                assert!(ds < 1e-5, "{name} S");
                assert!(dl1 < 1e-5, "{name} L1");
                assert!(dg < 1e-4 * gs, "{name} gradient");
            }
            let g1: Vec<f64> = v1.grad.iter().map(|&v| v as f64).collect();
            let (d12, gs) = max_abs_diff(&v2.grad, &g1);
            println!(
                "{w}x{h} kind {kind} v2 vs v1: max |d| {d12:.2e} = {:.2e} max |g|",
                d12 / gs
            );
            assert!(d12 < 1e-6 * gs, "v2 vs v1");
        }
    }
}

/// Error map: e = mean_c |x − y|, its min/max, and the same sums as the loss.
#[test]
#[ignore]
fn gpu_error_map_matches_cpu() {
    let gpu = gpu();
    let s = &gpu.stream;
    let (w, h) = (97usize, 45usize);
    let (x, y) = images(w, h, 3, 1);
    let (_, l1_ref, s_ref, _) = reference_loss(&x, &y, w, h, 0.8, 0.2);
    let mut loss = Loss::new(&gpu, w as u32, h as u32, LossConfig::default()).unwrap();
    let mut pd = IterParamsDev::alloc(&gpu).unwrap();
    pd.upload(&gpu, &params_for(w as u32, h as u32), None)
        .unwrap();
    let (dx, dy) = (s.clone_htod(&x).unwrap(), s.clone_htod(&y).unwrap());
    let mut out = ErrorMap::alloc(&gpu, w as u32, h as u32).unwrap();
    loss.error_map(&gpu, &pd, &dx, &dy, &mut out).unwrap();
    let err = s.clone_dtoh(&out.err).unwrap();
    let mm = s.clone_dtoh(&out.minmax).unwrap();
    let sums = s.clone_dtoh(&out.sums).unwrap();
    let hw = w * h;
    let want: Vec<f32> = (0..hw)
        .map(|i| {
            (0..3)
                .map(|c| (x[c * hw + i] - y[c * hw + i]).abs())
                .sum::<f32>()
                / 3.0
        })
        .collect();
    for i in 0..hw {
        assert!((err[i] - want[i]).abs() < 1e-6, "pixel {i}");
    }
    let lo = want.iter().cloned().fold(f32::MAX, f32::min);
    let hi = want.iter().cloned().fold(0f32, f32::max);
    assert!(
        (mm[0] - lo).abs() < 1e-6 && (mm[1] - hi).abs() < 1e-6,
        "{mm:?} vs {lo} {hi}"
    );
    assert!((sums[0] - l1_ref).abs() < 1e-5 * l1_ref);
    assert!((sums[1] - s_ref).abs() < 1e-5 * s_ref.abs());
}

/// Overflowed iterations leave the gradient and the sums untouched.
#[test]
#[ignore]
fn gpu_loss_respects_the_overflow_flag() {
    let gpu = gpu();
    let s = &gpu.stream;
    let (w, h) = (40u32, 30u32);
    let (x, y) = images(w as usize, h as usize, 1, 0);
    let mut pd = IterParamsDev::alloc(&gpu).unwrap();
    pd.upload(&gpu, &params_for(w, h), None).unwrap();
    s.memcpy_htod(&[1u32], &mut pd.status).unwrap();
    for kernel in [SsimKernel::TwoPass, SsimKernel::Fused] {
        let mut loss = Loss::new(
            &gpu,
            w,
            h,
            LossConfig {
                kernel,
                ..Default::default()
            },
        )
        .unwrap();
        let (dx, dy) = (s.clone_htod(&x).unwrap(), s.clone_htod(&y).unwrap());
        let mut dg = s.clone_htod(&vec![7f32; x.len()]).unwrap();
        loss.forward_backward(&gpu, &pd, &dx, &dy, &mut dg).unwrap();
        assert!(s.clone_dtoh(&dg).unwrap().iter().all(|&v| v == 7.0));
        assert_eq!(loss.take_sums(&gpu, 1).unwrap().l1, 0.0);
    }
}

/// Target resampling: exact u8/255 copy at scale 1; at scales 2, 3 and a
/// non-integer ratio the antialiased tent of PyTorch (f64 mirror) within 1e-5.
#[test]
#[ignore]
fn gpu_resample_target_matches_reference() {
    let gpu = gpu();
    let s = &gpu.stream;
    let (tw, th) = (157usize, 103usize);
    let p = Philox::new(11);
    let pool_host: Vec<u8> = (0..16 + 3 * tw * th)
        .map(|i| (p.draw4(i as u32, 0, 0)[0] & 0xff) as u8)
        .collect();
    let pool = s.clone_htod(&pool_host).unwrap();
    let k = LossKernels::load(&gpu).unwrap();
    let mut pd = IterParamsDev::alloc(&gpu).unwrap();
    for (w, h) in [(tw, th), (tw / 2, th / 2), (tw / 3, th / 3), (100, 70)] {
        let mut ip = params_for(w as u32, h as u32);
        ip.target_offset = 16;
        ip.target_w = tw as u32;
        ip.target_h = th as u32;
        pd.upload(&gpu, &ip, None).unwrap();
        let mut out = s.alloc_zeros::<f32>(3 * tw * th).unwrap();
        resample_target_raw(&gpu, &k, &pd, &pool, tw as u32, th as u32, &mut out).unwrap();
        let got = s.clone_dtoh(&out).unwrap();
        let src =
            |x: usize, y: usize, c: usize| pool_host[16 + 3 * (y * tw + x) + c] as f64 / 255.0;
        let mut worst = 0f64;
        for yy in 0..h {
            let (y0, wy) = aa_weights_ref(yy, th, h);
            for xx in 0..w {
                let (x0, wx) = aa_weights_ref(xx, tw, w);
                for c in 0..3 {
                    let mut v = 0.0;
                    for (j, a) in wy.iter().enumerate() {
                        for (i, b) in wx.iter().enumerate() {
                            v += a * b * src(x0 + i, y0 + j, c);
                        }
                    }
                    let g = got[c * w * h + yy * w + xx] as f64;
                    worst = worst.max((g - v).abs());
                }
            }
        }
        println!("resample {tw}x{th} -> {w}x{h}: max |d| {worst:.2e}");
        assert!(worst < 1e-5, "{w}x{h}");
    }
}

/// Stages fused-ssim oracle cases (`gs-oracle batch /out/image-to-3d/oracle`
/// on the host), then compares when the oracle output exists: |ΔS̄| < 1e-7
/// and max |Δ dL/dx| < 1e-8 (spec-train 3.3; documented in ORACLE.md).
#[test]
#[ignore]
fn gpu_ssim_matches_fused_ssim_oracle() {
    let gpu = gpu();
    let dir = oracle::stage_dir();
    std::fs::create_dir_all(&dir).unwrap();
    let mut missing = 0;
    for (w, h, kind) in [(37usize, 29usize, 0u32), (131, 77, 1), (1297, 840, 1)] {
        let (x, y) = images(w, h, 99 + kind as u64, kind);
        let name = format!("loss_{w}x{h}_{kind}");
        let mut z = Npz::new();
        z.f32("img1", &[1, 3, h, w], x.clone()).unwrap();
        z.f32("img2", &[1, 3, h, w], y.clone()).unwrap();
        oracle::stage(&dir, &name, "ssim", &z).unwrap();
        let mut runs = Vec::new();
        for kernel in [SsimKernel::TwoPass, SsimKernel::Fused] {
            let r = run_loss(&gpu, kernel, &x, &y, w as u32, h as u32);
            let mut ours = Npz::new();
            ours.f32("v_img1_loss", &[1, 3, h, w], r.grad.clone())
                .unwrap();
            ours.f32_scalar("ssim", r.sums.ssim as f32).unwrap();
            ours.f32_scalar("l1", r.sums.l1 as f32).unwrap();
            ours.f32_scalar("loss", r.sums.loss as f32).unwrap();
            if kernel == SsimKernel::Fused {
                oracle::stage_ours(&dir, &name, "ssim", &ours).unwrap();
            }
            runs.push((kernel, r));
        }
        let Ok(out) = oracle::load_out(&dir, &name, "ssim") else {
            missing += 1;
            continue;
        };
        let s_or = out.get("ssim").unwrap().to_f64()[0];
        let s_ref = out.get("ssim_ref").unwrap().to_f64()[0];
        let g_or = out.get("v_img1_loss").unwrap().to_f64();
        for (kernel, r) in &runs {
            let (dg, gs) = max_abs_diff(&r.grad, &g_or);
            println!(
                "{name} {kernel:?}: S ours {:.9} fused-ssim {s_or:.9} f64 {s_ref:.9} |dS| {:.2e}; \
                 grad max |d| {dg:.2e} (max |g| {gs:.2e})",
                r.sums.ssim,
                (r.sums.ssim - s_or).abs()
            );
            assert!(
                (r.sums.ssim - s_or).abs() < 2e-7,
                "{name} {kernel:?} S vs fused-ssim"
            );
            assert!(dg < 1e-8, "{name} {kernel:?} gradient vs fused-ssim");
        }
    }
    if missing > 0 {
        eprintln!(
            "{missing} oracle outputs missing: run `gs-oracle batch {}` on the host",
            dir.display()
        );
    }
}

/// Timing (CUDA events, median of 200 after 20 warm-ups), v1 vs v2, at the
/// garden 1/4 size and the spec's sizes. `SPL_GPU_LOCK=1` for reported numbers.
#[test]
#[ignore]
fn gpu_loss_bench() {
    let gpu = gpu();
    let s = &gpu.stream;
    println!("gpu before: {:?}", gpu_snapshot());
    for (w, h) in [(1297u32, 840u32), (1237, 822), (1600, 1063), (3840, 2160)] {
        let n = 3 * (w * h) as usize;
        let (x, y) = images(w as usize, h as usize, 5, 1);
        let mut pd = IterParamsDev::alloc(&gpu).unwrap();
        pd.upload(&gpu, &params_for(w, h), None).unwrap();
        let (dx, dy) = (s.clone_htod(&x).unwrap(), s.clone_htod(&y).unwrap());
        let mut dg = s.alloc_zeros::<f32>(n).unwrap();
        for kernel in [SsimKernel::TwoPass, SsimKernel::Fused] {
            let mut loss = Loss::new(
                &gpu,
                w,
                h,
                LossConfig {
                    kernel,
                    ..Default::default()
                },
            )
            .unwrap();
            let st = bench(&gpu, 20, 200, || {
                loss.forward_backward(&gpu, &pd, &dx, &dy, &mut dg)
            })
            .unwrap();
            println!(
                "loss {w}x{h} {kernel:?}: median {:.4} ms (p10 {:.4}, p90 {:.4})",
                st.median_ms, st.p10_ms, st.p90_ms
            );
        }
        let k = &Loss::new(&gpu, w, h, LossConfig::default())
            .unwrap()
            .kernels;
        let pool = s.alloc_zeros::<u8>(n).unwrap();
        let mut ip = params_for(w, h);
        ip.target_w = w;
        ip.target_h = h;
        pd.upload(&gpu, &ip, None).unwrap();
        let st = bench(&gpu, 20, 200, || {
            resample_target_raw(&gpu, k, &pd, &pool, w, h, &mut dg)
        })
        .unwrap();
        println!(
            "resample_target {w}x{h} copy: median {:.4} ms",
            st.median_ms
        );
    }
    println!("gpu after: {:?}", gpu_snapshot());
}
