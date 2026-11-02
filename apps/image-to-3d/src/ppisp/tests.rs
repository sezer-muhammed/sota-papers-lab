//! CPU tests (layout, init, schedule) and GPU tests against the upstream
//! PyTorch reference (cases from `tools/ppisp/ref.py`, staged under
//! `/out/image-to-3d/ppisp` or `SPL_PPISP_CASES`).

use std::path::PathBuf;

use super::*;
use crate::testkit::npz::Npz;
use crate::testkit::timing::bench;

#[test]
fn layout_offsets_and_len() {
    let l = Layout::new(2, 5);
    assert_eq!(
        (l.off_exposure, l.off_vig, l.off_color, l.off_crf),
        (0, 5, 35, 75)
    );
    assert_eq!(l.len(), 99);
}

#[test]
fn initial_crf_is_identity_like() {
    let l = Layout::new(1, 1);
    let p = l.initial();
    let sp = |x: f32| (1.0 + x.exp()).ln();
    for c in 0..3 {
        let r = &p[l.off_crf as usize + 4 * c..][..4];
        assert!((0.3 + sp(r[0]) - 1.0).abs() < 1e-6);
        assert!((0.3 + sp(r[1]) - 1.0).abs() < 1e-6);
        assert!((0.1 + sp(r[2]) - 1.0).abs() < 1e-6);
        assert_eq!(r[3], 0.0);
    }
    assert!(p[..l.off_crf as usize].iter().all(|&x| x == 0.0));
}

#[test]
fn lr_schedule_matches_torch_shape() {
    let c = PpispConfig::default();
    assert!((c.lr_at(0) - 2e-5).abs() < 1e-9);
    assert!((c.lr_at(250) - 0.002 * (0.01 + 0.99 * 0.5)).abs() < 1e-8);
    assert!((c.lr_at(500) - 0.002).abs() < 1e-8);
    assert!((c.lr_at(30_500) - 2e-5).abs() < 1e-8);
}

fn cases_dir() -> PathBuf {
    std::env::var_os("SPL_PPISP_CASES")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/out/image-to-3d/ppisp"))
}

fn f32s(z: &Npz, k: &str) -> Vec<f32> {
    z.get(k)
        .unwrap()
        .to_f64()
        .into_iter()
        .map(|x| x as f32)
        .collect()
}

/// max |a − b| / max(|b|_∞, floor)
fn rel(a: &[f32], b: &[f32], floor: f32) -> f32 {
    assert_eq!(a.len(), b.len());
    let m = b.iter().fold(floor, |m, x| m.max(x.abs()));
    a.iter()
        .zip(b)
        .fold(0.0f32, |e, (x, y)| e.max((x - y).abs()))
        / m
}

#[test]
fn lr_schedule_matches_torch_npz() {
    let p = cases_dir().join("lr.npz");
    if !p.exists() {
        eprintln!("skip: {} missing (run tools/ppisp/ref.py)", p.display());
        return;
    }
    let lr = Npz::read(&p).unwrap().get("lr").unwrap().to_f64();
    let c = PpispConfig::default();
    let worst = lr
        .iter()
        .enumerate()
        .map(|(i, &r)| ((c.lr_at(i as u32) as f64 - r) / r).abs())
        .fold(0.0, f64::max);
    println!("lr schedule: {} steps, max rel err {worst:.2e}", lr.len());
    assert!(worst < 1e-4);
}

#[test]
#[ignore]
fn gpu_ppisp_matches_torch_reference() {
    let gpu = Gpu::open().unwrap();
    for name in ["full", "cam_only", "frame_only"] {
        let z = Npz::read(&cases_dir().join(format!("{name}.npz"))).unwrap();
        let d: Vec<i32> = z
            .get("dims")
            .unwrap()
            .to_f64()
            .iter()
            .map(|&x| x as i32)
            .collect();
        let (w, h, cams, frames, cam, frame) = (
            d[0] as u32,
            d[1] as u32,
            d[2] as usize,
            d[3] as usize,
            d[4],
            d[5],
        );
        let mut pp =
            Ppisp::new(&gpu, cams, frames, (w * h) as usize, PpispConfig::default()).unwrap();
        let l = pp.layout;
        let mut p = vec![0.0; l.len()];
        let put = |p: &mut Vec<f32>, off: i32, v: Vec<f32>| {
            p[off as usize..off as usize + v.len()].copy_from_slice(&v)
        };
        put(&mut p, l.off_exposure, f32s(&z, "exposure"));
        put(&mut p, l.off_vig, f32s(&z, "vig"));
        put(&mut p, l.off_color, f32s(&z, "color"));
        put(&mut p, l.off_crf, f32s(&z, "crf"));
        pp.upload(&gpu, &p).unwrap();
        let s = &gpu.stream;
        let rgb = s.clone_htod(&f32s(&z, "rgb")).unwrap();
        let mut out = s.alloc_zeros::<f32>(rgb.len()).unwrap();
        pp.forward(&gpu, cam, frame, &rgb, &mut out, w, h).unwrap();
        let mut v = s.clone_htod(&f32s(&z, "wgt")).unwrap();
        pp.backward_inplace(&gpu, cam, frame, &rgb, &mut v, w, h)
            .unwrap();
        let got_out = s.clone_dtoh(&out).unwrap();
        let got_v = s.clone_dtoh(&v).unwrap();
        let g = s.clone_dtoh(&pp.grads).unwrap();
        let seg = |off: i32, n: usize| g[off as usize..off as usize + n].to_vec();
        let e_out = rel(&got_out, &f32s(&z, "out"), 1.0);
        let e_rgb = rel(&got_v, &f32s(&z, "g_rgb"), 1e-3);
        let e_exp = rel(&seg(l.off_exposure, frames), &f32s(&z, "g_exposure"), 1e-3);
        let e_vig = rel(&seg(l.off_vig, 15 * cams), &f32s(&z, "g_vig"), 1e-3);
        let e_col = rel(&seg(l.off_color, 8 * frames), &f32s(&z, "g_color"), 1e-3);
        let e_crf = rel(&seg(l.off_crf, 12 * cams), &f32s(&z, "g_crf"), 1e-3);
        println!(
            "{name} {w}x{h} cam {cam} frame {frame}: out {e_out:.2e} d_rgb {e_rgb:.2e} \
             d_exposure {e_exp:.2e} d_vig {e_vig:.2e} d_color {e_col:.2e} d_crf {e_crf:.2e}"
        );
        assert!(e_out < 1e-4, "{name} out");
        for (k, e) in [
            ("rgb", e_rgb),
            ("exp", e_exp),
            ("vig", e_vig),
            ("col", e_col),
            ("crf", e_crf),
        ] {
            assert!(e < 2e-3, "{name} grad {k}: {e}");
        }

        // Regulariser, alone on zeroed grads with the case's weights.
        let rw = f32s(&z, "reg_w");
        pp.cfg = PpispConfig {
            exposure_mean: rw[0],
            vig_center: rw[1],
            vig_channel: rw[2],
            vig_non_pos: rw[3],
            color_mean: rw[4],
            crf_channel: rw[5],
            ..PpispConfig::default()
        };
        s.memset_zeros(&mut pp.grads).unwrap();
        pp.regularize(&gpu).unwrap();
        let g = s.clone_dtoh(&pp.grads).unwrap();
        let seg = |off: i32, n: usize| g[off as usize..off as usize + n].to_vec();
        let r = s.clone_dtoh(&pp.reg_loss).unwrap()[0];
        let r_ref = f32s(&z, "reg")[0];
        let errs = [
            rel(&seg(l.off_exposure, frames), &f32s(&z, "r_exposure"), 1e-4),
            rel(&seg(l.off_vig, 15 * cams), &f32s(&z, "r_vig"), 1e-4),
            rel(&seg(l.off_color, 8 * frames), &f32s(&z, "r_color"), 1e-4),
            rel(&seg(l.off_crf, 12 * cams), &f32s(&z, "r_crf"), 1e-4),
        ];
        println!("{name} reg {r:.6} vs {r_ref:.6}; grad errs {errs:?}");
        assert!((r - r_ref).abs() <= 1e-4 * r_ref.abs().max(1e-3));
        assert!(errs.iter().all(|&e| e < 1e-3));
    }
}

/// Per-iteration cost at Mip-NeRF 360 1/4 resolution (garden: 1297×840).
#[test]
#[ignore]
fn gpu_ppisp_timing_mip360_quarter() {
    let gpu = Gpu::open().unwrap();
    let (w, h) = (1297u32, 840u32);
    let n = (w * h) as usize;
    let mut pp = Ppisp::new(&gpu, 1, 185, n, PpispConfig::default()).unwrap();
    let s = &gpu.stream;
    let img: Vec<f32> = (0..3 * n)
        .map(|i| ((i * 7919) % 1000) as f32 / 999.0)
        .collect();
    let rgb = s.clone_htod(&img).unwrap();
    let mut out = s.alloc_zeros::<f32>(3 * n).unwrap();
    let mut v = s.clone_htod(&img).unwrap();
    let fwd = bench(&gpu, 20, 200, || {
        pp.forward(&gpu, 0, 7, &rgb, &mut out, w, h)
    })
    .unwrap();
    let bwd = bench(&gpu, 20, 200, || {
        pp.backward_inplace(&gpu, 0, 7, &rgb, &mut v, w, h)
    })
    .unwrap();
    let opt = bench(&gpu, 20, 200, || {
        pp.regularize(&gpu)?;
        pp.step(&gpu)
    })
    .unwrap();
    let all = bench(&gpu, 20, 200, || {
        pp.forward(&gpu, 0, 7, &rgb, &mut out, w, h)?;
        pp.backward_inplace(&gpu, 0, 7, &rgb, &mut v, w, h)?;
        pp.regularize(&gpu)?;
        pp.step(&gpu)
    })
    .unwrap();
    println!(
        "ppisp {w}x{h}: fwd {:.4} ms, bwd+finalize {:.4} ms, reg+adam {:.4} ms, \
         iteration {:.4} ms (median; p90 {:.4}); {} KiB device",
        fwd.median_ms,
        bwd.median_ms,
        opt.median_ms,
        all.median_ms,
        all.p90_ms,
        pp.bytes() / 1024
    );
}
