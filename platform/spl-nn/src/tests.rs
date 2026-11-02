//! GPU tests. All are `#[ignore]`: on the GPU host, build the crate's test
//! binary and run it in the CUDA container with `--ignored` (as
//! `~/spl-work/bin/gpu-test` does for raw-retouch, with `-p spl-nn`).
//!
//! - `gpu_*`: every kernel against the f64 CPU references in [`crate::host`].
//! - `torch_*`: against PyTorch outputs written by `scripts/torch_ref.py`
//!   (fixtures in `$SPL_NN_FIXTURES`, default /out/nn-dense/fixtures).
//! - `bench_*`: throughput with CUDA events, with `nvidia-smi` state before
//!   and after; results go to `$SPL_OUT/bench_*.json` (default
//!   /out/nn-dense). `scripts/torch_bench.py` measures the PyTorch
//!   baselines on the same shapes.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use cudarc::driver::CudaSlice;
use cudarc::driver::sys::CUevent_flags;

use crate::attention::{AttnShape, FlashAttention, TILES, Tile, VFormat};
use crate::gemm::{Epilogue, Fp8Scales, Gemm, GemmShape, Tuning, WeightScale};
use crate::host::{self, fp8_to_f32, round_bf16};
use crate::norm::{Branch, LnParams, Norm, Quant, Rows, RowsMut};
use crate::qkv::{
    HEAD_DIM, NormKind, QkNorm, QkvIn, QkvOut, QkvPrologue, Rope, RopeLayout, RopeTable,
};
use crate::residual::Residual;
use crate::testkit::{Err, Npy, Rng, err, fixtures_dir, out_dir, widen};
use crate::{Act, Nn};

fn nn() -> Nn {
    Nn::open().expect("open GPU")
}

fn up_f32(nn: &Nn, v: &[f32]) -> CudaSlice<f32> {
    nn.stream.clone_htod(v).expect("upload")
}

/// Uploads as bf16 and returns the rounded values the GPU sees.
fn up_bf16(nn: &Nn, v: &[f32]) -> (CudaSlice<u16>, Vec<f32>) {
    let b = host::to_bf16(v);
    let r = host::from_bf16(&b);
    (nn.stream.clone_htod(&b).expect("upload"), r)
}

fn down_bf16(nn: &Nn, d: &CudaSlice<u16>) -> Vec<f32> {
    host::from_bf16(&nn.stream.clone_dtoh(d).expect("download"))
}

fn down<T: cudarc::driver::DeviceRepr + Default + Clone>(nn: &Nn, d: &CudaSlice<T>) -> Vec<T> {
    nn.stream.clone_dtoh(d).expect("download")
}

fn zeros<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
    nn: &Nn,
    n: usize,
) -> CudaSlice<T> {
    nn.stream.alloc_zeros::<T>(n).expect("alloc")
}

fn check(name: &str, e: Err, rel_rms: f64, max_abs: f64) {
    println!(
        "{name}: max_abs {:.3e} rms {:.3e} rel_rms {:.3e}",
        e.max_abs, e.rms, e.rel_rms
    );
    assert!(
        e.rel_rms <= rel_rms,
        "{name}: rel_rms {:.3e} > {rel_rms:.1e}",
        e.rel_rms
    );
    assert!(
        e.max_abs <= max_abs,
        "{name}: max_abs {:.3e} > {max_abs:.1e}",
        e.max_abs
    );
}

/// FP8 outputs agree when codes differ by at most one step and rarely.
fn check_fp8(name: &str, got: &[u8], want: &[u8], max_rate: f64) {
    assert_eq!(got.len(), want.len());
    let mut diff = 0usize;
    for (&g, &w) in got.iter().zip(want) {
        if g != w {
            diff += 1;
            let (a, b) = (fp8_to_f32(g), fp8_to_f32(w));
            let step = (fp8_to_f32(w.wrapping_add(1) & 0x7f | (w & 0x80)) - b)
                .abs()
                .max(2f32.powi(-9));
            assert!(
                (a - b).abs() <= step * 1.01 + 1e-12,
                "{name}: {g:#x} vs {w:#x}"
            );
        }
    }
    let rate = diff as f64 / got.len().max(1) as f64;
    println!(
        "{name}: {diff} of {} FP8 codes differ by one step ({rate:.2e})",
        got.len()
    );
    assert!(rate <= max_rate, "{name}: mismatch rate {rate:.2e}");
}

/// Mean per-iteration milliseconds over the best of 5 batches of `reps`.
fn time_ms(nn: &Nn, reps: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..3 {
        f();
    }
    nn.sync().unwrap();
    let ev = || {
        nn.ctx
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .unwrap()
    };
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let (a, b) = (ev(), ev());
        a.record(&nn.stream).unwrap();
        for _ in 0..reps {
            f();
        }
        b.record(&nn.stream).unwrap();
        best = best.min(a.elapsed_ms(&b).unwrap() as f64 / reps as f64);
    }
    best
}

// ---------------------------------------------------------------- norm --

struct LnCase {
    x: Vec<f32>,
    gamma: Vec<f32>,
    beta: Vec<f32>,
}

fn ln_case(rng: &mut Rng, rows: usize, c: usize) -> LnCase {
    // Mean offsets and a few outlier channels, like ViT residual streams.
    let mut x = rng.normals(rows * c, 1.0);
    for r in 0..rows {
        let off = rng.normal() * 3.0;
        for i in 0..c {
            x[r * c + i] += off + if i % 97 == 3 { 20.0 } else { 0.0 };
        }
    }
    LnCase {
        x,
        gamma: (0..c).map(|_| 1.0 + rng.normal() * 0.3).collect(),
        beta: rng.normals(c, 0.1),
    }
}

#[test]
#[ignore]
fn gpu_layer_norm_fp8_matches_cpu() {
    let nn = nn();
    let norm = Norm::new(&nn).unwrap();
    let mut rng = Rng::new(1);
    for (c, bf16_stream) in [
        (1024, false),
        (1024, true),
        (768, false),
        (2048, true),
        (256, false),
    ] {
        let rows = 689 * 2 + 3;
        let mut t = ln_case(&mut rng, rows, c);
        if bf16_stream {
            t.x = t.x.iter().map(|&v| round_bf16(v)).collect();
        }
        let gamma = up_f32(&nn, &t.gamma);
        let beta = up_f32(&nn, &t.beta);
        let p = LnParams {
            gamma: &gamma,
            beta: &beta,
            eps: 1e-6,
        };
        let mut out = zeros::<u16>(&nn, rows * c);
        let mut q = zeros::<u8>(&nn, rows * c);
        let mut scales = zeros::<f32>(&nn, rows);
        let x32 = up_f32(&nn, &t.x);
        let (x16, _) = up_bf16(&nn, &t.x);
        let x = if bf16_stream {
            Rows::Bf16(&x16)
        } else {
            Rows::F32(&x32)
        };
        norm.layer_norm(
            &nn,
            x,
            rows,
            c,
            &p,
            Some(&mut out),
            Quant::PerToken {
                q: &mut q,
                scales: &mut scales,
            },
        )
        .unwrap();
        let want = host::layer_norm(&t.x, c, &t.gamma, &t.beta, 1e-6);
        let name = format!("layer_norm c={c} bf16_stream={bf16_stream}");
        check(
            &format!("{name} bf16 out"),
            err(&down_bf16(&nn, &out), &want),
            3e-3,
            0.25,
        );
        let want32: Vec<f32> = want.iter().map(|&v| v as f32).collect();
        let (wq, ws) = host::quant_rows(&want32, c);
        let gs = down(&nn, &scales);
        for (g, w) in gs.iter().zip(&ws) {
            assert!((g / w - 1.0).abs() < 1e-5, "{name}: scale {g} vs {w}");
        }
        check_fp8(&format!("{name} fp8"), &down(&nn, &q), &wq, 2e-3);
        // Dequantised error against the exact norm: FP8's 3-bit mantissa.
        let deq: Vec<f32> = down(&nn, &q)
            .iter()
            .enumerate()
            .map(|(i, &b)| fp8_to_f32(b) * gs[i / c])
            .collect();
        check(&format!("{name} fp8 dequant"), err(&deq, &want), 0.04, 2.0);
    }
}

#[test]
#[ignore]
fn gpu_layer_norm_static_scale_saturates() {
    let nn = nn();
    let norm = Norm::new(&nn).unwrap();
    let mut rng = Rng::new(2);
    let (rows, c) = (37, 1024);
    let t = ln_case(&mut rng, rows, c);
    let gamma = up_f32(&nn, &t.gamma);
    let beta = up_f32(&nn, &t.beta);
    let p = LnParams {
        gamma: &gamma,
        beta: &beta,
        eps: 1e-5,
    };
    let scale = 0.01f32; // saturates beyond |4.48|
    let sd = up_f32(&nn, &[scale]);
    let mut q = zeros::<u8>(&nn, rows * c);
    let x = up_f32(&nn, &t.x);
    norm.layer_norm(
        &nn,
        Rows::F32(&x),
        rows,
        c,
        &p,
        None,
        Quant::Static {
            q: &mut q,
            scale: &sd,
        },
    )
    .unwrap();
    let want: Vec<u8> = host::layer_norm(&t.x, c, &t.gamma, &t.beta, 1e-5)
        .iter()
        .map(|&v| host::fp8(v as f32 * (1.0 / scale)))
        .collect();
    let got = down(&nn, &q);
    assert!(
        got.iter().any(|&b| b & 0x7f == 0x7e),
        "some values saturate"
    );
    check_fp8("layer_norm static", &got, &want, 2e-3);
}

#[test]
#[ignore]
fn gpu_residual_layer_norm_matches_cpu() {
    let nn = nn();
    let norm = Norm::new(&nn).unwrap();
    let res = Residual::new(&nn).unwrap();
    let mut rng = Rng::new(3);
    let (rows, c) = (689 + 11, 1024);
    for bf16_stream in [false, true] {
        let mut t = ln_case(&mut rng, rows, c);
        if bf16_stream {
            t.x = t.x.iter().map(|&v| round_bf16(v)).collect();
        }
        let y = rng.normals(rows * c, 2.0);
        let tok: Vec<f32> = (0..rows).map(|_| 0.5 + rng.uniform()).collect();
        let chan: Vec<f32> = (0..c).map(|_| 0.5 + rng.uniform()).collect();
        let bias = rng.normals(c, 0.2);
        let ls = rng.normals(c, 0.1);
        let (yd, yr) = up_bf16(&nn, &y);
        let (bd, br) = up_bf16(&nn, &bias);
        let (tokd, chand, lsd) = (up_f32(&nn, &tok), up_f32(&nn, &chan), up_f32(&nn, &ls));
        let branch = Branch {
            y: &yd,
            tok: Some(&tokd),
            chan: Some(&chand),
            bias: Some(&bd),
            layer_scale: Some(&lsd),
        };
        // Expected stream after the update.
        let mut x_new: Vec<f32> = (0..rows * c)
            .map(|i| {
                let (r, ch) = (i / c, i % c);
                t.x[i] + ls[ch] * (yr[i] * tok[r] * chan[ch] + br[ch])
            })
            .collect();
        if bf16_stream {
            x_new = x_new.iter().map(|&v| round_bf16(v)).collect();
        }
        let gamma = up_f32(&nn, &t.gamma);
        let beta = up_f32(&nn, &t.beta);
        let p = LnParams {
            gamma: &gamma,
            beta: &beta,
            eps: 1e-6,
        };
        let mut out = zeros::<u16>(&nn, rows * c);
        let mut q = zeros::<u8>(&nn, rows * c);
        let mut scales = zeros::<f32>(&nn, rows);
        let name = format!("residual+ln bf16_stream={bf16_stream}");
        let (got_x, got_x2) = if bf16_stream {
            let (mut xd, _) = up_bf16(&nn, &t.x);
            let (mut xd2, _) = up_bf16(&nn, &t.x);
            norm.residual_layer_norm(
                &nn,
                RowsMut::Bf16(&mut xd),
                &branch,
                rows,
                c,
                &p,
                Some(&mut out),
                Quant::PerToken {
                    q: &mut q,
                    scales: &mut scales,
                },
            )
            .unwrap();
            res.add(&nn, RowsMut::Bf16(&mut xd2), &branch, rows, c, c)
                .unwrap();
            (down_bf16(&nn, &xd), down_bf16(&nn, &xd2))
        } else {
            let mut xd = up_f32(&nn, &t.x);
            let mut xd2 = up_f32(&nn, &t.x);
            norm.residual_layer_norm(
                &nn,
                RowsMut::F32(&mut xd),
                &branch,
                rows,
                c,
                &p,
                Some(&mut out),
                Quant::PerToken {
                    q: &mut q,
                    scales: &mut scales,
                },
            )
            .unwrap();
            res.add(&nn, RowsMut::F32(&mut xd2), &branch, rows, c, c)
                .unwrap();
            (down(&nn, &xd), down(&nn, &xd2))
        };
        let tol = if bf16_stream { 1e-2 } else { 1e-5 };
        check(
            &format!("{name} stream"),
            err(&got_x, &widen(&x_new)),
            1e-3,
            tol * 30.0,
        );
        check(
            &format!("residual bf16_stream={bf16_stream}"),
            err(&got_x2, &widen(&x_new)),
            1e-3,
            tol * 30.0,
        );
        let want = host::layer_norm(&x_new, c, &t.gamma, &t.beta, 1e-6);
        check(
            &format!("{name} norm"),
            err(&down_bf16(&nn, &out), &want),
            3e-3,
            0.25,
        );
    }
}

#[test]
#[ignore]
fn gpu_row_epilogue_matches_cpu() {
    let nn = nn();
    let norm = Norm::new(&nn).unwrap();
    let mut rng = Rng::new(4);
    for (c, act) in [
        (4096, Act::GeluErf),
        (4096, Act::GeluTanh),
        (1024, Act::None),
        (3072, Act::GeluErf),
    ] {
        let rows = 203;
        let y = rng.normals(rows * c, 1.5);
        let tok: Vec<f32> = (0..rows).map(|_| 0.5 + rng.uniform()).collect();
        let chan: Vec<f32> = (0..c).map(|_| 0.5 + rng.uniform()).collect();
        let bias = rng.normals(c, 0.3);
        let (mut yd, yr) = up_bf16(&nn, &y);
        let (bd, br) = up_bf16(&nn, &bias);
        let (tokd, chand) = (up_f32(&nn, &tok), up_f32(&nn, &chan));
        let mut q = zeros::<u8>(&nn, rows * c);
        let mut scales = zeros::<f32>(&nn, rows);
        norm.row_epilogue(
            &nn,
            &mut yd,
            rows,
            c,
            c,
            Some(&tokd),
            Some(&chand),
            Some(&bd),
            act,
            true,
            Quant::PerToken {
                q: &mut q,
                scales: &mut scales,
            },
        )
        .unwrap();
        let want: Vec<f64> = (0..rows * c)
            .map(|i| {
                let (r, ch) = (i / c, i % c);
                let v = yr[i] as f64 * tok[r] as f64 * chan[ch] as f64 + br[ch] as f64;
                match act {
                    Act::None => v,
                    Act::GeluErf => host::gelu_erf(v),
                    Act::GeluTanh => host::gelu_tanh(v),
                }
            })
            .collect();
        let name = format!("row_epilogue c={c} {act:?}");
        check(
            &format!("{name} bf16"),
            err(&down_bf16(&nn, &yd), &want),
            3e-3,
            0.05,
        );
        let want32: Vec<f32> = want.iter().map(|&v| v as f32).collect();
        let (wq, _) = host::quant_rows(&want32, c);
        check_fp8(&format!("{name} fp8"), &down(&nn, &q), &wq, 2e-3);
    }
}

// ----------------------------------------------------------------- qkv --

#[test]
#[ignore]
fn gpu_qkv_prologue_matches_cpu() {
    let nn = nn();
    let pro = QkvPrologue::new(&nn).unwrap();
    let mut rng = Rng::new(5);
    let (grid_h, grid_w, special) = (21, 32, 17); // 512 × 336 at 16 px: 689 tokens
    let frame_len = special + grid_h * grid_w;
    let frames = 2;
    let tokens = frames * frame_len;
    for (heads, layout, kind, scaled) in [
        (16, Some(RopeLayout::Vggt), Some(NormKind::LayerNorm), false),
        (16, Some(RopeLayout::Dinov3), Some(NormKind::Rms), true),
        (6, Some(RopeLayout::Vggt), None, true),
        (3, None, Some(NormKind::LayerNorm), false),
    ] {
        let width = 3 * heads * HEAD_DIM;
        let ld = width + 8;
        let y = rng.normals(tokens * ld, 1.0);
        let (yd, yr) = up_bf16(&nn, &y);
        let tok: Vec<f32> = (0..tokens).map(|_| 0.5 + rng.uniform()).collect();
        let chan: Vec<f32> = (0..width).map(|_| 0.5 + rng.uniform()).collect();
        let bias = rng.normals(width, 0.2);
        let (bd, br) = up_bf16(&nn, &bias);
        let (tokd, chand) = (up_f32(&nn, &tok), up_f32(&nn, &chan));
        let nw: Vec<Vec<f32>> = (0..4)
            .map(|i| {
                if i % 2 == 0 {
                    (0..HEAD_DIM).map(|_| 1.0 + rng.normal() * 0.2).collect()
                } else {
                    rng.normals(HEAD_DIM, 0.1)
                }
            })
            .collect();
        let nwd: Vec<CudaSlice<f32>> = nw.iter().map(|v| up_f32(&nn, v)).collect();
        let table = layout.map(|l| match l {
            RopeLayout::Vggt => RopeTable::vggt(grid_h, grid_w, special, 100.0),
            RopeLayout::Dinov3 => RopeTable::dinov3(grid_h, grid_w, special, 100.0),
        });
        let table_d = table.as_ref().map(|t| t.upload(&nn).unwrap());
        let rope = table_d.as_ref().map(|d| Rope {
            table: d,
            layout: layout.unwrap(),
            frame_len,
        });
        let qk = kind.map(|k| QkNorm {
            kind: k,
            q_weight: &nwd[0],
            q_bias: Some(&nwd[1]),
            k_weight: &nwd[2],
            k_bias: Some(&nwd[3]),
            eps: 1e-6,
        });
        let n = tokens * heads * HEAD_DIM;
        let (mut q, mut k, mut v) = (
            zeros::<u16>(&nn, n),
            zeros::<u16>(&nn, n),
            zeros::<u16>(&nn, n),
        );
        let input = QkvIn {
            y: &yd,
            ld,
            tok: scaled.then_some(&tokd as _),
            chan: scaled.then_some(&chand as _),
            bias: Some(&bd),
        };
        pro.run(
            &nn,
            &input,
            tokens,
            heads,
            qk.as_ref(),
            rope.as_ref(),
            QkvOut {
                q: &mut q,
                k: &mut k,
                v: &mut v,
                v_format: VFormat::Bf16,
            },
        )
        .unwrap();
        let want = crate::qkv::reference(
            &yr,
            ld,
            tokens,
            heads,
            scaled.then_some(&tok[..]),
            scaled.then_some(&chan[..]),
            Some(&br),
            kind.map(|k| (k, [&nw[0][..], &nw[1][..], &nw[2][..], &nw[3][..]], 1e-6)),
            table.as_ref(),
        );
        let name = format!("qkv heads={heads} {layout:?} {kind:?} scaled={scaled}");
        if heads == 16 && !scaled {
            let mut v16 = zeros::<u16>(&nn, n);
            let (mut q2, mut k2) = (zeros::<u16>(&nn, n), zeros::<u16>(&nn, n));
            pro.run(
                &nn,
                &input,
                tokens,
                heads,
                qk.as_ref(),
                rope.as_ref(),
                QkvOut {
                    q: &mut q2,
                    k: &mut k2,
                    v: &mut v16,
                    v_format: VFormat::F16,
                },
            )
            .unwrap();
            let got: Vec<f32> = down(&nn, &v16)
                .iter()
                .map(|&h| host::f16_to_f32(h))
                .collect();
            check(&format!("{name} v (fp16)"), err(&got, &want[2]), 5e-4, 4e-3);
        }
        for (part, d) in [("q", &q), ("k", &k), ("v", &v)].into_iter().enumerate() {
            check(
                &format!("{name} {}", d.0),
                err(&down_bf16(&nn, d.1), &want[part]),
                3e-3,
                0.06,
            );
        }
    }
}

// ----------------------------------------------------------- attention --

fn ref_attention(q: &[f32], k: &[f32], v: &[f32], s: &AttnShape, scale: f64) -> Vec<f64> {
    let (l, total, h) = (s.seq_len, s.total_tokens(), s.heads);
    let mut out = vec![0.0f64; total * s.o_ld];
    for seq in 0..s.n_seq {
        let gather = |src: &[f32]| -> Vec<f32> {
            let mut o = Vec::with_capacity(h * l * HEAD_DIM);
            for hh in 0..h {
                let base = (hh * total + seq * l) * HEAD_DIM;
                o.extend_from_slice(&src[base..base + l * HEAD_DIM]);
            }
            o
        };
        let o = host::attention(&gather(q), &gather(k), &gather(v), h, HEAD_DIM, scale);
        for t in 0..l {
            let dst = (seq * l + t) * s.o_ld;
            out[dst..dst + h * HEAD_DIM]
                .copy_from_slice(&o[t * h * HEAD_DIM..(t + 1) * h * HEAD_DIM]);
        }
    }
    out
}

#[test]
#[ignore]
fn gpu_attention_matches_cpu_for_every_tile() {
    let nn = nn();
    let fa = FlashAttention::new(&nn).unwrap();
    let mut rng = Rng::new(6);
    let shapes = [
        AttnShape::frames(2, 689, 2),
        AttnShape::frames(3, 50, 3),
        AttnShape::frames(1, 129, 1),
        AttnShape::global(2, 689, 1),
        AttnShape {
            o_ld: 3 * HEAD_DIM + 16,
            ..AttnShape::frames(2, 200, 3)
        },
    ];
    for s in shapes {
        let n = s.heads * s.total_tokens() * HEAD_DIM;
        let (qd, qr) = up_bf16(&nn, &rng.normals(n, 2.0));
        let (kd, kr) = up_bf16(&nn, &rng.normals(n, 1.0));
        let v = rng.normals(n, 1.0);
        let (vd, vr) = up_bf16(&nn, &v);
        let v16: Vec<u16> = v.iter().map(|&x| host::f16(x)).collect();
        let vr16: Vec<f32> = v16.iter().map(|&h| host::f16_to_f32(h)).collect();
        let vd16 = nn.stream.clone_htod(&v16).unwrap();
        let scale = 1.0 / (HEAD_DIM as f64).sqrt();
        let mut want = ref_attention(&qr, &kr, &vr, &s, scale);
        let mut want16 = ref_attention(&qr, &kr, &vr16, &s, scale);
        // Columns past heads · 64 are never written.
        for t in 0..s.total_tokens() {
            for c in s.heads * HEAD_DIM..s.o_ld {
                want[t * s.o_ld + c] = 0.0;
                want16[t * s.o_ld + c] = 0.0;
            }
        }
        for tile in TILES {
            let mut o = zeros::<u16>(&nn, s.total_tokens() * s.o_ld);
            let (vdev, want) = match tile.v {
                VFormat::Bf16 => (&vd, &want),
                VFormat::F16 => (&vd16, &want16),
            };
            fa.forward(&nn, &qd, &kd, vdev, &mut o, &s, scale as f32, Some(tile))
                .unwrap();
            check(
                &format!("attention {s:?} {tile:?}"),
                err(&down_bf16(&nn, &o), want),
                6e-3,
                0.03,
            );
        }
    }
}

// ---------------------------------------------------------------- gemm --

#[test]
#[ignore]
fn gpu_gemm_bf16_matches_cpu() {
    let nn = nn();
    let gemm = Gemm::new(&nn, Tuning::Measure(8)).unwrap();
    println!("cuBLASLt {}", Gemm::version());
    let mut rng = Rng::new(7);
    for (m, n, k, bias, epi) in [
        (300, 512, 1024, true, Epilogue::Gelu),
        (77, 256, 512, false, Epilogue::None),
        (689, 1024, 256, true, Epilogue::None),
    ] {
        let (xd, xr) = up_bf16(&nn, &rng.normals(m * k, 1.0));
        let (wd, wr) = up_bf16(&nn, &rng.normals(n * k, 1.0 / (k as f32).sqrt()));
        let (bd, br) = up_bf16(&nn, &rng.normals(n, 0.5));
        let mut y = zeros::<u16>(&nn, m * n);
        let s = GemmShape::packed(m, n, k);
        gemm.linear_bf16(&nn, &xd, &wd, bias.then_some(&bd as _), epi, &mut y, s)
            .unwrap();
        let mut want = host::linear(&xr, &wr, bias.then_some(&br[..]), k, n);
        if epi == Epilogue::Gelu {
            want.iter_mut().for_each(|v| *v = host::gelu_tanh(*v));
        }
        check(
            &format!("gemm bf16 {m}x{n}x{k} bias={bias} {epi:?}"),
            err(&down_bf16(&nn, &y), &want),
            4e-3,
            0.05,
        );
    }
}

/// Dequantised `[rows][k]` from FP8 codes and per-row scales.
fn dequant_rows(q: &[u8], scales: &[f32], k: usize) -> Vec<f32> {
    q.iter()
        .enumerate()
        .map(|(i, &b)| fp8_to_f32(b) * scales[i / k])
        .collect()
}

/// Per-tensor FP8 quantisation: (codes, scale).
fn quant_tensor(v: &[f32]) -> (Vec<u8>, f32) {
    let amax = v.iter().fold(0.0f32, |a, x| a.max(x.abs()));
    let s = amax / host::FP8_MAX;
    (v.iter().map(|&x| host::fp8(x * (1.0 / s))).collect(), s)
}

#[test]
#[ignore]
fn gpu_fp8_scale_modes() {
    // Which FP8 scale configurations cuBLASLt accepts on this GPU.
    let nn = nn();
    let gemm = Gemm::new(&nn, Tuning::Heuristic).unwrap();
    let (major, minor) = nn.compute_capability();
    println!(
        "cuBLASLt {} on sm_{major}{minor}: FP8 per-channel alpha vector supported: {}",
        Gemm::version(),
        gemm.supports_channel_alpha()
    );
}

#[test]
#[ignore]
fn gpu_gemm_fp8_matches_cpu() {
    let nn = nn();
    let gemm = Gemm::new(&nn, Tuning::Measure(8)).unwrap();
    let mut rng = Rng::new(8);
    let (m, n, k) = (203, 768, 1024);
    let x = rng.normals(m * k, 1.0);
    // Weights with a spread of per-channel magnitudes (0.2x to 2x).
    let w: Vec<f32> = (0..n * k)
        .map(|i| rng.normal() * (0.2 + (i / k % 7) as f32 * 0.3) / (k as f32).sqrt())
        .collect();
    let bias = rng.normals(n, 0.5);
    let (bd, br) = up_bf16(&nn, &bias);
    let exact = host::linear(&x, &w, Some(&bias), k, n);
    let s = GemmShape::packed(m, n, k);

    let (xq, xs) = quant_tensor(&x);
    let xdeq: Vec<f32> = xq.iter().map(|&b| fp8_to_f32(b) * xs).collect();
    let (xqd, xsd) = (nn.stream.clone_htod(&xq).unwrap(), up_f32(&nn, &[xs]));

    // Per-tensor weight and activation scales: bias and GELU fused.
    let (wq, ws) = quant_tensor(&w);
    let wdeq: Vec<f32> = wq.iter().map(|&b| fp8_to_f32(b) * ws).collect();
    let (wqd, wsd) = (nn.stream.clone_htod(&wq).unwrap(), up_f32(&nn, &[ws]));
    for epi in [Epilogue::None, Epilogue::Gelu] {
        let mut y = zeros::<u16>(&nn, m * n);
        let scales = Fp8Scales {
            weight: WeightScale::Tensor(&wsd),
            act: Some(&xsd),
        };
        gemm.linear_fp8(&nn, &xqd, &wqd, &scales, Some(&bd), epi, &mut y, s)
            .unwrap();
        let act = |v: f64| {
            if epi == Epilogue::Gelu {
                host::gelu_tanh(v)
            } else {
                v
            }
        };
        let want: Vec<f64> = host::linear(&xdeq, &wdeq, Some(&br), k, n)
            .into_iter()
            .map(act)
            .collect();
        let got = down_bf16(&nn, &y);
        check(
            &format!("gemm fp8 tensor/tensor + bias {epi:?} vs dequantised"),
            err(&got, &want),
            4e-3,
            0.05,
        );
        let e = err(&got, &exact.iter().map(|&v| act(v)).collect::<Vec<_>>());
        println!("  FP8 error vs unquantised: rel_rms {:.3e}", e.rel_rms);
    }

    // Per-channel weight scales.
    let (wq, ws) = host::quant_weights(&w, k);
    let wdeq = dequant_rows(&wq, &ws, k);
    let (wqd, wsd) = (nn.stream.clone_htod(&wq).unwrap(), up_f32(&nn, &ws));
    let mut y = zeros::<u16>(&nn, m * n);
    let scales = Fp8Scales {
        weight: WeightScale::Channel(&wsd),
        act: Some(&xsd),
    };
    let r = gemm.linear_fp8(
        &nn,
        &xqd,
        &wqd,
        &scales,
        Some(&bd),
        Epilogue::Gelu,
        &mut y,
        s,
    );
    if gemm.supports_channel_alpha() {
        r.unwrap();
        let want: Vec<f64> = host::linear(&xdeq, &wdeq, Some(&br), k, n)
            .into_iter()
            .map(host::gelu_tanh)
            .collect();
        check(
            "gemm fp8 channel alpha + bias + gelu",
            err(&down_bf16(&nn, &y), &want),
            4e-3,
            0.05,
        );
    } else {
        let e = r.expect_err("channel alpha must be reported as unsupported");
        println!("channel alpha unsupported here, as reported: {e:#}");
    }

    // Deferred channel scales (the Ada path): the GEMM writes
    // act_scale · xq · wqᵀ and the consumer applies scales and bias.
    let mut y = zeros::<u16>(&nn, m * n);
    let scales = Fp8Scales {
        weight: WeightScale::Deferred,
        act: Some(&xsd),
    };
    assert!(
        gemm.linear_fp8(
            &nn,
            &xqd,
            &wqd,
            &scales,
            Some(&bd),
            Epilogue::None,
            &mut y,
            s
        )
        .is_err(),
        "bias before deferred channel scales is rejected"
    );
    gemm.linear_fp8(&nn, &xqd, &wqd, &scales, None, Epilogue::None, &mut y, s)
        .unwrap();
    let wq_f: Vec<f32> = wq.iter().map(|&b| fp8_to_f32(b)).collect();
    let want = host::linear(&xdeq, &wq_f, None, k, n);
    // Unscaled sums reach ~1e4; bf16 rounding bounds the error relatively.
    let peak = want.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    check(
        "gemm fp8 deferred channel scales",
        err(&down_bf16(&nn, &y), &want),
        4e-3,
        peak * 2f64.powi(-8),
    );
    let full: Vec<f32> = down_bf16(&nn, &y)
        .iter()
        .enumerate()
        .map(|(i, &v)| v * ws[i % n] + br[i % n])
        .collect();
    let e = err(&full, &exact);
    println!(
        "  FP8 per-channel error vs unquantised: rel_rms {:.3e}",
        e.rel_rms
    );
}

#[test]
#[ignore]
fn gpu_fp8_per_token_block_path() {
    // The Ada FP8 MLP hand-off with per-token activations and per-channel
    // weights: fc1 on cuBLASLt (scales deferred), then the row epilogue
    // applies token and channel scales, bias and GELU (erf) and
    // re-quantises per token for fc2.
    let nn = nn();
    let gemm = Gemm::new(&nn, Tuning::Heuristic).unwrap();
    let norm = Norm::new(&nn).unwrap();
    let mut rng = Rng::new(9);
    let (m, n, k) = (150, 4096, 1024);
    let x = rng.normals(m * k, 1.0);
    let w = rng.normals(n * k, 1.0 / (k as f32).sqrt());
    let bias = rng.normals(n, 0.5);
    let (wq, ws) = host::quant_weights(&w, k);
    let (xq, xs) = host::quant_rows(&x, k);
    let (wdeq, xdeq) = (dequant_rows(&wq, &ws, k), dequant_rows(&xq, &xs, k));
    let (wqd, wsd) = (nn.stream.clone_htod(&wq).unwrap(), up_f32(&nn, &ws));
    let (xqd, xsd) = (nn.stream.clone_htod(&xq).unwrap(), up_f32(&nn, &xs));
    let (bd, br) = up_bf16(&nn, &bias);
    let mut y = zeros::<u16>(&nn, m * n);
    let scales = Fp8Scales {
        weight: WeightScale::Deferred,
        act: None,
    };
    gemm.linear_fp8(
        &nn,
        &xqd,
        &wqd,
        &scales,
        None,
        Epilogue::None,
        &mut y,
        GemmShape::packed(m, n, k),
    )
    .unwrap();
    let mut hq = zeros::<u8>(&nn, m * n);
    let mut hs = zeros::<f32>(&nn, m);
    norm.row_epilogue(
        &nn,
        &mut y,
        m,
        n,
        n,
        Some(&xsd),
        Some(&wsd),
        Some(&bd),
        Act::GeluErf,
        true,
        Quant::PerToken {
            q: &mut hq,
            scales: &mut hs,
        },
    )
    .unwrap();
    let want: Vec<f64> = host::linear(&xdeq, &wdeq, Some(&br), k, n)
        .into_iter()
        .map(host::gelu_erf)
        .collect();
    check(
        "fc1 fp8 per-token + row epilogue (bf16)",
        err(&down_bf16(&nn, &y), &want),
        5e-3,
        0.08,
    );
    let deq = dequant_rows(&down(&nn, &hq), &down(&nn, &hs), n);
    check(
        "fc1 fp8 per-token + row epilogue (fp8 dequant)",
        err(&deq, &want),
        0.05,
        1.0,
    );
}

// --------------------------------------------------------------- torch --

fn fixture(name: &str) -> Npy {
    let p = fixtures_dir().join(format!("{name}.npy"));
    assert!(
        p.exists(),
        "{} missing: run scripts/torch_ref.py first (see scripts/README.md)",
        p.display()
    );
    Npy::read(&p)
}

fn fixture_f64(name: &str) -> Vec<f64> {
    widen(&fixture(name).f32())
}

#[test]
#[ignore]
fn torch_attention_matches_sdpa() {
    let nn = nn();
    let fa = FlashAttention::new(&nn).unwrap();
    for case in ["frame", "global"] {
        let q = fixture(&format!("attn_{case}_q"));
        let (heads, total) = (q.shape[0], q.shape[1]);
        let seq_len = fixture(&format!("attn_{case}_seq_len")).f32()[0] as usize;
        let s = AttnShape {
            seq_len,
            n_seq: total / seq_len,
            heads,
            o_ld: heads * HEAD_DIM,
        };
        let up = |n: &str| {
            nn.stream
                .clone_htod(&fixture(&format!("attn_{case}_{n}")).u16())
                .unwrap()
        };
        let (qd, kd, vd) = (up("q"), up("k"), up("v"));
        let v16: Vec<u16> = fixture(&format!("attn_{case}_v"))
            .u16()
            .iter()
            .map(|&b| host::f16(host::bf16_to_f32(b)))
            .collect();
        let vd16 = nn.stream.clone_htod(&v16).unwrap();
        let want = fixture_f64(&format!("attn_{case}_ref_fp32"));
        let sdpa = fixture(&format!("attn_{case}_sdpa_bf16")).u16();
        let e_torch = err(&host::from_bf16(&sdpa), &want);
        println!(
            "torch SDPA bf16 ({case}) vs fp32 reference: rel_rms {:.3e} max_abs {:.3e}",
            e_torch.rel_rms, e_torch.max_abs
        );
        for tile in [
            FlashAttention::default_tile(&s, VFormat::Bf16),
            Tile::new(4, 2, 64),
            FlashAttention::default_tile(&s, VFormat::F16),
        ] {
            let mut o = zeros::<u16>(&nn, total * s.o_ld);
            let v = if tile.v == VFormat::F16 { &vd16 } else { &vd };
            fa.forward(&nn, &qd, &kd, v, &mut o, &s, 0.125, Some(tile))
                .unwrap();
            let e = err(&down_bf16(&nn, &o), &want);
            check(
                &format!("ours ({case}, {tile:?}) vs torch fp32"),
                e,
                (e_torch.rel_rms * 1.5).max(2e-3),
                0.03,
            );
        }
    }
}

#[test]
#[ignore]
fn torch_layer_norm_fp8_matches() {
    let nn = nn();
    let norm = Norm::new(&nn).unwrap();
    let x = fixture("ln_x").f32();
    let (rows, c) = {
        let s = fixture("ln_x").shape;
        (s[0], s[1])
    };
    let (g, b) = (fixture("ln_gamma").f32(), fixture("ln_beta").f32());
    let (xd, gd, bd) = (up_f32(&nn, &x), up_f32(&nn, &g), up_f32(&nn, &b));
    let p = LnParams {
        gamma: &gd,
        beta: &bd,
        eps: 1e-6,
    };
    let mut out = zeros::<u16>(&nn, rows * c);
    let mut q = zeros::<u8>(&nn, rows * c);
    let mut scales = zeros::<f32>(&nn, rows);
    norm.layer_norm(
        &nn,
        Rows::F32(&xd),
        rows,
        c,
        &p,
        Some(&mut out),
        Quant::PerToken {
            q: &mut q,
            scales: &mut scales,
        },
    )
    .unwrap();
    check(
        "ln vs torch fp32 layer_norm",
        err(&down_bf16(&nn, &out), &fixture_f64("ln_ref_fp32")),
        3e-3,
        0.25,
    );
    let ts = fixture("ln_scale").f32();
    for (g, w) in down(&nn, &scales).iter().zip(&ts) {
        assert!((g / w - 1.0).abs() < 1e-5, "scale {g} vs torch {w}");
    }
    check_fp8(
        "ln fp8 vs torch float8_e4m3fn",
        &down(&nn, &q),
        &fixture("ln_q").u8(),
        2e-3,
    );
}

#[test]
#[ignore]
fn torch_gemm_matches() {
    let nn = nn();
    let gemm = Gemm::new(&nn, Tuning::Measure(8)).unwrap();
    let x = fixture("gemm_x");
    let (m, k) = (x.shape[0], x.shape[1]);
    let w = fixture("gemm_w");
    let n = w.shape[0];
    let s = GemmShape::packed(m, n, k);
    let (xd, wd) = (
        nn.stream.clone_htod(&x.u16()).unwrap(),
        nn.stream.clone_htod(&w.u16()).unwrap(),
    );
    let bd = nn.stream.clone_htod(&fixture("gemm_bias").u16()).unwrap();
    let mut y = zeros::<u16>(&nn, m * n);
    gemm.linear_bf16(&nn, &xd, &wd, Some(&bd), Epilogue::Gelu, &mut y, s)
        .unwrap();
    let want = fixture_f64("gemm_ref_gelu_tanh_fp32");
    let et = err(
        &host::from_bf16(&fixture("gemm_torch_gelu_tanh_bf16").u16()),
        &want,
    );
    println!(
        "torch bf16 linear+gelu(tanh) vs fp32: rel_rms {:.3e}",
        et.rel_rms
    );
    check(
        "ours bf16 linear+gelu vs torch fp32 (tanh)",
        err(&down_bf16(&nn, &y), &want),
        (et.rel_rms * 1.5).max(3e-3),
        0.05,
    );
    let e_erf = err(&down_bf16(&nn, &y), &fixture_f64("gemm_ref_gelu_erf_fp32"));
    println!(
        "ours (cuBLASLt GELU) vs torch erf GELU fp32: rel_rms {:.3e} max_abs {:.3e}",
        e_erf.rel_rms, e_erf.max_abs
    );

    // FP8: torch._scaled_mm with per-tensor scales vs ours.
    let xq = nn.stream.clone_htod(&fixture("gemm8_xq").u8()).unwrap();
    let wq = nn.stream.clone_htod(&fixture("gemm8_wq").u8()).unwrap();
    let ws = up_f32(&nn, &fixture("gemm8_ws").f32());
    let xs = up_f32(&nn, &fixture("gemm8_xs").f32());
    let mut y8 = zeros::<u16>(&nn, m * n);
    gemm.linear_fp8(
        &nn,
        &xq,
        &wq,
        &Fp8Scales {
            weight: WeightScale::Tensor(&ws),
            act: Some(&xs),
        },
        Some(&bd),
        Epilogue::None,
        &mut y8,
        s,
    )
    .unwrap();
    let want8 = fixture_f64("gemm8_ref_fp32");
    let et8 = err(&host::from_bf16(&fixture("gemm8_torch_bf16").u16()), &want8);
    println!(
        "torch _scaled_mm (per-tensor) bf16 vs fp32 dequantised: rel_rms {:.3e}",
        et8.rel_rms
    );
    check(
        "ours fp8 vs fp32 dequantised",
        err(&down_bf16(&nn, &y8), &want8),
        (et8.rel_rms * 1.5).max(3e-3),
        0.05,
    );
}

#[test]
#[ignore]
fn torch_qkv_prologue_matches() {
    let nn = nn();
    let pro = QkvPrologue::new(&nn).unwrap();
    let y = fixture("qkv_y");
    let (tokens, width) = (y.shape[0], y.shape[1]);
    let heads = width / (3 * HEAD_DIM);
    let meta = fixture("qkv_meta").f32(); // grid_h, grid_w, special
    let (gh, gw, sp) = (meta[0] as usize, meta[1] as usize, meta[2] as usize);
    let yd = nn.stream.clone_htod(&y.u16()).unwrap();
    let bd = nn.stream.clone_htod(&fixture("qkv_bias").u16()).unwrap();
    let nw: Vec<CudaSlice<f32>> = ["qn_w", "qn_b", "kn_w", "kn_b"]
        .iter()
        .map(|n| up_f32(&nn, &fixture(&format!("qkv_{n}")).f32()))
        .collect();
    let table = RopeTable::vggt(gh, gw, sp, 100.0);
    let td = table.upload(&nn).unwrap();
    let n = tokens * heads * HEAD_DIM;
    let (mut q, mut k, mut v) = (
        zeros::<u16>(&nn, n),
        zeros::<u16>(&nn, n),
        zeros::<u16>(&nn, n),
    );
    pro.run(
        &nn,
        &QkvIn {
            y: &yd,
            ld: width,
            tok: None,
            chan: None,
            bias: Some(&bd),
        },
        tokens,
        heads,
        Some(&QkNorm {
            kind: NormKind::LayerNorm,
            q_weight: &nw[0],
            q_bias: Some(&nw[1]),
            k_weight: &nw[2],
            k_bias: Some(&nw[3]),
            eps: 1e-6,
        }),
        Some(&Rope {
            table: &td,
            layout: RopeLayout::Vggt,
            frame_len: table.frame_len,
        }),
        QkvOut {
            q: &mut q,
            k: &mut k,
            v: &mut v,
            v_format: VFormat::Bf16,
        },
    )
    .unwrap();
    for (name, d) in [("q", &q), ("k", &k), ("v", &v)] {
        check(
            &format!("qkv {name} vs torch (VGGT-style RoPE)"),
            err(&down_bf16(&nn, d), &fixture_f64(&format!("qkv_ref_{name}"))),
            3e-3,
            0.06,
        );
    }
}

// --------------------------------------------------------------- bench --

/// GPU state next to timings, so contention is visible.
fn gpu_state() -> String {
    std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu,clocks.sm,clocks.mem,temperature.gpu,power.draw",
            "--format=csv,noheader",
        ])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "nvidia-smi unavailable".into())
}

fn write_bench(section: &str, rows: &BTreeMap<String, f64>) {
    let path = out_dir().join(format!("bench_{section}.json"));
    let mut s = String::from("{\n");
    for (i, (k, v)) in rows.iter().enumerate() {
        let _ = writeln!(
            s,
            "  \"{k}\": {v:.4}{}",
            if i + 1 < rows.len() { "," } else { "" }
        );
    }
    s.push_str("}\n");
    std::fs::write(&path, s).unwrap();
    println!("wrote {}", path.display());
}

#[test]
#[ignore]
fn bench_attention() {
    let nn = nn();
    let fa = FlashAttention::new(&nn).unwrap();
    for (t, regs, ctas) in fa.occupancy().unwrap() {
        println!(
            "{t:?}: {regs} registers, {} B shared, {ctas} CTAs/SM",
            t.smem_bytes()
        );
    }
    println!("GPU before: {}", gpu_state());
    let mut rng = Rng::new(10);
    let mut rows = BTreeMap::new();
    let cases = [
        ("frame_n2", AttnShape::frames(2, 689, 16)),
        ("frame_n16", AttnShape::frames(16, 689, 16)),
        ("frame_n64", AttnShape::frames(64, 689, 16)),
        ("frame_n200", AttnShape::frames(200, 689, 16)),
        ("global_n8", AttnShape::global(8, 689, 16)),
        ("global_n32", AttnShape::global(32, 689, 16)),
        ("global_n64", AttnShape::global(64, 689, 16)),
    ];
    for (name, s) in cases {
        let n = s.heads * s.total_tokens() * HEAD_DIM;
        let q = nn
            .stream
            .clone_htod(&host::to_bf16(&rng.normals(n, 1.0)))
            .unwrap();
        let k = nn
            .stream
            .clone_htod(&host::to_bf16(&rng.normals(n, 1.0)))
            .unwrap();
        let vn = rng.normals(n, 1.0);
        let v = nn.stream.clone_htod(&host::to_bf16(&vn)).unwrap();
        let v16 = nn
            .stream
            .clone_htod(&vn.iter().map(|&x| host::f16(x)).collect::<Vec<_>>())
            .unwrap();
        let mut o = zeros::<u16>(&nn, s.total_tokens() * s.o_ld);
        let reps = ((2e11 / s.flops()) as usize).clamp(2, 200);
        let mut best = (0.0, TILES[0]);
        let mut best16 = (0.0, TILES[0]);
        for tile in TILES {
            let vt = if tile.v == VFormat::F16 { &v16 } else { &v };
            let ms = time_ms(&nn, reps, || {
                fa.forward(&nn, &q, &k, vt, &mut o, &s, 0.125, Some(tile))
                    .unwrap();
            });
            let tflops = s.flops() / (ms * 1e-3) / 1e12;
            println!("attention {name:<11} {tile:?}: {ms:8.3} ms  {tflops:6.2} TFLOPS");
            rows.insert(
                format!(
                    "attn_{name}_w{}m{}n{}{}{}_tflops",
                    tile.warps,
                    tile.mt,
                    tile.bn,
                    if tile.alias_q { "qa" } else { "" },
                    if tile.v == VFormat::F16 { "pv16" } else { "" }
                ),
                tflops,
            );
            let slot = if tile.v == VFormat::F16 {
                &mut best16
            } else {
                &mut best
            };
            if tflops > slot.0 {
                *slot = (tflops, tile);
            }
        }
        println!(
            "attention {name}: best bf16 {:?} at {:.2} TFLOPS (default {:?}); best fp16-PV {:?} at {:.2}",
            best.1,
            best.0,
            FlashAttention::default_tile(&s, VFormat::Bf16),
            best16.1,
            best16.0,
        );
        rows.insert(format!("attn_{name}_best_tflops"), best.0);
        rows.insert(format!("attn_{name}_best_pv16_tflops"), best16.0);
    }
    println!("GPU after: {}", gpu_state());
    write_bench("attention", &rows);
}

#[test]
#[ignore]
fn bench_gemm() {
    let nn = nn();
    let gemm_h = Gemm::new(&nn, Tuning::Heuristic).unwrap();
    let gemm_m = Gemm::new(&nn, Tuning::Measure(16)).unwrap();
    println!("cuBLASLt {}; GPU before: {}", Gemm::version(), gpu_state());
    let mut rng = Rng::new(11);
    let mut rows = BTreeMap::new();
    for frames in [2usize, 32, 200] {
        let m = frames * 689;
        for (layer, n, k) in [
            ("qkv", 3072, 1024),
            ("proj", 1024, 1024),
            ("fc1", 4096, 1024),
            ("fc2", 1024, 4096),
        ] {
            let s = GemmShape::packed(m, n, k);
            let x = nn
                .stream
                .clone_htod(&host::to_bf16(&rng.normals(m * k, 1.0)))
                .unwrap();
            let w = nn
                .stream
                .clone_htod(&host::to_bf16(&rng.normals(n * k, 0.03)))
                .unwrap();
            let b = nn
                .stream
                .clone_htod(&host::to_bf16(&rng.normals(n, 0.1)))
                .unwrap();
            let x8 = nn.stream.clone_htod(&vec![0x38u8; m * k]).unwrap();
            let w8 = nn.stream.clone_htod(&vec![0x30u8; n * k]).unwrap();
            let ws = up_f32(&nn, &[0.01f32]);
            let xs = up_f32(&nn, &[0.02f32]);
            let mut y = zeros::<u16>(&nn, m * n);
            let epi = if layer == "fc1" {
                Epilogue::Gelu
            } else {
                Epilogue::None
            };
            let reps = ((1e12 / s.flops()) as usize).clamp(3, 200);
            for (tag, g) in [("heur", &gemm_h), ("tuned", &gemm_m)] {
                let ms = time_ms(&nn, reps, || {
                    g.linear_bf16(&nn, &x, &w, Some(&b), epi, &mut y, s)
                        .unwrap();
                });
                let t = s.flops() / (ms * 1e-3) / 1e12;
                println!("gemm bf16 {layer:<4} m={m:<6} {tag:<5}: {ms:8.3} ms {t:6.1} TFLOPS");
                rows.insert(format!("gemm_bf16_{layer}_n{frames}_{tag}_tflops"), t);
                let ms = time_ms(&nn, reps, || {
                    g.linear_fp8(
                        &nn,
                        &x8,
                        &w8,
                        &Fp8Scales {
                            weight: WeightScale::Tensor(&ws),
                            act: Some(&xs),
                        },
                        Some(&b),
                        epi,
                        &mut y,
                        s,
                    )
                    .unwrap();
                });
                let t = s.flops() / (ms * 1e-3) / 1e12;
                println!(
                    "gemm fp8  {layer:<4} m={m:<6} {tag:<5}: {ms:8.3} ms {t:6.1} TFLOPS (fused bias{})",
                    if epi == Epilogue::Gelu { "+gelu" } else { "" }
                );
                rows.insert(format!("gemm_fp8_{layer}_n{frames}_{tag}_tflops"), t);
                let ms = time_ms(&nn, reps, || {
                    let sc = Fp8Scales {
                        weight: WeightScale::Deferred,
                        act: None,
                    };
                    g.linear_fp8(&nn, &x8, &w8, &sc, None, Epilogue::None, &mut y, s)
                        .unwrap();
                });
                let t = s.flops() / (ms * 1e-3) / 1e12;
                println!(
                    "gemm fp8  {layer:<4} m={m:<6} {tag:<5}: {ms:8.3} ms {t:6.1} TFLOPS (deferred scales)"
                );
                rows.insert(
                    format!("gemm_fp8_deferred_{layer}_n{frames}_{tag}_tflops"),
                    t,
                );
            }
        }
    }
    println!("GPU after: {}", gpu_state());
    write_bench("gemm", &rows);
}

#[test]
#[ignore]
fn bench_memory_kernels() {
    let nn = nn();
    let norm = Norm::new(&nn).unwrap();
    let pro = QkvPrologue::new(&nn).unwrap();
    let res = Residual::new(&nn).unwrap();
    println!("GPU before: {}", gpu_state());
    let mut rng = Rng::new(12);
    let mut rows_out = BTreeMap::new();
    let frames = 64;
    let rows = frames * 689;
    let c = 1024;
    let x = up_f32(&nn, &rng.normals(rows * c, 1.0));
    let mut xm = up_f32(&nn, &rng.normals(rows * c, 1.0));
    let g = up_f32(&nn, &vec![1.0; c]);
    let b = up_f32(&nn, &vec![0.0; c]);
    let p = LnParams {
        gamma: &g,
        beta: &b,
        eps: 1e-6,
    };
    let mut q = zeros::<u8>(&nn, rows * c);
    let mut sc = zeros::<f32>(&nn, rows);
    let gbs = |bytes: f64, ms: f64| bytes / (ms * 1e-3) / 1e9;

    let ms = time_ms(&nn, 20, || {
        norm.layer_norm(
            &nn,
            Rows::F32(&x),
            rows,
            c,
            &p,
            None,
            Quant::PerToken {
                q: &mut q,
                scales: &mut sc,
            },
        )
        .unwrap();
    });
    let bytes = (rows * c * (4 + 1)) as f64;
    println!(
        "layer_norm f32 -> fp8 ({rows}x{c}): {ms:.3} ms, {:.0} GB/s",
        gbs(bytes, ms)
    );
    rows_out.insert("ln_f32_fp8_gbs".into(), gbs(bytes, ms));

    let y = nn
        .stream
        .clone_htod(&host::to_bf16(&rng.normals(rows * c, 1.0)))
        .unwrap();
    let ls = up_f32(&nn, &vec![0.1; c]);
    let yb = nn.stream.clone_htod(&host::to_bf16(&vec![0.0; c])).unwrap();
    let br = Branch {
        y: &y,
        tok: None,
        chan: None,
        bias: Some(&yb),
        layer_scale: Some(&ls),
    };
    let ms = time_ms(&nn, 20, || {
        norm.residual_layer_norm(
            &nn,
            RowsMut::F32(&mut xm),
            &br,
            rows,
            c,
            &p,
            None,
            Quant::PerToken {
                q: &mut q,
                scales: &mut sc,
            },
        )
        .unwrap();
    });
    let bytes = (rows * c * (4 + 2 + 4 + 1)) as f64;
    println!(
        "residual + layer_norm -> fp8: {ms:.3} ms, {:.0} GB/s",
        gbs(bytes, ms)
    );
    rows_out.insert("residual_ln_fp8_gbs".into(), gbs(bytes, ms));
    let ms_res = time_ms(&nn, 20, || {
        res.add(&nn, RowsMut::F32(&mut xm), &br, rows, c, c)
            .unwrap();
    });
    let bytes_res = (rows * c * (4 + 2 + 4)) as f64;
    println!(
        "residual alone: {ms_res:.3} ms, {:.0} GB/s",
        gbs(bytes_res, ms_res)
    );
    rows_out.insert("residual_gbs".into(), gbs(bytes_res, ms_res));

    let heads = 16;
    let yq = nn
        .stream
        .clone_htod(&host::to_bf16(&rng.normals(rows * 3 * c, 1.0)))
        .unwrap();
    let nw = up_f32(&nn, &vec![1.0; HEAD_DIM]);
    let nb = up_f32(&nn, &vec![0.0; HEAD_DIM]);
    let table = RopeTable::vggt(21, 32, 17, 100.0);
    let td = table.upload(&nn).unwrap();
    let n = rows * c;
    let (mut qq, mut kk, mut vv) = (
        zeros::<u16>(&nn, n),
        zeros::<u16>(&nn, n),
        zeros::<u16>(&nn, n),
    );
    let ms = time_ms(&nn, 20, || {
        pro.run(
            &nn,
            &QkvIn {
                y: &yq,
                ld: 3 * c,
                tok: None,
                chan: None,
                bias: None,
            },
            rows,
            heads,
            Some(&QkNorm {
                kind: NormKind::LayerNorm,
                q_weight: &nw,
                q_bias: Some(&nb),
                k_weight: &nw,
                k_bias: Some(&nb),
                eps: 1e-6,
            }),
            Some(&Rope {
                table: &td,
                layout: RopeLayout::Vggt,
                frame_len: 689,
            }),
            QkvOut {
                q: &mut qq,
                k: &mut kk,
                v: &mut vv,
                v_format: VFormat::Bf16,
            },
        )
        .unwrap();
    });
    let bytes = (rows * 3 * c * 4) as f64;
    println!("qkv prologue: {ms:.3} ms, {:.0} GB/s", gbs(bytes, ms));
    rows_out.insert("qkv_prologue_gbs".into(), gbs(bytes, ms));

    let mut h = nn
        .stream
        .clone_htod(&host::to_bf16(&rng.normals(rows * 4096, 1.0)))
        .unwrap();
    let hb = nn
        .stream
        .clone_htod(&host::to_bf16(&vec![0.0; 4096]))
        .unwrap();
    let mut hq = zeros::<u8>(&nn, rows * 4096);
    let tok = up_f32(&nn, &vec![0.01; rows]);
    let ms = time_ms(&nn, 20, || {
        norm.row_epilogue(
            &nn,
            &mut h,
            rows,
            4096,
            4096,
            Some(&tok),
            None,
            Some(&hb),
            Act::GeluErf,
            false,
            Quant::PerToken {
                q: &mut hq,
                scales: &mut sc,
            },
        )
        .unwrap();
    });
    let bytes = (rows * 4096 * 3) as f64;
    println!(
        "row epilogue (GELU erf -> fp8, 4096): {ms:.3} ms, {:.0} GB/s",
        gbs(bytes, ms)
    );
    rows_out.insert("row_epilogue_gelu_fp8_gbs".into(), gbs(bytes, ms));
    println!("GPU after: {}", gpu_state());
    write_bench("memory", &rows_out);
}
