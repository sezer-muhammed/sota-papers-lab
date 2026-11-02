//! GPU tests of [`crate::conv`] and [`crate::rowops`] (all `#[ignore]`, run
//! with `--ignored` on the GPU host like `tests.rs`), plus `bench_conv`:
//! CUDA-event timings of the VGGT-Ω dense-head convolutions.

use cudarc::driver::CudaSlice;
use cudarc::driver::sys::CUevent_flags;

use crate::conv::{Conv, ConvEpilogue, ConvShape, D2sOrder, conv2d_reference};
use crate::host;
use crate::rowops::{PackedQkv, RowOps};
use crate::testkit::{Err, Rng, err};
use crate::{Act, Nn};

fn nn() -> Nn {
    Nn::open().expect("open GPU")
}

fn up_bf16(nn: &Nn, v: &[f32]) -> (CudaSlice<u16>, Vec<f32>) {
    let b = host::to_bf16(v);
    (nn.stream.clone_htod(&b).unwrap(), host::from_bf16(&b))
}

fn down_bf16(nn: &Nn, d: &CudaSlice<u16>) -> Vec<f32> {
    host::from_bf16(&nn.stream.clone_dtoh(d).unwrap())
}

fn check(name: &str, e: Err, rel_rms: f64) {
    println!(
        "{name}: max_abs {:.3e} rms {:.3e} rel_rms {:.3e}",
        e.max_abs, e.rms, e.rel_rms
    );
    assert!(
        e.rel_rms <= rel_rms,
        "{name}: rel_rms {:.3e} > {rel_rms:.1e}",
        e.rel_rms
    );
}

fn conv_case(nn: &Nn, conv: &Conv, s: ConvShape, relu: bool, bias: bool, res: usize) {
    let mut rng = Rng::new(7 + s.cin as u64 + s.h as u64);
    let (x, xr) = up_bf16(nn, &rng.normals(s.n * s.h * s.w * s.cin, 1.0));
    let k = s.kh * s.kw * s.cin;
    let (w, wr) = up_bf16(nn, &rng.normals(s.cout * k, (1.0 / k as f32).sqrt()));
    let b: Vec<f32> = rng.normals(s.cout, 0.5);
    let bd = nn.stream.clone_htod(&b).unwrap();
    let m = s.pixels();
    let (r0, r0r) = up_bf16(nn, &rng.normals(m * s.cout, 1.0));
    let (r1, r1r) = up_bf16(nn, &rng.normals(m * s.cout, 1.0));
    let mut y = nn.stream.alloc_zeros::<u16>(m * s.cout).unwrap();
    let epi = ConvEpilogue {
        bias: bias.then_some(&bd as _),
        residual: [(res > 0).then_some(&r0 as _), (res > 1).then_some(&r1 as _)],
    };
    conv.conv2d(nn, &x, &w, &epi, relu, &mut y, s).unwrap();
    let got = down_bf16(nn, &y);
    let resv: Vec<&[f32]> = [&r0r[..], &r1r[..]].into_iter().take(res).collect();
    let want = conv2d_reference(&xr, &wr, bias.then_some(&b[..]), &resv, relu, s);
    check(
        &format!("conv2d {s:?} relu={relu} bias={bias} res={res}"),
        err(&got, &want),
        6e-3,
    );
}

#[test]
#[ignore]
fn gpu_conv2d_matches_cpu() {
    let nn = nn();
    let conv = Conv::new(&nn).unwrap();
    // Odd sizes (tile edges), stride 2, cout below one tile, relu + residuals.
    conv_case(
        &nn,
        &conv,
        ConvShape::same(2, 13, 21, 64, 256, 3, 1),
        true,
        true,
        2,
    );
    conv_case(
        &nn,
        &conv,
        ConvShape::same(1, 21, 32, 128, 128, 3, 2),
        false,
        true,
        0,
    );
    conv_case(
        &nn,
        &conv,
        ConvShape::same(1, 9, 7, 32, 72, 3, 1),
        false,
        false,
        1,
    );
    conv_case(
        &nn,
        &conv,
        ConvShape::same(3, 6, 5, 96, 40, 1, 1),
        true,
        false,
        0,
    );
}

#[test]
#[ignore]
fn gpu_depth_to_space_upsample_table() {
    let nn = nn();
    let conv = Conv::new(&nn).unwrap();
    let mut rng = Rng::new(3);
    let (n, h, w, c, s) = (2usize, 3usize, 5usize, 16usize, 4usize);
    let (x, xr) = up_bf16(&nn, &rng.normals(n * h * w * s * s * c, 1.0));
    let bias = rng.normals(c, 1.0);
    let bd = nn.stream.clone_htod(&bias).unwrap();
    for order in [D2sOrder::TapMajor, D2sOrder::ChannelMajor] {
        let mut y = nn.stream.alloc_zeros::<u16>(xr.len()).unwrap();
        conv.depth_to_space(&nn, &x, Some(&bd), order, &mut y, n, h, w, c, s)
            .unwrap();
        let got = down_bf16(&nn, &y);
        let mut want = vec![0f64; xr.len()];
        for ni in 0..n {
            for yy in 0..h * s {
                for xx in 0..w * s {
                    let (y0, i, x0, j) = (yy / s, yy % s, xx / s, xx % s);
                    for o in 0..c {
                        let ch = match order {
                            D2sOrder::TapMajor => (i * s + j) * c + o,
                            D2sOrder::ChannelMajor => o * s * s + i * s + j,
                        };
                        let src = ((ni * h + y0) * w + x0) * s * s * c + ch;
                        want[((ni * h * s + yy) * w * s + xx) * c + o] =
                            xr[src] as f64 + bias[o] as f64;
                    }
                }
            }
        }
        check(&format!("depth_to_space {order:?}"), err(&got, &want), 4e-3);
    }

    let (h, w, c, ho, wo) = (11usize, 16usize, 24usize, 21usize, 32usize);
    let (x, xr) = up_bf16(&nn, &rng.normals(n * h * w * c, 1.0));
    let mut y = nn.stream.alloc_zeros::<u16>(n * ho * wo * c).unwrap();
    conv.upsample_bilinear(&nn, &x, &mut y, n, h, w, c, ho, wo)
        .unwrap();
    let got = down_bf16(&nn, &y);
    let mut want = vec![0f64; got.len()];
    for ni in 0..n {
        for oy in 0..ho {
            let fy = oy as f64 * (h - 1) as f64 / (ho - 1) as f64;
            let y0 = (fy as usize).min(h - 1);
            let y1 = (y0 + 1).min(h - 1);
            let ly = fy - y0 as f64;
            for ox in 0..wo {
                let fx = ox as f64 * (w - 1) as f64 / (wo - 1) as f64;
                let x0 = (fx as usize).min(w - 1);
                let x1 = (x0 + 1).min(w - 1);
                let lx = fx - x0 as f64;
                for ch in 0..c {
                    let at = |yy: usize, xx: usize| xr[((ni * h + yy) * w + xx) * c + ch] as f64;
                    want[((ni * ho + oy) * wo + ox) * c + ch] = (1.0 - ly)
                        * ((1.0 - lx) * at(y0, x0) + lx * at(y0, x1))
                        + ly * ((1.0 - lx) * at(y1, x0) + lx * at(y1, x1));
                }
            }
        }
    }
    check("upsample_bilinear_ac", err(&got, &want), 4e-3);

    let (frames, rows, start, p, c) = (3usize, 23usize, 17usize, 6usize, 32usize);
    let (x, xr) = up_bf16(&nn, &rng.normals(frames * rows * c, 1.0));
    let tab = rng.normals(p * c, 1.0);
    let td = nn.stream.clone_htod(&tab).unwrap();
    let mut y = nn.stream.alloc_zeros::<u16>(frames * p * c).unwrap();
    conv.add_table(&nn, &x, Some(&td), 0.1, &mut y, frames, rows, start, p, c)
        .unwrap();
    let got = down_bf16(&nn, &y);
    let mut want = vec![0f64; got.len()];
    for f in 0..frames {
        for pi in 0..p {
            for ch in 0..c {
                want[(f * p + pi) * c + ch] =
                    xr[(f * rows + start + pi) * c + ch] as f64 + 0.1 * tab[pi * c + ch] as f64;
            }
        }
    }
    check("add_table gather", err(&got, &want), 4e-3);
    conv.add_table_inplace(&nn, &mut y, &td, -0.1, frames, p, c)
        .unwrap();
    let back = down_bf16(&nn, &y);
    let want: Vec<f64> = (0..frames * p * c)
        .map(|i| xr[(i / (p * c)) * rows * c + (start + (i / c) % p) * c + i % c] as f64)
        .collect();
    check("add_table in place", err(&back, &want), 6e-3);
}

#[test]
#[ignore]
fn gpu_rowops_match_cpu() {
    let nn = nn();
    let ops = RowOps::new(&nn).unwrap();
    let mut rng = Rng::new(11);
    let (mut x, xr) = up_bf16(&nn, &rng.normals(8192 * 3, 2.0));
    ops.activate(&nn, &mut x, xr.len(), Act::GeluErf).unwrap();
    let want: Vec<f64> = xr.iter().map(|&v| host::gelu_erf(v as f64)).collect();
    check("activate gelu_erf", err(&down_bf16(&nn, &x), &want), 4e-3);

    for seq in [17usize, 34, 136, 545] {
        let (heads, c) = (16usize, 2048usize);
        let (qkv, qr) = up_bf16(&nn, &rng.normals(seq * 3 * c, 1.0));
        let mut out = nn.stream.alloc_zeros::<u16>(seq * c).unwrap();
        let scale = 1.0 / (128f32).sqrt();
        ops.attention_d128(
            &nn,
            &qkv,
            PackedQkv::thirds(c),
            seq,
            heads,
            scale,
            &mut out,
            c,
        )
        .unwrap();
        // [heads][seq][128] copies for the reference.
        let split = |off: usize| -> Vec<f32> {
            let mut v = vec![0f32; seq * c];
            for h in 0..heads {
                for t in 0..seq {
                    for d in 0..128 {
                        v[(h * seq + t) * 128 + d] = qr[t * 3 * c + off + h * 128 + d];
                    }
                }
            }
            v
        };
        let want = host::attention(
            &split(0),
            &split(c),
            &split(2 * c),
            heads,
            128,
            scale as f64,
        );
        check(
            &format!("attention_d128 seq {seq}"),
            err(&down_bf16(&nn, &out), &want),
            6e-3,
        );
    }
}

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

/// Dense-head 3×3 convolutions for an 8-frame chunk at 512×336 (h×w = 21×32).
#[test]
#[ignore]
fn bench_conv() {
    let nn = nn();
    let conv = Conv::new(&nn).unwrap();
    let ops = RowOps::new(&nn).unwrap();
    let (h, w) = (21usize, 32usize);
    let cases = [
        (
            "rcu 256 @4h",
            ConvShape::same(8, 4 * h, 4 * w, 256, 256, 3, 1),
        ),
        (
            "rcu 256 @2h",
            ConvShape::same(8, 2 * h, 2 * w, 256, 256, 3, 1),
        ),
        (
            "rn 512->256 @2h",
            ConvShape::same(8, 2 * h, 2 * w, 512, 256, 3, 1),
        ),
        ("rn 1024->256 @h", ConvShape::same(8, h, w, 1024, 256, 3, 1)),
        (
            "resize 1024 s2 @h",
            ConvShape::same(8, h, w, 1024, 1024, 3, 2),
        ),
    ];
    for (name, s) in cases {
        let x = nn
            .stream
            .alloc_zeros::<u16>(s.n * s.h * s.w * s.cin)
            .unwrap();
        let wt = nn.stream.alloc_zeros::<u16>(s.cout * 9 * s.cin).unwrap();
        let mut y = nn.stream.alloc_zeros::<u16>(s.pixels() * s.cout).unwrap();
        let ms = time_ms(&nn, 20, || {
            conv.conv2d(&nn, &x, &wt, &ConvEpilogue::default(), true, &mut y, s)
                .unwrap()
        });
        println!(
            "bench_conv {name}: {ms:.3} ms, {:.1} TFLOPS",
            s.flops() / ms / 1e9
        );
    }
    for n in [8usize, 32, 200] {
        let seq = 17 * n;
        let qkv = nn.stream.alloc_zeros::<u16>(seq * 6144).unwrap();
        let mut out = nn.stream.alloc_zeros::<u16>(seq * 2048).unwrap();
        let ms = time_ms(&nn, 10, || {
            ops.attention_d128(
                &nn,
                &qkv,
                PackedQkv::thirds(2048),
                seq,
                16,
                0.088,
                &mut out,
                2048,
            )
            .unwrap()
        });
        let fl = 4.0 * (seq * seq) as f64 * 2048.0;
        println!(
            "bench attention_d128 N={n} seq {seq}: {ms:.3} ms, {:.2} TFLOPS",
            fl / ms / 1e9
        );
    }
}
