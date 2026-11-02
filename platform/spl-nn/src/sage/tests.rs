//! Sage attention tests. CPU tests run everywhere; the GPU ones are
//! `#[ignore]`d: on the GPU host run `cargo test --release -p spl-nn sage::
//! -- --ignored` inside the CUDA container. `gpu_bench` writes timings to
//! `$SAGE_OUT` (default /out/sage) and, with `SAGE_RUN=<dir>`, runs on the
//! Q/K/V files `torch_ref.py gen` wrote there (see that script).

use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use super::reference::*;
use super::*;

// ---------------------------------------------------------------- CPU tests

#[test]
fn engine_is_send_and_sync() {
    fn check<T: Send + Sync>() {}
    check::<SageAttention>();
    check::<Workspace>();
}

#[test]
fn e4m3_codes_round_trip() {
    for b in 0..=255u8 {
        if b & 0x7f == 0x7f {
            assert!(e4m3_to_f32(b).is_nan());
            continue;
        }
        let x = e4m3_to_f32(b);
        assert_eq!(f32_to_e4m3(x), b, "code {b:#04x} = {x}");
    }
    assert_eq!(e4m3_to_f32(0x7e), 448.0);
    assert_eq!(e4m3_to_f32(0x01), (2.0f32).powi(-9));
    assert_eq!(e4m3_to_f32(0x38), 1.0);
}

#[test]
fn e4m3_rounds_to_nearest_even_and_saturates() {
    // 1.0625 lies halfway between 1.0 (even) and 1.125.
    assert_eq!(e4m3_to_f32(f32_to_e4m3(1.0625)), 1.0);
    assert_eq!(e4m3_to_f32(f32_to_e4m3(1.1875)), 1.25);
    assert_eq!(e4m3_to_f32(f32_to_e4m3(-3.3)), -3.25);
    assert_eq!(f32_to_e4m3(1000.0), 0x7e);
    assert_eq!(f32_to_e4m3(-1000.0), 0xfe);
    // Subnormals: 2^-10 ties to 0, 3 * 2^-10 ties up to 2 * 2^-9.
    assert_eq!(f32_to_e4m3((2.0f32).powi(-10)), 0);
    assert_eq!(
        e4m3_to_f32(f32_to_e4m3(3.0 * (2.0f32).powi(-10))),
        (2.0f32).powi(-8)
    );
    // Mantissa carry into the next binade.
    assert_eq!(e4m3_to_f32(f32_to_e4m3(15.9)), 16.0);
}

#[test]
fn bf16_rounds_to_nearest_even() {
    assert_eq!(f32_to_bf16(1.0), 0x3f80);
    assert_eq!(bf16_to_f32(f32_to_bf16(1.0 + 1.0 / 256.0)), 1.0);
    assert_eq!(
        bf16_to_f32(f32_to_bf16(1.0 + 3.0 / 256.0)),
        1.0 + 4.0 / 256.0
    );
    assert_eq!(bf16_to_f32(f32_to_bf16(-2.5)), -2.5);
}

#[test]
fn v_permutation_matches_the_p_fragment() {
    // Thread t of a quad holds S columns {2t, 2t+1} of n-tiles 0 and 1; packed
    // into the FP8 A fragment they become k = 4t..4t+3.
    let mut seen = [false; 16];
    for t in 0..4 {
        for (e, key) in [2 * t, 2 * t + 1, 8 + 2 * t, 9 + 2 * t]
            .into_iter()
            .enumerate()
        {
            assert_eq!(vpos(key), 4 * t + e);
            seen[vpos(key)] = true;
        }
    }
    assert!(seen.iter().all(|&s| s));
}

fn vggt_layout(frames: usize) -> FrameLayout {
    FrameLayout {
        frames,
        special: 17,
        grid_h: 21,
        grid_w: 32,
        dense_frames: 0,
    }
}

#[test]
fn dense_first_frame_keeps_its_keys() {
    let layout = FrameLayout {
        dense_frames: 1,
        ..vggt_layout(3)
    };
    let plan = Plan::new(
        Shape {
            batch: 1,
            heads: 1,
            tokens: 3 * 689,
        },
        KvMerge::Pool { stride: 4, layout },
    )
    .unwrap();
    // 689 dense + 2 · (17 + 6 · 8) pooled.
    assert_eq!(plan.tk, 689 + 2 * 65);
    for j in 0..689 {
        assert_eq!(reference::members(&plan, j), vec![j]);
    }
    for j in 689..689 + 17 {
        assert_eq!(reference::members(&plan, j), vec![j]);
    }
    // The first pooled cell of frame 1.
    assert_eq!(
        reference::members(&plan, 689 + 17),
        vec![
            706, 707, 708, 709, 738, 739, 740, 741, 770, 771, 772, 773, 802, 803, 804, 805
        ]
    );
    let mut seen = vec![false; 3 * 689];
    for j in 0..plan.tk {
        for t in reference::members(&plan, j) {
            assert!(!seen[t], "token {t} pooled twice");
            seen[t] = true;
        }
    }
    assert!(seen.iter().all(|&s| s));
}

#[test]
fn vggt_merge_geometry() {
    let layout = vggt_layout(200);
    assert_eq!(layout.tokens_per_frame(), 689);
    let shape = Shape {
        batch: 1,
        heads: 16,
        tokens: 200 * 689,
    };
    let dense = Plan::new(shape, KvMerge::Dense).unwrap();
    assert_eq!(
        (dense.tq_pad, dense.tk, dense.tk_pad),
        (137856, 137800, 137856)
    );
    let m2 = Plan::new(shape, KvMerge::Pool { stride: 2, layout }).unwrap();
    assert_eq!(m2.tk, 200 * (17 + 11 * 16));
    let m4 = Plan::new(shape, KvMerge::Pool { stride: 4, layout }).unwrap();
    assert_eq!(m4.tk, 200 * (17 + 6 * 8));
    for plan in [m2, m4] {
        // One frame's merged keys cover each of its tokens exactly once.
        let per = layout.merged_per_frame(match plan.merge {
            KvMerge::Pool { stride, .. } => stride,
            KvMerge::Dense => 1,
        });
        let mut hit = vec![0u32; 689];
        for j in per..2 * per {
            for t in members(&plan, j) {
                hit[t - 689] += 1;
            }
        }
        assert!(hit.iter().all(|&h| h == 1));
        // Special tokens stay single; the last grid row pools partial cells.
        assert_eq!(members(&plan, per + 3), vec![689 + 3]);
    }
    assert_eq!(members(&m4, 17).len(), 16);
    assert_eq!(members(&m4, 17 + 5 * 8).len(), 4); // row 20 only
    assert_eq!(members(&m2, 17 + 10 * 16).len(), 2);
}

#[test]
fn plan_rejects_bad_layouts() {
    let shape = Shape {
        batch: 1,
        heads: 2,
        tokens: 1000,
    };
    let layout = vggt_layout(2);
    assert!(Plan::new(shape, KvMerge::Pool { stride: 2, layout }).is_err());
    let shape = Shape {
        tokens: 2 * 689,
        ..shape
    };
    assert!(Plan::new(shape, KvMerge::Pool { stride: 3, layout }).is_err());
    assert!(Plan::new(shape, KvMerge::Pool { stride: 4, layout }).is_ok());
    assert!(Plan::new(Shape { tokens: 0, ..shape }, KvMerge::Dense).is_err());
}

#[test]
fn merged_attention_equals_dense_on_uniform_cells() {
    // If every pooled cell holds identical keys and values, attention over
    // the pooled keys with the +log(count) bias is exactly dense attention.
    let layout = FrameLayout {
        frames: 2,
        special: 3,
        grid_h: 5,
        grid_w: 7,
        dense_frames: 0,
    };
    let shape = Shape {
        batch: 1,
        heads: 1,
        tokens: 2 * layout.tokens_per_frame(),
    };
    let d = 16;
    for stride in [2, 4] {
        let plan = Plan::new(shape, KvMerge::Pool { stride, layout }).unwrap();
        let mut rng = Rng::new(7);
        let mut cell_k = vec![0f32; plan.tk * d];
        let mut cell_v = vec![0f32; plan.tk * d];
        for x in cell_k.iter_mut().chain(cell_v.iter_mut()) {
            *x = rng.normal();
        }
        let mut k = vec![0f32; shape.tokens * d];
        let mut v = vec![0f32; shape.tokens * d];
        for j in 0..plan.tk {
            for t in members(&plan, j) {
                k[t * d..(t + 1) * d].copy_from_slice(&cell_k[j * d..(j + 1) * d]);
                v[t * d..(t + 1) * d].copy_from_slice(&cell_v[j * d..(j + 1) * d]);
            }
        }
        let q: Vec<f32> = (0..shape.tokens * d).map(|_| 2.0 * rng.normal()).collect();
        let dense = attention(&q, &k, &v, None, d, 0.25);
        let (km, vm, bias) = merge_kv(&plan, &k, &v, d);
        let merged = attention(&q, &km, &vm, Some(&bias), d, 0.25);
        let e = Errors::of(&merged, &dense);
        assert!(e.max_abs < 1e-5, "stride {stride}: {e:?}");
    }
}

#[test]
fn geom_matches_the_cuda_struct() {
    use std::mem::{offset_of, size_of};
    // SgGeom: 12 i64, 16 i32, 3 f32, 1 i32 padding.
    assert_eq!(size_of::<Geom>(), 12 * 8 + 16 * 4 + 4 * 4);
    assert_eq!(offset_of!(Geom, batch), 96);
    assert_eq!(offset_of!(Geom, flags), 96 + 15 * 4);
    assert_eq!(offset_of!(Geom, q_mul), 96 + 16 * 4);
    assert_eq!(offset_of!(Geom, v_range), 96 + 18 * 4);
}

#[test]
fn strides_cover_the_tensor() {
    let shape = Shape {
        batch: 2,
        heads: 3,
        tokens: 5,
    };
    let d = 64;
    assert_eq!(Strides::bthd(5, 3, d).extent(&shape, d), 2 * 5 * 3 * d);
    assert_eq!(Strides::bhtd(5, 3, d).extent(&shape, d), 2 * 3 * 5 * d);
    // A packed QKV view starting at V's offset ends at the buffer's end.
    assert_eq!(
        Strides::packed_qkv(5, 3, d).extent(&shape, d) + 2 * 3 * d,
        2 * 5 * 3 * 3 * d
    );
    assert!(Strides::bthd(5, 3, 60).check("x").is_err());
}

#[test]
fn sources_are_self_contained() {
    for src in [&*SOURCE, &*PEAK_SOURCE] {
        assert!(!src.contains("#include"));
    }
    for k in [
        "sg_prep",
        "sg_finalize",
        "sg_quant_kv",
        "sg_attn_dense",
        "sg_attn_merge",
    ] {
        assert!(SOURCE.contains(&format!("{k}(")), "{k}");
    }
    // The CUDA mirrors of the Rust constants.
    assert!(SOURCE.contains(&format!("#define SG_QGROUP {QGROUP}")));
    assert!(SOURCE.contains(&format!("#define SG_QTILE {QTILE}")));
    assert!(SOURCE.contains(&format!("#define SG_KBLOCK {KBLOCK}")));
    assert!(SOURCE.contains(&format!("#define SG_LN_Q {LN_Q}")));
    assert!(SOURCE.contains(&format!("#define SG_SMOOTH_V {SMOOTH_V}")));
}

// ---------------------------------------------------------------- test data

/// Synthetic activations shaped like a global attention layer's: clustered
/// tokens (so attention is peaked, not uniform), per-channel K and V offsets
/// (what smoothing removes) and a few outlier channels.
pub(crate) struct Data {
    pub shape: Shape,
    pub d: usize,
    /// `[b][t][h][d]` each.
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

impl Data {
    pub fn synthetic(shape: Shape, d: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let n = shape.batch * shape.tokens * shape.heads * d;
        let clusters = 37;
        let centers: Vec<f32> = (0..clusters * shape.heads * d)
            .map(|_| rng.normal())
            .collect();
        let k_off: Vec<f32> = (0..shape.heads * d).map(|_| 3.0 * rng.normal()).collect();
        let v_off: Vec<f32> = (0..shape.heads * d).map(|_| 2.0 * rng.normal()).collect();
        let (mut q, mut k, mut v) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
        for b in 0..shape.batch {
            for t in 0..shape.tokens {
                let c = (rng.next_u64() % clusters as u64) as usize;
                for h in 0..shape.heads {
                    for x in 0..d {
                        let i = ((b * shape.tokens + t) * shape.heads + h) * d + x;
                        let ctr = centers[(c * shape.heads + h) * d + x];
                        let outlier = if x % 23 == 5 { 6.0 } else { 1.0 };
                        q[i] = 1.6 * ctr + 0.6 * rng.normal();
                        k[i] = (1.6 * ctr + 0.6 * rng.normal()) * outlier + k_off[h * d + x];
                        v[i] = rng.normal() + 0.5 * ctr + v_off[h * d + x];
                    }
                }
            }
        }
        for x in q.iter_mut().chain(k.iter_mut()).chain(v.iter_mut()) {
            *x = bf16_to_f32(f32_to_bf16(*x));
        }
        Self { shape, d, q, k, v }
    }

    /// One (batch, head) slice `[t][d]` of `x`.
    pub fn head(&self, x: &[f32], b: usize, h: usize) -> Vec<f32> {
        let (t, hs, d) = (self.shape.tokens, self.shape.heads, self.d);
        let mut out = Vec::with_capacity(t * d);
        for tok in 0..t {
            let i = ((b * t + tok) * hs + h) * d;
            out.extend_from_slice(&x[i..i + d]);
        }
        out
    }

    /// Q, K and V packed as one `[b][t][3][h][d]` bf16 buffer.
    pub fn packed(&self) -> Vec<u16> {
        let (bt, hd) = (
            self.shape.batch * self.shape.tokens,
            self.shape.heads * self.d,
        );
        let mut out = vec![0u16; bt * 3 * hd];
        for r in 0..bt {
            for (s, src) in [&self.q, &self.k, &self.v].into_iter().enumerate() {
                for c in 0..hd {
                    out[(r * 3 + s) * hd + c] = f32_to_bf16(src[r * hd + c]);
                }
            }
        }
        out
    }
}

// ---------------------------------------------------------------- GPU tests

fn stream() -> Arc<CudaStream> {
    let ctx = spl_gpu::context(0).expect("CUDA device");
    ctx.new_stream().expect("stream")
}

struct Run {
    out: Vec<f32>,
    ws: Workspace,
}

fn run_gpu(
    sage: &SageAttention,
    data: &Data,
    merge: KvMerge,
    ln: Option<&CudaSlice<f32>>,
    smooth: bool,
) -> Run {
    let s = sage.stream();
    let (shape, d) = (data.shape, data.d);
    let packed = s.clone_htod(&data.packed()).unwrap();
    let hd = shape.heads * d;
    let strides = Strides::packed_qkv(shape.tokens, shape.heads, d);
    let mut ws = sage.workspace(shape, merge).unwrap();
    let mut out = s
        .alloc_zeros::<u16>(shape.batch * shape.tokens * hd)
        .unwrap();
    let opts = Options {
        smooth_k: smooth,
        smooth_v: smooth,
        qk_norm: ln.map(|p| QkNorm {
            params: p,
            eps: 1e-5,
        }),
        ..Default::default()
    };
    let (qv, kv, vv) = (
        packed.slice(0..),
        packed.slice(hd..),
        packed.slice(2 * hd..),
    );
    sage.forward(
        &mut ws,
        Tensor { data: &qv, strides },
        Tensor { data: &kv, strides },
        Tensor { data: &vv, strides },
        TensorMut {
            data: &mut out,
            strides: Strides::bthd(shape.tokens, shape.heads, d),
        },
        &opts,
    )
    .unwrap();
    let out = s.clone_dtoh(&out).unwrap();
    s.synchronize().unwrap();
    Run {
        out: out.into_iter().map(bf16_to_f32).collect(),
        ws,
    }
}

fn ln_params(d: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let mut p = Vec::with_capacity(4 * d);
    for part in 0..4 {
        for _ in 0..d {
            p.push(if part % 2 == 0 {
                1.0 + 0.5 * rng.normal()
            } else {
                0.8 * rng.normal()
            });
        }
    }
    p
}

/// One (batch, head) worked on the CPU: the f64 attention output, Q and K
/// after the optional LayerNorm, and the merged K and V.
struct RefHead {
    out: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    km: Vec<f32>,
    vm: Vec<f32>,
}

/// The f64 reference for one (batch, head) with the same merge and norm.
fn reference_head(data: &Data, plan: &Plan, ln: Option<&[f32]>, b: usize, h: usize) -> RefHead {
    let d = data.d;
    let (mut q, mut k, v) = (
        data.head(&data.q, b, h),
        data.head(&data.k, b, h),
        data.head(&data.v, b, h),
    );
    if let Some(p) = ln {
        for row in q.chunks_mut(d) {
            layer_norm(row, &p[..d], &p[d..2 * d], 1e-5);
        }
        for row in k.chunks_mut(d) {
            layer_norm(row, &p[2 * d..3 * d], &p[3 * d..], 1e-5);
        }
    }
    let scale = 1.0 / (d as f32).sqrt();
    let (km, vm, bias) = merge_kv(plan, &k, &v, d);
    let bias = matches!(plan.merge, KvMerge::Pool { .. }).then_some(bias);
    let out = attention(&q, &km, &vm, bias.as_deref(), d, scale);
    RefHead { out, q, k, km, vm }
}

fn out_head(run: &Run, data: &Data, b: usize, h: usize) -> Vec<f32> {
    data.head(&run.out, b, h)
}

/// Checks the prologue's buffers against the CPU: scales, quantisation
/// error within half a step, the V layout and the merge bias.
fn check_prologue(sage: &SageAttention, data: &Data, run: &Run, ln: Option<&[f32]>, smooth: bool) {
    let s = sage.stream();
    let ws = &run.ws;
    let plan = *ws.plan();
    let d = data.d;
    let q8 = s.clone_dtoh(&ws.q8).unwrap();
    let qsc = s.clone_dtoh(&ws.q_scale).unwrap();
    let k8 = s.clone_dtoh(&ws.k8).unwrap();
    let ksc = s.clone_dtoh(&ws.k_scale).unwrap();
    let v8 = s.clone_dtoh(&ws.v8).unwrap();
    let stats = s.clone_dtoh(&ws.stats).unwrap();
    let kbias = s.clone_dtoh(&ws.kbias).unwrap();
    let qmul = std::f32::consts::LOG2_E / (d as f32).sqrt();
    let nkb = plan.tk_pad / KBLOCK;
    for b in 0..plan.shape.batch {
        for h in 0..plan.shape.heads {
            let bh = b * plan.shape.heads + h;
            let RefHead {
                q, k: kn, km, vm, ..
            } = reference_head(data, &plan, ln, b, h);
            // Q: one scale per 16 rows; rows past the end are zero.
            for g in 0..plan.tq_pad / QGROUP {
                let rows = (g * QGROUP..(g + 1) * QGROUP).filter(|&r| r < plan.shape.tokens);
                let amax = rows
                    .clone()
                    .flat_map(|r| q[r * d..(r + 1) * d].iter())
                    .fold(0f32, |m, &x| m.max(x.abs()));
                let got = qsc[bh * plan.tq_pad / QGROUP + g];
                let want = amax / 127.0 * qmul;
                assert!(
                    (got - want).abs() <= 1e-4 * want.abs() + 1e-30,
                    "q scale {got} vs {want}"
                );
                let step = amax / 127.0;
                for r in g * QGROUP..(g + 1) * QGROUP {
                    for c in 0..d {
                        let qv = q8[(bh * plan.tq_pad + r) * d + c] as f32 * step;
                        let x = if r < plan.shape.tokens {
                            q[r * d + c]
                        } else {
                            0.0
                        };
                        assert!(
                            (qv - x).abs() <= 0.5 * step * 1.01 + 1e-3 * step,
                            "q {r} {c}: {qv} vs {x}"
                        );
                    }
                }
            }
            // K: smoothed with the GPU's mean (any constant is exact), one
            // scale per 64 keys.
            let st = &stats[bh * 4 * d..(bh + 1) * 4 * d];
            let ks: Vec<f32> = km
                .chunks(d)
                .flat_map(|row| row.iter().enumerate().map(|(c, &x)| x - st[c]))
                .collect();
            if smooth {
                for c in 0..d {
                    let m =
                        kn.chunks(d).map(|r| r[c] as f64).sum::<f64>() / plan.shape.tokens as f64;
                    assert!(
                        (st[c] as f64 - m).abs() < 1e-3 * (1.0 + m.abs()),
                        "k mean {c}: {} vs {m}",
                        st[c]
                    );
                }
            } else {
                assert!(st[..2 * d].iter().all(|&x| x == 0.0));
            }
            for blk in 0..nkb {
                let rows = blk * KBLOCK..((blk + 1) * KBLOCK).min(plan.tk);
                let amax = rows
                    .clone()
                    .flat_map(|r| ks[r * d..(r + 1) * d].iter())
                    .fold(0f32, |m, &x| m.max(x.abs()));
                let got = ksc[bh * nkb + blk];
                assert!(
                    (got - amax / 127.0).abs() <= 1e-4 * amax / 127.0 + 1e-30,
                    "k scale"
                );
                for r in rows {
                    for c in 0..d {
                        let kq = k8[(bh * plan.tk_pad + r) * d + c] as f32 * got;
                        let x = ks[r * d + c];
                        assert!(
                            (kq - x).abs() <= 0.5 * got * 1.01 + 1e-4 * amax,
                            "k {r} {c}: {kq} vs {x}"
                        );
                    }
                }
            }
            // V: per-channel scale, transposed per 64-key block, keys permuted.
            for j in 0..plan.tk {
                let (blk, kl) = (j / KBLOCK, j % KBLOCK);
                let col = (kl & !15) + vpos(kl & 15);
                for c in 0..d {
                    let byte = v8[((bh * nkb + blk) * d + c) * KBLOCK + col];
                    let got = e4m3_to_f32(byte) * st[2 * d + c] + st[d + c];
                    let want = vm[j * d + c];
                    let tol = (want - st[d + c]).abs() / 16.0 + st[2 * d + c] * (2.0f32).powi(-8);
                    assert!(
                        (got - want).abs() <= tol * 1.01,
                        "v {j} {c}: {got} vs {want}"
                    );
                }
            }
        }
    }
    if let KvMerge::Pool { .. } = plan.merge {
        for (j, &b) in kbias.iter().enumerate() {
            if j < plan.tk {
                let want = (members(&plan, j).len() as f32).log2();
                assert!((b - want).abs() <= 1e-6, "bias {j}: {b} vs {want}");
            } else {
                assert_eq!(b, f32::NEG_INFINITY);
            }
        }
    }
}

fn small_layout() -> FrameLayout {
    FrameLayout {
        frames: 3,
        special: 5,
        grid_h: 7,
        grid_w: 10,
        // The reference frame stays dense, as in the VGGT-Ω schedule.
        dense_frames: 1,
    }
}

#[test]
#[ignore = "needs a CUDA GPU (compute 8.9+)"]
fn gpu_prologue_matches_reference() {
    let s = stream();
    for d in [64, 128] {
        let sage = SageAttention::new(s.clone(), d).unwrap();
        let layout = small_layout();
        let shape = Shape {
            batch: 2,
            heads: 3,
            tokens: layout.frames * layout.tokens_per_frame(),
        };
        let data = Data::synthetic(shape, d, 11);
        let lnp = ln_params(d, 3);
        let ln_dev = s.clone_htod(&lnp).unwrap();
        for (merge, ln, smooth) in [
            (KvMerge::Dense, false, true),
            (KvMerge::Dense, false, false),
            (KvMerge::Dense, true, true),
            (KvMerge::Pool { stride: 2, layout }, false, true),
            (KvMerge::Pool { stride: 4, layout }, true, true),
        ] {
            let run = run_gpu(&sage, &data, merge, ln.then_some(&ln_dev), smooth);
            check_prologue(&sage, &data, &run, ln.then_some(&lnp[..]), smooth);
            println!("prologue ok: d {d} {merge:?} ln {ln} smooth {smooth}");
        }
        // FP16 PV accumulation scales V to ±2.25 instead of ±448.
        let config = Config {
            pv_accum: PvAccum::F16,
            ..Config::for_head_dim(d)
        };
        let sage16 = SageAttention::with_config(s.clone(), d, config).unwrap();
        let run = run_gpu(&sage16, &data, KvMerge::Dense, None, true);
        check_prologue(&sage16, &data, &run, None, true);
        let stats = s.clone_dtoh(&run.ws.stats).unwrap();
        let v8 = s.clone_dtoh(&run.ws.v8).unwrap();
        let vmax = v8
            .iter()
            .map(|&b| e4m3_to_f32(b).abs())
            .fold(0f32, f32::max);
        assert!(vmax <= PV16_V_RANGE, "V reaches {vmax}");
        assert!(stats[2 * d] > 0.0);
        println!("prologue ok: d {d} PV16");
    }
}

#[test]
#[ignore = "needs a CUDA GPU (compute 8.9+)"]
fn gpu_attention_matches_float_reference() {
    let s = stream();
    let layout = small_layout();
    for (d, pv_accum) in [
        (64, PvAccum::F32),
        (64, PvAccum::F16),
        (128, PvAccum::F32),
        (128, PvAccum::F16),
    ] {
        let config = Config {
            pv_accum,
            ..Config::for_head_dim(d)
        };
        let sage = SageAttention::with_config(s.clone(), d, config).unwrap();
        let (regs, spill) = sage.attn_resources().unwrap();
        println!("d {d} {config:?}: {regs} registers, {spill} spill bytes");
        assert_eq!(spill, 0, "the attention kernel spills");
        let shape = Shape {
            batch: 2,
            heads: 3,
            tokens: layout.frames * layout.tokens_per_frame(),
        };
        let data = Data::synthetic(shape, d, 5);
        let lnp = ln_params(d, 9);
        let ln_dev = s.clone_htod(&lnp).unwrap();
        for (merge, ln, smooth) in [
            (KvMerge::Dense, false, true),
            (KvMerge::Dense, false, false),
            (KvMerge::Dense, true, true),
            (KvMerge::Pool { stride: 2, layout }, false, true),
            (KvMerge::Pool { stride: 4, layout }, true, true),
        ] {
            let run = run_gpu(&sage, &data, merge, ln.then_some(&ln_dev), smooth);
            let plan = *run.ws.plan();
            let (mut ours, mut want) = (Vec::new(), Vec::new());
            for b in 0..shape.batch {
                for h in 0..shape.heads {
                    let r = reference_head(&data, &plan, ln.then_some(&lnp[..]), b, h);
                    want.extend(r.out);
                    ours.extend(out_head(&run, &data, b, h));
                }
            }
            let e = Errors::of(&ours, &want);
            println!("d {d} {pv_accum:?} {merge:?} ln {ln} smooth {smooth}: {e:?}");
            assert!(e.cos > 0.999 && e.rel_l2 < 0.04, "{e:?}");
        }
    }
}

#[test]
#[ignore = "needs a CUDA GPU (compute 8.9+)"]
fn gpu_long_sequence_single_head() {
    // A sequence long enough for many key blocks and every pipeline stage,
    // with a ragged tail on both queries and keys.
    let s = stream();
    let d = 64;
    let sage = SageAttention::new(s.clone(), d).unwrap();
    let shape = Shape {
        batch: 1,
        heads: 2,
        tokens: 3001,
    };
    let data = Data::synthetic(shape, d, 21);
    let run = run_gpu(&sage, &data, KvMerge::Dense, None, true);
    let plan = *run.ws.plan();
    let (mut ours, mut want) = (Vec::new(), Vec::new());
    for h in 0..shape.heads {
        let r = reference_head(&data, &plan, None, 0, h);
        want.extend(r.out);
        ours.extend(out_head(&run, &data, 0, h));
    }
    let e = Errors::of(&ours, &want);
    println!("T 3001: {e:?}");
    assert!(e.cos > 0.999 && e.rel_l2 < 0.04, "{e:?}");
}

// ---------------------------------------------------------------- benchmark

fn time_ms(s: &CudaStream, reps: usize, mut f: impl FnMut()) -> (f64, f64) {
    let ctx = s.context();
    let flags = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
    f(); // warm-up
    let mut times = Vec::with_capacity(reps);
    for _ in 0..reps {
        let a = ctx.new_event(flags).unwrap();
        let b = ctx.new_event(flags).unwrap();
        a.record(s).unwrap();
        f();
        b.record(s).unwrap();
        times.push(a.elapsed_ms(&b).unwrap() as f64);
    }
    times.sort_by(|a, b| a.total_cmp(b));
    (times[times.len() / 2], times[0])
}

fn out_dir() -> std::path::PathBuf {
    let dir = std::env::var("SAGE_OUT").unwrap_or_else(|_| "/out/sage".into());
    let dir = std::path::PathBuf::from(dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Measured tensor-core peaks (TOPS) of this GPU per mma shape.
fn peaks(s: &Arc<CudaStream>) -> Vec<(&'static str, f64)> {
    let ptx = compile(&PEAK_SOURCE, &[]).unwrap();
    let module = s.context().load_module(ptx).unwrap();
    let sms = s
        .context()
        .attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        )
        .unwrap() as u32;
    let blocks = sms * 8;
    let threads = 128u32;
    let mut sink = s.alloc_zeros::<i32>((blocks * threads) as usize).unwrap();
    let iters = 4096i32;
    let mut out = Vec::new();
    for (name, ops_per_mma) in [
        ("sg_peak_s8", 2.0 * 16.0 * 8.0 * 32.0),
        ("sg_peak_e4m3_f32", 2.0 * 16.0 * 8.0 * 32.0),
        ("sg_peak_e4m3_f16", 2.0 * 16.0 * 8.0 * 32.0),
        ("sg_peak_bf16_f32", 2.0 * 16.0 * 8.0 * 16.0),
        ("sg_peak_f16_f16", 2.0 * 16.0 * 8.0 * 16.0),
    ] {
        let f = module.load_function(name).unwrap();
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let (ms, _) = time_ms(s, 5, || {
            let mut l = s.launch_builder(&f);
            l.arg(&mut sink).arg(&iters);
            unsafe { l.launch(cfg) }.unwrap();
        });
        let mmas = (blocks * threads / 32) as f64 * iters as f64 * 8.0;
        out.push((name, mmas * ops_per_mma / (ms * 1e-3) / 1e12));
    }
    out
}

fn read_bf16(path: &std::path::Path) -> Vec<u16> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let (pairs, _) = bytes.as_chunks::<2>();
    pairs.iter().map(|c| u16::from_le_bytes(*c)).collect()
}

fn write_bf16(path: &std::path::Path, data: &[u16]) {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(path, bytes).unwrap();
}

/// `key=value` pairs from a run directory's `meta.txt` (written by
/// `torch_ref.py gen`).
fn read_meta(dir: &std::path::Path) -> std::collections::HashMap<String, String> {
    std::fs::read_to_string(dir.join("meta.txt"))
        .unwrap()
        .lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        .collect()
}

/// The global attention of VGGT-Ω at 512×336 (689 tokens per frame, 16
/// heads of 64): times the prologue and the attention, dense and merged,
/// against the measured tensor-core peaks. With `SAGE_RUN` set, runs on that
/// directory's `q.bin`/`k.bin`/`v.bin` and writes `out_<mode>.bin` there for
/// `torch_ref.py compare`.
#[test]
#[ignore = "needs a CUDA GPU (compute 8.9+); writes to $SAGE_OUT"]
fn gpu_bench() {
    let s = stream();
    let run_dir = std::env::var("SAGE_RUN").ok().map(std::path::PathBuf::from);
    let env = |k: &str, default: usize| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let (layout, heads, d) = match &run_dir {
        Some(dir) => {
            let m = read_meta(dir);
            let get = |k: &str| m[k].parse::<usize>().unwrap();
            (
                FrameLayout {
                    frames: get("frames"),
                    special: get("special"),
                    grid_h: get("grid_h"),
                    grid_w: get("grid_w"),
                    dense_frames: 0,
                },
                get("heads"),
                get("head_dim"),
            )
        }
        None => (
            vggt_layout(env("SAGE_FRAMES", 200)),
            env("SAGE_HEADS", 16),
            env("SAGE_D", 64),
        ),
    };
    let frames = layout.frames;
    let shape = Shape {
        batch: 1,
        heads,
        tokens: frames * layout.tokens_per_frame(),
    };
    let hd = heads * d;
    let n = shape.tokens * hd;
    let strides = Strides::bthd(shape.tokens, heads, d);
    let t0 = Instant::now();
    let (q, k, v) = match &run_dir {
        Some(dir) => (
            s.clone_htod(&read_bf16(&dir.join("q.bin"))).unwrap(),
            s.clone_htod(&read_bf16(&dir.join("k.bin"))).unwrap(),
            s.clone_htod(&read_bf16(&dir.join("v.bin"))).unwrap(),
        ),
        None => {
            // Timing does not depend on the values; a cheap pattern will do.
            let mut rng = Rng::new(1);
            let mut make = || -> Vec<u16> {
                (0..n)
                    .map(|_| f32_to_bf16(rng.normal() * 1.5))
                    .collect::<Vec<_>>()
            };
            let (a, b, c) = (make(), make(), make());
            (
                s.clone_htod(&a).unwrap(),
                s.clone_htod(&b).unwrap(),
                s.clone_htod(&c).unwrap(),
            )
        }
    };
    println!("inputs ready in {:.1} s", t0.elapsed().as_secs_f64());
    assert!(
        q.len() >= n && k.len() >= n && v.len() >= n,
        "input files too small"
    );

    let peak = peaks(&s);
    for (name, tops) in &peak {
        println!("peak {name}: {tops:.1} TOPS");
    }
    let base = Config::for_head_dim(d);
    let mut tiles = vec![base.tile];
    if std::env::var("SAGE_TILES").as_deref() == Ok("all") {
        for (warps, m_tiles, stages) in [(4, 2, 2), (4, 2, 4), (4, 1, 3), (8, 1, 3), (8, 1, 4)] {
            let t = Tile {
                warps,
                m_tiles,
                stages,
            };
            if t != base.tile && !(d > 64 && m_tiles > 1) {
                tiles.push(t);
            }
        }
    }
    let configs: Vec<Config> = [PvAccum::F32, PvAccum::F16]
        .into_iter()
        .flat_map(|pv_accum| tiles.iter().map(move |&tile| Config { tile, pv_accum }))
        .collect();
    // Kernel-source experiments: `SAGE_VARIANTS="SG_SPLIT=1;SG_FAST_EXP=0"`
    // runs every config once per `;`-separated set of defines.
    let variants: Vec<String> = std::env::var("SAGE_VARIANTS")
        .map(|v| v.split(';').map(str::to_string).collect())
        .unwrap_or_else(|_| vec![String::new()]);
    let mut out = s.alloc_zeros::<u16>(n).unwrap();
    let mut lines = Vec::new();
    for (config, variant) in configs
        .iter()
        .flat_map(|c| variants.iter().map(move |v| (*c, v)))
    {
        let tile = config.tile;
        let extra: Vec<String> = variant
            .split_whitespace()
            .map(|d| format!("-D{d}"))
            .collect();
        let sage = SageAttention::with_defines(s.clone(), d, config, &extra).unwrap();
        let (regs, spill) = sage.attn_resources().unwrap();
        let pv = match config.pv_accum {
            PvAccum::F32 => "f32",
            PvAccum::F16 => "f16",
        };
        for (mode, merge) in [
            ("dense", KvMerge::Dense),
            ("merge2", KvMerge::Pool { stride: 2, layout }),
            ("merge4", KvMerge::Pool { stride: 4, layout }),
        ] {
            let mut ws = sage.workspace(shape, merge).unwrap();
            let opts = Options::default();
            let prep = |ws: &mut Workspace| {
                sage.prepare(
                    ws,
                    Tensor { data: &q, strides },
                    Tensor { data: &k, strides },
                    Tensor { data: &v, strides },
                    &opts,
                )
                .unwrap()
            };
            let (prep_ms, _) = time_ms(&s, 5, || prep(&mut ws));
            let reps = if mode == "dense" { 3 } else { 5 };
            let (attn_ms, attn_best) = time_ms(&s, reps, || {
                sage.attend(
                    &ws,
                    TensorMut {
                        data: &mut out,
                        strides,
                    },
                )
                .unwrap()
            });
            let plan = *ws.plan();
            let tops = flops(&plan, d) / (attn_ms * 1e-3) / 1e12;
            let dense_tops = flops(&Plan::new(shape, KvMerge::Dense).unwrap(), d)
                / ((attn_ms + prep_ms) * 1e-3)
                / 1e12;
            let line = format!(
                "{{\"tile\": \"{}w{}m{}s\", \"variant\": \"{variant}\", \"pv_accum\": \"{pv}\", \"regs\": {regs}, \"spill\": {spill}, \"mode\": \"{mode}\", \
                 \"frames\": {frames}, \"heads\": {heads}, \"head_dim\": {d}, \"tokens\": {}, \"keys\": {}, \
                 \"prep_ms\": {prep_ms:.3}, \"attn_ms\": {attn_ms:.3}, \"attn_best_ms\": {attn_best:.3}, \
                 \"attn_tops\": {tops:.1}, \"dense_equiv_tops\": {dense_tops:.1}, \"workspace_mb\": {:.1}}}",
                tile.warps,
                tile.m_tiles,
                tile.stages,
                shape.tokens,
                plan.tk,
                ws.bytes() as f64 / 1e6
            );
            println!("{line}");
            lines.push(line);
            if let Some(dir) = &run_dir
                && tile == base.tile
            {
                let mut mode = match config.pv_accum {
                    PvAccum::F32 => mode.to_string(),
                    PvAccum::F16 => format!("{mode}_pv16"),
                };
                if !variant.is_empty() {
                    mode = format!("{mode}_{}", variant.replace([' ', '='], "_"));
                }
                prep(&mut ws);
                sage.attend(
                    &ws,
                    TensorMut {
                        data: &mut out,
                        strides,
                    },
                )
                .unwrap();
                let host = s.clone_dtoh(&out).unwrap();
                write_bf16(&dir.join(format!("out_{mode}.bin")), &host[..n]);
            }
        }
    }
    let peaks_json: Vec<String> = peak
        .iter()
        .map(|(n, t)| format!("\"{n}\": {t:.1}"))
        .collect();
    let json = format!(
        "{{\"peaks_tops\": {{{}}}, \"runs\": [\n  {}\n]}}\n",
        peaks_json.join(", "),
        lines.join(",\n  ")
    );
    let path = out_dir().join(format!("bench_d{d}_f{frames}.json"));
    std::fs::write(&path, json).unwrap();
    println!("wrote {}", path.display());
}
