//! Dense head (owner: heads; spec §5, `models/heads/dense_head.py`), one
//! chunk of frames at a time, NHWC bf16 activations.
//!
//! ```text
//! level k (layer 4, 11, 17, 23):
//!   LN(2048) over the chunk's rows → 1×1 project (GEMM, bias) → drop the 17
//!   special rows + 0.1·PE (one gather kernel) → resize:
//!     k0 ConvT 4×4/4: GEMM 256 → 16·256 + depth-to-space (+bias)
//!     k1 ConvT 2×2/2: GEMM 512 →  4·512 + depth-to-space (+bias)
//!     k2 identity;  k3 conv 3×3 stride 2 (+bias)
//!   r_k = conv 3×3 → 256 (no bias)
//! fusion: refinenet4..1 = [o + RCU1(r)] → RCU2 → bilinear (align_corners) →
//!   1×1 out_conv (GEMM); RCU = conv(relu(conv(relu x) + b)) + b + x with the
//!   ReLUs applied on the convolution's input load and the residual adds
//!   (x, and o for RCU1) in its epilogue
//! output: + 0.1·PE, proj | proj_conf (fp32 256 → 32), pixel shuffle ×4,
//!   exp / 1 + exp → f16 depth and conf, optional unprojection to world points
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use spl_core::{Result, bail};
use spl_nn::Nn;
use spl_nn::conv::{ConvEpilogue, ConvShape, D2sOrder};
use spl_nn::cudarc::driver::{
    CudaSlice, CudaView, CudaViewMut, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg,
};
use spl_nn::gemm::{Epilogue, GemmShape};
use spl_nn::norm::{LnParams, Quant, Rows};

use super::{
    DIM, Geometry, HeadSource, Kernels, SPECIAL, alloc, deconv_rows, ohwi, pos_embed, tensor,
    up_bf16, up_f32,
};

const F: usize = 256;
const OC: [usize; 4] = [256, 512, 1024, 1024];

struct ConvW {
    w: CudaSlice<u16>,
    b: Option<CudaSlice<f32>>,
}

impl ConvW {
    fn load(
        nn: &Nn,
        src: &dyn HeadSource,
        name: &str,
        o: usize,
        i: usize,
        k: usize,
        bias: bool,
    ) -> Result<Self> {
        let w = tensor(src, &format!("{name}.weight"), &[o, i, k, k])?;
        Ok(Self {
            w: up_bf16(nn, &ohwi(&w, o, i, k))?,
            b: if bias {
                Some(up_f32(nn, &tensor(src, &format!("{name}.bias"), &[o])?)?)
            } else {
                None
            },
        })
    }
}

/// A 1×1 convolution run as a GEMM: bf16 `[o][i]` weight, bf16 bias.
struct Linear {
    w: CudaSlice<u16>,
    b: CudaSlice<u16>,
}

impl Linear {
    fn load(nn: &Nn, src: &dyn HeadSource, name: &str, o: usize, i: usize) -> Result<Self> {
        Ok(Self {
            w: up_bf16(nn, &tensor(src, &format!("{name}.weight"), &[o, i, 1, 1])?)?,
            b: up_bf16(nn, &tensor(src, &format!("{name}.bias"), &[o])?)?,
        })
    }
}

struct Rcu {
    c1: ConvW,
    c2: ConvW,
}

struct Fusion {
    rcu1: Option<Rcu>,
    rcu2: Rcu,
    out: Linear,
}

/// Buffers and PE tables for one (chunk frames, grid) shape.
struct Work {
    pe: [CudaSlice<f32>; 4],
    pe_out: CudaSlice<f32>,
    ln: CudaSlice<u16>,
    proj: CudaSlice<u16>,
    feat: CudaSlice<u16>,
    deconv: CudaSlice<u16>,
    x: [CudaSlice<u16>; 3],
    r: [CudaSlice<u16>; 4],
    t: [CudaSlice<u16>; 3],
    o: [CudaSlice<u16>; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct OutArgs {
    fused: u64,
    pe: u64,
    w: u64,
    b: u64,
    depth: u64,
    conf: u64,
    extr: u64,
    intr: u64,
    world: u64,
    frames: i32,
    qh: i32,
    qw: i32,
}

// SAFETY: matches `struct OutArgs` in cuda/vggt_heads.cu.
unsafe impl spl_nn::cudarc::driver::DeviceRepr for OutArgs {}

/// (chunk frames, grid h, grid w, height, width).
type WorkKey = (usize, usize, usize, usize, usize);

/// Per-level sizes for an `h × w` grid.
fn level_hw(k: usize, h: usize, w: usize) -> (usize, usize) {
    match k {
        0 => (4 * h, 4 * w),
        1 => (2 * h, 2 * w),
        2 => (h, w),
        _ => (h.div_ceil(2), w.div_ceil(2)),
    }
}

pub struct DenseHead {
    k: Arc<Kernels>,
    norm_g: CudaSlice<f32>,
    norm_b: CudaSlice<f32>,
    projects: Vec<Linear>,
    deconv0_w: CudaSlice<u16>,
    deconv0_b: CudaSlice<f32>,
    deconv1_w: CudaSlice<u16>,
    deconv1_b: CudaSlice<f32>,
    resize3: ConvW,
    rn: Vec<ConvW>,
    /// refinenet1..4.
    fusion: Vec<Fusion>,
    out_w: CudaSlice<f32>,
    out_b: CudaSlice<f32>,
    work: Mutex<HashMap<WorkKey, Work>>,
}

impl DenseHead {
    pub fn new(nn: &Nn, k: Arc<Kernels>, src: &dyn HeadSource) -> Result<Self> {
        let p = "dense_head";
        let mut projects = Vec::new();
        let mut rn = Vec::new();
        for (i, &oc) in OC.iter().enumerate() {
            projects.push(Linear::load(
                nn,
                src,
                &format!("{p}.projects.{i}"),
                oc,
                DIM,
            )?);
            rn.push(ConvW::load(
                nn,
                src,
                &format!("{p}.scratch.layer{}_rn", i + 1),
                F,
                oc,
                3,
                false,
            )?);
        }
        let d0 = tensor(
            src,
            &format!("{p}.resize_layers.0.weight"),
            &[256, 256, 4, 4],
        )?;
        let d1 = tensor(
            src,
            &format!("{p}.resize_layers.1.weight"),
            &[512, 512, 2, 2],
        )?;
        let rcu = |name: String| -> Result<Rcu> {
            Ok(Rcu {
                c1: ConvW::load(nn, src, &format!("{name}.conv1"), F, F, 3, true)?,
                c2: ConvW::load(nn, src, &format!("{name}.conv2"), F, F, 3, true)?,
            })
        };
        let mut fusion = Vec::new();
        for r in 1..=4 {
            let base = format!("{p}.scratch.refinenet{r}");
            fusion.push(Fusion {
                rcu1: if r < 4 {
                    Some(rcu(format!("{base}.resConfUnit1"))?)
                } else {
                    None
                },
                rcu2: rcu(format!("{base}.resConfUnit2"))?,
                out: Linear::load(nn, src, &format!("{base}.out_conv"), F, F)?,
            });
        }
        let mut out_w = tensor(src, &format!("{p}.proj.weight"), &[16, F, 1, 1])?;
        out_w.extend(tensor(
            src,
            &format!("{p}.proj_conf.weight"),
            &[16, F, 1, 1],
        )?);
        let mut out_b = tensor(src, &format!("{p}.proj.bias"), &[16])?;
        out_b.extend(tensor(src, &format!("{p}.proj_conf.bias"), &[16])?);
        Ok(Self {
            norm_g: up_f32(nn, &tensor(src, &format!("{p}.norm.weight"), &[DIM])?)?,
            norm_b: up_f32(nn, &tensor(src, &format!("{p}.norm.bias"), &[DIM])?)?,
            projects,
            deconv0_w: up_bf16(nn, &deconv_rows(&d0, 256, 256, 4))?,
            deconv0_b: up_f32(
                nn,
                &tensor(src, &format!("{p}.resize_layers.0.bias"), &[256])?,
            )?,
            deconv1_w: up_bf16(nn, &deconv_rows(&d1, 512, 512, 2))?,
            deconv1_b: up_f32(
                nn,
                &tensor(src, &format!("{p}.resize_layers.1.bias"), &[512])?,
            )?,
            resize3: ConvW::load(
                nn,
                src,
                &format!("{p}.resize_layers.3"),
                1024,
                1024,
                3,
                true,
            )?,
            rn,
            fusion,
            out_w: up_f32(nn, &out_w)?,
            out_b: up_f32(nn, &out_b)?,
            k,
            work: Mutex::new(HashMap::new()),
        })
    }

    fn sizes(g: &Geometry) -> [usize; 9] {
        let (n, h, w) = (g.frames, g.grid_h, g.grid_w);
        let p = h * w;
        let q = 16 * p;
        [
            n * g.tokens() * DIM,                     // ln
            n * g.tokens() * 1024,                    // proj
            n * p * 1024,                             // feat
            n * p * 4096,                             // deconv (k0: 16·256, k1: 4·512)
            n * q * 256,                              // x0
            n * 4 * p * 512,                          // x1
            n * h.div_ceil(2) * w.div_ceil(2) * 1024, // x3
            n * q * F,                                // r0 / t / o (largest level)
            0,
        ]
    }

    /// Device bytes of the activations for a chunk of `g.frames` frames.
    pub fn scratch_bytes(g: &Geometry) -> usize {
        let s = Self::sizes(g);
        let (h, w) = (g.grid_h, g.grid_w);
        let r_rest: usize = (1..4)
            .map(|k| {
                let (a, b) = level_hw(k, h, w);
                g.frames * a * b * F
            })
            .sum();
        let pe: usize = OC.iter().map(|c| c * h * w).sum::<usize>() + 16 * h * w * F;
        2 * (s[0] + s[1] + s[2] + s[3] + s[4] + s[5] + s[6] + 6 * s[7] + r_rest) + 4 * pe
    }

    fn make_work(&self, nn: &Nn, g: &Geometry) -> Result<Work> {
        let s = Self::sizes(g);
        let (h, w) = (g.grid_h, g.grid_w);
        let aspect = g.width as f64 / g.height as f64;
        let pe = |c: usize, a: usize, b: usize| up_f32(nn, &pos_embed::table(c, a, b, aspect, 0.1));
        let r = |k: usize| {
            let (a, b) = level_hw(k, h, w);
            alloc::<u16>(nn, g.frames * a * b * F)
        };
        Ok(Work {
            pe: [
                pe(OC[0], h, w)?,
                pe(OC[1], h, w)?,
                pe(OC[2], h, w)?,
                pe(OC[3], h, w)?,
            ],
            pe_out: pe(F, 4 * h, 4 * w)?,
            ln: alloc(nn, s[0])?,
            proj: alloc(nn, s[1])?,
            feat: alloc(nn, s[2])?,
            deconv: alloc(nn, s[3])?,
            x: [alloc(nn, s[4])?, alloc(nn, s[5])?, alloc(nn, s[6])?],
            r: [r(0)?, r(1)?, r(2)?, r(3)?],
            t: [alloc(nn, s[7])?, alloc(nn, s[7])?, alloc(nn, s[7])?],
            o: [alloc(nn, s[7])?, alloc(nn, s[7])?],
        })
    }

    /// One chunk: `layers` are the chunk's rows of L_4, L_11, L_17, L_23
    /// (bf16 `[n·T][2048]`, `g.frames = n`); writes f16 `depth` and `conf`
    /// `[n][H][W]`. With `cams` (the chunk's extrinsics `[n][12]` and
    /// intrinsics `[n][9]`) and `world`, also writes world points
    /// `[n][H][W][3]`.
    #[allow(clippy::too_many_arguments)]
    pub fn run_chunk(
        &self,
        nn: &Nn,
        layers: [CudaView<'_, u16>; 4],
        g: &Geometry,
        depth: &mut CudaViewMut<'_, u16>,
        conf: &mut CudaViewMut<'_, u16>,
        cams: Option<(&CudaView<'_, f32>, &CudaView<'_, f32>)>,
        world: Option<&mut CudaViewMut<'_, f32>>,
    ) -> Result<()> {
        let (n, h, w) = (g.frames, g.grid_h, g.grid_w);
        let (t, p) = (g.tokens(), g.patches());
        if n == 0 {
            return Ok(());
        }
        if g.height != 16 * h || g.width != 16 * w {
            bail!(
                "dense head: image {}×{} is not 16 × grid {h}×{w}",
                g.height,
                g.width
            );
        }
        for l in &layers {
            if l.len() < n * t * DIM {
                bail!(
                    "dense head: layer view holds {} values, need {}",
                    l.len(),
                    n * t * DIM
                );
            }
        }
        let px = n * g.height * g.width;
        if depth.len() < px || conf.len() < px {
            bail!("dense head: depth/conf buffers smaller than {px}");
        }
        let key = (n, h, w, g.height, g.width);
        let mut map = self
            .work
            .lock()
            .map_err(|_| spl_core::anyhow!("dense work lock poisoned"))?;
        if !map.contains_key(&key) {
            // A run uses at most two chunk sizes (full chunks and the
            // remainder): keep both, or every run reallocates twice.
            if map.len() >= 2 {
                map.clear();
            }
            map.insert(key, self.make_work(nn, g)?);
        }
        let ws = map.get_mut(&key).unwrap();
        let k = &*self.k;
        let ln = LnParams {
            gamma: &self.norm_g,
            beta: &self.norm_b,
            eps: 1e-5,
        };

        for (lvl, layer) in layers.iter().enumerate() {
            let oc = OC[lvl];
            k.norm.layer_norm(
                nn,
                Rows::Bf16(layer),
                n * t,
                DIM,
                &ln,
                Some(&mut ws.ln),
                Quant::None,
            )?;
            let pr = &self.projects[lvl];
            k.gemm.linear_bf16(
                nn,
                &ws.ln,
                &pr.w,
                Some(&pr.b),
                Epilogue::None,
                &mut ws.proj,
                GemmShape::packed(n * t, oc, DIM),
            )?;
            let dst: &mut CudaSlice<u16> = if lvl == 2 { &mut ws.x[1] } else { &mut ws.feat };
            // Level 2 has no resize; its features go to x[1], which level 1
            // has finished with by then.
            k.conv.add_table(
                nn,
                &ws.proj,
                Some(&ws.pe[lvl]),
                1.0,
                dst,
                n,
                t,
                SPECIAL,
                p,
                oc,
            )?;
            match lvl {
                0 => {
                    k.gemm.linear_bf16(
                        nn,
                        &ws.feat,
                        &self.deconv0_w,
                        None,
                        Epilogue::None,
                        &mut ws.deconv,
                        GemmShape::packed(n * p, 16 * 256, 256),
                    )?;
                    k.conv.depth_to_space(
                        nn,
                        &ws.deconv,
                        Some(&self.deconv0_b),
                        D2sOrder::TapMajor,
                        &mut ws.x[0],
                        n,
                        h,
                        w,
                        256,
                        4,
                    )?;
                    let s = ConvShape::same(n, 4 * h, 4 * w, 256, F, 3, 1);
                    k.taps
                        .bf16(nn, "dense_resize0", &ws.x[0], s.n * s.h * s.w * 256)?;
                    k.conv.conv2d(
                        nn,
                        &ws.x[0],
                        &self.rn[0].w,
                        &ConvEpilogue::default(),
                        false,
                        &mut ws.r[0],
                        s,
                    )?;
                }
                1 => {
                    k.gemm.linear_bf16(
                        nn,
                        &ws.feat,
                        &self.deconv1_w,
                        None,
                        Epilogue::None,
                        &mut ws.deconv,
                        GemmShape::packed(n * p, 4 * 512, 512),
                    )?;
                    k.conv.depth_to_space(
                        nn,
                        &ws.deconv,
                        Some(&self.deconv1_b),
                        D2sOrder::TapMajor,
                        &mut ws.x[1],
                        n,
                        h,
                        w,
                        512,
                        2,
                    )?;
                    let s = ConvShape::same(n, 2 * h, 2 * w, 512, F, 3, 1);
                    k.taps
                        .bf16(nn, "dense_resize1", &ws.x[1], s.n * s.h * s.w * 512)?;
                    k.conv.conv2d(
                        nn,
                        &ws.x[1],
                        &self.rn[1].w,
                        &ConvEpilogue::default(),
                        false,
                        &mut ws.r[1],
                        s,
                    )?;
                }
                2 => {
                    let s = ConvShape::same(n, h, w, 1024, F, 3, 1);
                    k.taps.bf16(nn, "dense_resize2", &ws.x[1], n * p * 1024)?;
                    k.conv.conv2d(
                        nn,
                        &ws.x[1],
                        &self.rn[2].w,
                        &ConvEpilogue::default(),
                        false,
                        &mut ws.r[2],
                        s,
                    )?;
                }
                _ => {
                    let s = ConvShape::same(n, h, w, 1024, 1024, 3, 2);
                    let epi = ConvEpilogue {
                        bias: self.resize3.b.as_ref().map(|b| b as _),
                        residual: [None, None],
                    };
                    k.conv
                        .conv2d(nn, &ws.feat, &self.resize3.w, &epi, false, &mut ws.x[2], s)?;
                    k.taps
                        .bf16(nn, "dense_resize3", &ws.x[2], s.pixels() * 1024)?;
                    let s = ConvShape::same(n, h.div_ceil(2), w.div_ceil(2), 1024, F, 3, 1);
                    k.conv.conv2d(
                        nn,
                        &ws.x[2],
                        &self.rn[3].w,
                        &ConvEpilogue::default(),
                        false,
                        &mut ws.r[3],
                        s,
                    )?;
                }
            }
        }

        for lvl in 0..4 {
            let (a, b) = level_hw(lvl, h, w);
            k.taps
                .bf16(nn, &format!("dense_rn{lvl}"), &ws.r[lvl], n * a * b * F)?;
        }

        // Fusion: refinenet4 (r3 → size of r2), 3, 2, 1. `o` ping-pongs.
        let Work {
            r,
            t: tb,
            o,
            pe_out,
            ..
        } = ws;
        let [t0, t1, t2] = tb;
        let [o0, o1] = o;
        let mut cur: &mut CudaSlice<u16> = o0;
        let mut prev: &mut CudaSlice<u16> = o1;
        for lvl in [3usize, 2, 1, 0] {
            let fb = &self.fusion[lvl];
            let (a, b) = level_hw(lvl, h, w);
            let (oa, ob) = if lvl == 0 {
                (a, b)
            } else {
                level_hw(lvl - 1, h, w)
            };
            let s = ConvShape::same(n, a, b, F, F, 3, 1);
            // input = o_prev + RCU1(r)   (or r alone for refinenet4)
            let input: &CudaSlice<u16> = if let Some(rc) = &fb.rcu1 {
                self.rcu(nn, &r[lvl], Some(&*prev), rc, t0, t1, s)?;
                &*t1
            } else {
                &r[lvl]
            };
            self.rcu(nn, input, None, &fb.rcu2, t0, t2, s)?;
            let up: &CudaSlice<u16> = if (a, b) != (oa, ob) {
                k.conv.upsample_bilinear(nn, &*t2, t1, n, a, b, F, oa, ob)?;
                &*t1
            } else {
                &*t2
            };
            k.gemm.linear_bf16(
                nn,
                up,
                &fb.out.w,
                Some(&fb.out.b),
                Epilogue::None,
                cur,
                GemmShape::packed(n * oa * ob, F, F),
            )?;
            k.taps.bf16(
                nn,
                &format!("dense_refine{}", lvl + 1),
                cur,
                n * oa * ob * F,
            )?;
            std::mem::swap(&mut cur, &mut prev);
        }
        // `prev` now holds refinenet1's output at 4h × 4w.
        self.output(nn, prev, pe_out, g, depth, conf, cams, world)
    }

    /// `out = conv2(relu(conv1(relu x) + b1)) + b2 + x (+ extra)`.
    #[allow(clippy::too_many_arguments)]
    fn rcu(
        &self,
        nn: &Nn,
        x: &CudaSlice<u16>,
        extra: Option<&CudaSlice<u16>>,
        w: &Rcu,
        tmp: &mut CudaSlice<u16>,
        out: &mut CudaSlice<u16>,
        s: ConvShape,
    ) -> Result<()> {
        let k = &*self.k;
        let e1 = ConvEpilogue {
            bias: w.c1.b.as_ref().map(|b| b as _),
            residual: [None, None],
        };
        k.conv.conv2d(nn, x, &w.c1.w, &e1, true, tmp, s)?;
        let e2 = ConvEpilogue {
            bias: w.c2.b.as_ref().map(|b| b as _),
            residual: [Some(x), extra.map(|e| e as _)],
        };
        k.conv.conv2d(nn, &*tmp, &w.c2.w, &e2, true, out, s)
    }

    #[allow(clippy::too_many_arguments)]
    fn output(
        &self,
        nn: &Nn,
        fused: &CudaSlice<u16>,
        pe: &CudaSlice<f32>,
        g: &Geometry,
        depth: &mut CudaViewMut<'_, u16>,
        conf: &mut CudaViewMut<'_, u16>,
        cams: Option<(&CudaView<'_, f32>, &CudaView<'_, f32>)>,
        world: Option<&mut CudaViewMut<'_, f32>>,
    ) -> Result<()> {
        let s = &nn.stream;
        let n = g.frames;
        let (qh, qw) = (4 * g.grid_h, 4 * g.grid_w);
        let (fp, _a) = fused.device_ptr(s);
        let (pp, _b) = pe.device_ptr(s);
        let (wp, _c) = self.out_w.device_ptr(s);
        let (bp, _d) = self.out_b.device_ptr(s);
        let (dp, _e) = depth.device_ptr_mut(s);
        let (cp, _f) = conf.device_ptr_mut(s);
        let mut guards = Vec::new();
        let (mut ep, mut ip, mut wo) = (0u64, 0u64, 0u64);
        if let (Some((e, i)), Some(wv)) = (cams, world) {
            if e.len() < n * 12 || i.len() < n * 9 || wv.len() < n * g.height * g.width * 3 {
                bail!("dense head: camera or world-point buffers too small");
            }
            let (x, gx) = e.device_ptr(s);
            let (y, gy) = i.device_ptr(s);
            ep = x;
            ip = y;
            guards.push(gx);
            guards.push(gy);
            let (z, gz) = wv.device_ptr_mut(s);
            wo = z;
            guards.push(gz);
        }
        let a = OutArgs {
            fused: fp,
            pe: pp,
            w: wp,
            b: bp,
            depth: dp,
            conf: cp,
            extr: ep,
            intr: ip,
            world: wo,
            frames: n as i32,
            qh: qh as i32,
            qw: qw as i32,
        };
        let total = n * qh * qw;
        let cfg = LaunchConfig {
            grid_dim: (total.div_ceil(128) as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = s.launch_builder(&self.k.dense_output);
        b.arg(&a);
        // SAFETY: `a` matches `OutArgs`; all sizes checked above.
        unsafe { b.launch(cfg) }.map_err(|e| spl_core::anyhow!("dense_output: {e:?}"))?;
        drop(guards);
        Ok(())
    }
}
