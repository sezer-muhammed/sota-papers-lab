//! Camera head (owner: heads; spec §4, `models/heads/camera_head.py`).
//!
//! ```text
//! t   = token_norm(L_23[:, :17])                     one sequence [N·17, 2048]
//! t   = 4 × Block(2048, 16 heads × 128, qkv bias [b_q, 0, b_v], exact GELU MLP 8192, LayerScale)
//! c   = trunk_norm(t[frame, 0])
//! raw = Linear(2048→1024) → GELU → Linear(1024→9);  fov = relu(raw[7:9]) + 0.01
//! ```
//!
//! Per block: LN (fused with the previous residual update) → QKV GEMM (bias
//! in the epilogue) → head-dim-128 attention straight from the packed QKV →
//! proj GEMM → LayerScale residual + LN2 in one pass → fc1 GEMM (+bias) →
//! exact GELU in place → fc2 GEMM → LayerScale residual fused into the next
//! LN. The residual stream is fp32. The branch tail (1024 → 9), the
//! activation and the pose-encoding → camera conversion run in one kernel.

use std::sync::{Arc, Mutex};

use spl_core::{Context, Result, bail};
use spl_nn::Act;
use spl_nn::Nn;
use spl_nn::cudarc::driver::{CudaSlice, CudaView, DevicePtr, LaunchConfig, PushKernelArg};
use spl_nn::gemm::{Epilogue, GemmShape};
use spl_nn::norm::{Branch, LnParams, Quant, Rows, RowsMut};
use spl_nn::rowops::PackedQkv;

use super::{
    DIM, DeviceCams, Geometry, HeadSource, Kernels, SPECIAL, alloc, tensor, up_bf16, up_f32,
};

const HEADS: usize = 16;
const HIDDEN: usize = 4 * DIM;
const BRANCH: usize = DIM / 2;

struct Ln {
    gamma: CudaSlice<f32>,
    beta: CudaSlice<f32>,
}

impl Ln {
    fn load(nn: &Nn, src: &dyn HeadSource, prefix: &str) -> Result<Self> {
        Ok(Self {
            gamma: up_f32(nn, &tensor(src, &format!("{prefix}.weight"), &[DIM])?)?,
            beta: up_f32(nn, &tensor(src, &format!("{prefix}.bias"), &[DIM])?)?,
        })
    }

    fn params(&self) -> LnParams<'_> {
        LnParams {
            gamma: &self.gamma,
            beta: &self.beta,
            eps: 1e-5,
        }
    }
}

struct Block {
    norm1: Ln,
    qkv_w: CudaSlice<u16>,
    qkv_b: CudaSlice<u16>,
    proj_w: CudaSlice<u16>,
    proj_b: CudaSlice<u16>,
    ls1: CudaSlice<f32>,
    norm2: Ln,
    fc1_w: CudaSlice<u16>,
    fc1_b: CudaSlice<u16>,
    fc2_w: CudaSlice<u16>,
    fc2_b: CudaSlice<u16>,
    ls2: CudaSlice<f32>,
}

/// Activations for one frame count.
struct Work {
    frames: usize,
    gathered: CudaSlice<u16>,
    x: CudaSlice<f32>,
    hn: CudaSlice<u16>,
    qkv: CudaSlice<u16>,
    att: CudaSlice<u16>,
    y: CudaSlice<u16>,
    hidden: CudaSlice<u16>,
    branch: CudaSlice<u16>,
}

impl Work {
    fn new(nn: &Nn, frames: usize) -> Result<Self> {
        let rows = frames * SPECIAL;
        Ok(Self {
            frames,
            gathered: alloc(nn, rows * DIM)?,
            x: alloc(nn, rows * DIM)?,
            hn: alloc(nn, rows * DIM)?,
            qkv: alloc(nn, rows * 3 * DIM)?,
            att: alloc(nn, rows * DIM)?,
            y: alloc(nn, rows * DIM)?,
            hidden: alloc(nn, rows * HIDDEN)?,
            branch: alloc(nn, frames * BRANCH)?,
        })
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct CamArgs {
    h: u64,
    w2: u64,
    b2: u64,
    pose: u64,
    extr: u64,
    intr: u64,
    frames: i32,
    width: i32,
    height: i32,
    hidden: i32,
}

// SAFETY: matches `struct CamArgs` in cuda/vggt_heads.cu.
unsafe impl spl_nn::cudarc::driver::DeviceRepr for CamArgs {}

/// Head input: the bf16 layer cache or fp32 special tokens.
enum Input<'a, 'b> {
    Cache(&'a CudaView<'b, u16>),
    F32(&'a CudaView<'b, f32>),
}

pub struct CameraHead {
    k: Arc<Kernels>,
    token_norm: Ln,
    blocks: Vec<Block>,
    trunk_norm: Ln,
    b0_w: CudaSlice<u16>,
    b0_b: CudaSlice<u16>,
    b2_w: CudaSlice<f32>,
    b2_b: CudaSlice<f32>,
    work: Mutex<Option<Work>>,
}

impl CameraHead {
    pub fn new(nn: &Nn, k: Arc<Kernels>, src: &dyn HeadSource) -> Result<Self> {
        let mut blocks = Vec::with_capacity(4);
        for i in 0..4 {
            let p = format!("camera_head.trunk.{i}");
            let t = |n: &str, s: &[usize]| tensor(src, &format!("{p}.{n}"), s);
            let mut qkv_b = t("attn.qkv.bias", &[3 * DIM])?;
            // Masked K bias (bias_mask = [1, 0, 1]); zero it whatever the source did.
            qkv_b[DIM..2 * DIM].fill(0.0);
            blocks.push(Block {
                norm1: Ln::load(nn, src, &format!("{p}.norm1"))?,
                qkv_w: up_bf16(nn, &t("attn.qkv.weight", &[3 * DIM, DIM])?)?,
                qkv_b: up_bf16(nn, &qkv_b)?,
                proj_w: up_bf16(nn, &t("attn.proj.weight", &[DIM, DIM])?)?,
                proj_b: up_bf16(nn, &t("attn.proj.bias", &[DIM])?)?,
                ls1: up_f32(nn, &t("ls1.gamma", &[DIM])?)?,
                norm2: Ln::load(nn, src, &format!("{p}.norm2"))?,
                fc1_w: up_bf16(nn, &t("mlp.fc1.weight", &[HIDDEN, DIM])?)?,
                fc1_b: up_bf16(nn, &t("mlp.fc1.bias", &[HIDDEN])?)?,
                fc2_w: up_bf16(nn, &t("mlp.fc2.weight", &[DIM, HIDDEN])?)?,
                fc2_b: up_bf16(nn, &t("mlp.fc2.bias", &[DIM])?)?,
                ls2: up_f32(nn, &t("ls2.gamma", &[DIM])?)?,
            });
        }
        Ok(Self {
            token_norm: Ln::load(nn, src, "camera_head.token_norm")?,
            trunk_norm: Ln::load(nn, src, "camera_head.trunk_norm")?,
            b0_w: up_bf16(
                nn,
                &tensor(src, "camera_head.camera_branch.0.weight", &[BRANCH, DIM])?,
            )?,
            b0_b: up_bf16(
                nn,
                &tensor(src, "camera_head.camera_branch.0.bias", &[BRANCH])?,
            )?,
            b2_w: up_f32(
                nn,
                &tensor(src, "camera_head.camera_branch.2.weight", &[9, BRANCH])?,
            )?,
            b2_b: up_f32(nn, &tensor(src, "camera_head.camera_branch.2.bias", &[9])?)?,
            blocks,
            k,
            work: Mutex::new(None),
        })
    }

    /// Device bytes of the activations for `g.frames` frames.
    pub fn scratch_bytes(g: &Geometry) -> usize {
        let rows = g.frames * SPECIAL;
        rows * DIM * 4 + rows * (DIM * 5 + HIDDEN) * 2 + g.frames * BRANCH * 2
    }

    /// Runs the head on the last cached layer (`cache23`: bf16 `[N·T][2048]`),
    /// writes `cams` and returns `pose_enc` per frame. Synchronises once to
    /// download the 9 floats per frame.
    pub fn run(
        &self,
        nn: &Nn,
        cache23: &CudaView<'_, u16>,
        g: &Geometry,
        cams: &mut DeviceCams,
    ) -> Result<Vec<[f32; 9]>> {
        self.enqueue(nn, cache23, g, cams)?;
        Self::download(nn, g, cams)
    }

    /// [`Self::run`] on fp32 special tokens `[N·17][2048]` (`[F_23 | G_23]`
    /// rows 0..17 of every frame): no bf16 rounding of the head input, which
    /// is most of the fov error of a bf16 layer cache.
    pub fn run_f32(
        &self,
        nn: &Nn,
        tokens: &CudaView<'_, f32>,
        g: &Geometry,
        cams: &mut DeviceCams,
    ) -> Result<Vec<[f32; 9]>> {
        self.enqueue_f32(nn, tokens, g, cams)?;
        Self::download(nn, g, cams)
    }

    fn download(nn: &Nn, g: &Geometry, cams: &DeviceCams) -> Result<Vec<[f32; 9]>> {
        let pe = nn
            .stream
            .clone_dtoh(&cams.pose_enc)
            .context("download pose_enc")?;
        Ok(pe
            .as_chunks::<9>()
            .0
            .iter()
            .take(g.frames)
            .copied()
            .collect())
    }

    /// [`Self::run`] without the download (for timing and graph capture).
    pub fn enqueue(
        &self,
        nn: &Nn,
        cache23: &CudaView<'_, u16>,
        g: &Geometry,
        cams: &mut DeviceCams,
    ) -> Result<()> {
        if cache23.len() < g.frames * g.tokens() * DIM {
            bail!(
                "camera head: layer cache holds {} values, need {}",
                cache23.len(),
                g.frames * g.tokens() * DIM
            );
        }
        self.enqueue_input(nn, Input::Cache(cache23), g, cams)
    }

    /// [`Self::run_f32`] without the download.
    pub fn enqueue_f32(
        &self,
        nn: &Nn,
        tokens: &CudaView<'_, f32>,
        g: &Geometry,
        cams: &mut DeviceCams,
    ) -> Result<()> {
        if tokens.len() < g.frames * SPECIAL * DIM {
            bail!(
                "camera head: special tokens hold {} values, need {}",
                tokens.len(),
                g.frames * SPECIAL * DIM
            );
        }
        self.enqueue_input(nn, Input::F32(tokens), g, cams)
    }

    fn enqueue_input(
        &self,
        nn: &Nn,
        input: Input<'_, '_>,
        g: &Geometry,
        cams: &mut DeviceCams,
    ) -> Result<()> {
        let n = g.frames;
        if cams.frames < n {
            bail!(
                "camera head: DeviceCams holds {} frames, need {n}",
                cams.frames
            );
        }
        let mut guard = self
            .work
            .lock()
            .map_err(|_| spl_core::anyhow!("camera work lock poisoned"))?;
        if guard.as_ref().is_none_or(|w| w.frames < n) {
            *guard = Some(Work::new(nn, n)?);
        }
        let w = guard.as_mut().unwrap();
        let k = &*self.k;
        let rows = n * SPECIAL;

        // token_norm (from the 17 special rows of every frame of the bf16
        // cache, gathered, or from fp32 tokens), widen to the fp32 residual
        // stream, then norm1 of block 0.
        let rows_in = match input {
            Input::Cache(cache23) => {
                k.conv.add_table(
                    nn,
                    cache23,
                    None,
                    0.0,
                    &mut w.gathered,
                    n,
                    g.tokens(),
                    0,
                    SPECIAL,
                    DIM,
                )?;
                Rows::Bf16(&w.gathered)
            }
            Input::F32(t) => Rows::F32(t),
        };
        k.norm.layer_norm(
            nn,
            rows_in,
            rows,
            DIM,
            &self.token_norm.params(),
            Some(&mut w.hn),
            Quant::None,
        )?;
        self.k.taps.bf16(nn, "cam_token_norm", &w.hn, rows * DIM)?;
        self.widen(nn, &w.hn, &mut w.x, rows * DIM)?;
        k.norm.layer_norm(
            nn,
            Rows::F32(&w.x),
            rows,
            DIM,
            &self.blocks[0].norm1.params(),
            Some(&mut w.hn),
            Quant::None,
        )?;

        let scale = 1.0 / (128f32).sqrt();
        for (i, b) in self.blocks.iter().enumerate() {
            k.gemm.linear_bf16(
                nn,
                &w.hn,
                &b.qkv_w,
                Some(&b.qkv_b),
                Epilogue::None,
                &mut w.qkv,
                GemmShape::packed(rows, 3 * DIM, DIM),
            )?;
            k.rowops.attention_d128(
                nn,
                &w.qkv,
                PackedQkv::thirds(DIM),
                rows,
                HEADS,
                scale,
                &mut w.att,
                DIM,
            )?;
            k.gemm.linear_bf16(
                nn,
                &w.att,
                &b.proj_w,
                None,
                Epilogue::None,
                &mut w.y,
                GemmShape::packed(rows, DIM, DIM),
            )?;
            let br = Branch {
                y: &w.y,
                tok: None,
                chan: None,
                bias: Some(&b.proj_b),
                layer_scale: Some(&b.ls1),
            };
            k.norm.residual_layer_norm(
                nn,
                RowsMut::F32(&mut w.x),
                &br,
                rows,
                DIM,
                &b.norm2.params(),
                Some(&mut w.hn),
                Quant::None,
            )?;
            k.gemm.linear_bf16(
                nn,
                &w.hn,
                &b.fc1_w,
                Some(&b.fc1_b),
                Epilogue::None,
                &mut w.hidden,
                GemmShape::packed(rows, HIDDEN, DIM),
            )?;
            k.rowops
                .activate(nn, &mut w.hidden, rows * HIDDEN, Act::GeluErf)?;
            k.gemm.linear_bf16(
                nn,
                &w.hidden,
                &b.fc2_w,
                None,
                Epilogue::None,
                &mut w.y,
                GemmShape::packed(rows, DIM, HIDDEN),
            )?;
            let next = match self.blocks.get(i + 1) {
                Some(nb) => &nb.norm1,
                None => &self.trunk_norm,
            };
            let br = Branch {
                y: &w.y,
                tok: None,
                chan: None,
                bias: Some(&b.fc2_b),
                layer_scale: Some(&b.ls2),
            };
            k.norm.residual_layer_norm(
                nn,
                RowsMut::F32(&mut w.x),
                &br,
                rows,
                DIM,
                &next.params(),
                Some(&mut w.hn),
                Quant::None,
            )?;
            k.taps.f32(nn, &format!("cam_block{i}"), &w.x, rows * DIM)?;
        }

        // Camera token of each frame = row f·17 of the trunk_norm output.
        let shape = GemmShape {
            m: n,
            n: BRANCH,
            k: DIM,
            ldx: SPECIAL * DIM,
            ldy: BRANCH,
        };
        k.gemm.linear_bf16(
            nn,
            &w.hn,
            &self.b0_w,
            Some(&self.b0_b),
            Epilogue::None,
            &mut w.branch,
            shape,
        )?;
        k.rowops
            .activate(nn, &mut w.branch, n * BRANCH, Act::GeluErf)?;
        self.finish(nn, &w.branch, cams, g)
    }

    fn widen(
        &self,
        nn: &Nn,
        src: &CudaSlice<u16>,
        dst: &mut CudaSlice<f32>,
        len: usize,
    ) -> Result<()> {
        let (s, _g1) = src.device_ptr(&nn.stream);
        let (d, _g2) = spl_nn::cudarc::driver::DevicePtrMut::device_ptr_mut(dst, &nn.stream);
        let n = len as i64;
        let blocks = len.div_ceil(256).min(nn.multiprocessors() as usize * 16) as u32;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = nn.stream.launch_builder(&self.k.widen);
        b.arg(&s).arg(&d).arg(&n);
        // SAFETY: both buffers hold `len` elements (allocated by `Work`).
        unsafe { b.launch(cfg) }.map_err(|e| spl_core::anyhow!("widen_bf16: {e:?}"))?;
        Ok(())
    }

    fn finish(
        &self,
        nn: &Nn,
        h: &CudaSlice<u16>,
        cams: &mut DeviceCams,
        g: &Geometry,
    ) -> Result<()> {
        let s = &nn.stream;
        let (hp, _a) = h.device_ptr(s);
        let (w2, _b) = self.b2_w.device_ptr(s);
        let (b2, _c) = self.b2_b.device_ptr(s);
        use spl_nn::cudarc::driver::DevicePtrMut;
        let (pose, _d) = cams.pose_enc.device_ptr_mut(s);
        let (extr, _e) = cams.extrinsics.device_ptr_mut(s);
        let (intr, _f) = cams.intrinsics.device_ptr_mut(s);
        let a = CamArgs {
            h: hp,
            w2,
            b2,
            pose,
            extr,
            intr,
            frames: g.frames as i32,
            width: g.width as i32,
            height: g.height as i32,
            hidden: BRANCH as i32,
        };
        let cfg = LaunchConfig {
            grid_dim: (g.frames as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = s.launch_builder(&self.k.camera_finish);
        b.arg(&a);
        // SAFETY: `a` matches `CamArgs`; buffer sizes checked in `enqueue`.
        unsafe { b.launch(cfg) }.map_err(|e| spl_core::anyhow!("camera_finish: {e:?}"))?;
        Ok(())
    }
}
