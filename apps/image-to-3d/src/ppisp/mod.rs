//! PPISP (arXiv 2601.18336; model of nv-tlabs/ppisp, Apache-2.0): a
//! physically plausible ISP applied to the rendered image before the loss.
//! Owner: ppisp. Kernels: `cuda/ppisp.cu`.
//!
//! Per frame: exposure `2^e` and a colour homography on (r, g, r+g+b) from 8
//! latent chromaticity offsets. Per camera: radial vignetting per channel
//! (centre + 3 even coefficients, clamped to [0, 1]) and a toe/shoulder/gamma
//! response curve per channel. Forward and backward are one kernel each over
//! the planar `[3][H][W]` image; the backward writes dL/d(render) in place and
//! per-block parameter partials, reduced deterministically by a one-block
//! finalize that also maps dL/dH to the colour latents (forward-mode duals).
//! The regulariser and the optimizer (Adam, linear warm-up then exponential
//! decay, one group like upstream) are single small kernels.
//!
//! Novel views: upstream trains a per-camera CNN controller (distilled during
//! the last 20 % of training) that predicts exposure and colour for frames
//! without parameters. We use the documented upstream fallback instead (the
//! `use_controller = False` path): per-camera vignetting and CRF, identity
//! per-frame correction ([`NovelView::Identity`]), or a chosen training
//! frame's correction ([`NovelView::Frame`]). [`Ppisp::forward_override`]
//! takes externally predicted per-frame values and returns their gradient, the
//! hook a controller plugs into.
//!
//! Trainer integration (see `reports/image-to-3d/spec-train.md` 9 and the
//! ppisp agent's summary): after `raster.forward` call [`Ppisp::forward`]
//! (camera = `intr_id`, frame = training image index) into a second planar
//! buffer the loss reads; after the loss backward call [`Ppisp::backward`]
//! with `v_out = v_in = PixelGrads::d_rgb` (in place) before
//! `raster.backward`; once per iteration [`Ppisp::regularize`] then
//! [`Ppisp::step`]. Eval renders use [`Ppisp::forward_novel`].

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use spl_core::{Result, bail};
use spl_gpu::cudarc::driver::{CudaFunction, CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg};
use ts_rs::TS;

use crate::gpu::{CudaFile, CudaResultExt, Defines, Gpu, KernelSource};

pub const SOURCE: KernelSource = KernelSource {
    module: "ppisp",
    files: &[CudaFile {
        name: "ppisp.cu",
        text: include_str!("../../cuda/ppisp.cu"),
    }],
    kernels: &[
        "ppisp_prepare",
        "ppisp_fwd",
        "ppisp_bwd",
        "ppisp_bwd_finalize",
        "ppisp_reg",
        "ppisp_adam",
    ],
};

/// `PPISP_BLOCK` and `PPISP_NG` of `ppisp.cu`.
pub const BLOCK: u32 = 256;
pub const PARTIAL_COLS: usize = 37;
const CONSTS: usize = 48;

/// Upstream `PPISPConfig` defaults (regulariser weights, optimizer, schedule).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema, TS)]
#[serde(default)]
#[ts(export, export_to = "image-to-3d/")]
pub struct PpispConfig {
    pub exposure_mean: f32,
    pub vig_center: f32,
    pub vig_channel: f32,
    pub vig_non_pos: f32,
    pub color_mean: f32,
    pub crf_channel: f32,
    pub lr: f32,
    pub eps: f32,
    pub betas: (f32, f32),
    pub warmup_steps: u32,
    pub start_factor: f32,
    pub decay_max_steps: u32,
    pub final_factor: f32,
}

impl Default for PpispConfig {
    fn default() -> Self {
        Self {
            exposure_mean: 1.0,
            vig_center: 0.02,
            vig_channel: 0.1,
            vig_non_pos: 0.01,
            color_mean: 1.0,
            crf_channel: 0.1,
            lr: 0.002,
            eps: 1e-15,
            betas: (0.9, 0.999),
            warmup_steps: 500,
            start_factor: 0.01,
            decay_max_steps: 30_000,
            final_factor: 0.01,
        }
    }
}

impl PpispConfig {
    /// LR of optimizer step `step` (0-based), as torch's
    /// `SequentialLR(LinearLR(start_factor, warmup), ExponentialLR(gamma))`.
    pub fn lr_at(&self, step: u32) -> f32 {
        let w = self.warmup_steps;
        if step < w {
            let t = step as f32 / w as f32;
            self.lr * (self.start_factor + (1.0 - self.start_factor) * t)
        } else {
            let gamma = (self.final_factor as f64).powf(1.0 / self.decay_max_steps as f64);
            (self.lr as f64 * gamma.powi((step - w) as i32)) as f32
        }
    }
}

/// What per-frame correction a view without a trained frame gets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NovelView {
    /// Zero exposure, identity colour (upstream without controller).
    Identity,
    /// Reuse training frame `i`'s exposure and colour (e.g. the nearest pose).
    Frame(u32),
}

/// Offsets into the one parameter buffer; mirrors `PpispLayout` in `ppisp.cu`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Layout {
    pub n_frames: i32,
    pub n_cameras: i32,
    pub off_exposure: i32,
    pub off_vig: i32,
    pub off_color: i32,
    pub off_crf: i32,
}
// SAFETY: plain repr(C) i32s, the same as the CUDA struct.
unsafe impl DeviceRepr for Layout {}

impl Layout {
    pub fn new(n_cameras: usize, n_frames: usize) -> Self {
        let (f, c) = (n_frames as i32, n_cameras as i32);
        Self {
            n_frames: f,
            n_cameras: c,
            off_exposure: 0,
            off_vig: f,
            off_color: f + 15 * c,
            off_crf: f + 15 * c + 8 * f,
        }
    }

    pub fn len(&self) -> usize {
        (self.off_crf + 12 * self.n_cameras) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Upstream initial values: zeros, CRF raw = softplus⁻¹ of toe/shoulder
    /// 1 − 0.3, gamma 1 − 0.1, centre logit 0 (an identity-like curve).
    pub fn initial(&self) -> Vec<f32> {
        let mut p = vec![0.0f32; self.len()];
        let spi = |x: f64| (x.exp_m1()).ln() as f32;
        for c in 0..self.n_cameras as usize * 3 {
            let o = self.off_crf as usize + 4 * c;
            p[o] = spi(0.7);
            p[o + 1] = spi(0.7);
            p[o + 2] = spi(0.9);
        }
        p
    }
}

/// PPISP state on the device: parameters, gradients, Adam moments, scratch.
pub struct Ppisp {
    pub cfg: PpispConfig,
    pub layout: Layout,
    pub params: CudaSlice<f32>,
    pub grads: CudaSlice<f32>,
    m: CudaSlice<f32>,
    v: CudaSlice<f32>,
    consts: CudaSlice<f32>,
    partial: CudaSlice<f32>,
    /// Exposure + 8 colour values of the override path, and their gradient.
    pub override_ec: CudaSlice<f32>,
    pub grad_override: CudaSlice<f32>,
    /// Last regulariser value (device scalar; read only for logging).
    pub reg_loss: CudaSlice<f32>,
    steps: u32,
    last_override: bool,
    k: Kernels,
}

struct Kernels {
    prepare: CudaFunction,
    fwd: CudaFunction,
    bwd: CudaFunction,
    finalize: CudaFunction,
    reg: CudaFunction,
    adam: CudaFunction,
}

impl Ppisp {
    /// `max_pixels` sizes the partial buffer (largest render, W·H).
    pub fn new(
        gpu: &Gpu,
        n_cameras: usize,
        n_frames: usize,
        max_pixels: usize,
        cfg: PpispConfig,
    ) -> Result<Self> {
        if n_cameras == 0 || n_frames == 0 {
            bail!("ppisp: needs at least one camera and one frame");
        }
        let ks = gpu.kernels(&SOURCE, &Defines::new())?;
        let k = Kernels {
            prepare: ks.get("ppisp_prepare")?,
            fwd: ks.get("ppisp_fwd")?,
            bwd: ks.get("ppisp_bwd")?,
            finalize: ks.get("ppisp_bwd_finalize")?,
            reg: ks.get("ppisp_reg")?,
            adam: ks.get("ppisp_adam")?,
        };
        let layout = Layout::new(n_cameras, n_frames);
        let s = &gpu.stream;
        let n = layout.len();
        let blocks = max_pixels.div_ceil(BLOCK as usize).max(1);
        Ok(Self {
            params: s.clone_htod(&layout.initial()).cuda("ppisp params")?,
            grads: s.alloc_zeros(n).cuda("ppisp grads")?,
            m: s.alloc_zeros(n).cuda("ppisp adam m")?,
            v: s.alloc_zeros(n).cuda("ppisp adam v")?,
            consts: s.alloc_zeros(CONSTS).cuda("ppisp consts")?,
            partial: s
                .alloc_zeros(blocks * PARTIAL_COLS)
                .cuda("ppisp partials")?,
            override_ec: s.alloc_zeros(9).cuda("ppisp override")?,
            grad_override: s.alloc_zeros(9).cuda("ppisp override grad")?,
            reg_loss: s.alloc_zeros(1).cuda("ppisp reg loss")?,
            cfg,
            layout,
            steps: 0,
            last_override: false,
            k,
        })
    }

    /// Device bytes held.
    pub fn bytes(&self) -> usize {
        4 * (4 * self.layout.len() + CONSTS + self.partial.len() + 19)
    }

    fn check(&self, camera: i32, frame: i32, w: u32, h: u32, bufs: &[usize]) -> Result<()> {
        let n = 3 * (w as usize) * (h as usize);
        if camera >= self.layout.n_cameras || frame >= self.layout.n_frames {
            bail!("ppisp: camera {camera} / frame {frame} out of range");
        }
        if (w as usize * h as usize).div_ceil(BLOCK as usize) * PARTIAL_COLS > self.partial.len() {
            bail!("ppisp: {w}x{h} exceeds max_pixels");
        }
        if bufs.iter().any(|&l| l < n) {
            bail!("ppisp: image buffers smaller than 3x{w}x{h}");
        }
        Ok(())
    }

    fn prepare(&mut self, gpu: &Gpu, camera: i32, frame: i32, over: bool) -> Result<()> {
        self.last_override = over;
        let mut l = gpu.stream.launch_builder(&self.k.prepare);
        let ov = over as i32;
        l.arg(&self.params)
            .arg(&self.layout)
            .arg(&camera)
            .arg(&frame)
            .arg(&ov)
            .arg(&self.override_ec)
            .arg(&mut self.consts);
        // SAFETY: matches ppisp_prepare; indices checked by the caller.
        unsafe { l.launch(LaunchConfig::for_num_elems(1)) }.cuda("ppisp_prepare")?;
        Ok(())
    }

    fn launch_fwd(
        &mut self,
        gpu: &Gpu,
        rgb_in: &CudaSlice<f32>,
        rgb_out: &mut CudaSlice<f32>,
        w: u32,
        h: u32,
    ) -> Result<()> {
        let (wi, hi) = (w as i32, h as i32);
        let mut l = gpu.stream.launch_builder(&self.k.fwd);
        l.arg(&self.consts)
            .arg(rgb_in)
            .arg(rgb_out)
            .arg(&wi)
            .arg(&hi);
        let cfg = LaunchConfig {
            grid_dim: ((w * h).div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: buffers hold 3·w·h floats (checked).
        unsafe { l.launch(cfg) }.cuda("ppisp_fwd")?;
        Ok(())
    }

    /// Training-view forward: `camera` (per-camera index, `-1` off), `frame`
    /// (training image index, `-1` off). Planar `[3][h][w]` in and out.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &mut self,
        gpu: &Gpu,
        camera: i32,
        frame: i32,
        rgb_in: &CudaSlice<f32>,
        rgb_out: &mut CudaSlice<f32>,
        w: u32,
        h: u32,
    ) -> Result<()> {
        self.check(camera, frame, w, h, &[rgb_in.len(), rgb_out.len()])?;
        self.prepare(gpu, camera, frame, false)?;
        self.launch_fwd(gpu, rgb_in, rgb_out, w, h)
    }

    /// Forward for a view without per-frame parameters (eval / novel views).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_novel(
        &mut self,
        gpu: &Gpu,
        camera: i32,
        mode: NovelView,
        rgb_in: &CudaSlice<f32>,
        rgb_out: &mut CudaSlice<f32>,
        w: u32,
        h: u32,
    ) -> Result<()> {
        let frame = match mode {
            NovelView::Identity => -1,
            NovelView::Frame(i) => i as i32,
        };
        self.check(camera, frame, w, h, &[rgb_in.len(), rgb_out.len()])?;
        // Identity: frame effects with zero latents, i.e. upstream's override.
        gpu.stream
            .memset_zeros(&mut self.override_ec)
            .cuda("ppisp zero override")?;
        self.prepare(gpu, camera, frame, frame < 0)?;
        self.launch_fwd(gpu, rgb_in, rgb_out, w, h)
    }

    /// Forward with externally supplied per-frame values in `override_ec`
    /// (exposure, 8 colour latents), e.g. from a controller.
    pub fn forward_override(
        &mut self,
        gpu: &Gpu,
        camera: i32,
        rgb_in: &CudaSlice<f32>,
        rgb_out: &mut CudaSlice<f32>,
        w: u32,
        h: u32,
    ) -> Result<()> {
        self.check(camera, -1, w, h, &[rgb_in.len(), rgb_out.len()])?;
        self.prepare(gpu, camera, -1, true)?;
        self.launch_fwd(gpu, rgb_in, rgb_out, w, h)
    }

    /// Backward of the last forward's configuration. `v_out` = dL/d(output);
    /// writes dL/d(rgb_in) into `v_in` (pass the same buffer for in place:
    /// use [`Ppisp::backward_inplace`]) and adds parameter gradients into
    /// `grads` (or `grad_override` for the override path).
    #[allow(clippy::too_many_arguments)]
    pub fn backward(
        &mut self,
        gpu: &Gpu,
        camera: i32,
        frame: i32,
        rgb_in: &CudaSlice<f32>,
        v_out: &CudaSlice<f32>,
        v_in: &mut CudaSlice<f32>,
        w: u32,
        h: u32,
    ) -> Result<()> {
        self.check(
            camera,
            frame,
            w,
            h,
            &[rgb_in.len(), v_out.len(), v_in.len()],
        )?;
        let (wi, hi) = (w as i32, h as i32);
        let mut l = gpu.stream.launch_builder(&self.k.bwd);
        l.arg(&self.consts)
            .arg(rgb_in)
            .arg(v_out)
            .arg(v_in)
            .arg(&mut self.partial)
            .arg(&wi)
            .arg(&hi);
        // SAFETY: buffers checked; partial holds one row per block.
        unsafe { l.launch(bwd_cfg(w, h)) }.cuda("ppisp_bwd")?;
        self.finalize(gpu, camera, frame, w, h)
    }

    /// [`Ppisp::backward`] with dL/d(output) replaced by dL/d(input) in `v`.
    #[allow(clippy::too_many_arguments)]
    pub fn backward_inplace(
        &mut self,
        gpu: &Gpu,
        camera: i32,
        frame: i32,
        rgb_in: &CudaSlice<f32>,
        v: &mut CudaSlice<f32>,
        w: u32,
        h: u32,
    ) -> Result<()> {
        self.check(camera, frame, w, h, &[rgb_in.len(), v.len()])?;
        let (wi, hi) = (w as i32, h as i32);
        let mut l = gpu.stream.launch_builder(&self.k.bwd);
        // Each thread reads its own v_out pixel before writing v_in there.
        let vp = &*v;
        l.arg(&self.consts)
            .arg(rgb_in)
            .arg(vp)
            .arg(vp)
            .arg(&mut self.partial)
            .arg(&wi)
            .arg(&hi);
        // SAFETY: buffers checked; partial holds one row per block.
        unsafe { l.launch(bwd_cfg(w, h)) }.cuda("ppisp_bwd")?;
        self.finalize(gpu, camera, frame, w, h)
    }

    fn finalize(&self, gpu: &Gpu, camera: i32, frame: i32, w: u32, h: u32) -> Result<()> {
        let blocks = (w * h).div_ceil(BLOCK);
        // Same per-frame source as the forward that preceded this backward.
        let over = (frame < 0 && self.last_override) as i32;
        let nb = blocks as i32;
        let mut f = gpu.stream.launch_builder(&self.k.finalize);
        f.arg(&self.partial)
            .arg(&nb)
            .arg(&self.params)
            .arg(&self.layout)
            .arg(&camera)
            .arg(&frame)
            .arg(&over)
            .arg(&self.override_ec)
            .arg(&self.consts)
            .arg(&self.grads)
            .arg(&self.grad_override);
        let cfg1 = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: matches ppisp_bwd_finalize; grads hold layout.len() floats.
        unsafe { f.launch(cfg1) }.cuda("ppisp_bwd_finalize")?;
        Ok(())
    }

    /// Adds the regulariser's gradient into `grads`; value in `reg_loss`.
    pub fn regularize(&mut self, gpu: &Gpu) -> Result<()> {
        let c = &self.cfg;
        let mut l = gpu.stream.launch_builder(&self.k.reg);
        l.arg(&self.params)
            .arg(&self.layout)
            .arg(&c.exposure_mean)
            .arg(&c.vig_center)
            .arg(&c.vig_channel)
            .arg(&c.vig_non_pos)
            .arg(&c.color_mean)
            .arg(&c.crf_channel)
            .arg(&mut self.grads)
            .arg(&mut self.reg_loss);
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: matches ppisp_reg.
        unsafe { l.launch(cfg) }.cuda("ppisp_reg")?;
        Ok(())
    }

    /// One Adam step over all PPISP parameters; zeroes the gradients.
    pub fn step(&mut self, gpu: &Gpu) -> Result<()> {
        let lr = self.cfg.lr_at(self.steps);
        self.steps += 1;
        let t = self.steps as i32;
        let (b1, b2) = self.cfg.betas;
        let bc1 = 1.0 - b1.powi(t);
        let bc2 = 1.0 - b2.powi(t);
        let (lo, hi) = (0i32, self.layout.len() as i32);
        let eps = self.cfg.eps;
        let mut l = gpu.stream.launch_builder(&self.k.adam);
        l.arg(&mut self.params)
            .arg(&mut self.grads)
            .arg(&mut self.m)
            .arg(&mut self.v)
            .arg(&lo)
            .arg(&hi)
            .arg(&lr)
            .arg(&b1)
            .arg(&b2)
            .arg(&eps)
            .arg(&bc1)
            .arg(&bc2);
        // SAFETY: all four buffers hold layout.len() floats.
        unsafe { l.launch(LaunchConfig::for_num_elems(hi as u32)) }.cuda("ppisp_adam")?;
        Ok(())
    }

    pub fn steps(&self) -> u32 {
        self.steps
    }

    pub fn download(&self, gpu: &Gpu) -> Result<Vec<f32>> {
        gpu.stream.clone_dtoh(&self.params).cuda("ppisp download")
    }

    pub fn upload(&mut self, gpu: &Gpu, p: &[f32]) -> Result<()> {
        if p.len() != self.layout.len() {
            bail!(
                "ppisp: {} parameters, expected {}",
                p.len(),
                self.layout.len()
            );
        }
        gpu.stream
            .memcpy_htod(p, &mut self.params)
            .cuda("ppisp upload")
    }
}

fn bwd_cfg(w: u32, h: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((w * h).div_ceil(BLOCK), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

#[cfg(test)]
mod tests;
