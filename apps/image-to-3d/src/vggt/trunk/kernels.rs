//! Launchers for `cuda/vggt_trunk.cu`: patchify, token assembly, row moves
//! between frame layouts, the fp32 LayerNorm, bf16 cache writes.

use spl_core::{Result, anyhow, bail};
use spl_nn::Nn;
use spl_nn::cudarc::driver::{
    CudaFunction, DevicePtr, DevicePtrMut, DeviceRepr, LaunchConfig, PushKernelArg,
};

use super::DIM;

const SOURCE: &str = include_str!("../../../cuda/vggt_trunk.cu");

/// Row `p` of frame `f` lives at `f · frame_len + offset + p`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct RowMap {
    pub frame_len: i32,
    pub offset: i32,
}

// SAFETY: plain old data matching `struct RowMap` in cuda/vggt_trunk.cu.
unsafe impl DeviceRepr for RowMap {}

impl RowMap {
    pub fn new(frame_len: usize, offset: usize) -> Self {
        Self {
            frame_len: frame_len as i32,
            offset: offset as i32,
        }
    }
}

pub(crate) struct TrunkKernels {
    patchify: CudaFunction,
    put_prefix: CudaFunction,
    put_bf16_rows: CudaFunction,
    copy_rows: CudaFunction,
    ln_rows_f32: CudaFunction,
    to_cache: CudaFunction,
    rows_to_wide: CudaFunction,
    sms: u32,
}

fn grid(nn_sms: u32, work: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (
            work.div_ceil(256).clamp(1, nn_sms as usize * 16) as u32,
            1,
            1,
        ),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn check(len: usize, need: usize, what: &str) -> Result<()> {
    if len < need {
        bail!("{what}: buffer holds {len} elements, needs {need}");
    }
    Ok(())
}

impl TrunkKernels {
    pub fn new(nn: &Nn) -> Result<Self> {
        let src = [spl_nn::COMMON_CUH, SOURCE].concat();
        let m = nn.module("i3d_vggt_trunk", &src)?;
        let f = |n: &str| {
            m.load_function(n)
                .map_err(|e| anyhow!("load kernel {n}: {e:?}"))
        };
        Ok(Self {
            patchify: f("patchify")?,
            put_prefix: f("put_prefix")?,
            put_bf16_rows: f("put_bf16_rows")?,
            copy_rows: f("copy_rows")?,
            ln_rows_f32: f("ln_rows_f32")?,
            to_cache: f("to_cache")?,
            rows_to_wide: f("rows_to_wide")?,
            sms: nn.multiprocessors(),
        })
    }

    /// `img` f32 `[n][3][h][w]` → `out` bf16 `[n · h/16 · w/16][768]`.
    pub fn patchify(
        &self,
        nn: &Nn,
        img: &dyn DevicePtr<f32>,
        out: &mut dyn DevicePtrMut<u16>,
        n: usize,
        h: usize,
        w: usize,
    ) -> Result<()> {
        let rows = n * (h / 16) * (w / 16);
        check(img.len(), n * 3 * h * w, "patchify img")?;
        check(out.len(), rows * 768, "patchify out")?;
        let (ip, _g1) = img.device_ptr(&nn.stream);
        let (op, _g2) = out.device_ptr_mut(&nn.stream);
        let (n, h, w) = (n as i32, h as i32, w as i32);
        let mut b = nn.stream.launch_builder(&self.patchify);
        b.arg(&ip).arg(&op).arg(&n).arg(&h).arg(&w);
        // SAFETY: argument types match the kernel; sizes checked above.
        unsafe { b.launch(grid(self.sms, rows * 96)) }.map_err(|e| anyhow!("patchify: {e:?}"))?;
        Ok(())
    }

    /// Writes `count` rows per frame from `prefix` (`[variants][count][1024]`).
    #[allow(clippy::too_many_arguments)]
    pub fn put_prefix(
        &self,
        nn: &Nn,
        prefix: &dyn DevicePtr<f32>,
        x: &mut dyn DevicePtrMut<f32>,
        n: usize,
        count: usize,
        variants: usize,
        dst: RowMap,
    ) -> Result<()> {
        check(prefix.len(), variants * count * DIM, "put_prefix prefix")?;
        check(x.len(), rows_needed(n, count, dst) * DIM, "put_prefix x")?;
        let (pp, _g1) = prefix.device_ptr(&nn.stream);
        let (xp, _g2) = x.device_ptr_mut(&nn.stream);
        let (ni, ci, vi) = (n as i32, count as i32, variants as i32);
        let mut b = nn.stream.launch_builder(&self.put_prefix);
        b.arg(&pp).arg(&xp).arg(&ni).arg(&ci).arg(&vi).arg(&dst);
        // SAFETY: as above.
        unsafe { b.launch(grid(self.sms, n * count * DIM / 4)) }
            .map_err(|e| anyhow!("put_prefix: {e:?}"))?;
        Ok(())
    }

    /// Packed bf16 rows `[n · count][1024]` → f32 rows at `dst`.
    pub fn put_bf16_rows(
        &self,
        nn: &Nn,
        src: &dyn DevicePtr<u16>,
        x: &mut dyn DevicePtrMut<f32>,
        n: usize,
        count: usize,
        dst: RowMap,
    ) -> Result<()> {
        check(src.len(), n * count * DIM, "put_bf16_rows src")?;
        check(x.len(), rows_needed(n, count, dst) * DIM, "put_bf16_rows x")?;
        let (sp, _g1) = src.device_ptr(&nn.stream);
        let (xp, _g2) = x.device_ptr_mut(&nn.stream);
        let (ni, ci) = (n as i32, count as i32);
        let mut b = nn.stream.launch_builder(&self.put_bf16_rows);
        b.arg(&sp).arg(&xp).arg(&ni).arg(&ci).arg(&dst);
        // SAFETY: as above.
        unsafe { b.launch(grid(self.sms, n * count * DIM / 8)) }
            .map_err(|e| anyhow!("put_bf16_rows: {e:?}"))?;
        Ok(())
    }

    /// f32 rows `src` → `dst`, `count` per frame.
    #[allow(clippy::too_many_arguments)]
    pub fn copy_rows(
        &self,
        nn: &Nn,
        src: &dyn DevicePtr<f32>,
        dst: &mut dyn DevicePtrMut<f32>,
        n: usize,
        count: usize,
        sm: RowMap,
        dm: RowMap,
    ) -> Result<()> {
        check(src.len(), rows_needed(n, count, sm) * DIM, "copy_rows src")?;
        check(dst.len(), rows_needed(n, count, dm) * DIM, "copy_rows dst")?;
        let (sp, _g1) = src.device_ptr(&nn.stream);
        let (dp, _g2) = dst.device_ptr_mut(&nn.stream);
        let (ni, ci) = (n as i32, count as i32);
        let mut b = nn.stream.launch_builder(&self.copy_rows);
        b.arg(&sp).arg(&dp).arg(&ni).arg(&ci).arg(&sm).arg(&dm);
        // SAFETY: as above.
        unsafe { b.launch(grid(self.sms, n * count * DIM / 4)) }
            .map_err(|e| anyhow!("copy_rows: {e:?}"))?;
        Ok(())
    }

    /// LayerNorm with f32 output from `src` rows to `dst` rows.
    #[allow(clippy::too_many_arguments)]
    pub fn ln_rows_f32(
        &self,
        nn: &Nn,
        src: &dyn DevicePtr<f32>,
        dst: &mut dyn DevicePtrMut<f32>,
        gamma: &dyn DevicePtr<f32>,
        beta: &dyn DevicePtr<f32>,
        n: usize,
        count: usize,
        sm: RowMap,
        dm: RowMap,
        eps: f32,
    ) -> Result<()> {
        check(
            src.len(),
            rows_needed(n, count, sm) * DIM,
            "ln_rows_f32 src",
        )?;
        check(
            dst.len(),
            rows_needed(n, count, dm) * DIM,
            "ln_rows_f32 dst",
        )?;
        check(gamma.len(), DIM, "ln_rows_f32 gamma")?;
        check(beta.len(), DIM, "ln_rows_f32 beta")?;
        let (sp, _g1) = src.device_ptr(&nn.stream);
        let (dp, _g2) = dst.device_ptr_mut(&nn.stream);
        let (gp, _g3) = gamma.device_ptr(&nn.stream);
        let (bp, _g4) = beta.device_ptr(&nn.stream);
        let (ni, ci) = (n as i32, count as i32);
        let rows = n * count;
        let mut b = nn.stream.launch_builder(&self.ln_rows_f32);
        b.arg(&sp)
            .arg(&dp)
            .arg(&gp)
            .arg(&bp)
            .arg(&ni)
            .arg(&ci)
            .arg(&sm)
            .arg(&dm)
            .arg(&eps);
        let cfg = LaunchConfig {
            grid_dim: (rows.div_ceil(4) as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: as above; one warp per row.
        unsafe { b.launch(cfg) }.map_err(|e| anyhow!("ln_rows_f32: {e:?}"))?;
        Ok(())
    }

    /// f32 `[rows][1024]` → bf16 columns `col..col + 1024` of `[rows][ld]`.
    pub fn to_cache(
        &self,
        nn: &Nn,
        x: &dyn DevicePtr<f32>,
        cache: &mut dyn DevicePtrMut<u16>,
        rows: usize,
        ld: usize,
        col: usize,
    ) -> Result<()> {
        check(x.len(), rows * DIM, "to_cache x")?;
        check(cache.len(), rows * ld, "to_cache cache")?;
        if col + DIM > ld || !col.is_multiple_of(8) || !ld.is_multiple_of(8) {
            bail!("to_cache: column {col} / ld {ld} out of range");
        }
        let (xp, _g1) = x.device_ptr(&nn.stream);
        let (cp, _g2) = cache.device_ptr_mut(&nn.stream);
        let (r, l, c) = (rows as i64, ld as i32, col as i32);
        let mut b = nn.stream.launch_builder(&self.to_cache);
        b.arg(&xp).arg(&cp).arg(&r).arg(&l).arg(&c);
        // SAFETY: as above.
        unsafe { b.launch(grid(self.sms, rows * DIM / 8)) }
            .map_err(|e| anyhow!("to_cache: {e:?}"))?;
        Ok(())
    }
}

impl TrunkKernels {
    /// `count` rows per frame at `sm` → f32 columns `col..col + 1024` of
    /// packed `[n · count][ld]`.
    #[allow(clippy::too_many_arguments)]
    pub fn rows_to_wide(
        &self,
        nn: &Nn,
        x: &dyn DevicePtr<f32>,
        out: &mut dyn DevicePtrMut<f32>,
        n: usize,
        count: usize,
        sm: RowMap,
        ld: usize,
        col: usize,
    ) -> Result<()> {
        check(x.len(), rows_needed(n, count, sm) * DIM, "rows_to_wide x")?;
        check(out.len(), n * count * ld, "rows_to_wide out")?;
        if col + DIM > ld || !col.is_multiple_of(4) || !ld.is_multiple_of(4) {
            bail!("rows_to_wide: column {col} / ld {ld} out of range");
        }
        let (xp, _g1) = x.device_ptr(&nn.stream);
        let (op, _g2) = out.device_ptr_mut(&nn.stream);
        let (ni, ci, l, c) = (n as i32, count as i32, ld as i32, col as i32);
        let mut b = nn.stream.launch_builder(&self.rows_to_wide);
        b.arg(&xp)
            .arg(&op)
            .arg(&ni)
            .arg(&ci)
            .arg(&sm)
            .arg(&l)
            .arg(&c);
        // SAFETY: argument types match the kernel; sizes checked above.
        unsafe { b.launch(grid(self.sms, n * count * DIM / 4)) }
            .map_err(|e| anyhow!("rows_to_wide: {e:?}"))?;
        Ok(())
    }
}

/// Rows a row map touches for `n` frames of `count` rows.
fn rows_needed(n: usize, count: usize, m: RowMap) -> usize {
    if n == 0 || count == 0 {
        return 0;
    }
    (n - 1) * m.frame_len as usize + m.offset as usize + count
}
