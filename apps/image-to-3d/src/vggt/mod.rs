//! Native VGGT-Ω engine (`reports/Image to 3D.md` section 3, Stage A;
//! `reports/image-to-3d/spec-vggt.md`). Owner: runtime (this file,
//! `weights.rs`, `preprocess.rs`, `workspace.rs`, `graph.rs`, `tests.rs`);
//! trunk (`trunk/*`) and heads (`heads/*`) have their own owners.
//!
//! Load once, infer many: [`VggtEngine::load`] maps the converted weights
//! and uploads the aggregator in one copy; [`VggtEngine::run`] takes frames
//! prepared by [`VggtEngine::preprocess`] and returns cameras plus device
//! depth, confidence and world points. Workspaces are kept per geometry, so
//! a second run at the same (N, H, W) allocates nothing.
//!
//! Weights: `vggt_omega_1b_512.pt` under the FAIR non-commercial research
//! licence (gated on Hugging Face): research use only, never redistributed
//! or baked into an image.

pub mod graph;
pub mod heads;
pub mod preprocess;
pub mod trunk;
pub mod weights;
pub mod workspace;

#[cfg(test)]
mod integration;
#[cfg(test)]
mod phase2_tests;
#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use serde::Serialize;
use spl_core::{CancellationToken, Result, anyhow, bail};
use spl_nn::Nn;
use spl_nn::cudarc::driver::{CudaEvent, CudaSlice};

pub use preprocess::{Prepared, ResizeMode};
pub use weights::Weights;
pub use workspace::{Geometry, Plan, Workspace};

use crate::io::images::Rgb8Image;

/// What one inter-frame layer runs. Data, not code (Report §3 A.6): Phase 2
/// fills these with quantised or merged variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InterAttn {
    /// All N·T tokens as one sequence (dense).
    Global,
    /// The 17 special tokens of every frame as one sequence; patches pass.
    Register,
    /// Per-frame attention instead of global (Phase 2).
    Frame,
    /// Global with 2×2 / 4×4 merged keys and values of frames 1.. and a
    /// +log(count) bias (Phase 2); runs on the SageAttention kernel.
    Merge2,
    Merge4,
    /// SageAttention2 INT8/FP8 global (Phase 2).
    Sage,
}

/// The official layer types: register attention at 2, 6, 9, 14, 20.
pub const REGISTER_LAYERS: [usize; 5] = [2, 6, 9, 14, 20];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AttnSchedule(pub [InterAttn; 24]);

impl AttnSchedule {
    pub fn dense() -> Self {
        let mut s = [InterAttn::Global; 24];
        for i in REGISTER_LAYERS {
            s[i] = InterAttn::Register;
        }
        Self(s)
    }
}

/// Global layers the Phase 2 schedule runs as frame attention
/// (`spec-attn-schedule.md` §1).
pub const FRAME_LAYERS: [usize; 3] = [3, 4, 5];
/// Global layers the Phase 2 schedule merges 4×4.
pub const MERGE4_LAYERS: [usize; 4] = [0, 1, 7, 19];
/// The extra merge4 layers of the opt-in `ours_wide` schedule.
pub const WIDE_MERGE4_LAYERS: [usize; 2] = [8, 22];

impl AttnSchedule {
    /// The shipped Phase 2 schedule (`ours`): frame at 3, 4, 5; merge4 at
    /// 0, 1, 7, 19; the other global layers dense, through SageAttention
    /// when `sage`.
    pub fn phase2(sage: bool) -> Self {
        let mut s = Self::dense().0;
        for a in s.iter_mut() {
            if sage && *a == InterAttn::Global {
                *a = InterAttn::Sage;
            }
        }
        for i in FRAME_LAYERS {
            s[i] = InterAttn::Frame;
        }
        for i in MERGE4_LAYERS {
            s[i] = InterAttn::Merge4;
        }
        Self(s)
    }

    /// `ours_wide` (opt-in fast): [`Self::phase2`] plus merge4 at 8 and 22.
    pub fn phase2_wide(sage: bool) -> Self {
        let mut s = Self::phase2(sage);
        for i in WIDE_MERGE4_LAYERS {
            s.0[i] = InterAttn::Merge4;
        }
        s
    }

    /// Parses `dense`, `sage` (dense + sage), `ours`, `ours+sage`,
    /// `wide`, `wide+sage`.
    pub fn named(name: &str) -> Result<Self> {
        Ok(match name {
            "dense" => Self::dense(),
            "sage" => {
                let mut s = Self::dense();
                for a in s.0.iter_mut() {
                    if *a == InterAttn::Global {
                        *a = InterAttn::Sage;
                    }
                }
                s
            }
            "ours" => Self::phase2(false),
            "ours+sage" => Self::phase2(true),
            "wide" => Self::phase2_wide(false),
            "wide+sage" => Self::phase2_wide(true),
            other => bail!("unknown attention schedule {other:?}"),
        })
    }
}

impl Default for AttnSchedule {
    fn default() -> Self {
        Self::dense()
    }
}

#[derive(Debug, Clone)]
pub struct VggtConfig {
    /// Longest side (max_size) or token budget (balanced); 512 for the 512
    /// checkpoint.
    pub resolution: usize,
    pub resize: ResizeMode,
    /// Frames per dense-head chunk (the official head uses 8; the result
    /// does not depend on it).
    pub dense_chunk: usize,
    pub schedule: AttnSchedule,
    /// Capture the trunk in a CUDA graph per (N, H, W) for N up to this.
    pub graph_max_frames: usize,
    /// Also load `<weights>_fp8.safetensors` into the arena (Phase 2;
    /// implied by a non-empty `fp8_blocks`).
    pub fp8: bool,
    /// Trunk blocks whose GEMMs run in FP8 (Phase 2).
    pub fp8_blocks: trunk::Fp8Blocks,
    /// SageAttention layers accumulate each 64-key P·V block in FP16
    /// (SageAttention2++, faster) instead of FP32.
    pub sage_pv16: bool,
    /// Force (or forbid) moving levels 0..3 to pinned host memory between
    /// the trunk and the dense head; `None` lets the planner decide.
    pub offload: Option<bool>,
}

impl Default for VggtConfig {
    fn default() -> Self {
        Self {
            resolution: 512,
            resize: ResizeMode::MaxSize,
            dense_chunk: 8,
            schedule: AttnSchedule::dense(),
            graph_max_frames: 32,
            fp8: false,
            fp8_blocks: trunk::Fp8Blocks::NONE,
            sage_pv16: false,
            offload: None,
        }
    }
}

/// Engine results. Cameras are on the host; dense maps stay on the device
/// (f16 bits for depth/conf, f32 world points) inside the engine's
/// workspace, read through [`VggtEngine::download`].
#[derive(Debug, Clone, Default, Serialize)]
pub struct VggtCameras {
    /// Raw head output `[T(3), quat xyzw(4), fov_h, fov_w]`.
    pub pose_enc: Vec<[f32; 9]>,
    /// World-to-camera `[R | t]` row-major, OpenCV axes; world = camera 0.
    pub extrinsics: Vec<[f32; 12]>,
    /// `[fx, 0, cx, 0, fy, cy, 0, 0, 1]` in preprocessed pixels.
    pub intrinsics: Vec<[f32; 9]>,
    pub width: usize,
    pub height: usize,
}

/// Host copies of the dense outputs.
#[derive(Debug, Clone, Default)]
pub struct DenseHost {
    pub depth: Vec<f32>,
    pub conf: Vec<f32>,
    pub world: Vec<f32>,
}

/// Seconds per stage of the last run (CUDA events, so GPU time).
#[derive(Debug, Clone, Default, Serialize)]
pub struct StageTimes {
    pub upload_ms: f32,
    pub trunk_ms: f32,
    pub camera_ms: f32,
    pub dense_ms: f32,
    pub total_ms: f32,
    /// Wall time of `run` including host work.
    pub wall_ms: f32,
    pub graph: bool,
}

/// Progress callback: `(stage, fraction of the run)`.
pub type Progress<'a> = &'a mut dyn FnMut(&str, f32);

pub struct VggtEngine {
    pub nn: Arc<Nn>,
    pub weights: Weights,
    pub cfg: VggtConfig,
    trunk: trunk::Trunk,
    camera: heads::camera::CameraHead,
    dense: heads::dense::DenseHead,
    normalize: spl_nn::cudarc::driver::CudaFunction,
    workspaces: HashMap<Geometry, Workspace>,
    cams: HashMap<usize, heads::DeviceCams>,
    graphs: graph::GraphCache,
    pub times: StageTimes,
}

const RUNTIME_CU: &str = include_str!("../../cuda/vggt_runtime.cu");

impl VggtEngine {
    /// Maps and uploads the weights and builds every component.
    pub fn load(nn: Arc<Nn>, weights: &std::path::Path, cfg: VggtConfig) -> Result<Self> {
        let ctx = nn.ctx.clone();
        let _untracked = graph::Untracked::new(&ctx);
        let mut paths = vec![weights.to_path_buf()];
        if cfg.fp8 || cfg.fp8_blocks != trunk::Fp8Blocks::NONE {
            paths.push(weights::fp8_companion(weights));
        }
        let w = Weights::load(&nn, &paths)?;
        if w.has("text_alignment_head.token_norm.weight") {
            bail!("text-alignment weights are not supported; convert without them");
        }
        let trunk = trunk::Trunk::new(&nn, &w)?;
        let kernels = heads::Kernels::new(&nn)?;
        let camera = heads::camera::CameraHead::new(&nn, kernels.clone(), &w)?;
        let dense = heads::dense::DenseHead::new(&nn, kernels, &w)?;
        let m = nn.module("vggt_runtime", RUNTIME_CU)?;
        let normalize = m
            .load_function("vggt_normalize")
            .map_err(|e| anyhow!("load vggt_normalize: {e:?}"))?;
        Ok(Self {
            nn,
            weights: w,
            cfg,
            trunk,
            camera,
            dense,
            normalize,
            workspaces: HashMap::new(),
            cams: HashMap::new(),
            graphs: graph::GraphCache::default(),
            times: StageTimes::default(),
        })
    }

    /// The weights next to a model directory: `<dir>/vggt_omega_1b_512.safetensors`.
    pub fn default_weights(dir: &std::path::Path) -> PathBuf {
        dir.join("vggt_omega_1b_512.safetensors")
    }

    /// CPU part of §2 (crop, PIL-exact bicubic resize).
    pub fn preprocess(&self, frames: &[Rgb8Image]) -> Result<Prepared> {
        preprocess::prepare(frames, self.cfg.resolution, self.cfg.resize)
    }

    /// Memory plan for `geom` from the device's free memory.
    pub fn plan(&self, geom: Geometry) -> Result<Plan> {
        let (free, _total) = spl_nn::cudarc::driver::result::mem_get_info()
            .map_err(|e| anyhow!("cuMemGetInfo: {e:?}"))?;
        // Workspaces are replaced by the new one (one geometry at a time),
        // so their memory counts as free.
        let held: usize = self.workspaces.values().map(|w| w.plan.device_bytes).sum();
        let margin = 256 << 20;
        let trunk = trunk::Trunk::scratch_bytes_for(
            &trunk_geom(&geom),
            &trunk_schedule(&self.cfg.schedule)?,
        );
        let chunk = geom.with_frames(self.cfg.dense_chunk.min(geom.frames));
        let rest = geom.frames % self.cfg.dense_chunk.max(1);
        // The dense head keeps work buffers for the full chunk and the
        // remainder.
        let head = heads::dense::DenseHead::scratch_bytes(&chunk)
            + if rest > 0 && geom.frames > rest {
                heads::dense::DenseHead::scratch_bytes(&geom.with_frames(rest))
            } else {
                0
            }
            + heads::camera::CameraHead::scratch_bytes(&geom);
        Plan::new(
            geom,
            self.cfg.dense_chunk,
            (free + held).saturating_sub(margin),
            trunk,
            head,
        )
    }

    fn workspace(&mut self, geom: Geometry) -> Result<&mut Workspace> {
        if !self.workspaces.contains_key(&geom) {
            let mut plan = self.plan(geom)?;
            if let Some(o) = self.cfg.offload {
                plan.offload = o;
            }
            // One geometry at a time: a new size frees the old buffers.
            self.workspaces.clear();
            self.graphs.clear();
            let ws = Workspace::new(&self.nn, plan)?;
            self.workspaces.insert(geom, ws);
        }
        let ws = self.workspaces.get_mut(&geom).expect("inserted");
        ws.restore_cache(&self.nn)?;
        Ok(ws)
    }

    fn event(&self) -> Result<CudaEvent> {
        self.nn
            .ctx
            .new_event(Some(
                spl_nn::cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT,
            ))
            .map_err(|e| anyhow!("create CUDA event: {e:?}"))
    }

    /// Runs the model on prepared frames. Results: the returned cameras and
    /// the workspace's depth / conf / world buffers ([`Self::download`]).
    pub fn run(
        &mut self,
        input: &Prepared,
        cancel: &CancellationToken,
        progress: Progress<'_>,
    ) -> Result<VggtCameras> {
        let wall = Instant::now();
        let g = input.geom;
        if g.frames == 0 {
            bail!("no frames");
        }
        if input.pixels.len() != g.frames * g.pixels() * 3 {
            bail!("prepared pixels do not match {g:?}");
        }
        let nn = self.nn.clone();
        let _untracked = graph::Untracked::new(&nn.ctx);
        let ev: Vec<CudaEvent> = (0..5).map(|_| self.event()).collect::<Result<_>>()?;
        progress("upload", 0.0);
        record(&nn, &ev[0])?;
        let normalize = self.normalize.clone();
        let ws = self.workspace(g)?;
        nn.stream
            .memcpy_htod(&input.pixels, &mut ws.pixels)
            .map_err(|e| anyhow!("upload frames: {e:?}"))?;
        normalize_launch(&nn, &normalize, ws, g)?;
        record(&nn, &ev[1])?;

        if cancel.is_cancelled() {
            bail!("cancelled");
        }
        progress("trunk", 0.02);
        let sched = trunk_schedule(&self.cfg.schedule)?;
        self.trunk.set_fp8_blocks(self.cfg.fp8_blocks);
        self.trunk.set_pv_accum(if self.cfg.sage_pv16 {
            trunk::sched::PvAccum::F16
        } else {
            trunk::sched::PvAccum::F32
        });
        let use_graph = g.frames <= self.cfg.graph_max_frames;
        let mut graphed = false;
        {
            let ws = self.workspaces.get_mut(&g).expect("workspace");
            let trunk = &mut self.trunk;
            let w = &self.weights;
            let tg = trunk_geom(&g);
            let Workspace {
                images,
                cache,
                cam_tokens,
                ..
            } = &mut *ws;
            let images = images.slice(..);
            let mut cam_tokens = cam_tokens.slice_mut(..);
            let [mut c0, mut c1, mut c2, mut c3] = workspace::cache_mut(cache)?;
            let mut body = |nn: &Nn| -> Result<()> {
                trunk.run(
                    nn,
                    w,
                    tg,
                    &images,
                    [&mut c0, &mut c1, &mut c2, &mut c3],
                    Some(&mut cam_tokens),
                    &sched,
                    &mut trunk::NoTap,
                )
            };
            if use_graph {
                graphed = self.graphs.run(&nn, (g, run_key(&self.cfg)), &mut body)?;
            } else {
                body(&nn)?;
            }
        }
        record(&nn, &ev[2])?;
        if cancel.is_cancelled() {
            bail!("cancelled");
        }

        progress("camera head", 0.75);
        let cams = match self.cams.entry(g.frames) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(heads::DeviceCams::new(&nn, g.frames)?)
            }
        };
        let ws = self.workspaces.get_mut(&g).expect("workspace");
        // The camera head reads the fp32 special tokens: on the bf16 cache
        // its fov drifts to about 2x the official bf16-vs-fp32 distance.
        let pose_enc = self
            .camera
            .run_f32(&nn, &ws.cam_tokens.slice(..), &g, cams)?;
        record(&nn, &ev[3])?;

        progress("dense head", 0.8);
        if ws.plan.release_trunk && !graphed && !use_graph {
            self.trunk.release_buffers();
        }
        ws.offload_levels(&nn)?;
        let chunks: Vec<_> = ws.plan.chunks().collect();
        let nchunks = chunks.len();
        for (ci, r) in chunks.into_iter().enumerate() {
            if cancel.is_cancelled() {
                bail!("cancelled");
            }
            let cg = g.with_frames(r.len());
            let px = g.pixels();
            // Split borrows: levels from the cache, outputs from their own
            // buffers.
            let Workspace {
                depth,
                conf,
                world,
                cache,
                host_cache,
                stage,
                ..
            } = &mut *ws;
            let mut d = depth.slice_mut(r.start * px..r.end * px);
            let mut c = conf.slice_mut(r.start * px..r.end * px);
            let mut wp = world.slice_mut(r.start * px * 3..r.end * px * 3);
            let ex = cams.extrinsics.slice(r.start * 12..r.end * 12);
            let ix = cams.intrinsics.slice(r.start * 9..r.end * 9);
            let levels = chunk_levels(&nn, cache, host_cache, stage, g, r.clone())?;
            self.dense.run_chunk(
                &nn,
                levels,
                &cg,
                &mut d,
                &mut c,
                Some((&ex, &ix)),
                Some(&mut wp),
            )?;
            progress("dense head", 0.8 + 0.2 * (ci + 1) as f32 / nchunks as f32);
        }
        record(&nn, &ev[4])?;
        ev[4]
            .synchronize()
            .map_err(|e| anyhow!("wait for the run: {e:?}"))?;
        let ms = |a: usize, b: usize| ev[a].elapsed_ms(&ev[b]).unwrap_or(f32::NAN);
        self.times = StageTimes {
            upload_ms: ms(0, 1),
            trunk_ms: ms(1, 2),
            camera_ms: ms(2, 3),
            dense_ms: ms(3, 4),
            total_ms: ms(0, 4),
            wall_ms: wall.elapsed().as_secs_f32() * 1e3,
            graph: graphed,
        };
        let mut out = VggtCameras {
            width: g.width,
            height: g.height,
            ..Default::default()
        };
        for pe in &pose_enc {
            let (e, k) = heads::encoding_to_camera(pe, g.height, g.width);
            out.pose_enc.push(*pe);
            out.extrinsics.push(e);
            out.intrinsics.push(k);
        }
        progress("done", 1.0);
        Ok(out)
    }

    /// Copies depth, confidence and world points of the last run at `geom`
    /// to the host.
    pub fn download(&self, geom: &Geometry) -> Result<DenseHost> {
        let ws = self
            .workspaces
            .get(geom)
            .ok_or_else(|| anyhow!("no run at {geom:?}"))?;
        let s = &self.nn.stream;
        let depth: Vec<u16> = s.clone_dtoh(&ws.depth).map_err(|e| anyhow!("{e:?}"))?;
        let conf: Vec<u16> = s.clone_dtoh(&ws.conf).map_err(|e| anyhow!("{e:?}"))?;
        let world: Vec<f32> = s.clone_dtoh(&ws.world).map_err(|e| anyhow!("{e:?}"))?;
        Ok(DenseHost {
            depth: depth.iter().map(|&h| spl_nn::host::f16_to_f32(h)).collect(),
            conf: conf.iter().map(|&h| spl_nn::host::f16_to_f32(h)).collect(),
            world,
        })
    }

    /// Workspace of a geometry (tests read the cached levels).
    pub fn workspace_of(&self, geom: &Geometry) -> Option<&Workspace> {
        self.workspaces.get(geom)
    }
}

fn record(nn: &Nn, e: &CudaEvent) -> Result<()> {
    e.record(&nn.stream)
        .map_err(|e| anyhow!("record CUDA event: {e:?}"))
}

fn normalize_launch(
    nn: &Nn,
    f: &spl_nn::cudarc::driver::CudaFunction,
    ws: &mut Workspace,
    g: Geometry,
) -> Result<()> {
    use spl_nn::cudarc::driver::{LaunchConfig, PushKernelArg};
    let pixels = (g.frames * g.pixels()) as i64;
    let hw = g.pixels() as i32;
    let cfg = LaunchConfig {
        grid_dim: ((pixels as u64).div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = nn.stream.launch_builder(f);
    b.arg(&ws.pixels).arg(&mut ws.images).arg(&pixels).arg(&hw);
    // SAFETY: `pixels` is N·H·W·3 bytes and `images` N·3·H·W floats, as
    // the kernel indexes them.
    unsafe { b.launch(cfg) }.map_err(|e| anyhow!("vggt_normalize: {e:?}"))?;
    Ok(())
}

/// The four levels of one dense chunk; offloaded levels are copied from
/// pinned memory into the staging buffers first.
fn chunk_levels<'a>(
    nn: &Nn,
    cache: &'a [Option<CudaSlice<u16>>; 4],
    host: &'a Option<[spl_nn::cudarc::driver::PinnedHostSlice<u16>; 3]>,
    stage: &'a mut Option<[CudaSlice<u16>; 3]>,
    g: Geometry,
    frames: std::ops::Range<usize>,
) -> Result<[spl_nn::cudarc::driver::CudaView<'a, u16>; 4]> {
    let row = g.tokens() * workspace::CACHE_DIM;
    let (a, b) = (frames.start * row, frames.end * row);
    let level = |k: usize| -> Result<&'a CudaSlice<u16>> {
        cache[k]
            .as_ref()
            .ok_or_else(|| anyhow!("cache level {k} missing"))
    };
    if let (Some(host), Some(stage)) = (host.as_ref(), stage.as_mut()) {
        for (h, s) in host.iter().zip(stage.iter_mut()) {
            let src = &h.as_slice().map_err(|e| anyhow!("pinned: {e:?}"))?[a..b];
            let mut dst = s.slice_mut(..b - a);
            nn.stream
                .memcpy_htod(src, &mut dst)
                .map_err(|e| anyhow!("stage cache: {e:?}"))?;
        }
        let stage: &'a [CudaSlice<u16>; 3] = &*stage;
        return Ok([
            stage[0].slice(..b - a),
            stage[1].slice(..b - a),
            stage[2].slice(..b - a),
            level(3)?.slice(a..b),
        ]);
    }
    Ok([
        level(0)?.slice(a..b),
        level(1)?.slice(a..b),
        level(2)?.slice(a..b),
        level(3)?.slice(a..b),
    ])
}

fn trunk_schedule(s: &AttnSchedule) -> Result<[trunk::InterKind; 24]> {
    let mut out = [trunk::InterKind::Global; 24];
    for (i, a) in s.0.iter().enumerate() {
        out[i] = match a {
            InterAttn::Global => trunk::InterKind::Global,
            InterAttn::Register => trunk::InterKind::Register,
            InterAttn::Frame => trunk::InterKind::Frame,
            InterAttn::Sage => trunk::InterKind::Sage,
            InterAttn::Merge2 => trunk::InterKind::Merge2,
            InterAttn::Merge4 => trunk::InterKind::Merge4,
        };
        if REGISTER_LAYERS.contains(&i) != (*a == InterAttn::Register) {
            bail!("layer {i}: register attention is fixed by the checkpoint");
        }
    }
    Ok(out)
}

fn trunk_geom(g: &Geometry) -> trunk::TrunkGeom {
    trunk::TrunkGeom {
        frames: g.frames,
        grid_h: g.grid_h,
        grid_w: g.grid_w,
    }
}

/// Everything besides the geometry that changes the trunk's kernels.
fn run_key(cfg: &VggtConfig) -> u64 {
    sched_key(&cfg.schedule) ^ cfg.fp8_blocks.0.rotate_left(17) ^ u64::from(cfg.sage_pv16) << 63
}

fn sched_key(s: &AttnSchedule) -> u64 {
    s.0.iter()
        .enumerate()
        .fold(0u64, |h, (i, a)| h ^ ((*a as u64 + 1) << ((i * 3) % 60)))
}
