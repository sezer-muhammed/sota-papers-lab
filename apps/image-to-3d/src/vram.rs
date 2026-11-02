//! Sharing the GPU with other processes (owner: gpu-coexist).
//!
//! The ops run on a 12 GB card other programs also use. Before each stage
//! the free device memory decides the plan:
//!
//! - engine: frames per VGGT-Ω pass ([`engine_chunk_ladder`]); fewer frames
//!   per pass go through the chunked Sim(3) registration;
//! - trainer: Gaussian capacity and image resolution ([`fit_trainer`]).
//!
//! When memory is plentiful the plan is the quality path, unchanged. Every
//! step down is recorded ([`Coexist::notes`]) and shown in progress and
//! outputs. When even the smallest plan does not fit, [`wait_until`] polls
//! every 5 s for up to 30 min (`SPL_GPU_POLL_S`, `SPL_GPU_WAIT_S`) in a
//! "waiting for GPU memory" state, then fails with a clear message. When a
//! run ends, [`release_cached`] hands the stream-ordered pool's freed blocks
//! back to the driver so the idle server holds only its context.

use std::time::{Duration, Instant};

use spl_core::{CancellationToken, Result, anyhow, bail};
use spl_gpu::cudarc::driver::sys;

use crate::kernels::{Capacity, FrameBuffers};
use crate::model::bytes_per_gaussian;
use crate::presets::Preset;

const GB: f64 = 1e9;
const MIB: u64 = 1 << 20;

/// How long and how often to wait for memory.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WaitPolicy {
    pub poll: Duration,
    pub timeout: Duration,
}

impl Default for WaitPolicy {
    fn default() -> Self {
        Self {
            poll: Duration::from_secs(5),
            timeout: Duration::from_secs(30 * 60),
        }
    }
}

impl WaitPolicy {
    /// The default, with `SPL_GPU_POLL_S` and `SPL_GPU_WAIT_S` (seconds)
    /// when set.
    pub fn from_env() -> Self {
        let secs = |k: &str| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.trim().parse::<f64>().ok())
                .filter(|v| v.is_finite() && *v >= 0.0)
                .map(Duration::from_secs_f64)
        };
        let d = Self::default();
        Self {
            poll: secs("SPL_GPU_POLL_S").unwrap_or(d.poll),
            timeout: secs("SPL_GPU_WAIT_S").unwrap_or(d.timeout),
        }
    }
}

/// What a run gave up because the GPU was shared.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Coexist {
    /// One line per degradation, e.g. "engine: 64 frames per pass (…)".
    pub notes: Vec<String>,
    /// Seconds spent waiting for memory.
    pub waited_s: f64,
}

impl Coexist {
    pub fn extend(&mut self, o: Coexist) {
        self.notes.extend(o.notes);
        self.waited_s += o.waited_s;
    }
}

/// Why a plan does not fit.
#[derive(Debug, Clone, PartialEq)]
pub struct Shortfall {
    /// "the VGGT-Omega weights", "the trainer", ...
    pub what: String,
    /// Bytes the smallest plan needs free.
    pub need: u64,
}

/// One poll of [`wait_until`].
#[derive(Debug, Clone, PartialEq)]
pub struct WaitState {
    pub what: String,
    pub need: u64,
    pub free: u64,
    pub waited_s: f64,
    pub timeout_s: f64,
}

impl WaitState {
    pub fn message(&self) -> String {
        format!(
            "waiting for GPU memory: {} needs {:.2} GB, {:.2} GB free (other processes hold the rest); {:.0} s of {:.0} min",
            self.what,
            self.need as f64 / GB,
            self.free as f64 / GB,
            self.waited_s,
            self.timeout_s / 60.0
        )
    }

    /// Progress data for the page (`stage` as its stage bar names it).
    pub fn data(&self, stage: &str) -> serde_json::Value {
        serde_json::json!({
            "stage": stage,
            "waiting_for_gpu_memory": true,
            "need_bytes": self.need,
            "free_bytes": self.free,
            "waited_s": self.waited_s,
            "timeout_s": self.timeout_s,
        })
    }
}

/// `(free, total)` bytes of device 0.
pub fn device_memory() -> Result<(u64, u64)> {
    let ctx = spl_gpu::context(0)?;
    ctx.bind_to_thread()
        .map_err(|e| anyhow!("bind CUDA context: {e:?}"))?;
    let (free, total) = ctx
        .mem_get_info()
        .map_err(|e| anyhow!("cuMemGetInfo: {e:?}"))?;
    Ok((free as u64, total as u64))
}

/// Waits for the device, then returns the default memory pool's free
/// blocks to the driver (cudarc allocates stream-ordered; freed blocks stay
/// reserved by the process until a trim). Call after dropping buffers.
pub fn release_cached() -> Result<()> {
    let ctx = spl_gpu::context(0)?;
    ctx.bind_to_thread()
        .map_err(|e| anyhow!("bind CUDA context: {e:?}"))?;
    ctx.synchronize()
        .map_err(|e| anyhow!("synchronize: {e:?}"))?;
    let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
    // SAFETY: plain driver calls on the bound context's device; the pool
    // handle is the device's default pool and is not freed by us.
    unsafe {
        sys::cuDeviceGetDefaultMemPool(&mut pool, ctx.cu_device())
            .result()
            .map_err(|e| anyhow!("cuDeviceGetDefaultMemPool: {e:?}"))?;
        sys::cuMemPoolTrimTo(pool, 0)
            .result()
            .map_err(|e| anyhow!("cuMemPoolTrimTo: {e:?}"))?;
    }
    Ok(())
}

/// [`release_cached`], logging instead of failing (end-of-run cleanup).
pub fn release_quietly() {
    if let Err(e) = release_cached() {
        tracing::warn!("could not release cached GPU memory: {e:#}");
    }
}

/// Calls `fits(free)` until it returns a plan; while it reports a
/// shortfall, `report`s the wait and polls `free` every `policy.poll`.
/// Fails after `policy.timeout` or on cancel. Returns the plan and the
/// seconds waited.
pub fn wait_until<T>(
    policy: &WaitPolicy,
    cancel: &CancellationToken,
    mut free: impl FnMut() -> Result<u64>,
    mut fits: impl FnMut(u64) -> Result<std::result::Result<T, Shortfall>>,
    report: &mut dyn FnMut(&WaitState),
) -> Result<(T, f64)> {
    let start = Instant::now();
    loop {
        let f = free()?;
        let short = match fits(f)? {
            Ok(t) => return Ok((t, start.elapsed().as_secs_f64())),
            Err(s) => s,
        };
        let waited = start.elapsed();
        if waited >= policy.timeout {
            bail!(
                "not enough GPU memory: {} needs {:.2} GB, {:.2} GB are free after waiting {:.0} min; other processes hold the GPU (see nvidia-smi)",
                short.what,
                short.need as f64 / GB,
                f as f64 / GB,
                waited.as_secs_f64() / 60.0
            );
        }
        report(&WaitState {
            what: short.what,
            need: short.need,
            free: f,
            waited_s: waited.as_secs_f64(),
            timeout_s: policy.timeout.as_secs_f64(),
        });
        let until = Instant::now() + policy.poll.min(policy.timeout - waited);
        while Instant::now() < until {
            if cancel.is_cancelled() {
                bail!("cancelled");
            }
            std::thread::sleep(Duration::from_millis(100).min(policy.poll));
        }
    }
}

/// Free bytes the engine needs before loading: its weights plus room for
/// the components and a small workspace.
pub fn engine_load_need(weights_bytes: u64) -> u64 {
    weights_bytes + 768 * MIB
}

/// Frames per engine pass to try, largest first: all of them (up to
/// `max`), then fewer down to 16 (the chunked registration needs anchors
/// shared with the first chunk).
pub fn engine_chunk_ladder(n: usize, max: usize) -> Vec<usize> {
    let first = n.min(max).max(2);
    let mut v = vec![first];
    v.extend(
        [160, 128, 96, 64, 48, 32, 24, 16]
            .into_iter()
            .filter(|&k| k < first),
    );
    v
}

/// Frames each later chunk shares with the first: 32 (the measured
/// default) unless chunks are so small that half of them is less.
pub fn chunk_anchors(chunk: usize) -> usize {
    32.min(chunk / 2).max(2)
}

/// The trainer's plan for the free memory.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainerFit {
    /// Image scale (1 = the given resolution).
    pub scale: f64,
    /// Capacity ceiling to pass ([`crate::trainer::TrainConfig::max_gaussians`]);
    /// `None` = the trainer's own choice (the quality path).
    pub max_gaussians: Option<u32>,
    /// Gaussians the trainer will allocate (estimate).
    pub capacity: u64,
}

/// Mirrors `Trainer::new`: instances it starts with.
const INITIAL_INSTANCES: u32 = 1 << 24;
/// Mirrors `Trainer::new`: its raster reserve is a frame of this many
/// Gaussians (or the capacity ceiling when smaller).
const PROBE_GAUSSIANS: u32 = 4 << 20;
/// Capacity that counts as unconstrained (Mip-NeRF 360 Quality ends at
/// 1.25M Gaussians).
pub const COMFORT_GAUSSIANS: u64 = 1_500_000;

/// Resolutions to try, with the fewest Gaussians each is worth running at.
const TRAINER_LADDER: [(f64, u64); 4] = [
    (1.0, 500_000),
    (0.75, 400_000),
    (0.5, 300_000),
    (0.375, 200_000),
];

/// Gaussians `Trainer::new` would allocate with `free` bytes free before
/// the images go up (`images` bytes) and a raster reserve for `probe`.
fn trainer_capacity(free: u64, images: u64, w: u32, h: u32, probe: u32, preset: &Preset) -> u64 {
    let after = free.saturating_sub(images);
    let cap = Capacity::new(probe, INITIAL_INSTANCES, w, h, &preset.variant);
    let reserve = FrameBuffers::vram_bytes(cap, &preset.variant).total() + (1 << 30);
    let bf16 = preset.adam.moments != crate::kernels::adam::MomentPrecision::F32;
    let budget = (after - after / 10).saturating_sub(reserve + 512 * MIB);
    budget / bytes_per_gaussian(bf16) / 1024 * 1024
}

/// Picks resolution and capacity for `views` training images of `w × h`
/// with `free` bytes free (`extra` bytes are kept for the caller's own
/// buffers). The given resolution with the trainer's own capacity when
/// that reaches [`COMFORT_GAUSSIANS`] (or the preset's ceiling); else the
/// largest capacity at the largest scale whose floor it reaches.
pub fn fit_trainer(
    free: u64,
    extra: u64,
    w: u32,
    h: u32,
    views: usize,
    preset_cap: u64,
    preset: &Preset,
) -> std::result::Result<TrainerFit, Shortfall> {
    let free = free.saturating_sub(extra);
    let images = |s: f64| {
        let (sw, sh) = scaled(w, h, s);
        views as u64 * sw as u64 * sh as u64 * 3
    };
    let auto = trainer_capacity(free, images(1.0), w, h, PROBE_GAUSSIANS, preset).min(preset_cap);
    if auto >= COMFORT_GAUSSIANS.min(preset_cap) {
        return Ok(TrainerFit {
            scale: 1.0,
            max_gaussians: None,
            capacity: auto,
        });
    }
    for (s, floor) in TRAINER_LADDER {
        let (sw, sh) = scaled(w, h, s);
        let img = images(s);
        // Largest m with capacity(probe = m) ≥ m: binary search.
        let (mut lo, mut hi) = (0u64, (PROBE_GAUSSIANS as u64).min(preset_cap) / 1024);
        while lo < hi {
            let mid = (lo + hi).div_ceil(2);
            let m = mid * 1024;
            if trainer_capacity(free, img, sw, sh, m as u32, preset) >= m {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        let m = lo * 1024;
        if m >= floor.min(preset_cap) {
            return Ok(TrainerFit {
                scale: s,
                max_gaussians: Some(m as u32),
                capacity: m,
            });
        }
    }
    // The smallest plan's need: solve capacity(probe = floor) ≥ floor.
    let (s, floor) = TRAINER_LADDER[TRAINER_LADDER.len() - 1];
    let (sw, sh) = scaled(w, h, s);
    let mut need = free;
    while trainer_capacity(need, images(s), sw, sh, floor as u32, preset) < floor {
        need += 64 * MIB;
    }
    Err(Shortfall {
        what: format!("the trainer ({floor} Gaussians at {sw}x{sh})"),
        need: need + extra,
    })
}

/// `w × h` scaled by `s` (at least 1 pixel; `s = 1` keeps them).
pub fn scaled(w: u32, h: u32, s: f64) -> (u32, u32) {
    if s >= 1.0 {
        return (w, h);
    }
    (
        ((w as f64 * s).round() as u32).max(1),
        ((h as f64 * s).round() as u32).max(1),
    )
}

/// A note for a trainer plan below the quality path, if any.
pub fn trainer_note(fit: &TrainerFit, w: u32, h: u32, free: u64) -> Option<String> {
    fit.max_gaussians.map(|m| {
        let (sw, sh) = scaled(w, h, fit.scale);
        let res = if fit.scale < 1.0 {
            format!("images {sw}x{sh} instead of {w}x{h}, ")
        } else {
            String::new()
        };
        format!(
            "low GPU memory ({:.2} GB free): trainer {res}at most {m} Gaussians",
            free as f64 / GB
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::presets::PresetName;

    const G: u64 = 1_000_000_000;

    fn quality() -> Preset {
        Preset::new(PresetName::Quality)
    }

    #[test]
    fn chunk_ladder_starts_with_everything() {
        assert_eq!(engine_chunk_ladder(10, 200), vec![10]);
        assert_eq!(engine_chunk_ladder(60, 200), vec![60, 48, 32, 24, 16]);
        assert_eq!(engine_chunk_ladder(300, 200)[..3], [200, 160, 128]);
        assert_eq!(*engine_chunk_ladder(300, 200).last().unwrap(), 16);
        assert_eq!(chunk_anchors(200), 32);
        assert_eq!(chunk_anchors(48), 24);
        assert_eq!(chunk_anchors(16), 8);
    }

    #[test]
    fn plenty_of_memory_keeps_the_quality_path() {
        let f = fit_trainer(11 * G, 0, 1600, 1066, 160, 8 << 20, &quality()).unwrap();
        assert_eq!(f.scale, 1.0);
        assert_eq!(f.max_gaussians, None);
        assert!(f.capacity >= COMFORT_GAUSSIANS, "{f:?}");
    }

    #[test]
    fn less_memory_degrades_monotonically_then_waits() {
        let p = quality();
        let mut last = (2.0, u64::MAX);
        let mut fitted = 0;
        for gb10 in (5..=80).rev().step_by(5) {
            let free = gb10 as u64 * G / 10;
            match fit_trainer(free, 0, 1600, 1066, 160, 8 << 20, &p) {
                Ok(f) => {
                    fitted += 1;
                    // Never better with less memory.
                    assert!(
                        f.scale < last.0 || (f.scale == last.0 && f.capacity <= last.1),
                        "{free}: {f:?} after {last:?}"
                    );
                    last = (f.scale, f.capacity);
                }
                Err(s) => {
                    assert!(s.need > free, "{s:?} at {free}");
                    // Once short, every smaller amount is short too.
                    last = (0.0, 0);
                }
            }
        }
        assert!(fitted > 2);
        assert!(fit_trainer(G / 2, 0, 1600, 1066, 160, 8 << 20, &p).is_err());
    }

    #[test]
    fn wait_until_polls_then_succeeds_or_times_out() {
        let policy = WaitPolicy {
            poll: Duration::from_millis(5),
            timeout: Duration::from_millis(200),
        };
        let cancel = CancellationToken::new();
        let mut free = [1u64, 2, 3, 10].into_iter();
        let mut reports = Vec::new();
        let (got, _) = wait_until(
            &policy,
            &cancel,
            || Ok(free.next().unwrap_or(10)),
            |f| {
                Ok(if f >= 3 {
                    Ok(f)
                } else {
                    Err(Shortfall {
                        what: "x".into(),
                        need: 3,
                    })
                })
            },
            &mut |w| reports.push(w.clone()),
        )
        .unwrap();
        assert_eq!(got, 3);
        assert_eq!(reports.len(), 2);
        assert!(reports[0].message().starts_with("waiting for GPU memory"));
        assert_eq!(reports[0].data("posing")["waiting_for_gpu_memory"], true);

        let err = wait_until(
            &policy,
            &cancel,
            || Ok(1),
            |_| {
                Ok(Err::<(), _>(Shortfall {
                    what: "the engine".into(),
                    need: 2_000_000_000,
                }))
            },
            &mut |_| {},
        )
        .unwrap_err();
        assert!(format!("{err}").contains("not enough GPU memory"), "{err}");

        cancel.cancel();
        let err = wait_until(
            &policy,
            &cancel,
            || Ok(1),
            |_| {
                Ok(Err::<(), _>(Shortfall {
                    what: "x".into(),
                    need: 2,
                }))
            },
            &mut |_| {},
        )
        .unwrap_err();
        assert!(format!("{err}").contains("cancelled"));
    }

    #[test]
    fn wait_policy_reads_env_defaults() {
        let d = WaitPolicy::default();
        assert_eq!(d.poll, Duration::from_secs(5));
        assert_eq!(d.timeout, Duration::from_secs(1800));
    }
}

#[cfg(test)]
#[path = "vram_gpu_tests.rs"]
mod gpu_tests;
