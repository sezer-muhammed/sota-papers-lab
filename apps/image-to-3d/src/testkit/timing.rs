//! GPU timing: CUDA-event timers, warm-up + repeat benchmarks with order
//! statistics, and an `nvidia-smi` snapshot to record next to every timing
//! (so contention is visible). Timing runs that produce published numbers go
//! under `flock -x ~/spl-work/gpu.lock` on the host.

use serde::Serialize;
use spl_core::Result;
use spl_gpu::cudarc::driver::sys::CUevent_flags;
use spl_gpu::cudarc::driver::{CudaEvent, CudaStream};

use crate::gpu::{CudaResultExt, Gpu};

/// A pair of timing events on one stream.
pub struct GpuTimer {
    start: CudaEvent,
    stop: CudaEvent,
}

impl GpuTimer {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let ev = || {
            gpu.ctx
                .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .cuda("create CUDA event")
        };
        Ok(Self {
            start: ev()?,
            stop: ev()?,
        })
    }

    pub fn start(&self, s: &CudaStream) -> Result<()> {
        self.start.record(s).cuda("record start event")
    }

    /// Records the stop event, waits for it and returns the elapsed milliseconds.
    pub fn stop_ms(&self, s: &CudaStream) -> Result<f32> {
        self.stop.record(s).cuda("record stop event")?;
        self.start.elapsed_ms(&self.stop).cuda("event elapsed time")
    }
}

/// Order statistics of repeated timings, milliseconds.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Stats {
    pub n: usize,
    pub median_ms: f64,
    pub mean_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub p10_ms: f64,
    pub p90_ms: f64,
}

impl Stats {
    pub fn from_samples(samples: &[f64]) -> Self {
        if samples.is_empty() {
            return Self::default();
        }
        let mut s = samples.to_vec();
        s.sort_by(f64::total_cmp);
        let q = |p: f64| {
            let x = p * (s.len() - 1) as f64;
            let (i, f) = (x.floor() as usize, x.fract());
            s[i] + f * (s[(i + 1).min(s.len() - 1)] - s[i])
        };
        Self {
            n: s.len(),
            median_ms: q(0.5),
            mean_ms: s.iter().sum::<f64>() / s.len() as f64,
            min_ms: s[0],
            max_ms: s[s.len() - 1],
            p10_ms: q(0.1),
            p90_ms: q(0.9),
        }
    }
}

/// Runs `f` `warmup` times, then `reps` times between CUDA events on
/// `gpu.stream`; `f` only enqueues work.
pub fn bench(
    gpu: &Gpu,
    warmup: usize,
    reps: usize,
    mut f: impl FnMut() -> Result<()>,
) -> Result<Stats> {
    for _ in 0..warmup {
        f()?;
    }
    gpu.sync()?;
    let timer = GpuTimer::new(gpu)?;
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        timer.start(&gpu.stream)?;
        f()?;
        samples.push(timer.stop_ms(&gpu.stream)? as f64);
    }
    Ok(Stats::from_samples(&samples))
}

/// What `nvidia-smi` reports at one moment.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct GpuSnapshot {
    pub utilization_pct: u32,
    pub memory_used_mib: u32,
    pub sm_clock_mhz: u32,
    pub mem_clock_mhz: u32,
}

/// `nvidia-smi --query-gpu=...` for device 0; None when it is unavailable.
pub fn gpu_snapshot() -> Option<GpuSnapshot> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu,memory.used,clocks.sm,clocks.mem",
            "--format=csv,noheader,nounits",
            "-i",
            "0",
        ])
        .output()
        .ok()?;
    parse_snapshot(&String::from_utf8_lossy(&out.stdout))
}

fn parse_snapshot(line: &str) -> Option<GpuSnapshot> {
    let v: Vec<u32> = line
        .trim()
        .split(',')
        .map(|s| s.trim().parse().ok())
        .collect::<Option<_>>()?;
    match v.as_slice() {
        &[
            utilization_pct,
            memory_used_mib,
            sm_clock_mhz,
            mem_clock_mhz,
        ] => Some(GpuSnapshot {
            utilization_pct,
            memory_used_mib,
            sm_clock_mhz,
            mem_clock_mhz,
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_interpolate_percentiles() {
        let s = Stats::from_samples(&[5.0, 1.0, 3.0, 2.0, 4.0]);
        assert_eq!(
            (s.n, s.median_ms, s.min_ms, s.max_ms, s.mean_ms),
            (5, 3.0, 1.0, 5.0, 3.0)
        );
        assert!((s.p10_ms - 1.4).abs() < 1e-12 && (s.p90_ms - 4.6).abs() < 1e-12);
        assert_eq!(Stats::from_samples(&[]).n, 0);
    }

    #[test]
    fn parses_nvidia_smi_csv() {
        assert_eq!(
            parse_snapshot("3, 381, 2745, 10251\n"),
            Some(GpuSnapshot {
                utilization_pct: 3,
                memory_used_mib: 381,
                sm_clock_mhz: 2745,
                mem_clock_mhz: 10251
            })
        );
        assert_eq!(parse_snapshot("[N/A], 1, 2, 3"), None);
    }
}
