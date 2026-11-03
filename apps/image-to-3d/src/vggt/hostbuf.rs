//! Host buffers for device copies: page-locked when the driver grants it,
//! pageable otherwise. A container's memlock limit (or pinned memory held by
//! other processes) makes `cuMemHostAlloc` fail with `NOT_PERMITTED`; that
//! costs copy speed, never the run. Live pinned bytes are counted so tests
//! can check every allocation is released.

use std::sync::atomic::{AtomicUsize, Ordering};

use spl_core::{Result, anyhow};
use spl_nn::Nn;
use spl_nn::cudarc::driver::{DeviceRepr, PinnedHostSlice, ValidAsZeroBits};

static LIVE_PINNED: AtomicUsize = AtomicUsize::new(0);

/// Bytes of pinned memory currently held by [`HostBuf`]s in this process.
pub fn live_pinned_bytes() -> usize {
    LIVE_PINNED.load(Ordering::SeqCst)
}

pub enum HostBuf<T> {
    Pinned(PinnedHostSlice<T>),
    Pageable(Vec<T>),
}

impl<T: DeviceRepr + ValidAsZeroBits + Default + Clone> HostBuf<T> {
    /// `len` values; pinned if possible, else pageable with a log note.
    pub fn new(nn: &Nn, len: usize, what: &str) -> Self {
        // SAFETY: callers write every element (host or device copy) before
        // reading it.
        match unsafe { nn.ctx.alloc_pinned::<T>(len) } {
            Ok(p) => {
                LIVE_PINNED.fetch_add(len * size_of::<T>(), Ordering::SeqCst);
                Self::Pinned(p)
            }
            Err(e) => {
                tracing::warn!(
                    "{what}: pinned host allocation of {} MiB failed ({e:?}); using pageable memory (slower copies)",
                    (len * size_of::<T>()) >> 20
                );
                Self::Pageable(vec![T::default(); len])
            }
        }
    }
}

impl<T: ValidAsZeroBits> HostBuf<T> {
    pub fn is_pinned(&self) -> bool {
        matches!(self, Self::Pinned(_))
    }

    /// Waits for pending copies of a pinned buffer.
    pub fn as_slice(&self) -> Result<&[T]> {
        match self {
            Self::Pinned(p) => p.as_slice().map_err(|e| anyhow!("pinned: {e:?}")),
            Self::Pageable(v) => Ok(v),
        }
    }

    pub fn as_mut_slice(&mut self) -> Result<&mut [T]> {
        match self {
            Self::Pinned(p) => p.as_mut_slice().map_err(|e| anyhow!("pinned: {e:?}")),
            Self::Pageable(v) => Ok(v),
        }
    }
}

impl<T> Drop for HostBuf<T> {
    fn drop(&mut self) {
        if let Self::Pinned(p) = self {
            LIVE_PINNED.fetch_sub(p.len() * size_of::<T>(), Ordering::SeqCst);
        }
    }
}

/// Engine load + drop, ten times in one process, on the GPU host
/// (`#[ignore]`; weights `$SPL_VGGT_WEIGHTS`): pinned memory returns to the
/// baseline after every drop, and a large pinned allocation still succeeds.
#[cfg(test)]
mod gpu_tests {
    use std::sync::Arc;

    use super::*;
    use crate::vggt::{VggtConfig, VggtEngine};

    /// `VmLck` + `VmPin` of this process in KiB (Linux).
    fn locked_kib() -> u64 {
        let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        s.lines()
            .filter(|l| l.starts_with("VmLck:") || l.starts_with("VmPin:"))
            .filter_map(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
            .sum()
    }

    #[test]
    #[ignore]
    fn gpu_engine_load_drop_releases_pinned() {
        let weights = std::env::var("SPL_VGGT_WEIGHTS")
            .unwrap_or_else(|_| "/models/vggt-omega/vggt_omega_1b_512.safetensors".into());
        let base = (live_pinned_bytes(), locked_kib());
        eprintln!("baseline: live pinned {} B, locked {} KiB", base.0, base.1);
        for i in 0..10 {
            let nn = Arc::new(Nn::open().unwrap());
            let t = std::time::Instant::now();
            let e = VggtEngine::load(nn, weights.as_ref(), VggtConfig::default()).unwrap();
            let load = t.elapsed().as_secs_f64();
            let held = live_pinned_bytes();
            drop(e);
            let after = (live_pinned_bytes(), locked_kib());
            eprintln!(
                "cycle {i}: load {load:.2} s, pinned after load {held} B, after drop {} B, locked {} KiB",
                after.0, after.1
            );
            assert_eq!(after.0, base.0, "pinned bytes leaked in cycle {i}");
            assert!(after.1 <= base.1 + 1024, "locked memory grew in cycle {i}");
        }
        // 1 GiB pinned must still be available (the failing allocation was 2 × 64 MiB).
        let nn = Nn::open().unwrap();
        let probe = HostBuf::<u8>::new(&nn, 1 << 30, "probe");
        assert!(
            probe.is_pinned(),
            "pinned allocation refused after 10 cycles"
        );
    }
}
