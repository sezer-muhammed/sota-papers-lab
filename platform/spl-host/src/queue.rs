//! Which runs may execute: at most `max_concurrent_runs` ops at once, and an
//! exclusive GPU op never next to another GPU op.
//!
//! Two fair (first-in, first-out) locks do it. Every op takes one of the
//! run slots. A GPU op also takes the GPU lock first: shared ops as readers,
//! exclusive ops as the writer. A waiting writer holds back the readers that
//! come after it, so runs keep their submission order and an exclusive run
//! is never starved. CPU-only ops skip the GPU lock.

use std::sync::Arc;

use serde::Serialize;
use spl_core::{Context, GpuUse, Result};
use tokio::sync::{
    OwnedRwLockReadGuard, OwnedRwLockWriteGuard, OwnedSemaphorePermit, RwLock, Semaphore,
};
use ts_rs::TS;

/// What a queued run waits for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum WaitReason {
    /// The GPU: an exclusive run holds it or is next in line for it, or
    /// this exclusive run waits for the GPU runs ahead of it to end.
    Gpu,
    /// A run slot: `max_concurrent_runs` ops execute already.
    Slot,
}

pub(crate) struct RunQueue {
    slots: Arc<Semaphore>,
    gpu: Arc<RwLock<()>>,
}

/// The right to execute, held until the op returns.
pub(crate) struct Admission {
    _gpu: Option<GpuClaim>,
    _slot: OwnedSemaphorePermit,
}

/// Held, never read: dropping it releases the GPU lock.
enum GpuClaim {
    Shared { _guard: OwnedRwLockReadGuard<()> },
    Exclusive { _guard: OwnedRwLockWriteGuard<()> },
}

/// A run that could not start at once, holding what it already got.
pub(crate) struct Pending {
    gpu_use: GpuUse,
    gpu: Option<GpuClaim>,
}

impl Pending {
    pub(crate) fn waiting_for(&self) -> WaitReason {
        if self.needs_gpu() {
            WaitReason::Gpu
        } else {
            WaitReason::Slot
        }
    }

    fn needs_gpu(&self) -> bool {
        self.gpu.is_none() && self.gpu_use != GpuUse::CpuOnly
    }
}

impl RunQueue {
    pub(crate) fn new(max_concurrent_runs: usize) -> Self {
        Self {
            slots: Arc::new(Semaphore::new(max_concurrent_runs.max(1))),
            gpu: Arc::new(RwLock::new(())),
        }
    }

    /// Admits a run when it can start right now, without waiting. Otherwise
    /// the run keeps what it got (its GPU claim) and waits with [`admit`].
    ///
    /// [`admit`]: Self::admit
    pub(crate) fn try_admit(&self, gpu_use: GpuUse) -> Result<Admission, Pending> {
        let gpu = match gpu_use {
            GpuUse::CpuOnly => None,
            GpuUse::Shared => match self.gpu.clone().try_read_owned() {
                Ok(guard) => Some(GpuClaim::Shared { _guard: guard }),
                Err(_) => return Err(Pending { gpu_use, gpu: None }),
            },
            GpuUse::Exclusive => match self.gpu.clone().try_write_owned() {
                Ok(guard) => Some(GpuClaim::Exclusive { _guard: guard }),
                Err(_) => return Err(Pending { gpu_use, gpu: None }),
            },
        };
        match self.slots.clone().try_acquire_owned() {
            Ok(slot) => Ok(Admission {
                _gpu: gpu,
                _slot: slot,
            }),
            Err(_) => Err(Pending { gpu_use, gpu }),
        }
    }

    /// Waits in line, telling `on_wait` what it waits for. Dropping the
    /// future leaves the line and releases whatever the run held.
    pub(crate) async fn admit(
        &self,
        mut pending: Pending,
        mut on_wait: impl FnMut(WaitReason),
    ) -> Result<Admission> {
        if pending.needs_gpu() {
            on_wait(WaitReason::Gpu);
            pending.gpu = Some(match pending.gpu_use {
                GpuUse::Exclusive => GpuClaim::Exclusive {
                    _guard: self.gpu.clone().write_owned().await,
                },
                _ => GpuClaim::Shared {
                    _guard: self.gpu.clone().read_owned().await,
                },
            });
        }
        let slot = match self.slots.clone().try_acquire_owned() {
            Ok(slot) => slot,
            Err(_) => {
                on_wait(WaitReason::Slot);
                self.slots
                    .clone()
                    .acquire_owned()
                    .await
                    .context("run queue closed")?
            }
        };
        Ok(Admission {
            _gpu: pending.gpu,
            _slot: slot,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn admitted(queue: &RunQueue, gpu_use: GpuUse) -> Admission {
        match queue.try_admit(gpu_use) {
            Ok(admission) => admission,
            Err(pending) => panic!("{gpu_use:?} waits for {:?}", pending.waiting_for()),
        }
    }

    fn waits(queue: &RunQueue, gpu_use: GpuUse) -> Pending {
        match queue.try_admit(gpu_use) {
            Ok(_) => panic!("{gpu_use:?} was admitted"),
            Err(pending) => pending,
        }
    }

    /// Whether `future` is still pending after the scheduler had its turn.
    async fn blocked<T>(future: &mut (impl Future<Output = T> + Unpin)) -> bool {
        tokio::time::timeout(Duration::from_millis(20), future)
            .await
            .is_err()
    }

    #[tokio::test]
    async fn an_exclusive_run_waits_for_shared_ones_and_then_runs_alone() {
        let queue = RunQueue::new(3);
        let shared = admitted(&queue, GpuUse::Shared);
        let exclusive = waits(&queue, GpuUse::Exclusive);
        assert_eq!(exclusive.waiting_for(), WaitReason::Gpu);
        let mut exclusive = Box::pin(queue.admit(exclusive, |_| {}));
        assert!(blocked(&mut exclusive).await);

        // A slot is free, but the exclusive run is ahead in line.
        let later = waits(&queue, GpuUse::Shared);
        assert_eq!(later.waiting_for(), WaitReason::Gpu);
        let mut later = Box::pin(queue.admit(later, |_| {}));
        assert!(blocked(&mut later).await);
        // CPU-only work is not held back.
        let cpu = admitted(&queue, GpuUse::CpuOnly);

        drop(shared);
        let exclusive = exclusive.await.unwrap();
        assert!(
            blocked(&mut later).await,
            "no GPU op next to an exclusive one"
        );
        drop(exclusive);
        later.await.unwrap();
        drop(cpu);
    }

    #[tokio::test]
    async fn a_waiting_run_that_leaves_the_line_releases_it() {
        let queue = RunQueue::new(2);
        let shared = admitted(&queue, GpuUse::Shared);
        let mut exclusive = Box::pin(queue.admit(waits(&queue, GpuUse::Exclusive), |_| {}));
        assert!(blocked(&mut exclusive).await);
        let mut later = Box::pin(queue.admit(waits(&queue, GpuUse::Shared), |_| {}));
        assert!(blocked(&mut later).await);

        drop(exclusive); // cancelled while queued
        let later = later.await.unwrap();
        drop((shared, later));
        let _alone = admitted(&queue, GpuUse::Exclusive);
    }

    #[tokio::test]
    async fn runs_beyond_the_slots_wait_for_one() {
        let queue = RunQueue::new(1);
        let first = admitted(&queue, GpuUse::CpuOnly);
        let second = waits(&queue, GpuUse::Shared);
        assert_eq!(
            second.waiting_for(),
            WaitReason::Slot,
            "it holds the GPU lock"
        );
        let mut reasons = Vec::new();
        let mut second = Box::pin(queue.admit(second, |r| reasons.push(r)));
        assert!(blocked(&mut second).await);
        drop(first);
        second.await.unwrap();
        assert_eq!(reasons, [WaitReason::Slot]);

        let queue = RunQueue::new(1);
        let exclusive = admitted(&queue, GpuUse::Exclusive);
        let mut reasons = Vec::new();
        let mut shared = Box::pin(queue.admit(waits(&queue, GpuUse::Shared), |r| reasons.push(r)));
        assert!(blocked(&mut shared).await);
        drop(exclusive);
        shared.await.unwrap();
        assert_eq!(reasons, [WaitReason::Gpu]);
    }
}
