//! Decoded and developed RAWs, kept for the next run on the same file: the
//! auto op after inspect, and every live slider preview.

use std::collections::VecDeque;
use std::future::Future;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use spl_core::{Context, Result, RunCx};
use tokio::sync::OnceCell;

use super::work::{Span, blocking, ms};
use crate::decode::{RawFrame, decode};
use crate::develop::develop_scene;
use crate::finish::{self, LensCorrection};
use crate::lens::{self, LensSettings};
use crate::types::{Gpu, Params, Scene};

/// RAWs kept: about 40 to 150 MB each (the mosaic plus the preview images,
/// twice when lens-corrected).
const ENTRIES: usize = 4;

static DEVELOPED: Lru<Photo> = Lru::new(ENTRIES);

/// One RAW, decoded (for the full-resolution export) and developed to the
/// search and display images.
pub(super) struct Photo {
    pub frame: RawFrame,
    /// As developed, without lens corrections: the photo's info, and what
    /// renders when Lens Corrections are off.
    pub scene: Arc<Scene>,
    /// The camera's lens correction data, when the file carries it.
    pub lens: Option<lens::LensCorrection>,
    /// `scene` corrected, per lens setting asked for so far.
    corrected: Mutex<Vec<(LensSettings, Arc<Scene>)>>,
}

impl Photo {
    fn new(frame: RawFrame, scene: Scene, lens: Option<lens::LensCorrection>) -> Self {
        Self {
            frame,
            scene: Arc::new(scene),
            lens,
            corrected: Mutex::new(Vec::new()),
        }
    }

    /// The search and display images to render `p` from: with the lens
    /// corrections `p` switches on ([`finish::correct_scene`]), made on first
    /// use and kept. GPU work: call it on the blocking pool.
    pub fn scene_for(&self, gpu: &Gpu, p: &Params) -> Result<Arc<Scene>> {
        let settings = LensCorrection::new(p, &self.scene.info).settings();
        if !settings.profile && !settings.lateral_ca {
            return Ok(self.scene.clone());
        }
        // Held while correcting (a few milliseconds), so concurrent runs
        // on one photo correct it once.
        let mut corrected = self
            .corrected
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some((_, scene)) = corrected.iter().find(|(s, _)| *s == settings) {
            return Ok(scene.clone());
        }
        let scene = match finish::correct_scene(gpu, &self.scene, self.lens.as_ref(), p)? {
            Some(scene) => Arc::new(scene),
            None => self.scene.clone(),
        };
        corrected.push((settings, scene.clone()));
        Ok(scene)
    }
}

pub(super) struct Loaded {
    pub photo: Arc<Photo>,
    /// Zero when the photo came from the cache.
    pub decode_ms: f64,
    pub develop_ms: f64,
}

/// The RAW artifact `raw`, decoded and developed, from the cache or now.
/// Runs on the same file at the same time share one decode.
pub(super) async fn developed(cx: &RunCx, raw: &str, gpu: &Gpu, span: Span) -> Result<Loaded> {
    let mut timings = None;
    let photo = DEVELOPED
        .get_or_load(raw, || async {
            cx.progress(span.at(0.0), "Reading the RAW file");
            let started = Instant::now();
            let bytes = cx
                .artifacts
                .get(raw)
                .await
                .with_context(|| format!("read the RAW file (artifact {raw})"))?;
            let gpu = gpu.clone();
            let (photo, decode_ms, develop_ms) = blocking(cx, span, move |r| {
                r.progress(0.05, "Decoding the RAW file");
                let frame = decode(&bytes)?;
                let lens = lens::lens_profile(&bytes);
                drop(bytes);
                let decode_ms = ms(started);
                r.check()?;
                r.progress(0.7, "Developing");
                let developing = Instant::now();
                let scene = develop_scene(&gpu, &frame)?;
                let develop_ms = ms(developing);
                r.log(format!(
                    "Decoded {} {}, {}x{}, in {decode_ms:.0} ms; developed in {develop_ms:.0} ms",
                    frame.info.make, frame.info.model, frame.width, frame.height
                ));
                r.log(match &lens {
                    Some(lens) => format!("Lens correction data: {}", lens.summary()),
                    None => "No lens correction data in the file".to_owned(),
                });
                Ok((Photo::new(frame, scene, lens), decode_ms, develop_ms))
            })
            .await?;
            timings = Some((decode_ms, develop_ms));
            Ok(photo)
        })
        .await?;
    if timings.is_none() {
        cx.log("Reusing the decoded RAW from an earlier run");
    }
    let (decode_ms, develop_ms) = timings.unwrap_or_default();
    Ok(Loaded {
        photo,
        decode_ms,
        develop_ms,
    })
}

type Cell<V> = Arc<OnceCell<Arc<V>>>;

/// A least-recently-used map of values that load once: concurrent callers
/// for one key wait for a single load, and a failed load leaves no entry.
struct Lru<V> {
    capacity: usize,
    /// Most recently used last.
    slots: Mutex<VecDeque<(String, Cell<V>)>>,
}

impl<V> Lru<V> {
    const fn new(capacity: usize) -> Self {
        Self {
            capacity,
            slots: Mutex::new(VecDeque::new()),
        }
    }

    async fn get_or_load<F, Fut>(&self, key: &str, load: F) -> Result<Arc<V>>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<V>>,
    {
        let cell = self.cell(key);
        let loaded = cell
            .get_or_try_init(|| async { load().await.map(Arc::new) })
            .await
            .cloned();
        if loaded.is_err() {
            self.forget_if_empty(key, &cell);
        }
        loaded
    }

    /// The cell for `key`, now the most recent; evicts the least recent
    /// beyond capacity (runs holding an evicted value keep it).
    fn cell(&self, key: &str) -> Cell<V> {
        let mut slots = self.lock();
        if let Some(at) = slots.iter().position(|(k, _)| k == key)
            && let Some(slot) = slots.remove(at)
        {
            let cell = slot.1.clone();
            slots.push_back(slot);
            return cell;
        }
        let cell = Cell::<V>::default();
        slots.push_back((key.to_owned(), cell.clone()));
        while slots.len() > self.capacity.max(1) {
            slots.pop_front();
        }
        cell
    }

    fn forget_if_empty(&self, key: &str, cell: &Cell<V>) {
        let mut slots = self.lock();
        slots.retain(|(k, c)| !(k == key && Arc::ptr_eq(c, cell) && c.get().is_none()));
    }

    fn lock(&self) -> MutexGuard<'_, VecDeque<(String, Cell<V>)>> {
        // Every critical section is a few list operations: a poisoned lock
        // still holds a consistent list.
        self.slots.lock().unwrap_or_else(PoisonError::into_inner)
    }

    #[cfg(test)]
    fn keys(&self) -> Vec<String> {
        self.lock().iter().map(|(k, _)| k.clone()).collect()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use spl_core::bail;

    use super::*;

    #[tokio::test]
    async fn keeps_the_most_recent_and_loads_each_once() {
        let lru: Lru<String> = Lru::new(2);
        let loads = AtomicUsize::new(0);
        let get = |key: &'static str| {
            let lru = &lru;
            let loads = &loads;
            async move {
                lru.get_or_load(key, || async {
                    loads.fetch_add(1, Ordering::SeqCst);
                    Ok(key.to_uppercase())
                })
                .await
                .unwrap()
            }
        };
        assert_eq!(*get("a").await, "A");
        assert_eq!(*get("b").await, "B");
        assert_eq!(*get("a").await, "A");
        assert_eq!(loads.load(Ordering::SeqCst), 2);
        // "b" is now the least recent, so "c" evicts it.
        get("c").await;
        assert_eq!(lru.keys(), ["a", "c"]);
        get("b").await;
        assert_eq!(loads.load(Ordering::SeqCst), 4);
        assert_eq!(lru.keys(), ["c", "b"]);
    }

    #[tokio::test]
    async fn concurrent_callers_share_one_load() {
        let lru: Lru<u32> = Lru::new(4);
        let loads = AtomicUsize::new(0);
        let load = || async {
            loads.fetch_add(1, Ordering::SeqCst);
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            Ok(7)
        };
        let (a, b) = tokio::join!(lru.get_or_load("x", load), lru.get_or_load("x", load));
        assert_eq!((*a.unwrap(), *b.unwrap()), (7, 7));
        assert_eq!(loads.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn failed_loads_leave_nothing_behind() {
        let lru: Lru<u32> = Lru::new(4);
        let failed = lru
            .get_or_load("bad", || async { bail!("not a RAW file") })
            .await;
        assert_eq!(failed.unwrap_err().to_string(), "not a RAW file");
        assert!(lru.keys().is_empty());
        let ok = lru.get_or_load("bad", || async { Ok(1) }).await.unwrap();
        assert_eq!(*ok, 1);
        assert_eq!(lru.keys(), ["bad"]);
    }
}
