//! CUDA graphs per (geometry, schedule) for small N, where launch overhead
//! (about 900 kernels per forward) is a visible share of the run. Owner:
//! runtime.
//!
//! The first run of a key runs eagerly (it compiles kernels, picks GEMM
//! algorithms and lets every component allocate its buffers); the second
//! captures the same work on the stream and launches the graph; later runs
//! only launch it. Buffer addresses must not change between runs of a key:
//! the engine keeps the workspace per geometry and the components keep
//! theirs. If capture fails (a component synchronises or allocates inside
//! the body), the key is marked eager and runs without a graph from then on.

use std::collections::HashMap;

use spl_core::{Result, anyhow};
use spl_nn::Nn;
use spl_nn::cudarc::driver::CudaGraph;
use spl_nn::cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};

use super::workspace::Geometry;

pub type GraphKey = (Geometry, u64);

enum State {
    Warm,
    Graph(Exec),
    Eager,
}

/// A captured graph. cudarc does not mark graphs `Send`; the engine moves
/// between worker threads but runs on one at a time (it sits behind a
/// mutex), which is all the CUDA graph thread-safety rules ask for.
struct Exec(CudaGraph);

// SAFETY: see above; graphs are only touched through `&mut GraphCache`.
unsafe impl Send for Exec {}

#[derive(Default)]
pub struct GraphCache {
    graphs: HashMap<GraphKey, State>,
}

impl GraphCache {
    pub fn clear(&mut self) {
        self.graphs.clear();
    }

    /// Runs `body` for `key`, through a graph once one exists. Returns
    /// whether a graph ran.
    pub fn run(
        &mut self,
        nn: &Nn,
        key: GraphKey,
        body: &mut dyn FnMut(&Nn) -> Result<()>,
    ) -> Result<bool> {
        match self.graphs.get(&key) {
            None => {
                body(nn)?;
                self.graphs.insert(key, State::Warm);
                Ok(false)
            }
            Some(State::Eager) => body(nn).map(|_| false),
            Some(State::Graph(Exec(g))) => {
                g.launch()
                    .map_err(|e| anyhow!("launch CUDA graph: {e:?}"))?;
                Ok(true)
            }
            Some(State::Warm) => {
                // Everything before the capture must be done: a capture
                // cannot wait on work outside it.
                nn.sync()?;
                match capture(nn, body) {
                    Ok(g) => {
                        g.launch()
                            .map_err(|e| anyhow!("launch CUDA graph: {e:?}"))?;
                        self.graphs.insert(key, State::Graph(Exec(g)));
                        Ok(true)
                    }
                    Err(e) => {
                        tracing::warn!("CUDA graph capture failed, running eagerly: {e:#}");
                        self.graphs.insert(key, State::Eager);
                        body(nn).map(|_| false)
                    }
                }
            }
        }
    }

    pub fn graphs(&self) -> usize {
        self.graphs
            .values()
            .filter(|s| matches!(s, State::Graph(_)))
            .count()
    }
}

fn capture(nn: &Nn, body: &mut dyn FnMut(&Nn) -> Result<()>) -> Result<CudaGraph> {
    let s = &nn.stream;
    s.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        .map_err(|e| anyhow!("begin capture: {e:?}"))?;
    let r = body(nn);
    // Always end the capture so the stream is usable again.
    let g = s.end_capture(CUgraphInstantiate_flags(0));
    r?;
    let g = g.map_err(|e| anyhow!("end capture: {e:?}"))?;
    let g = g.ok_or_else(|| anyhow!("empty CUDA graph"))?;
    g.upload()
        .map_err(|e| anyhow!("upload CUDA graph: {e:?}"))?;
    Ok(g)
}
