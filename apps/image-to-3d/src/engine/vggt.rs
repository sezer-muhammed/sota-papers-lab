//! [`Engine`] for the native VGGT-Ω engine (research profile). The engine
//! itself is [`crate::vggt`]; this only adapts its API.

use spl_core::{CancellationToken, Result};

use super::{Engine, EngineInfo, EngineOut, EngineTimes};
use crate::io::images::Rgb8Image;
use crate::vggt::VggtEngine;

/// Licence of `vggt_omega_1b_512.pt`.
pub const VGGT_LICENCE: &str = "fair-noncommercial-research";

impl Engine for VggtEngine {
    fn info(&self) -> EngineInfo {
        EngineInfo {
            engine: "vggt-omega-1b-512".into(),
            licence: VGGT_LICENCE.into(),
            weights_sha256: None,
        }
    }

    fn infer(
        &mut self,
        frames: &[Rgb8Image],
        cancel: &CancellationToken,
        progress: &mut dyn FnMut(&str, f32),
    ) -> Result<EngineOut> {
        let t = std::time::Instant::now();
        let prepared = self.preprocess(frames)?;
        let preprocess_ms = t.elapsed().as_secs_f32() * 1e3;
        let cameras = self.run(&prepared, cancel, progress)?;
        let dense = self.download(&prepared.geom)?;
        Ok(EngineOut {
            prepared,
            cameras,
            dense,
            times: EngineTimes {
                preprocess_ms,
                trunk_ms: self.times.trunk_ms,
                dense_ms: self.times.camera_ms + self.times.dense_ms,
                forward_ms: self.times.total_ms,
            },
        })
    }
}
