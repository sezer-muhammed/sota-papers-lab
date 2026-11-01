//! A deterministic "Auto" first guess from the neutral render's statistics,
//! the way Lightroom's Auto button works. Owner: search.

use crate::types::{Params, Scene, Stats};

pub fn auto_params(neutral: &Stats, scene: &Scene) -> Params {
    let _ = (neutral, scene);
    unimplemented!("auto_params")
}
