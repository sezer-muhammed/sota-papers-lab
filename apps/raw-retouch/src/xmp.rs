//! Lightroom / Camera Raw XMP sidecar for the chosen sliders. Owner: judge.

use crate::types::{Params, RawInfo};

/// A complete `.xmp` file; name it like the RAW (`DSC01234.xmp`) next to it.
pub fn sidecar(params: &Params, info: &RawInfo) -> String {
    let _ = (params, info);
    unimplemented!("sidecar")
}
