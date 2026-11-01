//! STUB, for the integrator to delete together with `mod bw_stub;` in
//! `ops/pipeline.rs`: `LookSpec::black_and_white()` belongs to the search
//! owner (objective.rs), who adds it with the Black & White params. Until
//! then this keeps the request routing compiling; the look it returns has
//! natural's targets and no black and white conversion.

use crate::objective::LookSpec;

impl LookSpec {
    /// The black and white look.
    pub fn black_and_white() -> LookSpec {
        LookSpec {
            id: "bw".into(),
            label: "Black and white".into(),
            anchor: None,
        }
    }
}
