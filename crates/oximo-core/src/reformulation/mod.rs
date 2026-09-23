//! Explicit, solver-independent model reformulations.
//!
//! `reformulate*` operations append to the source model after validating the
//! complete plan. `to_reformulated_*` operations preserve the source and operate
//! on an independent clone. Both forms preserve existing variable and constraint
//! IDs.

mod indicator;
mod sos;

pub use indicator::{IndicatorReformulationArtifacts, IndicatorReformulationOptions};
pub use sos::{
    ReformulatedModel, ReformulationError, SosReformulationArtifacts, SosReformulationOptions,
};
