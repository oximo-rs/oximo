//! Solver-trait abstraction and shared option / result types for oximo.
//!
//! Concrete backends live in their own crates and are wired up through
//! cargo features on the umbrella `oximo` crate.
#![forbid(unsafe_code)]

pub mod options;
pub mod registry;
pub mod result;
pub mod solver;
pub mod status;

pub use options::{OptionValue, SolverOptions};
pub use registry::SolverRegistry;
pub use result::SolverResult;
pub use solver::Solver;
pub use status::{SolverError, SolverStatus};
