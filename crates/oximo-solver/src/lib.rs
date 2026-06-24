#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

pub mod options;
pub mod result;
pub mod solver;
pub mod status;

pub use options::{HasUniversal, UniversalOptions, UniversalOptionsExt};
pub use result::{ModelReport, SolutionPoint, SolverResult};
pub use solver::Solver;
pub use status::{PrimalStatus, SolverError, TerminationStatus};
