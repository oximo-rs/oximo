#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

pub mod incremental;
pub mod infeasibility;
pub mod options;
pub mod persistent;
pub mod result;
pub mod solver;
pub mod status;

pub use incremental::{Snapshot, snapshot};
pub use infeasibility::{Iis, IisReport, InfeasibilityDiagnosis, VarBoundKind, is_infeasible};
pub use options::{HasUniversal, UniversalOptions, UniversalOptionsExt};
pub use persistent::PersistentSolver;
pub use result::{ModelReport, SolutionPoint, SolverResult};
pub use solver::Solver;
pub use status::{PrimalStatus, SolverError, TerminationStatus};
