#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

pub mod translation;
pub use translation::{
    AffineRowTranslator, PreparedConstraint, TranslationContext, TypedConstraintTranslator,
};
pub mod capability;
pub use capability::{Capability, CapabilityDiagnostic, ConstraintSignature, ModelValidation};
pub mod incremental;
pub mod infeasibility;
pub mod options;
pub mod persistent;
pub mod prepare;
pub mod psd;
pub mod reconstruct;
#[doc(hidden)]
pub mod resident;
pub mod result;
pub mod solver;
pub mod status;

pub use incremental::{Snapshot, snapshot};
pub use infeasibility::{Iis, IisReport, InfeasibilityDiagnosis, VarBoundKind, is_infeasible};
pub use options::{HasUniversal, UniversalOptions, UniversalOptionsExt};
pub use persistent::PersistentSolver;
pub use result::{
    ConstraintEvaluation, DualStatus, ModelReport, ParameterSnapshot, SocEvaluation, SolutionPoint,
    SolverResult,
};
pub use solver::Solver;
pub use status::{PrimalStatus, SolverError, TerminationStatus};
