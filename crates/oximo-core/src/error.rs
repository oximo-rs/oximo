use smol_str::SmolStr;
use thiserror::Error;

/// Errors that can occur during model construction.
/// These are distinct from backend errors, which are opaque and translated into `SolverError`.
#[derive(Debug, Error)]
pub enum Error {
    #[error("variable name {0:?} already registered")]
    DuplicateVar(SmolStr),
    #[error("variable {0:?} not found")]
    UnknownVar(SmolStr),
    #[error("constraint {0:?} not found")]
    UnknownConstraint(SmolStr),
    #[error("objective already set on this model")]
    ObjectiveAlreadySet,
    #[error("nonlinear nodes are not supported by this backend")]
    NonlinearUnsupported,
    #[error("model has no objective")]
    NoObjective,
}

pub type Result<T> = std::result::Result<T, Error>;
