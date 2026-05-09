use thiserror::Error;

#[derive(Clone, Debug, PartialEq)]
pub enum SolverStatus {
    Optimal,
    Feasible,
    Infeasible,
    Unbounded,
    TimeLimit,
    NumericError,
    NotSolved,
    Other(String),
}

impl SolverStatus {
    pub fn has_solution(&self) -> bool {
        matches!(self, Self::Optimal | Self::Feasible)
    }
}

#[derive(Debug, Error)]
pub enum SolverError {
    #[error("solver does not support model kind {0:?}")]
    UnsupportedKind(oximo_core::ModelKind),
    #[error("model is missing an objective")]
    NoObjective,
    #[error("nonlinear constructs are not supported by this backend")]
    Nonlinear,
    #[error("backend error: {0}")]
    Backend(String),
    #[error(transparent)]
    Core(#[from] oximo_core::Error),
}
