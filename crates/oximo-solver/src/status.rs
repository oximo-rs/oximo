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

#[derive(Error)]
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

// Mirror `Display` in `Debug`. When a `main` returning `Result` propagates an
// error, Rust's `Termination` impl prints it with `{:?}`. The derived `Debug`
// would escape newlines in `Backend` messages (e.g. multi-line GAMS reports)
// onto a single line. These messages are human-facing, so render them as-is.
impl std::fmt::Debug for SolverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}
