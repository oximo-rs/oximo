#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

mod options;
mod solver_options;
mod translate;

pub use options::{GamsOptions, GamsSolver};
pub use solver_options::{
    GamsBaronOptions, GamsCbcCuts, GamsCbcOptions, GamsCbcPresolve, GamsCplexMipEmphasis,
    GamsCplexOptions, GamsGurobiMipFocus, GamsGurobiOptions, GamsHighsOptions, GamsHighsPresolve,
    GamsHighsSolver, GamsIpoptLinearSolver, GamsIpoptMuStrategy, GamsIpoptOptions,
    GamsKnitroAlgorithm, GamsKnitroOptions, GamsMosekOptions, GamsScipOptions, GamsSolverConfig,
    GamsXpressOptions,
};
pub use translate::solve;

use oximo_core::{Model, ModelKind};
use oximo_solver::{Solver, SolverError, SolverResult};

/// GAMS solver backend.
///
/// Writes the model to a temporary `.gms` file, invokes the GAMS executable,
/// and returns the parsed [`SolverResult`].
#[derive(Debug, Default, Clone)]
pub struct Gams {
    /// Optional override for the GAMS executable path. When `None`, `"gams"` is
    /// looked up from the system `PATH`. Overridden per-call by
    /// [`GamsOptions::gams_path`].
    pub exec: Option<String>,
}

impl Gams {
    /// Create a backend that uses `gams` from `PATH`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a backend pointing at an explicit GAMS executable path.
    pub fn with_exec(path: impl Into<String>) -> Self {
        Self { exec: Some(path.into()) }
    }
}

/// Display name for this backend; the single source for both [`Solver::name`]
/// and the `solver_name` stamped on every [`SolverResult`].
pub(crate) const NAME: &str = "GAMS";

impl Solver for Gams {
    type Options = GamsOptions;

    fn name(&self) -> &str {
        NAME
    }

    fn supports(&self, kind: ModelKind) -> bool {
        matches!(
            kind,
            ModelKind::LP
                | ModelKind::MILP
                | ModelKind::QP
                | ModelKind::MIQP
                | ModelKind::NLP
                | ModelKind::MINLP
        )
    }

    fn solve(&mut self, model: &Model, opts: &GamsOptions) -> Result<SolverResult, SolverError> {
        translate::solve(model, opts, self.exec.as_deref())
    }
}
