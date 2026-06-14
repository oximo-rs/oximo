#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

mod options;
mod translate;

pub use options::{HighsMethod, HighsOptions, HighsPresolve};
pub use translate::solve;

use oximo_core::{Model, ModelKind};
use oximo_solver::{Solver, SolverError, SolverResult};

/// HiGHS solver handle. Cheap to construct. The actual HiGHS instance is
/// created per `solve` call so models can be re-used or shared across solves.
///
/// TODO: Can we do this better in the future?
#[derive(Debug, Default, Clone, Copy)]
pub struct Highs;

/// Display name for this backend; the single source for both [`Solver::name`]
/// and the `solver_name` stamped on every [`SolverResult`].
pub(crate) const NAME: &str = "HiGHS";

impl Solver for Highs {
    type Options = HighsOptions;

    fn name(&self) -> &str {
        NAME
    }

    fn supports(&self, kind: ModelKind) -> bool {
        matches!(kind, ModelKind::LP | ModelKind::MILP | ModelKind::QP)
    }

    fn solve(&mut self, model: &Model, opts: &HighsOptions) -> Result<SolverResult, SolverError> {
        translate::solve(model, opts)
    }
}
