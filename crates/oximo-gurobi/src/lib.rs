#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

mod nonlinear;
mod options;
mod translate;

pub use options::{GurobiOptions, GurobiPresolve};
pub use translate::solve;

use oximo_core::{Model, ModelKind};
use oximo_solver::{Solver, SolverError, SolverResult};

#[derive(Debug, Default, Clone, Copy)]
pub struct Gurobi;

/// Display name for this backend; the single source for both [`Solver::name`]
/// and the `solver_name` stamped on every [`SolverResult`].
pub(crate) const NAME: &str = "Gurobi";

impl Solver for Gurobi {
    type Options = GurobiOptions;

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

    fn solve(&mut self, model: &Model, opts: &GurobiOptions) -> Result<SolverResult, SolverError> {
        translate::solve(model, opts)
    }
}
