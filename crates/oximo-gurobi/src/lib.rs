#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

mod nonlinear;
mod options;
mod persistent;
mod translate;

pub use options::{GurobiOptions, GurobiPresolve};
pub use persistent::GurobiPersistent;
pub use translate::solve;

use oximo_core::{Model, ModelKind};
use oximo_solver::{
    Iis, InfeasibilityDiagnosis, PersistentSolver, Solver, SolverError, SolverResult,
};

/// Gurobi solver handle.
///
/// [`Solver::solve`] builds a fresh Gurobi model for each call. For repeated solves
/// of one model (parameter sweeps, sensitivity studies, rolling horizons), build a
/// resident handle with [`Gurobi::persistent`](PersistentSolver::persistent).
#[derive(Debug, Default, Clone, Copy)]
pub struct Gurobi;

/// Display name for this backend; the single source for both [`Solver::name`]
/// and the `solver_name` stamped on every [`SolverResult`].
pub(crate) const NAME: &str = "Gurobi";

/// Gurobi handles every kind oximo classifies: linear, quadratic objectives
/// and constraints, second-order cones (lowered to quadratic rows), and
/// general nonlinear expressions.
pub(crate) const fn supported(kind: ModelKind) -> bool {
    matches!(
        kind,
        ModelKind::LP
            | ModelKind::MILP
            | ModelKind::QP
            | ModelKind::MIQP
            | ModelKind::QCP
            | ModelKind::MIQCP
            | ModelKind::SOCP
            | ModelKind::MISOCP
            | ModelKind::NLP
            | ModelKind::MINLP
    )
}

impl Solver for Gurobi {
    type Options = GurobiOptions;

    fn name(&self) -> &str {
        NAME
    }

    fn supports(&self, kind: ModelKind) -> bool {
        supported(kind)
    }

    fn solve(&mut self, model: &Model, opts: &GurobiOptions) -> Result<SolverResult, SolverError> {
        translate::solve(model, opts)
    }
}

impl PersistentSolver for Gurobi {
    type Handle = GurobiPersistent;

    fn persistent(&self) -> GurobiPersistent {
        GurobiPersistent::new()
    }
}

impl InfeasibilityDiagnosis for Gurobi {
    fn compute_iis(&mut self, model: &Model, opts: &GurobiOptions) -> Result<Iis, SolverError> {
        translate::compute_iis(model, opts)
    }
}
