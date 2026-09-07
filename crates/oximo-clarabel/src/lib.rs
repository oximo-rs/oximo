#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_cfg))]

mod options;
mod persistent;
mod translate;

#[cfg(feature = "benchmark-support")]
#[doc(hidden)]
pub use translate::benchmark_support;

pub use options::{ClarabelDirectSolve, ClarabelOptions};
pub use persistent::ClarabelPersistent;
pub use translate::solve;

use oximo_core::{Model, ModelKind};
use oximo_solver::{PersistentSolver, Solver, SolverError, SolverResult};

/// Clarabel solver handle.
///
/// Clarabel is a pure-Rust interior-point solver for convex conic programs:
/// linear programs, convex quadratic objectives, and second-order cone
/// constraints.
#[derive(Debug, Default, Clone, Copy)]
pub struct Clarabel;

/// Display name for this backend; the single source for both [`Solver::name`]
/// and the `solver_name` stamped on every [`SolverResult`].
pub(crate) const NAME: &str = "Clarabel";
pub(crate) const SUPPORTED_KINDS: &[ModelKind] = &[ModelKind::LP, ModelKind::QP, ModelKind::SOCP];

/// The model kinds Clarabel can solve: continuous LP, quadratic-objective QP,
/// and SOCP.
/// QCP is out until convex quadratic constraints are reformulated to SOC.
pub(crate) fn supported(kind: ModelKind) -> bool {
    SUPPORTED_KINDS.contains(&kind)
}

impl Solver for Clarabel {
    type Options = ClarabelOptions;

    fn name(&self) -> &str {
        NAME
    }

    fn supports(&self, kind: ModelKind) -> bool {
        supported(kind)
    }

    fn solve(
        &mut self,
        model: &Model,
        opts: &ClarabelOptions,
    ) -> Result<SolverResult, SolverError> {
        translate::solve(model, opts)
    }
}

impl PersistentSolver for Clarabel {
    type Handle = ClarabelPersistent;

    fn persistent(&self) -> ClarabelPersistent {
        ClarabelPersistent::new()
    }
}
