#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

mod options;
mod persistent;
mod translate;

#[cfg(feature = "benchmark-support")]
#[doc(hidden)]
pub use translate::benchmark_support;

pub use options::{HighsMethod, HighsOptions, HighsPresolve};
pub use persistent::HighsPersistent;
pub use translate::solve;

use oximo_core::{Model, ModelKind};
use oximo_solver::{PersistentSolver, Solver, SolverError, SolverResult};

/// HiGHS solver handle.
///
/// [`Solver::solve`] builds a fresh HiGHS instance for each call, so models can be
/// re-used or shared freely. For repeated solves of one model (parameter sweeps,
/// sensitivity studies, rolling horizons), build a resident handle with
/// [`Highs::persistent`](PersistentSolver::persistent).
#[derive(Debug, Default, Clone, Copy)]
pub struct Highs;

/// Display name for this backend; the single source for both [`Solver::name`]
/// and the `solver_name` stamped on every [`SolverResult`].
pub(crate) const NAME: &str = "HiGHS";

/// The model kinds HiGHS can solve: linear models and quadratic-objective QP.
pub(crate) const fn supported(kind: ModelKind) -> bool {
    matches!(kind, ModelKind::LP | ModelKind::MILP | ModelKind::QP)
}

impl Solver for Highs {
    type Options = HighsOptions;

    fn name(&self) -> &str {
        NAME
    }

    fn supports(&self, kind: ModelKind) -> bool {
        supported(kind)
    }

    fn domain_support(&self, _domain: oximo_core::Domain) -> oximo_solver::Capability {
        oximo_solver::Capability::Native
    }

    fn solve(&mut self, model: &Model, opts: &HighsOptions) -> Result<SolverResult, SolverError> {
        translate::solve(model, opts)
    }
}

impl PersistentSolver for Highs {
    type Handle = HighsPersistent;

    fn persistent(&self) -> HighsPersistent {
        HighsPersistent::new()
    }
}

impl<S: oximo_solver::translation::ScalarBoundSet>
    oximo_solver::TypedConstraintTranslator<oximo_core::ScalarFunctionData<oximo_core::Affine>, S>
    for Highs
where
    oximo_core::ScalarFunctionData<oximo_core::Affine>: oximo_core::FunctionInSet<S>,
{
    type Output = oximo_solver::translation::AffineRow;

    fn translate(
        &self,
        function: &oximo_core::ScalarFunctionData<oximo_core::Affine>,
        set: &S,
        context: &oximo_solver::TranslationContext,
    ) -> Result<Self::Output, oximo_solver::SolverError> {
        oximo_solver::TypedConstraintTranslator::translate(
            &oximo_solver::AffineRowTranslator,
            function,
            set,
            context,
        )
    }
}
