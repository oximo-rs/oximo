#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_cfg))]

mod options;
mod persistent;
mod translate;

#[cfg(feature = "benchmark-support")]
#[doc(hidden)]
pub use translate::benchmark_support;

#[cfg(feature = "sdp")]
#[cfg_attr(docsrs, doc(cfg(feature = "sdp")))]
pub use options::ClarabelChordalMerge;
pub use options::{ClarabelDirectSolve, ClarabelOptions};
pub use persistent::ClarabelPersistent;
pub use translate::solve;

use oximo_core::{Model, ModelKind};
use oximo_solver::{
    ExpressionOperator, MODEL_KINDS, PersistentSolver, Solver, SolverError, SolverProfile,
    SolverResult,
};
use std::{borrow::Cow, sync::LazyLock};

/// Clarabel solver handle.
///
/// Clarabel is a pure-Rust interior-point solver for convex conic programs:
/// linear programs, convex quadratic objectives, and second-order cone
/// constraints. PSD constraints require the optional SDP feature and BLAS/LAPACK.
#[derive(Debug, Default, Clone, Copy)]
pub struct Clarabel;

/// Display name for this backend; the single source for both [`Solver::name`]
/// and the `solver_name` stamped on every [`SolverResult`].
pub(crate) const NAME: &str = "Clarabel";

/// The model kinds Clarabel can solve: continuous LP, quadratic-objective QP,
/// and SOCP.
/// QCP is out until convex quadratic constraints are reformulated to SOC.
pub(crate) const fn supported(kind: ModelKind) -> bool {
    matches!(kind, ModelKind::LP | ModelKind::QP | ModelKind::SOCP)
        || (cfg!(feature = "sdp") && matches!(kind, ModelKind::SDP))
}

impl Solver for Clarabel {
    type Options = ClarabelOptions;

    fn profile(&self, _opts: &Self::Options) -> Cow<'_, SolverProfile> {
        Cow::Borrowed(&PROFILE)
    }

    fn name(&self) -> &str {
        NAME
    }

    fn solve(
        &mut self,
        model: &Model,
        opts: &ClarabelOptions,
    ) -> Result<SolverResult, SolverError> {
        translate::solve(model, opts)
    }
}

// Clarabel options do not change its supported interfaces. Borrow the immutable
// declaration on the solve path instead of allocating it for every check.
static PROFILE: LazyLock<SolverProfile> = LazyLock::new(build_profile);

fn build_profile() -> SolverProfile {
    use oximo_expr::ExprClass::{Linear, Quadratic};

    SolverProfile::new(NAME)
        .with_scalar_constraints([Linear])
        .with_scalar_objectives([Linear, Quadratic])
        .with_soc(true)
        .with_kinds(MODEL_KINDS.into_iter().filter(|kind| supported(*kind)))
        .with_domains([oximo_solver::VariableDomain::Real])
        .with_operators(ExpressionOperator::ALL.iter().copied())
        .with_psd(cfg!(feature = "sdp"))
}

impl PersistentSolver for Clarabel {
    type Handle = ClarabelPersistent;

    fn persistent(&self) -> ClarabelPersistent {
        ClarabelPersistent::new()
    }
}

impl<S: oximo_solver::translation::ScalarBoundSet>
    oximo_solver::TypedConstraintTranslator<oximo_core::ScalarFunctionData<oximo_core::Affine>, S>
    for Clarabel
where
    oximo_core::ScalarFunctionData<oximo_core::Affine>: oximo_core::FunctionInSet<S>,
{
    type Output = oximo_solver::translation::AffineRow;

    fn translate(
        &self,
        constraint: oximo_solver::PreparedConstraint<
            '_,
            oximo_core::ScalarFunctionData<oximo_core::Affine>,
            S,
        >,
    ) -> Result<Self::Output, oximo_solver::SolverError> {
        oximo_solver::TypedConstraintTranslator::translate(
            &oximo_solver::AffineRowTranslator,
            constraint,
        )
    }
}
