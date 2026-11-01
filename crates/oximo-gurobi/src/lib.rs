#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

mod nonlinear;
mod options;
mod persistent;
mod translate;

#[cfg(feature = "benchmark-support")]
#[doc(hidden)]
pub use translate::benchmark_support;

pub use gurobi_rs::callback::{Callback, CallbackLocation, CallbackMask, CbResult, Where};
pub use gurobi_rs::{GRB_METHOD_PDHG, Opcode, Status};
pub use options::{GurobiOptions, GurobiPresolve};
pub use persistent::GurobiPersistent;
pub use translate::solve;

use oximo_core::{Model, ModelKind};
use oximo_solver::{
    ExpressionOperator, Iis, InfeasibilityDiagnosis, MODEL_KINDS, PersistentSolver, Solver,
    SolverError, SolverProfile, SolverResult,
};
use std::{borrow::Cow, sync::LazyLock};

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

static PROFILE: LazyLock<SolverProfile> = LazyLock::new(build_profile);

fn build_profile() -> SolverProfile {
    use oximo_expr::ExprClass::{Linear, Nonlinear, Quadratic};
    use oximo_solver::VariableDomain::{Binary, Integer, Real, SemiContinuous, SemiInteger};

    SolverProfile::new(NAME)
        .with_scalar_constraints([Linear, Quadratic, Nonlinear])
        .with_scalar_objectives([Linear, Quadratic, Nonlinear])
        .with_soc(true)
        .with_kinds(MODEL_KINDS.into_iter().filter(|kind| supported(*kind)))
        .with_domains([Real, Integer, Binary, SemiContinuous, SemiInteger])
        .with_operators(
            ExpressionOperator::ALL.iter().copied().filter(|operator| supports_operator(*operator)),
        )
        .with_sos([oximo_core::SosType::Sos1, oximo_core::SosType::Sos2])
        .with_indicators(true)
}

impl Solver for Gurobi {
    type Options = GurobiOptions;

    fn profile(&self, _opts: &Self::Options) -> Cow<'_, SolverProfile> {
        Cow::Borrowed(&PROFILE)
    }

    fn name(&self) -> &str {
        NAME
    }

    fn solve(&mut self, model: &Model, opts: &GurobiOptions) -> Result<SolverResult, SolverError> {
        translate::solve(model, opts)
    }
}

impl Gurobi {
    /// Solve while receiving events from Gurobi's callback API.
    ///
    /// # Errors
    /// Returns a [`SolverError`] for model construction, callback, or Gurobi
    /// optimization failures.
    pub fn solve_with_callback<F>(
        &mut self,
        model: &Model,
        opts: &GurobiOptions,
        callback: &mut F,
    ) -> Result<SolverResult, SolverError>
    where
        F: Callback,
    {
        translate::solve_with_callback(model, opts, callback, None)
    }

    /// Solve while receiving only the selected Gurobi callback locations.
    ///
    /// # Errors
    /// Returns a [`SolverError`] for model construction, callback, or Gurobi
    /// optimization failures.
    pub fn solve_with_callback_filtered<F>(
        &mut self,
        model: &Model,
        opts: &GurobiOptions,
        callback: &mut F,
        locations: impl Into<CallbackMask>,
    ) -> Result<SolverResult, SolverError>
    where
        F: Callback,
    {
        translate::solve_with_callback(model, opts, callback, Some(locations.into()))
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

/// Translation and preflight share this vocabulary declaration.
pub(crate) fn supports_operator(operator: oximo_solver::ExpressionOperator) -> bool {
    use oximo_expr::UnaryOp::{Abs, Cos, Exp, Exp2, Log, Log2, Log10, Neg, Sin, Sqrt, Tan, Tanh};
    use oximo_solver::ExpressionOperator::{Atan2, Unary};
    match operator {
        Unary(op) => matches!(
            op,
            Neg | Abs | Sqrt | Exp | Exp2 | Log | Log2 | Log10 | Sin | Cos | Tan | Tanh
        ),
        Atan2 => false,
        _ => true,
    }
}
