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

use oximo_core::{Model, ModelKind, SosType};
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

/// Supported Models.
pub(crate) const SUPPORTED_KINDS: &[ModelKind] = &[
    ModelKind::LP,
    ModelKind::MILP,
    ModelKind::QP,
    ModelKind::MIQP,
    ModelKind::QCP,
    ModelKind::MIQCP,
    ModelKind::SOCP,
    ModelKind::MISOCP,
    ModelKind::NLP,
    ModelKind::MINLP,
];

/// Gurobi handles every kind oximo classifies: linear, quadratic objectives
/// and constraints, second-order cones (lowered to quadratic rows), and
/// general nonlinear expressions.
pub(crate) fn supported(kind: ModelKind) -> bool {
    SUPPORTED_KINDS.contains(&kind)
}

impl Solver for Gurobi {
    type Options = GurobiOptions;

    fn name(&self) -> &str {
        NAME
    }

    fn supports(&self, kind: ModelKind) -> bool {
        supported(kind)
    }

    fn supports_sos(&self, _sos_type: SosType) -> bool {
        true
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
