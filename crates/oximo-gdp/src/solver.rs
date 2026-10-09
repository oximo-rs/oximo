//! Apply GDP reformulation at solve time without modifying backend options.

use oximo_core::Model;
use oximo_solver::{Solver, SolverError, SolverResult};
use thiserror::Error;

use crate::{GdpError, GdpModelExt, GdpReformulationOptions};

/// A reformulation or backend failure from [`GdpSolver::solve`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum GdpSolveError {
    #[error(transparent)]
    Reformulation(#[from] GdpError),
    #[error(transparent)]
    Solver(#[from] SolverError),
}

/// A solver configured to reformulate pending GDP components before each solve.
/// Construct it through [`GdpSolverExt::with_gdp`].
#[derive(Debug)]
pub struct GdpSolver<S> {
    solver: S,
    options: GdpReformulationOptions,
}

/// Solve-time GDP configuration for any [`Solver`].
pub trait GdpSolverExt: Solver + Sized {
    /// Select a GDP method, or pass options for per-disjunction overrides.
    ///
    /// Reformulation occurs in place when [`GdpSolver::solve`] is called.
    /// Original model handles remain valid.
    #[must_use]
    fn with_gdp(self, options: impl Into<GdpReformulationOptions>) -> GdpSolver<Self> {
        GdpSolver { solver: self, options: options.into() }
    }
}

impl<S: Solver> GdpSolverExt for S {}

impl<S: Solver> GdpSolver<S> {
    /// Reformulate pending GDP in place, then solve with the backend's own options.
    /// The transformation report remains available through `model.gdp_reformulations()`.
    ///
    /// # Errors
    /// Returns [`GdpSolveError::Reformulation`] before calling the backend if GDP
    /// validation fails, leaving the model unchanged. Returns [`GdpSolveError::Solver`]
    /// for backend errors; a successful reformulation remains applied in that case.
    pub fn solve(
        &mut self,
        model: &Model,
        opts: &S::Options,
    ) -> Result<SolverResult, GdpSolveError> {
        model.reformulate_gdp(self.options.clone())?;

        Ok(self.solver.solve(model, opts)?)
    }

    /// Recover the underlying solver, retaining its backend state.
    #[must_use]
    pub fn into_inner(self) -> S {
        self.solver
    }
}
