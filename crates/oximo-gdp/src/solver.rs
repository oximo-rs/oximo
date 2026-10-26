//! Apply GDP reformulation at solve time without modifying backend options.

use std::ops::Range;

use oximo_core::{ConstraintId, Model, ModelId, VarId};
use oximo_solver::reconstruct::{
    ProjectionError, SourceCoordinates, SourceProjection, SourceProjectionPlan,
};
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
    Projection(#[from] ProjectionError),
    #[error(transparent)]
    Solver(#[from] SolverError),
}

/// A solver configured to reformulate pending GDP components before each solve.
/// Construct it through [`GdpSolverExt::with_gdp`].
#[derive(Debug)]
pub struct GdpSolver<S> {
    solver: S,
    options: GdpReformulationOptions,
    /// Reusable membership and activation sets for the last transformed state.
    source: Option<(SourceKey, SourceProjectionPlan)>,
}

/// Solve-time GDP configuration for any [`Solver`].
pub trait GdpSolverExt: Solver + Sized {
    /// Select a GDP method, or pass options for per-disjunction overrides.
    ///
    /// Reformulation occurs in place when [`GdpSolver::solve`] is called.
    /// Original model handles remain valid.
    #[must_use]
    fn with_gdp(self, options: impl Into<GdpReformulationOptions>) -> GdpSolver<Self> {
        GdpSolver { solver: self, options: options.into(), source: None }
    }
}

impl<S: Solver> GdpSolverExt for S {}

impl<S: Solver> GdpSolver<S> {
    /// Reformulate pending GDP in place, solve, and project the result back to
    /// source coordinates.
    ///
    /// The returned result carries only source variables plus the generated
    /// Hull activation variables that [`crate::GdpResultExt`] queries while
    /// other generated variables, rows, and multipliers from GDP, SOS, and indicator
    /// reformulations are dropped. Later user declarations remain source coordinates. The
    /// transformation report remains available through
    /// `model.gdp_reformulations()`, so activation handles stay valid.
    ///
    /// # Errors
    /// Returns [`GdpSolveError::Reformulation`] before calling the backend if GDP
    /// validation fails, leaving the model unchanged. Returns
    /// [`GdpSolveError::Solver`] for backend errors; a successful reformulation
    /// remains applied in that case. Returns [`GdpSolveError::Projection`] if the
    /// backend result does not belong to this model or the captured coordinates
    /// no longer exist.
    pub fn solve(
        &mut self,
        model: &Model,
        opts: &S::Options,
    ) -> Result<SolverResult, GdpSolveError> {
        model.reformulate_gdp(self.options.clone())?;

        // Without any in-place reformulation there are no generated
        // coordinates to remove, so the backend result is already in source
        // coordinates.
        if model.gdp_reformulations().is_empty()
            && model.sos_reformulations().is_empty()
            && model.indicator_reformulations().is_empty()
        {
            return Ok(self.solver.solve(model, opts)?);
        }

        // Capture after reformulation so the first repeated solve hits the
        // cache, but capture before the backend so callback additions belong
        // to the next solve's source coordinates.
        let key = SourceKey::capture(model);
        if self.source.as_ref().is_some_and(|(cached, _)| *cached != key) {
            self.source = None;
        }
        let (_, plan) = self.source.get_or_insert_with(|| (key, source_plan(model)));
        let result = self.solver.solve(model, opts)?;

        Ok(SourceProjection::with_plan(model, plan).project(result)?)
    }

    /// Recover the underlying solver, retaining its backend state.
    #[must_use]
    pub fn into_inner(self) -> S {
        self.solver
    }
}

/// Identity and sizes of a model's source-plus-generated state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SourceKey {
    model: ModelId,
    reports: usize,
    sos_reports: usize,
    indicator_reports: usize,
    variables: usize,
    algebraic_constraints: usize,
    soc_constraints: usize,
    psd_constraints: usize,
}

impl SourceKey {
    fn capture(model: &Model) -> Self {
        Self {
            model: model.id(),
            reports: model.gdp_reformulations().len(),
            sos_reports: model.sos_reformulations().len(),
            indicator_reports: model.indicator_reformulations().len(),
            variables: model.num_variables(),
            algebraic_constraints: model.constraints().algebraic().len(),
            soc_constraints: model.num_soc_constraints(),
            psd_constraints: model.num_psd_constraints(),
        }
    }
}

/// Derive original declaration membership from every registered reformulation.
fn source_plan(model: &Model) -> SourceProjectionPlan {
    let reports = model.gdp_reformulations();
    let total_variables = model.num_variables();
    let total_rows = model.constraints().algebraic().len();
    let provenance = oximo_core::ReformulationProvenance::capture(model);
    let variable_ranges = || provenance.variable_ranges().iter().cloned();
    let row_ranges = || provenance.row_ranges().iter().cloned();
    let variables = total_variables - variable_ranges().map(|range| range.len()).sum::<usize>();
    let rows = total_rows - row_ranges().map(|range| range.len()).sum::<usize>();
    let mut plan = SourceProjectionPlan::new(SourceCoordinates {
        variables,
        algebraic_constraints: rows,
        soc_constraints: model.num_soc_constraints(),
        psd_constraints: model.num_psd_constraints(),
    });
    if !source_is_prefix(variables, variable_ranges()) {
        plan = plan.with_source_variables(
            source_indices(total_variables, variable_ranges())
                .map(|index| VarId(u32::try_from(index).expect("variable index"))),
        );
    }
    if !source_is_prefix(rows, row_ranges()) {
        plan = plan.with_source_rows(
            source_indices(total_rows, row_ranges())
                .map(|index| ConstraintId(u32::try_from(index).expect("constraint index"))),
        );
    }
    plan.retain_variables(reports.iter().flat_map(|report| {
        report
            .hull_variables
            .iter()
            .filter_map(|copy| copy.activation_kind.variable().generated())
            .chain(report.hull_patterns.iter().map(|pattern| pattern.weight))
            .chain(report.hull_branch_sums.iter().map(|sum| sum.weight))
    }))
}

fn source_is_prefix(count: usize, mut generated: impl Iterator<Item = Range<usize>>) -> bool {
    generated.find(|range| !range.is_empty()).is_none_or(|first| first.start == count)
}

/// Gaps between chronologically ordered, non-overlapping generated ranges.
fn source_indices(
    total: usize,
    generated: impl Iterator<Item = Range<usize>>,
) -> impl Iterator<Item = usize> {
    let mut next = 0;
    generated.chain(std::iter::once(total..total)).flat_map(move |range| {
        let source = next..range.start;
        next = range.end;
        source
    })
}
