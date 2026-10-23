use crate::{
    Scip, ScipMapping, ScipOptions, ScipPluginRegistry, ScipSolveOutput,
    plugins::{Shared, SharedState},
    translate::Snapshot,
};
use oximo_core::{Model, ModelKind, SosType, VarId};
use oximo_solver::resident::PersistentPreparation;
use oximo_solver::{Solver, SolverError, SolverResult};
use russcip::{Model as Native, ProblemCreated, ProblemOrSolving, Solved};
use rustc_hash::FxHashMap;
use std::sync::Arc;
use std::{cell::RefCell, rc::Rc};

struct State {
    native: Native<Solved>,
    mapping: ScipMapping,
    snapshot: Arc<Snapshot>,
    options: ScipOptions,
    shared: SharedState,
    incumbent: FxHashMap<VarId, f64>,
}

/// Reuses original SCIP problems on unchanged and append-only solves. Existing
/// linear coefficient edits update original rows. Other entry edits, option
/// changes, and plugin-enabled solves rebuild. SCIP's
/// transformed search tree and LP basis are not retained by `free_transform`.
#[derive(Default)]
pub struct ScipPersistent {
    solver: Scip,
    state: Option<State>,
    preparation: PersistentPreparation,
    builds: u64,
    reuses: u64,
}
impl std::fmt::Debug for ScipPersistent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScipPersistent")
            .field("resident", &self.state.is_some())
            .field("builds", &self.builds)
            .field("reuses", &self.reuses)
            .finish_non_exhaustive()
    }
}
impl ScipPersistent {
    pub fn new() -> Self {
        Self::default()
    }
    pub(crate) fn with_solver(solver: Scip) -> Self {
        Self { solver, ..Self::default() }
    }
    pub fn reset(&mut self) {
        self.state = None;
        self.preparation.clear();
    }
    pub fn build_count(&self) -> u64 {
        self.builds
    }
    pub fn reuse_count(&self) -> u64 {
        self.reuses
    }
    /// Solve with the resident SCIP problem and return detailed results.
    ///
    /// # Errors
    ///
    /// Returns translation, registration, options, or solve errors. Any error
    /// clears the resident state so the next solve starts from a fresh problem.
    pub fn solve_detailed(
        &mut self,
        model: &Model,
        opts: &ScipOptions,
    ) -> Result<ScipSolveOutput, SolverError> {
        let result = crate::boundary(|| self.run(model, opts));
        if result.is_err() {
            self.state = None;
            self.preparation.clear();
        } else {
            self.preparation.commit();
        }
        result
    }
    fn run(&mut self, model: &Model, opts: &ScipOptions) -> Result<ScipSolveOutput, SolverError> {
        if opts.concurrent && self.solver.factory.is_some() {
            return Err(crate::backend("concurrent solving cannot copy custom Rust plugins"));
        }
        crate::check_name(&model.name)?;
        if model.has_active_psd_constraints() {
            return Err(SolverError::UnsupportedConstraint("PSD"));
        }
        let prepared = self.preparation.prepare(model)?;
        let snapshot = match &self.state {
            Some(state) => state.snapshot.refresh(model, &prepared, self.preparation.change())?,
            None => Arc::new(Snapshot::prepare_view(model, &prepared)?),
        };
        let old = self.state.take();
        let dense_edits = matches!(self.preparation.change(),oximo_solver::resident::PreparationChange::Selected(targets)
            if targets.len() > snapshot.rows.len().div_ceil(8).max(16));

        // On convex QP objectives, original-row coefficient mutation can make
        // SCIP's subsequent presolve/optimization substantially slower than a
        // cold build.
        let coefficient_reuse =
            !dense_edits && !matches!(snapshot.kind, ModelKind::QP | ModelKind::MIQP);
        let reusable = old.as_ref().is_some_and(|s| {
            self.solver.factory.is_none()
                && s.options == *opts
                && (Arc::ptr_eq(&snapshot, &s.snapshot)
                    || snapshot.extends(&s.snapshot)
                    || (coefficient_reuse && snapshot.linear_edits_compatible(&s.snapshot)))
        });
        let previous = old
            .as_ref()
            .filter(|s| snapshot.columns.starts_with(&s.snapshot.columns))
            .map(|s| s.incumbent.clone())
            .unwrap_or_default();
        let (mut native, mapping, shared) = if reusable {
            let s = old.expect("resident state");
            let mut native = s.native.free_transform();
            let mut mapping = s.mapping;
            *s.shared.borrow_mut() = Shared::default();
            if Arc::ptr_eq(&snapshot, &s.snapshot) {
                // Identical numeric data already belongs to the resident model.
            } else if snapshot.extends(&s.snapshot) {
                snapshot.append(&mut native, &mut mapping, Some(&s.snapshot))?;
            } else {
                snapshot.update_linear_rows(&s.snapshot, &mut native, &mapping)?;
            }
            self.reuses += 1;
            (native, mapping, s.shared)
        } else {
            // TODO: russcip should expose safe original-problem bound/objective
            // edits and nonlinear constraint deletion.
            drop(old);
            let (native, mapping) = snapshot.build(opts)?;
            self.builds += 1;
            (native, mapping, Rc::new(RefCell::new(Shared::default())))
        };
        if let Some(factory) = &self.solver.factory {
            factory(&mut ScipPluginRegistry::new(&mut native, &mapping, shared.clone()))?;
        }
        crate::plugins::refresh_prices(&mapping, &shared);
        if !reusable
            && snapshot.kind == ModelKind::LP
            && snapshot.columns.iter().all(|column| column.domain.semi_threshold().is_none())
            && !opts.concurrent
            && self.solver.factory.is_none()
        {
            crate::plugins::install_prices(&mut native, &mapping, shared.clone());
        }
        add_start(&native, &mapping, model, &previous)?;
        let solved =
            if opts.concurrent { native.try_solve_concurrent() } else { native.try_solve() }
                .map_err(crate::backend)?;
        if let Some(error) = shared.borrow().error.as_ref() {
            return Err(crate::backend(error));
        }
        let output = crate::result::collect(
            &solved,
            &mapping,
            &snapshot,
            model,
            &shared,
            &prepared.arena().parameter_snapshot(),
        )?;
        let incumbent = output.result.primal().cloned().unwrap_or_default();
        self.state = Some(State {
            native: solved,
            mapping,
            snapshot,
            options: opts.clone(),
            shared,
            incumbent,
        });
        Ok(output)
    }
}
impl Solver for ScipPersistent {
    type Options = ScipOptions;
    fn name(&self) -> &str {
        "SCIP"
    }

    fn supports(&self, kind: ModelKind) -> bool {
        self.solver.supports(kind)
    }

    fn domain_support(&self, domain: oximo_core::Domain) -> oximo_solver::Capability {
        self.solver.domain_support(domain)
    }

    fn supports_sos(&self, kind: SosType) -> bool {
        self.solver.supports_sos(kind)
    }

    fn supports_indicators(&self) -> bool {
        self.solver.supports_indicators()
    }

    fn supports_model(&self, model: &Model) -> bool {
        self.solver.supports_model(model)
    }

    fn validate_model_extra(
        &self,
        model: &Model,
        opts: &Self::Options,
    ) -> Vec<oximo_solver::CapabilityDiagnostic> {
        self.solver.validate_model_extra(model, opts)
    }

    fn solve(&mut self, model: &Model, opts: &ScipOptions) -> Result<SolverResult, SolverError> {
        Ok(self.solve_detailed(model, opts)?.result)
    }
}

fn add_start(
    native: &Native<ProblemCreated>,
    mapping: &ScipMapping,
    model: &Model,
    previous: &FxHashMap<VarId, f64>,
) -> Result<(), SolverError> {
    let values = model
        .variables()
        .iter()
        .filter_map(|v| v.initial.or_else(|| previous.get(&v.id).copied()).map(|x| (v.id, x)))
        .collect::<Vec<_>>();
    if values.is_empty() {
        return Ok(());
    }
    for &(_, value) in &values {
        crate::finite(value)?;
    }
    // Partial starts permit new variables and nonlinear objective auxiliaries
    // to be completed by SCIP rather than incorrectly fixing unknowns to zero.
    let sol = native.create_partial_sol();
    for (id, value) in values {
        sol.set_val(&mapping.variables[id.index()], value);
    }
    let _accepted = native.add_sol(sol);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximo_core::prelude::*;

    #[test]
    fn inverted_row_bounds_rebuild_after_coefficient_edits() {
        let model = Model::new("inverted resident row");
        param!(model, p = 1.0);
        variable!(model, 0.0 <= x <= 3.0);
        constraint!(model, inverted, 2.0 <= p * x <= 1.0);
        objective!(model, Min, x);
        let opts = ScipOptions::default();
        let mut solver = ScipPersistent::new();

        let first = solver.solve(&model, &opts).unwrap();
        assert_eq!(first.termination, oximo_solver::TerminationStatus::Infeasible);
        assert_eq!(solver.build_count(), 1);

        p.set_param_value(2.0).unwrap();
        let second = solver.solve(&model, &opts).unwrap();
        assert_eq!(second.termination, oximo_solver::TerminationStatus::Infeasible);
        assert_eq!(solver.build_count(), 2);
    }

    #[test]
    fn unused_parameter_updates_reuse_numeric_snapshot_and_preserve_result_snapshots() {
        let m = Model::new("resident snapshot reuse");

        param!(m, unused = 1.0);
        variable!(m, 0.0 <= x <= 2.0);
        objective!(m, Max, x);

        let mut solver = ScipPersistent::new();
        let opts = ScipOptions::default();
        let first = solver.solve(&m, &opts).unwrap();
        let snapshot = Arc::clone(&solver.state.as_ref().unwrap().snapshot);
        for value in [2.0, 3.0] {
            unused.set_param_value(value).unwrap();
            let result = solver.solve(&m, &opts).unwrap();
            assert_eq!(result.value_of(unused), Ok(Some(value)));
            assert_eq!(first.value_of(unused), Ok(Some(1.0)));
            assert!(Arc::ptr_eq(&snapshot, &solver.state.as_ref().unwrap().snapshot));
        }
        assert_eq!(solver.build_count(), 1);
    }
}
