use crate::{
    Scip, ScipMapping, ScipOptions, ScipPluginRegistry, ScipSolveOutput,
    plugins::{Shared, SharedState},
    translate::Snapshot,
};
use oximo_core::{Model, ModelKind, VarId};
use oximo_solver::SolutionValues;
use oximo_solver::resident::PersistentPreparation;
use oximo_solver::{Solver, SolverError, SolverResult};
use russcip::{Model as Native, ProblemCreated, ProblemOrSolving, Solved};
use std::sync::Arc;
use std::{cell::RefCell, rc::Rc};

struct State {
    native: Native<Solved>,
    mapping: ScipMapping,
    snapshot: Arc<Snapshot>,
    options: ScipOptions,
    shared: SharedState,
    incumbent: SolutionValues<VarId>,
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
    /// Preflight incompatibility preserves the resident state.
    pub fn solve_detailed(
        &mut self,
        model: &Model,
        opts: &ScipOptions,
    ) -> Result<ScipSolveOutput, SolverError> {
        self.profile(opts).check_model(model).into_result()?;

        // Translation is also local data validation. Reuse its result rather
        // than preparing a second snapshot just for preflight.
        let result = crate::boundary(|| self.run(model, opts));
        if result.is_err() {
            if !matches!(&result, Err(SolverError::IncompatibleModel(_))) {
                self.state = None;
            }
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
        crate::check_name(&model.name).map_err(crate::translation_rejection)?;
        if model.has_active_psd_constraints() {
            return Err(SolverError::UnsupportedConstraint("PSD"));
        }
        let prepared = self.preparation.prepare(model)?;
        let snapshot = match &self.state {
            Some(state) => state.snapshot.refresh(model, &prepared, self.preparation.change()),
            None => Snapshot::prepare_view(model, &prepared).map(Arc::new),
        }
        .map_err(crate::translation_rejection)?;
        let mut old = self.state.take();
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
            .as_mut()
            .filter(|s| snapshot.columns.starts_with(&s.snapshot.columns))
            .map(|s| std::mem::take(&mut s.incumbent))
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
    fn profile(&self, opts: &Self::Options) -> std::borrow::Cow<'_, oximo_solver::SolverProfile> {
        oximo_solver::Solver::profile(&self.solver, opts)
    }
    fn validate_requirements(
        &self,
        model: &oximo_core::Model,
        opts: &Self::Options,
        requirements: &oximo_solver::ModelRequirements,
    ) -> Vec<oximo_solver::RequirementDiagnostic> {
        oximo_solver::Solver::validate_requirements(&self.solver, model, opts, requirements)
    }
    fn name(&self) -> &str {
        "SCIP"
    }

    fn solve(&mut self, model: &Model, opts: &ScipOptions) -> Result<SolverResult, SolverError> {
        Ok(self.solve_detailed(model, opts)?.result)
    }
}

fn add_start(
    native: &Native<ProblemCreated>,
    mapping: &ScipMapping,
    model: &Model,
    previous: &SolutionValues<VarId>,
) -> Result<(), SolverError> {
    let values = model
        .variables()
        .iter()
        .filter_map(|v| v.initial.or_else(|| previous.get(v.id)).map(|x| (v.id, x)))
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
    #[test]
    fn preflight_rejection_preserves_the_resident_snapshot_and_build_count() {
        let model = Model::new("retained resident");
        variable!(model, 1.0 <= x <= 3.0);
        objective!(model, Min, x);

        let options = ScipOptions::default();
        let mut solver = ScipPersistent::new();
        solver.solve(&model, &options).unwrap();
        let snapshot = Arc::clone(&solver.state.as_ref().unwrap().snapshot);

        let rejected = Model::new("unsupported replacement");
        variable!(rejected, z);
        objective!(rejected, Min, z.tan());

        assert!(matches!(
            solver.solve(&rejected, &options),
            Err(SolverError::IncompatibleModel(_))
        ));
        assert!(Arc::ptr_eq(&snapshot, &solver.state.as_ref().unwrap().snapshot));

        solver.solve(&model, &options).unwrap();
        assert!(Arc::ptr_eq(&snapshot, &solver.state.as_ref().unwrap().snapshot));
        assert_eq!(solver.build_count(), 1);
    }

    #[test]
    fn local_preparation_rejection_preserves_native_state() {
        let model = Model::new("retained after data validation");
        variable!(model, 1.0 <= x <= 3.0);
        objective!(model, Min, x);

        let options = ScipOptions::default();
        let mut solver = ScipPersistent::new();
        solver.solve(&model, &options).unwrap();
        let invalid = Model::new("invalid semi domain");
        let z = invalid
            .__var("z")
            .domain(Domain::SemiContinuous { threshold: -1.0 })
            .lb(0.0)
            .ub(3.0)
            .build();
        objective!(invalid, Min, z);

        let expected = solver.validate_model(&invalid, &options).to_string();
        let error = solver.solve(&invalid, &options).unwrap_err();
        assert!(matches!(error, SolverError::IncompatibleModel(_)));
        assert_eq!(error.to_string(), expected);
        assert!(solver.state.is_some());

        let result = solver.solve(&model, &options).unwrap();
        assert!((result.objective().unwrap() - 1.0).abs() < 1e-7);
        assert_eq!(solver.build_count(), 1);
    }
}
