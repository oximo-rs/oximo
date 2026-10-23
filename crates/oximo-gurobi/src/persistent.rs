use gurobi_rs::prelude::*;
use oximo_core::{Model, ModelKind, SosType};
use oximo_solver::incremental::{refresh_snapshot, snapshot_prepared};
use oximo_solver::prepare::LoweringContext;
use oximo_solver::resident::{PersistentPreparation, PreparationChange};
use oximo_solver::{Iis, Snapshot, Solver, SolverError, SolverResult};

use crate::GurobiOptions;
use crate::options::apply as apply_options;
use crate::translate::{
    Built, build_prepared, compute_iis_resident, default_env, map_gurobi_err, run_and_collect,
};
use crate::{Callback, CallbackMask};

/// The resident Gurobi model plus the baseline snapshot the fast path diffs against
/// (`None` when the fast path is ineligible).
struct State {
    built: Built,
    snap: Option<Snapshot>,
}

/// A stateful Gurobi solver handle that keeps the built model resident across solves.
///
/// Created by [`Gurobi::persistent`](crate::Gurobi). It is a plain [`Solver`]: call
/// [`solve`](Solver::solve) on it repeatedly. When only objective coefficients or
/// variable bounds changed since the last call it pushes those deltas through
/// Gurobi's attribute API (`Obj`, `LB`, `UB`, `ObjCon`) and Gurobi warm-starts.
/// Parameter changes also update linear rows and replace affected quadratic,
/// nonlinear, conic, or indicator groups together with their owned auxiliaries.
/// A structural change (new rows/columns or flipped integrality or sense) triggers a
/// transparent full rebuild, so every result matches a cold [`Gurobi::solve`](crate::Gurobi).
///
/// Options passed to `solve` are applied to the resident instance and persist across calls.
/// Call [`reset`](Self::reset) for a clean slate.
/// A failed `solve` leaves the handle without a resident model and the next call rebuilds.
#[derive(Default)]
pub struct GurobiPersistent {
    /// Created once on the first rebuild and reused for every subsequent rebuild, so
    /// a structural change doesn't create a fresh Gurobi environment.
    env: Option<Env>,
    state: Option<State>,
    preparation: PersistentPreparation,
    callback_state: bool,
    #[cfg(test)]
    updates: usize,
}

impl std::fmt::Debug for GurobiPersistent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GurobiPersistent").field("resident", &self.state.is_some()).finish()
    }
}

impl GurobiPersistent {
    /// A fresh handle with no model loaded. The first solve builds it.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Drop the resident model so the next [`solve`](Solver::solve) rebuilds from
    /// scratch, clearing any solver options carried on the Gurobi instance.
    pub fn reset(&mut self) {
        self.state = None;
        self.preparation.clear();
        self.callback_state = false;
    }

    /// Compute an irreducible infeasible subsystem for the resident model, reusing the
    /// build and solve from the previous [`solve`](Solver::solve).
    ///
    /// Call this after a [`solve`](Solver::solve) that returned an infeasible status
    /// (see [`TerminationStatus::is_infeasible`](oximo_solver::TerminationStatus::is_infeasible)).
    /// It runs Gurobi's `computeIIS` directly on the resident instance. If the prior
    /// solve left the ambiguous `INF_OR_UNBD` status, it re-optimizes with dual
    /// reductions off to get a definite verdict first.
    ///
    /// For a one-shot diagnosis that builds and solves in a single call, use
    /// [`Gurobi::compute_iis`](oximo_solver::InfeasibilityDiagnosis::compute_iis) instead.
    ///
    /// # Errors
    ///
    /// Returns a [`SolverError`] if there is no resident model (never solved, or a
    /// prior solve failed and cleared it), the resident model is not infeasible, or
    /// Gurobi errors during the IIS computation.
    pub fn compute_iis(&mut self) -> Result<Iis, SolverError> {
        let st = self.state.as_mut().ok_or_else(|| {
            SolverError::Backend(
                "no resident model to diagnose; call solve first (it must be infeasible)".into(),
            )
        })?;
        compute_iis_resident(&mut st.built)
    }

    /// Discard any resident instance and rebuild from the current model state.
    fn rebuild(
        &mut self,
        model: &Model,
        prepared: &LoweringContext<'_>,
        opts: &GurobiOptions,
    ) -> Result<(), SolverError> {
        if model.has_active_psd_constraints() {
            return Err(SolverError::UnsupportedConstraint("PSD"));
        }
        let env = match self.env.as_ref() {
            Some(env) => env,
            None => self.env.insert(default_env()?),
        };
        let built = build_prepared(model, opts, env, prepared)?;
        // The fast path needs a linear snapshot and is unsafe for SC/SI variables
        // (whose Gurobi lower bound is the gap floor, not `Variable::lb`), so
        // those rebuild every solve.
        let snap = if matches!(prepared.kind(), ModelKind::LP | ModelKind::MILP) && !built.has_semi
        {
            Some(snapshot_prepared(model, prepared)?)
        } else {
            None
        };
        self.state = Some(State { built, snap });
        self.callback_state = false;
        Ok(())
    }

    /// Re-read the model, update the resident instance in place (or rebuild), and
    /// solve.
    fn solve_resident(
        &mut self,
        model: &Model,
        opts: &GurobiOptions,
    ) -> Result<SolverResult, SolverError> {
        let kind = self.update_resident(model, opts)?;
        let st = self.state.as_mut().expect("state present before solve");
        run_and_collect(&mut st.built, kind, model)
    }

    fn update_resident(
        &mut self,
        model: &Model,
        opts: &GurobiOptions,
    ) -> Result<ModelKind, SolverError> {
        if model.has_active_psd_constraints() {
            return Err(SolverError::UnsupportedConstraint("PSD"));
        }
        let prepared = self.preparation.prepare(model)?;
        let kind = prepared.kind();
        let callback_rebuild = self.callback_state
            && !matches!(self.preparation.change(), PreparationChange::Unchanged);
        if !callback_rebuild
            && let Some(st) = self.state.as_mut()
            && st.built.refresh(&prepared, self.preparation.change())?
        {
            apply_options(&mut st.built.model, opts).map_err(map_gurobi_err)?;
            st.built.model_id = model.id();
            let row_changes = matches!(self.preparation.change(),PreparationChange::Selected(targets)
                if targets.iter().any(|target| !matches!(target,oximo_core::ParameterTarget::Objective | oximo_core::ParameterTarget::VariableBound(_))));
            match &st.snap {
                Some(base) if !row_changes && matches!(kind, ModelKind::LP | ModelKind::MILP) => {
                    if let std::borrow::Cow::Owned(snapshot) =
                        refresh_snapshot(model, &prepared, self.preparation.change(), base)?
                    {
                        st.snap = Some(snapshot);
                    }
                }
                _ => st.snap = None,
            }
            #[cfg(test)]
            {
                self.updates += 1;
            }
            return Ok(kind);
        }
        let mut updated = false;
        if !callback_rebuild
            && matches!(kind, ModelKind::LP | ModelKind::MILP)
            && let Some(st) = self.state.as_mut()
            && let Some(base) = st.snap.as_ref()
        {
            let snap = snapshot_prepared(model, &prepared)?;
            if snap.fingerprint == base.fingerprint {
                push_deltas(&mut st.built, base, &snap, opts)?;
                st.built.model_id = model.id();
                st.built.parameters = snap.parameters.clone();
                st.snap = Some(snap);
                updated = true;
            }
        }
        if !updated {
            self.rebuild(model, &prepared, opts)?;
        }
        #[cfg(test)]
        if updated {
            self.updates += 1;
        }

        Ok(kind)
    }

    fn solve_resident_with_callback<F>(
        &mut self,
        model: &Model,
        opts: &GurobiOptions,
        callback: &mut F,
        mask: Option<CallbackMask>,
    ) -> Result<SolverResult, SolverError>
    where
        F: Callback,
    {
        let kind = self.update_resident(model, opts)?;
        self.callback_state = true;
        let st = self.state.as_mut().expect("state present before solve");
        crate::translate::run_and_collect_with_callback(&mut st.built, kind, model, callback, mask)
    }

    /// Solve the resident model while receiving Gurobi callback events.
    ///
    /// # Errors
    /// Returns a [`SolverError`] for model construction, callback, or Gurobi
    /// optimization failures. On failure, the resident state is cleared.
    pub fn solve_with_callback<F>(
        &mut self,
        model: &Model,
        opts: &GurobiOptions,
        callback: &mut F,
    ) -> Result<SolverResult, SolverError>
    where
        F: Callback,
    {
        match self.solve_resident_with_callback(model, opts, callback, None) {
            Ok(result) => {
                self.preparation.commit();
                Ok(result)
            }
            Err(error) => {
                self.state = None;
                self.preparation.clear();
                Err(error)
            }
        }
    }

    /// Solve the resident model with only the selected callback locations.
    ///
    /// # Errors
    /// Returns a [`SolverError`] for model construction, callback, or Gurobi
    /// optimization failures. On failure, the resident state is cleared.
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
        match self.solve_resident_with_callback(model, opts, callback, Some(locations.into())) {
            Ok(result) => {
                self.preparation.commit();
                Ok(result)
            }
            Err(error) => {
                self.state = None;
                self.preparation.clear();
                Err(error)
            }
        }
    }
}

impl Solver for GurobiPersistent {
    type Options = GurobiOptions;

    fn name(&self) -> &str {
        crate::NAME
    }

    fn supports(&self, kind: ModelKind) -> bool {
        crate::supported(kind)
    }

    fn domain_support(&self, _domain: oximo_core::Domain) -> oximo_solver::Capability {
        oximo_solver::Capability::Native
    }

    fn supports_sos(&self, _sos_type: SosType) -> bool {
        true
    }

    fn supports_indicators(&self) -> bool {
        true
    }

    fn solve(&mut self, model: &Model, opts: &GurobiOptions) -> Result<SolverResult, SolverError> {
        // A mid-update failure (a failed rebuild, a partial delta push, or a solve
        // error) can leave the resident model partially modified or its snapshot
        // stale. Drop the resident state on any error so the next solve rebuilds from
        // a clean slate, honoring the documented contract.
        match self.solve_resident(model, opts) {
            Ok(result) => {
                self.preparation.commit();
                Ok(result)
            }
            Err(e) => {
                self.state = None;
                self.preparation.clear();
                Err(e)
            }
        }
    }
}

/// Push the objective-coefficient, bound, and objective-constant deltas between the
/// resident `base` snapshot and the freshly read `snap` onto the live Gurobi model,
/// then re-apply options.
fn push_deltas(
    built: &mut Built,
    base: &Snapshot,
    snap: &Snapshot,
    opts: &GurobiOptions,
) -> Result<(), SolverError> {
    for i in 0..built.vars.len() {
        if snap.obj_costs[i].to_bits() != base.obj_costs[i].to_bits() {
            built
                .model
                .set_obj_attr(attr::Obj, &built.vars[i], snap.obj_costs[i])
                .map_err(map_gurobi_err)?;
        }
        if snap.lb[i].to_bits() != base.lb[i].to_bits() {
            built
                .model
                .set_obj_attr(attr::LB, &built.vars[i], snap.lb[i])
                .map_err(map_gurobi_err)?;
        }
        if snap.ub[i].to_bits() != base.ub[i].to_bits() {
            built
                .model
                .set_obj_attr(attr::UB, &built.vars[i], snap.ub[i])
                .map_err(map_gurobi_err)?;
        }
    }
    if snap.obj_constant.to_bits() != base.obj_constant.to_bits() {
        built.model.set_attr(attr::ObjCon, snap.obj_constant).map_err(map_gurobi_err)?;
    }
    apply_options(&mut built.model, opts).map_err(map_gurobi_err)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use oximo_core::prelude::*;
    use oximo_solver::{PersistentSolver, Solver};

    use crate::{Gurobi, GurobiOptions};

    #[test]
    fn switching_binary_to_integer_model_rebuilds_native_domain() {
        let first = Model::new("domain");

        let x = first.__var("x").domain(Domain::Binary).lb(0.0).ub(1.0).build();
        first.__maximize(x);
        let second = Model::new("domain");
        let y = second.__var("x").domain(Domain::Integer).lb(0.0).ub(2.0).build();
        second.__maximize(y);
        let mut solver = Gurobi.persistent();
        let opts = GurobiOptions::default();
        assert_eq!(solver.solve(&first, &opts).unwrap().objective(), Some(1.0));
        assert_eq!(solver.solve(&second, &opts).unwrap().objective(), Some(2.0));
        assert_eq!(solver.updates, 0);
    }

    #[test]
    fn replacing_nonlinear_and_range_groups_does_not_leak_auxiliaries() {
        use gurobi_rs::attr;
        let m = Model::new("owned groups");

        param!(m, p = 1.0);
        variable!(m,0.1<=x<=2.0);
        constraint!(m, nonlinear, p * x.exp() <= 5.0);
        constraint!(m,range,0.2<=p*x<=3.0);
        objective!(m, Min, (p * x).exp());

        let opts = GurobiOptions::default();
        let mut solver = Gurobi.persistent();
        let mut counts = None;
        for value in [1.0, 1.2, 1.4, 1.1, 1.3, 1.0] {
            p.set_param_value(value).unwrap();
            let result = solver.solve(&m, &opts).unwrap();
            let cold = Gurobi.solve(&m, &opts).unwrap();
            assert!((result.objective().unwrap() - cold.objective().unwrap()).abs() < 1e-5);
            let native = &solver.state.as_ref().unwrap().built.model;
            let current = (
                native.get_attr(attr::NumVars).unwrap(),
                native.get_attr(attr::NumConstrs).unwrap(),
                native.get_attr(attr::NumGenConstrs).unwrap(),
            );
            assert_eq!(*counts.get_or_insert(current), current, "native groups must stay bounded");
        }
        assert_eq!(solver.updates, 5);
    }

    #[test]
    fn persistent_parameter_snapshots_refresh_on_native_reuse() {
        let model = Model::new("resident parameter snapshots");
        param!(model, p = 2.0);
        param!(model, unused = 7.0);
        variable!(model, 0.0 <= x <= 3.0);
        objective!(model, Max, p * x);
        let opts = GurobiOptions::default();
        let mut solver = crate::Gurobi.persistent();
        let first = solver.solve(&model, &opts).unwrap();
        assert!(first.has_solution());
        assert_eq!(solver.updates, 0);

        model.set_param(unused, 9.0).unwrap();
        let second = solver.solve(&model, &opts).unwrap();
        assert!(second.has_solution());
        assert_eq!(solver.updates, 1, "unused-only change must reuse the native problem");
        assert!((second.objective().unwrap() - first.objective().unwrap()).abs() < 1e-5);
        assert_eq!(second.value_of(unused), Ok(Some(9.0)));
        assert_eq!(first.value_of(unused), Ok(Some(7.0)));

        model.set_param(p, 4.0).unwrap();
        let third = solver.solve(&model, &opts).unwrap();
        assert!(third.has_solution());
        assert_eq!(solver.updates, 2, "objective coefficient change must update in place");
        let x_value = third.value_of(x).unwrap().unwrap();
        assert_eq!(third.value_of(p * x), Ok(Some(4.0 * x_value)));
        assert_eq!(first.value_of(p), Ok(Some(2.0)));

        param!(model, added = 11.0);
        let fourth = solver.solve(&model, &opts).unwrap();
        assert!(fourth.has_solution());
        assert_eq!(solver.updates, 3, "adding an unused parameter must reuse the native problem");
        assert_eq!(fourth.value_of(added), Ok(Some(11.0)));
        assert_eq!(third.value_of(added), Ok(None));
        for point in &fourth.solutions {
            assert_eq!(point.parameters.values().as_ptr(), fourth.parameters.values().as_ptr());
        }
    }

    #[test]
    fn fast_path_retags_results_for_a_distinct_structurally_identical_model() {
        let first = Model::new("first");
        variable!(first, 0.0 <= x1 <= 1.0);
        objective!(first, Max, x1);

        let second = Model::new("second");
        variable!(second, 0.0 <= x2 <= 1.0);
        objective!(second, Max, x2);
        assert_ne!(first.id(), second.id());

        let mut solver = Gurobi.persistent();
        solver.solve(&first, &GurobiOptions::default()).unwrap();
        let result = solver.solve(&second, &GurobiOptions::default()).unwrap();

        assert_eq!(result.model_id(), second.id());
        assert_eq!(result.value_of(x2).unwrap(), Some(1.0));
        assert_eq!(result.value_of(x1), Err(ModelMismatchError::new(second.id(), first.id())));
    }
}
