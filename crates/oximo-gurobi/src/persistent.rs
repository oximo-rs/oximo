use grb::prelude::*;
use oximo_core::{Model, ModelKind};
use oximo_solver::{Iis, Snapshot, Solver, SolverError, SolverResult, snapshot};

use crate::GurobiOptions;
use crate::options::apply as apply_options;
use crate::translate::{Built, build, compute_iis_resident, default_env, map_grb_err, run_and_collect};

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
/// Gurobi's attribute API (`Obj`, `LB`, `UB`, `ObjCon`) and Gurobi warm-starts. Any
/// structural change (new rows/columns, changed matrix coefficients or row bounds,
/// flipped integrality or sense, or a quadratic/nonlinear objective) triggers a
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
    fn rebuild(&mut self, model: &Model, opts: &GurobiOptions) -> Result<(), SolverError> {
        let env = match self.env.as_ref() {
            Some(env) => env,
            None => self.env.insert(default_env()?),
        };
        let built = build(model, opts, env)?;
        // The fast path needs a linear snapshot and is unsafe for SC/SI variables
        // (whose Gurobi lower bound is the gap floor, not `Variable::lb`), so
        // those rebuild every solve.
        let snap = if matches!(model.kind(), ModelKind::LP | ModelKind::MILP) && !built.has_semi {
            Some(snapshot(model)?)
        } else {
            None
        };
        self.state = Some(State { built, snap });
        Ok(())
    }

    /// Re-read the model, update the resident instance in place (or rebuild), and
    /// solve.
    fn solve_resident(
        &mut self,
        model: &Model,
        opts: &GurobiOptions,
    ) -> Result<SolverResult, SolverError> {
        let kind = model.kind();
        let mut updated = false;
        if matches!(kind, ModelKind::LP | ModelKind::MILP) {
            if let Some(st) = self.state.as_mut() {
                if let Some(base) = st.snap.as_ref() {
                    let snap = snapshot(model)?;
                    if snap.fingerprint == base.fingerprint {
                        push_deltas(&mut st.built, base, &snap, opts)?;
                        st.snap = Some(snap);
                        updated = true;
                    }
                }
            }
        }
        if !updated {
            self.rebuild(model, opts)?;
        }

        let st = self.state.as_mut().expect("state present before solve");
        run_and_collect(&mut st.built, kind)
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

    fn solve(&mut self, model: &Model, opts: &GurobiOptions) -> Result<SolverResult, SolverError> {
        // A mid-update failure (a failed rebuild, a partial delta push, or a solve
        // error) can leave the resident model partially modified or its snapshot
        // stale. Drop the resident state on any error so the next solve rebuilds from
        // a clean slate, honoring the documented contract.
        match self.solve_resident(model, opts) {
            Ok(result) => Ok(result),
            Err(e) => {
                self.state = None;
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
                .map_err(map_grb_err)?;
        }
        if snap.lb[i].to_bits() != base.lb[i].to_bits() {
            built.model.set_obj_attr(attr::LB, &built.vars[i], snap.lb[i]).map_err(map_grb_err)?;
        }
        if snap.ub[i].to_bits() != base.ub[i].to_bits() {
            built.model.set_obj_attr(attr::UB, &built.vars[i], snap.ub[i]).map_err(map_grb_err)?;
        }
    }
    if snap.obj_constant.to_bits() != base.obj_constant.to_bits() {
        built.model.set_attr(attr::ObjCon, snap.obj_constant).map_err(map_grb_err)?;
    }
    apply_options(&mut built.model, opts).map_err(map_grb_err)?;
    Ok(())
}
