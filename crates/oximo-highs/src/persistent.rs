use std::time::Instant;

use highs::Model as HighsModel;
use oximo_core::{Model, ModelKind};
use oximo_solver::{Snapshot, Solver, SolverError, SolverResult, snapshot};

use crate::options::apply as apply_options;
use crate::translate::{Meta, build_problem, extract_result, make_live};
use crate::{HighsOptions, NAME};

/// The resident HiGHS instance plus the [`Meta`] needed to read results and to drive
/// incremental re-solves. `live` is taken on solve and put back (as a fresh
/// [`HighsModel`] over the same instance) so the basis carries to the next solve.
struct State {
    live: Option<HighsModel>,
    meta: Meta,
    snap: Option<Snapshot>,
}

/// A stateful HiGHS solver handle that keeps the built model resident across solves.
///
/// Created by [`Highs::persistent`](crate::Highs).
/// When only objective coefficients or variable bounds changed since the last call
/// it pushes those deltas and HiGHS warm-starts from the retained basis.
/// Any structural change (new rows/columns, changed matrix coefficients or row bounds,
/// flipped integrality or sense, or a quadratic objective) triggers a transparent
/// full rebuild, so every result matches a cold [`Highs::solve`](crate::Highs).
///
/// Options passed to each `solve` are applied to the resident instance and persist
/// across calls. Changing a field takes effect on the next solve, but a field you
/// stop setting keeps its previous value rather than reverting to the HiGHS default.
/// Call [`reset`](Self::reset) (or build a fresh handle) when you want the next
/// solve to behave exactly like a cold [`Highs::solve`](crate::Highs),
/// options and all.
///
/// A failed `solve` (HiGHS returning an error) leaves the handle without a resident
/// model, the next call rebuilds from scratch.
#[derive(Default)]
pub struct HighsPersistent {
    state: Option<State>,
}

impl std::fmt::Debug for HighsPersistent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HighsPersistent").field("resident", &self.state.is_some()).finish()
    }
}

impl HighsPersistent {
    /// A fresh handle with no model loaded. The first solve builds it.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Drop the resident model so the next [`solve`](Solver::solve) rebuilds from
    /// scratch, clearing the warm-start basis and any solver options carried on the
    /// HiGHS instance. After this, the next solve behaves exactly like a cold
    /// [`Highs::solve`](crate::Highs), regardless of earlier calls.
    pub fn reset(&mut self) {
        self.state = None;
    }

    /// Discard any resident instance and rebuild from the current model state.
    fn rebuild(&mut self, model: &Model, opts: &HighsOptions) -> Result<(), SolverError> {
        let (prob, meta) = build_problem(model)?;
        let live = make_live(prob, opts)?;
        let snap = match model.kind() {
            ModelKind::LP | ModelKind::MILP => Some(snapshot(model)?),
            _ => None,
        };
        self.state = Some(State { live: Some(live), meta, snap });
        Ok(())
    }

    /// Re-read the model, update the resident instance in place (or rebuild), and
    /// solve.
    fn solve_resident(
        &mut self,
        model: &Model,
        opts: &HighsOptions,
    ) -> Result<SolverResult, SolverError> {
        // The fast path is sound only for linear models (the snapshot extracts a
        // linear objective) once a resident model exists. A quadratic objective or
        // any structural change falls back to a full rebuild.
        let mut updated = false;
        if matches!(model.kind(), ModelKind::LP | ModelKind::MILP) {
            if let Some(st) = self.state.as_mut() {
                if let Some(base) = st.snap.as_ref() {
                    let snap = snapshot(model)?;
                    if snap.fingerprint == base.fingerprint {
                        let live = st.live.as_mut().expect("live model present on fast path");
                        for i in 0..st.meta.cols.len() {
                            if snap.obj_costs[i].to_bits() != base.obj_costs[i].to_bits() {
                                live.change_column_cost(st.meta.cols[i], snap.obj_costs[i]);
                            }
                            if snap.lb[i].to_bits() != base.lb[i].to_bits()
                                || snap.ub[i].to_bits() != base.ub[i].to_bits()
                            {
                                live.change_column_bounds(st.meta.cols[i], snap.lb[i]..=snap.ub[i]);
                            }
                        }
                        apply_options(live, opts)?;
                        st.meta.obj_constant = snap.obj_constant;
                        st.snap = Some(snap);
                        updated = true;
                    }
                }
            }
        }
        if !updated {
            self.rebuild(model, opts)?;
        }

        // Solve the resident model, then move it back so the basis is retained for
        // the next solve.
        let st = self.state.as_mut().expect("state present before solve");
        let live = st.live.take().expect("live model present before solve");
        let started = Instant::now();
        let solved = live
            .try_solve()
            .map_err(|e| SolverError::Backend(format!("HiGHS solve failed: {e:?}")))?;
        let elapsed = started.elapsed();
        let result =
            extract_result(&solved, st.meta.obj_constant, st.meta.num_constraints, elapsed);
        st.live = Some(HighsModel::from(solved));
        Ok(result)
    }
}

impl Solver for HighsPersistent {
    type Options = HighsOptions;

    fn name(&self) -> &str {
        NAME
    }

    fn supports(&self, kind: ModelKind) -> bool {
        crate::supported(kind)
    }

    fn solve(&mut self, model: &Model, opts: &HighsOptions) -> Result<SolverResult, SolverError> {
        match self.solve_resident(model, opts) {
            Ok(result) => Ok(result),
            Err(e) => {
                self.state = None;
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use oximo_core::prelude::*;
    use oximo_solver::{PersistentSolver, Solver, TerminationStatus};

    use crate::{Highs, HighsOptions};

    #[test]
    fn persistent_matches_cold_solve_on_objective_sweep() {
        let m = Model::new("pricing");
        param!(m, p1 = 0.0);
        variable!(m, x1 >= 0.0);
        variable!(m, x2 >= 0.0);
        constraint!(m, labor, 2.0 * x1 + x2 <= 100.0);
        constraint!(m, material, x1 + 3.0 * x2 <= 90.0);
        objective!(m, Max, p1 * x1 + 5.0 * x2);

        let mut solver = Highs.persistent();
        for price in [1.0, 1.6, 2.0, 5.0, 11.0] {
            p1.set_param_value(price);
            let s = solver.solve(&m, &HighsOptions::default()).unwrap();
            let c = Highs.solve(&m, &HighsOptions::default()).unwrap();
            assert_eq!(s.termination, TerminationStatus::Optimal, "price {price}");
            assert!(
                (s.objective().unwrap() - c.objective().unwrap()).abs() < 1e-9,
                "price {price}"
            );
            assert!((s.value_of(x1).unwrap() - c.value_of(x1).unwrap()).abs() < 1e-9);
            assert!((s.value_of(x2).unwrap() - c.value_of(x2).unwrap()).abs() < 1e-9);
        }
    }

    /// A parameter in a constraint right-hand side changes the row bound, which the
    /// fast path cannot push: the handle must fall back to a rebuild and still match
    /// the cold solve.
    #[test]
    fn persistent_rebuilds_on_constraint_rhs_change() {
        let m = Model::new("capacity");
        param!(m, cap = 100.0);
        variable!(m, x1 >= 0.0);
        variable!(m, x2 >= 0.0);
        constraint!(m, labor, 2.0 * x1 + x2 <= cap);
        constraint!(m, material, x1 + 3.0 * x2 <= 90.0);
        objective!(m, Max, 3.0 * x1 + 5.0 * x2);

        let mut solver = Highs.persistent();
        for c in [100.0, 60.0, 140.0] {
            cap.set_param_value(c);
            let s = solver.solve(&m, &HighsOptions::default()).unwrap();
            let cold = Highs.solve(&m, &HighsOptions::default()).unwrap();
            assert_eq!(s.termination, TerminationStatus::Optimal, "cap {c}");
            assert!((s.objective().unwrap() - cold.objective().unwrap()).abs() < 1e-9, "cap {c}");
        }
    }

    /// A feasibility model (no objective) solves through the persistent handle as
    /// `minimize 0`, both cold and on the warm fast path.
    #[test]
    fn persistent_feasibility_no_objective() {
        let m = Model::new("feas");
        variable!(m, 0.0 <= x <= 10.0);
        variable!(m, 0.0 <= y <= 10.0);
        constraint!(m, c, x + y == 5.0);
        objective!(m, Feasibility);

        let mut solver = Highs.persistent();
        let r = solver.solve(&m, &HighsOptions::default()).unwrap();
        assert!(r.has_solution(), "termination = {:?}", r.termination);
        m.fix(x, 2.0);
        let r2 = solver.solve(&m, &HighsOptions::default()).unwrap();
        assert!(r2.has_solution());
        assert!((r2.value_of(x).unwrap() - 2.0).abs() < 1e-9);
        assert!((r2.value_of(y).unwrap() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn persistent_reset_then_solve_ok() {
        let m = Model::new("pricing");
        param!(m, p1 = 0.0);
        variable!(m, x1 >= 0.0);
        variable!(m, x2 >= 0.0);
        constraint!(m, labor, 2.0 * x1 + x2 <= 100.0);
        objective!(m, Max, p1 * x1 + 5.0 * x2);

        let mut solver = Highs.persistent();
        p1.set_param_value(2.0);
        let first = solver.solve(&m, &HighsOptions::default()).unwrap();
        assert_eq!(first.termination, TerminationStatus::Optimal);

        solver.reset();
        let after = solver.solve(&m, &HighsOptions::default()).unwrap();
        assert_eq!(after.termination, TerminationStatus::Optimal);
        assert!((first.objective().unwrap() - after.objective().unwrap()).abs() < 1e-9);
    }

    #[test]
    fn persistent_warm_start_not_worse_than_cold() {
        let m = Model::new("mix");
        param!(m, bump = 0.0);
        variable!(m, p1 >= 0.0);
        variable!(m, p2 >= 0.0);
        variable!(m, p3 >= 0.0);
        variable!(m, p4 >= 0.0);
        variable!(m, p5 >= 0.0);
        constraint!(m, r1, 2.0 * p1 + p2 + 3.0 * p3 + p4 + p5 <= 100.0);
        constraint!(m, r2, p1 + 4.0 * p2 + p3 + 2.0 * p4 + p5 <= 120.0);
        constraint!(m, r3, 3.0 * p1 + p2 + 2.0 * p3 + p4 + 4.0 * p5 <= 150.0);
        constraint!(m, r4, p1 + p2 + p3 + p4 + p5 <= 80.0);
        objective!(m, Max, 10.0 * p1 + 8.0 * p2 + 7.0 * p3 + 6.0 * p4 + 9.0 * p5 + bump * p1);

        let mut solver = Highs.persistent();
        let mut persistent_iters = 0u64;
        let mut cold_iters = 0u64;
        for bump_val in [0.0, 1.0, 2.0, 3.0, 4.0] {
            bump.set_param_value(bump_val);
            let warm = solver.solve(&m, &HighsOptions::default()).unwrap();
            let cold = Highs.solve(&m, &HighsOptions::default()).unwrap();
            assert_eq!(warm.termination, TerminationStatus::Optimal, "bump {bump_val}");
            assert!(
                (warm.objective().unwrap() - cold.objective().unwrap()).abs() < 1e-9,
                "bump {bump_val}"
            );
            persistent_iters += warm.iterations;
            cold_iters += cold.iterations;
        }
        assert!(
            persistent_iters <= cold_iters,
            "warm-started handle used more iterations ({persistent_iters}) than cold ({cold_iters})"
        );
    }
}
