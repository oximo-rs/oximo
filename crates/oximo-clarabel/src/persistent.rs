use std::time::Instant;

use clarabel::solver::{DefaultSolver, IPSolver};
use oximo_core::{Model, ModelKind};
use oximo_solver::{Solver, SolverError, SolverResult};

use crate::translate::{Problem, build_problem, build_settings, read_result};
use crate::{ClarabelOptions, NAME};

/// The resident Clarabel solver plus the [`Problem`] it was built from. The
/// stored problem carries the sparsity pattern / cone layout used to decide
/// whether the next solve can update in place, and the [`crate::translate::Meta`]
/// used to read results back.
struct State {
    solver: DefaultSolver<f64>,
    problem: Problem,
}

/// A stateful Clarabel handle that keeps the built solver resident across
/// solves.
///
/// Created by [`Clarabel::persistent`](crate::Clarabel). When the next model
/// has the same dimensions, cone layout and `P`/`A` sparsity pattern, only the
/// numeric data (`P`, `q`, `A`, `b`) is overwritten in place via Clarabel's
/// [`update_data`](clarabel::solver::DefaultSolver::update_data), reusing the
/// KKT symbolic factorization instead of rebuilding it. Any structural change
/// — added/removed rows or columns, a changed sparsity pattern, a new cone, a
/// flipped constraint sense that moves a row between cones — triggers a
/// transparent full rebuild, so every result matches a cold
/// [`Clarabel::solve`](crate::Clarabel).
///
/// Clarabel is an interior-point method and does not warm-start from the
/// previous iterate, so iteration counts are unchanged; the saving is the
/// skipped setup (equilibration structure, symbolic factorization, allocations).
///
/// In-place updates are rejected by Clarabel when presolve or structural-zero
/// dropping altered the problem, in which case the handle falls back to a
/// rebuild. Set `presolve_enable(false)` and `input_sparse_dropzeros(false)`
/// (the defaults leave presolve on) to keep the fast path available.
///
/// A failed `solve` leaves the handle without a resident model; the next call
/// rebuilds from scratch.
#[derive(Default)]
pub struct ClarabelPersistent {
    state: Option<State>,
}

impl std::fmt::Debug for ClarabelPersistent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClarabelPersistent").field("resident", &self.state.is_some()).finish()
    }
}

impl ClarabelPersistent {
    /// A fresh handle with no model loaded. The first solve builds it.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Drop the resident solver so the next [`solve`](Solver::solve) rebuilds
    /// from scratch. After this, the next solve behaves exactly like a cold
    /// [`Clarabel::solve`](crate::Clarabel).
    pub fn reset(&mut self) {
        self.state = None;
    }

    /// Translate the model, update the resident solver in place when the
    /// structure is unchanged (else rebuild), then solve.
    fn solve_resident(
        &mut self,
        model: &Model,
        opts: &ClarabelOptions,
    ) -> Result<SolverResult, SolverError> {
        let new = build_problem(model)?;
        let settings = build_settings(opts);

        // Fast path: same structure, and Clarabel accepts both the settings
        // update (no immutable setting changed) and the data update (structure
        // not altered by presolve / dropped zeros). Any failure means rebuild.
        let mut updated = false;
        if let Some(state) = self.state.as_mut() {
            if state.problem.same_structure(&new)
                && state.solver.update_settings(settings.clone()).is_ok()
                && state.solver.update_data(&new.p_mat, &new.q, &new.a_mat, &new.b).is_ok()
            {
                updated = true;
            }
        }

        if updated {
            // Refresh the readback metadata (objective sign/constant, dual map).
            self.state.as_mut().expect("resident on fast path").problem = new;
        } else {
            let solver =
                DefaultSolver::new(&new.p_mat, &new.q, &new.a_mat, &new.b, &new.cones, settings)
                    .map_err(|e| SolverError::Backend(format!("Clarabel setup: {e:?}")))?;
            self.state = Some(State { solver, problem: new });
        }

        let state = self.state.as_mut().expect("state present before solve");
        let started = Instant::now();
        state.solver.solve();
        let elapsed = started.elapsed();
        Ok(read_result(&state.solver, &state.problem.meta, elapsed))
    }
}

impl Solver for ClarabelPersistent {
    type Options = ClarabelOptions;

    fn name(&self) -> &str {
        NAME
    }

    fn supports(&self, kind: ModelKind) -> bool {
        crate::supported(kind)
    }

    fn solve(
        &mut self,
        model: &Model,
        opts: &ClarabelOptions,
    ) -> Result<SolverResult, SolverError> {
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
    use oximo_solver::{PersistentSolver, Solver, SolverError, TerminationStatus};

    use crate::{Clarabel, ClarabelOptions};

    /// Relative closeness: two independent interior-point solves stop within
    /// the solver tolerance, not bit-for-bit, so scale by magnitude.
    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-5 * a.abs().max(b.abs()).max(1.0)
    }

    /// A parameter in an objective coefficient changes `q` only: the sparsity
    /// pattern is untouched, so the handle updates in place and must match the
    /// cold solve across the sweep.
    #[test]
    fn persistent_matches_cold_on_objective_sweep() {
        let m = Model::new("pricing");
        param!(m, p1 = 0.0);
        variable!(m, x1 >= 0.0);
        variable!(m, x2 >= 0.0);
        constraint!(m, labor, 2.0 * x1 + x2 <= 100.0);
        constraint!(m, material, x1 + 3.0 * x2 <= 90.0);
        objective!(m, Max, p1 * x1 + 5.0 * x2);

        let mut solver = Clarabel.persistent();
        for price in [1.0, 1.6, 2.0, 5.0, 11.0] {
            p1.set_param_value(price);
            let s = solver.solve(&m, &ClarabelOptions::default()).unwrap();
            let c = Clarabel.solve(&m, &ClarabelOptions::default()).unwrap();
            assert_eq!(s.termination, TerminationStatus::Optimal, "price {price}");
            assert!(close(s.objective().unwrap(), c.objective().unwrap()), "price {price}");
            assert!(close(s.value_of(x1).unwrap(), c.value_of(x1).unwrap()), "price {price}");
        }
    }

    /// A parameter in a right-hand side changes `b` only. Unlike the HiGHS
    /// handle (which rebuilds on rhs changes), Clarabel can push this through
    /// the in-place update; results must still match the cold solve.
    #[test]
    fn persistent_matches_cold_on_rhs_sweep() {
        let m = Model::new("capacity");
        param!(m, cap = 100.0);
        variable!(m, x1 >= 0.0);
        variable!(m, x2 >= 0.0);
        constraint!(m, labor, 2.0 * x1 + x2 <= cap);
        constraint!(m, material, x1 + 3.0 * x2 <= 90.0);
        objective!(m, Max, 3.0 * x1 + 5.0 * x2);

        let mut solver = Clarabel.persistent();
        for c in [100.0, 60.0, 140.0] {
            cap.set_param_value(c);
            let s = solver.solve(&m, &ClarabelOptions::default()).unwrap();
            let cold = Clarabel.solve(&m, &ClarabelOptions::default()).unwrap();
            assert_eq!(s.termination, TerminationStatus::Optimal, "cap {c}");
            assert!(close(s.objective().unwrap(), cold.objective().unwrap()), "cap {c}");
        }
    }

    /// A parameter in a matrix coefficient changes an `A` value, keeping the
    /// sparsity pattern: still the in-place path.
    #[test]
    fn persistent_matches_cold_on_matrix_coeff_sweep() {
        let m = Model::new("coeff");
        param!(m, a = 2.0);
        variable!(m, x1 >= 0.0);
        variable!(m, x2 >= 0.0);
        constraint!(m, labor, a * x1 + x2 <= 100.0);
        objective!(m, Max, 3.0 * x1 + 5.0 * x2);

        let mut solver = Clarabel.persistent();
        for av in [2.0, 1.0, 4.0] {
            a.set_param_value(av);
            let s = solver.solve(&m, &ClarabelOptions::default()).unwrap();
            let cold = Clarabel.solve(&m, &ClarabelOptions::default()).unwrap();
            assert_eq!(s.termination, TerminationStatus::Optimal, "a {av}");
            assert!(close(s.objective().unwrap(), cold.objective().unwrap()), "a {av}");
        }
    }

    /// Fixing a variable moves it from bound rows to a zero-cone row: a
    /// structural change the handle must absorb via a rebuild, still matching
    /// the cold solve.
    #[test]
    fn persistent_rebuilds_on_structural_change() {
        let m = Model::new("feas");
        variable!(m, 0.0 <= x <= 10.0);
        variable!(m, 0.0 <= y <= 10.0);
        constraint!(m, c, x + y == 5.0);
        objective!(m, Min, x + 2.0 * y);

        let mut solver = Clarabel.persistent();
        let r = solver.solve(&m, &ClarabelOptions::default()).unwrap();
        assert!(r.has_solution(), "termination = {:?}", r.termination);

        m.fix(x, 2.0);
        let r2 = solver.solve(&m, &ClarabelOptions::default()).unwrap();
        let cold = Clarabel.solve(&m, &ClarabelOptions::default()).unwrap();
        assert!(close(r2.value_of(x).unwrap(), 2.0));
        assert!(close(r2.value_of(y).unwrap(), 3.0));
        assert!(close(r2.objective().unwrap(), cold.objective().unwrap()));
    }

    /// An SOCP with a swept objective coefficient exercises the fast path with
    /// a second-order cone in the layout.
    #[test]
    fn persistent_socp_objective_sweep() {
        let m = Model::new("socp");
        param!(m, wt = 1.0);
        variable!(m, x);
        variable!(m, y);
        variable!(m, t >= 0.0);
        m.fix(t, 1.0);
        m.add_soc_constraint("disk", [x, y], t); // ||(x, y)|| <= 1
        objective!(m, Min, wt * x + y);

        let mut solver = Clarabel.persistent();
        for wv in [1.0, 2.0, 0.5] {
            wt.set_param_value(wv);
            let warm = solver.solve(&m, &ClarabelOptions::default()).unwrap();
            let cold = Clarabel.solve(&m, &ClarabelOptions::default()).unwrap();
            assert_eq!(warm.termination, TerminationStatus::Optimal, "wt {wv}");
            assert!(close(warm.objective().unwrap(), cold.objective().unwrap()), "wt {wv}");
        }
    }

    #[test]
    fn persistent_reset_then_solve_ok() {
        let m = Model::new("pricing");
        param!(m, p1 = 0.0);
        variable!(m, x1 >= 0.0);
        variable!(m, x2 >= 0.0);
        constraint!(m, labor, 2.0 * x1 + x2 <= 100.0);
        objective!(m, Max, p1 * x1 + 5.0 * x2);

        let mut solver = Clarabel.persistent();
        p1.set_param_value(2.0);
        let first = solver.solve(&m, &ClarabelOptions::default()).unwrap();
        assert_eq!(first.termination, TerminationStatus::Optimal);

        solver.reset();
        let after = solver.solve(&m, &ClarabelOptions::default()).unwrap();
        assert_eq!(after.termination, TerminationStatus::Optimal);
        assert!(close(first.objective().unwrap(), after.objective().unwrap()));
    }

    #[test]
    fn persistent_unsupported_kind_errors_and_clears() {
        let m = Model::new("milp");
        variable!(m, 0.0 <= x <= 5.0, Int);
        objective!(m, Min, x);
        let mut solver = Clarabel.persistent();
        let err = solver.solve(&m, &ClarabelOptions::default()).unwrap_err();
        assert!(matches!(err, SolverError::UnsupportedKind(ModelKind::MILP)));
    }
}
