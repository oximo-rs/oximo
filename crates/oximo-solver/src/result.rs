use std::time::Duration;

use oximo_core::{ConstraintId, Expr, ExprNode, IndexKey, IndexedVar, VarId};
use rustc_hash::FxHashMap;

use crate::status::SolverStatus;

/// A single primal point returned by a solver.
///
/// Most solves yield one point, but a global solver asked to enumerate solutions
/// may returns several. In a [`SolverResult`] the points live in [`SolverResult::solutions`].
/// Index `0` is always the best/incumbent.
#[derive(Clone, Debug, Default)]
pub struct SolutionPoint {
    pub primal: FxHashMap<VarId, f64>,
    pub objective: Option<f64>,
}

impl SolutionPoint {
    /// Look up a primal value by `VarId`.
    pub fn value(&self, id: VarId) -> Option<f64> {
        self.primal.get(&id).copied()
    }

    /// Look up the primal value for an `Expr` that points at a `Var` node.
    /// Returns `None` for any expression that is not a single variable, so
    /// callers that need the value of a derived expression should evaluate it
    /// against the primal solution explicitly.
    pub fn value_of(&self, expr: Expr<'_>) -> Option<f64> {
        let arena = expr.arena.borrow();
        match arena.get(expr.id) {
            ExprNode::Var(v) => self.primal.get(v).copied(),
            _ => None,
        }
    }

    /// Look up the primal value for a specific index of an [`IndexedVar`].
    ///
    /// Returns `None` if `key` is not in the variable's set or the solver did
    /// not return a primal value for that scalar.
    pub fn value_of_idx<K: Into<IndexKey>>(&self, var: &IndexedVar<'_>, key: K) -> Option<f64> {
        var.get(key).and_then(|e| self.value_of(e))
    }

    /// Iterate over primal values for all entries of an [`IndexedVar`].
    ///
    /// Yields `(&IndexKey, f64)` for every index whose primal value is present
    /// in the solution.
    pub fn values_of<'iv, 'a>(
        &'iv self,
        var: &'iv IndexedVar<'a>,
    ) -> impl Iterator<Item = (&'iv IndexKey, f64)> + 'iv {
        var.iter().filter_map(|(k, e)| self.value_of(*e).map(|v| (k, v)))
    }
}

/// A solver's final result on a model.
///
/// Primal points are held in `solutions` (index `0` is the best/incumbent,
/// empty when no solution was found). `dual` and `reduced_costs` apply to the
/// best continuous point and are sparse maps, so a solver that does not return
/// duals (e.g. MILP) can simply leave them empty.
#[derive(Clone, Debug)]
pub struct SolverResult {
    pub status: SolverStatus,
    pub solutions: Vec<SolutionPoint>,
    pub dual: FxHashMap<ConstraintId, f64>,
    pub reduced_costs: FxHashMap<VarId, f64>,
    pub solve_time: Duration,
    pub iterations: u64,
    pub raw_log: Option<String>,
}

impl Default for SolverResult {
    fn default() -> Self {
        Self {
            status: SolverStatus::NotSolved,
            solutions: Vec::new(),
            dual: FxHashMap::default(),
            reduced_costs: FxHashMap::default(),
            solve_time: Duration::ZERO,
            iterations: 0,
            raw_log: None,
        }
    }
}

impl SolverResult {
    /// The number of primal points the solver returned (`0` when infeasible or
    /// unsolved).
    pub fn result_count(&self) -> usize {
        self.solutions.len()
    }

    /// The `i`-th primal point, where index `0` is the best/incumbent.
    pub fn solution(&self, i: usize) -> Option<&SolutionPoint> {
        self.solutions.get(i)
    }

    /// The best primal point, or `None` when no solution was found.
    pub fn best(&self) -> Option<&SolutionPoint> {
        self.solutions.first()
    }

    /// The objective value of the best solution, or `None` when none was found.
    pub fn objective(&self) -> Option<f64> {
        self.solutions.first().and_then(|s| s.objective)
    }

    /// The best solution's primal map, or `None` when no solution was found.
    pub fn primal(&self) -> Option<&FxHashMap<VarId, f64>> {
        self.solutions.first().map(|s| &s.primal)
    }

    /// Look up a primal value by `VarId` in the best solution.
    pub fn value(&self, id: VarId) -> Option<f64> {
        self.solutions.first().and_then(|s| s.value(id))
    }

    /// Look up the best solution's primal value for an `Expr` that points at a
    /// `Var` node. Returns `None` for any expression that is not a single
    /// variable.
    pub fn value_of(&self, expr: Expr<'_>) -> Option<f64> {
        self.solutions.first().and_then(|s| s.value_of(expr))
    }

    pub fn dual_of(&self, c: ConstraintId) -> Option<f64> {
        self.dual.get(&c).copied()
    }

    /// Look up the best solution's primal value for a specific index of an
    /// [`IndexedVar`].
    pub fn value_of_idx<K: Into<IndexKey>>(&self, var: &IndexedVar<'_>, key: K) -> Option<f64> {
        var.get(key).and_then(|e| self.value_of(e))
    }

    /// Iterate over the best solution's primal values for all entries of an
    /// [`IndexedVar`]. Yields nothing when no solution was found.
    pub fn values_of<'iv, 'a>(
        &'iv self,
        var: &'iv IndexedVar<'a>,
    ) -> impl Iterator<Item = (&'iv IndexKey, f64)> + 'iv {
        var.iter().filter_map(|(k, e)| self.value_of(*e).map(|v| (k, v)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_result_has_no_solution() {
        let r = SolverResult::default();
        assert_eq!(r.result_count(), 0);
        assert!(r.best().is_none());
        assert!(r.objective().is_none());
        assert!(r.primal().is_none());
        assert!(r.value(VarId(0)).is_none());
        assert!(r.solution(0).is_none());
    }

    #[test]
    fn best_is_solution_zero() {
        let mut p0 = FxHashMap::default();
        p0.insert(VarId(0), 1.5);
        let mut p1 = FxHashMap::default();
        p1.insert(VarId(0), 2.5);
        let r = SolverResult {
            status: SolverStatus::Optimal,
            solutions: vec![
                SolutionPoint { primal: p0, objective: Some(10.0) },
                SolutionPoint { primal: p1, objective: Some(9.0) },
            ],
            ..Default::default()
        };
        assert_eq!(r.result_count(), 2);
        assert_eq!(r.objective(), Some(10.0));
        assert_eq!(r.value(VarId(0)), Some(1.5));
        assert_eq!(r.solution(1).unwrap().value(VarId(0)), Some(2.5));
    }
}
