use std::time::Duration;

use oximo_core::{ConstraintId, Expr, ExprNode, VarId};
use rustc_hash::FxHashMap;

use crate::status::SolverStatus;

/// A solver's final result on a model. `primal` and `dual` are sparse maps so a
/// solver that does not return duals (e.g. MILP) can simply leave them empty.
#[derive(Clone, Debug)]
pub struct SolverResult {
    pub status: SolverStatus,
    pub objective: Option<f64>,
    pub primal: FxHashMap<VarId, f64>,
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
            objective: None,
            primal: FxHashMap::default(),
            dual: FxHashMap::default(),
            reduced_costs: FxHashMap::default(),
            solve_time: Duration::ZERO,
            iterations: 0,
            raw_log: None,
        }
    }
}

impl SolverResult {
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

    pub fn dual_of(&self, c: ConstraintId) -> Option<f64> {
        self.dual.get(&c).copied()
    }
}
