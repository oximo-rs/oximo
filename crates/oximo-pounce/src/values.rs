//! Value-only oracle for the finite-difference (stable) path: compiles each
//! objective/constraint expression to a [`Tape`] once, so POUNCE's builder can
//! evaluate them at any point without borrowing the model. Derivatives are left
//! to POUNCE's finite differences.

use std::cell::RefCell;

use oximo_autodiff::tape::{Tape, params_snapshot};
use oximo_core::Model;
use oximo_expr::ExprId;

/// Objective and constraint values, evaluated from compiled tapes.
pub(crate) struct ValueOracle {
    obj: Option<Tape>,
    cons: Vec<Tape>,
    params: Vec<f64>,
    scratch: RefCell<Vec<f64>>,
    n_vars: usize,
    obj_expr: Option<ExprId>,
    con_exprs: Vec<ExprId>,
}

impl ValueOracle {
    pub(crate) fn new(model: &Model) -> Self {
        let arena = model.arena();
        let obj_expr = model.objective().as_ref().map(|o| o.expr);
        let con_exprs: Vec<ExprId> = model.constraints().iter().map(|c| c.lhs).collect();
        let obj = obj_expr.map(|e| Tape::compile(&arena, e));
        let cons: Vec<Tape> = con_exprs.iter().map(|&e| Tape::compile(&arena, e)).collect();
        let params = params_snapshot(&arena);
        let max_regs = obj.iter().chain(cons.iter()).map(Tape::n_regs).max().unwrap_or(0);
        Self {
            obj,
            cons,
            params,
            scratch: RefCell::new(vec![0.0; max_regs]),
            n_vars: model.variables().len(),
            obj_expr,
            con_exprs,
        }
    }

    pub(crate) fn num_constraints(&self) -> usize {
        self.cons.len()
    }

    /// Whether `model` has the same variables and the same objective/constraint
    /// expressions this oracle was built from. The tapes are then still valid
    /// and only a [`Self::refresh_params`] is needed.
    pub(crate) fn matches(&self, model: &Model) -> bool {
        self.n_vars == model.variables().len()
            && self.obj_expr == model.objective().as_ref().map(|o| o.expr)
            && self.con_exprs.len() == model.constraints().len()
            && model.constraints().iter().map(|c| c.lhs).eq(self.con_exprs.iter().copied())
    }

    /// Re-snapshot parameter values after `set_param`. The tapes are
    /// parameter-independent (parameters are `OP_PARAM` references resolved at
    /// evaluation), so only the value snapshot needs refreshing.
    pub(crate) fn refresh_params(&mut self, model: &Model) {
        self.params = params_snapshot(&model.arena());
    }

    /// Objective value at `x` (0 for a feasibility problem).
    pub(crate) fn objective(&self, x: &[f64]) -> f64 {
        match &self.obj {
            Some(tape) => tape.value(x, &self.params, &[], &mut self.scratch.borrow_mut()),
            None => 0.0,
        }
    }

    /// Value of constraint `i` at `x`.
    pub(crate) fn constraint(&self, i: usize, x: &[f64]) -> f64 {
        self.cons[i].value(x, &self.params, &[], &mut self.scratch.borrow_mut())
    }
}
