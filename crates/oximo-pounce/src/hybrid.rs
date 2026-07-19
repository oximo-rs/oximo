//! Stable-Rust derivative oracle: exact analytic derivatives for
//! closed-form, POUNCE's finite differences for the rest.
//!
//! Every objective/constraint expression is classified once with
//! [`FunctionSlot::classify`]. When the whole model is linear/quadratic
//! (LP/QP/QCP) this oracle serves exact gradients, Jacobian rows, and the
//! constant Hessian of the Lagrangian through [`DerivativeOracle`], and the
//! solve runs on POUNCE's low-level `TNLP` surface.
//! A nonlinear function anywhere routes the solve to POUNCE's `builder`
//! surface instead (see [`crate::stable`]), where POUNCE finite-differences
//! whatever this oracle cannot fill exactly.

use oximo_autodiff::slot::{
    linear_gradient_add, linear_value, quadratic_gradient_add, quadratic_value,
};
use oximo_autodiff::sparsity::{hessian_lagrangian_structure, jacobian_structure};
use oximo_autodiff::tape::params_snapshot;
use oximo_autodiff::{FunctionSlot, SlotKind};
use oximo_core::Model;
use oximo_expr::ExprId;

use crate::tnlp::DerivativeOracle;

/// Classified objective/constraint slots plus the precomputed sparsity and
/// scratch space to serve [`DerivativeOracle`] on stable Rust.
pub(crate) struct HybridOracle {
    obj: FunctionSlot,
    cons: Vec<FunctionSlot>,
    params: Vec<f64>,
    n_vars: usize,
    obj_expr: Option<ExprId>,
    con_exprs: Vec<ExprId>,
    /// Row-major `(constraint, variable)` pattern; row `i` is `cons[i].support`.
    jac_structure: Vec<(usize, usize)>,
    /// Sorted lower-triangle Lagrangian-Hessian pattern; empty unless
    /// `exact_hessian` (and for a pure LP, where it is genuinely empty).
    hess_structure: Vec<(usize, usize)>,
    /// True iff every slot is linear/quadratic, i.e. the Hessian is constant.
    exact_hessian: bool,
    /// Scatter positions of each quadratic slot's Hessian triples into
    /// `hess_structure`, aligned entry-for-entry.
    obj_hess_pos: Vec<usize>,
    con_hess_pos: Vec<Vec<usize>>,
    /// Tape evaluation registers (max `n_regs` over all nonlinear slots).
    regs: Vec<f64>,
    /// Dense row buffer for gathering per-constraint gradients.
    row_scratch: Vec<f64>,
}

impl HybridOracle {
    pub(crate) fn new(model: &Model) -> Self {
        let arena = model.arena();
        let obj_expr = model.objective().as_ref().map(|o| o.expr);
        let obj = obj_expr.map_or_else(FunctionSlot::zero, |e| FunctionSlot::classify(&arena, e));
        let con_exprs: Vec<ExprId> = model.constraints().iter().map(|c| c.lhs).collect();
        let cons: Vec<FunctionSlot> =
            con_exprs.iter().map(|&e| FunctionSlot::classify(&arena, e)).collect();
        let params = params_snapshot(&arena);
        let mut oracle = Self {
            obj,
            cons,
            params,
            n_vars: model.variables().len(),
            obj_expr,
            con_exprs,
            jac_structure: Vec::new(),
            hess_structure: Vec::new(),
            exact_hessian: false,
            obj_hess_pos: Vec::new(),
            con_hess_pos: Vec::new(),
            regs: Vec::new(),
            row_scratch: Vec::new(),
        };
        oracle.rebuild_structures();
        oracle
    }

    /// Whether `model` has the same variables and the same objective/constraint
    /// expressions this oracle was built from, so a [`Self::refresh`] suffices.
    pub(crate) fn matches(&self, model: &Model) -> bool {
        self.n_vars == model.variables().len()
            && self.obj_expr == model.objective().as_ref().map(|o| o.expr)
            && self.con_exprs.len() == model.constraints().len()
            && model.constraints().iter().map(|c| c.lhs).eq(self.con_exprs.iter().copied())
    }

    /// Re-extract every slot at the current parameter values and re-snapshot
    /// the params.
    /// `extract_linear`/`extract_quadratic` fold parameters into
    /// linear/quadratic coefficients, so a params-only refresh
    /// would serve stale derivatives after `set_param`.
    pub(crate) fn refresh(&mut self, model: &Model) {
        let arena = model.arena();
        if let Some(e) = self.obj_expr {
            self.obj = self.obj.reclassify(&arena, e);
        }
        for (slot, &e) in self.cons.iter_mut().zip(&self.con_exprs) {
            *slot = slot.reclassify(&arena, e);
        }
        self.params = params_snapshot(&arena);
        self.rebuild_structures();
    }

    /// Recompute the sparsity patterns, Hessian scatter, and scratch sizes
    /// from the current slots.
    fn rebuild_structures(&mut self) {
        self.jac_structure = jacobian_structure(&self.cons);
        self.exact_hessian =
            !self.obj.is_nonlinear() && self.cons.iter().all(|s| !s.is_nonlinear());
        if self.exact_hessian {
            self.hess_structure =
                hessian_lagrangian_structure(std::iter::once(&self.obj).chain(self.cons.iter()));
            self.obj_hess_pos = scatter_positions(&self.hess_structure, &self.obj);
            self.con_hess_pos =
                self.cons.iter().map(|s| scatter_positions(&self.hess_structure, s)).collect();
        } else {
            self.hess_structure = Vec::new();
            self.obj_hess_pos = Vec::new();
            self.con_hess_pos = vec![Vec::new(); self.cons.len()];
        }
        let n_regs = std::iter::once(&self.obj)
            .chain(self.cons.iter())
            .filter_map(|s| match &s.kind {
                SlotKind::Nonlinear(tape) => Some(tape.n_regs()),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        self.regs = vec![0.0; n_regs];
        self.row_scratch = vec![0.0; self.n_vars];
    }

    /// Whether every slot is closed-form (linear/quadratic).
    pub(crate) fn all_closed_form(&self) -> bool {
        self.exact_hessian
    }

    /// Fill the exact objective gradient when the objective is closed-form.
    /// Returns `false` for a nonlinear objective, so POUNCE's builder falls
    /// back to its own finite differences.
    pub(crate) fn try_exact_objective_gradient(&self, x: &[f64], grad: &mut [f64]) -> bool {
        match &self.obj.kind {
            SlotKind::Linear(t) => {
                grad.fill(0.0);
                linear_gradient_add(t, 1.0, grad);
                true
            }
            SlotKind::Quadratic(q) => {
                grad.fill(0.0);
                quadratic_gradient_add(q, x, 1.0, grad);
                true
            }
            SlotKind::Nonlinear(_) => false,
        }
    }

    /// Fill the exact dense row-major `m x n` Jacobian when every constraint is closed-form.
    /// Returns `false` when one or more constraints are nonlinear.
    pub(crate) fn try_exact_dense_jacobian(&self, x: &[f64], jac: &mut [f64]) -> bool {
        if self.cons.iter().any(FunctionSlot::is_nonlinear) {
            return false;
        }
        if self.n_vars == 0 {
            return true;
        }
        jac.fill(0.0);
        for (slot, row) in self.cons.iter().zip(jac.chunks_mut(self.n_vars)) {
            slot_gradient_add(slot, x, row);
        }
        true
    }
}

/// Positions of a quadratic slot's Hessian triples in the sorted pattern.
/// Empty for linear (no second derivatives) and nonlinear (no exact values)
/// slots.
fn scatter_positions(hess: &[(usize, usize)], slot: &FunctionSlot) -> Vec<usize> {
    match &slot.kind {
        SlotKind::Quadratic(q) => q
            .hessian
            .iter()
            .map(|&(r, c, _)| {
                hess.binary_search(&(r.index(), c.index()))
                    .expect("quadratic Hessian entry missing from the Lagrangian pattern")
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn slot_value(slot: &FunctionSlot, x: &[f64], params: &[f64], regs: &mut [f64]) -> f64 {
    match &slot.kind {
        SlotKind::Linear(t) => linear_value(t, x),
        SlotKind::Quadratic(q) => quadratic_value(q, x),
        SlotKind::Nonlinear(tape) => tape.value(x, params, &[], regs),
    }
}

/// Add the closed-form slot's gradient at `x` into `out`, which must be
/// zeroed on the slot's support beforehand. Entries off the support are never
/// written.
fn slot_gradient_add(slot: &FunctionSlot, x: &[f64], out: &mut [f64]) {
    match &slot.kind {
        SlotKind::Linear(t) => linear_gradient_add(t, 1.0, out),
        SlotKind::Quadratic(q) => quadratic_gradient_add(q, x, 1.0, out),
        SlotKind::Nonlinear(_) => {
            unreachable!("nonlinear slots are finite-differenced by POUNCE's builder")
        }
    }
}

impl DerivativeOracle for HybridOracle {
    fn num_variables(&self) -> usize {
        self.n_vars
    }

    fn num_constraints(&self) -> usize {
        self.cons.len()
    }

    fn jacobian_structure(&self) -> &[(usize, usize)] {
        &self.jac_structure
    }

    fn hessian_structure(&self) -> &[(usize, usize)] {
        &self.hess_structure
    }

    fn has_exact_hessian(&self) -> bool {
        self.exact_hessian
    }

    fn eval_objective(&mut self, x: &[f64]) -> f64 {
        slot_value(&self.obj, x, &self.params, &mut self.regs)
    }

    fn eval_objective_gradient(&mut self, x: &[f64], grad: &mut [f64]) {
        grad.fill(0.0);
        slot_gradient_add(&self.obj, x, grad);
    }

    fn eval_constraints(&mut self, x: &[f64], g: &mut [f64]) {
        for (out, slot) in g.iter_mut().zip(&self.cons) {
            *out = slot_value(slot, x, &self.params, &mut self.regs);
        }
    }

    fn eval_constraint_jacobian(&mut self, x: &[f64], vals: &mut [f64]) {
        let mut k = 0;
        for slot in &self.cons {
            for &j in &slot.support {
                self.row_scratch[j as usize] = 0.0;
            }
            slot_gradient_add(slot, x, &mut self.row_scratch);
            for &j in &slot.support {
                vals[k] = self.row_scratch[j as usize];
                k += 1;
            }
        }
        debug_assert_eq!(k, vals.len(), "jacobian nnz");
    }

    /// Constant-Hessian scatter.
    fn eval_hessian_lagrangian(
        &mut self,
        _x: &[f64],
        obj_factor: f64,
        lambda: &[f64],
        vals: &mut [f64],
    ) {
        vals.fill(0.0);
        if let SlotKind::Quadratic(q) = &self.obj.kind {
            for (&pos, &(_, _, h)) in self.obj_hess_pos.iter().zip(&q.hessian) {
                vals[pos] += obj_factor * h;
            }
        }
        for ((slot, pos), &l) in self.cons.iter().zip(&self.con_hess_pos).zip(lambda) {
            if let SlotKind::Quadratic(q) = &slot.kind {
                for (&p, &(_, _, h)) in pos.iter().zip(&q.hessian) {
                    vals[p] += l * h;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use oximo_core::prelude::*;

    use super::*;

    fn assert_close(got: f64, want: f64, tol: f64, what: &str) {
        assert!((got - want).abs() <= tol, "{what}: got {got}, want {want}");
    }

    /// Mixed model: linear + quadratic + nonlinear slots in one oracle.
    fn mixed_model() -> Model {
        let m = Model::new("mixed");
        variable!(m, 0.5 <= x <= 5.0);
        variable!(m, 0.5 <= y <= 5.0);
        variable!(m, 0.5 <= z <= 5.0);
        objective!(m, Min, x * y * z + x.powi(2));
        constraint!(m, lin, x + 2.0 * y - z >= 0.0);
        constraint!(m, quad, x.powi(2) + y * z <= 10.0);
        constraint!(m, nl, x * y * z >= 1.0);
        m
    }

    #[test]
    fn mixed_model_classification_and_structures() {
        let m = mixed_model();
        let o = HybridOracle::new(&m);
        assert!(o.obj.is_nonlinear());
        assert!(!o.cons[0].is_nonlinear());
        assert!(!o.cons[1].is_nonlinear());
        assert!(o.cons[2].is_nonlinear());
        assert!(!o.exact_hessian, "one nonlinear slot must drop the exact Hessian");
        assert!(!o.all_closed_form(), "mixed model must route to the builder");
        assert!(o.hess_structure.is_empty());
        assert_eq!(
            o.jac_structure,
            vec![
                (0, 0),
                (0, 1),
                (0, 2), // x + 2y - z
                (1, 0),
                (1, 1),
                (1, 2), // x^2 + y z
                (2, 0),
                (2, 1),
                (2, 2), // x y z
            ]
        );

        let sentinel = vec![42.0; 3];
        let mut grad = sentinel.clone();
        assert!(!o.try_exact_objective_gradient(&[1.0, 1.0, 1.0], &mut grad));
        assert_eq!(grad, sentinel, "declined gradient must not be written");
        let mut jac = vec![42.0; 9];
        assert!(!o.try_exact_dense_jacobian(&[1.0, 1.0, 1.0], &mut jac));
        assert_eq!(jac, vec![42.0; 9], "declined jacobian must not be written");
    }

    #[test]
    fn closed_form_model_serves_exact_gradient_and_jacobian() {
        // All-quadratic model: everything is analytic, no differencing.
        let m = Model::new("closed_form");
        variable!(m, -5.0 <= x <= 5.0);
        variable!(m, -5.0 <= y <= 5.0);
        objective!(m, Min, x.powi(2) + x * y + 3.0 * y);
        constraint!(m, lin, x + 2.0 * y <= 4.0);
        constraint!(m, ball, x.powi(2) + y.powi(2) <= 1.0);

        let mut oracle = HybridOracle::new(&m);
        assert!(oracle.all_closed_form());
        let point = [0.3, 0.5];

        // grad f = [2x + y, x + 3].
        let mut grad = vec![0.0; 2];
        oracle.eval_objective_gradient(&point, &mut grad);
        assert_close(grad[0], 1.1, 1e-12, "df/dx");
        assert_close(grad[1], 3.3, 1e-12, "df/dy");

        // Sparse rows: lin [1, 2], ball [2x, 2y].
        let mut jac = vec![0.0; oracle.jac_structure.len()];
        oracle.eval_constraint_jacobian(&point, &mut jac);
        assert_eq!(oracle.jac_structure, vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
        assert_eq!(jac, vec![1.0, 2.0, 0.6, 1.0]);
    }

    #[test]
    fn builder_hooks_fill_exact_closed_form_parts() {
        let m = Model::new("nl_obj");
        variable!(m, 0.5 <= x <= 5.0);
        variable!(m, 0.5 <= y <= 5.0);
        variable!(m, 0.5 <= z <= 5.0);
        objective!(m, Min, x * y * z);
        constraint!(m, lin, x + 2.0 * y - z >= 0.0);
        constraint!(m, quad, x.powi(2) + y * z <= 10.0);

        let oracle = HybridOracle::new(&m);
        assert!(!oracle.all_closed_form());

        let point = [1.3, 0.9, 2.1];
        let mut grad = vec![0.0; 3];
        assert!(
            !oracle.try_exact_objective_gradient(&point, &mut grad),
            "nonlinear objective declines"
        );

        let mut jac = vec![42.0; 6];
        assert!(oracle.try_exact_dense_jacobian(&point, &mut jac), "all-closed-form fill");
        assert_eq!(&jac[..3], &[1.0, 2.0, -1.0], "lin row");
        assert_close(jac[3], 2.6, 1e-12, "d(quad)/dx = 2x");
        assert_close(jac[4], 2.1, 1e-12, "d(quad)/dy = z");
        assert_close(jac[5], 0.9, 1e-12, "d(quad)/dz = y");
    }

    #[test]
    fn quadratic_model_serves_exact_constant_hessian() {
        let m = Model::new("qp");
        variable!(m, -5.0 <= x <= 5.0);
        variable!(m, -5.0 <= y <= 5.0);
        objective!(m, Min, x.powi(2) + x * y + y.powi(2) + 3.0 * x);
        constraint!(m, ball, x.powi(2) + y.powi(2) <= 1.0);
        constraint!(m, lin, x + y >= 0.2);

        let mut o = HybridOracle::new(&m);
        assert!(o.exact_hessian);
        // Lower triangle of the union pattern: obj {(0,0),(1,0),(1,1)},
        // ball {(0,0),(1,1)}, lin {}.
        assert_eq!(o.hess_structure, vec![(0, 0), (1, 0), (1, 1)]);

        // sigma * H_obj + lambda_0 * H_ball with H_obj = [[2,1],[1,2]],
        // H_ball = [[2,0],[0,2]].
        let mut vals = vec![0.0; 3];
        o.eval_hessian_lagrangian(&[0.0, 0.0], 2.0, &[0.5, 7.0], &mut vals);
        assert_close(vals[0], 2.0 * 2.0 + 0.5 * 2.0, 1e-12, "H[0,0]");
        assert_close(vals[1], 2.0 * 1.0, 1e-12, "H[1,0]");
        assert_close(vals[2], 2.0 * 2.0 + 0.5 * 2.0, 1e-12, "H[1,1]");
    }

    #[test]
    fn feasibility_objective_is_the_zero_slot() {
        let m = Model::new("feas");
        variable!(m, 0.0 <= x <= 1.0);
        constraint!(m, half, x >= 0.5);
        objective!(m, Feasibility);

        let mut o = HybridOracle::new(&m);
        assert!(o.exact_hessian);
        assert_close(o.eval_objective(&[0.7]), 0.0, 0.0, "feasibility objective");
        let mut grad = vec![1.0; 1];
        o.eval_objective_gradient(&[0.7], &mut grad);
        assert_close(grad[0], 0.0, 0.0, "feasibility gradient");
    }

    #[test]
    fn refresh_reextracts_folded_parameters() {
        let m = Model::new("fold");
        param!(m, w = 1.0);
        variable!(m, -5.0 <= x <= 5.0);
        objective!(m, Min, w * x.powi(2) + x);
        constraint!(m, lin, w * x >= -10.0);

        let mut o = HybridOracle::new(&m);
        assert!(o.exact_hessian);
        let mut vals = vec![0.0; o.hess_structure.len()];
        o.eval_hessian_lagrangian(&[0.0], 1.0, &[0.0], &mut vals);
        assert_close(vals[0], 2.0, 1e-12, "H with w=1");

        w.set_param_value(3.0);
        assert!(o.matches(&m), "same structure after set_param");
        o.refresh(&m);
        let mut vals = vec![0.0; o.hess_structure.len()];
        o.eval_hessian_lagrangian(&[0.0], 1.0, &[0.0], &mut vals);
        assert_close(vals[0], 6.0, 1e-12, "H with w=3: coefficients must re-fold");

        let mut jac = vec![0.0; o.jac_structure.len()];
        o.eval_constraint_jacobian(&[1.0], &mut jac);
        assert_close(jac[0], 3.0, 1e-12, "constraint coefficient after refresh");
    }
}
