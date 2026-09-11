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
use oximo_autodiff::sparsity::hessian_lagrangian_structure;
use oximo_autodiff::tape::params_snapshot;
use oximo_autodiff::{FunctionSlot, SlotKind};
use oximo_core::{Model, ModelKind};
use oximo_expr::ExprId;
use rayon::prelude::*;

use crate::tnlp::DerivativeOracle;

// Full-oracle initialization benefits from parallel slot classification for
// NLP at 32 rows and QCP at 128 rows. Other classes remain serial.
// All refresh/value/Jacobian work remains serial.
const PAR_CLASSIFY_THRESHOLD: usize = 32;
const PAR_QCP_CLASSIFY_THRESHOLD: usize = 128;

fn should_parallelize_classification(kind: ModelKind, len: usize) -> bool {
    let large_enough = match kind {
        ModelKind::NLP => len >= PAR_CLASSIFY_THRESHOLD,
        ModelKind::QCP => len >= PAR_QCP_CLASSIFY_THRESHOLD,
        _ => false,
    };
    large_enough && rayon::current_num_threads() > 1
}

fn classify_slots(
    arena: &oximo_expr::ExprArena,
    exprs: &[ExprId],
    parallel: bool,
) -> Vec<FunctionSlot> {
    if parallel {
        exprs.par_iter().map(|&e| FunctionSlot::classify(arena, e)).collect()
    } else {
        exprs.iter().map(|&e| FunctionSlot::classify(arena, e)).collect()
    }
}

struct JacobianMetadata {
    structure: Vec<(usize, usize)>,
    linear_scatter: Vec<usize>,
    hessian_scatter: Vec<(usize, usize)>,
}

fn jacobian_metadata(slots: &[FunctionSlot]) -> JacobianMetadata {
    let nnz = slots.iter().map(|slot| slot.support.len()).sum();
    let mut structure = Vec::with_capacity(nnz);
    let linear_count = slots
        .iter()
        .map(|slot| match &slot.kind {
            SlotKind::Linear(terms) => terms.coeffs.len(),
            SlotKind::Quadratic(terms) => terms.linear.len(),
            SlotKind::Nonlinear(_) => 0,
        })
        .sum();
    let hessian_count = slots
        .iter()
        .map(|slot| match &slot.kind {
            SlotKind::Quadratic(terms) => terms.hessian.len(),
            SlotKind::Linear(_) | SlotKind::Nonlinear(_) => 0,
        })
        .sum();
    let mut linear_scatter = Vec::with_capacity(linear_count);
    let mut hessian_scatter = Vec::with_capacity(hessian_count);
    for (row, slot) in slots.iter().enumerate() {
        structure.extend(slot.support.iter().map(|&var| (row, var as usize)));
        let position = |var: u32| {
            slot.support.binary_search(&var).expect("slot term missing from its Jacobian support")
        };
        match &slot.kind {
            SlotKind::Linear(terms) => {
                linear_scatter.extend(terms.coeffs.iter().map(|(var, _)| position(var.0)));
            }
            SlotKind::Quadratic(terms) => {
                linear_scatter.extend(terms.linear.iter().map(|(var, _)| position(var.0)));
                hessian_scatter.extend(
                    terms.hessian.iter().map(|(row, col, _)| (position(row.0), position(col.0))),
                );
            }
            SlotKind::Nonlinear(_) => {}
        }
    }
    JacobianMetadata { structure, linear_scatter, hessian_scatter }
}

/// Classified objective/constraint slots plus the precomputed sparsity and
/// scratch space to serve [`DerivativeOracle`] on stable Rust.
#[derive(Debug)]
pub(crate) struct HybridOracle {
    obj: FunctionSlot,
    cons: Vec<FunctionSlot>,
    params: Vec<f64>,
    n_vars: usize,
    obj_expr: Option<ExprId>,
    con_exprs: Vec<ExprId>,
    /// Row-major `(constraint, variable)` pattern; row `i` is `cons[i].support`.
    jac_structure: Vec<(usize, usize)>,
    /// Linear-term positions in compact Jacobian rows, concatenated in slot order.
    jac_linear_scatter: Vec<usize>,
    /// Hessian endpoint positions in compact Jacobian rows, concatenated in slot order.
    jac_hessian_scatter: Vec<(usize, usize)>,
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
}

impl HybridOracle {
    pub(crate) fn new(model: &Model) -> Self {
        let parallel =
            should_parallelize_classification(model.kind(), model.constraints().algebraic().len());
        Self::new_with(model, parallel)
    }

    fn new_with(model: &Model, parallel: bool) -> Self {
        let arena = model.arena();
        let obj_expr = model.objective().as_ref().map(|o| o.expr);
        let obj = obj_expr.map_or_else(FunctionSlot::zero, |e| FunctionSlot::classify(&arena, e));
        let con_exprs: Vec<ExprId> =
            model.constraints().algebraic().iter().map(|c| c.lhs).collect();
        let arena_ref = &*arena;
        let cons = classify_slots(arena_ref, &con_exprs, parallel);
        let params = params_snapshot(&arena);
        let mut oracle = Self {
            obj,
            cons,
            params,
            n_vars: model.variables().len(),
            obj_expr,
            con_exprs,
            jac_structure: Vec::new(),
            jac_linear_scatter: Vec::new(),
            jac_hessian_scatter: Vec::new(),
            hess_structure: Vec::new(),
            exact_hessian: false,
            obj_hess_pos: Vec::new(),
            con_hess_pos: Vec::new(),
            regs: Vec::new(),
        };
        oracle.rebuild_structures();
        oracle
    }

    /// Whether `model` has the same variables and the same objective/constraint
    /// expressions this oracle was built from, so a [`Self::refresh`] suffices.
    pub(crate) fn matches(&self, model: &Model) -> bool {
        let model_constraints = model.constraints();
        let constraints = model_constraints.algebraic();
        self.n_vars == model.variables().len()
            && self.obj_expr == model.objective().as_ref().map(|o| o.expr)
            && self.con_exprs.len() == constraints.len()
            && constraints.iter().map(|c| c.lhs).eq(self.con_exprs.iter().copied())
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
        let metadata = jacobian_metadata(&self.cons);
        self.jac_structure = metadata.structure;
        self.jac_linear_scatter = metadata.linear_scatter;
        self.jac_hessian_scatter = metadata.hessian_scatter;
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
        let mut linear_k = 0;
        let mut hessian_k = 0;
        for slot in &self.cons {
            let row = &mut vals[k..k + slot.support.len()];
            row.fill(0.0);
            match &slot.kind {
                SlotKind::Linear(terms) => {
                    let positions =
                        &self.jac_linear_scatter[linear_k..linear_k + terms.coeffs.len()];
                    for (&position, &(_, coefficient)) in positions.iter().zip(terms.coeffs.iter())
                    {
                        row[position] += coefficient;
                    }
                    linear_k += terms.coeffs.len();
                }
                SlotKind::Quadratic(terms) => {
                    let linear_positions =
                        &self.jac_linear_scatter[linear_k..linear_k + terms.linear.len()];
                    for (&position, &(_, coefficient)) in linear_positions.iter().zip(&terms.linear)
                    {
                        row[position] += coefficient;
                    }
                    linear_k += terms.linear.len();
                    let hessian_positions =
                        &self.jac_hessian_scatter[hessian_k..hessian_k + terms.hessian.len()];
                    for (&(row_position, col_position), &(row_var, col_var, hessian)) in
                        hessian_positions.iter().zip(&terms.hessian)
                    {
                        if row_var == col_var {
                            row[row_position] += hessian * x[row_var.index()];
                        } else {
                            row[row_position] += hessian * x[col_var.index()];
                            row[col_position] += hessian * x[row_var.index()];
                        }
                    }
                    hessian_k += terms.hessian.len();
                }
                SlotKind::Nonlinear(_) => {
                    unreachable!("nonlinear slots are finite-differenced by POUNCE's builder")
                }
            }
            k += row.len();
        }
        debug_assert_eq!(k, vals.len(), "jacobian nnz");
        debug_assert_eq!(linear_k, self.jac_linear_scatter.len(), "linear jacobian scatter");
        debug_assert_eq!(hessian_k, self.jac_hessian_scatter.len(), "Hessian jacobian scatter");
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

#[cfg(feature = "benchmark-support")]
#[doc(hidden)]
#[expect(clippy::cast_precision_loss)]
#[allow(clippy::wildcard_imports)]
pub mod benchmark_support {
    use oximo_core::constraint::Relate;

    use super::*;

    /// Crossover candidate used only to size the preprocessing benchmark cases.
    pub const CLASSIFY_THRESHOLD: usize = PAR_CLASSIFY_THRESHOLD;
    /// QCP full-oracle initialization crossover candidate.
    pub const QCP_CLASSIFY_THRESHOLD: usize = PAR_QCP_CLASSIFY_THRESHOLD;
    /// Crossover candidate used only to size the preprocessing benchmark cases.
    pub const VALUE_THRESHOLD: usize = 1_024;
    /// Crossover candidate used only to size the preprocessing benchmark cases.
    pub const JACOBIAN_THRESHOLD: usize = 1_024;
    // Keep each task large enough to amortize its register buffer.
    const VALUE_MIN_LEN: usize = 64;

    /// Exercise the production NLP routing and bound-snapshot path without
    /// constructing the derivative oracle or entering the optimizer.
    pub fn prepare_nlp(model: &Model) -> usize {
        let opts = crate::PounceOptions::default();
        let prepared = oximo_solver::prepare::LoweringContext::new(model).unwrap();
        assert_eq!(crate::convex::route(&prepared, &opts).unwrap(), crate::convex::Route::Nlp);
        let setup = crate::translate::setup_prepared(&prepared, &opts).unwrap();
        setup.x_l.len() + setup.x_u.len() + setup.x0.len() + setup.g_l.len() + setup.g_u.len()
    }

    pub fn model(rows: usize, nonlinear: bool) -> Model {
        let model = Model::new("pounce_bench");
        let x = model.__var("x").lb(-5.0).ub(5.0).build();
        let y = model.__var("y").lb(-5.0).ub(5.0).build();
        let z = model.__var("z").lb(-5.0).ub(5.0).build();
        model.__minimize(x.powi(2) + y + z);
        for i in 0..rows {
            let lhs = if nonlinear {
                x * y * z + (i as f64 + 1.0) * x
            } else if i % 2 == 0 {
                x.powi(2) + y * z + (i as f64 + 1.0) * x
            } else {
                x + 2.0 * y - z
            };
            model.__add_constraint_auto(lhs.le(i as f64 + 10.0));
        }
        model
    }

    pub fn jacobian_model(rows: usize, n_vars: usize) -> Model {
        assert!(n_vars >= 3);
        let model = Model::new("pounce_jacobian_bench");
        let vars: Vec<_> =
            (0..n_vars).map(|i| model.__var(format!("x{i}")).lb(-5.0).ub(5.0).build()).collect();
        model.__minimize(vars[0].powi(2));
        for i in 0..rows {
            let x = vars[i % n_vars];
            let y = vars[(i + 1) % n_vars];
            let z = vars[(i + 2) % n_vars];
            let lhs = if i % 2 == 0 { x.powi(2) + y * z } else { x + 2.0 * y - z };
            model.__add_constraint_auto(lhs.le(i as f64 + 10.0));
        }
        model
    }

    pub fn classify(model: &Model, parallel: bool) -> usize {
        let arena = model.arena();
        let exprs: Vec<ExprId> = model.constraints().algebraic().iter().map(|c| c.lhs).collect();
        let arena_ref = &*arena;
        if parallel {
            exprs.par_iter().map(|&e| FunctionSlot::classify(arena_ref, e).support.len()).sum()
        } else {
            exprs.iter().map(|&e| FunctionSlot::classify(arena_ref, e).support.len()).sum()
        }
    }

    pub fn initialize(model: &Model, parallel: bool) -> (usize, usize, bool) {
        let oracle = HybridOracle::new_with(model, parallel);
        (oracle.jac_structure.len(), oracle.hess_structure.len(), oracle.exact_hessian)
    }

    pub struct Oracle {
        inner: HybridOracle,
        point: Vec<f64>,
        values: Vec<f64>,
        sparse: Vec<f64>,
        dense: Vec<f64>,
    }

    impl std::fmt::Debug for Oracle {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Oracle").field("constraints", &self.values.len()).finish()
        }
    }

    impl Oracle {
        pub fn new(model: &Model) -> Self {
            let inner = HybridOracle::new(model);
            let values = vec![0.0; inner.cons.len()];
            let sparse = vec![0.0; inner.jac_structure.len()];
            let dense = Vec::new();
            let point = (0..inner.n_vars)
                .map(|i| match i % 3 {
                    0 => 0.7,
                    1 => -1.1,
                    _ => 0.4,
                })
                .collect();
            Self { inner, point, values, sparse, dense }
        }

        pub fn values(&mut self, parallel: bool) -> f64 {
            if parallel {
                self.values
                    .par_iter_mut()
                    .zip(self.inner.cons.par_iter())
                    .with_min_len(VALUE_MIN_LEN)
                    .for_each_with(vec![0.0; self.inner.regs.len()], |regs, (out, slot)| {
                        *out = slot_value(slot, &self.point, &self.inner.params, regs);
                    });
            } else {
                self.inner.eval_constraints(&self.point, &mut self.values);
            }
            self.values.iter().sum()
        }

        pub fn sparse_jacobian(&mut self, parallel: bool) -> f64 {
            if parallel {
                let rows: Vec<Vec<f64>> = self
                    .inner
                    .cons
                    .par_iter()
                    .map_init(
                        || vec![0.0; self.inner.n_vars],
                        |scratch, slot| {
                            for &j in &slot.support {
                                scratch[j as usize] = 0.0;
                            }
                            slot_gradient_add(slot, &self.point, scratch);
                            slot.support.iter().map(|&j| scratch[j as usize]).collect()
                        },
                    )
                    .collect();
                self.sparse.clear();
                self.sparse.extend(rows.into_iter().flatten());
            } else {
                self.inner.eval_constraint_jacobian(&self.point, &mut self.sparse);
            }
            self.sparse.iter().sum()
        }

        pub fn dense_jacobian(&mut self, parallel: bool) -> f64 {
            self.dense.resize(self.inner.cons.len() * self.inner.n_vars, 0.0);
            if parallel {
                assert!(!self.inner.cons.iter().any(FunctionSlot::is_nonlinear));
                self.dense
                    .par_chunks_mut(self.inner.n_vars)
                    .zip(self.inner.cons.par_iter())
                    .for_each(|(row, slot)| {
                        row.fill(0.0);
                        slot_gradient_add(slot, &self.point, row);
                    });
            } else {
                assert!(self.inner.try_exact_dense_jacobian(&self.point, &mut self.dense));
            }
            self.dense.iter().sum()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn parallel_scratch_matches_serial_results() {
            let model = super::model(33, false);
            let mut serial = Oracle::new(&model);
            let mut parallel = Oracle::new(&model);

            serial.values(false);
            parallel.values(true);
            assert_eq!(serial.values, parallel.values);
            serial.values(false);
            parallel.values(true);
            assert_eq!(serial.values, parallel.values);

            serial.sparse_jacobian(false);
            parallel.sparse_jacobian(true);
            assert_eq!(serial.sparse, parallel.sparse);
            serial.sparse_jacobian(false);
            parallel.sparse_jacobian(true);
            assert_eq!(serial.sparse, parallel.sparse);
        }
    }
}

#[cfg(test)]
#[expect(clippy::cast_precision_loss)]
mod tests {
    use oximo_core::prelude::*;
    use rayon::ThreadPoolBuilder;

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

    #[test]
    fn refresh_rebuilds_jacobian_scatter_when_support_changes() {
        let m = Model::new("scatter_refresh");
        param!(m, w = 1.0);
        variable!(m, -5.0 <= x <= 5.0);
        objective!(m, Min, x);
        constraint!(m, scaled_square, w * x.powi(2) <= 4.0);

        let mut oracle = HybridOracle::new(&m);
        assert_eq!(oracle.jac_structure, vec![(0, 0)]);
        let mut jac = vec![0.0; oracle.jac_structure.len()];
        oracle.eval_constraint_jacobian(&[3.0], &mut jac);
        assert_eq!(jac, vec![6.0]);

        w.set_param_value(0.0);
        oracle.refresh(&m);
        assert!(oracle.jac_structure.is_empty());
        let mut jac = Vec::new();
        oracle.eval_constraint_jacobian(&[3.0], &mut jac);
        assert!(jac.is_empty());

        w.set_param_value(2.0);
        oracle.refresh(&m);
        assert_eq!(oracle.jac_structure, vec![(0, 0)]);
        let mut jac = vec![0.0; oracle.jac_structure.len()];
        oracle.eval_constraint_jacobian(&[3.0], &mut jac);
        assert_eq!(jac, vec![12.0]);
    }

    fn repeated_model(rows: usize, nonlinear: bool) -> Model {
        let m = Model::new("repeated");
        variable!(m, -5.0 <= x <= 5.0);
        variable!(m, -5.0 <= y <= 5.0);
        variable!(m, -5.0 <= z <= 5.0);
        objective!(m, Min, x.powi(2) + y + z);
        for i in 0..rows {
            let lhs = if nonlinear {
                x * y * z + (i as f64 + 1.0) * x
            } else if i % 2 == 0 {
                x.powi(2) + y * z + (i as f64 + 1.0) * x
            } else {
                x + 2.0 * y - z
            };
            m.__add_constraint_auto(lhs.le(i as f64 + 10.0));
        }
        m
    }

    #[test]
    fn forced_serial_and_parallel_classification_are_identical() {
        for model in [
            repeated_model(PAR_CLASSIFY_THRESHOLD + 3, true),
            repeated_model(PAR_QCP_CLASSIFY_THRESHOLD + 3, false),
        ] {
            let serial = HybridOracle::new_with(&model, false);
            let parallel = HybridOracle::new_with(&model, true);
            assert_eq!(serial.jac_structure, parallel.jac_structure);
            assert_eq!(serial.hess_structure, parallel.hess_structure);
            assert_eq!(serial.exact_hessian, parallel.exact_hessian);
            assert_eq!(serial.jac_linear_scatter, parallel.jac_linear_scatter);
            assert_eq!(serial.jac_hessian_scatter, parallel.jac_hessian_scatter);
        }
    }

    #[test]
    fn automatic_classification_respects_kind_pool_and_threshold() {
        let one = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        one.install(|| {
            assert!(!should_parallelize_classification(ModelKind::NLP, PAR_CLASSIFY_THRESHOLD));
            assert!(!should_parallelize_classification(ModelKind::QCP, PAR_QCP_CLASSIFY_THRESHOLD));
        });
        let many = ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        many.install(|| {
            assert!(!should_parallelize_classification(ModelKind::NLP, PAR_CLASSIFY_THRESHOLD - 1));
            assert!(should_parallelize_classification(ModelKind::NLP, PAR_CLASSIFY_THRESHOLD));
            assert!(!should_parallelize_classification(
                ModelKind::QCP,
                PAR_QCP_CLASSIFY_THRESHOLD - 1
            ));
            assert!(should_parallelize_classification(ModelKind::QCP, PAR_QCP_CLASSIFY_THRESHOLD));
            assert!(!should_parallelize_classification(ModelKind::QP, PAR_QCP_CLASSIFY_THRESHOLD));
        });
    }
}
