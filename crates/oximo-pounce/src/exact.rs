//! Exact-derivative path (enzyme feature): plugs `oximo-autodiff`'s
//! [`NlpEvaluator`] (exact gradient, sparse Jacobian, and sparse Hessian of
//! the Lagrangian) into the shared [`crate::tnlp`] adapter.

use std::cell::RefCell;
use std::rc::Rc;

use oximo_autodiff::NlpEvaluator;
use oximo_core::Model;
use oximo_solver::SolverError;

use crate::tnlp::DerivativeOracle;

/// The resident derivative oracle, shared between the handle and the `TNLP`.
pub(crate) type Oracle = Rc<RefCell<NlpEvaluator>>;

/// Build a fresh evaluator.
///
/// Fails only when the model has no objective, which
/// [`crate::translate::setup`] already verifies.
pub(crate) fn build(model: &Model) -> Result<Oracle, SolverError> {
    let eval =
        NlpEvaluator::new(model).map_err(|e| SolverError::Backend(format!("autodiff: {e}")))?;
    Ok(Rc::new(RefCell::new(eval)))
}

/// Reuse the resident evaluator for `model` when the structure is unchanged.
pub(crate) fn try_reuse(oracle: &Oracle, model: &Model) -> bool {
    oracle.borrow_mut().try_refresh(model)
}

/// Exact derivatives always drive the `TNLP` surface directly.
pub(crate) fn run(
    oracle: &Oracle,
    prep: &crate::translate::Prepared,
    opts: &crate::options::PounceOptions,
    warm: Option<&crate::translate::WarmStart>,
) -> Result<crate::translate::Outcome, SolverError> {
    crate::tnlp::run(oracle, prep, opts, warm)
}

impl DerivativeOracle for NlpEvaluator {
    fn num_variables(&self) -> usize {
        NlpEvaluator::num_variables(self)
    }

    fn num_constraints(&self) -> usize {
        NlpEvaluator::num_constraints(self)
    }

    fn jacobian_structure(&self) -> &[(usize, usize)] {
        NlpEvaluator::jacobian_structure(self)
    }

    fn hessian_structure(&self) -> &[(usize, usize)] {
        self.hessian_lagrangian_structure()
    }

    fn has_exact_hessian(&self) -> bool {
        true
    }

    fn eval_objective(&mut self, x: &[f64]) -> f64 {
        NlpEvaluator::eval_objective(self, x)
    }

    fn eval_objective_gradient(&mut self, x: &[f64], grad: &mut [f64]) {
        NlpEvaluator::eval_objective_gradient(self, x, grad);
    }

    fn eval_constraints(&mut self, x: &[f64], g: &mut [f64]) {
        self.eval_constraint(x, g);
    }

    fn eval_constraint_jacobian(&mut self, x: &[f64], vals: &mut [f64]) {
        NlpEvaluator::eval_constraint_jacobian(self, x, vals);
    }

    fn eval_hessian_lagrangian(
        &mut self,
        x: &[f64],
        obj_factor: f64,
        lambda: &[f64],
        vals: &mut [f64],
    ) {
        NlpEvaluator::eval_hessian_lagrangian(self, x, obj_factor, lambda, vals);
    }
}
