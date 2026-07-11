//! Gradient values at a point, evaluated with reverse-mode autodiff.

use oximo_core::Model;
use oximo_expr::Expr;

use crate::enzyme::tape_gradient;
use crate::error::AutodiffError;
use crate::tape::Tape;

/// Dense gradient of `expr` (an expression on `model`'s arena) at `x`,
/// indexed by variable.
///
/// # Errors
///
/// Returns [`AutodiffError::DimensionMismatch`] if `x.len()` differs from the
/// model's variable count.
pub fn gradient_at(model: &Model, expr: Expr<'_>, x: &[f64]) -> Result<Vec<f64>, AutodiffError> {
    let n = model.variables().len();
    if x.len() != n {
        return Err(AutodiffError::DimensionMismatch { expected: n, got: x.len() });
    }
    let (tape, params) = compile(expr);
    let mut regs = vec![0.0; tape.n_regs()];
    let mut dregs = vec![0.0; tape.n_regs()];
    let mut grad = vec![0.0; n];
    tape_gradient(&tape, x, &params, &[], &mut regs, &mut dregs, &mut grad);
    Ok(grad)
}

fn compile(expr: Expr<'_>) -> (Tape, Vec<f64>) {
    let arena = expr.arena.borrow();
    let tape = Tape::compile(&arena, expr.id);
    let params = crate::tape::params_snapshot(&arena);
    (tape, params)
}
