//! Gradient values at a point, using closed forms or reverse-mode automatic differentiation.

use oximo_core::Model;
use oximo_expr::{Expr, extract_linear, extract_quadratic_shared};

use crate::enzyme::tape_gradient;
use crate::error::AutodiffError;
use crate::slot::{linear_gradient_add, quadratic_gradient_add};
use crate::tape::Tape;

/// Dense gradient of `expr` (an expression on `model`'s arena) at `x`,
/// indexed by variable. Affine and quadratic expressions use closed-form
/// derivatives at the current parameter values, while nonlinear expressions use Enzyme.
///
/// # Errors
///
/// Returns [`AutodiffError::DimensionMismatch`] if `x.len()` differs from the
/// model's variable count.
pub fn gradient_at(model: &Model, expr: Expr<'_>, x: &[f64]) -> Result<Vec<f64>, AutodiffError> {
    let n = model.num_variables();
    if x.len() != n {
        return Err(AutodiffError::DimensionMismatch { expected: n, got: x.len() });
    }
    let mut grad = vec![0.0; n];
    let (tape, params) = {
        let arena = expr.arena().borrow();
        let root = expr.id();
        if let Some(terms) = extract_linear(&arena, root) {
            linear_gradient_add(&terms, 1.0, &mut grad);
            return Ok(grad);
        }
        if let Some(terms) = extract_quadratic_shared(&arena, root) {
            quadratic_gradient_add(&terms, x, 1.0, &mut grad);
            return Ok(grad);
        }
        (Tape::compile(&arena, root), crate::tape::params_snapshot(&arena))
    };
    let mut regs = vec![0.0; tape.n_regs()];
    let mut dregs = vec![0.0; tape.n_regs()];
    let mut seed = vec![0.0; n];
    tape_gradient(&tape, x, &params, &[], &mut regs, &mut dregs, &mut seed, &mut grad);
    Ok(grad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximo_core::prelude::*;

    #[test]
    fn gradient_at_uses_affine_coefficients_and_current_parameters() {
        let m = Model::new("affine gradients");
        variable!(m, x);
        variable!(m, y);
        param!(m, p = 2.0);
        let affine = p * x + 3.0 * y + 7.0;
        assert_eq!(gradient_at(&m, affine.erase(), &[1.0, 2.0]).unwrap(), [2.0, 3.0]);

        m.set_param(p, -4.0).unwrap();
        assert_eq!(gradient_at(&m, affine.into_dynamic(), &[1.0, 2.0]).unwrap(), [-4.0, 3.0]);

        let constant = Expr::constant(m.__sum_context(), 5.0);
        assert_eq!(gradient_at(&m, constant.into_dynamic(), &[1.0, 2.0]).unwrap(), [0.0, 0.0]);
    }

    #[test]
    fn gradient_at_preserves_tiny_and_large_quadratic_derivatives() {
        let m = Model::new("quadratic gradient range");
        variable!(m, x);
        let tiny = f64::from_bits(1);
        for (coefficient, point, expected) in
            [(tiny, 0.5, tiny), (1e308, 0.0, 0.0), (1e308, 0.5, 1e308)]
        {
            let mut builder = m.quadratic_builder();
            builder.add_product(coefficient, x, x).unwrap();
            let gradient = gradient_at(&m, builder.build().into_dynamic(), &[point]).unwrap();
            assert_eq!(gradient[0].to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn gradient_at_reextracts_parameterized_quadratics() {
        let m = Model::new("parameterized quadratic gradients");
        variable!(m, x);
        variable!(m, y);
        param!(m, p = 2.0);
        let quadratic = p * x * x + x * y + 3.0 * y;
        assert_eq!(gradient_at(&m, quadratic.erase(), &[0.5, 2.0]).unwrap(), [4.0, 3.5]);

        m.set_param(p, -4.0).unwrap();
        assert_eq!(gradient_at(&m, quadratic.into_dynamic(), &[0.5, 2.0]).unwrap(), [-2.0, 3.5]);
    }
}
