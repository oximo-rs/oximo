//! Per-function classification.

use oximo_expr::{
    ExprArena, ExprId, LinearTerms, QuadraticTerms, extract_linear, extract_quadratic_shared,
};
use std::sync::Arc;

use crate::sparsity::{SparsityWorkspace, structural_sparsity_with_workspace};
use crate::tape::Tape;

/// One objective or constraint function, classified for derivative purposes.
#[derive(Clone, Debug)]
pub struct FunctionSlot {
    pub kind: SlotKind,
    /// Sorted, deduplicated structural derivative support. Variables appearing
    /// only in inactive subexpressions, such as a literal zero power's base,
    /// are excluded. Use [`crate::sparsity::variable_support`] for syntactic support.
    pub support: Vec<u32>,
    /// Exact structural lower-triangle Hessian pattern. Populated only for
    /// `Nonlinear` slots.
    pub hess_pairs: Vec<(u32, u32)>,
}

#[derive(Clone, Debug)]
pub enum SlotKind {
    /// Affine: constant gradient, zero Hessian.
    Linear(LinearTerms<'static>),
    /// Degree-2 polynomial: affine gradient `Qx + c`, constant Hessian.
    /// Polynomial coefficients, converted to derivatives at eval time.
    Quadratic(Arc<QuadraticTerms>),
    /// Everything else: evaluated and differentiated through the tape
    /// interpreter.
    Nonlinear(Tape),
}

impl FunctionSlot {
    /// The constant-zero function.
    /// A `Linear` slot with no terms, so it has no support,
    /// a zero gradient, and no Hessian.
    /// Used for a feasibility problem's absent objective.
    pub fn zero() -> Self {
        Self {
            kind: SlotKind::Linear(LinearTerms { coeffs: Vec::new().into(), constant: 0.0 }),
            support: Vec::new(),
            hess_pairs: Vec::new(),
        }
    }

    /// Classify the expression rooted at `root`.
    ///
    /// `extract_linear`/`extract_quadratic` fold parameters to
    /// their current values, so linear and quadratic slots must be
    /// re-classified after `set_param`.
    pub fn classify(arena: &ExprArena, root: ExprId) -> Self {
        Self::classify_with_workspace(arena, root, &mut SparsityWorkspace::default())
    }

    pub(crate) fn classify_with_workspace(
        arena: &ExprArena,
        root: ExprId,
        workspace: &mut SparsityWorkspace,
    ) -> Self {
        Self::closed_form(arena, root).unwrap_or_else(|| {
            let tape = Tape::compile(arena, root);
            let sparsity = structural_sparsity_with_workspace(arena, root, workspace);
            Self {
                kind: SlotKind::Nonlinear(tape),
                support: sparsity.support,
                hess_pairs: sparsity.hess_pairs,
            }
        })
    }

    /// The linear/quadratic closed-form classification, or `None` when the
    /// expression is nonlinear. Both `classify` and `reclassify` share this so
    /// the extraction-and-support logic lives in one place, the two differ only
    /// in their nonlinear fallback.
    fn closed_form(arena: &ExprArena, root: ExprId) -> Option<Self> {
        if let Some(linear) = extract_linear(arena, root) {
            let support = sorted_dedup(linear.coeffs.iter().map(|(v, _)| v.0));
            return Some(Self {
                kind: SlotKind::Linear(linear.into_owned()),
                support,
                hess_pairs: Vec::new(),
            });
        }
        if let Some(quadratic) = extract_quadratic_shared(arena, root) {
            let support = sorted_dedup(
                quadratic
                    .linear()
                    .iter()
                    .map(|(v, _)| v.0)
                    .chain(quadratic.quadratic().iter().flat_map(|&(r, c, _)| [r.0, c.0])),
            );
            return Some(Self {
                kind: SlotKind::Quadratic(quadratic),
                support,
                hess_pairs: Vec::new(),
            });
        }
        None
    }

    pub fn is_nonlinear(&self) -> bool {
        matches!(self.kind, SlotKind::Nonlinear(_))
    }

    /// Re-classify `root` at the current parameter values, reusing `self`'s
    /// compiled tape (and its support/Hessian pattern) when the function is
    /// still nonlinear.
    /// A nonlinear tape is parameter-independent, so it never needs recompiling.
    /// Linear/quadratic functions are re-extracted because their coefficients
    /// can move with parameters.
    ///
    /// Used by `NlpEvaluator::refresh_params` (behind the `enzyme` feature) to
    /// update a resident evaluator after `set_param` without retaping.
    pub fn reclassify(&self, arena: &ExprArena, root: ExprId) -> Self {
        Self::closed_form(arena, root).unwrap_or_else(|| {
            // Still nonlinear, reuse the existing tape/pattern when we have one,
            // otherwise fall back to a full classify.
            match self.kind {
                SlotKind::Nonlinear(_) => self.clone(),
                _ => Self::classify(arena, root),
            }
        })
    }
}

/// Collect variable indices into a sorted, deduplicated support vector.
fn sorted_dedup(vars: impl Iterator<Item = u32>) -> Vec<u32> {
    let mut support: Vec<u32> = vars.collect();
    support.sort_unstable();
    support.dedup();
    support
}

/// `t.constant + t.coeffs * x`
pub fn linear_value(t: &LinearTerms<'_>, x: &[f64]) -> f64 {
    t.coeffs.iter().fold(t.constant, |acc, &(v, c)| acc + c * x[v.index()])
}

/// Add the (constant) gradient of `t` into `grad`, scaled by `scale`.
pub fn linear_gradient_add(t: &LinearTerms<'_>, scale: f64, grad: &mut [f64]) {
    for &(v, c) in t.coeffs.iter() {
        grad[v.index()] += scale * c;
    }
}

/// Evaluate ordinary polynomial coefficients at `x`.
pub fn quadratic_value(q: &QuadraticTerms, x: &[f64]) -> f64 {
    let mut acc = q.constant();
    for &(v, c) in q.linear() {
        acc += c * x[v.index()];
    }
    for &(r, c, h) in q.quadratic() {
        acc += (h * x[r.index()]) * x[c.index()];
    }
    acc
}

/// Add the gradient of the quadratic at `x` into `grad`, scaled by `scale`.
pub fn quadratic_gradient_add(q: &QuadraticTerms, x: &[f64], scale: f64, grad: &mut [f64]) {
    for &(v, c) in q.linear() {
        grad[v.index()] += scale * c;
    }
    let normal_diagonals = q.diagonal_coefficient_magnitude_bounds().is_none_or(|(min, max)| {
        (min * scale.abs()).is_normal() && (2.0 * (max * scale.abs())).is_finite()
    });
    if !normal_diagonals {
        quadratic_gradient_extreme_add(q, x, scale, grad);
        return;
    }
    for &(r, c, h) in q.quadratic() {
        if r == c {
            grad[r.index()] += ((scale * h) * 2.0) * x[r.index()];
        } else {
            grad[r.index()] += quadratic_cross_gradient(h, x[c.index()], scale);
            grad[c.index()] += quadratic_cross_gradient(h, x[r.index()], scale);
        }
    }
}

#[cold]
fn quadratic_gradient_extreme_add(q: &QuadraticTerms, x: &[f64], scale: f64, grad: &mut [f64]) {
    for &(r, c, h) in q.quadratic() {
        if r == c {
            grad[r.index()] += quadratic_diagonal_gradient(h, x[r.index()], scale);
        } else {
            grad[r.index()] += quadratic_cross_gradient(h, x[c.index()], scale);
            grad[c.index()] += quadratic_cross_gradient(h, x[r.index()], scale);
        }
    }
}

/// Compute the weighted diagonal second derivative `2 * coefficient * scale`.
/// Place the factor of two before the product whenever that doubling is finite.
#[inline]
pub fn quadratic_diagonal_hessian(coefficient: f64, scale: f64) -> f64 {
    let twice_scale = 2.0 * scale;
    if twice_scale.is_finite() { coefficient * twice_scale } else { (coefficient * scale) * 2.0 }
}

/// Compute `2 * coefficient * scale * x` without losing the factor of two
/// to an intermediate underflow or overflow.
#[inline]
pub fn quadratic_diagonal_gradient(coefficient: f64, x: f64, scale: f64) -> f64 {
    let weighted = quadratic_diagonal_hessian(coefficient, scale);
    if weighted.is_normal() {
        weighted * x
    } else {
        gradient_product_extreme(coefficient, x, scale, 1)
    }
}

#[inline]
fn quadratic_cross_gradient(coefficient: f64, x: f64, scale: f64) -> f64 {
    if scale.to_bits() == 1.0_f64.to_bits() {
        return coefficient * x;
    }

    let weighted = scale * coefficient;
    if weighted.is_normal() {
        weighted * x
    } else {
        gradient_product_extreme(coefficient, x, scale, 0)
    }
}

/// Multiply normalized factors, with an optional binary exponent increment for
/// diagonal differentiation.
#[cold]
fn gradient_product_extreme(coefficient: f64, x: f64, scale: f64, extra_exponent: i32) -> f64 {
    if !coefficient.is_finite() || !x.is_finite() || !scale.is_finite() {
        let value = coefficient * scale * x;
        return if extra_exponent == 0 { value } else { value * 2.0 };
    }

    let sign = (coefficient.to_bits() ^ x.to_bits() ^ scale.to_bits()) & (1_u64 << 63);
    if coefficient == 0.0 || x == 0.0 || scale == 0.0 {
        return f64::from_bits(sign);
    }

    let (a, ea) = normalized_factor(coefficient);
    let (b, eb) = normalized_factor(scale);
    let (c, ec) = normalized_factor(x);
    let (mantissa, em) = normalized_factor((a * b) * c);
    let exponent = ea + eb + ec + em + extra_exponent;
    let magnitude = if exponent > 1023 {
        f64::INFINITY
    } else if exponent < -1075 {
        0.0
    } else if exponent < -1022 {
        let factor = power_of_two(exponent + 1074);
        (mantissa * factor) * f64::from_bits(1)
    } else {
        mantissa * power_of_two(exponent)
    };

    f64::from_bits(magnitude.to_bits() | sign)
}

fn normalized_factor(value: f64) -> (f64, i32) {
    let bits = value.to_bits() & !(1_u64 << 63);
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    if exponent == 0 {
        let (mantissa, exponent) = normalized_factor(value.abs() * 18_014_398_509_481_984.0);
        (mantissa, exponent - 54)
    } else {
        (f64::from_bits((bits & ((1_u64 << 52) - 1)) | (1023_u64 << 52)), exponent - 1023)
    }
}

fn power_of_two(exponent: i32) -> f64 {
    debug_assert!((-1022..=1023).contains(&exponent));
    f64::from_bits(u64::try_from(exponent + 1023).expect("normal exponent") << 52)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximo_expr::{ExprArenaCell, ExprNode, QuadraticBuilder, VarId};

    fn exact_power(exponent: i32) -> f64 {
        if exponent > 1023 {
            f64::INFINITY
        } else if exponent < -1074 {
            0.0
        } else if exponent < -1022 {
            f64::from_bits(1_u64 << (exponent + 1074))
        } else {
            power_of_two(exponent)
        }
    }

    #[test]
    fn quadratic_derivatives_preserve_range_across_power_of_two_factors() {
        let exponents = [-1074, -1073, -1022, -512, -1, 0, 1, 512, 1023];
        for ec in exponents {
            for es in exponents {
                let coefficient = exact_power(ec);
                let scale = exact_power(es);
                assert_eq!(
                    quadratic_diagonal_hessian(coefficient, scale).to_bits(),
                    exact_power(ec + es + 1).to_bits(),
                    "coefficient=2^{ec}, scale=2^{es}"
                );

                for ex in exponents {
                    for sign in [-1.0, 1.0] {
                        let actual =
                            quadratic_diagonal_gradient(coefficient, sign * exact_power(ex), scale);
                        assert_eq!(
                            actual.to_bits(),
                            (sign * exact_power(ec + es + ex + 1)).to_bits(),
                            "coefficient=2^{ec}, scale=2^{es}, x=2^{ex}, sign={sign}"
                        );
                        assert_eq!(
                            quadratic_cross_gradient(coefficient, sign * exact_power(ex), scale)
                                .to_bits(),
                            (sign * exact_power(ec + es + ex)).to_bits(),
                            "cross coefficient=2^{ec}, scale=2^{es}, x=2^{ex}, sign={sign}"
                        );
                    }
                }
            }
        }
        for coefficient in
            [f64::from_bits(1), f64::from_bits(3), f64::from_bits(f64::MIN_POSITIVE.to_bits() + 1)]
        {
            assert_eq!(
                quadratic_diagonal_hessian(coefficient, 0.5).to_bits(),
                coefficient.to_bits()
            );
            assert_eq!(
                quadratic_diagonal_gradient(coefficient, 0.5, 1.0).to_bits(),
                coefficient.to_bits()
            );
        }

        assert_eq!(quadratic_diagonal_gradient(1e308, 0.0, f64::MAX), 0.0);
        assert_eq!(quadratic_diagonal_hessian(1e308, 0.25), 5e307);
    }

    #[test]
    fn gradient_scatter_handles_normal_and_extreme_diagonals() {
        for coefficients in [[1.0, 2.0], [f64::from_bits(1), 1e308]] {
            let q = QuadraticTerms::from_coefficients(
                vec![
                    (VarId(0), VarId(0), coefficients[0]),
                    (VarId(1), VarId(1), coefficients[1]),
                    (VarId(1), VarId(0), 0.25),
                ],
                vec![(VarId(0), 0.5)],
                0.0,
            )
            .unwrap();
            for scale in [0.0, 0.5, 1.0, -1.0] {
                let point = [0.5, 0.0];
                let mut gradient = [0.0; 2];
                quadratic_gradient_add(&q, &point, scale, &mut gradient);
                assert_eq!(
                    gradient[0],
                    scale * 0.5 + quadratic_diagonal_gradient(coefficients[0], 0.5, scale)
                );
                assert_eq!(gradient[1], scale * 0.25 * 0.5);
            }
        }

        let tiny = f64::from_bits(1);
        let q = QuadraticTerms::from_coefficients(vec![(VarId(0), VarId(0), tiny)], vec![], 0.0)
            .unwrap();
        let mut gradient = [0.0];
        quadratic_gradient_add(&q, &[0.5], 1.0, &mut gradient);
        assert_eq!(gradient, [tiny]);
    }

    #[test]
    fn weighted_cross_gradients_preserve_range_in_both_scatter_paths() {
        let tiny = f64::from_bits(1);
        for (coefficient, scale, point, expected) in [
            (tiny, 0.5, 2.0, tiny),
            (3.0 * tiny, 0.5, 2.0, 3.0 * tiny),
            (tiny, -0.5, 2.0, -tiny),
            (1e308, 2.0, 1e-308, 2.0),
            (1e308, -2.0, 1e-308, -2.0),
            (1e308, 2.0, 0.0, 0.0),
        ] {
            for with_extreme_diagonal in [false, true] {
                let mut coefficients = vec![(VarId(1), VarId(0), coefficient)];
                if with_extreme_diagonal {
                    coefficients.push((VarId(2), VarId(2), tiny));
                }
                let q = QuadraticTerms::from_coefficients(coefficients, vec![], 0.0).unwrap();
                let mut gradient = [0.0; 3];
                quadratic_gradient_add(&q, &[point, point, 0.0], scale, &mut gradient);
                for actual in &gradient[..2] {
                    if expected.abs() <= tiny {
                        assert_eq!(*actual, expected);
                    } else {
                        assert!((actual - expected).abs() <= 4.0 * f64::EPSILON * expected.abs());
                    }
                }
                assert_eq!(gradient[2], 0.0);
            }
        }
    }

    #[test]
    fn compact_classification_shares_coefficients_and_keeps_reclassified_snapshots() {
        let arena = ExprArenaCell::new(ExprArena::new());
        let mut builder = QuadraticBuilder::new(&arena);
        for i in 0..4096 {
            builder.add_monomial(1.0, VarId(i), VarId(i)).unwrap();
        }

        let expression = builder.build();
        let root = expression.id();
        let snapshot = arena.borrow();
        let ExprNode::Quadratic(stored) = snapshot.get(root) else { panic!("compact quadratic") };
        let slot = FunctionSlot::classify(&snapshot, root);
        let refreshed = slot.reclassify(&snapshot, root);
        for kind in [&slot.kind, &refreshed.kind] {
            let SlotKind::Quadratic(terms) = kind else { panic!("quadratic slot") };
            assert!(Arc::ptr_eq(stored, terms));
        }
    }
}
