//! Per-function classification.

use oximo_expr::{
    ExprArena, ExprId, LinearTerms, QuadraticTerms, extract_linear, extract_quadratic,
};

use crate::sparsity::{hessian_pattern, variable_support};
use crate::tape::Tape;

/// One objective or constraint function, classified for derivative purposes.
#[derive(Clone, Debug)]
pub struct FunctionSlot {
    pub kind: SlotKind,
    /// Sorted, deduplicated indices of the variables this function touches.
    pub support: Vec<u32>,
    /// Exact structural lower-triangle Hessian pattern. Populated only for
    /// `Nonlinear` slots.
    pub hess_pairs: Vec<(u32, u32)>,
}

#[derive(Clone, Debug)]
pub enum SlotKind {
    /// Affine: constant gradient, zero Hessian.
    Linear(LinearTerms),
    /// Degree-2 polynomial: affine gradient `Qx + c`, constant Hessian.
    /// `QuadraticTerms.hessian` follows the `0.5 x'Qx` convention
    Quadratic(QuadraticTerms),
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
            kind: SlotKind::Linear(LinearTerms { coeffs: Vec::new(), constant: 0.0 }),
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
        Self::closed_form(arena, root).unwrap_or_else(|| {
            let tape = Tape::compile(arena, root);
            let support = variable_support(arena, root);
            let hess_pairs = hessian_pattern(arena, root);
            Self { kind: SlotKind::Nonlinear(tape), support, hess_pairs }
        })
    }

    /// The linear/quadratic closed-form classification, or `None` when the
    /// expression is nonlinear. Both `classify` and `reclassify` share this so
    /// the extraction-and-support logic lives in one place, the two differ only
    /// in their nonlinear fallback.
    fn closed_form(arena: &ExprArena, root: ExprId) -> Option<Self> {
        if let Some(linear) = extract_linear(arena, root) {
            let support = sorted_dedup(linear.coeffs.iter().map(|(v, _)| v.0));
            return Some(Self { kind: SlotKind::Linear(linear), support, hess_pairs: Vec::new() });
        }
        if let Some(quadratic) = extract_quadratic(arena, root) {
            let support = sorted_dedup(
                quadratic
                    .linear
                    .iter()
                    .map(|(v, _)| v.0)
                    .chain(quadratic.hessian.iter().flat_map(|&(r, c, _)| [r.0, c.0])),
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
pub fn linear_value(t: &LinearTerms, x: &[f64]) -> f64 {
    t.coeffs.iter().fold(t.constant, |acc, &(v, c)| acc + c * x[v.index()])
}

/// Add the (constant) gradient of `t` into `grad`, scaled by `scale`.
pub fn linear_gradient_add(t: &LinearTerms, scale: f64, grad: &mut [f64]) {
    for &(v, c) in &t.coeffs {
        grad[v.index()] += scale * c;
    }
}

/// Value of the quadratic at `x`: `constant + linear * x + 0.5 x' Q x`.
pub fn quadratic_value(q: &QuadraticTerms, x: &[f64]) -> f64 {
    let mut acc = q.constant;
    for &(v, c) in &q.linear {
        acc += c * x[v.index()];
    }
    for &(r, c, h) in &q.hessian {
        let term = h * x[r.index()] * x[c.index()];
        acc += if r == c { 0.5 * term } else { term };
    }
    acc
}

/// Add the gradient of the quadratic at `x` into `grad`, scaled by `scale`.
pub fn quadratic_gradient_add(q: &QuadraticTerms, x: &[f64], scale: f64, grad: &mut [f64]) {
    for &(v, c) in &q.linear {
        grad[v.index()] += scale * c;
    }
    for &(r, c, h) in &q.hessian {
        if r == c {
            grad[r.index()] += scale * h * x[r.index()];
        } else {
            grad[r.index()] += scale * h * x[c.index()];
            grad[c.index()] += scale * h * x[r.index()];
        }
    }
}
