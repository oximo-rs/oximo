use oximo_expr::{ExprArena, ExprId, LinearTerms, QuadraticTerms, VarId, extract_quadratic};
use smol_str::SmolStr;

use crate::constraint::{Constraint, Sense};
use crate::var::Variable;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SocConstraintId(pub u32);

impl SocConstraintId {
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

// TODO: Support rotated cones

/// An explicit second-order cone constraint `||terms||_2 <= bound`.
///
/// Every member of `terms` and the `bound` must be affine. This is validated
/// when the constraint is registered via [`crate::Model::add_soc_constraint`].
/// Rotated cones (`2uv >= ||w||^2`) are not supported yet.
#[derive(Clone, Debug)]
pub struct SocConstraint {
    pub name: SmolStr,
    pub terms: Vec<ExprId>,
    pub bound: ExprId,
    pub active: bool,
}

/// Normalized second-order cone data: `|| A x + a ||_2 <= b'x + beta`, one
/// [`LinearTerms`] per row of `A x + a` plus one for the bound side. Produced
/// by shared solver preparation for algebraic and explicit cones.
#[derive(Clone, Debug)]
pub struct SocForm {
    pub terms: Vec<LinearTerms<'static>>,
    pub bound: LinearTerms<'static>,
}

/// Extract and validate the diagonal quadratic form shared by SOC recognition
/// and model-kind inference.
fn soc_quadratic(vars: &[Variable], c: &Constraint, q: &QuadraticTerms) -> Option<(VarId, f64)> {
    let (sense, rhs) = c.as_single()?;
    if sense != Sense::Le {
        return None;
    }
    if !q.linear.is_empty() || q.constant - rhs != 0.0 {
        return None;
    }

    let mut positives = 0;
    let mut negative: Option<(VarId, f64)> = None;
    for &(row, col, h) in &q.hessian {
        if row != col {
            return None;
        }
        let coef = h / 2.0;
        if coef > 0.0 {
            positives += 1;
        } else if coef < 0.0 {
            if negative.is_some() {
                return None;
            }
            negative = Some((row, -coef));
        }
    }
    let (t, n) = negative?;
    if positives == 0 || vars[t.index()].lb < 0.0 {
        return None;
    }
    Some((t, n))
}

/// Whether an algebraic constraint has the supported detected-SOC shape.
///
/// This does not materialize a [`SocForm`]. Model-kind
/// inference only needs this predicate and can avoid allocating one
/// `LinearTerms` coefficient vector per cone member.
pub(crate) fn is_detected_soc(arena: &ExprArena, vars: &[Variable], c: &Constraint) -> bool {
    if !matches!(c.as_single(), Some((Sense::Le, _))) {
        return false;
    }
    extract_quadratic(arena, c.lhs).is_some_and(|q| soc_quadratic(vars, c, &q).is_some())
}

// TODO: Here we are deliberately conservative and purely structural

/// Recognize an algebraic quadratic constraint as second-order-cone shaped.
///
/// A constraint is recognized iff:
///
/// - it is single-sided `lhs <= rhs` (no ranges, `>=`, or equalities),
/// - `lhs - rhs` is a pure quadratic form: no linear terms, no constant,
/// - the Hessian is diagonal with exactly one negative entry `-n` (on the
///   bound variable `t`) and at least one positive entry `p_i`,
/// - `t` has lower bound `>= 0`.
///
/// That is `sum_i p_i x_i^2 <= n t^2` with `t >= 0`, equivalent to
/// `|| sqrt(p_i/n) x_i ||_2 <= t`. Cross-term (Cholesky-factorized) quadratic
/// forms are not detected, they classify as QCP instead.
///
/// Backend hook for recognizing a row that has already been decomposed.
/// `q` must describe `c.lhs` at the current parameter values.
#[doc(hidden)]
pub fn __detect_soc_from_quadratic(
    vars: &[Variable],
    c: &Constraint,
    q: &QuadraticTerms,
) -> Option<SocForm> {
    let (t, n) = soc_quadratic(vars, c, q)?;
    let terms = q
        .hessian
        .iter()
        .filter_map(|&(row, _, h)| {
            let coef = h / 2.0;
            (coef > 0.0).then(|| LinearTerms {
                coeffs: vec![(row, (coef / n).sqrt())].into(),
                constant: 0.0,
            })
        })
        .collect();
    let bound = LinearTerms { coeffs: vec![(t, 1.0)].into(), constant: 0.0 };
    Some(SocForm { terms, bound })
}
