use oximo_expr::{ExprArena, ExprId, LinearTerms, VarId, extract_linear, extract_quadratic};
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
/// by [`detect_soc`] (algebraic quadratic constraints) and
/// [`explicit_soc_form`] ([`SocConstraint`]s), so backends translate both
/// through a single shape.
#[derive(Clone, Debug)]
pub struct SocForm {
    pub terms: Vec<LinearTerms>,
    pub bound: LinearTerms,
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
pub fn detect_soc(arena: &ExprArena, vars: &[Variable], c: &Constraint) -> Option<SocForm> {
    let (sense, rhs) = c.as_single()?;
    if sense != Sense::Le {
        return None;
    }
    let q = extract_quadratic(arena, c.lhs)?;
    if !q.linear.is_empty() || q.constant - rhs != 0.0 {
        return None;
    }

    let mut positives: Vec<(VarId, f64)> = Vec::new();
    let mut negative: Option<(VarId, f64)> = None;
    for &(row, col, h) in &q.hessian {
        if row != col {
            return None;
        }
        let coef = h / 2.0;
        if coef > 0.0 {
            positives.push((row, coef));
        } else if coef < 0.0 {
            if negative.is_some() {
                return None;
            }
            negative = Some((row, -coef));
        }
    }
    let (t, n) = negative?;
    if positives.is_empty() || vars[t.index()].lb < 0.0 {
        return None;
    }

    let terms = positives
        .into_iter()
        .map(|(x, p)| LinearTerms { coeffs: vec![(x, (p / n).sqrt())], constant: 0.0 })
        .collect();
    let bound = LinearTerms { coeffs: vec![(t, 1.0)], constant: 0.0 };
    Some(SocForm { terms, bound })
}

/// The normalized [`SocForm`] view of an explicit [`SocConstraint`]. Members
/// are validated affine at registration, so this only returns `None` on a
/// corrupted model (e.g. an `ExprId` from a different arena).
pub fn explicit_soc_form(arena: &ExprArena, s: &SocConstraint) -> Option<SocForm> {
    let terms = s.terms.iter().map(|&e| extract_linear(arena, e)).collect::<Option<Vec<_>>>()?;
    let bound = extract_linear(arena, s.bound)?;
    Some(SocForm { terms, bound })
}
