use rustc_hash::{FxBuildHasher, FxHashMap};
use smallvec::smallvec;

use crate::arena::{ExprArena, ExprId, ExprNode, VarId};

/// Coefficients of a linear expression: `sum(coeff * var) + constant`.
#[derive(Clone, Debug, Default)]
pub struct LinearTerms {
    pub coeffs: Vec<(VarId, f64)>,
    pub constant: f64,
}

/// Try to interpret `id` as a linear expression. Returns `None` for any
/// nonlinear node (Mul of two non-constants, Pow, transcendentals, ...).
///
/// When `resolve_params` is set, a [`ExprNode::Param`] folds to its current
/// arena value and counts as a constant.
fn as_linear(arena: &ExprArena, id: ExprId, resolve_params: bool) -> Option<LinearTerms> {
    match arena.get(id) {
        ExprNode::Const(c) => Some(LinearTerms { coeffs: Vec::new(), constant: *c }),
        ExprNode::Param(p) if resolve_params => {
            Some(LinearTerms { coeffs: Vec::new(), constant: arena.param_value(*p) })
        }
        ExprNode::Var(v) => Some(LinearTerms { coeffs: vec![(*v, 1.0)], constant: 0.0 }),
        ExprNode::Linear { coeffs, constant } => {
            Some(LinearTerms { coeffs: coeffs.clone(), constant: *constant })
        }
        ExprNode::Neg(inner) => {
            let inner = *inner;
            as_linear(arena, inner, resolve_params).map(|mut t| {
                t.coeffs.iter_mut().for_each(|(_, c)| *c = -*c);
                t.constant = -t.constant;
                t
            })
        }
        ExprNode::Add(children) => {
            let children: smallvec::SmallVec<[ExprId; 4]> = children.iter().copied().collect();
            let mut acc = LinearTerms::default();
            let mut map: FxHashMap<VarId, f64> =
                FxHashMap::with_capacity_and_hasher(children.len() * 4, FxBuildHasher);
            for child in children {
                let t = as_linear(arena, child, resolve_params)?;
                for (v, c) in t.coeffs {
                    *map.entry(v).or_insert(0.0) += c;
                }
                acc.constant += t.constant;
            }
            acc.coeffs = map.into_iter().collect();
            Some(acc)
        }
        ExprNode::Mul(children) => {
            // Linear if and only if exactly one non-const child is linear and the rest are constants.
            let children: smallvec::SmallVec<[ExprId; 4]> = children.iter().copied().collect();
            let mut scalar = 1.0;
            let mut linear: Option<LinearTerms> = None;
            for child in children {
                match arena.get(child) {
                    ExprNode::Const(c) => scalar *= c,
                    ExprNode::Param(p) if resolve_params => scalar *= arena.param_value(*p),
                    _ if linear.is_none() => {
                        linear = Some(as_linear(arena, child, resolve_params)?);
                    }
                    _ => return None,
                }
            }
            Some(match linear {
                None => LinearTerms { coeffs: Vec::new(), constant: scalar },
                Some(mut t) => {
                    t.coeffs.iter_mut().for_each(|(_, c)| *c *= scalar);
                    t.constant *= scalar;
                    t
                }
            })
        }
        _ => None,
    }
}

/// Materialize a linear-terms struct into a fresh `Linear` node in the arena.
fn push_linear(arena: &mut ExprArena, mut t: LinearTerms) -> ExprId {
    t.coeffs.retain(|(_, c)| *c != 0.0);
    arena.push(ExprNode::Linear { coeffs: t.coeffs, constant: t.constant })
}

/// Build `lhs + rhs`, preserving the linear fast-path when both sides are
/// linear. Falls back to an n-ary `Add` node otherwise.
pub(crate) fn add_into(arena: &mut ExprArena, lhs: ExprId, rhs: ExprId) -> ExprId {
    if let (Some(lt), Some(rt)) = (as_linear(arena, lhs, false), as_linear(arena, rhs, false)) {
        let mut map: FxHashMap<VarId, f64> =
            FxHashMap::with_capacity_and_hasher(lt.coeffs.len() + rt.coeffs.len(), FxBuildHasher);
        for (v, c) in lt.coeffs.into_iter().chain(rt.coeffs) {
            *map.entry(v).or_insert(0.0) += c;
        }
        return push_linear(
            arena,
            LinearTerms { coeffs: map.into_iter().collect(), constant: lt.constant + rt.constant },
        );
    }
    arena.push(ExprNode::Add(smallvec![lhs, rhs]))
}

/// Build `lhs - rhs`. Same linear fast-path as `add_into`.
pub(crate) fn sub_into(arena: &mut ExprArena, lhs: ExprId, rhs: ExprId) -> ExprId {
    let neg = neg_into(arena, rhs);
    add_into(arena, lhs, neg)
}

/// Build `lhs * rhs`. If either side is constant and the other is linear, we
/// stay on the linear fast-path. Otherwise produce a generic n-ary `Mul`.
pub(crate) fn mul_into(arena: &mut ExprArena, lhs: ExprId, rhs: ExprId) -> ExprId {
    if let ExprNode::Const(c) = *arena.get(lhs) {
        if let Some(mut t) = as_linear(arena, rhs, false) {
            t.coeffs.iter_mut().for_each(|(_, co)| *co *= c);
            t.constant *= c;
            return push_linear(arena, t);
        }
    }
    if let ExprNode::Const(c) = *arena.get(rhs) {
        if let Some(mut t) = as_linear(arena, lhs, false) {
            t.coeffs.iter_mut().for_each(|(_, co)| *co *= c);
            t.constant *= c;
            return push_linear(arena, t);
        }
    }
    arena.push(ExprNode::Mul(smallvec![lhs, rhs]))
}

/// Build `num / den`. If `den` is a nonzero constant `c`, fold to `num * (1/c)`
/// so a constant-denominator division stays on the linear fast-path. Otherwise
/// produce a `Div` node (always nonlinear, even when the numerator is linear).
pub(crate) fn div_into(arena: &mut ExprArena, num: ExprId, den: ExprId) -> ExprId {
    if let ExprNode::Const(c) = *arena.get(den) {
        if c != 0.0 {
            if let Some(mut t) = as_linear(arena, num, false) {
                let inv = 1.0 / c;
                t.coeffs.iter_mut().for_each(|(_, co)| *co *= inv);
                t.constant *= inv;
                return push_linear(arena, t);
            }
            let inv = arena.push(ExprNode::Const(1.0 / c));
            return mul_into(arena, num, inv);
        }
    }
    arena.push(ExprNode::Div(num, den))
}

/// Build `-rhs`, preserving linearity.
pub(crate) fn neg_into(arena: &mut ExprArena, rhs: ExprId) -> ExprId {
    if let Some(mut t) = as_linear(arena, rhs, false) {
        t.coeffs.iter_mut().for_each(|(_, c)| *c = -*c);
        t.constant = -t.constant;
        return push_linear(arena, t);
    }
    arena.push(ExprNode::Neg(rhs))
}

/// Snapshot the linear terms of `id`, if any. Used by solver backends to
/// extract LP coefficients without walking the tree themselves.
///
/// Parameters are folded to their current arena values, so the returned
/// coefficients reflect the latest [`ExprArena::set_param_value`] binding.
///
/// [`ExprArena::set_param_value`]: crate::ExprArena::set_param_value
pub fn extract_linear(arena: &ExprArena, id: ExprId) -> Option<LinearTerms> {
    as_linear(arena, id, true)
}

/// A nonlinear residual summand: the existing arena node `id`, taken with a
/// leading negation when `neg` is set. Carrying the sign as a flag.
/// Lets [`split_linear`] run without a mutable arena.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SignedExpr {
    pub id: ExprId,
    pub neg: bool,
}

/// Split an expression into its linear part and a nonlinear residual. The
/// returned `(LinearTerms, Vec<SignedExpr>)` satisfies
///
/// ```text
/// value(id) == sum_i coef_i * var_i + constant + sum_j (-1)^neg_j value(id_j)
/// ```
///
/// where the residual is empty when the whole expression is linear and
/// otherwise lists the remaining nonlinear summands (each a pre-existing arena
/// node, optionally negated). `LinearTerms` may have empty `coeffs` and
/// `constant == 0.0` when the whole expression is purely nonlinear.
pub fn split_linear(arena: &ExprArena, id: ExprId) -> (LinearTerms, Vec<SignedExpr>) {
    if let Some(lt) = as_linear(arena, id, true) {
        return (lt, Vec::new());
    }
    let mut lin = LinearTerms::default();
    let mut residual: Vec<SignedExpr> = Vec::new();
    let mut sign_stack: smallvec::SmallVec<[(ExprId, f64); 8]> = smallvec![(id, 1.0)];
    while let Some((cur, sign)) = sign_stack.pop() {
        match arena.get(cur) {
            ExprNode::Add(children) => {
                for c in children.iter().copied() {
                    sign_stack.push((c, sign));
                }
            }
            ExprNode::Neg(inner) => sign_stack.push((*inner, -sign)),
            _ => {
                if let Some(mut t) = as_linear(arena, cur, true) {
                    if (sign - 1.0).abs() > 0.0 {
                        t.coeffs.iter_mut().for_each(|(_, c)| *c *= sign);
                        t.constant *= sign;
                    }
                    for (v, c) in t.coeffs {
                        if let Some((_, acc)) = lin.coeffs.iter_mut().find(|(vv, _)| *vv == v) {
                            *acc += c;
                        } else {
                            lin.coeffs.push((v, c));
                        }
                    }
                    lin.constant += t.constant;
                } else {
                    residual.push(SignedExpr { id: cur, neg: sign < 0.0 });
                }
            }
        }
    }
    lin.coeffs.retain(|(_, c)| *c != 0.0);
    (lin, residual)
}
