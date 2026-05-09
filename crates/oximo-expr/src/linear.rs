use rustc_hash::FxHashMap;
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
fn as_linear(arena: &ExprArena, id: ExprId) -> Option<LinearTerms> {
    match arena.get(id).clone() {
        ExprNode::Const(c) => Some(LinearTerms { coeffs: Vec::new(), constant: c }),
        ExprNode::Var(v) => Some(LinearTerms { coeffs: vec![(v, 1.0)], constant: 0.0 }),
        ExprNode::Linear { coeffs, constant } => Some(LinearTerms { coeffs, constant }),
        ExprNode::Neg(inner) => as_linear(arena, inner).map(|t| LinearTerms {
            coeffs: t.coeffs.into_iter().map(|(v, c)| (v, -c)).collect(),
            constant: -t.constant,
        }),
        ExprNode::Add(children) => {
            let mut acc = LinearTerms::default();
            let mut map: FxHashMap<VarId, f64> = FxHashMap::default();
            for child in children {
                let t = as_linear(arena, child)?;
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
            let mut scalar = 1.0;
            let mut linear: Option<LinearTerms> = None;
            for child in children {
                if let ExprNode::Const(c) = arena.get(child) {
                    scalar *= c;
                } else if linear.is_none() {
                    linear = Some(as_linear(arena, child)?);
                } else {
                    return None;
                }
            }
            Some(match linear {
                None => LinearTerms { coeffs: Vec::new(), constant: scalar },
                Some(t) => LinearTerms {
                    coeffs: t.coeffs.into_iter().map(|(v, c)| (v, c * scalar)).collect(),
                    constant: t.constant * scalar,
                },
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
    if let (Some(lt), Some(rt)) = (as_linear(arena, lhs), as_linear(arena, rhs)) {
        let mut map: FxHashMap<VarId, f64> = FxHashMap::default();
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
        if let Some(t) = as_linear(arena, rhs) {
            return push_linear(
                arena,
                LinearTerms {
                    coeffs: t.coeffs.into_iter().map(|(v, co)| (v, co * c)).collect(),
                    constant: t.constant * c,
                },
            );
        }
    }
    if let ExprNode::Const(c) = *arena.get(rhs) {
        if let Some(t) = as_linear(arena, lhs) {
            return push_linear(
                arena,
                LinearTerms {
                    coeffs: t.coeffs.into_iter().map(|(v, co)| (v, co * c)).collect(),
                    constant: t.constant * c,
                },
            );
        }
    }
    arena.push(ExprNode::Mul(smallvec![lhs, rhs]))
}

/// Build `-rhs`, preserving linearity.
pub(crate) fn neg_into(arena: &mut ExprArena, rhs: ExprId) -> ExprId {
    if let Some(t) = as_linear(arena, rhs) {
        return push_linear(
            arena,
            LinearTerms {
                coeffs: t.coeffs.into_iter().map(|(v, c)| (v, -c)).collect(),
                constant: -t.constant,
            },
        );
    }
    arena.push(ExprNode::Neg(rhs))
}

/// Snapshot the linear terms of `id`, if any. Used by solver backends to
/// extract LP coefficients without walking the tree themselves.
pub fn extract_linear(arena: &ExprArena, id: ExprId) -> Option<LinearTerms> {
    as_linear(arena, id)
}
