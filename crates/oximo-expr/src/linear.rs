use rustc_hash::{FxBuildHasher, FxHashMap};
use smallvec::smallvec;

use crate::arena::{ExprArena, ExprId, ExprNode, VarId};

/// Coefficients of a linear expression: `sum(coeff * var) + constant`.
#[derive(Clone, Debug, Default)]
pub struct LinearTerms {
    pub coeffs: Vec<(VarId, f64)>,
    pub constant: f64,
}

/// Accumulator that merges duplicate `(VarId, coeff)` terms while
/// preserving the order each variable is first seen.
struct CoeffAccum {
    coeffs: Vec<(VarId, f64)>,
    slot: FxHashMap<VarId, usize>,
}

impl CoeffAccum {
    fn with_capacity(n: usize) -> Self {
        Self {
            coeffs: Vec::with_capacity(n),
            slot: FxHashMap::with_capacity_and_hasher(n, FxBuildHasher),
        }
    }

    /// Add `c` to `v`'s running coefficient, appending `v` the first time it is
    /// seen.
    fn add(&mut self, v: VarId, c: f64) {
        if let Some(&i) = self.slot.get(&v) {
            self.coeffs[i].1 += c;
        } else {
            self.slot.insert(v, self.coeffs.len());
            self.coeffs.push((v, c));
        }
    }

    fn extend(&mut self, terms: impl IntoIterator<Item = (VarId, f64)>) {
        for (v, c) in terms {
            self.add(v, c);
        }
    }

    fn into_coeffs(self) -> Vec<(VarId, f64)> {
        self.coeffs
    }
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
            let mut acc = CoeffAccum::with_capacity(children.len() * 4);
            let mut constant = 0.0;
            for child in children {
                let t = as_linear(arena, child, resolve_params)?;
                acc.extend(t.coeffs);
                constant += t.constant;
            }
            Some(LinearTerms { coeffs: acc.into_coeffs(), constant })
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
        let mut acc = CoeffAccum::with_capacity(lt.coeffs.len() + rt.coeffs.len());
        acc.extend(lt.coeffs);
        acc.extend(rt.coeffs);
        return push_linear(
            arena,
            LinearTerms { coeffs: acc.into_coeffs(), constant: lt.constant + rt.constant },
        );
    }
    arena.push(ExprNode::Add(smallvec![lhs, rhs]))
}

/// Build a flat n-ary sum of `ids` as a single `Add` node.
/// `as_linear`/`split_linear` collapse the resulting `Add`
/// in one pass at extraction, so the linear fast-path is preserved.
///
/// # Panics
/// Panics if `ids` is empty (callers supply at least one term).
pub(crate) fn add_n(arena: &mut ExprArena, ids: &[ExprId]) -> ExprId {
    match ids {
        [] => panic!("add_n on an empty term list"),
        [one] => *one,
        _ => arena.push(ExprNode::Add(ids.iter().copied().collect())),
    }
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
    let mut lin = CoeffAccum::with_capacity(0);
    let mut constant = 0.0;
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
                    lin.extend(t.coeffs);
                    constant += t.constant;
                } else {
                    residual.push(SignedExpr { id: cur, neg: sign < 0.0 });
                }
            }
        }
    }
    let mut coeffs = lin.into_coeffs();
    coeffs.retain(|(_, c)| *c != 0.0);
    (LinearTerms { coeffs, constant }, residual)
}

/// Render the first nonlinear summand of `id` as a short infix string, resolving
/// each [`VarId`] to a display name via `resolve`. Returns `None` when `id` is
/// fully affine (no nonlinear residual).
pub fn describe_nonlinear_term(
    arena: &ExprArena,
    id: ExprId,
    resolve: &impl Fn(VarId) -> String,
) -> Option<String> {
    let (_, residual) = split_linear(arena, id);
    residual.first().map(|s| {
        if s.neg {
            format!("-{}", render_node(arena, s.id, resolve, PREC_UNARY))
        } else {
            render_node(arena, s.id, resolve, PREC_ADD)
        }
    })
}

// Precedence levels for parenthesizing `render_node` output.
const PREC_ADD: u8 = 1;
const PREC_MUL: u8 = 2;
const PREC_UNARY: u8 = 3;

/// Render an arena node as an infix string. `parent_prec` is the precedence of
/// the surrounding context.
fn render_node(
    arena: &ExprArena,
    id: ExprId,
    resolve: &impl Fn(VarId) -> String,
    parent_prec: u8,
) -> String {
    let (text, prec) = match arena.get(id) {
        ExprNode::Const(c) => (fmt_num(*c), PREC_UNARY),
        ExprNode::Var(v) => (resolve(*v), PREC_UNARY),
        ExprNode::Param(p) => (fmt_num(arena.param_value(*p)), PREC_UNARY),
        ExprNode::Neg(x) => {
            (format!("-{}", render_node(arena, *x, resolve, PREC_UNARY)), PREC_UNARY)
        }
        ExprNode::Add(children) => {
            let parts: Vec<String> =
                children.iter().map(|c| render_node(arena, *c, resolve, PREC_ADD)).collect();
            (parts.join(" + "), PREC_ADD)
        }
        ExprNode::Mul(children) => {
            let parts: Vec<String> =
                children.iter().map(|c| render_node(arena, *c, resolve, PREC_MUL)).collect();
            (parts.join(" * "), PREC_MUL)
        }
        ExprNode::Pow(b, e) => {
            let base = render_node(arena, *b, resolve, PREC_UNARY);
            let exp = render_node(arena, *e, resolve, PREC_UNARY);
            (format!("{base}^{exp}"), PREC_UNARY)
        }
        ExprNode::Div(num, den) => {
            let n = render_node(arena, *num, resolve, PREC_MUL);
            let d = render_node(arena, *den, resolve, PREC_MUL);
            (format!("{n} / {d}"), PREC_MUL)
        }
        ExprNode::Sin(x) => (fmt_call("sin", arena, *x, resolve), PREC_UNARY),
        ExprNode::Cos(x) => (fmt_call("cos", arena, *x, resolve), PREC_UNARY),
        ExprNode::Exp(x) => (fmt_call("exp", arena, *x, resolve), PREC_UNARY),
        ExprNode::Log(x) => (fmt_call("log", arena, *x, resolve), PREC_UNARY),
        ExprNode::Abs(x) => (fmt_call("abs", arena, *x, resolve), PREC_UNARY),
        ExprNode::Linear { coeffs, constant } => {
            let mut parts: Vec<String> = coeffs
                .iter()
                .map(|(v, c)| {
                    if (*c - 1.0).abs() < f64::EPSILON {
                        resolve(*v)
                    } else {
                        format!("{} * {}", fmt_num(*c), resolve(*v))
                    }
                })
                .collect();
            if *constant != 0.0 || parts.is_empty() {
                parts.push(fmt_num(*constant));
            }
            (parts.join(" + "), if parts.len() > 1 { PREC_ADD } else { PREC_MUL })
        }
    };
    if prec < parent_prec { format!("({text})") } else { text }
}

/// Format a call-like node `name(arg)`.
fn fmt_call(
    name: &str,
    arena: &ExprArena,
    arg: ExprId,
    resolve: &impl Fn(VarId) -> String,
) -> String {
    format!("{name}({})", render_node(arena, arg, resolve, PREC_ADD))
}

/// Render an `f64` compactly, used inside term descriptions.
fn fmt_num(v: f64) -> String {
    format!("{v}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::{ExprArena, ExprNode, VarId};

    #[test]
    fn param_times_var_stays_symbolic_until_extracted() {
        // Build `price * x` through the operator helper. The parameter must NOT
        // be folded into a Linear node at build time (so it stays re-bindable)
        let mut arena = ExprArena::new();
        let pid = arena.new_param(3.0);
        let price = arena.param(pid);
        let xnode = arena.push(ExprNode::Var(VarId(0)));
        let prod = mul_into(&mut arena, price, xnode);
        assert!(matches!(arena.get(prod), ExprNode::Mul(_)));

        let terms = extract_linear(&arena, prod).expect("linear");
        assert_eq!(terms.coeffs, vec![(VarId(0), 3.0)]);
        assert!(terms.constant.abs() < f64::EPSILON);
    }

    #[test]
    fn rebinding_param_updates_extracted_coeff() {
        let mut arena = ExprArena::new();
        let pid = arena.new_param(3.0);
        let price = arena.param(pid);
        let xnode = arena.push(ExprNode::Var(VarId(0)));
        let prod = mul_into(&mut arena, price, xnode);

        arena.set_param_value(pid, 10.0);
        let terms = extract_linear(&arena, prod).expect("linear");
        assert_eq!(terms.coeffs, vec![(VarId(0), 10.0)]);
    }

    #[test]
    fn param_plus_var_resolves_constant() {
        let mut arena = ExprArena::new();
        let pid = arena.new_param(5.0);
        let price = arena.param(pid);
        let xnode = arena.push(ExprNode::Var(VarId(0)));
        let sum = add_into(&mut arena, price, xnode);
        let terms = extract_linear(&arena, sum).expect("linear");
        assert_eq!(terms.coeffs, vec![(VarId(0), 1.0)]);
        assert!((terms.constant - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn add_extraction_is_first_seen_ordered_and_merges() {
        // `z + x + y + x`: coefficients come out in first-seen order [z, x, y]
        // and the repeated `x` is merged to coeff 2.
        let mut arena = ExprArena::new();
        let z = arena.push(ExprNode::Var(VarId(2)));
        let x = arena.push(ExprNode::Var(VarId(0)));
        let y = arena.push(ExprNode::Var(VarId(1)));
        let sum = arena.push(ExprNode::Add(smallvec::smallvec![z, x, y, x]));

        let terms = extract_linear(&arena, sum).expect("linear");
        assert_eq!(terms.coeffs, vec![(VarId(2), 1.0), (VarId(0), 2.0), (VarId(1), 1.0)]);
        assert!(terms.constant.abs() < f64::EPSILON);
        assert_eq!(extract_linear(&arena, sum).unwrap().coeffs, terms.coeffs);
    }

    #[test]
    fn wide_sum_merges_repeated_vars_in_order() {
        let mut arena = ExprArena::new();
        let n = 50u32;
        let mut ids = Vec::new();
        for _ in 0..3 {
            for v in 0..n {
                ids.push(arena.push(ExprNode::Var(VarId(v))));
            }
        }
        let sum = arena.push(ExprNode::Add(ids.into_iter().collect()));
        let terms = extract_linear(&arena, sum).expect("linear");
        let expected: Vec<(VarId, f64)> = (0..n).map(|v| (VarId(v), 3.0)).collect();
        assert_eq!(terms.coeffs, expected);
    }

    fn names(v: VarId) -> String {
        match v.0 {
            0 => "x".to_string(),
            1 => "y".to_string(),
            n => format!("v{n}"),
        }
    }

    #[test]
    fn describe_renders_the_first_nonlinear_summand() {
        let mut arena = ExprArena::new();
        let x = arena.push(ExprNode::Var(VarId(0)));
        let y = arena.push(ExprNode::Var(VarId(1)));

        let prod = arena.push(ExprNode::Mul(smallvec::smallvec![x, y]));
        assert_eq!(describe_nonlinear_term(&arena, prod, &names).as_deref(), Some("x * y"));

        let two = arena.constant(2.0);
        let pow = arena.push(ExprNode::Pow(x, two));
        assert_eq!(describe_nonlinear_term(&arena, pow, &names).as_deref(), Some("x^2"));

        let s = arena.push(ExprNode::Sin(x));
        assert_eq!(describe_nonlinear_term(&arena, s, &names).as_deref(), Some("sin(x)"));

        let div = arena.push(ExprNode::Div(x, y));
        assert_eq!(describe_nonlinear_term(&arena, div, &names).as_deref(), Some("x / y"));
    }

    #[test]
    fn describe_isolates_the_nonlinear_part_of_a_mixed_expression() {
        let mut arena = ExprArena::new();
        let x = arena.push(ExprNode::Var(VarId(0)));
        let y = arena.push(ExprNode::Var(VarId(1)));
        let z = arena.push(ExprNode::Var(VarId(2)));
        let two = arena.constant(2.0);
        let two_z = arena.push(ExprNode::Mul(smallvec::smallvec![two, z]));
        let prod = arena.push(ExprNode::Mul(smallvec::smallvec![x, y]));
        let sum = arena.push(ExprNode::Add(smallvec::smallvec![two_z, prod]));
        assert_eq!(describe_nonlinear_term(&arena, sum, &names).as_deref(), Some("x * y"));
    }

    #[test]
    fn describe_returns_none_for_affine() {
        let mut arena = ExprArena::new();
        let x = arena.push(ExprNode::Var(VarId(0)));
        let three = arena.constant(3.0);
        let sum = arena.push(ExprNode::Add(smallvec::smallvec![x, three]));
        assert_eq!(describe_nonlinear_term(&arena, sum, &names), None);
    }

    #[test]
    fn describe_falls_back_to_index_for_unknown_var() {
        let mut arena = ExprArena::new();
        let a = arena.push(ExprNode::Var(VarId(7)));
        let b = arena.push(ExprNode::Var(VarId(8)));
        let prod = arena.push(ExprNode::Mul(smallvec::smallvec![a, b]));
        assert_eq!(describe_nonlinear_term(&arena, prod, &names).as_deref(), Some("v7 * v8"));
    }
}
