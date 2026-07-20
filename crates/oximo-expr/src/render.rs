//! Shared infix rendering of arena expressions.
//! Used by the public display adapters in `oximo-core` and by
//! the nonlinear-term error messages ([`describe_nonlinear_term`](crate::describe_nonlinear_term)).

use crate::arena::{ExprArena, ExprId, ExprNode, VarId};
use crate::linear::{LinearTerms, split_linear};

// Precedence levels for parenthesizing `render_node` output.
pub(crate) const PREC_ADD: u8 = 1;
pub(crate) const PREC_MUL: u8 = 2;
pub(crate) const PREC_UNARY: u8 = 3;

/// One additive summand composed by the sign flag and the unsigned rendering.
type Part = (bool, String);

/// Render `id` as a canonical infix string, resolving each [`VarId`] to a
/// display name via `resolve`.
/// The linear part first, then any nonlinear residual summands.
/// Signs are folded into the joins, and a value-less expression
/// is rendered as `0`.
///
/// Parameters render as their current arena value.
pub fn render_expr(arena: &ExprArena, id: ExprId, resolve: &impl Fn(VarId) -> String) -> String {
    let (lin, residual) = split_linear(arena, id);
    let mut parts = linear_parts(&lin, resolve);
    for s in &residual {
        let prec = if s.neg { PREC_UNARY } else { PREC_ADD };
        parts.push((s.neg, render_node(arena, s.id, resolve, prec)));
    }
    join_parts(&parts)
}

/// Render a [`LinearTerms`] snapshot by:
/// - zero coefficients skipped,
/// - unit magnitudes omitted,
/// - multiplication implicit,
/// - the constant last.
///
/// An empty expression renders as `0`.
pub fn render_linear_terms(t: &LinearTerms, resolve: &impl Fn(VarId) -> String) -> String {
    join_parts(&linear_parts(t, resolve))
}

/// Split a linear expression into signed summands, ready for [`join_parts`].
fn linear_parts(t: &LinearTerms, resolve: &impl Fn(VarId) -> String) -> Vec<Part> {
    let mut parts = Vec::with_capacity(t.coeffs.len() + 1);
    for (v, c) in &t.coeffs {
        if *c == 0.0 {
            continue;
        }
        let mag = c.abs();
        let text = if (mag - 1.0).abs() < f64::EPSILON {
            resolve(*v)
        } else {
            format!("{} {}", fmt_num(mag), resolve(*v))
        };
        parts.push((*c < 0.0, text));
    }
    if t.constant != 0.0 {
        parts.push((t.constant < 0.0, fmt_num(t.constant.abs())));
    }
    parts
}

/// Join signed summands sign-aware.
fn join_parts(parts: &[Part]) -> String {
    let Some(((first_neg, first), rest)) = parts.split_first() else {
        return "0".to_string();
    };
    let mut out = String::new();
    if *first_neg {
        out.push('-');
    }
    out.push_str(first);
    for (neg, text) in rest {
        out.push_str(if *neg { " - " } else { " + " });
        out.push_str(text);
    }
    out
}

/// Render an arena node as an infix string. `parent_prec` is the precedence of
/// the surrounding context.
pub(crate) fn render_node(
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
            let mut parts: Vec<Part> = Vec::with_capacity(children.len());
            for c in children.iter().copied() {
                match arena.get(c) {
                    ExprNode::Neg(inner) => {
                        parts.push((true, render_node(arena, *inner, resolve, PREC_UNARY)));
                    }
                    ExprNode::Const(v) if *v < 0.0 => parts.push((true, fmt_num(-v))),
                    ExprNode::Linear { coeffs, constant } => parts.extend(linear_parts(
                        &LinearTerms { coeffs: coeffs.clone(), constant: *constant },
                        resolve,
                    )),
                    _ => parts.push((false, render_node(arena, c, resolve, PREC_ADD))),
                }
            }
            (join_parts(&parts), PREC_ADD)
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
            let parts =
                linear_parts(&LinearTerms { coeffs: coeffs.clone(), constant: *constant }, resolve);
            // A multi-term or negative-leading sum needs parens inside a product.
            let prec = match parts.as_slice() {
                [(false, _)] => PREC_MUL,
                [] => PREC_UNARY,
                _ => PREC_ADD,
            };
            (join_parts(&parts), prec)
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

/// Render an `f64` compactly (shortest round-trip).
pub(crate) fn fmt_num(v: f64) -> String {
    if v == 0.0 { "0".to_string() } else { format!("{v}") }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::{ExprArena, ExprNode, VarId};

    fn names(v: VarId) -> String {
        match v.0 {
            0 => "x".to_string(),
            1 => "y".to_string(),
            2 => "z".to_string(),
            n => format!("v{n}"),
        }
    }

    fn lt(coeffs: Vec<(u32, f64)>, constant: f64) -> LinearTerms {
        LinearTerms { coeffs: coeffs.into_iter().map(|(v, c)| (VarId(v), c)).collect(), constant }
    }

    #[test]
    fn linear_terms_are_sign_aware() {
        let t = lt(vec![(0, 1.0), (1, 2.0), (2, -3.0)], 0.0);
        assert_eq!(render_linear_terms(&t, &names), "x + 2 y - 3 z");
    }

    #[test]
    fn leading_negative_and_constant() {
        let t = lt(vec![(0, -1.0)], 2.0);
        assert_eq!(render_linear_terms(&t, &names), "-x + 2");
        let t = lt(vec![(0, 1.0)], -2.5);
        assert_eq!(render_linear_terms(&t, &names), "x - 2.5");
    }

    #[test]
    fn zero_coeffs_skipped_and_empty_is_zero() {
        let t = lt(vec![(0, 0.0), (1, 1.0)], 0.0);
        assert_eq!(render_linear_terms(&t, &names), "y");
        assert_eq!(render_linear_terms(&lt(vec![], 0.0), &names), "0");
        assert_eq!(render_linear_terms(&lt(vec![], -0.0), &names), "0");
    }

    #[test]
    fn constant_only() {
        assert_eq!(render_linear_terms(&lt(vec![], 5.0), &names), "5");
        assert_eq!(render_linear_terms(&lt(vec![], -5.0), &names), "-5");
    }

    #[test]
    fn expr_linear_and_nonlinear_mix() {
        let mut arena = ExprArena::new();
        let x = arena.push(ExprNode::Var(VarId(0)));
        let y = arena.push(ExprNode::Var(VarId(1)));
        let z = arena.push(ExprNode::Var(VarId(2)));
        let two = arena.constant(2.0);
        let two_z = arena.push(ExprNode::Mul(smallvec::smallvec![two, z]));
        let prod = arena.push(ExprNode::Mul(smallvec::smallvec![x, y]));
        let sum = arena.push(ExprNode::Add(smallvec::smallvec![two_z, prod]));
        assert_eq!(render_expr(&arena, sum, &names), "2 z + x * y");
    }

    #[test]
    fn expr_negated_residual() {
        let mut arena = ExprArena::new();
        let x = arena.push(ExprNode::Var(VarId(0)));
        let s = arena.push(ExprNode::Sin(x));
        let neg = arena.push(ExprNode::Neg(s));
        assert_eq!(render_expr(&arena, neg, &names), "-sin(x)");

        let y = arena.push(ExprNode::Var(VarId(1)));
        let sum = arena.push(ExprNode::Add(smallvec::smallvec![y, neg]));
        assert_eq!(render_expr(&arena, sum, &names), "y - sin(x)");
    }

    #[test]
    fn expr_pure_linear_uses_split() {
        let mut arena = ExprArena::new();
        let e = arena.push(ExprNode::Linear {
            coeffs: vec![(VarId(0), 3.0), (VarId(1), -1.0)],
            constant: 1.5,
        });
        assert_eq!(render_expr(&arena, e, &names), "3 x - y + 1.5");
    }

    #[test]
    fn precedence_parenthesizes_sums_in_products() {
        let mut arena = ExprArena::new();
        let x = arena.push(ExprNode::Var(VarId(0)));
        let y = arena.push(ExprNode::Var(VarId(1)));
        let one = arena.constant(1.0);
        let sum = arena.push(ExprNode::Add(smallvec::smallvec![x, one]));
        let prod = arena.push(ExprNode::Mul(smallvec::smallvec![sum, y]));
        assert_eq!(render_expr(&arena, prod, &names), "(x + 1) * y");
    }

    #[test]
    fn nested_add_is_sign_aware() {
        let mut arena = ExprArena::new();
        let x = arena.push(ExprNode::Var(VarId(0)));
        let y = arena.push(ExprNode::Var(VarId(1)));
        let prod = arena.push(ExprNode::Mul(smallvec::smallvec![x, y]));
        let neg_z = arena.push(ExprNode::Linear { coeffs: vec![(VarId(2), -1.0)], constant: 0.0 });
        let sum = arena.push(ExprNode::Add(smallvec::smallvec![prod, neg_z]));
        let s = arena.push(ExprNode::Sin(sum));
        assert_eq!(render_expr(&arena, s, &names), "sin(x * y - z)");
    }

    #[test]
    fn linear_node_inside_product_parenthesizes_when_needed() {
        let mut arena = ExprArena::new();
        let y = arena.push(ExprNode::Var(VarId(1)));
        let two_x = arena.push(ExprNode::Linear { coeffs: vec![(VarId(0), 2.0)], constant: 0.0 });
        let prod = arena.push(ExprNode::Mul(smallvec::smallvec![two_x, y]));
        assert_eq!(render_expr(&arena, prod, &names), "2 x * y");

        let sum = arena.push(ExprNode::Linear { coeffs: vec![(VarId(0), 1.0)], constant: 1.0 });
        let prod2 = arena.push(ExprNode::Mul(smallvec::smallvec![sum, y]));
        assert_eq!(render_expr(&arena, prod2, &names), "(x + 1) * y");
    }

    #[test]
    fn negative_zero_constant_renders_as_zero() {
        assert_eq!(fmt_num(-0.0), "0");
        let mut arena = ExprArena::new();
        let c = arena.constant(-0.0);
        assert_eq!(render_expr(&arena, c, &names), "0");
    }
}
