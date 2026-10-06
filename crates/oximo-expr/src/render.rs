//! Shared infix rendering of arena expressions.
//! Used by the public display adapters in `oximo-core` and by
//! the nonlinear-term error messages ([`describe_nonlinear_term`](crate::describe_nonlinear_term)).

use crate::arena::{ExprArena, ExprId, ExprNode, UnaryOp, VarId};
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
pub fn render_linear_terms(t: &LinearTerms<'_>, resolve: &impl Fn(VarId) -> String) -> String {
    join_parts(&linear_parts(t, resolve))
}

/// Split a linear expression into signed summands, ready for [`join_parts`].
fn linear_parts(t: &LinearTerms<'_>, resolve: &impl Fn(VarId) -> String) -> Vec<Part> {
    linear_parts_from_slice(&t.coeffs, t.constant, resolve)
}

fn linear_parts_from_slice(
    coeffs: &[(VarId, f64)],
    constant: f64,
    resolve: &impl Fn(VarId) -> String,
) -> Vec<Part> {
    let mut parts = Vec::with_capacity(coeffs.len() + 1);
    for (v, c) in coeffs {
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
    if constant != 0.0 {
        parts.push((constant < 0.0, fmt_num(constant.abs())));
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
    let mut out = String::new();
    let mut pending: RenderTasks<'_> = smallvec::smallvec![RenderTask::Node(id, parent_prec)];
    while let Some(task) = pending.pop() {
        let (id, parent_prec) = match task {
            RenderTask::Text(text) => {
                out.push_str(text);
                continue;
            }
            RenderTask::Add(children, first) => {
                render_add(arena, resolve, &mut out, &mut pending, children, first);
                continue;
            }
            RenderTask::Node(id, parent_prec) => (id, parent_prec),
        };
        let node = arena.get(id);
        if let ExprNode::Linear { coeffs, constant } = node {
            let parts = linear_parts_from_slice(coeffs, *constant, resolve);
            let prec = match parts.as_slice() {
                [(false, _)] => PREC_MUL,
                [] => PREC_UNARY,
                _ => PREC_ADD,
            };
            if prec < parent_prec {
                out.push('(');
            }
            out.push_str(&join_parts(&parts));
            if prec < parent_prec {
                out.push(')');
            }
            continue;
        }
        let prec = match node {
            ExprNode::Add(_) => PREC_ADD,
            ExprNode::Mul(_) | ExprNode::Div(_, _) => PREC_MUL,
            _ => PREC_UNARY,
        };
        if prec < parent_prec {
            out.push('(');
            pending.push(RenderTask::Text(")"));
        }
        match node {
            ExprNode::Const(c) => out.push_str(&fmt_num(*c)),
            ExprNode::Var(v) => out.push_str(&resolve(*v)),
            ExprNode::Param(p) => out.push_str(&fmt_num(arena.param_value(*p))),
            ExprNode::Add(children) => pending.push(RenderTask::Add(children, true)),
            ExprNode::Mul(children) => render_children(&mut pending, children, PREC_MUL, " * "),
            ExprNode::Unary(UnaryOp::Neg, child) => {
                out.push('-');
                pending.push(RenderTask::Node(*child, PREC_UNARY));
            }
            ExprNode::Pow(base, exponent) => {
                pending.push(RenderTask::Node(*exponent, PREC_UNARY));
                pending.push(RenderTask::Text("^"));
                pending.push(RenderTask::Node(*base, PREC_UNARY));
            }
            ExprNode::Div(num, den) => {
                pending.push(RenderTask::Node(*den, PREC_MUL));
                pending.push(RenderTask::Text(" / "));
                pending.push(RenderTask::Node(*num, PREC_MUL));
            }
            ExprNode::Unary(op, child) => {
                out.push_str(op.name());
                out.push('(');
                pending.push(RenderTask::Text(")"));
                pending.push(RenderTask::Node(*child, PREC_ADD));
            }
            ExprNode::Atan2(y, x) => {
                out.push_str("atan2(");
                pending.push(RenderTask::Text(")"));
                pending.push(RenderTask::Node(*x, PREC_ADD));
                pending.push(RenderTask::Text(", "));
                pending.push(RenderTask::Node(*y, PREC_ADD));
            }
            ExprNode::Min(children) | ExprNode::Max(children) => {
                out.push_str(if matches!(node, ExprNode::Min(_)) { "min(" } else { "max(" });
                pending.push(RenderTask::Text(")"));
                render_children(&mut pending, children, PREC_ADD, ", ");
            }
            ExprNode::Linear { .. } => unreachable!("handled above"),
        }
    }
    out
}

enum RenderTask<'a> {
    Node(ExprId, u8),
    Text(&'static str),
    Add(&'a [ExprId], bool),
}

type RenderTasks<'a> = smallvec::SmallVec<[RenderTask<'a>; 16]>;

fn render_add<'a>(
    arena: &ExprArena,
    resolve: &impl Fn(VarId) -> String,
    out: &mut String,
    pending: &mut RenderTasks<'a>,
    children: &'a [ExprId],
    mut first: bool,
) {
    let Some((&child, rest)) = children.split_first() else {
        if first {
            out.push('0');
        }
        return;
    };
    match arena.get(child) {
        ExprNode::Linear { coeffs, constant } => {
            for (negative, text) in linear_parts_from_slice(coeffs, *constant, resolve) {
                part_prefix(out, negative, first);
                out.push_str(&text);
                first = false;
            }
            pending.push(RenderTask::Add(rest, first));
        }
        ExprNode::Const(value) if *value < 0.0 => {
            part_prefix(out, true, first);
            out.push_str(&fmt_num(-value));
            pending.push(RenderTask::Add(rest, false));
        }
        ExprNode::Param(p) if arena.param_value(*p) < 0.0 => {
            part_prefix(out, true, first);
            out.push_str(&fmt_num(-arena.param_value(*p)));
            pending.push(RenderTask::Add(rest, false));
        }
        node => {
            let (negative, child, prec) = if let ExprNode::Unary(UnaryOp::Neg, inner) = node {
                (true, *inner, PREC_UNARY)
            } else {
                (false, child, PREC_ADD)
            };
            part_prefix(out, negative, first);
            pending.push(RenderTask::Add(rest, false));
            pending.push(RenderTask::Node(child, prec));
        }
    }
}

fn part_prefix(out: &mut String, negative: bool, first: bool) {
    if first {
        if negative {
            out.push('-');
        }
    } else {
        out.push_str(if negative { " - " } else { " + " });
    }
}

fn render_children<'a>(
    pending: &mut RenderTasks<'a>,
    children: &'a [ExprId],
    prec: u8,
    separator: &'static str,
) {
    for (i, &child) in children.iter().enumerate().rev() {
        pending.push(RenderTask::Node(child, prec));
        if i > 0 {
            pending.push(RenderTask::Text(separator));
        }
    }
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

    fn lt(coeffs: Vec<(u32, f64)>, constant: f64) -> LinearTerms<'static> {
        LinearTerms {
            coeffs: std::borrow::Cow::Owned(
                coeffs.into_iter().map(|(v, c)| (VarId(v), c)).collect(),
            ),
            constant,
        }
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
        let s = arena.push(ExprNode::Unary(UnaryOp::Sin, x));
        let neg = arena.push(ExprNode::Unary(UnaryOp::Neg, s));
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
        let s = arena.push(ExprNode::Unary(UnaryOp::Sin, sum));
        assert_eq!(render_expr(&arena, s, &names), "sin(x * y - z)");
    }

    #[test]
    fn negative_param_in_nonlinear_add_is_sign_aware() {
        let mut arena = ExprArena::new();
        let x = arena.push(ExprNode::Var(VarId(0)));
        let pid = arena.new_param(-3.0);
        let p = arena.param(pid);
        let sum = arena.push(ExprNode::Add(smallvec::smallvec![x, p]));
        let s = arena.push(ExprNode::Unary(UnaryOp::Sin, sum));
        assert_eq!(render_expr(&arena, s, &names), "sin(x - 3)");

        arena.set_param_value(pid, 3.0);
        assert_eq!(render_expr(&arena, s, &names), "sin(x + 3)");
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
