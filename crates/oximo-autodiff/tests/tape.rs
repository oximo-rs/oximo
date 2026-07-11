//! Stable tests: tape compilation/evaluation against `oximo_expr::evaluate`,
//! slot classification, and sparsity patterns.
#![allow(clippy::unreadable_literal, clippy::cast_precision_loss, clippy::float_cmp)]

use oximo_autodiff::slot::{
    FunctionSlot, SlotKind, linear_gradient_add, linear_value, quadratic_gradient_add,
    quadratic_value,
};
use oximo_autodiff::sparsity::{
    hessian_lagrangian_structure, jacobian_structure, variable_support,
};
use oximo_autodiff::tape::Tape;
use oximo_expr::{ExprArena, ExprId, ExprNode, VarId, evaluate};

fn assert_close(got: f64, want: f64, tol: f64, what: &str) {
    let denom = want.abs().max(1.0);
    assert!(((got - want) / denom).abs() < tol, "{what}: got {got}, want {want}");
}

/// Tape and recursive evaluator must agree to within a couple of ULPs (the
/// tape may associate sums differently, e.g. a Linear node's constant is
/// added last instead of first).
fn check_matches_evaluate(arena: &ExprArena, root: ExprId, points: &[Vec<f64>]) {
    let tape = Tape::compile(arena, root);
    let params: Vec<f64> = (0..arena.num_params())
        .map(|i| arena.param_value(oximo_expr::ParamId(u32::try_from(i).unwrap())))
        .collect();
    let mut regs = vec![0.0; tape.n_regs()];
    for x in points {
        let want = evaluate(arena, root, &x.as_slice()).unwrap();
        let got = tape.value(x, &params, &[], &mut regs);
        if got.is_nan() && want.is_nan() {
            continue;
        }
        assert_close(got, want, 1e-14, &format!("tape vs evaluate at {x:?}"));
    }
}

fn points(n_vars: usize, n_points: usize) -> Vec<Vec<f64>> {
    // Small deterministic LCG; values in roughly [-2, 2], kept away from 0.
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
        let v = 4.0 * unit - 2.0;
        if v.abs() < 0.1 { v + 0.5 } else { v }
    };
    (0..n_points).map(|_| (0..n_vars).map(|_| next()).collect()).collect()
}

#[test]
fn every_node_kind_matches_evaluate() {
    let mut arena = ExprArena::new();
    let x0 = arena.var(VarId(0));
    let x1 = arena.var(VarId(1));
    let p = arena.new_param(0.7);
    let p0 = arena.param(p);
    let c2 = arena.constant(2.0);
    let c3 = arena.constant(3.0);

    let add = arena.push(ExprNode::Add([x0, x1, c2].into_iter().collect()));
    let mul = arena.push(ExprNode::Mul([x0, x1, p0].into_iter().collect()));
    let neg = arena.push(ExprNode::Neg(mul));
    let powc = arena.push(ExprNode::Pow(x0, c3));
    let pow = arena.push(ExprNode::Pow(add, x1)); // expression exponent
    let div = arena.push(ExprNode::Div(powc, x1));
    let sin = arena.push(ExprNode::Sin(x0));
    let cos = arena.push(ExprNode::Cos(x1));
    let exp = arena.push(ExprNode::Exp(sin));
    let log = arena.push(ExprNode::Log(exp));
    let abs = arena.push(ExprNode::Abs(neg));
    let lin = arena.linear(vec![(VarId(0), 2.0), (VarId(1), -0.5)], 4.0);
    let root =
        arena.push(ExprNode::Add([add, neg, pow, div, cos, log, abs, lin].into_iter().collect()));

    // `pow` with an expression exponent needs a positive base: shift points.
    let pts: Vec<Vec<f64>> = points(2, 20)
        .into_iter()
        .map(|p| p.into_iter().map(f64::abs).map(|v| v + 0.2).collect())
        .collect();
    check_matches_evaluate(&arena, root, &pts);
}

#[test]
fn shared_subexpressions_lower_once() {
    let mut arena = ExprArena::new();
    let x0 = arena.var(VarId(0));
    let sin = arena.push(ExprNode::Sin(x0));
    // sin(x0) used three times: as a DAG the tape must reuse the register.
    let mul = arena.push(ExprNode::Mul([sin, sin].into_iter().collect()));
    let root = arena.push(ExprNode::Add([mul, sin].into_iter().collect()));

    let tape = Tape::compile(&arena, root);
    // x0, sin, mul, add, sharing means 4 instructions
    assert_eq!(tape.n_regs(), 4);
    check_matches_evaluate(&arena, root, &points(1, 10));
}

#[test]
fn weighted_tape_matches_manual_sum() {
    let mut arena = ExprArena::new();
    let x0 = arena.var(VarId(0));
    let x1 = arena.var(VarId(1));
    let sin = arena.push(ExprNode::Sin(x0));
    let f0 = arena.push(ExprNode::Mul([sin, x1].into_iter().collect()));
    let f1 = arena.push(ExprNode::Exp(x1));
    let f2 = arena.push(ExprNode::Mul([sin, sin].into_iter().collect())); // shares sin

    let tape = Tape::compile_weighted(&arena, &[f0, f1, f2]);
    assert_eq!(tape.n_mults(), 3);
    let mut regs = vec![0.0; tape.n_regs()];
    let mults = [1.5, -2.0, 0.25];
    for x in points(2, 10) {
        let want: f64 = [f0, f1, f2]
            .iter()
            .zip(mults)
            .map(|(&f, m)| m * evaluate(&arena, f, &x.as_slice()).unwrap())
            .sum();
        let got = tape.value(&x, &[], &mults, &mut regs);
        assert_close(got, want, 1e-14, "weighted tape");
    }
}

#[test]
fn empty_weighted_tape_is_zero() {
    let arena = ExprArena::new();
    let tape = Tape::compile_weighted(&arena, &[]);
    let mut regs = vec![0.0; tape.n_regs()];
    assert_eq!(tape.value(&[], &[], &[], &mut regs), 0.0);
}

#[test]
fn classification_fast_paths() {
    let mut arena = ExprArena::new();
    let x0 = arena.var(VarId(0));
    let x1 = arena.var(VarId(1));

    let lin = arena.linear(vec![(VarId(0), 2.0), (VarId(1), 3.0)], 1.0);
    let slot = FunctionSlot::classify(&arena, lin);
    assert!(matches!(slot.kind, SlotKind::Linear(_)), "linear slot");
    assert_eq!(slot.support, vec![0, 1]);

    let sq = arena.push(ExprNode::Mul([x0, x0].into_iter().collect()));
    let cross = arena.push(ExprNode::Mul([x0, x1].into_iter().collect()));
    let quad = arena.push(ExprNode::Add([sq, cross, lin].into_iter().collect()));
    let slot = FunctionSlot::classify(&arena, quad);
    assert!(matches!(slot.kind, SlotKind::Quadratic(_)), "quadratic slot");
    assert_eq!(slot.support, vec![0, 1]);

    let sin = arena.push(ExprNode::Sin(x0));
    let slot = FunctionSlot::classify(&arena, sin);
    assert!(slot.is_nonlinear(), "nonlinear slot");
    assert_eq!(slot.support, vec![0]);
}

#[test]
fn linear_and_quadratic_closed_forms() {
    let mut arena = ExprArena::new();
    let x0 = arena.var(VarId(0));
    let x1 = arena.var(VarId(1));
    let sq = arena.push(ExprNode::Mul([x0, x0].into_iter().collect()));
    let cross = arena.push(ExprNode::Mul([x0, x1].into_iter().collect()));
    let c3 = arena.constant(3.0);
    let scaled = arena.push(ExprNode::Mul([c3, sq].into_iter().collect()));
    let lin = arena.linear(vec![(VarId(0), 2.0), (VarId(1), -1.0)], 5.0);
    let quad = arena.push(ExprNode::Add([scaled, cross, lin].into_iter().collect()));

    let SlotKind::Quadratic(q) = FunctionSlot::classify(&arena, quad).kind else {
        panic!("expected quadratic");
    };
    let SlotKind::Linear(l) = FunctionSlot::classify(&arena, lin).kind else {
        panic!("expected linear");
    };

    for x in points(2, 10) {
        let want_q = evaluate(&arena, quad, &x.as_slice()).unwrap();
        assert_close(quadratic_value(&q, &x), want_q, 1e-13, "quadratic value");
        let want_l = evaluate(&arena, lin, &x.as_slice()).unwrap();
        assert_close(linear_value(&l, &x), want_l, 1e-13, "linear value");

        // Gradient of 3 x0^2 + x0 x1 + 2 x0 - x1 + 5 is (6 x0 + x1 + 2, x0 - 1).
        let mut g = vec![0.0; 2];
        quadratic_gradient_add(&q, &x, 1.0, &mut g);
        assert_close(g[0], 6.0 * x[0] + x[1] + 2.0, 1e-13, "quad grad x0");
        assert_close(g[1], x[0] - 1.0, 1e-13, "quad grad x1");

        let mut g = vec![0.0; 2];
        linear_gradient_add(&l, 2.0, &mut g);
        assert_close(g[0], 4.0, 1e-15, "lin grad x0 (scaled)");
        assert_close(g[1], -2.0, 1e-15, "lin grad x1 (scaled)");
    }
}

#[test]
fn sparsity_patterns() {
    let mut arena = ExprArena::new();
    let x0 = arena.var(VarId(0));
    let x2 = arena.var(VarId(2));
    let sin = arena.push(ExprNode::Sin(x2));
    let mul = arena.push(ExprNode::Mul([x0, sin].into_iter().collect()));
    let lin = arena.linear(vec![(VarId(1), 1.0), (VarId(3), 2.0)], 0.0);
    let root = arena.push(ExprNode::Add([mul, lin].into_iter().collect()));
    assert_eq!(variable_support(&arena, root), vec![0, 1, 2, 3]);

    let sq1 = arena.push(ExprNode::Mul([x2, x2].into_iter().collect()));
    let slots = vec![
        FunctionSlot::classify(&arena, lin), // linear in x1, x3
        FunctionSlot::classify(&arena, sq1), // quadratic in x2
        FunctionSlot::classify(&arena, mul), // nonlinear in x0, x2
    ];

    assert_eq!(jacobian_structure(&slots), vec![(0, 1), (0, 3), (1, 2), (2, 0), (2, 2)]);
    // Hessian: the quadratic contributes (2,2), and the nonlinear x0*sin(x2)
    // has the exact pattern {(2,0), (2,2)} with no (0,0) since d²/dx0² ≡ 0.
    // Linear contributes nothing.
    assert_eq!(hessian_lagrangian_structure(&slots), vec![(2, 0), (2, 2)]);
}
