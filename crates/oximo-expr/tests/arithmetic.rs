#![allow(clippy::float_cmp)]

use std::cell::RefCell;

use oximo_expr::{Expr, ExprArena, ExprNode, VarId, dot, evaluate, extract_linear};

fn make_var(arena: &RefCell<ExprArena>, idx: u32) -> Expr<'_> {
    Expr::from_var(arena, VarId(idx))
}

#[test]
fn linear_fast_path_collapses_add() {
    let arena = RefCell::new(ExprArena::new());
    let x = make_var(&arena, 0);
    let y = make_var(&arena, 1);

    let combo = 2.0 * x + 3.0 * y + 5.0;

    let snapshot = arena.borrow().get(combo.id).clone();
    match snapshot {
        ExprNode::Linear { coeffs, constant } => {
            assert_eq!(constant, 5.0);
            assert_eq!(coeffs.len(), 2);
            let mut sorted = coeffs;
            sorted.sort_by_key(|(v, _)| v.0);
            assert_eq!(sorted, vec![(VarId(0), 2.0), (VarId(1), 3.0)]);
        }
        n => panic!("expected Linear node, got {n:?}"),
    }
}

#[test]
fn linear_fast_path_handles_subtraction() {
    let arena = RefCell::new(ExprArena::new());
    let x = make_var(&arena, 0);
    let y = make_var(&arena, 1);

    let combo = 4.0 * x - y - 1.0;
    let terms = extract_linear(&arena.borrow(), combo.id).expect("must be linear");
    assert_eq!(terms.constant, -1.0);
    let mut sorted = terms.coeffs;
    sorted.sort_by_key(|(v, _)| v.0);
    assert_eq!(sorted, vec![(VarId(0), 4.0), (VarId(1), -1.0)]);
}

#[test]
fn nonlinear_pow_is_not_linear() {
    let arena = RefCell::new(ExprArena::new());
    let x = make_var(&arena, 0);
    let combo = x.powi(2) + x;
    assert!(extract_linear(&arena.borrow(), combo.id).is_none());
}

#[test]
fn evaluate_recovers_value() {
    let arena = RefCell::new(ExprArena::new());
    let x = make_var(&arena, 0);
    let y = make_var(&arena, 1);
    let combo = 2.0 * x + 3.0 * y + 5.0;
    let values: &[f64] = &[10.0, 7.0];
    let arena_ref = arena.borrow();
    let v = evaluate(&arena_ref, combo.id, &values).unwrap();
    assert_eq!(v, 2.0 * 10.0 + 3.0 * 7.0 + 5.0);
}

#[test]
fn negation_flips_coefficients() {
    let arena = RefCell::new(ExprArena::new());
    let x = make_var(&arena, 0);
    let y = make_var(&arena, 1);
    let combo = -(2.0 * x + 3.0 * y + 5.0);
    let terms = extract_linear(&arena.borrow(), combo.id).expect("linear");
    assert_eq!(terms.constant, -5.0);
    let mut sorted = terms.coeffs;
    sorted.sort_by_key(|(v, _)| v.0);
    assert_eq!(sorted, vec![(VarId(0), -2.0), (VarId(1), -3.0)]);
}

#[test]
fn dot_computes_weighted_sum() {
    let arena = RefCell::new(ExprArena::new());
    let xs: Vec<_> = (0..3).map(|i| make_var(&arena, i)).collect();
    let weights = [2.0, 3.0, 5.0];
    let result = dot(&xs, &weights);
    let terms = extract_linear(&arena.borrow(), result.id).expect("linear");
    let mut sorted = terms.coeffs;
    sorted.sort_by_key(|(v, _)| v.0);
    assert_eq!(sorted, vec![(VarId(0), 2.0), (VarId(1), 3.0), (VarId(2), 5.0)]);
}

#[test]
#[should_panic(expected = "dot: length mismatch")]
fn dot_panics_on_length_mismatch() {
    let arena = RefCell::new(ExprArena::new());
    let xs: Vec<_> = (0..3).map(|i| make_var(&arena, i)).collect();
    let weights = [2.0, 3.0];
    let _ = dot(&xs, &weights);
}

#[test]
#[should_panic(expected = "Expr::sum on empty iterator")]
fn dot_panics_on_empty() {
    let xs: Vec<Expr> = Vec::new();
    let coeffs: Vec<f64> = Vec::new();
    let _ = dot(&xs, &coeffs);
}

#[test]
fn div_by_constant_stays_linear() {
    let arena = RefCell::new(ExprArena::new());
    let x = make_var(&arena, 0);
    let combo = x / 2.0;

    let snapshot = arena.borrow().get(combo.id).clone();
    match snapshot {
        ExprNode::Linear { coeffs, constant } => {
            assert_eq!(constant, 0.0);
            assert_eq!(coeffs, vec![(VarId(0), 0.5)]);
        }
        n => panic!("expected Linear node, got {n:?}"),
    }
    assert_eq!(classify(&arena.borrow(), combo.id), ExprClass::Linear);
}

#[test]
fn div_two_vars_is_nonlinear() {
    let arena = RefCell::new(ExprArena::new());
    let a = make_var(&arena, 0);
    let b = make_var(&arena, 1);
    let q = a / b;

    assert!(matches!(arena.borrow().get(q.id), ExprNode::Div(_, _)));
    assert_eq!(classify(&arena.borrow(), q.id), ExprClass::Nonlinear);
    assert!(extract_linear(&arena.borrow(), q.id).is_none());
}

#[test]
fn scalar_over_var_is_nonlinear() {
    let arena = RefCell::new(ExprArena::new());
    let x = make_var(&arena, 0);
    let recip = 1.0 / x;
    assert!(matches!(arena.borrow().get(recip.id), ExprNode::Div(_, _)));
    assert_eq!(classify(&arena.borrow(), recip.id), ExprClass::Nonlinear);
}

#[test]
fn evaluate_division() {
    let arena = RefCell::new(ExprArena::new());
    let a = make_var(&arena, 0);
    let b = make_var(&arena, 1);
    let q = a / b;
    let values: &[f64] = &[12.0, 4.0];
    let arena_ref = arena.borrow();
    assert_eq!(evaluate(&arena_ref, q.id, &values).unwrap(), 3.0);
}

#[test]
fn evaluate_division_by_zero_is_infinite() {
    let arena = RefCell::new(ExprArena::new());
    let a = make_var(&arena, 0);
    let b = make_var(&arena, 1);
    let q = a / b;
    let values: &[f64] = &[1.0, 0.0];
    let arena_ref = arena.borrow();
    assert!(evaluate(&arena_ref, q.id, &values).unwrap().is_infinite());
}

#[test]
fn large_sum_extracts_correctly() {
    let arena = RefCell::new(ExprArena::new());
    let vars: Vec<_> = (0..100).map(|i| make_var(&arena, i)).collect();
    let total: Expr = vars.iter().copied().sum();
    let terms = extract_linear(&arena.borrow(), total.id).expect("linear");
    assert_eq!(terms.constant, 0.0);
    assert_eq!(terms.coeffs.len(), 100);
    for (_, c) in &terms.coeffs {
        assert!((*c - 1.0).abs() < f64::EPSILON);
    }
}
