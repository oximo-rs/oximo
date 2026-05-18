#![allow(clippy::float_cmp)]

use std::cell::RefCell;

use oximo_expr::{Expr, ExprArena, ExprNode, VarId, evaluate, extract_linear};

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
fn large_sum_extracts_correctly() {
    let arena = RefCell::new(ExprArena::new());
    let vars: Vec<_> = (0..100).map(|i| make_var(&arena, i)).collect();
    let total = oximo_expr::sum(vars.iter().copied());
    let terms = extract_linear(&arena.borrow(), total.id).expect("linear");
    assert_eq!(terms.constant, 0.0);
    assert_eq!(terms.coeffs.len(), 100);
    for (_, c) in &terms.coeffs {
        assert!((*c - 1.0).abs() < f64::EPSILON);
    }
}
