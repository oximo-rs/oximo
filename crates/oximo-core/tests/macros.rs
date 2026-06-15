//! End-to-end checks that the macros expand to the same model the
//! builder API would produce.

use oximo_core::prelude::*;

#[test]
fn scalar_variables_bounds_and_objective() {
    let m = Model::new("scalar");
    variable!(m, x >= 0.0);
    variable!(m, 0.0 <= y <= 10.0);
    variable!(m, z, Bin);

    constraint!(m, cap, x + y + z <= 10.0);
    objective!(m, Max, x + 2.0 * y);

    assert_eq!(m.num_variables(), 3);
    assert_eq!(m.num_constraints(), 1);
    assert_eq!(m.kind(), ModelKind::MILP);
}

#[test]
fn variable_bounds_apply() {
    let m = Model::new("bounds");
    variable!(m, 1.5 <= x <= 4.0);
    objective!(m, Min, x);
    let vars = m.variables();
    let v = &vars[0];
    assert!((v.lb - 1.5).abs() < f64::EPSILON);
    assert!((v.ub - 4.0).abs() < f64::EPSILON);
}

#[test]
fn indexed_variable_sum_and_family() {
    let m = Model::new("indexed");
    let assets = Set::range(0..3);
    variable!(m, 0.0 <= w[i in assets] <= 1.0);

    constraint!(m, budget, sum!(w[i] for i in assets) == 1);
    constraint!(m, ub[i in 0..3], w[i] <= 1.0);
    objective!(m, Min, sum!(w[i] for i in assets));

    assert_eq!(m.num_variables(), 3);
    assert_eq!(m.num_constraints(), 1 + 3);
    assert_eq!(m.kind(), ModelKind::LP);
    assert!(m.constraint_id("budget").is_some());
    assert!(m.constraint_id("ub[0]").is_some());
    assert!(m.constraint_id("ub[2]").is_some());
}

#[test]
fn anonymous_constraints_are_auto_named() {
    let m = Model::new("anon");
    variable!(m, x >= 0.0);
    variable!(m, y >= 0.0);
    constraint!(m, x + y >= 1.0);
    constraint!(m, x - y <= 2.0);
    assert_eq!(m.num_constraints(), 2);
    assert!(m.constraint_id("_c0").is_some());
    assert!(m.constraint_id("_c1").is_some());
}

#[test]
fn nested_sum_is_quadratic() {
    let m = Model::new("qp");
    let n = Set::range(0..2);
    let sigma = [[1.0, 0.2], [0.2, 1.0]];
    variable!(m, w[i in n] >= 0.0);
    constraint!(m, budget, sum!(w[i] for i in n) == 1);
    objective!(m, Min, sum!(sigma[i][j] * w[i] * w[j] for i in n, j in n));
    assert_eq!(m.kind(), ModelKind::QP);
}

#[test]
fn filtered_sum_skips_nonmatching_keys() {
    let m = Model::new("filter");
    let items = Set::range(0..5);
    variable!(m, x[i in items] >= 0.0);
    objective!(m, Max, sum!(x[i] for i in items if i % 2 == 0));
    constraint!(m, evens, sum!(x[i] for i in items if i % 2 == 0) <= 3.0);

    let arena = m.arena();
    let obj = m.try_objective().unwrap();
    let terms = oximo_expr::extract_linear(&arena, obj.expr).expect("linear");
    assert_eq!(terms.coeffs.len(), 3);
}

#[test]
fn param_handle_keeps_model_linear() {
    let m = Model::new("param");
    param!(m, rate = 0.05);
    variable!(m, x >= 0.0);
    constraint!(m, c, rate * x <= 1.0);
    objective!(m, Max, rate * x);
    assert_eq!(m.kind(), ModelKind::LP);
    assert!((m.param_value_of(rate).unwrap() - 0.05).abs() < f64::EPSILON);
}
