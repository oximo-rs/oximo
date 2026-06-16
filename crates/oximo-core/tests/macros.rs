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

#[test]
fn infers_string_key_without_annotation() {
    let m = Model::new("strkey");
    let plants = Set::strings(["a", "b", "c"]);
    variable!(m, x[p in plants] >= 0.0);
    constraint!(m, total, sum!(x[p] for p in plants) <= 1.0);
    objective!(m, Max, sum!(x[p] for p in plants));
    assert_eq!(m.num_variables(), 3);
    assert_eq!(m.kind(), ModelKind::LP);
}

#[test]
fn infers_tuple_key_without_annotation() {
    let m = Model::new("tuplekey");
    let plants = Set::strings(["p1", "p2"]);
    let times = Set::range(0..3); // Set<usize>
    let pt = &plants * &times; // Set<(String, usize)>
    variable!(m, b[(p, t) in pt] >= 0.0);
    constraint!(m, lim[(p, t) in pt], b[(p, t)] <= 10.0);
    assert_eq!(m.num_variables(), 6);
    assert_eq!(m.num_constraints(), 6);
}

#[test]
fn range_literal_defaults_usize_for_array_index() {
    let m = Model::new("arr");
    let cost = [1.0, 2.0, 3.0];
    variable!(m, x[i in 0..3] >= 0.0);
    // `i` defaults to usize, so it can index `cost` directly.
    objective!(m, Min, sum!(cost[i] * x[i] for i in 0..3));
    assert_eq!(m.num_variables(), 3);
}

#[test]
fn named_integer_set_infers_usize() {
    let m = Model::new("intset");
    let days = Set::range(0..4); // Set<usize>
    let demand = [5.0, 3.0, 8.0, 2.0];
    variable!(m, y[d in days] >= 0.0);
    constraint!(m, meet[d in days], y[d] >= demand[d]);
    assert_eq!(m.num_constraints(), 4);
}

#[test]
fn index_dependent_bound_infers_key() {
    let m = Model::new("bound");
    let items = Set::range(0..3);
    let cap = [2.0, 4.0, 6.0];
    variable!(m, 0.0 <= w[i in items] <= cap[i]);
    assert_eq!(w.len(), 3);
    let vars = m.variables();
    assert!((vars[2].ub - 6.0).abs() < f64::EPSILON);
}

// --- Multi-index access sugar: `q[i, j, k]` == `q[(&i, &j, &k)]`.

#[test]
fn multi_index_sugar_builds_the_family() {
    let m = Model::new("sugar");
    let hp = Set::strings(["H1", "H2"]);
    let cp = Set::strings(["C1", "C2"]);
    let st = Set::range(0..2);
    let hcs = &(&hp * &cp) * &st;
    variable!(m, q[(i, j, k) in hcs] >= 0.0);
    constraint!(m, lim[(i, j, k) in hcs], q[i, j, k] <= 10.0);
    objective!(m, Max, sum!(q[i, j, k] for (i, j, k) in hcs));
    assert_eq!(m.num_variables(), 8);
    assert_eq!(m.num_constraints(), 8);
    assert!(m.constraint_id("lim[H1,C2,1]").is_some());
}

#[test]
fn multi_index_sugar_allows_key_reuse() {
    let m = Model::new("reuse");
    let p = Set::strings(["a", "b"]);
    let n = Set::range(0..2);
    let pn = &p * &n;
    variable!(m, s[(pp, nn) in pn] >= 0.0);
    constraint!(m, c[(pp, nn) in pn], s[pp, nn] + s[pp, nn] <= 5.0);
    assert_eq!(m.num_constraints(), 4);
}

#[test]
fn multi_bind_declaration_not_mangled() {
    let m = Model::new("multibind");
    variable!(m, b[i in 0..2, n in 0..3] >= 0.0);
    constraint!(m, lim[i in 0..2, n in 0..3], b[i, n] <= 1.0);
    assert_eq!(m.num_variables(), 6);
    assert_eq!(m.num_constraints(), 6);
}

#[test]
fn index_sugar_leaves_arrays_untouched() {
    let m = Model::new("arrays");
    let cost = [3.0, 5.0];
    let mat = [[1.0, 0.0], [0.0, 1.0]];
    variable!(m, x[i in 0..2] >= 0.0);
    constraint!(m, c, sum!(cost[i] * x[i] for i in 0..2) <= 100.0);
    objective!(
        m,
        Max,
        sum!(mat[i][j] * x[i] for i in 0..2, j in 0..2) + sum!([3.0, 5.0][i] * x[i] for i in 0..2)
    );
    assert_eq!(m.num_variables(), 2);
    assert_eq!(m.num_constraints(), 1);
}
