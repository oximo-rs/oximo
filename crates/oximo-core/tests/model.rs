#![allow(clippy::float_cmp)]

use oximo_core::prelude::*;

#[test]
fn classifies_lp() {
    let m = Model::new("lp");
    let x = m.var("x").lb(0.0).build();
    m.constraint("c", x.le(10.0));
    m.minimize(x);
    assert_eq!(m.kind(), ModelKind::LP);
}

#[test]
fn classifies_milp() {
    let m = Model::new("milp");
    let x = m.var("x").lb(0.0).ub(1.0).integer().build();
    m.constraint("c", x.le(1.0));
    m.minimize(x);
    assert_eq!(m.kind(), ModelKind::MILP);
}

#[test]
fn classifies_qp() {
    let m = Model::new("qp");
    let x = m.var("x").lb(0.0).build();
    m.minimize(x.powi(2));
    assert_eq!(m.kind(), ModelKind::QP);
}

#[test]
fn classifies_miqp() {
    let m = Model::new("miqp");
    let x = m.var("x").lb(0.0).build();
    let y = m.var("y").lb(0.0).ub(1.0).integer().build();
    // Bilinear term keeps it quadratic, the integer var makes QP -> MIQP.
    m.minimize(x * y);
    assert_eq!(m.kind(), ModelKind::MIQP);
}

#[test]
fn quadratic_constraint_classifies_qp() {
    let m = Model::new("qp_con");
    let x = m.var("x").lb(0.0).build();
    // Linear objective but a quadratic constraint still makes the model a QP.
    m.constraint("c", x.powi(2).le(4.0));
    m.minimize(x);
    assert_eq!(m.kind(), ModelKind::QP);
}

#[test]
fn classifies_nlp() {
    let m = Model::new("nlp");
    let x = m.var("x").lb(0.0).build();
    // Degree-3, so it falls through to the nonlinear path.
    m.minimize(x.powi(3));
    assert_eq!(m.kind(), ModelKind::NLP);
}

#[test]
fn classifies_minlp_with_division() {
    let m = Model::new("minlp_div");
    let x = m.var("x").lb(1.0).build();
    let y = m.var("y").lb(0.0).ub(1.0).integer().build();
    // x / y is nonlinear (non-constant denominator)
    m.minimize(x / y);
    assert_eq!(m.kind(), ModelKind::MINLP);
}

#[test]
fn variable_count_matches_register() {
    let m = Model::new("vars");
    let _ = m.var("x").build();
    let _ = m.var("y").build();
    let _ = m.var("z").build();
    assert_eq!(m.num_variables(), 3);
}

#[test]
fn indexed_var_creates_named_scalars() {
    let m = Model::new("net");
    let nodes = Set::range(0..3);
    let flow = m.indexed_var("flow", &nodes).lb(0.0).build();
    assert_eq!(flow.len(), 3);
    assert!(m.variable_id("flow[0]").is_some());
    assert!(m.variable_id("flow[2]").is_some());
}

#[test]
fn kind_caches_and_invalidates() {
    let m = Model::new("cache");
    let x = m.var("x").lb(0.0).build();
    m.minimize(x);
    // First call computes, second call must return same value without traversal
    assert_eq!(m.kind(), ModelKind::LP);
    assert_eq!(m.kind(), ModelKind::LP);
    // Adding an integer variable invalidates the cache
    let _ = m.var("y").lb(0.0).integer().build();
    assert_eq!(m.kind(), ModelKind::MILP);
    // Adding a constraint invalidates again
    m.constraint("c", x.le(10.0));
    assert_eq!(m.kind(), ModelKind::MILP);
}

#[test]
fn fix_builder_sets_equal_bounds() {
    let m = Model::new("fix_builder");
    let _ = m.var("x").lb(0.0).ub(10.0).fix(3.5).build();
    let vars = m.variables();
    assert_eq!(vars[0].lb, 3.5);
    assert_eq!(vars[0].ub, 3.5);
}

#[test]
fn fix_var_mutates_bounds_post_build() {
    let m = Model::new("fix_post");
    let _ = m.var("x").lb(0.0).ub(10.0).build();
    let id = m.variable_id("x").unwrap();
    m.fix_var(id, 7.0);
    let vars = m.variables();
    assert_eq!(vars[0].lb, 7.0);
    assert_eq!(vars[0].ub, 7.0);
}

#[test]
fn unfix_var_restores_bounds() {
    let m = Model::new("unfix");
    let _ = m.var("x").lb(0.0).ub(10.0).build();
    let id = m.variable_id("x").unwrap();
    m.fix_var(id, 7.0);
    m.unfix_var(id, 0.0, 10.0);
    let vars = m.variables();
    assert_eq!(vars[0].lb, 0.0);
    assert_eq!(vars[0].ub, 10.0);
}

#[test]
fn initial_value_stored_on_variable() {
    let m = Model::new("init");
    let _ = m.var("x").lb(0.0).initial(3.5).build();
    let _ = m.var("y").lb(0.0).build();
    let vars = m.variables();
    assert_eq!(vars[0].initial, Some(3.5));
    assert_eq!(vars[1].initial, None);
}

#[test]
fn rhs_expr_folded_into_lhs() {
    use oximo_expr::extract_linear;
    let m = Model::new("rhs");
    let x = m.var("x").build();
    let y = m.var("y").build();
    // x <= y + 3   ↦  canonical: (x - y - 3) <= 0
    m.constraint("c", x.le(y + 3.0));
    let cs = m.constraints();
    assert_eq!(cs.len(), 1);
    assert_eq!(cs[0].rhs, 0.0);
    assert_eq!(cs[0].sense, Sense::Le);

    // Decode the LHS and confirm the original `y + 3` made it into the linear
    // form so `coeff*x - coeff*y - 3 <= 0` is what the solver will see.
    let arena = m.arena();
    let terms = extract_linear(&arena, cs[0].lhs).expect("linear");
    assert_eq!(terms.constant, -3.0);
    let mut sorted = terms.coeffs.clone();
    sorted.sort_by_key(|(v, _)| v.0);
    assert_eq!(sorted[0].1, 1.0);
    assert_eq!(sorted[1].1, -1.0);
}
