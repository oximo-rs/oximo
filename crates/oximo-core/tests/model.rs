#![expect(clippy::float_cmp)]

use oximo_core::prelude::*;

#[test]
fn classifies_lp() {
    let m = Model::new("lp");
    variable!(m, x >= 0.0);
    constraint!(m, c, x <= 10.0);
    objective!(m, Min, x);
    assert_eq!(m.kind(), ModelKind::LP);
}

#[test]
fn classifies_milp() {
    let m = Model::new("milp");
    variable!(m, 0.0 <= x <= 1.0, Int);
    constraint!(m, c, x <= 1.0);
    objective!(m, Min, x);
    assert_eq!(m.kind(), ModelKind::MILP);
}

#[test]
fn classifies_qp() {
    let m = Model::new("qp");
    variable!(m, x >= 0.0);
    objective!(m, Min, x.powi(2));
    assert_eq!(m.kind(), ModelKind::QP);
}

#[test]
fn classifies_miqp() {
    let m = Model::new("miqp");
    variable!(m, x >= 0.0);
    variable!(m, 0.0 <= y <= 1.0, Int);
    // Bilinear term keeps it quadratic, the integer var makes QP -> MIQP.
    objective!(m, Min, x * y);
    assert_eq!(m.kind(), ModelKind::MIQP);
}

#[test]
fn quadratic_constraint_classifies_qcp() {
    let m = Model::new("qcp");
    variable!(m, x >= 0.0);
    // A quadratic constraint (not the objective) makes the model a QCP,
    // not a QP.
    constraint!(m, c, x.powi(2) <= 4.0);
    objective!(m, Min, x);
    assert_eq!(m.kind(), ModelKind::QCP);
}

#[test]
fn quadratic_objective_and_constraint_classifies_qcp() {
    let m = Model::new("qcp_both");
    variable!(m, x >= 0.0);
    variable!(m, y >= 0.0);
    constraint!(m, c, x * y <= 4.0);
    objective!(m, Min, x.powi(2));
    assert_eq!(m.kind(), ModelKind::QCP);
}

#[test]
fn integer_var_promotes_qcp_to_miqcp() {
    let m = Model::new("miqcp");
    variable!(m, x >= 0.0);
    variable!(m, 0.0 <= y <= 1.0, Int);
    constraint!(m, c, x.powi(2) + y <= 4.0);
    objective!(m, Min, x);
    assert_eq!(m.kind(), ModelKind::MIQCP);
}

#[test]
fn soc_shaped_quadratic_constraint_classifies_socp() {
    let m = Model::new("socp_detected");
    variable!(m, x);
    variable!(m, y);
    variable!(m, t >= 0.0);
    constraint!(m, c, x.powi(2) + y.powi(2) <= t.powi(2));
    objective!(m, Min, t);
    assert_eq!(m.kind(), ModelKind::SOCP);
}

#[test]
fn soc_detection_requires_nonnegative_bound_var() {
    let m = Model::new("qcp_signed");
    variable!(m, x);
    variable!(m, t);
    constraint!(m, c, x.powi(2) <= t.powi(2));
    objective!(m, Min, t);
    assert_eq!(m.kind(), ModelKind::QCP);
}

#[test]
fn detected_socp_with_integer_var_is_misocp() {
    let m = Model::new("misocp");
    variable!(m, x);
    variable!(m, t >= 0.0);
    variable!(m, 0.0 <= z <= 1.0, Int);
    constraint!(m, c, x.powi(2) <= t.powi(2));
    constraint!(m, link, x + z >= 1.0);
    objective!(m, Min, t);
    assert_eq!(m.kind(), ModelKind::MISOCP);
}

#[test]
fn explicit_soc_constraint_classifies_socp() {
    let m = Model::new("socp_explicit");
    variable!(m, x);
    variable!(m, y);
    variable!(m, t >= 0.0);
    m.add_soc_constraint("cone", [x, y], t);
    objective!(m, Min, t);
    assert_eq!(m.kind(), ModelKind::SOCP);
    assert!(m.has_cones());
    assert!(m.soc_constraint_id("cone").is_some());
}

#[test]
fn explicit_soc_with_quadratic_objective_stays_socp() {
    let m = Model::new("socp_qobj");
    variable!(m, x);
    variable!(m, t >= 0.0);
    m.add_soc_constraint("cone", [x], t);
    objective!(m, Min, x.powi(2));
    assert_eq!(m.kind(), ModelKind::SOCP);
}

#[test]
fn explicit_soc_with_plain_quadratic_constraint_is_qcp() {
    let m = Model::new("qcp_over_socp");
    variable!(m, x);
    variable!(m, y);
    variable!(m, t >= 0.0);
    m.add_soc_constraint("cone", [x], t);
    constraint!(m, c, x * y <= 1.0);
    objective!(m, Min, t);
    assert_eq!(m.kind(), ModelKind::QCP);
}

#[test]
fn explicit_soc_with_nonlinear_constraint_is_nlp() {
    let m = Model::new("nlp_over_socp");
    variable!(m, x >= 0.1);
    variable!(m, t >= 0.0);
    m.add_soc_constraint("cone", [x], t);
    constraint!(m, c, x.sin() <= 0.5);
    objective!(m, Min, t);
    assert_eq!(m.kind(), ModelKind::NLP);
}

#[test]
#[should_panic(expected = "non-affine term")]
fn add_soc_constraint_rejects_quadratic_term() {
    let m = Model::new("bad_soc");
    variable!(m, x);
    variable!(m, t >= 0.0);
    m.add_soc_constraint("cone", [x * x], t);
}

#[test]
fn bound_change_invalidates_kind_cache() {
    // Kind depends on bounds: `x^2 <= t^2` is SOC only while `t >= 0`, so a
    // bound mutation after `kind()` was cached must recompute.
    let m = Model::new("soc_bounds");
    variable!(m, x);
    variable!(m, t);
    constraint!(m, c, x.powi(2) <= t.powi(2));
    objective!(m, Min, t);
    // t is free, so the squared form is not a cone.
    assert_eq!(m.kind(), ModelKind::QCP);

    // Fixing t to a nonnegative value makes the row a cone.
    m.fix(t, 1.0);
    assert_eq!(m.kind(), ModelKind::SOCP);

    // Unfixing back to a free variable demotes it again.
    m.unfix_var(t.var_id().unwrap(), f64::NEG_INFINITY, f64::INFINITY);
    assert_eq!(m.kind(), ModelKind::QCP);

    // A nonnegative lower bound restores the cone.
    m.unfix_var(t.var_id().unwrap(), 0.0, f64::INFINITY);
    assert_eq!(m.kind(), ModelKind::SOCP);
}

#[test]
fn adding_soc_constraint_invalidates_kind_cache() {
    let m = Model::new("soc_cache");
    variable!(m, x >= 0.0);
    variable!(m, t >= 0.0);
    objective!(m, Min, t);
    assert_eq!(m.kind(), ModelKind::LP);
    m.add_soc_constraint("cone", [x], t);
    assert_eq!(m.kind(), ModelKind::SOCP);
}

#[test]
fn classifies_nlp() {
    let m = Model::new("nlp");
    variable!(m, x >= 0.0);
    // Degree-3, so it falls through to the nonlinear path.
    objective!(m, Min, x.powi(3));
    assert_eq!(m.kind(), ModelKind::NLP);
}

#[test]
fn classifies_minlp_with_division() {
    let m = Model::new("minlp_div");
    variable!(m, x >= 1.0);
    variable!(m, 0.0 <= y <= 1.0, Int);
    // x / y is nonlinear (non-constant denominator)
    objective!(m, Min, x / y);
    assert_eq!(m.kind(), ModelKind::MINLP);
}

#[test]
fn variable_count_matches_register() {
    let m = Model::new("vars");
    variable!(m, x);
    variable!(m, y);
    variable!(m, z);
    let _ = (x, y, z);
    assert_eq!(m.num_variables(), 3);
}

#[test]
fn indexed_var_creates_named_scalars() {
    let m = Model::new("net");
    let nodes = Set::range(0..3);
    variable!(m, flow[i in nodes] >= 0.0);
    assert_eq!(flow.len(), 3);
    assert!(m.variable_id("flow[0]").is_some());
    assert!(m.variable_id("flow[2]").is_some());
}

#[test]
fn kind_caches_and_invalidates() {
    let m = Model::new("cache");
    variable!(m, x >= 0.0);
    objective!(m, Min, x);
    // First call computes, second call must return same value without traversal
    assert_eq!(m.kind(), ModelKind::LP);
    assert_eq!(m.kind(), ModelKind::LP);
    // Adding an integer variable invalidates the cache
    variable!(m, y >= 0.0, Int);
    let _ = y;
    assert_eq!(m.kind(), ModelKind::MILP);
    // Adding a constraint invalidates again
    constraint!(m, c, x <= 10.0);
    assert_eq!(m.kind(), ModelKind::MILP);
}

#[test]
fn fix_sets_equal_bounds() {
    let m = Model::new("fix_builder");
    variable!(m, 0.0 <= x <= 10.0);
    m.fix(x, 3.5);
    let vars = m.variables();
    assert_eq!(vars[0].lb, 3.5);
    assert_eq!(vars[0].ub, 3.5);
}

#[test]
fn fix_var_mutates_bounds_post_build() {
    let m = Model::new("fix_post");
    variable!(m, 0.0 <= x <= 10.0);
    let _ = x;
    let id = m.variable_id("x").unwrap();
    m.fix_var(id, 7.0);
    let vars = m.variables();
    assert_eq!(vars[0].lb, 7.0);
    assert_eq!(vars[0].ub, 7.0);
}

#[test]
fn fix_pins_var_expr_and_indexed_entry() {
    let m = Model::new("fix_expr");
    variable!(m, 0.0 <= x <= 10.0);
    m.fix(x, 3.0);
    let xid = m.variable_id("x").unwrap();
    let vars = m.variables();
    assert_eq!(vars[xid.index()].lb, 3.0);
    assert_eq!(vars[xid.index()].ub, 3.0);
    drop(vars);

    let keys = Set::strings(["a", "b"]);
    variable!(m, w[k in keys], Bin);
    m.fix(w["a"], 1.0);
    let aid = m.variable_id("w[a]").unwrap();
    let vars = m.variables();
    assert_eq!(vars[aid.index()].lb, 1.0);
    assert_eq!(vars[aid.index()].ub, 1.0);
}

#[test]
fn var_id_is_none_for_compound_expr() {
    let m = Model::new("var_id");
    variable!(m, x);
    variable!(m, y);
    assert!(x.var_id().is_some());
    assert!((x + 1.0).var_id().is_none());
    assert!((x + y).var_id().is_none());
    assert!((2.0 * x).var_id().is_none());
}

#[test]
fn unfix_var_restores_bounds() {
    let m = Model::new("unfix");
    variable!(m, 0.0 <= x <= 10.0);
    let _ = x;
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
    variable!(m, x >= 0.0);
    m.set_initial(x, 3.5);
    variable!(m, y >= 0.0);
    let _ = y;
    let vars = m.variables();
    assert_eq!(vars[0].initial, Some(3.5));
    assert_eq!(vars[1].initial, None);
}

#[test]
fn rhs_expr_folded_into_lhs() {
    use oximo_expr::extract_linear;
    let m = Model::new("rhs");
    variable!(m, x);
    variable!(m, y);
    // x <= y + 3   ↦  canonical: (x - y - 3) <= 0
    constraint!(m, c, x <= y + 3.0);
    let cs = m.constraints();
    assert_eq!(cs.len(), 1);
    assert_eq!(cs[0].as_single(), Some((Sense::Le, 0.0)));

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
