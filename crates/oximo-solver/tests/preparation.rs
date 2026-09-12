#![allow(clippy::float_cmp)]
use std::sync::Arc;

use oximo_core::{
    Model, ObjectiveSense, SosType, constraint, objective, param, soc_constraint, variable,
};
use oximo_solver::prepare::{AffineTerms, Extracted, LoweringContext, row_sides, shifted_bounds};

#[test]
fn preparation_freezes_parameters_and_shares_compound_decompositions() {
    let model = Model::new("snapshot");
    variable!(model, x);
    param!(model, p = 2.0);
    objective!(model, Max, p * x + x.powi(2) + 3.0);
    let prepared = LoweringContext::new(&model).unwrap();
    let expr = prepared.objective().unwrap().expr;
    assert!(matches!(prepared.quadratic(expr), Some(Extracted::Owned(_))));
    let Extracted::Shared(first) = prepared.quadratic(expr).unwrap() else {
        panic!("reuse not admitted")
    };
    let Extracted::Shared(second) = prepared.quadratic(expr).unwrap() else { panic!("cache miss") };
    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(prepared.sense(), ObjectiveSense::Maximize);
    model.set_param(p, 7.0);
    assert_eq!(prepared.quadratic(expr).unwrap().linear[0].1, 2.0);
    assert_eq!(LoweringContext::new(&model).unwrap().quadratic(expr).unwrap().linear[0].1, 7.0);
}

#[test]
fn affine_rows_keep_bounds_constants_and_borrowed_storage() {
    let model = Model::new("rows");
    variable!(model, x);
    constraint!(model, range, 1.0 <= 2.0 * x + 3.0 <= 5.0);
    objective!(model, Min, x);
    let prepared = LoweringContext::new(&model).unwrap();
    let row = &prepared.constraints().algebraic()[0];
    let terms = prepared.linear(row.lhs).unwrap();
    assert_eq!(shifted_bounds(row, terms.constant), (-2.0, 2.0));
    assert!(row_sides(row).iter().all(Option::is_some));
    let AffineTerms::Borrowed(t) = terms else { panic!("direct affine row was copied") };
    assert!(matches!(t.coeffs, std::borrow::Cow::Borrowed(_)));
}

#[test]
fn all_model_families_retain_original_entities() {
    let model = Model::new("families");
    variable!(model, x);
    variable!(model, t >= 0.0);
    let algebraic = constraint!(model, detected, x.powi(2) <= t.powi(2));
    let cone = soc_constraint!(model, explicit, [x] <= t);
    model.add_sos_constraint("choice", SosType::Sos1, [(x, 1.0), (t, 2.0)]);
    objective!(model, Min, x.exp());
    let prepared = LoweringContext::new(&model).unwrap();
    let row = &prepared.constraints().algebraic()[algebraic.index()];
    assert!(prepared.detected_soc(row).is_some());
    assert!(prepared.quadratic(row.lhs).is_some());
    assert!(
        prepared.explicit_soc(&prepared.constraints().second_order_cones()[cone.index()]).is_ok()
    );
    assert_eq!(prepared.constraints().special_ordered_sets()[0].members.len(), 2);
    let objective = prepared.objective().unwrap().expr;
    assert!(prepared.linear(objective).is_none());
    assert!(prepared.quadratic(objective).is_none());
    assert!(matches!(
        prepared.arena().get(objective),
        oximo_core::ExprNode::Unary(oximo_core::UnaryOp::Exp, _)
    ));
}

#[test]
fn feasibility_is_distinct_from_a_missing_objective_declaration() {
    let model = Model::new("feasibility");
    assert!(LoweringContext::new(&model).is_err());
    model.__feasibility();
    let prepared = LoweringContext::new(&model).unwrap();
    assert!(prepared.objective().is_none());
    assert_eq!(prepared.sense(), ObjectiveSense::Minimize);
}

#[test]
fn successful_extraction_never_formats_error_context() {
    let model = Model::new("lazy_errors");
    variable!(model, x);
    objective!(model, Min, x);
    let prepared = LoweringContext::new(&model).unwrap();
    let expr = prepared.objective().unwrap().expr;
    prepared.require_linear(expr, || panic!("eager affine diagnostic")).unwrap();
    prepared.require_linear_once(expr, || panic!("eager streaming diagnostic")).unwrap();
    prepared.require_quadratic(expr, || panic!("eager quadratic diagnostic")).unwrap();
    prepared.require_polynomial(expr, || panic!("eager polynomial diagnostic")).unwrap();
}

#[test]
fn failed_extraction_preserves_named_error_context() {
    let model = Model::new("named_errors");
    variable!(model, x);
    objective!(model, Min, x.exp());
    let prepared = LoweringContext::new(&model).unwrap();
    let expr = prepared.objective().unwrap().expr;
    let error = prepared.require_linear(expr, || "constraint custom".into()).unwrap_err();
    assert!(
        matches!(error, oximo_solver::SolverError::Nonlinear { location, .. } if location == "constraint custom")
    );
}

#[test]
fn explicit_form_recovers_affine_rows() {
    let model = Model::new("soc");
    variable!(model, x);
    variable!(model, y);
    variable!(model, t >= 0.0);
    model.add_soc_constraint("cone", [x - y, 2.0 * y + 1.0], t);
    objective!(model, Min, t);
    let prepared = LoweringContext::new(&model).unwrap();
    let form = prepared.explicit_soc(&prepared.constraints().second_order_cones()[0]).unwrap();
    assert_eq!(form.terms.len(), 2);
    assert_eq!(form.terms[0].coeffs.len(), 2);
    assert_eq!(form.terms[1].constant, 1.0);
    assert_eq!(form.bound.coeffs, vec![(model.variable_id("t").unwrap(), 1.0)]);
}

#[test]
fn explicit_cones_use_the_same_parameter_snapshot_as_polynomials() {
    let model = Model::new("cone_snapshot");
    variable!(model, x);
    variable!(model, t >= 0.0);
    param!(model, p = 2.0);
    soc_constraint!(model, cone, [p * x + 1.0] <= t + p);
    objective!(model, Min, t);
    let prepared = LoweringContext::new(&model).unwrap();
    let soc = &prepared.constraints().second_order_cones()[0];
    prepared.linear(soc.terms[0]).unwrap();
    let AffineTerms::Shared(first) = prepared.linear(soc.terms[0]).unwrap() else {
        panic!("parameterized member should use shared extraction");
    };
    model.set_param(p, 5.0);
    let form = prepared.explicit_soc(soc).unwrap();
    assert_eq!(form.terms[0].coeffs[0].1, 2.0);
    assert_eq!(form.bound.constant, 2.0);
    let AffineTerms::Shared(second) = prepared.linear(soc.terms[0]).unwrap() else {
        unreachable!()
    };
    assert!(Arc::ptr_eq(&first, &second));
    let fresh = LoweringContext::new(&model).unwrap();
    let form = fresh.explicit_soc(&fresh.constraints().second_order_cones()[0]).unwrap();
    assert_eq!(form.terms[0].coeffs[0].1, 5.0);
    assert_eq!(form.bound.constant, 5.0);
}
