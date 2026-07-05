//! Tests for SOC pattern detection (`detect_soc`) and the normalized
//! `SocForm` views backends translate from.

#![allow(clippy::float_cmp, clippy::many_single_char_names)]

use oximo_core::prelude::*;

fn detect_first(m: &Model) -> Option<SocForm> {
    let arena = m.arena();
    let vars = m.variables();
    let constraints = m.constraints();
    detect_soc(&arena, &vars, &constraints[0])
}

#[test]
fn accepts_scaled_sum_of_squares() {
    let m = Model::new("soc");
    variable!(m, x);
    variable!(m, y);
    variable!(m, t >= 0.0);
    constraint!(m, c, 2.0 * x * x + 3.0 * y * y <= 5.0 * t * t);

    let form = detect_first(&m).expect("should detect SOC");
    assert_eq!(form.terms.len(), 2);
    let coef_of = |name: &str| {
        let id = m.variable_id(name).unwrap();
        form.terms.iter().find(|lt| lt.coeffs[0].0 == id).map(|lt| lt.coeffs[0].1).unwrap()
    };
    assert!((coef_of("x") - (2.0f64 / 5.0).sqrt()).abs() < 1e-12);
    assert!((coef_of("y") - (3.0f64 / 5.0).sqrt()).abs() < 1e-12);
    assert_eq!(form.bound.coeffs, vec![(m.variable_id("t").unwrap(), 1.0)]);
    assert_eq!(form.bound.constant, 0.0);
}

#[test]
fn rejects_cross_terms() {
    let m = Model::new("soc");
    variable!(m, x);
    variable!(m, y);
    variable!(m, t >= 0.0);
    constraint!(m, c, x * x + x * y + y * y <= t * t);
    assert!(detect_first(&m).is_none());
}

#[test]
fn rejects_linear_part() {
    let m = Model::new("soc");
    variable!(m, x);
    variable!(m, t >= 0.0);
    constraint!(m, c, x * x + x <= t * t);
    assert!(detect_first(&m).is_none());
}

#[test]
fn rejects_nonzero_constant() {
    let m = Model::new("soc");
    variable!(m, x);
    variable!(m, t >= 0.0);
    constraint!(m, c, x * x + 1.0 <= t * t);
    assert!(detect_first(&m).is_none());
}

#[test]
fn rejects_ge_sense() {
    let m = Model::new("soc");
    variable!(m, x);
    variable!(m, t >= 0.0);
    constraint!(m, c, x * x >= t * t);
    assert!(detect_first(&m).is_none());
}

#[test]
fn rejects_two_negative_squares() {
    let m = Model::new("soc");
    variable!(m, x);
    variable!(m, t >= 0.0);
    variable!(m, u >= 0.0);
    constraint!(m, c, x * x - u * u <= t * t);
    assert!(detect_first(&m).is_none());
}

#[test]
fn rejects_all_positive_squares() {
    let m = Model::new("soc");
    variable!(m, x);
    variable!(m, y);
    constraint!(m, c, x * x + y * y <= 4.0);
    assert!(detect_first(&m).is_none());
}

#[test]
fn rejects_unsigned_bound_variable() {
    let m = Model::new("soc");
    variable!(m, x);
    variable!(m, t);
    constraint!(m, c, x * x <= t * t);
    assert!(detect_first(&m).is_none());
}

#[test]
fn explicit_form_recovers_affine_rows() {
    let m = Model::new("soc");
    variable!(m, x);
    variable!(m, y);
    variable!(m, t >= 0.0);
    m.add_soc_constraint("cone", [x - y, 2.0 * y + 1.0], t);

    let arena = m.arena();
    let socs = m.soc_constraints();
    let form = explicit_soc_form(&arena, &socs[0]).expect("affine members");
    assert_eq!(form.terms.len(), 2);
    assert_eq!(form.terms[0].coeffs.len(), 2);
    assert_eq!(form.terms[1].constant, 1.0);
    assert_eq!(form.bound.coeffs, vec![(m.variable_id("t").unwrap(), 1.0)]);
}
