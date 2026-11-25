//! Integration tests for the Clarabel backend through the umbrella crate.
//! Clarabel is pure Rust, so these run without any installed solver.
//! Compiled only with `--features clarabel`.

#![cfg(feature = "clarabel")]

use oximo::ClarabelOptions;
use oximo::prelude::*;
use oximo::solvers::Clarabel;

#[test]
fn lp_round_trip() {
    // max 3x + 2y  s.t.  x + y <= 4, 0 <= x, y <= 3. Optimum 11 at (3, 1).
    let m = Model::new("lp");
    variable!(m, 0.0 <= x <= 3.0);
    variable!(m, 0.0 <= y <= 3.0);
    constraint!(m, cap, x + y <= 4.0);
    objective!(m, Max, 3.0 * x + 2.0 * y);

    let res = Clarabel.solve(&m, &ClarabelOptions::default()).expect("solve");
    assert_eq!(res.termination, TerminationStatus::Optimal);
    assert!((res.objective().unwrap() - 11.0).abs() < 1e-6);
}

#[test]
fn socp_round_trip() {
    // min x + y  s.t.  ||(x, y)||_2 <= 1. Optimum -sqrt(2).
    let m = Model::new("socp");
    variable!(m, x);
    variable!(m, y);
    variable!(m, t >= 0.0);
    m.fix(t, 1.0).unwrap();
    soc_constraint!(m, disk, [x, y] <= t);
    objective!(m, Min, x + y);
    assert_eq!(m.kind(), ModelKind::SOCP);

    let res = Clarabel.solve(&m, &ClarabelOptions::default()).expect("solve");
    assert_eq!(res.termination, TerminationStatus::Optimal);
    assert!((res.objective().unwrap() + std::f64::consts::SQRT_2).abs() < 1e-6);
}

#[test]
fn milp_is_rejected_with_kind() {
    let m = Model::new("milp");
    variable!(m, 0.0 <= x <= 5.0, Int);
    objective!(m, Min, x);
    let err = Clarabel.solve(&m, &ClarabelOptions::default()).unwrap_err();
    assert!(
        matches!(&err, SolverError::IncompatibleModel(report)
        if report.errors.iter().any(|diagnostic|
            matches!(diagnostic.requirement, Requirement::Combination(features)
                if features.kind == ModelKind::MILP))),
        "{err}"
    );
}

// Rotated second-order cones: `||w||^2 <= 2 u v`, `u, v >= 0`.

#[expect(clippy::many_single_char_names)]
fn rotated_model(explicit: bool) -> (Model, Option<RotatedSocConstraintHandle>) {
    // min u + v  s.t.  x^2 + y^2 <= 2 u v, x = y = 1.
    // Optimum 2 at u = v = 1.
    let m = Model::new("rsocp");
    variable!(m, x);
    variable!(m, y);
    variable!(m, u >= 0.0);
    variable!(m, v >= 0.0);
    m.fix(x, 1.0).unwrap();
    m.fix(y, 1.0).unwrap();
    let handle = if explicit {
        Some(m.add_rotated_soc_constraint("r", [x, y], u, v))
    } else {
        constraint!(m, r, x * x + y * y <= 2.0 * u * v);
        None
    };
    objective!(m, Min, u + v);
    (m, handle)
}

#[test]
fn rotated_soc_explicit_round_trip_with_vector_dual() {
    let (m, handle) = rotated_model(true);
    assert_eq!(m.kind(), ModelKind::SOCP);
    let res = Clarabel.solve(&m, &ClarabelOptions::default()).expect("solve");
    assert_eq!(res.termination, TerminationStatus::Optimal);
    assert!((res.objective().unwrap() - 2.0).abs() < 1e-6);

    let z = res.soc_dual_of(handle.unwrap()).unwrap().expect("vector dual").as_slice().to_vec();
    assert_eq!(z.len(), 4);
    assert!((z[0] - 1.0).abs() < 1e-5 && (z[1] - 1.0).abs() < 1e-5, "{z:?}");
    assert!(2.0 * z[0] * z[1] >= z[2] * z[2] + z[3] * z[3] - 1e-6, "{z:?}");
}

#[test]
fn rotated_soc_detected_from_quadratic_matches_explicit() {
    let (m, _) = rotated_model(false);
    assert_eq!(m.kind(), ModelKind::SOCP);
    let res = Clarabel.solve(&m, &ClarabelOptions::default()).expect("solve");
    assert_eq!(res.termination, TerminationStatus::Optimal);
    assert!((res.objective().unwrap() - 2.0).abs() < 1e-6);
}

#[test]
#[expect(clippy::many_single_char_names)]
fn ge_rotated_soc_detected_matches_le_form() {
    let m = Model::new("rsocp ge");
    variable!(m, x);
    variable!(m, y);
    variable!(m, u >= 0.0);
    variable!(m, v >= 0.0);
    m.fix(x, 1.0).unwrap();
    m.fix(y, 1.0).unwrap();
    constraint!(m, r, 2.0 * u * v >= x * x + y * y);
    objective!(m, Min, u + v);
    assert_eq!(m.kind(), ModelKind::SOCP);

    let res = Clarabel.solve(&m, &ClarabelOptions::default()).expect("solve");
    assert_eq!(res.termination, TerminationStatus::Optimal);
    close_to(res.objective().unwrap(), 2.0);
}

#[test]
fn ge_standard_soc_detected_matches_le_form() {
    let m = Model::new("socp ge");
    variable!(m, x);
    variable!(m, y);
    variable!(m, t >= 0.0);
    m.fix(x, 3.0).unwrap();
    m.fix(y, 4.0).unwrap();
    constraint!(m, c, t * t >= x * x + y * y);
    objective!(m, Min, t);
    assert_eq!(m.kind(), ModelKind::SOCP);

    let res = Clarabel.solve(&m, &ClarabelOptions::default()).expect("solve");
    assert_eq!(res.termination, TerminationStatus::Optimal);
    close_to(res.objective().unwrap(), 5.0);
}

#[test]
#[expect(clippy::many_single_char_names)]
fn rotated_soc_persistent_resolve_after_parameter_change() {
    let m = Model::new("rsocp param");
    variable!(m, x);
    variable!(m, u >= 0.0);
    variable!(m, v >= 0.0);
    param!(m, p = 1.0);
    m.fix(x, 1.0).unwrap();
    m.add_rotated_soc_constraint("r", [p * x], u, v);
    objective!(m, Min, u + v);

    let mut solver = Clarabel.persistent();
    let first = solver.solve(&m, &ClarabelOptions::default()).expect("solve");
    assert!((first.objective().unwrap() - 2.0 * 0.5f64.sqrt()).abs() < 1e-5);
    m.set_param(p, 2.0).unwrap();
    let second = solver.solve(&m, &ClarabelOptions::default()).expect("resolve");
    assert!((second.objective().unwrap() - 2.0 * 2.0f64.sqrt()).abs() < 1e-5);
}

#[test]
#[expect(clippy::many_single_char_names)]
fn rotated_detected_persistent_after_parameter_change() {
    let m = Model::new("rsocp detected param");
    variable!(m, x);
    variable!(m, u >= 0.0);
    variable!(m, v >= 0.0);
    param!(m, p = 1.0);
    m.fix(x, 1.0).unwrap();
    constraint!(m, r, p * x * x <= 2.0 * u * v);
    objective!(m, Min, u + v);
    assert_eq!(m.kind(), ModelKind::SOCP);

    let mut solver = Clarabel.persistent();
    let first = solver.solve(&m, &ClarabelOptions::default()).expect("solve");
    close_to(first.objective().unwrap(), 2.0 * 0.5f64.sqrt());
    m.set_param(p, 2.0).unwrap();
    let second = solver.solve(&m, &ClarabelOptions::default()).expect("resolve");
    close_to(second.objective().unwrap(), 2.0);
}

#[test]
#[expect(clippy::many_single_char_names)]
fn ge_rotated_detected_persistent_after_parameter_change() {
    let m = Model::new("rsocp detected ge param");
    variable!(m, x);
    variable!(m, u >= 0.0);
    variable!(m, v >= 0.0);
    param!(m, p = 1.0);
    m.fix(x, 1.0).unwrap();
    constraint!(m, r, 2.0 * u * v >= p * x * x);
    objective!(m, Min, u + v);
    assert_eq!(m.kind(), ModelKind::SOCP);

    let mut solver = Clarabel.persistent();
    let first = solver.solve(&m, &ClarabelOptions::default()).expect("solve");
    close_to(first.objective().unwrap(), 2.0 * 0.5f64.sqrt());
    m.set_param(p, 2.0).unwrap();
    let second = solver.solve(&m, &ClarabelOptions::default()).expect("resolve");
    close_to(second.objective().unwrap(), 2.0);
}

#[test]
fn rotated_detected_persistent_after_bound_change() {
    let m = Model::new("rsocp detected bound");
    variable!(m, x);
    variable!(m, u >= 0.0);
    variable!(m, v >= 0.0);
    m.fix(x, 1.0).unwrap();
    constraint!(m, r, 2.0 * x * x <= 2.0 * u * v);
    objective!(m, Min, u + v);
    assert_eq!(m.kind(), ModelKind::SOCP);

    let mut solver = Clarabel.persistent();
    let first = solver.solve(&m, &ClarabelOptions::default()).expect("solve");
    close_to(first.objective().unwrap(), 2.0);
    m.fix(u, 2.0).unwrap();
    let second = solver.solve(&m, &ClarabelOptions::default()).expect("resolve");
    close_to(second.objective().unwrap(), 2.5);
}

fn close_to(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-5, "{a} != {b}");
}

#[test]
#[expect(clippy::many_single_char_names)]
fn rotated_explicit_cone_with_affine_heads_and_mixed_storage() {
    let m = Model::new("rsocp affine heads");
    variable!(m, x);
    variable!(m, u >= 0.0);
    variable!(m, v >= 0.0);
    variable!(m, t);
    m.fix(x, 1.0).unwrap();
    soc_constraint!(m, disk, [x] <= t);
    let cone = soc_constraint!(m, r, [x + 1.0] in rotated(u + 1.0, 2.0 * v));
    objective!(m, Min, u + v + t);

    let result = Clarabel.solve(&m, &ClarabelOptions::default()).expect("solve");
    close_to(result.objective().unwrap(), 2.0);
    let z = result.soc_dual_of(cone).unwrap().expect("rotated dual").as_slice();
    assert_eq!(z.len(), 3);
    assert!(z[0] >= -1e-7 && z[1] >= -1e-7, "{z:?}");
    assert!(2.0 * z[0] * z[1] >= z[2] * z[2] - 1e-6, "{z:?}");

    let evaluation = result.soc_evaluation(&m, cone).unwrap().unwrap();
    assert!(evaluation.violation < 1e-6, "{evaluation:?}");
}

#[test]
fn rotated_soc_with_constant_head() {
    // min t  s.t.  x^2 <= 2 t * 1, x = 3  =>  t = 4.5.
    let m = Model::new("rsoc_constant_head");
    variable!(m, x);
    variable!(m, t);
    m.fix(x, 3.0).unwrap();
    let cone = soc_constraint!(m, r, [x] in rotated(t, 1.0));
    objective!(m, Min, t);

    let result = Clarabel.solve(&m, &ClarabelOptions::default()).expect("solve");
    close_to(result.objective().unwrap(), 4.5);
    let z = result.soc_dual_of(cone).unwrap().expect("rotated dual");
    let SocHead::Rotated(zu, zv) = z.head() else { panic!("rotated dual") };
    assert!(zu >= -1e-7 && zv >= -1e-7, "{z:?}");
    assert!(2.0 * zu * zv >= z.terms()[0].powi(2) - 1e-6, "{z:?}");
}
