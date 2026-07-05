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
    m.fix(t, 1.0);
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
    assert!(matches!(err, SolverError::UnsupportedKind(ModelKind::MILP)));
}
