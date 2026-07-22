//! Integration test for the Gurobi backend's multi-solution support.
//!
//! Links against a licensed Gurobi installation, so compiled and run only with
//! `--features gurobi`.

#![cfg(feature = "gurobi")]

use oximo::GurobiOptions;
use oximo::prelude::*;
use oximo::solvers::Gurobi;

#[test]
fn gurobi_multi_optima_returns_pool() {
    // A MILP with many feasible/optimal assignments (pick any 2 of 4). Gurobi's
    // solution pool is requested with PoolSearchMode = 2 (find the n best) and
    // surfaced through the multi-solution API, best first.
    let m = Model::new("multi");
    let items = Set::range(0..4usize);
    variable!(m, x[i in items], Bin);
    constraint!(m, cap, sum!(x[i] for i in items) <= 2.0);
    objective!(m, Max, sum!(x[i] for i in items));

    let opts = GurobiOptions::default().pool_search_mode(2).pool_solutions(10);
    let r = Gurobi.solve(&m, &opts).unwrap();
    assert_eq!(r.termination, TerminationStatus::Optimal);
    assert!(r.result_count() > 1, "expected a solution pool, got {}", r.result_count());

    assert!((r.objective().unwrap() - 2.0).abs() < 1e-6);
    let mut prev = f64::INFINITY;
    for s in &r.solutions {
        let chosen: f64 = (0..4).filter_map(|i| s.value_of_idx(&x, i)).sum();
        assert!(chosen <= 2.0 + 1e-6, "infeasible pool point: sum={chosen}");
        let obj = s.objective.expect("pool point has an objective");
        assert!(obj <= prev + 1e-9, "pool not ordered best-first");
        prev = obj;
    }
}

#[test]
fn gurobi_qp_duals_linear_constraint() {
    // min x^2 + y^2  s.t.  x + y >= 2
    // Optimum: (1, 1), obj 2. KKT: 2x = lambda => dual of cap = 2,
    // reduced costs 0 (both variables interior to their bounds).
    let m = Model::new("qp_dual");
    variable!(m, -10.0 <= x <= 10.0);
    variable!(m, -10.0 <= y <= 10.0);
    let cap = constraint!(m, cap, x + y >= 2.0);
    objective!(m, Min, x * x + y * y);

    let result = Gurobi.solve(&m, &GurobiOptions::default()).unwrap();
    assert_eq!(result.termination, TerminationStatus::Optimal);
    assert!((result.objective().unwrap() - 2.0).abs() < 1e-5);
    assert!((result.value_of(x).unwrap() - 1.0).abs() < 1e-5);

    let dual = result.dual_of(cap).expect("dual missing for cap");
    assert!((dual.abs() - 2.0).abs() < 1e-5, "dual={dual}");
    let rc = result.reduced_costs.get(&x.var_id().unwrap()).copied();
    assert!(rc.expect("reduced cost missing").abs() < 1e-5, "rc={rc:?}");
}

#[test]
fn gurobi_qcp_duals_quadratic_constraint() {
    // min x + y  s.t.  x^2 + y^2 <= 2  (convex QCP)
    // Optimum: (-1, -1), obj -2. KKT: 1 + lambda*2x = 0 at x = -1
    // => dual of ball = 1/2 (QCPi). Gurobi computes QCP duals only when the
    // we set .qcp_dual(1).
    let m = Model::new("qcp_dual");
    variable!(m, -10.0 <= x <= 10.0);
    variable!(m, -10.0 <= y <= 10.0);
    let ball = constraint!(m, ball, x * x + y * y <= 2.0);
    objective!(m, Min, x + y);

    let opts = GurobiOptions::default().qcp_dual(1);
    let result = Gurobi.solve(&m, &opts).unwrap();
    assert_eq!(result.termination, TerminationStatus::Optimal);
    assert!((result.objective().unwrap() + 2.0).abs() < 1e-5);
    assert!((result.value_of(x).unwrap() + 1.0).abs() < 1e-5);

    let dual = result.dual_of(ball).expect("dual missing for ball");
    assert!((dual.abs() - 0.5).abs() < 1e-5, "dual={dual}");
}

#[test]
fn gurobi_semicontinuous_respects_threshold_gap() {
    // min s + t  s.t.  s >= 3,  t >= 3
    // s is semicontinuous (0 or [5, 10]), t is semi-integer (0 or int in [5, 10]).
    // The >= 3 constraints forbid 0, and the gap forbids (0, 5), so both jump to
    // 5 -> obj 10. If the threshold were dropped, the solver would settle at 3.
    let m = Model::new("semi");
    variable!(m, s <= 10.0, SemiCont(5.0));
    variable!(m, t <= 10.0, SemiInt(5.0));
    constraint!(m, cs, s >= 3.0);
    constraint!(m, ct, t >= 3.0);
    objective!(m, Min, s + t);

    let result = Gurobi.solve(&m, &GurobiOptions::default()).unwrap();
    assert_eq!(result.termination, TerminationStatus::Optimal);
    assert!((result.objective().unwrap() - 10.0).abs() < 1e-5, "obj={:?}", result.objective());
    assert!((result.value_of(s).unwrap() - 5.0).abs() < 1e-5, "s={:?}", result.value_of(s));
    assert!((result.value_of(t).unwrap() - 5.0).abs() < 1e-5, "t={:?}", result.value_of(t));
}

#[test]
fn gurobi_qcp_duals_skipped_by_default() {
    // Same convex QCP without .qcp_dual(1): the backend leaves Gurobi's
    // QCPDual default (0) untouched, so no duals are computed.
    let m = Model::new("qcp_no_dual");
    variable!(m, -10.0 <= x <= 10.0);
    variable!(m, -10.0 <= y <= 10.0);
    constraint!(m, ball, x * x + y * y <= 2.0);
    objective!(m, Min, x + y);

    let result = Gurobi.solve(&m, &GurobiOptions::default()).unwrap();
    assert_eq!(result.termination, TerminationStatus::Optimal);
    assert!((result.objective().unwrap() + 2.0).abs() < 1e-5);
    assert!(result.dual.is_empty(), "QCP duals must be empty without .qcp_dual(1)");
}

#[test]
fn gurobi_iis_pinpoints_conflicting_constraints() {
    // Infeasible: x >= 2 and x <= 1 cannot both hold. The unique minimal IIS is the
    // two conflicting constraints (the x >= 0 bound is not needed).
    let m = Model::new("iis");
    variable!(m, x >= 0.0);
    let floor = constraint!(m, floor, x >= 2.0);
    let ceil = constraint!(m, ceil, x <= 1.0);
    objective!(m, Min, x);

    let iis = Gurobi.compute_iis(&m, &GurobiOptions::default()).expect("compute_iis");
    assert!(!iis.is_empty());
    assert!(iis.constraints.contains(&floor), "floor missing from IIS: {iis:?}");
    assert!(iis.constraints.contains(&ceil), "ceil missing from IIS: {iis:?}");

    let report = iis.report(&m).to_string();
    assert!(report.contains("floor") && report.contains("ceil"), "{report}");
}

#[test]
fn gurobi_iis_errors_on_feasible_model() {
    let m = Model::new("feasible");
    variable!(m, 0.0 <= x <= 10.0);
    constraint!(m, c, x <= 5.0);
    objective!(m, Min, x);

    let err = Gurobi.compute_iis(&m, &GurobiOptions::default()).unwrap_err();
    match err {
        SolverError::Backend(msg) => assert!(msg.contains("not infeasible"), "{msg}"),
        other => panic!("expected Backend error, got {other:?}"),
    }
}

#[test]
fn gurobi_persistent_iis_reuses_resident_model() {
    let m = Model::new("iis_persist");
    variable!(m, x >= 0.0);
    let floor = constraint!(m, floor, x >= 2.0);
    let ceil = constraint!(m, ceil, x <= 1.0);
    objective!(m, Min, x);

    let mut h = Gurobi.persistent();
    let r = h.solve(&m, &GurobiOptions::default()).unwrap();
    assert!(r.termination.is_infeasible(), "expected infeasible, got {:?}", r.termination);

    let iis = h.compute_iis().expect("compute_iis on resident model");
    assert!(iis.constraints.contains(&floor), "floor missing from IIS: {iis:?}");
    assert!(iis.constraints.contains(&ceil), "ceil missing from IIS: {iis:?}");
}

#[test]
fn gurobi_persistent_iis_without_solve_errors() {
    let mut h = Gurobi.persistent();
    let err = h.compute_iis().unwrap_err();
    match err {
        SolverError::Backend(msg) => assert!(msg.contains("no resident model"), "{msg}"),
        other => panic!("expected Backend error, got {other:?}"),
    }
}
