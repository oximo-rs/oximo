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
    let x = m.indexed_var("x", &items).binary().build();
    m.constraint("cap", sum_over(&items, |i: usize| x[i]).le(2.0));
    m.maximize(sum_over(&items, |i: usize| x[i]));

    let opts = GurobiOptions::default().pool_search_mode(2).pool_solutions(10);
    let r = Gurobi.solve(&m, &opts).unwrap();
    assert_eq!(r.status, SolverStatus::Optimal);
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
    let x = m.var("x").lb(-10.0).ub(10.0).build();
    let y = m.var("y").lb(-10.0).ub(10.0).build();
    let cap = m.constraint("cap", (x + y).ge(2.0));
    m.minimize(x * x + y * y);

    let result = Gurobi.solve(&m, &GurobiOptions::default()).unwrap();
    assert_eq!(result.status, SolverStatus::Optimal);
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
    let x = m.var("x").lb(-10.0).ub(10.0).build();
    let y = m.var("y").lb(-10.0).ub(10.0).build();
    let ball = m.constraint("ball", (x * x + y * y).le(2.0));
    m.minimize(x + y);

    let opts = GurobiOptions::default().qcp_dual(1);
    let result = Gurobi.solve(&m, &opts).unwrap();
    assert_eq!(result.status, SolverStatus::Optimal);
    assert!((result.objective().unwrap() + 2.0).abs() < 1e-5);
    assert!((result.value_of(x).unwrap() + 1.0).abs() < 1e-5);

    let dual = result.dual_of(ball).expect("dual missing for ball");
    assert!((dual.abs() - 0.5).abs() < 1e-5, "dual={dual}");
}

#[test]
fn gurobi_qcp_duals_skipped_by_default() {
    // Same convex QCP without .qcp_dual(1): the backend leaves Gurobi's
    // QCPDual default (0) untouched, so no duals are computed.
    let m = Model::new("qcp_no_dual");
    let x = m.var("x").lb(-10.0).ub(10.0).build();
    let y = m.var("y").lb(-10.0).ub(10.0).build();
    m.constraint("ball", (x * x + y * y).le(2.0));
    m.minimize(x + y);

    let result = Gurobi.solve(&m, &GurobiOptions::default()).unwrap();
    assert_eq!(result.status, SolverStatus::Optimal);
    assert!((result.objective().unwrap() + 2.0).abs() < 1e-5);
    assert!(result.dual.is_empty(), "QCP duals must be empty without .qcp_dual(1)");
}
