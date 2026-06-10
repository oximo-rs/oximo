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
