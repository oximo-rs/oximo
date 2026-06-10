//! Integration test for the BARON backend's multi-solution support.
//!
//! Shells out to a BARON installation, so compiled and run only with
//! `--features baron`:
//! ```
//! cargo test -p oximo --features baron --test solve_baron
//! ```

#![cfg(feature = "baron")]

use std::time::Duration;

use oximo::BaronOptions;
use oximo::prelude::*;
use oximo::solvers::Baron;

#[test]
fn baron_enumerates_multiple_solutions() {
    // A MILP with many feasible solutions. With `NumSol > 1`
    // BARON enumerates distinct solutions, which the backend parses into the
    // result's solution pool (best first).
    let m = Model::new("multi");
    let items = Set::range(0..4usize);
    let x = m.indexed_var("x", &items).binary().build();
    m.constraint("cap", sum_over(&items, |i: usize| x[i]).le(2.0));
    m.maximize(sum_over(&items, |i: usize| x[i]));

    let opts = BaronOptions::default().num_sol(10).time_limit(Duration::from_secs(60));
    let r = Baron::new().solve(&m, &opts).unwrap();
    assert_eq!(r.status, SolverStatus::Optimal);
    assert!(r.result_count() > 1, "expected multiple solutions, got {}", r.result_count());

    assert!((r.objective().unwrap() - 2.0).abs() < 1e-4, "best obj={:?}", r.objective());
    for s in &r.solutions {
        let chosen: f64 = (0..4).filter_map(|i| s.value_of_idx(&x, i)).sum();
        assert!(chosen <= 2.0 + 1e-4, "infeasible point: sum={chosen}");
    }
}
