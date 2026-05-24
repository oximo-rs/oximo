//! Integration tests for the GAMS backend.
//!
//! These tests shell out to a GAMS installation and are therefore
//! compiled and run only when `--features gams` is passed.  Each test
//! mirrors the corresponding HiGHS test in `solve.rs` so that regressions
//! are caught on both backends.
//!
//! Run with:
//! ```
//! cargo test -p oximo --features gams --test solve_gams
//! ```

#![cfg(feature = "gams")]

use std::time::Duration;

use oximo::gams::{GamsHighsOptions, GamsHighsSolver, GamsSolverConfig};
use oximo::prelude::*;
use oximo::solvers::Gams;

#[test]
fn gams_lp_canonical() {
    let m = Model::new("lp");
    let x = m.var("x").lb(0.0).build();
    let y = m.var("y").lb(0.0).ub(4.0).build();
    m.constraint("c1", (x + 2.0 * y).le(14.0));
    m.constraint("c2", (3.0 * x - y).ge(0.0));
    m.constraint("c3", (x - y).le(2.0));
    m.maximize(3.0 * x + 4.0 * y);

    let opts = GamsOptions::default().time_limit(Duration::from_secs(60));
    let result = Gams::new().solve(&m, &opts).unwrap();
    assert_eq!(result.status, SolverStatus::Optimal);
    assert!((result.objective.unwrap() - 34.0).abs() < 1e-4, "obj={:?}", result.objective);
    assert!((result.value_of(x).unwrap() - 6.0).abs() < 1e-4);
    assert!((result.value_of(y).unwrap() - 4.0).abs() < 1e-4);
}

#[test]
fn gams_lp_duals_and_reduced_costs() {
    // max x  s.t.  x <= 5,  x >= 0
    // Optimal: x = 5, dual of (x <= 5) = +/-1.0, reduced cost of x = 0.
    let m = Model::new("lp_dual");
    let x = m.var("x").lb(0.0).build();
    let c = m.constraint("cap", x.le(5.0));
    m.maximize(x);

    let opts = GamsOptions::default().time_limit(Duration::from_secs(30));
    let result = Gams::new().solve(&m, &opts).unwrap();
    assert_eq!(result.status, SolverStatus::Optimal);
    assert!((result.objective.unwrap() - 5.0).abs() < 1e-6);

    let d = result.dual_of(c).expect("dual missing for cap constraint");
    assert!((d.abs() - 1.0).abs() < 1e-6, "dual={d}");

    // Only one variable in the model -> VarId(0).
    let rc = result.reduced_costs.get(&VarId(0)).copied().expect("reduced cost missing for x");
    assert!(rc.abs() < 1e-6, "reduced_cost(x)={rc}");
    let _ = x;
}

#[test]
fn gams_mip_has_no_duals() {
    let m = Model::new("mip_no_duals");
    let x = m.var("x").binary().build();
    let _c = m.constraint("c", x.le(1.0));
    m.maximize(x);

    let opts = GamsOptions::default().time_limit(Duration::from_secs(30));
    let result = Gams::new().solve(&m, &opts).unwrap();
    assert_eq!(result.status, SolverStatus::Optimal);
    assert!(result.dual.is_empty(), "MIP must not return duals");
    assert!(result.reduced_costs.is_empty(), "MIP must not return reduced costs");
}

#[test]
fn gams_knapsack_milp() {
    let weights = [3.0, 4.0, 2.0, 5.0, 1.0, 6.0, 7.0, 2.0];
    let values = [10.0, 12.0, 5.0, 14.0, 3.0, 18.0, 22.0, 6.0];

    let m = Model::new("knapsack");
    let xs: Vec<_> = (0..weights.len()).map(|i| m.var(format!("x{i}")).binary().build()).collect();
    m.constraint("cap", dot(&xs, &weights).le(15.0));
    m.maximize(dot(&xs, &values));

    let opts = GamsOptions::default().time_limit(Duration::from_secs(60));
    let result = Gams::new().solve(&m, &opts).unwrap();
    assert_eq!(result.status, SolverStatus::Optimal);
    assert!((result.objective.unwrap() - 47.0).abs() < 1e-4, "obj={:?}", result.objective);
}

#[test]
fn gams_infeasible_returns_status() {
    let m = Model::new("infeas");
    let x = m.var("x").lb(0.0).ub(1.0).build();
    m.constraint("c1", x.ge(5.0));
    m.minimize(x);

    let opts = GamsOptions::default().time_limit(Duration::from_secs(30));
    let result = Gams::new().solve(&m, &opts).unwrap();
    assert_eq!(result.status, SolverStatus::Infeasible);
}

#[test]
fn gams_mip_gap_option() {
    let weights = [3.0, 4.0, 2.0, 5.0, 1.0];
    let values = [10.0, 12.0, 5.0, 14.0, 3.0];
    let m = Model::new("ks");
    let xs: Vec<_> = (0..5).map(|i| m.var(format!("x{i}")).binary().build()).collect();
    m.constraint("cap", dot(&xs, &weights).le(8.0));
    m.maximize(dot(&xs, &values));

    let result = Gams::new().solve(&m, &GamsOptions::default().mip_gap(0.5)).unwrap();
    assert!(
        matches!(result.status, SolverStatus::Optimal | SolverStatus::Feasible),
        "unexpected status: {:?}",
        result.status
    );
    assert!(result.objective.unwrap() > 0.0);
}

/// Exercises the typed-options path: a `highs.opt` file is written with
/// `solver = simplex` and GAMS picks it up via `model.optfile = 1`.
///
/// Requires GAMS with HiGHS available as a sub-solver.
#[test]
fn gams_highs_opt_file_simplex() {
    let m = Model::new("lp");
    let x = m.var("x").lb(0.0).build();
    let y = m.var("y").lb(0.0).ub(4.0).build();
    m.constraint("c1", (x + 2.0 * y).le(14.0));
    m.constraint("c2", (3.0 * x - y).ge(0.0));
    m.constraint("c3", (x - y).le(2.0));
    m.maximize(3.0 * x + 4.0 * y);

    let opts = GamsOptions::default().solver(GamsSolverConfig::Highs(GamsHighsOptions {
        solver: Some(GamsHighsSolver::Simplex),
        ..Default::default()
    }));
    let result = Gams::new().solve(&m, &opts).unwrap();
    assert_eq!(result.status, SolverStatus::Optimal);
    assert!((result.objective.unwrap() - 34.0).abs() < 1e-4, "obj={:?}", result.objective);
}
