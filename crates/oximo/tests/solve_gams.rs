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

use oximo::gams::{GamsCplexOptions, GamsHighsOptions, GamsHighsSolver, GamsSolverConfig};
use oximo::prelude::*;
use oximo::solvers::Gams;

#[test]
fn gams_lp_canonical() {
    let m = Model::new("lp");
    variable!(m, x >= 0.0);
    variable!(m, 0.0 <= y <= 4.0);
    constraint!(m, c1, x + 2.0 * y <= 14.0);
    constraint!(m, c2, 3.0 * x - y >= 0.0);
    constraint!(m, c3, x - y <= 2.0);
    objective!(m, Max, 3.0 * x + 4.0 * y);

    let opts = GamsOptions::default().time_limit(Duration::from_secs(60));
    let result = Gams::new().solve(&m, &opts).unwrap();
    assert_eq!(result.termination, TerminationStatus::Optimal);
    assert!((result.objective().unwrap() - 34.0).abs() < 1e-4, "obj={:?}", result.objective());
    assert!((result.value_of(x).unwrap() - 6.0).abs() < 1e-4);
    assert!((result.value_of(y).unwrap() - 4.0).abs() < 1e-4);
}

#[test]
fn gams_lp_duals_and_reduced_costs() {
    // max x  s.t.  x <= 5,  x >= 0
    // Optimal: x = 5, dual of (x <= 5) = +/-1.0, reduced cost of x = 0.
    let m = Model::new("lp_dual");
    variable!(m, x >= 0.0);
    let c = constraint!(m, cap, x <= 5.0);
    objective!(m, Max, x);

    let opts = GamsOptions::default().time_limit(Duration::from_secs(30));
    let result = Gams::new().solve(&m, &opts).unwrap();
    assert_eq!(result.termination, TerminationStatus::Optimal);
    assert!((result.objective().unwrap() - 5.0).abs() < 1e-6);

    let d = result.dual_of(c).expect("dual missing for cap constraint");
    assert!((d.abs() - 1.0).abs() < 1e-6, "dual={d}");

    // Only one variable in the model -> VarId(0).
    let rc = result.reduced_costs.get(&VarId(0)).copied().expect("reduced cost missing for x");
    assert!(rc.abs() < 1e-6, "reduced_cost(x)={rc}");
}

#[test]
fn gams_nlp_duals_at_local_point() {
    // min exp(x)  s.t.  x >= 1
    // Optimum: x = 1, obj e. KKT: exp(x) = lambda => dual of (x >= 1) = +/-e,
    // reduced cost of x = 0.
    let m = Model::new("nlp_dual");
    variable!(m, -10.0 <= x <= 10.0);
    let cap = constraint!(m, cap, x >= 1.0);
    objective!(m, Min, x.exp());

    let opts = GamsOptions::default().time_limit(Duration::from_secs(30));
    let result = Gams::new().solve(&m, &opts).unwrap();
    assert!(
        result.has_solution(),
        "termination={:?}, primal={:?}",
        result.termination,
        result.primal_status
    );
    assert!((result.objective().unwrap() - std::f64::consts::E).abs() < 1e-5);
    assert!((result.value_of(x).unwrap() - 1.0).abs() < 1e-5);

    let dual = result.dual_of(cap).expect("dual missing for cap");
    assert!((dual.abs() - std::f64::consts::E).abs() < 1e-5, "dual={dual}");
    let rc = result.reduced_costs.get(&VarId(0)).copied().expect("reduced cost missing for x");
    assert!(rc.abs() < 1e-5, "reduced_cost(x)={rc}");
}

#[test]
fn gams_mip_duals_at_fixed_point() {
    // max 2a + 3b  s.t.  a + b <= 1,  a, b binary.
    // GAMS MIP links re-solve with integers fixed, so `.m` carries
    // the duals of the fixed problem at the optimum (0, 1).
    // The exact split between constraint duals and reduced costs is
    // solver-dependent. We assert presence, not values.
    let m = Model::new("mip_dual");
    variable!(m, a, Bin);
    variable!(m, b, Bin);
    let cap = constraint!(m, cap, a + b <= 1.0);
    objective!(m, Max, 2.0 * a + 3.0 * b);

    let opts = GamsOptions::default().time_limit(Duration::from_secs(30));
    let result = Gams::new().solve(&m, &opts).unwrap();
    assert_eq!(result.termination, TerminationStatus::Optimal);
    assert!((result.objective().unwrap() - 3.0).abs() < 1e-6);
    assert!(result.dual_of(cap).is_some(), "dual missing for cap");
    assert!(!result.reduced_costs.is_empty(), "reduced costs missing");
}

#[test]
fn gams_knapsack_milp() {
    let weights = [3.0, 4.0, 2.0, 5.0, 1.0, 6.0, 7.0, 2.0];
    let values = [10.0, 12.0, 5.0, 14.0, 3.0, 18.0, 22.0, 6.0];
    let n = weights.len();

    let m = Model::new("knapsack");
    variable!(m, x[i in 0..n], Bin);
    constraint!(m, cap, sum!(weights[i] * x[i] for i in 0..n) <= 15.0);
    objective!(m, Max, sum!(values[i] * x[i] for i in 0..n));

    let opts = GamsOptions::default().time_limit(Duration::from_secs(60));
    let result = Gams::new().solve(&m, &opts).unwrap();
    assert_eq!(result.termination, TerminationStatus::Optimal);
    assert!((result.objective().unwrap() - 47.0).abs() < 1e-4, "obj={:?}", result.objective());
}

#[test]
fn gams_semicontinuous_respects_threshold_gap() {
    // min s + t  s.t.  s >= 3,  t >= 3
    // s is semicontinuous (0 or [5, 10]), t is semi-integer (0 or int in [5, 10]).
    // The >= 3 constraints forbid 0 and the gap forbids (0, 5), so both jump to
    // 5 -> obj 10. A dropped threshold would let the solver settle at 3.
    let m = Model::new("semi");
    variable!(m, s <= 10.0, SemiCont(5.0));
    variable!(m, t <= 10.0, SemiInt(5.0));
    constraint!(m, cs, s >= 3.0);
    constraint!(m, ct, t >= 3.0);
    objective!(m, Min, s + t);

    let opts = GamsOptions::default().time_limit(Duration::from_secs(60));
    let result = Gams::new().solve(&m, &opts).unwrap();
    assert_eq!(result.termination, TerminationStatus::Optimal);
    assert!((result.objective().unwrap() - 10.0).abs() < 1e-4, "obj={:?}", result.objective());
    assert!((result.value_of(s).unwrap() - 5.0).abs() < 1e-4, "s={:?}", result.value_of(s));
    assert!((result.value_of(t).unwrap() - 5.0).abs() < 1e-4, "t={:?}", result.value_of(t));
}

#[test]
fn gams_infeasible_returns_status() {
    let m = Model::new("infeas");
    variable!(m, 0.0 <= x <= 1.0);
    constraint!(m, c1, x >= 5.0);
    objective!(m, Min, x);

    let opts = GamsOptions::default().time_limit(Duration::from_secs(30));
    let result = Gams::new().solve(&m, &opts).unwrap();
    assert_eq!(result.termination, TerminationStatus::Infeasible);
}

#[test]
fn gams_mip_gap_option() {
    let weights = [3.0, 4.0, 2.0, 5.0, 1.0];
    let values = [10.0, 12.0, 5.0, 14.0, 3.0];
    let n = weights.len();
    let m = Model::new("ks");
    variable!(m, x[i in 0..n], Bin);
    constraint!(m, cap, sum!(weights[i] * x[i] for i in 0..n) <= 8.0);
    objective!(m, Max, sum!(values[i] * x[i] for i in 0..n));

    let result = Gams::new().solve(&m, &GamsOptions::default().mip_gap(0.5)).unwrap();
    assert!(
        result.has_solution(),
        "unexpected status: {:?} / {:?}",
        result.termination,
        result.primal_status
    );
    assert!(result.objective().unwrap() > 0.0);
}

/// Exercises the typed-options path: a `highs.opt` file is written with
/// `solver = simplex` and GAMS picks it up via `model.optfile = 1`.
///
/// Requires GAMS with HiGHS available as a sub-solver.
#[test]
fn gams_highs_opt_file_simplex() {
    let m = Model::new("lp");
    variable!(m, x >= 0.0);
    variable!(m, 0.0 <= y <= 4.0);
    constraint!(m, c1, x + 2.0 * y <= 14.0);
    constraint!(m, c2, 3.0 * x - y >= 0.0);
    constraint!(m, c3, x - y <= 2.0);
    objective!(m, Max, 3.0 * x + 4.0 * y);

    let opts = GamsOptions::default().solver(GamsSolverConfig::Highs(GamsHighsOptions {
        solver: Some(GamsHighsSolver::Simplex),
        ..Default::default()
    }));
    let result = Gams::new().solve(&m, &opts).unwrap();
    assert_eq!(result.termination, TerminationStatus::Optimal);
    assert!((result.objective().unwrap() - 34.0).abs() < 1e-4, "obj={:?}", result.objective());
}

#[test]
fn gams_multi_optima_returns_single_best() {
    // A MILP with several optimal solutions. Without a sub-solver pool option the
    // GAMS bridge returns one optimum: exactly one valid point.
    let m = Model::new("multi");
    let items = Set::range(0..4usize);
    variable!(m, x[i in items], Bin);
    constraint!(m, cap, sum!(x[i] for i in items) <= 2.0);
    objective!(m, Max, sum!(x[i] for i in items));

    let opts = GamsOptions::default().time_limit(Duration::from_secs(60));
    let r = Gams::new().solve(&m, &opts).unwrap();
    assert_eq!(r.termination, TerminationStatus::Optimal);
    assert_eq!(r.result_count(), 1);
    assert!((r.objective().unwrap() - 2.0).abs() < 1e-4);
    let chosen: f64 = (0..4).filter_map(|i| r.value_of_idx(&x, i)).sum();
    assert!((chosen - 2.0).abs() < 1e-4, "best is not an optimum: sum={chosen}");
}

#[test]
fn gams_reads_cplex_solution_pool() {
    // Same multi-optima MILP. When the user enables CPLEX's `solnpool`, the
    // sub-solver writes a pool of GDX files into the run directory.
    // The GAMS backend reads them back and surfaces every point, best first.
    // Requires a GAMS install with a licensed CPLEX.
    let m = Model::new("multi");
    let items = Set::range(0..4usize);
    variable!(m, x[i in items], Bin);
    constraint!(m, cap, sum!(x[i] for i in items) <= 2.0);
    objective!(m, Max, sum!(x[i] for i in items));

    let cfg = GamsSolverConfig::Cplex(GamsCplexOptions {
        raw: vec![
            "solnpool oximo_pool.gdx".into(),
            "solnpoolpop 2".into(),
            "populatelim 20".into(),
        ],
        ..Default::default()
    });
    let opts = GamsOptions::default().solver(cfg).time_limit(Duration::from_secs(60));
    let r = Gams::new().solve(&m, &opts).unwrap();
    assert_eq!(r.termination, TerminationStatus::Optimal);
    assert!(r.result_count() > 1, "expected a solution pool, got {}", r.result_count());

    assert!((r.objective().unwrap() - 2.0).abs() < 1e-4);
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
#[expect(clippy::many_single_char_names)]
fn gams_soc_dual_matches_norm_form_multiplier() {
    // min x + y  s.t.  ||(x, y)||_2 <= 1 (explicit SOC, lowered to sqr rows).
    // KKT: z0 = ||grad obj|| = sqrt(2); the eq_soc0 marginal is rescaled by
    // 2 * bound_value.
    let m = Model::new("socp_dual");
    variable!(m, -10.0 <= x <= 10.0);
    variable!(m, -10.0 <= y <= 10.0);
    variable!(m, t >= 0.0);
    m.fix(t, 1.0);
    let disk = m.add_soc_constraint("disk", [x, y], t);
    objective!(m, Min, x + y);
    assert_eq!(m.kind(), ModelKind::SOCP);

    let r = Gams::new().solve(&m, &GamsOptions::default()).unwrap();
    assert!(r.has_solution());
    assert!((r.objective().unwrap() + std::f64::consts::SQRT_2).abs() < 1e-4);
    let z0 = r.soc_dual_of(disk).expect("SOC dual missing");
    assert!((z0 - std::f64::consts::SQRT_2).abs() < 1e-4, "z0 = {z0}");
}
