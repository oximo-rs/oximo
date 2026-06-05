//! Complete integration tests: write a model to `.nl`, hand it to a real
//! AMPL-compatible solver via subprocess, parse the resulting `.sol`, and
//! verify the objective value.
//!
//! Gated behind the env var `OXIMO_TEST_NL_SOLVER`.
//!
//! Usage:
//! ```bash
//! OXIMO_TEST_NL_SOLVER=ipopt cargo test -p oximo-io --test nl_solver -- --nocapture
//! ```

#[path = "../src/nl/sol.rs"]
mod sol;

use std::process::Command;

use oximo_core::prelude::*;
use oximo_expr::evaluate;
use oximo_io::to_nl_string;
use tempfile::TempDir;

const STUB: &str = "problem";

fn solver_bin() -> Option<String> {
    std::env::var("OXIMO_TEST_NL_SOLVER").ok().filter(|s| !s.is_empty())
}

fn write_and_solve(m: &Model, expected_obj: f64, tol: f64) {
    let Some(bin) = solver_bin() else {
        eprintln!("skipping: OXIMO_TEST_NL_SOLVER not set");
        return;
    };
    let dir = TempDir::new().expect("tempdir");
    let nl_path = dir.path().join(format!("{STUB}.nl"));
    let nl = to_nl_string(m).expect("nl writer");
    std::fs::write(&nl_path, &nl).expect("write .nl");

    let status = Command::new(&bin)
        .current_dir(dir.path())
        .arg(STUB)
        .arg("-AMPL")
        .status()
        .unwrap_or_else(|e| panic!("failed to run {bin:?}: {e}"));
    assert!(status.success(), "solver exited non-zero");

    let sol_path = dir.path().join(format!("{STUB}.sol"));
    let sol_text = std::fs::read_to_string(&sol_path).expect("read .sol");
    let parsed = sol::parse_sol(&sol_text).expect("parse sol");
    eprintln!("solver primals: {:?}", parsed.primals);

    let arena = m.arena();
    let objective = m.try_objective().expect("objective");
    let obj_val =
        evaluate(&arena, objective.expr, &parsed.primals.as_slice()).expect("evaluate obj");
    let diff = (obj_val - expected_obj).abs();
    assert!(
        diff <= tol,
        "objective {obj_val} differs from expected {expected_obj} by {diff} (tol {tol})"
    );
}

#[test]
fn rosenbrock_via_solver() {
    // min (1-x0)^2 + 100 (x1 - x0^2)^2, minimum 0 at (1, 1).
    let m = Model::new("rosenbrock");
    let x0 = m.var("x0").lb(-5.0).ub(5.0).initial(-1.2).build();
    let x1 = m.var("x1").lb(-5.0).ub(5.0).initial(1.0).build();
    m.minimize((1.0 - x0).powi(2) + 100.0 * (x1 - x0.powi(2)).powi(2));
    write_and_solve(&m, 0.0, 1e-4);
}

#[test]
fn small_lp_via_solver() {
    // min  x0 + 2*x1
    // s.t. x0 + x1 >= 3
    //      0 <= x0, x1 <= 10
    // Optimal: x0=3, x1=0, obj=3.
    let m = Model::new("smalllp");
    let x0 = m.var("x0").lb(0.0).ub(10.0).build();
    let x1 = m.var("x1").lb(0.0).ub(10.0).build();
    m.minimize(x0 + 2.0 * x1);
    m.constraint("c0", (x0 + x1).ge(3.0));
    write_and_solve(&m, 3.0, 1e-4);
}

/// Real `.sol` produced by Gurobi on NEOS for the HS71 fixture (2 constraints,
/// 4 variables).
/// Locks the dimension preamble parsing that the older parser skipped.
/// Runs without a solver.
#[test]
fn parse_neos_hs71_sol() {
    let sol = "\
Gurobi 13.0.0: optimal solution; objective 17.0140172898879
126 simplex iterations
23 branching nodes

Options
3
1
1
0
2
0
4
4
1
4.742999636766323
3.821149985043959
1.379408291839082
objno 0 0
";
    let parsed = sol::parse_sol(sol).expect("parse sol");
    assert!(parsed.duals.is_empty(), "MIP solve reports no duals");
    assert_eq!(
        parsed.primals,
        vec![1.0, 4.742_999_636_766_323, 3.821_149_985_043_959, 1.379_408_291_839_082]
    );
    assert_eq!(parsed.status, Some(0));
}
