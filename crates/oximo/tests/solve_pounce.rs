//! Integration tests for the POUNCE backend through the umbrella crate.
//! POUNCE runs on stable Rust (finite differences), so these need only
//! `--features pounce`. `--features pounce-enzyme` solves the same models with
//! exact derivatives.

#![cfg(feature = "pounce")]

use oximo::pounce::{MuStrategy, PounceOptions};
use oximo::prelude::*;
use oximo::solvers::PounceSolver;

fn assert_close(got: f64, want: f64, tol: f64, what: &str) {
    assert!((got - want).abs() < tol, "{what}: got {got}, want {want}");
}

/// Relative closeness for two independent interior-point solves.
fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-5 * a.abs().max(b.abs()).max(1.0)
}

/// Hock-Schittkowski 071:
/// min x1 x4 (x1 + x2 + x3) + x3 s.t. x1 x2 x3 x4 >= 25,
/// x1^2 + x2^2 + x3^2 + x4^2 == 40, 1 <= x <= 5.
/// Optimum approx 17.014 at (1.0, 4.743, 3.82115, 1.37941).
#[test]
fn hs071() {
    let m = Model::new("hs071");
    variable!(m, 1.0 <= x1 <= 5.0, initial = 1.0);
    variable!(m, 1.0 <= x2 <= 5.0, initial = 5.0);
    variable!(m, 1.0 <= x3 <= 5.0, initial = 5.0);
    variable!(m, 1.0 <= x4 <= 5.0, initial = 1.0);
    objective!(m, Min, x1 * x4 * (x1 + x2 + x3) + x3);
    constraint!(m, prod, x1 * x2 * x3 * x4 >= 25.0);
    constraint!(m, ssq, x1.powi(2) + x2.powi(2) + x3.powi(2) + x4.powi(2) == 40.0);

    let res = PounceSolver.solve(&m, &PounceOptions::default()).unwrap();
    assert!(res.has_solution(), "hs071 should solve");
    assert_close(res.value_of(x1).unwrap(), 1.0, 1e-3, "x1");
    assert_close(res.value_of(x2).unwrap(), 4.743, 1e-3, "x2");
    assert_close(res.value_of(x4).unwrap(), 1.379_408, 1e-3, "x4");
    assert_close(res.objective().unwrap(), 17.014, 1e-2, "objective");
}

#[test]
fn rosenbrock_unconstrained() {
    let m = Model::new("rosenbrock");
    variable!(m, -10.0 <= x <= 10.0, initial = -1.2);
    variable!(m, -10.0 <= y <= 10.0, initial = 1.0);
    objective!(m, Min, (1.0 - x).powi(2) + 100.0 * (y - x.powi(2)).powi(2));

    let res = PounceSolver.solve(&m, &PounceOptions::default()).unwrap();
    assert_eq!(res.termination, TerminationStatus::Optimal);
    assert_close(res.value_of(x).unwrap(), 1.0, 1e-4, "x");
    assert_close(res.value_of(y).unwrap(), 1.0, 1e-4, "y");
    assert!(res.objective().unwrap().abs() < 1e-6, "objective");
}

#[test]
fn maximize_flips_sign_back() {
    // max 4x - x^2 -> x = 2, objective 4.
    let m = Model::new("max");
    variable!(m, -10.0 <= x <= 10.0);
    objective!(m, Max, 4.0 * x - x.powi(2));

    let res = PounceSolver.solve(&m, &PounceOptions::default()).unwrap();
    assert_eq!(res.termination, TerminationStatus::Optimal);
    assert_close(res.value_of(x).unwrap(), 2.0, 1e-4, "x");
    assert_close(res.objective().unwrap(), 4.0, 1e-5, "objective");
}

#[test]
fn quadratic_constraint_qcp() {
    // min x + y s.t. x^2 + y^2 <= 1 -> x = y = -1/sqrt(2).
    let m = Model::new("qcp");
    variable!(m, -2.0 <= x <= 2.0);
    variable!(m, -2.0 <= y <= 2.0);
    constraint!(m, ball, x.powi(2) + y.powi(2) <= 1.0);
    objective!(m, Min, x + y);

    let res = PounceSolver.solve(&m, &PounceOptions::default()).unwrap();
    assert!(res.has_solution());
    let r = -1.0 / 2.0_f64.sqrt();
    assert_close(res.value_of(x).unwrap(), r, 1e-4, "x");
    assert_close(res.value_of(y).unwrap(), r, 1e-4, "y");
}

#[test]
fn lp_duals_match_lp_convention() {
    let m = Model::new("product_mix");
    variable!(m, x >= 0.0);
    variable!(m, y >= 0.0);
    variable!(m, z >= 0.0);
    let labor = constraint!(m, labor, x + y + z <= 12.0);
    let material = constraint!(m, material, 2.0 * x + y + 3.0 * z <= 16.0);
    objective!(m, Max, 40.0 * x + 30.0 * y + 20.0 * z);

    let res = PounceSolver.solve(&m, &PounceOptions::default()).unwrap();
    assert!(res.has_solution());
    assert_close(res.objective().unwrap(), 400.0, 1e-3, "objective");
    assert_close(res.value_of(x).unwrap(), 4.0, 1e-3, "x");
    assert_close(res.value_of(y).unwrap(), 8.0, 1e-3, "y");
    assert_close(res.dual_of(labor).unwrap(), 20.0, 1e-3, "labor dual");
    assert_close(res.dual_of(material).unwrap(), 10.0, 1e-3, "material dual");

    let z_id = m.variable_id("z").unwrap();
    assert_close(res.reduced_costs[&z_id], -30.0, 1e-3, "z reduced cost");
    assert!(res.iterations > 0, "iteration count should be reported");
}

#[test]
fn indexed_least_squares_qp() {
    // min sum_i (x_i - t_i)^2 s.t. sum_i x_i == 8 with t = [1, 2, 3, 4].
    // The projection shifts every x_i by (8 - 10) / 4: x_i = t_i - 0.5,
    // objective 4 * 0.25 = 1.
    let t = [1.0, 2.0, 3.0, 4.0];
    let n = t.len();

    let m = Model::new("least_squares");
    variable!(m, 0.0 <= x[i in 0..n] <= 10.0);
    constraint!(m, total, sum!(x[i] for i in 0..n) == 8.0);
    objective!(m, Min, sum!((x[i] - t[i]).powi(2) for i in 0..n));

    let res = PounceSolver.solve(&m, &PounceOptions::default()).unwrap();
    assert!(res.has_solution());
    assert_close(res.objective().unwrap(), 1.0, 1e-4, "objective");
    for i in 0..n {
        assert_close(res.value_of(x[i]).unwrap(), t[i] - 0.5, 1e-4, &format!("x[{i}]"));
    }
}

#[test]
fn feasibility_problem_returns_feasible_point() {
    let m = Model::new("feas");
    variable!(m, -2.0 <= x <= 2.0);
    variable!(m, -2.0 <= y <= 2.0);
    constraint!(m, disk, x.powi(2) + y.powi(2) <= 1.0);
    constraint!(m, line, x + y >= 1.0);
    objective!(m, Feasibility);

    let res = PounceSolver.solve(&m, &PounceOptions::default()).unwrap();
    assert!(res.has_solution(), "feasibility solve should return a point");
    let (xv, yv) = (res.value_of(x).unwrap(), res.value_of(y).unwrap());
    assert!(xv * xv + yv * yv <= 1.0 + 1e-5, "inside disk: ({xv}, {yv})");
    assert!(xv + yv >= 1.0 - 1e-5, "above line: ({xv}, {yv})");
}

#[test]
fn integer_models_are_rejected() {
    let m = Model::new("milp");
    variable!(m, 0.0 <= x <= 5.0, Int);
    objective!(m, Min, x);

    let err = PounceSolver.solve(&m, &PounceOptions::default()).unwrap_err();
    assert!(matches!(err, SolverError::UnsupportedKind(ModelKind::MILP)));
}

#[test]
fn persistent_matches_cold_on_parameter_sweep() {
    let m = Model::new("nlp_sweep");
    param!(m, w = 1.0);
    variable!(m, 0.1 <= x <= 10.0, initial = 1.0);
    variable!(m, 0.1 <= y <= 10.0, initial = 1.0);
    constraint!(m, prod, x * y >= 4.0);
    objective!(m, Min, w * x + y);

    let mut solver = PounceSolver.persistent();
    for wv in [1.0, 2.0, 0.5] {
        w.set_param_value(wv);
        let warm = solver.solve(&m, &PounceOptions::default()).unwrap();
        let cold = PounceSolver.solve(&m, &PounceOptions::default()).unwrap();
        assert!(warm.has_solution(), "w {wv}: no solution");
        assert!(close(warm.objective().unwrap(), cold.objective().unwrap()), "w {wv}: objective");
        assert!(close(warm.value_of(x).unwrap(), cold.value_of(x).unwrap()), "w {wv}: x");
        assert!(close(warm.value_of(y).unwrap(), cold.value_of(y).unwrap()), "w {wv}: y");
    }
}

#[test]
fn typed_options_still_solve() {
    let m = Model::new("hs071_opts");
    variable!(m, 1.0 <= x1 <= 5.0, initial = 1.0);
    variable!(m, 1.0 <= x2 <= 5.0, initial = 5.0);
    variable!(m, 1.0 <= x3 <= 5.0, initial = 5.0);
    variable!(m, 1.0 <= x4 <= 5.0, initial = 1.0);
    objective!(m, Min, x1 * x4 * (x1 + x2 + x3) + x3);
    constraint!(m, prod, x1 * x2 * x3 * x4 >= 25.0);
    constraint!(m, ssq, x1.powi(2) + x2.powi(2) + x3.powi(2) + x4.powi(2) == 40.0);

    let opts = PounceOptions::default()
        .tol(1e-8)
        .max_iter(500)
        .mu_strategy(MuStrategy::Adaptive)
        .mu_oracle("probing")
        .presolve(false);

    let res = PounceSolver.solve(&m, &opts).unwrap();
    assert!(res.has_solution(), "hs071 should solve with typed options");
    assert_close(res.objective().unwrap(), 17.014, 1e-2, "objective");
}

#[test]
fn unknown_option_is_rejected() {
    let m = Model::new("reject");
    variable!(m, 0.0 <= x <= 5.0, initial = 1.0);
    objective!(m, Min, (x - 2.0).powi(2));

    let opts = PounceOptions::default().set("not_a_real_option", 1);
    let err = PounceSolver.solve(&m, &opts).unwrap_err();
    assert!(matches!(err, SolverError::Backend(_)), "cold: got {err:?}");

    let mut solver = PounceSolver.persistent();
    let err = solver.solve(&m, &opts).unwrap_err();
    assert!(matches!(err, SolverError::Backend(_)), "persistent: got {err:?}");
}

#[test]
fn verbose_captures_raw_log() {
    let m = Model::new("logged");
    variable!(m, -10.0 <= x <= 10.0, initial = -1.2);
    objective!(m, Min, (x - 2.0).powi(2));

    let quiet = PounceSolver.solve(&m, &PounceOptions::default()).unwrap();
    assert!(quiet.raw_log.is_none(), "no log capture without verbose");

    let mut opts = PounceOptions::default();
    opts.universal.verbose = Some(true);
    let res = PounceSolver.solve(&m, &opts).unwrap();
    assert!(res.iterations > 0, "solve reports iterations");
    let log = res.raw_log.expect("verbose solve should capture a log");
    assert!(log.contains("EXIT:"), "log has the exit status: {log}");
}
