//! POUNCE backend integration tests.
//! These run on stable Rust (the default finite-difference path).
//! With `--features enzyme` the same models are solved with exact derivatives.

use std::time::Duration;

use oximo_core::prelude::*;
use oximo_pounce::{MuStrategy, Pounce, PounceAlgorithm, PounceOptions, PounceSolverSelection};
use oximo_solver::{PersistentSolver, Solver, SolverError, TerminationStatus, UniversalOptionsExt};

fn assert_close(got: f64, want: f64, tol: f64, what: &str) {
    assert!((got - want).abs() < tol, "{what}: got {got}, want {want}");
}

/// Relative closeness for two independent interior-point solves.
fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-5 * a.abs().max(b.abs()).max(1.0)
}

/// Hock-Schittkowski 071:
/// min x1 x4 (x1+x2+x3) + x3
/// s.t. x1 x2 x3 x4 >= 25,
///      x1^2+x2^2+x3^2+x4^2 == 40,
///      1 <= x <= 5.
/// Optimum approx 17.014.
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

    let res = Pounce.solve(&m, &PounceOptions::default()).unwrap();
    assert!(res.has_solution(), "hs071 should solve");
    assert_close(res.value_of(x1).unwrap(), 1.0, 1e-3, "x1");
    assert_close(res.value_of(x2).unwrap(), 4.743, 1e-3, "x2");
    assert_close(res.value_of(x4).unwrap(), 1.379_408, 1e-3, "x4");
    assert_close(res.objective().unwrap(), 17.014, 1e-2, "objective");
    assert!(res.iterations > 0, "builder path reports iterations");
    assert_eq!(res.reduced_costs.len(), 4, "builder returns bound multipliers");
}

#[test]
fn rosenbrock_unconstrained() {
    let m = Model::new("rosenbrock");
    variable!(m, -10.0 <= x <= 10.0, initial = -1.2);
    variable!(m, -10.0 <= y <= 10.0, initial = 1.0);
    objective!(m, Min, (1.0 - x).powi(2) + 100.0 * (y - x.powi(2)).powi(2));

    let res = Pounce.solve(&m, &PounceOptions::default()).unwrap();
    assert_eq!(res.termination, TerminationStatus::LocallyOptimal);
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

    let res = Pounce.solve(&m, &PounceOptions::default()).unwrap();
    assert_eq!(res.termination, TerminationStatus::LocallyOptimal);
    assert_close(res.value_of(x).unwrap(), 2.0, 1e-4, "x");
    assert_close(res.objective().unwrap(), 4.0, 1e-5, "objective");
}

#[test]
fn overflowed_print_level_is_rejected() {
    let m = Model::new("overflow");
    variable!(m, x >= 0.0);
    objective!(m, Min, x);

    let err = Pounce.solve(&m, &PounceOptions::default().print_level(u32::MAX)).unwrap_err();
    assert!(matches!(err, SolverError::Backend(_)), "{err}");
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

    let res = Pounce.solve(&m, &PounceOptions::default()).unwrap();
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
fn quadratic_constraint_qcp() {
    // min x + y s.t. x^2 + y^2 <= 1 -> x = y = -1/sqrt(2).
    let m = Model::new("qcp");
    variable!(m, -2.0 <= x <= 2.0);
    variable!(m, -2.0 <= y <= 2.0);
    constraint!(m, ball, x.powi(2) + y.powi(2) <= 1.0);
    objective!(m, Min, x + y);

    let res = Pounce.solve(&m, &PounceOptions::default()).unwrap();
    assert!(res.has_solution());
    let r = -1.0 / 2.0_f64.sqrt();
    assert_close(res.value_of(x).unwrap(), r, 1e-4, "x");
    assert_close(res.value_of(y).unwrap(), r, 1e-4, "y");
}

#[test]
fn feasibility_problem_returns_feasible_point() {
    let m = Model::new("feas");
    variable!(m, -2.0 <= x <= 2.0);
    variable!(m, -2.0 <= y <= 2.0);
    constraint!(m, disk, x.powi(2) + y.powi(2) <= 1.0);
    constraint!(m, line, x + y >= 1.0);
    objective!(m, Feasibility);

    let res = Pounce.solve(&m, &PounceOptions::default()).unwrap();
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

    let err = Pounce.solve(&m, &PounceOptions::default()).unwrap_err();
    assert!(matches!(err, SolverError::UnsupportedKind(ModelKind::MILP)));
}

#[test]
fn sos_constraints_are_rejected_consistently() {
    let m = Model::new("sos");
    variable!(m, x >= 0.0);
    variable!(m, y >= 0.0);
    objective!(m, Min, x + y);
    sos_constraint!(m, ordered, SOS1, [(x, 1.0), (y, 2.0)]);

    let cold = Pounce.solve(&m, &PounceOptions::default()).unwrap_err();
    assert!(matches!(cold, SolverError::UnsupportedSos));
    let mut persistent = Pounce.persistent();
    let resident = persistent.solve(&m, &PounceOptions::default()).unwrap_err();
    assert!(matches!(resident, SolverError::UnsupportedSos));
}

#[test]
fn soc_constraints_are_rejected_in_mixed_models() {
    let m = Model::new("mixed_soc");
    variable!(m, x >= 0.0);
    variable!(m, t >= 0.0);
    variable!(m, y >= 0.0);
    m.add_soc_constraint("cone", [x], t);
    constraint!(m, nonlinear, y.powi(3) <= 1.0);
    objective!(m, Min, y);

    assert_eq!(m.kind(), ModelKind::NLP);
    let err = Pounce.solve(&m, &PounceOptions::default()).unwrap_err();
    assert!(matches!(err, SolverError::UnsupportedKind(ModelKind::SOCP)));
}

#[test]
fn persistent_matches_cold_on_parameter_sweep() {
    let m = Model::new("nlp_sweep");
    param!(m, w = 1.0);
    variable!(m, 0.1 <= x <= 10.0, initial = 1.0);
    variable!(m, 0.1 <= y <= 10.0, initial = 1.0);
    constraint!(m, prod, x * y >= 4.0);
    objective!(m, Min, w * x + y);

    let mut solver = Pounce.persistent();
    for wv in [1.0, 2.0, 0.5, 3.0] {
        w.set_param_value(wv);
        let warm = solver.solve(&m, &PounceOptions::default()).unwrap();
        let cold = Pounce.solve(&m, &PounceOptions::default()).unwrap();
        assert!(warm.has_solution(), "w {wv}: no solution");
        assert!(close(warm.objective().unwrap(), cold.objective().unwrap()), "w {wv}: objective");
        assert!(close(warm.value_of(x).unwrap(), cold.value_of(x).unwrap()), "w {wv}: x");
        assert!(close(warm.value_of(y).unwrap(), cold.value_of(y).unwrap()), "w {wv}: y");
    }
}

#[test]
fn persistent_rebuilds_on_structural_change() {
    let m = Model::new("grow");
    variable!(m, 0.1 <= x <= 10.0, initial = 1.0);
    variable!(m, 0.1 <= y <= 10.0, initial = 1.0);
    constraint!(m, prod, x * y >= 4.0);
    objective!(m, Min, x + y);

    let mut solver = Pounce.persistent();
    let first = solver.solve(&m, &PounceOptions::default()).unwrap();
    assert!(first.has_solution());

    constraint!(m, sum, x + y <= 8.0);
    let grown = solver.solve(&m, &PounceOptions::default()).unwrap();
    let cold = Pounce.solve(&m, &PounceOptions::default()).unwrap();
    assert!(grown.has_solution(), "grown: no solution");
    assert!(close(grown.objective().unwrap(), cold.objective().unwrap()), "objective after growth");
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
        .mu_strategy(MuStrategy::Adaptive)
        .mu_oracle("probing")
        .barrier_tol_factor(10.0)
        .quality_function_max_section_steps(8)
        .feral_scaling("mc64")
        .presolve(false);

    let res = Pounce.solve(&m, &opts).unwrap();
    assert!(res.has_solution(), "hs071 should solve with typed options");
    assert_close(res.objective().unwrap(), 17.014, 1e-2, "objective");
}

#[test]
fn unknown_option_is_rejected() {
    let m = Model::new("reject");
    variable!(m, 0.0 <= x <= 5.0, initial = 1.0);
    objective!(m, Min, (x - 2.0).powi(2));

    let opts = PounceOptions::default().set("not_a_real_option", 1);
    let err = Pounce.solve(&m, &opts).unwrap_err();
    assert!(matches!(err, SolverError::Backend(_)), "cold: got {err:?}");

    let mut solver = Pounce.persistent();
    let err = solver.solve(&m, &opts).unwrap_err();
    assert!(matches!(err, SolverError::Backend(_)), "persistent: got {err:?}");
}

#[test]
fn invalid_option_value_is_rejected() {
    let m = Model::new("reject_val");
    variable!(m, 0.0 <= x <= 5.0, initial = 1.0);
    objective!(m, Min, (x - 2.0).powi(2));

    let err = Pounce.solve(&m, &PounceOptions::default().set("mu_init", -1.0)).unwrap_err();
    assert!(matches!(err, SolverError::Backend(_)), "out-of-range: got {err:?}");

    let err = Pounce.solve(&m, &PounceOptions::default().mu_oracle("nonsense")).unwrap_err();
    assert!(matches!(err, SolverError::Backend(_)), "bad enum: got {err:?}");
}

#[test]
fn verbose_captures_raw_log() {
    let m = Model::new("logged");
    variable!(m, -10.0 <= x <= 10.0, initial = -1.2);
    objective!(m, Min, (x - 2.0).powi(2));

    let quiet = Pounce
        .solve(&m, &PounceOptions::default().solver_selection(PounceSolverSelection::Nlp))
        .unwrap();
    assert!(quiet.raw_log.is_none(), "no log capture without verbose");

    let mut opts =
        PounceOptions::default().solver_selection(PounceSolverSelection::Nlp).print_level(0);
    opts.universal.verbose = Some(true);
    let res = Pounce.solve(&m, &opts).unwrap();
    assert!(res.iterations > 0, "TNLP solve reports iterations");
    let log = res.raw_log.expect("verbose solve should capture a log");
    assert!(log.contains("Number of Iterations....:"), "log has the summary: {log}");
    assert!(log.contains("objective function evaluations"), "log has eval counts: {log}");
    assert!(log.contains("KKT error above row noise:"), "log has noise-aware KKT error: {log}");
    assert!(log.contains("Iteration history:"), "log has iteration history: {log}");
    assert!(log.contains("EXIT:"), "log has the exit status: {log}");
}

#[test]
fn time_limit_is_honored_by_automatic_and_forced_convex_routes() {
    let m = Model::new("time_limited_lp");
    variable!(m, 0.0 <= x <= 10.0);
    objective!(m, Min, x);

    let automatic = Pounce
        .solve(
            &m,
            &PounceOptions::default()
                .time_limit(Duration::from_secs(1))
                .verbose(true)
                .print_level(0),
        )
        .unwrap();
    assert!(automatic.has_solution(), "automatic convex route failed: {:?}", automatic.termination);
    assert!(
        automatic.raw_log.as_deref().is_some_and(|log| log.contains("POUNCE convex route")),
        "a time-limited automatic solve should retain the convex route"
    );

    let forced = Pounce
        .solve(
            &m,
            &PounceOptions::default()
                .solver_selection(PounceSolverSelection::QpIpm)
                .time_limit(Duration::from_secs(1)),
        )
        .unwrap();
    assert!(forced.has_solution(), "forced convex route failed: {:?}", forced.termination);
}

#[test]
fn verbose_builder_path_logs_exit_status() {
    let m = Model::new("logged_builder");
    variable!(m, -10.0 <= x <= 10.0, initial = -1.2);
    variable!(m, -10.0 <= y <= 10.0, initial = 1.0);
    constraint!(m, exp_cap, x.exp() <= 10.0);
    objective!(m, Min, (1.0 - x).powi(2) + 100.0 * (y - x.powi(2)).powi(2));

    let mut opts = PounceOptions::default().print_level(0).presolve(true).presolve_fbbt(true);
    opts.universal.verbose = Some(true);
    let res = Pounce.solve(&m, &opts).unwrap();
    let log = res.raw_log.expect("verbose solve should capture a log");
    assert!(log.contains("KKT error above row noise:"), "log has noise-aware KKT error: {log}");
    assert!(log.contains("Iteration history:"), "log has iteration history: {log}");
    assert!(log.contains("EXIT:"), "log has the exit status: {log}");
}

#[cfg(not(feature = "enzyme"))]
#[test]
fn invalid_generated_fbbt_tape_is_returned_as_a_solver_error() {
    let m = Model::new("invalid_fbbt_tape");
    param!(m, offset = 0.0);
    variable!(m, -2.0 <= x <= 2.0, initial = 0.0);
    constraint!(m, nonlinear, offset + x.exp() <= 10.0);
    objective!(m, Min, x.powi(2));
    offset.set_param_value(f64::NAN);

    let options = PounceOptions::default()
        .solver_selection(PounceSolverSelection::Nlp)
        .presolve(true)
        .presolve_fbbt(true)
        .print_level(0);
    let error = Pounce.solve(&m, &options).unwrap_err();
    assert!(
        matches!(&error, SolverError::Backend(message) if message.contains("invalid FBBT tape")),
        "unexpected error: {error}"
    );
}

#[test]
fn active_set_sqp_solves_a_qp() {
    let m = Model::new("active_set_qp");
    variable!(m, 0.0 <= x <= 5.0, initial = 0.0);
    variable!(m, 0.0 <= y <= 5.0, initial = 0.0);
    constraint!(m, balance, x + y == 3.0);
    objective!(m, Min, (x - 1.0).powi(2) + (y - 2.0).powi(2));

    let opts = PounceOptions::default().algorithm(PounceAlgorithm::ActiveSetSqp);
    let res = Pounce.solve(&m, &opts).unwrap();
    assert!(res.has_solution(), "active-set SQP failed: {:?}", res.termination);
    assert!(res.iterations > 0);
    assert_close(res.value_of(x).unwrap(), 1.0, 1e-4, "x");
    assert_close(res.value_of(y).unwrap(), 2.0, 1e-4, "y");
}

#[test]
fn active_set_sqp_does_not_run_interior_point_infeasibility_retries() {
    let m = Model::new("active_set_infeasible");
    variable!(m, -2.0 <= x <= 2.0, initial = 0.0);
    constraint!(m, impossible, x.powi(2) <= -1.0);
    objective!(m, Min, x);

    let options = PounceOptions::default()
        .algorithm(PounceAlgorithm::ActiveSetSqp)
        .presolve(false)
        .print_level(0)
        .verbose(true);
    let result = Pounce.solve(&m, &options).unwrap();
    assert_eq!(
        result.termination,
        TerminationStatus::Infeasible,
        "{}",
        result.raw_log.as_deref().unwrap_or("no log")
    );
    assert!(
        result
            .raw_log
            .as_deref()
            .is_some_and(|log| !log.contains("local-infeasibility second opinion")),
        "interior-point retry controls must not be applied to active-set SQP"
    );
}

#[test]
fn convex_algorithm_selectors_are_available() {
    let m = Model::new("convex_selector");
    variable!(m, x >= 0.0);
    objective!(m, Min, x);

    for (selection, route) in [
        (PounceSolverSelection::LpIpm, "QpIpm"),
        (PounceSolverSelection::QpIpm, "QpIpm"),
        (PounceSolverSelection::QpActiveSet, "QpActiveSet"),
        (PounceSolverSelection::Socp, "Socp"),
    ] {
        let result = Pounce
            .solve(&m, &PounceOptions::default().solver_selection(selection).verbose(true))
            .unwrap();
        assert!(result.has_solution(), "{selection:?}: {:?}", result.termination);
        assert!(
            result
                .raw_log
                .as_deref()
                .is_some_and(|log| log.contains(&format!("POUNCE convex route: {route}"))),
            "{selection:?} did not use the expected {route} engine"
        );
    }
}

#[test]
fn automatic_convexity_respects_objective_sense() {
    let concave_max = Model::new("concave_max");
    variable!(concave_max, -10.0 <= x <= 10.0);
    objective!(concave_max, Max, 4.0 * x - x.powi(2));
    let result = Pounce.solve(&concave_max, &PounceOptions::default().verbose(true)).unwrap();
    assert_close(result.value_of(x).unwrap(), 2.0, 1e-5, "concave maximize x");
    assert!(
        result.raw_log.as_deref().is_some_and(|log| log.contains("POUNCE convex route: QpIpm")),
        "automatic concave-max normalization did not use the convex QP engine"
    );

    let convex_max = Model::new("convex_max");
    variable!(convex_max, -1.0 <= y <= 1.0, initial = 0.2);
    objective!(convex_max, Max, y.powi(2));
    let forced = Pounce
        .solve(
            &convex_max,
            &PounceOptions::default().solver_selection(PounceSolverSelection::QpIpm),
        )
        .unwrap_err();
    assert!(matches!(forced, SolverError::Backend(message) if message.contains("proven-convex")));
    let automatic = Pounce.solve(&convex_max, &PounceOptions::default()).unwrap();
    assert!(automatic.has_solution());
}

#[test]
fn convex_iteration_limit_is_not_retried_as_nlp() {
    let m = Model::new("limited_lp");
    variable!(m, 0.0 <= x <= 10.0);
    objective!(m, Min, x);

    let options = PounceOptions::default().max_iter(0).qp_presolve(false).verbose(true);
    let cold = Pounce.solve(&m, &options).unwrap();
    assert_eq!(cold.termination, TerminationStatus::IterationLimit);
    assert!(
        cold.raw_log.as_deref().is_some_and(|log| log.contains("POUNCE convex route")),
        "an iteration limit must retain the original convex result"
    );

    let mut persistent = Pounce.persistent();
    let resident = persistent.solve(&m, &options).unwrap();
    assert_eq!(resident.termination, TerminationStatus::IterationLimit);
    assert!(
        resident.raw_log.as_deref().is_some_and(|log| log.contains("POUNCE convex route")),
        "persistent solving must use the same fallback policy"
    );
}

#[test]
fn convex_objective_constants_survive_sense_normalization() {
    let min_model = Model::new("constant_min");
    variable!(min_model, -5.0 <= x <= 5.0);
    objective!(min_model, Min, (x - 2.0).powi(2) + 7.0);
    let min_result = Pounce.solve(&min_model, &PounceOptions::default()).unwrap();
    assert_close(min_result.objective().unwrap(), 7.0, 1e-6, "minimum constant");

    let max_model = Model::new("constant_max");
    variable!(max_model, -5.0 <= y <= 5.0);
    objective!(max_model, Max, 11.0 - (y + 1.0).powi(2));
    let max_result = Pounce.solve(&max_model, &PounceOptions::default()).unwrap();
    assert_close(max_result.objective().unwrap(), 11.0, 1e-6, "maximum constant");
}

#[test]
fn explicit_soc_routes_to_conic_ipm_and_reports_dual() {
    let m = Model::new("soc");
    variable!(m, -2.0 <= x <= 2.0);
    variable!(m, -2.0 <= y <= 2.0);
    variable!(m, 0.0 <= t <= 2.0);
    let cone = m.add_soc_constraint("disk", [x, y], t);
    constraint!(m, radius, t == 1.0);
    objective!(m, Max, x);

    let result = Pounce.solve(&m, &PounceOptions::default()).unwrap();
    assert!(result.has_solution(), "{:?}", result.termination);
    assert_close(result.value_of(x).unwrap(), 1.0, 1e-5, "soc x");
    assert!(result.soc_dual_of(cone).is_some());
}

#[test]
fn persistent_convex_ipm_and_active_set_follow_parameter_sweeps() {
    let m = Model::new("convex_sweep");
    param!(m, target = 1.0);
    variable!(m, -5.0 <= x <= 5.0);
    objective!(m, Min, x.powi(2) - 2.0 * target * x);

    for selection in [PounceSolverSelection::Auto, PounceSolverSelection::QpActiveSet] {
        let options = PounceOptions::default().solver_selection(selection);
        let mut persistent = Pounce.persistent();
        for value in [1.0, -2.0, 0.5, 3.0] {
            target.set_param_value(value);
            let warm = persistent.solve(&m, &options).unwrap();
            let cold = Pounce.solve(&m, &options).unwrap();
            assert!(warm.has_solution(), "{selection:?}, target={value}");
            assert_close(warm.value_of(x).unwrap(), cold.value_of(x).unwrap(), 1e-6, "x");
        }
    }
}

#[test]
fn persistent_conic_ipm_reuses_bound_expanded_warm_start() {
    let m = Model::new("soc_sweep");
    param!(m, weight = 1.0);
    variable!(m, -2.0 <= x <= 0.5);
    variable!(m, -2.0 <= y <= 2.0);
    variable!(m, 0.0 <= t <= 2.0);
    m.add_soc_constraint("disk", [x, y], t);
    constraint!(m, radius, t == 1.0);
    objective!(m, Max, weight * x + y);

    let mut persistent = Pounce.persistent();
    for value in [1.0, 2.0, 0.5] {
        weight.set_param_value(value);
        let warm = persistent.solve(&m, &PounceOptions::default()).unwrap();
        let cold = Pounce.solve(&m, &PounceOptions::default()).unwrap();
        assert!(warm.has_solution(), "weight={value}: {:?}", warm.termination);
        assert_close(warm.objective().unwrap(), cold.objective().unwrap(), 1e-5, "objective");
        assert_eq!(warm.reduced_costs.len(), 3);
        for variable in [x, y, t] {
            let id = variable.var_id().unwrap();
            assert_close(warm.reduced_costs[&id], cold.reduced_costs[&id], 1e-4, "reduced cost");
        }
    }
}

#[test]
fn persistent_reset_then_solve_ok() {
    let m = Model::new("reset");
    param!(m, w = 1.0);
    variable!(m, 0.1 <= x <= 10.0, initial = 1.0);
    variable!(m, 0.1 <= y <= 10.0, initial = 1.0);
    constraint!(m, prod, x * y >= 4.0);
    objective!(m, Min, w * x + y);

    let mut solver = Pounce.persistent();
    w.set_param_value(2.0);
    let first = solver.solve(&m, &PounceOptions::default()).unwrap();
    solver.reset();
    let after = solver.solve(&m, &PounceOptions::default()).unwrap();
    assert!(close(first.objective().unwrap(), after.objective().unwrap()));
}

#[test]
fn persistent_sweep_reclassifies_nonlinear_objective() {
    let m = Model::new("kind_flip");
    param!(m, w = 1.0);
    variable!(m, 0.1 <= x <= 10.0, initial = 1.0);
    variable!(m, 0.1 <= y <= 10.0, initial = 1.0);
    constraint!(m, prod, x * y >= 4.0);
    objective!(m, Min, x + y + w * x.powi(3));

    let mut solver = Pounce.persistent();
    for wv in [1.0, 0.0, 2.0] {
        w.set_param_value(wv);
        let warm = solver.solve(&m, &PounceOptions::default()).unwrap();
        let cold = Pounce.solve(&m, &PounceOptions::default()).unwrap();
        assert!(warm.has_solution(), "w {wv}: no solution");
        assert!(close(warm.objective().unwrap(), cold.objective().unwrap()), "w {wv}: objective");
        assert!(close(warm.value_of(x).unwrap(), cold.value_of(x).unwrap()), "w {wv}: x");
        assert!(close(warm.value_of(y).unwrap(), cold.value_of(y).unwrap()), "w {wv}: y");
    }
}

#[test]
fn public_solver_metadata_matches_supported_model_kinds() {
    assert_eq!(Pounce.name(), "pounce");
    for kind in [ModelKind::LP, ModelKind::QP, ModelKind::QCP, ModelKind::SOCP, ModelKind::NLP] {
        assert!(Pounce.supports(kind), "{kind:?}");
    }
    assert!(!Pounce.supports(ModelKind::MILP));

    let persistent = Pounce.persistent();
    assert_eq!(persistent.name(), "pounce");
    assert!(persistent.supports(ModelKind::SOCP));
    assert!(!persistent.supports(ModelKind::MINLP));
    assert!(format!("{persistent:?}").contains("resident: false"));
}

#[test]
fn forced_routes_reject_incompatible_models() {
    let qp = Model::new("not_an_lp");
    variable!(qp, -2.0 <= x <= 2.0);
    objective!(qp, Min, x.powi(2));
    let error = Pounce
        .solve(&qp, &PounceOptions::default().solver_selection(PounceSolverSelection::LpIpm))
        .unwrap_err();
    assert!(
        matches!(error, SolverError::Backend(message) if message.contains("anything other than an LP"))
    );

    let nlp = Model::new("not_convex");
    variable!(nlp, -2.0 <= y <= 2.0, initial = 0.5);
    objective!(nlp, Min, y.powi(3));
    for selection in [PounceSolverSelection::QpActiveSet, PounceSolverSelection::Socp] {
        let error =
            Pounce.solve(&nlp, &PounceOptions::default().solver_selection(selection)).unwrap_err();
        assert!(matches!(error, SolverError::Backend(_)), "{selection:?}: {error:?}");
    }

    let soc = Model::new("soc_is_not_nlp");
    variable!(soc, sx);
    variable!(soc, st >= 0.0);
    soc.add_soc_constraint("cone", [sx], st);
    objective!(soc, Min, st);
    for options in [
        PounceOptions::default().solver_selection(PounceSolverSelection::Nlp),
        PounceOptions::default().algorithm(PounceAlgorithm::ActiveSetSqp),
    ] {
        let error = Pounce.solve(&soc, &options).unwrap_err();
        assert!(matches!(error, SolverError::Backend(_)), "{error:?}");
    }
    let timed =
        Pounce.solve(&soc, &PounceOptions::default().time_limit(Duration::from_secs(1))).unwrap();
    assert!(timed.has_solution(), "automatic SOC route failed: {:?}", timed.termination);
}

#[test]
fn coupled_quadratic_hessians_are_certified_by_inertia() {
    let convex = Model::new("coupled_convex");
    variable!(convex, -5.0 <= x <= 5.0);
    variable!(convex, -5.0 <= y <= 5.0);
    objective!(convex, Min, (x + y).powi(2) + (x - 1.0).powi(2));
    let result = Pounce
        .solve(
            &convex,
            &PounceOptions::default().solver_selection(PounceSolverSelection::QpIpm).verbose(true),
        )
        .unwrap();
    assert!(result.has_solution(), "{:?}", result.termination);
    assert!(result.raw_log.as_deref().is_some_and(|log| log.contains("POUNCE convex route")));

    let indefinite = Model::new("coupled_indefinite");
    variable!(indefinite, -1.0 <= u <= 1.0);
    variable!(indefinite, -1.0 <= v <= 1.0);
    objective!(indefinite, Min, u * v);
    let error = Pounce
        .solve(
            &indefinite,
            &PounceOptions::default().solver_selection(PounceSolverSelection::QpIpm),
        )
        .unwrap_err();
    assert!(matches!(error, SolverError::Backend(message) if message.contains("proven-convex")));
}

#[test]
fn detected_socp_routes_to_conic_ipm() {
    let m = Model::new("detected_socp");
    variable!(m, x);
    variable!(m, y);
    variable!(m, t >= 0.0);
    constraint!(m, fix_x, x == 3.0);
    constraint!(m, fix_y, y == 4.0);
    constraint!(m, cone, x.powi(2) + y.powi(2) <= t.powi(2));
    objective!(m, Min, t);
    assert_eq!(m.kind(), ModelKind::SOCP);

    let result = Pounce.solve(&m, &PounceOptions::default().qp_presolve(false)).unwrap();
    assert!(result.has_solution(), "{:?}", result.termination);
    assert_close(result.value_of(t).unwrap(), 5.0, 1e-5, "t");
    assert_close(
        result.dual_of(m.constraint_id("cone").unwrap()).unwrap(),
        -1.0,
        1e-5,
        "cone dual",
    );
}

#[test]
fn ranged_linear_constraint_maps_both_sides_and_dual() {
    let m = Model::new("ranged_lp");
    variable!(m, x >= 0.0);
    variable!(m, y >= 0.0);
    constraint!(m, band, 1.0 <= x + y <= 3.0);
    objective!(m, Max, 3.0 * x + y + 2.0);

    let result = Pounce.solve(&m, &PounceOptions::default()).unwrap();
    assert_close(result.value_of(x).unwrap(), 3.0, 1e-5, "x");
    assert_close(result.value_of(y).unwrap(), 0.0, 1e-5, "y");
    assert_close(result.objective().unwrap(), 11.0, 1e-5, "objective");
    assert_close(result.dual_of(m.constraint_id("band").unwrap()).unwrap(), 3.0, 1e-5, "band dual");
}

#[test]
fn convex_options_validate_and_apply_to_each_qp_engine() {
    let m = Model::new("convex_options");
    variable!(m, -5.0 <= x <= 5.0);
    objective!(m, Min, (x - 2.0).powi(2));

    let ipm = PounceOptions::default()
        .solver_selection(PounceSolverSelection::QpIpm)
        .qp_presolve(true)
        .set("qp_presolve", false)
        .qp_tau(0.9)
        .qp_tau_max(0.95)
        .qp_reg(1e-9)
        .qp_infeas_tol(1e-7)
        .qp_hsde(false)
        .qp_equilibrate(false)
        .qp_crossover(false)
        .qp_gondzio_corr(2);
    assert!(Pounce.solve(&m, &ipm).unwrap().has_solution());

    let active = PounceOptions::default()
        .solver_selection(PounceSolverSelection::QpActiveSet)
        .sqp_qp_max_iter(100)
        .sqp_qp_feas_tol(1e-8)
        .sqp_qp_opt_tol(1e-8)
        .sqp_qp_elastic_gamma(1e4)
        .sqp_qp_use_schur_updates(false)
        .sqp_qp_use_homotopy(false)
        .sqp_qp_max_schur_updates_before_refactor(2)
        .sqp_qp_anti_cycling("bland")
        .sqp_qp_certify_second_order(true);
    assert!(Pounce.solve(&m, &active).unwrap().has_solution());
    assert!(Pounce.persistent().solve(&m, &active).unwrap().has_solution());

    for invalid in [
        PounceOptions::default().set("qp_tau", "not-a-number"),
        PounceOptions::default().qp_tau(1.0),
        PounceOptions::default().qp_tau(0.9).qp_tau_max(0.8),
        PounceOptions::default().sqp_qp_anti_cycling("invalid"),
    ] {
        let error =
            Pounce.solve(&m, &invalid.solver_selection(PounceSolverSelection::QpIpm)).unwrap_err();
        assert!(matches!(error, SolverError::Backend(_)), "{error:?}");
    }
}

#[test]
fn persistent_error_discards_resident_state() {
    let m = Model::new("clear_state");
    variable!(m, 0.0 <= x <= 5.0);
    objective!(m, Min, (x - 2.0).powi(2));

    let mut solver = Pounce.persistent();
    assert!(solver.solve(&m, &PounceOptions::default()).unwrap().has_solution());
    let error =
        solver.solve(&m, &PounceOptions::default().set("not_a_real_option", true)).unwrap_err();
    assert!(matches!(error, SolverError::Backend(_)));
    assert!(format!("{solver:?}").contains("resident: false"));
    assert!(solver.solve(&m, &PounceOptions::default()).unwrap().has_solution());
}

#[test]
fn semi_domains_are_rejected_consistently() {
    for (label, model) in
        [("semi-continuous", semi_model(false)), ("semi-integer", semi_model(true))]
    {
        let result = Pounce.solve(&model, &PounceOptions::default()).unwrap_err();
        assert!(matches!(result, SolverError::Backend(_)), "{label} cold: {result}");

        let mut persistent = Pounce.persistent();
        let result = persistent.solve(&model, &PounceOptions::default()).unwrap_err();
        assert!(matches!(result, SolverError::Backend(_)), "{label} persistent: {result}");
    }
}

fn semi_model(integral: bool) -> Model {
    let m = Model::new("semi");
    if integral {
        variable!(m, s <= 10, SemiInt(5.0));
        objective!(m, Min, s);
    } else {
        variable!(m, s <= 10.0, SemiCont(5.0));
        objective!(m, Min, s);
    }
    m
}
