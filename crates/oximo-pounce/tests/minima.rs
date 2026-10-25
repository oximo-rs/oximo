// Tests for the Pounce minima search algorithm.
use std::time::Duration;

use oximo_core::{Model, constraint, objective, param, variable};
use oximo_pounce::{
    Pounce, PounceAlgorithm, PounceMinimaMethod, PounceMinimaOptions, PounceOptions,
    PounceSolverSelection,
};
use oximo_solver::{
    DualStatus, PersistentSolver, PrimalStatus, Solver, SolverError, TerminationStatus,
    UniversalOptionsExt,
};

fn search_options(method: PounceMinimaMethod) -> PounceMinimaOptions {
    PounceMinimaOptions {
        method,
        n_minima: 2,
        max_solves: Some(60),
        patience: 30,
        seed: 0,
        step: Some(1.5),
        ..Default::default()
    }
}

#[test]
fn quadratic_search_uses_the_exact_derivative_path() {
    let model = Model::new("quadratic search");

    variable!(model, -3.0 <= x <= 3.0, initial = -1.0);
    variable!(model, -3.0 <= y <= 3.0, initial = 0.0);
    objective!(model, Min, (x - 1.0).powi(2) + (y - 2.0).powi(2));

    let search = PounceMinimaOptions { n_minima: 1, ..Default::default() };

    let result = Pounce.solve(&model, &PounceOptions::default().minima(search)).unwrap();

    assert_eq!(result.termination, TerminationStatus::SolutionLimit);
    assert_eq!(result.solutions.len(), 1);
    assert!((result.solutions[0].value_of(x).unwrap().unwrap() - 1.0).abs() < 1e-5);
    assert!((result.solutions[0].value_of(y).unwrap().unwrap() - 2.0).abs() < 1e-5);
}

#[test]
fn search_finds_optima_with_narrow_nonlinear_domains() {
    for constrained in [false, true] {
        let model = Model::new("narrow search domain");

        variable!(model, -1e-6 <= x <= 1e-6, initial = 0.0);
        if constrained {
            objective!(model, Min, (1e6 * x).powi(2));
            constraint!(model, nonlinear, (1e6 * x).asin() >= 0.3);
        } else {
            objective!(model, Min, ((1e6 * x).asin() - 0.3).powi(2));
        }

        let options = PounceOptions::default().bound_relax_factor(0.0).max_iter(200);

        let search = PounceMinimaOptions {
            method: PounceMinimaMethod::Multistart,
            n_minima: 1,
            max_solves: Some(2),
            ..Default::default()
        };

        let result = Pounce.solve(&model, &options.minima(search)).unwrap();

        assert_eq!(result.termination, TerminationStatus::SolutionLimit, "{result:?}");
        assert_eq!(result.solutions.len(), 1);

        let value = result.solutions[0].value_of(x).unwrap().unwrap();

        assert!((1e6 * value - 0.3_f64.sin()).abs() < 1e-5, "x = {value}, {result:?}");
    }
}

#[test]
fn search_preserves_the_optimum_inside_a_narrow_logarithmic_domain() {
    let model = Model::new("logarithmic search domain");

    variable!(model, 1e-9 <= x <= 1e-7, initial = 1e-8);
    objective!(model, Min, (1e8 * x).ln().powi(2));

    // Allow for fd error in the unscaled stationarity residual.
    let options = PounceOptions::default().bound_relax_factor(0.0).max_iter(500).tol(1e-6);

    let search = PounceMinimaOptions {
        method: PounceMinimaMethod::Multistart,
        n_minima: 1,
        max_solves: Some(1),
        ..Default::default()
    };

    let result = Pounce.solve(&model, &options.minima(search)).unwrap();

    assert_eq!(result.termination, TerminationStatus::SolutionLimit, "{result:?}");
    assert_eq!(result.solutions.len(), 1);

    let minimum = &result.solutions[0];
    let value = minimum.value_of(x).unwrap().unwrap();

    assert!((1e8 * value - 1.0).abs() < 1e-5, "x = {value}, {result:?}");
    assert!(minimum.objective.unwrap() < 1e-10, "{result:?}");
}

#[test]
fn exact_derivatives_allow_minima_at_active_bounds() {
    for maximize in [false, true] {
        let model = Model::new("boundary optima");

        variable!(model, -2.0 <= x <= 2.0, initial = -1.5);
        if maximize {
            objective!(model, Max, x.powi(2));
        } else {
            objective!(model, Min, -x.powi(2));
        }

        let result = Pounce
            .solve(
                &model,
                &PounceOptions::default().minima(search_options(PounceMinimaMethod::Multistart)),
            )
            .unwrap();

        assert_eq!(result.termination, TerminationStatus::SolutionLimit);
        assert_eq!(result.solutions.len(), 2);

        let mut points: Vec<_> =
            result.solutions.iter().map(|minimum| minimum.value_of(x).unwrap().unwrap()).collect();
        points.sort_by(f64::total_cmp);

        assert!((points[0] + 2.0).abs() < 1e-5);
        assert!((points[1] - 2.0).abs() < 1e-5);
        assert_eq!(result.dual_status, DualStatus::FeasiblePoint);
        assert!((result.reduced_costs[&model.variable_id("x").unwrap()].abs() - 4.0).abs() < 1e-4);
        assert!(result.raw_log.as_deref().unwrap().contains("saddle rejection skipped"));
    }
}

#[test]
fn algebraic_constraints_allow_minima_with_negative_objective_curvature() {
    let model = Model::new("constrained curvature");

    variable!(model, -2.0 <= x <= 2.0, initial = 0.5);
    objective!(model, Min, -x.powi(2));
    constraint!(model, interval, x.powi(2) <= 1.0);

    let result = Pounce
        .solve(
            &model,
            &PounceOptions::default().minima(search_options(PounceMinimaMethod::Multistart)),
        )
        .unwrap();

    assert_eq!(result.termination, TerminationStatus::SolutionLimit);
    assert_eq!(result.solutions.len(), 2);

    let mut points: Vec<_> =
        result.solutions.iter().map(|minimum| minimum.value_of(x).unwrap().unwrap()).collect();
    points.sort_by(f64::total_cmp);

    assert!((points[0] + 1.0).abs() < 1e-5);
    assert!((points[1] - 1.0).abs() < 1e-5);
}

#[test]
fn every_strategy_finds_both_wells_and_ranks_their_objectives() {
    let model = Model::new("two wells");

    variable!(model, -2.0 <= x <= 2.0, initial = -1.5);
    objective!(model, Min, (x.powi(2) - 1.0).powi(2) + 0.2 * x + 7.0);

    for method in [
        PounceMinimaMethod::Multistart,
        PounceMinimaMethod::Mlsl,
        PounceMinimaMethod::Basinhopping,
        PounceMinimaMethod::Flooding,
        PounceMinimaMethod::Deflation,
        PounceMinimaMethod::Tunneling,
    ] {
        let result = Pounce
            .solve(&model, &PounceOptions::default().max_iter(300).minima(search_options(method)))
            .unwrap();

        assert_eq!(result.termination, TerminationStatus::SolutionLimit, "{method:?}: {result:?}");
        assert_eq!(result.solutions.len(), 2, "{method:?}");
        assert!(result.iterations > 0);
        assert!(result.solutions[0].objective.unwrap() < result.solutions[1].objective.unwrap());
        assert!(result.solutions[0].value_of(x).unwrap().unwrap() < -0.9);
        assert!(result.solutions[1].value_of(x).unwrap().unwrap() > 0.9);
        assert!(result.best_bound.is_none());
        assert!(result.gap.is_none());
        assert_eq!(result.primal_status, PrimalStatus::FeasiblePoint);
    }
}

#[test]
fn maximization_returns_local_maxima_best_first() {
    let model = Model::new("two peaks");

    variable!(model, -2.0 <= x <= 2.0, initial = -1.5);
    objective!(model, Max, 10.0 - (x.powi(2) - 1.0).powi(2) - 0.2 * x);

    let result = Pounce
        .solve(
            &model,
            &PounceOptions::default().minima(search_options(PounceMinimaMethod::Multistart)),
        )
        .unwrap();

    assert_eq!(result.solutions.len(), 2);
    assert!(result.solutions[0].objective.unwrap() > result.solutions[1].objective.unwrap());
    assert!(result.solutions[0].objective.unwrap() > 10.19);
    assert!(result.solutions[0].value_of(x).unwrap().unwrap() < -0.9);
}

#[test]
fn minima_preserve_parameter_snapshots_constraint_duals_and_reduced_costs() {
    for maximize in [false, true] {
        let model = Model::new("constrained wells");

        param!(model, p = 2.0);
        variable!(model, -2.0 <= x <= 2.0, initial = -1.5);
        variable!(model, 0.0 <= y <= 4.0, initial = 2.5);
        variable!(model, 1.0 <= z <= 2.0, initial = 1.2);
        let expression = (x.powi(2) - 1.0).powi(2) + p * y.powi(2) + z.powi(2);

        if maximize {
            objective!(model, Max, -expression);
        } else {
            objective!(model, Min, expression);
        }

        let lower = constraint!(model, lower, y >= 2.0);

        let result = Pounce
            .solve(
                &model,
                &PounceOptions::default()
                    .max_iter(300)
                    .minima(search_options(PounceMinimaMethod::Multistart)),
            )
            .unwrap();

        assert_eq!(result.solutions.len(), 2);

        model.set_param(p, 3.0).unwrap();
        let sign = if maximize { -1.0 } else { 1.0 };

        assert!((result.dual_of(lower).unwrap().unwrap() - sign * 8.0).abs() < 1e-4);
        assert!((result.reduced_costs[&model.variable_id("z").unwrap()] - sign * 2.0).abs() < 1e-4);
        assert_eq!(result.dual_status, DualStatus::FeasiblePoint);

        for minimum in result.solutions {
            assert!((minimum.value_of(y).unwrap().unwrap() - 2.0).abs() < 1e-5);
            assert!((minimum.value_of(p * y).unwrap().unwrap() - 4.0).abs() < 1e-5);
        }
    }
}

#[test]
fn zero_search_budget_and_time_limit_return_no_minima() {
    let model = Model::new("limited search");

    variable!(model, -2.0 <= x <= 2.0);
    objective!(model, Min, (x.powi(2) - 1.0).powi(2));

    let search = PounceMinimaOptions { max_solves: Some(0), ..Default::default() };

    let result = Pounce.solve(&model, &PounceOptions::default().minima(search)).unwrap();

    assert!(result.solutions.is_empty());
    assert_eq!(result.termination, TerminationStatus::WorkLimit);

    let options = PounceOptions::default().time_limit(Duration::ZERO);

    for method in [
        PounceMinimaMethod::Multistart,
        PounceMinimaMethod::Mlsl,
        PounceMinimaMethod::Basinhopping,
        PounceMinimaMethod::Flooding,
        PounceMinimaMethod::Deflation,
        PounceMinimaMethod::Tunneling,
    ] {
        let result = Pounce.solve(&model, &options.clone().minima(search_options(method))).unwrap();

        assert!(result.solutions.is_empty());
        assert_eq!(result.iterations, 0);
        assert_eq!(result.termination, TerminationStatus::TimeLimit);
    }
}

#[test]
fn failed_local_solves_are_counted_and_reported() {
    let model = Model::new("failed search solves");

    variable!(model, -3.0 <= x <= 3.0, initial = -1.0);
    objective!(model, Min, (x - 1.0).powi(2));

    let options = PounceOptions::default().max_iter(0).solve_quality_audit(false);

    let search = PounceMinimaOptions {
        method: PounceMinimaMethod::Multistart,
        n_minima: 1,
        max_solves: Some(2),
        ..Default::default()
    };

    let result = Pounce.solve(&model, &options.minima(search)).unwrap();

    assert!(result.solutions.is_empty());
    assert_eq!(result.termination, TerminationStatus::WorkLimit);
    assert!(result.raw_log.as_deref().unwrap().contains("local solves did not converge: 2"));
}

#[test]
fn invalid_options_and_unsupported_models_return_errors() {
    let model = Model::new("invalid search");

    variable!(model, -2.0 <= x <= 2.0);
    objective!(model, Min, x.powi(2));

    for search in [
        PounceMinimaOptions { n_minima: 0, ..Default::default() },
        PounceMinimaOptions { dedup: f64::NAN, ..Default::default() },
        PounceMinimaOptions { sigma_per_dim: Some(vec![1.0, 2.0]), ..Default::default() },
        PounceMinimaOptions { n_minima: usize::MAX, ..Default::default() },
        PounceMinimaOptions { n_minima: usize::MAX, max_solves: Some(0), ..Default::default() },
        PounceMinimaOptions {
            method: PounceMinimaMethod::Mlsl,
            samples_per_round: Some(0),
            ..Default::default()
        },
    ] {
        assert!(matches!(
            Pounce.solve(&model, &PounceOptions::default().minima(search)),
            Err(SolverError::Backend(_))
        ));
    }

    for options in [
        PounceOptions::default().solver_selection(PounceSolverSelection::QpIpm),
        PounceOptions::default().set("not_a_pounce_option", true),
        PounceOptions::default().presolve(true).presolve_fbbt(true),
        PounceOptions::default().set("presolve", "yes").set("presolve_fbbt", "yes"),
    ] {
        assert!(matches!(
            Pounce.solve(&model, &options.clone().minima(PounceMinimaOptions::default())),
            Err(SolverError::Backend(_))
        ));

        assert!(matches!(
            Pounce.solve(
                &model,
                &options.time_limit(Duration::ZERO).minima(PounceMinimaOptions::default())
            ),
            Err(SolverError::Backend(_))
        ));
    }

    let integer = Model::new("integer search");

    variable!(integer, 0.0 <= y <= 2.0, Int);
    objective!(integer, Min, y);

    assert!(matches!(
        Pounce.solve(&integer, &PounceOptions::default().minima(PounceMinimaOptions::default())),
        Err(SolverError::UnsupportedKind(_))
    ));
}

#[test]
fn verbose_search_collects_local_iteration_logs() {
    let model = Model::new("logged search");

    variable!(model, -2.0 <= x <= 2.0, initial = -1.5);
    objective!(model, Min, (x.powi(2) - 1.0).powi(2));

    let result = Pounce
        .solve(
            &model,
            &PounceOptions::default()
                .verbose(true)
                .minima(PounceMinimaOptions { n_minima: 1, ..Default::default() }),
        )
        .unwrap();

    assert_eq!(result.solutions.len(), 1);

    let log = result.raw_log.unwrap();

    assert!(log.contains("Iteration history:"));
    assert!(log.contains("Local solve converged: true"));
    assert!(log.contains("Search solves: 1"));
    assert!(log.contains("Failed local solves: 0"));
    assert!(log.contains("Failed dual recoveries: 0"));
    assert!(log.contains("Search scale: [4.0]"));
}

#[test]
fn persistent_handles_switch_between_single_solves_and_minima_searches() {
    let model = Model::new("persistent minima");

    param!(model, p = 1.0);
    variable!(model, -2.0 <= x <= 2.0, initial = -1.5);
    objective!(model, Min, (x.powi(2) - p).powi(2));

    let mut solver = Pounce.persistent();
    let single = PounceOptions::default();
    let search = single.clone().minima(search_options(PounceMinimaMethod::Multistart));

    assert_eq!(solver.solve(&model, &single).unwrap().result_count(), 1);

    let found = solver.solve(&model, &search).unwrap();

    assert_eq!(found.termination, TerminationStatus::SolutionLimit);
    assert_eq!(found.result_count(), 2);
    assert_eq!(found.raw_status.as_deref(), Some("MinimaTargetReached"));

    model.set_param(p, 0.25).unwrap();
    let rebuilt = solver.solve(&model, &single).unwrap();

    assert_eq!(rebuilt.result_count(), 1);
    assert!((rebuilt.value_of(x).unwrap().unwrap().abs() - 0.5).abs() < 1e-5);

    let found = solver.solve(&model, &search).unwrap();

    assert_eq!(found.result_count(), 2);

    for point in &found.solutions {
        assert!((point.value_of(x).unwrap().unwrap().abs() - 0.5).abs() < 1e-5);
    }
    assert!(solver.solve(&model, &search.clone().set("invalid_option", true)).is_err());
    assert_eq!(solver.solve(&model, &single).unwrap().result_count(), 1);
}

#[test]
fn minima_route_validation_agrees_for_cold_and_persistent_handles() {
    let model = Model::new("search routes");

    variable!(model, -2.0 <= x <= 2.0);
    objective!(model, Min, x.powi(2));

    let search = PounceOptions::default().minima(PounceMinimaOptions::default());
    let mut persistent = Pounce.persistent();

    assert!(Pounce.validate_model(&model, &search).is_supported());
    assert!(persistent.validate_model(&model, &search).is_supported());

    for selection in [
        PounceSolverSelection::LpIpm,
        PounceSolverSelection::QpIpm,
        PounceSolverSelection::QpActiveSet,
        PounceSolverSelection::Socp,
    ] {
        let forced = search.clone().solver_selection(selection);

        assert!(!Pounce.validate_model(&model, &forced).is_supported());
        assert!(!persistent.validate_model(&model, &forced).is_supported());
        assert!(Pounce.solve(&model, &forced).is_err());
        assert!(persistent.solve(&model, &forced).is_err());
    }

    model.add_soc_constraint("cone", [x], x + 2.0);

    assert!(!Pounce.validate_model(&model, &search).is_supported());
    assert!(!persistent.validate_model(&model, &search).is_supported());
    assert!(Pounce.solve(&model, &search).is_err());
    assert!(persistent.solve(&model, &search).is_err());
}

#[test]
fn search_local_solves_and_dual_recovery_honor_limited_memory() {
    let model = Model::new("search Hessian options");

    variable!(model, -3.0 <= x <= 3.0, initial = -1.0);
    objective!(model, Min, (x - 1.0).powi(2));
    constraint!(model, lower, x >= 0.5);

    for hessian in ["exact", "limited-memory", "EXACT", "LIMITED-MEMORY"] {
        let options = PounceOptions::default()
            .set("hessian_approximation", hessian)
            .verbose(true)
            .print_level(0)
            .minima(PounceMinimaOptions { n_minima: 1, max_solves: Some(1), ..Default::default() });

        let result = Pounce.solve(&model, &options).unwrap();

        assert_eq!(result.termination, TerminationStatus::SolutionLimit);
        assert_eq!(result.dual_status, DualStatus::FeasiblePoint);
        assert!((result.value_of(x).unwrap().unwrap() - 1.0).abs() < 1e-5);

        let log = result.raw_log.unwrap();
        let hessian_evaluations: Vec<u64> = log
            .lines()
            .filter(|line| line.starts_with("Number of Lagrangian Hessian evaluations"))
            .map(|line| line.split_once('=').unwrap().1.trim().parse().unwrap())
            .collect();

        assert_eq!(hessian_evaluations.len(), 2, "search solve and dual recovery: {log}");

        if hessian.eq_ignore_ascii_case("limited-memory") {
            assert!(hessian_evaluations.iter().all(|&count| count == 0), "{log}");
        } else {
            assert!(hessian_evaluations.iter().any(|&count| count > 0), "{log}");
        }
    }
}

#[test]
fn unsupported_search_hessian_modes_fail_validation_and_solve() {
    let model = Model::new("unsupported search Hessian modes");

    variable!(model, -2.0 <= x <= 2.0);
    objective!(model, Min, x.powi(2));

    for hessian in ["partitioned", "finite-difference"] {
        let options = PounceOptions::default()
            .set("hessian_approximation", hessian)
            .minima(PounceMinimaOptions::default());

        assert!(!Pounce.validate_model(&model, &options).is_supported());
        let error = Pounce.solve(&model, &options).unwrap_err().to_string();

        assert!(error.contains("hessian_approximation"));
        assert!(error.contains(hessian));
    }
}

#[test]
fn minima_fbbt_validation_respects_effective_options() {
    let model = Model::new("search FBBT compatibility");

    variable!(model, -2.0 <= x <= 2.0);
    objective!(model, Min, x.powi(2));

    let search = PounceOptions::default()
        .minima(PounceMinimaOptions { max_solves: Some(0), ..Default::default() });

    for options in [
        search.clone().presolve(true).presolve_fbbt(true),
        search.clone().set("presolve", "yes").set("presolve_fbbt", "yes"),
        search.clone().set("PRESOLVE", "YES").set("PRESOLVE_FBBT", "YES"),
        search.clone().presolve(true).presolve_fbbt(false).set("presolve_fbbt", true),
        search.clone().presolve(false).presolve_fbbt(true).set("presolve", "yes"),
    ] {
        let mut persistent = Pounce.persistent();

        assert!(!Pounce.validate_model(&model, &options).is_supported());
        assert!(!persistent.validate_model(&model, &options).is_supported());

        for result in [Pounce.solve(&model, &options), persistent.solve(&model, &options)] {
            assert!(result.unwrap_err().to_string().contains("presolve_fbbt"));
        }
    }

    for options in [
        search.clone().presolve_fbbt(true),
        search.clone().presolve(true).presolve_fbbt(false),
        search.clone().presolve(true).presolve_fbbt(true).set("presolve_fbbt", "no"),
        search.clone().presolve(true).presolve_fbbt(true).set("presolve", false),
        search.clone().presolve(true).presolve_fbbt(true).set("PRESOLVE_FBBT", "NO"),
        search.clone().set("presolve", true).set("presolve", false).presolve_fbbt(true),
    ] {
        assert!(Pounce.validate_model(&model, &options).is_supported());
        assert!(Pounce.persistent().validate_model(&model, &options).is_supported());
        assert_eq!(
            Pounce.solve(&model, &options).unwrap().termination,
            TerminationStatus::WorkLimit
        );
    }
}

#[test]
fn searches_retain_minima_when_solve_sample_or_patience_limits_stop_them() {
    let model = Model::new("partial search");

    param!(model, p = 2.0);
    variable!(model, -3.0 <= x <= 3.0, initial = -1.0);
    objective!(model, Min, p * (x - 1.0).powi(2));

    let configurations = [
        (
            PounceMinimaOptions {
                method: PounceMinimaMethod::Multistart,
                n_minima: 2,
                max_solves: Some(1),
                ..Default::default()
            },
            TerminationStatus::WorkLimit,
            "MinimaBudgetExhausted",
        ),
        (
            PounceMinimaOptions {
                method: PounceMinimaMethod::Mlsl,
                n_minima: 2,
                max_samples: Some(0),
                ..Default::default()
            },
            TerminationStatus::WorkLimit,
            "MinimaSampleCapReached",
        ),
        (
            PounceMinimaOptions {
                method: PounceMinimaMethod::Multistart,
                n_minima: 2,
                patience: 1,
                ..Default::default()
            },
            TerminationStatus::LocallyOptimal,
            "MinimaConverged",
        ),
    ];

    for (search, termination, raw_status) in configurations {
        let options = PounceOptions::default().minima(search);

        let result = Pounce.solve(&model, &options).unwrap();

        assert_eq!(result.termination, termination, "{result:?}");
        assert_eq!(result.raw_status.as_deref(), Some(raw_status));
        assert_eq!(result.result_count(), 1);
        assert_eq!(result.primal_status, PrimalStatus::FeasiblePoint);
        assert!((result.value_of(x).unwrap().unwrap() - 1.0).abs() < 1e-5);
        assert!(result.objective().unwrap().abs() < 1e-10);
        assert_eq!(result.dual_status, DualStatus::FeasiblePoint);
        assert!(result.best_bound.is_none());
        assert!(result.gap.is_none());
        model.set_param(p, 3.0).unwrap();

        assert_eq!(result.value_of(p).unwrap(), Some(2.0));
        model.set_param(p, 2.0).unwrap();
    }
}

#[test]
fn mixed_case_route_and_algorithm_overrides_work_for_minima_search() {
    let model = Model::new("mixed-case search options");

    variable!(model, -2.0 <= x <= 2.0, initial = 0.5);
    objective!(model, Min, (x - 1.0).powi(2));

    let search = PounceOptions::default().minima(PounceMinimaOptions {
        n_minima: 1,
        max_solves: Some(1),
        ..Default::default()
    });

    for options in [
        search
            .clone()
            .solver_selection(PounceSolverSelection::QpIpm)
            .set("SOLVER_SELECTION", "NLP"),
        search.clone().set("solver_selection", "NLP"),
        search.clone().set("SOLVER_SELECTION", "QP-IPM").set("SoLvEr_SeLeCtIoN", "nLp"),
        search.clone().algorithm(PounceAlgorithm::ActiveSetSqp).set("ALGORITHM", "INTERIOR-POINT"),
        search.clone().algorithm(PounceAlgorithm::InteriorPoint).set("AlGoRiThM", "AcTiVe-SeT-SqP"),
    ] {
        let mut persistent = Pounce.persistent();

        assert!(Pounce.validate_model(&model, &options).is_supported());
        assert!(persistent.validate_model(&model, &options).is_supported());

        for result in [Pounce.solve(&model, &options), persistent.solve(&model, &options)] {
            let result = result.unwrap();

            assert_eq!(result.termination, TerminationStatus::SolutionLimit);
            assert_eq!(result.result_count(), 1);
            assert!((result.value_of(x).unwrap().unwrap() - 1.0).abs() < 1e-5);
        }
    }

    for options in [
        search.clone().set("SOLVER_SELECTION", "qp-ipm"),
        search.clone().set("SoLvEr_SeLeCtIoN", "QP-IPM"),
        search
            .clone()
            .solver_selection(PounceSolverSelection::Nlp)
            .set("solver_selection", "NLP")
            .set("SOLVER_SELECTION", "QP-IPM"),
    ] {
        let mut persistent = Pounce.persistent();

        assert!(!Pounce.validate_model(&model, &options).is_supported());
        assert!(!persistent.validate_model(&model, &options).is_supported());

        for result in [Pounce.solve(&model, &options), persistent.solve(&model, &options)] {
            assert!(
                result.unwrap_err().to_string().contains("requires solver_selection auto or nlp")
            );
        }
    }
}
