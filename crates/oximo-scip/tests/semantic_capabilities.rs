use oximo_core::prelude::*;
use oximo_core::{IndicatorFunctionData, IndicatorSet};
use oximo_scip::{Scip, ScipOptions};
use oximo_solver::{ConstraintSignature, PersistentSolver, RequirementSupport, Solver};

fn assert_validated_solve(
    solver: &mut dyn Solver<Options = ScipOptions>,
    model: &Model,
    options: &ScipOptions,
    expected: f64,
) {
    assert!(solver.validate_model(model, options).is_supported());
    let validation = solver.validate_model(model, options);
    assert!(validation.is_supported(), "{validation:?}");
    let result = solver.solve(model, options).unwrap();
    assert_eq!(result.model_id(), model.id());
    assert!((result.objective().unwrap() - expected).abs() < 1e-5);
}

#[test]
fn native_indicators_validate_and_translate_for_cold_and_persistent_solvers() {
    let model = Model::new("indicator capabilities");
    variable!(model, 0.0 <= x <= 10.0);
    variable!(model, enabled, Bin);
    model.fix(enabled, 1.0).unwrap();
    indicator_constraint!(model, cap, enabled == 1 => x <= 2.0);
    objective!(model, Max, x);
    let options = ScipOptions::default();
    let mut cold = Scip::new();
    let mut persistent = cold.persistent();

    for solver in [&mut cold as &mut dyn Solver<Options = ScipOptions>, &mut persistent] {
        assert_eq!(
            solver
                .profile(&options)
                .constraint_support(
                    ConstraintSignature::of::<IndicatorFunctionData, IndicatorSet>()
                ),
            RequirementSupport::Supported
        );
        assert_validated_solve(solver, &model, &options, 2.0);
    }
}

#[test]
fn semi_domains_validate_and_translate_for_cold_and_persistent_solvers() {
    for domain in
        [Domain::SemiContinuous { threshold: 1.0 }, Domain::SemiInteger { threshold: 1.0 }]
    {
        let model = Model::new("semi domain capabilities");
        let value = model.__var("value").domain(domain).lb(0.0).ub(4.0).build();
        constraint!(model, positive, value >= 0.5);
        objective!(model, Min, value);
        let options = ScipOptions::default();
        let mut cold = Scip::new();
        let mut persistent = cold.persistent();

        for solver in [&mut cold as &mut dyn Solver<Options = ScipOptions>, &mut persistent] {
            assert_eq!(
                solver.profile(&options).domain_support(domain),
                RequirementSupport::Supported
            );
            assert_validated_solve(solver, &model, &options, 1.0);
        }
    }
}

#[test]
fn unsupported_operators_fail_validation_and_translation_for_both_solvers() {
    for in_objective in [true, false] {
        let model = Model::new("unsupported operator capabilities");
        variable!(model, 0.0 <= x <= 1.0);
        if in_objective {
            objective!(model, Min, x.tan());
        } else {
            objective!(model, Min, x);
            constraint!(model, unsupported, x.tan() <= 1.0);
        }
        let options = ScipOptions::default();
        let mut cold = Scip::new();
        let mut persistent = cold.persistent();

        for solver in [&mut cold as &mut dyn Solver<Options = ScipOptions>, &mut persistent] {
            assert!(!solver.validate_model(&model, &options).is_supported());
            let validation = solver.validate_model(&model, &options);
            assert!(!validation.is_supported());
            assert!(validation.errors.iter().any(|diagnostic| matches!(&diagnostic.requirement, oximo_solver::Requirement::Operator(operator) if operator.to_string() == "tan")));
            assert!(solver.solve(&model, &options).unwrap_err().to_string().contains("tan"));
        }
    }
}

#[test]
fn aggregate_preflight_runs_before_plugin_callbacks_and_detailed_solves() {
    use std::{cell::Cell, rc::Rc};
    let calls = Rc::new(Cell::new(0));
    let observed = Rc::clone(&calls);
    let mut solver = Scip::new().with_plugins(move |_| {
        observed.set(observed.get() + 1);
        Ok(())
    });
    let mut persistent = solver.persistent();

    let model = Model::new("reject before plugins");
    variable!(model, x);
    constraint!(model, unsupported, x.tan() <= 1.0);
    objective!(model, Min, x.atan2(x));

    let options = ScipOptions::default();
    let expected = solver.validate_model(&model, &options).to_string();
    let error = solver.solve(&model, &options).unwrap_err();
    assert!(
        matches!(&error, oximo_solver::SolverError::IncompatibleModel(report) if report.errors.len() == 2)
    );
    assert_eq!(error.to_string(), expected);
    assert_eq!(solver.solve_detailed(&model, &options).unwrap_err().to_string(), expected);
    assert_eq!(persistent.solve(&model, &options).unwrap_err().to_string(), expected);
    assert_eq!(persistent.solve_detailed(&model, &options).unwrap_err().to_string(), expected);
    assert_eq!(calls.get(), 0);
}

#[test]
fn parameter_domain_errors_are_rechecked_without_discarding_native_state() {
    let model = Model::new("cached structure and live numeric data");
    variable!(model, 1.0 <= x <= 3.0);
    param!(model, p = 1.0);
    objective!(model, Min, p.sqrt() * x);

    let options = ScipOptions::default();
    let mut solver = Scip::new().persistent();
    let first = solver.solve(&model, &options).unwrap();
    assert!((first.objective().unwrap() - 1.0).abs() < 1e-7);

    p.set_param_value(-1.0).unwrap();
    let expected = solver.validate_model(&model, &options).to_string();
    let error = solver.solve(&model, &options).unwrap_err();
    assert!(matches!(error, oximo_solver::SolverError::IncompatibleModel(_)));
    assert_eq!(error.to_string(), expected);
    assert_eq!(solver.build_count(), 1);

    p.set_param_value(1.0).unwrap();
    let restored = solver.solve(&model, &options).unwrap();
    assert!((restored.objective().unwrap() - 1.0).abs() < 1e-7);
    assert_eq!(solver.build_count(), 1);
    assert_eq!(solver.reuse_count(), 1);

    p.set_param_value(4.0).unwrap();
    let changed = solver.solve(&model, &options).unwrap();
    assert!((changed.objective().unwrap() - 2.0).abs() < 1e-7);
}
