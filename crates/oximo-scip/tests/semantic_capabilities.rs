use oximo_core::prelude::*;
use oximo_core::{IndicatorFunctionData, IndicatorSet};
use oximo_scip::{Scip, ScipOptions};
use oximo_solver::{Capability, ConstraintSignature, PersistentSolver, Solver};

fn assert_validated_solve(
    solver: &mut dyn Solver<Options = ScipOptions>,
    model: &Model,
    options: &ScipOptions,
    expected: f64,
) {
    assert!(solver.supports_model(model));
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
            solver.constraint_support(
                ConstraintSignature::of::<IndicatorFunctionData, IndicatorSet>(),
                &options,
            ),
            Capability::Native
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
            assert_eq!(solver.domain_support(domain), Capability::Native);
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
            assert!(!solver.supports_model(&model));
            let validation = solver.validate_model(&model, &options);
            assert!(!validation.is_supported());
            assert!(validation.errors.iter().any(|diagnostic| diagnostic.reason.contains("tan")));
            assert!(solver.solve(&model, &options).unwrap_err().to_string().contains("tan"));
        }
    }
}
