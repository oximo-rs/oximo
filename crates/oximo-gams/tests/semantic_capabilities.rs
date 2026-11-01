use oximo_core::prelude::*;
use oximo_gams::{Gams, GamsOptions, GamsSolver};
use oximo_solver::{Solver, SolverError};

#[test]
fn mixed_integer_solver_profiles_accept_affine_interfaces_without_lp_support() {
    for selected in [GamsSolver::Scip, GamsSolver::OdhCplex] {
        let model = Model::new("affine pieces of a MILP");
        variable!(model, 0.0 <= x <= 2.0, Integer);
        constraint!(model, cap, x <= 1.0);
        objective!(model, Min, x);

        let options = GamsOptions::default().solver(selected);
        let mut solver = Gams::with_exec("definitely-not-a-gams-executable");
        assert!(!solver.profile(&options).supports_kind(ModelKind::LP));
        assert!(solver.validate_model(&model, &options).is_confirmed());
        assert!(matches!(solver.solve(&model, &options), Err(SolverError::Backend(_))));
    }
    for selected in [GamsSolver::Dicopt, GamsSolver::AlphaEcp] {
        let model = Model::new("affine pieces of a MINLP");

        variable!(model, 1.0 <= x <= 2.0, Integer);
        constraint!(model, affine, x <= 2.0);
        constraint!(model, nonlinear, x.exp() <= 8.0);
        objective!(model, Min, x);

        let options = GamsOptions::default().solver(selected);
        let mut solver = Gams::with_exec("definitely-not-a-gams-executable");
        assert!(!solver.profile(&options).supports_kind(ModelKind::LP));
        assert!(solver.validate_model(&model, &options).is_confirmed());
        assert!(matches!(solver.solve(&model, &options), Err(SolverError::Backend(_))));
    }
}

#[test]
fn affine_interface_support_does_not_enable_an_unsupported_gams_solve_type() {
    let model = Model::new("SCIP cannot be selected for an LP solve");

    variable!(model, x);
    constraint!(model, cap, x <= 1.0);
    objective!(model, Min, x);

    let options = GamsOptions::default().solver(GamsSolver::Scip);
    let solver = Gams::new();
    let report = solver.validate_model(&model, &options);
    assert_eq!(report.errors.len(), 1);
    assert!(matches!(report.errors[0].requirement,
        oximo_solver::Requirement::Combination(features) if features.kind == ModelKind::LP));
}

#[test]
fn validation_rejects_unsupported_operators_in_objectives_and_constraints() {
    for operator in ["cbrt", "expm1", "log1p", "asinh", "acosh", "atanh"] {
        for in_objective in [false, true] {
            let model = Model::new(operator);
            variable!(model, x);
            let expression = match operator {
                "cbrt" => x.cbrt(),
                "expm1" => x.expm1(),
                "log1p" => x.log1p(),
                "asinh" => x.asinh(),
                "acosh" => x.acosh(),
                "atanh" => x.atanh(),
                _ => unreachable!(),
            };
            if in_objective {
                objective!(model, Min, expression);
            } else {
                objective!(model, Min, x);
                constraint!(model, unsupported, expression <= 2.0);
            }

            let mut solver = Gams::with_exec("definitely-not-a-gams");
            let options = GamsOptions::default();
            let error = solver.solve(&model, &options).unwrap_err();
            let SolverError::IncompatibleModel(failed) = &error else { panic!("{error}") };
            assert!(failed.errors.iter().any(|diagnostic| matches!(&diagnostic.requirement, oximo_solver::Requirement::Operator(op) if op.to_string() == operator)));
            let report = solver.validate_model(&model, &options);
            assert!(
                report.errors.iter().any(|diagnostic| matches!(&diagnostic.requirement, oximo_solver::Requirement::Operator(op) if op.to_string() == operator)),
                "missing translation error for {operator} (objective={in_objective}): {report:?}"
            );
        }
    }
}

#[test]
fn validation_checks_only_active_expressions_and_observes_reactivation() {
    let model = Model::new("active expressions");
    variable!(model, x);
    let _unused = x.asinh();
    let row = constraint!(model, unsupported, x.cbrt() <= 2.0);
    objective!(model, Min, x.exp());
    model.set_constraint_active(row, false).unwrap();

    let solver = Gams::new();
    let options = GamsOptions::default();
    assert!(solver.validate_model(&model, &options).is_supported());
    model.set_constraint_active(row, true).unwrap();
    assert!(!solver.validate_model(&model, &options).is_supported());
}

#[test]
fn validation_uses_the_nonsmooth_solve_type_for_subsolver_diagnostics() {
    let model = Model::new("nonsmooth routing");
    variable!(model, x);
    objective!(model, Min, x.abs());

    let report =
        Gams::new().validate_model(&model, &GamsOptions::default().solver(GamsSolver::Cplex));
    assert!(report.errors.iter().any(|diagnostic| diagnostic.reason.contains("DNLP")));
    assert!(
        Gams::new()
            .validate_model(&model, &GamsOptions::default().solver(GamsSolver::Ipopt))
            .is_supported()
    );
}

#[test]
fn all_unmet_operator_requirements_are_reported_before_external_execution() {
    let model = Model::new("multiple unsupported operators");

    variable!(model, x);
    constraint!(model, row, x.cbrt() + x.asinh() <= 2.0);
    objective!(model, Min, x.cbrt());

    let options = GamsOptions::default();
    let mut solver = Gams::with_exec("definitely-not-an-installed-executable");
    let error = solver.solve(&model, &options).unwrap_err();
    let SolverError::IncompatibleModel(report) = error else { panic!("{error}") };
    let operators: Vec<_> = report
        .errors
        .iter()
        .filter(|d| matches!(d.requirement, oximo_solver::Requirement::Operator(_)))
        .collect();
    assert_eq!(operators.len(), 2);

    let cbrt = operators
        .iter()
        .find(|d| {
            d.requirement
                == oximo_solver::Requirement::Operator(oximo_solver::ExpressionOperator::Unary(
                    oximo_expr::UnaryOp::Cbrt,
                ))
        })
        .unwrap();
    assert_eq!(cbrt.origins.len(), 2);
    assert_eq!(report.to_string(), solver.validate_model(&model, &options).to_string());
}

#[test]
fn unknown_external_solver_is_conditional_and_can_still_be_invoked() {
    let model = Model::new("custom external solver");

    variable!(model, x);
    objective!(model, Min, x);

    let options = GamsOptions::default().solver(GamsSolver::Custom("UNKNOWN_CUSTOM_SOLVER".into()));
    let mut solver = Gams::with_exec("definitely-not-a-gams-executable");
    let report = solver.validate_model(&model, &options);
    assert!(report.is_supported());
    assert!(!report.is_confirmed());
    assert_eq!(report.conditions.len(), 1);
    assert!(matches!(solver.solve(&model, &options), Err(SolverError::Backend(_))));
}

#[test]
fn custom_solver_alias_cannot_advertise_uninstalled_indicator_mappings() {
    let model = Model::new("custom indicator interface");

    variable!(model, x);
    variable!(model, enabled, Bin);
    indicator_constraint!(model, cap, enabled == 1 => x <= 2.0);
    objective!(model, Min, x);

    let solver = Gams::new();
    let custom = GamsOptions::default().solver(GamsSolver::Custom("CPLEX".into()));
    let report = solver.profile(&custom).check(&oximo_solver::ModelRequirements::infer(&model));
    assert!(!report.is_supported());
    assert!(report.errors.iter().any(|diagnostic| matches!(diagnostic.requirement, oximo_solver::Requirement::Constraint(signature) if signature.set.is::<oximo_core::IndicatorSet>())));

    let named = GamsOptions::default().solver(GamsSolver::Cplex);
    assert!(solver.validate_model(&model, &named).is_confirmed());
}
