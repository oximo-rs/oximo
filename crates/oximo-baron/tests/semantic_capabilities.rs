use oximo_baron::{Baron, BaronOptions};
use oximo_core::prelude::*;
use oximo_solver::{Solver, SolverError};

#[test]
fn validation_rejects_unsupported_operators_in_objectives_and_constraints() {
    for operator in ["sin", "cbrt", "atan2", "min", "max"] {
        for in_objective in [false, true] {
            let model = Model::new(operator);
            variable!(model, x);
            let expression = match operator {
                "sin" => x.sin(),
                "cbrt" => x.cbrt(),
                "atan2" => x.atan2(x + 1.0),
                "min" => x.min(-x),
                "max" => x.max(-x),
                _ => unreachable!(),
            };
            if in_objective {
                objective!(model, Min, expression);
            } else {
                objective!(model, Min, x);
                constraint!(model, unsupported, expression <= 2.0);
            }

            let mut solver = Baron::with_exec("definitely-not-a-baron");
            let options = BaronOptions::default();
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
    let _unused = x.cbrt();
    let row = constraint!(model, unsupported, x.sin() <= 2.0);
    objective!(model, Min, x.exp());
    model.set_constraint_active(row, false).unwrap();

    let solver = Baron::new();
    let options = BaronOptions::default();
    assert!(solver.validate_model(&model, &options).is_supported());
    model.set_constraint_active(row, true).unwrap();
    assert!(!solver.validate_model(&model, &options).is_supported());
}

#[test]
fn validation_accepts_feasibility_and_rejects_missing_objectives() {
    let model = Model::new("feasibility");
    variable!(model, x >= 0.0);
    constraint!(model, bound, x <= 2.0);
    let solver = Baron::new();
    let options = BaronOptions::default();
    assert!(!solver.validate_model(&model, &options).is_supported());
    objective!(model, Feasibility);
    assert!(solver.validate_model(&model, &options).is_supported());
}

#[test]
fn all_unmet_operator_requirements_are_reported_before_external_execution() {
    let model = Model::new("multiple unsupported operators");

    variable!(model, x);
    constraint!(model, row, x.cbrt() + x.asinh() <= 2.0);
    objective!(model, Min, x.cbrt());

    let options = BaronOptions::default();
    let mut solver = Baron::with_exec("definitely-not-an-installed-executable");
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
