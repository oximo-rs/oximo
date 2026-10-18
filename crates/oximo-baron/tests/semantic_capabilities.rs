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
            assert!(matches!(
                error,
                SolverError::UnsupportedNonlinearOperator { backend: "BARON", operator: name }
                    if name == operator
            ));
            let report = solver.validate_model(&model, &options);
            assert!(
                report.errors.iter().any(|diagnostic| diagnostic.reason == error.to_string()),
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
