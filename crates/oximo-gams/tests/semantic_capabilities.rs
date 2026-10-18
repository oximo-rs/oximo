use oximo_core::prelude::*;
use oximo_gams::{Gams, GamsOptions, GamsSolver};
use oximo_solver::{Solver, SolverError};

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
            assert!(matches!(
                error,
                SolverError::UnsupportedNonlinearOperator { backend: "GAMS", operator: name }
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
