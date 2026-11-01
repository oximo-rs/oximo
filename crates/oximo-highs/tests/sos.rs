use oximo_core::prelude::*;
use oximo_highs::{Highs, HighsOptions};
use oximo_solver::{ConstraintSignature, PersistentSolver, Requirement, Solver, SolverError};

fn sos_model() -> (Model, SosConstraintHandle) {
    let m = Model::new("highs_sos");
    variable!(m, x);
    variable!(m, y);
    objective!(m, Feasibility);
    let choice = sos_constraint!(m, choice, SOS1, [x, y]);
    (m, choice)
}

#[test]
fn highs_rejects_sos_constraints() {
    let (model, choice) = sos_model();
    let mut solver = Highs;
    let error = solver.solve(&model, &HighsOptions::default()).unwrap_err();
    assert!(
        matches!(
            &error,
            SolverError::IncompatibleModel(report) if report.errors.iter().any(|diagnostic|
                diagnostic.requirement == Requirement::Sos(SosType::Sos1)
                    && diagnostic.constraint() == Some(choice.id()))
        ),
        "{error}"
    );
}

#[test]
fn highs_persistent_rejects_sos_constraints() {
    let (model, choice) = sos_model();
    let mut solver = Highs.persistent();
    let error = solver.solve(&model, &HighsOptions::default()).unwrap_err();
    assert!(
        matches!(
            &error,
            SolverError::IncompatibleModel(report) if report.errors.iter().any(|diagnostic|
                diagnostic.requirement == Requirement::Sos(SosType::Sos1)
                    && diagnostic.constraint() == Some(choice.id()))
        ),
        "{error}"
    );
}

#[test]
fn highs_rejects_native_indicators() {
    let model = Model::new("highs_native_indicator");
    variable!(model, b, Binary);
    variable!(model, 0.0 <= x <= 10.0);
    objective!(model, Feasibility);
    let cap = indicator_constraint!(model, cap, b == 1 => x <= 4.0);
    let expected = Requirement::Constraint(ConstraintSignature::of::<
        oximo_core::IndicatorFunctionData,
        oximo_core::IndicatorSet,
    >());
    let mut persistent = Highs.persistent();
    for error in [
        Highs.solve(&model, &HighsOptions::default()).unwrap_err(),
        persistent.solve(&model, &HighsOptions::default()).unwrap_err(),
    ] {
        assert!(
            matches!(
                &error,
                SolverError::IncompatibleModel(report) if report.errors.len() == 1
                    && report.errors[0].requirement == expected
                    && report.errors[0].constraint() == Some(cap.id())
            ),
            "{error}"
        );
    }
}

#[test]
fn highs_solves_reformulated_indicator_sides() {
    fn solve_case(active_value: f64, row_kind: usize, expected: f64) {
        let model = Model::new("highs_indicator_reformulation");
        variable!(model, b, Binary);
        variable!(model, 0.0 <= x <= 10.0);
        model.fix_var(b.var_id().unwrap(), active_value).unwrap();
        match row_kind {
            0 => {
                indicator_constraint!(model, cap, b == 1 => x <= 4.0);
            }
            1 => {
                indicator_constraint!(model, equal, b == 1 => x == 3.0);
            }
            _ => {
                indicator_constraint!(model, band, b == 1 => 2.0 <= x <= 4.0);
            }
        }
        objective!(model, Max, x);
        let transformed = model
            .to_reformulated_indicator_model(IndicatorReformulationOptions::default())
            .unwrap();
        let mut solver = Highs;
        let result = solver.solve(&transformed, &HighsOptions::default()).unwrap();
        let tx = transformed.variable_handle(x.var_id().unwrap());
        assert!((result.value_of(tx).unwrap().unwrap() - expected).abs() < 1e-7);
    }

    solve_case(1.0, 0, 4.0);
    solve_case(0.0, 0, 10.0);
    solve_case(1.0, 1, 3.0);
    solve_case(1.0, 2, 4.0);
}

#[test]
fn highs_solves_explicit_sos1_reformulation() {
    let model = Model::new("highs_reformulated_sos1");
    variable!(model, 0.0 <= x <= 1.0);
    variable!(model, 0.0 <= y <= 1.0);
    objective!(model, Max, x + y);
    sos_constraint!(model, choice, SOS1, [x, y]);
    let transformed = model.to_reformulated_sos_model(SosReformulationOptions::default()).unwrap();

    let mut solver = Highs;
    let result = solver.solve(&transformed, &HighsOptions::default()).unwrap();
    let tx = transformed.variable_handle(x.var_id().unwrap());
    let ty = transformed.variable_handle(y.var_id().unwrap());
    assert!((result.objective().unwrap() - 1.0).abs() < 1e-7);
    assert!(
        result.value_of(tx).unwrap().unwrap() + result.value_of(ty).unwrap().unwrap() <= 1.0 + 1e-7
    );
}

#[test]
fn highs_solves_weight_ordered_sos2_reformulation() {
    let model = Model::new("highs_reformulated_sos2");
    variable!(model, 0.0 <= x <= 1.0);
    variable!(model, 0.0 <= y <= 1.0);
    variable!(model, 0.0 <= z <= 1.0);
    objective!(model, Max, x + z);
    sos_constraint!(model, adjacent, SOS2, [(z, 3.0), (x, 1.0), (y, 2.0)]);
    let transformed = model.to_reformulated_sos_model(SosReformulationOptions::default()).unwrap();

    let mut solver = Highs;
    let result = solver.solve(&transformed, &HighsOptions::default()).unwrap();
    let tx = transformed.variable_handle(x.var_id().unwrap());
    let tz = transformed.variable_handle(z.var_id().unwrap());
    assert!((result.objective().unwrap() - 1.0).abs() < 1e-7);
    assert!(
        result.value_of(tx).unwrap().unwrap() + result.value_of(tz).unwrap().unwrap() <= 1.0 + 1e-7
    );
}
