use oximo_core::prelude::*;
use oximo_highs::{Highs, HighsOptions};
use oximo_solver::{PersistentSolver, Solver, SolverError};

fn sos_model() -> Model {
    let m = Model::new("highs_sos");
    variable!(m, x);
    variable!(m, y);
    sos_constraint!(m, choice, SOS1, [x, y]);
    m
}

#[test]
fn highs_rejects_sos_constraints() {
    let model = sos_model();
    let mut solver = Highs;
    assert!(matches!(
        solver.solve(&model, &HighsOptions::default()),
        Err(SolverError::UnsupportedSos)
    ));
}

#[test]
fn highs_persistent_rejects_sos_constraints() {
    let model = sos_model();
    let mut solver = Highs.persistent();
    assert!(matches!(
        solver.solve(&model, &HighsOptions::default()),
        Err(SolverError::UnsupportedSos)
    ));
}

#[test]
fn highs_rejects_native_indicators() {
    let model = Model::new("highs_native_indicator");
    variable!(model, b, Binary);
    variable!(model, 0.0 <= x <= 10.0);
    indicator_constraint!(model, cap, b == 1 => x <= 4.0);
    let mut solver = Highs;
    assert!(matches!(
        solver.solve(&model, &HighsOptions::default()),
        Err(SolverError::UnsupportedIndicator)
    ));
}

#[test]
fn highs_solves_reformulated_indicator_sides() {
    fn solve_case(active_value: f64, row_kind: usize, expected: f64) {
        let model = Model::new("highs_indicator_reformulation");
        variable!(model, b, Binary);
        variable!(model, 0.0 <= x <= 10.0);
        model.fix_var(b.var_id().unwrap(), active_value);
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
