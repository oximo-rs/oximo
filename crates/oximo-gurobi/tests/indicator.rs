use oximo_core::prelude::*;
use oximo_gurobi::{Gurobi, GurobiOptions};
use oximo_solver::Solver;

#[test]
fn native_indicators_enforce_only_the_selected_branch() {
    for (fixed, expected) in [(1.0, 3.0), (0.0, 10.0)] {
        let model = Model::new("indicator");
        variable!(model, b, Binary);
        variable!(model, 0.0 <= x <= 10.0);
        constraint!(model, fix, b == fixed);
        indicator_constraint!(model, cap, b == 1 => x <= 3.0);
        objective!(model, Max, x);
        assert!(Gurobi.supports_indicators());
        let result = Gurobi.solve(&model, &GurobiOptions::default()).unwrap();
        assert!((result.value_of(x).unwrap().unwrap() - expected).abs() < 1e-6);
    }
}

#[test]
fn native_indicator_equality_and_range() {
    let model = Model::new("indicator_forms");
    variable!(model, b, Binary);
    variable!(model, -10.0 <= x <= 10.0);
    variable!(model, -10.0 <= y <= 10.0);
    constraint!(model, fix, b == 0.0);
    indicator_constraint!(model, equal, b == 0 => x == 2.0);
    indicator_constraint!(model, band, b == 0 => -1.0 <= y <= 4.0);
    objective!(model, Max, x + y);
    let result = Gurobi.solve(&model, &GurobiOptions::default()).unwrap();
    assert!((result.value_of(x).unwrap().unwrap() - 2.0).abs() < 1e-6);
    assert!((result.value_of(y).unwrap().unwrap() - 4.0).abs() < 1e-6);
}
