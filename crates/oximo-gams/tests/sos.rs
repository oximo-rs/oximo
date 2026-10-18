use oximo_core::prelude::*;
use oximo_gams::Gams;
use oximo_solver::Solver;

#[test]
fn gams_advertises_sos1_and_sos2_support() {
    let model = Model::new("gams_sos");
    variable!(model, x);
    variable!(model, y);
    sos_constraint!(model, choice, SOS1, [(x, 1.0), (y, 2.0)]);
    sos_constraint!(model, adjacent, SOS2, [(x, 1.0), (y, 2.0)]);

    let solver = Gams::new();
    assert!(solver.supports_sos(SosType::Sos1));
    assert!(solver.supports_sos(SosType::Sos2));
    assert!(solver.supports_model(&model));
}

#[test]
fn gams_accepts_a_reformulated_model_without_active_sos() {
    let model = Model::new("gams_reformulated_sos");
    variable!(model, 0.0 <= x <= 1.0);
    variable!(model, 0.0 <= y <= 1.0);
    objective!(model, Max, x + y);
    let choice = sos_constraint!(model, choice, SOS1, [x, y]);
    let transformed =
        choice.to_reformulated_model(&model, SosReformulationOptions::default()).unwrap();

    let solver = Gams::new();
    assert!(!transformed.has_active_sos_constraints());
    assert!(solver.supports_model(&transformed));
}
