use oximo_clarabel::{Clarabel, ClarabelOptions};
use oximo_core::prelude::*;
use oximo_solver::{PersistentSolver, Requirement, Solver, SolverError};

fn sos_model() -> (Model, SosConstraintHandle) {
    let m = Model::new("clarabel_sos");
    variable!(m, x);
    variable!(m, y);
    objective!(m, Feasibility);
    let choice = sos_constraint!(m, choice, SOS1, [x, y]);
    (m, choice)
}

#[test]
fn clarabel_rejects_sos_constraints() {
    let (model, choice) = sos_model();
    let mut solver = Clarabel;
    let error = solver.solve(&model, &ClarabelOptions::default()).unwrap_err();
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
fn clarabel_persistent_rejects_sos_constraints() {
    let (model, choice) = sos_model();
    let mut solver = Clarabel.persistent();
    let error = solver.solve(&model, &ClarabelOptions::default()).unwrap_err();
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
