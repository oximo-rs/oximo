use oximo_baron::{Baron, BaronOptions};
use oximo_core::prelude::*;
use oximo_solver::{Requirement, Solver, SolverError};

#[test]
fn baron_rejects_sos_constraints_before_launching_executable() {
    let model = Model::new("baron_sos");
    variable!(model, x);
    variable!(model, y);
    objective!(model, Feasibility);
    let choice = sos_constraint!(model, choice, SOS1, [x, y]);
    let mut solver = Baron::with_exec("definitely-not-a-baron");
    let error = solver.solve(&model, &BaronOptions::default()).unwrap_err();
    assert!(
        matches!(
            &error,
            SolverError::IncompatibleModel(report) if report.errors.len() == 1
                && report.errors[0].requirement == Requirement::Sos(SosType::Sos1)
                && report.errors[0].constraint() == Some(choice.id())
        ),
        "{error}"
    );
}
