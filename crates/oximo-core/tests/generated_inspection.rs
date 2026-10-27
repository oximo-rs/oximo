use oximo_core::prelude::*;

#[test]
fn sos_inspection_excludes_source_entries_and_preserves_report_order() {
    let model = Model::new("SOS inspection");

    variable!(model, 0.0 <= x <= 2.0);
    variable!(model, 0.0 <= y <= 3.0);
    constraint!(model, original, x + y <= 4.0);
    sos_constraint!(model, choice, SOS1, [x, y]);

    let reports = model.reformulate_sos(SosReformulationOptions::default()).unwrap();
    let report = &reports[0];

    let variables = report.generated_variables(&model);
    assert_eq!(variables.len(), 2);

    for (variable, id) in variables.zip(&report.variables) {
        assert_eq!(variable.id.index(), id.index());
        assert_eq!(variable.domain, Domain::Binary);
        assert_ne!(variable.name, "x");
        assert_ne!(variable.name, "y");
    }

    let variable = report.generated_variables(&model).next().unwrap();
    drop(reports);
    assert_eq!(variable.domain, Domain::Binary);
}

#[test]
fn trivial_sos_inspection_is_empty() {
    let model = Model::new("trivial SOS inspection");

    variable!(model, 0.0 <= x <= 2.0);
    constraint!(model, x <= 1.0);
    sos_constraint!(model, SOS1, [x]);

    let reports = model.reformulate_sos(SosReformulationOptions::default()).unwrap();
    assert_eq!(reports[0].generated_variables(&model).len(), 0);
}

#[test]
fn sos_inspection_accepts_an_id_preserving_copy() {
    let model = Model::new("copied SOS inspection");

    variable!(model, 0.0 <= x <= 2.0);
    variable!(model, 0.0 <= y <= 3.0);
    sos_constraint!(model, SOS1, [x, y]);

    let transformed = model.to_reformulated_sos_model(SosReformulationOptions::default()).unwrap();
    let reports = transformed.sos_reformulations();
    let copy = transformed.fork();
    assert_ne!(transformed.id(), copy.id());
    assert_eq!(reports[0].generated_variables(&copy).len(), 2);
    assert!(reports[0].generated_variables(&copy).all(|v| v.domain == Domain::Binary));
}

#[test]
#[should_panic(expected = "unknown generated variable ID")]
fn sos_inspection_rejects_an_unregistered_variable() {
    let model = Model::new("missing generated variable");

    let report = SosReformulationArtifacts {
        source: SosConstraintId(0),
        variables: vec![GeneratedVarId::from_raw(VarId(0))],
        constraints: Vec::new(),
    };
    let _ = report.generated_variables(&model).next();
}
