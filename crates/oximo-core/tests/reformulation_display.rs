use oximo_core::prelude::*;

#[test]
fn sos_report_displays_a_summary_or_named_generated_algebra() {
    let model = Model::new("SOS display");

    variable!(model, 0.0 <= x <= 2.0);
    variable!(model, 0.0 <= y <= 3.0);
    constraint!(model, original, x + y <= 4.0);
    sos_constraint!(model, choice, SOS1, [x, y]);

    let reports = model.reformulate_sos(SosReformulationOptions::default()).unwrap();
    let report = &reports[0];

    let summary = "SOS reformulation #0: 2 generated variables, 3 generated constraints";
    assert_eq!(report.to_string(), summary);

    let display = report.display(&model);

    variable!(model, _unrelated);
    let detailed = display.to_string();
    assert!(detailed.starts_with(summary));
    assert!(detailed.contains("source: choice: SOS1 [x: 1, y: 2] (inactive)"));
    assert!(detailed.contains("vars\n  __oximo_sos0_member_0: binary, [0, 1]\n"));
    assert!(detailed.contains("  __oximo_sos0_member_1: binary, [0, 1]\n"));
    assert!(detailed.contains("s.t.\n"));

    for id in &report.constraints {
        assert!(detailed.contains(&model.display_row(id.raw_id()).to_string()));
    }

    assert!(!detailed.contains("original:"));
    assert!(!detailed.contains("_unrelated"));
}

#[test]
fn trivial_sos_report_omits_empty_generated_sections() {
    let model = Model::new("trivial SOS display");

    variable!(model, 0.0 <= x <= 2.0);
    sos_constraint!(model, choice, SOS1, [x]);
    let reports = model.reformulate_sos(SosReformulationOptions::default()).unwrap();
    assert_eq!(
        reports[0].display(&model).to_string(),
        "SOS reformulation #0: 0 generated variables, 0 generated constraints\n\
         source: choice: SOS1 [x: 1] (inactive)\n"
    );
}

#[test]
fn indicator_report_displays_its_trigger_and_both_equality_sides() {
    let model = Model::new("indicator display");

    variable!(model, 0.0 <= x <= 4.0);
    variable!(model, enabled, Binary);
    indicator_constraint!(model, fixed, enabled == 1 => x == 2.0);

    let reports = model.reformulate_indicators(IndicatorReformulationOptions::default()).unwrap();
    let report = &reports[0];
    assert_eq!(report.to_string(), "Indicator reformulation #0: 2 generated constraints");

    let detailed = report.display(&model).to_string();
    assert!(detailed.contains("source: fixed: enabled = 1 -> x = 2 (inactive)"));
    assert!(!detailed.contains("vars\n"));

    for id in &report.constraints {
        assert!(detailed.contains(&model.display_row(id.raw_id()).to_string()));
    }
}

#[test]
fn a_summary_needs_no_model_or_registered_ids() {
    let report = SosReformulationArtifacts {
        source: SosConstraintId(42),
        variables: vec![GeneratedVarId::from_raw(VarId(u32::MAX))],
        constraints: Vec::new(),
    };
    assert_eq!(
        report.to_string(),
        "SOS reformulation #42: 1 generated variable, 0 generated constraints"
    );
}
