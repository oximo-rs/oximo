use oximo_gdp::prelude::*;

fn model_with_hull() -> (Model, GdpReformulationReport) {
    let model = Model::new("Hull report inspection");

    variable!(model, 0.0 <= x <= 4.0);
    constraint!(model, original, x <= 4.0);
    boolean_variable!(model, left);
    boolean_variable!(model, right);
    disjunct_constraint!(model, left, x <= 1.0);
    disjunct_constraint!(model, right, x >= 2.0);
    disjunction!(model, [left, right]);
    let report = model.reformulate_gdp(Hull::default()).unwrap();
    (model, report)
}

#[test]
fn gdp_variable_inspection_covers_the_complete_generated_range() {
    let (model, report) = model_with_hull();
    let variables = report.generated_variables(&model);
    assert_eq!(variables.len(), report.generated_variables.len());

    for (variable, index) in variables.zip(report.generated_variables.clone()) {
        assert_eq!(variable.id.index(), index);
        assert_ne!(variable.name, "x");
        assert_ne!(variable.domain, Domain::Binary);
    }

    let variable = report.generated_variables(&model).next().unwrap();
    drop(report);
    assert_ne!(variable.name, "x");
}

#[test]
fn copied_gdp_history_inspects_its_owning_model() {
    let (model, _) = model_with_hull();
    let copy = model.fork();
    let reports = copy.gdp_reformulations();
    let report = &reports[0];
    assert_eq!(report.model_id(), Some(copy.id()));
    assert!(report.generated_variables(&copy).all(|v| v.id.index() < copy.num_variables()));
}

#[test]
fn gdp_variable_iterator_can_be_returned_from_a_local_report() {
    fn variables(
        model: &Model,
    ) -> impl ExactSizeIterator<Item = std::cell::Ref<'_, oximo_core::Variable>> {
        let report = model.gdp_reformulations()[0].clone();
        report.generated_variables(model)
    }

    let (model, report) = model_with_hull();
    let expected_variables = report.generated_variables.clone();
    drop(report);

    let variables = variables(&model);
    assert_eq!(variables.len(), expected_variables.len());
    for (variable, index) in variables.zip(expected_variables) {
        assert_eq!(variable.id.index(), index);
    }
}

#[test]
fn empty_unbound_gdp_report_inspection_is_empty() {
    let model = Model::new("unbound report inspection");
    let report = GdpReformulationReport::default();
    assert_eq!(report.generated_variables(&model).len(), 0);
}

#[test]
#[should_panic(expected = "reformulation report belongs to a different model")]
fn gdp_variable_inspection_rejects_a_foreign_model_even_with_matching_ids() {
    let (model, report) = model_with_hull();
    let foreign = model.fork();
    let _ = report.generated_variables(&foreign);
}

#[test]
fn gdp_display_includes_every_generated_row_and_excludes_declared_rows() {
    let (model, report) = model_with_hull();
    let summary = report.to_string();
    assert_eq!(
        summary,
        format!(
            "GDP reformulation: {} generated variables, {} generated constraints, {} Hull copies",
            report.generated_variables.len(),
            report.generated_rows.len(),
            report.hull_variables.len()
        )
    );
    let detailed = report.display(&model).to_string();
    assert!(detailed.starts_with(&summary));

    let mapped: std::collections::HashSet<_> =
        report.rows.iter().flat_map(|row| &row.constraints).map(|id| id.index()).collect();
    assert!(report.generated_rows.len() > mapped.len());

    let rows = model.constraints();
    for index in report.generated_rows.clone() {
        assert!(detailed.contains(rows.algebraic()[index].name.as_str()));
    }

    assert_eq!(detailed.lines().filter(|line| line.contains(": real, [")).count(), 2);
    assert!(!detailed.contains("original:"));
}

#[test]
fn an_empty_gdp_report_displays_only_its_summary() {
    let model = Model::new("empty GDP display");
    let report = GdpReformulationReport::default();
    assert_eq!(
        report.display(&model).to_string(),
        "GDP reformulation: 0 generated variables, 0 generated constraints, 0 Hull copies\n"
    );
}

#[test]
#[should_panic(expected = "reformulation report belongs to a different model")]
fn gdp_display_rejects_a_foreign_model() {
    let (model, report) = model_with_hull();
    let foreign = model.fork();
    let _ = report.display(&foreign);
}
