use oximo_gdp::prelude::*;

#[test]
fn generated_names_skip_existing_suffixes_without_colliding_with_other_planned_rows() {
    let model = Model::new("name collisions");
    variable!(model, 0.0 <= x <= 10.0);
    for suffix in 0..9 {
        let name = if suffix == 0 {
            "__oximo_gdp_row10".to_owned()
        } else {
            format!("__oximo_gdp_row10_{suffix}")
        };
        model.add_constraint(name, x.ge(0.0));
    }
    model.add_constraint("__oximo_gdp_row11", x.ge(0.0));
    boolean_variable!(model, active);
    disjunct_constraint!(model, active, 1.0 <= x <= 2.0);
    let report = model.reformulate_gdp(BigM::default()).unwrap();
    let constraints = model.constraints();
    assert_eq!(constraints.algebraic().len(), 12);
    assert_eq!(
        constraints.algebraic()[report.rows[0].constraints[0].index()].name,
        "__oximo_gdp_row10_9"
    );
    assert_eq!(
        constraints.algebraic()[report.rows[0].constraints[1].index()].name,
        "__oximo_gdp_row11_1"
    );
}

#[test]
fn checked_numeric_rows_remove_only_exactly_cancelled_indicator_coefficients() {
    for nested in [false, true] {
        let model = Model::new("cancelled indicator");
        variable!(model, 0.0 <= x <= 10.0);
        boolean_variable!(model, parent);
        boolean_variable!(model, child);
        boolean_variable!(model, other);
        let row = if nested {
            disjunct_constraint!(model, parent, x <= 4.0);
            disjunction!(model, [child, other], parent = parent);
            disjunct_constraint!(model, child, x - 9.0 * parent.binary() <= 1.0)
        } else {
            disjunct_constraint!(model, child, x - 9.0 * child.binary() <= 1.0)
        };
        let report = model
            .reformulate_gdp(
                GdpReformulationOptions::new(BigM::default())
                    .with_big_m(row, BigMValues::upper(9.0)),
            )
            .unwrap();
        let generated = report.rows.last().unwrap().constraints[0];
        let arena = model.arena();
        let rows = model.constraints();
        let terms =
            oximo_expr::extract_linear(&arena, rows.algebraic()[generated.index()].lhs).unwrap();
        assert_eq!(terms.coeffs[0], (x.var_id().unwrap(), 1.0));
        if nested {
            assert_eq!(terms.coeffs.len(), 2);
            assert_eq!(terms.coeffs[1].0, parent.binary().var_id().unwrap());
            assert!(terms.coeffs[1].1 > 0.0 && terms.coeffs[1].1 < 1e-10);
        } else {
            assert_eq!(terms.coeffs.len(), 1);
        }
        assert!((rows.algebraic()[generated.index()].upper - 10.0).abs() < 1e-8);
    }
}

#[test]
fn report_clones_share_data_and_mutations_preserve_model_and_clone_history() {
    let model = Model::new("report sharing");
    variable!(model, 0.0 <= x <= 10.0);
    boolean_variable!(model, active);
    disjunct_constraint!(model, active, x <= 2.0);
    let mut report = model.reformulate_gdp(BigM::default()).unwrap();
    let clone = report.clone();
    let history = model.gdp_reformulations();
    assert_eq!(report.rows.as_ptr(), clone.rows.as_ptr());
    assert_eq!(report.rows.as_ptr(), history[0].rows.as_ptr());
    let generated = history[0].rows[0].constraints[0];
    report.rows[0].constraints.clear();
    report.booleans.clear();
    assert_eq!(clone.rows[0].constraints, vec![generated]);
    assert_eq!(history[0].rows[0].constraints, vec![generated]);
    assert_eq!(clone.booleans.len(), 1);
    assert_eq!(history[0].booleans.len(), 1);
    let mut owned = clone.into_owned();
    owned.rows.clear();
    assert_eq!(history[0].rows.len(), 1);
    drop(history);
    drop(model);
    assert_eq!(report.rows.len(), 1);
}
