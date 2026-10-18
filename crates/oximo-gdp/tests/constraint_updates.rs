use oximo_gdp::prelude::*;

#[test]
fn supporting_global_rows_reject_edits_without_mutation() {
    for method in [GdpMethod::from(BigM::default()), GdpMethod::from(Hull::default())] {
        let model = Model::new("global row dependencies");
        variable!(model, 0.0 <= x <= 1000.0);
        let cap = constraint!(model, cap, x <= 10.0);
        boolean_variable!(model, active);
        disjunct_constraint!(model, active, x <= 2.0);
        model.reformulate_gdp(GdpReformulationOptions::new(method)).unwrap();
        let original = model.constraints().algebraic()[cap.index()].clone();

        for update in [
            model.set_constraint_active(cap, false),
            model.set_constraint_set(cap, LessThan(100.0)),
            model.set_constraint_function(cap, (0.1 * x).into_function()),
        ] {
            assert!(matches!(
                update,
                Err(oximo_core::ConstraintUpdateError::GdpDependency(id)) if id == cap.id()
            ));
        }

        let rows = model.constraints();
        let row = &rows.algebraic()[cap.index()];
        assert_eq!(row.lhs, original.lhs);
        assert_eq!(row.upper.to_bits(), original.upper.to_bits());
        assert_eq!(row.lower.to_bits(), original.lower.to_bits());
        assert!(row.active);
        let semantic = model.semantic_constraint(cap).unwrap().unwrap();
        assert!(semantic.active);
        assert_eq!(semantic.pair.set.scalar_bounds(), Some((original.lower, original.upper)));
        assert!(matches!(
            semantic.pair.function.representation(),
            oximo_core::FunctionRepresentationRef::Scalar(root) if root == original.lhs
        ));
    }
}

#[test]
fn global_row_locks_follow_actual_support_and_are_isolated_from_source_copies() {
    for tighten in [false, true] {
        let source = Model::new("global row lock scope");
        variable!(source, 0.0 <= x <= 1000.0);
        variable!(source, 0.0 <= z <= 1000.0);
        let cap = constraint!(source, cap, x <= 10.0);
        let redundant = constraint!(source, redundant, x <= 20.0);
        let unrelated = constraint!(source, unrelated, z <= 10.0);
        let inactive = constraint!(source, inactive, x <= 5.0);
        source.set_constraint_active(inactive, false).unwrap();
        boolean_variable!(source, active);
        disjunct_constraint!(source, active, x <= 2.0);
        let transformed = source
            .to_reformulated_gdp_model(
                GdpReformulationOptions::default().with_bound_tightening(tighten),
            )
            .unwrap();

        assert_eq!(
            transformed
                .model()
                .set_constraint_active(transformed.constraint_handle("cap").unwrap(), false)
                .is_err(),
            tighten
        );
        for name in ["redundant", "unrelated"] {
            transformed
                .model()
                .set_constraint_active(transformed.constraint_handle(name).unwrap(), false)
                .unwrap();
        }
        transformed
            .model()
            .set_constraint_active(transformed.constraint_handle("inactive").unwrap(), true)
            .unwrap();
        let fork = transformed.model().fork();
        assert_eq!(
            fork.set_constraint_active(fork.constraint_handle("cap").unwrap(), false).is_err(),
            tighten
        );

        for handle in [cap, redundant, unrelated] {
            source.set_constraint_active(handle, false).unwrap();
        }
    }
}

#[test]
fn failed_reformulation_does_not_lock_supporting_global_rows() {
    let model = Model::new("failed GDP row lock transaction");

    variable!(model, 0.0 <= x <= 1000.0);
    let cap = constraint!(model, cap, x <= 10.0);
    variable!(model, free);
    boolean_variable!(model, active);
    disjunct_constraint!(model, active, x <= 2.0);
    disjunct_constraint!(model, active, free <= 2.0);

    assert!(model.reformulate_gdp(BigM::default()).is_err());
    assert_eq!(model.constraints().algebraic().len(), 1);
    model.set_constraint_set(cap, LessThan(100.0)).unwrap();
    model.set_constraint_function(cap, (0.1 * x).into_function()).unwrap();
    model.set_constraint_active(cap, false).unwrap();
}

#[test]
fn incremental_reformulations_preserve_earlier_global_row_locks() {
    let model = Model::new("incremental global row dependencies");
    variable!(model, 0.0 <= x <= 1000.0);
    variable!(model, 0.0 <= y <= 1000.0);
    let x_cap = constraint!(model, x_cap, x <= 10.0);
    let y_cap = constraint!(model, y_cap, y <= 10.0);
    boolean_variable!(model, first);
    disjunct_constraint!(model, first, x <= 2.0);
    model.reformulate_gdp(BigM::default()).unwrap();
    model.set_constraint_set(y_cap, LessThan(20.0)).unwrap();

    boolean_variable!(model, second);
    disjunct_constraint!(model, second, y <= 2.0);
    model.reformulate_gdp(BigM::default()).unwrap();
    for cap in [x_cap, y_cap] {
        assert!(matches!(
            model.set_constraint_active(cap, false),
            Err(oximo_core::ConstraintUpdateError::GdpDependency(id)) if id == cap.id()
        ));
    }
}
