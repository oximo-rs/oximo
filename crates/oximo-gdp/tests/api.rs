use oximo_gdp::prelude::*;

#[test]
fn source_handles_rebind_to_independent_models_and_keep_option_provenance() {
    let model = Model::new("handles");

    variable!(model, 0.0 <= x <= 10.0);
    boolean_variable!(model, a);
    boolean_variable!(model, b);
    let row = disjunct_constraint!(model, a, capacity, x <= 2.0);
    let choice = disjunction!(model, choice, [a, b]);
    let logic = logical_constraint!(model, implication, implies(a, b));

    assert_eq!(model.boolean_handle_by_name("a").unwrap().id(), a.id());
    assert_eq!(model.disjunct_constraint_id("a::capacity"), Some(row.id()));
    assert_eq!(model.disjunct_constraint_handle("a::capacity"), Some(row));
    assert_eq!(model.disjunction_id("choice"), Some(choice.id()));
    assert_eq!(model.disjunction_handle("choice"), Some(choice));
    assert_eq!(model.logical_constraint_id("implication"), Some(logic.id()));
    assert_eq!(model.logical_constraint_handle("implication"), Some(logic));
    assert!(model.boolean_handle_by_name("missing").is_none());
    assert!(model.disjunction_handle("missing").is_none());
    assert!(model.logical_constraint_handle("missing").is_none());
    assert!(model.boolean_handle_from_id(BooleanId(u32::MAX)).is_none());
    assert!(model.disjunct_constraint_handle_from_id(DisjunctConstraintId(u32::MAX)).is_none());
    assert!(model.disjunction_handle_from_id(DisjunctionId(u32::MAX)).is_none());
    assert!(model.logical_constraint_handle_from_id(LogicalConstraintId(u32::MAX)).is_none());

    let clone = model.__gdp_clone();
    let rebound = clone.disjunct_constraint_handle_from_id(row.id()).unwrap();

    assert_eq!(rebound.id(), row.id());
    assert_ne!(rebound.model_id(), row.model_id());
    assert_ne!(clone.boolean_handle_from_id(a.id()).unwrap().model_id(), a.model_id());
    assert_ne!(clone.disjunction_handle_from_id(choice.id()).unwrap(), choice);
    assert_ne!(clone.logical_constraint_handle_from_id(logic.id()).unwrap(), logic);
    let before = clone.num_constraints();

    assert!(matches!(
        clone.reformulate_gdp(
            GdpReformulationOptions::default().with_big_m(row, BigMValues::upper(8.0))
        ),
        Err(GdpError::ForeignHandle(_))
    ));
    assert_eq!(clone.num_constraints(), before);
    let report = clone
        .reformulate_gdp(
            GdpReformulationOptions::default().with_big_m(rebound, BigMValues::upper(8.0)),
        )
        .unwrap();
    assert_eq!(report.booleans[0].source, a.id());
    assert_eq!(report.booleans[0].binary, a.binary().var_id().unwrap());
    assert_eq!(report.disjunctions[0].source, choice.id());
    assert_eq!(report.disjunctions[0].method, GdpMethodKind::BigM);
    assert_eq!(report.logical_constraints[0].source, logic.id());
    assert!(model.has_unreformulated_gdp());
    assert!(!clone.has_unreformulated_gdp());
}

#[test]
fn inspection_distinguishes_live_borrows_owned_snapshots_and_lifecycle() {
    let model = Model::new("inspection");

    variable!(model, 0.0 <= x <= 10.0);
    boolean_variable!(model, active);
    disjunct_constraint!(model, active, x <= 2.0);

    let snapshot = model.gdp_snapshot();
    assert_eq!(snapshot.rows[0].state(), ReformulationState::Pending);
    assert_eq!(snapshot.booleans[0].state(), DisjunctState::Open);
    {
        let view: GdpView<'_> = model.gdp();
        assert_eq!(view.rows.len(), 1);
        assert_eq!(view.rows[0].state(), ReformulationState::Pending);
    }
    model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
    assert_eq!(model.gdp().rows[0].state(), ReformulationState::Reformulated);
    assert_eq!(model.gdp().booleans[0].state(), DisjunctState::Sealed);
    assert_eq!(snapshot.rows[0].state(), ReformulationState::Pending);
    assert_eq!(snapshot.booleans[0].state(), DisjunctState::Open);

    let fresh = model.add_boolean("fresh");
    logical_constraint!(model, fresh);
    assert!(model.has_unreformulated_gdp());
    model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
    assert!(!model.has_unreformulated_gdp());
}

#[test]
fn copied_family_handles_and_borrowed_booleans_feed_logical_builders() {
    let model = Model::new("families");

    let indices = Set::from_ints([0, 1, 2]);
    boolean_variable!(model, decisions[i in indices]);
    let handles: Vec<_> = decisions.values().collect();
    let last: BooleanHandle<'_> = decisions.get(2).unwrap();

    assert_eq!(last.id(), handles[2].id());
    let entries: Vec<_> = decisions.iter().map(|(key, handle)| (key, handle.id())).collect();
    assert_eq!(entries.iter().map(|(key, _)| *key).collect::<Vec<_>>(), vec![0, 1, 2]);
    assert_eq!(decisions.get_ref(0).unwrap().id(), handles[0].id());
    assert_eq!(decisions.iter_ref().count(), 3);
    assert!(decisions.get(3).is_none());
    logical_constraint!(model, exactly(1, decisions.values()));
    logical_constraint!(model, logical_or(&handles));
    logical_constraint!(model, implies(handles.first().unwrap(), handles.last().unwrap()));
    model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
    assert!(!model.has_unreformulated_gdp());
}

#[test]
fn range_overrides_follow_lower_and_upper_sides_for_both_representations() {
    let model = Model::new("ranges");

    variable!(model, -10.0 <= x <= 10.0);
    variable!(model, -2.0 <= lo <= -1.0);
    variable!(model, 1.0 <= hi <= 2.0);
    boolean_variable!(model, active);
    let interval = disjunct_constraint!(model, active, numeric, -1.0 <= x <= 1.0);
    let split = disjunct_constraint!(model, active, symbolic, lo <= x <= hi);

    assert!(matches!(interval, GdpRangeHandles::Interval(_)));
    assert!(matches!(split, GdpRangeHandles::Split { .. }));
    let values = BigMValues { lower: Some(100.0), upper: Some(200.0) };
    let report = model
        .reformulate_gdp(
            GdpReformulationOptions::default()
                .with_range_big_m(interval, values)
                .with_range_big_m(split, values),
        )
        .unwrap();
    for row in &report.rows {
        if let Some(m) = row.lower_m {
            assert!((m.value - 100.0).abs() < f64::EPSILON);
            assert_eq!(m.origin, BigMOrigin::Explicit);
        }
        if let Some(m) = row.upper_m {
            assert!((m.value - 200.0).abs() < f64::EPSILON);
            assert_eq!(m.origin, BigMOrigin::Explicit);
        }
    }
    assert_eq!(report.rows.len(), 3);
    assert_eq!(report.rows.iter().filter(|row| row.lower_m.is_some()).count(), 2);
    assert_eq!(report.rows.iter().filter(|row| row.upper_m.is_some()).count(), 2);
    assert_eq!(BigMValues::lower(1.0).upper, None);
    assert_eq!(BigMValues::upper(1.0).lower, None);
}
