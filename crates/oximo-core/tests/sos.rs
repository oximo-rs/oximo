use oximo_core::prelude::*;
use oximo_expr::extract_linear;

#[test]
fn sos_macros_register_and_promote_kind() {
    let m = Model::new("sos");
    variable!(m, x);
    variable!(m, y);
    variable!(m, z);
    objective!(m, Min, x + y + z);
    sos_constraint!(m, first, SOS1, [(x, 1.0), (y, 2.0)]);
    sos_constraint!(m, curve, SOS2, [(x, 1.0), (y, 2.0), (z, 3.0)]);
    sos_constraint!(m, inferred, SOS1, [x, y, z]);
    assert_eq!(m.kind(), ModelKind::MILP);
    assert_eq!(m.num_sos_constraints(), 3);
    assert_eq!(m.sos_constraint_id("curve"), Some(SosConstraintId(1)));
    assert_eq!(m.sos_constraints()[1].sos_type, SosType::Sos2);
    assert_eq!(
        m.sos_constraints()[2].members.iter().map(|m| m.weight).collect::<Vec<_>>(),
        [1.0, 2.0, 3.0]
    );
}

#[test]
fn sos_family_and_auto_name_work() {
    let m = Model::new("sos_family");
    variable!(m, x[i in 0..2]);
    sos_constraint!(m, family[i in 0..2], SOS1, [(x[i], 1.0)]);
    sos_constraint!(m, SOS2, [(x[0], 1.0), (x[1], 2.0)]);
    assert_eq!(m.sos_constraints()[0].name, "family[0]");
    assert!(m.sos_constraints()[2].name.starts_with("_sos"));
}

#[test]
fn sos_indexed_family_infers_weights() {
    let m = Model::new("sos_indexed_family");
    variable!(m, x[i in 0..2]);
    variable!(m, y[i in 0..2]);
    variable!(m, z[i in 0..2]);
    sos_constraint!(m, choice[i in 0..2], SOS1, [x[i], y[i], z[i]]);

    assert_eq!(m.num_sos_constraints(), 2);
    assert_eq!(m.sos_constraints()[0].name, "choice[0]");
    assert_eq!(m.sos_constraints()[1].name, "choice[1]");
    for constraint in m.sos_constraints().iter() {
        assert_eq!(
            constraint.members.iter().map(|m| m.weight).collect::<Vec<_>>(),
            [1.0, 2.0, 3.0]
        );
    }
}

#[test]
fn sos_auto_weight_method_accepts_dynamic_members() {
    let m = Model::new("sos_dynamic");
    variable!(m, x);
    variable!(m, y);
    variable!(m, z);
    let members = vec![x, y, z];
    m.add_sos_constraint_auto_weights("dynamic", SosType::Sos2, members);

    assert_eq!(
        m.sos_constraints()[0].members.iter().map(|m| m.weight).collect::<Vec<_>>(),
        [1.0, 2.0, 3.0]
    );
}

#[test]
fn sos_registry_and_display_are_complete() {
    let m = Model::new("sos_display");
    variable!(m, x);
    variable!(m, y);
    let handle = m.add_sos_constraint("choice", SosType::Sos2, [(x, 10.0), (y, 20.0)]);
    let id = handle.id();

    assert_eq!(id.index(), 0);
    assert_eq!(SosType::Sos1.label(), "SOS1");
    assert_eq!(SosType::Sos2.label(), "SOS2");
    let labels = [SosType::Sos1.to_string(), SosType::Sos2.to_string()];
    assert_eq!(labels, ["SOS1", "SOS2"]);
    assert!(labels.iter().all(|label| label.is_ascii()));
    assert!(m.has_sos_constraints());
    assert_eq!(m.sos_constraint_id("choice"), Some(id));
    assert_eq!(m.sos_constraint_id("missing"), None);
    assert_eq!(m.display_sos(handle).to_string(), "choice: SOS2 [x: 10, y: 20]");
    assert!(m.to_string().contains("choice: SOS2 [x: 10, y: 20]"));
    assert_eq!(m.constraints().special_ordered_sets()[0].name, "choice");
}

#[test]
#[should_panic(expected = "has no members")]
fn sos_rejects_empty_members() {
    let m = Model::new("empty");
    m.add_sos_constraint("empty", SosType::Sos1, [] as [(Expr, f64); 0]);
}

#[test]
#[should_panic(expected = "duplicate variable")]
fn sos_rejects_duplicate_variables() {
    let m = Model::new("duplicate_variable");
    variable!(m, x);
    m.add_sos_constraint("duplicate", SosType::Sos1, [(x, 1.0), (x, 2.0)]);
}

#[test]
#[should_panic(expected = "non-finite weight")]
fn sos_rejects_nonfinite_weights() {
    let m = Model::new("nonfinite");
    variable!(m, x);
    m.add_sos_constraint("nonfinite", SosType::Sos1, [(x, f64::NAN)]);
}

#[test]
#[should_panic(expected = "must be bare variables")]
fn sos_rejects_compound_members() {
    let m = Model::new("compound");
    variable!(m, x);
    m.add_sos_constraint("compound", SosType::Sos1, [(x + 1.0, 1.0)]);
}

#[test]
#[should_panic(expected = "belongs to another model")]
fn sos_rejects_foreign_members() {
    let m = Model::new("owner");
    let other = Model::new("other");
    variable!(other, x);
    m.add_sos_constraint("foreign", SosType::Sos1, [(x, 1.0)]);
}

#[test]
#[should_panic(expected = "already registered")]
fn sos_rejects_duplicate_names() {
    let m = Model::new("duplicate_name");
    variable!(m, x);
    m.add_sos_constraint("same", SosType::Sos1, [(x, 1.0)]);
    m.add_sos_constraint("same", SosType::Sos1, [(x, 2.0)]);
}

#[test]
#[should_panic(expected = "duplicate weights")]
fn sos_rejects_duplicate_weights() {
    let m = Model::new("bad");
    variable!(m, x);
    variable!(m, y);
    m.add_sos_constraint("bad", SosType::Sos2, [(x, 1.0), (y, 1.0)]);
}

#[test]
#[should_panic(expected = "duplicate weights")]
fn sos_rejects_signed_zero_duplicate_weights() {
    let m = Model::new("signed_zero_weights");
    variable!(m, x);
    variable!(m, y);
    m.add_sos_constraint("signed_zero", SosType::Sos1, [(x, 0.0), (y, -0.0)]);
}

#[test]
fn single_sos_reformulation_is_independent_and_tracks_artifacts() {
    let m = Model::new("single_reformulation");
    variable!(m, -2.0 <= x <= 4.0);
    variable!(m, 0.0 <= y <= 5.0);
    objective!(m, Max, x + y);
    let choice = sos_constraint!(m, choice, SOS1, [(x, 1.0), (y, 2.0)]);

    let transformed = choice.to_reformulated_model(SosReformulationOptions::default()).unwrap();
    assert_ne!(transformed.id(), m.id());

    assert_eq!(m.num_variables(), 2);
    assert_eq!(m.constraints().algebraic().len(), 0);
    assert!(m.sos_constraints()[choice.index()].active);
    assert!(m.has_active_sos_constraints());

    assert_eq!(transformed.num_variables(), 4);
    assert_eq!(transformed.constraints().algebraic().len(), 4);
    assert!(!transformed.sos_constraints()[choice.index()].active);
    assert!(transformed.has_sos_constraints());
    assert!(!transformed.has_active_sos_constraints());
    assert_eq!(transformed.kind(), ModelKind::MILP);
    assert_eq!(transformed.sos_reformulations().len(), 1);
    let history = transformed.sos_reformulations();
    let artifacts = &history[0];
    assert_eq!(artifacts.source, choice.id());
    assert_eq!(artifacts.variables.len(), 2);
    assert_eq!(artifacts.constraints.len(), 4);
}

#[test]
fn sos_reformulation_requires_bounds_unless_fallback_is_explicit() {
    let m = Model::new("fallback_big_m");
    variable!(m, x <= 4.0);
    variable!(m, 0.0 <= y <= 5.0);
    let choice = sos_constraint!(m, choice, SOS1, [x, y]);

    assert!(matches!(
        choice.to_reformulated_model(SosReformulationOptions::default()),
        Err(ReformulationError::MissingFiniteBound { side: "lower", .. })
    ));
    assert_eq!(m.num_variables(), 2, "failed reformulation did not mutate its source");

    let transformed = choice
        .to_reformulated_model(SosReformulationOptions::default().with_fallback_big_m(100.0))
        .unwrap();
    assert_eq!(transformed.num_variables(), 4);
    assert!(!transformed.has_active_sos_constraints());

    assert!(matches!(
        choice.to_reformulated_model(SosReformulationOptions::default().with_fallback_big_m(0.0),),
        Err(ReformulationError::InvalidFallbackBigM(0.0))
    ));
}

#[test]
fn sos2_reformulation_uses_weight_order_and_adjacent_intervals() {
    let m = Model::new("sos2_order");
    variable!(m, 0.0 <= x <= 10.0);
    variable!(m, 0.0 <= y <= 10.0);
    variable!(m, 0.0 <= z <= 10.0);
    objective!(m, Max, x + y + z);
    let adjacent = sos_constraint!(m, adjacent, SOS2, [(z, 30.0), (x, 10.0), (y, 20.0)]);

    let transformed = adjacent.to_reformulated_model(SosReformulationOptions::default()).unwrap();
    let history = transformed.sos_reformulations();
    let artifacts = &history[0];
    assert_eq!(artifacts.variables.len(), 2, "one binary per adjacent interval");
    assert_eq!(artifacts.constraints.len(), 4, "selection plus one upper gate per member");

    let interval_0 = artifacts.variables[0];
    let interval_1 = artifacts.variables[1];
    let arena = transformed.arena();
    let model_constraints = transformed.constraints();
    let rows = model_constraints.algebraic();
    let coefficients = |row: usize| {
        extract_linear(&arena, rows[row].lhs)
            .unwrap()
            .coeffs
            .iter()
            .copied()
            .collect::<std::collections::HashMap<_, _>>()
    };

    let x_upper = coefficients(1);
    assert_eq!(x_upper.get(&x.var_id().unwrap()), Some(&1.0));
    assert_eq!(x_upper.get(&interval_0), Some(&-10.0));
    assert!(!x_upper.contains_key(&interval_1));

    let y_upper = coefficients(2);
    assert_eq!(y_upper.get(&y.var_id().unwrap()), Some(&1.0));
    assert_eq!(y_upper.get(&interval_0), Some(&-10.0));
    assert_eq!(y_upper.get(&interval_1), Some(&-10.0));

    let z_upper = coefficients(3);
    assert_eq!(z_upper.get(&z.var_id().unwrap()), Some(&1.0));
    assert!(!z_upper.contains_key(&interval_0));
    assert_eq!(z_upper.get(&interval_1), Some(&-10.0));
}

#[test]
fn whole_model_reformulation_skips_inactive_and_handles_degenerate_sets() {
    let m = Model::new("all_sos");
    variable!(m, 0.0 <= x <= 1.0);
    variable!(m, 0.0 <= y <= 1.0);
    objective!(m, Max, x + y);
    sos_constraint!(m, singleton, SOS1, [x]);
    sos_constraint!(m, pair, SOS2, [x, y]);

    let transformed = m.to_reformulated_sos_model(SosReformulationOptions::default()).unwrap();
    assert_eq!(transformed.num_variables(), 2, "degenerate sets need no binaries");
    assert_eq!(transformed.constraints().algebraic().len(), 0);
    assert_eq!(transformed.sos_reformulations().len(), 2);
    assert!(
        transformed
            .sos_reformulations()
            .iter()
            .all(|entry| { entry.variables.is_empty() && entry.constraints.is_empty() })
    );
    assert!(!transformed.has_active_sos_constraints());
    assert_eq!(transformed.kind(), ModelKind::LP);
}

#[test]
fn generated_names_avoid_user_collisions() {
    let m = Model::new("name_collision");
    variable!(m, 0.0 <= x <= 1.0);
    variable!(m, 0.0 <= y <= 1.0);
    variable!(m, __oximo_sos0_member_0, Binary);
    constraint!(m, __oximo_sos0_select, x <= 1.0);
    let choice = sos_constraint!(m, choice, SOS1, [x, y]);

    let transformed = choice.to_reformulated_model(SosReformulationOptions::default()).unwrap();
    assert!(transformed.variable_id("__oximo_sos0_member_0_1").is_some());
    assert!(transformed.constraint_id("__oximo_sos0_select_1").is_some());
}

#[test]
fn reformulation_omits_sign_redundant_big_m_rows() {
    let m = Model::new("sign_redundant_rows");
    variable!(m, -5.0 <= crossing <= 7.0);
    variable!(m, 2.0 <= nonnegative <= 8.0);
    variable!(m, -9.0 <= nonpositive <= -3.0);
    variable!(m, fixed, fix = 0.0);
    let choice = sos_constraint!(m, choice, SOS1, [crossing, nonnegative, nonpositive, fixed]);

    let transformed = choice.to_reformulated_model(SosReformulationOptions::default()).unwrap();
    let history = transformed.sos_reformulations();
    let artifacts = &history[0];

    assert_eq!(artifacts.variables.len(), 3, "fixed-zero members need no activation");
    assert_eq!(
        artifacts.constraints.len(),
        5,
        "one selection row, two crossing-sign gates, and one gate per sign-definite member"
    );
    assert!(transformed.constraint_id("__oximo_sos0_member_0_lower").is_some());
    assert!(transformed.constraint_id("__oximo_sos0_member_0_upper").is_some());
    assert!(transformed.constraint_id("__oximo_sos0_member_1_lower").is_none());
    assert!(transformed.constraint_id("__oximo_sos0_member_1_upper").is_some());
    assert!(transformed.constraint_id("__oximo_sos0_member_2_lower").is_some());
    assert!(transformed.constraint_id("__oximo_sos0_member_2_upper").is_none());
    assert!(transformed.constraint_id("__oximo_sos0_member_3_lower").is_none());
    assert!(transformed.constraint_id("__oximo_sos0_member_3_upper").is_none());
}

#[test]
fn trivially_satisfied_sos_sets_need_no_bounds_or_artifacts() {
    let m = Model::new("trivial_fixed_zero_members");
    variable!(m, unbounded);
    variable!(m, other_unbounded);
    variable!(m, zero_1, fix = 0.0);
    variable!(m, zero_2, fix = 0.0);
    let sos1 = sos_constraint!(m, one_possible_nonzero, SOS1, [unbounded, zero_1]);
    sos_constraint!(
        m,
        adjacent_possible_nonzeros,
        SOS2,
        [(unbounded, 30.0), (zero_2, 10.0), (other_unbounded, 20.0)]
    );

    let transformed = m.to_reformulated_sos_model(SosReformulationOptions::default()).unwrap();

    assert!(!transformed.has_active_sos_constraints());
    assert_eq!(transformed.num_variables(), 4);
    assert!(transformed.constraints().algebraic().is_empty());
    assert_eq!(transformed.sos_reformulations().len(), 2);
    assert!(
        transformed
            .sos_reformulations()
            .iter()
            .all(|artifacts| artifacts.variables.is_empty() && artifacts.constraints.is_empty())
    );
    assert_eq!(sos1.index(), 0);
}

#[test]
fn fixed_zero_member_still_separates_nonadjacent_sos2_members() {
    let m = Model::new("fixed_zero_sos2_gap");
    variable!(m, left);
    variable!(m, gap, fix = 0.0);
    variable!(m, right);
    sos_constraint!(m, separated, SOS2, [left, gap, right]);

    assert!(matches!(
        m.to_reformulated_sos_model(SosReformulationOptions::default()),
        Err(ReformulationError::MissingFiniteBound { .. })
    ));

    let transformed = m
        .to_reformulated_sos_model(SosReformulationOptions::default().with_fallback_big_m(10.0))
        .unwrap();
    let history = transformed.sos_reformulations();
    let artifacts = &history[0];
    assert_eq!(artifacts.variables.len(), 2);
    assert_eq!(artifacts.constraints.len(), 5);
}

#[test]
fn in_place_reformulation_preserves_handles_and_matches_independent_shape() {
    let m = Model::new("in_place");
    variable!(m, -2.0 <= x <= 4.0);
    variable!(m, 0.0 <= y <= 5.0);
    objective!(m, Max, x + y);
    let choice = sos_constraint!(m, choice, SOS1, [x, y]);
    let independent = choice.to_reformulated_model(SosReformulationOptions::default()).unwrap();

    let artifacts = m
        .reformulate_sos(SosReformulationOptions::default())
        .unwrap()
        .pop()
        .expect("active SOS produces artifacts");

    assert!(!m.has_active_sos_constraints());
    assert_eq!(m.num_variables(), independent.num_variables());
    assert_eq!(m.constraints().algebraic().len(), independent.constraints().algebraic().len());
    assert_eq!(artifacts.variables, independent.sos_reformulations()[0].variables);
    assert_eq!(artifacts.constraints, independent.sos_reformulations()[0].constraints);
    assert_eq!(m.display_expr(x + y).to_string(), "x + y");
    assert!(choice.reformulate(SosReformulationOptions::default()).unwrap().is_none());
}

#[test]
fn in_place_reformulation_validates_all_sets_before_mutating() {
    let m = Model::new("in_place_atomic_error");
    variable!(m, 0.0 <= bounded_x <= 1.0);
    variable!(m, 0.0 <= bounded_y <= 1.0);
    variable!(m, unbounded_x);
    variable!(m, unbounded_y);
    sos_constraint!(m, bounded, SOS1, [bounded_x, bounded_y]);
    sos_constraint!(m, unbounded, SOS1, [unbounded_x, unbounded_y]);

    assert!(matches!(
        m.reformulate_sos(SosReformulationOptions::default()),
        Err(ReformulationError::MissingFiniteBound { .. })
    ));
    assert_eq!(m.num_variables(), 4);
    assert!(m.constraints().algebraic().is_empty());
    assert!(m.sos_constraints().iter().all(|constraint| constraint.active));
}

#[test]
fn model_exposes_in_place_reformulation_history_after_artifacts_drop() {
    let m = Model::new("in_place_history");
    variable!(m, 0.0 <= x <= 1.0);
    variable!(m, 0.0 <= y <= 1.0);
    sos_constraint!(m, choice, SOS1, [x, y]);

    let artifacts = m.reformulate_sos(SosReformulationOptions::default()).unwrap();
    assert_eq!(artifacts.len(), 1);
    drop(artifacts);

    let history = m.sos_reformulations();
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].source, SosConstraintId(0));
    assert_eq!(history[0].variables.len(), 2);
}

#[test]
fn preserving_reformulation_leaves_source_member_bounds_mutable() {
    let m = Model::new("preserved_source_bounds");
    variable!(m, 0.0 <= x <= 1.0);
    variable!(m, 0.0 <= y <= 1.0);
    let choice = sos_constraint!(m, choice, SOS1, [x, y]);

    let transformed = choice.to_reformulated_model(SosReformulationOptions::default()).unwrap();
    m.unfix_var(x.var_id().unwrap(), -10.0, 100.0);

    assert!((m.variables()[x.var_id().unwrap().index()].ub - 100.0).abs() < f64::EPSILON);
    assert!((transformed.variables()[x.var_id().unwrap().index()].ub - 1.0).abs() < f64::EPSILON);
}

#[test]
fn reformulated_clone_rejects_source_handles_even_when_numeric_ids_collide() {
    let source = Model::new("forked_identity");
    variable!(source, 0.0 <= x <= 1.0);
    variable!(source, 0.0 <= y <= 1.0);
    sos_constraint!(source, choice, SOS1, [x, y]);

    let transformed = source.to_reformulated_sos_model(SosReformulationOptions::default()).unwrap();
    assert_ne!(source.id(), transformed.id());
    let auxiliary = transformed.sos_reformulations()[0].variables[0];

    // The independently mutable source allocates the same raw slot next. Its
    // model-bound handle must not be allowed to mutate the generated auxiliary.
    let late_source_variable = source.__var("late").build();
    assert_eq!(late_source_variable.var_id(), Some(auxiliary));
    assert_eq!(
        transformed.set_initial(late_source_variable, 0.75),
        Err(ModelMismatchError::new(transformed.id(), source.id()))
    );
    assert_eq!(transformed.variables()[auxiliary.index()].initial, None);

    let transformed_x = transformed.variable_handle(x.var_id().unwrap());
    transformed.set_initial(transformed_x, 0.25).unwrap();
    assert_eq!(transformed.variables()[x.var_id().unwrap().index()].initial, Some(0.25));
}

#[test]
#[should_panic(expected = "after SOS constraint \"choice\" was reformulated")]
fn reformulated_model_rejects_member_bound_changes() {
    let m = Model::new("frozen_reformulated_bounds");
    variable!(m, 0.0 <= x <= 1.0);
    variable!(m, 0.0 <= y <= 1.0);
    let choice = sos_constraint!(m, choice, SOS1, [x, y]);
    let transformed = choice.to_reformulated_model(SosReformulationOptions::default()).unwrap();

    transformed.unfix_var(x.var_id().unwrap(), -10.0, 100.0);
}

#[test]
#[should_panic(expected = "after SOS constraint \"choice\" was reformulated")]
fn trivial_reformulation_also_freezes_member_bounds() {
    let m = Model::new("frozen_trivial_reformulation");
    variable!(m, x);
    variable!(m, y, fix = 0.0);
    let choice = sos_constraint!(m, choice, SOS1, [x, y]);
    choice.reformulate(SosReformulationOptions::default()).unwrap();

    m.unfix_var(y.var_id().unwrap(), -1.0, 1.0);
}

#[test]
fn chained_preserving_reformulations_retain_complete_provenance() {
    let m = Model::new("chained_provenance");
    variable!(m, 0.0 <= x <= 1.0);
    variable!(m, 0.0 <= y <= 1.0);
    variable!(m, 0.0 <= z <= 1.0);
    let first = sos_constraint!(m, first, SOS1, [x, y]);
    let second = sos_constraint!(m, second, SOS1, [y, z]);

    let once = first.to_reformulated_model(SosReformulationOptions::default()).unwrap();
    let twice = once
        .to_reformulated_sos_constraint_model(second.id(), SosReformulationOptions::default())
        .unwrap();
    let history = twice.sos_reformulations();

    assert_eq!(history.len(), 2);
    assert_eq!(history[0].source, first.id());
    assert_eq!(history[1].source, second.id());
}
