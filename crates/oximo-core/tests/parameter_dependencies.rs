use oximo_core::prelude::*;

#[test]
fn declaration_snapshots_share_storage_and_detach_only_on_registry_writes() {
    let model = Model::new("declaration snapshots");

    param!(model, p = 2.0);
    let key = model.__update_key();
    let original = model.parameters().as_ptr();
    let first = model.parameter_declarations();
    assert_eq!(first.as_ptr(), original);

    let second = model.parameter_declarations();
    assert_eq!(first.model_id(), model.id());
    assert_eq!(first.as_ptr(), second.as_ptr());
    assert_eq!(key, model.__update_key());

    model.set_param(p, 7.0).unwrap();
    assert_eq!(first.as_ptr(), model.parameter_declarations().as_ptr());

    let fork = model.fork();
    assert_eq!(first.as_ptr(), fork.parameter_declarations().as_ptr());

    param!(model, later = 3.0);
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].id, p.param_id().unwrap());
    assert_eq!(first[0].name, "p");
    assert_eq!(model.parameter_declarations().len(), 2);
    assert_ne!(first.as_ptr(), model.parameter_declarations().as_ptr());
    assert_eq!(model.parameter_declarations()[1].id, later.param_id().unwrap());
}

#[test]
fn declaration_capture_remains_valid_while_the_registry_is_borrowed() {
    let model = Model::new("borrowed declaration snapshot");

    param!(model, _p = 2.0);
    let registry = model.parameters();
    let captured = model.parameter_declarations();
    assert_eq!(captured.len(), registry.len());
    assert_eq!(captured[0].name, registry[0].name);
    assert_ne!(captured.as_ptr(), registry.as_ptr());

    drop(registry);
    param!(model, _later = 3.0);
    assert_eq!(captured.len(), 1);
    assert_eq!(model.parameter_declarations().len(), 2);
}

#[test]
fn declaration_capture_during_a_read_borrow_keeps_already_shared_storage() {
    let model = Model::new("shared borrowed declarations");

    param!(model, _p = 2.0);
    let first = model.parameter_declarations();
    let key = model.__update_key();
    let registry = model.parameters();
    let second = model.parameter_declarations();
    assert_eq!(second.as_ptr(), first.as_ptr());
    assert_eq!(second.as_ptr(), registry.as_ptr());
    assert_eq!(key, model.__update_key());

    drop(registry);
    param!(model, _later = 3.0);
    assert_eq!(first.len(), 1);
    assert_eq!(second.len(), 1);
    assert_ne!(second.as_ptr(), model.parameter_declarations().as_ptr());
}

#[test]
fn declaration_prefix_checks_handle_shared_changed_and_missing_registries() {
    let source = Model::new("prefix metadata");

    param!(source, _p = 2.0);
    let source_declarations = source.parameter_declarations();
    let fork = source.fork();
    assert!(source_declarations.matches_prefix(&fork.parameter_declarations(), 1));

    param!(fork, _generated = 3.0);
    assert!(source_declarations.matches_prefix(&fork.parameter_declarations(), 1));
    assert!(!source_declarations.matches_prefix(&fork.parameter_declarations(), 2));

    let unrelated = Model::new("different names");
    param!(unrelated, _q = 2.0);
    assert!(!source_declarations.matches_prefix(&unrelated.parameter_declarations(), 1));
    assert!(!source_declarations.matches_prefix(&Model::new("empty").parameter_declarations(), 1));
    assert!(source_declarations.matches_prefix(&Model::new("empty").parameter_declarations(), 0));
}

#[test]
fn parameter_lineage_tracks_forks_siblings_and_indexed_family_boundaries() {
    let root = Model::new("parameter lineage");

    param!(root, p[i in 0..600] = 2.0);
    let left = root.fork();
    param!(root, later = 5.0);
    let root_later = later;
    let right = root.fork();
    param!(left, later = 99.0);

    assert_eq!(left.shared_parameter_prefix(&root), Some(600));
    assert_eq!(root.shared_parameter_prefix(&right), Some(601));
    assert_eq!(left.shared_parameter_prefix(&right), Some(600));
    assert_eq!(right.shared_parameter_prefix(&left), Some(600));

    let child = left.fork();
    assert_eq!(child.shared_parameter_prefix(&left), Some(601));
    assert_eq!(child.shared_parameter_prefix(&root), Some(600));

    root.set_param_idx(&p, 0, 7.0).unwrap();
    assert_eq!(root.shared_parameter_prefix(&right), Some(601));
    assert_eq!(left.shared_parameter_prefix(&left), Some(601));
    assert_eq!(root.shared_parameter_prefix(&Model::new("independent")), None);
    assert_eq!(root.param_value(root_later.param_id().unwrap()), 5.0);
    assert_eq!(left.param_value(later.param_id().unwrap()), 99.0);
}

#[test]
fn dependencies_cover_objective_bounds_and_native_rows() {
    let model = Model::new("parameter dependencies");
    param!(model, objective_scale = 2.0);
    param!(model, row_scale = 3.0);
    param!(model, lower = 0.0);
    param!(model, upper = 10.0);
    param!(model, unused = 11.0);

    variable!(model, lower <= x <= upper);
    variable!(model, y);
    let capacity = constraint!(model, capacity, row_scale * x + row_scale * y <= 4.0);
    objective!(model, Min, objective_scale * x);

    let dependencies = model.parameter_dependencies();
    let objective_id = objective_scale.param_id().unwrap();
    let row_id = row_scale.param_id().unwrap();
    let lower_id = lower.param_id().unwrap();
    let upper_id = upper.param_id().unwrap();
    let unused_id = unused.param_id().unwrap();
    let x_id = x.var_id().unwrap();

    assert_eq!(dependencies.model_id(), model.id());
    assert_eq!(dependencies.len(), 4);
    assert!(!dependencies.is_empty());
    assert_eq!(
        dependencies.parameters().collect::<Vec<_>>(),
        vec![objective_id, row_id, lower_id, upper_id]
    );

    assert_eq!(dependencies.targets(objective_id), &[ParameterTarget::Objective]);
    assert_eq!(dependencies.targets(row_id), &[ParameterTarget::Algebraic(capacity.row_id())]);
    assert_eq!(dependencies.targets(lower_id), &[ParameterTarget::VariableBound(x_id)]);
    assert_eq!(dependencies.targets(upper_id), &[ParameterTarget::VariableBound(x_id)]);
    assert_eq!(dependencies.targets(unused_id), &[]);

    assert_eq!(dependencies.parameters_for(ParameterTarget::Objective), &[objective_id]);
    assert_eq!(
        dependencies.parameters_for(ParameterTarget::Algebraic(capacity.row_id())),
        &[row_id]
    );
    assert_eq!(
        dependencies.parameters_for(ParameterTarget::VariableBound(x_id)),
        &[lower_id, upper_id]
    );

    assert_eq!(dependencies.semantic_constraints(row_id), &[capacity.id()]);
    assert_eq!(dependencies.parameters_for_semantic(capacity.id()), &[row_id]);
}

#[test]
fn dependencies_cover_cones_and_indicators() {
    let model = Model::new("cone dependencies");
    param!(model, soc_term = 2.0);
    param!(model, soc_bound = 3.0);
    param!(model, psd_entry = 4.0);
    param!(model, indicator_rhs = 5.0);
    variable!(model, x);
    variable!(model, t >= 0.0);
    variable!(model, binary, Binary);

    let cone = soc_constraint!(model, cone, [soc_term * x] <= soc_bound * t);
    let psd =
        psd_constraint!(model, matrix, SymmetricMatrix::from_upper_triangle(1, [psd_entry * x]));
    let guard = indicator_constraint!(model, guard, binary == 1 => indicator_rhs * x <= 5.0);

    let dependencies = model.parameter_dependencies();

    assert_eq!(
        dependencies.targets(soc_term.param_id().unwrap()),
        &[ParameterTarget::SecondOrderCone(cone.soc_id())]
    );
    assert_eq!(
        dependencies.targets(soc_bound.param_id().unwrap()),
        &[ParameterTarget::SecondOrderCone(cone.soc_id())]
    );
    assert_eq!(
        dependencies.targets(psd_entry.param_id().unwrap()),
        &[ParameterTarget::PositiveSemidefinite(psd.psd_id())]
    );
    assert_eq!(
        dependencies.targets(indicator_rhs.param_id().unwrap()),
        &[ParameterTarget::Indicator(guard.indicator_id())]
    );
}

#[test]
fn dependency_snapshot_is_not_live_across_structural_changes() {
    let model = Model::new("snapshot dependencies");
    param!(model, first = 1.0);
    variable!(model, x);
    let first_row = constraint!(model, first_row, first * x <= 1.0);
    let snapshot = model.parameter_dependencies();

    param!(model, second = 2.0);
    let second_row = constraint!(model, second_row, second * x <= 2.0);

    assert_eq!(snapshot.targets(second.param_id().unwrap()), &[]);
    assert_eq!(
        snapshot.targets(first.param_id().unwrap()),
        &[ParameterTarget::Algebraic(first_row.row_id())]
    );

    let current = model.parameter_dependencies();
    assert_eq!(
        current.targets(second.param_id().unwrap()),
        &[ParameterTarget::Algebraic(second_row.row_id())]
    );
}

#[test]
fn changed_parameter_queries_filter_unaffected_targets() {
    let model = Model::new("changed dependencies");
    param!(model, row_scale = 2.0);
    param!(model, unused = 3.0);
    variable!(model, x);
    let capacity = constraint!(model, capacity, row_scale * x <= 4.0);
    let dependencies = model.parameter_dependencies();
    let before = model.parameter_snapshot();

    model.set_param(unused, 8.0).unwrap();
    let after_unused = model.parameter_snapshot();
    assert_eq!(dependencies.changed_targets(&before, &after_unused).unwrap(), Vec::new());
    assert_eq!(
        dependencies.changed_semantic_constraints(&before, &after_unused).unwrap(),
        Vec::new()
    );

    model.set_param(row_scale, 5.0).unwrap();
    let after_row = model.parameter_snapshot();
    assert_eq!(
        dependencies.changed_targets(&before, &after_row).unwrap(),
        vec![ParameterTarget::Algebraic(capacity.row_id())]
    );
    assert_eq!(
        dependencies.changed_semantic_constraints(&before, &after_row).unwrap(),
        vec![capacity.id()]
    );
}

#[test]
fn inactive_algebraic_rows_retain_dependencies_across_parameter_changes() {
    let model = Model::new("inactive dependencies");
    param!(model, p = 2.0);
    variable!(model, x);
    let row = constraint!(model, row, p * x <= 4.0);
    model.set_constraint_active(row, false).unwrap();

    let dependencies = model.parameter_dependencies();
    let target = ParameterTarget::Algebraic(row.row_id());
    assert_eq!(dependencies.targets(p.param_id().unwrap()), &[target]);
    let before = model.parameter_snapshot();
    p.set_param_value(3.0).unwrap();
    let after = model.parameter_snapshot();
    assert_eq!(dependencies.changed_targets(&before, &after).unwrap(), vec![target]);
    assert_eq!(dependencies.changed_semantic_constraints(&before, &after).unwrap(), vec![row.id()]);

    model.set_constraint_active(row, true).unwrap();
    assert_eq!(model.parameter_dependencies().targets(p.param_id().unwrap()), &[target]);
}

struct ReplacedLowering<'a> {
    declared: Expr<'a>,
    native: Expr<'a>,
}

impl Function for ReplacedLowering<'_> {
    type Owned = oximo_core::ScalarFunctionData<oximo_expr::Dynamic>;

    fn semantic_data(&self) -> Self::Owned {
        oximo_core::ScalarFunctionData::from_expression(self.declared)
    }
}

impl FunctionInSet<LessThan> for ReplacedLowering<'_> {}

impl<'a> LowerConstraint<LessThan> for ReplacedLowering<'a> {
    type Ir = oximo_core::function_set::AlgebraicConstraintIr<'a>;

    fn lower(self, set: LessThan) -> Self::Ir {
        self.native.le(set.0).into_ir()
    }
}

#[test]
fn inactive_alternative_lowerings_keep_native_and_semantic_dependencies_distinct() {
    let model = Model::new("inactive alternative dependencies");
    param!(model, declared_scale = 2.0);
    param!(model, native_scale = 3.0);
    variable!(model, x);
    let row = model.add_lowered_constraint(
        "alternative",
        Constraint::new(
            ReplacedLowering {
                declared: (declared_scale * x).erase(),
                native: (native_scale * x).erase(),
            },
            LessThan(4.0),
        )
        .into_ir(),
    );
    let target = ParameterTarget::Algebraic(row.row_id());
    let native_id = native_scale.param_id().unwrap();
    let declared_id = declared_scale.param_id().unwrap();

    for active in [true, false, true] {
        model.set_constraint_active(row, active).unwrap();
        let dependencies = model.parameter_dependencies();
        assert_eq!(dependencies.targets(native_id), &[target]);
        assert_eq!(dependencies.targets(declared_id), &[]);
        assert_eq!(dependencies.semantic_constraints(native_id), &[]);
        assert_eq!(dependencies.semantic_constraints(declared_id), &[row.id()]);
        let before = model.parameter_snapshot();
        native_scale.set_param_value(model.param_value(native_id) + 1.0).unwrap();
        let after = model.parameter_snapshot();
        assert_eq!(dependencies.changed_targets(&before, &after).unwrap(), vec![target]);
        assert_eq!(dependencies.changed_semantic_constraints(&before, &after).unwrap(), Vec::new());
    }
}

#[test]
fn changed_queries_reject_foreign_snapshots_even_with_identical_values() {
    let model = Model::new("snapshot owner");
    param!(model, p = 2.0);
    variable!(model, x);
    constraint!(model, row, p * x <= 4.0);
    let dependencies = model.parameter_dependencies();
    let local = model.parameter_snapshot();
    let foreign_model = Model::new("foreign snapshot owner");
    param!(foreign_model, foreign_p = 2.0);
    assert_eq!(p.param_id(), foreign_p.param_id());
    let foreign = foreign_model.parameter_snapshot();
    assert_ne!(local.model_id(), foreign.model_id());
    assert!(local.parameters().same_bits(foreign.parameters()));

    for (previous, current) in [(&local, &foreign), (&foreign, &local), (&foreign, &foreign)] {
        let expected = ModelMismatchError::new(model.id(), foreign_model.id());
        assert_eq!(dependencies.changed_targets(previous, current), Err(expected));
        assert_eq!(dependencies.changed_semantic_constraints(previous, current), Err(expected));
    }
    assert_eq!(dependencies.changed_targets(&local, &local.clone()).unwrap(), Vec::new());
}

#[test]
fn model_parameter_snapshots_preserve_values_across_rebinding_append_and_drop() {
    let (saved, p_id, q_id, model_id) = {
        let model = Model::new("saved values");
        param!(model, p = 2.0);
        let saved = model.parameter_snapshot();
        p.set_param_value(3.0).unwrap();
        param!(model, q = 4.0);
        (saved, p.param_id().unwrap(), q.param_id().unwrap(), model.id())
    };
    assert_eq!(saved.model_id(), model_id);
    assert_eq!(saved.parameters().value(p_id), Some(2.0));
    assert_eq!(saved.parameters().value(q_id), None);
}

#[test]
fn long_parameter_chains_and_shared_roots_produce_complete_deduplicated_dependencies() {
    let model = Model::new("long shared chain");
    variable!(model, x);
    let mut root = x.erase();
    let mut parameters = Vec::new();
    for i in 0..20_000 {
        let parameter = model.__param(format!("p{i}"), 1.0);
        parameters.push(parameter.param_id().unwrap());
        root = (root.sin() + parameter).erase();
    }
    let shared = (root * root + root).erase();
    objective!(model, Min, shared);
    let row = constraint!(model, row, shared <= 1.0);
    let dependencies = model.parameter_dependencies();
    assert_eq!(dependencies.parameters_for(ParameterTarget::Objective), parameters);
    assert_eq!(dependencies.parameters_for_semantic(row.id()), parameters);
    for parameter in parameters {
        assert_eq!(
            dependencies.targets(parameter),
            &[ParameterTarget::Objective, ParameterTarget::Algebraic(row.row_id())]
        );
        assert_eq!(dependencies.semantic_constraints(parameter), &[row.id()]);
    }
}

#[test]
fn distinct_roots_sharing_different_prefixes_keep_complete_dependencies() {
    let model = Model::new("shared prefixes");
    variable!(model, x);
    let mut root = x.erase();
    let mut parameters = Vec::new();
    let mut prefixes = Vec::new();
    for i in 0..256 {
        let parameter = model.__param(format!("p{i}"), 1.0);
        parameters.push(parameter.param_id().unwrap());
        root = (root.sin() + parameter).erase();
        if (i + 1) % 8 == 0 {
            prefixes.push((root, parameters.len()));
        }
    }
    let rows: Vec<_> = prefixes
        .into_iter()
        .rev()
        .enumerate()
        .map(|(i, (prefix, count))| {
            (model.add_constraint(format!("row{i}"), prefix.sin().le(1.0)), count)
        })
        .collect();
    let dependencies = model.parameter_dependencies();
    for &(row, count) in &rows {
        assert_eq!(
            dependencies.parameters_for(ParameterTarget::Algebraic(row.row_id())),
            &parameters[..count]
        );
        assert_eq!(dependencies.parameters_for_semantic(row.id()), &parameters[..count]);
    }
    for (i, parameter) in parameters.into_iter().enumerate() {
        let expected: Vec<_> = rows
            .iter()
            .filter(|(_, count)| *count > i)
            .map(|(row, _)| ParameterTarget::Algebraic(row.row_id()))
            .collect();
        assert_eq!(dependencies.targets(parameter), expected);
    }
}

#[test]
fn lazy_queries_preserve_encounter_order_and_deduplicate_large_fan_outs() {
    let m = Model::new("dependency iteration");

    param!(m, first = 1.0);
    param!(m, second = 2.0);
    variable!(m, x);
    let early = constraint!(m, early, second * x <= 1.0);
    let rows: Vec<_> =
        (0..12).map(|_| m.__add_constraint_auto((first * x + second * x).le(2.0))).collect();
    objective!(m, Min, first * x);
    let index = m.parameter_dependencies();
    assert!(index.contains(ParameterTarget::Objective));
    assert!(!index.contains(ParameterTarget::VariableBound(x.var_id().unwrap())));
    let pairs: Vec<_> =
        index.iter().map(|(parameter, targets)| (parameter, targets.to_vec())).collect();
    assert_eq!(
        pairs[0],
        (first.param_id().unwrap(), index.targets(first.param_id().unwrap()).to_vec())
    );
    assert_eq!(pairs[1].0, second.param_id().unwrap());

    let before = m.parameter_snapshot();
    first.set_param_value(3.0).unwrap();
    second.set_param_value(4.0).unwrap();
    let after = m.parameter_snapshot();
    let changed = index.changed_targets(&before, &after).unwrap();
    let expected: Vec<_> = std::iter::once(ParameterTarget::Objective)
        .chain(rows.iter().map(|row| ParameterTarget::Algebraic(row.row_id())))
        .chain(std::iter::once(ParameterTarget::Algebraic(early.row_id())))
        .collect();
    assert_eq!(changed, expected);
    assert_eq!(index.changed_targets_iter(&before, &after).unwrap().collect::<Vec<_>>(), expected);
    assert_eq!(
        index.changed_semantic_constraints_iter(&before, &after).unwrap().collect::<Vec<_>>(),
        index.changed_semantic_constraints(&before, &after).unwrap()
    );

    let foreign = m.fork().parameter_snapshot();
    assert!(index.changed_targets_iter(&before, &foreign).is_err());
    assert!(index.changed_semantic_constraints_iter(&foreign, &after).is_err());
}
