#![allow(clippy::float_cmp, reason = "exact reference values and activation fixtures")]
use oximo_gdp::prelude::*;
use oximo_solver::{SolutionPoint, SolverResult};

fn size(model: &Model) -> GdpModelSize {
    GdpModelSize {
        variables: model.num_variables(),
        constraints: model.num_constraints(),
        expression_nodes: model.arena().len(),
    }
}

#[test]
fn indexed_reference_coordinates_shift_affine_copies_and_last_value_wins() {
    let model = Model::new("indexed shifts");

    variable!(model, 1.0 <= x[i in 0..3] <= 4.0);
    boolean_variable!(model, active);
    for (_, &variable) in x.iter() {
        disjunct_constraint!(model, active, variable <= 3.0);
    }

    let options = GdpReformulationOptions::new(Hull::default())
        .with_hull_reference_coordinates(x.iter().map(|(_, v)| (v, 1.0)))
        .with_hull_reference_coordinate(x[1], 2.0);

    let report = model.reformulate_gdp(options).unwrap();
    for copy in &report.hull_variables {
        let reference = if copy.source == x[1].var_id().unwrap() { 2.0 } else { 1.0 };
        assert_eq!(copy.reference, reference);
        assert_eq!(copy.bounds, [1.0, 4.0]);
        let generated = &model.variables()[copy.generated.index()];
        assert_eq!([generated.lb, generated.ub], [(1.0 - reference).min(0.0), 4.0 - reference]);
    }
}

#[test]
fn compound_references_are_distinct_from_foreign_handles() {
    let model = Model::new("reference diagnostics");

    let foreign = Model::new("foreign references");
    variable!(model, 0.0 <= x <= 1.0);
    variable!(foreign, y);
    boolean_variable!(model, active);
    disjunct_constraint!(model, active, x <= 0.5);
    let compound = x + 1.0;

    let before = size(&model);
    for (options, expected) in [
        (
            GdpReformulationOptions::new(Hull::default())
                .with_hull_reference_coordinate(compound, 0.0),
            GdpError::ExpectedHullReferenceVariable,
        ),
        (
            GdpReformulationOptions::new(Hull::default()).with_hull_reference_coordinate(y, 0.0),
            GdpError::ForeignHandle("Hull reference variable"),
        ),
    ] {
        assert_eq!(model.plan_gdp(options.clone()).unwrap_err(), expected);
        assert_eq!(model.reformulate_gdp(options.clone()).unwrap_err(), expected);
        assert_eq!(model.to_reformulated_gdp_model(options).unwrap_err(), expected);
        assert_eq!(size(&model), before);
        assert!(model.has_unreformulated_gdp());
    }
}

#[test]
fn restricted_domain_diagnostics_include_intervals_scope_and_validation_context() {
    for nested in [false, true] {
        let model = Model::new("log domain");

        variable!(model, 1.0 <= x <= 4.0);
        boolean_variable!(model, active);
        boolean_variable!(model, other);
        disjunct_constraint!(model, active, logarithm, x.ln() <= 3.0_f64.ln());
        let scope = if nested {
            GdpHullScope::Disjunction(disjunction!(model, [active, other]).id())
        } else {
            GdpHullScope::Standalone(active.id())
        };

        let before = size(&model);
        for reference in [0.0, 2.0, 4.0] {
            let options = GdpReformulationOptions::new(Hull::default())
                .with_hull_reference_coordinate(x, reference);

            let error = model.plan_gdp(options).unwrap_err();
            let message = error.to_string();
            match error {
                GdpError::UnsafeHullDomain {
                    operation,
                    arguments,
                    scope: actual_scope,
                    scope_name,
                    context,
                    constraint,
                } => {
                    assert_eq!(operation, "log");
                    assert_eq!(actual_scope, scope);
                    assert!(!scope_name.is_empty());
                    assert!(message.contains(scope_name.as_str()));
                    assert!(constraint.contains("logarithm"));
                    assert_eq!(arguments.len(), 1);
                    if reference == 0.0 {
                        assert_eq!(context, GdpHullDomainContext::Reference);
                        assert_eq!(arguments[0], [0.0, 0.0]);
                    } else {
                        assert_eq!(context, GdpHullDomainContext::GeneratedBox);
                        assert!(arguments[0][0] < 0.0 && arguments[0][1] > 0.0);
                    }
                    assert!(
                        message.contains("argument intervals") && message.contains("logarithm")
                    );
                }
                other => panic!("unexpected error: {other:?}"),
            }
            assert_eq!(size(&model), before);
        }
        model
            .plan_gdp(
                GdpReformulationOptions::new(Hull::default())
                    .with_hull_reference_coordinate(x, 1.0),
            )
            .unwrap();

        assert_eq!(size(&model), before);
    }
}

#[test]
fn planning_counts_match_committed_sizes_and_limits_are_atomic() {
    for inner_hull in [false, true] {
        let model = Model::new("planned nested enumeration");

        variable!(model, -2.0 <= x <= 4.0);
        boolean_variable!(model, parent);
        boolean_variable!(model, other);
        boolean_variable!(model, child);
        boolean_variable!(model, sibling);
        disjunct_constraint!(model, parent, x <= 3.0);
        disjunct_constraint!(model, child, 0.0 <= x * x <= 4.0);
        disjunct_constraint!(model, sibling, x >= 1.0);
        disjunction!(model, [parent, other], AtLeastOne);
        let inner = disjunction!(model, [child, sibling], parent = parent);
        logical_constraint!(model, parent | other);
        let hull = Hull::default()
            .with_inclusive_policy(HullInclusivePolicy::Enumerate { max_patterns: 3 });

        let options = GdpReformulationOptions::new(hull).with_method(
            inner,
            if inner_hull { GdpMethod::from(hull) } else { GdpMethod::from(BigM::default()) },
        );

        let before = size(&model);
        let summary = model.plan_gdp(options.clone()).unwrap();

        assert_eq!(size(&model), before);
        assert!(model.gdp_reformulations().is_empty());
        assert!(summary.hull_copies > 0);
        assert_eq!(summary.hull_patterns, 3);
        for (resource, limits) in [
            (
                "variables",
                GdpSizeLimits {
                    variables: Some(summary.total.variables - 1),
                    ..GdpSizeLimits::default()
                },
            ),
            (
                "constraints",
                GdpSizeLimits {
                    constraints: Some(summary.total.constraints - 1),
                    ..GdpSizeLimits::default()
                },
            ),
            (
                "expression nodes",
                GdpSizeLimits {
                    expression_nodes: Some(summary.total.expression_nodes - 1),
                    ..GdpSizeLimits::default()
                },
            ),
        ] {
            let limited = options.clone().with_size_limits(limits);
            for error in [
                model.plan_gdp(limited.clone()).unwrap_err(),
                model.reformulate_gdp(limited.clone()).unwrap_err(),
                model.to_reformulated_gdp_model(limited).unwrap_err(),
            ] {
                assert!(
                    matches!(error, GdpError::SizeLimit { resource: actual, required, limit } if actual == resource && required > limit),
                    "{error:?}"
                );
            }
            assert_eq!(size(&model), before);
            assert!(model.gdp_reformulations().is_empty());
        }
        let limits = GdpSizeLimits {
            variables: Some(summary.total.variables),
            constraints: Some(summary.total.constraints),
            expression_nodes: Some(summary.total.expression_nodes),
        };

        let report = model.reformulate_gdp(options.with_size_limits(limits)).unwrap();

        assert_eq!(summary.total, size(&model));
        assert_eq!(summary.generated.variables, size(&model).variables - before.variables);
        assert_eq!(summary.generated.constraints, size(&model).constraints - before.constraints);
        assert_eq!(
            summary.generated.expression_nodes,
            size(&model).expression_nodes - before.expression_nodes
        );
        assert_eq!(summary.hull_copies, report.hull_variables.len());
        assert_eq!(summary.hull_patterns, report.hull_patterns.len());
        if inner_hull {
            assert!(report.hull_variables.iter().any(|c| c.source.index() >= before.variables));
        }
        let no_op = model.plan_gdp(hull).unwrap();

        assert_eq!(no_op.generated, GdpModelSize::default());
        assert_eq!(no_op.total, size(&model));
    }
}

#[test]
fn typed_activations_query_fractional_values_without_an_expression_arena() {
    let model = Model::new("typed activations");

    variable!(model, 0.0 <= x <= 4.0);
    boolean_variable!(model, active);
    disjunct_constraint!(model, active, x <= 2.0);

    let report = model.reformulate_gdp(Hull::default()).unwrap();
    let point = SolutionPoint {
        model_id: model.id(),
        primal: [(active.binary().var_id().unwrap(), 0.25)].into_iter().collect(),
        objective: None,
        parameters: model.arena().parameter_snapshot(),
    };

    let result = SolverResult {
        model_id: model.id(),
        solutions: vec![point.clone()],
        ..SolverResult::default()
    };
    for (copy, activation) in report.hull_variables.iter().zip(report.hull_activations()) {
        let expected = match copy.activation_kind {
            GdpHullActivation::Indicator { indicator, .. } => {
                assert_eq!(copy.indicator, Some(indicator));
                0.25
            }
            GdpHullActivation::Complement { .. } => {
                assert_eq!(copy.indicator, None);
                0.75
            }
            other => panic!("unexpected activation: {other:?}"),
        };
        assert_eq!(point.hull_activation_value_of(activation), Ok(Some(expected)));
        assert_eq!(result.hull_activation_value_of(activation), Ok(Some(expected)));
        assert_eq!(
            SolutionPoint { model_id: model.id(), ..SolutionPoint::default() }
                .hull_activation_value_of(activation),
            Ok(None)
        );
        assert_eq!(
            SolverResult { model_id: model.id(), ..SolverResult::default() }
                .hull_activation_value_of(activation),
            Ok(None)
        );
    }
    let inclusive = Model::new("pattern activation");
    variable!(inclusive, 0.0 <= y <= 4.0);
    boolean_variable!(inclusive, left);
    boolean_variable!(inclusive, right);
    disjunct_constraint!(inclusive, left, y <= 2.0);
    disjunct_constraint!(inclusive, right, y >= 1.0);
    let disjunction = disjunction!(inclusive, [left, right], AtLeastOne);
    let hull =
        Hull::default().with_inclusive_policy(HullInclusivePolicy::Enumerate { max_patterns: 3 });

    let report = inclusive.reformulate_gdp(hull).unwrap();
    let point = SolutionPoint {
        model_id: inclusive.id(),
        primal: report.hull_patterns.iter().map(|pattern| (pattern.weight, 0.375)).collect(),
        objective: None,
        parameters: inclusive.arena().parameter_snapshot(),
    };
    for (copy, activation) in report.hull_variables.iter().zip(report.hull_activations()) {
        assert_eq!(copy.indicator, None);
        assert!(
            matches!(copy.activation_kind, GdpHullActivation::Pattern { disjunction: source, weight } if source == disjunction.id() && report.hull_patterns.iter().any(|p| p.weight == weight))
        );
        assert_eq!(point.hull_activation_value_of(activation), Ok(Some(0.375)));
    }
}

#[test]
fn planning_totals_include_native_constraints_and_pending_suffixes() {
    let model = Model::new("native constraint totals");

    variable!(model, 0.0 <= x <= 4.0);
    variable!(model, 0.0 <= t <= 10.0);
    soc_constraint!(model, [x] <= t);
    boolean_variable!(model, first);
    disjunct_constraint!(model, first, x <= 2.0);

    let before = size(&model);
    let summary = model.plan_gdp(BigM::default()).unwrap();

    assert_eq!(size(&model), before);
    model.reformulate_gdp(BigM::default()).unwrap();

    assert_eq!(size(&model), summary.total);
    assert_eq!(summary.hull_copies, 0);
    boolean_variable!(model, second);
    disjunct_constraint!(model, second, x >= 1.0);

    let before = size(&model);
    let summary = model.plan_gdp(Hull::default()).unwrap();

    assert_eq!(size(&model), before);
    assert_eq!(summary.hull_copies, 2);
    let transformed = model.to_reformulated_gdp_model(Hull::default()).unwrap();

    assert_eq!(size(&model), before);
    assert_eq!(size(&transformed), summary.total);
    assert!(matches!(
        model.plan_gdp(GdpReformulationOptions::new(Hull::default()).with_size_limits(
            GdpSizeLimits { constraints: Some(before.constraints - 1), ..GdpSizeLimits::default() }
        )),
        Err(GdpError::SizeLimit { resource: "constraints", .. })
    ));
}

#[test]
fn division_diagnostics_preserve_operand_order_and_planning_leaves_dependencies_mutable() {
    let model = Model::new("division intervals");

    variable!(model, 1.0 <= x <= 4.0);
    variable!(model, 1.0 <= y <= 2.0);
    param!(model, limit = 3.0);
    boolean_variable!(model, active);
    disjunct_constraint!(model, active, x / y <= limit);

    let options = GdpReformulationOptions::new(Hull::default())
        .with_hull_reference_coordinates([(x, 1.0), (y, 2.0)]);
    match model.plan_gdp(options).unwrap_err() {
        GdpError::UnsafeHullDomain { operation: "division", arguments, .. } => {
            assert_eq!(arguments.len(), 2);
            assert!(arguments[0][0] > 0.0);
            assert!(arguments[1][0] < 0.0 && arguments[1][1] > 0.0);
        }
        error => panic!("unexpected error: {error:?}"),
    }

    let options = GdpReformulationOptions::new(Hull::default())
        .with_hull_reference_coordinates([(x, 1.0), (y, 1.0)]);
    model.plan_gdp(options).unwrap();
    limit.set_param_value(2.0);
    model.fix_var(x.var_id().unwrap(), 2.0);
    assert_eq!(model.param_value(limit.param_id().unwrap()), 2.0);
}

#[test]
fn nested_expansion_is_limited_across_groups_and_temporary_planning() {
    let model = Model::new("depth six inclusive expansion");

    variable!(model, -2.0 <= x <= 4.0);
    let mut parent = None;
    for level in 0..6 {
        let left = model.add_boolean(format!("left{level}"));
        let right = model.add_boolean(format!("right{level}"));
        left.context().add_constraint("limit", x.le(3.0));
        if let Some(parent) = parent {
            disjunction!(model, [left, right], parent = parent, AtLeastOne);
        } else {
            disjunction!(model, [left, right], AtLeastOne);
        }
        parent = Some(left);
    }

    let options = GdpReformulationOptions::new(
        Hull::default().with_inclusive_policy(HullInclusivePolicy::Enumerate { max_patterns: 3 }),
    );

    let before = size(&model);
    let summary = model.plan_gdp(options.clone()).unwrap();

    assert_eq!(summary.hull_patterns, 18);
    assert!(summary.hull_copies > 4_000);
    assert!(summary.generated.constraints > 19_000);
    assert!(summary.planning.row_expansions > summary.generated.constraints);
    for (resource, limits) in [
        ("variables", GdpSizeLimits { variables: Some(2_000), ..GdpSizeLimits::default() }),
        ("constraints", GdpSizeLimits { constraints: Some(2_000), ..GdpSizeLimits::default() }),
        (
            "expression nodes",
            GdpSizeLimits { expression_nodes: Some(2_000), ..GdpSizeLimits::default() },
        ),
    ] {
        let limited = options.clone().with_size_limits(limits);
        assert!(matches!(model.reformulate_gdp(limited),
            Err(GdpError::SizeLimit { resource: actual, required, limit })
            if actual == resource && required > limit));
        assert_eq!(size(&model), before);
    }
    for (resource, limits) in [
        (
            "row expansions",
            GdpPlanningLimits {
                row_expansions: Some(summary.planning.row_expansions - 1),
                ..GdpPlanningLimits::default()
            },
        ),
        (
            "private expression nodes",
            GdpPlanningLimits {
                expression_nodes: Some(summary.planning.private_expression_nodes - 1),
                ..GdpPlanningLimits::default()
            },
        ),
    ] {
        let limited = options.clone().with_planning_limits(limits);
        for error in [
            model.plan_gdp(limited.clone()).unwrap_err(),
            model.reformulate_gdp(limited.clone()).unwrap_err(),
            model.to_reformulated_gdp_model(limited).unwrap_err(),
        ] {
            assert!(
                matches!(error,
                GdpError::PlanningLimit { resource: actual, required, limit }
                if actual == resource && required > limit),
                "{error:?}"
            );
        }
        assert_eq!(size(&model), before);
        assert!(model.gdp_reformulations().is_empty());
    }
    let limit = before.expression_nodes + 5;
    let limited = options.clone().with_planning_limits(GdpPlanningLimits {
        expression_nodes: Some(limit),
        ..GdpPlanningLimits::default()
    });
    assert!(matches!(model.plan_gdp(limited),
        Err(GdpError::PlanningLimit { resource: "private expression nodes", required, limit: actual })
        if actual == limit && required == limit + 1));
    let limits = GdpSizeLimits {
        variables: Some(summary.total.variables),
        constraints: Some(summary.total.constraints),
        expression_nodes: Some(summary.total.expression_nodes),
    };
    let planning_limits = GdpPlanningLimits {
        row_expansions: Some(summary.planning.row_expansions),
        expression_nodes: Some(summary.planning.private_expression_nodes),
    };

    let report = model
        .reformulate_gdp(options.with_size_limits(limits).with_planning_limits(planning_limits))
        .unwrap();

    assert_eq!(report.planning, summary.planning);
    assert_eq!(size(&model), summary.total);
}

#[test]
fn branch_sum_activations_query_fractional_weight_values() {
    let model = Model::new("branch sum query");

    variable!(model, 0.0 <= x <= 4.0);
    let branches: Vec<_> = (0..4).map(|i| model.add_boolean(format!("b{i}"))).collect();
    for branch in &branches[..2] {
        branch.context().add_constraint("limit", x.le(2.0));
    }
    let choice = disjunction!(model, branches);

    let report = model.reformulate_gdp(Hull::default()).unwrap();
    let sum = &report.hull_branch_sums[0];
    let kind = GdpHullActivation::BranchSum { disjunction: choice.id(), weight: sum.weight };
    let activation = report.hull_activations().find(|a| a.kind() == kind).unwrap();
    let point = SolutionPoint {
        model_id: model.id(),
        primal: [(sum.weight, 0.375)].into_iter().collect(),
        objective: None,
        parameters: model.arena().parameter_snapshot(),
    };
    assert_eq!(point.hull_activation_value_of(activation), Ok(Some(0.375)));
    assert_eq!(
        SolutionPoint { model_id: model.id(), ..SolutionPoint::default() }
            .hull_activation_value_of(activation),
        Ok(None)
    );
}

#[test]
fn hull_activation_queries_check_report_result_and_solution_ownership() {
    let model = Model::new("activation ownership");

    variable!(model, 0.0 <= x <= 4.0);
    boolean_variable!(model, active);
    disjunct_constraint!(model, active, x <= 2.0);
    let transformed = model.to_reformulated_gdp_model(Hull::default()).unwrap();
    let reports = transformed.gdp_reformulations();

    let report = &reports[0];
    assert_eq!(report.model_id(), Some(transformed.id()));
    let activation = report.hull_activation(0).unwrap();

    assert_eq!(activation.model_id(), transformed.id());
    assert_eq!(activation.kind(), report.hull_variables[0].activation_kind);
    assert_eq!(report.hull_activation(report.hull_variables.len()), None);
    let point = SolutionPoint {
        model_id: transformed.id(),
        primal: [(active.binary().var_id().unwrap(), 0.25)].into_iter().collect(),
        ..SolutionPoint::default()
    };
    assert_eq!(point.hull_activation_value_of(activation), Ok(Some(0.25)));

    // Source and transformed models share numeric IDs but have different identities.
    let foreign = SolutionPoint { model_id: model.id(), ..point.clone() };
    let mismatch = ModelMismatchError::new(model.id(), transformed.id());
    assert_eq!(foreign.hull_activation_value_of(activation), Err(mismatch));
    let empty = SolutionPoint { model_id: model.id(), ..SolutionPoint::default() };
    assert_eq!(empty.hull_activation_value_of(activation), Err(mismatch));
    let empty_result = SolverResult { model_id: model.id(), ..SolverResult::default() };
    assert_eq!(empty_result.hull_activation_value_of(activation), Err(mismatch));
    let inconsistent_result = SolverResult {
        model_id: transformed.id(),
        solutions: vec![foreign],
        ..SolverResult::default()
    };
    assert_eq!(inconsistent_result.hull_activation_value_of(activation), Err(mismatch));

    let cloned = transformed.to_reformulated_gdp_model(Hull::default()).unwrap();
    let cloned_reports = cloned.gdp_reformulations();
    assert_eq!(cloned_reports[0].model_id(), Some(cloned.id()));
    assert!(std::ptr::eq(
        report.hull_variables.as_ptr(),
        cloned_reports[0].hull_variables.as_ptr()
    ));
    let cloned_activation = cloned_reports[0].hull_activation(0).unwrap();
    let cloned_point = SolutionPoint { model_id: cloned.id(), ..point.clone() };
    assert_eq!(cloned_point.hull_activation_value_of(cloned_activation), Ok(Some(0.25)));
    assert_eq!(
        cloned_point.hull_activation_value_of(activation),
        Err(ModelMismatchError::new(cloned.id(), transformed.id()))
    );
    assert_eq!(point.hull_activation_value_of(activation), Ok(Some(0.25)));

    let unbound = GdpReformulationReport::from(report.clone().into_owned());
    assert_eq!(unbound.model_id(), None);
    assert_eq!(unbound.hull_activation(0), None);
    assert_eq!(unbound.hull_activations().count(), 0);
}
