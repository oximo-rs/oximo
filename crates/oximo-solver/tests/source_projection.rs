use oximo_core::SemanticConstraintId;
use oximo_core::VarId;
use oximo_core::prelude::*;
use oximo_solver::reconstruct::{
    ObjectiveTransform, ProjectionError, SourceCoordinates, SourceProjection, SourceProjectionPlan,
};
use oximo_solver::{DualStatus, PrimalStatus, SolutionPoint, SolverResult};

#[test]
fn descendant_projection_rejects_late_source_variable_collisions_before_reevaluation() {
    let source = Model::new("late source variable");

    variable!(source, x);
    let plan = SourceProjectionPlan::capture(&source);
    let transformed = source.fork();
    variable!(source, y);
    let source_row = constraint!(source, y <= 1.0);
    variable!(transformed, z);
    let native_row = constraint!(transformed, z <= 9.0);
    assert_eq!(y.var_id(), z.var_id());
    assert_eq!(source_row.row_id(), native_row.row_id());

    let mut result = SolverResult::for_model(&transformed);
    result.dual_status = DualStatus::FeasiblePoint;
    result.add_solution(
        [(x.var_id().unwrap(), 1.0), (z.var_id().unwrap(), 9.0)].into_iter().collect(),
        None,
    );
    result.dual.insert(native_row.row_id(), 7.0);
    let calls = std::cell::Cell::new(0);
    let projection = SourceProjection::capture(&source)
        .from_transformed(&transformed)
        .reevaluate_objective(|point| {
            calls.set(calls.get() + 1);
            point.value_of(2.0 * y).ok().flatten()
        });
    let error =
        ProjectionError::CoordinatesNotInherited { kind: "variable", count: 2, inherited: 1 };
    assert_eq!(projection.validate(&result).unwrap_err(), error);
    assert_eq!(projection.project(result.clone()).unwrap_err(), error);
    assert_eq!(calls.get(), 0);

    let explicit = SourceProjection::with_plan(&source, &plan)
        .with_source_variables([y.var_id().unwrap()])
        .from_transformed(&transformed);
    assert_eq!(explicit.project(result.clone()).unwrap_err(), error);

    let retained = SourceProjection::with_plan(&source, &plan)
        .retain_variables([oximo_core::GeneratedVarId::from_raw(z.var_id().unwrap())])
        .from_transformed(&transformed);
    assert_eq!(retained.project(result.clone()).unwrap_err(), error);

    let projected = SourceProjection::with_plan(&source, &plan)
        .from_transformed(&transformed)
        .project(result)
        .unwrap();
    assert_eq!(projected.value_of(x), Ok(Some(1.0)));
    assert_eq!(projected.value_of(y), Ok(None));
    assert_eq!(projected.value_of(2.0 * y), Ok(None));
    assert_eq!(projected.dual_of(source_row), Ok(None));
}

#[test]
fn descendant_projection_rejects_late_source_row_collisions_for_prefix_and_explicit_plans() {
    let source = Model::new("late source row");

    variable!(source, x);
    let inherited_row = constraint!(source, x <= 10.0);
    let plan = SourceProjectionPlan::capture(&source);
    let transformed = source.fork().fork();
    let source_row = constraint!(source, x <= 1.0);
    let native_x = transformed.variable_handle(x.var_id().unwrap());
    let native_row = constraint!(transformed, native_x <= 9.0);
    assert_eq!(source_row.row_id(), native_row.row_id());

    let mut result = SolverResult::for_model(&transformed);
    result.dual_status = DualStatus::FeasiblePoint;
    result.add_solution([(x.var_id().unwrap(), 9.0)].into_iter().collect(), None);
    result.dual.insert(inherited_row.row_id(), 3.0);
    result.dual.insert(native_row.row_id(), 7.0);
    result.semantic_dual.insert(native_row.id(), Some(7.0));
    let error =
        ProjectionError::CoordinatesNotInherited { kind: "algebraic row", count: 2, inherited: 1 };

    for projection in [
        SourceProjection::capture(&source),
        SourceProjection::with_plan(&source, &plan).with_source_rows([source_row.row_id()]),
    ] {
        let projection = projection.from_transformed(&transformed);
        assert_eq!(projection.validate(&result).unwrap_err(), error);
        assert_eq!(projection.project(result.clone()).unwrap_err(), error);
    }
    let projected = SourceProjection::with_plan(&source, &plan)
        .from_transformed(&transformed)
        .project(result)
        .unwrap();
    assert_eq!(projected.dual_of(inherited_row), Ok(Some(3.0)));
    assert_eq!(projected.dual_of(source_row), Ok(None));
    assert!(!projected.semantic_dual.contains_key(&source_row.id()));
}

#[test]
fn descendant_projection_rejects_late_source_cones() {
    let source = Model::new("late source cones");

    variable!(source, x);
    variable!(source, t >= 0.0);
    let transformed = source.fork();
    soc_constraint!(source, [x] <= t);
    psd_constraint!(source, SymmetricMatrix::from_upper_triangle(1, [x]));
    let result = SolverResult::for_model(&transformed);
    assert_eq!(
        SourceProjection::capture(&source).from_transformed(&transformed).validate(&result),
        Err(ProjectionError::CoordinatesNotInherited {
            kind: "SOC constraint",
            count: 1,
            inherited: 0,
        }),
    );

    let coordinates =
        SourceCoordinates { soc_constraints: 0, ..SourceCoordinates::capture(&source) };
    assert_eq!(
        SourceProjection::with_coordinates(&source, coordinates)
            .from_transformed(&transformed)
            .project(result)
            .unwrap_err(),
        ProjectionError::CoordinatesNotInherited { kind: "PSD constraint", count: 1, inherited: 0 },
    );
}

#[test]
fn projection_rejects_unrelated_parameterless_models_with_matching_coordinate_counts() {
    let source = Model::new("parameterless source");

    variable!(source, x);
    let transformed = Model::new("unrelated parameterless native");
    variable!(transformed, z);
    assert_eq!(x.var_id(), z.var_id());

    let mut result = SolverResult::for_model(&transformed);
    result.add_solution([(z.var_id().unwrap(), 9.0)].into_iter().collect(), None);
    let projection = SourceProjection::capture(&source).from_transformed(&transformed);
    let error = ProjectionError::UnrelatedTransformation {
        source_model: source.id(),
        transformed_model: transformed.id(),
    };
    assert_eq!(projection.validate(&result).unwrap_err(), error);
    assert_eq!(result.value_of(z), Ok(Some(9.0)));
    assert_eq!(projection.project(result).unwrap_err(), error);
}

#[test]
fn projection_rejects_sibling_forks_with_colliding_variable_and_row_ids() {
    let root = Model::new("sibling root");

    param!(root, _p = 2.0);
    variable!(root, x);
    constraint!(root, shared, x <= 10.0);
    let source = root.fork();
    let transformed = root.fork();
    variable!(source, y);
    variable!(transformed, z);
    let source_row = constraint!(source, y <= 1.0);
    let native_row = constraint!(transformed, z <= 9.0);
    assert_eq!(y.var_id(), z.var_id());
    assert_eq!(source_row.row_id(), native_row.row_id());
    assert_eq!(source.shared_parameter_prefix(&transformed), Some(1));

    let mut result = SolverResult::for_model(&transformed);
    result.add_solution(
        [(x.var_id().unwrap(), 1.0), (z.var_id().unwrap(), 9.0)].into_iter().collect(),
        None,
    );
    let projection = SourceProjection::capture(&source).from_transformed(&transformed);
    let error = ProjectionError::UnrelatedTransformation {
        source_model: source.id(),
        transformed_model: transformed.id(),
    };
    assert_eq!(projection.validate(&result).unwrap_err(), error);
    assert_eq!(projection.project(result).unwrap_err(), error);
}

#[test]
fn projection_rejects_a_transformed_model_that_is_an_ancestor_of_the_source() {
    let transformed = Model::new("native ancestor");

    param!(transformed, _p = 2.0);
    variable!(transformed, x);
    let source = transformed.fork();
    let mut result = SolverResult::for_model(&transformed);
    result.add_solution([(x.var_id().unwrap(), 3.0)].into_iter().collect(), None);
    let projection = SourceProjection::capture(&source).from_transformed(&transformed);
    let error = ProjectionError::UnrelatedTransformation {
        source_model: source.id(),
        transformed_model: transformed.id(),
    };
    assert_eq!(projection.validate(&result).unwrap_err(), error);
    assert_eq!(projection.project(result).unwrap_err(), error);
}

#[test]
fn projection_accepts_a_transitive_descendant_and_uses_its_shared_parameter_values() {
    let source = Model::new("descendant source");

    param!(source, p = 2.0);
    variable!(source, x);
    let child = source.fork();
    let transformed = child.fork();
    transformed.set_param_id(p.param_id().unwrap(), 4.0).unwrap();
    variable!(transformed, generated);
    let mut result = SolverResult::for_model(&transformed);
    result.add_solution(
        [(x.var_id().unwrap(), 3.0), (generated.var_id().unwrap(), 9.0)].into_iter().collect(),
        None,
    );
    let projection = SourceProjection::capture(&source)
        .from_transformed(&transformed)
        .reevaluate_objective_expr(p * x)
        .unwrap();
    assert!(projection.validate(&result).is_ok());

    let projected = projection.project(result).unwrap();
    assert_eq!(projected.objective(), Some(12.0));
    assert_eq!(projected.value_of(x), Ok(Some(3.0)));
    assert_eq!(projected.primal().unwrap().get(generated.var_id().unwrap()), None);
}

#[test]
fn captured_plan_debug_summarizes_large_parameter_registries() {
    let source = Model::new("compact plan debug");

    param!(source, _debug_parameter[i in 0..600] = 2.0);
    let plan = SourceProjectionPlan::capture(&source);
    let debug = format!("{plan:?}");
    assert!(debug.contains("parameter_count: Some(600)"));
    assert!(debug.contains("source_model: Some("));
    assert!(!debug.contains("_debug_parameter"));
    assert!(debug.len() < 512);
}

#[test]
fn projection_validates_every_point_before_rebinding_or_evaluating() {
    use std::cell::Cell;

    let source = Model::new("source");

    param!(source, _p = 2.0);
    variable!(source, x);
    let transformed = source.fork();
    let foreign = source.fork();

    for native in [&source, &transformed] {
        for point_owner in [&source, &transformed, &foreign] {
            if point_owner.id() == native.id() {
                continue;
            }

            let evaluated = Cell::new(false);
            let mut result = SolverResult::for_model(native);
            result.add_solution([(x.var_id().unwrap(), 1.0)].into_iter().collect(), None);
            result.solutions.push(SolutionPoint::new(
                point_owner,
                [(x.var_id().unwrap(), 3.0)].into_iter().collect(),
                None,
            ));
            let error = SourceProjection::capture(&source)
                .from_transformed(native)
                .reevaluate_objective(|_| {
                    evaluated.set(true);
                    Some(0.0)
                })
                .project(result)
                .unwrap_err();
            assert!(matches!(
                error,
                ProjectionError::ForeignPoint { solution: 1, expected, actual }
                    if expected == native.id() && actual == point_owner.id()
            ));
            assert!(!evaluated.get());
        }
    }
}

#[test]
fn projection_returns_parameter_mismatches_before_objective_evaluation() {
    use std::cell::Cell;

    let source = Model::new("conflicting point parameters");

    param!(source, p = 2.0);
    variable!(source, x);

    for detached in [false, true] {
        source.set_param(p, 2.0).unwrap();
        let mut result = SolverResult::for_model(&source);
        source.set_param(p, 3.0).unwrap();
        let mut point =
            SolutionPoint::new(&source, [(x.var_id().unwrap(), 1.0)].into_iter().collect(), None);
        if detached {
            point.model_id = ModelId::UNASSIGNED;
        }
        result.solutions.push(point);
        let evaluated = Cell::new(false);
        let error = SourceProjection::capture(&source)
            .reevaluate_objective(|_| {
                evaluated.set(true);
                Some(0.0)
            })
            .project(result)
            .unwrap_err();
        assert_eq!(
            error,
            ProjectionError::ParameterSnapshotMismatch {
                solution: 0,
                result_parameters: 1,
                point_parameters: 1,
                changed_parameters: vec![p.param_id().unwrap()],
            }
        );
        assert!(!evaluated.get());
    }
}

#[test]
fn detached_points_inherit_solve_parameters_before_objective_evaluation() {
    let source = Model::new("detached projection");

    param!(source, p = 2.0);
    variable!(source, x);
    let transformed = source.fork();
    let mut result = SolverResult::for_model(&transformed);
    result.solutions.push(SolutionPoint {
        primal: [(x.var_id().unwrap(), 3.0)].into_iter().collect(),
        ..Default::default()
    });
    source.set_param(p, 9.0).unwrap();
    let projected = SourceProjection::capture(&source)
        .from_transformed(&transformed)
        .reevaluate_objective(|point| point.value_of(p * x).unwrap())
        .project(result)
        .unwrap();
    assert_eq!(projected.objective(), Some(6.0));
    assert_eq!(projected.best().unwrap().parameters, projected.parameters);
}

#[test]
fn expression_projection_keeps_the_captured_function_and_solve_parameters() {
    let source = Model::new("captured source objective");

    param!(source, p = 2.0);
    variable!(source, x);
    let expression = p * x * x;
    objective!(source, Min, expression);
    let transformed = source.fork();
    let projection = SourceProjection::capture(&source)
        .from_transformed(&transformed)
        .reevaluate_objective_expr(expression)
        .unwrap();

    let mut result = SolverResult::for_model(&transformed);
    result.termination = oximo_solver::TerminationStatus::TimeLimit;
    result.add_solution([(x.var_id().unwrap(), 3.0)].into_iter().collect(), Some(99.0));
    result.add_solution([(x.var_id().unwrap(), 4.0)].into_iter().collect(), Some(99.0));
    result.best_bound = Some(1.0);
    source.set_param(p, 9.0).unwrap();
    transformed.set_param_id(p.param_id().unwrap(), 7.0).unwrap();
    objective!(source, Min, 100.0 * x);

    let projected = projection.project(result).unwrap();
    assert_eq!(projected.objective(), Some(18.0));
    assert_eq!(projected.solution(1).unwrap().objective, Some(32.0));
    assert_eq!(projected.best_bound, Some(1.0));
    assert_eq!(projected.primal_status, PrimalStatus::FeasiblePoint);
}

#[test]
fn expression_projection_rejects_foreign_expressions_at_registration() {
    let source = Model::new("expression owner");

    variable!(source, _x);
    let foreign = Model::new("foreign expression owner");
    variable!(foreign, y);

    let error = SourceProjection::capture(&source).reevaluate_objective_expr(y).unwrap_err();
    assert_eq!(
        error,
        ProjectionError::ForeignExpression { expected: source.id(), actual: foreign.id() }
    );
}

#[test]
fn projection_rejects_aliased_parameters_and_preserves_the_result_during_preflight() {
    let source = Model::new("source parameter meanings");

    param!(source, p = 2.0);
    variable!(source, x);
    let transformed = Model::new("unrelated parameter meanings");
    param!(transformed, q = 7.0);
    variable!(transformed, x2);
    assert_eq!(p.param_id(), q.param_id());

    let mut result = SolverResult::for_model(&transformed);
    result.add_solution([(x2.var_id().unwrap(), 3.0)].into_iter().collect(), None);
    let projection = SourceProjection::capture(&source)
        .from_transformed(&transformed)
        .reevaluate_objective_expr(p * x)
        .unwrap();
    let error = projection.validate(&result).unwrap_err();
    assert_eq!(
        error,
        ProjectionError::UnrelatedTransformation {
            source_model: source.id(),
            transformed_model: transformed.id(),
        }
    );
    assert_eq!(result.model_id, transformed.id());
    assert_eq!(result.value_of(q * x2), Ok(Some(21.0)));
    assert_eq!(projection.project(result).unwrap_err(), error);
}

#[test]
fn transformed_parameter_prefix_requires_all_source_ids_but_allows_generated_suffixes() {
    let source = Model::new("source prefix");

    param!(source, p = 2.0);
    variable!(source, x);
    let missing = Model::new("missing parameter");
    let projection = SourceProjection::capture(&source).from_transformed(&missing);
    assert_eq!(
        projection.validate(&SolverResult::for_model(&missing)),
        Err(ProjectionError::UnrelatedTransformation {
            source_model: source.id(),
            transformed_model: missing.id(),
        })
    );

    let transformed = source.fork();
    param!(transformed, _generated = 99.0);
    transformed.set_param_id(p.param_id().unwrap(), 7.0).unwrap();
    let mut result = SolverResult::for_model(&transformed);
    result.add_solution([(x.var_id().unwrap(), 3.0)].into_iter().collect(), None);
    let projected = SourceProjection::capture(&source)
        .from_transformed(&transformed)
        .reevaluate_objective_expr(p * x)
        .unwrap()
        .project(result)
        .unwrap();
    assert_eq!(projected.objective(), Some(21.0));
    assert_eq!(projected.parameters.values(), &[7.0]);
    param!(source, later = 5.0);
    assert_eq!(projected.value_of(later), Ok(None));
    assert_eq!(projected.best().unwrap().value_of(later), Ok(None));
}

#[test]
fn projection_rejects_an_empty_result_snapshot_and_allows_repair_after_preflight() {
    let source = Model::new("empty result snapshot");

    param!(source, p = 2.0);
    variable!(source, x);
    let point =
        SolutionPoint::new(&source, [(x.var_id().unwrap(), 3.0)].into_iter().collect(), None);
    let mut result =
        SolverResult { model_id: source.id(), solutions: vec![point], ..Default::default() };
    let projection = SourceProjection::capture(&source).reevaluate_objective_expr(p * x).unwrap();
    let error = ProjectionError::ParameterSnapshotMismatch {
        solution: 0,
        result_parameters: 0,
        point_parameters: 1,
        changed_parameters: vec![p.param_id().unwrap()],
    };
    assert_eq!(projection.validate(&result).unwrap_err(), error);
    assert_eq!(projection.project(result.clone()).unwrap_err(), error);

    result.parameters = result.solutions[0].parameters.clone();
    assert!(projection.validate(&result).is_ok());
    assert_eq!(projection.project(result).unwrap().objective(), Some(6.0));
}

#[test]
fn late_source_parameters_are_unavailable_after_a_fork_even_with_a_late_projector() {
    let source = Model::new("late source parameter");

    param!(source, p = 2.0);
    variable!(source, x);
    let transformed = source.fork();
    let mut result = SolverResult::for_model(&transformed);
    result.add_solution([(x.var_id().unwrap(), 3.0)].into_iter().collect(), None);
    param!(source, later = 5.0);

    for (expression, expected) in [(p * x, Some(6.0)), (later * x, None)] {
        let projection = SourceProjection::capture(&source)
            .from_transformed(&transformed)
            .reevaluate_objective_expr(expression)
            .unwrap();
        assert!(projection.validate(&result).is_ok());

        let projected = projection.project(result.clone()).unwrap();
        assert_eq!(projected.objective(), expected);
        assert_eq!(projected.value_of(later), Ok(None));
        assert_eq!(projected.parameters.values(), &[2.0]);
    }
}

#[test]
fn late_same_named_source_parameters_do_not_alias_a_generated_fork_suffix() {
    let source = Model::new("same names after divergence");

    param!(source, p = 2.0);
    variable!(source, x);
    let transformed = source.fork();
    param!(transformed, generated = 99.0);
    let mut result = SolverResult::for_model(&transformed);
    result.add_solution([(x.var_id().unwrap(), 3.0)].into_iter().collect(), None);
    let native_id = generated.param_id();
    param!(source, generated = 5.0);
    assert_eq!(native_id, generated.param_id());

    let projected = SourceProjection::capture(&source)
        .from_transformed(&transformed)
        .reevaluate_objective_expr(generated * x)
        .unwrap()
        .project(result)
        .unwrap();
    assert!(projected.has_solution());
    assert_eq!(projected.objective(), None);
    assert_eq!(projected.value_of(generated), Ok(None));
    assert_eq!(projected.value_of(p), Ok(Some(2.0)));
    assert_eq!(projected.parameters.values(), &[2.0]);
}

#[test]
fn captured_plans_freeze_parameters_and_reject_a_different_source_owner() {
    let source = Model::new("captured plan parameters");

    param!(source, p = 2.0);
    variable!(source, x);
    let plan = SourceProjectionPlan::capture(&source);
    let projection = SourceProjection::capture(&source);
    param!(source, later = 5.0);
    let mut result = SolverResult::for_model(&source);
    result.add_solution([(x.var_id().unwrap(), 3.0)].into_iter().collect(), None);

    for projection in [projection, SourceProjection::with_plan(&source, &plan)] {
        let projected = projection.project(result.clone()).unwrap();
        assert_eq!(projected.value_of(p), Ok(Some(2.0)));
        assert_eq!(projected.value_of(later), Ok(None));
    }

    let foreign = source.fork();
    let error = SourceProjection::with_plan(&foreign, &plan)
        .validate(&SolverResult::for_model(&foreign))
        .unwrap_err();
    assert_eq!(
        error,
        ProjectionError::ForeignSourcePlan { expected: foreign.id(), actual: source.id() }
    );
}

#[test]
fn independently_registered_same_named_parameters_are_not_shared_lineage() {
    let source = Model::new("independent source");

    let early_fork = source.fork();
    param!(source, _p = 2.0);
    let native = Model::new("independent native");
    param!(native, _p = 7.0);

    let error = SourceProjection::capture(&source)
        .from_transformed(&early_fork)
        .from_transformed(&native)
        .validate(&SolverResult::for_model(&native))
        .unwrap_err();
    assert_eq!(
        error,
        ProjectionError::UnrelatedTransformation {
            source_model: source.id(),
            transformed_model: native.id(),
        }
    );
}

#[test]
fn callbacks_observe_restored_objectives_and_can_explicitly_return_unavailable() {
    let source = Model::new("callback objective convention");

    variable!(source, x);
    let mut result = SolverResult::for_model(&source);

    result.add_solution([(x.var_id().unwrap(), 3.0)].into_iter().collect(), Some(-10.0));
    result.best_bound = Some(-8.0);
    for fallback in [true, false] {
        let projected = SourceProjection::capture(&source)
            .with_objective(ObjectiveTransform { sign: -1.0, offset: 5.0 })
            .reevaluate_objective(|point| {
                assert_eq!(point.objective, Some(15.0));
                if fallback { point.objective } else { None }
            })
            .project(result.clone())
            .unwrap();
        assert_eq!(projected.objective(), fallback.then_some(15.0));
        assert_eq!(projected.best_bound, Some(13.0));
    }
}

#[test]
fn mismatch_diagnostics_identify_changed_ids_in_equal_length_snapshots() {
    let source = Model::new("snapshot bit diagnostics");

    param!(source, zero = 0.0);
    param!(source, weight = 2.0);
    let mut result = SolverResult::for_model(&source);
    source.set_params(&[(zero, -0.0), (weight, 3.0)]).unwrap();
    result.solutions.push(SolutionPoint::new(
        &source,
        oximo_solver::SolutionValues::default(),
        None,
    ));

    let error = SourceProjection::capture(&source).validate(&result).unwrap_err();
    assert_eq!(
        error,
        ProjectionError::ParameterSnapshotMismatch {
            solution: 0,
            result_parameters: 2,
            point_parameters: 2,
            changed_parameters: vec![zero.param_id().unwrap(), weight.param_id().unwrap()],
        }
    );
    assert!(error.to_string().contains("ParamId(0)"));
    assert!(error.to_string().contains("ParamId(1)"));
}

#[test]
fn projection_preserves_matching_nondetached_snapshot_storage() {
    let source = Model::new("independent matching snapshots");

    param!(source, p = 2.0);
    variable!(source, x);
    let mut result = SolverResult::for_model(&source);
    source.set_param(p, 3.0).unwrap();
    source.set_param(p, 2.0).unwrap();

    let point =
        SolutionPoint::new(&source, [(x.var_id().unwrap(), 3.0)].into_iter().collect(), None);
    let pointer = point.parameters.values().as_ptr();
    assert_ne!(pointer, result.parameters.values().as_ptr());

    let projection = SourceProjection::capture(&source).reevaluate_objective_expr(p * x).unwrap();
    for owner in [source.id(), ModelId::UNASSIGNED] {
        let mut point = point.clone();
        point.model_id = owner;
        result.solutions = vec![point];
        let projected = projection.project(result.clone()).unwrap();
        assert_eq!(projected.best().unwrap().model_id, source.id());
        assert_eq!(projected.best().unwrap().parameters.values().as_ptr(), pointer);
        assert_eq!(projected.objective(), Some(6.0));
    }
}

#[test]
fn captured_expression_queries_validate_point_ownership_and_saved_parameters() {
    let source = Model::new("mutable source graph");

    param!(source, p = 2.0);
    variable!(source, x);
    let expression = p * x * x;
    let captured = expression.capture();
    let mut result = SolverResult::for_model(&source);
    result.add_solution([(x.var_id().unwrap(), 3.0)].into_iter().collect(), None);
    let projection =
        SourceProjection::capture(&source).reevaluate_objective_expr(expression).unwrap();
    expression.arena().borrow_mut().constant(99.0);
    source.set_param(p, 7.0).unwrap();
    let projected = projection.project(result).unwrap();
    assert_eq!(projected.objective(), Some(18.0));
    assert_eq!(projected.best().unwrap().value_of_captured(&captured), Ok(Some(18.0)));

    let foreign = source.fork();
    let point =
        SolutionPoint::new(&foreign, [(x.var_id().unwrap(), 3.0)].into_iter().collect(), None);
    assert!(point.value_of_captured(&captured).is_err());
}

#[test]
fn builder_errors_convert_to_projection_errors_for_question_mark_chains() {
    fn project(
        source: &Model,
        expression: Expr<'_>,
        result: SolverResult,
    ) -> Result<SolverResult, ProjectionError> {
        SourceProjection::capture(source).reevaluate_objective_expr(expression)?.project(result)
    }

    let source = Model::new("source");
    variable!(source, x);
    let foreign = Model::new("foreign");
    variable!(foreign, y);

    let error = oximo_core::ModelMismatchError::new(source.id(), foreign.id());
    assert_eq!(
        ProjectionError::from(error),
        ProjectionError::ForeignExpression { expected: source.id(), actual: foreign.id() }
    );
    assert!(project(&source, y.erase(), SolverResult::for_model(&source)).is_err());
    assert!(project(&source, x.erase(), SolverResult::for_model(&source)).is_ok());
}

#[test]
fn expression_projection_does_not_fall_back_to_current_parameters_or_native_values() {
    let source = Model::new("missing saved parameter");

    variable!(source, x);
    let mut result = SolverResult::for_model(&source);
    result.add_solution([(x.var_id().unwrap(), 3.0)].into_iter().collect(), Some(99.0));
    param!(source, later = 2.0);

    let projected = SourceProjection::capture(&source)
        .reevaluate_objective_expr(later * x)
        .unwrap()
        .project(result)
        .unwrap();
    assert_eq!(projected.objective(), None);
    assert!(projected.has_solution());
}

#[test]
fn expression_projection_reports_nonfinite_objectives_as_unavailable() {
    let source = Model::new("undefined source objective");

    variable!(source, x);
    let mut result = SolverResult::for_model(&source);
    result.add_solution([(x.var_id().unwrap(), 0.0)].into_iter().collect(), Some(99.0));
    let projected = SourceProjection::capture(&source)
        .reevaluate_objective_expr(1.0 / x)
        .unwrap()
        .project(result)
        .unwrap();
    assert_eq!(projected.objective(), None);
    assert!(projected.has_solution());
}

#[test]
fn borrowed_plan_preserves_interleaved_source_rows_and_reuses_membership() {
    let source = Model::new("reusable projection plan");
    variable!(source, x);
    variable!(source, generated);
    variable!(source, z);
    constraint!(source, first, x <= 1.0);
    constraint!(source, generated_row, generated <= 2.0);
    constraint!(source, last, z <= 3.0);
    let first = source.constraint_id("first").unwrap();
    let last = source.constraint_id("last").unwrap();
    let generated_row = source.constraint_id("generated_row").unwrap();
    let x_id = x.var_id().unwrap();
    let z_id = z.var_id().unwrap();
    let generated_id = generated.var_id().unwrap();
    let cone = soc_constraint!(source, [x] <= z);
    let scalar_handle = |id| {
        source.require_scalar_dual_handle(source.constraint_handle_from_id(id).unwrap()).unwrap()
    };
    let plan = SourceProjectionPlan::new(SourceCoordinates::capture(&source))
        .with_source_variables([x_id, z_id, x_id])
        .with_source_rows([first, last, first])
        .retain_variables([oximo_core::GeneratedVarId::from_raw(generated_id)]);
    let projection = SourceProjection::with_plan(&source, &plan);
    let mut result = SolverResult::for_model(&source);
    result.dual_status = DualStatus::FeasiblePoint;
    result
        .add_solution([(x_id, 1.0), (z_id, 2.0), (generated_id, 0.5)].into_iter().collect(), None);
    result.dual = [(first, 1.0), (generated_row, 9.0), (last, 2.0)].into_iter().collect();
    result.semantic_dual = [
        (SemanticConstraintId::Algebraic(first), Some(5.0)),
        (SemanticConstraintId::Algebraic(generated_row), Some(7.0)),
        (SemanticConstraintId::Algebraic(last), None),
        (cone.id(), Some(3.0)),
    ]
    .into_iter()
    .collect();
    result.reduced_costs = [(x_id, 1.0), (generated_id, 9.0), (z_id, 2.0)].into_iter().collect();
    for generated_multiplier in [Some(7.0), None, Some(7.0)] {
        result
            .semantic_dual
            .insert(SemanticConstraintId::Algebraic(generated_row), generated_multiplier);
        let projected = projection.project(result.clone()).unwrap();
        assert_eq!(projected.primal().unwrap().len(), 3);
        assert_eq!(projected.dual.len(), 2);
        assert_eq!(projected.dual.get(last), Some(2.0));
        assert!(projected.dual.get(generated_row).is_none());
        assert!(
            !projected.semantic_dual.contains_key(&SemanticConstraintId::Algebraic(generated_row))
        );
        assert_eq!(projected.dual_of(scalar_handle(generated_row)).unwrap(), None);
        assert_eq!(projected.dual_of(scalar_handle(first)).unwrap(), Some(5.0));
        assert_eq!(
            projected.semantic_dual.get(&SemanticConstraintId::Algebraic(last)),
            Some(&None)
        );
        assert_eq!(projected.dual_of(scalar_handle(last)).unwrap(), None);
        assert_eq!(projected.semantic_dual.get(&cone.id()), Some(&Some(3.0)));
        assert_eq!(projected.reduced_costs.len(), 2);
        assert!(projected.reduced_costs.get(generated_id).is_none());
    }
    // The same immutable plan remains usable by another projector.
    assert!(SourceProjection::with_plan(&source, &plan).project(result).unwrap().has_solution());
}

#[test]
fn explicit_source_rows_validate_indices_instead_of_only_set_length() {
    let source = Model::new("explicit source row bounds");
    variable!(source, x);
    constraint!(source, row, x <= 1.0);
    let error = SourceProjection::capture(&source)
        .with_source_rows([oximo_core::ConstraintId(1)])
        .project(SolverResult::for_model(&source))
        .unwrap_err();
    assert!(matches!(
        error,
        ProjectionError::CoordinatesExceedSource { kind: "algebraic row", count: 2, available: 1 }
    ));
}

#[test]
fn projection_filters_generated_entities_and_rebuilds_the_source_objective() {
    let source = Model::new("source projection");
    variable!(source, 0.0 <= x <= 1.0);
    variable!(source, 0.0 <= y <= 1.0);
    param!(source, weight = 1.0);
    let capacity = source.add_constraint("capacity", (x + y).le(1.0));
    sos_constraint!(source, choice, SOS1, [x, y]);
    objective!(source, Max, weight * (x + y));

    let coordinates = SourceCoordinates::capture(&source);

    source.set_param(weight, 2.0).unwrap();
    let transformed = source.to_reformulated_sos_model(SosReformulationOptions::default()).unwrap();
    source.set_param(weight, 1.0).unwrap();

    let history = transformed.sos_reformulations();
    let artifacts = &history[0];
    let generated_variable = artifacts.variables[0].raw_id();
    let generated_constraint = artifacts.constraints[0].raw_id();
    let sos_id = artifacts.source;

    let solve_parameters = transformed.arena().parameter_snapshot();
    let mut result = SolverResult::for_model(&transformed);
    result.dual_status = DualStatus::FeasiblePoint;

    let mut primal = oximo_solver::SolutionValues::default();
    primal.insert(x.var_id().unwrap(), 0.5);
    primal.insert(y.var_id().unwrap(), 0.0);
    primal.insert(generated_variable, 0.0);
    result.add_solution(primal, Some(7.0));

    result.dual.insert(capacity.row_id(), 1.0);
    result.dual.insert(generated_constraint, 9.0);
    result.semantic_dual.insert(capacity.id(), Some(5.0));
    result.semantic_dual.insert(SemanticConstraintId::Algebraic(generated_constraint), Some(7.0));
    result.reduced_costs.insert(x.var_id().unwrap(), 0.5);
    result.reduced_costs.insert(generated_variable, 9.0);

    let projected = SourceProjection::with_coordinates(&source, coordinates)
        .from_transformed(&transformed)
        .reevaluate_objective(|point| point.value_of(weight * (x + y)).ok().flatten())
        .project(result)
        .unwrap();

    assert_eq!(projected.model_id, source.id());
    assert_eq!(projected.solutions.len(), 1);
    assert_eq!(projected.solutions[0].model_id, source.id());
    assert_eq!(projected.solutions[0].primal.len(), 2);
    assert!(projected.solutions[0].primal.get(generated_variable).is_none());
    assert_eq!(projected.solutions[0].primal.get(x.var_id().unwrap()).unwrap(), 0.5);

    assert_eq!(projected.dual.get(capacity.row_id()), Some(1.0));
    assert_eq!(projected.dual_of(capacity).unwrap(), Some(5.0));
    assert!(
        !projected
            .semantic_dual
            .contains_key(&SemanticConstraintId::Algebraic(generated_constraint))
    );
    assert!(projected.dual.get(generated_constraint).is_none());
    assert!(projected.reduced_costs.get(generated_variable).is_none());

    assert_eq!(projected.solutions[0].objective, Some(1.0));
    assert!(projected.parameters.same_bits(&solve_parameters));
    assert!(projected.solutions[0].parameters.same_bits(&solve_parameters));

    assert!(
        projected
            .semantic_dual
            .get(&SemanticConstraintId::SpecialOrderedSet(sos_id))
            .copied()
            .flatten()
            .is_none()
    );
}

#[test]
fn projection_restores_the_objective_convention() {
    let source = Model::new("objective restore");
    variable!(source, x);

    let mut result = SolverResult::for_model(&source);
    result.add_solution([(x.var_id().unwrap(), 1.0)].into_iter().collect(), Some(-10.0));
    result.best_bound = Some(-8.0);

    let projected = SourceProjection::capture(&source)
        .with_objective(ObjectiveTransform { sign: -1.0, offset: 5.0 })
        .project(result)
        .unwrap();

    assert_eq!(projected.solutions[0].objective, Some(15.0));
    assert_eq!(projected.best_bound, Some(13.0));
}

#[test]
fn projection_rejects_coordinates_larger_than_the_source() {
    let source = Model::new("source bounds");
    variable!(source, _x);

    let coordinates = SourceCoordinates { variables: 2, ..SourceCoordinates::capture(&source) };
    let error = SourceProjection::with_coordinates(&source, coordinates)
        .project(SolverResult::for_model(&source))
        .unwrap_err();

    assert!(matches!(
        error,
        ProjectionError::CoordinatesExceedSource { kind: "variable", count: 2, available: 1 }
    ));
}

#[test]
fn projection_rejects_a_result_from_another_model() {
    let source = Model::new("projection source");
    let transformed = Model::new("unrelated transformed model");

    let error = SourceProjection::capture(&source)
        .from_transformed(&transformed)
        .project(SolverResult::for_model(&source))
        .unwrap_err();

    assert!(matches!(error, ProjectionError::ForeignResult { .. }));
}

#[test]
fn projection_rejects_a_foreign_result_without_a_transformed_model() {
    let source = Model::new("in-place source");
    let foreign = Model::new("foreign model");

    let error =
        SourceProjection::capture(&source).project(SolverResult::for_model(&foreign)).unwrap_err();

    assert!(matches!(
        error,
        ProjectionError::ForeignResult { expected, actual }
            if expected == source.id() && actual == foreign.id()
    ));
}

#[test]
fn in_place_reformulation_projects_without_cloning() {
    let source = Model::new("in place projection");
    variable!(source, 0.0 <= x <= 1.0);
    variable!(source, 0.0 <= y <= 1.0);
    sos_constraint!(source, choice, SOS1, [x, y]);
    objective!(source, Max, x + y);

    let coordinates = SourceCoordinates::capture(&source);

    source.reformulate_sos(SosReformulationOptions::default()).unwrap();

    let history = source.sos_reformulations();
    let artifacts = &history[0];
    let generated_variable = artifacts.variables[0].raw_id();
    let generated_constraint = artifacts.constraints[0].raw_id();

    let mut result = SolverResult::for_model(&source);
    result.dual_status = DualStatus::FeasiblePoint;

    let mut primal = oximo_solver::SolutionValues::default();
    primal.insert(x.var_id().unwrap(), 1.0);
    primal.insert(y.var_id().unwrap(), 0.0);
    primal.insert(generated_variable, 0.0);
    result.add_solution(primal, Some(7.0));
    result.best_bound = Some(6.0);
    result.dual.insert(generated_constraint, 9.0);
    result.reduced_costs.insert(generated_variable, 9.0);

    let projected = SourceProjection::with_coordinates(&source, coordinates)
        .reevaluate_objective(|point| point.value_of(x + y).ok().flatten())
        .project(result)
        .unwrap();

    assert_eq!(projected.model_id, source.id());
    assert_eq!(projected.solutions.len(), 1);
    assert_eq!(projected.solutions[0].primal.len(), 2);
    assert!(projected.solutions[0].primal.get(generated_variable).is_none());
    assert_eq!(projected.solutions[0].objective, Some(1.0));

    // `reevaluate_objective` changes points only.
    assert_eq!(projected.best_bound, Some(6.0));
    assert!(projected.dual.get(generated_constraint).is_none());
    assert!(projected.reduced_costs.get(generated_variable).is_none());
}

#[test]
fn projection_preserves_every_valid_solution_point() {
    let source = Model::new("multi point projection");
    variable!(source, x);
    variable!(source, y);
    objective!(source, Min, x + y);

    let mut result = SolverResult::for_model(&source);
    result.primal_status = PrimalStatus::FeasiblePoint;
    result.add_solution(
        [(x.var_id().unwrap(), 1.0), (y.var_id().unwrap(), 2.0)].into_iter().collect(),
        Some(99.0),
    );
    result.add_solution(
        [(x.var_id().unwrap(), 3.0), (y.var_id().unwrap(), 4.0)].into_iter().collect(),
        Some(99.0),
    );

    // Missing `y`
    result.add_solution([(x.var_id().unwrap(), 5.0)].into_iter().collect(), Some(99.0));

    let projected = SourceProjection::capture(&source)
        .reevaluate_objective(|point| point.value_of(x + y).ok().flatten())
        .project(result)
        .unwrap();

    assert_eq!(projected.solutions.len(), 2);
    assert_eq!(projected.solutions[0].objective, Some(3.0));
    assert_eq!(projected.solutions[1].objective, Some(7.0));
    assert_eq!(projected.primal_status, PrimalStatus::FeasiblePoint);
}

#[test]
fn projection_retains_explicitly_requested_generated_variables() {
    let source = Model::new("retained projection");
    variable!(source, x);
    variable!(source, y);

    let x_id = x.var_id().unwrap();
    let y_id = y.var_id().unwrap();
    let retained = VarId(2);
    let dropped = VarId(3);

    let mut result = SolverResult::for_model(&source);
    result.add_solution(
        [(x_id, 1.0), (y_id, 2.0), (retained, 0.5), (dropped, 0.75)].into_iter().collect(),
        Some(0.0),
    );

    let projected = SourceProjection::capture(&source)
        .retain_variables([oximo_core::GeneratedVarId::from_raw(retained)])
        .project(result)
        .unwrap();

    let primal = projected.primal().unwrap();
    assert_eq!(primal.len(), 3);
    assert_eq!(primal.get(retained), Some(0.5));
    assert_eq!(primal.get(x_id), Some(1.0));
    assert!(primal.get(dropped).is_none());
}

#[test]
fn projection_drops_non_finite_retained_values() {
    let source = Model::new("non-finite retained projection");
    variable!(source, x);

    let x_id = x.var_id().unwrap();
    let retained = VarId(1);

    let mut result = SolverResult::for_model(&source);
    result.add_solution([(x_id, 1.0), (retained, f64::NAN)].into_iter().collect(), Some(0.0));

    let projected = SourceProjection::capture(&source)
        .retain_variables([oximo_core::GeneratedVarId::from_raw(retained)])
        .project(result)
        .unwrap();

    let primal = projected.primal().unwrap();
    assert_eq!(primal.len(), 1);
    assert!(primal.get(retained).is_none());
}

#[test]
fn projection_still_requires_every_source_variable_when_retaining() {
    let source = Model::new("incomplete retained projection");
    variable!(source, x);
    variable!(source, _y);

    let mut result = SolverResult::for_model(&source);
    result.add_solution(
        [(x.var_id().unwrap(), 1.0), (VarId(2), 0.5)].into_iter().collect(),
        Some(0.0),
    );

    let projected = SourceProjection::capture(&source)
        .retain_variables([oximo_core::GeneratedVarId::from_raw(VarId(2))])
        .project(result)
        .unwrap();

    assert!(projected.solutions.is_empty());
    assert_eq!(projected.primal_status, PrimalStatus::NoSolution);
}

#[test]
fn explicit_source_variables_can_interleave_with_generated_ids() {
    let source = Model::new("interleaved source projection");
    variable!(source, x);
    variable!(source, y);
    variable!(source, z);

    let x_id = x.var_id().unwrap();
    let y_id = y.var_id().unwrap();
    let z_id = z.var_id().unwrap();
    let coordinates = SourceCoordinates { variables: 2, ..SourceCoordinates::capture(&source) };

    let mut result = SolverResult::for_model(&source);
    result.dual_status = DualStatus::FeasiblePoint;
    result.add_solution([(x_id, 1.0), (y_id, 0.5), (z_id, 2.0)].into_iter().collect(), Some(0.0));
    result.reduced_costs.insert(x_id, 1.0);
    result.reduced_costs.insert(y_id, 9.0);
    result.reduced_costs.insert(z_id, 2.0);

    let projected = SourceProjection::with_coordinates(&source, coordinates)
        .with_source_variables([x_id, z_id])
        .project(result)
        .unwrap();

    let primal = projected.primal().unwrap();
    assert_eq!(primal.len(), 2);
    assert_eq!(primal.get(x_id), Some(1.0));
    assert_eq!(primal.get(z_id), Some(2.0));
    assert!(primal.get(y_id).is_none());

    assert_eq!(projected.reduced_costs.get(x_id), Some(1.0));
    assert_eq!(projected.reduced_costs.get(z_id), Some(2.0));
    assert!(projected.reduced_costs.get(y_id).is_none());
}

#[test]
fn explicit_source_variables_still_require_complete_source_points() {
    let source = Model::new("incomplete explicit projection");
    variable!(source, x);
    variable!(source, _y);
    variable!(source, z);

    let x_id = x.var_id().unwrap();
    let z_id = z.var_id().unwrap();
    let coordinates = SourceCoordinates { variables: 2, ..SourceCoordinates::capture(&source) };

    let mut result = SolverResult::for_model(&source);
    result.add_solution([(x_id, 1.0), (VarId(1), 0.5)].into_iter().collect(), Some(0.0));

    let projected = SourceProjection::with_coordinates(&source, coordinates)
        .with_source_variables([x_id, z_id])
        .project(result)
        .unwrap();

    assert!(projected.solutions.is_empty());
    assert_eq!(projected.primal_status, PrimalStatus::NoSolution);
}

#[test]
fn explicit_source_variables_must_belong_to_the_source_model() {
    let source = Model::new("explicit source bounds");
    variable!(source, _x);

    let error = SourceProjection::capture(&source)
        .with_source_variables([VarId(0), VarId(1)])
        .project(SolverResult::for_model(&source))
        .unwrap_err();

    assert!(matches!(
        error,
        ProjectionError::CoordinatesExceedSource { kind: "variable", count: 2, available: 1 }
    ));
}

#[test]
fn projection_filters_generated_cone_duals() {
    let source = Model::new("cone dual projection");
    variable!(source, x);
    variable!(source, t >= 0.0);
    let cone = soc_constraint!(source, cone, [x] <= t);
    variable!(source, z);
    let positivity =
        psd_constraint!(source, positivity, SymmetricMatrix::from_upper_triangle(1, [z]));

    let coordinates = SourceCoordinates::capture(&source);
    assert_eq!(coordinates.soc_constraints, 1);
    assert_eq!(coordinates.psd_constraints, 1);

    let generated_cone = soc_constraint!(source, [x] <= t);
    let generated_positivity =
        psd_constraint!(source, SymmetricMatrix::from_upper_triangle(1, [z]));
    let projection = SourceProjection::with_coordinates(&source, coordinates);

    for generated_multiplier in [Some(7.0), None] {
        let mut result = SolverResult::for_model(&source);
        result.dual_status = DualStatus::FeasiblePoint;
        result.add_solution(
            [(x.var_id().unwrap(), 1.0), (t.var_id().unwrap(), 2.0), (z.var_id().unwrap(), 3.0)]
                .into_iter()
                .collect(),
            Some(0.0),
        );
        result.soc_dual.insert(cone.soc_id(), 2.0);
        result.soc_dual.insert(generated_cone.soc_id(), 9.0);
        result.psd_dual.insert(positivity.psd_id(), SymmetricMatrix::from_upper_triangle(1, [4.0]));
        result
            .psd_dual
            .insert(generated_positivity.psd_id(), SymmetricMatrix::from_upper_triangle(1, [9.0]));
        result.semantic_dual.insert(cone.id(), Some(3.0));
        result.semantic_dual.insert(positivity.id(), None);
        result.semantic_dual.insert(generated_cone.id(), generated_multiplier);
        result.semantic_dual.insert(generated_positivity.id(), generated_multiplier);

        let projected = projection.project(result).unwrap();

        assert_eq!(projected.soc_dual.len(), 1);
        assert_eq!(projected.soc_dual.get(cone.soc_id()), Some(2.0));
        assert_eq!(projected.psd_dual.len(), 1);
        assert!(projected.psd_dual.contains_key(&positivity.psd_id()));
        assert_eq!(projected.semantic_dual.get(&cone.id()), Some(&Some(3.0)));
        assert_eq!(projected.semantic_dual.get(&positivity.id()), Some(&None));
        assert!(!projected.semantic_dual.contains_key(&generated_cone.id()));
        assert!(!projected.semantic_dual.contains_key(&generated_positivity.id()));
    }
}

#[test]
fn projection_without_source_rows_preserves_non_algebraic_semantic_mappings() {
    let model = Model::new("cone semantics without scalar source rows");

    variable!(model, x);
    variable!(model, t);
    let cone = soc_constraint!(model, [x] <= t);
    let coordinates = SourceCoordinates::capture(&model);
    let generated = constraint!(model, x <= 2.0);
    let projection = SourceProjection::with_coordinates(&model, coordinates);

    for source_multiplier in [Some(3.0), Some(f64::NAN), None] {
        let mut result = SolverResult::for_model(&model);
        result.dual_status = DualStatus::FeasiblePoint;
        result.add_solution(
            [(x.var_id().unwrap(), 1.0), (t.var_id().unwrap(), 2.0)].into_iter().collect(),
            Some(0.0),
        );

        result.dual.insert(generated.row_id(), 7.0);
        result.semantic_dual.insert(generated.id(), Some(7.0));
        result.semantic_dual.insert(cone.id(), source_multiplier);
        result.soc_dual.insert(cone.soc_id(), 2.0);

        let projected = projection.project(result).unwrap();
        assert!(projected.dual.is_empty());
        assert!(!projected.semantic_dual.contains_key(&generated.id()));
        assert_eq!(projected.dual_of(generated).unwrap(), None);
        assert_eq!(
            projected.semantic_dual.get(&cone.id()),
            Some(&source_multiplier.filter(|value| value.is_finite()))
        );
        assert_eq!(projected.soc_dual_of(cone).unwrap(), Some(2.0));
    }
}
