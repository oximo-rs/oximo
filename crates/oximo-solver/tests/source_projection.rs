use oximo_core::SemanticConstraintId;
use oximo_core::VarId;
use oximo_core::prelude::*;
use oximo_solver::reconstruct::{
    ObjectiveTransform, ProjectionError, SourceCoordinates, SourceProjection, SourceProjectionPlan,
};
use oximo_solver::{DualStatus, PrimalStatus, SolverResult};

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
    let plan = SourceProjectionPlan::new(SourceCoordinates::capture(&source))
        .with_source_variables([x_id, z_id, x_id])
        .with_source_rows([first, last, first])
        .retain_variables([generated_id]);
    let projection = SourceProjection::with_plan(&source, &plan);
    let mut result = SolverResult::for_model(&source);
    result.dual_status = DualStatus::FeasiblePoint;
    result
        .add_solution([(x_id, 1.0), (z_id, 2.0), (generated_id, 0.5)].into_iter().collect(), None);
    result.dual = [(first, 1.0), (generated_row, 9.0), (last, 2.0)].into_iter().collect();
    result.reduced_costs = [(x_id, 1.0), (generated_id, 9.0), (z_id, 2.0)].into_iter().collect();
    for _ in 0..3 {
        let projected = projection.project(result.clone()).unwrap();
        assert_eq!(projected.primal().unwrap().len(), 3);
        assert_eq!(projected.dual.len(), 2);
        assert_eq!(projected.dual.get(&last), Some(&2.0));
        assert!(!projected.dual.contains_key(&generated_row));
        assert_eq!(projected.reduced_costs.len(), 2);
        assert!(!projected.reduced_costs.contains_key(&generated_id));
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
    let generated_variable = artifacts.variables[0];
    let generated_constraint = artifacts.constraints[0];
    let sos_id = artifacts.source;

    let solve_parameters = transformed.arena().parameter_snapshot();
    let mut result = SolverResult::for_model(&transformed);
    result.dual_status = DualStatus::FeasiblePoint;

    let mut primal = rustc_hash::FxHashMap::default();
    primal.insert(x.var_id().unwrap(), 0.5);
    primal.insert(y.var_id().unwrap(), 0.0);
    primal.insert(generated_variable, 0.0);
    result.add_solution(primal, Some(7.0));

    result.dual.insert(capacity.row_id(), 1.0);
    result.dual.insert(generated_constraint, 9.0);
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
    assert!(!projected.solutions[0].primal.contains_key(&generated_variable));
    assert_eq!(projected.solutions[0].primal[&x.var_id().unwrap()], 0.5);

    assert_eq!(projected.dual.get(&capacity.row_id()), Some(&1.0));
    assert!(!projected.dual.contains_key(&generated_constraint));
    assert!(!projected.reduced_costs.contains_key(&generated_variable));

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
    let generated_variable = artifacts.variables[0];
    let generated_constraint = artifacts.constraints[0];

    let mut result = SolverResult::for_model(&source);
    result.dual_status = DualStatus::FeasiblePoint;

    let mut primal = rustc_hash::FxHashMap::default();
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
    assert!(!projected.solutions[0].primal.contains_key(&generated_variable));
    assert_eq!(projected.solutions[0].objective, Some(1.0));

    // `reevaluate_objective` changes points only.
    assert_eq!(projected.best_bound, Some(6.0));
    assert!(!projected.dual.contains_key(&generated_constraint));
    assert!(!projected.reduced_costs.contains_key(&generated_variable));
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

    let projected =
        SourceProjection::capture(&source).retain_variables([retained]).project(result).unwrap();

    let primal = projected.primal().unwrap();
    assert_eq!(primal.len(), 3);
    assert_eq!(primal.get(&retained), Some(&0.5));
    assert_eq!(primal.get(&x_id), Some(&1.0));
    assert!(!primal.contains_key(&dropped));
}

#[test]
fn projection_drops_non_finite_retained_values() {
    let source = Model::new("non-finite retained projection");
    variable!(source, x);

    let x_id = x.var_id().unwrap();
    let retained = VarId(1);

    let mut result = SolverResult::for_model(&source);
    result.add_solution([(x_id, 1.0), (retained, f64::NAN)].into_iter().collect(), Some(0.0));

    let projected =
        SourceProjection::capture(&source).retain_variables([retained]).project(result).unwrap();

    let primal = projected.primal().unwrap();
    assert_eq!(primal.len(), 1);
    assert!(!primal.contains_key(&retained));
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

    let projected =
        SourceProjection::capture(&source).retain_variables([VarId(2)]).project(result).unwrap();

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
    assert_eq!(primal.get(&x_id), Some(&1.0));
    assert_eq!(primal.get(&z_id), Some(&2.0));
    assert!(!primal.contains_key(&y_id));

    assert_eq!(projected.reduced_costs.get(&x_id), Some(&1.0));
    assert_eq!(projected.reduced_costs.get(&z_id), Some(&2.0));
    assert!(!projected.reduced_costs.contains_key(&y_id));
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

    let mut result = SolverResult::for_model(&source);
    result.dual_status = DualStatus::FeasiblePoint;
    result.add_solution(
        [(x.var_id().unwrap(), 1.0), (t.var_id().unwrap(), 2.0), (z.var_id().unwrap(), 3.0)]
            .into_iter()
            .collect(),
        Some(0.0),
    );
    result.soc_dual.insert(cone.soc_id(), 2.0);
    result.soc_dual.insert(SocConstraintId(1), 9.0);
    result.psd_dual.insert(positivity.psd_id(), SymmetricMatrix::from_upper_triangle(1, [4.0]));
    result.psd_dual.insert(PsdConstraintId(1), SymmetricMatrix::from_upper_triangle(1, [9.0]));

    let projected = SourceProjection::capture(&source).project(result).unwrap();

    assert_eq!(projected.soc_dual.len(), 1);
    assert_eq!(projected.soc_dual.get(&cone.soc_id()), Some(&2.0));
    assert_eq!(projected.psd_dual.len(), 1);
    assert!(projected.psd_dual.contains_key(&positivity.psd_id()));
}
