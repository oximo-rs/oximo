use oximo_core::prelude::*;
use oximo_solver::{SolutionPoint, SolverResult, reconstruct::finalize_result};

#[test]
fn captured_result_queries_check_both_owners_and_use_the_best_saved_point() {
    let model = Model::new("captured result query");

    param!(model, p = 2.0);
    variable!(model, x);
    variable!(model, missing);
    let captured = (p * x).capture();
    let mut result = SolverResult::for_model(&model);
    assert_eq!(result.value_of_captured(&captured), Ok(None));

    let foreign = Model::new("foreign captured query");

    variable!(foreign, other);
    let foreign_captured = other.capture();
    let foreign_error = ModelMismatchError::new(model.id(), foreign.id());
    assert_eq!(result.value_of_captured(&foreign_captured), Err(foreign_error));

    model.set_param(p, 7.0).unwrap();
    for value in [3.0, 5.0] {
        result.add_solution([(x.var_id().unwrap(), value)].into_iter().collect(), None);
    }
    assert_eq!(result.value_of_captured(&captured), Ok(Some(6.0)));
    assert_eq!(result.solution(1).unwrap().value_of_captured(&captured), Ok(Some(10.0)));
    assert_eq!(result.value_of_captured(&missing.capture()), Ok(None));

    param!(model, later = 4.0);
    assert_eq!(result.value_of_captured(&(later * x).capture()), Ok(None));
    result.solutions[0] =
        SolutionPoint::new(&foreign, [(other.var_id().unwrap(), 9.0)].into_iter().collect(), None);
    assert_eq!(
        result.value_of_captured(&captured),
        Err(ModelMismatchError::new(foreign.id(), model.id())),
    );
}

#[test]
fn added_solutions_use_the_results_snapshot_even_after_the_model_changes() {
    let model = Model::new("result assembly");

    param!(model, p = 2.0);
    variable!(model, x);
    let mut result = SolverResult::for_model(&model);

    model.set_param(p, 4.0).unwrap();
    for value in [3.0, 4.0] {
        result
            .add_solution([(x.var_id().unwrap(), value)].into_iter().collect(), Some(2.0 * value));
    }

    for (point, expected) in result.solutions.iter().zip([6.0, 8.0]) {
        assert_eq!(point.model_id(), result.model_id());
        assert_eq!(point.parameters.values().as_ptr(), result.parameters.values().as_ptr());
        assert_eq!(point.value_of(p * x), Ok(Some(expected)));
    }
    assert_eq!(result.value_of(p * x), Ok(Some(6.0)));
    assert_eq!(result.value_of_with_current_parameters(p * x), Ok(Some(12.0)));

    let normalized = finalize_result(result, &model, 1);
    assert_eq!(normalized.value_of(p * x), Ok(Some(6.0)));
    assert_eq!(normalized.solution(1).unwrap().value_of(p * x), Ok(Some(8.0)));
}

#[test]
fn scalar_queries_preserve_saved_parameters_and_allow_explicit_scenarios() {
    let model = Model::new("parameters");

    param!(model, p = 2.0);
    variable!(model, x);
    variable!(model, missing);
    let point =
        SolutionPoint::new(&model, [(x.var_id().unwrap(), 3.0)].into_iter().collect(), Some(6.0));
    let mut result = SolverResult::for_model(&model);
    result.solutions.push(point.clone());
    let expr = p * x;
    model.set_param(p, 4.0).unwrap();

    assert_eq!(point.value_of(expr), Ok(Some(6.0)));
    assert_eq!(point.value_of_with_current_parameters(expr), Ok(Some(12.0)));
    assert_eq!(result.value_of(expr), Ok(Some(6.0)));
    assert_eq!(result.value_of_with_current_parameters(expr), Ok(Some(12.0)));
    assert_eq!(result.objective(), Some(6.0));
    assert_eq!(point.value_of(p), Ok(Some(2.0)));
    assert_eq!(point.value_of_with_current_parameters(p), Ok(Some(4.0)));

    // Queries built after solving may still use parameters captured at solve time.
    assert_eq!(point.value_of(p * x.powi(2)), Ok(Some(18.0)));
    assert_eq!(point.value_of((p * x).sin()), Ok(Some(6.0_f64.sin())));
    assert_eq!(point.value_of_with_current_parameters((p * x).sin()), Ok(Some(12.0_f64.sin())));
    assert_eq!(point.value_of(p * missing), Ok(None));
    assert_eq!(point.value_of_with_current_parameters(p * missing), Ok(None));
    param!(model, added = 5.0);
    assert_eq!(point.value_of(added), Ok(None));
    assert_eq!(point.value_of(p * x + added), Ok(None));
    assert_eq!(point.value_of_with_current_parameters(p * x + added), Ok(Some(17.0)));

    let matrix = SymmetricMatrix::from_upper_triangle(1, [added * x]);
    assert_eq!(point.value_of_matrix(&matrix), Ok(None));
    assert_eq!(result.value_of_matrix(&matrix), Ok(None));
    assert_eq!(
        point.value_of_matrix_with_current_parameters(&matrix).unwrap().unwrap()[(0, 0)],
        15.0
    );
    assert_eq!(point.parameters.values(), &[2.0]);
    assert_eq!(point.clone().value_of(expr), Ok(Some(6.0)));
}

#[test]
fn constraint_and_soc_queries_use_saved_or_current_parameters_at_each_point() {
    let model = Model::new("feasibility");

    param!(model, p = 2.0);
    variable!(model, x);
    let row = constraint!(model, row, p * x <= 8.0);
    let cone = soc_constraint!(model, cone, [p * x] <= p + 5.0);
    let mut result = SolverResult::for_model(&model);
    result.solutions = [3.0, 4.0]
        .into_iter()
        .map(|v| SolutionPoint::new(&model, [(x.var_id().unwrap(), v)].into_iter().collect(), None))
        .collect();
    model.set_param(p, 4.0).unwrap();
    let saved = result.constraint_evaluation(&model, row).unwrap().unwrap();
    let current =
        result.constraint_evaluation_with_current_parameters(&model, row).unwrap().unwrap();

    assert_eq!((saved.activity, saved.upper_slack, saved.violation), (6.0, Some(2.0), 0.0));
    assert_eq!((current.activity, current.upper_slack, current.violation), (12.0, Some(-4.0), 4.0));
    assert_eq!(result.constraint_evaluation_at(&model, row, 1).unwrap().unwrap().activity, 8.0);
    assert_eq!(
        result
            .constraint_evaluation_at_with_current_parameters(&model, row, 1)
            .unwrap()
            .unwrap()
            .activity,
        16.0
    );

    let saved = result.soc_evaluation(&model, cone).unwrap().unwrap();
    let current = result.soc_evaluation_with_current_parameters(&model, cone).unwrap().unwrap();
    assert_eq!((saved.norm, saved.bound, saved.slack, saved.violation), (6.0, 7.0, 1.0, 0.0));
    assert_eq!(
        (current.norm, current.bound, current.slack, current.violation),
        (12.0, 9.0, -3.0, 3.0)
    );
    assert_eq!(result.soc_evaluation_at(&model, cone, 1).unwrap().unwrap().norm, 8.0);
    assert_eq!(
        result.soc_evaluation_at_with_current_parameters(&model, cone, 1).unwrap().unwrap().norm,
        16.0
    );
    assert_eq!(result.constraint_evaluation_at_with_current_parameters(&model, row, 2), Ok(None));
    assert_eq!(result.soc_evaluation_at_with_current_parameters(&model, cone, 2), Ok(None));
    param!(model, added = 10.0);

    let new_row = constraint!(model, new_row, added * x <= 50.0);
    let new_cone = soc_constraint!(model, new_cone, [x] <= added);
    assert_eq!(result.constraint_evaluation(&model, new_row), Ok(None));
    assert_eq!(result.soc_evaluation(&model, new_cone), Ok(None));
    assert_eq!(
        result
            .constraint_evaluation_with_current_parameters(&model, new_row)
            .unwrap()
            .unwrap()
            .activity,
        30.0
    );
    assert_eq!(
        result.soc_evaluation_with_current_parameters(&model, new_cone).unwrap().unwrap().bound,
        10.0
    );
}

#[test]
fn both_modes_validate_ownership_including_empty_results_and_inconsistent_points() {
    let model = Model::new("source");

    variable!(model, x);
    let row = constraint!(model, row, x <= 1.0);
    let cone = soc_constraint!(model, cone, [x] <= x + 1.0);
    let other = model.to_reformulated_sos_model(SosReformulationOptions::default()).unwrap();
    let foreign = other.variable_handle(x.var_id().unwrap());
    let foreign_row = other.constraint_handle_from_id(row.row_id()).unwrap();
    let foreign_cone = other.soc_constraint_handle_from_id(cone.soc_id()).unwrap();
    let mut result = SolverResult::for_model(&model);

    assert!(result.value_of(foreign).is_err());
    assert!(result.value_of_with_current_parameters(foreign).is_err());
    assert_eq!(result.value_of_with_current_parameters(x), Ok(None));
    assert!(result.constraint_evaluation(&model, foreign_row).is_err());
    assert!(result.constraint_evaluation_with_current_parameters(&model, foreign_row).is_err());
    assert!(result.soc_evaluation_with_current_parameters(&model, foreign_cone).is_err());
    assert!(result.soc_evaluation_with_current_parameters(&other, cone).is_err());

    result.solutions.push(SolutionPoint::new(
        &other,
        oximo_solver::SolutionValues::default(),
        None,
    ));
    assert!(result.value_of_with_current_parameters(x).is_err());
    assert!(result.constraint_evaluation_with_current_parameters(&model, row).is_err());
    assert!(result.soc_evaluation_with_current_parameters(&model, cone).is_err());
}

#[test]
fn normalization_shares_original_parameters_with_all_points_and_is_repeatable() {
    let model = Model::new("pool");

    param!(model, p = 2.0);
    variable!(model, x);
    let mut result = SolverResult::for_model(&model);
    result.solutions = [3.0, 4.0]
        .into_iter()
        .map(|v| SolutionPoint {
            primal: [(x.var_id().unwrap(), v)].into_iter().collect(),
            ..Default::default()
        })
        .collect();
    model.set_param(p, 4.0).unwrap();

    let result = finalize_result(finalize_result(result, &model, 1), &model, 1);
    assert_eq!(result.parameters.values(), &[2.0]);
    assert_eq!(result.solution(0).unwrap().value_of(p * x), Ok(Some(6.0)));
    assert_eq!(result.solution(1).unwrap().value_of(p * x), Ok(Some(8.0)));
    assert_eq!(
        result.solution(1).unwrap().clone().value_of_with_current_parameters(p * x),
        Ok(Some(16.0))
    );
}

#[test]
fn normalization_rejects_foreign_points_even_when_parameter_snapshots_match() {
    for with_parameters in [false, true] {
        let model = Model::new("result owner");
        let other = Model::new("foreign point owner");
        if with_parameters {
            param!(model, _p = 2.0);
            param!(other, _q = 2.0);
        }
        variable!(model, x);
        variable!(other, y);
        let mut result = SolverResult::for_model(&model);
        let foreign = SolutionPoint::new(
            &other,
            [(y.var_id().unwrap(), 123.0)].into_iter().collect(),
            Some(246.0),
        );
        assert_eq!(x.var_id(), y.var_id());
        assert_eq!(result.parameters, foreign.parameters);
        result.solutions.push(foreign);
        assert!(result.value_of(x).is_err());
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            finalize_result(result, &model, 1)
        }))
        .unwrap_err();
        assert!(
            panic
                .downcast_ref::<String>()
                .unwrap()
                .contains("model identity mismatch for solution 0")
        );
    }
}

#[test]
fn normalization_retags_unassigned_points_with_matching_parameter_snapshots() {
    let model = Model::new("detached adapter point");

    param!(model, p = 2.0);
    variable!(model, x);
    let mut result = SolverResult::for_model(&model);
    let mut point =
        SolutionPoint::new(&model, [(x.var_id().unwrap(), 3.0)].into_iter().collect(), Some(6.0));
    point.model_id = ModelId::UNASSIGNED;
    result.solutions.push(point);
    let result = finalize_result(result, &model, 1);

    assert_eq!(result.best().unwrap().model_id(), model.id());
    assert_eq!(result.value_of(p * x), Ok(Some(6.0)));
}

#[test]
#[should_panic(expected = "parameter snapshot mismatch for solution 0")]
fn normalization_rejects_an_empty_result_snapshot_instead_of_erasing_point_parameters() {
    let model = Model::new("missing result snapshot");

    param!(model, p = 2.0);
    variable!(model, x);
    let point =
        SolutionPoint::new(&model, [(x.var_id().unwrap(), 3.0)].into_iter().collect(), Some(6.0));
    let result =
        SolverResult { model_id: model.id(), solutions: vec![point], ..Default::default() };

    assert_eq!(result.value_of(p * x), Ok(Some(6.0)));
    finalize_result(result, &model, 1);
}

#[test]
fn normalization_rejects_conflicting_parameter_bits_even_for_detached_points() {
    let finite = 1.25_f64;
    for (before, after) in [(2.0, 4.0), (0.0, -0.0), (finite, f64::from_bits(finite.to_bits() + 1))]
    {
        let model = Model::new("conflicting snapshots");
        param!(model, p = before);
        variable!(model, x);
        let mut result = SolverResult::for_model(&model);
        model.set_param(p, after).unwrap();
        let point =
            SolutionPoint::new(&model, [(x.var_id().unwrap(), 3.0)].into_iter().collect(), None);
        for model_id in [model.id(), ModelId::UNASSIGNED] {
            let mut point = point.clone();
            point.model_id = model_id;
            result.solutions = vec![point];
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| finalize_result(
                    result.clone(),
                    &model,
                    1
                )))
                .is_err()
            );
        }
        // An assigned point with empty parameters is not a detached placeholder.
        result.solutions[0] = SolutionPoint { model_id: model.id(), ..Default::default() };
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| finalize_result(
                result, &model, 1
            )))
            .is_err()
        );
    }
}

#[test]
fn normalization_preserves_independent_snapshots_with_identical_finite_and_signed_zero_bits() {
    let model = Model::new("equivalent snapshots");

    let finite = 1.25_f64;
    param!(model, p = finite);
    param!(model, zero = -0.0);
    variable!(model, x);
    let mut result = SolverResult::for_model(&model);
    model.set_param(p, 1.0).unwrap();
    model.set_param(p, finite).unwrap();
    result.solutions.push(SolutionPoint::new(
        &model,
        [(x.var_id().unwrap(), 3.0)].into_iter().collect(),
        None,
    ));

    let point_storage = result.best().unwrap().parameters.values().as_ptr();
    assert_ne!(
        result.parameters.values().as_ptr(),
        result.best().unwrap().parameters.values().as_ptr()
    );

    let result = finalize_result(finalize_result(result, &model, 1), &model, 1);
    assert_eq!(point_storage, result.best().unwrap().parameters.values().as_ptr());
    assert_eq!(result.value_of(p).unwrap().unwrap().to_bits(), finite.to_bits());
    assert_eq!(result.value_of(zero).unwrap().unwrap().to_bits(), (-0.0_f64).to_bits());
}

#[test]
fn incremental_coefficients_and_parameters_are_captured_together() {
    let model = Model::new("incremental snapshot");

    param!(model, p = 2.0);
    param!(model, unused = 7.0);
    variable!(model, x >= 0.0);
    objective!(model, Min, p * x);

    let first = oximo_solver::snapshot(&model).unwrap();
    model.set_param(p, 4.0).unwrap();
    model.set_param(unused, 9.0).unwrap();
    let second = oximo_solver::snapshot(&model).unwrap();

    assert_eq!(first.obj_costs, [2.0]);
    assert_eq!(first.parameters.values(), &[2.0, 7.0]);
    assert_eq!(second.obj_costs, [4.0]);
    assert_eq!(second.parameters.values(), &[4.0, 9.0]);
    assert_eq!(first.fingerprint, second.fingerprint);
}
