use oximo_core::{ConstraintId, VarId};
use oximo_solver::reconstruct::{
    DualProjection, ObjectiveTransform, SourceCoordinates, accumulate_dual, finalize_result,
    finalize_result_with_coordinates, project_dense_primal, project_primal, relative_gap,
    require_active_dual_evidence, try_accumulate_dual,
};
use oximo_solver::{DualStatus, PrimalStatus, SolutionPoint, SolverResult, TerminationStatus};

fn model_with_variables(count: usize) -> oximo_core::Model {
    let model = oximo_core::Model::new("result finalization");
    for index in 0..count {
        model.__var(format!("x{index}")).build();
    }
    model
}

#[test]
fn active_dual_evidence_reports_all_missing_groups_and_preserves_native_logs() {
    use oximo_core::prelude::*;

    let model = Model::new("active dual evidence");
    variable!(model, x);
    let inactive_row = constraint!(model, inactive_row, x <= 1.0);
    let row = constraint!(model, row, x >= 0.0);
    let inactive_cone = soc_constraint!(model, inactive_cone, [x] <= x);
    let cone = soc_constraint!(model, cone, [x] <= x);
    model.set_constraint_active(inactive_row, false).unwrap();
    model.set_constraint_active(inactive_cone, false).unwrap();

    for (row_value, soc_value, missing) in [
        (None, None, Some("algebraic and SOC")),
        (Some(0.0), None, Some("SOC")),
        (None, Some(0.0), Some("algebraic")),
        (Some(0.0), Some(0.0), None),
        (Some(-1.0), Some(1.0), None),
        (Some(f64::NAN), Some(f64::INFINITY), Some("algebraic and SOC")),
        (Some(f64::INFINITY), Some(0.0), Some("algebraic")),
        (Some(0.0), Some(-1.0), Some("SOC")),
    ] {
        let mut result = SolverResult::for_model(&model);
        result.dual_status = DualStatus::FeasiblePoint;
        result.raw_log = Some("native log".into());
        result.reduced_costs.insert(VarId(0), 0.0);
        result.dual.insert(inactive_row.row_id(), 0.0);
        result.soc_dual.insert(inactive_cone.soc_id(), 0.0);
        if let Some(value) = row_value {
            result.dual.insert(row.row_id(), value);
        }
        if let Some(value) = soc_value {
            result.soc_dual.insert(cone.soc_id(), value);
        }

        require_active_dual_evidence(&mut result, [row.row_id()], &model.soc_constraints());

        if let Some(group) = missing {
            assert_eq!(result.dual_status, DualStatus::Unknown);
            assert_eq!(
                result.raw_log.as_deref(),
                Some(format!("native log\nNo multiplier captured for active {group} constraints; dual status is Unknown.").as_str())
            );
        } else {
            assert_eq!(result.dual_status, DualStatus::FeasiblePoint);
            assert_eq!(result.raw_log.as_deref(), Some("native log"));
        }
    }

    for status in [DualStatus::NoSolution, DualStatus::Unknown] {
        let mut result = SolverResult::for_model(&model);
        result.dual_status = status;
        require_active_dual_evidence(&mut result, [row.row_id()], &model.soc_constraints());
        assert_eq!(result.dual_status, status);
        assert!(result.raw_log.is_none());
    }
}

#[test]
fn custom_solver_can_grow_public_result_fields_without_preallocating() {
    use oximo_core::prelude::*;

    let model = Model::new("custom result assembly");
    variable!(model, x);
    let row = constraint!(model, row, x >= 0.0);
    param!(model, cap = 1.0);
    let cone = soc_constraint!(model, cone, [x] <= cap);
    let mut result = SolverResult::for_model(&model);
    assert_eq!(result.dual.slot_count(), 0);
    assert_eq!(result.soc_dual.slot_count(), 0);
    assert_eq!(result.reduced_costs.slot_count(), 0);

    result.dual.insert(row.row_id(), 1.0);
    result.soc_dual.extend([(cone.soc_id(), 2.0)]);
    result.reduced_costs.insert(x.var_id().unwrap(), 0.0);
    assert_eq!(result.dual_of(row), Ok(Some(1.0)));
    assert_eq!(result.soc_dual_of(cone), Ok(Some(2.0)));
    assert_eq!(result.reduced_costs.get(x.var_id().unwrap()), Some(0.0));
}

#[test]
fn native_dual_accumulation_reports_bounds_without_mutating_values() {
    let mut values = oximo_solver::SolutionValues::<ConstraintId>::missing(2);
    try_accumulate_dual(&mut values, ConstraintId(0), 2.0, -1.0).unwrap();
    let before = values.clone();
    assert!(try_accumulate_dual(&mut values, ConstraintId(2), 9.0, 1.0).is_err());
    assert_eq!(values, before);

    let projection = DualProjection { source: ConstraintId(2), scale: -2.0 };
    assert!(projection.try_accumulate(&mut values, 3.0, -1.0).is_err());
    assert_eq!(values, before);
}

#[test]
fn finalization_bounds_all_scalar_fields_without_a_source_projection() {
    use oximo_core::SocConstraintId;
    use oximo_core::prelude::*;
    let model = Model::new("scalar coordinate bounds");

    variable!(model, x);
    variable!(model, later);
    assert_eq!(later.var_id(), Some(VarId(1)));
    let row = constraint!(model, row, x >= 0.0);
    param!(model, cap = 1.0);
    let cone = soc_constraint!(model, cone, [x] <= cap);

    let mut result = SolverResult::for_model(&model);
    result.add_solution([(VarId(0), 0.0)].into_iter().collect(), None);
    result.dual_status = DualStatus::FeasiblePoint;
    result.dual.extend([(row.row_id(), 0.0), (ConstraintId(1_000_000), 1.0)]);
    result.soc_dual.extend([(cone.soc_id(), 0.0), (SocConstraintId(1_000_000), 1.0)]);
    result.reduced_costs.extend([(VarId(0), 0.0), (VarId(1_000_000), 1.0)]);
    result
        .semantic_dual
        .insert(oximo_core::SemanticConstraintId::Algebraic(ConstraintId(1_000_000)), Some(1.0));

    for _ in 0..2 {
        result = finalize_result(result, &model, 1);
        assert!(result.has_solution());
        assert_eq!(result.dual.slot_count(), 1);
        assert_eq!(result.soc_dual.slot_count(), 1);
        assert_eq!(result.reduced_costs.slot_count(), 1);
        assert_eq!(result.dual.as_dense(), Some([0.0].as_slice()));
        assert_eq!(result.soc_dual.as_dense(), Some([0.0].as_slice()));
        assert_eq!(result.reduced_costs.as_dense(), Some([0.0].as_slice()));
        assert!(
            result
                .semantic_dual
                .keys()
                .all(|id| *id
                    != oximo_core::SemanticConstraintId::Algebraic(ConstraintId(1_000_000)))
        );
    }
}

#[test]
fn finalization_does_not_pad_short_or_unavailable_scalar_fields() {
    use oximo_core::prelude::*;
    let model = Model::new("short scalar buffers");

    variable!(model, x);
    constraint!(model, first, x >= 0.0);
    constraint!(model, trailing, x <= 1.0);

    let mut result = SolverResult::for_model(&model);
    result.add_solution([(VarId(0), 0.0)].into_iter().collect(), None);
    result.dual_status = DualStatus::FeasiblePoint;
    result.dual.insert(ConstraintId(0), 0.0);
    let result = finalize_result(result, &model, 1);
    assert_eq!(model.constraints().algebraic().len(), 2);
    assert_eq!(result.dual.slot_count(), 1);
    assert_eq!(result.dual.as_dense().unwrap().len(), 1);
    assert_eq!(result.reduced_costs.slot_count(), 0);
    assert_eq!(result.reduced_costs.as_dense(), Some([].as_slice()));
    assert_eq!(result.soc_dual.slot_count(), 0);
}

#[test]
fn captured_finalization_excludes_later_declarations_in_every_registry() {
    use oximo_core::prelude::*;
    let model = Model::new("callback declarations");

    variable!(model, x);
    let row = constraint!(model, row, x >= 0.0);
    let cone = soc_constraint!(model, cone, [x] <= x);
    let coordinates = SourceCoordinates::capture(&model);
    variable!(model, later);
    let later_row = constraint!(model, later_row, x <= 2.0);
    let later_cone = soc_constraint!(model, later_cone, [x] <= x);
    let later_psd = psd_constraint!(model, later_psd, SymmetricMatrix::from_upper_triangle(1, [x]));

    let mut result = SolverResult::for_model(&model);
    result.add_solution(
        [(x.var_id().unwrap(), 0.0), (later.var_id().unwrap(), 9.0)].into_iter().collect(),
        None,
    );
    result.dual_status = DualStatus::FeasiblePoint;
    result.dual.extend([(row.row_id(), 0.0), (later_row.row_id(), 5.0)]);
    result.soc_dual.extend([(cone.soc_id(), 0.0), (later_cone.soc_id(), 5.0)]);
    result.reduced_costs.extend([(x.var_id().unwrap(), 0.0), (later.var_id().unwrap(), 5.0)]);
    result.psd_dual.insert(later_psd.psd_id(), SymmetricMatrix::from_upper_triangle(1, [5.0]));
    let current = finalize_result(result.clone(), &model, coordinates.variables);
    assert_eq!(current.dual.slot_count(), 2);
    assert_eq!(current.soc_dual.slot_count(), 2);
    assert_eq!(current.psd_dual.len(), 1);

    let captured = finalize_result_with_coordinates(result, &model, coordinates);
    assert_eq!(captured.primal().unwrap().slot_count(), 1);
    assert_eq!(captured.dual.as_dense(), Some([0.0].as_slice()));
    assert_eq!(captured.soc_dual.as_dense(), Some([0.0].as_slice()));
    assert_eq!(captured.reduced_costs.as_dense(), Some([0.0].as_slice()));
    assert!(captured.psd_dual.is_empty());
}

#[test]
#[should_panic(expected = "source algebraic row coordinates")]
fn captured_finalization_rejects_counts_beyond_the_model() {
    let model = model_with_variables(1);
    let coordinates =
        SourceCoordinates { algebraic_constraints: 1, ..SourceCoordinates::capture(&model) };
    finalize_result_with_coordinates(SolverResult::for_model(&model), &model, coordinates);
}

fn point(values: &[(u32, f64)], objective: f64) -> SolutionPoint {
    SolutionPoint {
        model_id: oximo_core::ModelId::UNASSIGNED,
        primal: values.iter().map(|&(id, value)| (VarId(id), value)).collect(),
        objective: Some(objective),
        parameters: oximo_solver::ParameterSnapshot::default(),
    }
}

#[test]
fn projection_handles_reordering_auxiliaries_and_missing_columns() {
    let columns = [Some(VarId(1)), None, Some(VarId(0))];
    let projected = project_primal(&[4.0, f64::NAN, 3.0], &columns, 2).unwrap();
    assert_eq!(projected.get(VarId(0)).unwrap(), 3.0);
    assert_eq!(projected.get(VarId(1)).unwrap(), 4.0);
    assert!(project_primal(&[4.0], &columns, 2).is_none());
    assert!(project_primal(&[4.0, 3.0], &[Some(VarId(0)); 2], 2).is_none());
    assert!(project_primal(&[f64::NAN, 0.0, 3.0], &columns, 2).is_none());
    assert!(project_primal(&[], &[], 0).is_some());
}

#[test]
fn dense_native_projection_requires_matching_extent_and_finite_values() {
    let empty = project_dense_primal(&[], 0).unwrap();
    assert_eq!(empty.as_dense(), Some([].as_slice()));
    assert!(project_dense_primal(&[1.0], 2).is_none());

    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(project_dense_primal(&[1.0, invalid], 2).is_none());
    }

    let complete = project_dense_primal(&[-0.0, 2.0], 2).unwrap();
    assert_eq!(complete.get(VarId(0)).unwrap().to_bits(), (-0.0_f64).to_bits());
    assert_eq!(complete.as_dense().unwrap().len(), 2);
}

#[test]
fn limits_require_points_and_invalid_incumbents_do_not_keep_their_duals() {
    let model = model_with_variables(1);
    let mut result = SolverResult {
        termination: TerminationStatus::TimeLimit,
        ..SolverResult::for_model(&model)
    };
    assert_eq!(finalize_result(result.clone(), &model, 1).primal_status, PrimalStatus::NoSolution);
    result.solutions = vec![point(&[(0, 3.0)], 7.0)];
    assert_eq!(
        finalize_result(result.clone(), &model, 1).primal_status,
        PrimalStatus::FeasiblePoint
    );
    result.solutions.insert(0, point(&[], 2.0));
    result.dual_status = DualStatus::FeasiblePoint;
    result.gap = Some(0.25);
    result.dual.insert(ConstraintId(0), 99.0);
    let result = finalize_result(result, &model, 1);
    assert_eq!(result.result_count(), 1);
    assert!(result.dual.is_empty());
    assert_eq!(result.dual_status, DualStatus::Unknown);
    assert_eq!(result.gap, None);
}

#[test]
fn discarded_only_incumbent_clears_gap_but_preserves_independent_bound() {
    let model = model_with_variables(1);
    let result = finalize_result(
        SolverResult {
            termination: TerminationStatus::TimeLimit,
            solutions: vec![point(&[(0, f64::NAN)], 7.0)],
            best_bound: Some(2.0),
            gap: Some(0.5),
            ..SolverResult::for_model(&model)
        },
        &model,
        1,
    );
    assert_eq!(result.primal_status, PrimalStatus::NoSolution);
    assert_eq!(result.best_bound, Some(2.0));
    assert_eq!(result.gap, None);
}

#[test]
fn empty_solutions_clear_gap() {
    let model = model_with_variables(1);
    let result = finalize_result(
        SolverResult {
            termination: TerminationStatus::TimeLimit,
            gap: Some(0.5),
            ..SolverResult::for_model(&model)
        },
        &model,
        1,
    );
    assert_eq!(result.primal_status, PrimalStatus::NoSolution);
    assert_eq!(result.gap, None);
}

#[test]
fn gaps_use_restored_units_and_retain_native_conventions_separately() {
    let model = model_with_variables(1);
    let transform = ObjectiveTransform { sign: -1.0, offset: 5.0 };
    let p = transform.restore(-10.0).unwrap();
    let b = transform.restore(-8.0).unwrap();
    let result = finalize_result(
        SolverResult {
            termination: TerminationStatus::TimeLimit,
            solutions: vec![point(&[(0, 1.0)], p)],
            best_bound: Some(b),
            gap: Some(0.25),
            ..SolverResult::for_model(&model)
        },
        &model,
        1,
    );
    assert_eq!(result.gap, Some(0.25));
    assert_eq!(relative_gap(Some(f64::MAX), Some(-f64::MAX)), Some(2.0));
    assert_eq!(relative_gap(Some(0.0), Some(0.0)), Some(0.0));
    assert_eq!(relative_gap(None, Some(2.0)), None);
    assert!(transform.restore(f64::INFINITY).is_none());
}

#[test]
fn only_global_optimality_supplies_a_missing_bound() {
    let model = model_with_variables(0);
    for (termination, bound) in [
        (TerminationStatus::Optimal, Some(3.0)),
        (TerminationStatus::LocallyOptimal, None),
        (TerminationStatus::Feasible, None),
    ] {
        let result = finalize_result(
            SolverResult {
                termination,
                solutions: vec![point(&[], 3.0)],
                ..SolverResult::for_model(&model)
            },
            &model,
            0,
        );
        assert!(result.has_solution());
        assert_eq!(result.best_bound, bound);
    }
}

#[test]
fn split_rows_accumulate_sign_and_objective_transform() {
    let mut dual = oximo_solver::SolutionValues::missing(5);
    accumulate_dual(&mut dual, ConstraintId(4), 3.0, -1.0);
    accumulate_dual(&mut dual, ConstraintId(4), 5.0, 1.0);
    assert_eq!(dual.get(ConstraintId(4)).unwrap(), 2.0);
}

#[test]
fn discarded_optimal_incumbent_does_not_certify_another_pool_point() {
    let model = model_with_variables(1);
    for bound in [None, Some(2.0)] {
        let mut result = SolverResult {
            termination: TerminationStatus::Optimal,
            primal_status: PrimalStatus::OptimalPoint,
            solutions: vec![point(&[], 2.0), point(&[(0, 3.0)], 7.0)],
            best_bound: bound,
            gap: Some(0.0),
            dual_status: DualStatus::FeasiblePoint,
            ..SolverResult::for_model(&model)
        };
        result.dual.insert(ConstraintId(0), 99.0);
        // Repeated normalization must not upgrade the retained pool point.
        for _ in 0..2 {
            result = finalize_result(result, &model, 1);
            assert_eq!(result.termination, TerminationStatus::Optimal);
            assert_eq!(result.primal_status, PrimalStatus::FeasiblePoint);
            assert_eq!(result.objective(), Some(7.0));
            assert_eq!(result.best_bound, bound);
            assert_eq!(result.gap, None);
            assert_eq!(result.dual_status, DualStatus::Unknown);
            assert!(result.dual.is_empty());
        }
    }
}

#[test]
fn normalization_filters_auxiliaries_but_rejects_invalid_original_values() {
    let model = model_with_variables(1);
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let result = finalize_result(
            SolverResult {
                termination: TerminationStatus::Optimal,
                primal_status: PrimalStatus::OptimalPoint,
                solutions: vec![
                    point(&[(0, bad), (7, 1.0)], 1.0),
                    point(&[(0, 2.0), (7, bad)], 2.0),
                ],
                dual_status: DualStatus::FeasiblePoint,
                ..SolverResult::for_model(&model)
            },
            &model,
            1,
        );
        assert_eq!(result.result_count(), 1);
        assert_eq!(result.solutions[0].primal.len(), 1);
        assert_eq!(result.solutions[0].primal.get(VarId(0)).unwrap(), 2.0);
        assert_eq!(result.primal_status, PrimalStatus::FeasiblePoint);
        assert_eq!(result.dual_status, DualStatus::Unknown);
        assert_eq!(result.best_bound, None);
    }
}
