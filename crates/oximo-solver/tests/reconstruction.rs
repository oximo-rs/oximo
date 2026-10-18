use oximo_core::{ConstraintId, VarId};
use oximo_solver::reconstruct::{
    ObjectiveTransform, accumulate_dual, finalize_result, project_primal, relative_gap,
};
use oximo_solver::{DualStatus, PrimalStatus, SolutionPoint, SolverResult, TerminationStatus};

fn model_with_variables(count: usize) -> oximo_core::Model {
    let model = oximo_core::Model::new("result finalization");
    for index in 0..count {
        model.__var(format!("x{index}")).build();
    }
    model
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
    assert_eq!(projected[&VarId(0)], 3.0);
    assert_eq!(projected[&VarId(1)], 4.0);
    assert!(project_primal(&[4.0], &columns, 2).is_none());
    assert!(project_primal(&[4.0, 3.0], &[Some(VarId(0)); 2], 2).is_none());
    assert!(project_primal(&[f64::NAN, 0.0, 3.0], &columns, 2).is_none());
    assert!(project_primal(&[], &[], 0).is_some());
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
    let mut dual = rustc_hash::FxHashMap::default();
    accumulate_dual(&mut dual, ConstraintId(4), 3.0, -1.0);
    accumulate_dual(&mut dual, ConstraintId(4), 5.0, 1.0);
    assert_eq!(dual[&ConstraintId(4)], 2.0);
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
        assert_eq!(result.solutions[0].primal[&VarId(0)], 2.0);
        assert_eq!(result.primal_status, PrimalStatus::FeasiblePoint);
        assert_eq!(result.dual_status, DualStatus::Unknown);
        assert_eq!(result.best_bound, None);
    }
}
