use oximo_gdp::prelude::*;
use oximo_solver::{SolutionPoint, SolverResult};

fn point(boolean: BooleanHandle<'_>, value: Option<f64>) -> SolutionPoint {
    SolutionPoint {
        model_id: boolean.model_id(),
        primal: value
            .into_iter()
            .map(|value| (boolean.binary().var_id().unwrap(), value))
            .collect(),
        objective: None,
    }
}

fn result(point: SolutionPoint) -> SolverResult {
    SolverResult { model_id: point.model_id, solutions: vec![point], ..SolverResult::default() }
}

#[test]
fn boolean_queries_accept_integrality_tolerance_on_both_sides() {
    let model = Model::new("Boolean values");
    boolean_variable!(model, selected);
    let tolerance = 1e-5_f64;
    for (value, expected) in [
        (-tolerance, false),
        (-0.0, false),
        (0.0, false),
        (tolerance, false),
        ((1.0 - tolerance).next_up(), true),
        (1.0, true),
        ((1.0 + tolerance).next_down(), true),
    ] {
        let point = point(selected, Some(value));
        assert_eq!(point.boolean_value_of(selected), Ok(Some(expected)), "value={value}");
        assert_eq!(result(point).boolean_value_of(selected), Ok(Some(expected)), "value={value}");
    }
}

#[test]
fn absent_boolean_values_remain_missing() {
    let model = Model::new("missing Boolean");
    boolean_variable!(model, selected);
    let point = point(selected, None);
    assert_eq!(point.boolean_value_of(selected), Ok(None));
    assert_eq!(result(point).boolean_value_of(selected), Ok(None));
    let empty = SolverResult { model_id: model.id(), ..SolverResult::default() };
    assert_eq!(empty.boolean_value_of(selected), Ok(None));
}

#[test]
fn fractional_and_non_finite_values_are_errors_and_numeric_queries_remain_available() {
    let model = Model::new("fractional Boolean");
    boolean_variable!(model, selected);
    let tolerance = 1e-5_f64;
    for value in [
        -tolerance.next_up(),
        tolerance.next_up(),
        (1.0 - tolerance).next_down(),
        (1.0 + tolerance).next_up(),
        0.5,
        0.6,
        -1.0,
        2.0,
        f64::NAN,
        f64::NEG_INFINITY,
        f64::INFINITY,
    ] {
        let point = point(selected, Some(value));
        let result = result(point.clone());
        for error in [
            point.boolean_value_of(selected).unwrap_err(),
            result.boolean_value_of(selected).unwrap_err(),
        ] {
            let BooleanValueError::NonBooleanValue { value: actual, tolerance: actual_tolerance } =
                error
            else {
                panic!("unexpected error: {error:?}");
            };
            assert_eq!(actual.to_bits(), value.to_bits());
            assert_eq!(actual_tolerance, tolerance);
        }
        assert_eq!(point.value_of(selected.binary()).unwrap().unwrap().to_bits(), value.to_bits());
        assert_eq!(result.value_of(selected.binary()).unwrap().unwrap().to_bits(), value.to_bits());
    }
}

#[test]
fn foreign_boolean_handles_are_rejected_even_without_a_solution() {
    let model = Model::new("own Boolean");
    boolean_variable!(model, selected);
    let other = Model::new("foreign Boolean");
    boolean_variable!(other, foreign);
    let expected =
        BooleanValueError::ModelMismatch(ModelMismatchError::new(model.id(), other.id()));
    for value in [None, Some(1.0)] {
        let point = point(selected, value);
        assert_eq!(point.boolean_value_of(foreign).unwrap_err(), expected);
        assert_eq!(result(point).boolean_value_of(foreign).unwrap_err(), expected);
    }
    let empty = SolverResult { model_id: model.id(), ..SolverResult::default() };
    assert_eq!(empty.boolean_value_of(foreign).unwrap_err(), expected);
}

#[test]
fn result_queries_the_best_point_and_individual_points_keep_their_own_values() {
    let model = Model::new("multiple selections");
    boolean_variable!(model, selected);
    let mut result = result(point(selected, Some(1.0)));
    result.solutions.push(point(selected, Some(0.0)));
    assert_eq!(result.boolean_value_of(selected), Ok(Some(true)));
    assert_eq!(result.solution(1).unwrap().boolean_value_of(selected), Ok(Some(false)));
    result.solutions[0].primal.clear();
    assert_eq!(result.boolean_value_of(selected), Ok(None));
}
