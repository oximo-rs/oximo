#![expect(clippy::float_cmp)]

use oximo_core::prelude::*;
use oximo_expr::extract_linear;

#[test]
fn derives_tight_m_and_preserves_history_and_ids() {
    let model = Model::new("indicator_big_m");
    variable!(model, b, Binary);
    variable!(model, 0.0 <= x <= 10.0);
    variable!(model, -2.0 <= y <= 4.0);
    let source = indicator_constraint!(model, row, b == 1 => 2.0 * x - 3.0 * y + 7.0 <= 20.0);

    // Maximum at the inactive trigger value is 2*10 - 3*(-2) + 7 = 33.
    let transformed =
        source.to_reformulated_model(IndicatorReformulationOptions::default()).unwrap();
    assert_eq!(transformed.num_variables(), model.num_variables());
    assert!(!transformed.indicator_constraints()[0].active);
    assert_eq!(transformed.constraints().algebraic()[0].name, "__oximo_indicator0_upper");
    let arena = transformed.arena();
    let terms = extract_linear(&arena, transformed.constraints().algebraic()[0].lhs).unwrap();
    assert_eq!(
        terms.coeffs.iter().map(|(v, c)| (v.index(), *c)).collect::<Vec<_>>(),
        [(1, 2.0), (2, -3.0), (0, 13.0)]
    );
    assert_eq!(terms.constant, -26.0);
    assert_eq!(transformed.constraints().algebraic()[0].upper, 0.0);
    assert_eq!(transformed.indicator_reformulations()[0].source, source.id());

    // The original remains native and active.
    assert!(model.indicator_constraints()[0].active);
    assert!(model.constraints().algebraic().is_empty());
}

#[test]
fn trigger_zero_range_and_zero_m_sides() {
    let model = Model::new("indicator_sides");
    variable!(model, b, Binary);
    variable!(model, -3.0 <= x <= 5.0);
    indicator_constraint!(model, range, b == 0 => -2.0 <= x <= 4.0);
    indicator_constraint!(model, always, b == 1 => -4.0 <= x <= 6.0);
    let result =
        model.to_reformulated_indicator_model(IndicatorReformulationOptions::default()).unwrap();
    assert_eq!(result.indicator_reformulations()[0].source, IndicatorConstraintId(0));
    assert_eq!(result.indicator_reformulations()[0].constraints.len(), 2);
    assert_eq!(result.indicator_reformulations()[1].source, IndicatorConstraintId(1));
    assert_eq!(result.indicator_reformulations()[1].constraints.len(), 2);
    assert_eq!(result.constraints().algebraic().len(), 4);
    let arena = result.arena();
    let rows = result.constraints();
    for (row, expected_constant) in rows.algebraic()[2..].iter().zip([4.0, -6.0]) {
        let terms = extract_linear(&arena, row.lhs).unwrap();
        assert_eq!(terms.coeffs.as_ref(), &[(x.var_id().unwrap(), 1.0)]);
        assert_eq!(terms.constant, expected_constant);
    }
    assert_eq!(rows.algebraic()[2].lower, 0.0);
    assert_eq!(rows.algebraic()[3].upper, 0.0);
}

#[test]
fn exact_equality_rows_and_trigger_in_body() {
    let model = Model::new("indicator_equality");
    variable!(model, b, Binary);
    variable!(model, -1.0 <= x <= 3.0);
    let row = indicator_constraint!(model, equal, b == 1 => x + 2.0 * b == 1.0);
    let transformed = row.to_reformulated_model(IndicatorReformulationOptions::default()).unwrap();
    assert_eq!(transformed.constraints().algebraic().len(), 2);
    let arena = transformed.arena();
    let lower = extract_linear(&arena, transformed.constraints().algebraic()[0].lhs).unwrap();
    let upper = extract_linear(&arena, transformed.constraints().algebraic()[1].lhs).unwrap();
    assert_eq!(lower.coeffs.iter().map(|(_, c)| *c).collect::<Vec<_>>(), [1.0]);
    assert_eq!(upper.coeffs.iter().map(|(_, c)| *c).collect::<Vec<_>>(), [1.0, 4.0]);
}

#[test]
fn semi_variable_effective_bounds_and_nonfinite_body() {
    let model = Model::new("indicator_semi");
    variable!(model, b, Binary);
    variable!(model, x <= 10.0, SemiCont(2.0));
    indicator_constraint!(model, cap, b == 1 => x <= 1.0);
    let transformed =
        model.to_reformulated_indicator_model(IndicatorReformulationOptions::default()).unwrap();
    let arena = transformed.arena();
    let row = extract_linear(&arena, transformed.constraints().algebraic()[0].lhs).unwrap();
    assert_eq!(row.coeffs.iter().find(|(id, _)| id.index() == 0).unwrap().1, 9.0);

    let invalid = Model::new("indicator_nonfinite");
    variable!(invalid, trigger, Binary);
    variable!(invalid, value);
    indicator_constraint!(invalid, bad, trigger == 1 => f64::INFINITY * value <= 1.0);
    assert!(matches!(
        invalid.to_reformulated_indicator_model(IndicatorReformulationOptions::default()),
        Err(ReformulationError::InvalidIndicatorExpression { .. })
    ));
}

#[test]
fn failures_are_atomic_and_fallback_is_a_row_m() {
    let model = Model::new("indicator_errors");
    variable!(model, b, Binary);
    variable!(model, x);
    let first = indicator_constraint!(model, first, b == 1 => x <= 2.0);
    param!(model, rhs = 1.0);
    indicator_constraint!(model, second, b == 1 => x <= rhs);
    assert!(matches!(
        model.reformulate_indicators(IndicatorReformulationOptions::default()),
        Err(ReformulationError::MissingIndicatorBigM { side: "upper", .. })
    ));
    assert!(model.indicator_constraints().iter().all(|c| c.active));
    assert!(model.constraints().algebraic().is_empty());
    assert!(matches!(
        model.reformulate_indicator_constraint(
            IndicatorConstraintId(1),
            IndicatorReformulationOptions::default()
        ),
        Err(ReformulationError::ParameterDependentIndicator { .. })
    ));
    first.reformulate(IndicatorReformulationOptions::default().with_fallback_big_m(50.0)).unwrap();
    assert_eq!(model.constraints().algebraic().len(), 1);
    let arena = model.arena();
    let fallback_row = extract_linear(&arena, model.constraints().algebraic()[0].lhs).unwrap();
    assert_eq!(fallback_row.coeffs.iter().find(|(id, _)| id.index() == 0).unwrap().1, 50.0);
    assert_eq!(model.parameters()[0].id.index(), 0);
    assert!(matches!(
        model.to_reformulated_indicator_constraint_model(
            IndicatorConstraintId(99),
            IndicatorReformulationOptions::default()
        ),
        Err(ReformulationError::UnknownIndicatorConstraint(99))
    ));
    assert!(matches!(
        model.to_reformulated_indicator_model(IndicatorReformulationOptions::default().with_fallback_big_m(f64::INFINITY)),
        Err(ReformulationError::InvalidFallbackBigM(value)) if value.is_infinite()
    ));
}

#[test]
fn collision_safe_names_and_inactive_sources_are_stable() {
    let model = Model::new("indicator_names");
    variable!(model, b, Binary);
    variable!(model, -1.0 <= x <= 1.0);
    constraint!(model, __oximo_indicator0_upper, x <= 1.0);
    let indicator = indicator_constraint!(model, cap, b == 1 => x <= 0.0);
    let artifacts =
        indicator.reformulate(IndicatorReformulationOptions::default()).unwrap().unwrap();
    assert_eq!(model.constraints().algebraic()[1].name, "__oximo_indicator0_upper_1");
    assert_eq!(model.indicator_reformulations()[0].constraints, artifacts.constraints);
    assert!(indicator.reformulate(IndicatorReformulationOptions::default()).unwrap().is_none());
    assert_eq!(model.indicator_reformulations().len(), 1);
}

#[test]
fn in_place_history_prevents_bound_changes_and_chains_with_sos() {
    let model = Model::new("indicator_chain");
    variable!(model, b, Binary);
    variable!(model, -2.0 <= x <= 5.0);
    variable!(model, -1.0 <= y <= 1.0);
    indicator_constraint!(model, row, b == 1 => x <= 3.0);
    sos_constraint!(model, choice, SOS1, [x, y]);
    model.reformulate_indicators(IndicatorReformulationOptions::default()).unwrap();
    assert_eq!(model.indicator_reformulations().len(), 1);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || model.fix_var(x.var_id().unwrap(), 1.0)
        ))
        .is_err()
    );
    let transformed = model.to_reformulated_sos_model(SosReformulationOptions::default()).unwrap();
    assert_eq!(transformed.indicator_reformulations().len(), 1);
    assert_eq!(transformed.sos_reformulations().len(), 1);
}
