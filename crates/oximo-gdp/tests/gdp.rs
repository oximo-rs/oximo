use oximo_expr::{ExprNode, evaluate};
use oximo_gdp::prelude::*;
use std::panic::{AssertUnwindSafe, catch_unwind};

fn satisfied(model: &Model, values: &[f64]) -> bool {
    let arena = model.arena();
    model.constraints().algebraic().iter().all(|r| {
        let value = evaluate(&arena, r.lhs, &values).unwrap();
        value.is_finite() && value >= r.lower - 1e-9 && value <= r.upper + 1e-9
    })
}

fn auxiliary_feasible(model: &Model, a: bool, b: bool) -> bool {
    let n = model.num_variables();
    assert!(n < 20, "truth-table model should remain small");
    (0..1_usize << (n - 2)).any(|mask| {
        let mut values = vec![f64::from(a), f64::from(b)];
        values.extend((0..n - 2).map(|i| f64::from(mask & (1 << i) != 0)));
        satisfied(model, &values)
    })
}

#[test]
fn named_blocks_preserve_sources_ids_and_history() {
    let m = Model::new("blocks");
    variable!(m,0.0<=x<=10.0);
    let prior = constraint!(m, global, x >= 0.0);
    let a = disjunct!(m, low, |d| {
        constraint!(d, cap, x <= 2.0);
    });
    let b = disjunct!(m, high, |d| {
        constraint!(d, floor, x >= 5.0);
    });
    let choice = disjunction!(m, choice, [a, b]);
    assert!(m.has_unreformulated_gdp());
    assert_eq!(m.constraints().algebraic().len(), 1);
    objective!(m, Min, x);
    let clone = m.to_reformulated_gdp_model(BigM::default()).unwrap();
    assert_ne!(clone.id(), m.id());
    assert!(m.has_unreformulated_gdp());
    assert!(!clone.has_unreformulated_gdp());
    assert_eq!(clone.constraint_id("global"), Some(prior.id()));
    let history = clone.gdp_reformulations();
    let report = &history[0];
    assert_eq!(report.disjunctions[0].source, choice.id());
    assert!((report.rows[0].upper_m.unwrap().value - 8.0).abs() < 1e-10);
    assert!((report.rows[1].lower_m.unwrap().value - 5.0).abs() < 1e-10);
    assert!(satisfied(&clone, &[1.0, 1.0, 0.0]));
    assert!(!satisfied(&clone, &[3.0, 1.0, 0.0]));
    assert!(satisfied(&clone, &[7.0, 0.0, 1.0]));
    assert!(!satisfied(&clone, &[7.0, 0.0, 0.0]));
    assert_eq!(clone.boolean_handle(a.id()).binary().var_id(), a.binary().var_id());
}

#[test]
fn indexed_filtered_tagged_rows_and_block_ranges() {
    let m = Model::new("indexed");
    variable!(m,0.0<=x[i in 0..3]<=10.0);
    boolean_variable!(m,y[i in 0..3]);
    let rows = disjunct_constraint!(m,y[i],cap[i in 0..3 if i!=1],x[i]<=2.0);
    assert_eq!(rows.len(), 2);
    assert!(rows.get(1).is_none());
    let branches = disjunct!(m,b[i in 0..3],|d| {
        constraint!(d,ranges[j in 0..2],0.0<=x[j]<=4.0);
        constraint!(d,0.0<=x[2]<=6.0);
        constraint!(d,sum!(x[j] for j in 0..0)<=1.0);
        logical_constraint!(d,!y[1]);
    });
    assert_eq!(branches.len(), 3);
    disjunction!(m, [branches[0], branches[1], branches[2]]);
    let indexed = logical_constraint!(m,l[i in 0..3 if i!=1],y[i].implies(!y[1]));
    assert_eq!(indexed.len(), 2);
    m.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
    assert!(!m.has_unreformulated_gdp());
}

#[test]
fn nested_disjunctions_disable_all_children_with_parent() {
    let model = Model::new("nesting");
    boolean_variable!(model, p);
    boolean_variable!(model, q);
    boolean_variable!(model, a);
    boolean_variable!(model, b);
    disjunction!(model, outer, [p, q]);
    disjunction!(model, inner, [a, b], parent = p);
    logical_constraint!(p.context(), conditional, a | b);
    model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
    let aux_count = model.num_variables() - 4;
    for mask in 0..16_usize {
        let original: Vec<_> = (0..4).map(|i| f64::from(mask & (1 << i) != 0)).collect();
        let feasible = (0..1_usize << aux_count).any(|aux| {
            let mut values = original.clone();
            values.extend((0..aux_count).map(|i| f64::from(aux & (1 << i) != 0)));
            satisfied(&model, &values)
        });
        let parent = mask & 1 != 0;
        let alternative = mask & 2 != 0;
        let child_count = usize::from(mask & 4 != 0) + usize::from(mask & 8 != 0);
        assert_eq!(
            feasible,
            parent != alternative && child_count == usize::from(parent),
            "assignment {original:?}"
        );
    }
}

#[test]
fn inclusive_disjunctions_and_empty_branches() {
    let m = Model::new("inclusive");
    let a = disjunct!(m, a, |_d| {});
    let b = disjunct!(m, b, |_d| {});
    disjunction!(m, [a, b], AtLeastOne);
    m.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
    assert!(satisfied(&m, &[1.0, 1.0]));
    assert!(!satisfied(&m, &[0.0, 0.0]));
}

#[test]
fn truth_tables_for_all_logic_and_nested_cardinality() {
    for case in 0..15 {
        let m = Model::new("logic");
        boolean_variable!(m, a);
        boolean_variable!(m, b);
        let expr = match case {
            0 => !a,
            1 => a & b,
            2 => a | b,
            3 => a ^ b,
            4 => implies(a, b),
            5 => iff(a, b),
            6 => exactly(1, [a, b]),
            7 => at_most(1, [a, b]),
            8 => at_least(1, [a, b]),
            9 => exactly(1, [!a, a & b]),
            10 => at_least(2, [LogicalExpr::from(a), iff(a, b), !b]),
            11 => exactly(0, std::iter::empty::<BooleanHandle<'_>>()),
            12 => at_least(3, [a, b]),
            13 => true | a,
            14 => false & a,
            _ => unreachable!(),
        };
        logical_constraint!(m, expr);
        m.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
        for a in [false, true] {
            for b in [false, true] {
                let expected = match case {
                    0 => !a,
                    1 => a && b,
                    2 | 8 => a || b,
                    3 => a ^ b,
                    4 => !a || b,
                    5 => a == b,
                    6 => a != b,
                    7 => !(a && b),
                    9 => usize::from(!a) + usize::from(a && b) == 1,
                    10 => usize::from(a) + usize::from(a == b) + usize::from(!b) >= 2,
                    11 | 13 => true,
                    12 | 14 => false,
                    _ => unreachable!(),
                };
                assert_eq!(auxiliary_feasible(&m, a, b), expected, "case {case}, {a}, {b}");
            }
        }
    }
}

#[test]
fn independent_method_overrides_and_m_precedence() {
    let model = Model::new("overrides");
    variable!(model, value);
    boolean_variable!(model, a);
    boolean_variable!(model, b);
    boolean_variable!(model, c);
    boolean_variable!(model, d);
    let first = disjunct_constraint!(model, a, first, value <= 2.0);
    disjunct_constraint!(model, c, second, value >= 5.0);
    let d1 = disjunction!(model, d1, [a, b]);
    let d2 = disjunction!(model, d2, [c, d]);
    let report = model
        .reformulate_gdp(
            GdpReformulationOptions::default()
                .with_fallback_big_m(100.0)
                .with_method(d1, BigM::default().with_fallback_big_m(40.0))
                .with_method(d2, BigM::default().with_fallback_big_m(50.0))
                .with_big_m(first, BigMValues { lower: None, upper: Some(20.0) }),
        )
        .unwrap();
    assert_eq!(
        report.rows[0].upper_m.unwrap(),
        BigMSide { value: 20.0, origin: BigMOrigin::Explicit }
    );
    assert_eq!(
        report.rows[1].lower_m.unwrap(),
        BigMSide { value: 50.0, origin: BigMOrigin::DisjunctionFallback }
    );
    assert_eq!(report.disjunctions.len(), 2);
    let empty = model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
    assert!(empty.rows.is_empty());
    assert_eq!(model.gdp_reformulations().len(), 1);
}

#[test]
fn unconfigured_disjunctions_nested_children_and_standalone_rows_use_base_method() {
    let model = Model::new("mixed methods");
    variable!(model, x);
    boolean_variable!(model, selected[i in 0..9]);
    disjunct_constraint!(model, selected[0], x <= 2.0);
    disjunct_constraint!(model, selected[2], x >= 5.0);
    disjunct_constraint!(model, selected[4], x <= 8.0);
    disjunct_constraint!(model, selected[6], x <= 10.0);
    disjunct_constraint!(model, selected[7], x >= 1.0);
    let choice_a = disjunction!(model, [selected[0], selected[1]]);
    let choice_b = disjunction!(model, [selected[2], selected[3]]);
    disjunction!(model, [selected[4], selected[5]]);
    disjunction!(model, [selected[7], selected[8]], parent = selected[0]);

    let report = model
        .reformulate_gdp(
            GdpReformulationOptions::new(BigM::default().with_fallback_big_m(100.0))
                .with_method(choice_a, BigM::default().with_fallback_big_m(40.0))
                .with_method(choice_b, BigM::default().with_fallback_big_m(50.0)),
        )
        .unwrap();

    assert_eq!(
        report.rows[0].upper_m.unwrap(),
        BigMSide { value: 40.0, origin: BigMOrigin::DisjunctionFallback }
    );
    assert_eq!(
        report.rows[1].lower_m.unwrap(),
        BigMSide { value: 50.0, origin: BigMOrigin::DisjunctionFallback }
    );
    for side in [report.rows[2].upper_m, report.rows[3].upper_m, report.rows[4].lower_m] {
        assert_eq!(side.unwrap(), BigMSide { value: 100.0, origin: BigMOrigin::GlobalFallback });
    }
}

#[test]
fn method_objects_and_enums_preserve_configuration_in_both_reformulation_apis() {
    let model = Model::new("direct method");
    variable!(model, x);
    boolean_variable!(model, active);
    disjunct_constraint!(model, active, x <= 2.0);
    let method = BigM::default().with_fallback_big_m(20.0);

    let transformed = model.to_reformulated_gdp_model(GdpMethod::from(method)).unwrap();
    assert!(model.has_unreformulated_gdp());
    assert_ne!(transformed.id(), model.id());
    assert_eq!(transformed.gdp_reformulations()[0].rows[0].upper_m.unwrap().value, 20.0);

    let report = model.reformulate_gdp(method).unwrap();
    assert!(!model.has_unreformulated_gdp());
    assert_eq!(report.rows[0].upper_m.unwrap().value, 20.0);
}

#[test]
fn estimates_win_over_fallback_and_zero_m_is_valid() {
    let m = Model::new("zero");
    variable!(m,0.0<=x<=1.0);
    boolean_variable!(m, a);
    let row = disjunct_constraint!(m, a, row, x <= 2.0);
    let report =
        m.reformulate_gdp(GdpReformulationOptions::default().with_fallback_big_m(100.0)).unwrap();
    assert_eq!(report.rows[0].source, row.id());
    assert_eq!(
        report.rows[0].upper_m.unwrap(),
        BigMSide { value: 0.0, origin: BigMOrigin::Estimated }
    );
    assert_eq!(report.rows[0].constraints.len(), 1);
}

#[test]
fn nonlinear_equality_and_parameter_bounds() {
    let m = Model::new("nonlinear");
    variable!(m,-2.0<=x<=3.0);
    boolean_variable!(m, a);
    param!(m, p = 2.0);
    disjunct_constraint!(m, a, equality, p * x.square() == 4.0);
    let report = m.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
    assert!((report.rows[0].lower_m.unwrap().value - 4.0).abs() < 1e-10);
    assert!((report.rows[0].upper_m.unwrap().value - 14.0).abs() < 1e-10);
    assert!(satisfied(&m, &[0.0, 0.0]));
    assert!(!satisfied(&m, &[0.0, 1.0]));
    assert!(satisfied(&m, &[2.0_f64.sqrt(), 1.0]));
    assert!(catch_unwind(AssertUnwindSafe(|| m.set_param(p, 3.0).unwrap())).is_err());
    assert!(catch_unwind(AssertUnwindSafe(|| m.fix(x, 1.0).unwrap())).is_err());
    m.set_initial(x, 1.0).unwrap();
    variable!(m, unrelated);
    m.fix(unrelated, 1.0).unwrap();
}

#[test]
fn failures_are_atomic_and_domains_are_checked_with_explicit_m() {
    let model = Model::new("unsafe");
    variable!(model,-1.0<=x<=3.0);
    boolean_variable!(model, a);
    disjunct_constraint!(model, a, valid, x <= 1.0);
    let bad = disjunct_constraint!(model, a, invalid, x.ln() <= 1.0);
    let before =
        (model.num_variables(), model.arena().len(), model.constraints().algebraic().len());
    assert!(matches!(
        model.reformulate_gdp(
            GdpReformulationOptions::default().with_big_m(bad, BigMValues::symmetric(100.0))
        ),
        Err(GdpError::UnsafeDomain { operation: "log", .. })
    ));
    assert_eq!(
        before,
        (model.num_variables(), model.arena().len(), model.constraints().algebraic().len())
    );
    assert!(model.gdp().rows.iter().all(|r| r.state() == ReformulationState::Pending));
    assert!(model.gdp_reformulations().is_empty());
    let unbounded = Model::new("unbounded");
    variable!(unbounded, z);
    boolean_variable!(unbounded, b);
    disjunct_constraint!(unbounded, b, row, z <= 1.0);
    assert!(matches!(
        unbounded.reformulate_gdp(GdpReformulationOptions::default()),
        Err(GdpError::MissingBigM { side: BoundSide::Upper, .. })
    ));
    assert!(matches!(
        unbounded.reformulate_gdp(GdpReformulationOptions::default().with_fallback_big_m(f64::NAN)),
        Err(GdpError::InvalidBigM(_))
    ));
}

#[test]
fn foreign_handles_and_structural_errors_are_rejected() {
    let model = Model::new("ownership");
    boolean_variable!(model, a);
    boolean_variable!(model, b);
    let choice = disjunction!(model, choice, [a, b]);
    assert!(catch_unwind(AssertUnwindSafe(|| disjunction!(model, [a, b]))).is_err());
    let other = Model::new("other");
    assert!(matches!(
        other.reformulate_gdp(
            GdpReformulationOptions::default().with_method(choice, BigM::default())
        ),
        Err(GdpError::ForeignHandle("disjunction"))
    ));
    assert!(catch_unwind(AssertUnwindSafe(|| logical_constraint!(other, a))).is_err());
    let cycle = Model::new("cycle");
    boolean_variable!(cycle, p);
    boolean_variable!(cycle, q);
    disjunction!(cycle, child, [q], parent = p);
    assert!(
        catch_unwind(AssertUnwindSafe(|| disjunction!(cycle, parent, [p], parent = q))).is_err()
    );
}

#[test]
fn newly_added_gdp_is_transformed_once_and_sealed_groups_cannot_change() {
    let m = Model::new("append");
    variable!(m,0.0<=x<=3.0);
    boolean_variable!(m, a);
    disjunct_constraint!(m, a, row, x <= 1.0);
    m.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
    assert!(catch_unwind(AssertUnwindSafe(|| disjunct_constraint!(m, a, late, x <= 2.0))).is_err());
    boolean_variable!(m, b);
    disjunct_constraint!(m, b, new_row, x >= 2.0);
    let report = m.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
    assert_eq!(report.rows.len(), 1);
    assert_eq!(m.gdp_reformulations().len(), 2);
}

#[test]
fn clone_keeps_source_mutable_and_rebinds_logical_records() {
    let m = Model::new("clone");
    variable!(m,0.0<=x<=3.0);
    boolean_variable!(m, a);
    disjunct_constraint!(m, a, row, x <= 1.0);
    logical_constraint!(m, original, a);
    let clone = m.to_reformulated_gdp_model(GdpReformulationOptions::default()).unwrap();
    m.fix(x, 2.0).unwrap();
    let expression = clone.gdp().logical_constraints[0].expression.clone();
    logical_constraint!(clone, copy, expression);
    clone.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
    assert!(m.has_unreformulated_gdp());
    assert!(!clone.has_unreformulated_gdp());
}

#[test]
fn solver_lowering_rejects_pending_logic_even_without_conditional_rows() {
    let m = Model::new("reject");
    boolean_variable!(m, a);
    logical_constraint!(m, a);
    objective!(m, Feasibility);
    assert!(matches!(
        oximo_solver::prepare::LoweringContext::new(&m),
        Err(oximo_solver::SolverError::Core(oximo_core::Error::UnreformulatedGdp))
    ));
    m.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
    assert!(oximo_solver::prepare::LoweringContext::new(&m).is_ok());
}

#[test]
fn semicontinuous_enclosures_include_zero_even_with_singleton_positive_branch() {
    let m = Model::new("semi");
    let x = m.__var("x").bounds(2.0, 2.0).domain(Domain::SemiContinuous { threshold: 2.0 }).build();
    boolean_variable!(m, y);
    disjunct_constraint!(m, y, row, x >= 1.0);
    let report = m.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
    assert!((report.rows[0].lower_m.unwrap().value - 1.0).abs() < 1e-10);
    assert!(satisfied(&m, &[0.0, 0.0]));
    assert!(!satisfied(&m, &[0.0, 1.0]));
}

#[test]
fn boolean_bounds_remain_binary_but_fixing_a_selector_is_allowed() {
    let m = Model::new("binary bounds");
    boolean_variable!(m, y);
    logical_constraint!(m, y);
    m.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
    m.fix(y.binary(), 1.0).unwrap();
    m.unfix_var(y.binary().var_id().unwrap(), 0.0, 1.0);
    assert!(
        catch_unwind(AssertUnwindSafe(|| m.unfix_var(y.binary().var_id().unwrap(), 0.0, 2.0)))
            .is_err()
    );
}

#[test]
fn parameter_dependent_affine_cancellation_and_integer_power() {
    let m = Model::new("parameter algebra");
    variable!(m, x);
    param!(m, p = 2.0);
    boolean_variable!(m, y);
    disjunct_constraint!(m, y, cancel, p * x - p * x <= 1.0);
    let report = m.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
    assert_eq!(report.rows[0].upper_m.unwrap().value, 0.0);
    let power = Model::new("power");
    variable!(power,-2.0<=x<=3.0);
    param!(power, p = 2.0);
    boolean_variable!(power, y);
    disjunct_constraint!(power, y, pow, x.pow(p + 1.0) <= 1.0);
    let report = power.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
    assert!((report.rows[0].upper_m.unwrap().value - 26.0).abs() < 1e-10);
}

#[test]
fn generated_names_avoid_user_collisions_and_display_marks_pending_gdp() {
    let m = Model::new("names");
    variable!(m,0.0<=x<=3.0);
    variable!(m, __oximo_gdp_boolean0, Binary);
    constraint!(m, name = String::from("__oximo_gdp_row1"), x >= 0.0);
    boolean_variable!(m, y);
    disjunct_constraint!(m, y, row, x <= 2.0);
    assert!(m.to_string().contains("pending reformulation"));
    m.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
    assert!(m.to_string().contains("reformulated"));
    assert!(m.constraint_id("__oximo_gdp_row1_1").is_some());
    assert_ne!(y.binary().var_id(), __oximo_gdp_boolean0.var_id());
}

#[test]
fn every_parameter_setter_preserves_reformulated_rows() {
    let model = Model::new("parameter locks");
    variable!(model, 0.0 <= flow <= 10.0);
    param!(model, coefficient = 1.0);
    param!(model, unrelated = 1.0);
    boolean_variable!(model, active);
    disjunct_constraint!(model, active, capacity, coefficient * flow <= 2.0);
    let report = model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
    assert!((report.rows[0].upper_m.unwrap().value - 8.0).abs() < 1e-10);
    let parameter_id = coefficient.param_id().unwrap();
    assert!(catch_unwind(AssertUnwindSafe(|| coefficient.set_param_value(100.0))).is_err());
    assert!(
        catch_unwind(AssertUnwindSafe(|| model.set_param(coefficient, 100.0).unwrap())).is_err()
    );
    assert!(catch_unwind(AssertUnwindSafe(|| model.set_param_id(parameter_id, 100.0))).is_err());
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            model.__sum_context().borrow_mut().set_param_value(parameter_id, 100.0);
        }))
        .is_err()
    );
    assert_eq!(model.param_value(parameter_id), 1.0);
    assert!(satisfied(&model, &[10.0, 0.0]));
    unrelated.set_param_value(100.0);
    assert_eq!(model.param_value(unrelated.param_id().unwrap()), 100.0);
}

#[test]
fn indexed_parameter_handles_and_model_setters_share_locks() {
    let model = Model::new("indexed locks");
    variable!(model, 0.0 <= flow <= 10.0);
    param!(model, coefficient[i in 0..2] = 1.0);
    boolean_variable!(model, active);
    disjunct_constraint!(model, active, capacity, coefficient[0] * flow <= 2.0);
    model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
    assert!(catch_unwind(AssertUnwindSafe(|| coefficient[0].set_param_value(100.0))).is_err());
    assert!(
        catch_unwind(AssertUnwindSafe(|| model.set_param_idx(&coefficient, 0, 100.0).unwrap()))
            .is_err()
    );
    coefficient[1].set_param_value(100.0);
    assert_eq!(model.param_value_idx(&coefficient, 0).unwrap(), Some(1.0));
    assert_eq!(model.param_value_idx(&coefficient, 1).unwrap(), Some(100.0));
}

#[test]
fn parameter_locks_are_preserved_by_clones_and_isolated_from_the_source() {
    let model = Model::new("clone locks");
    variable!(model, 0.0 <= flow <= 10.0);
    param!(model, coefficient = 1.0);
    boolean_variable!(model, active);
    disjunct_constraint!(model, active, capacity, coefficient * flow <= 2.0);
    let transformed = model.to_reformulated_gdp_model(GdpReformulationOptions::default()).unwrap();
    let parameter_id = coefficient.param_id().unwrap();
    let transformed_parameter = transformed.__sum_context().borrow_mut().param_expr(parameter_id);
    assert!(
        catch_unwind(AssertUnwindSafe(|| transformed_parameter.set_param_value(100.0))).is_err()
    );
    coefficient.set_param_value(100.0);
    assert_eq!(model.param_value(parameter_id), 100.0);
    assert_eq!(transformed.param_value(parameter_id), 1.0);
    assert!(satisfied(&transformed, &[10.0, 0.0]));
    let cloned = transformed.to_reformulated_gdp_model(GdpReformulationOptions::default()).unwrap();
    assert!(catch_unwind(AssertUnwindSafe(|| cloned.set_param_id(parameter_id, 100.0))).is_err());
    assert_eq!(cloned.param_value(parameter_id), 1.0);
}

#[test]
fn anonymous_declarations_use_available_names_in_each_disjunct_scope() {
    let model = Model::new("scoped names");
    boolean_variable!(model, parent);
    let context = parent.context();
    logical_constraint!(context, _gdp0, true);
    logical_constraint!(context, true);
    logical_constraint!(context, true);
    for _ in 0..2 {
        let left = disjunct!(context, |_child| {});
        let right = disjunct!(context, |_child| {});
        disjunction!(context, [left, right]);
    }
    boolean_variable!(model, other);
    logical_constraint!(other.context(), true);
    logical_constraint!(other.context(), true);
    let data = model.gdp();
    assert_eq!(data.disjunctions.len(), 2);
    assert_eq!(data.logical_constraints.len(), 5);
    assert_eq!(data.logical_constraints[1].name, "parent::_gdp1");
    assert_eq!(data.logical_constraints[2].name, "parent::_gdp2");
    assert_eq!(data.logical_constraints[3].name, "other::_gdp0");
    assert_eq!(data.logical_constraints[4].name, "other::_gdp1");
    drop(data);
    model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
}

#[test]
fn anonymous_conditional_rows_and_ranges_skip_explicit_names() {
    for case in 0..3 {
        let model = Model::new("conditional names");
        variable!(model, 0.0 <= flow <= 10.0);
        variable!(model, 0.0 <= lower_bound <= 10.0);
        variable!(model, 0.0 <= upper_bound <= 10.0);
        boolean_variable!(model, active);
        let context = active.context();
        constraint!(context, _c0, flow <= 10.0);
        constraint!(context, _c2, flow <= 10.0);
        match case {
            0 => {
                constraint!(context, flow >= 1.0);
            }
            1 => {
                constraint!(context, 1.0 <= flow <= 8.0);
            }
            _ => {
                constraint!(context, _c3_lo, flow <= 10.0);
                constraint!(context, lower_bound <= flow <= upper_bound);
            }
        }
        let names: Vec<_> = model.gdp().rows.iter().map(|row| row.name.clone()).collect();
        assert!(names.iter().any(|name| name == "active::_c1"));
        assert_eq!(names.len(), if case == 2 { 5 } else { 3 });
        model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
    }
}

#[test]
fn anonymous_disjunctions_accept_identifier_branch_lists_and_options() {
    for case in 0..3 {
        let model = Model::new("identifier disjunction");
        boolean_variable!(model, left);
        boolean_variable!(model, right);
        boolean_variable!(model, parent);
        let branches = [left, right];
        match case {
            0 => {
                disjunction!(model, branches, AtLeastOne);
                model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
                assert!(satisfied(&model, &[1.0, 1.0, 0.0]));
                assert!(!satisfied(&model, &[0.0, 0.0, 0.0]));
            }
            1 => {
                disjunction!(model, branches, ExactlyOne);
                model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
                assert!(satisfied(&model, &[1.0, 0.0, 0.0]));
                assert!(!satisfied(&model, &[1.0, 1.0, 0.0]));
            }
            _ => {
                disjunction!(model, branches, parent = parent, AtLeastOne);
                model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
                assert!(satisfied(&model, &[1.0, 1.0, 1.0]));
                assert!(satisfied(&model, &[0.0, 0.0, 0.0]));
                assert!(!satisfied(&model, &[1.0, 0.0, 0.0]));
            }
        }
    }
}

#[test]
fn wide_selection_emits_one_compact_linear_node() {
    let model = Model::new("wide selection");
    let branches: Vec<_> = (0..4096).map(|i| model.add_boolean(format!("b{i}"))).collect();
    let selection = disjunction!(model, branches);
    let before = model.arena().len();
    let report = model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
    assert_eq!(report.variables, []);
    assert_eq!(report.disjunctions[0].source, selection.id());
    assert_eq!(report.disjunctions[0].constraints.len(), 1);
    assert_eq!(model.arena().len(), before + 1);
    let rows = model.constraints();
    let arena = model.arena();
    let ExprNode::Linear { coeffs, .. } = arena.get(rows.algebraic()[0].lhs) else {
        panic!("selection should be a single Linear node");
    };
    assert_eq!(coeffs.len(), 4096);
    assert!(coeffs.iter().all(|(_, c)| (*c - 1.0).abs() < f64::EPSILON));
}

#[test]
fn nested_exactly_one_omits_only_implied_child_rows() {
    for inclusive in [false, true] {
        let model = Model::new("nested relaxation");
        boolean_variable!(model, parent);
        boolean_variable!(model, a);
        boolean_variable!(model, b);
        if inclusive {
            disjunction!(model, [a, b], parent = parent, AtLeastOne);
        } else {
            disjunction!(model, [a, b], parent = parent);
        }
        let report = model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
        assert_eq!(report.disjunctions[0].constraints.len(), if inclusive { 3 } else { 1 });
        assert!(satisfied(&model, &[0.5, 0.2, 0.3]));
        assert!(!satisfied(&model, &[0.0, 1.0, 0.0]));
        assert!(!satisfied(&model, &[0.5, 0.8, 0.0]));
    }
}

#[test]
fn direct_assertions_preserve_guarded_cardinality_truth_tables() {
    for guarded in [false, true] {
        for case in 0..17 {
            let model = Model::new("direct assertions");
            boolean_variable!(model, a);
            boolean_variable!(model, b);
            boolean_variable!(model, parent);
            let count = case / 3;
            let formula = if case == 15 {
                logical_and([LogicalExpr::from(a), !b])
            } else if case == 16 {
                logical_or([LogicalExpr::from(a), !b])
            } else {
                let terms = [LogicalExpr::from(a), !b, true.into()];
                match case % 3 {
                    0 => exactly(count, terms),
                    1 => at_most(count, terms),
                    _ => at_least(count, terms),
                }
            };
            if guarded {
                logical_constraint!(parent.context(), formula);
            } else {
                logical_constraint!(model, formula);
            }
            let report = model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
            assert_eq!(report.variables, []);
            assert!(report.logical_constraints[0].constraints.len() <= if guarded { 2 } else { 1 });
            if guarded && case == 15 {
                assert!(!satisfied(&model, &[1.0, 1.0, 0.5]));
            }
            for mask in 0..8 {
                let (a, b, p) = (mask & 1 != 0, mask & 2 != 0, mask & 4 != 0);
                let sum = usize::from(a) + usize::from(!b) + 1;
                let predicate = if case == 15 {
                    a && !b
                } else if case == 16 {
                    a || !b
                } else {
                    match case % 3 {
                        0 => sum == count,
                        1 => sum <= count,
                        _ => sum >= count,
                    }
                };
                assert_eq!(
                    satisfied(&model, &[f64::from(a), f64::from(b), f64::from(p)]),
                    guarded && !p || predicate,
                    "guarded={guarded}, case={case}, mask={mask}"
                );
            }
        }
    }
}

#[test]
fn compact_rows_merge_repeated_boolean_coefficients() {
    let model = Model::new("duplicate coefficients");
    boolean_variable!(model, a);
    boolean_variable!(model, b);
    logical_constraint!(model, exactly(1, [a, a, b]));
    logical_constraint!(model, exactly(1, [LogicalExpr::from(a), !a, b.into()]));
    model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
    let arena = model.arena();
    let constraints = model.constraints();
    for row in constraints.algebraic() {
        let terms = oximo_expr::extract_linear(&arena, row.lhs).unwrap();
        let ids: std::collections::HashSet<_> = terms.coeffs.iter().map(|(v, _)| *v).collect();
        assert_eq!(ids.len(), terms.coeffs.len());
    }
    assert_eq!(
        oximo_expr::extract_linear(&arena, constraints.algebraic()[1].lhs).unwrap().coeffs.len(),
        1
    );
}

#[test]
fn no_op_reformulation_validates_options_and_preserves_pending_lifecycle() {
    let model = Model::new("pending lifecycle");
    boolean_variable!(model, parent);
    logical_constraint!(parent.context(), at_least(0, [parent]));
    assert!(model.has_unreformulated_gdp());
    let transformed = model.to_reformulated_gdp_model(GdpReformulationOptions::default()).unwrap();
    assert!(model.has_unreformulated_gdp());
    assert!(!transformed.has_unreformulated_gdp());
    let first = model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
    assert_eq!(first.logical_constraints[0].constraints, [] as [oximo_core::ConstraintId; 0]);
    assert!(!model.has_unreformulated_gdp());
    let before = (model.arena().len(), model.gdp_reformulations().len());
    let empty = model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
    assert_eq!(empty.booleans.len(), 1);
    assert_eq!(before, (model.arena().len(), model.gdp_reformulations().len()));
    assert!(matches!(
        model.reformulate_gdp(GdpReformulationOptions::default().with_fallback_big_m(-1.0)),
        Err(GdpError::InvalidBigM(_))
    ));
    logical_constraint!(model, !parent);
    assert!(model.has_unreformulated_gdp());
    model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
    assert!(!model.has_unreformulated_gdp());
    assert!(satisfied(&model, &[0.0]));
}

#[test]
fn direct_binary_predicates_preserve_fractional_guard_relaxations() {
    for guarded in [false, true] {
        for case in 0..3 {
            let model = Model::new("binary predicate");
            boolean_variable!(model, a);
            boolean_variable!(model, b);
            boolean_variable!(model, parent);
            let predicate = match case {
                0 => implies(a, !b),
                1 => iff(a, !b),
                _ => a ^ !b,
            };
            if guarded {
                logical_constraint!(parent.context(), predicate);
            } else {
                logical_constraint!(model, predicate);
            }
            let report = model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
            assert_eq!(report.variables, []);
            assert_eq!(
                report.logical_constraints[0].constraints.len(),
                if guarded && case != 0 { 2 } else { 1 }
            );
            for av in [0.0_f64, 0.25, 0.5, 0.75, 1.0] {
                for bv in [0.0_f64, 0.25, 0.5, 0.75, 1.0] {
                    for pv in [0.0_f64, 0.5, 1.0] {
                        let guard = if guarded { pv } else { 1.0 };
                        let rhs = 1.0 - bv;
                        let expected = match case {
                            0 => rhs - av >= guard - 1.0,
                            1 => (av - rhs).abs() <= 1.0 - guard,
                            _ => av + rhs >= guard && av + rhs <= 2.0 - guard,
                        };
                        assert_eq!(satisfied(&model, &[av, bv, pv]), expected);
                    }
                }
            }
        }
    }
}

#[test]
fn flattening_and_shared_definitions_preserve_cardinality_multiplicity() {
    for case in 0..4 {
        let model = Model::new("shared logical terms");
        boolean_variable!(model, a);
        boolean_variable!(model, b);
        boolean_variable!(model, c);
        let formula = match case {
            0 => (a & b) & c,
            1 => a | (b | c),
            2 => exactly(1, [(a & b) & c, a & (b & c)]),
            _ => (a | b) & (a | b),
        };
        logical_constraint!(model, formula);
        let report = model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
        assert_eq!(report.variables.len(), usize::from(case >= 2));
        for mask in 0..8 {
            let (av, bv, cv) = (mask & 1 != 0, mask & 2 != 0, mask & 4 != 0);
            let expected = match case {
                0 => av && bv && cv,
                1 => av || bv || cv,
                2 => false,
                _ => av || bv,
            };
            let feasible = (0..1_usize << report.variables.len()).any(|aux| {
                let mut values = vec![f64::from(av), f64::from(bv), f64::from(cv)];
                values.extend((0..report.variables.len()).map(|i| f64::from(aux & (1 << i) != 0)));
                satisfied(&model, &values)
            });
            assert_eq!(feasible, expected, "case={case}, mask={mask}");
        }
    }
}

#[test]
fn shared_predicate_definitions_work_across_distinct_guards() {
    let model = Model::new("shared guarded predicates");
    boolean_variable!(model, a);
    boolean_variable!(model, b);
    boolean_variable!(model, p);
    boolean_variable!(model, q);
    logical_constraint!(p.context(), (a | b) & !a);
    logical_constraint!(q.context(), (a | b) & !b);
    let report = model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
    assert_eq!(report.variables.len(), 1);
    for mask in 0..16 {
        let bits: Vec<_> = (0..4).map(|i| mask & (1 << i) != 0).collect();
        let expected = (!bits[2] || (!bits[0] && bits[1])) && (!bits[3] || (bits[0] && !bits[1]));
        let feasible = [0.0, 1.0].into_iter().any(|aux| {
            let mut values: Vec<_> = bits.iter().map(|b| f64::from(*b)).collect();
            values.push(aux);
            satisfied(&model, &values)
        });
        assert_eq!(feasible, expected, "mask={mask}");
    }
}

#[test]
fn global_affine_bounds_tighten_both_m_sides_and_lock_supporting_dependencies() {
    let model = Model::new("global propagation");
    variable!(model, -1000.0 <= x <= 1000.0);
    variable!(model, 0.0 <= y <= 1000.0);
    variable!(model, 0.0 <= z <= 1000.0);
    param!(model, cap = 3.0);
    param!(model, unrelated = 7.0);
    constraint!(model, coupled, x - y <= 2.0);
    constraint!(model, cap_row, y <= cap);
    constraint!(model, floor, -x <= -1.0);
    constraint!(model, unrelated_row, z <= unrelated);
    boolean_variable!(model, active);
    disjunct_constraint!(model, active, band, 2.0 <= x <= 4.0);
    let report = model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
    assert!((report.rows[0].lower_m.unwrap().value - 1.0).abs() < 1e-10);
    assert!((report.rows[0].upper_m.unwrap().value - 1.0).abs() < 1e-10);
    assert!(satisfied(&model, &[5.0, 3.0, 0.0, 0.0]));
    assert!(satisfied(&model, &[1.0, 0.0, 0.0, 0.0]));
    assert!(satisfied(&model, &[3.0, 1.0, 0.0, 1.0]));
    assert!(!satisfied(&model, &[5.0, 3.0, 0.0, 1.0]));
    assert!(catch_unwind(AssertUnwindSafe(|| cap.set_param_value(20.0))).is_err());
    assert!(
        catch_unwind(AssertUnwindSafe(|| model.unfix_var(y.var_id().unwrap(), 0.0, 2000.0)))
            .is_err()
    );
    unrelated.set_param_value(8.0);
    model.unfix_var(z.var_id().unwrap(), 0.0, 2000.0);
    let clone = model.__gdp_clone();
    assert!(
        catch_unwind(AssertUnwindSafe(|| clone.set_param_id(cap.param_id().unwrap(), 20.0)))
            .is_err()
    );
}

#[test]
fn conditional_sibling_bounds_are_not_used_for_m_estimates() {
    let model = Model::new("conditional bounds");
    variable!(model, 0.0 <= x <= 1000.0);
    let a = disjunct!(model, a, |d| {
        constraint!(d, x <= 10.0);
    });
    let b = disjunct!(model, b, |d| {
        constraint!(d, x <= 2.0);
    });
    disjunction!(model, [a, b]);
    let report = model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
    assert!((report.rows[1].upper_m.unwrap().value - 998.0).abs() < 1e-9);
    assert!(satisfied(&model, &[10.0, 1.0, 0.0]));
}

#[test]
fn affine_sources_materialize_at_locked_parameter_values() {
    let model = Model::new("materialized source");
    variable!(model, 0.0 <= x <= 10.0);
    param!(model, coefficient = 2.0);
    boolean_variable!(model, active);
    let source = disjunct_constraint!(model, active, capacity, coefficient * x + 1.0 <= 5.0);
    objective!(model, Feasibility);
    let report = model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
    assert_eq!(report.rows[0].source, source.id());
    let arena = model.arena();
    let rows = model.constraints();
    assert!(matches!(arena.get(rows.algebraic()[0].lhs), ExprNode::Linear { .. }));
    assert!(matches!(arena.get(model.gdp().rows[0].lhs), ExprNode::Add(_)));
    drop(rows);
    drop(arena);
    let prepared = oximo_solver::prepare::LoweringContext::new(&model).unwrap();
    let expr = prepared.constraints().algebraic()[0].lhs;
    assert!((prepared.linear(expr).unwrap().coeffs[0].1 - 2.0).abs() < f64::EPSILON);
    assert!(satisfied(&model, &[2.0, 1.0]));
    assert!(!satisfied(&model, &[3.0, 1.0]));
}

#[test]
fn failed_incremental_numeric_reformulation_keeps_pending_suffix_for_retry() {
    let model = Model::new("pending retry");
    boolean_variable!(model, first);
    logical_constraint!(model, first);
    model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
    variable!(model, flow);
    boolean_variable!(model, second);
    let source = disjunct_constraint!(model, second, capacity, flow <= 2.0);
    logical_constraint!(model, implies(first, second));
    let before = model.num_constraints();
    assert!(matches!(
        model.reformulate_gdp(GdpReformulationOptions::default()),
        Err(GdpError::MissingBigM { .. })
    ));
    assert_eq!(model.num_constraints(), before);
    assert!(model.has_unreformulated_gdp());
    constraint!(model, global_box, 0.0 <= flow <= 10.0);
    let report = model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
    assert_eq!(report.rows.len(), 1);
    assert_eq!(report.rows[0].source, source.id());
    assert_eq!(report.logical_constraints.len(), 1);
    assert!((report.rows[0].upper_m.unwrap().value - 8.0).abs() < 1e-10);
    assert!(!model.has_unreformulated_gdp());
    boolean_variable!(model, left);
    boolean_variable!(model, right);
    disjunction!(model, [left, right]);
    disjunct_constraint!(model, left, new_capacity, flow <= 1.0);
    logical_constraint!(model, implies(left, !right));
    let clone = model.__gdp_clone();
    let report = clone.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
    assert_eq!(
        (report.rows.len(), report.disjunctions.len(), report.logical_constraints.len()),
        (1, 1, 1)
    );
    assert!(model.has_unreformulated_gdp());
    model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
    assert!(!model.has_unreformulated_gdp());
}

#[test]
fn global_bound_tightening_can_be_disabled_without_locking_global_parameters() {
    for tighten in [false, true] {
        let model = Model::new("optional tightening");
        variable!(model, 0.0 <= x <= 1000.0);
        param!(model, cap = 10.0);
        constraint!(model, x <= cap);
        boolean_variable!(model, b);
        disjunct_constraint!(model, b, x <= 2.0);
        let report = model
            .reformulate_gdp(GdpReformulationOptions::default().with_bound_tightening(tighten))
            .unwrap();
        let expected = if tighten { 8.0 } else { 998.0 };
        assert!((report.rows[0].upper_m.unwrap().value - expected).abs() < 1e-9);
        assert_eq!(catch_unwind(AssertUnwindSafe(|| cap.set_param_value(20.0))).is_err(), tighten);
        if tighten {
            assert!(!satisfied(&model, &[10.0, 0.5]));
        } else {
            assert!(satisfied(&model, &[10.0, 0.5]));
        }
    }
}

#[test]
fn unconditional_domain_bounds_can_validate_nonlinear_sources() {
    let model = Model::new("global log domain");

    variable!(model, 0.0 <= x <= 1000.0);
    param!(model, cap = 10.0);
    constraint!(model, lower, x >= 1.0);
    constraint!(model, upper, x <= cap);
    boolean_variable!(model, active);
    disjunct_constraint!(model, active, log_cap, x.ln() <= 1.0);
    let before = model.num_constraints();
    assert!(matches!(
        model.reformulate_gdp(GdpReformulationOptions::default().with_bound_tightening(false)),
        Err(GdpError::UnsafeDomain { operation: "log", .. })
    ));
    assert_eq!(model.num_constraints(), before);
    let report = model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
    assert!((report.rows[0].upper_m.unwrap().value - (10.0_f64.ln() - 1.0)).abs() < 1e-10);
    assert!(satisfied(&model, &[10.0, 0.0]));
    assert!(!satisfied(&model, &[10.0, 1.0]));
    assert!(satisfied(&model, &[2.0, 1.0]));
    assert!(catch_unwind(AssertUnwindSafe(|| cap.set_param_value(20.0))).is_err());
}

#[test]
fn nonnegative_sums_keep_safe_square_root_domains() {
    for squared in [false, true] {
        let model = Model::new("nonnegative sum");
        variable!(model, 0.0 <= x <= 1.0);
        variable!(model, 0.0 <= z <= 1.0);
        boolean_variable!(model, active);
        let sum = if squared { (x.square() + z.square()).erase() } else { (x + z).erase() };
        disjunct_constraint!(model, active, sum.sqrt() <= 1.0);
        model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
        assert!(satisfied(&model, &[0.0, 0.0, 1.0]));
        assert!(!satisfied(&model, &[1.0, 1.0, 1.0]));
        assert!(satisfied(&model, &[1.0, 1.0, 0.0]));
    }
}

#[test]
fn global_zero_domain_bounds_work_with_either_coefficient_sign() {
    for negative in [false, true] {
        let model = Model::new("global zero domain");
        variable!(model, -1.0 <= x <= 1.0);
        boolean_variable!(model, active);
        if negative {
            constraint!(model, -2.0 * x <= 0.0);
        } else {
            constraint!(model, 2.0 * x >= 0.0);
        }
        disjunct_constraint!(model, active, x.sqrt() <= 1.0);
        assert!(matches!(
            model.reformulate_gdp(GdpReformulationOptions::default().with_bound_tightening(false)),
            Err(GdpError::UnsafeDomain { operation: "sqrt", .. })
        ));
        model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
        assert!(satisfied(&model, &[0.0, 1.0]));
        assert!(satisfied(&model, &[1.0, 1.0]));
    }
    let model = Model::new("actually negative domain");
    let x = model.__var("x").bounds(-f64::from_bits(1), 1.0).build();
    boolean_variable!(model, active);
    disjunct_constraint!(model, active, x.sqrt() <= 1.0);
    assert!(matches!(
        model.reformulate_gdp(GdpReformulationOptions::default()),
        Err(GdpError::UnsafeDomain { operation: "sqrt", .. })
    ));
}

#[test]
fn affine_constant_normalization_avoids_big_m_overflow() {
    for upper in [false, true] {
        let model = Model::new("large source constant");
        variable!(model, -1e308 <= x <= 0.0);
        boolean_variable!(model, active);
        if upper {
            disjunct_constraint!(model, active, -x - 1e308 <= -1e308);
        } else {
            disjunct_constraint!(model, active, x + 1e308 >= 1e308);
        }
        model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
        let arena = model.arena();
        for row in model.constraints().algebraic() {
            let ExprNode::Linear { coeffs, constant } = arena.get(row.lhs) else {
                panic!("affine source should produce a linear row");
            };
            assert!(constant.is_finite());
            assert!(coeffs.iter().all(|(_, value)| value.is_finite()));
            assert!(if upper { row.upper.is_finite() } else { row.lower.is_finite() });
        }
        assert!(satisfied(&model, &[0.0, 1.0]));
        assert!(!satisfied(&model, &[-1e308, 1.0]));
        assert!(satisfied(&model, &[-1e308, 0.0]));
    }
}

#[test]
fn generated_numeric_overflow_is_rejected_before_any_mutation() {
    for case in 0..3 {
        let model = Model::new("numeric overflow");
        variable!(model, -1.0 <= x <= 1.0);
        param!(model, offset = 0.0);
        boolean_variable!(model, active);
        disjunct_constraint!(model, active, valid, x <= 0.0);
        let row = match case {
            0 => disjunct_constraint!(model, active, overflow, x + offset <= 1e308),
            1 => disjunct_constraint!(model, active, overflow, x + offset >= -1e308),
            _ => disjunct_constraint!(model, active, overflow, 1e308 * active.binary() <= 0.0),
        };
        let before = (model.num_variables(), model.num_constraints(), model.arena().len());
        let options =
            GdpReformulationOptions::default().with_big_m(row, BigMValues::symmetric(1e308));
        assert!(matches!(model.reformulate_gdp(options.clone()), Err(GdpError::NumericOverflow)));
        assert!(matches!(model.to_reformulated_gdp_model(options), Err(GdpError::NumericOverflow)));
        assert_eq!(before, (model.num_variables(), model.num_constraints(), model.arena().len()));
        assert!(model.has_unreformulated_gdp());
        assert!(model.gdp().rows.iter().all(|row| row.state() == ReformulationState::Pending));
        assert!(model.gdp_reformulations().is_empty());
        model.unfix_var(x.var_id().unwrap(), -2.0, 2.0);
        offset.set_param_value(1.0);
    }
}

#[test]
fn regression_global_tightening_merges_repeated_linear_coefficients() {
    for (coefficients, bound, expected_m, inactive_x) in
        [([1.0, -2.0], 1.0, 8.0, 10.0), ([1.0, -1.0], 1.0, 8.0, 10.0), ([1.0, 1.0], 10.0, 3.0, 5.0)]
    {
        let model = Model::new("repeated global coefficients");
        variable!(model, -10.0 <= x <= 10.0);
        boolean_variable!(model, active);
        let variable = x.var_id().unwrap();
        let root = model.__sum_context().borrow_mut().linear(
            coefficients.into_iter().map(|coefficient| (variable, coefficient)).collect(),
            0.0,
        );
        let expression: Expr = Expr::new(root, model.__sum_context());
        constraint!(model, global, expression <= bound);
        disjunct_constraint!(model, active, x <= 2.0);
        let report = model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
        assert!((report.rows[0].upper_m.unwrap().value - expected_m).abs() < 1e-10);
        assert!(satisfied(&model, &[inactive_x, 0.0]));
        assert!(satisfied(&model, &[2.0, 1.0]));
        assert!(!satisfied(&model, &[3.0, 1.0]));
    }
}

#[test]
fn regression_big_m_precision_loss_is_rejected_atomically_on_both_sides() {
    for nonlinear in [false, true] {
        for upper in [false, true] {
            for method in 0..3 {
                let model = Model::new("Big-M precision loss");
                let extent = match method {
                    0 => 1e14,
                    1 => 10.0,
                    _ => f64::INFINITY,
                };
                variable!(model, -extent <= x <= extent);
                let offset_value = if method == 0 { 0.0 } else { 0.125 };
                param!(model, offset = offset_value);
                boolean_variable!(model, active);
                disjunct_constraint!(model, active, valid, x <= 1e14);
                let residual = if nonlinear { (x + x.sin()).erase() } else { x.erase() };
                let row = if upper {
                    disjunct_constraint!(
                        model,
                        active,
                        ill_scaled,
                        residual + offset <= offset_value + 0.001
                    )
                } else {
                    disjunct_constraint!(
                        model,
                        active,
                        ill_scaled,
                        residual + offset >= offset_value - 0.001
                    )
                };
                let options = match method {
                    0 => GdpReformulationOptions::default(),
                    1 => GdpReformulationOptions::default()
                        .with_big_m(row, BigMValues::symmetric(1e14)),
                    _ => GdpReformulationOptions::default().with_fallback_big_m(1e14),
                };
                let before = (model.num_variables(), model.num_constraints(), model.arena().len());
                let expected = GdpError::PrecisionLoss {
                    constraint: "active::ill_scaled".into(),
                    side: if upper { BoundSide::Upper } else { BoundSide::Lower },
                };
                assert_eq!(model.reformulate_gdp(options.clone()).unwrap_err(), expected);
                assert_eq!(model.to_reformulated_gdp_model(options).unwrap_err(), expected);
                assert_eq!(
                    before,
                    (model.num_variables(), model.num_constraints(), model.arena().len())
                );
                assert!(model.has_unreformulated_gdp());
                assert!(
                    model.gdp().rows.iter().all(|row| row.state() == ReformulationState::Pending)
                );
                assert!(model.gdp_reformulations().is_empty());
                model.unfix_var(x.var_id().unwrap(), -10.0, 10.0);
                offset.set_param_value(0.0);
                let report = model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
                assert_eq!(report.rows.len(), 2);
                assert!(!model.has_unreformulated_gdp());
            }
        }
    }
}

#[test]
fn big_m_indicator_coefficient_precision_loss_is_atomic() {
    for nonlinear in [false, true] {
        for upper in [false, true] {
            for coefficient in [-0.001, 0.001] {
                for method in 0..3 {
                    let model = Model::new("indicator coefficient precision");
                    let extent = match method {
                        0 => 1e14,
                        1 => 1.0,
                        _ => f64::INFINITY,
                    };
                    variable!(model, -extent <= x <= extent);
                    param!(model, scale = coefficient);
                    boolean_variable!(model, active);
                    disjunct_constraint!(model, active, valid, x <= 1e14);
                    let residual = if nonlinear { (x + x.sin()).erase() } else { x.erase() };
                    let body = residual + scale * active.binary();
                    let row = if upper {
                        disjunct_constraint!(model, active, ill_scaled, body <= 0.0)
                    } else {
                        disjunct_constraint!(model, active, ill_scaled, body >= 0.0)
                    };
                    let options = match method {
                        0 => GdpReformulationOptions::default(),
                        1 => GdpReformulationOptions::default()
                            .with_big_m(row, BigMValues::symmetric(1e14)),
                        _ => GdpReformulationOptions::default().with_fallback_big_m(1e14),
                    };
                    let before =
                        (model.num_variables(), model.num_constraints(), model.arena().len());
                    let expected = GdpError::PrecisionLoss {
                        constraint: "active::ill_scaled".into(),
                        side: if upper { BoundSide::Upper } else { BoundSide::Lower },
                    };
                    assert_eq!(model.reformulate_gdp(options.clone()).unwrap_err(), expected);
                    assert_eq!(model.to_reformulated_gdp_model(options).unwrap_err(), expected);
                    assert_eq!(
                        before,
                        (model.num_variables(), model.num_constraints(), model.arena().len())
                    );
                    assert!(model.has_unreformulated_gdp());
                    assert!(model.gdp_reformulations().is_empty());
                    assert!(
                        model.gdp().rows.iter().all(|r| r.state() == ReformulationState::Pending)
                    );

                    model.unfix_var(x.var_id().unwrap(), -1.0, 1.0);
                    scale.set_param_value(coefficient);
                    model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
                    // Tighter bounds retain the selector's coefficient and restore
                    // the original active inequality on either side of its boundary.
                    let delta = if upper { -0.01 } else { 0.01 };
                    assert!(satisfied(&model, &[-coefficient + delta, 1.0]));
                    assert!(!satisfied(&model, &[-coefficient - delta, 1.0]));
                    assert!(satisfied(&model, &[if upper { 1.0 } else { -1.0 }, 0.0]));
                }
            }
        }
    }
}

#[test]
fn big_m_checks_the_combined_source_indicator_coefficient() {
    for nonlinear in [false, true] {
        let model = Model::new("repeated source indicator coefficients");
        variable!(model, -1e14 <= x <= 1e14);
        boolean_variable!(model, active);
        let variable = x.var_id().unwrap();
        let indicator = active.binary().var_id().unwrap();
        let source = model
            .__sum_context()
            .borrow_mut()
            .linear(vec![(variable, 1.0), (indicator, 0.0005), (indicator, 0.0005)], 0.0);
        let affine: Expr = Expr::new(source, model.__sum_context());
        let body = if nonlinear { affine + x.sin() } else { affine };
        disjunct_constraint!(model, active, body <= 0.0);
        assert!(matches!(
            model.reformulate_gdp(GdpReformulationOptions::default()),
            Err(GdpError::PrecisionLoss { .. })
        ));
        assert_eq!(model.num_constraints(), 0);
        assert!(model.has_unreformulated_gdp());
    }
}

#[test]
fn big_m_preserves_the_effective_active_bound_after_coefficient_merging() {
    for nonlinear in [false, true] {
        for upper in [false, true] {
            let model = Model::new("active bound cancellation");
            variable!(model, -8.0 <= x <= 0.0);
            boolean_variable!(model, active);
            let bound = if upper { 1e14_f64 } else { -1e14_f64 };
            let coefficient = if upper { bound.next_down() } else { bound.next_up() };
            let residual = if nonlinear {
                if upper { x.exp().erase() } else { (-x.exp()).erase() }
            } else {
                x.erase()
            };
            let body = residual + coefficient * active.binary();
            let row = if upper {
                disjunct_constraint!(model, active, ill_scaled, body <= bound)
            } else {
                disjunct_constraint!(model, active, ill_scaled, body >= bound)
            };
            // Both the bound and coefficient individually round-trip within the
            // relative tolerance, but shifting them erases their small difference.
            // For a nonlinear residual, this would change exp(x) <= 0.015625
            // into exp(x) <= 0, eliminating all feasible active points.
            let options = GdpReformulationOptions::new(BigM::default())
                .with_big_m(row, BigMValues::symmetric(1e14));
            let before = (model.num_variables(), model.num_constraints(), model.arena().len());
            let expected = GdpError::PrecisionLoss {
                constraint: "active::ill_scaled".into(),
                side: if upper { BoundSide::Upper } else { BoundSide::Lower },
            };
            assert_eq!(model.reformulate_gdp(options.clone()).unwrap_err(), expected);
            assert_eq!(model.to_reformulated_gdp_model(options).unwrap_err(), expected);
            assert_eq!(
                before,
                (model.num_variables(), model.num_constraints(), model.arena().len())
            );
            assert!(model.has_unreformulated_gdp());
            assert!(model.gdp().rows.iter().all(|row| row.state() == ReformulationState::Pending));
            assert!(model.gdp_reformulations().is_empty());
        }
    }
}

#[test]
fn mixed_nonlinear_rows_preserve_affine_terms_constants_and_residual_signs() {
    let model = Model::new("mixed nonlinear range");
    variable!(model, -2.0 <= x <= 2.0);
    variable!(model, -1.0 <= z <= 1.0);
    param!(model, rate = 0.5);
    param!(model, coefficient = 0.25);
    param!(model, offset = 0.125);
    boolean_variable!(model, active);
    let body = x.square() - ((rate * x).exp() + coefficient * active.binary() + offset + z);
    disjunct_constraint!(model, active, interval, -2.0 <= body <= 3.0);
    let report = model.reformulate_gdp(BigM::default()).unwrap();
    assert_eq!(report.rows.len(), 1);
    assert_eq!(report.rows[0].constraints.len(), 2);
    assert_eq!(model.gdp().rows[0].lhs, body.id());

    for x_value in [-2.0_f64, -1.0, 0.0, 1.0, 2.0] {
        for z_value in [-1.0, 0.0, 1.0] {
            for selected in [false, true] {
                let value = x_value * x_value
                    - (0.5 * x_value).exp()
                    - 0.25 * f64::from(selected)
                    - 0.125
                    - z_value;
                assert_eq!(
                    satisfied(&model, &[x_value, z_value, f64::from(selected)]),
                    !selected || (-2.0..=3.0).contains(&value),
                    "x={x_value}, z={z_value}, selected={selected}"
                );
            }
        }
    }
}

#[test]
fn logical_auxiliary_bounds_remain_binary_and_preserve_truth_tables() {
    let model = Model::new("logical auxiliary bounds");
    boolean_variable!(model, a);
    boolean_variable!(model, b);
    boolean_variable!(model, c);
    logical_constraint!(model, exactly(2, [!(a & b), c.into()]));
    let report = model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
    assert_eq!(report.variables.len(), 1);
    let auxiliary = report.variables[0];
    for (lower, upper) in [(-1.0, 1.0), (0.0, 2.0)] {
        assert!(
            catch_unwind(AssertUnwindSafe(|| model.unfix_var(auxiliary, lower, upper))).is_err()
        );
    }
    model.fix_var(auxiliary, 0.0);
    model.unfix_var(auxiliary, 0.0, 1.0);
    for mask in 0..8 {
        let a = mask & 1 != 0;
        let b = mask & 2 != 0;
        let c = mask & 4 != 0;
        let feasible = [0.0, 1.0].into_iter().any(|auxiliary| {
            satisfied(&model, &[f64::from(a), f64::from(b), f64::from(c), auxiliary])
        });
        assert_eq!(feasible, !(a && b) && c);
    }
}

#[test]
fn deep_logic_snapshots_and_reformulated_copies_are_stack_safe() {
    for negate in [false, true] {
        let model = Model::new("deep logical model");
        boolean_variable!(model, selected);
        let mut formula: LogicalExpr = selected.into();
        for _ in 0..20_000 {
            formula = if negate { !formula } else { formula & selected };
        }
        logical_constraint!(model, formula);
        let snapshot = model.gdp_snapshot();
        assert_eq!(snapshot.logical_constraints[0].state(), ReformulationState::Pending);
        let copy = model.to_reformulated_gdp_model(GdpReformulationOptions::default()).unwrap();
        assert!(model.has_unreformulated_gdp());
        assert!(!copy.has_unreformulated_gdp());
        assert_ne!(copy.id(), model.id());
        assert!(satisfied(&copy, &[1.0]));
        assert!(!satisfied(&copy, &[0.0]));
        model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
        drop(snapshot);
        drop(copy);
        drop(model);
    }
}
