use oximo_expr::evaluate;
use oximo_gdp::prelude::*;

fn satisfied(model: &Model, values: &[f64]) -> bool {
    let arena = model.arena();
    model.constraints().algebraic().iter().all(|row| {
        let value = evaluate(&arena, row.lhs, &values).unwrap();
        value >= row.lower - 1e-8 && value <= row.upper + 1e-8
    })
}

fn near(actual: Option<f64>, expected: f64) {
    assert!((actual.unwrap() - expected).abs() < 1e-8, "{actual:?} != {expected}");
}

#[test]
fn parent_bounds_strengthen_relaxations_independently_of_declaration_order() {
    for inner_first in [false, true] {
        for enabled in [false, true] {
            let model = Model::new("nested relaxation");
            variable!(model, 0.0 <= x <= 10.0);
            boolean_variable!(model, parent);
            boolean_variable!(model, other);
            boolean_variable!(model, child);
            boolean_variable!(model, sibling);
            disjunct_constraint!(model, parent, x <= 4.0);
            disjunct_constraint!(model, child, x <= 1.0);
            disjunct_constraint!(model, sibling, x >= 9.0);
            if inner_first {
                disjunction!(model, [child, sibling], parent = parent);
                disjunction!(model, [parent, other]);
            } else {
                disjunction!(model, [parent, other]);
                disjunction!(model, [child, sibling], parent = parent);
            }
            let report = model
                .reformulate_gdp(BigM::default().with_hierarchical_tightening(enabled))
                .unwrap();
            assert_eq!(model.num_constraints(), 5);
            assert_eq!(report.variables.len(), 0);
            near(report.rows[1].upper_m.map(|m| m.value), 9.0);
            assert_eq!(satisfied(&model, &[7.0, 0.5, 0.5, 0.25, 0.25]), !enabled);
            assert!(satisfied(&model, &[6.25, 0.5, 0.5, 0.25, 0.25]));
            assert!(satisfied(&model, &[10.0, 0.0, 1.0, 0.0, 0.0]));
            assert!(satisfied(&model, &[1.0, 1.0, 0.0, 1.0, 0.0]));
            assert!(!satisfied(&model, &[2.0, 1.0, 0.0, 1.0, 0.0]));
            if enabled {
                let terms = &report.rows[1].hierarchical_m;
                assert_eq!(terms.len(), 2);
                assert_eq!(terms[0].indicator, child.id());
                assert_eq!(terms[1].indicator, parent.id());
                near(terms[0].upper, 3.0);
                near(terms[1].upper, 6.0);
            } else {
                assert!(report.rows.iter().all(|row| row.hierarchical_m.is_empty()));
            }
        }
    }
}

#[test]
fn ranges_and_inclusive_disjunctions_preserve_integer_feasibility() {
    for inclusive in [false, true] {
        let model = Model::new("nested range");
        variable!(model, 0.0 <= x <= 10.0);
        boolean_variable!(model, parent);
        boolean_variable!(model, other);
        boolean_variable!(model, child);
        boolean_variable!(model, sibling);
        disjunct_constraint!(model, parent, 2.0 <= x <= 4.0);
        disjunct_constraint!(model, child, 3.0 <= x <= 3.5);
        disjunction!(model, [parent, other]);
        if inclusive {
            disjunction!(model, [child, sibling], parent = parent, AtLeastOne);
        } else {
            disjunction!(model, [child, sibling], parent = parent);
        }
        let report = model.reformulate_gdp(BigM::default()).unwrap();
        let terms = &report.rows[1].hierarchical_m;
        near(terms[0].lower, 1.0);
        near(terms[0].upper, 0.5);
        near(terms[1].lower, 2.0);
        near(terms[1].upper, 6.0);
        for bits in 0..16 {
            let parent = bits & 1 != 0;
            let other = bits & 2 != 0;
            let child = bits & 4 != 0;
            let sibling = bits & 8 != 0;
            let selected = usize::from(child) + usize::from(sibling);
            let logic = parent != other
                && if inclusive {
                    if parent { selected >= 1 } else { selected == 0 }
                } else {
                    selected == usize::from(parent)
                };
            for x in [0.0, 2.0, 3.0, 3.25, 3.5, 4.0, 10.0] {
                let expected = logic
                    && (!parent || (2.0..=4.0).contains(&x))
                    && (!child || (3.0..=3.5).contains(&x));
                assert_eq!(
                    satisfied(
                        &model,
                        &[
                            x,
                            f64::from(parent),
                            f64::from(other),
                            f64::from(child),
                            f64::from(sibling)
                        ]
                    ),
                    expected
                );
            }
        }
    }
}

#[test]
fn multiple_ancestors_have_telescoping_relaxations() {
    let model = Model::new("three levels");
    variable!(model, 0.0 <= x <= 10.0);
    boolean_variable!(model, root);
    boolean_variable!(model, root_other);
    boolean_variable!(model, parent);
    boolean_variable!(model, parent_other);
    boolean_variable!(model, child);
    boolean_variable!(model, child_other);
    disjunct_constraint!(model, root, x <= 8.0);
    disjunct_constraint!(model, parent, x <= 4.0);
    disjunct_constraint!(model, child, x <= 1.0);
    disjunction!(model, [child, child_other], parent = parent);
    disjunction!(model, [parent, parent_other], parent = root);
    disjunction!(model, [root, root_other]);
    let report = model.reformulate_gdp(BigM::default()).unwrap();
    let terms = &report.rows[2].hierarchical_m;
    assert_eq!(
        terms.iter().map(|term| term.indicator).collect::<Vec<_>>(),
        [child.id(), root.id(), parent.id()]
    );
    near(terms[0].upper, 3.0);
    near(terms[1].upper, 2.0);
    near(terms[2].upper, 4.0);
    assert!(!satisfied(&model, &[6.0, 0.75, 0.25, 0.5, 0.25, 0.25, 0.25]));
    for (root, parent, child) in
        [(false, false, false), (true, false, false), (true, true, false), (true, true, true)]
    {
        for x in [0.0, 1.0, 2.0, 4.0, 8.0, 10.0] {
            assert_eq!(
                satisfied(
                    &model,
                    &[
                        x,
                        f64::from(root),
                        f64::from(!root),
                        f64::from(parent),
                        f64::from(root && !parent),
                        f64::from(child),
                        f64::from(parent && !child)
                    ]
                ),
                (!root || x <= 8.0) && (!parent || x <= 4.0) && (!child || x <= 1.0)
            );
        }
    }
}

#[test]
fn nonlinear_children_use_affine_ancestor_bounds_and_keep_global_domain_checks() {
    let model = Model::new("nonlinear child");
    variable!(model, 0.0 <= x <= 3.0);
    param!(model, rate = 2.0);
    boolean_variable!(model, parent);
    boolean_variable!(model, other);
    boolean_variable!(model, child);
    boolean_variable!(model, sibling);
    disjunct_constraint!(model, parent, rate * x <= 2.0);
    disjunct_constraint!(model, child, x.exp() <= 2.0);
    disjunction!(model, [parent, other]);
    disjunction!(model, [child, sibling], parent = parent);
    let copy = model.to_reformulated_gdp_model(BigM::default()).unwrap();
    model.reformulate_gdp(BigM::default()).unwrap();
    near(model.gdp_reformulations()[0].rows[1].hierarchical_m[0].upper, std::f64::consts::E - 2.0);
    for x in [0.0, 0.5, 0.7, 1.0, 3.0] {
        for (p, c) in [(false, false), (true, false), (true, true)] {
            let values = [x, f64::from(p), f64::from(!p), f64::from(c), f64::from(p && !c)];
            let expected = (!p || x <= 1.0) && (!c || x <= 2.0_f64.ln());
            assert_eq!(satisfied(&model, &values), expected);
            assert_eq!(satisfied(&copy, &values), expected);
        }
    }
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rate.set_param_value(1.0)))
            .is_err()
    );

    let unsafe_model = Model::new("unsafe global domain");
    variable!(unsafe_model, 0.0 <= z <= 3.0);
    boolean_variable!(unsafe_model, p);
    boolean_variable!(unsafe_model, c);
    boolean_variable!(unsafe_model, spare);
    disjunct_constraint!(unsafe_model, p, z >= 1.0);
    disjunct_constraint!(unsafe_model, c, z.ln() <= 1.0);
    disjunction!(unsafe_model, [c, spare], parent = p);
    let before = unsafe_model.arena().len();
    assert!(matches!(
        unsafe_model.reformulate_gdp(BigM::default()),
        Err(GdpError::UnsafeDomain { .. })
    ));
    assert_eq!(unsafe_model.num_constraints(), 0);
    assert_eq!(unsafe_model.arena().len(), before);
}

#[test]
fn shared_nonlinear_rows_use_each_parents_bounds_in_small_and_large_models() {
    for padding in [0, 5000] {
        let model = Model::new("shared nonlinear scopes");
        variable!(model, 0.0 <= x <= 3.0);
        for index in 0..padding {
            model.__var(format!("unrelated{index}")).build();
        }
        boolean_variable!(model, left);
        boolean_variable!(model, right);
        boolean_variable!(model, left_child);
        boolean_variable!(model, left_other);
        boolean_variable!(model, right_child);
        boolean_variable!(model, right_other);
        let body = x.exp();
        disjunct_constraint!(model, left, x <= 1.0);
        disjunct_constraint!(model, right, x <= 2.0);
        disjunct_constraint!(model, left_child, body <= 1.0);
        disjunct_constraint!(model, right_child, body <= 1.0);
        disjunction!(model, [left, right]);
        disjunction!(model, [left_child, left_other], parent = left);
        disjunction!(model, [right_child, right_other], parent = right);
        let report = model.reformulate_gdp(BigM::default()).unwrap();
        near(report.rows[2].hierarchical_m[0].upper, 1.0_f64.exp() - 1.0);
        near(report.rows[3].hierarchical_m[0].upper, 2.0_f64.exp() - 1.0);
        near(report.rows[2].upper_m.map(|m| m.value), 3.0_f64.exp() - 1.0);
        near(report.rows[3].upper_m.map(|m| m.value), 3.0_f64.exp() - 1.0);
    }
}

#[test]
fn explicit_global_m_values_keep_their_origin_and_disabling_propagation_disables_splitting() {
    for enabled in [false, true] {
        let model = Model::new("explicit nested M");
        variable!(model, 0.0 <= x <= 10.0);
        boolean_variable!(model, parent);
        boolean_variable!(model, child);
        boolean_variable!(model, other);
        disjunct_constraint!(model, parent, x <= 4.0);
        let row = disjunct_constraint!(model, child, x <= 1.0);
        disjunction!(model, [child, other], parent = parent);
        let report = model
            .reformulate_gdp(
                GdpReformulationOptions::new(BigM::default())
                    .with_bound_tightening(enabled)
                    .with_big_m(row, BigMValues::upper(20.0)),
            )
            .unwrap();
        assert_eq!(
            report.rows[1].upper_m,
            Some(BigMSide { value: 20.0, origin: BigMOrigin::Explicit })
        );
        if enabled {
            near(report.rows[1].hierarchical_m[0].upper, 3.0);
            near(report.rows[1].hierarchical_m[1].upper, 17.0);
        } else {
            assert!(report.rows[1].hierarchical_m.is_empty());
        }
    }
}

#[test]
fn ancestor_indicator_coefficient_precision_loss_is_atomic() {
    let model = Model::new("ancestor cancellation");
    variable!(model, -8.0 <= x <= 0.0);
    boolean_variable!(model, parent);
    boolean_variable!(model, child);
    boolean_variable!(model, other);
    disjunct_constraint!(model, parent, x <= -1.0);
    let bound = 1e14_f64;
    let row =
        disjunct_constraint!(model, child, x.exp() + bound.next_down() * parent.binary() <= bound);
    disjunction!(model, [child, other], parent = parent);
    let options =
        GdpReformulationOptions::new(BigM::default()).with_big_m(row, BigMValues::upper(bound));
    let before = (model.num_variables(), model.num_constraints(), model.arena().len());
    assert!(matches!(model.reformulate_gdp(options.clone()), Err(GdpError::PrecisionLoss { .. })));
    assert!(matches!(
        model.to_reformulated_gdp_model(options),
        Err(GdpError::PrecisionLoss { .. })
    ));
    assert_eq!(before, (model.num_variables(), model.num_constraints(), model.arena().len()));
    assert!(model.has_unreformulated_gdp());
    assert!(model.gdp_reformulations().is_empty());
}

#[test]
fn zero_active_bounds_survive_multiple_ancestor_shifts() {
    let model = Model::new("zero nested bound");
    variable!(model, -10.0 <= x <= 10.0);
    boolean_variable!(model, root);
    boolean_variable!(model, parent);
    boolean_variable!(model, parent_other);
    boolean_variable!(model, child);
    boolean_variable!(model, child_other);
    disjunct_constraint!(model, root, x <= 4.0);
    disjunct_constraint!(model, parent, x <= 2.0);
    disjunct_constraint!(model, child, x <= 0.0);
    disjunction!(model, [parent, parent_other], parent = root);
    disjunction!(model, [child, child_other], parent = parent);
    model.reformulate_gdp(BigM::default()).unwrap();
    assert!(satisfied(&model, &[0.0, 1.0, 1.0, 0.0, 1.0, 0.0]));
    assert!(!satisfied(&model, &[0.001, 1.0, 1.0, 0.0, 1.0, 0.0]));
}
