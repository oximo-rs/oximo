#![cfg(all(feature = "gdp", feature = "highs"))]
use oximo::prelude::*;
use oximo::{HighsOptions, solvers::Highs};

fn paper_model(equivalent: bool, flattened: bool) -> Model {
    let model = Model::new("paper Example 4.1");

    variable!(model, 1.0 <= x <= 9.0);
    variable!(model, 1.0 <= z <= 6.0);
    let w1 = disjunct!(model, w1, |d| {
        constraint!(d, 1.0 <= x <= 2.0);
        constraint!(d, 5.0 <= z <= 6.0);
    });
    let w2 = disjunct!(model, w2, |d| {
        constraint!(d, 2.0 <= x <= 3.0);
        constraint!(d, 4.0 <= z <= 5.0);
    });
    let y2 = disjunct!(model, y2, |d| {
        constraint!(d, 8.0 <= x <= 9.0);
        constraint!(d, 1.0 <= z <= 2.0);
    });
    if flattened {
        disjunction!(model, [w1, w2, y2]);
    } else {
        let y1 = disjunct!(model, y1, |d| {
            constraint!(d, 1.0 <= x <= 3.0);
            constraint!(d, 4.0 <= z <= 6.0);
        });
        disjunction!(model, [y1, y2]);
        if equivalent {
            boolean_variable!(model, slack);
            disjunction!(model, [w1, w2, slack]);
            constraint!(model, y1.binary() == w1.binary() + w2.binary());
        } else {
            disjunction!(model, [w1, w2], parent = y1);
        }
    }
    objective!(model, Max, x + 2.0 * z);
    model
}

fn support(model: &Model, direction: [f64; 2]) -> f64 {
    linear_support(model, &[(VarId(0), direction[0]), (VarId(1), direction[1])])
}

fn linear_support(model: &Model, direction: &[(VarId, f64)]) -> f64 {
    let relaxed = Model::new("Hull LP relaxation");
    for v in model.variables().iter() {
        relaxed.__var(v.name.clone()).bounds(v.lb, v.ub).build();
    }
    let source_arena = model.arena();
    for row in model.constraints().algebraic() {
        let terms = oximo::expr::extract_linear(&source_arena, row.lhs).unwrap();
        let id =
            relaxed.__sum_context().borrow_mut().linear(terms.coeffs.into_owned(), terms.constant);
        let expression: Expr = Expr::new(id, relaxed.__sum_context());
        relaxed.add_constraint(
            row.name.clone(),
            Constraint::new(
                ScalarDynamicFunction::from(expression),
                Interval { lower: row.lower, upper: row.upper },
            ),
        );
    }
    let id = relaxed.__sum_context().borrow_mut().linear(direction.to_vec(), 0.0);
    let expression: Expr = Expr::new(id, relaxed.__sum_context());
    objective!(relaxed, Max, expression);

    let result = Highs.solve(&relaxed, &HighsOptions::default()).unwrap();

    assert_eq!(result.termination, TerminationStatus::Optimal);
    result.objective().unwrap()
}

fn inner_relaxation(inclusive: bool) -> Model {
    let model = Model::new("independent child Hull");

    variable!(model, 1.0 <= x <= 9.0);
    variable!(model, 1.0 <= z <= 9.0);
    let c = disjunct!(model, c, |d| {
        constraint!(d, 1.0 <= x <= 3.0);
        constraint!(d, 1.0 <= z <= 3.0);
    });
    let d = disjunct!(model, d, |d| {
        constraint!(d, 7.0 <= x <= 9.0);
        constraint!(d, 7.0 <= z <= 9.0);
    });
    if inclusive {
        disjunction!(model, [c, d], AtLeastOne);
    } else {
        disjunction!(model, [c, d]);
    }
    model
        .reformulate_gdp(
            Hull::default()
                .with_inclusive_policy(HullInclusivePolicy::Enumerate { max_patterns: 3 }),
        )
        .unwrap();
    model
}

fn flattened_inclusive_relaxation(inner: &Model) -> Model {
    let model = Model::new("independent convex combination of nested pattern blocks");

    for v in inner.variables().iter() {
        model.__var(v.name.clone()).bounds(v.lb, v.ub).build();
    }
    let x = model.variable_handle(VarId(0));
    let z = model.variable_handle(VarId(1));
    let only_a = model.add_boolean("only a");
    let only_b = model.add_boolean("only b");
    let both = model.add_boolean("both");
    let arena = inner.arena();
    for branch in [only_a, both] {
        for (i, row) in inner.constraints().algebraic().iter().enumerate() {
            let terms = oximo::expr::extract_linear(&arena, row.lhs).unwrap();
            let id = model
                .__sum_context()
                .borrow_mut()
                .linear(terms.coeffs.into_owned(), terms.constant);
            let body: Expr = Expr::new(id, model.__sum_context());
            if row.lower.is_finite() {
                branch.context().add_constraint(format!("inner lower {i}"), body.ge(row.lower));
            }
            if row.upper.is_finite() {
                branch.context().add_constraint(format!("inner upper {i}"), body.le(row.upper));
            }
        }
    }
    for branch in [only_b, both] {
        let context = branch.context();
        constraint!(context, 3.0 <= x <= 7.0);
        constraint!(context, z == 4.0);
    }

    // In the B-only pattern the child selectors and all child copies are absent.
    for v in inner.variables().iter().skip(2) {
        only_b.context().add_constraint(
            format!("absent {}", v.id.index()),
            model.variable_handle(v.id).eq(0.0),
        );
    }
    disjunction!(model, [only_a, only_b, both]);
    model.reformulate_gdp(Hull::default()).unwrap();
    constraint!(model, only_a.binary() + both.binary() == 0.8);
    constraint!(model, only_b.binary() + both.binary() == 0.6);
    model
}

#[test]
fn inclusive_ancestor_relaxation_matches_independent_pattern_polytopes() {
    for inclusive_child in [false, true] {
        let model = Model::new("nested inclusive relaxation");

        variable!(model, 1.0 <= x <= 9.0);
        variable!(model, 1.0 <= z <= 9.0);
        boolean_variable!(model, parent);
        let other = disjunct!(model, other, |context| {
            constraint!(context, 3.0 <= x <= 7.0);
            constraint!(context, z == 4.0);
        });
        let low = disjunct!(model, low, |context| {
            constraint!(context, 1.0 <= x <= 3.0);
            constraint!(context, 1.0 <= z <= 3.0);
        });
        let high = disjunct!(model, high, |context| {
            constraint!(context, 7.0 <= x <= 9.0);
            constraint!(context, 7.0 <= z <= 9.0);
        });
        disjunction!(model, [parent, other], AtLeastOne);
        if inclusive_child {
            disjunction!(model, [low, high], parent = parent, AtLeastOne);
        } else {
            disjunction!(model, [low, high], parent = parent);
        }
        model
            .reformulate_gdp(
                Hull::default()
                    .with_inclusive_policy(HullInclusivePolicy::Enumerate { max_patterns: 3 }),
            )
            .unwrap();
        constraint!(model, parent.binary() == 0.8);
        constraint!(model, other.binary() == 0.6);
        let independent = flattened_inclusive_relaxation(&inner_relaxation(inclusive_child));
        for x in [-1.0, 0.0, 1.0] {
            for z in [-1.0, 0.0, 1.0] {
                for c in [-2.0, 0.0, 2.0] {
                    let nested =
                        linear_support(&model, &[(VarId(0), x), (VarId(1), z), (VarId(4), c)]);
                    let flat = linear_support(
                        &independent,
                        &[(VarId(0), x), (VarId(1), z), (VarId(2), c)],
                    );
                    assert!(
                        (nested - flat).abs() < 1e-7,
                        "direction [{x}, {z}, {c}]: nested={nested}, flat={flat}"
                    );
                }
            }
        }
    }
}

#[test]
fn paper_nested_relaxation_matches_flattened_hull_and_strengthens_single_level() {
    let nested = paper_model(false, false);
    let equivalent = paper_model(true, false);
    let flattened = paper_model(false, true);
    for model in [&nested, &equivalent, &flattened] {
        model.reformulate_gdp(Hull::default()).unwrap();
    }
    let mut strict = false;
    for a in [-2.0, -1.0, 0.0, 1.0, 2.0] {
        for b in [-2.0, -1.0, 0.0, 1.0, 2.0] {
            let n = support(&nested, [a, b]);
            let e = support(&equivalent, [a, b]);
            let f = support(&flattened, [a, b]);
            assert!((n - f).abs() < 1e-7, "direction [{a}, {b}]: nested {n}, flattened {f}");
            assert!(n <= e + 1e-7);
            strict |= e - n > 1e-5;
        }
    }
    assert!(strict);

    let result = Highs.with_gdp(Hull::default()).solve(&nested, &HighsOptions::default()).unwrap();

    assert_eq!(result.termination, TerminationStatus::Optimal);
    assert!((result.objective().unwrap() - 14.0).abs() < 1e-7);
}

#[test]
fn inclusive_mip_matches_all_selected_source_constraints() {
    let model = Model::new("inclusive Hull solve");

    variable!(model, 0.0 <= x <= 10.0);
    let a = disjunct!(model, a, |d| {
        constraint!(d, x <= 3.0);
    });
    let b = disjunct!(model, b, |d| {
        constraint!(d, x >= 2.0);
    });
    disjunction!(model, [a, b], AtLeastOne);
    logical_constraint!(model, logical_and([a, b]));
    objective!(model, Max, x);
    let hull =
        Hull::default().with_inclusive_policy(HullInclusivePolicy::Enumerate { max_patterns: 3 });

    let result = Highs.with_gdp(hull).solve(&model, &HighsOptions::default()).unwrap();

    assert!((result.value_of(x).unwrap().unwrap() - 3.0).abs() < 1e-7);
    assert_eq!(result.boolean_value_of(a).unwrap(), Some(true));
    assert_eq!(result.boolean_value_of(b).unwrap(), Some(true));
}

#[test]
fn highs_solves_mixed_nested_methods_with_active_and_inactive_parents() {
    for parent_hull in [false, true] {
        for inactive_parent in [false, true] {
            let model = Model::new("mixed nested MILP");

            variable!(model, 0.0 <= x <= 10.0);
            let parent = disjunct!(model, parent, |d| {
                constraint!(d, 2.0 <= x <= 8.0);
            });
            let other = disjunct!(model, other, |d| {
                constraint!(d, x <= 4.0);
            });
            let low = disjunct!(model, low, |d| {
                constraint!(d, x <= 3.0);
            });
            let high = disjunct!(model, high, |d| {
                constraint!(d, 5.0 <= x <= 6.0);
            });
            let outer = disjunction!(model, [parent, other]);
            let inner = disjunction!(model, [low, high], parent = parent);
            if inactive_parent {
                logical_constraint!(model, !parent);
            }
            objective!(model, Max, x);
            let (parent_method, child_method) = if parent_hull {
                (GdpMethod::from(Hull::default()), GdpMethod::from(BigM::default()))
            } else {
                (GdpMethod::from(BigM::default()), GdpMethod::from(Hull::default()))
            };

            let options = GdpReformulationOptions::new(parent_method)
                .with_method(inner, child_method)
                .with_hull_reference_coordinate(x, 0.5);

            let result = Highs.with_gdp(options).solve(&model, &HighsOptions::default()).unwrap();

            assert_eq!(result.termination, TerminationStatus::Optimal);
            let expected = if inactive_parent { 4.0 } else { 6.0 };

            assert!((result.objective().unwrap() - expected).abs() < 1e-7);
            assert!((result.value_of(x).unwrap().unwrap() - expected).abs() < 1e-7);
            assert_eq!(result.boolean_value_of(parent).unwrap(), Some(!inactive_parent));
            assert_eq!(result.boolean_value_of(other).unwrap(), Some(inactive_parent));
            assert_eq!(result.boolean_value_of(low).unwrap(), Some(false));
            assert_eq!(result.boolean_value_of(high).unwrap(), Some(!inactive_parent));
            assert!(!model.has_unreformulated_gdp());
            let reports = model.gdp_reformulations();
            for (source, expected_method) in [
                (outer.id(), if parent_hull { GdpMethodKind::Hull } else { GdpMethodKind::BigM }),
                (inner.id(), if parent_hull { GdpMethodKind::BigM } else { GdpMethodKind::Hull }),
            ] {
                assert_eq!(
                    reports[0].disjunctions.iter().find(|d| d.source == source).unwrap().method,
                    expected_method,
                );
            }
            #[cfg(feature = "io")]
            {
                let mut bytes = Vec::new();
                oximo::io::write_lp(&model, &mut bytes).unwrap();
                let imported = oximo::io::read_lp(std::io::Cursor::new(bytes)).unwrap();

                let result = Highs.solve(&imported, &HighsOptions::default()).unwrap();

                assert_eq!(result.termination, TerminationStatus::Optimal);
                assert!((result.objective().unwrap() - expected).abs() < 1e-7);
            }
        }
    }
}

#[cfg(feature = "io")]
#[test]
fn affine_hull_exports_as_an_lp() {
    let model = paper_model(false, false);
    model.reformulate_gdp(Hull::default()).unwrap();
    let mut bytes = Vec::new();
    oximo::io::write_lp(&model, &mut bytes).unwrap();

    assert_ne!(bytes, [] as [u8; 0]);
}

#[test]
fn inactive_three_level_group_remains_feasible_with_a_descendant_reference() {
    for reference in [0.0, 0.25] {
        let model = Model::new("inactive nested reference");

        variable!(model, 0.0 <= x <= 4.0);
        boolean_variable!(model, parent);
        boolean_variable!(model, other);
        boolean_variable!(model, child);
        boolean_variable!(model, sibling);
        boolean_variable!(model, grandchild);
        boolean_variable!(model, grand_sibling);
        disjunct_constraint!(model, child, x <= 1.0);
        disjunct_constraint!(model, grandchild, x >= 0.5);
        disjunction!(model, [parent, other]);
        disjunction!(model, [child, sibling], parent = parent, AtLeastOne);
        disjunction!(model, [grandchild, grand_sibling], parent = child);
        logical_constraint!(model, !parent);
        objective!(model, Max, x);

        let options = GdpReformulationOptions::new(
            Hull::default()
                .with_inclusive_policy(HullInclusivePolicy::Enumerate { max_patterns: 3 }),
        )
        .with_hull_reference_point(grandchild.binary(), reference);

        let result = Highs.with_gdp(options).solve(&model, &HighsOptions::default()).unwrap();

        assert_eq!(result.termination, TerminationStatus::Optimal);
        assert!((result.objective().unwrap() - 4.0).abs() < 1e-7);
        assert_eq!(result.boolean_value_of(parent).unwrap(), Some(false));
        assert_eq!(result.boolean_value_of(other).unwrap(), Some(true));
        assert_eq!(result.boolean_value_of(grandchild).unwrap(), Some(false));
    }
}

fn sparse_flat_model(full_copies: bool, selectors: Option<[f64; 5]>) -> Model {
    let model = Model::new("sparse branch support relaxation");

    variable!(model, -2.0 <= x <= 4.0);
    variable!(model, -3.0 <= y <= 5.0);
    variable!(model, 1.0 <= z <= 6.0);
    let branches: Vec<_> = (0..5).map(|i| model.add_boolean(format!("b{i}"))).collect();
    branches[0].context().add_constraint("xz", (x + z).le(3.0));
    branches[1].context().add_constraint("xy", (x - y).ge(1.0));
    branches[2].context().add_constraint("y", y.le(0.0));
    if full_copies {
        // Tautologies force an independent traditional copy in every branch,
        // but they do not change any source feasible set or its convex hull.
        for branch in &branches {
            branch.context().add_constraint("x box", x.ge(-2.0));
            branch.context().add_constraint("y box", y.le(5.0));
            branch.context().add_constraint("z box", z.le(6.0));
        }
    }
    disjunction!(model, branches.clone());

    let report = model
        .reformulate_gdp(
            GdpReformulationOptions::new(Hull::default()).with_hull_reference_coordinates([
                (x, 1.0),
                (y, -0.5),
                (z, 2.0),
            ]),
        )
        .unwrap();

    assert_eq!(report.hull_variables.len(), if full_copies { 15 } else { 8 });
    if let Some(selectors) = selectors {
        for (branch, value) in branches.iter().zip(selectors) {
            constraint!(model, branch.binary() == value);
        }
    }
    model
}

#[test]
fn combined_unused_copies_preserve_the_full_copy_projected_relaxation() {
    for selectors in [
        None,
        Some([0.1, 0.2, 0.3, 0.1, 0.3]),
        Some([0.5, 0.5, 0.0, 0.0, 0.0]),
        Some([0.0, 0.0, 0.0, 0.25, 0.75]),
        Some([1.0, 0.0, 0.0, 0.0, 0.0]),
    ] {
        let combined = sparse_flat_model(false, selectors);
        let traditional = sparse_flat_model(true, selectors);
        for x in [-1.0, 0.0, 1.0] {
            for y in [-1.0, 0.0, 1.0] {
                for z in [-1.0, 0.0, 1.0] {
                    let direction = [
                        (VarId(0), x),
                        (VarId(1), y),
                        (VarId(2), z),
                        (VarId(3), 0.7),
                        (VarId(6), -0.3),
                    ];
                    let actual = linear_support(&combined, &direction);
                    let expected = linear_support(&traditional, &direction);

                    assert!(
                        (actual - expected).abs() < 1e-7,
                        "selectors {selectors:?}, direction [{x}, {y}, {z}]: combined={actual}, traditional={expected}"
                    );
                }
            }
        }
    }
}
