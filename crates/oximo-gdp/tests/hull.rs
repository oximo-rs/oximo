#![allow(
    clippy::float_cmp,
    clippy::many_single_char_names,
    reason = "exact integer truth tables and mathematical branch labels"
)]
use oximo_expr::evaluate;
use oximo_gdp::prelude::*;

fn satisfied(model: &Model, values: &[f64]) -> bool {
    let arena = model.arena();
    model
        .resolved_variables()
        .iter()
        .all(|v| values[v.id.index()] >= v.lb - 1e-8 && values[v.id.index()] <= v.ub + 1e-8)
        && model.constraints().algebraic().iter().all(|r| {
            let value = evaluate(&arena, r.lhs, &values).unwrap();
            value.is_finite() && value >= r.lower - 1e-8 && value <= r.upper + 1e-8
        })
}

/// Construct the exact integer lift, independently of generated row bodies.
fn lift(model: &Model, report: &GdpReformulationReport, source: &[f64]) -> Vec<f64> {
    let mut values = source.to_vec();
    values.resize(model.num_variables(), 0.0);
    for pattern in &report.hull_patterns {
        let d = &model.gdp().disjunctions[pattern.source.index()];
        values[pattern.weight.index()] = f64::from(d.branches.iter().all(|b| {
            let active = source[model.gdp().booleans[b.index()].binary.index()] == 1.0;
            active == pattern.selected.contains(b)
        }));
    }
    for sum in &report.hull_branch_sums {
        values[sum.weight.index()] = 1.0
            - sum
                .excluded
                .iter()
                .map(|b| source[model.gdp().booleans[b.index()].binary.index()])
                .sum::<f64>();
    }
    let arena = model.arena();
    for copy in &report.hull_variables {
        let activation = evaluate(&arena, copy.activation, &&values[..]).unwrap();

        assert_eq!(
            copy.activation_kind.value_with(|v| values.get(v.index()).copied()),
            Some(activation)
        );
        values[copy.generated.index()] =
            activation * (values[copy.source.index()] - copy.reference);
    }
    values
}

#[test]
fn flat_hull_is_affine_and_preserves_integer_feasibility() {
    let model = Model::new("flat Hull");

    variable!(model, -5.0 <= x <= 10.0);
    let a = disjunct!(model, a, |d| {
        constraint!(d, 1.0 <= x <= 3.0);
    });
    let b = disjunct!(model, b, |d| {
        constraint!(d, x == -2.0);
    });
    disjunction!(model, [a, b]);

    let report = model.reformulate_gdp(Hull::default()).unwrap();

    assert_eq!(report.hull_variables.len(), 2);
    assert_eq!(report.disjunctions[0].method, GdpMethodKind::Hull);
    assert!(report.rows.iter().all(|r| r.lower_m.is_none() && r.upper_m.is_none()));
    assert!(
        model.constraints().algebraic().iter().all(|r| oximo_expr::extract_linear(
            &model.arena(),
            r.lhs
        )
        .is_some())
    );
    for bits in 0..4 {
        for value in [-5.0, -2.0, 0.0, 1.0, 2.0, 3.0, 10.0] {
            let av = f64::from(bits & 1 != 0);
            let bv = f64::from(bits & 2 != 0);
            let expected = av + bv == 1.0
                && (av == 0.0 || (1.0..=3.0).contains(&value))
                && (bv == 0.0 || value == -2.0);
            assert_eq!(satisfied(&model, &lift(&model, &report, &[value, av, bv])), expected);
        }
    }
}

#[test]
fn nested_and_mixed_hull_preserve_integer_feasibility_in_both_orders() {
    for inner_first in [false, true] {
        for parent_hull in [false, true] {
            for child_hull in [false, true] {
                let model = Model::new("nested mixed");

                variable!(model, -2.0 <= x <= 10.0);
                boolean_variable!(model, a);
                boolean_variable!(model, b);
                boolean_variable!(model, c);
                boolean_variable!(model, e);
                disjunct_constraint!(model, a, 1.0 <= x <= 6.0);
                disjunct_constraint!(model, b, x >= 8.0);
                disjunct_constraint!(model, c, x <= 3.0);
                disjunct_constraint!(model, e, x >= 5.0);
                let (outer, inner) = if inner_first {
                    let inner = disjunction!(model, [c, e], parent = a);
                    (disjunction!(model, [a, b]), inner)
                } else {
                    let outer = disjunction!(model, [a, b]);
                    (outer, disjunction!(model, [c, e], parent = a))
                };
                let method = |h| {
                    if h {
                        GdpMethod::from(Hull::default())
                    } else {
                        GdpMethod::from(BigM::default())
                    }
                };

                let report = model
                    .reformulate_gdp(
                        GdpReformulationOptions::new(method(parent_hull))
                            .with_method(inner, method(child_hull)),
                    )
                    .unwrap();

                assert_eq!(
                    report.disjunctions.iter().find(|d| d.source == outer.id()).unwrap().method,
                    if parent_hull { GdpMethodKind::Hull } else { GdpMethodKind::BigM }
                );
                for bits in 0..16 {
                    let av = f64::from(bits & 1 != 0);
                    let bv = f64::from(bits & 2 != 0);
                    let cv = f64::from(bits & 4 != 0);
                    let ev = f64::from(bits & 8 != 0);
                    for x in [-2.0, 0.0, 1.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0] {
                        let expected = av + bv == 1.0
                            && cv + ev == av
                            && (av == 0.0 || (1.0..=6.0).contains(&x))
                            && (bv == 0.0 || x >= 8.0)
                            && (cv == 0.0 || x <= 3.0)
                            && (ev == 0.0 || x >= 5.0);
                        assert_eq!(
                            satisfied(&model, &lift(&model, &report, &[x, av, bv, cv, ev])),
                            expected,
                            "inner_first={inner_first}, parent_hull={parent_hull}, child_hull={child_hull}, bits={bits}, x={x}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn paper_nested_links_use_parent_copies() {
    let model = Model::new("paper equation 18");

    variable!(model, 1.0 <= x <= 9.0);
    variable!(model, 1.0 <= z <= 6.0);
    let y1 = disjunct!(model, y1, |d| {
        constraint!(d, 1.0 <= x <= 3.0);
        constraint!(d, 4.0 <= z <= 6.0);
    });
    let y2 = disjunct!(model, y2, |d| {
        constraint!(d, 8.0 <= x <= 9.0);
        constraint!(d, 1.0 <= z <= 2.0);
    });
    let w1 = disjunct!(model, w1, |d| {
        constraint!(d, 1.0 <= x <= 2.0);
        constraint!(d, 5.0 <= z <= 6.0);
    });
    let w2 = disjunct!(model, w2, |d| {
        constraint!(d, 2.0 <= x <= 3.0);
        constraint!(d, 4.0 <= z <= 5.0);
    });
    disjunction!(model, [y1, y2]);
    let child = disjunction!(model, [w1, w2], parent = y1);

    let report = model.reformulate_gdp(Hull::default()).unwrap();

    assert_eq!(report.hull_variables.len(), 8);

    let parent = report
        .hull_variables
        .iter()
        .find(|c| c.source == x.var_id().unwrap() && c.indicator == Some(y1.id()))
        .unwrap()
        .generated;
    let child_vars: Vec<_> = report
        .hull_variables
        .iter()
        .filter(|c| c.source == x.var_id().unwrap() && c.disjunction == Some(child.id()))
        .map(|c| c.generated)
        .collect();
    let rows = model.constraints();
    let arena = model.arena();

    assert!(rows.algebraic().iter().any(|r| {
        if r.lower != 0.0 || r.upper != 0.0 {
            return false;
        }
        let Some(terms) = oximo_expr::extract_linear(&arena, r.lhs) else {
            return false;
        };
        terms.coeffs.contains(&(parent, -1.0))
            && child_vars.iter().all(|v| terms.coeffs.contains(&(*v, 1.0)))
            && !terms.coeffs.iter().any(|(v, _)| *v == x.var_id().unwrap())
    }));
}

#[test]
fn nonlinear_perspectives_and_explicit_references_are_exact_at_integer_points() {
    for shifted in [false, true] {
        let model = Model::new("nonlinear Hull");

        variable!(model, 1.0 <= x <= 4.0);
        boolean_variable!(model, a);
        if shifted {
            disjunct_constraint!(model, a, x.ln() <= 2.0_f64.ln());
        } else {
            disjunct_constraint!(model, a, x.exp() + x * x <= 3.0_f64.exp() + 9.0);
        }
        let mut options = GdpReformulationOptions::new(Hull::default());
        if shifted {
            options = options.with_hull_reference_coordinate(x, 1.0);
        }

        let report = model.reformulate_gdp(options).unwrap();
        for active in [0.0, 1.0] {
            for value in [1.0, 2.0, 3.0, 4.0] {
                assert_eq!(
                    satisfied(&model, &lift(&model, &report, &[value, active])),
                    active == 0.0 || value <= if shifted { 2.0 } else { 3.0 }
                );
            }
        }
        assert!(
            report.hull_variables.iter().all(|c| c.reference == if shifted { 1.0 } else { 0.0 })
        );
    }
}

#[test]
fn inclusive_enumeration_preserves_simultaneous_selection() {
    let model = Model::new("inclusive patterns");

    variable!(model, -2.0 <= x <= 10.0);
    let a = disjunct!(model, a, |d| {
        constraint!(d, x >= 1.0);
    });
    let b = disjunct!(model, b, |d| {
        constraint!(d, x <= 3.0);
    });
    let c = disjunct!(model, c, |d| {
        constraint!(d, x == 2.0);
    });
    disjunction!(model, [a, b, c], AtLeastOne);

    let report = model
        .reformulate_gdp(
            Hull::default()
                .with_inclusive_policy(HullInclusivePolicy::Enumerate { max_patterns: 7 }),
        )
        .unwrap();

    assert_eq!(report.hull_patterns.len(), 7);
    for bits in 0..8 {
        let av = f64::from(bits & 1 != 0);
        let bv = f64::from(bits & 2 != 0);
        let cv = f64::from(bits & 4 != 0);
        for value in [-2.0, 0.0, 1.0, 2.0, 3.0, 10.0] {
            let expected = bits != 0
                && (av == 0.0 || value >= 1.0)
                && (bv == 0.0 || value <= 3.0)
                && (cv == 0.0 || value == 2.0);
            assert_eq!(satisfied(&model, &lift(&model, &report, &[value, av, bv, cv])), expected);
        }
    }
}

#[test]
fn hull_validation_is_atomic_in_place_and_on_copy() {
    for case in 0..5 {
        let model = Model::new("invalid Hull");

        let x = if case == 0 {
            model.__var("x").build()
        } else {
            model.__var("x").bounds(1.0, 3.0).build()
        };
        boolean_variable!(model, a);
        boolean_variable!(model, b);
        if case == 1 {
            disjunct_constraint!(model, a, x.ln() <= 1.0);
        } else {
            disjunct_constraint!(model, a, x <= 2.0);
        }
        if case >= 3 {
            disjunction!(model, [a, b], AtLeastOne);
        } else {
            disjunction!(model, [a, b]);
        }
        let method = match case {
            2 => Hull::default().with_epsilon(f64::NAN),
            4 => Hull::default()
                .with_inclusive_policy(HullInclusivePolicy::Enumerate { max_patterns: 2 }),
            _ => Hull::default(),
        };

        let before = (
            model.num_variables(),
            model.num_constraints(),
            model.arena().len(),
            model.to_string(),
        );
        assert!(model.reformulate_gdp(method).is_err());
        assert!(model.to_reformulated_gdp_model(method).is_err());
        assert_eq!(
            before,
            (
                model.num_variables(),
                model.num_constraints(),
                model.arena().len(),
                model.to_string()
            )
        );
        assert!(model.gdp_reformulations().is_empty());
    }
}

#[test]
fn inclusive_ancestor_recombines_nested_selectors_and_copies() {
    for nonlinear in [false, true] {
        for outer_hull in [false, true] {
            let model = Model::new("nested inclusive");

            variable!(model, 0.0 <= x <= 6.0);
            boolean_variable!(model, a);
            boolean_variable!(model, b);
            boolean_variable!(model, c);
            boolean_variable!(model, d);
            boolean_variable!(model, e);
            boolean_variable!(model, f);
            disjunct_constraint!(model, a, x >= 1.0);
            disjunct_constraint!(model, b, x <= 5.0);
            if nonlinear {
                disjunct_constraint!(model, c, x.exp() <= 3.0_f64.exp());
            } else {
                disjunct_constraint!(model, c, x <= 3.0);
            }
            disjunct_constraint!(model, d, x >= 2.0);
            disjunct_constraint!(model, e, x <= 4.0);
            disjunct_constraint!(model, f, x >= 3.0);
            let outer = disjunction!(model, [a, b], AtLeastOne);
            disjunction!(model, [c, d], parent = a);
            disjunction!(model, [e, f], parent = b, AtLeastOne);
            let hull = Hull::default()
                .with_inclusive_policy(HullInclusivePolicy::Enumerate { max_patterns: 3 });

            let options = GdpReformulationOptions::new(hull).with_method(
                outer,
                if outer_hull { GdpMethod::from(hull) } else { GdpMethod::from(BigM::default()) },
            );

            let report = model.reformulate_gdp(options).unwrap();
            for bits in 0..64 {
                let flags: Vec<_> = (0..6).map(|i| f64::from(bits & (1 << i) != 0)).collect();
                let (av, bv, cv, dv, ev, fv) =
                    (flags[0], flags[1], flags[2], flags[3], flags[4], flags[5]);
                for x in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0] {
                    let expected = av + bv >= 1.0
                        && cv + dv == av
                        && ev + fv >= bv
                        && ev <= bv
                        && fv <= bv
                        && (av == 0.0 || x >= 1.0)
                        && (bv == 0.0 || x <= 5.0)
                        && (cv == 0.0 || x <= 3.0)
                        && (dv == 0.0 || x >= 2.0)
                        && (ev == 0.0 || x <= 4.0)
                        && (fv == 0.0 || x >= 3.0);
                    let mut source = vec![x];
                    source.extend(&flags);
                    assert_eq!(
                        satisfied(&model, &lift(&model, &report, &source)),
                        expected,
                        "outer_hull={outer_hull}, nonlinear={nonlinear}, bits={bits}, x={x}"
                    );
                }
            }
        }
    }
}

#[test]
fn nonlinear_fractional_perspective_matches_formula_and_ranges() {
    let model = Model::new("fractional perspective");

    variable!(model, -2.0 <= x <= 2.0);
    boolean_variable!(model, a);
    disjunct_constraint!(model, a, 0.5 <= x.exp() + 3.0 * x <= 2.0);

    let epsilon = 0.1;

    let report = model.reformulate_gdp(Hull::default().with_epsilon(epsilon)).unwrap();
    let active_copy =
        report.hull_variables.iter().find(|c| c.indicator == Some(a.id())).unwrap().generated;
    let mut values = vec![0.0; model.num_variables()];
    values[a.binary().var_id().unwrap().index()] = 0.4;
    values[active_copy.index()] = 0.2;
    let s = epsilon + (1.0 - epsilon) * 0.4;
    let expected = s * (0.2_f64 / s).exp() - epsilon * 0.6 + 3.0 * 0.2;
    let rows = model.constraints();
    let arena = model.arena();
    for (&id, bound) in report.rows[0].constraints.iter().zip([0.5, 2.0]) {
        let value = evaluate(&arena, rows.algebraic()[id.index()].lhs, &&values[..]).unwrap();

        assert!((value - (expected - 0.4 * bound)).abs() < 1e-10);
    }
}

#[test]
fn copy_lifecycle_and_unconditional_tightening_remain_atomic() {
    let model = Model::new("copy and tight bounds");

    variable!(model, x);
    constraint!(model, -2.0 <= x <= 5.0);
    boolean_variable!(model, a);
    disjunct_constraint!(model, a, x <= 1.0);

    let before = (model.num_variables(), model.num_constraints(), model.arena().len());
    let copy = model.to_reformulated_gdp_model(Hull::default()).unwrap();

    assert_eq!(before, (model.num_variables(), model.num_constraints(), model.arena().len()));
    assert!(model.has_unreformulated_gdp());
    assert_eq!(copy.gdp_reformulations()[0].hull_variables.len(), 2);

    let report = model.reformulate_gdp(Hull::default()).unwrap();

    assert!(report.hull_variables.iter().all(|v| v.bounds[0] <= -2.0 && v.bounds[1] >= 5.0));

    let after = (model.num_variables(), model.num_constraints(), model.arena().len());
    assert_eq!(model.reformulate_gdp(Hull::default()).unwrap().variables, [] as [VarId; 0]);
    assert_eq!(after, (model.num_variables(), model.num_constraints(), model.arena().len()));
    boolean_variable!(model, fresh);
    logical_constraint!(model, fresh);

    model.reformulate_gdp(Hull::default()).unwrap();

    assert_eq!(model.gdp_reformulations().len(), 2);
}

#[test]
fn deep_exclusive_trees_generate_linear_size_on_a_small_stack() {
    std::thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(|| {
            let model = Model::new("deep Hull");

            variable!(model, 0.0 <= x <= 10.0);
            let mut parent = None;
            for level in 0..500 {
                let a = model.add_boolean(format!("a{level}"));
                let b = model.add_boolean(format!("b{level}"));
                a.context().add_constraint("limit", x.le(5.0));
                if let Some(p) = parent {
                    disjunction!(model, [a, b], parent = p);
                } else {
                    disjunction!(model, [a, b]);
                }
                parent = Some(a);
            }

            let report = model.reformulate_gdp(Hull::default()).unwrap();

            assert_eq!(report.hull_variables.len(), 1000);
            assert_eq!(model.num_constraints(), 3500);
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn numeric_selectors_preserve_own_ancestor_sibling_and_unrelated_values() {
    let model = Model::new("numeric selectors");

    let x = model.__var("x").domain(Domain::Integer).bounds(-1.0, 2.0).build();
    boolean_variable!(model, a);
    boolean_variable!(model, b);
    boolean_variable!(model, c);
    boolean_variable!(model, d);
    boolean_variable!(model, unrelated);
    let body = x.exp()
        + 2.0 * c.binary()
        + 3.0 * a.binary()
        + unrelated.binary().exp()
        + d.binary() * d.binary();
    disjunct_constraint!(model, c, body <= 9.0);
    disjunction!(model, [a, b]);
    disjunction!(model, [c, d], parent = a);

    let report = model
        .reformulate_gdp(
            GdpReformulationOptions::new(Hull::default())
                .with_hull_reference_coordinate(d.binary(), 0.25)
                .with_hull_reference_coordinate(unrelated.binary(), 0.5),
        )
        .unwrap();

    assert_eq!(model.resolved_variables()[0].domain, Domain::Integer);
    assert!(
        report
            .variables
            .iter()
            .all(|v| model.resolved_variables()[v.index()].domain == Domain::Real)
    );

    for bits in 0..32 {
        let a = f64::from(bits & 1 != 0);
        let b = f64::from(bits & 2 != 0);
        let c = f64::from(bits & 4 != 0);
        let d = f64::from(bits & 8 != 0);
        let unrelated = f64::from(bits & 16 != 0);
        for x in [-1.0_f64, 0.0, 1.0, 2.0] {
            let expected = a + b == 1.0
                && c + d == a
                && (c == 0.0 || x.exp() + 2.0 * c + 3.0 * a + unrelated.exp() + d * d <= 9.0);
            assert_eq!(
                satisfied(&model, &lift(&model, &report, &[x, a, b, c, d, unrelated])),
                expected,
                "bits={bits}, x={x}"
            );
        }
    }
}

#[test]
fn shifted_negative_nonlinear_equalities_are_exact_at_integer_points() {
    let model = Model::new("negative equality");

    variable!(model, -2.0 <= x <= 4.0);
    boolean_variable!(model, active);
    disjunct_constraint!(model, active, 2.0 * x - x * x == -3.0);

    let report = model
        .reformulate_gdp(
            GdpReformulationOptions::new(Hull::default()).with_hull_reference_coordinate(x, -1.0),
        )
        .unwrap();

    assert_eq!(report.rows[0].constraints.len(), 1);
    for active in [0.0, 1.0] {
        for x in [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0] {
            assert_eq!(
                satisfied(&model, &lift(&model, &report, &[x, active])),
                active == 0.0 || x == -1.0 || x == 3.0
            );
        }
    }
}

#[test]
fn independent_big_m_trees_retain_hierarchical_estimates_beside_hull() {
    let model = Model::new("mixed forest");

    variable!(model, 0.0 <= x <= 10.0);
    boolean_variable!(model, parent);
    boolean_variable!(model, other);
    boolean_variable!(model, child);
    boolean_variable!(model, sibling);
    disjunct_constraint!(model, parent, x <= 4.0);
    disjunct_constraint!(model, child, x <= 1.0);
    disjunct_constraint!(model, sibling, x >= 9.0);
    let outer = disjunction!(model, [parent, other]);
    let inner = disjunction!(model, [child, sibling], parent = parent);
    let left = disjunct!(model, left, |d| {
        constraint!(d, x <= 7.0);
    });
    let right = disjunct!(model, right, |d| {
        constraint!(d, x >= 3.0);
    });
    disjunction!(model, [left, right]);

    let report = model
        .reformulate_gdp(
            GdpReformulationOptions::new(Hull::default())
                .with_method(outer, BigM::default())
                .with_method(inner, BigM::default()),
        )
        .unwrap();
    let terms = &report.rows[1].hierarchical_m;

    assert_eq!(terms.len(), 2);
    assert_eq!(terms[0].indicator, child.id());
    assert_eq!(terms[1].indicator, parent.id());
    assert!((terms[0].upper.unwrap() - 3.0).abs() < 1e-8);
    assert!((terms[1].upper.unwrap() - 6.0).abs() < 1e-8);
    assert_eq!(report.disjunctions.len(), 3);
}

#[test]
fn nonlinear_parameter_snapshots_and_clone_locks_are_isolated() {
    let model = Model::new("Hull parameter snapshot");

    variable!(model, -1.0 <= x <= 2.0);
    param!(model, coefficient = 2.0);
    param!(model, unrelated = 1.0);
    boolean_variable!(model, active);
    disjunct_constraint!(model, active, (coefficient * x).exp() <= 1.0);

    let transformed = model.to_reformulated_gdp_model(Hull::default()).unwrap();
    let id = coefficient.param_id().unwrap();
    coefficient.set_param_value(3.0).unwrap();

    assert_eq!(model.param_value(id), 3.0);
    assert_eq!(transformed.param_value(id), 2.0);
    assert!(transformed.set_param_id(id, 4.0).is_err());
    assert!(transformed.fix_var(VarId(0), 0.0).is_err());

    let generated = transformed.gdp_reformulations()[0].hull_variables[0].generated;
    assert!(transformed.fix_var(generated, 0.0).is_err());

    let clone = transformed.to_reformulated_gdp_model(Hull::default()).unwrap();

    assert!(clone.set_param_id(id, 4.0).is_err());

    clone.set_param_id(unrelated.param_id().unwrap(), 5.0).unwrap();
    assert_eq!(transformed.param_value(unrelated.param_id().unwrap()), 1.0);

    let report = &transformed.gdp_reformulations()[0];
    for active in [0.0, 1.0] {
        for x in [-1.0, 0.0, 1.0, 2.0] {
            assert_eq!(
                satisfied(&transformed, &lift(&transformed, report, &[x, active])),
                active == 0.0 || x <= 0.0
            );
        }
    }
}

#[test]
fn invalid_reference_handles_and_coordinates_do_not_mutate_the_model() {
    let model = Model::new("invalid references");

    let foreign = Model::new("foreign");

    variable!(model, -1.0 <= x <= 2.0);
    variable!(foreign, z);
    boolean_variable!(model, active);
    disjunct_constraint!(model, active, x.exp() <= 1.0);
    let compound = x + 1.0;

    let before = (model.num_variables(), model.num_constraints(), model.arena().len());

    for options in [
        GdpReformulationOptions::new(Hull::default()).with_hull_reference_coordinate(z, 0.0),
        GdpReformulationOptions::new(Hull::default()).with_hull_reference_coordinate(compound, 0.0),
        GdpReformulationOptions::new(Hull::default())
            .with_hull_reference_coordinate(x, f64::INFINITY),
        GdpReformulationOptions::new(Hull::default()).with_hull_reference_coordinate(x, 1e20),
    ] {
        assert!(model.reformulate_gdp(options.clone()).is_err());
        assert!(model.to_reformulated_gdp_model(options).is_err());
        assert_eq!(before, (model.num_variables(), model.num_constraints(), model.arena().len()));
        assert!(model.gdp_reformulations().is_empty());
    }
}

#[test]
fn standalone_parents_leave_physical_variables_free_when_inactive() {
    let model = Model::new("standalone nested on/off Hull");

    variable!(model, 1.0 <= x <= 4.0);
    boolean_variable!(model, parent);
    boolean_variable!(model, child);
    boolean_variable!(model, sibling);
    disjunct_constraint!(model, child, x.ln() <= 2.0_f64.ln());
    disjunct_constraint!(model, sibling, x >= 3.0);
    disjunction!(model, [child, sibling], parent = parent);

    let report = model
        .reformulate_gdp(
            GdpReformulationOptions::new(Hull::default()).with_hull_reference_coordinate(x, 1.0),
        )
        .unwrap();
    for bits in 0..8 {
        let parent = f64::from(bits & 1 != 0);
        let child = f64::from(bits & 2 != 0);
        let sibling = f64::from(bits & 4 != 0);
        for x in [1.0, 2.0, 2.5, 3.0, 4.0] {
            let expected = child + sibling == parent
                && (child == 0.0 || x <= 2.0)
                && (sibling == 0.0 || x >= 3.0);
            assert_eq!(
                satisfied(&model, &lift(&model, &report, &[x, parent, child, sibling])),
                expected
            );
        }
    }
}

#[test]
fn enumeration_uses_zero_references_for_local_descendants() {
    let model = Model::new("local references in three levels");

    variable!(model, 0.0 <= x <= 8.0);
    boolean_variable!(model, parent);
    boolean_variable!(model, other);
    boolean_variable!(model, child);
    boolean_variable!(model, sibling);
    boolean_variable!(model, grandchild);
    boolean_variable!(model, grand_sibling);
    disjunct_constraint!(model, child, (x + grandchild.binary()).exp() <= 4.0_f64.exp());
    disjunct_constraint!(model, sibling, x <= 2.0);
    disjunct_constraint!(model, grandchild, x >= 3.0);
    disjunction!(model, [parent, other]);
    disjunction!(model, [child, sibling], parent = parent, AtLeastOne);
    disjunction!(model, [grandchild, grand_sibling], parent = child);

    let options = GdpReformulationOptions::new(
        Hull::default().with_inclusive_policy(HullInclusivePolicy::Enumerate { max_patterns: 3 }),
    )
    .with_hull_reference_coordinate(grandchild.binary(), 0.25)
    .with_hull_reference_coordinate(x, 1.0);

    let report = model.reformulate_gdp(options).unwrap();
    let selector = grandchild.binary().var_id().unwrap();

    assert!(report.hull_variables.iter().any(|v| v.source == selector));
    assert!(
        report.hull_variables.iter().filter(|v| v.source == selector).all(|v| v.reference == 0.0)
    );
    assert!(
        report.hull_variables.iter().filter(|v| v.source == VarId(0)).all(|v| v.reference == 1.0)
    );

    for bits in 0..64 {
        let parent = f64::from(bits & 1 != 0);
        let other = f64::from(bits & 2 != 0);
        let child = f64::from(bits & 4 != 0);
        let sibling = f64::from(bits & 8 != 0);
        let grandchild = f64::from(bits & 16 != 0);
        let grand_sibling = f64::from(bits & 32 != 0);
        for x in [0.0, 1.0, 2.0, 3.0, 4.0, 8.0] {
            let expected = parent + other == 1.0
                && child <= parent
                && sibling <= parent
                && child + sibling >= parent
                && grandchild + grand_sibling == child
                && (child == 0.0 || x + grandchild <= 4.0)
                && (sibling == 0.0 || x <= 2.0)
                && (grandchild == 0.0 || x >= 3.0);
            assert_eq!(
                satisfied(
                    &model,
                    &lift(
                        &model,
                        &report,
                        &[x, parent, other, child, sibling, grandchild, grand_sibling]
                    )
                ),
                expected,
                "bits={bits}, x={x}"
            );
        }
    }
}

#[test]
fn reference_shifts_must_preserve_small_active_row_bounds_atomically() {
    for side in [BoundSide::Lower, BoundSide::Upper] {
        let model = Model::new("row reference cancellation");

        variable!(model, -1e16 <= x <= 1e16);
        boolean_variable!(model, active);
        if side == BoundSide::Lower {
            disjunct_constraint!(model, active, x >= 1.0);
        } else {
            disjunct_constraint!(model, active, x <= -1.0);
        }
        let reference = if side == BoundSide::Lower { 1e16 } else { -1e16 };

        let options = GdpReformulationOptions::new(Hull::default())
            .with_hull_reference_coordinate(x, reference);

        let before = (
            model.num_variables(),
            model.num_constraints(),
            model.arena().len(),
            model.to_string(),
        );
        for result in [
            model.reformulate_gdp(options.clone()).map(|_| ()),
            model.to_reformulated_gdp_model(options).map(|_| ()),
        ] {
            assert!(
                matches!(result, Err(GdpError::HullRowPrecisionLoss { side: actual, .. }) if actual == side)
            );
            assert_eq!(
                before,
                (
                    model.num_variables(),
                    model.num_constraints(),
                    model.arena().len(),
                    model.to_string()
                )
            );
            assert!(model.gdp_reformulations().is_empty());
        }
    }
}

#[test]
fn rounded_reference_products_are_rejected_atomically_on_both_sides_and_equalities() {
    for (coefficient, reference) in [
        (1e16, 1.0 + f64::EPSILON),
        (-1e16, 1.0 + f64::EPSILON),
        (1e16, -(1.0 + f64::EPSILON)),
        (-1e16, -(1.0 + f64::EPSILON)),
    ] {
        for scope in 0..3 {
            for kind in 0..3 {
                let model = Model::new("rounded reference product");

                variable!(model, -2.0 <= x <= 2.0);
                boolean_variable!(model, active);
                let side = match kind {
                    0 => {
                        disjunct_constraint!(model, active, capacity, coefficient * x <= 2.0);
                        BoundSide::Upper
                    }
                    1 => {
                        disjunct_constraint!(model, active, capacity, coefficient * x >= 2.0);
                        BoundSide::Lower
                    }
                    _ => {
                        disjunct_constraint!(model, active, capacity, coefficient * x == 2.0);
                        BoundSide::Lower
                    }
                };
                match scope {
                    1 => {
                        boolean_variable!(model, other);
                        disjunction!(model, [active, other]);
                    }
                    2 => {
                        boolean_variable!(model, other);
                        disjunction!(model, [active, other], AtLeastOne);
                    }
                    _ => {}
                }
                let hull = Hull::default()
                    .with_inclusive_policy(HullInclusivePolicy::Enumerate { max_patterns: 3 });

                let options =
                    GdpReformulationOptions::new(hull).with_hull_reference_coordinate(x, reference);

                let before = (
                    model.num_variables(),
                    model.num_constraints(),
                    model.arena().len(),
                    model.to_string(),
                );
                for result in [
                    model.plan_gdp(options.clone()).map(|_| ()),
                    model.reformulate_gdp(options.clone()).map(|_| ()),
                    model.to_reformulated_gdp_model(options).map(|_| ()),
                ] {
                    let Err(GdpError::HullRowPrecisionLoss {
                        constraint,
                        side: actual,
                        error,
                        tolerance,
                    }) = result
                    else {
                        panic!(
                            "expected reference product precision loss: {result:?}; coefficient={coefficient}, reference={reference}, scope={scope}, kind={kind}"
                        );
                    };
                    assert!(constraint.contains("capacity"));
                    assert_eq!(actual, side);
                    assert!((error - (1e16 * f64::EPSILON - 2.0)).abs() < 1e-12);
                    assert_eq!(tolerance, 2e-9);
                    assert_eq!(
                        before,
                        (
                            model.num_variables(),
                            model.num_constraints(),
                            model.arena().len(),
                            model.to_string(),
                        )
                    );
                    assert!(model.gdp_reformulations().is_empty());
                    assert!(model.has_unreformulated_gdp());
                }
                // An exactly representable product should still reformulate successfully.
                model
                    .reformulate_gdp(
                        GdpReformulationOptions::new(hull)
                            .with_hull_reference_coordinate(x, reference.signum()),
                    )
                    .unwrap();
            }
        }
    }
}

#[test]
fn row_recovery_checks_equalities_ranges_constants_and_negative_coefficients() {
    for kind in 0..4 {
        let model = Model::new("reference recovery variants");

        variable!(model, -1e16 <= x <= 1e16);
        boolean_variable!(model, active);
        let side = match kind {
            0 => {
                disjunct_constraint!(model, active, x == 1.0);
                BoundSide::Lower
            }
            1 => {
                disjunct_constraint!(model, active, 0.5 <= x <= 1.0);
                BoundSide::Lower
            }
            2 => {
                disjunct_constraint!(model, active, -x <= -1.0);
                BoundSide::Upper
            }
            _ => {
                disjunct_constraint!(model, active, x + 0.25 >= 1.25);
                BoundSide::Lower
            }
        };

        let options =
            GdpReformulationOptions::new(Hull::default()).with_hull_reference_coordinate(x, 1e16);

        let before = (model.num_variables(), model.num_constraints(), model.arena().len());
        assert!(matches!(model.reformulate_gdp(options),
            Err(GdpError::HullRowPrecisionLoss { side: actual, .. }) if actual == side));
        assert_eq!(before, (model.num_variables(), model.num_constraints(), model.arena().len()));
        assert!(model.gdp_reformulations().is_empty());
    }
}

#[test]
fn row_recovery_rejects_lost_nonlinear_corrections_and_accepts_a_safe_shift() {
    let model = Model::new("large nonlinear perspective correction");

    variable!(model, -100.0 <= x <= 0.0);
    boolean_variable!(model, active);
    disjunct_constraint!(model, active, 1e22 * x.exp() >= 1.0);

    let before = (model.num_variables(), model.num_constraints(), model.arena().len());
    assert!(matches!(
        model.reformulate_gdp(Hull::default()),
        Err(GdpError::HullRowPrecisionLoss { side: BoundSide::Lower, .. })
    ));
    assert_eq!(before, (model.num_variables(), model.num_constraints(), model.arena().len()));

    let report = model
        .reformulate_gdp(
            GdpReformulationOptions::new(Hull::default()).with_hull_reference_coordinate(x, -50.0),
        )
        .unwrap();
    for active in [0.0, 1.0] {
        for x in [-100.0_f64, -60.0, -50.0, -40.0, 0.0] {
            assert_eq!(
                satisfied(&model, &lift(&model, &report, &[x, active])),
                active == 0.0 || 1e22 * x.exp() >= 1.0
            );
        }
    }
}

#[test]
fn row_recovery_preserves_representable_and_cancelling_reference_offsets() {
    let model = Model::new("representable bound");

    variable!(model, -1e16 <= x <= 1e16);
    boolean_variable!(model, active);
    disjunct_constraint!(model, active, x >= 2.0);

    let report = model
        .reformulate_gdp(
            GdpReformulationOptions::new(Hull::default()).with_hull_reference_coordinate(x, 1e16),
        )
        .unwrap();
    for active in [0.0, 1.0] {
        for x in [-2.0, 0.0, 2.0, 4.0] {
            assert_eq!(
                satisfied(&model, &lift(&model, &report, &[x, active])),
                active == 0.0 || x >= 2.0
            );
        }
    }
    // Large offsets cancel exactly.
    let model = Model::new("cancelling offsets");

    variable!(model, -1e16 <= x <= 1e16);
    variable!(model, -2.0 <= z <= 2.0);
    boolean_variable!(model, active);
    disjunct_constraint!(model, active, x + z >= 1.0);
    model
        .reformulate_gdp(
            GdpReformulationOptions::new(Hull::default())
                .with_hull_reference_coordinate(x, 1e16)
                .with_hull_reference_coordinate(z, 1.0),
        )
        .unwrap();

    let model = Model::new("fractional offset");

    variable!(model, -1.0 <= x <= 2.0);
    boolean_variable!(model, active);
    disjunct_constraint!(model, active, 0.3 * x == 0.0);

    let report = model
        .reformulate_gdp(
            GdpReformulationOptions::new(Hull::default()).with_hull_reference_coordinate(x, 0.1),
        )
        .unwrap();

    assert!(satisfied(&model, &lift(&model, &report, &[0.0, 1.0])));
    assert!(!satisfied(&model, &lift(&model, &report, &[1.0, 1.0])));
}

#[test]
fn nonlinear_local_rows_survive_exclusive_and_inclusive_ancestors_with_shifted_coordinates() {
    let model = Model::new("typed local row scopes");

    variable!(model, -1.0 <= x <= 3.0);
    boolean_variable!(model, grandparent);
    boolean_variable!(model, other);
    boolean_variable!(model, parent);
    boolean_variable!(model, sibling);
    boolean_variable!(model, child);
    boolean_variable!(model, grand_sibling);
    disjunction!(model, [grandparent, other]);
    disjunction!(model, [parent, sibling], parent = grandparent, AtLeastOne);
    disjunction!(model, [child, grand_sibling], parent = parent);
    disjunct_constraint!(model, parent, x >= -0.5);
    disjunct_constraint!(model, child, x * x <= 1.0);
    disjunct_constraint!(model, grand_sibling, x >= 1.0);
    disjunct_constraint!(model, sibling, x >= 2.0);
    let hull = Hull::default()
        .with_epsilon(0.1)
        .with_inclusive_policy(HullInclusivePolicy::Enumerate { max_patterns: 3 });

    let report = model
        .reformulate_gdp(GdpReformulationOptions::new(hull).with_hull_reference_coordinate(x, 0.25))
        .unwrap();
    for bits in 0..64 {
        let decisions: Vec<_> = (0..6).map(|i| f64::from(bits & (1 << i) != 0)).collect();
        let [grandparent, other, parent, sibling, child, grand_sibling] = decisions[..] else {
            unreachable!()
        };
        for x_value in [-1.0, 0.0, 0.5, 1.0, 2.0, 3.0] {
            let expected = grandparent + other == 1.0
                && parent <= grandparent
                && sibling <= grandparent
                && parent + sibling >= grandparent
                && child + grand_sibling == parent
                && (parent == 0.0 || x_value >= -0.5)
                && (child == 0.0 || x_value * x_value <= 1.0)
                && (grand_sibling == 0.0 || x_value >= 1.0)
                && (sibling == 0.0 || x_value >= 2.0);
            let source: Vec<_> =
                std::iter::once(x_value).chain(decisions.iter().copied()).collect();
            assert_eq!(
                satisfied(&model, &lift(&model, &report, &source)),
                expected,
                "bits={bits}, x={x_value}"
            );
        }
    }
}

#[test]
fn wide_flat_group_combines_unused_copies_and_has_no_affine_rewrite_nodes() {
    let model = Model::new("wide sparse support");

    variable!(model, -2.0 <= x <= 4.0);
    let branches: Vec<_> = (0..1000).map(|i| model.add_boolean(format!("b{i}"))).collect();
    branches[0].context().add_constraint("limit", x.le(2.0));
    disjunction!(model, branches.clone());
    let before_nodes = model.arena().len();

    let report = model
        .reformulate_gdp(
            GdpReformulationOptions::new(Hull::default()).with_hull_reference_coordinate(x, 1.0),
        )
        .unwrap();

    assert_eq!(report.hull_variables.len(), 2);
    assert_eq!(model.num_constraints(), 7);

    // Only the two shared report activation expressions enter the private arena.
    assert_eq!(report.planning.private_expression_nodes, before_nodes + 2);
    assert!(report.hull_patterns.is_empty());
    for selected in [0, 1, 999] {
        for value in [-2.0, 1.0, 2.0, 3.0, 4.0] {
            let mut source = vec![0.0; 1001];
            source[0] = value;
            source[branches[selected].binary().var_id().unwrap().index()] = 1.0;
            assert_eq!(
                satisfied(&model, &lift(&model, &report, &source)),
                selected != 0 || value <= 2.0
            );
        }
    }
}

#[test]
fn unused_activation_weights_are_shared_by_coordinates_with_the_same_support() {
    let model = Model::new("shared unused activation");

    variable!(model, -2.0 <= x <= 4.0);
    variable!(model, -3.0 <= y <= 5.0);
    let branches: Vec<_> = (0..5).map(|i| model.add_boolean(format!("b{i}"))).collect();
    branches[0].context().add_constraint("sum", (x + y).le(3.0));
    branches[1].context().add_constraint("difference", (x - y).ge(1.0));
    disjunction!(model, branches.clone());

    let report = model
        .reformulate_gdp(
            GdpReformulationOptions::new(Hull::default())
                .with_hull_reference_coordinates([(x, 1.0), (y, -0.5)]),
        )
        .unwrap();

    assert_eq!(report.hull_variables.len(), 6);
    assert_eq!(report.hull_branch_sums.len(), 1);
    for copy in &report.hull_variables {
        for other in &report.hull_variables {
            if copy.activation_kind == other.activation_kind {
                assert_eq!(copy.activation, other.activation);
            }
        }
    }
    let sum = &report.hull_branch_sums[0];
    assert_eq!(sum.excluded, [branches[0].id(), branches[1].id()]);
    assert_eq!(
        report
            .hull_variables
            .iter()
            .filter(|c| matches!(c.activation_kind,
            GdpHullActivation::BranchSum { weight, .. } if weight == sum.weight))
            .count(),
        2
    );
    for selected in 0..5 {
        for xv in [-2.0, 0.0, 4.0] {
            for yv in [-3.0, 0.0, 5.0] {
                let mut source = vec![0.0; 7];
                source[0] = xv;
                source[1] = yv;
                source[branches[selected].binary().var_id().unwrap().index()] = 1.0;
                let expected = match selected {
                    0 => xv + yv <= 3.0,
                    1 => xv - yv >= 1.0,
                    _ => true,
                };
                assert_eq!(satisfied(&model, &lift(&model, &report, &source)), expected);
            }
        }
    }
}

#[test]
fn nonlinear_sparse_group_keeps_shifted_perspectives_and_combined_unused_bounds() {
    let model = Model::new("sparse log Hull");

    variable!(model, 1.0 <= x <= 4.0);
    let branches: Vec<_> = (0..4).map(|i| model.add_boolean(format!("b{i}"))).collect();
    branches[0].context().add_constraint("log limit", x.ln().le(3.0_f64.ln()));
    disjunction!(model, branches.clone());

    let report = model
        .reformulate_gdp(
            GdpReformulationOptions::new(Hull::default()).with_hull_reference_coordinate(x, 1.0),
        )
        .unwrap();

    assert_eq!(report.hull_variables.len(), 2);
    for selected in 0..4 {
        for value in [1.0, 2.0, 3.0, 4.0] {
            let mut source = vec![0.0; 5];
            source[0] = value;
            source[branches[selected].binary().var_id().unwrap().index()] = 1.0;
            assert_eq!(
                satisfied(&model, &lift(&model, &report, &source)),
                selected != 0 || value <= 3.0
            );
        }
    }
}

#[test]
fn active_selector_substitution_preserves_small_constants_and_ancestor_offsets() {
    for nested in [false, true] {
        for nonlinear in [false, true] {
            for mixed in [false, true] {
                let model = Model::new("selector cancellation before Hull");

                variable!(model, -2.0 <= x <= 2.0);
                boolean_variable!(model, active);
                boolean_variable!(model, other);
                boolean_variable!(model, child);
                boolean_variable!(model, sibling);
                let owner = if nested { child } else { active };
                let selector = active.binary();
                if nonlinear {
                    disjunct_constraint!(model, owner, x * x + 1e16 * selector - 1.0 <= 1e16);
                } else {
                    disjunct_constraint!(model, owner, x + 1e16 * selector + 1.0 <= 1e16);
                }
                disjunction!(model, [active, other]);
                let mut options = GdpReformulationOptions::new(Hull::default());
                if nested {
                    let inner = disjunction!(model, [child, sibling], parent = active);
                    if mixed {
                        options = options.with_method(inner, BigM::default());
                    }
                    // Own and ancestor selectors cancel each other; the unit
                    // constant must survive regardless of their summation order.
                    disjunct_constraint!(
                        model,
                        child,
                        x + 1e16 * child.binary() - 1e16 * active.binary() + 1.0 <= 0.0
                    );
                }

                let report = model.reformulate_gdp(options).unwrap();
                for selected in 0..3 {
                    if !nested && selected == 2 {
                        continue;
                    }
                    let (av, bv, cv, sv) = match selected {
                        0 => (0.0, 1.0, 0.0, 0.0),
                        1 => (1.0, 0.0, f64::from(nested), 0.0),
                        _ => (1.0, 0.0, 0.0, 1.0),
                    };
                    for value in [-2.0, -1.0, 0.0, 1.0, 2.0] {
                        let enforce = if nested { cv == 1.0 } else { av == 1.0 };
                        let expected = !enforce
                            || (if nonlinear { value * value <= 1.0 } else { value <= -1.0 })
                                && (!nested || value <= -1.0);
                        assert_eq!(
                            satisfied(&model, &lift(&model, &report, &[value, av, bv, cv, sv])),
                            expected,
                            "nested={nested}, nonlinear={nonlinear}, mixed={mixed}, selected={selected}, x={value}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn zero_bound_mass_balances_allow_source_scaled_roundoff() {
    for scale in [1e-12, 1.0, 1e6] {
        for references in [[0.0, 0.0], [0.1, 0.1], [0.1, 0.2], [1.0, 2.0]] {
            for nested in [false, true] {
                let model = Model::new("shifted mass balance");

                variable!(model, -1.0 <= x <= 1.0);
                variable!(model, -1.0 <= z <= 1.0);
                boolean_variable!(model, active);
                boolean_variable!(model, other);
                boolean_variable!(model, child);
                boolean_variable!(model, sibling);
                let owner = if nested { child } else { active };
                disjunct_constraint!(model, owner, scale * x + scale * z == 0.0);
                disjunction!(model, [active, other], AtLeastOne);
                if nested {
                    disjunction!(model, [child, sibling], parent = active);
                }

                let report = model
                    .reformulate_gdp(
                        GdpReformulationOptions::new(Hull::default().with_inclusive_policy(
                            HullInclusivePolicy::Enumerate { max_patterns: 3 },
                        ))
                        .with_hull_reference_coordinates([(x, references[0]), (z, references[1])]),
                    )
                    .unwrap();
                for value in [-1.0, -0.25, 0.0, 0.5, 1.0] {
                    assert!(satisfied(
                        &model,
                        &lift(&model, &report, &[value, -value, 1.0, 0.0, f64::from(nested), 0.0])
                    ));
                }
            }
        }
    }
}

#[test]
fn zero_bound_roundoff_allowance_does_not_scale_with_references() {
    let model = Model::new("lost unit in zero-bound equation");

    variable!(model, -1e16 <= x <= 1e16);
    variable!(model, -2.0 <= z <= 2.0);
    boolean_variable!(model, active);
    disjunct_constraint!(model, active, x + z == 0.0);

    let options = GdpReformulationOptions::new(Hull::default())
        .with_hull_reference_coordinates([(x, 1e16), (z, 1.0)]);

    let before = (model.num_variables(), model.num_constraints(), model.arena().len());
    assert!(matches!(model.reformulate_gdp(options), Err(GdpError::HullRowPrecisionLoss { .. })));
    assert_eq!(before, (model.num_variables(), model.num_constraints(), model.arena().len()));
}

#[test]
fn distinct_sparse_coordinates_use_two_copies_each_with_bulk_references() {
    let model = Model::new("inverted support incidence");

    let coordinates: Vec<_> =
        (0..1000).map(|i| model.__var(format!("x{i}")).bounds(-2.0, 2.0).build()).collect();
    let branches: Vec<_> = (0..1000).map(|i| model.add_boolean(format!("b{i}"))).collect();
    for (&x, branch) in coordinates.iter().zip(&branches) {
        branch.context().add_constraint("limit", x.le(1.0));
    }
    disjunction!(model, branches.clone());

    let report = model
        .reformulate_gdp(
            GdpReformulationOptions::new(Hull::default())
                .with_hull_reference_coordinates(coordinates.iter().map(|x| (x, 0.25))),
        )
        .unwrap();

    assert_eq!(report.hull_variables.len(), 2000);
    assert_eq!(model.num_constraints(), 6001);
    assert!(report.hull_branch_sums.is_empty());

    // Each selected branch constrains only its own coordinate.
    // All others can use their common unused box.
    for selected in [0, 499, 999] {
        for value in [1.0, 2.0] {
            let mut source = vec![2.0; 2000];
            source[1000..].fill(0.0);
            source[selected] = value;
            source[1000 + selected] = 1.0;
            assert_eq!(satisfied(&model, &lift(&model, &report, &source)), value <= 1.0);
        }
    }
}

#[test]
fn nonzero_bound_recovery_keeps_relative_precision_with_large_source_coefficients() {
    let model = Model::new("nonzero scaled bound recovery");

    variable!(model, -1e16 <= x <= 1e16);
    boolean_variable!(model, active);
    disjunct_constraint!(model, active, 1e16 * x >= 1.0);

    let options =
        GdpReformulationOptions::new(Hull::default()).with_hull_reference_coordinate(x, 1e16);
    assert!(matches!(
        model.reformulate_gdp(options),
        Err(GdpError::HullRowPrecisionLoss { side: BoundSide::Lower, .. })
    ));
}

#[test]
fn shared_nonlinear_source_preparation_keeps_each_bound_and_parameter_snapshot() {
    let model = Model::new("cached selector substitution");

    variable!(model, -2.0 <= x <= 2.0);
    param!(model, factor = 1.0);
    boolean_variable!(model, active);
    let body = factor * x * x + 1e16 * active.binary() - 1.0;
    disjunct_constraint!(model, active, first, body <= 1e16);
    disjunct_constraint!(model, active, second, body <= 1e16 + 2.0);
    model.plan_gdp(Hull::default()).unwrap();
    factor.set_param_value(2.0).unwrap();

    let report = model.reformulate_gdp(Hull::default()).unwrap();

    assert_eq!(report.rows.len(), 2);
    let values = lift(&model, &report, &[1.0, 1.0]);
    let arena = model.arena();
    for (source, expected) in report.rows.iter().zip([false, true]) {
        assert_eq!(
            source.constraints.iter().all(|id| {
                let rows = model.constraints();
                let row = &rows.algebraic()[id.index()];
                let value = evaluate(&arena, row.lhs, &&values[..]).unwrap();
                value >= row.lower - 1e-8 && value <= row.upper + 1e-8
            }),
            expected
        );
    }
}

#[test]
fn nonlinear_selector_terms_that_become_constant_share_compensated_substitution() {
    for nested in [false, true] {
        let model = Model::new("linearized selector cancellation");

        variable!(model, -2.0 <= x <= 2.0);
        boolean_variable!(model, active);
        boolean_variable!(model, other);
        boolean_variable!(model, child);
        boolean_variable!(model, sibling);
        let owner = if nested { child } else { active };
        disjunct_constraint!(
            model,
            owner,
            x + 1e16 * owner.binary() + 1.0 - 1e16 * active.binary() * active.binary() == 0.0
        );
        disjunction!(model, [active, other]);
        if nested {
            disjunction!(model, [child, sibling], parent = active);
        }

        let report = model
            .reformulate_gdp(
                GdpReformulationOptions::new(Hull::default())
                    .with_hull_reference_coordinate(x, 0.2),
            )
            .unwrap();
        for value in [-2.0, -1.0, 0.0, 1.0, 2.0] {
            assert_eq!(
                satisfied(
                    &model,
                    &lift(&model, &report, &[value, 1.0, 0.0, f64::from(nested), 0.0])
                ),
                value == -1.0
            );
            assert!(satisfied(&model, &lift(&model, &report, &[value, 0.0, 1.0, 0.0, 0.0])));
        }
    }
}

#[test]
fn reused_affine_scratch_preserves_cancellation_and_changing_row_supports() {
    let model = Model::new("normalization scratch reuse");

    variable!(model, -2.0 <= x <= 2.0);
    variable!(model, -2.0 <= z <= 2.0);
    variable!(model, -2.0 <= w <= 2.0);
    boolean_variable!(model, active);

    // At these references, the guard coefficient cancels to zero.
    disjunct_constraint!(model, active, x + z == 1.0);
    disjunct_constraint!(model, active, w >= -0.5);
    disjunct_constraint!(model, active, 2.0 * z - x <= 1.0);

    let report = model
        .reformulate_gdp(
            GdpReformulationOptions::new(Hull::default()).with_hull_reference_coordinates([
                (x, 0.25),
                (z, 0.75),
                (w, -0.5),
            ]),
        )
        .unwrap();
    for active in [0.0, 1.0] {
        for x in [-1.0, 0.25, 1.0, 1.5] {
            for z in [-0.5, 0.0, 0.75, 2.0] {
                for w in [-1.0, -0.5, 1.0] {
                    assert_eq!(
                        satisfied(&model, &lift(&model, &report, &[x, z, w, active])),
                        active == 0.0 || (x + z == 1.0 && w >= -0.5 && 2.0 * z - x <= 1.0),
                    );
                }
            }
        }
    }
}
