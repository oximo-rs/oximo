#![allow(clippy::float_cmp, reason = "exact fixture values and provenance comparisons")]
use oximo_gdp::prelude::*;

/// An empty Hull child forces the ancestor through the mixed compiler while
/// leaving the tested source row and its numeric enclosure unchanged.
fn mixed_child(
    model: &Model,
    parent: BooleanHandle<'_>,
    options: GdpReformulationOptions,
) -> GdpReformulationOptions {
    boolean_variable!(model, child);
    boolean_variable!(model, sibling);
    let child = disjunction!(model, [child, sibling], parent = parent);
    options.with_method(child, Hull::default())
}

#[test]
fn big_m_paths_agree_on_estimates_precedence_and_parameter_snapshots() {
    for case in 0..5 {
        let mut comparisons = Vec::new();
        for mixed in [false, true] {
            let model = Model::new("shared Big-M precedence");

            let bounded = case < 2;
            let x = model
                .__var("x")
                .bounds(
                    if bounded { -2.0 } else { f64::NEG_INFINITY },
                    if bounded { 4.0 } else { f64::INFINITY },
                )
                .build();
            param!(model, offset = 1.0);
            boolean_variable!(model, active);
            boolean_variable!(model, other);
            let GdpRangeHandles::Interval(row) =
                disjunct_constraint!(model, active, capacity, -2.0 <= 2.0 * x + x + offset <= 10.0)
            else {
                panic!("expected an affine interval")
            };
            let outer = disjunction!(model, [active, other]);
            let mut options = GdpReformulationOptions::new(BigM::default())
                .with_bound_tightening(false)
                .with_fallback_big_m(100.0);
            if case < 4 {
                options = options.with_method(outer, BigM::default().with_fallback_big_m(50.0));
            }
            if case == 1 || case == 2 {
                options =
                    options.with_big_m(row, BigMValues { lower: Some(20.0), upper: Some(30.0) });
            }
            if mixed {
                options = mixed_child(&model, active, options);
            }

            let report = model.reformulate_gdp(options).unwrap();
            let artifact = report.rows.iter().find(|r| r.source == row.id()).unwrap();
            let amounts = [artifact.lower_m.unwrap(), artifact.upper_m.unwrap()];
            let origin = match case {
                0 => BigMOrigin::Estimated,
                1 | 2 => BigMOrigin::Explicit,
                3 => BigMOrigin::DisjunctionFallback,
                _ => BigMOrigin::GlobalFallback,
            };
            assert!(amounts.iter().all(|amount| amount.origin == origin));
            let expected = match case {
                0 => [3.0, 3.0],
                1 | 2 => [20.0, 30.0],
                3 => [50.0, 50.0],
                _ => [100.0, 100.0],
            };
            for (amount, expected) in amounts.into_iter().zip(expected) {
                assert!((amount.value - expected).abs() < 1e-10);
            }
            comparisons.push(amounts);

            let rows = model.constraints();
            let arena = model.arena();
            let mut values = vec![0.0; model.num_variables()];
            values[x.var_id().unwrap().index()] = 0.0;
            values[active.binary().var_id().unwrap().index()] = 1.0;
            assert!(artifact.constraints.iter().all(|id| {
                let row = &rows.algebraic()[id.index()];
                let value = oximo_expr::evaluate(&arena, row.lhs, &&values[..]).unwrap();
                value >= row.lower && value <= row.upper
            }));
        }
        assert_eq!(comparisons[0], comparisons[1]);
    }
}

#[test]
fn big_m_paths_share_domain_missing_m_and_precision_diagnostics() {
    for case in 0..3 {
        let mut errors = Vec::new();
        for mixed in [false, true] {
            let model = Model::new("shared Big-M errors");

            let x = model
                .__var("x")
                .bounds(
                    if case == 1 { f64::NEG_INFINITY } else { -1e16 },
                    if case == 1 { f64::INFINITY } else { 1e16 },
                )
                .build();
            boolean_variable!(model, active);
            boolean_variable!(model, other);
            if case == 0 {
                disjunct_constraint!(model, active, failure, x.ln() <= 1.0);
            } else {
                disjunct_constraint!(model, active, failure, x >= 1.0);
            }
            disjunction!(model, [active, other]);
            let mut options =
                GdpReformulationOptions::new(BigM::default()).with_bound_tightening(false);
            if mixed {
                options = mixed_child(&model, active, options);
            }

            let before = (model.num_variables(), model.num_constraints(), model.arena().len());

            let error = model.reformulate_gdp(options).unwrap_err();

            assert!(
                match case {
                    0 => matches!(error, GdpError::UnsafeDomain { operation: "log", .. }),
                    1 => matches!(error, GdpError::MissingBigM { side: BoundSide::Lower, .. }),
                    _ => matches!(error, GdpError::PrecisionLoss { side: BoundSide::Lower, .. }),
                },
                "{error:?}"
            );
            assert_eq!(
                (model.num_variables(), model.num_constraints(), model.arena().len()),
                before
            );
            assert!(model.gdp_reformulations().is_empty());
            errors.push(error);
        }
        assert_eq!(errors[0], errors[1]);
    }
}

#[test]
fn hull_numeric_boxes_preserve_semicontinuous_domains_and_tightened_bounds() {
    for domain in
        [Domain::SemiContinuous { threshold: 2.0 }, Domain::SemiInteger { threshold: 2.0 }]
    {
        for tighten in [false, true] {
            let model = Model::new("semi-domain Hull boxes");

            let x = model.__var("x").bounds(2.0, 4.0).domain(domain).build();
            boolean_variable!(model, active);
            if tighten {
                constraint!(model, x >= 1.0);
            }
            disjunct_constraint!(model, active, x >= 3.0);

            let report = model.reformulate_gdp(Hull::default()).unwrap();

            assert_eq!(model.resolved_variables()[x.var_id().unwrap().index()].domain, domain);
            for copy in &report.hull_variables {
                assert_eq!(model.resolved_variables()[copy.generated.index()].domain, Domain::Real);
                if tighten {
                    assert!(copy.bounds[0] > 0.0 && copy.bounds[0] <= 1.0);
                } else {
                    assert_eq!(copy.bounds[0], 0.0);
                }
                assert_eq!(copy.bounds[1], 4.0);
            }
            let mut values = vec![0.0; model.num_variables()];
            for active_value in [0.0, 1.0] {
                for x_value in [0.0, 2.0, 3.0, 4.0] {
                    values[active.binary().var_id().unwrap().index()] = active_value;
                    values[x.var_id().unwrap().index()] = x_value;
                    for copy in &report.hull_variables {
                        let activation =
                            copy.activation_kind.value_with(|v| Some(values[v.index()])).unwrap();
                        values[copy.generated.index()] = activation * x_value;
                    }
                    let rows = model.constraints();
                    let arena = model.arena();
                    let feasible = rows.algebraic().iter().all(|row| {
                        let value = oximo_expr::evaluate(&arena, row.lhs, &&values[..]).unwrap();
                        value >= row.lower - 1e-10 && value <= row.upper + 1e-10
                    });
                    let expected =
                        (!tighten || x_value >= 1.0) && (active_value == 0.0 || x_value >= 3.0);
                    assert_eq!(feasible, expected);
                }
            }
        }
    }
}
