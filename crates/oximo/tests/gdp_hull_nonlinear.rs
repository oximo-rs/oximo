#![cfg(all(feature = "gdp", feature = "scip"))]
use oximo::prelude::*;
use oximo::{ScipOptions, solvers::Scip};

#[test]
fn scip_solves_nested_nonlinear_hull_and_shifted_logarithm() {
    let model = Model::new("nested nonlinear Hull");

    variable!(model, 0.0 <= x <= 5.0);
    let parent = disjunct!(model, parent, |d| {
        constraint!(d, x <= 4.0);
    });
    let other = disjunct!(model, other, |d| {
        constraint!(d, x <= 1.0);
    });
    let square = disjunct!(model, square, |d| {
        constraint!(d, x * x <= 4.0);
    });
    let exponential = disjunct!(model, exponential, |d| {
        constraint!(d, x.exp() <= 3.0_f64.exp());
    });
    disjunction!(model, [parent, other]);
    disjunction!(model, [square, exponential], parent = parent);
    objective!(model, Max, x);

    let result =
        Scip::default().with_gdp(Hull::default()).solve(&model, &ScipOptions::default()).unwrap();

    assert_eq!(result.termination, TerminationStatus::Optimal);
    assert!((result.value_of(x).unwrap().unwrap() - 3.0).abs() < 1e-5);
    assert_eq!(result.boolean_value_of(exponential).unwrap(), Some(true));

    let model = Model::new("shifted logarithm Hull");

    variable!(model, 1.0 <= flow <= 4.0);
    boolean_variable!(model, active);
    disjunct_constraint!(model, active, flow.ln() <= 3.0_f64.ln());
    logical_constraint!(model, active);
    objective!(model, Max, flow);

    let options =
        GdpReformulationOptions::new(Hull::default()).with_hull_reference_point(flow, 1.0);

    let result = Scip::default().with_gdp(options).solve(&model, &ScipOptions::default()).unwrap();

    assert_eq!(result.termination, TerminationStatus::Optimal);
    assert!((result.value_of(flow).unwrap().unwrap() - 3.0).abs() < 1e-5);
}

#[test]
fn scip_solves_mixed_nested_methods_with_active_and_inactive_parents() {
    for parent_hull in [false, true] {
        for inactive_parent in [false, true] {
            let model = Model::new("mixed nested MINLP");

            variable!(model, 0.0 <= x <= 5.0);
            let parent = disjunct!(model, parent, |d| {
                constraint!(d, x * x <= 16.0);
            });
            let other = disjunct!(model, other, |d| {
                constraint!(d, x <= 1.0);
            });
            let square = disjunct!(model, square, |d| {
                constraint!(d, x * x <= 4.0);
            });
            let exponential = disjunct!(model, exponential, |d| {
                constraint!(d, x.exp() <= 3.0_f64.exp());
            });
            let outer = disjunction!(model, [parent, other]);
            let inner = disjunction!(model, [square, exponential], parent = parent);
            if inactive_parent {
                logical_constraint!(model, !parent);
            }
            objective!(model, Max, x);
            let (parent_method, child_method) = if parent_hull {
                (GdpMethod::from(Hull::default()), GdpMethod::from(BigM::default()))
            } else {
                (GdpMethod::from(BigM::default()), GdpMethod::from(Hull::default()))
            };

            let options =
                GdpReformulationOptions::new(parent_method).with_method(inner, child_method);

            let result =
                Scip::default().with_gdp(options).solve(&model, &ScipOptions::default()).unwrap();

            assert_eq!(result.termination, TerminationStatus::Optimal);
            let expected = if inactive_parent { 1.0 } else { 3.0 };

            assert!((result.objective().unwrap() - expected).abs() < 1e-5);
            assert!((result.value_of(x).unwrap().unwrap() - expected).abs() < 1e-5);
            assert_eq!(result.boolean_value_of(parent).unwrap(), Some(!inactive_parent));
            assert_eq!(result.boolean_value_of(other).unwrap(), Some(inactive_parent));
            assert_eq!(result.boolean_value_of(square).unwrap(), Some(false));
            assert_eq!(result.boolean_value_of(exponential).unwrap(), Some(!inactive_parent));
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
                oximo::io::write_nl(&model, &mut bytes).unwrap();
                let imported = oximo::io::read_nl(std::io::Cursor::new(bytes)).unwrap();

                let result = Scip::default().solve(&imported, &ScipOptions::default()).unwrap();

                assert_eq!(result.termination, TerminationStatus::Optimal);
                assert!((result.objective().unwrap() - expected).abs() < 1e-5);
            }
        }
    }
}

#[test]
fn nonlinear_hull_and_logical_auxiliaries_share_the_committed_arena() {
    let model = Model::new("Hull with compound logic");

    variable!(model, -1.0 <= x <= 3.0);
    boolean_variable!(model, active);
    boolean_variable!(model, left);
    boolean_variable!(model, right);
    disjunct_constraint!(model, active, x.exp() <= 2.0_f64.exp());
    logical_constraint!(model, iff(active, left | right));
    logical_constraint!(model, left);
    objective!(model, Max, x);
    model.reformulate_gdp(Hull::default()).unwrap();

    let report = &model.gdp_reformulations()[0];
    assert!(report.variables.len() > report.hull_variables.len());

    let result = Scip::default().solve(&model, &ScipOptions::default()).unwrap();

    assert_eq!(result.termination, TerminationStatus::Optimal);
    assert!((result.value_of(x).unwrap().unwrap() - 2.0).abs() < 1e-5);
    assert_eq!(result.boolean_value_of(active).unwrap(), Some(true));
    #[cfg(feature = "io")]
    {
        let mut bytes = Vec::new();
        oximo::io::write_nl(&model, &mut bytes).unwrap();
        let imported = oximo::io::read_nl(std::io::Cursor::new(bytes)).unwrap();

        assert_eq!(imported.num_variables(), model.num_variables());
        assert_eq!(imported.num_constraints(), model.num_constraints());

        let result = Scip::default().solve(&imported, &ScipOptions::default()).unwrap();

        assert_eq!(result.termination, TerminationStatus::Optimal);
        assert!((result.objective().unwrap() - 2.0).abs() < 1e-5);
    }
}
