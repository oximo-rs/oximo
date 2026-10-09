#![cfg(all(feature = "gdp", any(feature = "io", feature = "highs", feature = "scip")))]
use oximo::prelude::*;

#[test]
fn exports_reject_unresolved_gdp_before_writing() {
    #[cfg(feature = "io")]
    {
        let m = Model::new("export GDP");
        boolean_variable!(m, y);
        logical_constraint!(m, y);
        objective!(m, Feasibility);
        let mut bytes = Vec::new();
        assert!(matches!(
            oximo::io::write_lp(&m, &mut bytes),
            Err(oximo::io::IoError::UnreformulatedGdp)
        ));
        assert_eq!(bytes, [] as [u8; 0]);
        assert!(matches!(
            oximo::io::write_mps(&m, &mut bytes),
            Err(oximo::io::IoError::UnreformulatedGdp)
        ));
        assert_eq!(bytes, [] as [u8; 0]);
        assert!(matches!(
            oximo::io::write_nl(&m, &mut bytes),
            Err(oximo::io::IoError::UnreformulatedGdp)
        ));
        assert_eq!(bytes, [] as [u8; 0]);
        m.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
        oximo::io::write_lp(&m, &mut bytes).unwrap();
        assert_ne!(bytes, [] as [u8; 0]);
    }
}

#[cfg(feature = "highs")]
#[test]
fn solve_time_reformulation_preserves_handles_and_applies_overrides() {
    use oximo::{GdpSolveError, HighsOptions, solvers::Highs};

    let model = Model::new("solve-time choices");
    variable!(model, x);
    boolean_variable!(model, active[i in 0..3]);
    boolean_variable!(model, spare[i in 0..3]);
    let limits = [3.0, 2.0, 4.0];
    disjunct_constraint!(model, active[i], limit[i in 0..3], x <= limits[i]);
    let choice_a = disjunction!(model, [active[0], spare[0]]);
    let choice_b = disjunction!(model, [active[1], spare[1]]);
    disjunction!(model, [active[2], spare[2]]);
    logical_constraint!(model, logical_and(active.values()));
    objective!(model, Max, x);

    let gdp = GdpReformulationOptions::new(BigM::default().with_fallback_big_m(100.0))
        .with_method(choice_a, BigM::default().with_fallback_big_m(20.0))
        .with_method(choice_b, BigM::default().with_fallback_big_m(30.0));
    let result = Highs.with_gdp(gdp).solve(&model, &HighsOptions::default()).unwrap();

    assert_eq!(result.termination, TerminationStatus::Optimal);
    assert!((result.value_of(x).unwrap().unwrap() - 2.0).abs() < 1e-7);
    for indicator in active.values() {
        assert_eq!(result.boolean_value_of(indicator).unwrap(), Some(true));
        assert_eq!(result.best().unwrap().boolean_value_of(indicator).unwrap(), Some(true));
    }
    for indicator in spare.values() {
        assert_eq!(result.boolean_value_of(indicator).unwrap(), Some(false));
    }
    assert!(!model.has_unreformulated_gdp());
    let history = model.gdp_reformulations();
    for (row, expected) in history[0].rows.iter().zip([20.0, 30.0, 100.0]) {
        assert_eq!(row.upper_m.unwrap().value, expected);
    }

    let nonlinear = Model::new("unsupported backend");
    variable!(nonlinear, 0.0 <= z <= 3.0);
    let branch = disjunct!(nonlinear, |d| {
        constraint!(d, z.exp() <= 4.0);
    });
    disjunction!(nonlinear, [branch]);
    objective!(nonlinear, Max, z);
    assert!(matches!(
        Highs.with_gdp(BigM::default()).solve(&nonlinear, &HighsOptions::default()),
        Err(GdpSolveError::Solver(_))
    ));
    assert!(!nonlinear.has_unreformulated_gdp());
}

#[cfg(feature = "highs")]
#[test]
fn linear_gdp_matches_enumerated_branch_optimum_and_persistent_rejects_new_logic() {
    use oximo::{HighsOptions, solvers::Highs};
    let model = Model::new("GDP MILP");
    variable!(model,0.0<=x<=10.0);
    variable!(model,0.0<=z<=10.0);
    let a = disjunct!(model, a, |d| {
        constraint!(d, x <= 2.0);
        constraint!(d, z <= 8.0);
    });
    let b = disjunct!(model, b, |d| {
        constraint!(d, x <= 7.0);
        constraint!(d, z <= 1.0);
    });
    disjunction!(model, [a, b]);
    objective!(model, Max, 3.0 * x + z);

    assert!(!Highs.supports_model(&model));
    assert!(Highs.solve(&model, &HighsOptions::default()).is_err());
    let mut best = f64::NEG_INFINITY;
    for (x_bound, z_bound) in [(2.0, 8.0), (7.0, 1.0)] {
        let branch = Model::new("branch");
        variable!(branch,0.0<=x<=x_bound);
        variable!(branch,0.0<=z<=z_bound);
        objective!(branch, Max, 3.0 * x + z);
        best =
            best.max(Highs.solve(&branch, &HighsOptions::default()).unwrap().objective().unwrap());
    }

    model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
    let result = Highs.solve(&model, &HighsOptions::default()).unwrap();
    assert_eq!(result.termination, TerminationStatus::Optimal);
    assert!((result.objective().unwrap() - best).abs() < 1e-7);
    assert_eq!(result.boolean_value_of(a).unwrap(), Some(false));
    assert_eq!(result.boolean_value_of(b).unwrap(), Some(true));
    let mut persistent = Highs.persistent();
    persistent.solve(&model, &HighsOptions::default()).unwrap();
    logical_constraint!(model, !b);
    assert!(persistent.solve(&model, &HighsOptions::default()).is_err());
    model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
    let result = persistent.solve(&model, &HighsOptions::default()).unwrap();
    assert!((result.objective().unwrap() - 14.0).abs() < 1e-7);
}

#[cfg(feature = "highs")]
#[test]
fn direct_cardinality_with_repeated_terms_solves_correctly() {
    use oximo::{HighsOptions, solvers::Highs};
    for cancel in [false, true] {
        let model = Model::new("compact cardinality");
        boolean_variable!(model, a);
        boolean_variable!(model, b);
        if cancel {
            logical_constraint!(model, exactly(1, [LogicalExpr::from(a), !a, b.into()]));
        } else {
            logical_constraint!(model, exactly(1, [a, a, b]));
        }
        objective!(model, Max, 2.0 * a.binary() + b.binary());
        let report = model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
        assert_eq!(report.variables, []);
        assert_eq!(report.logical_constraints[0].constraints.len(), 1);
        let result = Highs.solve(&model, &HighsOptions::default()).unwrap();
        assert_eq!(result.termination, TerminationStatus::Optimal);
        let expected = if cancel { 2.0 } else { 1.0 };
        assert!((result.objective().unwrap() - expected).abs() < 1e-7);
    }
}

#[cfg(feature = "scip")]
#[test]
fn nonlinear_gdp_matches_enumerated_branch_models() {
    use oximo::{ScipOptions, solvers::Scip};
    let m = Model::new("GDP MINLP");
    variable!(m,0.0<=x<=3.0);
    let a = disjunct!(m, a, |d| {
        constraint!(d, x.exp() <= 2.0);
    });
    let b = disjunct!(m, b, |d| {
        constraint!(d, x.exp() <= 4.0);
    });
    disjunction!(m, [a, b]);
    objective!(m, Max, x);
    m.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
    let result = Scip::default().solve(&m, &ScipOptions::default()).unwrap();
    assert_eq!(result.termination, TerminationStatus::Optimal);
    let mut best = f64::NEG_INFINITY;
    for bound in [2.0, 4.0] {
        let branch = Model::new("nonlinear branch");
        variable!(branch,0.0<=x<=3.0);
        constraint!(branch, x.exp() <= bound);
        objective!(branch, Max, x);
        best = best.max(
            Scip::default().solve(&branch, &ScipOptions::default()).unwrap().objective().unwrap(),
        );
    }
    assert!((result.objective().unwrap() - best).abs() < 1e-5);
    assert!((best - 4.0_f64.ln()).abs() < 1e-5);
    assert_eq!(result.boolean_value_of(a).unwrap(), Some(false));
    assert_eq!(result.boolean_value_of(b).unwrap(), Some(true));
}

#[cfg(feature = "highs")]
#[test]
fn global_bound_tightening_strengthens_lp_relaxation_without_changing_milp_optimum() {
    use oximo::{HighsOptions, solvers::Highs};
    for tighten in [false, true] {
        let model = Model::new("GDP global cap");
        variable!(model, 0.0 <= x <= 1000.0);
        constraint!(model, x <= 10.0);
        boolean_variable!(model, active);
        disjunct_constraint!(model, active, x <= 2.0);
        objective!(model, Max, x + 10.0 * active.binary());
        let report = model
            .reformulate_gdp(GdpReformulationOptions::default().with_bound_tightening(tighten))
            .unwrap();
        let m = report.rows[0].upper_m.unwrap().value;
        let integer = Highs.solve(&model, &HighsOptions::default()).unwrap();
        assert_eq!(integer.termination, TerminationStatus::Optimal);
        assert!((integer.objective().unwrap() - 12.0).abs() < 1e-7);

        let relaxation = Model::new("GDP LP relaxation");
        variable!(relaxation, 0.0 <= flow <= 1000.0);
        variable!(relaxation, 0.0 <= selection <= 1.0);
        constraint!(relaxation, flow <= 10.0);
        constraint!(relaxation, flow + m * selection <= 2.0 + m);
        objective!(relaxation, Max, flow + 10.0 * selection);
        let result = Highs.solve(&relaxation, &HighsOptions::default()).unwrap();
        let expected = if tighten { 12.0 } else { 20.0 - 80.0 / 998.0 };
        assert_eq!(result.termination, TerminationStatus::Optimal);
        assert!((result.objective().unwrap() - expected).abs() < 1e-7);
    }
}

#[cfg(feature = "highs")]
#[test]
fn hierarchical_big_m_strengthens_the_lp_and_preserves_the_milp_optimum() {
    use oximo::{HighsOptions, solvers::Highs};
    for enabled in [false, true] {
        let model = Model::new("hierarchical Big-M");
        variable!(model, 0.0 <= x <= 10.0);
        boolean_variable!(model, parent);
        boolean_variable!(model, other);
        boolean_variable!(model, child);
        boolean_variable!(model, sibling);
        disjunct_constraint!(model, parent, x <= 4.0);
        disjunct_constraint!(model, child, x <= 1.0);
        disjunct_constraint!(model, sibling, x <= 2.0);
        disjunction!(model, [parent, other]);
        disjunction!(model, [child, sibling], parent = parent);
        // This reward exposes a fractional optimum in the original formulation.
        objective!(model, Max, x + 7.0 * parent.binary());
        model.reformulate_gdp(BigM::default().with_hierarchical_tightening(enabled)).unwrap();
        let integer = Highs.solve(&model, &HighsOptions::default()).unwrap();
        assert_eq!(integer.termination, TerminationStatus::Optimal);
        assert!((integer.objective().unwrap() - 10.0).abs() < 1e-7);

        // Copy the actual generated rows, replacing all variables with continuous
        // ones. This exercises the emitted coefficients rather than rebuilding M rows.
        let relaxation = Model::new("hierarchical LP relaxation");
        let variables: Vec<_> = model
            .variables()
            .iter()
            .map(|v| relaxation.__var(v.name.clone()).bounds(v.lb, v.ub).build())
            .collect();
        let arena = model.arena();
        for row in model.constraints().algebraic() {
            let terms = oximo::expr::extract_linear(&arena, row.lhs).unwrap();
            let body = terms
                .coeffs
                .iter()
                .fold(0.0 * variables[0] + terms.constant, |body, &(variable, coefficient)| {
                    body + coefficient * variables[variable.index()]
                });
            constraint!(relaxation, row.lower <= body <= row.upper);
        }
        objective!(relaxation, Max, variables[0] + 7.0 * variables[1]);
        let relaxed = Highs.solve(&relaxation, &HighsOptions::default()).unwrap();
        assert_eq!(relaxed.termination, TerminationStatus::Optimal);
        let expected = if enabled { 10.0 } else { 11.0 };
        assert!((relaxed.objective().unwrap() - expected).abs() < 1e-7);
    }
}

#[cfg(feature = "highs")]
#[test]
fn ill_scaled_big_m_is_rejected_and_tighter_bounds_preserve_the_active_optimum() {
    use oximo::{HighsOptions, solvers::Highs};
    for upper in [false, true] {
        let model = Model::new("small active bound");
        variable!(model, -1e14 <= x <= 1e14);
        boolean_variable!(model, active);
        logical_constraint!(model, active);
        if upper {
            disjunct_constraint!(model, active, x <= 0.001);
            objective!(model, Max, x);
        } else {
            disjunct_constraint!(model, active, x >= -0.001);
            objective!(model, Min, x);
        }
        assert!(matches!(
            model.reformulate_gdp(GdpReformulationOptions::default()),
            Err(GdpError::PrecisionLoss { .. })
        ));
        assert_eq!(model.num_constraints(), 0);
        assert!(model.has_unreformulated_gdp());

        constraint!(model, -1.0 <= x <= 1.0);
        model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
        let result = Highs.solve(&model, &HighsOptions::default()).unwrap();
        assert_eq!(result.termination, TerminationStatus::Optimal);
        let expected = if upper { 0.001 } else { -0.001 };
        assert!((result.objective().unwrap() - expected).abs() < 1e-9);
    }
}
