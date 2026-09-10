//! Common adapter contracts.

#![cfg(any(
    feature = "highs",
    feature = "clarabel",
    feature = "pounce",
    feature = "gurobi",
    feature = "mosek",
    feature = "gams",
    feature = "baron"
))]

use oximo::prelude::*;

fn close(actual: Option<f64>, expected: f64) {
    let actual = actual.expect("missing result value");
    assert!((actual - expected).abs() < 1e-5, "got {actual}, expected {expected}");
}

fn range_and_offset<S: Solver>(mut solver: S, opts: &S::Options, require_dual: bool) {
    for slope in [2.0, -2.0] {
        let model = Model::new("range_offset_conformance");
        variable!(model, -10.0 <= x <= 10.0);
        constraint!(model, band, 1.0 <= 2.0 * x + 3.0 <= 9.0);
        objective!(model, Max, slope * x + 7.0);
        let result = solver.solve(&model, opts).unwrap();
        let expected_x = if slope > 0.0 { 3.0 } else { -1.0 };
        assert_eq!(result.termination, TerminationStatus::Optimal);
        assert_eq!(result.primal_status, PrimalStatus::OptimalPoint);
        close(result.value_of(x), expected_x);
        close(result.objective(), slope * expected_x + 7.0);
        close(result.best_bound, slope * expected_x + 7.0);
        assert!(result.gap.is_none());
        if require_dual {
            assert_eq!(result.dual_status, DualStatus::FeasiblePoint);
            // Both the native interval and split-row representations must
            // return sensitivity in the original row and objective units.
            close(result.dual_of(model.constraint_id("band").unwrap()), slope / 2.0);
        }
    }
}

fn parameter_refresh<S: Solver>(mut solver: S, opts: &S::Options) {
    let model = Model::new("parameter_refresh_conformance");
    variable!(model, 0.0 <= x <= 20.0);
    param!(model, a = 2.0);
    param!(model, shift = 3.0);
    param!(model, price = 2.0);
    param!(model, offset = 7.0);
    constraint!(model, cap, a * x + shift <= 9.0);
    objective!(model, Max, price * x + offset);
    // Objective-only updates followed by row coefficient/constant updates.
    for (av, sv, pv, ov) in [(2.0, 3.0, 2.0, 7.0), (2.0, 3.0, 3.0, 11.0), (4.0, 1.0, 3.0, 11.0)] {
        model.set_param(a, av);
        model.set_param(shift, sv);
        model.set_param(price, pv);
        model.set_param(offset, ov);
        let result = solver.solve(&model, opts).unwrap();
        let expected_x = (9.0 - sv) / av;
        assert_eq!(result.termination, TerminationStatus::Optimal);
        close(result.value_of(x), expected_x);
        close(result.objective(), pv * expected_x + ov);
        close(result.best_bound, pv * expected_x + ov);
        assert!(result.gap.is_none());
    }
}

macro_rules! adapter {
    ($feature:literal, $name:ident, $solver:expr, $opts:expr, $dual:expr) => {
        #[cfg(feature = $feature)]
        mod $name {
            use super::*;
            #[test]
            fn maximization_offset_and_both_range_sides() {
                range_and_offset($solver, &$opts, $dual);
            }
            #[test]
            fn parameters_between_cold_solves() {
                parameter_refresh($solver, &$opts);
            }
        }
    };
}

adapter!("highs", highs, oximo::solvers::Highs, oximo::HighsOptions::default(), true);
adapter!("clarabel", clarabel, oximo::solvers::Clarabel, oximo::ClarabelOptions::default(), true);
adapter!("pounce", pounce, oximo::solvers::Pounce, oximo::pounce::PounceOptions::default(), true);
adapter!("gurobi", gurobi, oximo::solvers::Gurobi, oximo::GurobiOptions::default(), true);
adapter!("mosek", mosek, oximo::solvers::Mosek, oximo::MosekOptions::default(), true);
adapter!("gams", gams, oximo::solvers::Gams::new(), oximo::GamsOptions::default(), true);
adapter!("baron", baron, oximo::solvers::Baron::new(), oximo::BaronOptions::default(), false);

macro_rules! resident {
    ($feature:literal, $name:ident, $solver:expr, $opts:expr) => {
        #[cfg(feature = $feature)]
        #[test]
        fn $name() {
            parameter_refresh($solver.persistent(), &$opts);
        }
    };
}

resident!("highs", highs_resident, oximo::solvers::Highs, oximo::HighsOptions::default());
resident!(
    "clarabel",
    clarabel_resident,
    oximo::solvers::Clarabel,
    oximo::ClarabelOptions::default()
);
resident!(
    "pounce",
    pounce_resident,
    oximo::solvers::Pounce,
    oximo::pounce::PounceOptions::default()
);
resident!("gurobi", gurobi_resident, oximo::solvers::Gurobi, oximo::GurobiOptions::default());
resident!("mosek", mosek_resident, oximo::solvers::Mosek, oximo::MosekOptions::default());

#[cfg(feature = "clarabel")]
#[test]
fn clarabel_iteration_limit_without_a_certified_point() {
    let model = Model::new("limit_without_point");
    variable!(model, x >= 0.0);
    constraint!(model, cap, x <= 3.0);
    objective!(model, Max, x + 7.0);
    let result = oximo::solvers::Clarabel
        .solve(&model, &oximo::ClarabelOptions::default().max_iter(0))
        .unwrap();
    assert_eq!(result.termination, TerminationStatus::IterationLimit);
    assert_eq!(result.primal_status, PrimalStatus::NoSolution);
    assert!(result.solutions.is_empty());
    assert!(result.best_bound.is_none());
    assert!(result.gap.is_none());
    assert!(result.dual.is_empty());
}

#[cfg(feature = "pounce")]
#[test]
fn pounce_nlp_limit_requires_feasibility_evidence() {
    for initial in [0.0, 1.5] {
        let model = Model::new("limit_point_evidence");
        variable!(model, 0.0 <= x <= 3.0, initial = initial);
        constraint!(model, floor, x >= 1.0);
        objective!(model, Min, (x - 2.0).powi(2) + 7.0);
        let opts = oximo::pounce::PounceOptions::default()
            .solver_selection(oximo::pounce::PounceSolverSelection::Nlp)
            .max_iter(0);
        let result = oximo::solvers::Pounce.solve(&model, &opts).unwrap();
        assert_eq!(result.termination, TerminationStatus::IterationLimit);
        assert_eq!(result.has_solution(), initial >= 1.0);
        assert!(result.best_bound.is_none());
        assert!(result.gap.is_none());
        assert!(result.dual.is_empty());
    }
}
