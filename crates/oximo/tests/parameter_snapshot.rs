#![cfg(any(
    feature = "highs",
    feature = "clarabel",
    feature = "pounce",
    feature = "gurobi",
    feature = "mosek",
    feature = "scip",
    feature = "scip-system",
    feature = "gams",
    feature = "baron"
))]

use oximo::prelude::*;

fn check_scenarios<S: Solver>(mut solver: S, opts: &S::Options) {
    let model = Model::new("solution parameters");

    param!(model, p = 2.0);
    param!(model, unused = 7.0);
    variable!(model, 0.0 <= x <= 3.0);
    objective!(model, Max, p * x);
    let first = solver.solve(&model, opts).unwrap();
    assert!(first.has_solution());

    let original_x = first.value_of(x).unwrap().unwrap();
    assert!((original_x - 3.0).abs() < 1e-5);
    assert_eq!(first.parameters.values(), &[2.0, 7.0]);

    let certificate = (first.objective(), first.primal_status, first.termination.clone());
    model.set_param(p, 4.0).unwrap();
    model.set_param(unused, 9.0).unwrap();

    assert_eq!(first.value_of(p * x), Ok(Some(2.0 * original_x)));
    assert_eq!(first.value_of_with_current_parameters(p * x), Ok(Some(4.0 * original_x)));
    assert_eq!(first.best().unwrap().clone().value_of(p * x), Ok(Some(2.0 * original_x)));
    assert_eq!((first.objective(), first.primal_status, first.termination.clone()), certificate);

    let second = solver.solve(&model, opts).unwrap();
    assert!(second.has_solution());

    let second_x = second.value_of(x).unwrap().unwrap();
    assert_eq!(second.value_of(p * x), Ok(Some(4.0 * second_x)));
    assert_eq!(second.parameters.values(), &[4.0, 9.0]);
    assert_eq!(first.value_of(unused), Ok(Some(7.0)));
    assert_eq!(second.value_of(unused), Ok(Some(9.0)));
    for point in &second.solutions {
        assert_eq!(point.parameters.values(), &[4.0, 9.0]);
    }

    model.set_param(unused, 11.0).unwrap();
    let unused_only = solver.solve(&model, opts).unwrap();
    assert!(unused_only.has_solution());
    assert!((unused_only.objective().unwrap() - second.objective().unwrap()).abs() < 1e-5);
    assert_eq!(unused_only.value_of(unused), Ok(Some(11.0)));
    assert_eq!(second.value_of(unused), Ok(Some(9.0)));
    for point in &unused_only.solutions {
        assert_eq!(point.parameters.values(), &[4.0, 11.0]);
        assert_eq!(point.parameters.values().as_ptr(), unused_only.parameters.values().as_ptr());
    }

    let cloned = model.to_reformulated_sos_model(SosReformulationOptions::default()).unwrap();
    let third = solver.solve(&cloned, opts).unwrap();
    assert_eq!(third.model_id(), cloned.id());
    assert!(third.value_of(x).is_err());
    assert!(third.value_of_with_current_parameters(x).is_err());
    model.set_param(p, 6.0).unwrap();
    assert_eq!(second.value_of(p * x), Ok(Some(4.0 * second_x)));
    assert_eq!(first.value_of(p * x), Ok(Some(2.0 * original_x)));
    check_infeasible_scenarios(&mut solver, opts);
}

fn check_infeasible_scenarios<S: Solver>(solver: &mut S, opts: &S::Options) {
    let model = Model::new("infeasible solution parameters");
    param!(model, p = 2.0);
    param!(model, unused = 7.0);
    variable!(model, 0.0 <= x <= 3.0);
    constraint!(model, impossible, x >= 5.0);
    objective!(model, Max, p * x);
    let mut results = Vec::new();

    for (coefficient, unused_value) in [(2.0, 7.0), (2.0, 9.0), (4.0, 9.0)] {
        model.set_param(p, coefficient).unwrap();
        model.set_param(unused, unused_value).unwrap();
        let result = solver.solve(&model, opts).unwrap();
        assert!(!result.has_solution(), "{:?}", result.termination);
        assert!(result.solutions.is_empty());
        assert_eq!(result.model_id(), model.id());
        assert_eq!(result.parameters.values(), &[coefficient, unused_value]);
        results.push(result);
    }
    model.set_param(p, 6.0).unwrap();
    model.set_param(unused, 11.0).unwrap();
    param!(model, added = 13.0);

    for (result, expected) in results.iter().zip([[2.0, 7.0], [2.0, 9.0], [4.0, 9.0]]) {
        assert_eq!(result.parameters.values(), &expected);
        assert_eq!(result.parameters.value(p.param_id().unwrap()), Some(expected[0]));
        assert_eq!(result.parameters.value(unused.param_id().unwrap()), Some(expected[1]));
        assert_eq!(result.parameters.value(added.param_id().unwrap()), None);
        assert_eq!(result.value_of(p), Ok(None));
        assert_eq!(result.value_of_with_current_parameters(p), Ok(None));
        assert_eq!(result.objective(), None);
    }
}

#[cfg(feature = "highs")]
#[test]
fn highs_cold_and_persistent_parameters() {
    use oximo::solvers::Highs;
    check_scenarios(Highs, &HighsOptions::default());
    check_scenarios(Highs.persistent(), &HighsOptions::default());
}

#[cfg(feature = "clarabel")]
#[test]
fn clarabel_cold_and_persistent_parameters() {
    use oximo::solvers::Clarabel;
    check_scenarios(Clarabel, &ClarabelOptions::default());
    check_scenarios(Clarabel.persistent(), &ClarabelOptions::default());
}

#[cfg(feature = "pounce")]
#[test]
fn pounce_convex_and_nlp_cold_and_persistent_parameters() {
    use oximo::solvers::Pounce;
    for opts in [
        PounceOptions::default(),
        PounceOptions::default().solver_selection(PounceSolverSelection::Nlp),
    ] {
        check_scenarios(Pounce, &opts);
        check_scenarios(Pounce.persistent(), &opts);
    }
}

#[cfg(feature = "gurobi")]
#[test]
fn gurobi_cold_and_persistent_parameters() {
    use oximo::solvers::Gurobi;
    check_scenarios(Gurobi, &GurobiOptions::default());
    check_scenarios(Gurobi.persistent(), &GurobiOptions::default());
}

#[cfg(feature = "mosek")]
#[test]
fn mosek_cold_and_persistent_parameters() {
    use oximo::solvers::Mosek;
    check_scenarios(Mosek, &MosekOptions::default());
    check_scenarios(Mosek.persistent(), &MosekOptions::default());
}

#[cfg(any(feature = "scip", feature = "scip-system"))]
#[test]
fn scip_cold_and_persistent_parameters() {
    use oximo::ScipOptions;
    use oximo::solvers::Scip;
    check_scenarios(Scip::default(), &ScipOptions::default());
    check_scenarios(Scip::default().persistent(), &ScipOptions::default());
}

#[cfg(feature = "gams")]
#[test]
fn gams_parameters() {
    use oximo::solvers::Gams;
    check_scenarios(Gams::default(), &GamsOptions::default());
}

#[cfg(feature = "baron")]
#[test]
fn baron_parameters() {
    use oximo::solvers::Baron;
    check_scenarios(Baron::default(), &BaronOptions::default());
}
