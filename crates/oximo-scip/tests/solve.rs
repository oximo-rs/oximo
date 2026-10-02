#![expect(clippy::many_single_char_names, reason = "mathematical test models")]
use oximo_core::prelude::*;
use oximo_scip::{Scip, ScipOptions, ScipSetting};
use oximo_solver::{PersistentSolver, Solver, TerminationStatus, UniversalOptionsExt};
use std::time::Duration;

fn solve(m: &Model) -> oximo_solver::SolverResult {
    Scip::new().solve(m, &ScipOptions::default().time_limit(Duration::from_secs(15))).unwrap()
}
fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-4, "{a} != {b}");
}

#[test]
fn linear_objective_senses_constants_and_fixed_variables() {
    let m = Model::new("linear");
    variable!(m,0.0<=x<=10.0);
    variable!(m, y, initial = 2.0);
    m.fix(y, 2.0).unwrap();
    constraint!(m, cap, x + y <= 7.0);
    objective!(m, Max, 3.0 * x + 10.0);
    let r = solve(&m);
    assert_eq!(r.termination, TerminationStatus::Optimal);
    close(r.objective().unwrap(), 25.0);
    close(r.best_bound.unwrap(), 25.0);
    assert_eq!(r.primal().unwrap().len(), 2);
    objective!(m, Min, 3.0 * x - 10.0);
    close(solve(&m).objective().unwrap(), -10.0);
}

#[test]
fn constant_objective_without_variables() {
    let m = Model::new("constant");
    objective!(m, Min, m.__constant(2.0));
    let r = solve(&m);
    assert_eq!(r.termination, TerminationStatus::Optimal);
    close(r.objective().unwrap(), 2.0);
    assert!(r.primal().unwrap().is_empty());
}

#[test]
fn milp_and_solution_pool() {
    let m = Model::new("integer");
    variable!(m,0.0<=x<=5.0,Int);
    variable!(m, y, Bin);
    constraint!(m, cap, x + 2.0 * y <= 4.5);
    objective!(m, Max, x + 3.0 * y);
    let output = Scip::new().solve_detailed(&m, &ScipOptions::default().max_solutions(20)).unwrap();
    close(output.result.objective().unwrap(), 5.0);
    assert_eq!(m.kind(), ModelKind::MILP);
    assert_ne!(output.statistics_json, "");
    assert!(output.result.dual.is_empty());
    let objs = output.result.solutions.iter().map(|s| s.objective.unwrap()).collect::<Vec<_>>();
    assert!(objs.windows(2).all(|v| v[0] >= v[1]));
}

#[test]
fn quadratic_objectives() {
    let m = Model::new("qp");
    variable!(m,-5.0<=x<=5.0);
    objective!(m, Min, (x - 2.0).powi(2) + 7.0);
    assert_eq!(m.kind(), ModelKind::QP);
    let r = solve(&m);
    close(r.objective().unwrap(), 7.0);
    close(r.value_of(x).unwrap().unwrap(), 2.0);
    assert_eq!(r.primal().unwrap().len(), 1);
    objective!(m, Max, 4.0 * x - x.powi(2));
    close(solve(&m).objective().unwrap(), 4.0);
    let m = Model::new("miqp");
    variable!(m,-5.0<=x<=5.0,Int);
    objective!(m, Min, (x - 1.7).powi(2));
    assert_eq!(m.kind(), ModelKind::MIQP);
    close(solve(&m).objective().unwrap(), 0.09);
}

#[test]
fn quadratic_constraints_and_cross_term_scaling() {
    for integer in [false, true] {
        let m = Model::new("qcp");
        let x = m
            .__var("x")
            .lb(0.0)
            .ub(4.0)
            .domain(if integer { Domain::Integer } else { Domain::Real })
            .build();
        variable!(m,0.0<=y<=4.0);
        m.fix(y, 2.0).unwrap();
        constraint!(m, quad, x.powi(2) + 2.0 * x * y <= 8.0);
        objective!(m, Max, x);
        let r = solve(&m);
        close(r.value_of(x).unwrap().unwrap(), if integer { 1.0 } else { -2.0 + 12.0_f64.sqrt() });
    }
}

#[test]
fn explicit_cones_and_negative_rhs() {
    let m = Model::new("soc");
    variable!(m, x);
    variable!(m, y);
    variable!(m,-2.0<=t<=10.0);
    m.fix(x, 3.0).unwrap();
    m.fix(y, 4.0).unwrap();
    m.add_soc_constraint("cone", [x, y], t);
    objective!(m, Min, t);
    assert_eq!(m.kind(), ModelKind::SOCP);
    close(solve(&m).objective().unwrap(), 5.0);
    variable!(m, z, Bin);
    constraint!(m, integer, z <= t);
    assert_eq!(m.kind(), ModelKind::MISOCP);
    close(solve(&m).objective().unwrap(), 5.0);
    m.fix(t, -1.0).unwrap();
    assert_eq!(solve(&m).termination, TerminationStatus::Infeasible);
}

#[test]
fn nonlinear_objective_and_constraints() {
    let m = Model::new("nlp");
    variable!(m,0.0<=x<=2.0);
    constraint!(m, cap, x.exp() <= 2.0);
    objective!(m, Max, x);
    assert_eq!(m.kind(), ModelKind::NLP);
    close(solve(&m).objective().unwrap(), 2.0_f64.ln());
    objective!(m, Min, (x - 1.0).exp());
    close(solve(&m).objective().unwrap(), (-1.0_f64).exp());
    let m = Model::new("minlp");
    variable!(m,0.0<=x<=2.0,Int);
    constraint!(m, cap, x.exp() <= 4.0);
    objective!(m, Max, x);
    assert_eq!(m.kind(), ModelKind::MINLP);
    close(solve(&m).objective().unwrap(), 1.0);
}

#[test]
fn supported_functions_at_fixed_values() {
    let m = Model::new("functions");
    variable!(m,0.5<=x<=2.0);
    m.fix(x, 1.0).unwrap();
    objective!(m, Min, x.log() + x.sqrt() + x.abs() + x.sin() + x.cos() + 1.0 / x + x.powf(1.5));
    close(solve(&m).objective().unwrap(), 4.0 + 1.0_f64.sin() + 1.0_f64.cos());
    let m = Model::new("division");
    variable!(m, 0.5 <= x <= 2.0);
    objective!(m, Min, 1.0 / x);
    close(solve(&m).objective().unwrap(), 0.5);
}

#[test]
fn sos1_and_sos2() {
    let m = Model::new("sos");
    variable!(m,0.0<=x<=2.0);
    variable!(m,0.0<=y<=3.0);
    m.add_sos_constraint("choice", SosType::Sos1, [(x, 1.0), (y, 2.0)]);
    objective!(m, Max, x + y);
    close(solve(&m).objective().unwrap(), 3.0);
    m.add_sos_constraint("ordered", SosType::Sos2, [(x, 1.0), (y, 2.0)]);
    assert!(Scip::new().supports_sos(SosType::Sos2));
    assert!(Scip::new().supports_model(&m));
    close(solve(&m).objective().unwrap(), 3.0);
    let d = Model::new("duplicate-native-names");
    variable!(d, 0.0 <= x <= 1.0);
    constraint!(d, same, x <= 1.0);
    d.add_sos_constraint("same", SosType::Sos1, [(x, 1.0)]);
    objective!(d, Max, x);
    close(solve(&d).objective().unwrap(), 1.0);
}

#[test]
fn native_indicators_enforce_trigger_values_and_both_sides() {
    let m = Model::new("indicator_one");
    variable!(m, b, Bin);
    variable!(m, 0.0 <= x <= 10.0);
    m.fix(b, 1.0).unwrap();
    indicator_constraint!(m, cap, b == 1 => x <= 2.0);
    indicator_constraint!(m, inactive, b == 0 => x <= 1.0);
    objective!(m, Max, x);
    assert!(Scip::new().supports_model(&m));
    close(solve(&m).objective().unwrap(), 2.0);

    let m = Model::new("indicator_zero");
    variable!(m, b, Bin);
    variable!(m, 0.0 <= x <= 10.0);
    m.fix(b, 0.0).unwrap();
    indicator_constraint!(m, band, b == 0 => 1.0 <= x <= 3.0);
    objective!(m, Max, x);
    close(solve(&m).objective().unwrap(), 3.0);
    objective!(m, Min, x);
    close(solve(&m).objective().unwrap(), 1.0);

    let m = Model::new("indicator_equal");
    variable!(m, b, Bin);
    variable!(m, 0.0 <= x <= 10.0);
    m.fix(b, 1.0).unwrap();
    indicator_constraint!(m, equal, b == 1 => x == 4.0);
    objective!(m, Max, x);
    close(solve(&m).objective().unwrap(), 4.0);

    let m = Model::new("indicator_choice");
    variable!(m, b, Bin);
    variable!(m, 0.0 <= x <= 10.0);
    indicator_constraint!(m, conditional_cap, b == 0 => x <= 2.0);
    objective!(m, Max, x - 100.0 * b);
    let r = solve(&m);
    close(r.objective().unwrap(), 2.0);
    close(r.value_of(b).unwrap().unwrap(), 0.0);

    let m = Model::new("indicator_constant");
    variable!(m, b, Bin);
    m.fix(b, 1.0).unwrap();
    indicator_constraint!(m, impossible, b == 1 => m.__constant(1.0) <= 0.0);
    objective!(m, Feasibility);
    assert_eq!(solve(&m).termination, TerminationStatus::Infeasible);
}

#[test]
fn persistent_indicators_append_and_rebuild_after_parameter_edits() {
    let m = Model::new("indicator_persistent");
    param!(m, cap = 3.0);
    variable!(m, b, Bin);
    variable!(m, 0.0 <= x <= 10.0);
    m.fix(b, 1.0).unwrap();
    objective!(m, Max, x);
    let mut solver = Scip::new().persistent();
    let opts = ScipOptions::default();
    close(solver.solve(&m, &opts).unwrap().objective().unwrap(), 10.0);
    indicator_constraint!(m, cap_row, b == 1 => x <= cap);
    close(solver.solve(&m, &opts).unwrap().objective().unwrap(), 3.0);
    assert_eq!(solver.reuse_count(), 1);
    cap.set_param_value(2.0);
    close(solver.solve(&m, &opts).unwrap().objective().unwrap(), 2.0);
    assert_eq!(solver.build_count(), 2);
}

#[test]
fn unsupported_operators_and_semi_variables() {
    let m = Model::new("tan");
    variable!(m,0.0<=x<=1.0);
    objective!(m, Min, x.tan());
    assert!(!Scip::new().supports_model(&m));
    assert!(
        Scip::new().solve(&m, &ScipOptions::default()).unwrap_err().to_string().contains("tan")
    );
    let m = Model::new("constant-unsupported");
    param!(m, p = 1.0);
    objective!(m, Min, p.tan());
    assert!(
        Scip::new().solve(&m, &ScipOptions::default()).unwrap_err().to_string().contains("tan")
    );
    let m = Model::new("semi");
    variable!(m,0.0<=x<=3.0,SemiCont(1.0));
    variable!(m,0.0<=y<=4.0,SemiInt(2.0));
    objective!(m, Min, x + y);
    assert!(Scip::new().supports_model(&m));
    close(solve(&m).objective().unwrap(), 0.0);
    constraint!(m, require_positive, x + y >= 2.5);
    close(solve(&m).objective().unwrap(), 2.5);
}

#[test]
fn feasibility_infeasible_unbounded_and_limits() {
    let m = Model::new("feasible");
    variable!(m,0.0<=x<=1.0);
    objective!(m, Feasibility);
    let r = solve(&m);
    assert!(r.has_solution());
    close(r.objective().unwrap(), 0.0);
    constraint!(m, impossible, x >= 2.0);
    assert_eq!(solve(&m).termination, TerminationStatus::Infeasible);
    let m = Model::new("unbounded");
    variable!(m, x);
    objective!(m, Max, x);
    assert!(matches!(
        solve(&m).termination,
        TerminationStatus::Unbounded | TerminationStatus::InfeasibleOrUnbounded
    ));
    let m = Model::new("limit");
    variable!(m,0.0<=x<=2.0);
    objective!(m, Min, x);
    let r = Scip::new().solve(&m, &ScipOptions::default().time_limit(Duration::ZERO)).unwrap();
    assert_eq!(r.termination, TerminationStatus::TimeLimit);
}

#[test]
fn persistence_reuses_and_rebuilds_without_stale_parameters() {
    let m = Model::new("resident");
    param!(m, p = 2.0);
    variable!(m,0.0<=x<=5.0);
    objective!(m, Max, p * x + 3.0);
    let opts = ScipOptions::default();
    let mut solver = Scip::new().persistent();
    close(solver.solve(&m, &opts).unwrap().objective().unwrap(), 13.0);
    solver.solve(&m, &opts).unwrap();
    assert_eq!(solver.build_count(), 1);
    assert_eq!(solver.reuse_count(), 1);
    variable!(m,0.0<=y<=1.0);
    constraint!(m, row, x + y <= 3.0);
    objective!(m, Max, p * x + 3.0 + y);
    close(solver.solve(&m, &opts).unwrap().objective().unwrap(), 9.0);
    assert_eq!(solver.build_count(), 1);
    p.set_param_value(4.0);
    let warm = solver.solve(&m, &opts).unwrap();
    close(warm.objective().unwrap(), solve(&m).objective().unwrap());
    assert_eq!(solver.build_count(), 2);
    m.fix(x, 1.0).unwrap();
    close(solver.solve(&m, &opts).unwrap().objective().unwrap(), 8.0);
    assert!(solver.solve(&m, &opts.clone().real_param("no/such/parameter", 1.0)).is_err());
    close(solver.solve(&m, &opts).unwrap().objective().unwrap(), 8.0);
    solver.reset();
    close(solver.solve(&m, &opts).unwrap().objective().unwrap(), 8.0);
}

#[test]
fn nonlinear_persistence_and_new_options() {
    let m = Model::new("persistent_nlp");
    param!(m, cap = 2.0);
    variable!(m,0.0<=x<=2.0);
    constraint!(m, c, x.exp() <= cap);
    objective!(m, Max, x);
    let mut s = Scip::new().persistent();
    let opts = ScipOptions::default();
    s.solve(&m, &opts).unwrap();
    s.solve(&m, &opts).unwrap();
    assert_eq!(s.build_count(), 1);
    cap.set_param_value(3.0);
    close(s.solve(&m, &opts).unwrap().objective().unwrap(), 3.0_f64.ln());
    assert_eq!(s.build_count(), 2);
    s.solve(&m, &opts.clone().node_limit(0)).unwrap();
    close(s.solve(&m, &opts).unwrap().objective().unwrap(), 3.0_f64.ln());
}

#[test]
fn concurrent_option_solves_without_plugins() {
    let m = Model::new("concurrent");
    variable!(m, 0.0 <= x <= 3.0);
    objective!(m, Max, x);
    let r = Scip::new().solve(&m, &ScipOptions::default().concurrent(true).threads(2)).unwrap();
    close(r.objective().unwrap(), 3.0);
    let mut persistent = Scip::new().persistent();
    let opts = ScipOptions::default().concurrent(true).threads(2);
    close(persistent.solve(&m, &opts).unwrap().objective().unwrap(), 3.0);
    close(persistent.solve(&m, &opts).unwrap().objective().unwrap(), 3.0);
}

#[test]
fn lp_duals_are_a_valid_original_certificate() {
    let m = Model::new("duals");
    variable!(m,0.0<=x<=10.0);
    variable!(m,0.0<=y<=10.0);
    let row = constraint!(m, cap, x + y >= 3.0);
    objective!(m, Min, x + 2.0 * y);
    let opts = ScipOptions::default()
        .presolving(ScipSetting::Off)
        .heuristics(ScipSetting::Off)
        .separating(ScipSetting::Off);
    let r = Scip::new().solve(&m, &opts).unwrap();
    close(r.objective().unwrap(), 3.0);
    assert_eq!(r.dual_status, oximo_solver::DualStatus::FeasiblePoint);
    close(r.dual[&row.id()], 1.0);
    objective!(m, Max, -x - 2.0 * y);
    let r = Scip::new().solve(&m, &opts).unwrap();
    assert_eq!(r.dual_status, oximo_solver::DualStatus::FeasiblePoint);
    close(r.dual[&row.id()], -1.0);
}
