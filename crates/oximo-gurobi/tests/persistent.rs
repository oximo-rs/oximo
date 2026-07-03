//! Live Gurobi tests for the persistent solver handle.

use oximo_core::prelude::*;
use oximo_gurobi::{Gurobi, GurobiOptions};
use oximo_solver::{PersistentSolver, Solver, TerminationStatus};

#[test]
fn persistent_matches_cold_solve_on_objective_sweep() {
    let m = Model::new("pricing");
    param!(m, p1 = 0.0);
    variable!(m, x1 >= 0.0);
    variable!(m, x2 >= 0.0);
    constraint!(m, labor, 2.0 * x1 + x2 <= 100.0);
    constraint!(m, material, x1 + 3.0 * x2 <= 90.0);
    objective!(m, Max, p1 * x1 + 5.0 * x2);

    let mut solver = Gurobi.persistent();
    for price in [1.0, 1.6, 2.0, 5.0, 11.0] {
        p1.set_param_value(price);
        let s = solver.solve(&m, &GurobiOptions::default()).expect("persistent solve");
        let c = Gurobi.solve(&m, &GurobiOptions::default()).expect("cold solve");
        assert_eq!(s.termination, TerminationStatus::Optimal, "price {price}");
        assert!((s.objective().unwrap() - c.objective().unwrap()).abs() < 1e-6, "price {price}");
        assert!((s.value_of(x1).unwrap() - c.value_of(x1).unwrap()).abs() < 1e-6);
        assert!((s.value_of(x2).unwrap() - c.value_of(x2).unwrap()).abs() < 1e-6);
    }
}

#[test]
fn persistent_rebuilds_on_constraint_rhs_change() {
    let m = Model::new("capacity");
    param!(m, cap = 100.0);
    variable!(m, x1 >= 0.0);
    variable!(m, x2 >= 0.0);
    constraint!(m, labor, 2.0 * x1 + x2 <= cap);
    constraint!(m, material, x1 + 3.0 * x2 <= 90.0);
    objective!(m, Max, 3.0 * x1 + 5.0 * x2);

    let mut solver = Gurobi.persistent();
    for c in [100.0, 60.0, 140.0] {
        cap.set_param_value(c);
        let s = solver.solve(&m, &GurobiOptions::default()).expect("persistent solve");
        let cold = Gurobi.solve(&m, &GurobiOptions::default()).expect("cold solve");
        assert_eq!(s.termination, TerminationStatus::Optimal, "cap {c}");
        assert!((s.objective().unwrap() - cold.objective().unwrap()).abs() < 1e-6, "cap {c}");
    }
}

#[test]
fn persistent_bound_change_via_fix() {
    let m = Model::new("bounds");
    variable!(m, x1 >= 0.0);
    variable!(m, x2 >= 0.0);
    constraint!(m, labor, 2.0 * x1 + x2 <= 100.0);
    objective!(m, Max, 3.0 * x1 + 5.0 * x2);

    let mut solver = Gurobi.persistent();
    solver.solve(&m, &GurobiOptions::default()).expect("first solve");
    m.fix(x1, 10.0);
    let s = solver.solve(&m, &GurobiOptions::default()).expect("after fix");
    let cold = Gurobi.solve(&m, &GurobiOptions::default()).expect("cold solve");
    assert_eq!(s.termination, TerminationStatus::Optimal);
    assert!((s.value_of(x1).unwrap() - 10.0).abs() < 1e-6);
    assert!((s.objective().unwrap() - cold.objective().unwrap()).abs() < 1e-6);
}

#[test]
fn persistent_feasibility_no_objective() {
    let m = Model::new("feas");
    variable!(m, 0.0 <= x <= 10.0);
    variable!(m, 0.0 <= y <= 10.0);
    constraint!(m, c, x + y == 5.0);

    let mut solver = Gurobi.persistent();
    let r = solver.solve(&m, &GurobiOptions::default()).expect("feasibility solve");
    assert!(r.has_solution(), "termination = {:?}", r.termination);
    m.fix(x, 2.0);
    let r2 = solver.solve(&m, &GurobiOptions::default()).expect("after fix");
    assert!(r2.has_solution());
    assert!((r2.value_of(x).unwrap() - 2.0).abs() < 1e-6);
    assert!((r2.value_of(y).unwrap() - 3.0).abs() < 1e-6);
}

#[test]
fn persistent_milp_objective_sweep() {
    let m = Model::new("knap");
    param!(m, v0 = 1.0);
    variable!(m, x0 >= 0.0, Int);
    variable!(m, x1 >= 0.0, Int);
    constraint!(m, cap, 3.0 * x0 + 4.0 * x1 <= 12.0);
    objective!(m, Max, v0 * x0 + 5.0 * x1);

    let mut solver = Gurobi.persistent();
    for v in [1.0, 2.0, 7.0] {
        v0.set_param_value(v);
        let s = solver.solve(&m, &GurobiOptions::default()).expect("persistent solve");
        let cold = Gurobi.solve(&m, &GurobiOptions::default()).expect("cold solve");
        assert_eq!(s.termination, TerminationStatus::Optimal, "v {v}");
        assert!((s.objective().unwrap() - cold.objective().unwrap()).abs() < 1e-6, "v {v}");
    }
}
