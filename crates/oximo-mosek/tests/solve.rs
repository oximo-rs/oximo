use oximo_core::prelude::*;
use oximo_mosek::{Mosek, MosekOptions, solve};
use oximo_solver::{PersistentSolver, Solver, SolverError, TerminationStatus};

fn close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-6, "actual={actual}, expected={expected}");
}

#[test]
fn lp_range_duals_reduced_costs_and_maximization() {
    let model = Model::new("lp");
    variable!(model, x >= 0.0);
    variable!(model, y >= 0.0);
    constraint!(model, band, 1.0 <= x + y <= 3.0);
    objective!(model, Max, 3.0 * x + y + 2.0);

    let result = solve(&model, &MosekOptions::default()).unwrap();
    assert_eq!(result.termination, TerminationStatus::Optimal);
    close(result.value_of(x).unwrap().unwrap(), 3.0);
    close(result.value_of(y).unwrap().unwrap(), 0.0);
    close(result.objective().unwrap(), 11.0);
    close(result.dual_of(model.constraint_handle("band").unwrap()).unwrap().unwrap(), 3.0);
    close(*result.reduced_costs.get(&y.var_id().unwrap()).unwrap(), -2.0);
}

#[test]
fn milp_and_quadratic_model_classes() {
    let milp = Model::new("milp");
    variable!(milp, x, Bin);
    variable!(milp, y, Bin);
    constraint!(milp, cap, 2.0 * x + 3.0 * y <= 4.0);
    objective!(milp, Max, 3.0 * x + 4.0 * y);
    let result = solve(&milp, &MosekOptions::default()).unwrap();
    close(result.value_of(x).unwrap().unwrap(), 0.0);
    close(result.value_of(y).unwrap().unwrap(), 1.0);

    let qp = Model::new("qp");
    variable!(qp, qx >= 0.0);
    variable!(qp, qy >= 0.0);
    constraint!(qp, sum, qx + qy == 1.0);
    objective!(qp, Min, qx.powi(2) + qy.powi(2));
    let result = solve(&qp, &MosekOptions::default()).unwrap();
    close(result.value_of(qx).unwrap().unwrap(), 0.5);
    close(result.value_of(qy).unwrap().unwrap(), 0.5);
    close(result.objective().unwrap(), 0.5);

    let miqp = Model::new("miqp");
    variable!(miqp, 0.0 <= z <= 5.0, Int);
    objective!(miqp, Min, (z - 2.2).powi(2));
    let result = solve(&miqp, &MosekOptions::default()).unwrap();
    close(result.value_of(z).unwrap().unwrap(), 2.0);
    close(result.objective().unwrap(), 0.04);
}

#[test]
fn qcp_and_miqcp() {
    let qcp = Model::new("qcp");
    variable!(qcp, x >= 0.0);
    constraint!(qcp, ball, (x - 2.0).powi(2) <= 1.0);
    objective!(qcp, Min, x);
    let result = solve(&qcp, &MosekOptions::default()).unwrap();
    close(result.value_of(x).unwrap().unwrap(), 1.0);

    let miqcp = Model::new("miqcp");
    variable!(miqcp, 0.0 <= z <= 4.0, Int);
    constraint!(miqcp, ball, (z - 2.0).powi(2) <= 1.0);
    objective!(miqcp, Max, z);
    let result = solve(&miqcp, &MosekOptions::default()).unwrap();
    close(result.value_of(z).unwrap().unwrap(), 3.0);
}

#[test]
fn explicit_detected_and_mixed_integer_socp() {
    let explicit = Model::new("explicit");
    variable!(explicit, x);
    variable!(explicit, y);
    variable!(explicit, t >= 0.0);
    constraint!(explicit, fix_x, x == 3.0);
    constraint!(explicit, fix_y, y == 4.0);
    let cone = explicit.add_soc_constraint("cone", [x, y], t);
    objective!(explicit, Min, t);
    let result = solve(&explicit, &MosekOptions::default()).unwrap();
    close(result.value_of(t).unwrap().unwrap(), 5.0);
    close(result.soc_dual_of(cone).unwrap().unwrap(), 1.0);

    let detected = Model::new("detected");
    variable!(detected, dx);
    variable!(detected, dy);
    variable!(detected, dt >= 0.0);
    constraint!(detected, fix_x, dx == 3.0);
    constraint!(detected, fix_y, dy == 4.0);
    constraint!(detected, cone, dx.powi(2) + dy.powi(2) <= dt.powi(2));
    objective!(detected, Min, dt);
    let result = solve(&detected, &MosekOptions::default()).unwrap();
    close(result.value_of(dt).unwrap().unwrap(), 5.0);

    let mixed = Model::new("mixed_socp");
    variable!(mixed, 3.0 <= ix <= 3.0, Int);
    variable!(mixed, iy);
    variable!(mixed, it >= 0.0);
    constraint!(mixed, fix_y, iy == 4.0);
    mixed.add_soc_constraint("cone", [ix, iy], it);
    objective!(mixed, Min, it);
    let result = solve(&mixed, &MosekOptions::default()).unwrap();
    close(result.value_of(it).unwrap().unwrap(), 5.0);
}

#[test]
fn infeasible_unbounded_and_public_solver_trait() {
    let infeasible = Model::new("infeasible");
    variable!(infeasible, x);
    constraint!(infeasible, lo, x >= 1.0);
    constraint!(infeasible, hi, x <= 0.0);
    objective!(infeasible, Min, x);
    let result = solve(&infeasible, &MosekOptions::default()).unwrap();
    assert_eq!(result.termination, TerminationStatus::Infeasible);
    assert!(!result.has_solution());

    let unbounded = Model::new("unbounded");
    variable!(unbounded, y);
    objective!(unbounded, Min, y);
    let result = solve(&unbounded, &MosekOptions::default()).unwrap();
    assert_eq!(result.termination, TerminationStatus::Unbounded);

    let mut solver = Mosek;
    let result = solver.solve(&infeasible, &MosekOptions::default()).unwrap();
    assert_eq!(solver.name(), "MOSEK");
    assert_eq!(result.solver_name.as_deref(), Some("MOSEK"));
}

#[test]
fn feasibility_model_returns_a_point() {
    let model = Model::new("feasibility");
    variable!(model, x);
    constraint!(model, bounds, 2.0 <= x <= 3.0);
    objective!(model, Feasibility);

    let result = solve(&model, &MosekOptions::default()).unwrap();
    assert_eq!(result.termination, TerminationStatus::Optimal);
    assert!((2.0..=3.0).contains(&result.value_of(x).unwrap().unwrap()));
    close(result.objective().unwrap(), 0.0);
}

#[test]
fn unsupported_nonlinear_and_semi_domains() {
    let nonlinear = Model::new("nonlinear");
    variable!(nonlinear, x);
    objective!(nonlinear, Min, x.sin());
    assert!(matches!(
        solve(&nonlinear, &MosekOptions::default()),
        Err(SolverError::UnsupportedKind(ModelKind::NLP))
    ));

    let mixed_nonlinear = Model::new("mixed_nonlinear");
    variable!(mixed_nonlinear, z, Int);
    objective!(mixed_nonlinear, Min, z.sin());
    assert!(matches!(
        solve(&mixed_nonlinear, &MosekOptions::default()),
        Err(SolverError::UnsupportedKind(ModelKind::MINLP))
    ));

    let semi = Model::new("semi");
    variable!(semi, s <= 10.0, SemiCont(2.0));
    objective!(semi, Min, s);
    let error = solve(&semi, &MosekOptions::default()).unwrap_err();
    assert!(error.to_string().contains("semi-continuous or semi-integer"));

    let semi_integer = Model::new("semi_integer");
    variable!(semi_integer, si <= 10.0, SemiInt(2.0));
    objective!(semi_integer, Min, si);
    let error = solve(&semi_integer, &MosekOptions::default()).unwrap_err();
    assert!(error.to_string().contains("semi-continuous or semi-integer"));
}

#[test]
fn unsupported_sos_reports_explicit_reformulation_hint() {
    let model = Model::new("mosek_sos");
    variable!(model, 0.0 <= x <= 1.0);
    variable!(model, 0.0 <= y <= 1.0);
    objective!(model, Max, x + y);
    sos_constraint!(model, choice, SOS1, [x, y]);

    let error = solve(&model, &MosekOptions::default()).unwrap_err();
    assert!(matches!(error, SolverError::UnsupportedSos));
    assert!(error.to_string().contains("Model::to_reformulated_sos_model"));
}

#[test]
fn native_indicators_cover_active_inactive_equality_and_range() {
    for (fixed, expected) in [(1.0, 3.0), (0.0, 10.0)] {
        let model = Model::new("indicator");
        variable!(model, b, Binary);
        variable!(model, 0.0 <= x <= 10.0);
        constraint!(model, fix, b == fixed);
        indicator_constraint!(model, cap, b == 1 => x <= 3.0);
        objective!(model, Max, x);
        let result = solve(&model, &MosekOptions::default()).unwrap();
        close(result.value_of(x).unwrap().unwrap(), expected);
    }

    let model = Model::new("indicator_forms");
    variable!(model, b, Binary);
    variable!(model, -10.0 <= x <= 10.0);
    variable!(model, -10.0 <= y <= 10.0);
    constraint!(model, fix, b == 0.0);
    indicator_constraint!(model, equal, b == 0 => x == 2.0);
    indicator_constraint!(model, band, b == 0 => -1.0 <= y <= 4.0);
    objective!(model, Max, x + y);
    let result = solve(&model, &MosekOptions::default()).unwrap();
    close(result.value_of(x).unwrap().unwrap(), 2.0);
    close(result.value_of(y).unwrap().unwrap(), 4.0);
}

#[test]
fn mosek_reports_nonconvex_quadratic_data() {
    let model = Model::new("nonconvex");
    variable!(model, -1.0 <= x <= 1.0);
    objective!(model, Min, -x.powi(2));
    let error = solve(&model, &MosekOptions::default()).unwrap_err();
    let message = error.to_string();
    assert!(message.contains("MOSEK:"), "{message}");
    assert!(
        message.to_ascii_lowercase().contains("convex")
            || message.to_ascii_lowercase().contains("semidefinite"),
        "{message}"
    );
}

#[test]
fn branch_limit_keeps_constructed_incumbent() {
    let model = Model::new("limited_mip");
    let items = Set::range(0..40);
    variable!(model, x[i in items], Bin);
    for i in 0..40 {
        model.set_initial(x[i], 0.0).unwrap();
    }
    constraint!(
        model,
        capacity,
        sum!(f64::from(u32::try_from(i % 9 + 3).unwrap()) * x[i] for i in 0..40) <= 105.0
    );
    objective!(
        model,
        Max,
        sum!(f64::from(u32::try_from(i * 17 % 31 + 1).unwrap()) * x[i] for i in 0..40)
    );

    let options =
        MosekOptions::default().presolve_use(mosek::Presolvemode::OFF).mio_max_num_branches(0);
    let result = solve(&model, &options).unwrap();
    assert_eq!(result.termination, TerminationStatus::NodeLimit);
    assert!(result.has_solution());
    // With zero allowed branches MOSEK may stop after constructing an incumbent,
    // before a best bound or relative gap has been computed.
}

#[test]
fn persistent_updates_linear_objective_and_bounds() {
    let model = Model::new("persistent_lp");
    param!(model, price = 1.0);
    variable!(model, 0.0 <= x <= 10.0);
    objective!(model, Max, price * x);

    let mut solver = Mosek.persistent();
    for (coefficient, upper) in [(1.0, 10.0), (3.0, 4.0), (2.0, 7.0)] {
        price.set_param_value(coefficient);
        model.unfix_var(x.var_id().unwrap(), 0.0, upper);
        let resident = solver.solve(&model, &MosekOptions::default()).unwrap();
        let cold = Mosek.solve(&model, &MosekOptions::default()).unwrap();
        assert_eq!(resident.termination, TerminationStatus::Optimal);
        close(resident.value_of(x).unwrap().unwrap(), cold.value_of(x).unwrap().unwrap());
        close(resident.objective().unwrap(), cold.objective().unwrap());
    }
}

#[test]
fn persistent_rebuilds_after_linear_row_change() {
    let model = Model::new("persistent_rebuild");
    param!(model, capacity = 5.0);
    variable!(model, x >= 0.0);
    constraint!(model, cap, x <= capacity);
    objective!(model, Max, x);

    let mut solver = Mosek.persistent();
    for bound in [5.0, 2.0, 9.0] {
        capacity.set_param_value(bound);
        let resident = solver.solve(&model, &MosekOptions::default()).unwrap();
        let cold = Mosek.solve(&model, &MosekOptions::default()).unwrap();
        close(resident.value_of(x).unwrap().unwrap(), cold.value_of(x).unwrap().unwrap());
        close(resident.objective().unwrap(), cold.objective().unwrap());
    }
}
