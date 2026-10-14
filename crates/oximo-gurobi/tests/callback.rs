//! Live callback-surface tests for the Gurobi 13 backend.

use oximo_core::prelude::*;
use oximo_gurobi::{
    Callback, CallbackLocation, CallbackMask, CbResult, Gurobi, GurobiOptions, GurobiPresolve,
    Where,
};
use oximo_solver::{PersistentSolver, Solver, UniversalOptionsExt};

#[derive(Default)]
struct Locations(Vec<CallbackLocation>);

impl Callback for Locations {
    fn callback(&mut self, where_: Where) -> CbResult {
        let location = match where_ {
            Where::Polling(_) => CallbackLocation::Polling,
            Where::PreSolve(_) => CallbackLocation::PreSolve,
            Where::Simplex(_) => CallbackLocation::Simplex,
            Where::MIP(_) => CallbackLocation::Mip,
            Where::MIPSol(_) => CallbackLocation::MipSol,
            Where::MIPNode(_) => CallbackLocation::MipNode,
            Where::Message(_) => CallbackLocation::Message,
            Where::Barrier(_) => CallbackLocation::Barrier,
            Where::MultiObj(_) => CallbackLocation::MultiObj,
            Where::IIS(_) => CallbackLocation::Iis,
            _ => return Err(std::io::Error::other("unexpected callback location").into()),
        };
        self.0.push(location);
        Ok(())
    }
}

#[test]
fn callback_solve_uses_common_result_collection() {
    let m = Model::new("callback");
    variable!(m, x[i in 0..4], Bin);
    constraint!(m, cap, sum!(x[i] for i in 0..4) <= 2.0);
    objective!(m, Max, sum!(x[i] for i in 0..4));

    let mut callback = Locations::default();
    let mut solver = Gurobi;
    let mask = CallbackMask::from(CallbackLocation::Polling) | CallbackLocation::Mip;
    let result = solver
        .solve_with_callback_filtered(
            &m,
            &GurobiOptions::default().presolve(GurobiPresolve::Off),
            &mut callback,
            mask,
        )
        .expect("callback solve");

    assert!(result.has_solution());
    assert!(callback.0.contains(&CallbackLocation::Mip), "Gurobi did not invoke a MIP callback");
    for location in &callback.0 {
        assert!(mask.contains(*location), "callback location {location:?} was not selected");
    }
}

struct FailingCallback;

struct RebindParameter<'a> {
    model: &'a Model,
    parameter: Expr<'a, oximo_expr::Constant>,
    value: f64,
    calls: usize,
}

impl Callback for RebindParameter<'_> {
    fn callback(&mut self, _: Where) -> CbResult {
        self.model.set_param(self.parameter, self.value)?;
        self.calls += 1;
        Ok(())
    }
}

#[test]
fn callback_parameter_rebinding_does_not_change_solved_snapshot() {
    let model = Model::new("callback parameters");
    param!(model, p = 2.0);
    variable!(model, 0.0 <= x <= 3.0);
    objective!(model, Max, p * x);
    let opts = GurobiOptions::default().presolve(GurobiPresolve::Off).threads(1);
    let mut callback = RebindParameter { model: &model, parameter: p, value: 4.0, calls: 0 };
    let first = Gurobi.solve_with_callback(&model, &opts, &mut callback).unwrap();
    assert!(callback.calls > 0);
    let value = first.value_of(x).unwrap().unwrap();
    assert_eq!(first.value_of(p * x), Ok(Some(2.0 * value)));
    assert_eq!(first.value_of_with_current_parameters(p * x), Ok(Some(4.0 * value)));

    let mut resident = Gurobi.persistent();
    let second = resident.solve(&model, &opts).unwrap();
    model.set_param(p, 6.0).unwrap();
    callback.value = 8.0;
    callback.calls = 0;
    let third = resident.solve_with_callback(&model, &opts, &mut callback).unwrap();
    assert!(callback.calls > 0);
    let value = third.value_of(x).unwrap().unwrap();
    assert_eq!(third.value_of(p * x), Ok(Some(6.0 * value)));
    assert_eq!(third.value_of_with_current_parameters(p * x), Ok(Some(8.0 * value)));
    assert_eq!(second.parameters.values(), &[4.0]);
    assert_eq!(first.parameters.values(), &[2.0]);
}

impl Callback for FailingCallback {
    fn callback(&mut self, _where: Where) -> CbResult {
        Err(std::io::Error::other("callback failure").into())
    }
}

#[test]
fn callback_errors_are_solver_errors() {
    let m = Model::new("callback_error");
    variable!(m, x, Bin);
    objective!(m, Max, x);

    let mut callback = FailingCallback;
    let mut solver = Gurobi;
    let error = solver
        .solve_with_callback(&m, &GurobiOptions::default(), &mut callback)
        .expect_err("callback should fail");
    assert!(error.to_string().contains("Problem with callback"), "{error}");
}

#[test]
fn persistent_callback_error_clears_and_rebuilds_resident_model() {
    let m = Model::new("persistent_callback_error");
    variable!(m, x, Bin);
    objective!(m, Max, x);

    let opts = GurobiOptions::default().presolve(GurobiPresolve::Off);
    let mut solver = Gurobi.persistent();
    let first = solver.solve(&m, &opts).expect("initial persistent solve");
    assert!(first.has_solution());

    let mut callback = FailingCallback;
    let error =
        solver.solve_with_callback(&m, &opts, &mut callback).expect_err("callback should fail");
    assert!(error.to_string().contains("Problem with callback"), "{error}");

    let cleared = solver.compute_iis().expect_err("callback failure should clear resident state");
    assert!(cleared.to_string().contains("no resident model"), "{cleared}");

    let recovered =
        solver.solve(&m, &opts).expect("persistent solve should rebuild after callback failure");
    assert!(recovered.has_solution());
}
