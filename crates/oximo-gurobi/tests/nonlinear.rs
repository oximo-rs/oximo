//! Live Gurobi tests for QP, NLP, and MINLP models.

use oximo_core::prelude::*;
use oximo_gurobi::{Gurobi, GurobiOptions};
use oximo_solver::{InfeasibilityDiagnosis, Solver};

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

fn assert_solved(r: &oximo_solver::SolverResult) {
    assert!(r.has_solution(), "termination = {:?}, primal = {:?}", r.termination, r.primal_status);
}

#[test]
fn qp_min_sum_of_squares() {
    // min x^2 + y^2 s.t. x + y >= 1.
    // Optimum at x = y = 0.5, objective = 0.5.
    let m = Model::new("qp");
    variable!(m, -10.0 <= x <= 10.0);
    variable!(m, -10.0 <= y <= 10.0);
    constraint!(m, c, x + y >= 1.0);
    objective!(m, Min, x.powi(2) + y.powi(2));

    let r = Gurobi.solve(&m, &GurobiOptions::default()).expect("solve");
    assert!(r.has_solution());
    let obj = r.objective().expect("obj");
    assert!(close(obj, 0.5, 1e-4), "obj = {obj}");
}

#[test]
fn nlp_with_sin_objective() {
    // min (x - 1)^2 + 0.1 * sin(x)^2 over x in [-3, 3].
    // Local minimum near x = 1, objective near 0.
    let m = Model::new("nlp_sin");
    variable!(m, -3.0 <= x <= 3.0);
    m.set_initial(x, 0.5);
    objective!(m, Min, (x - 1.0).powi(2) + 0.1 * x.sin().powi(2));

    let r = Gurobi.solve(&m, &GurobiOptions::default()).expect("solve");
    assert!(r.has_solution());
    let primal_x = r.value(VarId(0)).expect("primal");
    assert!(close(primal_x, 1.0, 0.1), "x = {primal_x}");
}

#[test]
fn nlp_with_abs_objective() {
    // min |x - 2| over x in [-10, 10]. Optimum at x = 2, objective = 0.
    let m = Model::new("nlp_abs");
    variable!(m, -10.0 <= x <= 10.0);
    m.set_initial(x, 0.5);
    objective!(m, Min, (x - 2.0).abs());

    let r = Gurobi.solve(&m, &GurobiOptions::default()).expect("solve");
    assert_solved(&r);
    let primal_x = r.value(VarId(0)).expect("primal");
    assert!(close(primal_x, 2.0, 1e-3), "x = {primal_x}");
    let obj = r.objective().expect("obj");
    assert!(close(obj, 0.0, 1e-3), "obj = {obj}");
}

#[test]
fn minlp_binary_with_log() {
    // Binary b, continuous x in [0.1, 10]. Min (x - 1)^2 + b * log(1 + x).
    // Optimal: b = 0, x = 1, objective = 0.
    let m = Model::new("minlp_log");
    variable!(m, b, Bin);
    variable!(m, 0.1 <= x <= 10.0);
    m.set_initial(x, 0.5);
    objective!(m, Min, (x - 1.0).powi(2) + b * (1.0 + x).log());

    let r = Gurobi.solve(&m, &GurobiOptions::default()).expect("solve");
    assert!(r.has_solution());
    let obj = r.objective().expect("obj");
    assert!(close(obj, 0.0, 1e-3), "obj = {obj}");
}

// Division lowering

#[test]
fn div_by_linear_denominator() {
    // x / (y + z) == 3, with x = 12 and z = 1 fixed -> y + 1 = 4 -> y = 3.
    let m = Model::new("div_linear");
    variable!(m, x);
    m.fix(x, 12.0);
    variable!(m, 0.1 <= y <= 100.0);
    variable!(m, z);
    m.fix(z, 1.0);
    constraint!(m, c, x / (y + z) == 3.0);
    objective!(m, Min, y);

    let sol = Gurobi.solve(&m, &GurobiOptions::default()).expect("solve");
    assert_solved(&sol);
    let yv = sol.value(VarId(1)).expect("primal y");
    assert!(close(yv, 3.0, 1e-4), "y = {yv}");
}

#[test]
fn div_by_negative_denominator() {
    // x / d == -3, with x = 12 fixed and d in [-100, -0.1] -> d = -4.
    // A `pow(den, -1)` lowering could not represent a negative denominator,
    // the bilinear `d * recip == 1` pin can.
    let m = Model::new("div_negative");
    variable!(m, x);
    m.fix(x, 12.0);
    variable!(m, -100.0 <= d <= -0.1);
    constraint!(m, c, x / d == -3.0);
    objective!(m, Min, d);

    let r = Gurobi.solve(&m, &GurobiOptions::default()).expect("solve");
    assert_solved(&r);
    let dv = r.value(VarId(1)).expect("primal d");
    assert!(close(dv, -4.0, 1e-4), "d = {dv}");
}

#[test]
fn div_constant_numerator() {
    // 6 / d == 2 -> d = 3. Exercises the constant-numerator fold (`6 * recip`,
    // which stays linear in `recip` rather than materializing a product).
    let m = Model::new("div_const_num");
    variable!(m, 0.1 <= d <= 100.0);
    constraint!(m, c, 6.0 / d == 2.0);
    objective!(m, Min, d);

    let r = Gurobi.solve(&m, &GurobiOptions::default()).expect("solve");
    assert_solved(&r);
    let dv = r.value(VarId(0)).expect("primal d");
    assert!(close(dv, 3.0, 1e-4), "d = {dv}");
}

#[test]
fn div_by_quadratic_denominator() {
    // x / (x * y) == 0.5 reduces to 1 / y == 0.5 -> y = 2 for any nonzero x.
    // The quadratic denominator is first materialized into an aux variable so
    // the reciprocal pin stays bilinear rather than cubic.
    let m = Model::new("div_quadratic");
    variable!(m, 1.0 <= x <= 10.0);
    variable!(m, 0.1 <= y <= 10.0);
    constraint!(m, c, x / (x * y) == 0.5);
    objective!(m, Min, x);

    let r = Gurobi.solve(&m, &GurobiOptions::default()).expect("solve");
    assert_solved(&r);
    let yv = r.value(VarId(1)).expect("primal y");
    assert!(close(yv, 2.0, 1e-3), "y = {yv}");
}

#[test]
fn div_by_zero_constant_errors() {
    // A literal zero denominator survives construction as a `Div` node (only
    // nonzero constants are folded into the linear path), so lowering must
    // reject it rather than emit an infeasible `0 * recip == 1`.
    let m = Model::new("div_zero");
    variable!(m, 0.0 <= x <= 10.0);
    objective!(m, Min, x / 0.0);

    let err = Gurobi.solve(&m, &GurobiOptions::default()).expect_err("expected error");
    assert!(err.to_string().contains("division by zero"), "err = {err}");
}

#[test]
fn nonlinear_iis_maps_generated_abs_definition() {
    let m = Model::new("nl_iis_abs");
    variable!(m, x);
    let conflict = constraint!(m, impossible, x.abs() <= -1.0);
    objective!(m, Min, x);

    let iis = Gurobi.compute_iis(&m, &GurobiOptions::default()).expect("compute nonlinear IIS");
    assert!(iis.constraints.contains(&conflict));
}

#[test]
fn unsupported_operator_errors_before_environment_or_optimize() {
    let m = Model::new("unsupported_asin");
    variable!(m, -0.5 <= x <= 0.5);
    objective!(m, Min, x.asin());
    assert!(matches!(
        Gurobi.solve(&m, &GurobiOptions::default()),
        Err(oximo_solver::SolverError::UnsupportedNonlinearOperator {
            backend: "Gurobi",
            operator: "asin"
        })
    ));
}

#[test]
fn newly_supported_native_unary_operators_solve_at_fixed_point() {
    let point: f64 = 0.5;

    macro_rules! check_unary {
        ($method:ident, $expected:expr) => {{
            let m = Model::new(concat!("native_", stringify!($method)));
            variable!(m, -5.0 <= x <= 5.0);
            m.fix(x, point);
            objective!(m, Min, x.$method());

            let result = Gurobi
                .solve(&m, &GurobiOptions::default())
                .unwrap_or_else(|error| panic!("{} solve failed: {error}", stringify!($method)));
            assert_solved(&result);
            let actual = result.objective().expect("objective");
            assert!(
                close(actual, $expected, 1e-6),
                "{}({point}) = {actual}, expected {}",
                stringify!($method),
                $expected
            );
        }};
    }

    check_unary!(sqrt, point.sqrt());
    check_unary!(exp2, point.exp2());
    check_unary!(log2, point.log2());
    check_unary!(log10, point.log10());
    check_unary!(tan, point.tan());
    check_unary!(tanh, point.tanh());
}

#[test]
fn nested_min_max_use_native_general_constraints() {
    let m = Model::new("nested_extrema");
    variable!(m, -5.0 <= x <= 5.0);
    variable!(m, -5.0 <= y <= 5.0);
    m.fix(x, -2.0);
    m.fix(y, 3.0);
    objective!(m, Min, x.min(y).max(x + 1.0));

    let result = Gurobi.solve(&m, &GurobiOptions::default()).expect("solve nested extrema");
    assert_solved(&result);
    assert!(close(result.objective().unwrap(), -1.0, 1e-8));
}
