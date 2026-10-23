//! Each persistent update is checked against the same adapter's cold solve.
#![cfg(any(
    feature = "highs",
    feature = "clarabel",
    feature = "pounce",
    feature = "scip",
    feature = "gurobi",
    feature = "mosek"
))]

use oximo_core::prelude::*;
use oximo_solver::{PersistentSolver, Solver};

fn parity<S: Solver>(
    cold: &mut S,
    warm: &mut impl Solver<Options = S::Options>,
    m: &Model,
    opts: &S::Options,
) {
    let a = warm.solve(m, opts).unwrap();
    let b = cold.solve(m, opts).unwrap();
    assert!(a.has_solution(), "resident: {:?}", a.termination);
    assert!(b.has_solution(), "cold: {:?}", b.termination);

    let (a, b) = (a.objective().unwrap(), b.objective().unwrap());
    assert!((a - b).abs() <= 2e-4 * a.abs().max(b.abs()).max(1.0), "{a} vs {b}");
}

fn linear_and_quadratic<S: PersistentSolver>(mut cold: S, opts: S::Options) {
    let m = Model::new("parameter updates");

    param!(m, p = 1.0);
    param!(m, cap = 4.0);
    param!(m, limit = 5.0);
    param!(m, unused = 7.0);
    variable!(m,0.0<=x<=limit);
    variable!(m,0.0<=y<=3.0);
    let row = constraint!(m, row, p * x + y <= cap);
    objective!(m, Max, 2.0 * x + y);

    let mut warm = cold.persistent();
    let first = warm.solve(&m, &opts).unwrap();
    unused.set_param_value(9.0).unwrap();
    let reused = warm.solve(&m, &opts).unwrap();
    assert_eq!(reused.value_of(unused), Ok(Some(9.0)));
    assert_eq!(first.value_of(unused), Ok(Some(7.0)));

    parity(&mut cold, &mut warm, &m, &opts);
    for value in [1.0, 0.0, -1.0, 2.0] {
        p.set_param_value(value).unwrap();
        cap.set_param_value(3.0 + value).unwrap();
        limit.set_param_value(4.0 + value.abs()).unwrap();
        unused.set_param_value(value).unwrap();
        parity(&mut cold, &mut warm, &m, &opts);
    }
    assert_eq!(first.value_of(p), Ok(Some(1.0)));
    assert_eq!(first.value_of(unused), Ok(Some(7.0)));

    m.set_constraint_active(row, false).unwrap();
    parity(&mut cold, &mut warm, &m, &opts);
    m.set_constraint_active(row, true).unwrap();
    parity(&mut cold, &mut warm, &m, &opts);
    m.set_initial(x, 1.0).unwrap();
    parity(&mut cold, &mut warm, &m, &opts);
    objective!(m, Min, p * x * x - 2.0 * x + y * y);
    for value in [2.0, 1.0, 0.0, 3.0] {
        p.set_param_value(value).unwrap();
        parity(&mut cold, &mut warm, &m, &opts);
    }
    let fork = m.fork();
    parity(&mut cold, &mut warm, &fork, &opts);
}

#[cfg(any(
    feature = "clarabel",
    feature = "pounce",
    feature = "scip",
    feature = "gurobi",
    feature = "mosek"
))]
fn conic<S: PersistentSolver>(mut cold: S, opts: S::Options) {
    let m = Model::new("cone updates");

    param!(m, p = 1.0);
    param!(m, radius = 3.0);
    variable!(m,-4.0<=x<=4.0);
    variable!(m,-4.0<=y<=4.0);
    soc_constraint!(m, cone, [p * x, y] <= radius);
    objective!(m, Max, x + y);

    let mut warm = cold.persistent();
    for value in [1.0, 2.0, 0.0, -1.0, 3.0] {
        p.set_param_value(value).unwrap();
        radius.set_param_value(2.0 + value.abs()).unwrap();
        parity(&mut cold, &mut warm, &m, &opts);
    }
}

#[cfg(any(feature = "pounce", feature = "scip", feature = "gurobi"))]
fn nonlinear<S: PersistentSolver>(mut cold: S, opts: S::Options) {
    let m = Model::new("nonlinear updates");

    param!(m, p = 1.0);
    param!(m, cap = 3.0);
    variable!(m,0.1<=x<=2.0);
    constraint!(m, row, p * x.exp() <= cap);
    objective!(m, Max, x);

    let mut warm = cold.persistent();
    for value in [1.0, 2.0, 0.0, 1.5, 1.0] {
        p.set_param_value(value).unwrap();
        parity(&mut cold, &mut warm, &m, &opts);
    }

    let first = Model::new("colliding nonlinear IDs");
    variable!(first,0.1<=first_x<=2.0);
    constraint!(first, row, first_x.exp() <= 1.5);
    objective!(first, Max, first_x);
    parity(&mut cold, &mut warm, &first, &opts);

    let second = Model::new("colliding nonlinear IDs");
    variable!(second,0.1<=second_x<=2.0);
    constraint!(second, row, second_x.ln() <= 1.5);
    objective!(second, Max, second_x);

    parity(&mut cold, &mut warm, &second, &opts);
}

#[cfg(any(feature = "pounce", feature = "scip", feature = "gurobi", feature = "mosek"))]
fn quadratic_rows<S: PersistentSolver>(mut cold: S, opts: S::Options) {
    let model = Model::new("quadratic row clearing");

    param!(model, scale = 1.0);
    variable!(model,0.0<=x<=3.0);
    variable!(model,0.0<=y<=3.0);
    constraint!(model, row, scale * x * x + y <= 4.0);
    objective!(model, Max, x + 0.2 * y);

    let mut warm = cold.persistent();
    for value in [1.0, 0.0, 2.0, 1.0] {
        scale.set_param_value(value).unwrap();
        parity(&mut cold, &mut warm, &model, &opts);
    }
}

#[cfg(feature = "highs")]
#[test]
fn highs() {
    linear_and_quadratic(oximo_highs::Highs, oximo_highs::HighsOptions::default());
}

#[cfg(feature = "clarabel")]
#[test]
fn clarabel() {
    let opts = oximo_clarabel::ClarabelOptions::default()
        .presolve_enable(false)
        .input_sparse_dropzeros(false);
    linear_and_quadratic(oximo_clarabel::Clarabel, opts.clone());
    conic(oximo_clarabel::Clarabel, opts);
}

#[cfg(feature = "pounce")]
#[test]
fn pounce() {
    linear_and_quadratic(oximo_pounce::Pounce, oximo_pounce::PounceOptions::default());
    conic(oximo_pounce::Pounce, oximo_pounce::PounceOptions::default());
    nonlinear(oximo_pounce::Pounce, oximo_pounce::PounceOptions::default());
    quadratic_rows(oximo_pounce::Pounce, oximo_pounce::PounceOptions::default());
}

#[cfg(feature = "scip")]
#[test]
fn scip() {
    linear_and_quadratic(oximo_scip::Scip::new(), oximo_scip::ScipOptions::default());
    conic(oximo_scip::Scip::new(), oximo_scip::ScipOptions::default());
    nonlinear(oximo_scip::Scip::new(), oximo_scip::ScipOptions::default());
    quadratic_rows(oximo_scip::Scip::new(), oximo_scip::ScipOptions::default());

    let model = Model::new("coefficient cancellation");
    param!(model, scale = 1e16);
    variable!(model,0.0<=x<=2.0);
    constraint!(model, row, scale * x <= 1.0);
    objective!(model, Max, x);

    let mut warm = oximo_scip::Scip::new().persistent();
    let options = oximo_scip::ScipOptions::default();
    warm.solve(&model, &options).unwrap();
    scale.set_param_value(1.0).unwrap();
    parity(&mut oximo_scip::Scip::new(), &mut warm, &model, &options);
    assert_eq!(warm.build_count(), 2, "an inexact additive delta must rebuild");
}

#[cfg(feature = "gurobi")]
#[test]
fn gurobi() {
    linear_and_quadratic(oximo_gurobi::Gurobi, oximo_gurobi::GurobiOptions::default());
    conic(oximo_gurobi::Gurobi, oximo_gurobi::GurobiOptions::default());
    nonlinear(oximo_gurobi::Gurobi, oximo_gurobi::GurobiOptions::default());
    quadratic_rows(oximo_gurobi::Gurobi, oximo_gurobi::GurobiOptions::default());
}

#[cfg(feature = "mosek")]
#[test]
fn mosek() {
    linear_and_quadratic(oximo_mosek::Mosek, oximo_mosek::MosekOptions::default());
    conic(oximo_mosek::Mosek, oximo_mosek::MosekOptions::default());
    quadratic_rows(oximo_mosek::Mosek, oximo_mosek::MosekOptions::default());

    let m = Model::new("PSD updates");
    param!(m, p = 1.0);
    variable!(m,0.0<=t<=5.0);
    psd_constraint!(
        m,
        psd,
        SymmetricMatrix::from_upper_triangle(2, [t.erase(), p.erase(), t.erase()])
    );
    objective!(m, Min, t);

    let mut cold = oximo_mosek::Mosek;
    let mut warm = cold.persistent();
    for value in [1.0, 2.0, 0.0, -1.0, 3.0] {
        p.set_param_value(value).unwrap();
        parity(&mut cold, &mut warm, &m, &oximo_mosek::MosekOptions::default());
    }
}
