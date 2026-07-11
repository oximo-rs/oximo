//! Enzyme-feature tests: gradients, Jacobians, and Hessians of the Lagrangian
//! against finite differences and exact quadratic extraction.
//!
//! Run with:
//! ```text
//! RUSTFLAGS="-Zautodiff=Enable" cargo +nightly test -p oximo-autodiff --features enzyme --profile enzyme
//! ```
#![cfg(feature = "enzyme")]
#![allow(clippy::cast_precision_loss)]

use oximo_autodiff::{AutodiffError, NlpEvaluator, gradient_at};
use oximo_core::Model;
use oximo_core::prelude::*;

fn assert_close(got: f64, want: f64, tol: f64, what: &str) {
    let denom = want.abs().max(1.0);
    assert!(((got - want) / denom).abs() < tol, "{what}: got {got}, want {want}");
}

/// Central finite-difference gradient of the evaluator's objective.
fn fd_objective_gradient(ev: &NlpEvaluator, x: &[f64]) -> Vec<f64> {
    let h = 1e-6;
    (0..x.len())
        .map(|i| {
            let mut xp = x.to_vec();
            let mut xm = x.to_vec();
            xp[i] += h;
            xm[i] -= h;
            (ev.eval_objective(&xp) - ev.eval_objective(&xm)) / (2.0 * h)
        })
        .collect()
}

/// Dense gradient of the Lagrangian function via the evaluator's first-derivative callbacks.
fn lagrangian_gradient(ev: &NlpEvaluator, x: &[f64], sigma: f64, lambda: &[f64]) -> Vec<f64> {
    let n = ev.num_variables();
    let mut g = vec![0.0; n];
    ev.eval_objective_gradient(x, &mut g);
    for v in &mut g {
        *v *= sigma;
    }
    let mut jac = vec![0.0; ev.jacobian_structure().len()];
    ev.eval_constraint_jacobian(x, &mut jac);
    for (&(row, col), v) in ev.jacobian_structure().iter().zip(&jac) {
        g[col] += lambda[row] * v;
    }
    g
}

/// Assemble the dense symmetric Hessian from the evaluator's sparse lower
/// triangle.
fn dense_hessian(ev: &NlpEvaluator, x: &[f64], sigma: f64, lambda: &[f64]) -> Vec<Vec<f64>> {
    let n = ev.num_variables();
    let mut vals = vec![0.0; ev.hessian_lagrangian_structure().len()];
    ev.eval_hessian_lagrangian(x, sigma, lambda, &mut vals);
    let mut dense = vec![vec![0.0; n]; n];
    for (&(r, c), &v) in ev.hessian_lagrangian_structure().iter().zip(&vals) {
        dense[r][c] += v;
        if r != c {
            dense[c][r] += v;
        }
    }
    dense
}

#[test]
fn objective_gradient_matches_fd() {
    let m = Model::new("grad");
    variable!(m, -5.0 <= x <= 5.0);
    variable!(m, -5.0 <= y <= 5.0);
    objective!(m, Min, x.sin() * y + x.powi(3) / y + 2.0 * x - 0.5 * y);

    let ev = NlpEvaluator::new(&m).unwrap();
    for point in [[0.8, 1.7], [-1.2, 0.4], [2.5, -3.0]] {
        let mut grad = vec![0.0; 2];
        ev.eval_objective_gradient(&point, &mut grad);
        let fd = fd_objective_gradient(&ev, &point);
        for i in 0..2 {
            assert_close(grad[i], fd[i], 1e-6, &format!("grad[{i}] at {point:?}"));
        }
    }
}

#[test]
fn constraint_jacobian_matches_fd() {
    let m = Model::new("jac");
    variable!(m, -5.0 <= x <= 5.0);
    variable!(m, -5.0 <= y <= 5.0);
    objective!(m, Min, x + y);
    constraint!(m, lin, 2.0 * x + 3.0 * y <= 10.0);
    constraint!(m, quad, x.powi(2) + x * y <= 5.0);
    constraint!(m, nl, x.sin() * y.exp() <= 1.0);

    let ev = NlpEvaluator::new(&m).unwrap();
    let point = [0.7, 1.3];
    let mut jac = vec![0.0; ev.jacobian_structure().len()];
    ev.eval_constraint_jacobian(&point, &mut jac);

    let h = 1e-6;
    let m_con = ev.num_constraints();
    for (&(row, col), &v) in ev.jacobian_structure().iter().zip(&jac) {
        let mut xp = point.to_vec();
        let mut xm = point.to_vec();
        xp[col] += h;
        xm[col] -= h;
        let mut gp = vec![0.0; m_con];
        let mut gm = vec![0.0; m_con];
        ev.eval_constraint(&xp, &mut gp);
        ev.eval_constraint(&xm, &mut gm);
        let fd = (gp[row] - gm[row]) / (2.0 * h);
        assert_close(v, fd, 1e-6, &format!("jac[{row},{col}]"));
    }
}

#[test]
fn hessian_exact_on_quadratic() {
    // f = 3x^2 + xy + y^2 + 2x - y, Hessian [[6, 1], [1, 2]].
    let m = Model::new("qp");
    variable!(m, -5.0 <= x <= 5.0);
    variable!(m, -5.0 <= y <= 5.0);
    objective!(m, Min, 3.0 * x.powi(2) + x * y + y.powi(2) + 2.0 * x - y);

    let ev = NlpEvaluator::new(&m).unwrap();
    assert_eq!(ev.hessian_lagrangian_structure(), &[(0, 0), (1, 0), (1, 1)]);
    let mut vals = vec![0.0; 3];
    ev.eval_hessian_lagrangian(&[0.3, -0.4], 2.0, &[], &mut vals);
    assert_close(vals[0], 12.0, 1e-14, "H[0,0] * sigma");
    assert_close(vals[1], 2.0, 1e-14, "H[1,0] * sigma");
    assert_close(vals[2], 4.0, 1e-14, "H[1,1] * sigma");
}

fn lagrangian_test_model() -> Model {
    let m = Model::new("lagr");
    variable!(m, -5.0 <= x <= 5.0);
    variable!(m, -5.0 <= y <= 5.0);
    objective!(m, Min, x.sin() * y + x.powi(3));
    constraint!(m, quad, x.powi(2) + y.powi(2) <= 40.0);
    constraint!(m, nl, x * y.exp() >= 1.0);
    constraint!(m, lin, x + y <= 10.0);
    m
}

#[test]
fn hessian_lagrangian_matches_fd_of_gradient() {
    let m = lagrangian_test_model();
    let ev = NlpEvaluator::new(&m).unwrap();
    let x = [0.9, -1.1];
    let sigma = 0.7;
    let lambda = [1.3, -0.6, 2.0];

    let dense = dense_hessian(&ev, &x, sigma, &lambda);
    let h = 1e-5;
    for j in 0..2 {
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        xp[j] += h;
        xm[j] -= h;
        let gp = lagrangian_gradient(&ev, &xp, sigma, &lambda);
        let gm = lagrangian_gradient(&ev, &xm, sigma, &lambda);
        for i in 0..2 {
            let fd = (gp[i] - gm[i]) / (2.0 * h);
            assert_close(dense[i][j], fd, 1e-5, &format!("H[{i},{j}]"));
        }
    }
}

#[test]
fn hessian_is_linear_in_sigma_and_lambda() {
    let m = lagrangian_test_model();
    let ev = NlpEvaluator::new(&m).unwrap();
    let x = [0.9, -1.1];
    let nnz = ev.hessian_lagrangian_structure().len();

    let eval = |sigma: f64, lambda: [f64; 3]| {
        let mut vals = vec![0.0; nnz];
        ev.eval_hessian_lagrangian(&x, sigma, &lambda, &mut vals);
        vals
    };

    let sigma = 0.7;
    let lambda = [1.3, -0.6, 2.0];
    let combined = eval(sigma, lambda);
    let obj = eval(1.0, [0.0; 3]);
    let parts: Vec<Vec<f64>> = (0..3)
        .map(|i| {
            let mut l = [0.0; 3];
            l[i] = 1.0;
            eval(0.0, l)
        })
        .collect();
    for k in 0..nnz {
        let want = sigma * obj[k] + lambda.iter().zip(&parts).map(|(l, p)| l * p[k]).sum::<f64>();
        assert_close(combined[k], want, 1e-10, &format!("linearity at nnz {k}"));
    }
}

#[test]
fn repeated_evaluations_are_deterministic() {
    let m = lagrangian_test_model();
    let ev = NlpEvaluator::new(&m).unwrap();
    let x = [0.9, -1.1];
    let lambda = [1.3, -0.6, 2.0];

    let mut g1 = vec![0.0; 2];
    let mut g2 = vec![0.0; 2];
    ev.eval_objective_gradient(&x, &mut g1);
    ev.eval_objective_gradient(&x, &mut g2);
    assert_eq!(g1, g2, "gradient must not accumulate across calls");

    let nnz = ev.hessian_lagrangian_structure().len();
    let mut h1 = vec![0.0; nnz];
    let mut h2 = vec![0.0; nnz];
    ev.eval_hessian_lagrangian(&x, 0.7, &lambda, &mut h1);
    ev.eval_hessian_lagrangian(&x, 0.7, &lambda, &mut h2);
    assert_eq!(h1, h2, "hessian must not accumulate across calls");
}

#[test]
fn params_refresh_without_retaping() {
    let m = Model::new("params");
    variable!(m, -5.0 <= x <= 5.0);
    param!(m, p = 2.0);
    objective!(m, Min, p * x.sin());

    let mut ev = NlpEvaluator::new(&m).unwrap();
    let point = [0.8];
    let mut grad = vec![0.0; 1];
    ev.eval_objective_gradient(&point, &mut grad);
    assert_close(grad[0], 2.0 * point[0].cos(), 1e-12, "grad with p=2");

    m.set_param(p, 3.0);
    ev.refresh_params(&m);
    ev.eval_objective_gradient(&point, &mut grad);
    assert_close(grad[0], 3.0 * point[0].cos(), 1e-12, "grad with p=3");
}

#[test]
fn gradient_at_matches_fd() {
    let m = Model::new("gat");
    variable!(m, -5.0 <= x <= 5.0);
    variable!(m, -5.0 <= y <= 5.0);
    let g = x.powi(2) * y + y.exp();

    let point = [1.5, 0.5];
    let grad = gradient_at(&m, g, &point).unwrap();
    assert_close(grad[0], 2.0 * point[0] * point[1], 1e-12, "d/dx");
    assert_close(grad[1], point[0].powi(2) + point[1].exp(), 1e-12, "d/dy");
}

#[test]
fn feasibility_model_has_zero_objective() {
    let m = Model::new("feas");
    variable!(m, -5.0 <= x <= 5.0);
    variable!(m, -5.0 <= y <= 5.0);
    constraint!(m, c, x.sin() + y.powi(2) <= 4.0);
    objective!(m, Feasibility);

    let ev = NlpEvaluator::new(&m).unwrap();
    let point = [1.0, -2.0];
    assert_close(ev.eval_objective(&point), 0.0, 1e-12, "objective");

    let mut grad = vec![0.0; 2];
    ev.eval_objective_gradient(&point, &mut grad);
    for (i, &g) in grad.iter().enumerate() {
        assert_close(g, 0.0, 1e-12, &format!("objective grad[{i}]"));
    }

    let mut g = vec![0.0; 1];
    ev.eval_constraint(&point, &mut g);
    assert_close(g[0], point[0].sin() + point[1].powi(2), 1e-12, "constraint value");
}

/// Mostly-separable objective.
#[test]
fn separable_hessian_is_sparse_and_compressed() {
    let m = Model::new("separable");
    variable!(m, -5.0 <= x0 <= 5.0);
    variable!(m, -5.0 <= x1 <= 5.0);
    variable!(m, -5.0 <= x2 <= 5.0);
    variable!(m, -5.0 <= x3 <= 5.0);
    variable!(m, -5.0 <= x4 <= 5.0);
    variable!(m, -5.0 <= x5 <= 5.0);
    let xs = [x0, x1, x2, x3, x4, x5];
    // sum sin(x_i) + x0*x1, true Hessian is diagonal plus one cross entry.
    let mut obj = xs[0] * xs[1];
    for x in &xs {
        obj = obj + x.sin();
    }
    objective!(m, Min, obj);

    let ev = NlpEvaluator::new(&m).unwrap();
    // 7 entries, not the 21 a support clique would produce.
    assert_eq!(
        ev.hessian_lagrangian_structure(),
        &[(0, 0), (1, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
    );
    // Columns {0, 2..5} share one seed while column 1 conflicts with 0.
    assert_eq!(ev.num_hessian_seeds(), 2);

    let x = [0.4, -1.3, 2.1, 0.9, -0.2, 1.7];
    let dense = dense_hessian(&ev, &x, 1.0, &[]);
    let h = 1e-5;
    for j in 0..6 {
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        xp[j] += h;
        xm[j] -= h;
        let gp = lagrangian_gradient(&ev, &xp, 1.0, &[]);
        let gm = lagrangian_gradient(&ev, &xm, 1.0, &[]);
        for i in 0..6 {
            let fd = (gp[i] - gm[i]) / (2.0 * h);
            assert_close(dense[i][j], fd, 1e-5, &format!("H[{i},{j}]"));
        }
    }
}

#[test]
fn try_refresh_reuses_tapes_when_structure_preserved() {
    let m = Model::new("refresh_ok");
    variable!(m, -5.0 <= x <= 5.0);
    param!(m, p = 2.0);
    objective!(m, Min, p * x.sin());

    let mut ev = NlpEvaluator::new(&m).unwrap();
    let point = [0.8];
    let mut grad = vec![0.0; 1];
    ev.eval_objective_gradient(&point, &mut grad);
    assert_close(grad[0], 2.0 * point[0].cos(), 1e-12, "grad with p=2");

    m.set_param(p, 3.0);
    assert!(ev.try_refresh(&m), "structure preserved -> refreshed in place");
    ev.eval_objective_gradient(&point, &mut grad);
    assert_close(grad[0], 3.0 * point[0].cos(), 1e-12, "grad after try_refresh with p=3");
}

#[test]
fn try_refresh_declines_when_model_structure_differs() {
    let base = Model::new("base");
    variable!(base, -5.0 <= x <= 5.0);
    variable!(base, -5.0 <= y <= 5.0);
    objective!(base, Min, x.sin() * y);
    constraint!(base, c0, x.powi(2) + y.powi(2) <= 4.0);

    let mut ev = NlpEvaluator::new(&base).unwrap();
    let jac_before = ev.jacobian_structure().to_vec();

    let other = Model::new("other");
    variable!(other, -5.0 <= x <= 5.0);
    variable!(other, -5.0 <= y <= 5.0);
    objective!(other, Min, x.sin() * y);
    constraint!(other, c0, x.powi(2) + y.powi(2) <= 4.0);
    constraint!(other, c1, x * y.exp() >= 1.0);

    assert!(!ev.try_refresh(&other), "different constraint set -> caller must rebuild");
    assert_eq!(ev.jacobian_structure(), jac_before.as_slice(), "declined refresh is a no-op");
    assert_eq!(ev.num_constraints(), 1, "evaluator still describes the base model");
}

#[test]
fn gradient_at_rejects_wrong_dimension() {
    let m = Model::new("gat_dim");
    variable!(m, -5.0 <= x <= 5.0);
    variable!(m, -5.0 <= y <= 5.0);
    let g = x.powi(2) + y.exp();

    let err = gradient_at(&m, g, &[1.0]).unwrap_err();
    assert!(
        matches!(err, AutodiffError::DimensionMismatch { expected: 2, got: 1 }),
        "unexpected error: {err:?}"
    );
}

/// A dense Hessian yields one seed per column, crossing the parallel-HVP
/// threshold, so this tests the rayon Hessian path and checks it
/// against finite differences.
#[test]
fn parallel_hessian_matches_fd() {
    let n = 20usize;
    let m = Model::new("par_hess");
    variable!(m, -1.0 <= x[i in 0..n] <= 1.0);
    objective!(m, Min, sum!(x[i] for i in 0..n).sin());

    let ev = NlpEvaluator::new(&m).unwrap();
    assert_eq!(ev.num_hessian_seeds(), n, "dense Hessian -> one seed per column");

    let x: Vec<f64> = (0..n).map(|i| 0.05 + 0.01 * i as f64).collect();
    let dense = dense_hessian(&ev, &x, 1.0, &[]);
    let h = 1e-5;
    for j in 0..n {
        let mut xp = x.clone();
        let mut xm = x.clone();
        xp[j] += h;
        xm[j] -= h;
        let gp = lagrangian_gradient(&ev, &xp, 1.0, &[]);
        let gm = lagrangian_gradient(&ev, &xm, 1.0, &[]);
        for i in 0..n {
            let fd = (gp[i] - gm[i]) / (2.0 * h);
            assert_close(dense[i][j], fd, 1e-5, &format!("H[{i},{j}]"));
        }
    }
}

/// Enough constraints to cross the parallel threshold, testing the rayon
/// constraint-value and Jacobian paths against closed-form derivatives.
#[test]
fn parallel_constraints_and_jacobian_match_analytic() {
    let n = 70usize;
    let m = Model::new("par_con");
    variable!(m, -2.0 <= x[i in 0..n] <= 2.0);
    constraint!(m, c[i in 0..n], x[i].sin() <= 0.5);
    objective!(m, Min, sum!(x[i] for i in 0..n));

    let ev = NlpEvaluator::new(&m).unwrap();
    assert_eq!(ev.num_constraints(), n);

    let point: Vec<f64> = (0..n).map(|i| 0.1 + 0.01 * i as f64).collect();

    let mut g = vec![0.0; n];
    ev.eval_constraint(&point, &mut g);
    for i in 0..n {
        assert_close(g[i], point[i].sin(), 1e-12, &format!("constraint[{i}] value"));
    }

    // Row i is sin(x_i), so its only Jacobian entry is at column i with value
    // cos(x_i).
    let mut jac = vec![0.0; ev.jacobian_structure().len()];
    ev.eval_constraint_jacobian(&point, &mut jac);
    for (&(row, col), &v) in ev.jacobian_structure().iter().zip(&jac) {
        assert_eq!(row, col, "each constraint touches exactly its own variable");
        assert_close(v, point[col].cos(), 1e-9, &format!("jac[{row},{col}]"));
    }
}

#[test]
fn hub_objective_star_compresses_seeds() {
    let n = 7usize;
    let m = Model::new("hub");
    variable!(m, -1.0 <= x[i in 0..n] <= 1.0);
    objective!(m, Min, x[0] * sum!(x[i].sin() for i in 1..n));

    let ev = NlpEvaluator::new(&m).unwrap();
    // 2 * (n-1) + 0 entries: (i,0) and (i,i) for each spoke, no (0,0).
    assert_eq!(ev.hessian_lagrangian_structure().len(), 2 * (n - 1));
    assert_eq!(ev.num_hessian_seeds(), 2, "arrow Hessian compresses to two seeds");

    let x: Vec<f64> = (0..n).map(|i| 0.2 + 0.1 * i as f64).collect();
    let dense = dense_hessian(&ev, &x, 1.0, &[]);
    let h = 1e-5;
    for j in 0..n {
        let mut xp = x.clone();
        let mut xm = x.clone();
        xp[j] += h;
        xm[j] -= h;
        let gp = lagrangian_gradient(&ev, &xp, 1.0, &[]);
        let gm = lagrangian_gradient(&ev, &xm, 1.0, &[]);
        for i in 0..n {
            let fd = (gp[i] - gm[i]) / (2.0 * h);
            assert_close(dense[i][j], fd, 1e-5, &format!("H[{i},{j}]"));
        }
    }
}
