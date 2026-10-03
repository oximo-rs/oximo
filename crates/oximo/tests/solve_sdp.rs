#![allow(non_snake_case)]
#![cfg(any(feature = "clarabel-sdp", feature = "mosek"))]

use oximo::prelude::*;

fn close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 2e-5, "actual={actual}, expected={expected}");
}

fn check_trace_problem<S: Solver>(solver: &mut S, options: &S::Options) {
    let m = Model::new("trace");
    symmetric_variable!(m, X[2]);
    let cone = psd_constraint!(m, positivity, &X);
    constraint!(m, X[0, 1] == 1.0);
    objective!(m, Min, X.trace());
    assert!(solver.supports_psd());
    assert!(solver.supports_model(&m));
    let result = solver.solve(&m, options).unwrap();
    assert_eq!(result.termination, TerminationStatus::Optimal);
    close(result.objective().unwrap(), 2.0);
    let primal = result.value_of_matrix(&X).unwrap().unwrap();
    for &entry in primal.upper_triangle() {
        close(entry, 1.0);
    }
    let dual = result.psd_dual_of(cone).unwrap().unwrap();
    close(dual[(0, 0)], 1.0);
    close(dual[(0, 1)], -1.0);
    close(dual[(1, 1)], 1.0);
    assert!(primal[(0, 0)] >= -2e-5 && primal[(1, 1)] >= -2e-5);
    assert!(primal[(0, 0)] * primal[(1, 1)] - primal[(0, 1)].powi(2) >= -2e-5);
    assert!(dual[(0, 0)] * dual[(1, 1)] - dual[(0, 1)].powi(2) >= -2e-5);
    close(primal.frobenius(dual), 0.0);
    assert!(result.report(&m).unwrap().to_string().contains("positivity"));
}

fn check_eigenvalue_problem<S: Solver>(solver: &mut S, options: &S::Options) {
    let m = Model::new("eigenvalue");
    variable!(m, t);
    let identity = SymmetricMatrix::<f64>::identity(3);
    let c = SymmetricMatrix::try_from_rows([[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]])
        .unwrap();
    let matrix = t * &identity - &c;
    let cone = psd_constraint!(m, eigenvalue, &matrix);
    objective!(m, Min, t);
    let result = solver.solve(&m, options).unwrap();
    close(result.objective().unwrap(), 2.0 + 2.0_f64.sqrt());
    let dual = result.psd_dual_of(cone).unwrap().unwrap();
    let expected = [0.25, 2.0_f64.sqrt() / 4.0, 0.5, 0.25, 2.0_f64.sqrt() / 4.0, 0.25];
    for (&actual, expected) in dual.upper_triangle().iter().zip(expected) {
        close(actual, expected);
    }
    close(dual.trace(), 1.0); // stationarity with respect to t
    close(result.value_of_matrix(&matrix).unwrap().unwrap().frobenius(dual), 0.0);
}

fn check_maximization_and_mixed_cones<S: Solver>(solver: &mut S, options: &S::Options) {
    let m = Model::new("max");
    symmetric_variable!(m, X[2]);
    let inactive = psd_constraint!(m, inactive, SymmetricMatrix::from_upper_triangle(1, [-1.0]));
    m.set_psd_active(inactive, false).unwrap();
    let cone = psd_constraint!(m, positivity, &X);
    let scalar = psd_constraint!(m, scalar, SymmetricMatrix::from_upper_triangle(1, [X[(0, 0)]]));
    constraint!(m, X[0, 0] == 1.0);
    constraint!(m, X[1, 1] == 1.0);
    soc_constraint!(m, disk, [X[0, 1]] <= 2.0 * X[0, 0]);
    objective!(m, Max, X[0, 1]);
    let result = solver.solve(&m, options).unwrap();
    close(result.objective().unwrap(), 1.0);
    assert!(result.psd_dual_of(inactive).unwrap().is_none());
    assert!(result.psd_dual_of(scalar).unwrap().is_some());
    let dual = result.psd_dual_of(cone).unwrap().unwrap();
    close(dual[(0, 0)], 0.5);
    close(dual[(0, 1)], -0.5);
    close(dual[(1, 1)], 0.5);
    close(result.value_of_matrix(&X).unwrap().unwrap().frobenius(dual), 0.0);
}

fn check_persistent<S: Solver>(solver: &mut S, options: &S::Options) {
    let m = Model::new("sweep");
    variable!(m, t);
    param!(m, p = 1.0);
    let matrix = SymmetricMatrix::from_upper_triangle(2, [t, p.into(), t]);
    let cone = psd_constraint!(m, &matrix);
    objective!(m, Min, t);
    for expected in [1.0, 2.0, 3.0] {
        m.set_param(p, expected).unwrap();
        let result = solver.solve(&m, options).unwrap();
        close(result.objective().unwrap(), expected);
        close(result.psd_dual_of(cone).unwrap().unwrap()[(0, 1)], -0.5);
    }
    psd_constraint!(m, SymmetricMatrix::from_upper_triangle(1, [t - 4.0]));
    close(solver.solve(&m, options).unwrap().objective().unwrap(), 4.0);
}

fn check_sparse_persistent<S: PersistentSolver>(solver: &mut S, options: &S::Options) {
    let m = Model::new("sparse path eigenvalue");
    variable!(m, t);
    param!(m, p = 1.0);
    let identity = SymmetricMatrix::from_upper_fn(8, |i, j| f64::from(u8::from(i == j)));
    let path = SymmetricMatrix::from_upper_fn(8, |i, j| f64::from(u8::from(j == i + 1)));
    let matrix = t * identity - p * path;
    let cone = psd_constraint!(m, eigenvalue, &matrix);
    objective!(m, Min, t);
    let angle = std::f64::consts::PI / 9.0;
    let eigenvector: Vec<_> =
        (1..=8).map(|i| (2.0_f64 / 9.0).sqrt() * (f64::from(i) * angle).sin()).collect();
    let expected_dual = SymmetricMatrix::from_upper_fn(8, |i, j| eigenvector[i] * eigenvector[j]);
    let mut persistent = solver.persistent();
    for scale in [1.0, 2.0, 0.5] {
        m.set_param(p, scale).unwrap();
        let warm = persistent.solve(&m, options).unwrap();
        let cold = solver.solve(&m, options).unwrap();
        for result in [&warm, &cold] {
            assert_eq!(result.termination, TerminationStatus::Optimal);
            close(result.objective().unwrap(), scale * 2.0 * angle.cos());
            let primal = result.value_of_matrix(&matrix).unwrap().unwrap();
            let dual = result.psd_dual_of(cone).unwrap().unwrap();
            for (&actual, &expected) in
                dual.upper_triangle().iter().zip(expected_dual.upper_triangle())
            {
                close(actual, expected);
            }
            close(dual.trace(), 1.0);
            close(primal.frobenius(dual), 0.0);
        }
        close(warm.value_of(t).unwrap().unwrap(), cold.value_of(t).unwrap().unwrap());
    }
}

fn check_persistent_coefficients<S: PersistentSolver>(solver: &mut S, options: &S::Options) {
    let m = Model::new("PSD coefficient sweep");
    variable!(m, t >= -10.0);
    variable!(m, x);
    m.fix(x, 1.0).unwrap();
    param!(m, p = 1.0);
    let matrix = SymmetricMatrix::from_upper_triangle(2, [t, p * x, t]);
    let cone = psd_constraint!(m, &matrix);
    objective!(m, Min, t);
    let mut persistent = solver.persistent();
    for coefficient in [1.0_f64, 2.0, 0.0, -2.0, -0.5, 0.0, 1.0] {
        m.set_param(p, coefficient).unwrap();
        let warm = persistent.solve(&m, options).unwrap();
        let cold = solver.solve(&m, options).unwrap();
        for result in [&warm, &cold] {
            assert_eq!(result.termination, TerminationStatus::Optimal);
            close(result.objective().unwrap(), coefficient.abs());
            close(result.value_of(x).unwrap().unwrap(), 1.0);
            let primal = result.value_of_matrix(&matrix).unwrap().unwrap();
            let dual = result.psd_dual_of(cone).unwrap().unwrap();
            close(primal[(0, 1)], coefficient);
            close(dual.trace(), 1.0);
            if coefficient.abs() > 0.0 {
                close(dual[(0, 0)], 0.5);
                close(dual[(0, 1)], -0.5 * coefficient.signum());
                close(dual[(1, 1)], 0.5);
            }
            close(primal.frobenius(dual), 0.0);
        }
        close(warm.value_of(t).unwrap().unwrap(), cold.value_of(t).unwrap().unwrap());
    }
    // Remove and restore the PSD block itself to exercise a structural rebuild
    // in both directions, including transition through a scalar-only model.
    for (active, expected) in [(false, -10.0), (true, 1.0)] {
        m.set_psd_active(cone, active).unwrap();
        let warm = persistent.solve(&m, options).unwrap();
        let cold = solver.solve(&m, options).unwrap();
        assert_eq!(warm.termination, TerminationStatus::Optimal);
        close(warm.objective().unwrap(), expected);
        close(warm.objective().unwrap(), cold.objective().unwrap());
        assert_eq!(warm.psd_dual_of(cone).unwrap().is_some(), active);
    }
}

#[cfg(feature = "clarabel-sdp")]
#[test]
fn clarabel_sparse_sdp_primal_dual_and_persistent() {
    use oximo::ClarabelOptions;
    use oximo::solvers::Clarabel;
    // Resolve individual entries of the rank-one dual more accurately than
    // the default objective-gap stopping criterion requires.
    let options = ClarabelOptions::default()
        .presolve_enable(false)
        .tol_gap_abs(1e-10)
        .tol_gap_rel(1e-10)
        .tol_feas(1e-10);
    check_sparse_persistent(&mut Clarabel, &options);
}

#[cfg(feature = "clarabel-sdp")]
#[test]
fn clarabel_persistent_sdp_coefficient_sweep() {
    use oximo::ClarabelOptions;
    use oximo::solvers::Clarabel;
    let options = ClarabelOptions::default().presolve_enable(false);
    check_persistent_coefficients(&mut Clarabel, &options);
}

#[cfg(feature = "mosek")]
#[test]
fn mosek_sparse_sdp_primal_dual_and_persistent() {
    use oximo::MosekOptions;
    use oximo::solvers::Mosek;
    check_sparse_persistent(&mut Mosek, &MosekOptions::default());
}

#[cfg(feature = "mosek")]
#[test]
fn mosek_persistent_sdp_coefficient_sweep() {
    use oximo::MosekOptions;
    use oximo::solvers::Mosek;
    check_persistent_coefficients(&mut Mosek, &MosekOptions::default());
}

#[cfg(feature = "clarabel-sdp")]
#[test]
fn clarabel_sdp_primal_dual_and_persistent() {
    use oximo::ClarabelOptions;
    use oximo::solvers::Clarabel;
    let options = ClarabelOptions::default();
    check_trace_problem(&mut Clarabel, &options);
    check_eigenvalue_problem(&mut Clarabel, &options);
    check_maximization_and_mixed_cones(&mut Clarabel, &options);
    check_persistent(&mut Clarabel.persistent(), &options);
}

#[cfg(feature = "clarabel-sdp")]
#[test]
fn clarabel_quadratic_objective_and_infeasible_sdp() {
    use oximo::ClarabelOptions;
    use oximo::solvers::Clarabel;
    let m = Model::new("quadratic");
    variable!(m, t);
    let cone = psd_constraint!(m, SymmetricMatrix::from_upper_triangle(1, [t - 1.0]));
    objective!(m, Min, t.square());
    let result = Clarabel.solve(&m, &ClarabelOptions::default()).unwrap();
    close(result.objective().unwrap(), 1.0);
    close(result.psd_dual_of(cone).unwrap().unwrap()[(0, 0)], 2.0);
    constraint!(m, t <= 0.0);
    let result = Clarabel.solve(&m, &ClarabelOptions::default()).unwrap();
    assert_eq!(result.termination, TerminationStatus::Infeasible);
    assert!(result.psd_dual.is_empty());
    assert!(
        result.value_of_matrix(&SymmetricMatrix::from_upper_triangle(1, [t])).unwrap().is_none()
    );
}

#[cfg(feature = "mosek")]
#[test]
fn mosek_sdp_primal_dual_and_persistent() {
    use oximo::MosekOptions;
    use oximo::solvers::Mosek;
    let options = MosekOptions::default();
    check_trace_problem(&mut Mosek, &options);
    check_eigenvalue_problem(&mut Mosek, &options);
    check_maximization_and_mixed_cones(&mut Mosek, &options);
    check_persistent(&mut Mosek.persistent(), &options);
}

#[cfg(feature = "mosek")]
#[test]
fn mosek_rejects_quadratic_sdp_before_optimization() {
    use oximo::MosekOptions;
    use oximo::solvers::Mosek;
    let m = Model::new("quadratic");
    variable!(m, t);
    psd_constraint!(m, SymmetricMatrix::from_upper_triangle(1, [t]));
    objective!(m, Min, t.square());
    assert!(!Mosek.supports_model(&m));
    assert!(matches!(
        Mosek.solve(&m, &MosekOptions::default()),
        Err(SolverError::UnsupportedConstraint(_))
    ));
}
