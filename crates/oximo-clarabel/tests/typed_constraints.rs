use oximo_clarabel::ClarabelOptions;
use oximo_core::prelude::*;
use oximo_solver::TerminationStatus;

#[test]
fn checked_affine_powers_solve_as_soc_and_keep_parameters_live() {
    let m = Model::new("affine powers in SOC");
    variable!(m, x);
    variable!(m, t);
    param!(m, p = 2.0);
    m.fix(x, 2.0).unwrap();
    objective!(m, Min, t);
    soc_constraint!(m, cone, [p.powi(2) * x.powi(1), x.powi(0)] <= t);
    for parameter in [2.0_f64, 3.0] {
        m.set_param(p, parameter).unwrap();
        let result = oximo_clarabel::solve(&m, &ClarabelOptions::default()).unwrap();
        assert_eq!(result.termination, TerminationStatus::Optimal);
        let expected = (4.0 * parameter.powi(4) + 1.0).sqrt();
        assert!((result.value_of(t).unwrap().unwrap() - expected).abs() < 1e-6);
    }
}

#[test]
fn inverted_finite_intervals_remain_infeasible() {
    let m = Model::new("inverted interval");
    variable!(m, x);
    objective!(m, Min, x);
    constraint!(m, impossible, 2.0 <= x <= 1.0);
    let result = oximo_clarabel::solve(&m, &ClarabelOptions::default()).unwrap();
    assert_eq!(result.termination, TerminationStatus::Infeasible);
    assert!(!result.has_solution());
}
