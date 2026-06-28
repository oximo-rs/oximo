use oximo::prelude::*;
use oximo::solvers::Highs;

#[test]
fn lp_canonical() {
    let m = Model::new("transport");
    variable!(m, x >= 0.0);
    variable!(m, 0.0 <= y <= 4.0);
    constraint!(m, c1, x + 2.0 * y <= 14.0);
    constraint!(m, c2, 3.0 * x - y >= 0.0);
    constraint!(m, c3, x - y <= 2.0);
    objective!(m, Max, 3.0 * x + 4.0 * y);

    let result = Highs.solve(&m, &HighsOptions::default()).unwrap();
    assert_eq!(result.termination, TerminationStatus::Optimal);
    assert!((result.objective().unwrap() - 34.0).abs() < 1e-6);
    assert!((result.value_of(x).unwrap() - 6.0).abs() < 1e-6);
    assert!((result.value_of(y).unwrap() - 4.0).abs() < 1e-6);
}

#[test]
fn highs_multi_optima_returns_single_best() {
    // A MILP with several optimal solutions.
    // HiGHS has no solution pool, so the multi-solution API surfaces exactly one optimum.
    let m = Model::new("multi");
    let items = Set::range(0..4usize);
    variable!(m, x[i in items], Bin);
    constraint!(m, cap, sum!(x[i] for i in items) <= 2.0);
    objective!(m, Max, sum!(x[i] for i in items));

    let r = Highs.solve(&m, &HighsOptions::default()).unwrap();
    assert_eq!(r.termination, TerminationStatus::Optimal);
    assert_eq!(r.result_count(), 1);
    assert!((r.objective().unwrap() - 2.0).abs() < 1e-6);
    let chosen: f64 = (0..4).filter_map(|i| r.value_of_idx(&x, i)).sum();
    assert!((chosen - 2.0).abs() < 1e-6, "best is not an optimum: sum={chosen}");
}

#[test]
fn indexed_param_rebind_changes_solution() {
    // maximize sum_i price[i] * x[i] s.t. sum_i x[i] <= 1, 0 <= x <= 1.
    // With price = [1, 3, 2] the optimum loads x[1] (price 3) -> obj 3.
    // Re-binding price[2] to 5 shifts the optimum to x[2] -> obj 5.
    let m = Model::new("ip_solve");
    let items = Set::range(0..3usize);
    let price = [1.0, 3.0, 2.0];
    param!(m, p[i in items] = price[i]);
    variable!(m, 0.0 <= x[i in items] <= 1.0);
    constraint!(m, budget, sum!(x[i] for i in items) <= 1.0);
    objective!(m, Max, sum!(p[i] * x[i] for i in items));

    let r1 = Highs.solve(&m, &HighsOptions::default()).unwrap();
    assert_eq!(r1.termination, TerminationStatus::Optimal);
    assert!((r1.objective().unwrap() - 3.0).abs() < 1e-6);
    assert!((r1.value_of_idx(&x, 1usize).unwrap() - 1.0).abs() < 1e-6);

    m.set_param_idx(&p, 2usize, 5.0);
    let r2 = Highs.solve(&m, &HighsOptions::default()).unwrap();
    assert_eq!(r2.termination, TerminationStatus::Optimal);
    assert!((r2.objective().unwrap() - 5.0).abs() < 1e-6);
    assert!((r2.value_of_idx(&x, 2usize).unwrap() - 1.0).abs() < 1e-6);
}

#[test]
fn highs_rejects_semi_domains() {
    // HiGHS can't represent the semicontinuity gap through the `highs` crate, so
    // a semicontinuous (or semi-integer) variable must error.
    let m = Model::new("semi");
    variable!(m, s <= 10.0, SemiCont(2.0));
    constraint!(m, c, s >= 3.0);
    objective!(m, Min, s);

    let err = Highs.solve(&m, &HighsOptions::default()).unwrap_err();
    assert!(matches!(err, SolverError::Backend(_)), "expected Backend error, got {err:?}");
}

#[test]
fn param_coefficient_lp_rebinds_without_rebuild() {
    let m = Model::new("param_lp");
    param!(m, price = 3.0);
    variable!(m, 0.0 <= x <= 10.0);
    objective!(m, Max, price * x);
    assert_eq!(m.kind(), ModelKind::LP);

    let r = Highs.solve(&m, &HighsOptions::default()).unwrap();
    assert_eq!(r.termination, TerminationStatus::Optimal);
    assert!((r.objective().unwrap() - 30.0).abs() < 1e-6);

    m.set_param(price, 5.0);
    let r2 = Highs.solve(&m, &HighsOptions::default()).unwrap();
    assert!((r2.objective().unwrap() - 50.0).abs() < 1e-6);
}

#[test]
fn param_coefficient_qp_rebinds() {
    let m = Model::new("param_qp");
    param!(m, t = 2.0);
    variable!(m, -10.0 <= x <= 10.0);
    objective!(m, Min, (x - t).powi(2));
    assert_eq!(m.kind(), ModelKind::QP);

    let r = Highs.solve(&m, &HighsOptions::default()).unwrap();
    assert_eq!(r.termination, TerminationStatus::Optimal);
    assert!((r.value_of(x).unwrap() - 2.0).abs() < 1e-5);

    m.set_param(t, 4.0);
    let r2 = Highs.solve(&m, &HighsOptions::default()).unwrap();
    assert!((r2.value_of(x).unwrap() - 4.0).abs() < 1e-5);
}

#[cfg(feature = "io")]
#[test]
fn io_linear_writers_fold_param_coefficient() {
    use oximo::io::{to_lp_string, to_mps_string};

    let m = Model::new("io_param");
    param!(m, cost = 3.0);
    variable!(m, x >= 0.0);
    constraint!(m, c, x >= 2.0);
    objective!(m, Min, cost * x);

    let mps = to_mps_string(&m).unwrap();
    assert!(mps.contains("x         OBJ       3"), "got:\n{mps}");
    assert!(to_lp_string(&m).is_ok());

    m.set_param(cost, 5.0);
    let mps2 = to_mps_string(&m).unwrap();
    assert!(mps2.contains("x         OBJ       5"), "got:\n{mps2}");
}

#[cfg(feature = "io")]
#[test]
fn nl_writer_emits_param_in_nonlinear_term() {
    use oximo::io::to_nl_string;

    let m = Model::new("nl_param");
    param!(m, k = 2.0);
    variable!(m, x >= 0.0);
    constraint!(m, c, k * x.powi(2) <= 10.0);
    objective!(m, Min, x);
    let nl = to_nl_string(&m).unwrap();
    assert!(!nl.is_empty());
}

#[test]
fn knapsack_milp() {
    let weights = [3.0, 4.0, 2.0, 5.0, 1.0, 6.0, 7.0, 2.0];
    let values = [10.0, 12.0, 5.0, 14.0, 3.0, 18.0, 22.0, 6.0];
    let n = weights.len();

    let m = Model::new("knapsack");
    variable!(m, x[i in 0..n], Bin);
    constraint!(m, cap, sum!(weights[i] * x[i] for i in 0..n) <= 15.0);
    objective!(m, Max, sum!(values[i] * x[i] for i in 0..n));

    let result = Highs.solve(&m, &HighsOptions::default()).unwrap();
    assert_eq!(result.termination, TerminationStatus::Optimal);
    assert!((result.objective().unwrap() - 47.0).abs() < 1e-6);
}

#[test]
fn lp_initial_values_do_not_affect_optimum() {
    let m = Model::new("lp_warm");
    variable!(m, x >= 0.0);
    variable!(m, 0.0 <= y <= 4.0);
    m.set_initial(x, 6.0);
    m.set_initial(y, 4.0);
    constraint!(m, c1, x + 2.0 * y <= 14.0);
    constraint!(m, c2, 3.0 * x - y >= 0.0);
    constraint!(m, c3, x - y <= 2.0);
    objective!(m, Max, 3.0 * x + 4.0 * y);

    let result = Highs.solve(&m, &HighsOptions::default()).unwrap();
    assert_eq!(result.termination, TerminationStatus::Optimal);
    assert!((result.objective().unwrap() - 34.0).abs() < 1e-6);
    assert!((result.value_of(x).unwrap() - 6.0).abs() < 1e-6);
    assert!((result.value_of(y).unwrap() - 4.0).abs() < 1e-6);
}

#[test]
fn milp_warm_start_finds_optimum() {
    // Pass initial values that are not optimal,
    // should still find the true optimum of 47 (items 4 and 6).
    let weights = [3.0, 4.0, 2.0, 5.0, 1.0, 6.0, 7.0, 2.0];
    let values = [10.0, 12.0, 5.0, 14.0, 3.0, 18.0, 22.0, 6.0];
    let warm_start = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let n = weights.len();

    let m = Model::new("knapsack_warm");
    variable!(m, x[i in 0..n], Bin);
    for i in 0..n {
        m.set_initial(x[i], warm_start[i]);
    }
    constraint!(m, cap, sum!(weights[i] * x[i] for i in 0..n) <= 15.0);
    objective!(m, Max, sum!(values[i] * x[i] for i in 0..n));

    let result = Highs.solve(&m, &HighsOptions::default()).unwrap();
    assert_eq!(result.termination, TerminationStatus::Optimal);
    assert!((result.objective().unwrap() - 47.0).abs() < 1e-6);
}

#[test]
fn infeasible_returns_status() {
    let m = Model::new("infeas");
    variable!(m, 0.0 <= x <= 1.0);
    constraint!(m, c1, x >= 5.0);
    objective!(m, Min, x);
    let result = Highs.solve(&m, &HighsOptions::default()).unwrap();
    assert_eq!(result.termination, TerminationStatus::Infeasible);
}

#[test]
fn presolve_off_gives_correct_result() {
    let m = Model::new("canon");
    variable!(m, x >= 0.0);
    variable!(m, 0.0 <= y <= 4.0);
    constraint!(m, c1, x + 2.0 * y <= 14.0);
    constraint!(m, c2, 3.0 * x - y >= 0.0);
    constraint!(m, c3, x - y <= 2.0);
    objective!(m, Max, 3.0 * x + 4.0 * y);
    let result = Highs.solve(&m, &HighsOptions::default().presolve(HighsPresolve::Off)).unwrap();
    assert_eq!(result.termination, TerminationStatus::Optimal);
    assert!((result.objective().unwrap() - 34.0).abs() < 1e-6);
    assert!((result.value_of(x).unwrap() - 6.0).abs() < 1e-6);
    assert!((result.value_of(y).unwrap() - 4.0).abs() < 1e-6);
}

#[test]
fn ipm_method_gives_correct_result() {
    let m = Model::new("canon");
    variable!(m, x >= 0.0);
    variable!(m, 0.0 <= y <= 4.0);
    constraint!(m, c1, x + 2.0 * y <= 14.0);
    constraint!(m, c2, 3.0 * x - y >= 0.0);
    constraint!(m, c3, x - y <= 2.0);
    objective!(m, Max, 3.0 * x + 4.0 * y);
    let result = Highs.solve(&m, &HighsOptions::default().method(HighsMethod::Ipm)).unwrap();
    assert_eq!(result.termination, TerminationStatus::Optimal);
    assert!((result.objective().unwrap() - 34.0).abs() < 1e-6);
    assert!((result.value_of(x).unwrap() - 6.0).abs() < 1e-6);
    assert!((result.value_of(y).unwrap() - 4.0).abs() < 1e-6);
}

#[test]
fn threads_one_gives_correct_result() {
    let m = Model::new("canon");
    variable!(m, x >= 0.0);
    variable!(m, 0.0 <= y <= 4.0);
    constraint!(m, c1, x + 2.0 * y <= 14.0);
    constraint!(m, c2, 3.0 * x - y >= 0.0);
    constraint!(m, c3, x - y <= 2.0);
    objective!(m, Max, 3.0 * x + 4.0 * y);
    let result = Highs.solve(&m, &HighsOptions::default().threads(1)).unwrap();
    assert_eq!(result.termination, TerminationStatus::Optimal);
    assert!((result.objective().unwrap() - 34.0).abs() < 1e-6);
    assert!((result.value_of(x).unwrap() - 6.0).abs() < 1e-6);
    assert!((result.value_of(y).unwrap() - 4.0).abs() < 1e-6);
}

#[test]
fn mip_gap_accepted_and_solves() {
    // Loose gap on a tiny knapsack, still gets optimal on small instances.
    let weights = [3.0, 4.0, 2.0, 5.0, 1.0];
    let values = [10.0, 12.0, 5.0, 14.0, 3.0];
    let n = weights.len();
    let m = Model::new("ks");
    variable!(m, x[i in 0..n], Bin);
    constraint!(m, cap, sum!(weights[i] * x[i] for i in 0..n) <= 8.0);
    objective!(m, Max, sum!(values[i] * x[i] for i in 0..n));
    let opts = HighsOptions::default().mip_gap(0.5).verbose(false);
    let result = Highs.solve(&m, &opts).unwrap();
    assert!(
        result.has_solution(),
        "unexpected status: {:?} / {:?}",
        result.termination,
        result.primal_status
    );
    assert!(result.objective().unwrap() > 0.0);
}

#[test]
fn indexed_var_retrieval() {
    let m = Model::new("indexed");
    let routes = Set::strings(["a", "b"]);
    variable!(m, flow[r in routes] >= 0.0);

    constraint!(m, ca, flow["a"] >= 3.0);
    constraint!(m, cb, flow["b"] >= 7.0);
    objective!(m, Min, flow["a"] + flow["b"]);

    let result = Highs.solve(&m, &HighsOptions::default()).unwrap();
    assert_eq!(result.termination, TerminationStatus::Optimal);

    assert!((result.value_of_idx(&flow, "a").unwrap() - 3.0).abs() < 1e-6);
    assert!((result.value_of_idx(&flow, "b").unwrap() - 7.0).abs() < 1e-6);

    let mut vals: Vec<_> = result.values_of(&flow).collect();
    vals.sort_by(|(a, _), (b, _)| format!("{a:?}").cmp(&format!("{b:?}")));
    assert_eq!(vals.len(), 2);

    let nonzero: Vec<_> = result.values_of(&flow).filter(|(_, v)| *v != 0.0).collect();
    assert_eq!(nonzero.len(), 2);
}

#[cfg(feature = "io")]
#[test]
fn mps_coefficients_and_rhs() {
    // Verify that COLUMNS carries the right coefficients and RHS is correct.
    // Model: min 3x + 4y, s.t. x + 2y <= 14 (y bounded 0..4)
    let m = Model::new("transport");
    variable!(m, x >= 0.0);
    variable!(m, 0.0 <= y <= 4.0);
    constraint!(m, c1, x + 2.0 * y <= 14.0);
    objective!(m, Min, 3.0 * x + 4.0 * y);

    let s = oximo::io::to_mps_string(&m).unwrap();

    // Structure
    assert!(s.contains("NAME"));
    assert!(s.contains("ROWS"));
    assert!(s.contains(" N  OBJ"));
    assert!(s.contains(" L  c1"));
    assert!(s.contains("COLUMNS"));
    assert!(s.contains("RHS"));
    assert!(s.contains("BOUNDS"));
    assert!(s.contains("ENDATA"));

    // Objective coefficients
    assert!(s.contains("x         OBJ       3"));
    assert!(s.contains("y         OBJ       4"));

    // Constraint coefficients
    assert!(s.contains("x         c1        1"));
    assert!(s.contains("y         c1        2"));

    // RHS for c1
    assert!(s.contains("RHS       c1        14"));

    // y upper bound
    assert!(s.contains("UP BND       y         4"));

    // Sense comment preserved
    assert!(s.contains("* sense: minimize"));
}

#[cfg(feature = "io")]
#[test]
fn mps_fixed_variable_emits_fx_bound() {
    let m = Model::new("fixed");
    variable!(m, 0.0 <= x <= 10.0);
    variable!(m, y);
    m.fix(y, 3.5);
    constraint!(m, c, x + y <= 20.0);
    objective!(m, Min, x + y);

    let s = oximo::io::to_mps_string(&m).unwrap();

    // y is fixed, must appear as FX at col 2, value at col 25
    assert!(s.contains(" FX BND       y         3.5"), "got:\n{s}");
    // x is not fixed, no FX line for x
    assert!(!s.contains(" FX BND       x"), "got:\n{s}");
    // y still appears in COLUMNS (constraint coefficients unchanged)
    assert!(s.contains('y'), "got:\n{s}");
}

#[cfg(feature = "io")]
#[test]
fn mps_free_and_integer_bounds() {
    let m = Model::new("mixed");
    variable!(m, x, Bin);
    variable!(m, 0.0 <= y <= 10.0, Int);
    variable!(m, z >= f64::NEG_INFINITY);
    objective!(m, Min, z);
    let _ = (x, y);

    let mps = oximo::io::to_mps_string(&m).unwrap();

    // Integer markers present
    assert!(mps.contains("'INTORG'"));
    assert!(mps.contains("'INTEND'"));

    // z is free
    assert!(mps.contains("FR BND       z"));

    // y upper bound
    assert!(mps.contains("UP BND       y         10"));
}

#[cfg(feature = "io")]
#[test]
fn lp_coefficients_and_bounds() {
    // Verify coefficients, sense, constraint operators, and bound declarations.
    let m = Model::new("transport");
    variable!(m, x >= 0.0);
    variable!(m, 0.0 <= y <= 4.0);
    constraint!(m, c1, x + 2.0 * y <= 14.0);
    constraint!(m, c2, 3.0 * x - y >= 0.0);
    objective!(m, Max, 3.0 * x + 4.0 * y);

    let s = oximo::io::to_lp_string(&m).unwrap();

    // Sense
    assert!(s.contains("Maximize"));
    assert!(!s.contains("Minimize"));

    // Objective coefficients (unit coeff omits magnitude)
    assert!(s.contains("obj:"));
    assert!(s.contains("3 x"));
    assert!(s.contains("4 y"));

    // Constraints present with correct operators
    assert!(s.contains("c1:"));
    assert!(s.contains("<= 14"));
    assert!(s.contains("c2:"));
    assert!(s.contains(">= 0"));

    // y ub=4 is non-default -> must appear in Bounds
    assert!(s.contains("Bounds"));
    assert!(s.contains("<= 4"));

    // x has default bounds (lb=0, ub=inf) -> must NOT appear in Bounds
    let bounds_start = s.find("Bounds").unwrap();
    let end_start = s.find("End").unwrap();
    let bounds_section = &s[bounds_start..end_start];
    assert!(!bounds_section.contains(" x "), "x should not appear in Bounds");

    assert!(s.contains("End"));
}

#[cfg(feature = "io")]
#[test]
fn lp_free_variable_emits_free_keyword() {
    let m = Model::new("free_test");
    variable!(m, x >= f64::NEG_INFINITY);
    objective!(m, Min, x);

    let s = oximo::io::to_lp_string(&m).unwrap();
    assert!(s.contains("x free"), "free variable must use 'x free' syntax");
}

#[cfg(feature = "io")]
#[test]
fn lp_export_lists_binaries_and_integers() {
    let m = Model::new("mixed");
    variable!(m, x, Bin);
    variable!(m, 0.0 <= y <= 10.0, Int);
    variable!(m, z >= 0.0);
    objective!(m, Min, z);
    let _ = (x, y);

    let lp = oximo::io::to_lp_string(&m).unwrap();
    assert!(lp.contains("General"));
    assert!(lp.contains("Binaries"));

    let gen_start = lp.find("General").unwrap();
    let bin_start = lp.find("Binaries").unwrap();
    // y in General section, not Binaries
    assert!(lp[gen_start..bin_start].contains(" y"));
    assert!(!lp[gen_start..bin_start].contains(" x"));
    // x in Binaries section
    assert!(lp[bin_start..].contains(" x"));
    assert!(!lp[bin_start..].contains(" y"));
}

#[cfg(feature = "io")]
#[test]
fn mps_columns_nonzero_count() {
    // 5 variables, 5 constraints, each constraint involves all 5 variables.
    // COLUMNS section must have exactly 5*5 + 5 = 30 data lines
    let m = Model::new("dense");
    variable!(m, x[i in 0..5] >= 0.0);
    for _ in 0..5usize {
        constraint!(m, sum!(x[j] for j in 0..5) <= 10.0);
    }
    objective!(m, Min, sum!(x[j] for j in 0..5));

    let s = oximo::io::to_mps_string(&m).unwrap();

    let cols_start = s.find("COLUMNS\n").unwrap() + "COLUMNS\n".len();
    let rhs_start = s.find("RHS\n").unwrap();
    let cols_section = &s[cols_start..rhs_start];

    let data_lines: Vec<&str> = cols_section
        .lines()
        .filter(|l| !l.contains("'MARKER'"))
        .filter(|l| !l.trim().is_empty())
        .collect();
    assert_eq!(data_lines.len(), 30, "expected 30 COLUMNS entries, got {}", data_lines.len());
}
