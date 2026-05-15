use oximo::prelude::*;
use oximo::solvers::Highs;

#[test]
fn lp_canonical() {
    let m = Model::new("transport");
    let x = m.var("x").lb(0.0).build();
    let y = m.var("y").lb(0.0).ub(4.0).build();
    m.constraint("c1", (x + 2.0 * y).le(14.0));
    m.constraint("c2", (3.0 * x - y).ge(0.0));
    m.constraint("c3", (x - y).le(2.0));
    m.maximize(3.0 * x + 4.0 * y);

    let result = Highs.solve(&m, &HighsOptions::default()).unwrap();
    assert_eq!(result.status, SolverStatus::Optimal);
    assert!((result.objective.unwrap() - 34.0).abs() < 1e-6);
    assert!((result.value_of(x).unwrap() - 6.0).abs() < 1e-6);
    assert!((result.value_of(y).unwrap() - 4.0).abs() < 1e-6);
}

#[test]
fn knapsack_milp() {
    let weights = [3.0, 4.0, 2.0, 5.0, 1.0, 6.0, 7.0, 2.0];
    let values = [10.0, 12.0, 5.0, 14.0, 3.0, 18.0, 22.0, 6.0];

    let m = Model::new("knapsack");
    let xs: Vec<_> = (0..weights.len()).map(|i| m.var(format!("x{i}")).binary().build()).collect();
    let weight_sum = sum(xs.iter().zip(weights.iter()).map(|(x, w)| *w * *x));
    m.constraint("cap", weight_sum.le(15.0));
    m.maximize(sum(xs.iter().zip(values.iter()).map(|(x, v)| *v * *x)));

    let result = Highs.solve(&m, &HighsOptions::default()).unwrap();
    assert_eq!(result.status, SolverStatus::Optimal);
    assert!((result.objective.unwrap() - 47.0).abs() < 1e-6);
}

#[test]
fn infeasible_returns_status() {
    let m = Model::new("infeas");
    let x = m.var("x").lb(0.0).ub(1.0).build();
    m.constraint("c1", x.ge(5.0));
    m.minimize(x);
    let result = Highs.solve(&m, &HighsOptions::default()).unwrap();
    assert_eq!(result.status, SolverStatus::Infeasible);
}

#[test]
fn presolve_off_gives_correct_result() {
    let m = Model::new("canon");
    let x = m.var("x").lb(0.0).build();
    let y = m.var("y").lb(0.0).ub(4.0).build();
    m.constraint("c1", (x + 2.0 * y).le(14.0));
    m.constraint("c2", (3.0 * x - y).ge(0.0));
    m.constraint("c3", (x - y).le(2.0));
    m.maximize(3.0 * x + 4.0 * y);
    let result = Highs.solve(&m, &HighsOptions::default().presolve(HighsPresolve::Off)).unwrap();
    assert_eq!(result.status, SolverStatus::Optimal);
    assert!((result.objective.unwrap() - 34.0).abs() < 1e-6);
    assert!((result.value_of(x).unwrap() - 6.0).abs() < 1e-6);
    assert!((result.value_of(y).unwrap() - 4.0).abs() < 1e-6);
}

#[test]
fn ipm_method_gives_correct_result() {
    let m = Model::new("canon");
    let x = m.var("x").lb(0.0).build();
    let y = m.var("y").lb(0.0).ub(4.0).build();
    m.constraint("c1", (x + 2.0 * y).le(14.0));
    m.constraint("c2", (3.0 * x - y).ge(0.0));
    m.constraint("c3", (x - y).le(2.0));
    m.maximize(3.0 * x + 4.0 * y);
    let result = Highs.solve(&m, &HighsOptions::default().method(HighsMethod::Ipm)).unwrap();
    assert_eq!(result.status, SolverStatus::Optimal);
    assert!((result.objective.unwrap() - 34.0).abs() < 1e-6);
    assert!((result.value_of(x).unwrap() - 6.0).abs() < 1e-6);
    assert!((result.value_of(y).unwrap() - 4.0).abs() < 1e-6);
}

#[test]
fn threads_one_gives_correct_result() {
    let m = Model::new("canon");
    let x = m.var("x").lb(0.0).build();
    let y = m.var("y").lb(0.0).ub(4.0).build();
    m.constraint("c1", (x + 2.0 * y).le(14.0));
    m.constraint("c2", (3.0 * x - y).ge(0.0));
    m.constraint("c3", (x - y).le(2.0));
    m.maximize(3.0 * x + 4.0 * y);
    let result = Highs.solve(&m, &HighsOptions::default().threads(1)).unwrap();
    assert_eq!(result.status, SolverStatus::Optimal);
    assert!((result.objective.unwrap() - 34.0).abs() < 1e-6);
    assert!((result.value_of(x).unwrap() - 6.0).abs() < 1e-6);
    assert!((result.value_of(y).unwrap() - 4.0).abs() < 1e-6);
}

#[test]
fn mip_gap_accepted_and_solves() {
    // Loose gap on a tiny knapsack, still gets optimal on small instances.
    let weights = [3.0, 4.0, 2.0, 5.0, 1.0];
    let values = [10.0, 12.0, 5.0, 14.0, 3.0];
    let m = oximo_core::Model::new("ks");
    let xs: Vec<_> = (0..5).map(|i| m.var(format!("x{i}")).binary().build()).collect();
    let ws = xs.iter().zip(weights.iter()).map(|(x, w)| *w * *x);
    m.constraint("cap", sum(ws).le(8.0));
    m.maximize(sum(xs.iter().zip(values.iter()).map(|(x, v)| *v * *x)));
    let opts = HighsOptions::default().mip_gap(0.5).verbose(false);
    let result = Highs.solve(&m, &opts).unwrap();
    assert!(
        matches!(result.status, SolverStatus::Optimal | SolverStatus::Feasible),
        "unexpected status: {:?}",
        result.status
    );
    assert!(result.objective.unwrap() > 0.0);
}

#[cfg(feature = "io")]
#[test]
fn mps_coefficients_and_rhs() {
    // Verify that COLUMNS carries the right coefficients and RHS is correct.
    // Model: min 3x + 4y, s.t. x + 2y <= 14 (y bounded 0..4)
    let m = Model::new("transport");
    let x = m.var("x").lb(0.0).build();
    let y = m.var("y").lb(0.0).ub(4.0).build();
    m.constraint("c1", (x + 2.0 * y).le(14.0));
    m.minimize(3.0 * x + 4.0 * y);

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
fn mps_free_and_integer_bounds() {
    let m = Model::new("mixed");
    let _x = m.var("x").binary().build();
    let _y = m.var("y").lb(0.0).ub(10.0).integer().build();
    let z = m.var("z").lb(f64::NEG_INFINITY).build();
    m.minimize(z);

    let s = oximo::io::to_mps_string(&m).unwrap();

    // Integer markers present
    assert!(s.contains("'INTORG'"));
    assert!(s.contains("'INTEND'"));

    // z is free
    assert!(s.contains("FR BND       z"));

    // y upper bound
    assert!(s.contains("UP BND       y         10"));
}

#[cfg(feature = "io")]
#[test]
fn lp_coefficients_and_bounds() {
    // Verify coefficients, sense, constraint operators, and bound declarations.
    let m = Model::new("transport");
    let x = m.var("x").lb(0.0).build();
    let y = m.var("y").lb(0.0).ub(4.0).build();
    m.constraint("c1", (x + 2.0 * y).le(14.0));
    m.constraint("c2", (3.0 * x - y).ge(0.0));
    m.maximize(3.0 * x + 4.0 * y);

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
    let x = m.var("x").lb(f64::NEG_INFINITY).build();
    m.minimize(x);

    let s = oximo::io::to_lp_string(&m).unwrap();
    assert!(s.contains("x free"), "free variable must use 'x free' syntax");
}

#[cfg(feature = "io")]
#[test]
fn lp_export_lists_binaries_and_integers() {
    let m = Model::new("mixed");
    let _x = m.var("x").binary().build();
    let _y = m.var("y").lb(0.0).ub(10.0).integer().build();
    let z = m.var("z").lb(0.0).build();
    m.minimize(z);

    let s = oximo::io::to_lp_string(&m).unwrap();
    assert!(s.contains("General"));
    assert!(s.contains("Binaries"));

    let gen_start = s.find("General").unwrap();
    let bin_start = s.find("Binaries").unwrap();
    // y in General section, not Binaries
    assert!(s[gen_start..bin_start].contains(" y"));
    assert!(!s[gen_start..bin_start].contains(" x"));
    // x in Binaries section
    assert!(s[bin_start..].contains(" x"));
    assert!(!s[bin_start..].contains(" y"));
}
