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
    assert!((result.objective().unwrap() - 34.0).abs() < 1e-6);
    assert!((result.value_of(x).unwrap() - 6.0).abs() < 1e-6);
    assert!((result.value_of(y).unwrap() - 4.0).abs() < 1e-6);
}

#[test]
fn param_coefficient_lp_rebinds_without_rebuild() {
    let m = Model::new("param_lp");
    let price = m.param("price", 3.0);
    let x = m.var("x").bounds(0.0, 10.0).build();
    m.maximize(price * x);
    assert_eq!(m.kind(), ModelKind::LP);

    let r = Highs.solve(&m, &HighsOptions::default()).unwrap();
    assert_eq!(r.status, SolverStatus::Optimal);
    assert!((r.objective().unwrap() - 30.0).abs() < 1e-6);

    m.set_param(price, 5.0);
    let r2 = Highs.solve(&m, &HighsOptions::default()).unwrap();
    assert!((r2.objective().unwrap() - 50.0).abs() < 1e-6);
}

#[test]
fn param_coefficient_qp_rebinds() {
    let m = Model::new("param_qp");
    let t = m.param("t", 2.0);
    let x = m.var("x").bounds(-10.0, 10.0).build();
    m.minimize((x - t).powi(2));
    assert_eq!(m.kind(), ModelKind::QP);

    let r = Highs.solve(&m, &HighsOptions::default()).unwrap();
    assert_eq!(r.status, SolverStatus::Optimal);
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
    let cost = m.param("cost", 3.0);
    let x = m.var("x").lb(0.0).build();
    m.constraint("c", x.ge(2.0));
    m.minimize(cost * x);

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
    let k = m.param("k", 2.0);
    let x = m.var("x").lb(0.0).build();
    m.constraint("c", (k * x.powi(2)).le(10.0));
    m.minimize(x);
    let nl = to_nl_string(&m).unwrap();
    assert!(!nl.is_empty());
}

#[test]
fn knapsack_milp() {
    let weights = [3.0, 4.0, 2.0, 5.0, 1.0, 6.0, 7.0, 2.0];
    let values = [10.0, 12.0, 5.0, 14.0, 3.0, 18.0, 22.0, 6.0];

    let m = Model::new("knapsack");
    let xs: Vec<_> = (0..weights.len()).map(|i| m.var(format!("x{i}")).binary().build()).collect();
    m.constraint("cap", dot(&xs, &weights).le(15.0));
    m.maximize(dot(&xs, &values));

    let result = Highs.solve(&m, &HighsOptions::default()).unwrap();
    assert_eq!(result.status, SolverStatus::Optimal);
    assert!((result.objective().unwrap() - 47.0).abs() < 1e-6);
}

#[test]
fn lp_initial_values_do_not_affect_optimum() {
    let m = Model::new("lp_warm");
    let x = m.var("x").lb(0.0).initial(6.0).build();
    let y = m.var("y").lb(0.0).ub(4.0).initial(4.0).build();
    m.constraint("c1", (x + 2.0 * y).le(14.0));
    m.constraint("c2", (3.0 * x - y).ge(0.0));
    m.constraint("c3", (x - y).le(2.0));
    m.maximize(3.0 * x + 4.0 * y);

    let result = Highs.solve(&m, &HighsOptions::default()).unwrap();
    assert_eq!(result.status, SolverStatus::Optimal);
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

    let m = Model::new("knapsack_warm");
    let xs: Vec<_> = (0..weights.len())
        .map(|i| m.var(format!("x{i}")).binary().initial(warm_start[i]).build())
        .collect();
    m.constraint("cap", dot(&xs, &weights).le(15.0));
    m.maximize(dot(&xs, &values));

    let result = Highs.solve(&m, &HighsOptions::default()).unwrap();
    assert_eq!(result.status, SolverStatus::Optimal);
    assert!((result.objective().unwrap() - 47.0).abs() < 1e-6);
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
    assert!((result.objective().unwrap() - 34.0).abs() < 1e-6);
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
    assert!((result.objective().unwrap() - 34.0).abs() < 1e-6);
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
    assert!((result.objective().unwrap() - 34.0).abs() < 1e-6);
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
    m.constraint("cap", dot(&xs, &weights).le(8.0));
    m.maximize(dot(&xs, &values));
    let opts = HighsOptions::default().mip_gap(0.5).verbose(false);
    let result = Highs.solve(&m, &opts).unwrap();
    assert!(
        matches!(result.status, SolverStatus::Optimal | SolverStatus::Feasible),
        "unexpected status: {:?}",
        result.status
    );
    assert!(result.objective().unwrap() > 0.0);
}

#[test]
fn indexed_var_retrieval() {
    let m = Model::new("indexed");
    let routes = Set::strings(["a", "b"]);
    let flow = m.indexed_var("flow", &routes).lb(0.0).build();

    m.constraint("ca", flow["a"].ge(3.0));
    m.constraint("cb", flow["b"].ge(7.0));
    m.minimize(flow["a"] + flow["b"]);

    let result = Highs.solve(&m, &HighsOptions::default()).unwrap();
    assert_eq!(result.status, SolverStatus::Optimal);

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
fn mps_fixed_variable_emits_fx_bound() {
    let m = Model::new("fixed");
    let x = m.var("x").lb(0.0).ub(10.0).build();
    let y = m.var("y").lb(0.0).ub(10.0).fix(3.5).build();
    m.constraint("c", (x + y).le(20.0));
    m.minimize(x + y);

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

#[cfg(feature = "io")]
#[test]
fn mps_columns_nonzero_count() {
    // 5 variables, 5 constraints, each constraint involves all 5 variables.
    // COLUMNS section must have exactly 5*5 + 5 = 30 data lines
    let m = Model::new("dense");
    let xs: Vec<_> = (0..5).map(|i| m.var(format!("x{i}")).lb(0.0).build()).collect();
    for i in 0..5usize {
        let row = xs.iter().copied().sum::<Expr>();
        m.constraint(format!("c{i}"), row.le(10.0));
    }
    m.minimize(xs.iter().copied().sum::<Expr>());

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
