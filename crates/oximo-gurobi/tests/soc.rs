//! Live Gurobi tests for second-order cone models (explicit and detected).

#![expect(clippy::many_single_char_names)]

use oximo_core::prelude::*;
use oximo_gurobi::{Gurobi, GurobiOptions};
use oximo_solver::Solver;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

#[test]
fn explicit_socp_min_linear_over_disk() {
    let m = Model::new("socp");
    variable!(m, -10.0 <= x <= 10.0);
    variable!(m, -10.0 <= y <= 10.0);
    variable!(m, t >= 0.0);
    m.fix(t, 1.0);
    m.add_soc_constraint("disk", [x, y], t);
    objective!(m, Min, x + y);
    assert_eq!(m.kind(), ModelKind::SOCP);

    let r = Gurobi.solve(&m, &GurobiOptions::default()).expect("solve");
    assert!(r.has_solution());
    let obj = r.objective().expect("obj");
    assert!(close(obj, -std::f64::consts::SQRT_2, 1e-4), "obj = {obj}");
}

#[test]
fn explicit_soc_dual_matches_norm_form_multiplier() {
    // KKT gives z0 = ||grad obj|| = sqrt(2)
    let m = Model::new("socp_dual");
    variable!(m, -10.0 <= x <= 10.0);
    variable!(m, -10.0 <= y <= 10.0);
    variable!(m, t >= 0.0);
    m.fix(t, 1.0);
    let disk = m.add_soc_constraint("disk", [x, y], t);
    objective!(m, Min, x + y);

    let opts = GurobiOptions::default().qcp_dual(1);
    let r = Gurobi.solve(&m, &opts).expect("solve");
    assert!(r.has_solution());
    let z0 = r.soc_dual_of(disk).expect("SOC dual missing");
    assert!(close(z0, std::f64::consts::SQRT_2, 1e-4), "z0 = {z0}");

    // Without QCPDual=1 Gurobi computes no QCP duals; the map stays empty.
    let r = Gurobi.solve(&m, &GurobiOptions::default()).expect("solve");
    assert!(r.soc_dual.is_empty());
}

#[test]
fn detected_socp_hypotenuse() {
    let m = Model::new("socp_detected");
    variable!(m, x);
    variable!(m, y);
    variable!(m, t >= 0.0);
    m.fix(x, 3.0);
    m.fix(y, 4.0);
    constraint!(m, cone, x.powi(2) + y.powi(2) <= t.powi(2));
    objective!(m, Min, t);
    assert_eq!(m.kind(), ModelKind::SOCP);

    let r = Gurobi.solve(&m, &GurobiOptions::default()).expect("solve");
    assert!(r.has_solution());
    let obj = r.objective().expect("obj");
    assert!(close(obj, 5.0, 1e-4), "obj = {obj}");
}

#[test]
fn misocp_with_binary_var() {
    let m = Model::new("misocp");
    variable!(m, x);
    variable!(m, y >= 1.0);
    variable!(m, t >= 0.0);
    variable!(m, z, Bin);
    m.add_soc_constraint("cone", [x, y], t);
    constraint!(m, cx, x >= 1.0 + z);
    objective!(m, Min, t + 10.0 * z);
    assert_eq!(m.kind(), ModelKind::MISOCP);

    let r = Gurobi.solve(&m, &GurobiOptions::default()).expect("solve");
    assert!(r.has_solution());
    let obj = r.objective().expect("obj");
    assert!(close(obj, std::f64::consts::SQRT_2, 1e-4), "obj = {obj}");
}

#[test]
fn soc_with_affine_members() {
    let m = Model::new("socp_affine");
    variable!(m, x);
    variable!(m, y);
    variable!(m, u >= -1.0);
    m.fix(x, 1.0);
    m.fix(y, 1.0);
    m.add_soc_constraint("cone", [x - y, y + 1.0], u + 2.0);
    objective!(m, Min, u);
    assert_eq!(m.kind(), ModelKind::SOCP);

    let r = Gurobi.solve(&m, &GurobiOptions::default()).expect("solve");
    assert!(r.has_solution());
    let obj = r.objective().expect("obj");
    assert!(close(obj, 0.0, 1e-4), "obj = {obj}");
}
