//! Live Gurobi tests for second-order cone models (explicit and detected).

#![expect(clippy::many_single_char_names)]

use oximo_core::prelude::*;
use oximo_gurobi::{Gurobi, GurobiOptions};
use oximo_solver::{InfeasibilityDiagnosis, PersistentSolver, Solver};

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

#[test]
fn explicit_socp_min_linear_over_disk() {
    let m = Model::new("socp");
    variable!(m, -10.0 <= x <= 10.0);
    variable!(m, -10.0 <= y <= 10.0);
    variable!(m, t >= 0.0);
    m.fix(t, 1.0).unwrap();
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
    m.fix(t, 1.0).unwrap();
    let disk = m.add_soc_constraint("disk", [x, y], t);
    objective!(m, Min, x + y);

    let opts = GurobiOptions::default().qcp_dual(1);
    let r = Gurobi.solve(&m, &opts).expect("solve");
    assert!(r.has_solution());
    let z0 = r
        .soc_dual_of(disk)
        .expect("matching model")
        .expect("SOC dual missing")
        .standard_bound()
        .expect("standard cone");
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
    m.fix(x, 3.0).unwrap();
    m.fix(y, 4.0).unwrap();
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
    m.fix(x, 1.0).unwrap();
    m.fix(y, 1.0).unwrap();
    m.add_soc_constraint("cone", [x - y, y + 1.0], u + 2.0);
    objective!(m, Min, u);
    assert_eq!(m.kind(), ModelKind::SOCP);

    let r = Gurobi.solve(&m, &GurobiOptions::default()).expect("solve");
    assert!(r.has_solution());
    let obj = r.objective().expect("obj");
    assert!(close(obj, 0.0, 1e-4), "obj = {obj}");
}

#[test]
fn soc_iis_maps_the_generated_head_bound() {
    let m = Model::new("socp_iis_sign");
    variable!(m, x);
    variable!(m, t);
    m.fix(x, 0.0).unwrap();
    m.fix(t, -1.0).unwrap();
    let cone = m.add_soc_constraint("cone", [x], t);
    objective!(m, Min, x);

    let iis = Gurobi.compute_iis(&m, &GurobiOptions::default()).expect("compute SOC IIS");
    assert!(iis.soc_constraints.contains(&cone.soc_id()), "cone missing from IIS: {iis:?}");
}

#[test]
fn rotated_soc_is_recognized_as_convex_with_vector_dual() {
    // min u + v
    // s.t.  x^2 + y^2 <= 2 u v, x = y = 1.
    // Optimum 2 at u = v = 1.
    // `NonConvex=0` makes Gurobi reject the model unless it sees the cone.
    let m = Model::new("rsocp");
    variable!(m, x);
    variable!(m, y);
    variable!(m, u >= 0.0);
    variable!(m, v >= 0.0);
    m.fix(x, 1.0).unwrap();
    m.fix(y, 1.0).unwrap();
    let cone = m.add_rotated_soc_constraint("r", [x, y], u, v);
    objective!(m, Min, u + v);
    assert_eq!(m.kind(), ModelKind::SOCP);

    let opts = GurobiOptions::default().non_convex(0).qcp_dual(1);
    let r = Gurobi.solve(&m, &opts).expect("solve");
    assert!(r.has_solution());
    assert!(close(r.objective().unwrap(), 2.0, 1e-5));

    let z = r.soc_dual_of(cone).unwrap().expect("vector dual").as_slice().to_vec();
    assert_eq!(z.len(), 4);
    assert!(close(z[0], 1.0, 1e-3) && close(z[1], 1.0, 1e-3), "{z:?}");
    assert!(2.0 * z[0] * z[1] >= z[2] * z[2] + z[3] * z[3] - 1e-3, "{z:?}");
}

#[test]
fn rotated_soc_detected_from_quadratic_solves() {
    let m = Model::new("rsocp_detected");
    variable!(m, x);
    variable!(m, u >= 0.0);
    variable!(m, v >= 0.0);
    m.fix(x, 2.0).unwrap();
    constraint!(m, cone, x * x <= 2.0 * u * v);
    objective!(m, Min, u + v);
    assert_eq!(m.kind(), ModelKind::SOCP);

    let r = Gurobi.solve(&m, &GurobiOptions::default().non_convex(0)).expect("solve");
    assert!(close(r.objective().unwrap(), 2.0 * std::f64::consts::SQRT_2, 1e-5));
}

#[test]
fn rotated_soc_with_free_and_affine_heads_stays_convex() {
    // `NonConvex=0` rejects the model unless every cone keeps Gurobi's shape.
    let opts = GurobiOptions::default().non_convex(0).qcp_dual(1);

    let m = Model::new("rsocp_free_heads");
    variable!(m, x);
    variable!(m, u);
    variable!(m, v);
    m.fix(x, 1.0).unwrap();
    let cone = m.add_rotated_soc_constraint("r", [x], u, v);
    objective!(m, Min, u + v);
    let r = Gurobi.solve(&m, &opts).expect("free heads solve");
    assert!(close(r.objective().unwrap(), std::f64::consts::SQRT_2, 1e-5));
    let z = r.soc_dual_of(cone).unwrap().expect("vector dual").as_slice().to_vec();
    assert!(close(z[0], 1.0, 1e-3) && close(z[1], 1.0, 1e-3), "{z:?}");

    let m = Model::new("rsocp_affine_heads");
    variable!(m, x);
    variable!(m, u >= 0.0);
    variable!(m, v >= 0.0);
    m.fix(x, 1.0).unwrap();
    m.add_rotated_soc_constraint("r", [x + 1.0], u + 1.0, 2.0 * v);
    objective!(m, Min, u + v);
    let r = Gurobi.solve(&m, &opts).expect("affine heads solve");
    assert!(close(r.objective().unwrap(), 1.0, 1e-5));

    let m = Model::new("socp_free_bound");
    variable!(m, x);
    variable!(m, t);
    m.fix(x, 1.0).unwrap();
    soc_constraint!(m, c, [x - 3.0] <= t);
    objective!(m, Min, t);
    let r = Gurobi.solve(&m, &opts).expect("free bound solve");
    assert!(close(r.objective().unwrap(), 2.0, 1e-5));
}

#[test]
fn persistent_cone_head_bound_turning_negative_keeps_the_cone() {
    let m = Model::new("rsocp_head_bound");
    param!(m, lower = 0.0);
    variable!(m, x);
    variable!(m, u >= lower);
    variable!(m, v >= 0.0);
    m.fix(x, 1.0).unwrap();
    m.add_rotated_soc_constraint("r", [x], u, v);
    objective!(m, Min, u + v);

    let opts = GurobiOptions::default().non_convex(0);
    let mut solver = Gurobi.persistent();
    for value in [0.0, -1.0] {
        lower.set_param_value(value).unwrap();
        let r = solver.solve(&m, &opts).expect("solve");
        assert!(close(r.objective().unwrap(), std::f64::consts::SQRT_2, 1e-5), "lower={value}");
    }
}

#[test]
fn rotated_soc_constant_and_shared_heads_stay_convex() {
    let opts = GurobiOptions::default().non_convex(0).qcp_dual(1);

    let m = Model::new("rsocp_constant_head");
    variable!(m, x);
    variable!(m, t);
    m.fix(x, 3.0).unwrap();
    let cone = soc_constraint!(m, r, [x] in rotated(t, 1.0));
    objective!(m, Min, t);
    let r = Gurobi.solve(&m, &opts).expect("constant head solve");
    assert!(close(r.objective().unwrap(), 4.5, 1e-5));
    let z = r.soc_dual_of(cone).unwrap().expect("vector dual");
    assert_eq!(z.as_slice().len(), 3);
    let SocHead::Rotated(zu, zv) = z.head() else { panic!("rotated dual") };
    assert!(zu >= -1e-7 && zv >= -1e-7 && 2.0 * zu * zv >= z.terms()[0].powi(2) - 1e-4);

    let m = Model::new("rsocp_shared_head_term");
    variable!(m, x);
    variable!(m, u >= 0.0);
    variable!(m, v >= 0.0);
    m.fix(x, 1.0).unwrap();
    m.add_rotated_soc_constraint("r", [u, x], u, v);
    objective!(m, Min, u + v);
    let r = Gurobi.solve(&m, &opts).expect("shared head solve");
    assert!(close(r.objective().unwrap(), 3.0_f64.sqrt(), 1e-5));

    let m = Model::new("rsocp_negative_head");
    variable!(m, x);
    variable!(m, t);
    m.fix(x, 1.0).unwrap();
    soc_constraint!(m, r, [x] in rotated(t, -1.0));
    objective!(m, Min, t);
    let r = Gurobi.solve(&m, &opts).expect("negative head solve");
    assert!(!r.has_solution(), "{:?}", r.termination);
}

#[test]
fn persistent_rebuilds_when_a_direct_head_loses_its_sign() {
    let m = Model::new("rsocp_head_bound");
    param!(m, lower = 0.0);
    variable!(m, x);
    variable!(m, u >= lower);
    variable!(m, v >= 0.0);
    m.fix(x, 1.0).unwrap();
    m.add_rotated_soc_constraint("r", [x], u, v);
    objective!(m, Min, u + v);

    let opts = GurobiOptions::default().non_convex(0);
    let mut solver = Gurobi.persistent();
    let first = solver.solve(&m, &opts).expect("first solve");
    assert!(close(first.objective().unwrap(), std::f64::consts::SQRT_2, 1e-5));

    m.set_param(lower, -1.0).unwrap();
    let second = solver.solve(&m, &opts).expect("second solve");
    assert!(close(second.objective().unwrap(), std::f64::consts::SQRT_2, 1e-5));

    m.set_param(lower, 0.5).unwrap();
    let third = solver.solve(&m, &opts).expect("third solve");
    assert!(close(third.objective().unwrap(), std::f64::consts::SQRT_2, 1e-5));
}
