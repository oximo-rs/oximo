#![cfg(feature = "mosek")]

use oximo::prelude::*;
use oximo::solvers::Mosek;
use oximo::{MosekOptions, MosekPersistent};

#[test]
fn mosek_feature_exposes_public_solver_and_options() {
    let model = Model::new("umbrella_mosek");
    variable!(model, x >= 1.0);
    objective!(model, Min, x + 2.0);

    let result = Mosek.solve(&model, &MosekOptions::default()).unwrap();
    assert_eq!(result.termination, TerminationStatus::Optimal);
    assert!((result.value_of(x).unwrap().unwrap() - 1.0).abs() < 1e-7);
    assert!((result.objective().unwrap() - 3.0).abs() < 1e-7);
}

#[test]
fn mosek_feature_solves_public_mip_qp_and_socp_apis() {
    let mip = Model::new("umbrella_mosek_mip");
    variable!(mip, x, Bin);
    objective!(mip, Max, 2.0 * x);
    let result = Mosek.solve(&mip, &MosekOptions::default().mio_tol_rel_gap(1e-4)).unwrap();
    assert_eq!(result.termination, TerminationStatus::Optimal);
    assert!((result.value_of(x).unwrap().unwrap() - 1.0).abs() < 1e-7);

    let qp = Model::new("umbrella_mosek_qp");
    variable!(qp, qx >= 0.0);
    constraint!(qp, total, qx >= 2.0);
    objective!(qp, Min, qx.powi(2));
    let result = Mosek.solve(&qp, &MosekOptions::default()).unwrap();
    assert_eq!(result.termination, TerminationStatus::Optimal);
    assert!((result.value_of(qx).unwrap().unwrap() - 2.0).abs() < 1e-6);

    let socp = Model::new("umbrella_mosek_socp");
    variable!(socp, sx);
    variable!(socp, sy);
    variable!(socp, t >= 0.0);
    constraint!(socp, fix_x, sx == 3.0);
    constraint!(socp, fix_y, sy == 4.0);
    socp.add_soc_constraint("norm", [sx, sy], t);
    objective!(socp, Min, t);
    let result = Mosek.solve(&socp, &MosekOptions::default()).unwrap();
    assert_eq!(result.termination, TerminationStatus::Optimal);
    assert!((result.value_of(t).unwrap().unwrap() - 5.0).abs() < 1e-6);
}

#[test]
fn mosek_feature_exposes_persistent_solver_through_prelude() {
    let model = Model::new("umbrella_mosek_persistent");
    param!(model, price = 1.0);
    variable!(model, 0.0 <= x <= 10.0);
    objective!(model, Max, price * x);

    let mut solver: MosekPersistent = Mosek.persistent();
    for (coefficient, upper) in [(1.0, 10.0), (4.0, 3.0)] {
        price.set_param_value(coefficient).unwrap();
        model.unfix_var(x.var_id().unwrap(), 0.0, upper).unwrap();
        let result = solver.solve(&model, &MosekOptions::default()).unwrap();
        assert_eq!(result.termination, TerminationStatus::Optimal);
        assert!((result.value_of(x).unwrap().unwrap() - upper).abs() < 1e-7);
        assert!((result.objective().unwrap() - coefficient * upper).abs() < 1e-7);
    }
}
