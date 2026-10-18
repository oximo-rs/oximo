use oximo_core::prelude::*;
use oximo_gurobi::{Gurobi, GurobiOptions};
use oximo_solver::Solver;

#[test]
fn native_sos1_and_sos2_solve() {
    let m = Model::new("sos");
    variable!(m, 0.0 <= x <= 10.0);
    variable!(m, 0.0 <= y <= 20.0);
    variable!(m, 0.0 <= z <= 30.0);
    objective!(m, Max, x + y + z);
    sos_constraint!(m, one, SOS1, [(x, 1.0), (y, 2.0)]);
    sos_constraint!(m, two, SOS2, [(x, 1.0), (y, 2.0), (z, 3.0)]);
    assert!(Gurobi.supports_model(&m));
    let result = Gurobi.solve(&m, &GurobiOptions::default()).expect("Gurobi SOS solve");
    assert!(result.has_solution());
    assert_eq!(m.kind(), ModelKind::MILP);
    assert!(result.value_of(y).unwrap().unwrap() > 19.0);
    assert!(result.value_of(x).unwrap().unwrap().abs() < 1e-6);
}

#[test]
fn native_sos2_rejects_nonadjacent_optimum() {
    let m = Model::new("sos2_adjacency");
    variable!(m, 0.0 <= x <= 1.0);
    variable!(m, 0.0 <= y <= 1.0);
    variable!(m, 0.0 <= z <= 1.0);
    constraint!(m, budget, x + y + z <= 2.0);
    objective!(m, Max, x + z);
    sos_constraint!(m, adjacent, SOS2, [(x, 1.0), (y, 2.0), (z, 3.0)]);

    let result = Gurobi.solve(&m, &GurobiOptions::default()).expect("Gurobi SOS2 solve");
    let x_value = result.value_of(x).expect("matching model").expect("x solution");
    let z_value = result.value_of(z).expect("matching model").expect("z solution");
    assert!(result.objective().expect("objective") <= 1.0 + 1e-6);
    assert!(
        x_value <= 1e-6 || z_value <= 1e-6,
        "non-adjacent members selected: x={x_value}, z={z_value}"
    );
}

#[test]
fn reformulated_sos1_solves_and_preserves_original_ids() {
    let m = Model::new("reformulated_sos1");
    variable!(m, 0.0 <= x <= 1.0);
    variable!(m, 0.0 <= y <= 1.0);
    objective!(m, Max, x + y);
    let choice = sos_constraint!(m, choice, SOS1, [x, y]);

    let transformed = choice.to_reformulated_model(&m, SosReformulationOptions::default()).unwrap();
    assert!(Gurobi.supports_model(&transformed));
    let result = Gurobi
        .solve(&transformed, &GurobiOptions::default())
        .expect("Gurobi reformulated SOS solve");
    let tx = transformed.variable_handle(x.var_id().unwrap());
    let ty = transformed.variable_handle(y.var_id().unwrap());

    assert!((result.objective().unwrap() - 1.0).abs() < 1e-6);
    assert!(
        result.value_of(tx).unwrap().unwrap() + result.value_of(ty).unwrap().unwrap() <= 1.0 + 1e-6
    );
    assert_eq!(choice.id().index(), 0);
    assert!(!transformed.sos_constraints()[choice.index()].active);
}
