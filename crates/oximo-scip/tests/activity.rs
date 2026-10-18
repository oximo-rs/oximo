use oximo_core::prelude::*;
use oximo_scip::{Scip, ScipOptions};
use oximo_solver::{PersistentSolver, Solver};

#[test]
fn inactive_cones_are_omitted_and_active_cones_retain_source_ids() {
    let model = Model::new("SOC activity");

    variable!(model, 3.0 <= fixed_x <= 3.0);
    variable!(model, bound >= 0.0);
    let inactive = model.add_soc_constraint("disabled", [fixed_x], bound);
    let active = model.add_soc_constraint("retained", [fixed_x], bound);
    objective!(model, Min, bound);
    model.set_constraint_active(inactive, false).unwrap();

    let options = ScipOptions::default();
    let mut solver = Scip::new();
    let result = solver.solve(&model, &options).unwrap();
    assert!((result.value_of(bound).unwrap().unwrap() - 3.0).abs() < 1e-5);
    assert!(result.soc_dual_of(inactive).unwrap().is_none());

    let mut resident = solver.persistent();
    let _ = resident.solve(&model, &options).unwrap();
    model.set_constraint_active(active, false).unwrap();
    assert_eq!(model.kind(), ModelKind::LP);

    for result in
        [solver.solve(&model, &options).unwrap(), resident.solve(&model, &options).unwrap()]
    {
        assert!(result.value_of(bound).unwrap().unwrap().abs() < 1e-5);
        assert!(result.soc_dual_of(active).unwrap().is_none());
    }
    assert_eq!(inactive.soc_id().index(), 0);
    assert_eq!(active.soc_id().index(), 1);
}
