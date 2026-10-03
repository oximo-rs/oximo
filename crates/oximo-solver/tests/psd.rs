use oximo_core::prelude::*;
use oximo_solver::{DualStatus, SolutionPoint, Solver, SolverError, SolverResult, snapshot};

#[derive(Debug)]
struct ScalarOnly;
impl Solver for ScalarOnly {
    type Options = ();
    fn name(&self) -> &str {
        "scalar"
    }
    fn supports(&self, _: ModelKind) -> bool {
        true
    }
    fn solve(&mut self, _: &Model, (): &()) -> Result<SolverResult, SolverError> {
        unreachable!()
    }
}

#[test]
fn psd_capability_is_independent_of_model_kind() {
    let m = Model::new("nonlinear and psd");
    variable!(m, x);
    psd_constraint!(m, SymmetricMatrix::from_upper_triangle(1, [x]));
    objective!(m, Min, x.exp());
    assert_eq!(m.kind(), ModelKind::NLP);
    assert!(!ScalarOnly.supports_model(&m));
    assert!(matches!(snapshot(&m), Err(SolverError::UnsupportedConstraint("PSD"))));
}

#[test]
fn psd_lowering_uses_live_parameters_and_rejects_nonfinite_entries() {
    let m = Model::new("rebind");
    variable!(m, x);
    param!(m, p = 2.0);
    let cone = psd_constraint!(m, positivity, SymmetricMatrix::from_upper_triangle(1, [p * x + p]));
    objective!(m, Min, x);
    for value in [2.0, 5.0] {
        m.set_param(p, value).unwrap();
        let prepared = oximo_solver::prepare::LoweringContext::new(&m).unwrap();
        let matrix = prepared.explicit_psd(&m.psd_constraints()[cone.index()]).unwrap();
        let entry = &matrix.upper_triangle()[0];
        assert_eq!(entry.constant, value);
        assert_eq!(entry.coeffs[0].1, value);
    }
    m.set_param(p, f64::INFINITY).unwrap();
    let prepared = oximo_solver::prepare::LoweringContext::new(&m).unwrap();
    let error = prepared.explicit_psd(&m.psd_constraints()[cone.index()]).unwrap_err();
    assert!(error.to_string().contains("positivity"));
}

#[test]
fn matrix_readback_checks_ownership_missing_entries_and_dual_status() {
    let m = Model::new("source");
    symmetric_variable!(m, x[2]);
    let handle = psd_constraint!(m, &x);
    let foreign = Model::new("foreign");
    symmetric_variable!(foreign, y[2]);
    let foreign_handle = psd_constraint!(foreign, &y);
    let mut result = SolverResult { model_id: m.id(), ..SolverResult::default() };
    assert!(result.value_of_matrix(&y).is_err());
    assert!(result.psd_dual_of(foreign_handle).is_err());
    assert!(result.value_of_matrix(&x).unwrap().is_none());
    result.solutions.push(SolutionPoint {
        model_id: m.id(),
        primal: [(x[(0, 0)].var_id().unwrap(), 1.0)].into_iter().collect(),
        objective: None,
    });
    assert!(result.value_of_matrix(&x).unwrap().is_none());
    result.solutions[0].primal = x
        .upper_triangle()
        .iter()
        .enumerate()
        .map(|(i, entry)| (entry.var_id().unwrap(), f64::from(u32::try_from(i).unwrap())))
        .collect();
    assert_eq!(result.value_of_matrix(&x).unwrap().unwrap().upper_triangle(), &[0.0, 1.0, 2.0]);
    result.psd_dual.insert(handle.id(), SymmetricMatrix::from_upper_triangle(2, [1.0, -1.0, 1.0]));
    result.dual_status = DualStatus::FeasiblePoint;
    let result = oximo_solver::reconstruct::normalize_result(result, 3);
    assert_eq!(result.psd_dual_of(handle).unwrap().unwrap()[(0, 1)], -1.0);
    let mut result = result;
    result.dual_status = DualStatus::NoSolution;
    let result = oximo_solver::reconstruct::normalize_result(result, 3);
    assert!(result.psd_dual.is_empty());
}

#[test]
fn matrix_readback_uses_live_parameters_and_validates_all_owners_before_missing_values() {
    let m = Model::new("readback");
    variable!(m, x);
    variable!(m, missing);
    param!(m, p = 2.0);
    let point = SolutionPoint {
        model_id: m.id(),
        primal: [(x.var_id().unwrap(), 3.0)].into_iter().collect(),
        objective: None,
    };
    let matrix = SymmetricMatrix::from_upper_triangle(2, [p * x + 1.0, p * x, x]);
    assert_eq!(point.value_of_matrix(&matrix).unwrap().unwrap().upper_triangle(), &[7.0, 6.0, 3.0]);
    m.set_param(p, 4.0).unwrap();
    assert_eq!(
        point.value_of_matrix(&matrix).unwrap().unwrap().upper_triangle(),
        &[13.0, 12.0, 3.0]
    );
    let other = Model::new("other");
    variable!(other, foreign);
    let mixed = SymmetricMatrix::from_upper_triangle(2, [missing, x, foreign]);
    assert!(point.value_of_matrix(&mixed).is_err());
    let result =
        SolverResult { model_id: m.id(), solutions: vec![point], ..SolverResult::default() };
    assert!(result.value_of_matrix(&mixed).is_err());
    assert!(
        result
            .value_of_matrix(&SymmetricMatrix::from_upper_triangle(1, [missing]))
            .unwrap()
            .is_none()
    );
}
