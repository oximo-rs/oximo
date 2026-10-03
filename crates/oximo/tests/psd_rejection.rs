#![cfg(any(
    feature = "highs",
    all(feature = "clarabel", not(feature = "clarabel-sdp")),
    feature = "gurobi",
    feature = "pounce",
    feature = "scip",
    feature = "scip-system",
    feature = "gams",
    feature = "baron",
))]
use oximo::prelude::*;

fn check<S: Solver>(solver: &mut S, options: &S::Options) {
    let m = Model::new("unsupported PSD");
    variable!(m, x);
    psd_constraint!(m, SymmetricMatrix::from_upper_triangle(1, [x]));
    objective!(m, Min, x.exp());
    assert_eq!(m.kind(), ModelKind::NLP);
    assert!(!solver.supports_model(&m));
    assert!(matches!(solver.solve(&m, options), Err(SolverError::UnsupportedConstraint(_))));
}

#[cfg(feature = "highs")]
#[test]
fn highs_rejects_psd() {
    let mut s = oximo::solvers::Highs;
    check(&mut s, &HighsOptions::default());
    check(&mut s.persistent(), &HighsOptions::default());
}

#[cfg(all(feature = "clarabel", not(feature = "clarabel-sdp")))]
#[test]
fn clarabel_without_sdp_feature_rejects_psd() {
    let mut s = oximo::solvers::Clarabel;
    check(&mut s, &ClarabelOptions::default());
    check(&mut s.persistent(), &ClarabelOptions::default());
}

#[cfg(feature = "gurobi")]
#[test]
fn gurobi_rejects_psd() {
    let mut s = oximo::solvers::Gurobi;
    check(&mut s, &GurobiOptions::default());
    check(&mut s.persistent(), &GurobiOptions::default());
}

#[cfg(feature = "pounce")]
#[test]
fn pounce_rejects_psd() {
    let mut s = oximo::solvers::Pounce;
    check(&mut s, &PounceOptions::default());
    check(&mut s.persistent(), &PounceOptions::default());
}

#[cfg(any(feature = "scip", feature = "scip-system"))]
#[test]
fn scip_rejects_psd() {
    let mut s = oximo::solvers::Scip::default();
    check(&mut s, &oximo::ScipOptions::default());
    check(&mut s.persistent(), &oximo::ScipOptions::default());
}

#[cfg(feature = "gams")]
#[test]
fn gams_rejects_psd_before_external_execution() {
    let mut s = oximo::solvers::Gams::with_exec("missing-gams-executable");
    check(&mut s, &GamsOptions::default());
}

#[cfg(feature = "baron")]
#[test]
fn baron_rejects_psd_before_external_execution() {
    let mut s = oximo::solvers::Baron::with_exec("missing-baron-executable");
    check(&mut s, &BaronOptions::default());
}
