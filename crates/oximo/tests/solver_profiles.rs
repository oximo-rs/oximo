#![cfg(any(
    feature = "highs",
    feature = "clarabel",
    feature = "scip",
    feature = "scip-system",
    feature = "gurobi",
    feature = "mosek",
    feature = "baron",
))]

use oximo_solver::{Solver, SolverProfile};
use std::borrow::Cow;

fn borrowed_profile<S: Solver>(solver: &S, options: &S::Options) -> *const SolverProfile {
    let profile = solver.profile(options);
    assert!(matches!(profile, Cow::Borrowed(_)));
    let again = solver.profile(options);
    assert!(std::ptr::eq(profile.as_ref(), again.as_ref()));
    std::ptr::from_ref(profile.as_ref())
}

#[cfg(any(
    feature = "highs",
    feature = "clarabel",
    feature = "scip",
    feature = "scip-system",
    feature = "gurobi",
    feature = "mosek"
))]
fn borrowed_resident_profile<S: oximo_solver::PersistentSolver>(solver: &S, options: &S::Options) {
    let cold = borrowed_profile(solver, options);
    let resident = solver.persistent();
    assert_eq!(cold, borrowed_profile(&resident, options));
}

#[cfg(feature = "highs")]
#[test]
fn highs_borrows_the_same_profile_for_cold_and_resident_queries() {
    borrowed_resident_profile(&oximo_highs::Highs, &oximo_highs::HighsOptions::default());
}

#[cfg(feature = "clarabel")]
#[test]
fn clarabel_borrows_the_same_profile_for_cold_and_resident_queries() {
    borrowed_resident_profile(
        &oximo_clarabel::Clarabel,
        &oximo_clarabel::ClarabelOptions::default(),
    );
}

#[cfg(any(feature = "scip", feature = "scip-system"))]
#[test]
fn scip_borrows_the_same_profile_for_cold_and_resident_queries() {
    borrowed_resident_profile(&oximo_scip::Scip::new(), &oximo_scip::ScipOptions::default());
}

#[cfg(feature = "gurobi")]
#[test]
fn gurobi_borrows_the_same_profile_for_cold_and_resident_queries() {
    borrowed_resident_profile(&oximo_gurobi::Gurobi, &oximo_gurobi::GurobiOptions::default());
}

#[cfg(feature = "mosek")]
#[test]
fn mosek_borrows_the_same_profile_for_cold_and_resident_queries() {
    borrowed_resident_profile(&oximo_mosek::Mosek, &oximo_mosek::MosekOptions::default());
}

#[cfg(feature = "baron")]
#[test]
fn baron_borrows_its_profile_across_backend_instances() {
    assert_eq!(
        borrowed_profile(&oximo_baron::Baron::new(), &oximo_baron::BaronOptions::default()),
        borrowed_profile(
            &oximo_baron::Baron::with_exec("other-baron"),
            &oximo_baron::BaronOptions::default()
        ),
    );
}
