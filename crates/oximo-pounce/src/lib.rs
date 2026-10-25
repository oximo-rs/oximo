#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

mod convex;
mod fbbt;
mod minima;
mod options;
mod persistent;
mod tnlp;
mod translate;

#[cfg(feature = "enzyme")]
mod exact;
#[cfg(not(feature = "enzyme"))]
mod hybrid;
#[cfg(not(feature = "enzyme"))]
mod stable;

#[cfg(all(feature = "benchmark-support", not(feature = "enzyme")))]
#[doc(hidden)]
pub use hybrid::benchmark_support;

pub use minima::{PounceMinimaMethod, PounceMinimaOptions};
pub use options::{
    MuStrategy, PounceAlgorithm, PounceOptionValue, PounceOptions, PounceSolverSelection,
};
pub use persistent::PouncePersistent;

/// The POUNCE backend:
/// specialized convex LP/QP/SOCP engines with a general IPOPT-lineage
/// QCP/NLP fallback.
///
/// On stable Rust an all-linear/quadratic model gets exact analytic
/// derivatives (including the Hessian).
/// A model with nonlinear functions is handed to POUNCE's builder, which
/// finite-differences them with an L-BFGS Hessian. With the `enzyme`
/// feature everything is exact via `oximo-autodiff`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Pounce;

impl oximo_solver::Solver for Pounce {
    type Options = PounceOptions;

    fn name(&self) -> &str {
        "pounce"
    }

    fn supports(&self, kind: oximo_core::ModelKind) -> bool {
        use oximo_core::ModelKind;
        matches!(
            kind,
            ModelKind::LP | ModelKind::QP | ModelKind::QCP | ModelKind::SOCP | ModelKind::NLP
        )
    }

    fn domain_support(&self, domain: oximo_core::Domain) -> oximo_solver::Capability {
        use oximo_core::Domain;
        match domain {
            Domain::Real => oximo_solver::Capability::Native,
            _ => oximo_solver::Capability::Unsupported,
        }
    }

    fn validate_model_extra(
        &self,
        model: &oximo_core::Model,
        opts: &Self::Options,
    ) -> Vec<oximo_solver::CapabilityDiagnostic> {
        convex::validate_route(model, opts)
            .err()
            .map(|error| oximo_solver::CapabilityDiagnostic {
                constraint: None,
                location: "POUNCE route".into(),
                reason: error.to_string(),
            })
            .into_iter()
            .collect()
    }

    fn solve(
        &mut self,
        model: &oximo_core::Model,
        opts: &Self::Options,
    ) -> Result<oximo_solver::SolverResult, oximo_solver::SolverError> {
        translate::solve(model, opts)
    }
}

impl oximo_solver::PersistentSolver for Pounce {
    type Handle = PouncePersistent;

    fn persistent(&self) -> Self::Handle {
        PouncePersistent::new()
    }
}
