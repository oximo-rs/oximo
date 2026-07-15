#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

mod options;
mod persistent;
mod translate;

#[cfg(feature = "enzyme")]
mod exact;
#[cfg(not(feature = "enzyme"))]
mod stable;
#[cfg(not(feature = "enzyme"))]
mod values;

pub use options::{MuStrategy, PounceOptionValue, PounceOptions};
pub use persistent::PouncePersistent;

/// The POUNCE interior-point backend: a pure-Rust IPOPT port.
/// Solves continuous LP/QP/QCP/NLP models.
/// On stable Rust POUNCE finite-differences the derivatives.
/// With the `enzyme` feature it uses exact first and second
/// derivatives from `oximo-autodiff`.
#[derive(Clone, Copy, Debug, Default)]
pub struct PounceSolver;

impl oximo_solver::Solver for PounceSolver {
    type Options = PounceOptions;

    fn name(&self) -> &str {
        "pounce"
    }

    fn supports(&self, kind: oximo_core::ModelKind) -> bool {
        use oximo_core::ModelKind;
        matches!(kind, ModelKind::LP | ModelKind::QP | ModelKind::QCP | ModelKind::NLP)
    }

    fn solve(
        &mut self,
        model: &oximo_core::Model,
        opts: &Self::Options,
    ) -> Result<oximo_solver::SolverResult, oximo_solver::SolverError> {
        translate::solve(model, opts)
    }
}

impl oximo_solver::PersistentSolver for PounceSolver {
    type Handle = PouncePersistent;

    fn persistent(&self) -> Self::Handle {
        PouncePersistent::new()
    }
}
