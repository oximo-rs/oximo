#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

mod options;
mod persistent;
mod translate;

#[cfg(feature = "benchmark-support")]
#[doc(hidden)]
pub use translate::benchmark_support;

pub use options::MosekOptions;
pub use persistent::MosekPersistent;
pub use translate::solve;

use oximo_core::{Model, ModelKind};
use oximo_solver::{PersistentSolver, Solver, SolverError, SolverResult};

/// MOSEK solver backend.
#[derive(Debug, Default, Clone, Copy)]
pub struct Mosek;

pub(crate) const NAME: &str = "MOSEK";

pub(crate) const fn supported(kind: ModelKind) -> bool {
    matches!(
        kind,
        ModelKind::SDP
            | ModelKind::LP
            | ModelKind::MILP
            | ModelKind::QP
            | ModelKind::MIQP
            | ModelKind::QCP
            | ModelKind::MIQCP
            | ModelKind::SOCP
            | ModelKind::MISOCP
    )
}

impl Solver for Mosek {
    type Options = MosekOptions;

    fn name(&self) -> &str {
        NAME
    }

    fn supports(&self, kind: ModelKind) -> bool {
        supported(kind)
    }

    fn supports_psd(&self) -> bool {
        true
    }

    fn supports_model(&self, model: &Model) -> bool {
        supported(model.kind()) && !model.has_active_sos_constraints() && psd_compatible(model)
    }

    fn supports_indicators(&self) -> bool {
        true
    }

    fn solve(&mut self, model: &Model, opts: &MosekOptions) -> Result<SolverResult, SolverError> {
        translate::solve(model, opts)
    }
}

impl PersistentSolver for Mosek {
    type Handle = MosekPersistent;

    fn persistent(&self) -> MosekPersistent {
        MosekPersistent::new()
    }
}

/// Native quadratic objectives are not included in the initial MOSEK SDP path.
pub(crate) fn psd_compatible(model: &Model) -> bool {
    if !model.has_active_psd_constraints() {
        return true;
    }
    model.kind() == ModelKind::SDP
        && !model.has_active_indicator_constraints()
        && model
            .objective()
            .as_ref()
            .is_none_or(|o| oximo_expr::extract_linear(&model.arena(), o.expr).is_some())
}
