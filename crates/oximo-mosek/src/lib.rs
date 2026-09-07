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
pub(crate) const SUPPORTED_KINDS: &[ModelKind] = &[
    ModelKind::LP,
    ModelKind::MILP,
    ModelKind::QP,
    ModelKind::MIQP,
    ModelKind::QCP,
    ModelKind::MIQCP,
    ModelKind::SOCP,
    ModelKind::MISOCP,
];

pub(crate) fn supported(kind: ModelKind) -> bool {
    SUPPORTED_KINDS.contains(&kind)
}

impl Solver for Mosek {
    type Options = MosekOptions;

    fn name(&self) -> &str {
        NAME
    }

    fn supports(&self, kind: ModelKind) -> bool {
        supported(kind)
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
