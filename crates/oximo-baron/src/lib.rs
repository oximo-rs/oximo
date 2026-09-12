#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

//! Reference: [N. Sahinidis, *BARON User Manual and Installation Guide*, version
//! 2026.9.10, The Optimization Firm, LLC, September 10, 2026][baron-manual].
//!
//! [baron-manual]: https://minlp-downloads.nyc3.cdn.digitaloceanspaces.com/docs/baron%20manual.pdf

mod options;
mod translate;

#[cfg(feature = "benchmark-support")]
#[doc(hidden)]
pub use translate::benchmark_support;

pub use options::BaronOptions;
pub use translate::solve;

use oximo_core::{Model, ModelKind};
use oximo_solver::{Iis, InfeasibilityDiagnosis, Solver, SolverError, SolverResult};

/// BARON backend. Writes an oximo [`Model`] to a temporary `.bar` file, invokes
/// the `baron` executable, and parses the result.
#[derive(Debug, Default, Clone)]
pub struct Baron {
    /// Optional override for the BARON executable path. When `None`, `"baron"`
    /// is looked up from `PATH`. Overridden per-call by
    /// [`BaronOptions::baron_path`].
    pub exec: Option<String>,
}

impl Baron {
    /// Create a backend that uses `baron` from `PATH`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a backend pointing at an explicit BARON executable path.
    #[must_use]
    pub fn with_exec(path: impl Into<String>) -> Self {
        Self { exec: Some(path.into()) }
    }
}

/// Display name for this backend; the single source for both [`Solver::name`]
/// and the `solver_name` stamped on every [`SolverResult`].
pub(crate) const NAME: &str = "BARON";

pub(crate) const fn supported(kind: ModelKind) -> bool {
    matches!(
        kind,
        ModelKind::LP
            | ModelKind::MILP
            | ModelKind::QP
            | ModelKind::MIQP
            | ModelKind::QCP
            | ModelKind::MIQCP
            | ModelKind::SOCP
            | ModelKind::MISOCP
            | ModelKind::NLP
            | ModelKind::MINLP
    )
}

impl Solver for Baron {
    type Options = BaronOptions;

    fn name(&self) -> &str {
        NAME
    }

    fn supports(&self, kind: ModelKind) -> bool {
        supported(kind)
    }

    fn solve(&mut self, model: &Model, opts: &BaronOptions) -> Result<SolverResult, SolverError> {
        translate::solve(model, opts, self.exec.as_deref())
    }
}

impl InfeasibilityDiagnosis for Baron {
    fn compute_iis(&mut self, model: &Model, opts: &BaronOptions) -> Result<Iis, SolverError> {
        translate::compute_iis(model, opts, self.exec.as_deref())
    }
}
