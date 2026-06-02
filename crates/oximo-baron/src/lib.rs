#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

//! References:
//! N. Sahinidis, BARON User Manual, version 2026.4.12.
//! The Optimization Firm, LLC, Apr. 12, 2026.

// TODO: Add support for absolute values, reformulating as |x| = (x^2)^(1/2).
// First we need to have an Abs node in oximo Expr.

mod options;
mod translate;

pub use options::BaronOptions;
pub use translate::solve;

use oximo_core::{Model, ModelKind};
use oximo_solver::{Solver, SolverError, SolverResult};

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

impl Solver for Baron {
    type Options = BaronOptions;

    fn name(&self) -> &str {
        "baron"
    }

    fn supports(&self, kind: ModelKind) -> bool {
        // BARON is a global solver for all of the following model classes.
        matches!(
            kind,
            ModelKind::LP
                | ModelKind::MILP
                | ModelKind::QP
                | ModelKind::MIQP
                | ModelKind::NLP
                | ModelKind::MINLP
        )
    }

    fn solve(&mut self, model: &Model, opts: &BaronOptions) -> Result<SolverResult, SolverError> {
        translate::solve(model, opts, self.exec.as_deref())
    }
}
