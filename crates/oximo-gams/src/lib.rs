#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

mod options;
mod solver_options;
mod translate;

#[cfg(feature = "benchmark-support")]
#[doc(hidden)]
pub use translate::benchmark_support;

pub use options::{GamsOptions, GamsSolver};

// The per-solver `Gams<Name>Options` structs and `GamsSolverConfig` are generated
// from `option-snapshots/` (see `build.rs`).
pub use solver_options::*;
pub use translate::solve;

use oximo_core::{Model, ModelKind, SosType};
use oximo_solver::{Solver, SolverError, SolverResult};

/// GAMS solver backend.
///
/// Writes the model to a temporary `.gms` file, invokes the GAMS executable,
/// and returns the parsed [`SolverResult`].
#[derive(Debug, Default, Clone)]
pub struct Gams {
    /// Optional override for the GAMS executable path. When `None`, `"gams"` is
    /// looked up from the system `PATH`. Overridden per-call by
    /// [`GamsOptions::gams_path`].
    pub exec: Option<String>,
}

impl Gams {
    /// Create a backend that uses `gams` from `PATH`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a backend pointing at an explicit GAMS executable path.
    pub fn with_exec(path: impl Into<String>) -> Self {
        Self { exec: Some(path.into()) }
    }
}

/// Display name for this backend; the single source for both [`Solver::name`]
/// and the `solver_name` stamped on every [`SolverResult`].
pub(crate) const NAME: &str = "GAMS";

/// GAMS handles every kind oximo classifies: quadratic constraints route
/// through the QCP/MIQCP solve types and explicit SOC constraints are emitted
/// as quadratic `sqr(..)` rows. Whether the selected sub-solver copes is
/// checked separately (see `validate_solver`).
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

impl Solver for Gams {
    type Options = GamsOptions;

    fn name(&self) -> &str {
        NAME
    }

    fn supports(&self, kind: ModelKind) -> bool {
        supported(kind)
    }

    fn domain_support(&self, _domain: oximo_core::Domain) -> oximo_solver::Capability {
        oximo_solver::Capability::Native
    }

    fn supports_sos(&self, _sos_type: SosType) -> bool {
        true
    }

    fn supports_indicators(&self) -> bool {
        true
    }

    fn validate_model_extra(
        &self,
        model: &Model,
        opts: &Self::Options,
    ) -> Vec<oximo_solver::CapabilityDiagnostic> {
        let mut diagnostics = Vec::new();
        if !matches!(model.kind(), ModelKind::SDP | ModelKind::MISDP)
            && let Err(error) = translate::validate_solver_type(
                opts,
                model.kind(),
                translate::gams_solve_type(model.kind()),
            )
        {
            diagnostics.push(oximo_solver::CapabilityDiagnostic {
                constraint: None,
                location: "GAMS subsolver".into(),
                reason: error.to_string(),
            });
        }
        if let Err(error) = translate::validate_indicator_solver(model, opts) {
            diagnostics.push(oximo_solver::CapabilityDiagnostic {
                constraint: None,
                location: "GAMS indicators".into(),
                reason: error.to_string(),
            });
        }
        diagnostics
    }

    fn solve(&mut self, model: &Model, opts: &GamsOptions) -> Result<SolverResult, SolverError> {
        translate::solve(model, opts, self.exec.as_deref())
    }
}
