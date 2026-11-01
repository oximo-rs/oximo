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
use oximo_solver::{
    ExpressionOperator, Iis, InfeasibilityDiagnosis, MODEL_KINDS, Solver, SolverError,
    SolverProfile, SolverResult,
};
use std::{borrow::Cow, sync::LazyLock};

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

static PROFILE: LazyLock<SolverProfile> = LazyLock::new(build_profile);

fn build_profile() -> SolverProfile {
    use oximo_expr::ExprClass::{Linear, Nonlinear, Quadratic};
    use oximo_solver::VariableDomain::{Binary, Integer, Real};

    SolverProfile::new(NAME)
        .with_scalar_constraints([Linear, Quadratic, Nonlinear])
        .with_scalar_objectives([Linear, Quadratic, Nonlinear])
        .with_soc(true)
        .with_kinds(MODEL_KINDS.into_iter().filter(|kind| supported(*kind)))
        .with_domains([Real, Integer, Binary])
        .with_operators(
            ExpressionOperator::ALL.iter().copied().filter(|operator| supports_operator(*operator)),
        )
}

impl Solver for Baron {
    type Options = BaronOptions;

    fn profile(&self, _opts: &Self::Options) -> Cow<'_, SolverProfile> {
        Cow::Borrowed(&PROFILE)
    }

    fn name(&self) -> &str {
        NAME
    }

    fn validate_requirements(
        &self,
        model: &Model,
        opts: &Self::Options,
        requirements: &oximo_solver::ModelRequirements,
    ) -> Vec<oximo_solver::RequirementDiagnostic> {
        if !self.profile(opts).check(requirements).is_supported() {
            return Vec::new();
        }

        translate::validate_model(model, opts)
            .err()
            .map(|error| {
                oximo_solver::RequirementDiagnostic::adapter("BARON translation", error.to_string())
            })
            .into_iter()
            .collect()
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

pub(crate) fn supports_operator(operator: oximo_solver::ExpressionOperator) -> bool {
    use oximo_expr::UnaryOp::{Abs, Exp, Exp2, Log, Log10, Neg, Sqrt};
    use oximo_solver::ExpressionOperator::{Atan2, Max, Min, Unary};
    match operator {
        Unary(op) => matches!(op, Neg | Abs | Sqrt | Exp | Exp2 | Log | Log10),
        Atan2 | Min | Max => false,
        _ => true,
    }
}
