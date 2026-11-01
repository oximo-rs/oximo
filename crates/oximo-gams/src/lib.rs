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

use oximo_core::{Model, ModelKind};
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

    fn profile(&self, opts: &Self::Options) -> std::borrow::Cow<'_, oximo_solver::SolverProfile> {
        std::borrow::Cow::Owned({
            use oximo_expr::ExprClass::{Linear, Nonlinear, Quadratic};
            use oximo_solver::VariableDomain::{
                Binary, Integer, Real, SemiContinuous, SemiInteger,
            };

            let profile = oximo_solver::SolverProfile::new(self.name())
                .with_scalar_constraints([Linear, Quadratic, Nonlinear])
                .with_scalar_objectives([Linear, Quadratic, Nonlinear])
                .with_soc(true)
                .with_kinds(oximo_solver::MODEL_KINDS.into_iter().filter(|kind| {
                    supported(*kind)
                        && opts.solver.as_ref().is_none_or(|config| config.supports(*kind))
                }))
                .with_domains([Real, Integer, Binary, SemiContinuous, SemiInteger])
                .with_operators(
                    oximo_solver::ExpressionOperator::ALL
                        .iter()
                        .copied()
                        .filter(|operator| supports_operator(*operator)),
                )
                .with_sos([oximo_core::SosType::Sos1, oximo_core::SosType::Sos2])
                .with_indicators(translate::indicator_supported(opts));

            let profile = if opts.solver.as_ref().is_some_and(|config| {
                !crate::solver_options::solver_supports_type(config.gams_name(), "DNLP")
            }) {
                profile.reject_combination(
                    |f| f.kind == oximo_core::ModelKind::NLP && f.has_variable_nonsmooth,
                    "the selected GAMS solver does not support DNLP models",
                )
            } else {
                profile
            };

            let profile = if let Some(config) = &opts.solver {
                profile.named(format!("GAMS ({})", config.gams_name()))
            } else {
                profile
            };

            if opts.solver.as_ref().is_none_or(|config| {
                crate::solver_options::supported_solve_types(config.gams_name()).is_none()
            }) {
                profile.with_external_uncertainty(
                    "the externally selected GAMS solver has no locally confirmed capability table",
                )
            } else {
                profile
            }
        })
    }

    fn name(&self) -> &str {
        NAME
    }

    fn solve(&mut self, model: &Model, opts: &GamsOptions) -> Result<SolverResult, SolverError> {
        translate::solve(model, opts, self.exec.as_deref())
    }
}

pub(crate) fn supports_operator(operator: oximo_solver::ExpressionOperator) -> bool {
    use oximo_expr::UnaryOp::{Acosh, Asinh, Atanh, Cbrt, Expm1, Log1p};
    use oximo_solver::ExpressionOperator::Unary;
    !matches!(operator, Unary(Cbrt | Expm1 | Log1p | Asinh | Acosh | Atanh))
}
