#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

#[cfg(all(feature = "bundled", feature = "system"))]
compile_error!("SCIP installation modes are exclusive: disable default features to select system");
#[cfg(not(any(feature = "bundled", feature = "system")))]
compile_error!("select bundled (default) or system for oximo-scip");

mod expression;
mod options;
mod persistent;
pub mod plugins;
mod result;
mod translate;

pub use options::{ScipOptionValue, ScipOptions, ScipSetting};
pub use persistent::ScipPersistent;
pub use plugins::*;
pub use result::ScipSolveOutput;
pub use russcip as native;

use oximo_core::Model;
use oximo_solver::{
    ExpressionOperator, MODEL_KINDS, PersistentSolver, Solver, SolverError, SolverProfile,
    SolverResult,
};
use std::{borrow::Cow, fmt::Debug, rc::Rc, sync::LazyLock};

type Factory = dyn Fn(&mut ScipPluginRegistry<'_>) -> Result<(), SolverError>;

/// SCIP backend.
///
/// `solve` builds a fresh native problem, while `persistent` keeps
/// compatible original problems resident. Plugin factories run on every build.
#[derive(Clone, Default)]
pub struct Scip {
    pub(crate) factory: Option<Rc<Factory>>,
}
impl Debug for Scip {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scip").field("plugins", &self.factory.is_some()).finish()
    }
}
impl Scip {
    pub fn new() -> Self {
        Self::default()
    }
    #[must_use]
    pub fn with_plugins(
        mut self,
        factory: impl Fn(&mut ScipPluginRegistry<'_>) -> Result<(), SolverError> + 'static,
    ) -> Self {
        self.factory = Some(Rc::new(factory));
        self
    }
    /// Solve and include native statistics and priced-column solutions.
    ///
    /// # Errors
    ///
    /// Returns translation, option, plugin, or native solve errors.
    pub fn solve_detailed(
        &mut self,
        model: &Model,
        opts: &ScipOptions,
    ) -> Result<ScipSolveOutput, SolverError> {
        self.persistent().solve_detailed(model, opts)
    }
    /// Export the translated original problem using a SCIP writer, e.g. `cip`,
    /// `lp`, or `mps`. The writer reports unsupported format features.
    ///
    /// # Errors
    ///
    /// Returns errors from translation, registration, or SCIP's writer.
    pub fn write_model(
        &self,
        model: &Model,
        opts: &ScipOptions,
        path: &str,
        format: &str,
    ) -> Result<(), SolverError> {
        use russcip::ModelWithProblem;
        check_name(path)?;
        check_name(format)?;
        let snapshot = translate::Snapshot::prepare(model)?;
        boundary(|| {
            let (mut native, mapping) = snapshot.build(opts)?;
            if let Some(factory) = &self.factory {
                factory(&mut ScipPluginRegistry::new(&mut native, &mapping, Rc::default()))?;
            }
            native.write(path, format, true).map_err(backend)
        })
    }
}
static PROFILE: LazyLock<SolverProfile> = LazyLock::new(build_profile);

fn build_profile() -> SolverProfile {
    use oximo_expr::ExprClass::{Linear, Nonlinear, Quadratic};
    use oximo_solver::VariableDomain::{Binary, Integer, Real, SemiContinuous, SemiInteger};

    SolverProfile::new("SCIP")
        .with_scalar_constraints([Linear, Quadratic, Nonlinear])
        .with_scalar_objectives([Linear, Quadratic, Nonlinear])
        .with_soc(true)
        .with_kinds(MODEL_KINDS.into_iter().filter(|kind| {
            !matches!(kind, oximo_core::ModelKind::SDP | oximo_core::ModelKind::MISDP)
        }))
        .with_domains([Real, Integer, Binary, SemiContinuous, SemiInteger])
        .with_operators(
            ExpressionOperator::ALL.iter().copied().filter(|operator| supports_operator(*operator)),
        )
        .with_sos([oximo_core::SosType::Sos1, oximo_core::SosType::Sos2])
        .with_indicators(true)
}

impl Solver for Scip {
    type Options = ScipOptions;
    fn profile(&self, _opts: &Self::Options) -> Cow<'_, SolverProfile> {
        Cow::Borrowed(&PROFILE)
    }

    fn name(&self) -> &str {
        "SCIP"
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

        translate::Snapshot::prepare(model)
            .err()
            .map(|error| {
                oximo_solver::RequirementDiagnostic::adapter("SCIP translation", error.to_string())
            })
            .into_iter()
            .collect()
    }

    fn solve(&mut self, model: &Model, opts: &ScipOptions) -> Result<SolverResult, SolverError> {
        Ok(self.solve_detailed(model, opts)?.result)
    }
}
impl PersistentSolver for Scip {
    type Handle = ScipPersistent;
    fn persistent(&self) -> Self::Handle {
        ScipPersistent::with_solver(self.clone())
    }
}

pub(crate) fn backend(e: impl Debug) -> SolverError {
    SolverError::Backend(format!("SCIP: {e:?}"))
}
pub(crate) fn finite(v: f64) -> Result<(), SolverError> {
    if v.is_finite() { Ok(()) } else { Err(backend("non-finite numeric data")) }
}
pub(crate) fn check_name(name: &str) -> Result<(), SolverError> {
    if name.contains('\0') {
        Err(backend("names and string parameters cannot contain NUL"))
    } else {
        Ok(())
    }
}
pub(crate) fn boundary<T>(f: impl FnOnce() -> Result<T, SolverError>) -> Result<T, SolverError> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(r) => r,
        Err(p) => Err(backend(
            p.downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| p.downcast_ref::<&str>().copied())
                .unwrap_or("upstream panicked"),
        )),
    }
}

pub(crate) fn supports_operator(operator: oximo_solver::ExpressionOperator) -> bool {
    use oximo_expr::UnaryOp::{Abs, Cos, Exp, Log, Neg, Sin, Sqrt};
    use oximo_solver::ExpressionOperator::{Atan2, Max, Min, Unary, VariablePower};
    match operator {
        Unary(op) => matches!(op, Neg | Abs | Sqrt | Exp | Log | Sin | Cos),
        Atan2 | Min | Max | VariablePower => false,
        _ => true,
    }
}

/// Called only for local translation failures, before any native state changes.
pub(crate) fn translation_rejection(error: SolverError) -> SolverError {
    SolverError::IncompatibleModel(Box::new(oximo_solver::CompatibilityReport {
        solver: "SCIP".into(),
        errors: vec![oximo_solver::RequirementDiagnostic::adapter(
            "SCIP translation",
            error.to_string(),
        )],
        ..oximo_solver::CompatibilityReport::default()
    }))
}
