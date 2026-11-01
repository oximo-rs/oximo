use oximo_core::Model;

use crate::result::SolverResult;
use crate::status::SolverError;

/// Concrete solver backend.
///
/// Backends live in their own crates and the umbrella `oximo` crate
/// gates them behind cargo features. Implementors translate the
/// `Model` into their internal representation, solve, and return
/// a populated [`SolverResult`].
///
/// Each backend defines its own [`Options`](Solver::Options) type so users get
/// LSP autocomplete and compile-time validation on the options that actually
/// apply. The `oximo_solver` crate ships shared building blocks
/// ([`UniversalOptions`](crate::UniversalOptions),
/// [`UniversalOptionsExt`](crate::UniversalOptionsExt))
/// for backends to compose into their own structs.
pub trait Solver {
    /// Backend-specific options struct. Use `()` for solvers without any
    /// tunables.
    type Options;

    fn name(&self) -> &str;

    /// Declare the interfaces supported by this backend and option configuration.
    fn profile(&self, opts: &Self::Options) -> std::borrow::Cow<'_, crate::SolverProfile>;

    /// Match current requirements and perform local, side-effect-free data checks.
    fn validate_model(&self, model: &Model, opts: &Self::Options) -> crate::CompatibilityReport {
        let requirements = crate::requirements::for_matching(model);
        let mut report = self.profile(opts).check_with_origins(model, &requirements);

        // Data checks can depend on an objective and an already reformulated model.
        if model.ensure_objective_declared().is_ok() && !model.has_unreformulated_gdp() {
            report.errors.extend(self.validate_requirements(model, opts, &requirements));
        }
        report
    }

    /// Resolve local numeric, representation, or route restrictions without native setup.
    /// The supplied summary may omit origins. Inspect `model` or explicitly call
    /// `ModelRequirements::infer` when a local diagnostic needs source locations.
    fn validate_requirements(
        &self,
        _model: &Model,
        _opts: &Self::Options,
        _requirements: &crate::ModelRequirements,
    ) -> Vec<crate::RequirementDiagnostic> {
        Vec::new()
    }

    /// Solves the given `Model` using this solver.
    /// Results and all solution points must carry the model identity and the
    /// parameter snapshot used to translate or update the native problem.
    /// Use [`crate::reconstruct::finalize_result`] to normalize the result and suppress
    /// semantic dual fallback for alternative lowerings without mappings.
    /// Implementors must validate before native setup, external execution, callbacks,
    /// or resident-state changes. Use `self.validate_model(model, opts).into_result()?`,
    /// or `self.profile(opts).check_model(model)` together with local data validation
    /// that reuses the translation needed by the solve.
    ///
    /// # Errors
    ///
    /// Returns a [`SolverError`] if the model is unsupported or if the solver backend fails.
    fn solve(&mut self, model: &Model, opts: &Self::Options) -> Result<SolverResult, SolverError>;
}
