use oximo_core::{Model, ModelKind, SosType};

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

    fn supports(&self, kind: ModelKind) -> bool;

    /// Whether this backend can consume native SOS constraints of `sos_type`.
    fn supports_sos(&self, _sos_type: SosType) -> bool {
        false
    }

    /// Whether this backend can consume native indicator constraints.
    fn supports_indicators(&self) -> bool {
        false
    }

    /// Whether this backend supports real affine PSD matrix constraints.
    fn supports_psd(&self) -> bool {
        false
    }

    /// Additional backend-specific restrictions after the shared capability checks.
    ///
    /// Override this hook to restrict supported model structures while retaining
    /// the default GDP, model-kind, PSD, SOS, and indicator checks.
    fn supports_model_extra(&self, _model: &Model) -> bool {
        true
    }

    /// Whether this backend can consume all features present in `model`.
    ///
    /// Prefer overriding [`Solver::supports_model_extra`] for additional restrictions
    /// so this method continues to apply the shared capability checks.
    fn supports_model(&self, model: &Model) -> bool {
        !model.has_unreformulated_gdp()
            && self.supports(model.kind())
            && (!model.has_active_psd_constraints() || self.supports_psd())
            && model
                .sos_constraints()
                .iter()
                .filter(|constraint| constraint.active)
                .all(|constraint| self.supports_sos(constraint.sos_type))
            && (!model.has_active_indicator_constraints() || self.supports_indicators())
            && self.supports_model_extra(model)
    }

    /// Solves the given `Model` using this solver.
    ///
    /// # Errors
    ///
    /// Returns a [`SolverError`] if the model is unsupported or if the solver backend fails.
    fn solve(&mut self, model: &Model, opts: &Self::Options) -> Result<SolverResult, SolverError>;
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use oximo_core::{SosType, constraint, variable};

    use super::*;

    #[derive(Debug)]
    struct NoSos;

    impl Solver for NoSos {
        type Options = ();

        fn name(&self) -> &str {
            "no-sos"
        }

        fn supports(&self, kind: ModelKind) -> bool {
            matches!(kind, ModelKind::LP | ModelKind::MILP)
        }

        fn solve(&mut self, _model: &Model, _opts: &()) -> Result<SolverResult, SolverError> {
            unreachable!("capability test solver is never solved")
        }
    }

    #[derive(Debug)]
    struct NativeSos;

    impl Solver for NativeSos {
        type Options = ();

        fn name(&self) -> &str {
            "native-sos"
        }

        fn supports(&self, kind: ModelKind) -> bool {
            matches!(kind, ModelKind::LP | ModelKind::MILP)
        }

        fn supports_sos(&self, _sos_type: SosType) -> bool {
            true
        }

        fn solve(&mut self, _model: &Model, _opts: &()) -> Result<SolverResult, SolverError> {
            unreachable!("capability test solver is never solved")
        }
    }

    fn sos_model() -> Model {
        let m = Model::new("capabilities");
        variable!(m, x);
        variable!(m, y);
        constraint!(m, bound, x + y <= 1.0);
        m.add_sos_constraint("choice", SosType::Sos1, [(x, 1.0), (y, 2.0)]);
        m
    }

    #[test]
    fn supports_model_checks_native_sos_capability() {
        let model = sos_model();
        assert!(!NoSos.supports_model(&model));
        assert!(NativeSos.supports_model(&model));

        let transformed = model
            .to_reformulated_sos_model(
                oximo_core::SosReformulationOptions::default().with_fallback_big_m(100.0),
            )
            .unwrap();
        assert!(NoSos.supports_model(&transformed));
        assert!(NativeSos.supports_model(&transformed));
    }

    #[derive(Debug, Default)]
    struct RestrictedModel {
        checks: Cell<usize>,
    }

    impl Solver for RestrictedModel {
        type Options = ();

        fn name(&self) -> &str {
            "restricted-model"
        }

        fn supports(&self, kind: ModelKind) -> bool {
            matches!(kind, ModelKind::LP | ModelKind::MILP)
        }

        fn supports_model_extra(&self, model: &Model) -> bool {
            self.checks.set(self.checks.get() + 1);
            model.num_variables() <= 1
        }

        fn solve(&mut self, _model: &Model, _opts: &()) -> Result<SolverResult, SolverError> {
            unreachable!("capability test solver is never solved")
        }
    }

    #[test]
    fn backend_hook_can_restrict_support_and_runs_after_shared_checks() {
        let solver = RestrictedModel::default();
        let model = Model::new("backend restriction");
        variable!(model, _x);
        assert!(solver.supports_model(&model));
        assert_eq!(solver.checks.get(), 1);

        variable!(model, _y);
        assert!(!solver.supports_model(&model));
        assert_eq!(solver.checks.get(), 2);

        assert!(!solver.supports_model(&sos_model()));
        assert_eq!(solver.checks.get(), 2);

        let nonlinear = Model::new("unsupported kind");
        variable!(nonlinear, z);
        constraint!(nonlinear, z.exp() <= 2.0);
        assert!(!solver.supports_model(&nonlinear));
        assert_eq!(solver.checks.get(), 2);
    }
}
