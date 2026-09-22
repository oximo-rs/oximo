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

    /// Whether this backend can consume all features present in `model`.
    fn supports_model(&self, model: &Model) -> bool {
        self.supports(model.kind())
            && model
                .sos_constraints()
                .iter()
                .filter(|constraint| constraint.active)
                .all(|constraint| self.supports_sos(constraint.sos_type))
            && (!model.has_active_indicator_constraints() || self.supports_indicators())
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
}
