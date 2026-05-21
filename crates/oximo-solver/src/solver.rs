use oximo_core::{Model, ModelKind};

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

    /// Solves the given `Model` using this solver.
    ///
    /// # Errors
    ///
    /// Returns a [`SolverError`] if the model is unsupported or if the solver backend fails.
    fn solve(&mut self, model: &Model, opts: &Self::Options) -> Result<SolverResult, SolverError>;
}
