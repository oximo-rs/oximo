use oximo_core::{Model, ModelKind};

use crate::options::SolverOptions;
use crate::result::SolverResult;
use crate::status::SolverError;

/// Concrete solver backend.
///
/// Backends live in their own crates and the umbrella `oximo` crate 
/// gates them behind cargo features. Implementors translate the 
/// `Model` into their internal representation, solve, and return 
/// a populated [`SolverResult`].
pub trait Solver {
    fn name(&self) -> &str;

    fn supports(&self, kind: ModelKind) -> bool;

    /// Solves the given `Model` using this solver.
    ///
    /// # Errors
    ///
    /// Returns a [`SolverError`] if the model is unsupported or if the solver backend fails.
    fn solve(&mut self, model: &Model, opts: &SolverOptions) -> Result<SolverResult, SolverError>;
}
