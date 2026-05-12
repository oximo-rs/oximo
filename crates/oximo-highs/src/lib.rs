//! HiGHS LP / MILP backend for oximo.
//!
//! ```no_run
//! use oximo_core::prelude::*;
//! use oximo_highs::{Highs, HighsOptions};
//! use oximo_solver::Solver;
//!
//! let m = Model::new("toy");
//! let x = m.var("x").lb(0.0).build();
//! m.minimize(x);
//! let mut s = Highs::default();
//! let res = s.solve(&m, &HighsOptions::default()).unwrap();
//! assert!(res.status.has_solution());
//! ```
#![forbid(unsafe_code)]

mod options;
mod translate;

pub use options::{HighsMethod, HighsOptions};
pub use translate::solve;

use oximo_core::{Model, ModelKind};
use oximo_solver::{Solver, SolverError, SolverResult};

/// HiGHS solver handle. Cheap to construct. The actual HiGHS instance is
/// created per `solve` call so models can be re-used or shared across solves.
///
/// TODO: Can we do this better in the future?
#[derive(Debug, Default, Clone, Copy)]
pub struct Highs;

impl Solver for Highs {
    type Options = HighsOptions;

    fn name(&self) -> &str {
        "highs"
    }

    fn supports(&self, kind: ModelKind) -> bool {
        matches!(kind, ModelKind::LP | ModelKind::MILP)
    }

    fn solve(&mut self, model: &Model, opts: &HighsOptions) -> Result<SolverResult, SolverError> {
        translate::solve(model, opts)
    }
}
