//! A resident POUNCE handle that keeps the built derivative oracle alive across
//! solves and warm-starts each solve from the previous iterate.

use std::time::Instant;

use oximo_core::{Model, ModelKind};
use oximo_solver::{Solver, SolverError, SolverResult};

use crate::options::PounceOptions;
use crate::translate::{WarmStart, assemble, setup};

#[cfg(feature = "enzyme")]
use crate::exact as backend;
#[cfg(not(feature = "enzyme"))]
use crate::stable as backend;

struct State {
    oracle: backend::Oracle,
    warm: Option<WarmStart>,
}

/// A stateful POUNCE handle that keeps the derivative build resident across
/// solves. Created by [`Pounce::persistent`](crate::Pounce).
///
/// When the next model has the same variables, objective, and constraint
/// expressions with an unchanged sparsity pattern, the resident oracle
/// is refreshed in place, reusing the compiled tapes (and, on the `enzyme`
/// path, the exact jacobians/Hessians structure) instead of rebuilding.
/// Also, the solve is warm-started from the previous iterate.
/// Any structural change rebuilds.
///
/// A failed solve clears the resident state. The next call rebuilds from scratch.
#[derive(Default)]
pub struct PouncePersistent {
    state: Option<State>,
}

impl std::fmt::Debug for PouncePersistent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PouncePersistent").field("resident", &self.state.is_some()).finish()
    }
}

impl PouncePersistent {
    /// A fresh handle with no model loaded. The first solve builds it.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Drop the resident oracle so the next [`solve`](Solver::solve) rebuilds
    /// from scratch (and starts from the model's initial point).
    pub fn reset(&mut self) {
        self.state = None;
    }

    fn solve_resident(
        &mut self,
        model: &Model,
        opts: &PounceOptions,
    ) -> Result<SolverResult, SolverError> {
        let prep = setup(model)?;
        let started = Instant::now();

        let state = match &mut self.state {
            Some(state) if backend::try_reuse(&state.oracle, model) => state,
            state => state.insert(State { oracle: backend::build(model)?, warm: None }),
        };
        let mut outcome = backend::run(&state.oracle, &prep, opts, state.warm.as_ref())?;
        let elapsed = started.elapsed();

        state.warm = outcome.warm.take();
        Ok(assemble(prep.sign, outcome, elapsed))
    }
}

impl Solver for PouncePersistent {
    type Options = PounceOptions;

    fn name(&self) -> &str {
        "pounce"
    }

    fn supports(&self, kind: ModelKind) -> bool {
        matches!(kind, ModelKind::LP | ModelKind::QP | ModelKind::QCP | ModelKind::NLP)
    }

    fn solve(&mut self, model: &Model, opts: &PounceOptions) -> Result<SolverResult, SolverError> {
        match self.solve_resident(model, opts) {
            Ok(result) => Ok(result),
            Err(e) => {
                self.state = None;
                Err(e)
            }
        }
    }
}
