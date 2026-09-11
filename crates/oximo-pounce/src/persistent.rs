//! A resident POUNCE handle that keeps the built derivative oracle alive across
//! solves and warm-starts each solve from the previous iterate.

use std::time::Instant;

use oximo_core::{Model, ModelKind};
use oximo_solver::prepare::LoweringContext;
use oximo_solver::{Solver, SolverError, SolverResult};

use crate::convex::{self, Route};
use crate::options::PounceOptions;
use crate::translate::{
    WarmStart, assemble, reject_semi_domains, run_nlp_with_retries, setup_prepared,
};

#[cfg(feature = "enzyme")]
use crate::exact as backend;
#[cfg(not(feature = "enzyme"))]
use crate::stable as backend;

struct NlpState {
    oracle: backend::Oracle,
    warm: Option<WarmStart>,
}

struct ConvexState {
    route: Route,
    problem: convex::Problem,
    warm: Option<pounce_rs::convex::QpWarmStart>,
    active: Option<convex::ActivePersistent>,
}

enum State {
    Nlp(NlpState),
    Convex(Box<ConvexState>),
}

struct CachedValidation {
    options: PounceOptions,
    result: Result<(), String>,
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
    validation: Option<CachedValidation>,
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
        self.validation = None;
    }

    fn solve_resident(
        &mut self,
        model: &Model,
        opts: &PounceOptions,
    ) -> Result<SolverResult, SolverError> {
        if model.has_active_sos_constraints() {
            return Err(SolverError::UnsupportedSos);
        }
        reject_semi_domains(model)?;

        let prepared = LoweringContext::new(model)?;
        let route = convex::route(&prepared, opts)?;
        if route == Route::Nlp {
            return self.solve_nlp(model, &prepared, opts);
        }
        self.solve_convex(model, &prepared, opts, route)
    }

    fn solve_nlp(
        &mut self,
        model: &Model,
        prepared: &LoweringContext<'_>,
        opts: &PounceOptions,
    ) -> Result<SolverResult, SolverError> {
        self.solve_nlp_since(model, prepared, opts, Instant::now())
    }

    fn solve_nlp_since(
        &mut self,
        model: &Model,
        prepared: &LoweringContext<'_>,
        opts: &PounceOptions,
        started: Instant,
    ) -> Result<SolverResult, SolverError> {
        let prep = setup_prepared(prepared, opts)?;
        let state = match &mut self.state {
            Some(State::Nlp(state)) if backend::try_reuse(&state.oracle, model) => state,
            slot => {
                *slot = Some(State::Nlp(NlpState { oracle: backend::build(model)?, warm: None }));
                let Some(State::Nlp(state)) = slot else { unreachable!() };
                state
            }
        };
        let mut outcome =
            run_nlp_with_retries(model, &state.oracle, &prep, opts, state.warm.as_ref())?;
        let elapsed = started.elapsed();
        state.warm = outcome.warm.take();
        Ok(assemble(prep.sign, outcome, elapsed, model.num_variables()))
    }

    fn solve_convex(
        &mut self,
        model: &Model,
        prepared: &LoweringContext<'_>,
        opts: &PounceOptions,
        route: Route,
    ) -> Result<SolverResult, SolverError> {
        self.validate_convex_options(opts)?;
        let problem = convex::build_problem(prepared, opts)?;
        let started = Instant::now();
        let state = match &mut self.state {
            Some(State::Convex(state))
                if state.route == route && state.problem.same_structure(&problem) =>
            {
                state.problem = problem;
                state
            }
            slot => {
                let active = (route == Route::QpActiveSet).then(convex::ActivePersistent::new);
                *slot = Some(State::Convex(Box::new(ConvexState {
                    route,
                    problem,
                    warm: None,
                    active,
                })));
                let Some(State::Convex(state)) = slot else { unreachable!() };
                state
            }
        };
        let solution = if route == Route::QpActiveSet {
            state
                .active
                .as_mut()
                .expect("active-set state exists for active-set route")
                .solve(&state.problem, opts)
        } else {
            convex::run(&state.problem, opts, route, state.warm.as_ref())
        };
        if convex::should_fallback_to_nlp(model, opts, &solution)? {
            return self.solve_nlp_since(model, prepared, opts, started);
        }
        let elapsed = started.elapsed();
        let mut outcome = convex::outcome(&state.problem, opts, route, &solution);
        state.warm = (route != Route::QpActiveSet && outcome.termination.admits_primal())
            .then(|| convex::warm_from_solution(route, &state.problem, &solution));
        let sign = state.problem.sign();
        outcome.warm = None;
        Ok(assemble(sign, outcome, elapsed, model.num_variables()))
    }

    fn validate_convex_options(&mut self, opts: &PounceOptions) -> Result<(), SolverError> {
        if let Some(cached) = &self.validation
            && cached.options == *opts
        {
            return cached.result.clone().map_err(SolverError::Backend);
        }
        self.validation = None;
        let result = convex::validate_options(opts);
        let cached = match &result {
            Ok(()) => Ok(()),
            Err(SolverError::Backend(message)) => Err(message.clone()),
            Err(_) => return result,
        };
        self.validation = Some(CachedValidation { options: opts.clone(), result: cached });
        result
    }
}

impl Solver for PouncePersistent {
    type Options = PounceOptions;

    fn name(&self) -> &str {
        "pounce"
    }

    fn supports(&self, kind: ModelKind) -> bool {
        matches!(
            kind,
            ModelKind::LP | ModelKind::QP | ModelKind::QCP | ModelKind::SOCP | ModelKind::NLP
        )
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convex_validation_cache_tracks_options_and_errors() {
        let mut solver = PouncePersistent::new();
        let valid = PounceOptions::default().qp_tau(0.9);
        solver.validate_convex_options(&valid).unwrap();
        assert!(
            solver
                .validation
                .as_ref()
                .is_some_and(|cached| { cached.options == valid && cached.result.is_ok() })
        );
        solver.validate_convex_options(&valid).unwrap();

        let invalid = PounceOptions::default().set("not_a_real_option", true);
        let first = solver.validate_convex_options(&invalid).unwrap_err().to_string();
        let second = solver.validate_convex_options(&invalid).unwrap_err().to_string();
        assert_eq!(first, second);
        assert!(
            solver
                .validation
                .as_ref()
                .is_some_and(|cached| { cached.options == invalid && cached.result.is_err() })
        );

        solver.reset();
        assert!(solver.validation.is_none());
    }
}
