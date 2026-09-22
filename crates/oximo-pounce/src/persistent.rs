//! A resident POUNCE handle that keeps the built derivative oracle alive across
//! solves and warm-starts each solve from the previous iterate.

use std::time::Instant;

use oximo_core::{Model, ModelKind};
use oximo_solver::prepare::LoweringContext;
use oximo_solver::{Solver, SolverError, SolverResult};

use crate::convex::{self, Route};
use crate::options::{PounceAlgorithm, PounceOptions};
use crate::translate::{
    WarmStart, assemble, reject_semi_domains, run_nlp_retries_after, selected_algorithm,
    setup_prepared,
};

#[cfg(feature = "enzyme")]
use crate::exact as backend;
#[cfg(not(feature = "enzyme"))]
use crate::stable as backend;

struct NlpState {
    oracle: backend::Oracle,
    warm: Option<WarmStart>,
    resident: Option<backend::Resident>,
    params: Vec<u64>,
}

struct ConvexState {
    route: Route,
    problem: convex::Problem,
    warm: Option<pounce_rs::convex::QpWarmStart>,
    ipm: Option<convex::IpmPersistent>,
    active: Option<convex::ActivePersistent>,
}

enum State {
    Nlp(Box<NlpState>),
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
        if model.has_active_indicator_constraints() {
            return Err(SolverError::UnsupportedIndicator);
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
        let params = parameter_snapshot(model);
        let mut params_changed = false;
        let state = match &mut self.state {
            Some(State::Nlp(state)) if backend::try_reuse(&state.oracle, model) => {
                params_changed = state.params != params;
                state.params = params;
                state
            }
            slot => {
                *slot = Some(State::Nlp(Box::new(NlpState {
                    oracle: backend::build(model)?,
                    warm: None,
                    resident: None,
                    params,
                })));
                let Some(State::Nlp(state)) = slot else { unreachable!() };
                state
            }
        };
        let primary_started = Instant::now();
        let primary = if selected_algorithm(opts)? == PounceAlgorithm::InteriorPoint
            && backend::supports_resident(&state.oracle)
        {
            let rebuild =
                state.resident.as_ref().is_none_or(|resident| !resident.matches_options(opts));
            if rebuild {
                state.resident = Some(backend::Resident::new(&state.oracle, &prep, opts)?);
            }
            let resident = state.resident.as_mut().expect("resident TNLP session exists");
            if params_changed && opts.effective_bool("presolve_auxiliary") == Some(true) {
                resident.invalidate();
            }
            resident.solve(&prep, opts, state.warm.as_ref())?
        } else {
            state.resident = None;
            backend::run(model, &state.oracle, &prep, opts, state.warm.as_ref())?
        };
        let mut outcome =
            run_nlp_retries_after(model, &state.oracle, &prep, opts, primary_started, primary)?;
        let elapsed = started.elapsed();
        state.warm = outcome.warm.take();
        Ok(assemble(prep.sign, outcome, elapsed, model.id(), model.num_variables()))
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
                let ipm = (route == Route::QpIpm).then(convex::IpmPersistent::new);
                let active = (route == Route::QpActiveSet).then(convex::ActivePersistent::new);
                *slot = Some(State::Convex(Box::new(ConvexState {
                    route,
                    problem,
                    warm: None,
                    ipm,
                    active,
                })));
                let Some(State::Convex(state)) = slot else { unreachable!() };
                state
            }
        };
        let solution = match route {
            Route::QpIpm => state.ipm.as_mut().expect("IPM state exists for IPM route").solve(
                &state.problem,
                opts,
                state.warm.as_ref(),
            ),
            Route::QpActiveSet => state
                .active
                .as_mut()
                .expect("active-set state exists for active-set route")
                .solve(&state.problem, opts),
            Route::Socp => convex::run(&state.problem, opts, route, state.warm.as_ref()),
            Route::Nlp => unreachable!("NLP route passed to convex resident solver"),
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
        Ok(assemble(sign, outcome, elapsed, model.id(), model.num_variables()))
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

fn parameter_snapshot(model: &Model) -> Vec<u64> {
    let arena = model.arena();
    oximo_autodiff::params_snapshot(&arena).into_iter().map(f64::to_bits).collect()
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
    use crate::options::PounceSolverSelection;
    use oximo_core::{constraint, objective, param, variable};

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

    #[test]
    fn convex_ipm_retains_and_rebuilds_presolve_by_numeric_fingerprint() {
        let model = Model::new("convex_presolve_session");
        param!(model, weight = 1.0);
        variable!(model, 0.0 <= x <= 5.0);
        constraint!(model, fix, x == 1.0);
        objective!(model, Min, x.powi(2) + weight * x);

        let opts = PounceOptions::default().solver_selection(PounceSolverSelection::QpIpm);
        let mut solver = PouncePersistent::new();
        assert!(solver.solve(&model, &opts).unwrap().has_solution());
        let Some(State::Convex(state)) = &solver.state else { panic!("convex state") };
        assert!(!state.ipm.as_ref().unwrap().last_reused_transform());

        assert!(solver.solve(&model, &opts).unwrap().has_solution());
        let Some(State::Convex(state)) = &solver.state else { panic!("convex state") };
        assert!(state.ipm.as_ref().unwrap().last_reused_transform());

        weight.set_param_value(2.0);
        assert!(solver.solve(&model, &opts).unwrap().has_solution());
        let Some(State::Convex(state)) = &solver.state else { panic!("convex state") };
        assert!(!state.ipm.as_ref().unwrap().last_reused_transform());
    }

    #[test]
    fn nlp_session_reuses_objective_only_and_rebuilds_constraint_changes() {
        let model = Model::new("nlp_presolve_session");
        param!(model, target = 2.0);
        param!(model, scale = 1.0);
        variable!(model, 0.0 <= x <= 5.0, initial = 1.0);
        variable!(model, 0.0 <= y <= 5.0, initial = 1.0);
        constraint!(model, balance, scale * x + y == 3.0);
        objective!(model, Min, (x - target).powi(2));

        let opts =
            PounceOptions::default().solver_selection(PounceSolverSelection::Nlp).presolve(true);
        let mut solver = PouncePersistent::new();
        assert!(solver.solve(&model, &opts).unwrap().has_solution());
        let Some(State::Nlp(state)) = &solver.state else { panic!("NLP state") };
        assert_eq!(state.resident.as_ref().unwrap().last_presolve_reused(), Some(false));

        target.set_param_value(1.5);
        assert!(solver.solve(&model, &opts).unwrap().has_solution());
        let Some(State::Nlp(state)) = &solver.state else { panic!("NLP state") };
        assert_eq!(state.resident.as_ref().unwrap().last_presolve_reused(), Some(true));

        scale.set_param_value(2.0);
        assert!(solver.solve(&model, &opts).unwrap().has_solution());
        let Some(State::Nlp(state)) = &solver.state else { panic!("NLP state") };
        assert_eq!(state.resident.as_ref().unwrap().last_presolve_reused(), Some(false));
    }

    #[test]
    fn nlp_auxiliary_parameter_and_option_changes_invalidate_the_session() {
        let model = Model::new("nlp_presolve_invalidation");
        param!(model, target = 2.0);
        variable!(model, 0.0 <= x <= 5.0, initial = 1.0);
        objective!(model, Min, (x - target).powi(2));

        let opts = PounceOptions::default()
            .solver_selection(PounceSolverSelection::Nlp)
            .presolve(true)
            .presolve_auxiliary(true);
        let mut solver = PouncePersistent::new();
        assert!(solver.solve(&model, &opts).unwrap().has_solution());
        assert!(solver.solve(&model, &opts).unwrap().has_solution());
        let Some(State::Nlp(state)) = &solver.state else { panic!("NLP state") };
        assert_eq!(state.resident.as_ref().unwrap().last_presolve_reused(), Some(true));

        target.set_param_value(1.5);
        assert!(solver.solve(&model, &opts).unwrap().has_solution());
        let Some(State::Nlp(state)) = &solver.state else { panic!("NLP state") };
        assert_eq!(state.resident.as_ref().unwrap().last_presolve_reused(), Some(false));

        let changed_opts = opts.clone().tol(1e-7);
        assert!(solver.solve(&model, &changed_opts).unwrap().has_solution());
        let Some(State::Nlp(state)) = &solver.state else { panic!("NLP state") };
        assert_eq!(state.resident.as_ref().unwrap().last_presolve_reused(), Some(false));
    }
}
