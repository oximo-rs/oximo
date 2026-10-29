//! Multiple local optima through POUNCE's shared search driver.

use std::cell::RefCell;
use std::fmt::Write;
use std::rc::Rc;
use std::time::{Duration, Instant};

use oximo_core::Model;
use oximo_solver::prepare::LoweringContext;
use oximo_solver::reconstruct::{ObjectiveTransform, finalize_result, project_owned_primal};
use oximo_solver::{DualStatus, SolutionPoint, SolverError, SolverResult, TerminationStatus};
use pounce_rs::minima::{
    HessianMode, IpoptLocalSolve, LocalOutcome, LocalSolve, Minimum, SolveQuality, Stop,
};
use pounce_rs::{IpoptApplication, TNLP};

use crate::options::{PounceOptions, PounceSolverSelection};
use crate::tnlp::{DerivativeOracle, OximoTnlp, SearchCurvature};
use crate::translate::{self, Outcome};

#[cfg(feature = "enzyme")]
use crate::exact as backend;
#[cfg(not(feature = "enzyme"))]
use crate::stable as backend;

pub use pounce_rs::minima::{Method as PounceMinimaMethod, MinimaOptions as PounceMinimaOptions};

/// Why a multiple-minima search stopped.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SearchStop {
    TargetReached,
    Converged,
    BudgetExhausted,
    SampleCapReached,
    Aborted,
    TimeLimit,
}

/// Search output before mapping into the common solver result.
#[derive(Clone, Debug)]
struct SearchResult {
    /// Best objective first, respecting the model's minimization/maximization sense.
    minima: Vec<Minimum>,
    stop: SearchStop,
    /// Search solves charged to `max_solves`, excluding dual-recovery solves.
    n_solves: usize,
    /// Local solves that ran and did not converge, including dual recovery.
    n_failed_solves: usize,
    /// Returned minima without complete, finite recovered multipliers.
    n_failed_dual_recoveries: usize,
    n_kkt_rejected: usize,
    /// Per-variable scale used for deduplication, in model variable order.
    scale: Vec<f64>,
    /// Search and local-solver quality warnings, with duplicates removed.
    warnings: Vec<String>,
    /// Total time, including setup and dual recovery.
    solve_time: Duration,
    /// Sum of reported local-solver iteration counts, including dual recovery.
    iterations: u64,
    /// All local-solve logs, when `PounceOptions::universal.verbose` is true.
    raw_log: Option<String>,
}

pub(crate) fn solve(
    model: &Model,
    opts: &PounceOptions,
    search: &PounceMinimaOptions,
) -> Result<SolverResult, SolverError> {
    let started = Instant::now();

    if model.has_active_sos_constraints() {
        return Err(SolverError::UnsupportedSos);
    }

    if model.has_active_indicator_constraints() {
        return Err(SolverError::UnsupportedIndicator);
    }

    translate::reject_semi_domains(model)?;
    validate_route(opts)?;

    let prepared = LoweringContext::new(model)?;
    let prep = translate::setup_prepared(&prepared, opts)?;
    let empty = SolverResult {
        parameters: prepared.arena().parameter_snapshot(),
        solver_name: Some("pounce".into()),
        ..SolverResult::for_model(model)
    };

    validate_options(search, prep.x0.len())?;

    let mut app = search_application(opts)?;

    if let Some(result) = expired_search(&prep, opts, started) {
        return Ok(result.into_solver_result(&prep, model, empty));
    }

    #[cfg(feature = "enzyme")]
    let oracle = backend::build_prepared(model, &prepared)?;
    #[cfg(not(feature = "enzyme"))]
    let oracle = backend::build_prepared(model, &prepared);

    let exact_hessian = oracle.borrow().has_exact_hessian();
    let local_exact_hessian = exact_hessian
        && app
            .options()
            .get_string_value("hessian_approximation", "")
            .expect("hessian_approximation is registered")
            .0
            == "exact";

    let curvature = Rc::new(SearchCurvature::default());
    let problem: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(
        OximoTnlp::new(oracle, &prep, None).with_search_curvature(Rc::clone(&curvature)),
    ));

    let verbose = opts.universal.verbose == Some(true);
    let _collector = verbose.then(pounce_rs::collector_scope);

    if verbose {
        app.enable_iter_history();
    }

    let local_time_limit = app
        .options()
        .get_numeric_value("max_wall_time", "")
        .expect("max_wall_time is registered")
        .0;
    let mut solver = SearchSolver {
        app,
        exact_hessian,
        local_exact_hessian,
        curvature,
        started,
        time_limit: opts.universal.time_limit,
        local_time_limit,
        timed_out: false,
        n_failed_solves: 0,
        iterations: 0,
        warnings: Vec::new(),
        raw_log: verbose.then(String::new),
    };

    // TODO: Use upstream search cancellation once POUNCE supports it (v0.14).
    if let Some(result) = expired_search(&prep, opts, started) {
        return Ok(result.into_solver_result(&prep, model, empty));
    }

    let result = pounce_rs::minima::find_minima(&problem, &mut solver, search)
        .map_err(|error| SolverError::Backend(format!("pounce minima: {error}")))?;

    let n_failed_dual_recoveries =
        result.minima.iter().filter(|minimum| !has_recovered_duals(minimum, &prep)).count();
    let warnings = solver.collect_warnings(result.warnings, n_failed_dual_recoveries);

    let stop = if solver.timed_out
        || opts.universal.time_limit.is_some_and(|limit| started.elapsed() >= limit)
    {
        SearchStop::TimeLimit
    } else {
        match result.stop {
            Stop::TargetReached => SearchStop::TargetReached,
            Stop::Converged => SearchStop::Converged,
            Stop::BudgetExhausted => SearchStop::BudgetExhausted,
            Stop::SampleCapReached => SearchStop::SampleCapReached,
            Stop::Aborted => SearchStop::Aborted,
        }
    };

    Ok(SearchResult {
        minima: result.minima,
        stop,
        n_solves: result.n_solves,
        n_failed_solves: solver.n_failed_solves,
        n_failed_dual_recoveries,
        n_kkt_rejected: result.n_kkt_rejected,
        scale: result.scale,
        warnings,
        solve_time: started.elapsed(),
        iterations: solver.iterations,
        raw_log: solver.raw_log,
    }
    .into_solver_result(&prep, model, empty))
}

/// Minima search always uses the general NLP route.
pub(crate) fn validate_route(opts: &PounceOptions) -> Result<(), SolverError> {
    if !matches!(
        translate::selected_solver(opts)?,
        PounceSolverSelection::Auto | PounceSolverSelection::Nlp
    ) {
        return Err(SolverError::Backend(
            "pounce multiple-minima search requires solver_selection auto or nlp".into(),
        ));
    }

    if let Some(hessian) = opts.effective_string("hessian_approximation") {
        validate_hessian_approximation(&hessian)?;
    }

    let defaults = pounce_rs::presolve::PresolveOptions::defaults();
    validate_search_presolve(
        opts.effective_bool("presolve").unwrap_or(defaults.enabled),
        opts.effective_bool("presolve_fbbt").unwrap_or(defaults.fbbt),
    )
}

fn validate_search_presolve(enabled: bool, fbbt: bool) -> Result<(), SolverError> {
    // Seeded and augmented search TNLPs do not carry an FBBT expression provider.
    if enabled && fbbt {
        return Err(SolverError::Backend(
            "pounce multiple-minima search does not support presolve_fbbt".into(),
        ));
    }

    Ok(())
}

impl SearchResult {
    fn into_solver_result(
        self,
        prep: &translate::Prepared,
        model: &Model,
        empty: SolverResult,
    ) -> SolverResult {
        let mut minima = self.minima.into_iter();

        // The common dual fields belong to solution zero.
        // Additional minima contribute only their primal points.
        let mut result = match minima.next() {
            Some(best) => assemble_minimum(best, prep, model, empty.parameters.clone()),
            None => empty,
        };

        result.solutions.extend(minima.map(|minimum| SolutionPoint {
            model_id: result.model_id,
            primal: project_owned_primal(minimum.x, prep.x0.len()).unwrap_or_default(),
            objective:
                ObjectiveTransform { sign: prep.sign, offset: 0.0 }.restore(minimum.objective),
            parameters: result.parameters.clone(),
        }));

        result.termination = match self.stop {
            SearchStop::TargetReached => TerminationStatus::SolutionLimit,
            SearchStop::Converged if result.has_solution() => TerminationStatus::LocallyOptimal,
            SearchStop::Converged => TerminationStatus::Other("MinimaConverged".into()),
            SearchStop::BudgetExhausted | SearchStop::SampleCapReached => {
                TerminationStatus::WorkLimit
            }
            SearchStop::Aborted => TerminationStatus::Interrupted,
            SearchStop::TimeLimit => TerminationStatus::TimeLimit,
        };
        result.raw_status = Some(format!("Minima{:?}", self.stop).into());
        result.solve_time = self.solve_time;
        result.iterations = self.iterations;

        let mut log = self.raw_log;

        if let Some(log) = &mut log {
            writeln!(log, "Minima search stop: {:?}", self.stop).unwrap();
            writeln!(log, "Search solves: {}", self.n_solves).unwrap();
            writeln!(log, "Failed local solves: {}", self.n_failed_solves).unwrap();
            writeln!(log, "Failed dual recoveries: {}", self.n_failed_dual_recoveries).unwrap();
            writeln!(log, "KKT rejected: {}", self.n_kkt_rejected).unwrap();
            writeln!(log, "Search scale: {:?}", self.scale).unwrap();
        }

        // Preserve search diagnostics even when iteration logging is disabled.
        for warning in self.warnings {
            writeln!(log.get_or_insert_with(String::new), "Warning: {warning}").unwrap();
        }

        result.raw_log = log;

        finalize_result(result, model, prep.x0.len())
    }
}

fn search_application(opts: &PounceOptions) -> Result<IpoptApplication, SolverError> {
    let mut app = IpoptApplication::new();
    app.initialize().map_err(|error| SolverError::Backend(format!("pounce init: {error:?}")))?;

    let mut local_opts = opts.clone();

    // Share the universal budget across local solves.
    local_opts.universal.time_limit = None;
    translate::apply_options(app.options_mut(), &local_opts, false)?;

    let presolve = pounce_rs::presolve::PresolveOptions::from_options_list(app.options())
        .map_err(|error| SolverError::Backend(format!("pounce presolve: {error}")))?;
    validate_search_presolve(presolve.enabled, presolve.fbbt)?;

    let hessian = app
        .options()
        .get_string_value("hessian_approximation", "")
        .expect("hessian_approximation is registered")
        .0;
    validate_hessian_approximation(&hessian)?;

    Ok(app)
}

fn validate_hessian_approximation(hessian: &str) -> Result<(), SolverError> {
    if !["exact", "limited-memory"].iter().any(|mode| hessian.eq_ignore_ascii_case(mode)) {
        return Err(SolverError::Backend(format!(
            "pounce multiple-minima search does not support hessian_approximation `{hessian}`; \
             use exact or limited-memory",
        )));
    }

    Ok(())
}

fn expired_search(
    prep: &translate::Prepared,
    opts: &PounceOptions,
    started: Instant,
) -> Option<SearchResult> {
    if opts.universal.time_limit.is_none_or(|limit| started.elapsed() < limit) {
        return None;
    }

    // Match POUNCE 0.13's search scales.
    let has_box = prep.x_l.iter().zip(&prep.x_u).all(|(&lo, &hi)| lo > -1e19 && hi < 1e19);
    let scale = prep
        .x_l
        .iter()
        .zip(&prep.x_u)
        .map(|(&lo, &hi)| if has_box && hi > lo { hi - lo } else { 1.0 })
        .collect();

    Some(SearchResult {
        minima: Vec::new(),
        stop: SearchStop::TimeLimit,
        n_solves: 0,
        n_failed_solves: 0,
        n_failed_dual_recoveries: 0,
        n_kkt_rejected: 0,
        scale,
        warnings: Vec::new(),
        solve_time: started.elapsed(),
        iterations: 0,
        raw_log: (opts.universal.verbose == Some(true))
            .then(|| "Search time limit reached before any local solve.\n".into()),
    })
}

fn has_recovered_duals(minimum: &Minimum, prep: &translate::Prepared) -> bool {
    minimum.lambda.len() == prep.g_l.len()
        && minimum.z_l.len() == prep.x0.len()
        && minimum.z_u.len() == prep.x0.len()
        && minimum.lambda.iter().chain(&minimum.z_l).chain(&minimum.z_u).all(|v| v.is_finite())
}

fn assemble_minimum(
    minimum: Minimum,
    prep: &translate::Prepared,
    model: &Model,
    parameters: oximo_solver::ParameterSnapshot,
) -> SolverResult {
    let recovered = has_recovered_duals(&minimum, prep);
    let reduced =
        recovered.then(|| minimum.z_l.iter().zip(&minimum.z_u).map(|(lo, hi)| lo - hi).collect());

    translate::assemble(
        prep.sign,
        Outcome {
            has_point: true,
            termination: TerminationStatus::LocallyOptimal,
            dual_status: if recovered { DualStatus::FeasiblePoint } else { DualStatus::Unknown },
            raw_status: "MinimumFound".into(),
            x: minimum.x,
            lambda: minimum.lambda,
            soc_dual: Vec::new(),
            reduced,
            objective: Some(minimum.objective),
            iterations: 0,
            warm: None,
            raw_log: None,
        },
        Duration::ZERO,
        model,
        parameters,
    )
}

struct SearchSolver {
    app: IpoptApplication,
    exact_hessian: bool,
    local_exact_hessian: bool,
    curvature: Rc<SearchCurvature>,
    started: Instant,
    time_limit: Option<Duration>,
    local_time_limit: f64,
    timed_out: bool,
    n_failed_solves: usize,
    iterations: u64,
    warnings: Vec<String>,
    raw_log: Option<String>,
}

impl SearchSolver {
    fn collect_warnings(
        &mut self,
        mut warnings: Vec<String>,
        n_failed_dual_recoveries: usize,
    ) -> Vec<String> {
        if self.curvature.skipped.get() {
            warnings.push(
                "saddle rejection skipped at constrained points or active bounds: \
                 the objective Hessian alone does not establish constrained optimality"
                    .into(),
            );
        }

        if self.n_failed_solves > 0 {
            warnings.push(format!(
                "local solves did not converge: {} (including dual recovery)",
                self.n_failed_solves
            ));
        }

        if n_failed_dual_recoveries > 0 {
            warnings.push(format!(
                "accepted minima without recovered duals: {n_failed_dual_recoveries}; \
                 constraint duals and reduced costs are unavailable"
            ));
        }

        for warning in self.warnings.drain(..) {
            if !warnings.contains(&warning) {
                warnings.push(warning);
            }
        }

        warnings
    }
}

impl LocalSolve for SearchSolver {
    fn solve(&mut self, problem: Rc<RefCell<dyn TNLP>>, mode: HessianMode) -> LocalOutcome {
        if let Some(limit) = self.time_limit {
            let remaining = limit.saturating_sub(self.started.elapsed());

            if remaining.is_zero() {
                self.timed_out = true;

                return LocalOutcome {
                    converged: false,
                    x: Vec::new(),
                    lambda: Vec::new(),
                    z_l: Vec::new(),
                    z_u: Vec::new(),
                    quality: SolveQuality::default(),
                };
            }

            self.app
                .options_mut()
                .set_numeric_value(
                    "max_wall_time",
                    remaining.as_secs_f64().min(self.local_time_limit),
                    true,
                    true,
                )
                .expect("remaining time is a valid wall time limit");
        }

        let mode = if self.local_exact_hessian { mode } else { HessianMode::QuasiNewton };
        self.curvature.local_solve.set(true);
        let outcome = IpoptLocalSolve::new(&mut self.app).solve(problem, mode);
        self.curvature.local_solve.set(false);

        if !outcome.converged {
            self.n_failed_solves = self.n_failed_solves.saturating_add(1);
        }

        let stats = self.app.statistics();
        self.iterations = self
            .iterations
            .saturating_add(u64::try_from(stats.iteration_count.max(0)).unwrap_or(u64::MAX));

        for warning in &stats.warnings {
            if !self.warnings.contains(warning) {
                self.warnings.push(warning.clone());
            }
        }

        if let Some(log) = &mut self.raw_log {
            use std::fmt::Write as _;

            log.push_str(&crate::tnlp::format_statistics(&stats));
            let _ = writeln!(log, "Local solve converged: {}\n", outcome.converged);
        }

        if self.time_limit.is_some_and(|limit| self.started.elapsed() >= limit) {
            self.timed_out = true;
        }

        outcome
    }

    fn has_exact_hessian(&self) -> bool {
        self.exact_hessian
    }
}

fn validate_options(options: &PounceMinimaOptions, n: usize) -> Result<(), SolverError> {
    let invalid = |name| SolverError::Backend(format!("invalid pounce minima option `{name}`"));

    for (name, value) in [("n_minima", options.n_minima), ("patience", options.patience)] {
        if value == 0 {
            return Err(invalid(name));
        }
    }

    // POUNCE v0.13 eagerly evaluates the default even with an explicit budget.
    if options.n_minima.checked_mul(8).is_none() {
        return Err(invalid("n_minima"));
    }

    if options.samples_per_round == Some(0) {
        return Err(invalid("samples_per_round"));
    }

    for (name, value) in [
        ("dedup", Some(options.dedup)),
        ("sigma", options.sigma),
        ("sigma_frac", options.sigma_frac),
        ("amplitude", options.amplitude),
        ("amp_margin", options.amp_margin),
        ("eta", options.eta),
        ("power", options.power),
        ("soft", options.soft),
        ("length", options.length),
        ("length_frac", options.length_frac),
        ("gamma", options.gamma),
        ("step", options.step),
        ("amp_bump", options.amp_bump),
        ("amp_bump_cap", options.amp_bump_cap),
        ("amp_fallback", options.amp_fallback),
    ] {
        if value.is_some_and(|value| !value.is_finite() || value <= 0.0) {
            return Err(invalid(name));
        }
    }

    for (name, value) in [
        ("psd_tol", Some(options.psd_tol)),
        ("kkt_tol", options.kkt_tol),
        ("temperature", options.temperature),
        ("restart_jitter", options.restart_jitter),
    ] {
        if value.is_some_and(|value| !value.is_finite() || value < 0.0) {
            return Err(invalid(name));
        }
    }

    for (name, values) in
        [("sigma_per_dim", &options.sigma_per_dim), ("length_per_dim", &options.length_per_dim)]
    {
        if values.as_ref().is_some_and(|values| {
            values.len() != n || values.iter().any(|value| !value.is_finite() || *value <= 0.0)
        }) {
            return Err(invalid(name));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximo_core::{constraint, objective, param, variable};
    use oximo_solver::UniversalOptionsExt;

    #[test]
    fn deadlines_expired_during_setup_preserve_search_scales() {
        for upper in [4.0, f64::INFINITY, 1e19] {
            let model = Model::new("expired search scales");

            variable!(model, -2.0 <= x <= 2.0);
            variable!(model, 0.0 <= y <= upper);
            variable!(model, 1.0 <= z <= 1.0);
            objective!(model, Min, x + y + z);

            let options = PounceOptions::default().time_limit(Duration::from_secs(1)).verbose(true);
            let prepared = LoweringContext::new(&model).unwrap();
            let prep = translate::setup_prepared(&prepared, &options).unwrap();
            let started = Instant::now().checked_sub(Duration::from_secs(2)).unwrap();
            let result = expired_search(&prep, &options, started).unwrap();
            let expected = if upper < 1e19 { vec![4.0, upper, 1.0] } else { vec![1.0; 3] };

            assert_eq!(result.scale, expected);
            assert_eq!(result.n_solves, 0);
            assert_eq!(result.n_failed_solves, 0);
            assert_eq!(result.n_failed_dual_recoveries, 0);
            assert_eq!(result.stop, SearchStop::TimeLimit);
            assert!(result.solve_time >= Duration::from_secs(2));
            assert!(result.raw_log.unwrap().contains("before any local solve"));
        }
    }

    #[test]
    fn failed_dual_recovery_preserves_minima_and_hides_all_multipliers() {
        let model = Model::new("unavailable recovered duals");

        param!(model, p = 2.0);
        variable!(model, 0.0 <= x <= 3.0);
        objective!(model, Min, p * x);
        let lower = constraint!(model, lower, x >= 1.0);

        let prepared = LoweringContext::new(&model).unwrap();
        let prep = translate::setup_prepared(&prepared, &PounceOptions::default()).unwrap();
        let parameters = prepared.arena().parameter_snapshot();
        model.set_param(p, 3.0).unwrap();

        for (lambda, z_l, z_u) in [
            (vec![0.0], vec![], vec![]),
            (vec![], vec![0.0], vec![0.0]),
            (vec![f64::NAN], vec![0.0], vec![0.0]),
            (vec![-2.0], vec![f64::INFINITY], vec![0.0]),
            (vec![-2.0], vec![0.0], vec![f64::NEG_INFINITY]),
        ] {
            let minimum = Minimum { x: vec![1.0], objective: 2.0, lambda, z_l, z_u };
            let result = assemble_minimum(minimum, &prep, &model, parameters.clone());

            assert_eq!(result.termination, TerminationStatus::LocallyOptimal);
            assert_eq!(result.value_of(x).unwrap(), Some(1.0));
            assert_eq!(result.objective(), Some(2.0));
            assert_eq!(result.value_of(p * x).unwrap(), Some(2.0));
            assert_eq!(result.dual_status, DualStatus::Unknown);
            assert_eq!(result.dual_of(lower).unwrap(), None);
            assert!(result.dual.is_empty());
            assert!(result.semantic_dual.is_empty());
            assert!(result.reduced_costs.is_empty());
        }
    }

    #[test]
    fn timed_out_search_preserves_ranked_points_snapshots_and_best_duals() {
        let model = Model::new("partial search result");

        param!(model, p = 2.0);
        variable!(model, -2.0 <= x <= 2.0);
        variable!(model, 0.0 <= y <= 4.0);
        objective!(model, Max, -(x.powi(2) - 1.0).powi(2) - p * y.powi(2));
        let lower = constraint!(model, lower, y >= 2.0);

        let prepared = LoweringContext::new(&model).unwrap();
        let prep = translate::setup_prepared(&prepared, &PounceOptions::default()).unwrap();
        let empty = SolverResult::for_model(&model);
        let search = SearchResult {
            minima: vec![
                Minimum {
                    x: vec![-1.0, 2.0],
                    objective: 8.0,
                    lambda: vec![-8.0],
                    z_l: vec![0.0; 2],
                    z_u: vec![0.0; 2],
                },
                Minimum {
                    x: vec![1.0, 2.0],
                    objective: 8.0,
                    lambda: vec![],
                    z_l: vec![],
                    z_u: vec![],
                },
            ],
            stop: SearchStop::TimeLimit,
            n_solves: 2,
            n_failed_solves: 0,
            n_failed_dual_recoveries: 1,
            n_kkt_rejected: 0,
            scale: vec![4.0; 2],
            warnings: Vec::new(),
            solve_time: Duration::from_secs(1),
            iterations: 12,
            raw_log: None,
        };

        model.set_param(p, 3.0).unwrap();
        let result = search.into_solver_result(&prep, &model, empty);

        assert_eq!(result.termination, TerminationStatus::TimeLimit);
        assert_eq!(result.raw_status.as_deref(), Some("MinimaTimeLimit"));
        assert_eq!(result.result_count(), 2);
        assert_eq!(result.dual_status, DualStatus::FeasiblePoint);
        assert_eq!(result.dual_of(lower).unwrap(), Some(-8.0));
        assert_eq!(result.value_of(x).unwrap(), Some(-1.0));
        assert_eq!(result.solution(1).unwrap().value_of(x).unwrap(), Some(1.0));

        for point in &result.solutions {
            assert_eq!(point.model_id(), model.id());
            assert_eq!(point.objective, Some(-8.0));
            assert_eq!(point.value_of(p * y).unwrap(), Some(4.0));
        }

        assert_eq!(result.solve_time, Duration::from_secs(1));
        assert_eq!(result.iterations, 12);
        assert!(result.best_bound.is_none());
        assert!(result.gap.is_none());
    }

    #[test]
    fn failure_summaries_distinguish_local_solves_from_unavailable_duals() {
        let mut solver = SearchSolver {
            app: IpoptApplication::new(),
            exact_hessian: true,
            local_exact_hessian: true,
            curvature: Rc::new(SearchCurvature::default()),
            started: Instant::now(),
            time_limit: None,
            local_time_limit: 1e20,
            timed_out: false,
            n_failed_solves: 2,
            iterations: 0,
            warnings: vec!["quality warning".into()],
            raw_log: None,
        };
        let warnings = solver.collect_warnings(vec!["quality warning".into()], 1);

        assert_eq!(warnings.iter().filter(|warning| *warning == "quality warning").count(), 1);
        assert!(
            warnings.iter().any(|warning| warning.contains("local solves did not converge: 2"))
        );
        assert!(
            warnings.iter().any(|warning| warning.contains("minima without recovered duals: 1"))
        );
    }
}
