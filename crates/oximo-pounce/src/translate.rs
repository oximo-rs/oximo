//! Model to POUNCE adapter.
//! Shared setup, status/result mapping, and the option helpers used by
//! both the exact-derivative path ([`crate::exact`], enzyme) and the
//! stable hybrid path ([`crate::stable`]).

use std::time::{Duration, Instant};

use oximo_core::{ConstraintId, Model, ModelKind, ObjectiveSense, SocConstraintId, VarId};
use oximo_solver::{
    DualStatus, PrimalStatus, SolutionPoint, SolverError, SolverResult, TerminationStatus,
};
use pounce_rs::ApplicationReturnStatus;
use pounce_rs::pounce_common::options_list::OptionsList;
use rustc_hash::FxHashMap;

use crate::options::{PounceAlgorithm, PounceOptionValue, PounceOptions, PounceSolverSelection};

#[cfg(feature = "enzyme")]
use crate::exact as backend;
#[cfg(not(feature = "enzyme"))]
use crate::stable as backend;

/// POUNCE treats bounds at or beyond +-2e19 as infinite.
pub(crate) const POUNCE_INFINITY: f64 = 2.0e19;

/// Bounds and objective sense snapshotted from the model, shared by both
/// derivative paths.
pub(crate) struct Prepared {
    pub sign: f64,
    pub x_l: Vec<f64>,
    pub x_u: Vec<f64>,
    pub g_l: Vec<f64>,
    pub g_u: Vec<f64>,
    pub x0: Vec<f64>,
}

/// A full primal-dual point, kept by a persistent handle to warm-start the
/// next solve. Both derivative paths reuse the whole point.
#[derive(Clone, Debug)]
pub(crate) struct WarmStart {
    pub x: Vec<f64>,
    pub z_l: Vec<f64>,
    pub z_u: Vec<f64>,
    pub lambda: Vec<f64>,
    pub sqp_working: Option<pounce_rs::sqp::WorkingSet>,
}

/// Backend-agnostic solve outcome, mapped into a [`SolverResult`] by
/// [`assemble`].
///
/// Objective and multipliers are in POUNCE's minimization sense
/// (a `Maximize` model is posed as `min -f`) and [`assemble`] undoes the sign.
pub(crate) struct Outcome {
    pub termination: TerminationStatus,
    pub dual_status: DualStatus,
    /// Native POUNCE/convex-route status label.
    pub raw_status: String,
    pub x: Vec<f64>,
    pub lambda: Vec<f64>,
    /// Explicit second-order-cone bound multipliers in model order.
    pub soc_dual: Vec<f64>,
    pub reduced: Option<Vec<f64>>,
    pub objective: Option<f64>,
    pub iterations: u64,
    pub warm: Option<WarmStart>,
    /// Reconstructed solver log (minimization sense), captured when `verbose`.
    pub raw_log: Option<String>,
}

pub(crate) fn reject_semi_domains(model: &Model) -> Result<(), SolverError> {
    let vars = model.variables();
    if let Some(v) = vars.iter().find(|v| v.domain.semi_threshold().is_some()) {
        return Err(SolverError::Backend(format!(
            "variable {} has a semicontinuous/semi-integer domain, \
                which POUNCE does not support",
            v.name
        )));
    }
    Ok(())
}

/// Translate `model`, solve with POUNCE (cold), and map the outcome.
///
/// # Errors
///
/// [`SolverError::UnsupportedKind`] for integer/cone model kinds,
/// [`SolverError::Backend`] for a model with semi-continuous/semi-integer variables and
/// [`SolverError::Core`] for a model with neither an objective nor a declared
/// feasibility problem.
pub fn solve(model: &Model, opts: &PounceOptions) -> Result<SolverResult, SolverError> {
    if model.has_active_sos_constraints() {
        return Err(SolverError::UnsupportedSos);
    }
    reject_semi_domains(model)?;
    let route = crate::convex::route(model, opts)?;
    if route != crate::convex::Route::Nlp {
        return crate::convex::solve(model, opts, route);
    }
    solve_nlp(model, opts)
}

pub(crate) fn solve_nlp(model: &Model, opts: &PounceOptions) -> Result<SolverResult, SolverError> {
    solve_nlp_since(model, opts, Instant::now())
}

/// Solve through the NLP route while charging time already spent by an
/// automatic convex attempt to the reported solve duration.
pub(crate) fn solve_nlp_since(
    model: &Model,
    opts: &PounceOptions,
    started: Instant,
) -> Result<SolverResult, SolverError> {
    let prep = setup(model, opts)?;
    let oracle = backend::build(model)?;
    let outcome = run_nlp_with_retries(model, &oracle, &prep, opts, None)?;
    Ok(assemble(prep.sign, outcome, started.elapsed()))
}

/// Mirror POUNCE's two-rung second opinion for a local-infeasibility verdict.
/// A retry is promoted only when its own convergence check succeeds.
pub(crate) fn run_nlp_with_retries(
    model: &Model,
    oracle: &backend::Oracle,
    prep: &Prepared,
    opts: &PounceOptions,
    warm: Option<&WarmStart>,
) -> Result<Outcome, SolverError> {
    let started = Instant::now();
    let mut original = backend::run(model, oracle, prep, opts, warm)?;
    if original.termination != TerminationStatus::Infeasible {
        return Ok(original);
    }

    // These retries change interior-point controls and cannot provide a
    // meaningful second opinion for the active-set SQP algorithm.
    if selected_algorithm(opts)? != PounceAlgorithm::InteriorPoint {
        return Ok(original);
    }

    let scaling = effective_string(opts, "feral_scaling");
    if effective_bool(opts, "feral_infeasibility_scaling_retry").unwrap_or(true)
        && scaling.as_deref() != Some("mc64")
    {
        if let Some(retry_opts) = retry_options(opts, started, "feral_scaling", "mc64") {
            match backend::run(model, oracle, prep, &retry_opts, None) {
                Ok(mut retry) if retry.termination == TerminationStatus::LocallyOptimal => {
                    merge_retry_log(&original, &mut retry, "MC64 scaling", true);
                    return Ok(retry);
                }
                Ok(retry) => merge_retry_log(&retry, &mut original, "MC64 scaling", false),
                Err(error) => record_retry_error(&mut original, "MC64 scaling", &error),
            }
        } else {
            record_retry_skip(&mut original, "MC64 scaling");
        }
    }

    let adaptive = effective_mu_is_adaptive(opts);
    if effective_bool(opts, "infeasibility_mu_strategy_retry").unwrap_or(true) && !adaptive {
        if let Some(retry_opts) = retry_options(opts, started, "mu_strategy", "adaptive") {
            match backend::run(model, oracle, prep, &retry_opts, None) {
                Ok(mut retry) if retry.termination == TerminationStatus::LocallyOptimal => {
                    merge_retry_log(&original, &mut retry, "adaptive mu", true);
                    return Ok(retry);
                }
                Ok(retry) => merge_retry_log(&retry, &mut original, "adaptive mu", false),
                Err(error) => record_retry_error(&mut original, "adaptive mu", &error),
            }
        } else {
            record_retry_skip(&mut original, "adaptive mu");
        }
    }
    Ok(original)
}

fn effective_mu_is_adaptive(opts: &PounceOptions) -> bool {
    let mut adaptive = opts.mu_strategy == Some(crate::options::MuStrategy::Adaptive);
    for (name, value) in &opts.extra {
        if name == "mu_strategy"
            && let PounceOptionValue::Str(value) = value
        {
            adaptive = value == "adaptive";
        }
    }
    adaptive
}

fn retry_options(
    opts: &PounceOptions,
    started: Instant,
    name: &str,
    value: &str,
) -> Option<PounceOptions> {
    let mut retry = opts.clone().set(name, value);
    if let Some(limit) = opts.universal.time_limit {
        let remaining = limit.saturating_sub(started.elapsed());
        if remaining.is_zero() {
            return None;
        }
        retry.universal.time_limit = Some(remaining);
    }
    Some(retry)
}

fn merge_retry_log(source: &Outcome, target: &mut Outcome, label: &str, promoted: bool) {
    let Some(target_log) = &mut target.raw_log else {
        return;
    };
    let disposition = if promoted { "promoted" } else { "did not converge, original retained" };
    let mut merged = if promoted {
        source.raw_log.clone().unwrap_or_default()
    } else {
        std::mem::take(target_log)
    };
    use std::fmt::Write as _;
    let _ = writeln!(merged, "POUNCE local-infeasibility second opinion ({label}): {disposition}");
    if promoted {
        merged.push_str(target_log);
    } else if let Some(source_log) = &source.raw_log {
        merged.push_str(source_log);
    }
    *target_log = merged;
}

fn record_retry_error(original: &mut Outcome, label: &str, error: &SolverError) {
    let Some(log) = &mut original.raw_log else {
        return;
    };
    use std::fmt::Write as _;
    let _ = writeln!(
        log,
        "POUNCE local-infeasibility second opinion ({label}) failed: {error}, original retained"
    );
}

fn record_retry_skip(original: &mut Outcome, label: &str) {
    let Some(log) = &mut original.raw_log else {
        return;
    };
    use std::fmt::Write as _;
    let _ = writeln!(
        log,
        "POUNCE local-infeasibility second opinion ({label}) skipped: time budget exhausted"
    );
}

fn effective_bool(opts: &PounceOptions, name: &str) -> Option<bool> {
    opts.effective_bool(name)
}

fn effective_string(opts: &PounceOptions, name: &str) -> Option<String> {
    opts.effective_string(name)
}

/// Kind gate, objective declaration check, sign, and bound snapshot.
pub(crate) fn setup(model: &Model, opts: &PounceOptions) -> Result<Prepared, SolverError> {
    let kind = model.kind();
    if !matches!(
        kind,
        ModelKind::LP | ModelKind::QP | ModelKind::QCP | ModelKind::SOCP | ModelKind::NLP
    ) {
        return Err(SolverError::UnsupportedKind(kind));
    }
    validate_algorithm(kind, opts)?;
    model.ensure_objective_declared().map_err(SolverError::Core)?;
    let sign = match model.objective().as_ref().map(|o| o.sense) {
        Some(ObjectiveSense::Maximize) => -1.0,
        _ => 1.0,
    };

    let vars = model.variables();
    let mut x_l = Vec::with_capacity(vars.len());
    let mut x_u = Vec::with_capacity(vars.len());
    let mut x0 = Vec::with_capacity(vars.len());
    for v in vars.iter() {
        x_l.push(v.lb.max(-POUNCE_INFINITY));
        x_u.push(v.ub.min(POUNCE_INFINITY));
        x0.push(v.initial.unwrap_or_else(|| initial_guess(v.lb, v.ub)));
    }
    drop(vars);

    let model_constraints = model.constraints();
    if !model_constraints.second_order_cones().is_empty() {
        return Err(SolverError::UnsupportedKind(ModelKind::SOCP));
    }
    let constraints = model_constraints.algebraic();
    let mut g_l = Vec::with_capacity(constraints.len());
    let mut g_u = Vec::with_capacity(constraints.len());
    for c in constraints {
        g_l.push(c.lower.max(-POUNCE_INFINITY));
        g_u.push(c.upper.min(POUNCE_INFINITY));
    }

    Ok(Prepared { sign, x_l, x_u, g_l, g_u, x0 })
}

/// Resolve the NLP algorithm after raw options (which apply last).
pub(crate) fn selected_algorithm(opts: &PounceOptions) -> Result<PounceAlgorithm, SolverError> {
    let mut algorithm = opts.algorithm.unwrap_or_default();
    for (name, value) in &opts.extra {
        match (name.as_str(), value) {
            ("algorithm", PounceOptionValue::Str(value)) => {
                algorithm = match value.as_str() {
                    "interior-point" => PounceAlgorithm::InteriorPoint,
                    "active-set-sqp" => PounceAlgorithm::ActiveSetSqp,
                    other => {
                        return Err(SolverError::Backend(format!(
                            "pounce algorithm `{other}` is unsupported; use `interior-point` or `active-set-sqp`"
                        )));
                    }
                };
            }
            ("algorithm", _) => {
                return Err(SolverError::Backend(
                    "pounce option `algorithm` must be a string".into(),
                ));
            }
            _ => {}
        }
    }
    Ok(algorithm)
}

/// Resolve structural routing after raw options (which apply last).
pub(crate) fn selected_solver(opts: &PounceOptions) -> Result<PounceSolverSelection, SolverError> {
    let mut selection = opts.solver_selection.unwrap_or_default();
    for (name, value) in &opts.extra {
        if name == "solver_selection" {
            let PounceOptionValue::Str(value) = value else {
                return Err(SolverError::Backend(
                    "pounce option `solver_selection` must be a string".into(),
                ));
            };
            selection = PounceSolverSelection::parse(value).ok_or_else(|| {
                SolverError::Backend(format!(
                    "pounce solver_selection `{value}` is unsupported. Use auto, nlp, lp-ipm, qp-ipm, qp-active-set, or socp"
                ))
            })?;
        }
    }
    Ok(selection)
}

pub(crate) fn validate_algorithm(
    kind: ModelKind,
    opts: &PounceOptions,
) -> Result<PounceAlgorithm, SolverError> {
    let algorithm = selected_algorithm(opts)?;
    if algorithm.supports(kind) { Ok(algorithm) } else { Err(SolverError::UnsupportedKind(kind)) }
}

/// Midpoint of finite bounds, otherwise zero clipped into the bounds.
fn initial_guess(lb: f64, ub: f64) -> f64 {
    if lb.is_finite() && ub.is_finite() { f64::midpoint(lb, ub) } else { 0.0_f64.clamp(lb, ub) }
}

/// Map a POUNCE application status onto oximo's termination taxonomy.
pub(crate) fn map_status(s: ApplicationReturnStatus) -> TerminationStatus {
    use ApplicationReturnStatus as A;
    match s {
        A::SolveSucceeded | A::SolvedToAcceptableLevel => TerminationStatus::LocallyOptimal,
        A::FeasiblePointFound => TerminationStatus::Feasible,
        A::InfeasibleProblemDetected => TerminationStatus::Infeasible,
        A::MaximumIterationsExceeded => TerminationStatus::IterationLimit,
        A::MaximumCpuTimeExceeded | A::MaximumWallTimeExceeded => TerminationStatus::TimeLimit,
        A::UserRequestedStop => TerminationStatus::Interrupted,
        A::SearchDirectionBecomesTooSmall
        | A::RestorationFailed
        | A::ErrorInStepComputation
        | A::InvalidNumberDetected => TerminationStatus::NumericError,
        other => TerminationStatus::Other(format!("{other:?}")),
    }
}

/// Assemble a [`SolverResult`], undoing the maximize sign flip: the reported
/// objective is `sign * pounce_obj`, the LP-convention dual is `−sign * lambda`,
/// and the reduced cost is `sign * (z_l − z_u)`.
pub(crate) fn assemble(sign: f64, o: Outcome, elapsed: Duration) -> SolverResult {
    let has_point = o.termination.admits_primal() && !o.x.is_empty();

    let mut solutions = Vec::new();
    let mut dual: FxHashMap<ConstraintId, f64> = FxHashMap::default();
    let mut reduced_costs: FxHashMap<VarId, f64> = FxHashMap::default();
    let mut soc_dual: FxHashMap<SocConstraintId, f64> = FxHashMap::default();

    if has_point {
        let mut primal: FxHashMap<VarId, f64> = FxHashMap::default();
        for (i, &v) in o.x.iter().enumerate() {
            primal.insert(VarId(u32::try_from(i).expect("variable count overflow")), v);
        }
        for (i, &l) in o.lambda.iter().enumerate() {
            dual.insert(
                ConstraintId(u32::try_from(i).expect("constraint count overflow")),
                -sign * l,
            );
        }
        if let Some(red) = &o.reduced {
            for (i, &r) in red.iter().enumerate() {
                reduced_costs
                    .insert(VarId(u32::try_from(i).expect("variable count overflow")), sign * r);
            }
        }
        for (i, &value) in o.soc_dual.iter().enumerate() {
            soc_dual.insert(
                SocConstraintId(u32::try_from(i).expect("SOC constraint count overflow")),
                value,
            );
        }
        solutions.push(SolutionPoint { primal, objective: o.objective.map(|f| sign * f) });
    }

    let primal_status = PrimalStatus::infer(&o.termination, has_point);
    SolverResult {
        termination: o.termination,
        primal_status,
        dual_status: o.dual_status,
        solutions,
        dual,
        soc_dual,
        reduced_costs,
        best_bound: None,
        gap: None,
        solve_time: elapsed,
        iterations: o.iterations,
        node_count: None,
        raw_status: Some(o.raw_status.into()),
        raw_log: o.raw_log,
        solver_name: Some("pounce".into()),
        solver_version: None,
    }
}

/// The effective `print_level`:
/// Explicit `print_level`, else 5 when `verbose`, else 0 (quiet).
pub(crate) fn print_level(opts: &PounceOptions) -> i32 {
    opts.print_level.map_or(if opts.universal.verbose == Some(true) { 5 } else { 0 }, |v| {
        i32::try_from(v).unwrap_or(i32::MAX)
    })
}

pub(crate) fn mu_strategy_str(s: crate::options::MuStrategy) -> &'static str {
    match s {
        crate::options::MuStrategy::Monotone => "monotone",
        crate::options::MuStrategy::Adaptive => "adaptive",
    }
}

/// An option POUNCE rejected (unknown name or out-of-range value) as a
/// [`SolverError`].
fn opt_error(name: &str, e: &impl std::fmt::Display) -> SolverError {
    SolverError::Backend(format!("pounce rejected option `{name}`: {e}"))
}

fn set_num(list: &mut OptionsList, name: &str, v: f64) -> Result<(), SolverError> {
    list.set_numeric_value(name, v, true, true).map(|_| ()).map_err(|e| opt_error(name, &e))
}

fn set_int(list: &mut OptionsList, name: &str, v: i32) -> Result<(), SolverError> {
    list.set_integer_value(name, v, true, true).map(|_| ()).map_err(|e| opt_error(name, &e))
}

pub(crate) fn set_str(list: &mut OptionsList, name: &str, v: &str) -> Result<(), SolverError> {
    list.set_string_value(name, v, true, true).map(|_| ()).map_err(|e| opt_error(name, &e))
}

fn set_bool(list: &mut OptionsList, name: &str, v: bool) -> Result<(), SolverError> {
    list.set_bool_value(name, v, true, true).map(|_| ()).map_err(|e| opt_error(name, &e))
}

pub(crate) fn convex_only_option(name: &str) -> bool {
    matches!(
        name,
        "qp_presolve"
            | "qp_tau"
            | "qp_tau_max"
            | "qp_reg"
            | "qp_infeas_tol"
            | "qp_hsde"
            | "qp_equilibrate"
            | "qp_crossover"
            | "qp_gondzio_corr"
    )
}

/// Apply [`PounceOptions`] onto POUNCE's option list (from `IpoptApplication::options_mut()`),
/// surfacing an invalid name or out-of-range value as a [`SolverError::Backend`].
/// Both derivative paths apply onto the live application via [`crate::tnlp::run`].
pub(crate) fn apply_options(
    list: &mut OptionsList,
    opts: &PounceOptions,
    warm: bool,
) -> Result<(), SolverError> {
    set_int(list, "print_level", print_level(opts))?;
    if warm {
        set_str(list, "warm_start_init_point", "yes")?;
    }
    if let Some(tol) = opts.tol {
        set_num(list, "tol", tol)?;
    }
    if let Some(n) = opts.max_iter {
        set_int(list, "max_iter", i32::try_from(n).unwrap_or(i32::MAX))?;
    }
    if let Some(limit) = opts.universal.time_limit {
        set_num(list, "max_wall_time", limit.as_secs_f64())?;
    }
    if let Some(s) = opts.mu_strategy {
        set_str(list, "mu_strategy", mu_strategy_str(s))?;
    }
    if let Some(algorithm) = opts.algorithm {
        set_str(list, "algorithm", algorithm.as_str())?;
    }
    if let Some(selection) = opts.solver_selection {
        set_str(list, "solver_selection", selection.as_str())?;
    }
    for &(name, v) in opts.num_opts() {
        if !convex_only_option(name) {
            set_num(list, name, v)?;
        }
    }
    for &(name, v) in opts.int_opts() {
        if !convex_only_option(name) {
            set_int(list, name, v)?;
        }
    }
    for (name, v) in opts.str_opts() {
        set_str(list, name, v)?;
    }
    for &(name, v) in opts.bool_opts() {
        if !convex_only_option(name) {
            set_bool(list, name, v)?;
        }
    }
    for (name, value) in &opts.extra {
        if convex_only_option(name) {
            continue;
        }
        match value {
            PounceOptionValue::Num(v) => set_num(list, name, *v)?,
            PounceOptionValue::Int(v) => set_int(list, name, *v)?,
            PounceOptionValue::Str(v) => set_str(list, name, v)?,
            PounceOptionValue::Bool(v) => set_bool(list, name, *v)?,
        }
    }
    Ok(())
}

#[cfg(test)]
mod retry_tests {
    use super::*;
    use crate::options::MuStrategy;

    #[test]
    fn raw_mu_strategy_has_final_precedence() {
        let monotone = PounceOptions::default()
            .mu_strategy(MuStrategy::Adaptive)
            .set("mu_strategy", "monotone");
        assert!(!effective_mu_is_adaptive(&monotone));

        let adaptive = PounceOptions::default()
            .mu_strategy(MuStrategy::Monotone)
            .set("mu_strategy", "adaptive");
        assert!(effective_mu_is_adaptive(&adaptive));
    }

    #[test]
    fn retries_do_not_start_after_the_shared_budget_expires() {
        let mut opts = PounceOptions::default();
        opts.universal.time_limit = Some(Duration::ZERO);
        assert!(retry_options(&opts, Instant::now(), "mu_strategy", "adaptive").is_none());
    }

    #[test]
    fn failed_retry_log_retains_both_attempts_in_order() {
        let mut original = logged_outcome("original\n");
        let retry = logged_outcome("retry\n");
        merge_retry_log(&retry, &mut original, "MC64 scaling", false);
        assert_eq!(
            original.raw_log.as_deref(),
            Some(
                "original\nPOUNCE local-infeasibility second opinion (MC64 scaling): did not converge, original retained\nretry\n"
            )
        );
    }

    #[test]
    fn promoted_retry_log_retains_both_attempts_in_order() {
        let original = logged_outcome("original\n");
        let mut retry = logged_outcome("retry\n");
        merge_retry_log(&original, &mut retry, "adaptive mu", true);
        assert_eq!(
            retry.raw_log.as_deref(),
            Some(
                "original\nPOUNCE local-infeasibility second opinion (adaptive mu): promoted\nretry\n"
            )
        );
    }

    #[test]
    fn retry_diagnostics_explain_errors_and_exhausted_budgets() {
        let mut error = logged_outcome("original\n");
        record_retry_error(&mut error, "MC64 scaling", &SolverError::Backend("failed".into()));
        assert!(error.raw_log.as_deref().is_some_and(|log| {
            log.contains("(MC64 scaling) failed: backend error: failed, original retained")
        }));

        let mut skipped = logged_outcome("original\n");
        record_retry_skip(&mut skipped, "adaptive mu");
        assert!(
            skipped
                .raw_log
                .as_deref()
                .is_some_and(|log| log.contains("(adaptive mu) skipped: time budget exhausted"))
        );
    }

    #[test]
    fn raw_route_and_algorithm_options_take_precedence_and_validate_types() {
        let options = PounceOptions::default()
            .algorithm(PounceAlgorithm::ActiveSetSqp)
            .solver_selection(PounceSolverSelection::Nlp)
            .set("algorithm", "interior-point")
            .set("solver_selection", "qp-active-set");
        assert_eq!(selected_algorithm(&options).unwrap(), PounceAlgorithm::InteriorPoint);
        assert_eq!(selected_solver(&options).unwrap(), PounceSolverSelection::QpActiveSet);

        for options in [
            PounceOptions::default().set("algorithm", "invalid"),
            PounceOptions::default().set("algorithm", true),
            PounceOptions::default().set("solver_selection", "invalid"),
            PounceOptions::default().set("solver_selection", true),
        ] {
            assert!(matches!(
                selected_algorithm(&options).and_then(|_| selected_solver(&options)),
                Err(SolverError::Backend(_))
            ));
        }
    }

    #[test]
    fn retry_controls_use_the_last_well_typed_value() {
        let options = PounceOptions::default()
            .feral_infeasibility_scaling_retry(false)
            .set("feral_infeasibility_scaling_retry", "wrong-type")
            .set("feral_infeasibility_scaling_retry", true)
            .feral_scaling("none")
            .feral_scaling("matching")
            .set("feral_scaling", false)
            .set("feral_scaling", "mc64");
        assert_eq!(effective_bool(&options, "feral_infeasibility_scaling_retry"), Some(true));
        assert_eq!(effective_string(&options, "feral_scaling").as_deref(), Some("mc64"));
        assert_eq!(effective_bool(&options, "missing"), None);
        assert_eq!(effective_string(&options, "missing"), None);
    }

    #[test]
    fn initial_guesses_respect_finite_and_one_sided_bounds() {
        for (actual, expected) in [
            (initial_guess(-2.0, 4.0), 1.0),
            (initial_guess(3.0, f64::INFINITY), 3.0),
            (initial_guess(f64::NEG_INFINITY, -3.0), -3.0),
            (initial_guess(f64::NEG_INFINITY, f64::INFINITY), 0.0),
        ] {
            assert!((actual - expected).abs() <= f64::EPSILON);
        }
    }

    #[test]
    fn only_standalone_convex_options_bypass_the_pounce_registry() {
        for name in [
            "qp_presolve",
            "qp_tau",
            "qp_tau_max",
            "qp_reg",
            "qp_infeas_tol",
            "qp_hsde",
            "qp_equilibrate",
            "qp_crossover",
            "qp_gondzio_corr",
        ] {
            assert!(convex_only_option(name), "{name}");
        }
        for name in ["solver_selection", "feral_infeasibility_scaling_retry", "sqp_qp_max_iter"] {
            assert!(!convex_only_option(name), "{name}");
        }
    }

    #[test]
    fn apply_options_skips_convex_only_integer_options_on_the_nlp_path() {
        let mut app = pounce_rs::IpoptApplication::new();
        app.initialize().unwrap();
        let opts = PounceOptions::default().qp_gondzio_corr(3).sqp_qp_max_iter(12);
        apply_options(app.options_mut(), &opts, false).unwrap();
    }

    #[test]
    fn assemble_preserves_explicit_dual_status_for_partial_and_limit_points() {
        for (termination, raw_status) in [
            (TerminationStatus::Feasible, "OptimalInaccurate"),
            (TerminationStatus::IterationLimit, "MaximumIterationsExceeded"),
            (TerminationStatus::TimeLimit, "MaximumWallTimeExceeded"),
        ] {
            let result = assemble(
                1.0,
                Outcome {
                    termination,
                    dual_status: DualStatus::Unknown,
                    raw_status: raw_status.into(),
                    x: vec![1.0],
                    lambda: vec![2.0],
                    soc_dual: Vec::new(),
                    reduced: None,
                    objective: Some(1.0),
                    iterations: 1,
                    warm: None,
                    raw_log: None,
                },
                Duration::ZERO,
            );
            assert_eq!(result.dual_status, DualStatus::Unknown, "{raw_status}");
        }

        let result = assemble(
            1.0,
            Outcome {
                termination: TerminationStatus::LocallyOptimal,
                dual_status: DualStatus::FeasiblePoint,
                raw_status: "SolveSucceeded".into(),
                x: vec![1.0],
                lambda: vec![2.0],
                soc_dual: Vec::new(),
                reduced: None,
                objective: Some(1.0),
                iterations: 1,
                warm: None,
                raw_log: None,
            },
            Duration::ZERO,
        );
        assert_eq!(result.dual_status, DualStatus::FeasiblePoint);
    }

    fn logged_outcome(log: &str) -> Outcome {
        Outcome {
            termination: TerminationStatus::Infeasible,
            dual_status: DualStatus::Unknown,
            raw_status: "InfeasibleProblemDetected".into(),
            x: Vec::new(),
            lambda: Vec::new(),
            soc_dual: Vec::new(),
            reduced: None,
            objective: None,
            iterations: 0,
            warm: None,
            raw_log: Some(log.to_owned()),
        }
    }
}
