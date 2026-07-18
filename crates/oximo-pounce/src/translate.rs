//! Model to POUNCE adapter.
//! Shared setup, status/result mapping, and the option helpers used by
//! both the exact-derivative path ([`crate::exact`], enzyme) and the
//! stable hybrid path ([`crate::stable`]).

use std::time::{Duration, Instant};

use oximo_core::{ConstraintId, Model, ModelKind, ObjectiveSense, VarId};
use oximo_solver::{PrimalStatus, SolutionPoint, SolverError, SolverResult, TerminationStatus};
use pounce_rs::ApplicationReturnStatus;
use pounce_rs::pounce_common::options_list::OptionsList;
use rustc_hash::FxHashMap;

use crate::options::{PounceOptionValue, PounceOptions};

#[cfg(feature = "enzyme")]
use crate::exact as backend;
#[cfg(not(feature = "enzyme"))]
use crate::stable as backend;

/// POUNCE treats bounds at or beyond +-2e19 as infinite.
const POUNCE_INFINITY: f64 = 2.0e19;

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
}

/// Backend-agnostic solve outcome, mapped into a [`SolverResult`] by
/// [`assemble`].
///
/// Objective and multipliers are in POUNCE's minimization sense
/// (a `Maximize` model is posed as `min -f`) and [`assemble`] undoes the sign.
pub(crate) struct Outcome {
    pub termination: TerminationStatus,
    pub x: Vec<f64>,
    pub lambda: Vec<f64>,
    pub reduced: Option<Vec<f64>>,
    pub objective: Option<f64>,
    pub iterations: u64,
    pub warm: Option<WarmStart>,
    /// Reconstructed solver log (minimization sense), captured when `verbose`.
    pub raw_log: Option<String>,
}

/// Translate `model`, solve with POUNCE (cold), and map the outcome.
///
/// # Errors
///
/// [`SolverError::UnsupportedKind`] for integer/cone model kinds and
/// [`SolverError::Core`] for a model with neither an objective nor a declared
/// feasibility problem.
pub fn solve(model: &Model, opts: &PounceOptions) -> Result<SolverResult, SolverError> {
    let prep = setup(model)?;
    let started = Instant::now();
    let oracle = backend::build(model)?;
    let outcome = backend::run(&oracle, &prep, opts, None)?;
    Ok(assemble(prep.sign, outcome, started.elapsed()))
}

/// Kind gate, objective declaration check, sign, and bound snapshot.
pub(crate) fn setup(model: &Model) -> Result<Prepared, SolverError> {
    let kind = model.kind();
    if !matches!(kind, ModelKind::LP | ModelKind::QP | ModelKind::QCP | ModelKind::NLP) {
        return Err(SolverError::UnsupportedKind(kind));
    }
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

    let constraints = model.constraints();
    let mut g_l = Vec::with_capacity(constraints.len());
    let mut g_u = Vec::with_capacity(constraints.len());
    for c in constraints.iter() {
        g_l.push(c.lower.max(-POUNCE_INFINITY));
        g_u.push(c.upper.min(POUNCE_INFINITY));
    }

    Ok(Prepared { sign, x_l, x_u, g_l, g_u, x0 })
}

/// Midpoint of finite bounds, otherwise zero clipped into the bounds.
fn initial_guess(lb: f64, ub: f64) -> f64 {
    if lb.is_finite() && ub.is_finite() { f64::midpoint(lb, ub) } else { 0.0_f64.clamp(lb, ub) }
}

/// Map a POUNCE application status onto oximo's termination taxonomy.
pub(crate) fn map_status(s: ApplicationReturnStatus) -> TerminationStatus {
    use ApplicationReturnStatus as A;
    match s {
        A::SolveSucceeded => TerminationStatus::Optimal,
        A::SolvedToAcceptableLevel | A::FeasiblePointFound => TerminationStatus::LocallyOptimal,
        A::InfeasibleProblemDetected => TerminationStatus::Infeasible,
        A::DivergingIterates => TerminationStatus::Unbounded,
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
        solutions.push(SolutionPoint { primal, objective: o.objective.map(|f| sign * f) });
    }

    let primal_status = PrimalStatus::infer(&o.termination, has_point);
    SolverResult {
        termination: o.termination,
        primal_status,
        solutions,
        dual,
        soc_dual: FxHashMap::default(),
        reduced_costs,
        best_bound: None,
        gap: None,
        solve_time: elapsed,
        iterations: o.iterations,
        raw_log: o.raw_log,
        solver_name: Some("pounce".into()),
    }
}

/// The effective `print_level`:
/// Explicit `print_level`, else 5 when `verbose`, else 0 (quiet).
pub(crate) fn print_level(opts: &PounceOptions) -> i32 {
    opts.print_level.map_or(if opts.universal.verbose == Some(true) { 5 } else { 0 }, |v| {
        i32::try_from(v).unwrap_or(12)
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
        set_num(list, "max_cpu_time", limit.as_secs_f64())?;
    }
    if let Some(s) = opts.mu_strategy {
        set_str(list, "mu_strategy", mu_strategy_str(s))?;
    }
    for &(name, v) in opts.num_opts() {
        set_num(list, name, v)?;
    }
    for &(name, v) in opts.int_opts() {
        set_int(list, name, v)?;
    }
    for (name, v) in opts.str_opts() {
        set_str(list, name, v)?;
    }
    for &(name, v) in opts.bool_opts() {
        set_bool(list, name, v)?;
    }
    for (name, value) in &opts.extra {
        match value {
            PounceOptionValue::Num(v) => set_num(list, name, *v)?,
            PounceOptionValue::Int(v) => set_int(list, name, *v)?,
            PounceOptionValue::Str(v) => set_str(list, name, v)?,
            PounceOptionValue::Bool(v) => set_bool(list, name, *v)?,
        }
    }
    Ok(())
}
