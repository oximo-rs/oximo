//! The shared POUNCE `TNLP` adapter and solve driver.
//!
//! Both derivative paths ([`crate::exact`] on enzyme, [`crate::hybrid`] on
//! stable) plug a [`DerivativeOracle`] into the one [`OximoTnlp`] adapter and
//! solve through [`run`], so warm starts, option handling, statistics, and log
//! capture behave identically regardless of where derivatives come from.

use std::cell::RefCell;
use std::fmt::Write as _;
use std::rc::Rc;

use oximo_core::Model;
use oximo_solver::{DualStatus, SolverError};
use pounce_rs::pounce_nlp::solve_statistics::SolveStatistics;
use pounce_rs::presolve::{ExpressionProvider, FbbtTape, PresolveOptions, WarmPoint};
use pounce_rs::restoration::run_second_opinion_ladder;
use pounce_rs::session::{SessionSolution, TnlpPresolveSession};
use pounce_rs::{
    ApplicationReturnStatus, BoundsInfo, Index, IndexStyle, IpoptApplication, IpoptCq, IpoptData,
    NlpInfo, Solution, SparsityRequest, StartingPoint, TNLP,
};

use crate::options::{PounceAlgorithm, PounceOptions};
use crate::translate::{
    Outcome, Prepared, WarmStart, apply_options, map_status, selected_algorithm, set_str,
};

/// A derivative source for [`OximoTnlp`].
///
/// Values and derivatives are all in POUNCE's minimization sense before the
/// maximize sign flip.
pub(crate) trait DerivativeOracle {
    fn num_variables(&self) -> usize;
    fn num_constraints(&self) -> usize;
    /// Sparse `(constraint, variable)` Jacobian pattern in row-major order.
    fn jacobian_structure(&self) -> &[(usize, usize)];
    /// Sorted lower-triangle Hessian-of-the-Lagrangian pattern.
    fn hessian_structure(&self) -> &[(usize, usize)];
    /// Whether [`Self::eval_hessian_lagrangian`] provides exact values for
    /// [`Self::hessian_structure`].
    fn has_exact_hessian(&self) -> bool;
    fn eval_objective(&mut self, x: &[f64]) -> f64;
    fn eval_objective_gradient(&mut self, x: &[f64], grad: &mut [f64]);
    fn eval_constraints(&mut self, x: &[f64], g: &mut [f64]);
    /// Jacobian values aligned with [`Self::jacobian_structure`].
    fn eval_constraint_jacobian(&mut self, x: &[f64], vals: &mut [f64]);
    /// Hessian values aligned with [`Self::hessian_structure`].
    fn eval_hessian_lagrangian(
        &mut self,
        x: &[f64],
        obj_factor: f64,
        lambda: &[f64],
        vals: &mut [f64],
    );
}

fn to_index(v: usize) -> Index {
    Index::try_from(v).expect("index exceeds i32")
}

/// Drive one POUNCE solve of `oracle` over the snapshot in `prep`.
///
/// Applies options onto the live application.
/// Warm-starts from `warm` when given, and reads iteration
/// statistics back off the application after the solve.
pub(crate) fn run<O: DerivativeOracle + 'static>(
    model: &Model,
    oracle: &Rc<RefCell<O>>,
    prep: &Prepared,
    opts: &PounceOptions,
    warm: Option<&WarmStart>,
) -> Result<Outcome, SolverError> {
    let tnlp = Rc::new(RefCell::new(OximoTnlp::new(Rc::clone(oracle), prep, warm.cloned())));

    let mut app = IpoptApplication::new();
    app.initialize().map_err(|e| SolverError::Backend(format!("pounce init: {e:?}")))?;
    if !oracle.borrow().has_exact_hessian() {
        set_str(app.options_mut(), "hessian_approximation", "limited-memory")?;
    }
    apply_options(app.options_mut(), opts, warm.is_some())?;
    let collect_iterations = opts.universal.verbose == Some(true);
    let _collector = collect_iterations.then(pounce_rs::collector_scope);
    if collect_iterations {
        app.enable_iter_history();
    }
    if selected_algorithm(opts)? == PounceAlgorithm::ActiveSetSqp
        && let Some(warm) = warm
    {
        app.set_sqp_warm_start(pounce_rs::sqp::SqpIterates {
            x: warm.x.clone(),
            lambda_g: warm.lambda.clone(),
            lambda_x: warm.z_l.iter().zip(&warm.z_u).map(|(l, u)| l - u).collect(),
            working: warm.sqp_working.clone(),
        });
    }

    let inner = wrap_with_fbbt_if_enabled(&mut app, model, oracle, prep, &tnlp)?;
    let interior_point = selected_algorithm(opts)? == PounceAlgorithm::InteriorPoint;
    if interior_point {
        app.defer_end_verdict();
    }
    let base_status = app.optimize_tnlp(Rc::clone(&inner));
    let base_stats = app.statistics();
    let mut ladder_log = Vec::new();
    let (status, stats, iterations) = if interior_point {
        let ladder =
            run_second_opinion_ladder(&mut app, inner, base_status, base_stats, &mut |line| {
                ladder_log.push(line.to_owned());
            });
        let iterations = u64::try_from(ladder.total_iteration_count()).unwrap_or(u64::MAX);
        (ladder.status, ladder.statistics, iterations)
    } else {
        let iterations = u64::try_from(base_stats.iteration_count.max(0)).unwrap_or(0);
        (base_status, base_stats, iterations)
    };
    let sqp_working = app.last_sqp_working_set().cloned();
    let termination = map_status(status);
    let raw_log = (opts.universal.verbose == Some(true)).then(|| {
        let mut log = format_raw_log(&stats, status);
        if !ladder_log.is_empty() {
            use std::fmt::Write as _;
            for line in &ladder_log {
                let _ = writeln!(log, "{line}");
            }
        }
        log
    });

    if let Some(captured) = &mut tnlp.borrow_mut().captured {
        captured.warm.sqp_working = sqp_working;
        captured.warm.mu = Some(stats.final_mu);
    }
    let t = tnlp.borrow();
    Ok(match &t.captured {
        Some(c) => Outcome {
            has_point: crate::translate::nlp_has_point(status, &stats, opts),
            termination,
            dual_status: if matches!(status, ApplicationReturnStatus::SolveSucceeded) {
                DualStatus::FeasiblePoint
            } else {
                DualStatus::Unknown
            },
            raw_status: format!("{status:?}"),
            x: c.warm.x.clone(),
            lambda: c.warm.lambda.clone(),
            soc_dual: Vec::new(),
            reduced: Some(c.reduced.clone()),
            objective: Some(c.obj),
            iterations,
            warm: Some(c.warm.clone()),
            raw_log,
        },
        None => Outcome {
            has_point: false,
            termination,
            dual_status: DualStatus::Unknown,
            raw_status: format!("{status:?}"),
            x: Vec::new(),
            lambda: Vec::new(),
            soc_dual: Vec::new(),
            reduced: None,
            objective: None,
            iterations,
            warm: None,
            raw_log,
        },
    })
}

fn wrap_with_fbbt_if_enabled<O: DerivativeOracle + 'static>(
    app: &mut IpoptApplication,
    model: &Model,
    oracle: &Rc<RefCell<O>>,
    prep: &Prepared,
    tnlp: &Rc<RefCell<OximoTnlp<O>>>,
) -> Result<Rc<RefCell<dyn TNLP>>, SolverError> {
    let inner = Rc::clone(tnlp) as Rc<RefCell<dyn TNLP>>;
    let presolve_opts = PresolveOptions::from_options_list(app.options())
        .map_err(|error| SolverError::Backend(format!("pounce presolve: {error}")))?;
    if !presolve_opts.enabled || !presolve_opts.fbbt {
        return Ok(inner);
    }
    let tapes = crate::fbbt::checked_constraint_tapes(model)?;
    validate_fbbt_values(oracle, prep, &tapes)?;
    tnlp.borrow_mut().fbbt_constraints = tapes;
    let provider = Rc::clone(tnlp) as Rc<RefCell<dyn ExpressionProvider>>;
    let wrapped = pounce_rs::presolve::wrap_with_presolve_provider(inner, provider, presolve_opts)
        .map_err(|error| SolverError::Backend(format!("pounce presolve: {error}")))?;
    app.set_presolve_already_applied(true);
    Ok(wrapped)
}

fn validate_fbbt_values<O: DerivativeOracle>(
    oracle: &Rc<RefCell<O>>,
    prep: &Prepared,
    tapes: &[Option<FbbtTape>],
) -> Result<(), SolverError> {
    use pounce_rs::presolve::pounce_presolve::fbbt::{forward_pass, forward_result};

    if tapes.len() != oracle.borrow().num_constraints() {
        return Err(SolverError::Backend(
            "oximo-pounce: generated FBBT tape rejected: constraint count differs from TNLP".into(),
        ));
    }
    let midpoint = prep
        .x_l
        .iter()
        .zip(&prep.x_u)
        .zip(&prep.x0)
        .map(|((&lo, &hi), &start)| {
            if lo > -crate::translate::POUNCE_INFINITY && hi < crate::translate::POUNCE_INFINITY {
                lo + 0.5 * (hi - lo)
            } else {
                start
            }
        })
        .collect::<Vec<_>>();

    for (label, point) in [("starting point", prep.x0.as_slice()), ("box midpoint", &midpoint)] {
        let mut values = vec![0.0; tapes.len()];
        oracle.borrow_mut().eval_constraints(point, &mut values);
        for (constraint, tape) in tapes.iter().enumerate() {
            let Some(tape) = tape.as_ref().filter(|tape| !tape.is_empty()) else {
                continue;
            };
            let actual = values[constraint];
            if !actual.is_finite() {
                continue;
            }
            let slots = forward_pass(tape, point, point).map_err(|error| {
                SolverError::Backend(format!(
                    "oximo-pounce: generated FBBT tape rejected for constraint {constraint} at {label}: {error:?}"
                ))
            })?;
            let range = forward_result(&slots);
            let scale = [actual, range.lo, range.hi]
                .into_iter()
                .filter(|value| value.is_finite())
                .fold(1.0_f64, |scale, value| scale.max(value.abs()));
            let tolerance = f64::EPSILON.sqrt() * scale;
            if !range.contains(actual)
                && (range.is_empty()
                    || actual < range.lo - tolerance
                    || actual > range.hi + tolerance)
            {
                return Err(SolverError::Backend(format!(
                    "oximo-pounce: generated FBBT tape rejected for constraint {constraint} at {label}: callback value {actual} is outside [{}, {}]",
                    range.lo, range.hi
                )));
            }
        }
    }
    Ok(())
}

/// A live TNLP plus POUNCE's presolve-aware warm session.
///
/// The typed handle is retained alongside the erased handle owned by POUNCE so
/// refreshed model data can be installed before each solve.
pub(crate) struct Resident<O> {
    tnlp: Rc<RefCell<OximoTnlp<O>>>,
    session: TnlpPresolveSession,
    options: PounceOptions,
}

impl<O: DerivativeOracle + 'static> Resident<O> {
    pub(crate) fn new(
        oracle: &Rc<RefCell<O>>,
        prep: &Prepared,
        opts: &PounceOptions,
    ) -> Result<Self, SolverError> {
        let tnlp = Rc::new(RefCell::new(OximoTnlp::new(Rc::clone(oracle), prep, None)));
        let inner = Rc::clone(&tnlp) as Rc<RefCell<dyn TNLP>>;
        let mut session = TnlpPresolveSession::new(inner)
            .map_err(|error| SolverError::Backend(format!("pounce session: {error}")))?;
        if !oracle.borrow().has_exact_hessian() {
            set_str(session.options_mut(), "hessian_approximation", "limited-memory")?;
        }
        apply_options(session.options_mut(), opts, true)?;
        if let Some(mu) = opts.effective_num("mu_init") {
            session
                .set_option_num("mu_init", mu)
                .map_err(|error| SolverError::Backend(format!("pounce session: {error}")))?;
        }
        if opts.universal.verbose == Some(true) {
            session.enable_iter_history();
        }
        Ok(Self { tnlp, session, options: opts.clone() })
    }

    pub(crate) fn matches_options(&self, opts: &PounceOptions) -> bool {
        self.options == *opts
    }

    pub(crate) fn invalidate(&mut self) {
        self.session.invalidate();
    }

    pub(crate) fn solve(
        &mut self,
        prep: &Prepared,
        opts: &PounceOptions,
        warm: Option<&WarmStart>,
    ) -> Result<Outcome, SolverError> {
        self.tnlp.borrow_mut().refresh(prep);
        let result = match warm {
            Some(warm) => {
                let point = WarmPoint {
                    x: warm.x.clone(),
                    lambda: warm.lambda.clone(),
                    z_l: warm.z_l.clone(),
                    z_u: warm.z_u.clone(),
                    mu: warm.mu,
                };
                self.session.solve_warm(&point)
            }
            None => self.session.solve_cold(),
        }
        .map_err(|error| SolverError::Backend(format!("pounce session: {error}")))?;
        Ok(session_outcome(result, opts, &self.tnlp.borrow()))
    }

    #[cfg(test)]
    pub(crate) fn last_presolve_reused(&self) -> Option<bool> {
        self.session.last().map(|solution| solution.presolve_reused)
    }
}

fn session_outcome<O: DerivativeOracle>(
    solution: SessionSolution,
    opts: &PounceOptions,
    tnlp: &OximoTnlp<O>,
) -> Outcome {
    let SessionSolution { status, x, objective, lambda, z_l, z_u, stats, .. } = solution;
    let termination = map_status(status);
    let has_point = crate::translate::nlp_has_point(status, &stats, opts);
    let iterations = u64::try_from(stats.iteration_count.max(0)).unwrap_or(0);
    let raw_log = (opts.universal.verbose == Some(true)).then(|| format_raw_log(&stats, status));
    let captured = x.len() == tnlp.oracle.borrow().num_variables()
        && lambda.len() == tnlp.oracle.borrow().num_constraints()
        && z_l.len() == x.len()
        && z_u.len() == x.len();
    let reduced = captured.then(|| z_l.iter().zip(&z_u).map(|(&zl, &zu)| zl - zu).collect());
    let warm = captured.then(|| WarmStart {
        x: x.clone(),
        z_l: z_l.clone(),
        z_u: z_u.clone(),
        lambda: lambda.clone(),
        mu: Some(stats.final_mu),
        sqp_working: None,
    });
    Outcome {
        has_point,
        termination,
        dual_status: if matches!(status, ApplicationReturnStatus::SolveSucceeded) {
            DualStatus::FeasiblePoint
        } else {
            DualStatus::Unknown
        },
        raw_status: format!("{status:?}"),
        x,
        lambda,
        soc_dual: Vec::new(),
        reduced,
        objective: captured.then_some(objective),
        iterations,
        warm,
        raw_log,
    }
}

/// The Ipopt-style end-of-solve report off the application's statistics
/// (values are in POUNCE's minimization sense).
#[expect(
    clippy::too_many_lines,
    reason = "the backend log intentionally mirrors POUNCE's complete end-of-solve report"
)]
pub(crate) fn format_raw_log(stats: &SolveStatistics, status: ApplicationReturnStatus) -> String {
    let mut log = String::new();
    let _ = writeln!(log, "Number of Iterations....: {}", stats.iteration_count);
    let _ =
        writeln!(log, "\n                                   (scaled)                 (unscaled)");
    let _ = writeln!(
        log,
        "Objective...............: {:24.16e} {:24.16e}",
        stats.final_scaled_objective, stats.final_objective
    );
    let _ = writeln!(
        log,
        "Dual infeasibility......: {:24.16e} {:24.16e}",
        stats.final_dual_inf, stats.final_unscaled_dual_inf
    );
    let _ = writeln!(
        log,
        "Constraint violation....: {:24.16e} {:24.16e}",
        stats.final_constr_viol, stats.final_unscaled_constr_viol
    );
    let _ = writeln!(
        log,
        "Complementarity.........: {:24.16e} {:24.16e}",
        stats.final_compl, stats.final_unscaled_compl
    );
    let _ = writeln!(
        log,
        "Overall NLP error.......: {:24.16e} {:24.16e}",
        stats.final_kkt_error, stats.final_unscaled_kkt_error
    );
    let _ =
        writeln!(log, "KKT error above row noise: {:24.16e}", stats.final_kkt_error_above_noise);
    let _ = writeln!(log);
    let _ = writeln!(
        log,
        "Number of objective function evaluations             = {}",
        stats.num_obj_evals
    );
    let _ = writeln!(
        log,
        "Number of objective gradient evaluations             = {}",
        stats.num_obj_grad_evals
    );
    let _ = writeln!(
        log,
        "Number of constraint evaluations                     = {}",
        stats.num_constr_evals
    );
    let _ = writeln!(
        log,
        "Number of constraint Jacobian evaluations            = {}",
        stats.num_constr_jac_evals
    );
    let _ = writeln!(
        log,
        "Number of Lagrangian Hessian evaluations             = {}",
        stats.num_hess_evals
    );
    if stats.restoration_calls > 0 {
        let _ = writeln!(
            log,
            "Restoration phase calls                              = {} ({} outer / {} inner iters, {:.3} s)",
            stats.restoration_calls,
            stats.restoration_outer_iters,
            stats.restoration_inner_iters,
            stats.restoration_wall_secs
        );
    }
    if stats.sqp_qp_solves > 0 {
        let _ = writeln!(
            log,
            "SQP QP solves / working-set changes                   = {} / {}",
            stats.sqp_qp_solves, stats.sqp_qp_working_set_changes
        );
    }
    if !stats.iterations.is_empty() {
        let _ = writeln!(log, "\nIteration history:");
        for row in &stats.iterations {
            let _ = writeln!(
                log,
                "iter={} obj={:.9e} inf_pr={:.3e} inf_du={:.3e} mu={:.3e} alpha_pr={:.3e} alpha_du={:.3e} resto={}",
                row.iter,
                row.objective,
                row.inf_pr,
                row.inf_du,
                row.mu,
                row.alpha_primal,
                row.alpha_dual,
                row.alpha_primal_char == 'r'
            );
        }
    }
    let _ = writeln!(
        log,
        "Total seconds (CPU)                                  = {:.3}",
        stats.total_cpu_time_secs
    );
    let _ = writeln!(
        log,
        "Total seconds (wallclock)                            = {:.3}",
        stats.total_wallclock_time_secs
    );
    let _ = writeln!(log, "\nEXIT: {status:?}");
    log
}

/// Captured final iterate from `finalize_solution`.
struct Captured {
    warm: WarmStart,
    /// `z_l − z_u` per variable (bound multipliers), for reduced costs.
    reduced: Vec<f64>,
    obj: f64,
}

/// POUNCE `TNLP` backed by a shared derivative oracle.
/// `sign` is `-1.0` for a Maximize model.
struct OximoTnlp<O> {
    oracle: Rc<RefCell<O>>,
    sign: f64,
    x_l: Vec<f64>,
    x_u: Vec<f64>,
    g_l: Vec<f64>,
    g_u: Vec<f64>,
    x0: Vec<f64>,
    warm: Option<WarmStart>,
    fbbt_constraints: Vec<Option<FbbtTape>>,
    captured: Option<Captured>,
}

impl<O> OximoTnlp<O> {
    fn new(oracle: Rc<RefCell<O>>, prep: &Prepared, warm: Option<WarmStart>) -> Self {
        Self {
            oracle,
            sign: prep.sign,
            x_l: prep.x_l.clone(),
            x_u: prep.x_u.clone(),
            g_l: prep.g_l.clone(),
            g_u: prep.g_u.clone(),
            x0: prep.x0.clone(),
            warm,
            fbbt_constraints: Vec::new(),
            captured: None,
        }
    }

    fn refresh(&mut self, prep: &Prepared) {
        self.sign = prep.sign;
        self.x_l.clone_from(&prep.x_l);
        self.x_u.clone_from(&prep.x_u);
        self.g_l.clone_from(&prep.g_l);
        self.g_u.clone_from(&prep.g_u);
        self.x0.clone_from(&prep.x0);
        self.warm = None;
        self.captured = None;
    }
}

impl<O> ExpressionProvider for OximoTnlp<O> {
    fn constraint_expression(&self, index: usize) -> Option<FbbtTape> {
        self.fbbt_constraints.get(index).and_then(Clone::clone)
    }
}

impl<O: DerivativeOracle> TNLP for OximoTnlp<O> {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        let e = self.oracle.borrow();
        Some(NlpInfo {
            n: to_index(e.num_variables()),
            m: to_index(e.num_constraints()),
            nnz_jac_g: to_index(e.jacobian_structure().len()),
            nnz_h_lag: to_index(e.hessian_structure().len()),
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&self.x_l);
        b.x_u.copy_from_slice(&self.x_u);
        b.g_l.copy_from_slice(&self.g_l);
        b.g_u.copy_from_slice(&self.g_u);
        true
    }

    fn get_starting_point(&mut self, mut sp: StartingPoint<'_>) -> bool {
        match &self.warm {
            Some(w) => {
                sp.x.copy_from_slice(&w.x);
                if w.z_l.len() == sp.z_l.len() && w.z_u.len() == sp.z_u.len() {
                    sp.init_z = true;
                    sp.z_l.copy_from_slice(&w.z_l);
                    sp.z_u.copy_from_slice(&w.z_u);
                }
                if w.lambda.len() == sp.lambda.len() {
                    sp.init_lambda = true;
                    sp.lambda.copy_from_slice(&w.lambda);
                }
            }
            None => sp.x.copy_from_slice(&self.x0),
        }
        true
    }

    fn eval_f(&mut self, x: &[f64], _new_x: bool) -> Option<f64> {
        Some(self.sign * self.oracle.borrow_mut().eval_objective(x))
    }

    fn eval_grad_f(&mut self, x: &[f64], _new_x: bool, grad_f: &mut [f64]) -> bool {
        self.oracle.borrow_mut().eval_objective_gradient(x, grad_f);
        for g in grad_f.iter_mut() {
            *g *= self.sign;
        }
        true
    }

    fn eval_g(&mut self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        self.oracle.borrow_mut().eval_constraints(x, g);
        true
    }

    fn eval_jac_g(&mut self, x: Option<&[f64]>, _new_x: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let e = self.oracle.borrow();
                for (k, &(r, c)) in e.jacobian_structure().iter().enumerate() {
                    irow[k] = to_index(r);
                    jcol[k] = to_index(c);
                }
            }
            SparsityRequest::Values { values } => {
                self.oracle
                    .borrow_mut()
                    .eval_constraint_jacobian(x.expect("jacobian values need x"), values);
            }
        }
        true
    }

    fn eval_h(
        &mut self,
        x: Option<&[f64]>,
        _new_x: bool,
        obj_factor: f64,
        lambda: Option<&[f64]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        if !self.oracle.borrow().has_exact_hessian() {
            return false;
        }
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let e = self.oracle.borrow();
                for (k, &(r, c)) in e.hessian_structure().iter().enumerate() {
                    irow[k] = to_index(r);
                    jcol[k] = to_index(c);
                }
            }
            SparsityRequest::Values { values } => {
                self.oracle.borrow_mut().eval_hessian_lagrangian(
                    x.expect("hessian values need x"),
                    self.sign * obj_factor,
                    lambda.expect("hessian values need lambda"),
                    values,
                );
            }
        }
        true
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _ip_data: &IpoptData, _ip_cq: &IpoptCq) {
        let reduced = sol.z_l.iter().zip(sol.z_u).map(|(&zl, &zu)| zl - zu).collect::<Vec<f64>>();
        self.captured = Some(Captured {
            warm: WarmStart {
                x: sol.x.to_vec(),
                z_l: sol.z_l.to_vec(),
                z_u: sol.z_u.to_vec(),
                lambda: sol.lambda.to_vec(),
                mu: None,
                sqp_working: None,
            },
            reduced,
            obj: sol.obj_value,
        });
    }
}

#[cfg(all(test, not(feature = "enzyme")))]
mod fbbt_tests {
    use super::*;
    use oximo_core::{constraint, objective, variable};
    use oximo_solver::prepare::LoweringContext;

    #[test]
    fn tnlp_expression_provider_tightens_bounds() {
        let model = Model::new("tnlp_fbbt");
        variable!(model, -10.0 <= x <= 10.0);
        objective!(model, Min, x.powi(2));
        constraint!(model, square_cap, x.powi(2) <= 4.0);

        let options = PounceOptions::default();
        let lowering = LoweringContext::new(&model).unwrap();
        let prep = crate::translate::setup_prepared(&lowering, &options).unwrap();
        let oracle = Rc::new(RefCell::new(crate::hybrid::HybridOracle::new(&model)));
        let mut tnlp = OximoTnlp::new(oracle, &prep, None);
        tnlp.fbbt_constraints = crate::fbbt::checked_constraint_tapes(&model).unwrap();
        let tnlp = Rc::new(RefCell::new(tnlp));
        let inner = Rc::clone(&tnlp) as Rc<RefCell<dyn TNLP>>;
        let provider = Rc::clone(&tnlp) as Rc<RefCell<dyn ExpressionProvider>>;
        let mut presolve_opts = PresolveOptions::defaults();
        presolve_opts.enabled = true;
        presolve_opts.fbbt = true;
        let wrapped =
            Rc::new(RefCell::new(pounce_rs::presolve::PresolveTnlp::with_expression_provider(
                inner,
                provider,
                presolve_opts,
            )));

        let mut app = IpoptApplication::new();
        app.initialize().unwrap();
        app.options_mut().set_integer_value("print_level", 0, true, true).unwrap();
        app.set_presolve_already_applied(true);
        let status = app.optimize_tnlp(Rc::clone(&wrapped) as Rc<RefCell<dyn TNLP>>);
        assert_eq!(status, ApplicationReturnStatus::SolveSucceeded);
        let report = wrapped.borrow().fbbt_report().expect("FBBT ran");
        assert!(report.bound_updates > 0, "{report:?}");
    }
}
