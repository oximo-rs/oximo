//! Structural extraction and the specialized POUNCE convex routes.

use std::fmt::Write as _;
use std::time::Instant;

use oximo_core::{Model, ModelKind, ObjectiveSense, Sense, SocForm, detect_soc, explicit_soc_form};
use oximo_expr::{LinearTerms, extract_linear, extract_quadratic};
use oximo_solver::{DualStatus, SolverError, SolverResult, TerminationStatus};
use pounce_rs::IpoptApplication;
use pounce_rs::convex::{
    ActiveSetOverrides, ConeSpec, QpOptions, QpProblem, QpSolution, QpStatus, QpWarmStart, Triplet,
    solve_qp_active_set, solve_qp_ipm, solve_qp_ipm_warm, solve_socp_ipm, solve_socp_ipm_warm,
};
use pounce_rs::linsol::backend;

use crate::options::{PounceAlgorithm, PounceOptionValue, PounceOptions, PounceSolverSelection};
use crate::translate::{
    Outcome, apply_options, assemble, selected_algorithm, selected_solver, solve_nlp_since,
};

const PSD_TOL: f64 = 1e-9;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Route {
    Nlp,
    QpIpm,
    QpActiveSet,
    Socp,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Class {
    Lp,
    ConvexQp,
    Socp,
    General,
}

/// Resolve automatic/forced routing, including the safe time-limit policy.
pub(crate) fn route(model: &Model, opts: &PounceOptions) -> Result<Route, SolverError> {
    let selection = selected_solver(opts)?;
    let algorithm = selected_algorithm(opts)?;
    let class = classify(model);
    let has_explicit_soc = !model.constraints().second_order_cones().is_empty();

    if selection == PounceSolverSelection::Auto && algorithm == PounceAlgorithm::ActiveSetSqp {
        return if has_explicit_soc {
            Err(incompatible("active-set-sqp", "a model with explicit SOC constraints"))
        } else {
            Ok(Route::Nlp)
        };
    }

    match selection {
        PounceSolverSelection::Auto => match class {
            Class::Lp | Class::ConvexQp => Ok(Route::QpIpm),
            Class::Socp => Ok(Route::Socp),
            Class::General => {
                if has_explicit_soc {
                    Err(SolverError::UnsupportedKind(ModelKind::SOCP))
                } else {
                    Ok(Route::Nlp)
                }
            }
        },
        PounceSolverSelection::Nlp => {
            if has_explicit_soc {
                Err(incompatible("nlp", "explicit SOC constraints"))
            } else {
                Ok(Route::Nlp)
            }
        }
        PounceSolverSelection::LpIpm => {
            if class == Class::Lp {
                Ok(Route::QpIpm)
            } else {
                Err(incompatible("lp-ipm", "anything other than an LP"))
            }
        }
        PounceSolverSelection::QpIpm => {
            if matches!(class, Class::Lp | Class::ConvexQp) {
                Ok(Route::QpIpm)
            } else {
                Err(incompatible("qp-ipm", "a model that is not an LP or proven-convex QP"))
            }
        }
        PounceSolverSelection::QpActiveSet => {
            if matches!(class, Class::Lp | Class::ConvexQp) {
                Ok(Route::QpActiveSet)
            } else {
                Err(incompatible("qp-active-set", "a model that is not an LP or proven-convex QP"))
            }
        }
        PounceSolverSelection::Socp => {
            if matches!(class, Class::Lp | Class::ConvexQp | Class::Socp) {
                Ok(Route::Socp)
            } else {
                Err(incompatible("socp", "a model outside the convex LP/QP/SOCP family"))
            }
        }
    }
}

fn incompatible(selection: &str, detail: &str) -> SolverError {
    SolverError::Backend(format!(
        "pounce solver_selection `{selection}` is incompatible with {detail}"
    ))
}

fn classify(model: &Model) -> Class {
    match model.kind() {
        ModelKind::LP => Class::Lp,
        ModelKind::QP => {
            let arena = model.arena();
            let sign = objective_sign(model);
            let psd = model
                .objective()
                .as_ref()
                .and_then(|obj| extract_quadratic(&arena, obj.expr))
                .is_some_and(|q| hessian_is_psd(&q.hessian, sign));
            if psd { Class::ConvexQp } else { Class::General }
        }
        ModelKind::SOCP => {
            let arena = model.arena();
            let sign = objective_sign(model);
            let convex = model.objective().as_ref().is_none_or(|obj| {
                extract_quadratic(&arena, obj.expr)
                    .is_some_and(|q| hessian_is_psd(&q.hessian, sign))
            });
            if convex { Class::Socp } else { Class::General }
        }
        _ => Class::General,
    }
}

fn objective_sign(model: &Model) -> f64 {
    if model.objective().as_ref().is_some_and(|o| o.sense == ObjectiveSense::Maximize) {
        -1.0
    } else {
        1.0
    }
}

/// POUNCE's safe classifier using the reusable PSD certificate.
fn hessian_is_psd(hessian: &[(oximo_expr::VarId, oximo_expr::VarId, f64)], sign: f64) -> bool {
    let n = hessian
        .iter()
        .flat_map(|(row, col, _)| [row.index(), col.index()])
        .max()
        .map_or(0, |index| index + 1);
    let triplets = hessian
        .iter()
        .map(|&(row, col, value)| {
            Triplet::new(row.index().max(col.index()), row.index().min(col.index()), sign * value)
        })
        .collect::<Vec<_>>();
    pounce_rs::convex::certify_psd_lower_triangle(n, &triplets, PSD_TOL, backend).unwrap_or(false)
}

#[derive(Clone, Debug)]
enum ConstraintMap {
    None,
    Eq(usize),
    Ineq { upper: Option<usize>, lower: Option<usize> },
}

#[derive(Clone, Debug)]
pub(crate) struct Problem {
    pub qp: QpProblem,
    pub cones: Vec<ConeSpec>,
    maps: Vec<ConstraintMap>,
    explicit_soc_starts: Vec<usize>,
    sign: f64,
    objective_constant: f64,
}

impl Problem {
    pub(crate) fn same_structure(&self, other: &Self) -> bool {
        fn triplets_same(left: &[Triplet], right: &[Triplet]) -> bool {
            left.len() == right.len()
                && left.iter().zip(right).all(|(a, b)| a.row == b.row && a.col == b.col)
        }
        self.qp.n == other.qp.n
            && self.qp.b.len() == other.qp.b.len()
            && self.qp.h.len() == other.qp.h.len()
            && self.cones == other.cones
            && triplets_same(&self.qp.p_lower, &other.qp.p_lower)
            && triplets_same(&self.qp.a, &other.qp.a)
            && triplets_same(&self.qp.g, &other.qp.g)
            && self.qp.lb.iter().zip(&other.qp.lb).all(|(a, b)| a.is_finite() == b.is_finite())
            && self.qp.ub.iter().zip(&other.qp.ub).all(|(a, b)| a.is_finite() == b.is_finite())
    }

    pub(crate) const fn sign(&self) -> f64 {
        self.sign
    }
}

fn push_row(out: &mut Vec<Triplet>, row: usize, terms: &LinearTerms<'_>, scale: f64) {
    out.extend(
        terms.coeffs.iter().map(|&(var, value)| Triplet::new(row, var.index(), scale * value)),
    );
}

/// Extract a proven-convex model into POUNCE's standard form.
#[expect(
    clippy::many_single_char_names,
    reason = "A, b, G, h, c, and n are the conventional standard-form QP symbols"
)]
#[expect(clippy::too_many_lines)]
pub(crate) fn build_problem(model: &Model, opts: &PounceOptions) -> Result<Problem, SolverError> {
    if model.has_active_sos_constraints() {
        return Err(SolverError::UnsupportedSos);
    }
    model.ensure_objective_declared().map_err(SolverError::Core)?;
    let arena = model.arena();
    let vars = model.variables();
    let n = vars.len();
    let sign = objective_sign(model);
    let mut p_lower = Vec::new();
    let mut c = vec![0.0; n];
    let mut objective_constant = 0.0;
    if let Some(obj) = model.objective().as_ref() {
        let q = extract_quadratic(&arena, obj.expr).ok_or_else(|| {
            SolverError::Backend("pounce convex route requires a quadratic objective".into())
        })?;
        p_lower.extend(
            q.hessian
                .into_iter()
                .map(|(row, col, value)| Triplet::new(row.index(), col.index(), sign * value)),
        );
        for (var, value) in q.linear {
            c[var.index()] += sign * value;
        }
        objective_constant = sign * q.constant;
    }

    let model_constraints = model.constraints();
    let algebraic = model_constraints.algebraic();
    let mut maps = vec![ConstraintMap::None; algebraic.len()];
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut g = Vec::new();
    let mut h = Vec::new();
    let mut detected: Vec<(usize, SocForm)> = Vec::new();

    for (index, constraint) in algebraic.iter().enumerate() {
        if let Some(terms) = extract_linear(&arena, constraint.lhs) {
            if let Some((Sense::Eq, rhs)) = constraint.as_single() {
                let row = b.len();
                push_row(&mut a, row, &terms, 1.0);
                b.push(rhs - terms.constant);
                maps[index] = ConstraintMap::Eq(row);
                continue;
            }
            let mut upper = None;
            let mut lower = None;
            if constraint.upper.is_finite() {
                let row = h.len();
                push_row(&mut g, row, &terms, 1.0);
                h.push(relaxed_row_upper(constraint.upper, opts) - terms.constant);
                upper = Some(row);
            }
            if constraint.lower.is_finite() {
                let row = h.len();
                push_row(&mut g, row, &terms, -1.0);
                h.push(-(relaxed_row_lower(constraint.lower, opts) - terms.constant));
                lower = Some(row);
            }
            maps[index] = ConstraintMap::Ineq { upper, lower };
        } else if let Some(form) = detect_soc(&arena, &vars, constraint) {
            detected.push((index, form));
        } else {
            return Err(SolverError::Backend(format!(
                "constraint {:?} cannot be represented by the selected convex route",
                constraint.name
            )));
        }
    }

    let linear_ineq = h.len();
    let mut cones = Vec::new();
    if linear_ineq > 0 {
        cones.push(ConeSpec::Nonneg(linear_ineq));
    }
    let mut explicit_soc_starts = Vec::new();
    for soc in model_constraints.second_order_cones() {
        let form = explicit_soc_form(&arena, soc).ok_or_else(|| {
            SolverError::Backend(format!("invalid explicit SOC constraint {:?}", soc.name))
        })?;
        explicit_soc_starts.push(h.len());
        append_soc(&mut g, &mut h, &form);
        cones.push(ConeSpec::SecondOrder(1 + form.terms.len()));
    }
    for (index, form) in detected {
        maps[index] = ConstraintMap::Ineq { upper: Some(h.len()), lower: None };
        append_soc(&mut g, &mut h, &form);
        cones.push(ConeSpec::SecondOrder(1 + form.terms.len()));
    }

    Ok(Problem {
        qp: QpProblem {
            n,
            p_lower,
            c,
            a,
            b,
            g,
            h,
            lb: vars.iter().map(|v| relaxed_lower(v.lb, opts)).collect(),
            ub: vars.iter().map(|v| relaxed_upper(v.ub, opts)).collect(),
        },
        cones,
        maps,
        explicit_soc_starts,
        sign,
        objective_constant,
    })
}

fn variable_relaxation_delta(bound: f64, opts: &PounceOptions) -> f64 {
    let Some(factor) = num_value(opts, "bound_relax_factor") else {
        return 0.0;
    };
    if !factor.is_finite() || factor <= 0.0 || !bound.is_finite() {
        return 0.0;
    }
    let cap = num_value(opts, "constr_viol_tol").unwrap_or(1e-4).max(0.0);
    (factor * bound.abs().max(1.0)).min(cap)
}

fn relaxed_lower(bound: f64, opts: &PounceOptions) -> f64 {
    if bound.is_finite() { bound - variable_relaxation_delta(bound, opts) } else { bound }
}

fn relaxed_upper(bound: f64, opts: &PounceOptions) -> f64 {
    if bound.is_finite() { bound + variable_relaxation_delta(bound, opts) } else { bound }
}

fn row_relaxation_delta(bound: f64, opts: &PounceOptions) -> f64 {
    let Some(factor) = num_value(opts, "bound_relax_factor") else {
        return 0.0;
    };
    if !factor.is_finite() || factor <= 0.0 || !bound.is_finite() {
        return 0.0;
    }
    let cap = num_value(opts, "constr_viol_tol").unwrap_or(1e-4).max(0.0);
    let scale = match bound {
        0.0 => 1.0,
        _ => bound.abs(),
    };
    (factor * scale).min(cap)
}

fn relaxed_row_lower(bound: f64, opts: &PounceOptions) -> f64 {
    if bound.is_finite() { bound - row_relaxation_delta(bound, opts) } else { bound }
}

fn relaxed_row_upper(bound: f64, opts: &PounceOptions) -> f64 {
    if bound.is_finite() { bound + row_relaxation_delta(bound, opts) } else { bound }
}

fn append_soc(g: &mut Vec<Triplet>, h: &mut Vec<f64>, form: &SocForm) {
    let row = h.len();
    push_row(g, row, &form.bound, -1.0);
    h.push(form.bound.constant);
    for term in &form.terms {
        let row = h.len();
        push_row(g, row, term, -1.0);
        h.push(term.constant);
    }
}

pub(crate) fn solve(
    model: &Model,
    opts: &PounceOptions,
    route: Route,
) -> Result<SolverResult, SolverError> {
    validate_options(opts)?;
    let problem = build_problem(model, opts)?;
    let started = Instant::now();
    let sol = run(&problem, opts, route, None);
    if should_fallback_to_nlp(model, opts, &sol)? {
        return solve_nlp_since(model, opts, started);
    }
    let outcome = outcome(&problem, opts, route, &sol);
    Ok(assemble(problem.sign, outcome, started.elapsed()))
}

/// Match POUNCE's automatic fallback policy.
/// An iteration limit is an authoritative convex result. Only a numerical
/// failure gets a second opinion, and only for representations the NLP route
/// can express.
pub(crate) fn should_fallback_to_nlp(
    model: &Model,
    opts: &PounceOptions,
    sol: &QpSolution,
) -> Result<bool, SolverError> {
    Ok(selected_solver(opts)? == PounceSolverSelection::Auto
        && sol.status == QpStatus::NumericalFailure
        && (model.kind() == ModelKind::LP
            || (model.kind() == ModelKind::SOCP
                && model.constraints().second_order_cones().is_empty())))
}

pub(crate) fn validate_options(opts: &PounceOptions) -> Result<(), SolverError> {
    let mut validator = IpoptApplication::new();
    validator.initialize().map_err(|error| {
        SolverError::Backend(format!("pounce option validator initialization failed: {error:?}"))
    })?;
    apply_options(validator.options_mut(), opts, false)?;
    validate_convex_options(opts)
}

pub(crate) fn run(
    problem: &Problem,
    opts: &PounceOptions,
    route: Route,
    warm: Option<&QpWarmStart>,
) -> QpSolution {
    let qp_opts = qp_options(opts);
    let presolve = presolve_enabled(opts);
    match route {
        Route::QpIpm => {
            let solve = |qp: &QpProblem| match warm {
                Some(w) if same_warm_shape(qp, w) => solve_qp_ipm_warm(qp, &qp_opts, w, backend),
                _ => solve_qp_ipm(qp, &qp_opts, backend),
            };
            if presolve && warm.is_none() {
                pounce_rs::convex::pounce_convex::presolve::solve_with_presolve(&problem.qp, solve)
            } else {
                solve(&problem.qp)
            }
        }
        Route::QpActiveSet => {
            let engine = active_set_options(opts);
            let mut factory = backend;
            let mut solve =
                |qp: &QpProblem| solve_qp_active_set(qp, &qp_opts, &engine, &mut factory);
            if presolve {
                pounce_rs::convex::pounce_convex::presolve::solve_with_presolve(&problem.qp, solve)
            } else {
                solve(&problem.qp)
            }
        }
        Route::Socp => run_socp(problem, &qp_opts, presolve, warm),
        Route::Nlp => unreachable!("NLP route passed to convex runner"),
    }
}

fn run_socp(
    problem: &Problem,
    opts: &QpOptions,
    presolve: bool,
    warm: Option<&QpWarmStart>,
) -> QpSolution {
    let (expanded, cones) = expand_bounds_for_conic_warm(&problem.qp, &problem.cones);
    if let Some(warm) = warm.filter(|w| same_warm_shape(&expanded, w)) {
        let mut solution = solve_socp_ipm_warm(&expanded, &cones, warm, opts, backend);
        restore_conic_bound_multipliers(&problem.qp, &mut solution);
        return solution;
    }
    if !presolve {
        return solve_socp_ipm(&problem.qp, &problem.cones, opts, backend);
    }
    use pounce_rs::convex::pounce_convex::presolve::{PresolveOutcome, presolve_conic};
    match presolve_conic(&problem.qp, &problem.cones) {
        PresolveOutcome::Reduced(ps) => {
            let reduced_cones = ps.reduced_cones(&problem.cones);
            let reduced = solve_socp_ipm(&ps.reduced, &reduced_cones, opts, backend);
            ps.postsolve(&reduced)
        }
        PresolveOutcome::Infeasible(_) => empty_solution(&problem.qp, QpStatus::PrimalInfeasible),
        PresolveOutcome::Unbounded => empty_solution(&problem.qp, QpStatus::DualInfeasible),
    }
}

fn restore_conic_bound_multipliers(qp: &QpProblem, solution: &mut QpSolution) {
    let mut row = qp.h.len();
    solution.z_lb = vec![0.0; qp.n];
    solution.z_ub = vec![0.0; qp.n];
    for i in 0..qp.n {
        if qp.ub[i].is_finite() {
            solution.z_ub[i] = solution.z.get(row).copied().unwrap_or(0.0);
            row += 1;
        }
        if qp.lb[i].is_finite() {
            solution.z_lb[i] = solution.z.get(row).copied().unwrap_or(0.0);
            row += 1;
        }
    }
}

fn empty_solution(qp: &QpProblem, status: QpStatus) -> QpSolution {
    QpSolution {
        status,
        x: vec![0.0; qp.n],
        y: vec![0.0; qp.b.len()],
        z: vec![0.0; qp.h.len()],
        z_lb: vec![0.0; qp.n],
        z_ub: vec![0.0; qp.n],
        obj: 0.0,
        iters: 0,
        iterates: Vec::new(),
    }
}

fn same_warm_shape(qp: &QpProblem, warm: &QpWarmStart) -> bool {
    warm.x.len() == qp.n && warm.y.len() == qp.b.len() && warm.z.len() == qp.h.len()
}

/// Warm conic solves require bounds to be explicit orthant rows.
pub(crate) fn expand_bounds_for_conic_warm(
    qp: &QpProblem,
    cones: &[ConeSpec],
) -> (QpProblem, Vec<ConeSpec>) {
    let mut out = qp.clone();
    let mut added = 0;
    for i in 0..qp.n {
        if qp.ub[i].is_finite() {
            let row = out.h.len();
            out.g.push(Triplet::new(row, i, 1.0));
            out.h.push(qp.ub[i]);
            added += 1;
        }
        if qp.lb[i].is_finite() {
            let row = out.h.len();
            out.g.push(Triplet::new(row, i, -1.0));
            out.h.push(-qp.lb[i]);
            added += 1;
        }
    }
    out.lb.clear();
    out.ub.clear();
    let mut out_cones = cones.to_vec();
    if added > 0 {
        out_cones.push(ConeSpec::Nonneg(added));
    }
    (out, out_cones)
}

pub(crate) fn outcome(
    problem: &Problem,
    opts: &PounceOptions,
    route: Route,
    sol: &QpSolution,
) -> Outcome {
    let termination = match sol.status {
        QpStatus::Optimal => TerminationStatus::LocallyOptimal,
        QpStatus::OptimalInaccurate => TerminationStatus::Feasible,
        QpStatus::PrimalInfeasible => TerminationStatus::Infeasible,
        QpStatus::DualInfeasible => TerminationStatus::Unbounded,
        QpStatus::IterationLimit => TerminationStatus::IterationLimit,
        QpStatus::TimeLimit => TerminationStatus::TimeLimit,
        QpStatus::NumericalFailure => TerminationStatus::NumericError,
    };
    let mut lambda = vec![0.0; problem.maps.len()];
    for (value, map) in lambda.iter_mut().zip(&problem.maps) {
        *value = match map {
            ConstraintMap::None => 0.0,
            ConstraintMap::Eq(row) => sol.y.get(*row).copied().unwrap_or(0.0),
            ConstraintMap::Ineq { upper, lower } => {
                upper.and_then(|r| sol.z.get(r)).copied().unwrap_or(0.0)
                    - lower.and_then(|r| sol.z.get(r)).copied().unwrap_or(0.0)
            }
        };
    }
    let soc_dual = problem
        .explicit_soc_starts
        .iter()
        .map(|&row| sol.z.get(row).copied().unwrap_or(0.0))
        .collect();
    let reduced = Some(sol.z_lb.iter().zip(&sol.z_ub).map(|(l, u)| l - u).collect());
    Outcome {
        termination,
        dual_status: if matches!(sol.status, QpStatus::Optimal) {
            DualStatus::FeasiblePoint
        } else {
            DualStatus::Unknown
        },
        raw_status: format!("{:?}", sol.status),
        x: sol.x.clone(),
        lambda,
        soc_dual,
        reduced,
        objective: Some(sol.obj + problem.objective_constant),
        iterations: u64::try_from(sol.iters).unwrap_or(u64::MAX),
        warm: None,
        raw_log: (opts.universal.verbose == Some(true)).then(|| convex_log(route, sol)),
    }
}

fn convex_log(route: Route, sol: &QpSolution) -> String {
    let mut log = format!(
        "POUNCE convex route: {route:?}\nstatus: {:?}\nNumber of Iterations....: {}\nobjective: {:.16e}\n0 objective function evaluations (specialized convex engine)\n",
        sol.status, sol.iters, sol.obj
    );
    for it in &sol.iterates {
        let _ = writeln!(
            log,
            "iter={} obj={:.9e} inf_pr={:.3e} inf_du={:.3e} mu={:.3e} alpha_pr={:.3e} alpha_du={:.3e}",
            it.iter,
            it.objective,
            it.primal_infeasibility,
            it.dual_infeasibility,
            it.mu,
            it.alpha_primal,
            it.alpha_dual
        );
    }
    let _ = writeln!(log, "EXIT: {:?}", sol.status);
    log
}

fn qp_options(opts: &PounceOptions) -> QpOptions {
    let mut out = QpOptions::default();
    if let Some(value) = opts.tol {
        out.tol = value;
    }
    if let Some(value) = opts.max_iter {
        out.max_iter = value as usize;
    }
    if let Some(value) = num_value(opts, "qp_tau") {
        out.tau = value;
    }
    if let Some(value) = num_value(opts, "qp_tau_max") {
        out.tau_max = value;
    }
    if let Some(value) = num_value(opts, "qp_reg") {
        out.reg = value;
    }
    if let Some(value) = num_value(opts, "qp_infeas_tol") {
        out.infeas_tol = value;
    }
    if let Some(value) = bool_value(opts, "qp_hsde") {
        out.use_hsde = value;
    }
    if let Some(value) = bool_value(opts, "qp_equilibrate") {
        out.equilibrate = value;
    }
    if let Some(value) = bool_value(opts, "qp_crossover") {
        out.crossover = value;
    }
    if let Some(value) = int_value(opts, "qp_gondzio_corr") {
        out.gondzio_max_corr = usize::try_from(value).unwrap_or(0);
    }
    out.time_limit = opts.universal.time_limit;
    out.collect_iterates = opts.universal.verbose == Some(true);
    out
}

fn active_set_options(opts: &PounceOptions) -> ActiveSetOverrides {
    use pounce_rs::qp::AntiCyclingChoice;
    ActiveSetOverrides {
        max_iter: int_value(opts, "sqp_qp_max_iter").and_then(|v| u32::try_from(v).ok()),
        anti_cycling: str_value(opts, "sqp_qp_anti_cycling").and_then(|v| match v.as_str() {
            "expand" => Some(AntiCyclingChoice::Expand),
            "bland" => Some(AntiCyclingChoice::Bland),
            "none" => Some(AntiCyclingChoice::None),
            _ => None,
        }),
        feas_tol: num_value(opts, "sqp_qp_feas_tol"),
        opt_tol: num_value(opts, "sqp_qp_opt_tol"),
        elastic_gamma: num_value(opts, "sqp_qp_elastic_gamma"),
        use_schur_updates: bool_value(opts, "sqp_qp_use_schur_updates"),
        use_homotopy: bool_value(opts, "sqp_qp_use_homotopy"),
        max_schur_updates_before_refactor: int_value(
            opts,
            "sqp_qp_max_schur_updates_before_refactor",
        )
        .and_then(|v| u32::try_from(v).ok()),
        certify_second_order: bool_value(opts, "sqp_qp_certify_second_order"),
    }
}

fn presolve_enabled(opts: &PounceOptions) -> bool {
    bool_value(opts, "qp_presolve").or_else(|| bool_value(opts, "presolve")).unwrap_or(true)
}

fn num_value(opts: &PounceOptions, sought: &str) -> Option<f64> {
    opts.effective_num(sought)
}

fn int_value(opts: &PounceOptions, sought: &str) -> Option<i32> {
    opts.effective_int(sought)
}

fn bool_value(opts: &PounceOptions, sought: &str) -> Option<bool> {
    opts.effective_bool(sought)
}

fn str_value(opts: &PounceOptions, sought: &str) -> Option<String> {
    opts.effective_string(sought)
}

fn validate_convex_options(opts: &PounceOptions) -> Result<(), SolverError> {
    for (name, value) in &opts.extra {
        let expected =
            if matches!(name.as_str(), "qp_tau" | "qp_tau_max" | "qp_reg" | "qp_infeas_tol") {
                Some("number")
            } else if name == "qp_gondzio_corr" {
                Some("integer")
            } else if matches!(
                name.as_str(),
                "qp_hsde" | "qp_equilibrate" | "qp_crossover" | "qp_presolve"
            ) {
                Some("boolean")
            } else {
                None
            };
        let valid = match expected {
            Some("number") => matches!(value, PounceOptionValue::Num(_)),
            Some("integer") => matches!(value, PounceOptionValue::Int(_)),
            Some("boolean") => {
                matches!(value, PounceOptionValue::Bool(_))
                    || matches!(value, PounceOptionValue::Str(v) if matches!(v.as_str(), "yes" | "no" | "true" | "false" | "on" | "off"))
            }
            _ => true,
        };
        if !valid {
            return Err(SolverError::Backend(format!(
                "pounce option `{name}` must be a {}",
                expected.expect("expected kind exists when invalid")
            )));
        }
    }
    let numeric = [
        ("qp_tau", 0.0, 1.0, true),
        ("qp_tau_max", 0.0, 1.0, true),
        ("qp_reg", 0.0, f64::INFINITY, false),
        ("qp_infeas_tol", 0.0, f64::INFINITY, true),
        ("sqp_qp_feas_tol", 0.0, f64::INFINITY, true),
        ("sqp_qp_opt_tol", 0.0, f64::INFINITY, true),
        ("sqp_qp_elastic_gamma", 0.0, f64::INFINITY, true),
    ];
    for (name, lower, upper, strict) in numeric {
        if let Some(value) = num_value(opts, name) {
            let lower_ok = if strict { value > lower } else { value >= lower };
            if !value.is_finite() || !lower_ok || value >= upper {
                return Err(SolverError::Backend(format!(
                    "pounce rejected option `{name}`: value {value} is outside its valid range"
                )));
            }
        }
    }
    if let Some(value) = int_value(opts, "qp_gondzio_corr")
        && !(0..=10).contains(&value)
    {
        return Err(SolverError::Backend(
            "pounce rejected option `qp_gondzio_corr`: value must be between 0 and 10".into(),
        ));
    }
    if let (Some(tau), Some(tau_max)) = (num_value(opts, "qp_tau"), num_value(opts, "qp_tau_max"))
        && tau_max < tau
    {
        return Err(SolverError::Backend(
            "pounce rejected option `qp_tau_max`: it must be at least qp_tau".into(),
        ));
    }
    if let Some(value) = str_value(opts, "sqp_qp_anti_cycling")
        && !matches!(value.as_str(), "expand" | "bland" | "none")
    {
        return Err(SolverError::Backend(format!(
            "pounce rejected option `sqp_qp_anti_cycling`: `{value}` is invalid"
        )));
    }
    Ok(())
}

pub(crate) fn warm_from_solution(route: Route, problem: &Problem, sol: &QpSolution) -> QpWarmStart {
    if route == Route::Socp {
        let (expanded, _) = expand_bounds_for_conic_warm(&problem.qp, &problem.cones);
        let mut z = sol.z.clone();
        if z.len() == problem.qp.h.len() {
            for i in 0..problem.qp.n {
                if problem.qp.ub[i].is_finite() {
                    z.push(sol.z_ub.get(i).copied().unwrap_or(0.0));
                }
                if problem.qp.lb[i].is_finite() {
                    z.push(sol.z_lb.get(i).copied().unwrap_or(0.0));
                }
            }
        }
        debug_assert_eq!(z.len(), expanded.h.len());
        QpWarmStart { x: sol.x.clone(), y: sol.y.clone(), z, z_lb: Vec::new(), z_ub: Vec::new() }
    } else {
        QpWarmStart::from_solution(sol)
    }
}

/// Resident active-set session.
pub(crate) struct ActivePersistent {
    session: pounce_rs::convex::ActiveSetSession,
    engine: ActiveSetOverrides,
    presolve: bool,
}

impl ActivePersistent {
    pub(crate) fn new() -> Self {
        let engine = ActiveSetOverrides::default();
        let presolve = true;
        let session = pounce_rs::convex::ActiveSetSession::new(backend)
            .with_engine_overrides(engine)
            .with_presolve(presolve);
        Self { session, engine, presolve }
    }

    pub(crate) fn solve(&mut self, problem: &Problem, opts: &PounceOptions) -> QpSolution {
        let engine = active_set_options(opts);
        let presolve = presolve_enabled(opts);
        if engine != self.engine || presolve != self.presolve {
            self.session = pounce_rs::convex::ActiveSetSession::new(backend)
                .with_engine_overrides(engine)
                .with_presolve(presolve);
            self.engine = engine;
            self.presolve = presolve;
        }
        self.session.set_options(qp_options(opts));
        self.session.solve(&problem.qp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relaxation_uses_distinct_variable_and_scale_relative_row_formulas() {
        let opts = PounceOptions::default().bound_relax_factor(1e-8).constr_viol_tol(1e-4);

        for (actual, expected) in [
            (variable_relaxation_delta(2e-12, &opts), 1e-8),
            (row_relaxation_delta(2e-12, &opts), 2e-20),
            (row_relaxation_delta(20_000.0, &opts), 1e-4),
            (row_relaxation_delta(0.0, &opts), 1e-8),
        ] {
            assert!((actual - expected).abs() <= expected * f64::EPSILON);
        }
    }
}
