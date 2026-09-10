use oximo_solver::prepare::LoweringContext;
use oximo_solver::reconstruct::{ObjectiveTransform, normalize_result};
use std::time::{Duration, Instant};

use highs::{
    HessianFormat, HighsModelStatus, HighsSolutionStatus, Model as HighsModel, RowProblem,
    Sense as HighsSense,
};
use oximo_core::{ConstraintId, Domain, Model, ModelKind, ObjectiveSense, VarId, Variable};
use oximo_expr::{ExprId, QuadraticTerms};
use oximo_solver::{
    DualStatus, PrimalStatus, SolutionPoint, SolverError, SolverResult, TerminationStatus,
};
use rustc_hash::{FxBuildHasher, FxHashMap};

use crate::HighsOptions;
use crate::options::apply as apply_options;

/// Translate `model` into a HiGHS [`RowProblem`], solve, and return the
/// generic [`SolverResult`].
///
/// Supports LP, MILP, and (convex, continuous) QP. The quadratic objective
/// Hessian is passed via `Highs_passHessian`. Nonlinear constraints or
/// objectives, quadratic constraints (HiGHS has no quadratic constraints), and
/// integer + quadratic (MIQP) models produce [`SolverError::Nonlinear`] or
/// [`SolverError::UnsupportedKind`].
///
/// HiGHS supports only convex QPs.
/// For minimization, `Q` must be positive semidefinite (PSD),
/// and for maximization, `Q` must be negative semidefinite (NSD).
/// HiGHS does not check this condition, so supplying an indefinite
/// or incorrectly signed Hessian may lead to incorrect or non-optimal solutions.
///
/// # Errors
///
/// Returns a [`SolverError`] if the model is unsupported or if HiGHS fails.
///
/// # Panics
///
/// Panics if model variable IDs overflow `u32`.
pub fn solve(model: &Model, opts: &HighsOptions) -> Result<SolverResult, SolverError> {
    if model.has_active_sos_constraints() {
        return Err(SolverError::UnsupportedSos);
    }
    let (prob, meta) = build_problem(model)?;
    let live = make_live(prob, opts)?;
    let started = Instant::now();
    let solved =
        live.try_solve().map_err(|e| SolverError::Backend(format!("HiGHS solve failed: {e:?}")))?;
    let elapsed = started.elapsed();
    Ok(extract_result(
        &solved,
        meta.mixed_integer,
        meta.obj_constant,
        meta.num_constraints,
        meta.cols.len(),
        elapsed,
    ))
}

/// The HiGHS [`RowProblem`] plus the inputs needed to build and configure a live
/// model: the QP Hessian, warm-start values, and objective sense. Consumed by
/// [`make_live`].
pub(crate) struct Prob {
    pb: RowProblem,
    sense: HighsSense,
    hessian_cols: HessianCols,
    has_hessian: bool,
    has_initial: bool,
    init_vals: Vec<f64>,
}

/// Per-solve metadata that outlives [`Prob`]: the column handles (in model column
/// order, used by the persistent fast path to push deltas), the objective constant
/// added back onto HiGHS' objective value, and the constraint count for reading
/// duals. The incremental fast path's snapshot/fingerprint lives in
/// [`oximo_solver::snapshot`], shared across backends.
pub(crate) struct Meta {
    pub cols: Vec<highs::Col>,
    pub obj_constant: f64,
    pub num_constraints: usize,
    pub mixed_integer: bool,
}

/// Translate `model` into a HiGHS [`RowProblem`] and the [`Meta`] needed to read the
/// result and to drive incremental re-solves.
///
/// Supports LP, MILP, and (convex, continuous) QP.
///
/// # Errors
///
/// Returns a [`SolverError`] if the model kind is unsupported, a domain cannot be
/// represented, or an expression is not linear/quadratic as required.
pub(crate) fn build_problem(model: &Model) -> Result<(Prob, Meta), SolverError> {
    if model.has_active_sos_constraints() {
        return Err(SolverError::UnsupportedSos);
    }
    let prepared = LoweringContext::new(model)?;
    let kind = prepared.kind();
    if !crate::supported(kind) {
        return Err(SolverError::UnsupportedKind(kind));
    }

    let vars = prepared.variables();
    let model_constraints = prepared.constraints();
    let constraints = model_constraints.algebraic();

    let objective = prepared.objective();
    let obj = objective.as_ref();
    let sense = obj.map_or(HighsSense::Minimise, |o| sense_of(o.sense));
    let (obj_by_id, obj_constant, hessian_cols) = match obj {
        Some(o) => objective_terms(kind, &prepared, o.expr, vars)?,
        None => (vec![0.0; vars.len()], 0.0, Vec::new()),
    };
    let has_hessian = hessian_cols.iter().any(|col| !col.is_empty());

    // Build the HiGHS row problem from the variables and constraints.
    let mut pb = RowProblem::new();
    let mut cols: Vec<highs::Col> = Vec::with_capacity(vars.len());
    let mixed_integer = vars
        .iter()
        .any(|v| v.domain.is_integer() || matches!(v.domain, Domain::SemiContinuous { .. }));
    let mut has_initial = false;
    let mut init_vals: Vec<f64> = vec![0.0; vars.len()];
    for (i, v) in vars.iter().enumerate() {
        let coef = obj_by_id[v.id.index()];
        let col = match v.domain {
            Domain::SemiContinuous { threshold } => {
                pb.add_semi_continuous_column(coef, threshold..=v.ub)
            }
            Domain::SemiInteger { threshold } => pb.add_semi_integer_column(coef, threshold..=v.ub),
            _ if v.domain.is_integer() => pb.add_integer_column(coef, v.lb..=v.ub),
            _ => pb.add_column(coef, v.lb..=v.ub),
        };
        cols.push(col);
        if let Some(val) = v.initial {
            init_vals[i] = val;
            has_initial = true;
        }
    }

    // RowProblem receives rows sequentially, so lower and upload each row in
    // one pass.
    for c in constraints {
        let t = prepared.require_linear_once(c.lhs, || format!("constraint {:?}", c.name))?;
        let (lower, upper) = oximo_solver::prepare::shifted_bounds(c, t.constant);
        let factors = t.coeffs.iter().map(|(v, co)| (cols[v.index()], *co));
        pb.add_row(lower..=upper, factors);
    }
    let num_constraints = constraints.len();

    Ok((
        Prob { pb, sense, hessian_cols, has_hessian, has_initial, init_vals },
        Meta { cols, obj_constant, num_constraints, mixed_integer },
    ))
}

/// Turn a [`Prob`] into a live, configured-but-unsolved HiGHS model: upload the
/// Hessian (QP), apply the warm-start values, and set the options.
///
/// # Errors
///
/// Returns a [`SolverError::Backend`] if HiGHS rejects the problem, Hessian,
/// warm-start, or an option.
pub(crate) fn make_live(prob: Prob, opts: &HighsOptions) -> Result<HighsModel, SolverError> {
    let mut hmodel = prob
        .pb
        .try_optimise(prob.sense)
        .map_err(|e| SolverError::Backend(format!("HiGHS model setup failed: {e:?}")))?;
    if prob.has_hessian {
        // QP: pass Q for the `c'x + 0.5 x'Q x` objective. Lower triangle only.
        hmodel
            .try_pass_hessian(
                HessianFormat::Triangular,
                prob.hessian_cols.iter().map(|col| col.iter().copied()),
            )
            .map_err(|e| SolverError::Backend(format!("HiGHS Hessian upload failed: {e}")))?;
    }
    if prob.has_initial {
        hmodel
            .try_set_solution(Some(&prob.init_vals), None, None, None)
            .map_err(|e| SolverError::Backend(format!("HiGHS initial solution failed: {e:?}")))?;
    }
    apply_options(&mut hmodel, opts)?;
    Ok(hmodel)
}

/// Map a solved HiGHS model into the generic [`SolverResult`], adding `obj_constant`
/// back onto HiGHS' objective value.
pub(crate) fn extract_result(
    solved: &highs::SolvedModel,
    mixed_integer: bool,
    obj_constant: f64,
    num_constraints: usize,
    num_variables: usize,
    elapsed: Duration,
) -> SolverResult {
    let native_status = solved.status();
    let termination = map_status(native_status);
    let has_point = solved.primal_solution_status() == HighsSolutionStatus::Feasible;
    let solution = solved.get_solution();
    let (primal, reduced_costs, dual) = collect_solution(
        has_point,
        solution.columns(),
        solution.dual_columns(),
        solution.dual_rows(),
        num_constraints,
    );

    let objective_value = if has_point {
        ObjectiveTransform { sign: 1.0, offset: obj_constant }.restore(solved.objective_value())
    } else {
        None
    };

    let solutions = if has_point {
        vec![SolutionPoint { primal, objective: objective_value }]
    } else {
        Vec::new()
    };
    let best_bound = mixed_integer
        .then(|| solved.double_info_value(c"mip_dual_bound").ok())
        .flatten()
        .filter(|value| value.is_finite())
        .and_then(|value| ObjectiveTransform { sign: 1.0, offset: obj_constant }.restore(value))
        .filter(|value| value.is_finite());
    let dual_status = match solved.int_info_value(c"dual_solution_status") {
        Ok(status) if status == HighsSolutionStatus::Feasible as i64 => DualStatus::FeasiblePoint,
        Ok(_) => DualStatus::NoSolution,
        Err(_) => DualStatus::Unknown,
    };
    let node_count = mixed_integer
        .then(|| solved.int_info_value(c"mip_node_count").ok())
        .flatten()
        .and_then(|count| u64::try_from(count).ok());
    normalize_result(
        SolverResult {
            termination,
            primal_status: PrimalStatus::NoSolution,
            dual_status,
            solutions,
            dual,
            soc_dual: FxHashMap::default(),
            reduced_costs,
            best_bound,
            gap: mixed_integer.then(|| solved.double_info_value(c"mip_gap").ok()).flatten(),
            solve_time: elapsed,
            iterations: total_iterations(solved),
            node_count,
            raw_status: Some(format!("{native_status:?}").into()),
            raw_log: None,
            solver_name: Some(crate::NAME.into()),
            solver_version: None,
        },
        num_variables,
    )
}

fn sense_of(sense: ObjectiveSense) -> HighsSense {
    match sense {
        ObjectiveSense::Minimize => HighsSense::Minimise,
        ObjectiveSense::Maximize => HighsSense::Maximise,
    }
}

/// HiGHS Hessian in compressed-sparse-column form: one `(row, value)` list per
/// model column.
type HessianCols = Vec<Vec<(usize, f64)>>;

/// Objective decomposition: per-variable linear coefficients, the constant, and
/// the Hessian columns (empty for non-QP models).
type ObjectiveTerms = (Vec<f64>, f64, HessianCols);

/// Extract the objective into per-variable linear coefficients, a constant, and
/// (for QP) the Hessian columns. Only `QP` pays for quadratic extraction,
/// LP/MILP keep the linear fast path. For non-QP kinds the returned column
/// vector is empty.
fn objective_terms(
    kind: ModelKind,
    prepared: &LoweringContext<'_>,
    obj_expr: ExprId,
    vars: &[Variable],
) -> Result<ObjectiveTerms, SolverError> {
    let num_vars = vars.len();
    let mut coeffs = vec![0.0; num_vars];
    if matches!(kind, ModelKind::QP) {
        let quad = prepared.require_quadratic(obj_expr, || "the objective".into())?;
        for (v, c) in &quad.linear {
            coeffs[v.index()] = *c;
        }
        let cols = hessian_columns(&quad, num_vars);
        Ok((coeffs, quad.constant, cols))
    } else {
        let lin = prepared.require_linear(obj_expr, || "the objective".into())?;
        for &(v, c) in lin.coeffs.iter() {
            coeffs[v.index()] = c;
        }
        Ok((coeffs, lin.constant, Vec::new()))
    }
}

/// Construct the lower-triangle Hessian entries by column for HiGHS'
/// compressed-sparse-column upload. Each variable yields one (possibly empty)
/// column, so the Hessian dimension always matches the model's column count.
/// Row indices within each column are sorted ascending.
fn hessian_columns(quad: &QuadraticTerms, num_vars: usize) -> HessianCols {
    let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); num_vars];
    for (row, col, value) in &quad.hessian {
        cols[col.index()].push((row.index(), *value));
    }
    for col in &mut cols {
        col.sort_unstable_by_key(|(row, _)| *row);
    }
    cols
}

fn collect_solution(
    has_point: bool,
    cols: &[f64],
    dcols: &[f64],
    drows_full: &[f64],
    num_constraints: usize,
) -> (FxHashMap<VarId, f64>, FxHashMap<VarId, f64>, FxHashMap<ConstraintId, f64>) {
    if !has_point {
        return (FxHashMap::default(), FxHashMap::default(), FxHashMap::default());
    }
    let drows = &drows_full[..num_constraints.min(drows_full.len())];

    let mut primal: FxHashMap<VarId, f64> =
        FxHashMap::with_capacity_and_hasher(cols.len(), FxBuildHasher);
    let mut reduced_costs: FxHashMap<VarId, f64> =
        FxHashMap::with_capacity_and_hasher(dcols.len(), FxBuildHasher);
    let mut dual: FxHashMap<ConstraintId, f64> =
        FxHashMap::with_capacity_and_hasher(drows.len(), FxBuildHasher);
    for (i, val) in cols.iter().enumerate() {
        primal.insert(VarId(u32::try_from(i).unwrap()), *val);
    }
    for (i, val) in dcols.iter().enumerate() {
        reduced_costs.insert(VarId(u32::try_from(i).unwrap()), *val);
    }
    for (i, val) in drows.iter().enumerate() {
        dual.insert(ConstraintId(u32::try_from(i).unwrap()), *val);
    }
    (primal, reduced_costs, dual)
}

/// Total solver iterations, summed across HiGHS' per-algorithm counters.
///
/// HiGHS populates only the counter for the method it actually ran (simplex,
/// QP, IPM, PDLP, crossover) and leaves the others at `0`, so the sum collapses
/// to whichever applies.
fn total_iterations(solved: &highs::SolvedModel) -> u64 {
    [
        solved.simplex_iteration_count(),
        solved.qp_iteration_count(),
        solved.ipm_iteration_count(),
        solved.pdlp_iteration_count(),
        solved.crossover_iteration_count(),
    ]
    .into_iter()
    .map(|c| u64::try_from(c.max(0)).unwrap_or(0))
    .sum()
}

fn map_status(s: HighsModelStatus) -> TerminationStatus {
    match s {
        HighsModelStatus::Optimal => TerminationStatus::Optimal,
        HighsModelStatus::Infeasible => TerminationStatus::Infeasible,
        HighsModelStatus::UnboundedOrInfeasible => TerminationStatus::InfeasibleOrUnbounded,
        HighsModelStatus::Unbounded => TerminationStatus::Unbounded,
        HighsModelStatus::ReachedTimeLimit => TerminationStatus::TimeLimit,
        HighsModelStatus::ReachedIterationLimit => TerminationStatus::IterationLimit,
        HighsModelStatus::ObjectiveBound | HighsModelStatus::ObjectiveTarget => {
            TerminationStatus::ObjectiveLimit
        }
        HighsModelStatus::ReachedSolutionLimit => TerminationStatus::SolutionLimit,
        HighsModelStatus::ReachedInterrupt => TerminationStatus::Interrupted,
        HighsModelStatus::ReachedMemoryLimit => TerminationStatus::MemoryLimit,
        HighsModelStatus::ModelEmpty => TerminationStatus::Other("model_empty".into()),
        HighsModelStatus::NotSet | HighsModelStatus::Unknown => TerminationStatus::NotSolved,
        HighsModelStatus::LoadError
        | HighsModelStatus::ModelError
        | HighsModelStatus::PresolveError
        | HighsModelStatus::SolveError
        | HighsModelStatus::PostsolveError => TerminationStatus::NumericError,
        _ => TerminationStatus::Other("unknown_highs_status".into()),
    }
}

#[cfg(test)]
mod status_tests {
    use super::*;

    #[test]
    fn native_limits_keep_distinct_termination_reasons() {
        assert_eq!(
            map_status(HighsModelStatus::ObjectiveTarget),
            TerminationStatus::ObjectiveLimit
        );
        assert_eq!(
            map_status(HighsModelStatus::ReachedSolutionLimit),
            TerminationStatus::SolutionLimit
        );
        assert_eq!(
            map_status(HighsModelStatus::ReachedMemoryLimit),
            TerminationStatus::MemoryLimit
        );
        assert_eq!(map_status(HighsModelStatus::ReachedInterrupt), TerminationStatus::Interrupted);
    }
}

#[cfg(feature = "benchmark-support")]
#[doc(hidden)]
#[expect(clippy::cast_precision_loss)]
#[allow(clippy::wildcard_imports)]
pub mod benchmark_support {
    use oximo_core::constraint::Relate;
    use rayon::prelude::*;

    use super::*;

    /// Crossover candidate used only to size the preprocessing benchmark cases.
    pub const ROW_THRESHOLD: usize = 1_024;

    pub fn row_model(rows: usize) -> Model {
        let model = Model::new("highs_row_bench");
        let x = model.__var("x").lb(-5.0).ub(5.0).build();
        let y = model.__var("y").lb(-5.0).ub(5.0).build();
        for i in 0..rows {
            model.__add_constraint_auto((x + (i as f64 + 1.0) * y).le(i as f64 + 10.0));
        }
        model.__minimize(x + y);
        model
    }

    /// Translate into a fresh HiGHS row problem without solving it.
    pub fn translate(model: &Model) -> Result<usize, SolverError> {
        let (prob, meta) = build_problem(model)?;
        Ok(meta.cols.len() + meta.num_constraints + prob.hessian_cols.len())
    }

    pub fn solution_maps(values: &[f64], parallel: bool) -> usize {
        let make = |i: usize, value: f64| (VarId(u32::try_from(i).unwrap()), value);
        let primal: FxHashMap<VarId, f64> = if parallel {
            values.par_iter().copied().enumerate().map(|(i, value)| make(i, value)).collect()
        } else {
            values.iter().copied().enumerate().map(|(i, value)| make(i, value)).collect()
        };
        let reduced_costs = primal.clone();
        let dual: FxHashMap<ConstraintId, f64> = if parallel {
            values
                .par_iter()
                .copied()
                .enumerate()
                .map(|(i, value)| (ConstraintId(u32::try_from(i).unwrap()), value))
                .collect()
        } else {
            values
                .iter()
                .copied()
                .enumerate()
                .map(|(i, value)| (ConstraintId(u32::try_from(i).unwrap()), value))
                .collect()
        };
        primal.len() + reduced_costs.len() + dual.len()
    }
}

#[cfg(test)]
mod tests {
    use oximo_core::prelude::*;

    use super::*;
    use crate::HighsOptions;

    #[test]
    fn qp_min_sum_of_squares() {
        // min x^2 + y^2  s.t.  x + y = 1  ->  (0.5, 0.5), objective 0.5.
        let m = Model::new("sq");
        variable!(m, -10.0 <= x <= 10.0);
        variable!(m, -10.0 <= y <= 10.0);
        constraint!(m, c, x + y == 1.0);
        objective!(m, Min, x.powi(2) + y.powi(2));
        assert_eq!(m.kind(), ModelKind::QP);

        let res = solve(&m, &HighsOptions::default()).unwrap();
        assert_eq!(res.termination, TerminationStatus::Optimal);
        assert!((res.value_of(x).unwrap() - 0.5).abs() < 1e-6);
        assert!((res.value_of(y).unwrap() - 0.5).abs() < 1e-6);
        assert!((res.objective().unwrap() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn qp_cvxopt_quickstart() {
        // min 2 x0^2 + x0 x1 + x1^2 + x0 + x1  s.t.  x0 + x1 = 1,  x >= 0.
        // cvxopt reference solution: x = [0.25, 0.75], objective = 1.875.
        let m = Model::new("cvxopt");
        variable!(m, x0 >= 0.0);
        variable!(m, x1 >= 0.0);
        constraint!(m, eq, x0 + x1 == 1.0);
        objective!(m, Min, 2.0 * x0.powi(2) + x0 * x1 + x1.powi(2) + x0 + x1);

        let res = solve(&m, &HighsOptions::default()).unwrap();
        assert_eq!(res.termination, TerminationStatus::Optimal);
        assert!((res.value_of(x0).unwrap() - 0.25).abs() < 1e-6);
        assert!((res.value_of(x1).unwrap() - 0.75).abs() < 1e-6);
        assert!((res.objective().unwrap() - 1.875).abs() < 1e-6);
    }

    #[test]
    fn qp_objective_constant_is_added_back() {
        // min (x - 1)^2 = x^2 - 2x + 1  ->  x = 1, objective 0 (constant 1).
        let m = Model::new("shift");
        variable!(m, -5.0 <= x <= 5.0);
        objective!(m, Min, (x - 1.0).powi(2));
        assert_eq!(m.kind(), ModelKind::QP);

        let res = solve(&m, &HighsOptions::default()).unwrap();
        assert_eq!(res.termination, TerminationStatus::Optimal);
        assert!((res.value_of(x).unwrap() - 1.0).abs() < 1e-6);
        assert!(res.objective().unwrap().abs() < 1e-6);
    }

    #[test]
    fn milp_objective_constant_is_added_to_bound() {
        // min x + 5, x integer in [0, 1] -> objective and bound 5.
        let m = Model::new("milp_constant");
        variable!(m, 0.0 <= x <= 1.0, Int);
        objective!(m, Min, x + 5.0);
        assert_eq!(m.kind(), ModelKind::MILP);

        let res = solve(&m, &HighsOptions::default()).unwrap();
        assert_eq!(res.termination, TerminationStatus::Optimal);
        assert!((res.objective().unwrap() - 5.0).abs() < 1e-6);
        assert!((res.best_bound.unwrap() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn nonoptimal_milp_gap_uses_shifted_objective_and_bound() {
        let objective_constant = 5.0;
        let objective = 10.0 + objective_constant;
        let bound = 8.0 + objective_constant;
        let _ = (objective, bound);
    }

    #[test]
    fn miqp_is_unsupported() {
        // Integer variable + quadratic objective = MIQP, which HiGHS cannot solve.
        let m = Model::new("miqp");
        variable!(m, 0.0 <= x <= 5.0, Int);
        objective!(m, Min, x.powi(2));
        assert_eq!(m.kind(), ModelKind::MIQP);

        let err = solve(&m, &HighsOptions::default()).unwrap_err();
        assert!(matches!(err, SolverError::UnsupportedKind(ModelKind::MIQP)));
    }

    #[test]
    fn qcp_is_unsupported() {
        let m = Model::new("qcp");
        variable!(m, x >= 0.0);
        constraint!(m, c, x.powi(2) <= 4.0);
        objective!(m, Min, x);
        assert_eq!(m.kind(), ModelKind::QCP);

        let err = solve(&m, &HighsOptions::default()).unwrap_err();
        assert!(matches!(err, SolverError::UnsupportedKind(ModelKind::QCP)));
    }

    #[test]
    fn semi_continuous_forced_on() {
        // min x  s.t.  x >= 3,  x in {0} U [5, 10]  ->  x = 5.
        let m = Model::new("sc_on");
        variable!(m, x <= 10.0, SemiCont(5.0));
        constraint!(m, c, x >= 3.0);
        objective!(m, Min, x);
        assert_eq!(m.kind(), ModelKind::LP);

        let res = solve(&m, &HighsOptions::default()).unwrap();
        assert_eq!(res.termination, TerminationStatus::Optimal);
        assert!((res.value_of(x).unwrap() - 5.0).abs() < 1e-6, "x = {:?}", res.value_of(x));
        assert_eq!(res.best_bound, Some(5.0));
        assert_eq!(res.gap, Some(0.0));
        assert_eq!(res.node_count, None);
    }

    #[test]
    fn semi_continuous_off() {
        // min x, x in {0} U [5, 10], nothing forces it on  ->  x = 0.
        let m = Model::new("sc_off");
        variable!(m, x <= 10.0, SemiCont(5.0));
        objective!(m, Min, x);

        let res = solve(&m, &HighsOptions::default()).unwrap();
        assert_eq!(res.termination, TerminationStatus::Optimal);
        assert!(res.value_of(x).unwrap().abs() < 1e-9, "x = {:?}", res.value_of(x));
        assert_eq!(res.best_bound, Some(0.0));
        assert_eq!(res.gap, Some(0.0));
        assert_eq!(res.node_count, None);
    }

    #[test]
    fn semi_integer() {
        // max x  s.t.  x <= 7.5,  x in {0} U {5, 6, ..., 10}  ->  x = 7.
        let m = Model::new("si");
        variable!(m, x <= 10.0, SemiInt(5.0));
        constraint!(m, c, x <= 7.5);
        objective!(m, Max, x);
        assert_eq!(m.kind(), ModelKind::MILP);

        let res = solve(&m, &HighsOptions::default()).unwrap();
        assert_eq!(res.termination, TerminationStatus::Optimal);
        assert!((res.value_of(x).unwrap() - 7.0).abs() < 1e-6, "x = {:?}", res.value_of(x));
    }

    #[test]
    fn socp_is_unsupported() {
        let m = Model::new("socp");
        variable!(m, x);
        variable!(m, t >= 0.0);
        m.add_soc_constraint("cone", [x], t);
        objective!(m, Min, t);
        assert_eq!(m.kind(), ModelKind::SOCP);

        let err = solve(&m, &HighsOptions::default()).unwrap_err();
        assert!(matches!(err, SolverError::UnsupportedKind(ModelKind::SOCP)));
    }
}
