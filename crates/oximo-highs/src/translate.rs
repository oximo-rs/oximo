use std::time::Instant;

use highs::{
    HessianFormat, HighsModelStatus, HighsSolutionStatus, RowProblem, Sense as HighsSense,
};
use oximo_core::{ConstraintId, Model, ModelKind, ObjectiveSense, Sense, VarId, Variable};
use oximo_expr::{
    ExprArena, ExprId, LinearTerms, QuadraticTerms, extract_linear, extract_quadratic,
};
use oximo_solver::{PrimalStatus, SolutionPoint, SolverError, SolverResult, TerminationStatus};
use rayon::prelude::*;
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
/// [`SolverError::UnsupportedKind`]. Semicontinuous / semi-integer variables are
/// rejected with [`SolverError::Backend`] (the `highs` crate exposes no way to
/// set them).
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
#[allow(clippy::too_many_lines)]
pub fn solve(model: &Model, opts: &HighsOptions) -> Result<SolverResult, SolverError> {
    let kind = model.kind();
    if !matches!(kind, ModelKind::LP | ModelKind::MILP | ModelKind::QP) {
        return Err(SolverError::UnsupportedKind(kind));
    }

    let arena = model.arena();
    let vars = model.variables();
    reject_semi_domains(&vars)?;
    let constraints = model.constraints();
    let objective = model.try_objective().map_err(SolverError::Core)?;

    let (obj_coeffs, obj_constant, hessian_cols) =
        objective_terms(kind, &arena, objective.expr, vars.len())?;
    let has_hessian = hessian_cols.iter().any(|col| !col.is_empty());

    // Build the HiGHS row problem
    let mut pb = RowProblem::new();
    let mut cols: Vec<highs::Col> = Vec::with_capacity(vars.len());
    let mut has_initial = false;
    let mut init_vals: Vec<f64> = vec![0.0; vars.len()];
    for (i, v) in vars.iter().enumerate() {
        let bounds = v.lb..=v.ub;
        let coef = obj_coeffs[v.id.index()];
        let col = if v.domain.is_integer() {
            pb.add_integer_column(coef, bounds)
        } else {
            pb.add_column(coef, bounds)
        };
        cols.push(col);
        if let Some(val) = v.initial {
            init_vals[i] = val;
            has_initial = true;
        }
    }

    let arena_ref: &ExprArena = &arena;
    let con_terms: Vec<LinearTerms> = constraints
        .par_iter()
        .map(|c| extract_linear(arena_ref, c.lhs).ok_or(SolverError::Nonlinear))
        .collect::<Result<Vec<_>, _>>()?;

    for (c, t) in constraints.iter().zip(con_terms) {
        let adjusted_rhs = c.rhs - t.constant;
        let factors = t.coeffs.iter().map(|(v, co)| (cols[v.index()], *co));
        match c.sense {
            Sense::Le => pb.add_row(f64::NEG_INFINITY..=adjusted_rhs, factors),
            Sense::Ge => pb.add_row(adjusted_rhs..=f64::INFINITY, factors),
            Sense::Eq => pb.add_row(adjusted_rhs..=adjusted_rhs, factors),
        }
    }

    // The arena / vars / constraints borrows are released before we move into
    // HiGHS land, `pb` already owns the matrix data.
    drop(arena);
    drop(vars);
    let num_constraints = constraints.len();
    drop(constraints);

    let sense = match objective.sense {
        ObjectiveSense::Minimize => HighsSense::Minimise,
        ObjectiveSense::Maximize => HighsSense::Maximise,
    };

    let started = Instant::now();
    let mut hmodel = pb
        .try_optimise(sense)
        .map_err(|e| SolverError::Backend(format!("HiGHS model setup failed: {e:?}")))?;
    if has_hessian {
        // QP: pass Q for the `c'x + 0.5 x'Q x` objective. Lower triangle only.
        hmodel
            .try_pass_hessian(
                HessianFormat::Triangular,
                hessian_cols.iter().map(|col| col.iter().copied()),
            )
            .map_err(|e| SolverError::Backend(format!("HiGHS Hessian upload failed: {e}")))?;
    }
    if has_initial {
        hmodel
            .try_set_solution(Some(&init_vals), None, None, None)
            .map_err(|e| SolverError::Backend(format!("HiGHS initial solution failed: {e:?}")))?;
    }
    apply_options(&mut hmodel, opts)?;
    let solved = hmodel
        .try_solve()
        .map_err(|e| SolverError::Backend(format!("HiGHS solve failed: {e:?}")))?;
    let elapsed = started.elapsed();

    let termination = map_status(solved.status());
    let has_point = solved.primal_solution_status() == HighsSolutionStatus::Feasible;
    let solution = solved.get_solution();
    let (primal, reduced_costs, dual) = collect_solution(
        has_point,
        solution.columns(),
        solution.dual_columns(),
        solution.dual_rows(),
        num_constraints,
    );

    let objective_value =
        if has_point { Some(solved.objective_value() + obj_constant) } else { None };

    let solutions = if has_point {
        vec![SolutionPoint { primal, objective: objective_value }]
    } else {
        Vec::new()
    };
    let primal_status = PrimalStatus::infer(&termination, has_point);
    let raw_gap = solved.mip_gap();
    let gap = raw_gap.is_finite().then_some(raw_gap);
    let best_bound = solved.double_info_value(c"mip_dual_bound").ok().filter(|b| b.is_finite());
    Ok(SolverResult {
        termination,
        primal_status,
        solutions,
        dual,
        reduced_costs,
        best_bound,
        gap,
        solve_time: elapsed,
        iterations: total_iterations(&solved),
        raw_log: None,
        solver_name: Some(crate::NAME.into()),
    })
}

/// HiGHS supports semicontinuous/semi-integer variables, but the `highs` crate
/// only exposes continuous/integer integrality, so we cannot mark them. Reject
/// such a model.
fn reject_semi_domains(vars: &[Variable]) -> Result<(), SolverError> {
    for v in vars {
        if v.domain.semi_threshold().is_some() {
            return Err(SolverError::Backend(format!(
                "variable x{} has a semicontinuous/semi-integer domain, \
                 which the HiGHS backend does not support yet",
                v.id.index()
            )));
        }
    }
    Ok(())
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
    arena: &ExprArena,
    obj_expr: ExprId,
    num_vars: usize,
) -> Result<ObjectiveTerms, SolverError> {
    let mut coeffs = vec![0.0; num_vars];
    if matches!(kind, ModelKind::QP) {
        let quad = extract_quadratic(arena, obj_expr).ok_or(SolverError::Nonlinear)?;
        for (v, c) in &quad.linear {
            coeffs[v.index()] = *c;
        }
        let cols = hessian_columns(&quad, num_vars);
        Ok((coeffs, quad.constant, cols))
    } else {
        let lin = extract_linear(arena, obj_expr).ok_or(SolverError::Nonlinear)?;
        for (v, c) in &lin.coeffs {
            coeffs[v.index()] = *c;
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

    // Below this, rayon's HashMap collect overhead exceeds the gain.
    // TODO: benchmark and tune this threshold.
    const PAR_THRESHOLD: usize = 8192;
    if cols.len() + dcols.len() + drows.len() < PAR_THRESHOLD {
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
        return (primal, reduced_costs, dual);
    }

    let primal: FxHashMap<VarId, f64> =
        cols.par_iter().enumerate().map(|(i, v)| (VarId(u32::try_from(i).unwrap()), *v)).collect();
    let reduced_costs: FxHashMap<VarId, f64> =
        dcols.par_iter().enumerate().map(|(i, v)| (VarId(u32::try_from(i).unwrap()), *v)).collect();
    let dual: FxHashMap<ConstraintId, f64> = drows
        .par_iter()
        .enumerate()
        .map(|(i, v)| (ConstraintId(u32::try_from(i).unwrap()), *v))
        .collect();
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
            TerminationStatus::Interrupted
        }
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
    fn miqp_is_unsupported() {
        // Integer variable + quadratic objective = MIQP, which HiGHS cannot solve.
        let m = Model::new("miqp");
        variable!(m, 0.0 <= x <= 5.0, Int);
        objective!(m, Min, x.powi(2));
        assert_eq!(m.kind(), ModelKind::MIQP);

        let err = solve(&m, &HighsOptions::default()).unwrap_err();
        assert!(matches!(err, SolverError::UnsupportedKind(ModelKind::MIQP)));
    }
}
