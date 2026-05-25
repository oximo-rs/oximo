use std::time::Instant;

use highs::{HighsModelStatus, RowProblem, Sense as HighsSense};
use oximo_core::{ConstraintId, Model, ModelKind, ObjectiveSense, Sense, VarId};
use oximo_expr::{ExprArena, LinearTerms, extract_linear};
use oximo_solver::{SolverError, SolverResult, SolverStatus};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::HighsOptions;
use crate::options::apply as apply_options;

/// Translate `model` into a HiGHS [`RowProblem`], solve, and return the
/// generic [`SolverResult`].
///
/// For now we only support LP and MILP. Nonlinear constraints
/// or objectives produce [`SolverError::Nonlinear`].
///
/// # Errors
///
/// Returns a [`SolverError`] if the model is unsupported or if HiGHS fails.
///
/// # Panics
///
/// Panics if model variable IDs overflow `u32`.
pub fn solve(model: &Model, opts: &HighsOptions) -> Result<SolverResult, SolverError> {
    let kind = model.kind();
    if !matches!(kind, ModelKind::LP | ModelKind::MILP) {
        return Err(SolverError::UnsupportedKind(kind));
    }

    let arena = model.arena();
    let vars = model.variables();
    let constraints = model.constraints();
    let objective = model.try_objective().map_err(SolverError::Core)?;

    // Objective: extract linear coefficients per variable + constant
    let mut obj_coeffs: Vec<f64> = vec![0.0; vars.len()];
    let obj_constant = match extract_linear(&arena, objective.expr) {
        Some(t) => {
            for (v, c) in t.coeffs {
                obj_coeffs[v.index()] = c;
            }
            t.constant
        }
        None => return Err(SolverError::Nonlinear),
    };

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
        let factors: Vec<(highs::Col, f64)> =
            t.coeffs.iter().map(|(v, co)| (cols[v.index()], *co)).collect();
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
    let mut hmodel = pb.optimise(sense);
    if has_initial {
        hmodel.set_solution(Some(&init_vals), None, None, None);
    }
    apply_options(&mut hmodel, opts);
    let solved = hmodel.solve();
    let elapsed = started.elapsed();

    let status = map_status(solved.status());
    let solution = solved.get_solution();

    let mut primal: FxHashMap<VarId, f64> = FxHashMap::default();
    let mut reduced_costs: FxHashMap<VarId, f64> = FxHashMap::default();
    let mut dual: FxHashMap<ConstraintId, f64> = FxHashMap::default();

    if status.has_solution() {
        for (i, val) in solution.columns().iter().enumerate() {
            primal.insert(VarId(u32::try_from(i).unwrap()), *val);
        }
        for (i, val) in solution.dual_columns().iter().enumerate() {
            reduced_costs.insert(VarId(u32::try_from(i).unwrap()), *val);
        }
        for (i, val) in solution.dual_rows().iter().take(num_constraints).enumerate() {
            dual.insert(ConstraintId(u32::try_from(i).unwrap()), *val);
        }
    }

    let objective_value =
        if status.has_solution() { Some(solved.objective_value() + obj_constant) } else { None };

    Ok(SolverResult {
        status,
        objective: objective_value,
        primal,
        dual,
        reduced_costs,
        solve_time: elapsed,
        iterations: 0,
        raw_log: None,
    })
}

fn map_status(s: HighsModelStatus) -> SolverStatus {
    match s {
        HighsModelStatus::Optimal => SolverStatus::Optimal,
        HighsModelStatus::Infeasible => SolverStatus::Infeasible,
        HighsModelStatus::UnboundedOrInfeasible | HighsModelStatus::Unbounded => {
            SolverStatus::Unbounded
        }
        HighsModelStatus::ReachedTimeLimit | HighsModelStatus::ReachedIterationLimit => {
            SolverStatus::TimeLimit
        }
        HighsModelStatus::ModelEmpty => SolverStatus::Other("model_empty".into()),
        HighsModelStatus::NotSet | HighsModelStatus::Unknown => SolverStatus::NotSolved,
        HighsModelStatus::ObjectiveBound | HighsModelStatus::ObjectiveTarget => {
            SolverStatus::Feasible
        }
        HighsModelStatus::LoadError
        | HighsModelStatus::ModelError
        | HighsModelStatus::PresolveError
        | HighsModelStatus::SolveError
        | HighsModelStatus::PostsolveError => SolverStatus::NumericError,
        _ => SolverStatus::Other("unknown_highs_status".into()),
    }
}
