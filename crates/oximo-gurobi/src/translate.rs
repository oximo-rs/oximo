use std::time::Instant;

use grb::expr::LinExpr;
use grb::prelude::*;
use oximo_core::{ConstraintId, Domain, Model, ModelKind, ObjectiveSense, Sense, VarId};
use oximo_expr::{ExprArena, LinearTerms, extract_linear};
use oximo_solver::{SolverError, SolverResult, SolverStatus};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::GurobiOptions;
use crate::options::apply as apply_options;

fn map_grb_err(e: grb::Error) -> SolverError {
    SolverError::Backend(e.to_string())
}

/// Translate `model` into a Gurobi model, solve, and return the generic
/// [`SolverResult`].
///
/// # Errors
///
/// Returns a [`SolverError`] if the model is unsupported, contains nonlinear
/// expressions, or if Gurobi reports an error during setup or optimization.
///
/// # Panics
///
/// Panics if model variable or constraint indices overflow `u32`.
#[allow(clippy::unnecessary_cast, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn solve(model: &Model, opts: &GurobiOptions) -> Result<SolverResult, SolverError> {
    let kind = model.kind();
    if !matches!(kind, ModelKind::LP | ModelKind::MILP) {
        return Err(SolverError::UnsupportedKind(kind));
    }

    let arena = model.arena();
    let vars = model.variables();
    let constraints = model.constraints();
    let objective = model.try_objective().map_err(SolverError::Core)?;

    let mut obj_coeffs = vec![0.0; vars.len()];
    let obj_constant = match extract_linear(&arena, objective.expr) {
        Some(t) => {
            for (v, c) in t.coeffs {
                obj_coeffs[v.index()] = c;
            }
            t.constant
        }
        None => return Err(SolverError::Nonlinear),
    };

    let env = Env::new("").map_err(|e| SolverError::Backend(format!("Gurobi env: {e}")))?;
    let mut grb_model = grb::Model::with_env("oximo", &env).map_err(map_grb_err)?;

    let mut gurobi_vars = Vec::with_capacity(vars.len());
    for (i, v) in vars.iter().enumerate() {
        let vtype = match v.domain {
            Domain::Real => VarType::Continuous,
            Domain::Integer => VarType::Integer,
            Domain::Binary => VarType::Binary,
            Domain::SemiContinuous { .. } => VarType::SemiCont,
            Domain::SemiInteger { .. } => VarType::SemiInt,
        };
        let gvar = add_var!(grb_model, vtype, obj: obj_coeffs[i], bounds: v.lb..v.ub, name: &format!("x{i}"))
            .map_err(map_grb_err)?;
        gurobi_vars.push(gvar);
        if let Some(val) = v.initial {
            grb_model.set_obj_attr(attr::Start, &gvar, val).map_err(map_grb_err)?;
        }
    }

    let arena_ref: &ExprArena = &arena;
    let con_terms: Vec<LinearTerms> = constraints
        .par_iter()
        .map(|c| extract_linear(arena_ref, c.lhs).ok_or(SolverError::Nonlinear))
        .collect::<Result<Vec<_>, _>>()?;

    let mut gurobi_constrs = Vec::with_capacity(constraints.len());
    for (c_id, (c, t)) in constraints.iter().zip(con_terms).enumerate() {
        let adjusted_rhs = c.rhs - t.constant;

        let mut expr = LinExpr::new();
        for (v, co) in t.coeffs {
            expr.add_term(co, gurobi_vars[v.index()]);
        }

        let name = format!("c{c_id}");
        let constr = match c.sense {
            Sense::Le => {
                grb_model.add_constr(&name, c!(expr <= adjusted_rhs)).map_err(map_grb_err)?
            }
            Sense::Ge => {
                grb_model.add_constr(&name, c!(expr >= adjusted_rhs)).map_err(map_grb_err)?
            }
            Sense::Eq => {
                grb_model.add_constr(&name, c!(expr == adjusted_rhs)).map_err(map_grb_err)?
            }
        };
        gurobi_constrs.push(constr);
    }

    grb_model
        .set_attr(
            attr::ModelSense,
            match objective.sense {
                ObjectiveSense::Minimize => 1,
                ObjectiveSense::Maximize => -1,
            },
        )
        .map_err(map_grb_err)?;

    apply_options(&mut grb_model, opts).map_err(map_grb_err)?;

    let started = Instant::now();
    grb_model.optimize().map_err(map_grb_err)?;
    let elapsed = started.elapsed();

    let status = map_status(&grb_model)?;
    let (primal, reduced_costs, dual) =
        collect_solution(&status, &grb_model, &gurobi_vars, &gurobi_constrs);

    let objective_value = grb_model.get_attr(attr::ObjVal).ok().map(|v| v + obj_constant);
    let iterations = grb_model.get_attr(attr::IterCount).unwrap_or(0.0) as u64;

    Ok(SolverResult {
        status,
        objective: objective_value,
        primal,
        dual,
        reduced_costs,
        solve_time: elapsed,
        iterations,
        raw_log: None,
    })
}

fn collect_solution(
    status: &SolverStatus,
    model: &grb::Model,
    vars: &[grb::Var],
    constrs: &[grb::Constr],
) -> (FxHashMap<VarId, f64>, FxHashMap<VarId, f64>, FxHashMap<ConstraintId, f64>) {
    let mut primal = FxHashMap::default();
    let mut reduced_costs = FxHashMap::default();
    let mut dual = FxHashMap::default();

    if status.has_solution() {
        for (i, gvar) in vars.iter().enumerate() {
            if let Ok(val) = model.get_obj_attr(attr::X, gvar) {
                primal.insert(VarId(u32::try_from(i).unwrap()), val);
            }
            // Reduced costs
            if let Ok(val) = model.get_obj_attr(attr::RC, gvar) {
                reduced_costs.insert(VarId(u32::try_from(i).unwrap()), val);
            }
        }
        for (i, c) in constrs.iter().enumerate() {
            if let Ok(val) = model.get_obj_attr(attr::Pi, c) {
                dual.insert(ConstraintId(u32::try_from(i).unwrap()), val);
            }
        }
    }

    (primal, reduced_costs, dual)
}

fn map_status(model: &grb::Model) -> Result<SolverStatus, SolverError> {
    let status = model.get_attr(attr::Status).map_err(map_grb_err)?;
    Ok(match status {
        Status::Optimal => SolverStatus::Optimal,
        Status::Infeasible => SolverStatus::Infeasible,
        Status::Unbounded | Status::InfOrUnbd => SolverStatus::Unbounded,
        Status::Numeric => SolverStatus::NumericError,
        Status::TimeLimit | Status::IterationLimit | Status::NodeLimit => SolverStatus::TimeLimit,
        Status::SubOptimal => SolverStatus::Feasible,
        Status::Loaded => SolverStatus::NotSolved,
        _ => SolverStatus::Other(format!("Status: {status:?}")),
    })
}
