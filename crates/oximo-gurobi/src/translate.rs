use std::time::Instant;

use grb::expr::{LinExpr, QuadExpr};
use grb::prelude::*;
use oximo_core::{
    Constraint, ConstraintId, Domain, Model, ModelKind, ObjectiveSense, Sense, SocConstraint,
    SocConstraintId, VarId, Variable,
};
use oximo_expr::{ExprArena, ExprId, LinearTerms, extract_linear};
use oximo_solver::{PrimalStatus, SolutionPoint, SolverError, SolverResult, TerminationStatus};
use rustc_hash::FxHashMap;

use crate::GurobiOptions;
use crate::nonlinear::{LoweredExpr, LoweringCtx, lower};
use crate::options::apply as apply_options;

pub(crate) fn map_grb_err(e: grb::Error) -> SolverError {
    SolverError::Backend(e.to_string())
}

/// Translate `model` into a Gurobi model, solve, and return the generic
/// [`SolverResult`].
///
/// # Errors
///
/// Returns a [`SolverError`] if the model is unsupported, contains nonlinear
/// expressions Gurobi cannot represent, or if Gurobi reports an error during
/// setup or optimization.
///
/// # Panics
///
/// Panics if model variable or constraint indices overflow `u32`.
pub fn solve(model: &Model, opts: &GurobiOptions) -> Result<SolverResult, SolverError> {
    let kind = model.kind();
    let env = default_env()?;
    let mut built = build(model, opts, &env)?;
    run_and_collect(&mut built, kind)
}

/// Create the default Gurobi [`Env`].
pub(crate) fn default_env() -> Result<Env, SolverError> {
    Env::new("").map_err(|e| SolverError::Backend(format!("Gurobi env: {e}")))
}

/// A built Gurobi model plus the handles needed to read its solution and to drive
/// incremental re-solves.
pub(crate) struct Built {
    pub model: grb::Model,
    pub vars: Vec<grb::Var>,
    pub constrs: Vec<GrbRow>,
    pub soc_rows: Vec<(grb::QConstr, LinearTerms)>,
    pub obj_constant: f64,
    pub has_semi: bool,
}

/// Translate `model` into a configured-but-unsolved Gurobi model.
///
/// # Errors
///
/// Returns a [`SolverError`] if the model contains nonlinear expressions Gurobi
/// cannot represent or Gurobi reports an error during setup.
pub(crate) fn build(model: &Model, opts: &GurobiOptions, env: &Env) -> Result<Built, SolverError> {
    let kind = model.kind();
    let nonlinear_kind = matches!(
        kind,
        ModelKind::QP
            | ModelKind::MIQP
            | ModelKind::QCP
            | ModelKind::MIQCP
            | ModelKind::NLP
            | ModelKind::MINLP
    );

    let arena = model.arena();
    let vars = model.variables();
    let constraints = model.constraints();
    let socs = model.soc_constraints();
    let objective = model.objective();
    let has_semi = vars.iter().any(|v| v.domain.semi_threshold().is_some());

    let mut grb_model = grb::Model::with_env("oximo", env).map_err(map_grb_err)?;

    let gurobi_vars = add_variables(&mut grb_model, &vars)?;

    // Aux-variable counter shared by the constraint and objective lowering so the
    // synthetic variable names stay unique across both.
    let mut aux_counter = 0_u32;
    let gurobi_constrs =
        add_constraints(&arena, &constraints, &mut grb_model, &gurobi_vars, &mut aux_counter)?;
    let soc_rows = add_soc_rows(&arena, &socs, &mut grb_model, &gurobi_vars)?;

    let obj_constant = match objective.as_ref() {
        Some(o) => {
            set_objective(&arena, o.expr, o.sense, &mut grb_model, &gurobi_vars, &mut aux_counter)?
        }
        None => 0.0,
    };

    apply_options(&mut grb_model, opts).map_err(map_grb_err)?;
    if nonlinear_kind {
        // Gurobi requires NonConvex=2 for general nonlinear constraints and
        // bilinear non-convex objectives. Skip if the user already set it.
        let current = grb_model.get_param(grb::param::NonConvex).map_err(map_grb_err)?;
        if current < 2 {
            grb_model.set_param(grb::param::NonConvex, 2).map_err(map_grb_err)?;
        }
    }

    Ok(Built {
        model: grb_model,
        vars: gurobi_vars,
        constrs: gurobi_constrs,
        soc_rows,
        obj_constant,
        has_semi,
    })
}

/// Optimize a built model and assemble the generic [`SolverResult`]. Shared by the
/// one-shot [`solve`] and the persistent handle's re-solve.
///
/// # Errors
///
/// Returns a [`SolverError`] if Gurobi reports an error during optimization.
pub(crate) fn run_and_collect(
    built: &mut Built,
    kind: ModelKind,
) -> Result<SolverResult, SolverError> {
    let started = Instant::now();
    built.model.optimize().map_err(map_grb_err)?;
    let elapsed = started.elapsed();

    let termination = map_status(&built.model)?;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let iterations = built.model.get_attr(attr::IterCount).unwrap_or(0.0) as u64;
    let (solutions, reduced_costs, dual, soc_dual) = collect_solution(
        kind,
        &mut built.model,
        &built.vars,
        &built.constrs,
        &built.soc_rows,
        built.obj_constant,
    );

    let primal_status = PrimalStatus::infer(&termination, !solutions.is_empty());
    let best_bound = built.model.get_attr(attr::ObjBound).ok().filter(|b| b.is_finite());
    let gap = built.model.get_attr(attr::MIPGap).ok().filter(|g| g.is_finite());

    Ok(SolverResult {
        termination,
        primal_status,
        solutions,
        dual,
        soc_dual,
        reduced_costs,
        best_bound,
        gap,
        solve_time: elapsed,
        iterations,
        raw_log: None,
        solver_name: Some(crate::NAME.into()),
    })
}

/// Add one Gurobi variable per model variable, in `VarId` order, applying its
/// domain, bounds, and any warm-start value.
fn add_variables(
    grb_model: &mut grb::Model,
    vars: &[Variable],
) -> Result<Vec<grb::Var>, SolverError> {
    let mut gurobi_vars = Vec::with_capacity(vars.len());
    for (i, v) in vars.iter().enumerate() {
        let vtype = match v.domain {
            Domain::Real => VarType::Continuous,
            Domain::Integer => VarType::Integer,
            Domain::Binary => VarType::Binary,
            Domain::SemiContinuous { .. } => VarType::SemiCont,
            Domain::SemiInteger { .. } => VarType::SemiInt,
        };
        // For a semi-continuous/semi-integer variable the gap floor (`threshold`)
        // is Gurobi's lower bound: the value is 0 or in `[lb, ub]`.
        let floor = v.domain.semi_threshold().unwrap_or(v.lb);
        // `add_var!` expands the f64 bounds with an `as f64` cast.
        #[allow(clippy::unnecessary_cast)]
        let gvar = add_var!(grb_model, vtype, bounds: floor..v.ub, name: &format!("x{i}"))
            .map_err(map_grb_err)?;
        gurobi_vars.push(gvar);
        if let Some(val) = v.initial {
            grb_model.set_obj_attr(attr::Start, &gvar, val).map_err(map_grb_err)?;
        }
    }
    Ok(gurobi_vars)
}

/// Gurobi's handle for an added constraint, kept so its dual can be queried
/// after the solve (`Pi` for linear rows, `QCPi` for quadratic rows).
pub(crate) enum GrbRow {
    Lin(grb::Constr),
    Quad(grb::QConstr),
}

/// Add every model constraint, returning the per-constraint row handle.
/// Each constraint takes the linear fast path when its LHS is linear
/// and falls back to the general lowering otherwise.
fn add_constraints(
    arena: &ExprArena,
    constraints: &[Constraint],
    grb_model: &mut grb::Model,
    gurobi_vars: &[grb::Var],
    aux_counter: &mut u32,
) -> Result<Vec<GrbRow>, SolverError> {
    // One `GrbRow` per oximo constraint keeps `gurobi_constrs` 1:1 with
    // `ConstraintId`, which `collect_solution` relies on to key duals. A
    // two-sided range becomes a single native `add_range` row.
    let mut gurobi_constrs: Vec<GrbRow> = Vec::with_capacity(constraints.len());
    for (c_id, c) in constraints.iter().enumerate() {
        let name = format!("c{c_id}");
        if let Some((sense, rhs)) = c.as_single() {
            if let Some(t) = extract_linear(arena, c.lhs) {
                let adjusted_rhs = rhs - t.constant;
                let mut expr = LinExpr::new();
                for (v, co) in t.coeffs {
                    expr.add_term(co, gurobi_vars[v.index()]);
                }
                let constr = match sense {
                    Sense::Le => grb_model.add_constr(&name, c!(expr <= adjusted_rhs)),
                    Sense::Ge => grb_model.add_constr(&name, c!(expr >= adjusted_rhs)),
                    Sense::Eq => grb_model.add_constr(&name, c!(expr == adjusted_rhs)),
                }
                .map_err(map_grb_err)?;
                gurobi_constrs.push(GrbRow::Lin(constr));
            } else {
                let row = add_nonlinear_constraint(
                    arena,
                    c.lhs,
                    sense,
                    rhs,
                    c_id,
                    grb_model,
                    gurobi_vars,
                    aux_counter,
                )?;
                gurobi_constrs.push(row);
            }
        } else {
            // Genuine two-sided range. Gurobi's `add_range` is linear-only.
            let Some(t) = extract_linear(arena, c.lhs) else {
                return Err(SolverError::Backend(format!(
                    "Gurobi does not support a two-sided range on a nonlinear constraint ('{}')",
                    c.name
                )));
            };
            let lower = c.lower - t.constant;
            let upper = c.upper - t.constant;
            let mut expr = LinExpr::new();
            for (v, co) in t.coeffs {
                expr.add_term(co, gurobi_vars[v.index()]);
            }
            #[allow(clippy::unnecessary_cast)]
            let (_slack, constr) =
                grb_model.add_range(&name, c!(expr in lower..upper)).map_err(map_grb_err)?;
            gurobi_constrs.push(GrbRow::Lin(constr));
        }
    }
    Ok(gurobi_constrs)
}

/// Lower each explicit SOC constraint `||terms||_2 <= bound` to the quadratic
/// row `sum(term_i^2) - bound^2 <= 0` plus the linear side condition
/// `bound >= 0`.
///
/// Returns each cone's quadratic-row handle plus its affine bound side, in
/// `SocConstraintId` order, so `collect_solution` can rescale the squared-form
/// `QCPi` multiplier back to the norm form.
fn add_soc_rows(
    arena: &ExprArena,
    socs: &[SocConstraint],
    grb_model: &mut grb::Model,
    gurobi_vars: &[grb::Var],
) -> Result<Vec<(grb::QConstr, LinearTerms)>, SolverError> {
    let mut rows = Vec::with_capacity(socs.len());
    for (i, s) in socs.iter().enumerate() {
        let mut q = QuadExpr::new();
        for &term in &s.terms {
            let t = extract_linear(arena, term).ok_or(SolverError::Nonlinear)?;
            add_squared_affine(&mut q, &t, 1.0, gurobi_vars);
        }
        let b = extract_linear(arena, s.bound).ok_or(SolverError::Nonlinear)?;
        add_squared_affine(&mut q, &b, -1.0, gurobi_vars);
        let qrow = grb_model.add_qconstr(&format!("soc{i}"), c!(q <= 0.0)).map_err(map_grb_err)?;

        let mut e = LinExpr::new();
        for &(v, co) in &b.coeffs {
            e.add_term(co, gurobi_vars[v.index()]);
        }
        grb_model.add_constr(&format!("soc{i}_sign"), c!(e >= -b.constant)).map_err(map_grb_err)?;
        rows.push((qrow, b));
    }
    Ok(rows)
}

/// Expand `sign * (a'x + c)^2` into `q`.
fn add_squared_affine(q: &mut QuadExpr, t: &LinearTerms, sign: f64, gurobi_vars: &[grb::Var]) {
    for (i, &(vi, ci)) in t.coeffs.iter().enumerate() {
        q.add_qterm(sign * ci * ci, gurobi_vars[vi.index()], gurobi_vars[vi.index()]);
        for &(vj, cj) in &t.coeffs[i + 1..] {
            q.add_qterm(sign * 2.0 * ci * cj, gurobi_vars[vi.index()], gurobi_vars[vj.index()]);
        }
        if t.constant != 0.0 {
            q.add_term(sign * 2.0 * ci * t.constant, gurobi_vars[vi.index()]);
        }
    }
    if t.constant != 0.0 {
        q.add_constant(sign * t.constant * t.constant);
    }
}

#[allow(clippy::too_many_arguments)]
fn add_nonlinear_constraint(
    arena: &ExprArena,
    lhs: ExprId,
    sense: Sense,
    rhs: f64,
    c_id: usize,
    grb_model: &mut grb::Model,
    gurobi_vars: &[grb::Var],
    aux_counter: &mut u32,
) -> Result<GrbRow, SolverError> {
    let mut ctx = LoweringCtx { model: grb_model, gurobi_vars, aux_counter: *aux_counter };
    let lowered = lower(arena, lhs, &mut ctx)?;
    *aux_counter = ctx.aux_counter;
    let name = format!("c{c_id}");
    let row = match lowered {
        LoweredExpr::Linear(e) => GrbRow::Lin(
            match sense {
                Sense::Le => grb_model.add_constr(&name, c!(e <= rhs)),
                Sense::Ge => grb_model.add_constr(&name, c!(e >= rhs)),
                Sense::Eq => grb_model.add_constr(&name, c!(e == rhs)),
            }
            .map_err(map_grb_err)?,
        ),
        LoweredExpr::Quadratic(e) => GrbRow::Quad(
            match sense {
                Sense::Le => grb_model.add_qconstr(&name, c!(e <= rhs)),
                Sense::Ge => grb_model.add_qconstr(&name, c!(e >= rhs)),
                Sense::Eq => grb_model.add_qconstr(&name, c!(e == rhs)),
            }
            .map_err(map_grb_err)?,
        ),
        LoweredExpr::Var(v) => GrbRow::Lin(
            match sense {
                Sense::Le => grb_model.add_constr(&name, c!(v <= rhs)),
                Sense::Ge => grb_model.add_constr(&name, c!(v >= rhs)),
                Sense::Eq => grb_model.add_constr(&name, c!(v == rhs)),
            }
            .map_err(map_grb_err)?,
        ),
    };
    Ok(row)
}

fn set_objective(
    arena: &ExprArena,
    obj_expr: ExprId,
    sense: ObjectiveSense,
    grb_model: &mut grb::Model,
    gurobi_vars: &[grb::Var],
    aux_counter: &mut u32,
) -> Result<f64, SolverError> {
    let grb_sense = match sense {
        ObjectiveSense::Minimize => ModelSense::Minimize,
        ObjectiveSense::Maximize => ModelSense::Maximize,
    };
    if let Some(t) = extract_linear(arena, obj_expr) {
        let mut e = LinExpr::new();
        for (v, c) in t.coeffs {
            e.add_term(c, gurobi_vars[v.index()]);
        }
        // Gurobi's set_objective absorbs LinExpr offsets into ObjCon, so we do
        // not need to track the constant separately.
        e.add_constant(t.constant);
        grb_model.set_objective(e, grb_sense).map_err(map_grb_err)?;
        return Ok(0.0);
    }
    let mut ctx = LoweringCtx { model: grb_model, gurobi_vars, aux_counter: *aux_counter };
    let lowered = lower(arena, obj_expr, &mut ctx)?;
    *aux_counter = ctx.aux_counter;
    grb_model.set_objective(lowered.into_expr_for_objective(), grb_sense).map_err(map_grb_err)?;
    Ok(0.0)
}

/// Build `(solutions, reduced_costs, dual, soc_dual)` from a solved Gurobi
/// model.
///
/// `solutions` holds every point in Gurobi's solution pool, best first (index 0
/// is the incumbent). The pool is populated automatically during a MIP solve,
/// set `PoolSearchMode`/`PoolSolutions` (via [`crate::GurobiOptions`]) to force
/// Gurobi to enumerate alternative optima. Duals and reduced costs are returned
/// for continuous models (`LP` and `QP`). For quadratically constrained models
/// (`QCP`/`SOCP`, including the lowered SOC rows) Gurobi computes duals only
/// with `QCPDual=1`.
/// `(solutions, reduced_costs, dual, soc_dual)` bundle read from a solved model.
type Collected = (
    Vec<SolutionPoint>,
    FxHashMap<VarId, f64>,
    FxHashMap<ConstraintId, f64>,
    FxHashMap<SocConstraintId, f64>,
);

fn collect_solution(
    kind: ModelKind,
    model: &mut grb::Model,
    vars: &[grb::Var],
    constrs: &[GrbRow],
    soc_rows: &[(grb::QConstr, LinearTerms)],
    obj_constant: f64,
) -> Collected {
    // A primal point exists only when Gurobi actually stored one. `SolCount` is
    // the number of available solutions and it stays `> 0` for an incumbent
    // kept at a time/iteration/node limit.
    let sol_count = model.get_attr(attr::SolCount).unwrap_or(0);
    if sol_count <= 0 {
        return (Vec::new(), FxHashMap::default(), FxHashMap::default(), FxHashMap::default());
    }

    let solutions = collect_pool(model, vars, obj_constant, sol_count);

    // Skip retrieval of duals and reduced costs for integer model classes,
    // where Gurobi refuses the attributes.
    // LP/QP always have duals, QCP/SOCP rows only when the user opted into
    // `QCPDual=1`.
    if !matches!(kind, ModelKind::LP | ModelKind::QP | ModelKind::QCP | ModelKind::SOCP) {
        return (solutions, FxHashMap::default(), FxHashMap::default(), FxHashMap::default());
    }

    let reduced_costs = model
        .get_obj_attr_batch(attr::RC, vars.iter().copied())
        .map(|v| index_map(&v))
        .unwrap_or_default();

    let mut dual = FxHashMap::default();
    for (i, row) in constrs.iter().enumerate() {
        let pi = match row {
            GrbRow::Lin(c) => model.get_obj_attr(attr::Pi, c),
            GrbRow::Quad(q) => model.get_obj_attr(attr::QCPi, q),
        };
        if let Ok(pi) = pi {
            dual.insert(ConstraintId(u32::try_from(i).unwrap()), pi);
        }
    }

    // Convert the squared-form multiplier back to the norm-form
    // bound multiplier `z0 = 2 * bound_value * |QCPi|`.
    // Available only under `QCPDual=1`.
    let mut soc_dual = FxHashMap::default();
    let primal = &solutions[0].primal;
    for (i, (qrow, bound)) in soc_rows.iter().enumerate() {
        if let Ok(pi) = model.get_obj_attr(attr::QCPi, qrow) {
            let b_val = bound.constant
                + bound
                    .coeffs
                    .iter()
                    .map(|&(v, c)| c * primal.get(&v).copied().unwrap_or(0.0))
                    .sum::<f64>();
            soc_dual.insert(SocConstraintId(u32::try_from(i).unwrap()), 2.0 * b_val * pi.abs());
        }
    }

    (solutions, reduced_costs, dual, soc_dual)
}

/// Collect the `n` pooled primal points, best first.
fn collect_pool(
    model: &mut grb::Model,
    vars: &[grb::Var],
    obj_constant: f64,
    n: i32,
) -> Vec<SolutionPoint> {
    // Single point (and the only path for LP/continuous models):
    // read the incumbent directly via `X` / `ObjVal`.
    if n <= 1 {
        let primal = model
            .get_obj_attr_batch(attr::X, vars.iter().copied())
            .map(|v| index_map(&v))
            .unwrap_or_default();
        let objective = model.get_attr(attr::ObjVal).ok().map(|v| v + obj_constant);
        return vec![SolutionPoint { primal, objective }];
    }

    // A MIP solution pool. Gurobi sorts it best-first.
    // `SolutionNumber` selects which one `Xn` / `PoolObjVal` report.
    let mut out = Vec::with_capacity(usize::try_from(n).unwrap_or(0));
    for k in 0..n {
        if model.set_param(grb::param::SolutionNumber, k).is_err() {
            break;
        }
        let Ok(vals) = model.get_obj_attr_batch(attr::Xn, vars.iter().copied()) else {
            break;
        };
        let objective = model.get_attr(attr::PoolObjVal).ok().map(|v| v + obj_constant);
        out.push(SolutionPoint { primal: index_map(&vals), objective });
    }
    out
}

/// Map a dense per-variable value array (in `VarId` order) to a sparse map.
fn index_map(vals: &[f64]) -> FxHashMap<VarId, f64> {
    vals.iter().enumerate().map(|(i, &val)| (VarId(u32::try_from(i).unwrap()), val)).collect()
}

fn map_status(model: &grb::Model) -> Result<TerminationStatus, SolverError> {
    let status = model.get_attr(attr::Status).map_err(map_grb_err)?;
    Ok(match status {
        Status::Optimal => TerminationStatus::Optimal,
        Status::Infeasible => TerminationStatus::Infeasible,
        Status::Unbounded => TerminationStatus::Unbounded,
        Status::InfOrUnbd => TerminationStatus::InfeasibleOrUnbounded,
        Status::Numeric => TerminationStatus::NumericError,
        Status::TimeLimit => TerminationStatus::TimeLimit,
        Status::IterationLimit => TerminationStatus::IterationLimit,
        Status::NodeLimit => TerminationStatus::NodeLimit,
        // Gurobi could not meet optimality tolerances but holds a usable point.
        Status::SubOptimal => TerminationStatus::Interrupted,
        Status::Loaded => TerminationStatus::NotSolved,
        _ => TerminationStatus::Other(format!("Status: {status:?}")),
    })
}
