use oximo_solver::prepare::{LoweringContext, PreparedExpressions};
use oximo_solver::reconstruct::{ObjectiveTransform, normalize_result};
use std::time::Instant;

use gurobi_rs::constr::RangeExpr;
use gurobi_rs::expr::{LinExpr, QuadExpr};
use gurobi_rs::prelude::*;
use oximo_core::{
    Constraint, ConstraintId, Domain, Model, ModelKind, ObjectiveSense, Sense, SocConstraint,
    SocConstraintId, SosConstraint, SosConstraintId, SosType, VarId, Variable, var_name,
};
use oximo_expr::{ExprArena, ExprId, LinearTerms, describe_nonlinear_term};
use oximo_solver::{
    DualStatus, Iis, PrimalStatus, SolutionPoint, SolverError, SolverResult, TerminationStatus,
    VarBoundKind,
};
use rustc_hash::FxHashMap;

use crate::GurobiOptions;
use crate::nonlinear::{GeneratedConstraint, LoweredExpr, LoweringCtx, lower, validate_supported};
use crate::options::apply as apply_options;

pub(crate) fn map_gurobi_err(e: gurobi_rs::Error) -> SolverError {
    SolverError::Backend(format!("Gurobi: {e}"))
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
    validate_model_operators(model)?;
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
    pub model: gurobi_rs::Model,
    pub vars: Vec<gurobi_rs::Var>,
    pub constrs: Vec<ConstraintHandle>,
    pub objective_generated: Vec<GeneratedConstraint>,
    pub soc_rows: Vec<SocHandle>,
    pub sos: Vec<(SosConstraintId, gurobi_rs::SOS)>,
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
    validate_model_operators(model)?;
    let prepared = LoweringContext::new(model)?;
    let kind = prepared.kind();
    let nonlinear_kind = matches!(
        kind,
        ModelKind::QP
            | ModelKind::MIQP
            | ModelKind::QCP
            | ModelKind::MIQCP
            | ModelKind::NLP
            | ModelKind::MINLP
    );

    let vars = prepared.variables();
    let model_constraints = prepared.constraints();
    let constraints = model_constraints.algebraic();
    let socs = prepared.constraints().second_order_cones();
    let sos = prepared.constraints().special_ordered_sets();
    let objective = prepared.objective();
    let has_semi = vars.iter().any(|v| v.domain.semi_threshold().is_some());

    let mut gurobi_model = gurobi_rs::Model::with_env("oximo", env).map_err(map_gurobi_err)?;

    let gurobi_vars = add_variables(&mut gurobi_model, vars)?;

    // Aux-variable counter shared by the constraint and objective lowering so the
    // synthetic variable names stay unique across both.
    let mut aux_counter = 0_u32;
    let gurobi_constrs =
        add_constraints(&prepared, constraints, &mut gurobi_model, &gurobi_vars, &mut aux_counter)?;
    let soc_rows = add_soc_rows(&prepared, vars, socs, &mut gurobi_model, &gurobi_vars)?;
    let special_ordered_sets = add_sos_constraints(sos, &mut gurobi_model, &gurobi_vars)?;

    let (obj_constant, objective_generated) = match objective.as_ref() {
        Some(o) => set_objective(
            &prepared,
            o.expr,
            o.sense,
            &mut gurobi_model,
            &gurobi_vars,
            &mut aux_counter,
        )?,
        None => (0.0, Vec::new()),
    };

    // Warm starts are assigned only after the full structural model has been
    // built. This keeps the batch variable construction and the start vector
    // compatible with Gurobi's pending-update rules.
    apply_initial_values(&gurobi_model, vars, &gurobi_vars)?;

    apply_options(&mut gurobi_model, opts).map_err(map_gurobi_err)?;
    if nonlinear_kind && !opts.has_non_convex() {
        // Gurobi requires NonConvex=2 for general nonlinear constraints and
        // bilinear non-convex objectives. Preserve an explicit user setting.
        gurobi_model.set_param(gurobi_rs::param::NonConvex, 2).map_err(map_gurobi_err)?;
    }

    Ok(Built {
        model: gurobi_model,
        vars: gurobi_vars,
        constrs: gurobi_constrs,
        objective_generated,
        soc_rows,
        sos: special_ordered_sets,
        obj_constant,
        has_semi,
    })
}

fn validate_model_operators(model: &Model) -> Result<(), SolverError> {
    let arena = model.arena();
    let mut roots = Vec::new();
    if let Some(objective) = model.objective().as_ref() {
        roots.push(objective.expr);
    }
    roots.extend(
        model
            .constraints()
            .algebraic()
            .iter()
            .filter(|constraint| constraint.active)
            .map(|constraint| constraint.lhs),
    );
    validate_supported(&arena, roots)
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
    built.model.optimize().map_err(map_gurobi_err)?;
    collect_after_optimize(built, kind, started.elapsed())
}

/// Optimize with a user callback and collect results through the same path as
/// an ordinary solve. Callback failures are surfaced as [`SolverError`].
pub(crate) fn run_and_collect_with_callback<F>(
    built: &mut Built,
    kind: ModelKind,
    callback: &mut F,
    mask: Option<gurobi_rs::callback::CallbackMask>,
) -> Result<SolverResult, SolverError>
where
    F: gurobi_rs::callback::Callback,
{
    let started = Instant::now();
    match mask {
        Some(mask) => built.model.optimize_with_callback_filtered(callback, mask),
        None => built.model.optimize_with_callback(callback),
    }
    .map_err(map_gurobi_err)?;
    collect_after_optimize(built, kind, started.elapsed())
}

fn collect_after_optimize(
    built: &mut Built,
    kind: ModelKind,
    elapsed: std::time::Duration,
) -> Result<SolverResult, SolverError> {
    let native_status = built.model.get_attr(attr::Status).map_err(map_gurobi_err)?;
    let termination = map_status(native_status);
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let iterations = built.model.get_attr(attr::IterCount).unwrap_or(0.0) as u64;
    let (solutions, reduced_costs, dual, soc_dual, dual_available) = collect_solution(
        kind,
        &mut built.model,
        &built.vars,
        &built.constrs,
        &built.soc_rows,
        built.obj_constant,
    );

    let best_bound = built.model.get_attr(attr::ObjBound).ok().filter(|b| b.is_finite());
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let node_count = matches!(
        kind,
        ModelKind::MILP | ModelKind::MIQP | ModelKind::MIQCP | ModelKind::MISOCP | ModelKind::MINLP
    )
    .then(|| built.model.get_attr(attr::NodeCount).ok())
    .flatten()
    .filter(|count| count.is_finite() && *count >= 0.0)
    .map(|count| count as u64);
    let (major, minor, technical) = gurobi_rs::version();

    Ok(normalize_result(
        SolverResult {
            termination,
            primal_status: PrimalStatus::NoSolution,
            dual_status: collected_dual_status(kind, dual_available, !solutions.is_empty()),
            solutions,
            dual,
            soc_dual,
            reduced_costs,
            best_bound,
            gap: built.model.get_attr(attr::MIPGap).ok(),
            solve_time: elapsed,
            iterations,
            node_count,
            raw_status: Some(format!("{native_status:?}").into()),
            raw_log: None,
            solver_name: Some(crate::NAME.into()),
            solver_version: Some(format!("{major}.{minor}.{technical}").into()),
        },
        built.vars.len(),
    ))
}

/// Build, optimize with a callback, and collect the ordinary solver result.
pub fn solve_with_callback<F>(
    model: &Model,
    opts: &GurobiOptions,
    callback: &mut F,
    mask: Option<gurobi_rs::callback::CallbackMask>,
) -> Result<SolverResult, SolverError>
where
    F: gurobi_rs::callback::Callback,
{
    let kind = model.kind();
    let env = default_env()?;
    let mut built = build(model, opts, &env)?;
    run_and_collect_with_callback(&mut built, kind, callback, mask)
}

/// Build `model`, solve it, and when Gurobi finds it infeasible compute an
/// irreducible infeasible subsystem via Gurobi's `computeIIS`.
///
/// `DualReductions` is turned off so an ambiguous "infeasible or unbounded" presolve
/// outcome is resolved to a definite status before the IIS is requested.
///
/// # Errors
///
/// Returns a [`SolverError`] if the model is unsupported, Gurobi errors during setup
/// or optimization, or the model is not infeasible.
///
/// # Panics
///
/// Panics if a constraint, SOC, or variable index overflows `u32`.
pub fn compute_iis(model: &Model, opts: &GurobiOptions) -> Result<Iis, SolverError> {
    let env = default_env()?;
    let mut built = build(model, opts, &env)?;
    // Force a definite Infeasible/Unbounded verdict so we don't ask for an IIS on an
    // unbounded model.
    built.model.set_param(gurobi_rs::param::DualReductions, 0).map_err(map_gurobi_err)?;
    built.model.optimize().map_err(map_gurobi_err)?;
    compute_iis_resident(&mut built)
}

/// Run Gurobi's `computeIIS` on an already-optimized [`Built`] and read the IIS
/// membership back. Shared by the one-shot [`compute_iis`] and the persistent handle.
///
/// If the last optimize left the ambiguous `INF_OR_UNBD` status (the standalone path
/// forces `DualReductions` off up front, but a persistent handle's prior `solve` may
/// not have), this turns `DualReductions` off and re-optimizes to get a definite
/// verdict first, since Gurobi's `computeIIS` requires a proven-infeasible model.
///
/// # Errors
///
/// Returns a [`SolverError`] if the model is not infeasible or Gurobi errors
/// during IIS computation.
///
/// # Panics
///
/// Panics if a constraint, SOC, or variable index overflows `u32`.
pub(crate) fn compute_iis_resident(built: &mut Built) -> Result<Iis, SolverError> {
    let mut termination = map_status(built.model.get_attr(attr::Status).map_err(map_gurobi_err)?);
    if matches!(termination, TerminationStatus::InfeasibleOrUnbounded) {
        // Dual reductions can leave "infeasible or unbounded". Turn them off just
        // for a disambiguating re-optimize, then restore the saved value so the
        // resident model's parameter state is unchanged for later solves.
        let saved =
            built.model.get_param(gurobi_rs::param::DualReductions).map_err(map_gurobi_err)?;
        built.model.set_param(gurobi_rs::param::DualReductions, 0).map_err(map_gurobi_err)?;
        let outcome = (|| {
            built.model.optimize().map_err(map_gurobi_err)?;
            built.model.get_attr(attr::Status).map(map_status).map_err(map_gurobi_err)
        })();
        built.model.set_param(gurobi_rs::param::DualReductions, saved).map_err(map_gurobi_err)?;
        termination = outcome?;
    }
    if !termination.is_infeasible() {
        return Err(SolverError::Backend(format!(
            "cannot compute an IIS: model is not infeasible ({termination:?})"
        )));
    }

    built.model.compute_iis().map_err(map_gurobi_err)?;
    let minimal = built.model.get_attr(attr::IISMinimal).map_err(map_gurobi_err)?;
    if minimal != 1 {
        return Err(SolverError::Backend(
            "Gurobi IIS computation was interrupted or did not produce a minimal subsystem".into(),
        ));
    }
    read_iis(built)
}

/// Read the IIS membership attributes off a model on which `computeIIS` has already
/// run, mapping Gurobi's per-row and per-column flags back to oximo ids through the
/// 1:1 handle vectors on [`Built`].
fn read_iis(built: &Built) -> Result<Iis, SolverError> {
    let model = &built.model;
    let mut iis = Iis::default();

    for (i, handle) in built.constrs.iter().enumerate() {
        let mut in_iis = false;
        for row in &handle.rows {
            let selected = match row {
                GurobiRow::Lin(c) => model.get_obj_attr(attr::IISConstr, c),
                GurobiRow::Quad(q) => model.get_obj_attr(attr::IISQConstr, q),
            }
            .map_err(map_gurobi_err)?;
            in_iis |= selected != 0;
        }
        for generated in &handle.generated {
            in_iis |= generated_selected(model, generated)?;
        }
        if in_iis {
            iis.constraints
                .push(ConstraintId(u32::try_from(i).expect("constraint index fits u32")));
        }
    }

    for generated in &built.objective_generated {
        if generated_selected(model, generated)? {
            return Err(SolverError::Backend(
                "Gurobi IIS includes an objective-generated definition that cannot be mapped to an original model member"
                    .into(),
            ));
        }
    }

    for (i, handle) in built.soc_rows.iter().enumerate() {
        let quadratic =
            model.get_obj_attr(attr::IISQConstr, &handle.quadratic).map_err(map_gurobi_err)?;
        let sign =
            model.get_obj_attr(attr::IISConstr, &handle.bound_sign).map_err(map_gurobi_err)?;
        if quadratic != 0 || sign != 0 {
            iis.soc_constraints
                .push(SocConstraintId(u32::try_from(i).expect("soc index fits u32")));
        }
    }

    for (id, sos) in &built.sos {
        if model.get_obj_attr(attr::IISSOS, sos).map_err(map_gurobi_err)? != 0 {
            iis.sos_constraints.push(*id);
        }
    }

    // Only model variables are scanned.
    for (i, v) in built.vars.iter().enumerate() {
        let id = VarId(u32::try_from(i).expect("var index fits u32"));
        if model.get_obj_attr(attr::IISLB, v).map_err(map_gurobi_err)? != 0 {
            iis.var_bounds.push((id, VarBoundKind::Lower));
        }
        if model.get_obj_attr(attr::IISUB, v).map_err(map_gurobi_err)? != 0 {
            iis.var_bounds.push((id, VarBoundKind::Upper));
        }
    }

    Ok(iis)
}

fn generated_selected(
    model: &gurobi_rs::Model,
    generated: &GeneratedConstraint,
) -> Result<bool, SolverError> {
    let selected = match generated {
        GeneratedConstraint::Linear(c) => model.get_obj_attr(attr::IISConstr, c),
        GeneratedConstraint::Quadratic(q) => model.get_obj_attr(attr::IISQConstr, q),
        GeneratedConstraint::General(g) => model.get_obj_attr(attr::IISGenConstr, g),
    }
    .map_err(map_gurobi_err)?;
    Ok(selected != 0)
}

/// Add one Gurobi variable per model variable, in `VarId` order, applying its
/// domain, bounds, and any warm-start value.
fn add_variables(
    gurobi_model: &mut gurobi_rs::Model,
    vars: &[Variable],
) -> Result<Vec<gurobi_rs::Var>, SolverError> {
    let specs = vars.iter().map(|v| {
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
        gurobi_rs::VarSpec::new(v.name.as_str(), vtype, 0.0, floor, v.ub, [])
    });
    gurobi_model.add_vars(specs).map_err(map_gurobi_err)
}

fn add_sos_constraints(
    constraints: &[SosConstraint],
    gurobi_model: &mut gurobi_rs::Model,
    gurobi_vars: &[gurobi_rs::Var],
) -> Result<Vec<(SosConstraintId, gurobi_rs::SOS)>, SolverError> {
    active_sos_constraints(constraints)
        .map(|(id, s)| {
            let members = s.members.iter().map(|m| (gurobi_vars[m.variable.index()], m.weight));
            let ty = match s.sos_type {
                SosType::Sos1 => SOSType::Ty1,
                SosType::Sos2 => SOSType::Ty2,
            };
            gurobi_model.add_sos(members, ty).map(|handle| (id, handle)).map_err(map_gurobi_err)
        })
        .collect()
}

fn active_sos_constraints(
    constraints: &[SosConstraint],
) -> impl Iterator<Item = (SosConstraintId, &SosConstraint)> {
    constraints.iter().enumerate().filter(|(_, constraint)| constraint.active).map(
        |(index, constraint)| {
            (SosConstraintId(u32::try_from(index).expect("sos index fits u32")), constraint)
        },
    )
}

fn apply_initial_values(
    gurobi_model: &gurobi_rs::Model,
    vars: &[Variable],
    gurobi_vars: &[gurobi_rs::Var],
) -> Result<(), SolverError> {
    let starts = vars
        .iter()
        .zip(gurobi_vars.iter().copied())
        .filter_map(|(v, var)| v.initial.map(|value| (var, value)));
    gurobi_model.set_obj_attr_batch(attr::Start, starts).map_err(map_gurobi_err)
}

/// Gurobi's handle for an added constraint, kept so its dual can be queried
/// after the solve (`Pi` for linear rows, `QCPi` for quadratic rows).
pub(crate) enum GurobiRow {
    Lin(gurobi_rs::Constr),
    Quad(gurobi_rs::QConstr),
}

/// All native rows and generated definitions associated with one original
/// algebraic constraint. A nonlinear range may have two comparison rows.
pub(crate) struct ConstraintHandle {
    pub rows: Vec<GurobiRow>,
    pub generated: Vec<GeneratedConstraint>,
}

/// The native rows associated with one explicit second-order cone.
///
/// Squaring `||terms|| <= bound` also requires the generated `bound >= 0`
/// side row. Both rows belong to the original cone for IIS reporting.
pub(crate) struct SocHandle {
    pub quadratic: gurobi_rs::QConstr,
    pub bound_sign: gurobi_rs::Constr,
    pub bound: LinearTerms<'static>,
}

/// Add every model constraint, returning the per-constraint row handle.
/// Each constraint takes the linear fast path when its LHS is linear
/// and falls back to the general lowering otherwise.
#[expect(clippy::too_many_lines)]
fn add_constraints(
    prepared: &PreparedExpressions,
    constraints: &[Constraint],
    gurobi_model: &mut gurobi_rs::Model,
    gurobi_vars: &[gurobi_rs::Var],
    aux_counter: &mut u32,
) -> Result<Vec<ConstraintHandle>, SolverError> {
    let arena = prepared.arena();
    struct PendingSingle {
        index: usize,
        name: String,
        expr: LinExpr,
        sense: Sense,
        rhs: f64,
    }
    struct PendingRange {
        index: usize,
        name: String,
        expr: LinExpr,
        lower: f64,
        upper: f64,
    }

    let mut slots: Vec<Option<ConstraintHandle>> = (0..constraints.len()).map(|_| None).collect();
    let mut singles = Vec::new();
    let mut ranges = Vec::new();
    let mut nonlinear = Vec::new();

    for (index, c) in constraints.iter().enumerate() {
        if let Some((sense, rhs)) = c.as_single() {
            if let Some(t) = prepared.linear(c.lhs) {
                let mut expr = LinExpr::new();
                for &(v, coeff) in t.coeffs.iter() {
                    expr.add_term(coeff, gurobi_vars[v.index()]);
                }
                singles.push(PendingSingle {
                    index,
                    name: c.name.to_string(),
                    expr,
                    sense,
                    rhs: rhs - t.constant,
                });
            } else {
                nonlinear.push(index);
            }
        } else if c.is_range() {
            if let Some(t) = prepared.linear(c.lhs) {
                let mut expr = LinExpr::new();
                for &(v, coeff) in t.coeffs.iter() {
                    expr.add_term(coeff, gurobi_vars[v.index()]);
                }
                ranges.push(PendingRange {
                    index,
                    name: c.name.to_string(),
                    expr,
                    lower: c.lower - t.constant,
                    upper: c.upper - t.constant,
                });
            } else {
                nonlinear.push(index);
            }
        } else if c.lower.is_infinite() && c.upper.is_infinite() {
            slots[index] = Some(ConstraintHandle { rows: Vec::new(), generated: Vec::new() });
        } else {
            nonlinear.push(index);
        }
    }

    let mut single_names = Vec::with_capacity(singles.len());
    let mut single_indices = Vec::with_capacity(singles.len());
    let single_batch: Vec<_> = singles
        .into_iter()
        .map(|p| {
            single_names.push(p.name);
            single_indices.push(p.index);
            let expr = p.expr;
            match p.sense {
                Sense::Le => c!(expr <= p.rhs),
                Sense::Ge => c!(expr >= p.rhs),
                Sense::Eq => c!(expr == p.rhs),
            }
        })
        .collect();
    let single_batch_iter = single_batch.into_iter();
    let single_handles = gurobi_model
        .add_constrs(single_names.iter().zip(single_batch_iter))
        .map_err(map_gurobi_err)?;
    for (index, handle) in single_indices.into_iter().zip(single_handles) {
        slots[index] =
            Some(ConstraintHandle { rows: vec![GurobiRow::Lin(handle)], generated: Vec::new() });
    }

    let mut range_names = Vec::with_capacity(ranges.len());
    let mut range_indices = Vec::with_capacity(ranges.len());
    let range_batch: Vec<_> = ranges
        .into_iter()
        .map(|p| {
            range_names.push(p.name);
            range_indices.push(p.index);
            RangeExpr { expr: p.expr.into(), lb: p.lower, ub: p.upper }
        })
        .collect();
    let range_batch_iter = range_batch.into_iter();
    let range_handles = gurobi_model
        .add_ranges(range_names.iter().zip(range_batch_iter))
        .map_err(map_gurobi_err)?;
    for (index, handle) in range_indices.into_iter().zip(range_handles.1) {
        slots[index] =
            Some(ConstraintHandle { rows: vec![GurobiRow::Lin(handle)], generated: Vec::new() });
    }

    for index in nonlinear {
        let c = &constraints[index];
        let (rows, generated) = add_nonlinear_constraint(
            arena,
            c.lhs,
            c.as_single(),
            c.lower,
            c.upper,
            c.name.as_str(),
            gurobi_model,
            gurobi_vars,
            aux_counter,
        )?;
        slots[index] = Some(ConstraintHandle { rows, generated });
    }

    slots
        .into_iter()
        .map(|slot| {
            slot.ok_or_else(|| SolverError::Backend("internal constraint batching error".into()))
        })
        .collect()
}

/// Lower each explicit SOC constraint `||terms||_2 <= bound` to the quadratic
/// row `sum(term_i^2) - bound^2 <= 0` plus the linear side condition
/// `bound >= 0`.
///
/// Returns each cone's quadratic-row handle plus its affine bound side, in
/// `SocConstraintId` order, so `collect_solution` can rescale the squared-form
/// `QCPi` multiplier back to the norm form.
fn add_soc_rows(
    prepared: &PreparedExpressions,
    vars: &[Variable],
    socs: &[SocConstraint],
    gurobi_model: &mut gurobi_rs::Model,
    gurobi_vars: &[gurobi_rs::Var],
) -> Result<Vec<SocHandle>, SolverError> {
    let arena = prepared.arena();
    let mut rows = Vec::with_capacity(socs.len());
    for (i, s) in socs.iter().enumerate() {
        let mut q = QuadExpr::new();
        for &term in &s.terms {
            let t = prepared.linear(term).ok_or_else(|| SolverError::Nonlinear {
                location: format!("second-order cone {:?} term {i}", s.name),
                term: describe_nonlinear_term(arena, term, &|v| var_name(vars, v))
                    .unwrap_or_else(|| "<nonlinear>".into()),
            })?;
            add_squared_affine(&mut q, &t, 1.0, gurobi_vars);
        }
        let b = prepared.linear(s.bound).ok_or_else(|| SolverError::Nonlinear {
            location: format!("second-order cone {:?} bound", s.name),
            term: describe_nonlinear_term(arena, s.bound, &|v| var_name(vars, v))
                .unwrap_or_else(|| "<nonlinear>".into()),
        })?;
        add_squared_affine(&mut q, &b, -1.0, gurobi_vars);
        let qrow =
            gurobi_model.add_qconstr(s.name.as_str(), c!(q <= 0.0)).map_err(map_gurobi_err)?;

        let mut e = LinExpr::new();
        for &(v, co) in b.coeffs.iter() {
            e.add_term(co, gurobi_vars[v.index()]);
        }
        let sign_name = format!("{}_sign", s.name);
        let sign_row =
            gurobi_model.add_constr(&sign_name, c!(e >= -b.constant)).map_err(map_gurobi_err)?;
        rows.push(SocHandle { quadratic: qrow, bound_sign: sign_row, bound: b.into_owned() });
    }
    Ok(rows)
}

/// Expand `sign * (a'x + c)^2` into `q`.
fn add_squared_affine(
    q: &mut QuadExpr,
    t: &LinearTerms,
    sign: f64,
    gurobi_vars: &[gurobi_rs::Var],
) {
    for (i, &(vi, ci)) in t.coeffs.iter().enumerate() {
        q.add_qterm(sign * ci * ci, gurobi_vars[vi.index()], gurobi_vars[vi.index()]);
        for &(vj, cj) in &t.coeffs.as_ref()[i + 1..] {
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

#[expect(clippy::too_many_arguments)]
fn add_nonlinear_constraint(
    arena: &ExprArena,
    lhs: ExprId,
    single: Option<(Sense, f64)>,
    lower_bound: f64,
    upper: f64,
    name: &str,
    gurobi_model: &mut gurobi_rs::Model,
    gurobi_vars: &[gurobi_rs::Var],
    aux_counter: &mut u32,
) -> Result<(Vec<GurobiRow>, Vec<GeneratedConstraint>), SolverError> {
    let mut ctx = LoweringCtx::new(gurobi_model, gurobi_vars, *aux_counter);
    let lowered = lower(arena, lhs, &mut ctx)?;
    *aux_counter = ctx.aux_counter;
    let generated = std::mem::take(&mut ctx.generated);
    drop(ctx);
    let rows = add_comparison_rows(gurobi_model, name, &lowered, single, lower_bound, upper)?;
    Ok((rows, generated))
}

fn add_comparison_rows(
    model: &mut gurobi_rs::Model,
    name: &str,
    value: &LoweredExpr,
    single: Option<(Sense, f64)>,
    lower: f64,
    upper: f64,
) -> Result<Vec<GurobiRow>, SolverError> {
    let senses = single
        .map_or_else(|| vec![(Sense::Ge, lower), (Sense::Le, upper)], |pair| vec![pair])
        .into_iter()
        .filter(|(_, rhs)| rhs.is_finite());
    senses
        .into_iter()
        .enumerate()
        .map(|(i, (sense, rhs))| {
            let row_name =
                if i == 0 && single.is_some() { name.to_string() } else { format!("{name}_{i}") };
            match value {
                LoweredExpr::Linear(e) => {
                    let row = match sense {
                        Sense::Le => model.add_constr(&row_name, c!(e.clone() <= rhs)),
                        Sense::Ge => model.add_constr(&row_name, c!(e.clone() >= rhs)),
                        Sense::Eq => model.add_constr(&row_name, c!(e.clone() == rhs)),
                    }
                    .map_err(map_gurobi_err)?;
                    Ok(GurobiRow::Lin(row))
                }
                LoweredExpr::Quadratic(e) => {
                    let row = match sense {
                        Sense::Le => model.add_qconstr(&row_name, c!(e.clone() <= rhs)),
                        Sense::Ge => model.add_qconstr(&row_name, c!(e.clone() >= rhs)),
                        Sense::Eq => model.add_qconstr(&row_name, c!(e.clone() == rhs)),
                    }
                    .map_err(map_gurobi_err)?;
                    Ok(GurobiRow::Quad(row))
                }
                LoweredExpr::Var(v) => {
                    let row = match sense {
                        Sense::Le => model.add_constr(&row_name, c!(v <= rhs)),
                        Sense::Ge => model.add_constr(&row_name, c!(v >= rhs)),
                        Sense::Eq => model.add_constr(&row_name, c!(v == rhs)),
                    }
                    .map_err(map_gurobi_err)?;
                    Ok(GurobiRow::Lin(row))
                }
            }
        })
        .collect()
}

fn set_objective(
    prepared: &PreparedExpressions,
    obj_expr: ExprId,
    sense: ObjectiveSense,
    gurobi_model: &mut gurobi_rs::Model,
    gurobi_vars: &[gurobi_rs::Var],
    aux_counter: &mut u32,
) -> Result<(f64, Vec<GeneratedConstraint>), SolverError> {
    let arena = prepared.arena();
    let gurobi_sense = match sense {
        ObjectiveSense::Minimize => ModelSense::Minimize,
        ObjectiveSense::Maximize => ModelSense::Maximize,
    };
    if let Some(t) = prepared.linear(obj_expr) {
        let mut e = LinExpr::new();
        for &(v, c) in t.coeffs.iter() {
            e.add_term(c, gurobi_vars[v.index()]);
        }
        // Gurobi's set_objective absorbs LinExpr offsets into ObjCon, so we do
        // not need to track the constant separately.
        e.add_constant(t.constant);
        gurobi_model.set_objective(e, gurobi_sense).map_err(map_gurobi_err)?;
        return Ok((0.0, Vec::new()));
    }
    let mut ctx = LoweringCtx::new(gurobi_model, gurobi_vars, *aux_counter);
    let lowered = lower(arena, obj_expr, &mut ctx)?;
    *aux_counter = ctx.aux_counter;
    ctx.model
        .set_objective(lowered.into_expr_for_objective(), gurobi_sense)
        .map_err(map_gurobi_err)?;
    let generated = std::mem::take(&mut ctx.generated);
    drop(ctx);
    Ok((0.0, generated))
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
    bool,
);

fn supports_dual_attributes(kind: ModelKind) -> bool {
    matches!(kind, ModelKind::LP | ModelKind::QP | ModelKind::QCP | ModelKind::SOCP)
}

fn collected_dual_status(kind: ModelKind, dual_available: bool, has_solution: bool) -> DualStatus {
    if dual_available {
        DualStatus::FeasiblePoint
    } else if !has_solution || !supports_dual_attributes(kind) {
        DualStatus::NoSolution
    } else {
        DualStatus::Unknown
    }
}

fn collect_solution(
    kind: ModelKind,
    model: &mut gurobi_rs::Model,
    vars: &[gurobi_rs::Var],
    constrs: &[ConstraintHandle],
    soc_rows: &[SocHandle],
    obj_constant: f64,
) -> Collected {
    // A primal point exists only when Gurobi actually stored one. `SolCount` is
    // the number of available solutions and it stays `> 0` for an incumbent
    // kept at a time/iteration/node limit.
    let sol_count = model.get_attr(attr::SolCount).unwrap_or(0);
    if sol_count <= 0 {
        return (
            Vec::new(),
            FxHashMap::default(),
            FxHashMap::default(),
            FxHashMap::default(),
            false,
        );
    }

    let solutions = collect_pool(model, vars, obj_constant, sol_count);

    // Skip retrieval of duals and reduced costs for integer model classes,
    // where Gurobi refuses the attributes.
    // LP/QP always have duals, QCP/SOCP rows only when the user opted into
    // `QCPDual=1`.
    if !supports_dual_attributes(kind) {
        return (
            solutions,
            FxHashMap::default(),
            FxHashMap::default(),
            FxHashMap::default(),
            false,
        );
    }

    let reduced_cost_values = model.get_obj_attr_batch(attr::RC, vars.iter().copied());
    let mut dual_available = reduced_cost_values.is_ok();
    let reduced_costs = reduced_cost_values.map(|v| index_map(&v)).unwrap_or_default();

    let mut dual = FxHashMap::default();
    dual.reserve(constrs.len());
    for (i, handle) in constrs.iter().enumerate() {
        let mut value = 0.0;
        let mut available = false;
        for row in &handle.rows {
            let pi = match row {
                GurobiRow::Lin(c) => model.get_obj_attr(attr::Pi, c),
                GurobiRow::Quad(q) => model.get_obj_attr(attr::QCPi, q),
            };
            if let Ok(pi) = pi {
                value += pi;
                available = true;
                dual_available = true;
            }
        }
        if available {
            dual.insert(ConstraintId(u32::try_from(i).unwrap()), value);
        }
    }

    // Convert the squared-form multiplier back to the norm-form
    // bound multiplier `z0 = 2 * bound_value * |QCPi|`.
    // Available only under `QCPDual=1`.
    let mut soc_dual = FxHashMap::default();
    soc_dual.reserve(soc_rows.len());
    if let Some(first) = solutions.first() {
        let primal = &first.primal;
        for (i, handle) in soc_rows.iter().enumerate() {
            if let Ok(pi) = model.get_obj_attr(attr::QCPi, &handle.quadratic) {
                let b_val = handle.bound.constant
                    + handle
                        .bound
                        .coeffs
                        .iter()
                        .map(|&(v, c)| c * primal.get(&v).copied().unwrap_or(0.0))
                        .sum::<f64>();
                soc_dual.insert(SocConstraintId(u32::try_from(i).unwrap()), 2.0 * b_val * pi.abs());
            }
        }
    }

    (solutions, reduced_costs, dual, soc_dual, dual_available)
}

/// Collect the `n` pooled primal points, best first.
fn collect_pool(
    model: &mut gurobi_rs::Model,
    vars: &[gurobi_rs::Var],
    obj_constant: f64,
    n: i32,
) -> Vec<SolutionPoint> {
    // Single point (and the only path for LP/continuous models):
    // read the incumbent directly via `X` / `ObjVal`.
    if n <= 1 {
        return vec![collect_incumbent(model, vars, obj_constant)];
    }

    // A MIP solution pool. Gurobi sorts it best-first.
    // `SolutionNumber` selects which one `Xn` / `PoolObjVal` report.
    let mut out = Vec::with_capacity(usize::try_from(n).unwrap_or(0));
    for k in 0..n {
        if model.set_param(gurobi_rs::param::SolutionNumber, k).is_err() {
            break;
        }
        let Ok(vals) = model.get_obj_attr_batch(attr::Xn, vars.iter().copied()) else {
            break;
        };
        let objective = model
            .get_attr(attr::PoolObjVal)
            .ok()
            .and_then(|v| ObjectiveTransform { sign: 1.0, offset: obj_constant }.restore(v));
        out.push(SolutionPoint { primal: index_map(&vals), objective });
    }
    if out.is_empty() {
        out.push(collect_incumbent(model, vars, obj_constant));
    }
    out
}

fn collect_incumbent(
    model: &mut gurobi_rs::Model,
    vars: &[gurobi_rs::Var],
    obj_constant: f64,
) -> SolutionPoint {
    let primal = model
        .get_obj_attr_batch(attr::X, vars.iter().copied())
        .map(|v| index_map(&v))
        .unwrap_or_default();
    let objective = model
        .get_attr(attr::ObjVal)
        .ok()
        .and_then(|v| ObjectiveTransform { sign: 1.0, offset: obj_constant }.restore(v));
    SolutionPoint { primal, objective }
}

/// Map a dense per-variable value array (in `VarId` order) to a sparse map.
fn index_map(vals: &[f64]) -> FxHashMap<VarId, f64> {
    oximo_solver::reconstruct::project_dense_primal(vals, vals.len()).unwrap_or_default()
}

fn map_status(status: Status) -> TerminationStatus {
    match status {
        Status::Optimal => TerminationStatus::Optimal,
        Status::Infeasible => TerminationStatus::Infeasible,
        Status::Unbounded => TerminationStatus::Unbounded,
        Status::InfOrUnbd => TerminationStatus::InfeasibleOrUnbounded,
        Status::Numeric => TerminationStatus::NumericError,
        Status::TimeLimit => TerminationStatus::TimeLimit,
        Status::IterationLimit => TerminationStatus::IterationLimit,
        Status::NodeLimit => TerminationStatus::NodeLimit,
        Status::Interrupted => TerminationStatus::Interrupted,
        Status::CutOff | Status::UserObjLimit => TerminationStatus::ObjectiveLimit,
        Status::SolutionLimit => TerminationStatus::SolutionLimit,
        Status::WorkLimit => TerminationStatus::WorkLimit,
        Status::MemLimit => TerminationStatus::MemoryLimit,
        Status::LocallyOptimal => TerminationStatus::LocallyOptimal,
        Status::LocallyInfeasible => TerminationStatus::LocallyInfeasible,
        // Gurobi could not meet optimality tolerances but holds a feasible point.
        Status::SubOptimal => TerminationStatus::Feasible,
        Status::Loaded => TerminationStatus::NotSolved,
        Status::InProgress => TerminationStatus::Other("Status: InProgress".into()),
    }
}

#[cfg(test)]
mod status_tests {
    use oximo_core::{SosConstraint, SosMember, VarId};

    use super::*;

    #[test]
    fn integer_incumbents_do_not_report_dual_status() {
        assert_eq!(collected_dual_status(ModelKind::MILP, false, true), DualStatus::NoSolution);
        assert_eq!(collected_dual_status(ModelKind::MINLP, false, true), DualStatus::NoSolution);
        assert_eq!(collected_dual_status(ModelKind::QP, false, true), DualStatus::Unknown);
        assert_eq!(collected_dual_status(ModelKind::QP, false, false), DualStatus::NoSolution);
        assert_eq!(collected_dual_status(ModelKind::MILP, true, true), DualStatus::FeasiblePoint);
    }

    #[test]
    fn native_limits_keep_distinct_termination_reasons() {
        assert_eq!(map_status(Status::UserObjLimit), TerminationStatus::ObjectiveLimit);
        assert_eq!(map_status(Status::SolutionLimit), TerminationStatus::SolutionLimit);
        assert_eq!(map_status(Status::WorkLimit), TerminationStatus::WorkLimit);
        assert_eq!(map_status(Status::MemLimit), TerminationStatus::MemoryLimit);
        assert_eq!(map_status(Status::LocallyInfeasible), TerminationStatus::LocallyInfeasible);
        assert_eq!(map_status(Status::Interrupted), TerminationStatus::Interrupted);
    }

    #[test]
    fn inactive_sos_constraints_are_skipped_without_compacting_ids() {
        let constraints = vec![
            SosConstraint {
                name: "inactive".into(),
                sos_type: SosType::Sos1,
                members: vec![SosMember { variable: VarId(0), weight: 1.0 }],
                active: false,
            },
            SosConstraint {
                name: "first".into(),
                sos_type: SosType::Sos1,
                members: vec![SosMember { variable: VarId(1), weight: 1.0 }],
                active: true,
            },
            SosConstraint {
                name: "second".into(),
                sos_type: SosType::Sos2,
                members: vec![SosMember { variable: VarId(2), weight: 1.0 }],
                active: true,
            },
        ];
        let ids: Vec<_> = active_sos_constraints(&constraints).map(|(id, _)| id).collect();
        assert_eq!(ids, [SosConstraintId(1), SosConstraintId(2)]);
    }
}

#[cfg(feature = "benchmark-support")]
#[doc(hidden)]
#[expect(clippy::cast_precision_loss)]
#[allow(clippy::wildcard_imports)]
pub mod benchmark_support {
    use oximo_core::constraint::Relate;

    use super::*;

    /// Crossover candidate used only to size the preprocessing benchmark cases.
    pub const THRESHOLD: usize = 1_024;

    pub fn model(rows: usize, degree: usize) -> Model {
        let model = Model::new("gurobi_extract_bench");
        let x = model.__var("x").lb(-5.0).ub(5.0).build();
        let y = model.__var("y").lb(-5.0).ub(5.0).build();
        let z = model.__var("z").lb(-5.0).ub(5.0).build();
        for i in 0..rows {
            let lhs = match degree {
                1 => x + 2.0 * y - z,
                2 => x.powi(2) + y,
                _ => x * y * z,
            };
            model.__add_constraint_auto(lhs.le(i as f64 + 10.0));
        }
        model.__minimize(x + y + z);
        model
    }

    /// Create a Gurobi environment once for repeated translation benchmarks.
    pub fn environment() -> Result<Env, SolverError> {
        default_env()
    }

    /// Translate into a fresh native Gurobi model without optimizing it.
    pub fn translate(model: &Model, env: &Env) -> Result<usize, SolverError> {
        let built = build(model, &GurobiOptions::default(), env)?;
        Ok(built.vars.len() + built.constrs.len() + built.soc_rows.len())
    }
}
