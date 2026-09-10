use oximo_solver::prepare::{LoweringContext, PolynomialTerms};
use oximo_solver::reconstruct::normalize_result;
use std::time::{Duration, Instant};

use mosek::{
    Boundkey, Dinfitem, Iinfitem, Liinfitem, Objsense, Prosta, Rescode, Solsta, Soltype,
    Streamtype, Task, TaskCB, Variabletype,
};
use oximo_core::{ConstraintId, Domain, Model, ModelKind, ObjectiveSense, SocConstraintId};
use oximo_expr::{LinearTerms, QuadraticTerms, VarId};
use oximo_solver::{
    DualStatus, PrimalStatus, SolutionPoint, SolverError, SolverResult, TerminationStatus,
};
use rustc_hash::FxHashMap;

use crate::MosekOptions;

#[derive(Debug)]
pub(crate) struct Meta {
    kind: ModelKind,
    row_by_constraint: Vec<Option<i32>>,
    explicit_accs: Vec<(SocConstraintId, i64, usize)>,
}

/// Reusable native upload buffers.
#[derive(Default)]
struct TaskScratch {
    indices: Vec<i32>,
    columns: Vec<i32>,
    values: Vec<f64>,
    acc_rhs: Vec<f64>,
}

/// Translate and solve an oximo model with a fresh MOSEK task.
///
/// # Errors
///
/// Returns an error for unsupported model kinds or variable domains, invalid
/// expressions, and native MOSEK setup, optimization, or solution-query errors.
pub fn solve(model: &Model, opts: &MosekOptions) -> Result<SolverResult, SolverError> {
    if model.has_active_sos_constraints() {
        return Err(SolverError::UnsupportedSos);
    }
    let (mut task, meta) = build_task(model, opts)?;
    solve_task(model, &mut task, &meta)
}

/// Build a callback-capable MOSEK task from an oximo model.
///
/// This is shared by one-shot and persistent solves. A persistent handle retains
/// the returned task only when its model snapshot remains compatible.
pub(crate) fn build_task(
    model: &Model,
    opts: &MosekOptions,
) -> Result<(TaskCB, Meta), SolverError> {
    if model.has_active_sos_constraints() {
        return Err(SolverError::UnsupportedSos);
    }
    let prepared = LoweringContext::new(model)?;
    let kind = prepared.kind();
    if !crate::supported(kind) {
        return Err(SolverError::UnsupportedKind(kind));
    }
    reject_semi_domains(model)?;

    let task =
        Task::new().ok_or_else(|| SolverError::Backend("MOSEK: failed to create task".into()))?;
    let mut task = task.with_callbacks();
    build_base(&model.name, &prepared, &mut task)?;
    opts.apply_cb(&mut task)?;

    if opts.universal.verbose.unwrap_or(false) {
        task.put_stream_callback(Streamtype::LOG, |message| print!("{message}"))
            .map_err(backend)?;
    }
    let meta = build_rows_and_cones(&prepared, kind, &mut task)?;

    Ok((task, meta))
}

/// Optimize an already built task and read its result.
pub(crate) fn solve_task(
    model: &Model,
    task: &mut TaskCB,
    meta: &Meta,
) -> Result<SolverResult, SolverError> {
    let started = Instant::now();
    let trm = task.optimize().map_err(backend)?;
    let elapsed = started.elapsed();
    extract_result(model, task, meta, trm, elapsed)
}

fn reject_semi_domains(model: &Model) -> Result<(), SolverError> {
    for variable in model.variables().iter() {
        if matches!(variable.domain, Domain::SemiContinuous { .. } | Domain::SemiInteger { .. }) {
            return Err(SolverError::Backend(format!(
                "MOSEK: variable '{}' uses a semi-continuous or semi-integer domain; \
                 this backend does not reformulate semi-variable domains",
                variable.name
            )));
        }
    }
    Ok(())
}

fn build_base(
    name: &str,
    prepared: &LoweringContext<'_>,
    task: &mut TaskCB,
) -> Result<(), SolverError> {
    let variables = prepared.variables();
    let objective = prepared.objective();
    let quad = objective.as_ref().map_or_else(
        || Ok(oximo_solver::prepare::Extracted::Owned(QuadraticTerms::default())),
        |obj| prepared.require_quadratic(obj.expr, || "the objective".into()),
    )?;

    task.put_task_name(name).map_err(backend)?;
    task.append_vars(count_i32(variables.len(), "variables")?).map_err(backend)?;
    for variable in variables {
        let j = index_i32(variable.id.index(), "variable")?;
        task.put_var_name(j, &variable.name).map_err(backend)?;
        let (key, lower, upper) = bounds(variable.lb, variable.ub);
        task.put_var_bound(j, key, lower, upper).map_err(backend)?;
        if variable.domain.is_integer() {
            task.put_var_type(j, Variabletype::TYPE_INT).map_err(backend)?;
        }
    }

    for &(variable, coefficient) in &quad.linear {
        task.put_c_j(index_i32(variable.index(), "variable")?, coefficient).map_err(backend)?;
    }
    task.put_cfix(quad.constant).map_err(backend)?;
    let mut scratch = TaskScratch::default();
    put_q_objective(task, &quad, &mut scratch)?;
    task.put_obj_sense(objective.as_ref().map_or(Objsense::MINIMIZE, |obj| match obj.sense {
        ObjectiveSense::Minimize => Objsense::MINIMIZE,
        ObjectiveSense::Maximize => Objsense::MAXIMIZE,
    }))
    .map_err(backend)?;

    let initial_type =
        if variables.iter().any(|v| v.domain.is_integer()) { Soltype::ITG } else { Soltype::ITR };
    let mut has_initial = false;
    for variable in variables {
        if let Some(value) = variable.initial {
            let j = index_i32(variable.id.index(), "variable")?;
            task.put_xx_slice(initial_type, j, j + 1, &[value]).map_err(backend)?;
            has_initial = true;
        }
    }
    if has_initial && initial_type == Soltype::ITG {
        task.put_int_param(mosek::Iparam::MIO_CONSTRUCT_SOL, mosek::Onoffkey::ON)
            .map_err(backend)?;
    }
    Ok(())
}

fn build_rows_and_cones(
    prepared: &LoweringContext<'_>,
    kind: ModelKind,
    task: &mut TaskCB,
) -> Result<Meta, SolverError> {
    let model_constraints = prepared.constraints();
    let constraints = model_constraints.algebraic();
    let mut row_by_constraint = vec![None; constraints.len()];
    let mut detected = Vec::new();
    let mut scratch = TaskScratch::default();

    for (id, constraint) in constraints.iter().enumerate() {
        // Direct affine rows remain borrowed. Every other row is traversed once
        // as a quadratic and retained through detection/native emission.
        let terms = prepared
            .require_polynomial(constraint.lhs, || format!("constraint {:?}", constraint.name))?;
        if let PolynomialTerms::Quadratic(terms) = &terms
            && let Some(form) =
                oximo_core::__detect_soc_from_quadratic(prepared.variables(), constraint, terms)
        {
            detected.push((id, form));
            continue;
        }
        let (linear, constant) = match &terms {
            PolynomialTerms::Affine(terms) => (terms.coeffs.as_ref(), terms.constant),
            PolynomialTerms::Quadratic(terms) => (terms.linear.as_slice(), terms.constant),
        };
        let row = task.get_num_con().map_err(backend)?;
        task.append_cons(1).map_err(backend)?;
        task.put_con_name(row, &constraint.name).map_err(backend)?;
        put_linear_row(task, row, linear, &mut scratch)?;
        let (key, lower, upper) = {
            let (lower, upper) = oximo_solver::prepare::shifted_bounds(constraint, constant);
            bounds(lower, upper)
        };
        task.put_con_bound(row, key, lower, upper).map_err(backend)?;
        if let PolynomialTerms::Quadratic(terms) = terms {
            put_q_constraint(task, row, &terms, &mut scratch)?;
        }
        row_by_constraint[id] = Some(row);
    }

    let socs = prepared.constraints().second_order_cones();
    let mut explicit_accs = Vec::with_capacity(socs.len());
    let mut next_afe = 0_i64;
    let mut next_acc = 0_i64;
    for (id, soc) in socs.iter().enumerate() {
        let form = prepared.explicit_soc(soc)?;
        let dim = append_soc(task, &form, &mut next_afe, &mut scratch)?;
        task.put_acc_name(next_acc, &soc.name).map_err(backend)?;
        explicit_accs.push((
            SocConstraintId(u32::try_from(id).map_err(|_| overflow("SOC constraint"))?),
            next_acc,
            dim,
        ));
        next_acc += 1;
    }
    for (id, form) in detected {
        append_soc(task, &form, &mut next_afe, &mut scratch)?;
        task.put_acc_name(next_acc, &constraints[id].name).map_err(backend)?;
        next_acc += 1;
    }

    Ok(Meta { kind, row_by_constraint, explicit_accs })
}

fn append_soc(
    task: &mut TaskCB,
    form: &oximo_core::SocForm,
    next_afe: &mut i64,
    scratch: &mut TaskScratch,
) -> Result<usize, SolverError> {
    let dim = 1 + form.terms.len();
    task.append_afes(i64::try_from(dim).map_err(|_| overflow("SOC dimension"))?)
        .map_err(backend)?;
    let first = *next_afe;
    put_afe(task, first, &form.bound, scratch)?;
    for (offset, terms) in form.terms.iter().enumerate() {
        put_afe(
            task,
            first + i64::try_from(offset + 1).map_err(|_| overflow("AFE index"))?,
            terms,
            scratch,
        )?;
    }
    let domain = task
        .append_quadratic_cone_domain(i64::try_from(dim).map_err(|_| overflow("SOC dimension"))?)
        .map_err(backend)?;
    scratch.acc_rhs.clear();
    scratch.acc_rhs.resize(dim, 0.0);
    task.append_acc_seq(domain, first, &scratch.acc_rhs).map_err(backend)?;
    *next_afe += i64::try_from(dim).map_err(|_| overflow("AFE count"))?;
    Ok(dim)
}

fn put_afe(
    task: &mut TaskCB,
    afe: i64,
    terms: &LinearTerms<'_>,
    scratch: &mut TaskScratch,
) -> Result<(), SolverError> {
    fill_linear_buffers(terms.coeffs.as_ref(), scratch)?;
    task.put_afe_f_row(afe, &scratch.indices, &scratch.values).map_err(backend)?;
    task.put_afe_g(afe, terms.constant).map_err(backend)
}

fn put_linear_row(
    task: &mut TaskCB,
    row: i32,
    terms: &[(VarId, f64)],
    scratch: &mut TaskScratch,
) -> Result<(), SolverError> {
    fill_linear_buffers(terms, scratch)?;
    task.put_a_row(row, &scratch.indices, &scratch.values).map_err(backend)
}

fn fill_linear_buffers(
    terms: &[(VarId, f64)],
    scratch: &mut TaskScratch,
) -> Result<(), SolverError> {
    scratch.indices.clear();
    scratch.values.clear();
    scratch.indices.reserve(terms.len().saturating_sub(scratch.indices.capacity()));
    scratch.values.reserve(terms.len().saturating_sub(scratch.values.capacity()));
    for &(variable, value) in terms {
        scratch.indices.push(index_i32(variable.index(), "variable")?);
        scratch.values.push(value);
    }
    Ok(())
}

fn put_q_objective(
    task: &mut TaskCB,
    terms: &QuadraticTerms,
    scratch: &mut TaskScratch,
) -> Result<(), SolverError> {
    if terms.hessian.is_empty() {
        return Ok(());
    }
    fill_q_triplets(terms, scratch)?;
    task.put_q_obj(&scratch.indices, &scratch.columns, &scratch.values).map_err(backend)
}

fn put_q_constraint(
    task: &mut TaskCB,
    row: i32,
    terms: &QuadraticTerms,
    scratch: &mut TaskScratch,
) -> Result<(), SolverError> {
    if terms.hessian.is_empty() {
        return Ok(());
    }
    fill_q_triplets(terms, scratch)?;
    task.put_q_con_k(row, &scratch.indices, &scratch.columns, &scratch.values).map_err(backend)
}

fn fill_q_triplets(terms: &QuadraticTerms, scratch: &mut TaskScratch) -> Result<(), SolverError> {
    scratch.indices.clear();
    scratch.columns.clear();
    scratch.values.clear();
    let len = terms.hessian.len();
    scratch.indices.reserve(len.saturating_sub(scratch.indices.capacity()));
    scratch.columns.reserve(len.saturating_sub(scratch.columns.capacity()));
    scratch.values.reserve(len.saturating_sub(scratch.values.capacity()));
    for &(row, column, value) in &terms.hessian {
        scratch.indices.push(index_i32(row.index(), "variable")?);
        scratch.columns.push(index_i32(column.index(), "variable")?);
        scratch.values.push(value);
    }
    Ok(())
}

fn extract_result(
    model: &Model,
    task: &TaskCB,
    meta: &Meta,
    trm: i32,
    elapsed: Duration,
) -> Result<SolverResult, SolverError> {
    let mixed_integer = matches!(
        meta.kind,
        ModelKind::MILP | ModelKind::MIQP | ModelKind::MIQCP | ModelKind::MISOCP
    );
    let solution_type = select_solution(task, mixed_integer);
    let solution_status = task.get_sol_sta(solution_type).unwrap_or(Solsta::UNKNOWN);
    let problem_status = task.get_pro_sta(solution_type).unwrap_or(Prosta::UNKNOWN);
    let termination = map_status(solution_status, problem_status, trm);
    let solution_defined = task.solution_def(solution_type).unwrap_or(false);
    let has_point = solution_defined && has_primal_solution(solution_status);
    let has_dual = !mixed_integer && solution_defined && has_dual_solution(solution_status);

    let variables = model.variables();
    let mut solutions = Vec::new();
    let mut dual = FxHashMap::default();
    let mut soc_dual = FxHashMap::default();
    let mut reduced_costs = FxHashMap::default();
    if has_point {
        let mut values = vec![0.0; variables.len()];
        task.get_xx(solution_type, &mut values).map_err(backend)?;
        let primal = oximo_solver::reconstruct::project_dense_primal(&values, variables.len())
            .unwrap_or_default();
        let objective = task.get_primal_obj(solution_type).ok().filter(|value| value.is_finite());
        solutions.push(SolutionPoint { primal, objective });
    }
    if has_dual {
        collect_continuous_duals(
            task,
            solution_type,
            meta,
            variables.len(),
            &mut dual,
            &mut reduced_costs,
        )?;
        collect_continuous_soc_duals(task, solution_type, meta, &mut soc_dual);
    }

    let bound_defined = mixed_integer
        && task.get_int_inf(Iinfitem::MIO_OBJ_BOUND_DEFINED).is_ok_and(|defined| defined != 0);
    let best_bound = bound_defined
        .then(|| task.get_dou_inf(Dinfitem::MIO_OBJ_BOUND).ok())
        .flatten()
        .filter(|value| value.is_finite());
    let iterations = iteration_count(task, mixed_integer);
    let node_count = mixed_integer
        .then(|| task.get_int_inf(Iinfitem::MIO_NUM_SOLVED_NODES).ok())
        .flatten()
        .and_then(|count| u64::try_from(count).ok());
    let dual_status = if has_dual { DualStatus::FeasiblePoint } else { DualStatus::NoSolution };
    let mut major = 0;
    let mut minor = 0;
    let mut revision = 0;
    let solver_version = mosek::get_version(&mut major, &mut minor, &mut revision)
        .ok()
        .map(|()| format!("{major}.{minor}.{revision}").into());
    Ok(normalize_result(
        SolverResult {
            termination,
            primal_status: PrimalStatus::NoSolution,
            dual_status,
            solutions,
            dual,
            soc_dual,
            reduced_costs,
            best_bound,
            gap: mixed_integer
                .then(|| task.get_dou_inf(Dinfitem::MIO_OBJ_REL_GAP).ok())
                .flatten(),
            solve_time: elapsed,
            iterations,
            node_count,
            raw_status: Some(
                format!("solution={solution_status}, problem={problem_status}, termination={trm}")
                    .into(),
            ),
            raw_log: None,
            solver_name: Some(crate::NAME.into()),
            solver_version,
        },
        variables.len(),
    ))
}

fn collect_continuous_duals(
    task: &TaskCB,
    solution_type: i32,
    meta: &Meta,
    num_variables: usize,
    dual: &mut FxHashMap<ConstraintId, f64>,
    reduced_costs: &mut FxHashMap<VarId, f64>,
) -> Result<(), SolverError> {
    let num_rows = usize::try_from(task.get_num_con().map_err(backend)?)
        .map_err(|_| overflow("constraint"))?;
    let mut row_duals = vec![0.0; num_rows];
    task.get_y(solution_type, &mut row_duals).map_err(backend)?;
    for (id, row) in meta.row_by_constraint.iter().enumerate() {
        if let Some(row) = row
            && let Some(&value) = row_duals.get(usize::try_from(*row).unwrap_or(usize::MAX))
        {
            dual.insert(
                ConstraintId(u32::try_from(id).map_err(|_| overflow("constraint"))?),
                value,
            );
        }
    }

    let mut lower = vec![0.0; num_variables];
    let mut upper = vec![0.0; num_variables];
    task.get_slx(solution_type, &mut lower).map_err(backend)?;
    task.get_sux(solution_type, &mut upper).map_err(backend)?;
    for (index, (lower, upper)) in lower.into_iter().zip(upper).enumerate() {
        reduced_costs
            .insert(VarId(u32::try_from(index).map_err(|_| overflow("variable"))?), lower - upper);
    }
    Ok(())
}

fn collect_continuous_soc_duals(
    task: &TaskCB,
    solution_type: i32,
    meta: &Meta,
    soc_dual: &mut FxHashMap<SocConstraintId, f64>,
) {
    for &(id, acc, dim) in &meta.explicit_accs {
        let mut dot_y = vec![0.0; dim];
        if task.get_acc_dot_y(solution_type, acc, &mut dot_y).is_ok()
            && let Some(&bound_multiplier) = dot_y.first()
        {
            soc_dual.insert(id, bound_multiplier);
        }
    }
}

fn has_primal_solution(solution_status: i32) -> bool {
    matches!(
        solution_status,
        Solsta::OPTIMAL | Solsta::INTEGER_OPTIMAL | Solsta::PRIM_FEAS | Solsta::PRIM_AND_DUAL_FEAS
    )
}

fn has_dual_solution(solution_status: i32) -> bool {
    matches!(solution_status, Solsta::OPTIMAL | Solsta::DUAL_FEAS | Solsta::PRIM_AND_DUAL_FEAS)
}

fn select_solution(task: &TaskCB, mixed_integer: bool) -> i32 {
    if mixed_integer {
        return Soltype::ITG;
    }
    if task.solution_def(Soltype::ITR).unwrap_or(false) {
        Soltype::ITR
    } else if task.solution_def(Soltype::BAS).unwrap_or(false) {
        Soltype::BAS
    } else {
        Soltype::ITR
    }
}

fn iteration_count(task: &TaskCB, mixed_integer: bool) -> u64 {
    let nonnegative = |value: i64| u64::try_from(value.max(0)).unwrap_or(0);
    if mixed_integer {
        return [
            task.get_lint_inf(Liinfitem::MIO_INTPNT_ITER).unwrap_or(0),
            task.get_lint_inf(Liinfitem::MIO_SIMPLEX_ITER).unwrap_or(0),
        ]
        .into_iter()
        .map(nonnegative)
        .sum();
    }
    nonnegative(i64::from(task.get_int_inf(Iinfitem::INTPNT_ITER).unwrap_or(0)))
        + nonnegative(task.get_lint_inf(Liinfitem::SIMPLEX_ITER).unwrap_or(0))
}

fn map_status(solution: i32, problem: i32, trm: i32) -> TerminationStatus {
    if matches!(solution, Solsta::OPTIMAL | Solsta::INTEGER_OPTIMAL) {
        return TerminationStatus::Optimal;
    }
    if matches!(solution, Solsta::PRIM_INFEAS_CER)
        || matches!(problem, Prosta::PRIM_INFEAS | Prosta::PRIM_AND_DUAL_INFEAS)
    {
        return TerminationStatus::Infeasible;
    }
    if matches!(solution, Solsta::DUAL_INFEAS_CER) || problem == Prosta::DUAL_INFEAS {
        return TerminationStatus::Unbounded;
    }
    if problem == Prosta::PRIM_INFEAS_OR_UNBOUNDED {
        return TerminationStatus::InfeasibleOrUnbounded;
    }
    if matches!(solution, Solsta::PRIM_ILLPOSED_CER | Solsta::DUAL_ILLPOSED_CER)
        || problem == Prosta::ILL_POSED
    {
        return TerminationStatus::NumericError;
    }
    match trm {
        Rescode::TRM_MAX_TIME | Rescode::TRM_SERVER_MAX_TIME => TerminationStatus::TimeLimit,
        Rescode::TRM_MAX_ITERATIONS => TerminationStatus::IterationLimit,
        Rescode::TRM_MIO_NUM_BRANCHES => TerminationStatus::NodeLimit,
        Rescode::TRM_MIO_NUM_RELAXS => TerminationStatus::WorkLimit,
        Rescode::TRM_NUM_MAX_NUM_INT_SOLUTIONS => TerminationStatus::SolutionLimit,
        Rescode::TRM_OBJECTIVE_RANGE => TerminationStatus::ObjectiveLimit,
        Rescode::TRM_SERVER_MAX_MEMORY => TerminationStatus::MemoryLimit,
        Rescode::TRM_USER_CALLBACK | Rescode::TRM_LOST_RACE => TerminationStatus::Interrupted,
        Rescode::TRM_NUMERICAL_PROBLEM | Rescode::TRM_STALL => TerminationStatus::NumericError,
        _ if matches!(solution, Solsta::PRIM_FEAS | Solsta::PRIM_AND_DUAL_FEAS) => {
            TerminationStatus::Feasible
        }
        Rescode::OK => TerminationStatus::NotSolved,
        _ => TerminationStatus::Other(format!("MOSEK termination code {trm}")),
    }
}

fn bounds(lower: f64, upper: f64) -> (i32, f64, f64) {
    match (lower.is_finite(), upper.is_finite()) {
        (false, false) => (Boundkey::FR, 0.0, 0.0),
        (true, false) => (Boundkey::LO, lower, 0.0),
        (false, true) => (Boundkey::UP, 0.0, upper),
        (true, true) if lower.total_cmp(&upper).is_eq() => (Boundkey::FX, lower, upper),
        (true, true) => (Boundkey::RA, lower, upper),
    }
}

fn count_i32(count: usize, what: &str) -> Result<i32, SolverError> {
    i32::try_from(count).map_err(|_| overflow(what))
}

fn index_i32(index: usize, what: &str) -> Result<i32, SolverError> {
    i32::try_from(index).map_err(|_| overflow(what))
}

fn overflow(what: &str) -> SolverError {
    SolverError::Backend(format!("MOSEK: {what} count exceeds the native index range"))
}

fn backend(message: String) -> SolverError {
    SolverError::Backend(format!("MOSEK: {message}"))
}

#[cfg(feature = "benchmark-support")]
#[doc(hidden)]
#[expect(clippy::cast_precision_loss)]
#[allow(clippy::wildcard_imports)]
pub mod benchmark_support {
    use oximo_core::constraint::Relate;

    use super::*;

    /// Crossover candidate used only to size the preprocessing benchmark cases.
    pub const ROW_THRESHOLD: usize = 1_024;
    /// Crossover candidate used only to size the preprocessing benchmark cases.
    pub const SOC_THRESHOLD: usize = 1_024;

    pub fn row_model(rows: usize, degree: usize) -> Model {
        let model = Model::new("mosek_prepare_bench");
        let x = model.__var("x").lb(-5.0).ub(5.0).build();
        let y = model.__var("y").lb(-5.0).ub(5.0).build();
        let t = model.__var("t").lb(0.0).build();
        for i in 0..rows {
            let (lhs, rhs) = match degree {
                1 => (x + 2.0 * y - t, i as f64 + 10.0),
                2 => (x.powi(2) + y, i as f64 + 10.0),
                _ => (x.powi(2) + y.powi(2) - t.powi(2), 0.0),
            };
            model.__add_constraint_auto(lhs.le(rhs));
        }
        model.__minimize(t);
        model
    }

    pub fn explicit_soc_model(count: usize) -> Model {
        let model = Model::new("mosek_explicit_soc_bench");
        let x = model.__var("x").build();
        let y = model.__var("y").build();
        let t = model.__var("t").lb(0.0).build();
        for i in 0..count {
            model.add_soc_constraint(format!("soc{i}"), [x, y], t);
        }
        model.__minimize(t);
        model
    }

    /// Build and destroy a complete native task, without optimization.
    pub fn translate(model: &Model) -> Result<usize, SolverError> {
        let built = build_task(model, &MosekOptions::default())?;
        std::hint::black_box(&built);
        Ok(model.num_variables())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn separates_primal_and_dual_solution_statuses() {
        assert!(has_primal_solution(Solsta::OPTIMAL));
        assert!(has_dual_solution(Solsta::OPTIMAL));
        assert!(has_primal_solution(Solsta::PRIM_FEAS));
        assert!(!has_dual_solution(Solsta::PRIM_FEAS));
        assert!(!has_primal_solution(Solsta::DUAL_FEAS));
        assert!(has_dual_solution(Solsta::DUAL_FEAS));
        assert!(has_primal_solution(Solsta::PRIM_AND_DUAL_FEAS));
        assert!(has_dual_solution(Solsta::PRIM_AND_DUAL_FEAS));
        assert!(has_primal_solution(Solsta::INTEGER_OPTIMAL));
        assert!(!has_dual_solution(Solsta::INTEGER_OPTIMAL));
    }

    #[test]
    fn maps_native_limit_and_failure_codes() {
        assert_eq!(
            map_status(Solsta::PRIM_FEAS, Prosta::UNKNOWN, Rescode::TRM_MAX_TIME),
            TerminationStatus::TimeLimit
        );
        assert_eq!(
            map_status(Solsta::PRIM_FEAS, Prosta::UNKNOWN, Rescode::TRM_MAX_ITERATIONS),
            TerminationStatus::IterationLimit
        );
        assert_eq!(
            map_status(Solsta::PRIM_FEAS, Prosta::UNKNOWN, Rescode::TRM_MIO_NUM_BRANCHES),
            TerminationStatus::NodeLimit
        );
        assert_eq!(
            map_status(Solsta::PRIM_FEAS, Prosta::UNKNOWN, Rescode::TRM_MIO_NUM_RELAXS),
            TerminationStatus::WorkLimit
        );
        assert_eq!(
            map_status(Solsta::PRIM_FEAS, Prosta::UNKNOWN, Rescode::TRM_NUM_MAX_NUM_INT_SOLUTIONS),
            TerminationStatus::SolutionLimit
        );
        assert_eq!(
            map_status(Solsta::PRIM_FEAS, Prosta::UNKNOWN, Rescode::TRM_SERVER_MAX_MEMORY),
            TerminationStatus::MemoryLimit
        );
        assert_eq!(
            map_status(Solsta::UNKNOWN, Prosta::UNKNOWN, Rescode::TRM_USER_CALLBACK),
            TerminationStatus::Interrupted
        );
        assert_eq!(
            map_status(Solsta::UNKNOWN, Prosta::ILL_POSED, Rescode::TRM_NUMERICAL_PROBLEM),
            TerminationStatus::NumericError
        );
    }

    #[test]
    fn maps_problem_certificates() {
        assert_eq!(
            map_status(Solsta::UNKNOWN, Prosta::PRIM_INFEAS, Rescode::OK),
            TerminationStatus::Infeasible
        );
        assert_eq!(
            map_status(Solsta::UNKNOWN, Prosta::DUAL_INFEAS, Rescode::OK),
            TerminationStatus::Unbounded
        );
        assert_eq!(
            map_status(Solsta::UNKNOWN, Prosta::PRIM_INFEAS_OR_UNBOUNDED, Rescode::OK),
            TerminationStatus::InfeasibleOrUnbounded
        );
    }
}
