use crate::{
    ScipGeneratedVarId, ScipMapping,
    plugins::SharedState,
    translate::{Body, Snapshot},
};
use oximo_core::{Model, ModelKind, ObjectiveSense, VarId};
use oximo_expr::{EvalContext, ParamId, WithParameters, evaluate};
use oximo_solver::{
    DualStatus, PrimalStatus, SolutionPoint, SolverError, SolverResult, TerminationStatus,
};
use russcip::{Model as Native, ModelWithProblem, Solved, Status, WithSolutions, WithSolvingStats};
use rustc_hash::FxHashMap;
use std::time::Duration;

/// Extended results. Generated-column maps align with `result.solutions` and
/// use their own ID space.
#[derive(Clone, Debug)]
pub struct ScipSolveOutput {
    pub result: SolverResult,
    pub generated_solutions: Vec<FxHashMap<ScipGeneratedVarId, f64>>,
    pub generated_names: FxHashMap<ScipGeneratedVarId, String>,
    pub statistics_json: String,
}

pub(crate) fn termination(status: Status) -> TerminationStatus {
    use TerminationStatus as T;
    match status {
        Status::Optimal => T::Optimal,
        Status::Infeasible => T::Infeasible,
        Status::Unbounded => T::Unbounded,
        Status::Inforunbd => T::InfeasibleOrUnbounded,
        Status::TimeLimit => T::TimeLimit,
        Status::NodeLimit | Status::TotalNodeLimit | Status::StallNodeLimit => T::NodeLimit,
        Status::MemoryLimit => T::MemoryLimit,
        Status::SolutionLimit | Status::BestSolutionLimit => T::SolutionLimit,
        Status::PrimalLimit | Status::DualLimit => T::ObjectiveLimit,
        Status::RestartLimit => T::WorkLimit,
        Status::UserInterrupt | Status::Terminate => T::Interrupted,
        Status::GapLimit => T::Feasible,
        Status::Unknown => T::NotSolved,
    }
}
struct Values<'a>(&'a FxHashMap<VarId, f64>);
impl EvalContext for Values<'_> {
    fn var(&self, id: VarId) -> Option<f64> {
        self.0.get(&id).copied()
    }
    fn param(&self, _: ParamId) -> Option<f64> {
        None
    }
}

fn objective_at(
    model: &Model,
    primal: &FxHashMap<VarId, f64>,
    parameters: &oximo_solver::ParameterSnapshot,
) -> Result<f64, SolverError> {
    model
        .objective()
        .as_ref()
        .map_or(Ok(0.0), |o| {
            evaluate(&model.arena(), o.expr, &WithParameters::new(&Values(primal), parameters))
        })
        .map_err(crate::backend)
}

pub(crate) fn collect(
    native: &Native<Solved>,
    mapping: &ScipMapping,
    snapshot: &Snapshot,
    model: &Model,
    shared: &SharedState,
) -> Result<ScipSolveOutput, SolverError> {
    let parameters = &snapshot.parameters;
    let shared = shared.borrow();
    let status = termination(native.status());
    let infinity = native.real_param("numerics/infinity");
    let valid = |v: f64| v.is_finite() && v.abs() < infinity;
    let best_bound = Some(native.best_bound()).filter(|v| valid(*v));
    let native_vars = native.vars();
    let mut points = Vec::new();
    let mut generated_names = FxHashMap::default();
    for (i, g) in shared.generated.iter().enumerate() {
        generated_names.insert(ScipGeneratedVarId(i), g.name.clone());
    }
    for sol in native.get_sols().unwrap_or_default() {
        let primal = mapping
            .variables
            .iter()
            .enumerate()
            .map(|(i, v)| (VarId(u32::try_from(i).expect("variable id")), sol.val(v)))
            .collect::<FxHashMap<_, _>>();
        if !primal.values().all(|v| v.is_finite()) {
            continue;
        }
        let mut generated = FxHashMap::default();
        let mut extra = 0.0;
        for (i, g) in shared.generated.iter().enumerate() {
            let v = native_vars
                .iter()
                .find(|v| v.index() == g.index)
                .ok_or_else(|| crate::backend("priced column missing from solved model"))?;
            let value = sol.val(v);
            generated.insert(ScipGeneratedVarId(i), value);
            extra += g.objective * value;
        }
        let objective = objective_at(model, &primal, parameters)? + extra;
        if !objective.is_finite() {
            return Err(crate::backend("non-finite reconstructed objective"));
        }
        points.push((
            SolutionPoint {
                model_id: model.id(),
                primal,
                objective: Some(objective),
                parameters: parameters.clone(),
            },
            generated,
        ));
    }
    points.sort_by(|a, b| {
        let order = a.0.objective.unwrap().total_cmp(&b.0.objective.unwrap());
        if snapshot.sense == ObjectiveSense::Minimize { order } else { order.reverse() }
    });
    let (solutions, generated_solutions): (Vec<_>, Vec<_>) = points.into_iter().unzip();
    let primal_status = PrimalStatus::infer(&status, !solutions.is_empty());
    let primal = solutions.first().and_then(|s| s.objective);
    let gap = primal.zip(best_bound).and_then(|(p, d)| {
        if (p - d).abs() <= 1e-12 {
            Some(0.0)
        } else if p * d > 0.0 {
            Some((p - d).abs() / p.abs().min(d.abs()))
        } else {
            None
        }
    });
    let mut result = SolverResult {
        model_id: model.id(),
        termination: status,
        primal_status,
        solutions,
        best_bound,
        gap,
        solve_time: Duration::from_secs_f64(native.solving_time().max(0.0)),
        iterations: native.n_lp_iterations() as u64,
        node_count: Some(native.n_nodes() as u64),
        raw_status: Some(format!("{:?}", native.status()).into()),
        solver_name: Some("SCIP".into()),
        parameters: parameters.clone(),
        ..Default::default()
    };
    // Require a complete valid original LP certificate. A presolved-away row
    // or a node relaxation must not be presented as an original-model dual.
    if snapshot.kind == ModelKind::LP
        && snapshot.columns.iter().all(|column| column.domain.semi_threshold().is_none())
        && result.termination == TerminationStatus::Optimal
        && shared.generated.is_empty()
        && let Some(point) = result.best()
    {
        for sign in [1.0, -1.0] {
            if valid_prices(snapshot, point, &shared, sign) {
                result.dual = shared.dual.iter().map(|(&id, &v)| (id, sign * v)).collect();
                result.reduced_costs =
                    shared.reduced.iter().map(|(&id, &v)| (id, sign * v)).collect();
                result.dual_status = DualStatus::FeasiblePoint;
                break;
            }
        }
    }
    Ok(ScipSolveOutput {
        result: result.with_semantic_dual_defaults(model),
        generated_solutions,
        generated_names,
        statistics_json: native.stats_json(),
    })
}

fn valid_prices(
    snapshot: &Snapshot,
    point: &SolutionPoint,
    s: &crate::plugins::Shared,
    sign: f64,
) -> bool {
    let Body::Linear(costs, _) = &snapshot.objective else {
        return false;
    };
    if s.lp_objective.is_none() || s.reduced.len() != snapshot.columns.len() {
        return false;
    }
    let tol = 1e-6;
    let sense = if snapshot.sense == ObjectiveSense::Minimize { 1.0 } else { -1.0 };
    let mut residual = vec![0.0; snapshot.columns.len()];
    for &(id, c) in costs {
        residual[id.index()] += c;
    }
    for (i, row) in snapshot.rows.iter().enumerate() {
        if !row.active {
            continue;
        }
        let Body::Linear(cs, k) = &row.body else {
            return false;
        };
        let Some(&price) = s.dual.get(&crate::translate::id(i)) else {
            return false;
        };
        let price = sign * price;
        let activity = k + cs.iter().map(|(v, c)| point.primal[v] * c).sum::<f64>();
        if price * sense > tol && (!row.lower.is_finite() || (activity - row.lower).abs() > tol) {
            return false;
        }
        if price * sense < -tol && (!row.upper.is_finite() || (activity - row.upper).abs() > tol) {
            return false;
        }
        for &(v, c) in cs {
            residual[v.index()] -= price * c;
        }
    }
    for c in &snapshot.columns {
        let x = point.primal[&c.id];
        let rc = sign * s.reduced[&c.id];
        if s.lp_values.get(&c.id).is_none_or(|v| (x - v).abs() > tol) {
            return false;
        }
        if rc * sense > tol && (x - c.lb).abs() > tol {
            return false;
        }
        if rc * sense < -tol && (x - c.ub).abs() > tol {
            return false;
        }
        if (residual[c.id.index()] - rc).abs() > tol * (1.0 + rc.abs()) {
            return false;
        }
    }
    true
}
