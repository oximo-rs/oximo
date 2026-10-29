use crate::{
    ScipGeneratedVarId, ScipMapping,
    plugins::SharedState,
    translate::{Body, Snapshot},
};
use oximo_core::{Model, ModelKind, ObjectiveSense, VarId};
use oximo_expr::{WithParameters, evaluate};
use oximo_solver::{
    DualStatus, PrimalStatus, SolutionPoint, SolverError, SolverResult, TerminationStatus,
};
use oximo_solver::{SolutionValueId, SolutionValues};
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
fn objective_at(
    model: &Model,
    primal: &SolutionValues<VarId>,
    parameters: &oximo_solver::ParameterSnapshot,
) -> Result<f64, SolverError> {
    model
        .objective()
        .as_ref()
        .map_or(Ok(0.0), |o| {
            evaluate(&model.arena(), o.expr, &WithParameters::new(primal, parameters))
        })
        .map_err(crate::backend)
}

pub(crate) fn collect(
    native: &Native<Solved>,
    mapping: &ScipMapping,
    snapshot: &Snapshot,
    model: &Model,
    shared: &SharedState,
    parameters: &oximo_solver::ParameterSnapshot,
) -> Result<ScipSolveOutput, SolverError> {
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
        let primal =
            SolutionValues::from_dense(mapping.variables.iter().map(|v| sol.val(v)).collect());
        if !primal.values().all(f64::is_finite) {
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
    recover_prices(snapshot, &mut result, &shared);
    Ok(ScipSolveOutput {
        result: result.with_semantic_dual_defaults(model),
        generated_solutions,
        generated_names,
        statistics_json: native.stats_json(),
    })
}

fn recover_prices(snapshot: &Snapshot, result: &mut SolverResult, shared: &crate::plugins::Shared) {
    // Require a complete valid original LP certificate. A presolved-away row
    // or a node relaxation must not be presented as an original-model dual.
    if snapshot.kind == ModelKind::LP
        && snapshot.columns.iter().all(|column| column.domain.semi_threshold().is_none())
        && result.termination == TerminationStatus::Optimal
        && shared.generated.is_empty()
        && let Some(point) = result.best()
    {
        for sign in [1.0, -1.0] {
            if valid_prices(snapshot, point, shared, sign) {
                let (dual, skipped_rows) = scaled_prices(snapshot.rows.len(), &shared.dual, sign);
                let (reduced, skipped_columns) =
                    scaled_prices(snapshot.columns.len(), &shared.reduced, sign);
                result.dual = dual;
                result.reduced_costs = reduced;
                if skipped_rows + skipped_columns != 0 {
                    result.raw_log = Some(format!(
                        "Ignored {} native price coordinates outside captured slot counts; known coordinates passed certificate checks.",
                        skipped_rows + skipped_columns
                    ));
                }
                result.dual_status = DualStatus::FeasiblePoint;
                break;
            }
        }
    }
}

fn scaled_prices<I: SolutionValueId>(
    slot_count: usize,
    prices: &FxHashMap<I, f64>,
    sign: f64,
) -> (SolutionValues<I>, usize) {
    let mut values = SolutionValues::default();
    let mut ignored = 0;
    for (&id, &value) in prices {
        // Native/plugin prices may contain coordinates outside this snapshot.
        if id.index() >= slot_count {
            ignored += 1;
            continue;
        }
        if values.slot_count() == 0 {
            values = SolutionValues::missing(slot_count);
        }
        let _ = values.try_set(id, Some(sign * value));
    }
    (values, ignored)
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
    if s.lp_objective.is_none_or(|value| !value.is_finite()) {
        return false;
    }
    let tol = 1e-6;
    let sense = if snapshot.sense == ObjectiveSense::Minimize { 1.0 } else { -1.0 };
    let mut residual = vec![0.0; snapshot.columns.len()];
    for &(id, c) in costs {
        let Some(entry) = residual.get_mut(id.index()) else {
            return false;
        };
        *entry += c;
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
        if !price.is_finite() {
            return false;
        }
        let Some(activity) =
            cs.iter().try_fold(*k, |sum, &(v, c)| point.primal.get(v).map(|x| sum + x * c))
        else {
            return false;
        };
        if !activity.is_finite() {
            return false;
        }
        if price * sense > tol && (!row.lower.is_finite() || (activity - row.lower).abs() > tol) {
            return false;
        }
        if price * sense < -tol && (!row.upper.is_finite() || (activity - row.upper).abs() > tol) {
            return false;
        }
        for &(v, c) in cs {
            let Some(entry) = residual.get_mut(v.index()) else {
                return false;
            };
            *entry -= price * c;
        }
    }
    for c in &snapshot.columns {
        let (Some(x), Some(&reduced), Some(&stationarity)) =
            (point.primal.get(c.id), s.reduced.get(&c.id), residual.get(c.id.index()))
        else {
            return false;
        };
        let rc = sign * reduced;
        if !x.is_finite() || !rc.is_finite() || !stationarity.is_finite() {
            return false;
        }
        if s.lp_values.get(&c.id).is_none_or(|v| !v.is_finite() || (x - v).abs() > tol) {
            return false;
        }
        if rc * sense > tol && (x - c.lb).abs() > tol {
            return false;
        }
        if rc * sense < -tol && (x - c.ub).abs() > tol {
            return false;
        }
        if (stationarity - rc).abs() > tol * (1.0 + rc.abs()) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximo_core::prelude::*;

    #[test]
    fn native_extra_prices_are_skipped_and_missing_values_reject_certificates() {
        let model = Model::new("native price boundaries");

        variable!(model, x >= 0.0);
        let row = constraint!(model, row, x >= 0.0);
        objective!(model, Min, x);

        let snapshot = Snapshot::prepare(&model).unwrap();
        let point = SolutionPoint::new(&model, SolutionValues::from_dense(vec![0.0]), Some(0.0));
        let id = x.var_id().unwrap();
        let mut shared = crate::plugins::Shared::default();
        shared.lp_objective = Some(0.0);
        shared.lp_values.insert(id, 0.0);
        shared.reduced.insert(id, 0.0);
        shared.reduced.insert(VarId(999), 8.0);
        shared.dual.insert(row.row_id(), 1.0);
        shared.dual.insert(oximo_core::ConstraintId(999), 9.0);
        assert!(valid_prices(&snapshot, &point, &shared, 1.0));

        let mut result = SolverResult::for_model(&model);
        result.termination = TerminationStatus::Optimal;
        result.solutions.push(point.clone());
        recover_prices(&snapshot, &mut result, &shared);
        assert_eq!(result.dual_status, DualStatus::FeasiblePoint);
        assert_eq!(result.dual.get(row.row_id()), Some(1.0));
        assert_eq!(result.reduced_costs.get(id), Some(0.0));
        assert!(!result.dual.contains(oximo_core::ConstraintId(999)));
        assert!(!result.reduced_costs.contains(VarId(999)));
        assert!(result.raw_log.as_deref().unwrap().contains("Ignored 2"));

        let mut incomplete = point.clone();
        incomplete.primal.clear();
        assert!(!valid_prices(&snapshot, &incomplete, &shared, 1.0));

        shared.reduced.remove(&id);
        assert!(!valid_prices(&snapshot, &point, &shared, 1.0));
    }
}
