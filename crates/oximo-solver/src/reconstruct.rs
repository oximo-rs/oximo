//! Reversible coordinate changes and the common result contract.
//!
//! Adapters must establish native solution quality before supplying points,
//! duals or global bounds.

use std::collections::HashMap;
use std::hash::{BuildHasher, Hash};

use oximo_core::VarId;
use rustc_hash::FxHashMap;

use crate::{DualStatus, PrimalStatus, SolverResult, TerminationStatus};

/// One native row's contribution to an original multiplier. Store `None` in
/// the native row map when no reversible multiplier transformation is known.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DualProjection<K> {
    pub source: K,
    pub scale: f64,
}

impl<K: Copy + Eq + Hash> DualProjection<K> {
    pub fn accumulate<S: BuildHasher>(
        self,
        map: &mut HashMap<K, f64, S>,
        value: f64,
        objective_sign: f64,
    ) {
        accumulate_dual(map, self.source, value, self.scale * objective_sign);
    }
}

/// Maps a native objective or bound to the original model convention.
/// A native solver that already includes the constant must use offset zero.
#[derive(Clone, Copy, Debug)]
pub struct ObjectiveTransform {
    pub sign: f64,
    pub offset: f64,
}

impl ObjectiveTransform {
    pub fn restore(self, value: f64) -> Option<f64> {
        let mapped = self.sign * value + self.offset;
        (value.is_finite() && mapped.is_finite()).then_some(mapped)
    }
}

/// Project a native column vector, excluding auxiliaries (`None`). Every
/// original variable must occur exactly once and have a finite value.
pub fn project_primal(
    values: &[f64],
    columns: &[Option<VarId>],
    num_variables: usize,
) -> Option<FxHashMap<VarId, f64>> {
    if values.len() != columns.len() {
        return None;
    }
    let mut primal = FxHashMap::default();
    for (&value, &column) in values.iter().zip(columns) {
        if let Some(id) = column
            && (id.index() >= num_variables
                || !value.is_finite()
                || primal.insert(id, value).is_some())
        {
            return None;
        }
    }
    (primal.len() == num_variables).then_some(primal)
}

/// Use when native columns are original variables in model order.
pub fn project_dense_primal(values: &[f64], num_variables: usize) -> Option<FxHashMap<VarId, f64>> {
    if values.len() != num_variables {
        return None;
    }
    values
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let id = VarId(u32::try_from(index).ok()?);
            value.is_finite().then_some((id, value))
        })
        .collect()
}

/// Accumulate a known native multiplier with its adapter-supplied coordinate
/// factor. Missing values must not call this helper with a fabricated zero.
pub fn accumulate_dual<K: Eq + Hash, S: BuildHasher>(
    map: &mut HashMap<K, f64, S>,
    id: K,
    value: f64,
    factor: f64,
) {
    *map.entry(id).or_insert(0.0) += value * factor;
}

/// Overflow-resistant symmetric relative gap in original objective units.
pub fn relative_gap(primal: Option<f64>, bound: Option<f64>) -> Option<f64> {
    let (p, b) = (primal?, bound?);
    if !p.is_finite() || !b.is_finite() {
        return None;
    }
    let scale = p.abs().max(b.abs()) + 1e-10;
    Some((p / scale - b / scale).abs())
}

/// Enforce the public result contract after the adapter has established native
/// evidence and restored coordinates. Does not infer feasibility from a stop
/// reason, or re-check feasibility using an unrelated common tolerance.
/// An explicit `FeasiblePoint` status caps the point's quality even if native
/// termination was `Optimal`. Invalid incumbents cause the same downgrade for
/// surviving pool points. Normalizing an already normalized result is safe.
pub fn normalize_result(mut result: SolverResult, num_variables: usize) -> SolverResult {
    let mut first = true;
    let mut lost_incumbent = false;
    result.solutions.retain_mut(|point| {
        point.primal.retain(|id, value| id.index() < num_variables && value.is_finite());
        point.objective = point.objective.filter(|v| v.is_finite());
        let valid = point.primal.len() == num_variables;
        if first {
            lost_incumbent = !valid;
            first = false;
        }
        valid
    });
    // A surviving pool point does not inherit the discarded incumbent's
    // optimality certificate, so we preserve an explicit feasibility-only
    // status on subsequent normalization too.
    let optimal_point = result.termination == TerminationStatus::Optimal
        && !lost_incumbent
        && result.primal_status != PrimalStatus::FeasiblePoint;
    result.primal_status = if result.solutions.is_empty() {
        PrimalStatus::NoSolution
    } else if optimal_point {
        PrimalStatus::OptimalPoint
    } else {
        PrimalStatus::FeasiblePoint
    };
    if lost_incumbent {
        result.dual_status = DualStatus::Unknown;
        result.gap = None;
    }
    if result.dual_status != DualStatus::FeasiblePoint {
        result.dual.clear();
        result.soc_dual.clear();
        result.reduced_costs.clear();
    }
    result.dual.retain(|_, v| v.is_finite());
    result.soc_dual.retain(|_, v| v.is_finite() && *v >= 0.0);
    result.reduced_costs.retain(|_, v| v.is_finite());
    result.best_bound = result.best_bound.filter(|v| v.is_finite());
    if result.primal_status == PrimalStatus::OptimalPoint && result.best_bound.is_none() {
        result.best_bound = result.objective();
    }
    result.gap = result.gap.filter(|v| v.is_finite() && *v >= 0.0);
    result
}
