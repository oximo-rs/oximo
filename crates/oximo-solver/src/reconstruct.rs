//! Reversible coordinate changes and the common result contract.
//!
//! Adapters must establish native solution quality before supplying points,
//! duals or global bounds. [`SourceProjection`] composes these helpers for
//! callers that solve an independently transformed model and need results in
//! the original model's coordinates.

use std::collections::HashMap;
use std::hash::{BuildHasher, Hash};

use oximo_core::{Model, ModelId, VarId};
use rustc_hash::FxHashMap;

use crate::{DualStatus, PrimalStatus, SolutionPoint, SolverResult, TerminationStatus};

/// Finalize a backend result in the source model's coordinates.
/// Applies semantic dual safeguards and the common result contract in one pass.
/// The result and its points retain the parameter snapshot captured for the solve.
/// Pass the variable count captured during translation, even if the source model
/// has since gained variables (for example, from a solver callback).
///
/// # Panics
/// Panics for a foreign result, foreign points, or conflicting parameter snapshots.
#[inline]
pub fn finalize_result(
    result: SolverResult,
    model: &oximo_core::Model,
    num_variables: usize,
) -> SolverResult {
    normalize_result(result, num_variables).with_semantic_dual_defaults(model)
}

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
/// Detached points with an unassigned model identity and empty parameters inherit
/// the result's snapshot. Other points must already have the same parameter bits.
/// Assigned point identities must match the result's model identity.
///
/// # Panics
/// Panics if a point's model identity or captured parameters conflict with the result.
/// Use [`SolverResult::for_model`] and [`SolverResult::add_solution`] to assemble
/// points from a single captured parameter state.
fn normalize_result(mut result: SolverResult, num_variables: usize) -> SolverResult {
    for (index, point) in result.solutions.iter().enumerate() {
        assert!(
            point.model_id == oximo_core::ModelId::UNASSIGNED || point.model_id == result.model_id,
            "model identity mismatch for solution {index}; assigned points must belong to the result's model"
        );
        let detached = point.model_id == oximo_core::ModelId::UNASSIGNED
            && point.parameters.values().is_empty();
        assert!(
            detached || point.parameters.same_bits(&result.parameters),
            "parameter snapshot mismatch for solution {index}; use SolverResult::for_model and add_solution to capture consistent parameters"
        );
    }
    let model_id = result.model_id;
    for point in &mut result.solutions {
        point.model_id = model_id;
        point.parameters = result.parameters.clone();
    }
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
        result.semantic_dual.clear();
        result.dual.clear();
        result.soc_dual.clear();
        result.psd_dual.clear();
        result.reduced_costs.clear();
    }
    for value in result.semantic_dual.values_mut() {
        if value.is_some_and(|v| !v.is_finite()) {
            *value = None;
        }
    }
    result.dual.retain(|_, v| v.is_finite());
    result.soc_dual.retain(|_, v| v.is_finite() && *v >= 0.0);
    result.psd_dual.retain(|_, matrix| matrix.upper_triangle().iter().all(|v| v.is_finite()));
    result.reduced_costs.retain(|_, v| v.is_finite());
    result.best_bound = result.best_bound.filter(|v| v.is_finite());
    if result.primal_status == PrimalStatus::OptimalPoint && result.best_bound.is_none() {
        result.best_bound = result.objective();
    }
    result.gap = result.gap.filter(|v| v.is_finite() && *v >= 0.0);
    if result.primal_status == PrimalStatus::NoSolution {
        result.gap = None;
    }
    result
}

/// Coordinate counts captured from a source model before a reformulation.
///
/// Entities appended by a reformulation have IDs at or above these bounds.
/// For an in-place reformulation, capture this before modifying the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceCoordinates {
    pub variables: usize,
    /// Number of algebraic rows (`ConstraintId` indexes).
    pub algebraic_constraints: usize,
    pub soc_constraints: usize,
    pub psd_constraints: usize,
}

impl SourceCoordinates {
    /// Capture the source model's current coordinate counts.
    #[must_use]
    pub fn capture(model: &Model) -> Self {
        Self {
            variables: model.num_variables(),
            algebraic_constraints: model.constraints().algebraic().len(),
            soc_constraints: model.num_soc_constraints(),
            psd_constraints: model.num_psd_constraints(),
        }
    }
}

/// Errors from projecting a transformed-model result back to its source.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum ProjectionError {
    #[error("source {kind} coordinates ({count}) exceed the source model's {available}")]
    CoordinatesExceedSource { kind: &'static str, count: usize, available: usize },
    #[error("result belongs to model {actual:?}, expected transformed model {expected:?}")]
    ForeignResult { expected: ModelId, actual: ModelId },
}

type SourceObjective<'a> = Box<dyn Fn(&SolutionPoint) -> Option<f64> + 'a>;

/// Projects a result produced on a transformed model back into the source
/// model's coordinates.
///
/// The projection:
/// - drops generated variables, rows, SOC/PSD cones, and reduced costs;
/// - rebinds the result and its points to the source model identity;
/// - preserves the solve-time parameter snapshot;
/// - optionally restores the objective convention and re-evaluates the source
///   objective on each projected point;
/// - applies [`finalize_result`] so the usual result contract holds.
///
/// Generated multipliers are dropped rather than reinterpreted in source
/// coordinates. Semantic duals remain subject to the usual explicit-mapping
/// rules; an unavailable value stays unavailable.
pub struct SourceProjection<'a> {
    source: &'a Model,
    coordinates: SourceCoordinates,
    objective: ObjectiveTransform,
    reevaluate: Option<SourceObjective<'a>>,
    transformed: Option<&'a Model>,
}

impl<'a> SourceProjection<'a> {
    /// Capture the source counts now.
    ///
    /// Use [`Self::with_coordinates`] when an in-place reformulation has already
    /// appended generated entities.
    #[must_use]
    pub fn capture(source: &'a Model) -> Self {
        Self::with_coordinates(source, SourceCoordinates::capture(source))
    }

    /// Use counts captured before the model was transformed.
    #[must_use]
    pub fn with_coordinates(source: &'a Model, coordinates: SourceCoordinates) -> Self {
        Self {
            source,
            coordinates,
            objective: ObjectiveTransform { sign: 1.0, offset: 0.0 },
            reevaluate: None,
            transformed: None,
        }
    }

    /// Require the result to belong to this transformed model.
    #[must_use]
    pub fn from_transformed(mut self, transformed: &'a Model) -> Self {
        self.transformed = Some(transformed);
        self
    }

    /// Restore the objective convention used by the source declaration.
    ///
    /// This transforms solution objectives and the best bound. It does not
    /// change duals, which adapters must already have mapped into source
    /// coordinates and sign.
    #[must_use]
    pub fn with_objective(mut self, transform: ObjectiveTransform) -> Self {
        self.objective = transform;
        self
    }

    /// Replace each projected objective with the source expression evaluated at
    /// the projected point.
    ///
    /// The callback receives source coordinates and the solve-time parameter
    /// snapshot. Returning `None` reports the value as unavailable instead of
    /// retaining a native epigraph value. When supplied, this takes precedence
    /// over [`Self::with_objective`] for solution objectives.
    #[must_use]
    pub fn reevaluate_objective<F>(mut self, evaluate: F) -> Self
    where
        F: Fn(&SolutionPoint) -> Option<f64> + 'a,
    {
        self.reevaluate = Some(Box::new(evaluate));
        self
    }

    /// Project the result into the source model's coordinates.
    ///
    /// # Errors
    /// Returns [`ProjectionError`] when the captured coordinates exceed the
    /// source model or when [`Self::from_transformed`] was set and the result
    /// belongs to another model.
    pub fn project(self, result: SolverResult) -> Result<SolverResult, ProjectionError> {
        let Self { source, coordinates, objective, reevaluate, transformed } = self;
        check_source_coordinates(source, coordinates)?;
        if let Some(transformed) = transformed
            && result.model_id != transformed.id()
        {
            return Err(ProjectionError::ForeignResult {
                expected: transformed.id(),
                actual: result.model_id,
            });
        }

        let mut result = result;

        // Drop generated entities before rebinding.
        for point in &mut result.solutions {
            point.primal.retain(|id, _| id.index() < coordinates.variables);
        }
        result.dual.retain(|id, _| id.index() < coordinates.algebraic_constraints);
        result.soc_dual.retain(|id, _| id.index() < coordinates.soc_constraints);
        result.psd_dual.retain(|id, _| id.index() < coordinates.psd_constraints);
        result.reduced_costs.retain(|id, _| id.index() < coordinates.variables);

        // Restore objective units on the native certificate.
        for point in &mut result.solutions {
            point.objective = point.objective.and_then(|value| objective.restore(value));
        }
        result.best_bound = result.best_bound.and_then(|value| objective.restore(value));

        // Rebind provenance before evaluating source expressions.
        result.model_id = source.id();
        for point in &mut result.solutions {
            point.model_id = source.id();
            if let Some(evaluate) = &reevaluate {
                point.objective = evaluate(point);
            }
        }

        Ok(finalize_result(result, source, coordinates.variables))
    }
}

impl std::fmt::Debug for SourceProjection<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SourceProjection")
            .field("source", &self.source.id())
            .field("coordinates", &self.coordinates)
            .field("objective", &self.objective)
            .field("reevaluate", &self.reevaluate.is_some())
            .field("transformed", &self.transformed.map(Model::id))
            .finish()
    }
}

fn check_source_coordinates(
    model: &Model,
    coordinates: SourceCoordinates,
) -> Result<(), ProjectionError> {
    for (kind, count, available) in [
        ("variable", coordinates.variables, model.num_variables()),
        ("algebraic row", coordinates.algebraic_constraints, model.constraints().algebraic().len()),
        ("SOC constraint", coordinates.soc_constraints, model.num_soc_constraints()),
        ("PSD constraint", coordinates.psd_constraints, model.num_psd_constraints()),
    ] {
        if count > available {
            return Err(ProjectionError::CoordinatesExceedSource { kind, count, available });
        }
    }
    Ok(())
}
