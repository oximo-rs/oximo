//! Reversible coordinate changes and the common result contract.
//!
//! Adapters must establish native solution quality before supplying points,
//! duals or global bounds. [`SourceProjection`] composes these helpers for
//! callers that solve an independently transformed model and need results in
//! the original model's coordinates.

use std::borrow::Cow;
use std::fmt::Write as FmtWrite;

use oximo_core::{
    ConstraintId, Model, ModelId, ParameterDeclarations, SemanticConstraintId, SocConstraint,
    SocConstraintId, VarId,
};
use rustc_hash::FxHashSet;

use crate::{
    DualStatus, PrimalStatus, SolutionIndexError, SolutionPoint, SolutionValueId, SolutionValues,
    SolverResult, TerminationStatus,
};

/// Require at least one usable multiplier for each active constraint group.
/// Backends select the algebraic row IDs; SOC IDs are indices into `socs`.
/// Algebraic multipliers must be finite, and SOC multipliers must also be nonnegative.
///
/// # Panics
/// Panics if an active SOC index exceeds `u32::MAX`.
pub fn require_active_dual_evidence(
    result: &mut SolverResult,
    active_rows: impl IntoIterator<Item = ConstraintId>,
    socs: &[SocConstraint],
) {
    if result.dual_status != DualStatus::FeasiblePoint {
        return;
    }

    let mut active_rows = active_rows.into_iter().peekable();
    let missing_rows = active_rows.peek().is_some()
        && (result.dual.is_empty()
            || !active_rows.any(|id| result.dual.get(id).is_some_and(f64::is_finite)));
    let mut active_socs = socs.iter().enumerate().filter(|(_, soc)| soc.active).peekable();
    let missing_socs = active_socs.peek().is_some()
        && (result.soc_dual.is_empty()
            || !active_socs.any(|(index, _)| {
                result
                    .soc_dual
                    .get(SocConstraintId(u32::try_from(index).unwrap()))
                    .is_some_and(|value| value.is_finite() && value >= 0.0)
            }));

    if missing_rows || missing_socs {
        result.dual_status = DualStatus::Unknown;
        let log = result.raw_log.get_or_insert_with(String::new);
        if !log.is_empty() {
            log.push('\n');
        }

        let group = match (missing_rows, missing_socs) {
            (true, true) => "algebraic and SOC",
            (true, false) => "algebraic",
            _ => "SOC",
        };
        let _ = write!(
            log,
            "No multiplier captured for active {group} constraints; dual status is Unknown."
        );
    }
}

/// Finalize a backend result in the source model's coordinates.
/// Applies semantic dual safeguards and the common result contract in one pass.
/// The result and its points retain the parameter snapshot captured for the solve.
/// Pass the variable count captured during translation, even if the source model
/// has since gained variables (for example, from a solver callback).
///
/// # Panics
/// Panics if the captured variable count exceeds the model's current count,
/// or for a foreign result, foreign points, or conflicting parameter snapshots.
#[inline]
#[track_caller]
pub fn finalize_result(
    result: SolverResult,
    model: &oximo_core::Model,
    num_variables: usize,
) -> SolverResult {
    let coordinates =
        SourceCoordinates { variables: num_variables, ..SourceCoordinates::capture(model) };
    finalize_result_with_coordinates(result, model, coordinates)
}

/// Finalize using variable, algebraic-row, SOC, and PSD counts captured before
/// solving.
///
/// Counts describe ID prefixes.
/// Use [`SourceProjection`] for interleaved source and generated IDs or
/// a fallible coordinate/provenance check.
///
/// # Panics
/// Panics if captured counts exceed the model's current counts, or for a foreign
/// result, foreign points, or conflicting parameter snapshots.
#[inline]
#[track_caller]
pub fn finalize_result_with_coordinates(
    result: SolverResult,
    model: &Model,
    coordinates: SourceCoordinates,
) -> SolverResult {
    let plan = SourceProjectionPlan::new(coordinates);
    check_source_coordinates(model, &plan).unwrap_or_else(|error| panic!("{error}"));
    finalize_result_retaining(
        result,
        model,
        &plan.variables,
        &plan.retained,
        ProjectionCoordinates {
            rows: &plan.rows,
            soc: coordinates.soc_constraints,
            psd: coordinates.psd_constraints,
        },
    )
}

/// Membership test for one source coordinate registry.
#[derive(Clone, Debug)]
enum SourceIndices {
    /// Source IDs are the prefix `0..count`.
    Prefix(usize),
    /// Source IDs are interleaved with generated IDs.
    Explicit { ids: FxHashSet<usize>, upper_bound: usize },
}

impl SourceIndices {
    fn explicit(indices: impl IntoIterator<Item = usize>) -> Self {
        let ids: FxHashSet<_> = indices.into_iter().collect();
        let upper_bound = ids.iter().max().map_or(0, |id| id.saturating_add(1));
        Self::Explicit { ids, upper_bound }
    }

    fn len(&self) -> usize {
        match self {
            Self::Prefix(count) => *count,
            Self::Explicit { ids, .. } => ids.len(),
        }
    }

    fn contains(&self, index: usize) -> bool {
        match self {
            Self::Prefix(count) => index < *count,
            Self::Explicit { ids, .. } => ids.contains(&index),
        }
    }

    fn upper_bound(&self) -> usize {
        match self {
            Self::Prefix(count) => *count,
            Self::Explicit { upper_bound, .. } => *upper_bound,
        }
    }
}

#[derive(Clone, Copy)]
struct ProjectionCoordinates<'a> {
    rows: &'a SourceIndices,
    soc: usize,
    psd: usize,
}

impl ProjectionCoordinates<'_> {
    fn includes_semantic(self, id: SemanticConstraintId) -> bool {
        match id {
            SemanticConstraintId::Algebraic(row) => self.rows.contains(row.index()),
            SemanticConstraintId::SecondOrderCone(cone) => cone.index() < self.soc,
            SemanticConstraintId::PositiveSemidefinite(cone) => cone.index() < self.psd,
            SemanticConstraintId::SpecialOrderedSet(_) | SemanticConstraintId::Indicator(_) => true,
        }
    }
}

#[inline]
fn finalize_result_retaining(
    result: SolverResult,
    model: &oximo_core::Model,
    variables: &SourceIndices,
    retained: &FxHashSet<VarId>,
    projection: ProjectionCoordinates<'_>,
) -> SolverResult {
    let result = normalize_result(result, variables, retained, projection);
    result.with_semantic_dual_defaults_filtered(model, |id| projection.includes_semantic(id))
}

/// One native row's contribution to an original multiplier. Store `None` in
/// the native row map when no reversible multiplier transformation is known.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DualProjection<K> {
    pub source: K,
    pub scale: f64,
}

impl<K: SolutionValueId> DualProjection<K> {
    /// Accumulate a mapped native multiplier into fixed-slot storage.
    ///
    /// # Panics
    /// Panics if the source coordinate is outside the destination slot count.
    /// Adapters processing native output should use `try_accumulate` instead.
    #[track_caller]
    pub fn accumulate(self, values: &mut SolutionValues<K>, value: f64, objective_sign: f64) {
        accumulate_dual(values, self.source, value, self.scale * objective_sign);
    }

    /// Accumulate a mapped native multiplier without panicking on a count mismatch.
    ///
    /// # Errors
    /// Returns an out-of-range coordinate error without changing stored values.
    pub fn try_accumulate(
        self,
        values: &mut SolutionValues<K>,
        value: f64,
        objective_sign: f64,
    ) -> Result<(), SolutionIndexError> {
        try_accumulate_dual(values, self.source, value, self.scale * objective_sign)
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
) -> Option<SolutionValues<VarId>> {
    if values.len() != columns.len() || num_variables > columns.len() {
        return None;
    }

    if values.len() == num_variables
        && columns.iter().enumerate().all(|(index, id)| id.is_some_and(|id| id.index() == index))
    {
        return project_dense_primal(values, num_variables);
    }

    let mut primal = SolutionValues::missing(num_variables);
    for (&value, &column) in values.iter().zip(columns) {
        if let Some(id) = column
            && (id.index() >= num_variables
                || !value.is_finite()
                || primal.set(id, Some(value)).is_some())
        {
            return None;
        }
    }
    (primal.len() == num_variables).then_some(primal)
}

/// Use when native columns are original variables in model order.
pub fn project_dense_primal(values: &[f64], num_variables: usize) -> Option<SolutionValues<VarId>> {
    if values.len() != num_variables {
        return None;
    }
    if !values.iter().all(|value| value.is_finite()) {
        return None;
    }
    Some(SolutionValues::from_dense(values.to_vec()))
}

/// Project an owned native vector in model order without copying its allocation.
/// Every source variable must have a finite value and the slot count must match.
pub fn project_owned_primal(
    values: Vec<f64>,
    num_variables: usize,
) -> Option<SolutionValues<VarId>> {
    if values.len() != num_variables || !values.iter().all(|value| value.is_finite()) {
        return None;
    }
    Some(SolutionValues::from_dense(values))
}

/// Accumulate a known native multiplier with its adapter-supplied coordinate
/// factor. Missing multipliers must not be replaced by fabricated zeros.
///
/// # Panics
/// Panics if `id` is outside the fixed slot count. Native adapters should use
/// [`try_accumulate_dual`] and handle a count mismatch deliberately.
#[track_caller]
pub fn accumulate_dual<K: SolutionValueId>(
    values: &mut SolutionValues<K>,
    id: K,
    value: f64,
    factor: f64,
) {
    try_accumulate_dual(values, id, value, factor).unwrap_or_else(|error| {
        panic!("solution ID {id:?} outside slot count {}", error.slot_count)
    });
}

/// Fallible native multiplier accumulation.
///
/// # Errors
/// Returns [`SolutionIndexError`] for a coordinate outside the fixed slot count.
pub fn try_accumulate_dual<K: SolutionValueId>(
    values: &mut SolutionValues<K>,
    id: K,
    value: f64,
    factor: f64,
) -> Result<(), SolutionIndexError> {
    values.try_accumulate(id, value * factor)
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
#[inline]
fn normalize_result(
    result: SolverResult,
    source_variables: &SourceIndices,
    retained: &FxHashSet<VarId>,
    projection: ProjectionCoordinates<'_>,
) -> SolverResult {
    validate_result_points(&result).unwrap_or_else(|error| panic!("{error}"));
    normalize_validated_result(
        ValidatedResult(result),
        source_variables,
        retained,
        projection,
        |_| {},
    )
}

/// Proof that point provenance was checked before any rebinding or callbacks.
struct ValidatedResult(SolverResult);

#[inline]
fn normalize_validated_result(
    ValidatedResult(mut result): ValidatedResult,
    source_variables: &SourceIndices,
    retained: &FxHashSet<VarId>,
    projection: ProjectionCoordinates<'_>,
    mut prepare_point: impl FnMut(&mut SolutionPoint),
) -> SolverResult {
    let model_id = result.model_id;
    let parameter_count = result.parameters.values().len();
    let mut first = true;
    let mut lost_incumbent = false;
    let expected = source_variables.len();
    let primal_extent = retained
        .iter()
        .map(|id| id.index().saturating_add(1))
        .fold(source_variables.upper_bound(), usize::max);
    result.solutions.retain_mut(|point| {
        let detached =
            point.model_id == ModelId::UNASSIGNED && point.parameters.values().is_empty();
        point.model_id = model_id;

        // Projection may also have removed a transformed-only parameter suffix.
        if detached || point.parameters.values().len() > parameter_count {
            point.parameters = result.parameters.clone();
        }
        point.primal.truncate(primal_extent);
        let source_count =
            if matches!(source_variables, SourceIndices::Prefix(_)) && retained.is_empty() {
                // Truncation already establishes membership for a source prefix.
                // We don't repeat ID/hash checks and a counter update for every value.
                point.primal.retain(|_, value| value.is_finite());
                point.primal.len()
            } else {
                let mut count = 0usize;
                point.primal.retain(|id, value| {
                    let source = source_variables.contains(id.index());
                    let keep = (source || retained.contains(&id)) && value.is_finite();
                    if keep && source {
                        count += 1;
                    }
                    keep
                });
                count
            };
        // Retained generated coordinates do not count toward source completeness.
        let valid = source_count == expected;
        if valid {
            point.primal.compact_if_excess();
            prepare_point(point);
            point.objective = point.objective.filter(|v| v.is_finite());
        }
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

    normalize_duals(&mut result, source_variables, projection);
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

/// Normalize dual evidence and project coordinates together.
fn normalize_duals(
    result: &mut SolverResult,
    source_variables: &SourceIndices,
    coordinates: ProjectionCoordinates<'_>,
) {
    if result.dual_status != DualStatus::FeasiblePoint {
        result.semantic_dual.clear();
        result.dual = SolutionValues::default();
        result.soc_dual = SolutionValues::default();
        result.psd_dual.clear();
        result.reduced_costs = SolutionValues::default();
        return;
    }

    match coordinates.rows {
        SourceIndices::Prefix(count) => {
            normalize_projected_duals(result, source_variables, coordinates, |row| {
                row.index() < *count
            });
        }
        SourceIndices::Explicit { ids, .. } => {
            normalize_projected_duals(result, source_variables, coordinates, |row| {
                ids.contains(&row.index())
            });
        }
    }
}

#[inline]
fn normalize_semantic_multiplier(value: &mut Option<f64>) {
    if value.is_some_and(|v| !v.is_finite()) {
        *value = None;
    }
}

fn normalize_projected_duals(
    result: &mut SolverResult,
    source_variables: &SourceIndices,
    coordinates: ProjectionCoordinates<'_>,
    keep_row: impl Fn(ConstraintId) -> bool,
) {
    result.dual.truncate(coordinates.rows.upper_bound());
    result.soc_dual.truncate(coordinates.soc);
    result.reduced_costs.truncate(source_variables.upper_bound());
    let keep_semantic = |id| match id {
        SemanticConstraintId::Algebraic(row) => keep_row(row),
        _ => coordinates.includes_semantic(id),
    };
    // Without scalar source rows, an all-algebraic map is entirely generated.
    // Check for preserved non-algebraic mappings before clearing in bulk.
    if coordinates.rows.len() == 0
        && result.semantic_dual.keys().all(|id| matches!(id, SemanticConstraintId::Algebraic(_)))
    {
        result.semantic_dual.clear();
    } else {
        result.semantic_dual.retain(|id, value| {
            if !keep_semantic(*id) {
                return false;
            }
            normalize_semantic_multiplier(value);
            true
        });
    }
    if coordinates.rows.len() == 0 {
        result.dual = SolutionValues::default();
    } else {
        result.dual.retain(|id, value| keep_row(id) && value.is_finite());
    }

    result
        .soc_dual
        .retain(|id, value| id.index() < coordinates.soc && value.is_finite() && value >= 0.0);
    result.psd_dual.retain(|id, matrix| {
        id.index() < coordinates.psd && matrix.upper_triangle().iter().all(|v| v.is_finite())
    });
    result
        .reduced_costs
        .retain(|id, value| source_variables.contains(id.index()) && value.is_finite());
    result.dual.compact_if_excess();
    result.soc_dual.compact_if_excess();
    result.reduced_costs.compact_if_excess();
}

/// Coordinate counts captured from a source model before solving or reformulation.
///
/// Entities appended by a reformulation have IDs at or above these bounds.
/// For an in-place reformulation, capture this before modifying the model.
/// Later source declarations can interleave with generated IDs; provide an
/// explicit source sets through [`SourceProjection::with_source_variables`] and
/// [`SourceProjection::with_source_rows`] in that case.
/// These counts do not include parameter declarations. Use
/// [`SourceProjectionPlan::capture`] to freeze those at the same boundary.
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
    #[error(
        "source {kind} coordinates extend to {count}, but the transformed model inherited only the first {inherited}"
    )]
    CoordinatesNotInherited { kind: &'static str, count: usize, inherited: usize },
    #[error("result belongs to model {actual:?}, expected {expected:?}")]
    ForeignResult { expected: ModelId, actual: ModelId },
    #[error("expression belongs to model {actual:?}, expected {expected:?}")]
    ForeignExpression { expected: ModelId, actual: ModelId },
    #[error(
        "parameter {parameter:?} is {source_name:?} in the source model, but {transformed_name:?} in the transformed model"
    )]
    IncompatibleParameter {
        parameter: oximo_expr::ParamId,
        source_name: String,
        transformed_name: Option<String>,
    },
    #[error(
        "transformed model {transformed_model:?} does not descend from source model {source_model:?}"
    )]
    UnrelatedTransformation { source_model: ModelId, transformed_model: ModelId },
    #[error("projection plan belongs to source model {actual:?}, expected {expected:?}")]
    ForeignSourcePlan { expected: ModelId, actual: ModelId },
    #[error(
        "model identity mismatch for solution {solution}: point belongs to model {actual:?}, expected {expected:?}"
    )]
    ForeignPoint { solution: usize, expected: ModelId, actual: ModelId },
    #[error(
        "parameter snapshot mismatch for solution {solution}: changed IDs {changed_parameters:?}; result has {result_parameters} parameters, point has {point_parameters}; use SolverResult::for_model and add_solution to capture consistent parameters"
    )]
    ParameterSnapshotMismatch {
        solution: usize,
        result_parameters: usize,
        point_parameters: usize,
        changed_parameters: Vec<oximo_expr::ParamId>,
    },
}

impl From<oximo_core::ModelMismatchError> for ProjectionError {
    fn from(error: oximo_core::ModelMismatchError) -> Self {
        Self::ForeignExpression { expected: error.expected, actual: error.actual }
    }
}

/// Validate native point provenance before any source-coordinate rebinding.
#[inline]
fn validate_result_points(result: &SolverResult) -> Result<(), ProjectionError> {
    for (solution, point) in result.solutions.iter().enumerate() {
        if point.model_id != ModelId::UNASSIGNED && point.model_id != result.model_id {
            return Err(ProjectionError::ForeignPoint {
                solution,
                expected: result.model_id,
                actual: point.model_id,
            });
        }
        let detached =
            point.model_id == ModelId::UNASSIGNED && point.parameters.values().is_empty();
        if !detached && !point.parameters.same_bits(&result.parameters) {
            return Err(ProjectionError::ParameterSnapshotMismatch {
                solution,
                result_parameters: result.parameters.values().len(),
                point_parameters: point.parameters.values().len(),
                changed_parameters: result.parameters.changed_parameters(&point.parameters),
            });
        }
    }
    Ok(())
}

type SourceObjective<'a> = Box<dyn Fn(&SolutionPoint) -> Option<f64> + 'a>;

/// Reusable coordinate membership for source-result projection.
///
/// Build once, then borrow with [`SourceProjection::with_plan`] for each solve.
/// Rebuild after source or generated entities change.
/// Plans captured with [`Self::capture`] or [`Self::with_coordinates`] are pinned
/// to the exact source model identity.
#[derive(Clone)]
pub struct SourceProjectionPlan {
    coordinates: SourceCoordinates,
    variables: SourceIndices,
    rows: SourceIndices,
    retained: FxHashSet<VarId>,
    parameters: Option<ParameterDeclarations>,
}

impl std::fmt::Debug for SourceProjectionPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SourceProjectionPlan")
            .field("coordinates", &self.coordinates)
            .field("source_variables", &self.variables.len())
            .field("source_rows", &self.rows.len())
            .field("retained_variables", &self.retained.len())
            .field("source_model", &self.parameters.as_ref().map(ParameterDeclarations::model_id))
            .field("parameter_count", &self.parameters.as_ref().map(|prefix| prefix.len()))
            .finish()
    }
}

impl SourceProjectionPlan {
    /// Use raw coordinate counts. Parameters are captured when a projector is
    /// constructed from this plan; use [`Self::capture`] to freeze them earlier.
    #[must_use]
    pub fn new(coordinates: SourceCoordinates) -> Self {
        Self {
            variables: SourceIndices::Prefix(coordinates.variables),
            rows: SourceIndices::Prefix(coordinates.algebraic_constraints),
            coordinates,
            retained: FxHashSet::default(),
            parameters: None,
        }
    }

    /// Capture coordinate counts and an immutable parameter declaration prefix
    /// before solving or transforming.
    #[must_use]
    pub fn capture(source: &Model) -> Self {
        Self::with_coordinates(source, SourceCoordinates::capture(source))
    }

    /// Capture parameters now while using previously captured coordinate counts.
    #[must_use]
    pub fn with_coordinates(source: &Model, coordinates: SourceCoordinates) -> Self {
        Self { parameters: Some(source.parameter_declarations()), ..Self::new(coordinates) }
    }

    /// Replace the source-variable prefix with exactly these IDs.
    #[must_use]
    pub fn with_source_variables(mut self, variables: impl IntoIterator<Item = VarId>) -> Self {
        self.set_variables(variables);
        self
    }

    /// Replace the source algebraic-row prefix with exactly these IDs.
    #[must_use]
    pub fn with_source_rows(mut self, rows: impl IntoIterator<Item = ConstraintId>) -> Self {
        self.set_rows(rows);
        self
    }

    /// Keep additional generated primal coordinates, without their reduced costs.
    #[must_use]
    pub fn retain_variables(
        mut self,
        variables: impl IntoIterator<Item = oximo_core::GeneratedVarId>,
    ) -> Self {
        self.retained.extend(variables.into_iter().map(oximo_core::GeneratedVarId::raw_id));
        self
    }

    fn set_variables(&mut self, variables: impl IntoIterator<Item = VarId>) {
        self.variables = SourceIndices::explicit(variables.into_iter().map(VarId::index));
        self.coordinates.variables = self.variables.len();
    }

    fn set_rows(&mut self, rows: impl IntoIterator<Item = ConstraintId>) {
        self.rows = SourceIndices::explicit(rows.into_iter().map(ConstraintId::index));
        self.coordinates.algebraic_constraints = self.rows.len();
    }
}

/// Projects a result produced on a transformed model back into the source
/// model's coordinates.
///
/// The projection:
/// - drops generated variables, rows, SOC/PSD cones, and reduced costs, except
///   for generated variables explicitly listed with [`Self::retain_variables`];
/// - rebinds the result and its points to the source model identity;
/// - preserves solve-time source parameters and removes transformed-only suffixes;
/// - optionally restores the objective convention and re-evaluates the source
///   objective on each projected point;
/// - applies [`finalize_result`] so the usual result contract holds.
///
/// Source variables and algebraic rows are ID prefixes by default. Use
/// [`Self::with_source_variables`] and [`Self::with_source_rows`] for interleaved
/// source and generated IDs. Borrow a [`SourceProjectionPlan`] to reuse membership
/// sets without cloning or rebuilding them, and reuse the projector across solves.
///
/// Generated multipliers are dropped rather than reinterpreted in source
/// coordinates. Semantic duals remain subject to the usual explicit-mapping
/// rules; an unavailable value stays unavailable.
///
/// Model and point provenance are checked at runtime.
pub struct SourceProjection<'a> {
    source: &'a Model,
    plan: Cow<'a, SourceProjectionPlan>,
    objective: ObjectiveTransform,
    reevaluate: Option<SourceObjective<'a>>,
    transformed: Option<&'a Model>,
    parameters: ParameterDeclarations,
    parameter_count: usize,
}

impl<'a> SourceProjection<'a> {
    /// Capture coordinate counts and the source parameter declaration prefix now.
    /// Later parameters remain unavailable through this projector.
    ///
    /// Use [`Self::with_coordinates`] when an in-place reformulation has already
    /// appended generated entities.
    #[must_use]
    pub fn capture(source: &'a Model) -> Self {
        Self::with_coordinates(source, SourceCoordinates::capture(source))
    }

    /// Use coordinate counts captured earlier and freeze parameter declarations
    /// now.
    #[must_use]
    pub fn with_coordinates(source: &'a Model, coordinates: SourceCoordinates) -> Self {
        let parameters = source.parameter_declarations();
        let plan = SourceProjectionPlan::new(coordinates);
        let parameter_count = parameters.len();
        Self {
            source,
            plan: Cow::Owned(plan),
            objective: ObjectiveTransform { sign: 1.0, offset: 0.0 },
            reevaluate: None,
            transformed: None,
            parameters,
            parameter_count,
        }
    }

    /// Borrow a reusable plan without copying its membership sets.
    #[must_use]
    pub fn with_plan(source: &'a Model, plan: &'a SourceProjectionPlan) -> Self {
        let parameters = plan.parameters.clone().unwrap_or_else(|| source.parameter_declarations());
        let parameter_count = parameters.len();
        Self {
            source,
            plan: Cow::Borrowed(plan),
            objective: ObjectiveTransform { sign: 1.0, offset: 0.0 },
            reevaluate: None,
            transformed: None,
            parameters,
            parameter_count,
        }
    }

    /// Treat exactly these variables as source coordinates.
    ///
    /// Use when source entities declared after an in-place reformulation
    /// interleave with generated IDs, so [`SourceCoordinates::variables`] no
    /// longer identifies a prefix. [`Self::retain_variables`] still applies to
    /// generated coordinates that must remain queryable.
    #[must_use]
    pub fn with_source_variables(mut self, variables: impl IntoIterator<Item = VarId>) -> Self {
        self.plan.to_mut().set_variables(variables);
        self
    }

    /// Treat exactly these algebraic rows as source coordinates.
    ///
    /// Use when rows declared after an in-place reformulation interleave with
    /// generated rows. SOC and PSD coordinates retain their captured prefixes.
    #[must_use]
    pub fn with_source_rows(mut self, rows: impl IntoIterator<Item = ConstraintId>) -> Self {
        self.plan.to_mut().set_rows(rows);
        self
    }

    /// Keep explicitly listed generated variables in each projected primal point.
    ///
    /// Source variables are always kept. Use this for generated coordinates
    /// that remain intentionally queryable after projection, such as GDP Hull
    /// activation variables. Only primal values are retained; generated rows,
    /// duals, SOC/PSD multipliers, and reduced costs are still dropped.
    #[must_use]
    pub fn retain_variables(
        mut self,
        variables: impl IntoIterator<Item = oximo_core::GeneratedVarId>,
    ) -> Self {
        self.plan
            .to_mut()
            .retained
            .extend(variables.into_iter().map(oximo_core::GeneratedVarId::raw_id));
        self
    }

    /// Require the result to belong to this transformed model.
    ///
    /// The transformed model must be the source itself or a descendant fork or
    /// reformulation.
    #[must_use]
    pub fn from_transformed(mut self, transformed: &'a Model) -> Self {
        let inherited = self.source.shared_parameter_prefix(transformed).unwrap_or(usize::MAX);
        self.parameter_count = self.parameters.len().min(inherited);
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
    /// The callback receives source coordinates, the solve-time parameter
    /// snapshot, and an objective already restored by [`Self::with_objective`].
    /// Returning `None` reports the value as unavailable instead of
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

    /// Re-evaluate a source expression using a captured expression graph and each
    /// point's solve-time parameters.
    ///
    /// Captures the graph now, so later objective declarations and parameter
    /// writes do not change the expression being evaluated. Missing variables or
    /// solve-time parameters, and non-finite values, make the objective unavailable.
    ///
    /// # Errors
    /// Returns an ownership error if the expression belongs to another model.
    pub fn reevaluate_objective_expr<D: oximo_expr::Degree>(
        self,
        expression: oximo_expr::Expr<'_, D>,
    ) -> Result<Self, ProjectionError> {
        if expression.model_id() != self.source.id() {
            return Err(oximo_core::ModelMismatchError::new(
                self.source.id(),
                expression.model_id(),
            )
            .into());
        }
        let expression = expression.capture();
        Ok(self.reevaluate_objective(move |point| {
            expression
                .evaluate(&point.primal, &point.parameters)
                .ok()
                .filter(|value| value.is_finite())
        }))
    }

    /// Check coordinates, parameter identity, and all point provenance without
    /// consuming the result or invoking objective reevaluation.
    ///
    /// # Errors
    /// Returns the same validation errors as [`Self::project`].
    pub fn validate(&self, result: &SolverResult) -> Result<(), ProjectionError> {
        check_source_coordinates(self.source, &self.plan)?;
        if self.parameters.model_id() != self.source.id() {
            return Err(ProjectionError::ForeignSourcePlan {
                expected: self.source.id(),
                actual: self.parameters.model_id(),
            });
        }
        let native = self.transformed.unwrap_or(self.source);
        if result.model_id != native.id() {
            return Err(ProjectionError::ForeignResult {
                expected: native.id(),
                actual: result.model_id,
            });
        }
        if native.id() != self.source.id() {
            let inherited = native.inherited_prefix(self.source).ok_or(
                ProjectionError::UnrelatedTransformation {
                    source_model: self.source.id(),
                    transformed_model: native.id(),
                },
            )?;
            for (kind, count, inherited) in [
                ("variable", self.plan.variables.upper_bound(), inherited.variables),
                ("algebraic row", self.plan.rows.upper_bound(), inherited.algebraic_constraints),
                (
                    "SOC constraint",
                    self.plan.coordinates.soc_constraints,
                    inherited.soc_constraints,
                ),
                (
                    "PSD constraint",
                    self.plan.coordinates.psd_constraints,
                    inherited.psd_constraints,
                ),
                ("SOS constraint", self.source.num_sos_constraints(), inherited.sos_constraints),
                (
                    "indicator constraint",
                    self.source.num_indicator_constraints(),
                    inherited.indicator_constraints,
                ),
            ] {
                if count > inherited {
                    return Err(ProjectionError::CoordinatesNotInherited {
                        kind,
                        count,
                        inherited,
                    });
                }
            }

            // Retained generated primals must not alias later source variables.
            if let Some(id) = self.plan.retained.iter().find(|id| {
                id.index() >= inherited.variables && id.index() < self.source.num_variables()
            }) {
                return Err(ProjectionError::CoordinatesNotInherited {
                    kind: "variable",
                    count: id.index() + 1,
                    inherited: inherited.variables,
                });
            }
        }
        if native.id() != self.source.id() && self.parameter_count != 0 {
            let transformed = native.parameters();
            if !self.parameters.matches_prefix(transformed.as_slice(), self.parameter_count) {
                for parameter in self.parameters.iter().take(self.parameter_count) {
                    let other = transformed.get(parameter.id.index());
                    if !other.is_some_and(|p| p.id == parameter.id && p.name == parameter.name) {
                        return Err(ProjectionError::IncompatibleParameter {
                            parameter: parameter.id,
                            source_name: parameter.name.to_string(),
                            transformed_name: other.map(|p| p.name.to_string()),
                        });
                    }
                }
            }
        }
        validate_result_points(result)
    }

    /// Project the result into the source model's coordinates.
    ///
    /// # Errors
    /// Returns [`ProjectionError`] when the captured coordinates exceed the
    /// source model, or when the result belongs to a model other than the
    /// expected one: the transformed model if [`Self::from_transformed`] was
    /// set, otherwise the source model. Rejects non-descendant transformations,
    /// source coordinates outside the inherited prefixes, incompatible parameter IDs/names,
    /// foreign solution points and conflicting point parameters before invoking
    /// any reevaluation callback.
    pub fn project(&self, mut result: SolverResult) -> Result<SolverResult, ProjectionError> {
        self.validate(&result)?;
        let Self { source, plan, objective, reevaluate, parameter_count, .. } = self;
        let SourceProjectionPlan {
            coordinates, variables: source_variables, rows, retained, ..
        } = plan.as_ref();
        // Restore objective units on the native certificate.
        result.best_bound = result.best_bound.and_then(|value| objective.restore(value));
        if result.parameters.values().len() > *parameter_count {
            result.parameters = result.parameters.prefix(*parameter_count);
        }
        result.model_id = source.id();
        let projection = ProjectionCoordinates {
            rows,
            soc: coordinates.soc_constraints,
            psd: coordinates.psd_constraints,
        };
        let result = normalize_validated_result(
            ValidatedResult(result),
            source_variables,
            retained,
            projection,
            |point| {
                point.objective = point.objective.and_then(|value| objective.restore(value));
                if let Some(evaluate) = reevaluate {
                    point.objective = evaluate(point);
                }
            },
        );
        Ok(result
            .with_semantic_dual_defaults_filtered(source, |id| projection.includes_semantic(id)))
    }
}

impl std::fmt::Debug for SourceProjection<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SourceProjection")
            .field("source", &self.source.id())
            .field("plan", &self.plan)
            .field("objective", &self.objective)
            .field("reevaluate", &self.reevaluate.is_some())
            .field("transformed", &self.transformed.map(Model::id))
            .field("parameters", &self.parameter_count)
            .finish()
    }
}

fn check_source_coordinates(
    model: &Model,
    plan: &SourceProjectionPlan,
) -> Result<(), ProjectionError> {
    let coordinates = plan.coordinates;
    for (kind, count, available) in [
        ("variable", plan.variables.upper_bound(), model.num_variables()),
        ("algebraic row", plan.rows.upper_bound(), model.constraints().algebraic().len()),
        ("SOC constraint", coordinates.soc_constraints, model.num_soc_constraints()),
        ("PSD constraint", coordinates.psd_constraints, model.num_psd_constraints()),
    ] {
        if count > available {
            return Err(ProjectionError::CoordinatesExceedSource { kind, count, available });
        }
    }
    Ok(())
}
