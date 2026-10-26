//! Reversible coordinate changes and the common result contract.
//!
//! Adapters must establish native solution quality before supplying points,
//! duals or global bounds. [`SourceProjection`] composes these helpers for
//! callers that solve an independently transformed model and need results in
//! the original model's coordinates.

use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::{BuildHasher, Hash};

use oximo_core::{ConstraintId, Model, ModelId, SemanticConstraintId, VarId};
use rustc_hash::{FxHashMap, FxHashSet};

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
    finalize_result_retaining(
        result,
        model,
        &SourceIndices::Prefix(num_variables),
        &FxHashSet::default(),
        None,
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
    projection: Option<ProjectionCoordinates<'_>>,
) -> SolverResult {
    let result = normalize_result(result, variables, retained, projection);
    result.with_semantic_dual_defaults_filtered(model, |id| {
        projection.is_none_or(|coordinates| coordinates.includes_semantic(id))
    })
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
fn normalize_result(
    mut result: SolverResult,
    source_variables: &SourceIndices,
    retained: &FxHashSet<VarId>,
    projection: Option<ProjectionCoordinates<'_>>,
) -> SolverResult {
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
    let expected = source_variables.len();
    result.solutions.retain_mut(|point| {
        let mut source_count = 0usize;
        point.primal.retain(|id, value| {
            let source = source_variables.contains(id.index());
            let keep = (source || retained.contains(id)) && value.is_finite();
            if keep && source {
                source_count += 1;
            }
            keep
        });
        point.objective = point.objective.filter(|v| v.is_finite());
        // Retained generated coordinates do not count toward source completeness.
        let valid = source_count == expected;
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
    projection: Option<ProjectionCoordinates<'_>>,
) {
    if result.dual_status != DualStatus::FeasiblePoint {
        result.semantic_dual.clear();
        result.dual.clear();
        result.soc_dual.clear();
        result.psd_dual.clear();
        result.reduced_costs.clear();
        return;
    }

    if let Some(coordinates) = projection {
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
    } else {
        result.semantic_dual.values_mut().for_each(normalize_semantic_multiplier);
        result.dual.retain(|_, value| value.is_finite());
        result.soc_dual.retain(|_, value| value.is_finite() && *value >= 0.0);
        result.psd_dual.retain(|_, matrix| matrix.upper_triangle().iter().all(|v| v.is_finite()));
        result.reduced_costs.retain(|_, value| value.is_finite());
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
        result.dual.clear();
    } else {
        result.dual.retain(|id, value| keep_row(*id) && value.is_finite());
    }

    result
        .soc_dual
        .retain(|id, value| id.index() < coordinates.soc && value.is_finite() && *value >= 0.0);
    result.psd_dual.retain(|id, matrix| {
        id.index() < coordinates.psd && matrix.upper_triangle().iter().all(|v| v.is_finite())
    });
    result
        .reduced_costs
        .retain(|id, value| source_variables.contains(id.index()) && value.is_finite());
}

/// Coordinate counts captured from a source model before a reformulation.
///
/// Entities appended by a reformulation have IDs at or above these bounds.
/// For an in-place reformulation, capture this before modifying the model.
/// Later source declarations can interleave with generated IDs; provide an
/// explicit source sets through [`SourceProjection::with_source_variables`] and
/// [`SourceProjection::with_source_rows`] in that case.
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
    #[error("result belongs to model {actual:?}, expected {expected:?}")]
    ForeignResult { expected: ModelId, actual: ModelId },
}

type SourceObjective<'a> = Box<dyn Fn(&SolutionPoint) -> Option<f64> + 'a>;

/// Reusable coordinate membership for source-result projection.
///
/// Build once, then borrow with [`SourceProjection::with_plan`] for each solve.
/// Rebuild after source or generated entities change.
#[derive(Clone, Debug)]
pub struct SourceProjectionPlan {
    coordinates: SourceCoordinates,
    variables: SourceIndices,
    rows: SourceIndices,
    retained: FxHashSet<VarId>,
}

impl SourceProjectionPlan {
    #[must_use]
    pub fn new(coordinates: SourceCoordinates) -> Self {
        Self {
            variables: SourceIndices::Prefix(coordinates.variables),
            rows: SourceIndices::Prefix(coordinates.algebraic_constraints),
            coordinates,
            retained: FxHashSet::default(),
        }
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
/// - preserves the solve-time parameter snapshot;
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
pub struct SourceProjection<'a> {
    source: &'a Model,
    plan: Cow<'a, SourceProjectionPlan>,
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
            plan: Cow::Owned(SourceProjectionPlan::new(coordinates)),
            objective: ObjectiveTransform { sign: 1.0, offset: 0.0 },
            reevaluate: None,
            transformed: None,
        }
    }

    /// Borrow a reusable plan without allocating or copying its membership sets.
    #[must_use]
    pub fn with_plan(source: &'a Model, plan: &'a SourceProjectionPlan) -> Self {
        Self {
            source,
            plan: Cow::Borrowed(plan),
            objective: ObjectiveTransform { sign: 1.0, offset: 0.0 },
            reevaluate: None,
            transformed: None,
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

    /// Keep explicitly listed generated variables in each projected primal map.
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
    /// source model, or when the result belongs to a model other than the
    /// expected one: the transformed model if [`Self::from_transformed`] was
    /// set, otherwise the source model.
    pub fn project(&self, result: SolverResult) -> Result<SolverResult, ProjectionError> {
        let Self { source, plan, objective, reevaluate, transformed } = self;
        let SourceProjectionPlan { coordinates, variables: source_variables, rows, retained } =
            plan.as_ref();
        check_source_coordinates(source, plan)?;
        let expected = transformed.map_or_else(|| source.id(), Model::id);
        if result.model_id != expected {
            return Err(ProjectionError::ForeignResult { expected, actual: result.model_id });
        }

        let mut result = result;

        // Generated multipliers and reduced costs have no source meaning.
        // Primals are filtered by `finalize_result` below, which also keeps
        // explicitly retained generated variables.
        if reevaluate.is_some() {
            for point in &mut result.solutions {
                point
                    .primal
                    .retain(|id, _| source_variables.contains(id.index()) || retained.contains(id));
            }
        }

        // Restore objective units on the native certificate.
        for point in &mut result.solutions {
            point.objective = point.objective.and_then(|value| objective.restore(value));
        }
        result.best_bound = result.best_bound.and_then(|value| objective.restore(value));

        // Rebind provenance before evaluating source expressions.
        result.model_id = source.id();
        for point in &mut result.solutions {
            point.model_id = source.id();
            if let Some(evaluate) = reevaluate {
                point.objective = evaluate(point);
            }
        }

        Ok(finalize_result_retaining(
            result,
            source,
            source_variables,
            retained,
            Some(ProjectionCoordinates {
                rows,
                soc: coordinates.soc_constraints,
                psd: coordinates.psd_constraints,
            }),
        ))
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
