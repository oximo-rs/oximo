//! Plan first, then append solver-independent algebraic rows.

use oximo_core::gdp::{
    BooleanId, BoundSide, DisjunctConstraintHandle, DisjunctionHandle, DisjunctionId,
    GdpPlanningStatistics, GdpRangeHandles, GdpReformulationReport,
};
use oximo_core::{Expr, Model, ModelId, ReformulatedModel, VarId};
use rustc_hash::FxHashMap;
use smol_str::SmolStr;
use thiserror::Error;

mod big_m;
mod hull;
mod logic;
mod plan;

use plan::PlanBuilder;

/// Big-M settings for a disjunction. Automatic estimates take precedence over fallbacks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BigM {
    fallback: Option<f64>,
    hierarchical_tightening: bool,
}

impl Default for BigM {
    fn default() -> Self {
        Self { fallback: None, hierarchical_tightening: true }
    }
}

impl BigM {
    /// Use ancestor constraints to tighten nested rows, retaining relaxation
    /// terms for inactive ancestors. Enabled by default.
    ///
    /// Disabling this keeps the single-indicator formulation for comparisons or
    /// lower preprocessing cost.
    #[must_use]
    pub const fn with_hierarchical_tightening(mut self, enabled: bool) -> Self {
        self.hierarchical_tightening = enabled;
        self
    }

    pub const fn hierarchical_tightening(self) -> bool {
        self.hierarchical_tightening
    }

    #[must_use]
    pub const fn with_fallback_big_m(mut self, value: f64) -> Self {
        self.fallback = Some(value);
        self
    }

    pub const fn fallback_big_m(self) -> Option<f64> {
        self.fallback
    }
}

// TODO: Add more methods (Multiple Big-M, P-split, cover cuts, etc.).

/// Treatment of inclusive disjunctions by Hull.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HullInclusivePolicy {
    /// Require exactly-one selection (the usual Hull formulation).
    #[default]
    Reject,
    /// Expand nonempty activation subsets locally. This has exponential size.
    Enumerate { max_patterns: usize },
}

/// Hierarchical Hull settings. Nonlinear perspectives are exact at binary selectors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hull {
    epsilon: f64,
    inclusive_policy: HullInclusivePolicy,
}

impl Default for Hull {
    fn default() -> Self {
        Self { epsilon: 1e-6, inclusive_policy: HullInclusivePolicy::Reject }
    }
}

impl Hull {
    /// Set the nonlinear perspective regularization, strictly between zero and one.
    /// Invalid values are rejected during planning, before any model mutation.
    #[must_use]
    pub const fn with_epsilon(mut self, value: f64) -> Self {
        self.epsilon = value;
        self
    }

    pub const fn epsilon(self) -> f64 {
        self.epsilon
    }

    #[must_use]
    pub const fn with_inclusive_policy(mut self, policy: HullInclusivePolicy) -> Self {
        self.inclusive_policy = policy;
        self
    }

    pub const fn inclusive_policy(self) -> HullInclusivePolicy {
        self.inclusive_policy
    }
}

/// Method selected independently for each disjunction.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GdpMethod {
    BigM(BigM),
    Hull(Hull),
}

impl From<Hull> for GdpMethod {
    fn from(value: Hull) -> Self {
        Self::Hull(value)
    }
}

impl From<BigM> for GdpMethod {
    fn from(value: BigM) -> Self {
        Self::BigM(value)
    }
}

impl Default for GdpMethod {
    fn default() -> Self {
        BigM::default().into()
    }
}

/// Explicit nonnegative relaxation amounts. An absent side is estimated normally.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BigMValues {
    pub lower: Option<f64>,
    pub upper: Option<f64>,
}

impl BigMValues {
    pub const fn lower(value: f64) -> Self {
        Self { lower: Some(value), upper: None }
    }

    pub const fn upper(value: f64) -> Self {
        Self { lower: None, upper: Some(value) }
    }

    pub const fn symmetric(value: f64) -> Self {
        Self { lower: Some(value), upper: Some(value) }
    }
}

/// Counts of model variables, constraints (including native constraints), and arena nodes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GdpModelSize {
    pub variables: usize,
    pub constraints: usize,
    pub expression_nodes: usize,
}

/// Exact sizes of a validated reformulation, without modifying the source model.
/// Counts describe the pending GDP components under the supplied options.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GdpPlanningSummary {
    pub planning: GdpPlanningStatistics,
    pub generated: GdpModelSize,
    pub total: GdpModelSize,
    pub hull_copies: usize,
    pub hull_patterns: usize,
}

/// Limits on the final model size, including existing model objects.
///
/// `None` leaves a resource unlimited. Variables and rows are checked during
/// planning.
/// These limits do not bound temporary planning memory.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GdpSizeLimits {
    pub variables: Option<usize>,
    pub constraints: Option<usize>,
    pub expression_nodes: Option<usize>,
}

/// Limits on planning work across all groups, including repeated nested expansions.
/// These supplement [`GdpSizeLimits`], which caps the committed model.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GdpPlanningLimits {
    /// Row objects built or cloned during planning, including final planned rows.
    pub row_expansions: Option<usize>,
    /// Maximum private arena nodes, including cloned source nodes and temporary
    /// rewrites.
    pub expression_nodes: Option<usize>,
}

/// Scope in which a Hull reference or perspective fails domain validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GdpHullScope {
    Disjunction(DisjunctionId),
    Standalone(BooleanId),
}

/// Domain checks distinguish the reference evaluation from the complete
/// independent box of generated variable bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GdpHullDomainContext {
    Reference,
    GeneratedBox,
}

impl std::fmt::Display for GdpHullDomainContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Reference => "reference evaluation",
            Self::GeneratedBox => "generated variable box",
        })
    }
}

/// A method for all disjunctions, optional per-disjunction overrides, and per-row M assertions.
#[derive(Clone, Debug)]
pub struct GdpReformulationOptions {
    default_method: GdpMethod,
    methods: FxHashMap<DisjunctionHandle, GdpMethod>,
    big_m: FxHashMap<DisjunctConstraintHandle, BigMValues>,
    fallback: Option<f64>,
    bound_tightening: bool,
    reference_points: FxHashMap<(ModelId, Option<VarId>), f64>,
    size_limits: GdpSizeLimits,
    planning_limits: GdpPlanningLimits,
}

impl Default for GdpReformulationOptions {
    fn default() -> Self {
        Self::new(GdpMethod::default())
    }
}

impl From<BigM> for GdpReformulationOptions {
    fn from(method: BigM) -> Self {
        Self::new(method)
    }
}

impl From<Hull> for GdpReformulationOptions {
    fn from(method: Hull) -> Self {
        Self::new(method)
    }
}

impl From<GdpMethod> for GdpReformulationOptions {
    fn from(method: GdpMethod) -> Self {
        Self::new(method)
    }
}

impl GdpReformulationOptions {
    /// Select the method used by disjunctions without a [`Self::with_method`] override.
    /// Standalone conditional rows also use this method's settings.
    #[must_use]
    pub fn new(method: impl Into<GdpMethod>) -> Self {
        Self {
            default_method: method.into(),
            methods: FxHashMap::default(),
            big_m: FxHashMap::default(),
            fallback: None,
            bound_tightening: true,
            reference_points: FxHashMap::default(),
            size_limits: GdpSizeLimits::default(),
            planning_limits: GdpPlanningLimits::default(),
        }
    }

    /// Enable bounded affine propagation over unconditional and ancestor constraints.
    /// Enabled by default.
    /// Disable to use only declared variable bounds or to avoid the preprocessing cost
    /// on models whose global rows cannot help.
    #[must_use]
    pub const fn with_bound_tightening(mut self, enabled: bool) -> Self {
        self.bound_tightening = enabled;
        self
    }

    /// Set the reference coordinate `a` shared across affine and nonlinear Hull scopes.
    ///
    /// A generated copy represents `z = activation * (source - a)`, with bounds
    /// `(lower - a) * activation <= z <= (upper - a) * activation`. Affine rows
    /// and aggregation equations use this same shift. Nonlinear perspectives
    /// evaluate the source at `z / (epsilon + (1 - epsilon) * activation) + a`.
    /// Unspecified coordinates are zero. Local selectors and child auxiliaries
    /// copied by inclusive ancestors use zero.
    ///
    /// The reference doesn't need to satisfy branch constraints. Both its evaluation
    /// and the entire independent generated variable box must have safe domains.
    /// A reference inside the source domain can therefore fail: for `ln(x)` with
    /// `x` in `[1, 4]`, reference `1` works while `2` and `4` introduce negative
    /// arguments in the generated box. Failures include argument intervals and scope.
    /// Shifts that lose precision in source bounds or transformed rows are rejected.
    /// Pass a bare variable handle. Compound expressions are rejected at planning time.
    #[must_use]
    pub fn with_hull_reference_coordinate<D: oximo_expr::Degree>(
        mut self,
        variable: Expr<'_, D>,
        value: f64,
    ) -> Self {
        self.reference_points.insert((variable.model_id(), variable.var_id()), value);
        self
    }

    /// Set multiple reference coordinates, including entries of indexed variables.
    /// Later entries replace earlier values for the same variable.
    /// See [`Self::with_hull_reference_coordinate`] for the coordinate convention.
    ///
    /// ```
    /// use oximo_gdp::prelude::*;
    /// let model = Model::new("indexed references");
    /// variable!(model, 1.0 <= x[i in 0..3] <= 4.0);
    /// let options = GdpReformulationOptions::new(Hull::default())
    ///     .with_hull_reference_coordinates(x.iter().map(|(_, variable)| (variable, 1.0)));
    /// ```
    #[must_use]
    pub fn with_hull_reference_coordinates<
        'a,
        D: oximo_expr::Degree,
        V: std::borrow::Borrow<Expr<'a, D>>,
    >(
        mut self,
        coordinates: impl IntoIterator<Item = (V, f64)>,
    ) -> Self {
        for (variable, value) in coordinates {
            self = self.with_hull_reference_coordinate(*variable.borrow(), value);
        }
        self
    }

    /// Compatibility alias for [`Self::with_hull_reference_coordinate`].
    #[must_use]
    pub fn with_hull_reference_point<D: oximo_expr::Degree>(
        self,
        variable: Expr<'_, D>,
        value: f64,
    ) -> Self {
        self.with_hull_reference_coordinate(variable, value)
    }

    /// Limit total variables, constraints, and expression nodes in the final model.
    /// A limit failure leaves the source model unchanged.
    #[must_use]
    pub const fn with_size_limits(mut self, limits: GdpSizeLimits) -> Self {
        self.size_limits = limits;
        self
    }

    /// Cap total planning work and temporary private nodes across nested groups.
    /// See [`GdpPlanningLimits`] and [`Self::with_size_limits`].
    #[must_use]
    pub const fn with_planning_limits(mut self, limits: GdpPlanningLimits) -> Self {
        self.planning_limits = limits;
        self
    }

    /// Override the method for one disjunction. Other disjunctions keep the
    /// method supplied to [`Self::new`]. Nested disjunctions do not inherit
    /// their parent's override.
    #[must_use]
    pub fn with_method(
        mut self,
        disjunction: DisjunctionHandle,
        method: impl Into<GdpMethod>,
    ) -> Self {
        self.methods.insert(disjunction, method.into());
        self
    }

    #[must_use]
    pub fn with_big_m(mut self, row: DisjunctConstraintHandle, values: BigMValues) -> Self {
        self.big_m.insert(row, values);
        self
    }

    /// Set overrides on a range, including its separately normalized rows.
    /// Each side is applied only to the corresponding side of a split range.
    #[must_use]
    pub fn with_range_big_m(mut self, range: GdpRangeHandles, values: BigMValues) -> Self {
        match range {
            GdpRangeHandles::Interval(row) => {
                self.big_m.insert(row, values);
            }
            GdpRangeHandles::Split { lower, upper } => {
                self.big_m.insert(lower, BigMValues { lower: values.lower, upper: None });
                self.big_m.insert(upper, BigMValues { lower: None, upper: values.upper });
            }
        }
        self
    }

    #[must_use]
    pub const fn with_fallback_big_m(mut self, value: f64) -> Self {
        self.fallback = Some(value);
        self
    }
}

/// An invalid GDP transformation. Errors are reported before mutation.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum GdpError {
    #[error("GDP option refers to an unknown or foreign-model {0}")]
    ForeignHandle(&'static str),
    #[error("Big-M must be finite and nonnegative, got {0}")]
    InvalidBigM(f64),
    #[error("Hull epsilon must be finite and strictly between zero and one, got {0}")]
    InvalidHullEpsilon(f64),
    #[error("Hull reference coordinate must be finite, got {0}")]
    InvalidHullReference(f64),
    #[error(
        "Hull reference coordinate expects a bare variable, got a compound expression or constant"
    )]
    ExpectedHullReferenceVariable,
    #[error(
        "Hull reference {reference} loses precision in the bounds of variable {variable:?}; choose a closer reference"
    )]
    HullReferencePrecisionLoss { variable: SmolStr, reference: f64 },
    #[error(
        "Hull loses precision in the active {side} inequality of row {constraint:?}: recovery error {error} exceeds tolerance {tolerance}; choose a closer reference or rescale the row"
    )]
    HullRowPrecisionLoss {
        constraint: SmolStr,
        side: BoundSide,
        /// Absolute error recovering an active coefficient or effective bound.
        error: f64,
        /// Maximum permitted absolute error for that recovery check.
        tolerance: f64,
    },
    #[error("Hull requires finite bounds for variable {variable:?}")]
    MissingHullBounds { variable: SmolStr },
    #[error(
        "Hull requires exactly-one selection for {0:?}; opt into activation-pattern enumeration to handle AtLeastOne"
    )]
    InclusiveHull(SmolStr),
    #[error("activation-pattern enumeration for {disjunction:?} exceeds limit {limit}")]
    HullPatternLimit { disjunction: SmolStr, limit: usize },
    #[error(
        "Hull row {constraint:?} in scope {scope_name:?} ({scope:?}) has an unsafe {context} domain for {operation} with argument intervals {arguments:?}; choose a safe reference coordinate or use Big-M"
    )]
    UnsafeHullDomain {
        constraint: SmolStr,
        operation: &'static str,
        /// Intervals of the offending operation's arguments, in operand order.
        arguments: Vec<[f64; 2]>,
        scope: GdpHullScope,
        scope_name: SmolStr,
        context: GdpHullDomainContext,
    },
    #[error(
        "cannot derive finite {side} Big-M for {constraint:?}; provide an explicit M or fallback"
    )]
    MissingBigM { constraint: SmolStr, side: BoundSide },
    #[error(
        "conditional row {constraint:?} has an unsafe global domain for {operation}; provide global domain-safe variable bounds"
    )]
    UnsafeDomain { constraint: SmolStr, operation: &'static str },
    #[error("conditional row {0:?} contains invalid numeric values or variable bounds")]
    InvalidExpression(SmolStr),
    #[error("GDP reformulation generates non-finite coefficients or bounds")]
    NumericOverflow,
    #[error(
        "Big-M loses precision in the active {side} inequality of conditional row {constraint:?}; tighten global bounds or provide a smaller valid M"
    )]
    PrecisionLoss { constraint: SmolStr, side: BoundSide },
    #[error("GDP reformulation exceeds the model's numeric ID capacity")]
    Capacity,
    #[error("GDP reformulation needs {required} total {resource}, exceeding limit {limit}")]
    SizeLimit { resource: &'static str, required: usize, limit: usize },
    #[error("GDP planning needs {required} {resource}, exceeding work limit {limit}")]
    PlanningLimit { resource: &'static str, required: usize, limit: usize },
}

/// Explicit transformations of an ordinary `Model`.
pub trait GdpModelExt {
    /// Fully plan and validate pending GDP components without modifying the model.
    /// Returns exact committed sizes, including nested copies and pattern weights.
    /// Planning builds temporary artifacts and costs approximately as much as
    /// reformulation.
    /// Subsequent reformulation plans again using the then-current source model.
    ///
    /// # Errors
    /// Returns the same validation and size-limit errors as [`Self::reformulate_gdp`].
    fn plan_gdp(
        &self,
        options: impl Into<GdpReformulationOptions>,
    ) -> Result<GdpPlanningSummary, GdpError>;

    /// Validate all pending components and reformulate them in place.
    ///
    /// Accepts [`BigM`] or [`Hull`] directly, or [`GdpReformulationOptions`]
    /// for per-disjunction overrides and other configuration.
    ///
    /// # Errors
    /// Returns a diagnostic without changing the model on invalid options,
    /// unsafe nonlinear domains, missing finite bounds or M values, unsupported
    /// inclusive Hull selection, excessive pattern counts, numeric overflow or
    /// significant precision loss in reference shifts or generated rows.
    fn reformulate_gdp(
        &self,
        options: impl Into<GdpReformulationOptions>,
    ) -> Result<GdpReformulationReport, GdpError>;

    /// Transform an independent copy while preserving numeric source IDs.
    ///
    /// # Errors
    /// Returns the same validation errors as the in-place method.
    fn to_reformulated_gdp_model(
        &self,
        options: impl Into<GdpReformulationOptions>,
    ) -> Result<ReformulatedModel, GdpError>;
}

impl GdpModelExt for Model {
    // TODO: We can expose a reusable validated plan to avoid repeating expensive planning
    // for large enumerations. Before applying it, detect changes to parameters,
    // variable bounds, and model structure, and reject or revalidate stale plans.
    fn plan_gdp(
        &self,
        options: impl Into<GdpReformulationOptions>,
    ) -> Result<GdpPlanningSummary, GdpError> {
        Ok(PlanBuilder::build(self, &options.into())?.summary())
    }

    fn reformulate_gdp(
        &self,
        options: impl Into<GdpReformulationOptions>,
    ) -> Result<GdpReformulationReport, GdpError> {
        let options = options.into();
        let plan = PlanBuilder::build(self, &options)?;

        Ok(plan.apply(self))
    }

    fn to_reformulated_gdp_model(
        &self,
        options: impl Into<GdpReformulationOptions>,
    ) -> Result<ReformulatedModel, GdpError> {
        let options = options.into();
        let plan = PlanBuilder::build(self, &options)?;
        let clone = self.__gdp_clone();
        plan.apply(&clone);

        Ok(ReformulatedModel::__from_model(clone))
    }
}
