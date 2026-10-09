//! Plan first, then append solver-independent algebraic rows.

use oximo_core::gdp::{
    BoundSide, DisjunctConstraintHandle, DisjunctionHandle, GdpRangeHandles, GdpReformulationReport,
};
use oximo_core::{Model, ReformulatedModel};
use rustc_hash::FxHashMap;
use smol_str::SmolStr;
use thiserror::Error;

mod big_m;
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

// TODO: Add more methods (Multiple Big-M, Hull, P-split, cover cuts, etc.).

/// Method selected independently for each disjunction.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GdpMethod {
    BigM(BigM),
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

/// A method for all disjunctions, optional per-disjunction overrides, and per-row M assertions.
#[derive(Clone, Debug)]
pub struct GdpReformulationOptions {
    default_method: GdpMethod,
    methods: FxHashMap<DisjunctionHandle, GdpMethod>,
    big_m: FxHashMap<DisjunctConstraintHandle, BigMValues>,
    fallback: Option<f64>,
    bound_tightening: bool,
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
}

/// Explicit transformations of an ordinary `Model`.
pub trait GdpModelExt {
    /// Validate all pending components and reformulate them in place.
    ///
    /// Accepts a method such as [`BigM`] directly, or [`GdpReformulationOptions`]
    /// for per-disjunction overrides and other configuration.
    ///
    /// # Errors
    /// Returns a diagnostic without changing the model on invalid options,
    /// unsafe nonlinear domains, missing finite M values, numeric overflow or
    /// significant precision loss in generated Big-M rows.
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
