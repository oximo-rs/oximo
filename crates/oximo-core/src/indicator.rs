use oximo_expr::{ExprId, ModelId, VarId};
use smol_str::SmolStr;

use crate::model::Model;

/// Stable identifier for a native indicator constraint.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct IndicatorConstraintId(pub u32);

impl IndicatorConstraintId {
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// A binary-triggered affine row, stored canonically as
/// `trigger == active_value => lower <= lhs <= upper`.
#[derive(Clone, Debug)]
pub struct IndicatorConstraint {
    pub name: SmolStr,
    pub trigger: VarId,
    pub active_value: bool,
    pub lhs: ExprId,
    pub lower: f64,
    pub upper: f64,
    pub active: bool,
}

impl IndicatorConstraint {
    #[must_use]
    pub fn is_range(&self) -> bool {
        self.lower.is_finite()
            && self.upper.is_finite()
            && !self.lower.total_cmp(&self.upper).is_eq()
    }

    #[must_use]
    pub fn as_single(&self) -> Option<(crate::Sense, f64)> {
        match (self.lower.is_finite(), self.upper.is_finite()) {
            (false, true) => Some((crate::Sense::Le, self.upper)),
            (true, false) => Some((crate::Sense::Ge, self.lower)),
            (true, true) if self.lower.total_cmp(&self.upper).is_eq() => {
                Some((crate::Sense::Eq, self.lower))
            }
            _ => None,
        }
    }
}

/// Model-bound handle returned by a scalar indicator declaration.
#[derive(Copy, Clone)]
pub struct IndicatorConstraintHandle<'a> {
    pub(crate) model: &'a Model,
    pub(crate) id: IndicatorConstraintId,
}

impl std::fmt::Debug for IndicatorConstraintHandle<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndicatorConstraintHandle").field("id", &self.id).finish()
    }
}

impl IndicatorConstraintHandle<'_> {
    #[must_use]
    pub const fn id(self) -> IndicatorConstraintId {
        self.id
    }
    #[must_use]
    pub fn index(self) -> usize {
        self.id.index()
    }
    #[must_use]
    pub fn model_id(self) -> ModelId {
        self.model.id()
    }
}

impl From<IndicatorConstraintHandle<'_>> for IndicatorConstraintId {
    fn from(value: IndicatorConstraintHandle<'_>) -> Self {
        value.id
    }
}

/// IDs produced by a two-sided indicator declaration.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RangeIndicatorConstraintIds {
    Interval(IndicatorConstraintId),
    Split { lower: IndicatorConstraintId, upper: IndicatorConstraintId },
}

/// Model-bound handles produced by a two-sided indicator declaration.
#[derive(Copy, Clone, Debug)]
pub enum RangeIndicatorConstraintHandles<'a> {
    Interval(IndicatorConstraintHandle<'a>),
    Split { lower: IndicatorConstraintHandle<'a>, upper: IndicatorConstraintHandle<'a> },
}

impl RangeIndicatorConstraintHandles<'_> {
    #[must_use]
    pub const fn ids(self) -> RangeIndicatorConstraintIds {
        match self {
            Self::Interval(h) => RangeIndicatorConstraintIds::Interval(h.id()),
            Self::Split { lower, upper } => {
                RangeIndicatorConstraintIds::Split { lower: lower.id(), upper: upper.id() }
            }
        }
    }
}
