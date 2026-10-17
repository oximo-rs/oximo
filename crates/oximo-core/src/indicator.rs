use oximo_expr::{ExprId, VarId};
use smol_str::SmolStr;

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

/// Typed semantic reference to a binary-triggered affine function in an indicator set.
pub type IndicatorConstraintHandle =
    crate::ConstraintHandle<crate::IndicatorFunctionData, crate::IndicatorSet>;

/// IDs produced by a two-sided indicator declaration.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RangeIndicatorConstraintIds {
    Interval(IndicatorConstraintId),
    Split { lower: IndicatorConstraintId, upper: IndicatorConstraintId },
}

/// Model-bound handles produced by a two-sided indicator declaration.
#[derive(Copy, Clone, Debug)]
pub enum RangeIndicatorConstraintHandles {
    Interval(IndicatorConstraintHandle),
    Split { lower: IndicatorConstraintHandle, upper: IndicatorConstraintHandle },
}

impl RangeIndicatorConstraintHandles {
    #[must_use]
    pub const fn ids(self) -> RangeIndicatorConstraintIds {
        match self {
            Self::Interval(h) => RangeIndicatorConstraintIds::Interval(h.indicator_id()),
            Self::Split { lower, upper } => RangeIndicatorConstraintIds::Split {
                lower: lower.indicator_id(),
                upper: upper.indicator_id(),
            },
        }
    }
}
