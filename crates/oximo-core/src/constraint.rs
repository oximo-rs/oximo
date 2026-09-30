use std::fmt;

use oximo_expr::{Expr, ExprId, ModelId};
use smol_str::SmolStr;

/// The sense of a constraint: less-than-or-equal, greater-than-or-equal, or equality.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Sense {
    Le,
    Ge,
    Eq,
}

impl fmt::Display for Sense {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Le => "<=",
            Self::Ge => ">=",
            Self::Eq => "=",
        })
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ConstraintId(pub u32);

/// A model-bound handle to an algebraic constraint.
///
/// Constraint declaration macros return this type.
/// Use [`Self::id`] when a backend-facing raw numeric ID
/// is explicitly required.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ConstraintHandle {
    id: ConstraintId,
    model_id: ModelId,
}

impl ConstraintHandle {
    pub(crate) const fn new(id: ConstraintId, model_id: ModelId) -> Self {
        Self { id, model_id }
    }

    #[must_use]
    pub const fn id(self) -> ConstraintId {
        self.id
    }

    #[must_use]
    pub const fn model_id(self) -> ModelId {
        self.model_id
    }

    #[must_use]
    pub fn index(self) -> usize {
        self.id.index()
    }
}

impl From<ConstraintHandle> for ConstraintId {
    fn from(value: ConstraintHandle) -> Self {
        value.id
    }
}

impl PartialEq<ConstraintId> for ConstraintHandle {
    fn eq(&self, other: &ConstraintId) -> bool {
        self.id == *other
    }
}

impl PartialEq<ConstraintHandle> for ConstraintId {
    fn eq(&self, other: &ConstraintHandle) -> bool {
        *self == other.id
    }
}

/// Model row IDs produced by a two-sided range declaration.
///
/// Constant bounds with a linear body produce one interval row. Other ranges
/// produce separate lower and upper rows. You can query each row's dual individually.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RangeConstraintIds {
    Interval(ConstraintId),
    Split { lower: ConstraintId, upper: ConstraintId },
}

/// Model-bound handles produced by a two-sided range declaration.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RangeConstraintHandles {
    Interval(ConstraintHandle),
    Split { lower: ConstraintHandle, upper: ConstraintHandle },
}

impl RangeConstraintHandles {
    #[must_use]
    pub const fn ids(self) -> RangeConstraintIds {
        match self {
            Self::Interval(handle) => RangeConstraintIds::Interval(handle.id()),
            Self::Split { lower, upper } => {
                RangeConstraintIds::Split { lower: lower.id(), upper: upper.id() }
            }
        }
    }
}

impl From<RangeConstraintHandles> for RangeConstraintIds {
    fn from(value: RangeConstraintHandles) -> Self {
        value.ids()
    }
}

impl PartialEq<RangeConstraintIds> for RangeConstraintHandles {
    fn eq(&self, other: &RangeConstraintIds) -> bool {
        self.ids() == *other
    }
}

impl PartialEq<RangeConstraintHandles> for RangeConstraintIds {
    fn eq(&self, other: &RangeConstraintHandles) -> bool {
        *self == other.ids()
    }
}

impl ConstraintId {
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// A single algebraic constraint, canonicalized as the interval
/// `lower <= lhs <= upper` with numeric bounds. RHS expressions are folded into
/// `lhs` during construction, so backends only ever see this canonical shape.
///
/// The single-sided senses map onto the interval as `Le(rhs) => [-inf, rhs]`,
/// `Ge(rhs) => [rhs, +inf]`, `Eq(rhs) => [rhs, rhs]`. A two-sided range with
/// constant bounds is `[lo, hi]`. Use [`AlgebraicConstraint::as_single`] to recover the
/// single-sided sense (for backends without native two-sided rows) and
/// [`AlgebraicConstraint::is_range`] to detect a genuine range.
#[derive(Clone, Debug)]
pub struct AlgebraicConstraint {
    pub name: SmolStr,
    pub lhs: ExprId,
    pub lower: f64,
    pub upper: f64,
    pub active: bool,
}

impl AlgebraicConstraint {
    /// The two bounds are equal, i.e. this is an equality row. Uses `total_cmp`
    /// for an exact comparison: the bounds are literals (`Eq` copies the same
    /// value into both).
    fn is_equality(&self) -> bool {
        self.lower.total_cmp(&self.upper).is_eq()
    }

    /// Whether this is a genuine two-sided range (both bounds finite and not an
    /// equality), as opposed to a single-sided `Le`/`Ge`/`Eq` row. An inverted
    /// `[hi, lo]` (`lo > hi`, an infeasible user range) is also a range, so the
    /// solver reports the infeasibility rather than it collapsing to an equality.
    #[must_use]
    pub fn is_range(&self) -> bool {
        self.lower.is_finite() && self.upper.is_finite() && !self.is_equality()
    }

    /// Recover the single-sided `(sense, rhs)` view, or `None` for a genuine
    /// range (or an unconstrained `[-inf, +inf]` row). Backends and writers
    /// without native two-sided rows branch on this.
    #[must_use]
    pub fn as_single(&self) -> Option<(Sense, f64)> {
        match (self.lower.is_finite(), self.upper.is_finite()) {
            (false, true) => Some((Sense::Le, self.upper)),
            (true, false) => Some((Sense::Ge, self.lower)),
            (true, true) if self.is_equality() => Some((Sense::Eq, self.lower)),
            _ => None,
        }
    }
}

/// Numeric or symbolic right-hand side of a relation.
pub trait IntoRhs<'a, D: crate::function_set::FunctionDegree = oximo_expr::Dynamic> {
    type Degree: crate::function_set::FunctionDegree;
    fn fold_rhs(self, lhs: Expr<'a, D>) -> (Expr<'a, Self::Degree>, f64);
    fn const_bound(&self) -> Option<f64> {
        None
    }
}
impl<'a, D: crate::function_set::FunctionDegree> IntoRhs<'a, D> for f64 {
    type Degree = D;
    #[inline]
    fn fold_rhs(self, lhs: Expr<'a, D>) -> (Expr<'a, D>, f64) {
        (lhs, self)
    }
    fn const_bound(&self) -> Option<f64> {
        Some(*self)
    }
}
impl<'a, D: crate::function_set::FunctionDegree> IntoRhs<'a, D> for i32 {
    type Degree = D;
    #[inline]
    fn fold_rhs(self, lhs: Expr<'a, D>) -> (Expr<'a, D>, f64) {
        (lhs, f64::from(self))
    }
    fn const_bound(&self) -> Option<f64> {
        Some(f64::from(*self))
    }
}
impl<'a, D, R> IntoRhs<'a, D> for Expr<'a, R>
where
    D: crate::function_set::FunctionDegree + oximo_expr::degree::AddDegree<R>,
    R: oximo_expr::Degree,
    <D as oximo_expr::degree::AddDegree<R>>::Output: crate::function_set::FunctionDegree,
{
    type Degree = <D as oximo_expr::degree::AddDegree<R>>::Output;
    #[inline]
    fn fold_rhs(self, lhs: Expr<'a, D>) -> (Expr<'a, Self::Degree>, f64) {
        (lhs - self, 0.0)
    }
}
use crate::function_set::{Constraint, EqualTo, FunctionDegree, GreaterThan, LessThan};
/// Build typed relations from the expression DSL.
pub trait Relate<'a, D: FunctionDegree = oximo_expr::Dynamic> {
    fn le<R: IntoRhs<'a, D>>(
        self,
        rhs: R,
    ) -> Constraint<<R::Degree as FunctionDegree>::RelationFunction<'a>, LessThan>;
    fn ge<R: IntoRhs<'a, D>>(
        self,
        rhs: R,
    ) -> Constraint<<R::Degree as FunctionDegree>::RelationFunction<'a>, GreaterThan>;
    fn eq<R: IntoRhs<'a, D>>(
        self,
        rhs: R,
    ) -> Constraint<<R::Degree as FunctionDegree>::RelationFunction<'a>, EqualTo>;
}
impl<'a, D: FunctionDegree> Relate<'a, D> for Expr<'a, D> {
    #[inline]
    fn le<R: IntoRhs<'a, D>>(
        self,
        rhs: R,
    ) -> Constraint<<R::Degree as FunctionDegree>::RelationFunction<'a>, LessThan> {
        let (lhs, rhs) = rhs.fold_rhs(self);
        Constraint::new(R::Degree::relation_function(lhs), LessThan(rhs))
    }
    #[inline]
    fn ge<R: IntoRhs<'a, D>>(
        self,
        rhs: R,
    ) -> Constraint<<R::Degree as FunctionDegree>::RelationFunction<'a>, GreaterThan> {
        let (lhs, rhs) = rhs.fold_rhs(self);
        Constraint::new(R::Degree::relation_function(lhs), GreaterThan(rhs))
    }
    #[inline]
    fn eq<R: IntoRhs<'a, D>>(
        self,
        rhs: R,
    ) -> Constraint<<R::Degree as FunctionDegree>::RelationFunction<'a>, EqualTo> {
        let (lhs, rhs) = rhs.fold_rhs(self);
        Constraint::new(R::Degree::relation_function(lhs), EqualTo(rhs))
    }
}
#[cfg(test)]
mod tests {
    use super::Sense;

    #[test]
    fn display_uses_ascii_relations() {
        let labels = [Sense::Le.to_string(), Sense::Ge.to_string(), Sense::Eq.to_string()];
        assert_eq!(labels, ["<=", ">=", "="]);
        assert!(labels.iter().all(|label| label.is_ascii()));
    }
}
