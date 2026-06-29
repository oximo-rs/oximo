use oximo_expr::{Expr, ExprId};
use smol_str::SmolStr;

/// The sense of a constraint: less-than-or-equal, greater-than-or-equal, or equality.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Sense {
    Le,
    Ge,
    Eq,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ConstraintId(pub u32);

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
/// constant bounds is `[lo, hi]`. Use [`Constraint::as_single`] to recover the
/// single-sided sense (for backends without native two-sided rows) and
/// [`Constraint::is_range`] to detect a genuine range.
#[derive(Clone, Debug)]
pub struct Constraint {
    pub name: SmolStr,
    pub lhs: ExprId,
    pub lower: f64,
    pub upper: f64,
    pub active: bool,
}

impl Constraint {
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

/// In-progress constraint produced by [`Relate::le`] / [`Relate::ge`] /
/// [`Relate::eq`]. Registered through the `constraint!` macro.
#[derive(Copy, Clone, Debug)]
pub struct ConstraintExpr<'a> {
    pub lhs: Expr<'a>,
    pub sense: Sense,
    pub rhs: f64,
}

/// Build a constraint from an expression. Lives on `Expr` itself.
pub trait Relate<'a> {
    fn le<R: IntoRhs<'a>>(self, rhs: R) -> ConstraintExpr<'a>;
    fn ge<R: IntoRhs<'a>>(self, rhs: R) -> ConstraintExpr<'a>;
    fn eq<R: IntoRhs<'a>>(self, rhs: R) -> ConstraintExpr<'a>;
}

/// What can appear on the right-hand side of a constraint. Numeric scalars
/// stay as the canonical `rhs`. Expressions get subtracted into the LHS.
pub trait IntoRhs<'a> {
    fn fold_rhs(self, lhs: Expr<'a>) -> (Expr<'a>, f64);

    /// The numeric value when this RHS is a pure constant bound (a literal),
    /// else `None`. Used by the range-constraint registration to decide whether
    /// `lo <= e <= hi` collapses to one interval row: expression/param bounds
    /// return `None` so they stay two general constraints (keeping the symbolic
    /// bound re-bindable). Defaults to `None`.
    fn const_bound(&self) -> Option<f64> {
        None
    }
}

impl<'a> IntoRhs<'a> for f64 {
    fn fold_rhs(self, lhs: Expr<'a>) -> (Expr<'a>, f64) {
        (lhs, self)
    }
    fn const_bound(&self) -> Option<f64> {
        Some(*self)
    }
}

impl<'a> IntoRhs<'a> for i32 {
    fn fold_rhs(self, lhs: Expr<'a>) -> (Expr<'a>, f64) {
        (lhs, f64::from(self))
    }
    fn const_bound(&self) -> Option<f64> {
        Some(f64::from(*self))
    }
}

impl<'a> IntoRhs<'a> for Expr<'a> {
    fn fold_rhs(self, lhs: Expr<'a>) -> (Expr<'a>, f64) {
        (lhs - self, 0.0)
    }
}

impl<'a> Relate<'a> for Expr<'a> {
    fn le<R: IntoRhs<'a>>(self, rhs: R) -> ConstraintExpr<'a> {
        let (lhs, rhs) = rhs.fold_rhs(self);
        ConstraintExpr { lhs, sense: Sense::Le, rhs }
    }

    fn ge<R: IntoRhs<'a>>(self, rhs: R) -> ConstraintExpr<'a> {
        let (lhs, rhs) = rhs.fold_rhs(self);
        ConstraintExpr { lhs, sense: Sense::Ge, rhs }
    }

    fn eq<R: IntoRhs<'a>>(self, rhs: R) -> ConstraintExpr<'a> {
        let (lhs, rhs) = rhs.fold_rhs(self);
        ConstraintExpr { lhs, sense: Sense::Eq, rhs }
    }
}
