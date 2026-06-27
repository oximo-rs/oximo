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

/// A single algebraic constraint already canonicalized as `lhs <op> rhs`,
/// where `rhs` is a numeric constant. RHS expressions are folded into `lhs`
/// during construction, so backends only ever see this canonical shape.
#[derive(Clone, Debug)]
pub struct Constraint {
    pub name: SmolStr,
    pub lhs: ExprId,
    pub sense: Sense,
    pub rhs: f64,
    pub active: bool,
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
}

impl<'a> IntoRhs<'a> for f64 {
    fn fold_rhs(self, lhs: Expr<'a>) -> (Expr<'a>, f64) {
        (lhs, self)
    }
}

impl<'a> IntoRhs<'a> for i32 {
    fn fold_rhs(self, lhs: Expr<'a>) -> (Expr<'a>, f64) {
        (lhs, f64::from(self))
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
