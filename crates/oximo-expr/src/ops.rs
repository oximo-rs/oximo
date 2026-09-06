use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::arena::{Children, ExprId};
use crate::handle::Expr;
use crate::linear::{add_into, add_n, div_into, mul_into, neg_into, sub_into};

// -----------------------------------------------------------------------------
// Expr <op> Expr
// -----------------------------------------------------------------------------

impl<'a> Add for Expr<'a> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        self.assert_same_arena(rhs);
        let id = self.arena.with_mut(|arena| add_into(arena, self.id, rhs.id));
        Self::new(id, self.arena)
    }
}

impl<'a> Sub for Expr<'a> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self.assert_same_arena(rhs);
        let id = self.arena.with_mut(|arena| sub_into(arena, self.id, rhs.id));
        Self::new(id, self.arena)
    }
}

impl<'a> Mul for Expr<'a> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        self.assert_same_arena(rhs);
        let id = self.arena.with_mut(|arena| mul_into(arena, self.id, rhs.id));
        Self::new(id, self.arena)
    }
}

impl<'a> Div for Expr<'a> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        self.assert_same_arena(rhs);
        let id = self.arena.with_mut(|arena| div_into(arena, self.id, rhs.id));
        Self::new(id, self.arena)
    }
}

impl<'a> Neg for Expr<'a> {
    type Output = Self;
    fn neg(self) -> Self {
        let id = self.arena.with_mut(|arena| neg_into(arena, self.id));
        Self::new(id, self.arena)
    }
}

// -----------------------------------------------------------------------------
// Expr <op> f64 / f64 <op> Expr, and the same for i32 because `2 * x`
// without type annotation is the most common ergonomic case.
// -----------------------------------------------------------------------------

macro_rules! impl_scalar_ops {
    ($scalar:ty, $to_f64:expr) => {
        impl<'a> Add<$scalar> for Expr<'a> {
            type Output = Self;
            fn add(self, rhs: $scalar) -> Self {
                let id = self.arena.with_mut(|arena| {
                    let rhs_id = arena.constant($to_f64(rhs));
                    add_into(arena, self.id, rhs_id)
                });
                Self::new(id, self.arena)
            }
        }

        impl<'a> Add<Expr<'a>> for $scalar {
            type Output = Expr<'a>;
            fn add(self, rhs: Expr<'a>) -> Expr<'a> {
                rhs + self
            }
        }

        impl<'a> Sub<$scalar> for Expr<'a> {
            type Output = Self;
            fn sub(self, rhs: $scalar) -> Self {
                let id = self.arena.with_mut(|arena| {
                    let rhs_id = arena.constant($to_f64(rhs));
                    sub_into(arena, self.id, rhs_id)
                });
                Self::new(id, self.arena)
            }
        }

        impl<'a> Sub<Expr<'a>> for $scalar {
            type Output = Expr<'a>;
            fn sub(self, rhs: Expr<'a>) -> Expr<'a> {
                let id = rhs.arena.with_mut(|arena| {
                    let lhs_id = arena.constant($to_f64(self));
                    sub_into(arena, lhs_id, rhs.id)
                });
                Expr::new(id, rhs.arena)
            }
        }

        impl<'a> Mul<$scalar> for Expr<'a> {
            type Output = Self;
            fn mul(self, rhs: $scalar) -> Self {
                let id = self.arena.with_mut(|arena| {
                    let rhs_id = arena.constant($to_f64(rhs));
                    mul_into(arena, self.id, rhs_id)
                });
                Self::new(id, self.arena)
            }
        }

        impl<'a> Mul<Expr<'a>> for $scalar {
            type Output = Expr<'a>;
            fn mul(self, rhs: Expr<'a>) -> Expr<'a> {
                rhs * self
            }
        }

        impl<'a> Div<$scalar> for Expr<'a> {
            type Output = Self;
            fn div(self, rhs: $scalar) -> Self {
                let id = self.arena.with_mut(|arena| {
                    let rhs_id = arena.constant($to_f64(rhs));
                    div_into(arena, self.id, rhs_id)
                });
                Self::new(id, self.arena)
            }
        }

        impl<'a> Div<Expr<'a>> for $scalar {
            type Output = Expr<'a>;
            fn div(self, rhs: Expr<'a>) -> Expr<'a> {
                let id = rhs.arena.with_mut(|arena| {
                    let lhs_id = arena.constant($to_f64(self));
                    div_into(arena, lhs_id, rhs.id)
                });
                Expr::new(id, rhs.arena)
            }
        }
    };
}

impl_scalar_ops!(f64, core::convert::identity);
impl_scalar_ops!(i32, f64::from);

// -----------------------------------------------------------------------------
// std::iter::Sum: the first element of the iterator carries the arena handle,
// so no external zero is required. Collected into a single flat n-ary `Add`.
// -----------------------------------------------------------------------------

fn sum_children(first: ExprId, rest: impl Iterator<Item = ExprId>) -> Children {
    let capacity = rest.size_hint().0.saturating_add(1);
    let mut children = Children::new();
    if capacity > children.inline_size() {
        // We use Vec's extend loop to avoid checking SmallVec's storage mode per term.
        let mut ids = Vec::with_capacity(capacity);
        ids.push(first);
        ids.extend(rest);
        Children::from_vec(ids)
    } else {
        children.push(first);
        children.extend(rest);
        children
    }
}

impl<'a> std::iter::Sum for Expr<'a> {
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        let first = iter.next().expect("Expr::sum on empty iterator");
        let ids = sum_children(
            first.id,
            iter.map(|expr| {
                first.assert_same_arena(expr);
                expr.id
            }),
        );
        let id = first.arena.with_mut(|arena| add_n(arena, ids));
        Self::new(id, first.arena)
    }
}

impl<'a> Expr<'a> {
    /// Macro-helper summation that evaluates every term before checking arena
    /// ownership, matching the former collect-then-sum behavior. Returns `None`
    /// for an empty domain so the caller can preserve its own diagnostic.
    #[doc(hidden)]
    pub fn __sum_terms(mut iter: impl Iterator<Item = Self>) -> Option<Self> {
        let first = iter.next()?;
        let mut same_arena = true;
        let ids = sum_children(
            first.id,
            iter.map(|expr| {
                same_arena &= std::ptr::eq(first.arena, expr.arena);
                expr.id
            }),
        );
        assert!(same_arena, "expressions belong to different arenas");
        let id = first.arena.with_mut(|arena| add_n(arena, ids));
        Some(Self::new(id, first.arena))
    }
}

impl<'a, 'b> std::iter::Sum<&'b Expr<'a>> for Expr<'a> {
    fn sum<I: Iterator<Item = &'b Expr<'a>>>(iter: I) -> Self {
        iter.copied().sum()
    }
}

/// Dot product of expressions with scalar coefficients: `sum_{i} c_i * e_i`.
///
/// Both arguments are slices. Pass owned containers by reference:
/// `&vec`, `vec.as_slice()`, or `&array`.
///
/// # Panics
/// Panics if `exprs` and `coeffs` have different lengths, or if `exprs`
/// is empty (the result needs an arena handle).
pub fn dot<'a>(exprs: &[Expr<'a>], coeffs: &[f64]) -> Expr<'a> {
    assert_eq!(
        exprs.len(),
        coeffs.len(),
        "dot: length mismatch (exprs.len() = {}, coeffs.len() = {})",
        exprs.len(),
        coeffs.len(),
    );
    exprs.iter().zip(coeffs).map(|(e, c)| *c * *e).sum()
}
