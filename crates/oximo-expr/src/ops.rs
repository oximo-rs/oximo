use crate::degree::{AddDegree, Degree, DivDegree, MulDegree, Nonlinear};
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::arena::{Children, ExprId, ExprNode};
use crate::handle::Expr;
use crate::linear::{add_into, add_n, div_into, mul_into, neg_into, sub_into};

// -----------------------------------------------------------------------------
// Expr <op> Expr
// -----------------------------------------------------------------------------

impl<'a, L: Degree + AddDegree<R>, R: Degree> Add<Expr<'a, R>> for Expr<'a, L> {
    type Output = Expr<'a, <L as AddDegree<R>>::Output>;
    fn add(self, rhs: Expr<'a, R>) -> Self::Output {
        self.assert_same_arena(rhs);
        let id = self.arena.with_mut(|arena| add_into(arena, self.id, rhs.id));
        Expr::from_id(id, self.arena)
    }
}

impl<'a, L: Degree + AddDegree<R>, R: Degree> Sub<Expr<'a, R>> for Expr<'a, L> {
    type Output = Expr<'a, <L as AddDegree<R>>::Output>;
    fn sub(self, rhs: Expr<'a, R>) -> Self::Output {
        self.assert_same_arena(rhs);
        let id = self.arena.with_mut(|arena| sub_into(arena, self.id, rhs.id));
        Expr::from_id(id, self.arena)
    }
}

impl<'a, L: Degree + MulDegree<R>, R: Degree> Mul<Expr<'a, R>> for Expr<'a, L> {
    type Output = Expr<'a, <L as MulDegree<R>>::Output>;
    fn mul(self, rhs: Expr<'a, R>) -> Self::Output {
        self.assert_same_arena(rhs);
        let id = self.arena.with_mut(|arena| mul_into(arena, self.id, rhs.id));
        Expr::from_id(id, self.arena)
    }
}

impl<'a, L: Degree + DivDegree<R>, R: Degree> Div<Expr<'a, R>> for Expr<'a, L> {
    type Output = Expr<'a, <L as DivDegree<R>>::Output>;
    fn div(self, rhs: Expr<'a, R>) -> Self::Output {
        self.assert_same_arena(rhs);
        let id = self.arena.with_mut(|arena| div_into(arena, self.id, rhs.id));
        Expr::from_id(id, self.arena)
    }
}

impl<'a, D: Degree> Neg for Expr<'a, D> {
    type Output = Self;
    fn neg(self) -> Self {
        let id = self.arena.with_mut(|arena| neg_into(arena, self.id));
        Expr::from_id(id, self.arena)
    }
}

// -----------------------------------------------------------------------------
// Expr <op> f64 / f64 <op> Expr, and the same for i32 because `2 * x`
// without type annotation is the most common ergonomic case.
// -----------------------------------------------------------------------------

macro_rules! impl_scalar_ops {
    ($scalar:ty, $to_f64:expr) => {
        impl<'a, D: Degree> Add<$scalar> for Expr<'a, D> {
            type Output = Self;
            fn add(self, rhs: $scalar) -> Self {
                let id = self.arena.with_mut(|arena| {
                    let rhs_id = arena.constant($to_f64(rhs));
                    add_into(arena, self.id, rhs_id)
                });
                Expr::from_id(id, self.arena)
            }
        }

        impl<'a, D: Degree> Add<Expr<'a, D>> for $scalar {
            type Output = Expr<'a, D>;
            fn add(self, rhs: Expr<'a, D>) -> Expr<'a, D> {
                rhs + self
            }
        }

        impl<'a, D: Degree> Sub<$scalar> for Expr<'a, D> {
            type Output = Self;
            fn sub(self, rhs: $scalar) -> Self {
                let id = self.arena.with_mut(|arena| {
                    let rhs_id = arena.constant($to_f64(rhs));
                    sub_into(arena, self.id, rhs_id)
                });
                Expr::from_id(id, self.arena)
            }
        }

        impl<'a, D: Degree> Sub<Expr<'a, D>> for $scalar {
            type Output = Expr<'a, D>;
            fn sub(self, rhs: Expr<'a, D>) -> Expr<'a, D> {
                let id = rhs.arena.with_mut(|arena| {
                    let lhs_id = arena.constant($to_f64(self));
                    sub_into(arena, lhs_id, rhs.id)
                });
                Expr::from_id(id, rhs.arena)
            }
        }

        impl<'a, D: Degree> Mul<$scalar> for Expr<'a, D> {
            type Output = Self;
            fn mul(self, rhs: $scalar) -> Self {
                let id = self.arena.with_mut(|arena| {
                    let rhs_id = arena.constant($to_f64(rhs));
                    mul_into(arena, self.id, rhs_id)
                });
                Expr::from_id(id, self.arena)
            }
        }

        impl<'a, D: Degree> Mul<Expr<'a, D>> for $scalar {
            type Output = Expr<'a, D>;
            fn mul(self, rhs: Expr<'a, D>) -> Expr<'a, D> {
                rhs * self
            }
        }

        impl<'a, D: Degree> Div<$scalar> for Expr<'a, D> {
            type Output = Self;
            fn div(self, rhs: $scalar) -> Self {
                assert!($to_f64(rhs) != 0.0, "expression division by zero");
                let id = self.arena.with_mut(|arena| {
                    let rhs_id = arena.constant($to_f64(rhs));
                    div_into(arena, self.id, rhs_id)
                });
                Expr::from_id(id, self.arena)
            }
        }

        impl<'a, D: Degree> Div<Expr<'a, D>> for $scalar {
            type Output = Expr<'a, Nonlinear>;
            fn div(self, rhs: Expr<'a, D>) -> Expr<'a, Nonlinear> {
                let id = rhs.arena.with_mut(|arena| {
                    let lhs_id = arena.constant($to_f64(self));
                    div_into(arena, lhs_id, rhs.id)
                });
                Expr::from_id(id, rhs.arena)
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

impl<'a, D: Degree> std::iter::Sum for Expr<'a, D> {
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
        Expr::from_id(id, first.arena)
    }
}

impl<'a, D: Degree> Expr<'a, D> {
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
        Some(Expr::from_id(id, first.arena))
    }

    /// Model-anchored summation. Validate all term owners before emitting the
    /// sum, and construct zero in the supplied arena when there are no terms.
    #[doc(hidden)]
    pub fn __sum_terms_in(
        arena: &'a crate::ExprArenaCell,
        mut iter: impl Iterator<Item = Self>,
    ) -> Self {
        let Some(first) = iter.next() else {
            return Expr::from_id(arena.with_mut(|a| a.constant(0.0)), arena);
        };
        let mut same_arena = std::ptr::eq(arena, first.arena);
        let ids = sum_children(
            first.id,
            iter.map(|expr| {
                same_arena &= std::ptr::eq(arena, expr.arena);
                expr.id
            }),
        );
        assert!(same_arena, "sum! terms belong to a different model");
        let id = arena.with_mut(|arena| add_n(arena, ids));
        Expr::from_id(id, arena)
    }

    fn extrema_terms(
        mut iter: impl Iterator<Item = Self>,
        is_min: bool,
    ) -> Option<Expr<'a, Nonlinear>> {
        let first = iter.next()?;
        let mut same_arena = true;
        let mut ids = Children::new();
        let mut append = |expr: Self| {
            same_arena &= std::ptr::eq(first.arena, expr.arena);
            expr.arena.with_ref(|arena| match arena.get(expr.id) {
                ExprNode::Min(children) if is_min => ids.extend_from_slice(children),
                ExprNode::Max(children) if !is_min => ids.extend_from_slice(children),
                _ => ids.push(expr.id),
            });
        };
        append(first);
        for expr in iter {
            append(expr);
        }
        assert!(same_arena, "expressions belong to different arenas");
        if ids.len() == 1 {
            return Some(Expr::from_id(ids[0], first.arena));
        }
        let id = first.arena.with_mut(|arena| {
            arena.push(if is_min { ExprNode::Min(ids) } else { ExprNode::Max(ids) })
        });
        Some(Expr::from_id(id, first.arena))
    }

    /// Macro support for constructing a flat n-ary minimum.
    #[doc(hidden)]
    pub fn __min_terms(iter: impl Iterator<Item = Self>) -> Option<Expr<'a, Nonlinear>> {
        Self::extrema_terms(iter, true)
    }

    /// Macro support for constructing a flat n-ary maximum.
    #[doc(hidden)]
    pub fn __max_terms(iter: impl Iterator<Item = Self>) -> Option<Expr<'a, Nonlinear>> {
        Self::extrema_terms(iter, false)
    }
}

impl<'a, 'b, D: Degree> std::iter::Sum<&'b Expr<'a, D>> for Expr<'a, D> {
    fn sum<I: Iterator<Item = &'b Expr<'a, D>>>(iter: I) -> Self {
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
pub fn dot<'a, D: Degree>(exprs: &[Expr<'a, D>], coeffs: &[f64]) -> Expr<'a, D> {
    assert_eq!(
        exprs.len(),
        coeffs.len(),
        "dot: length mismatch (exprs.len() = {}, coeffs.len() = {})",
        exprs.len(),
        coeffs.len(),
    );
    exprs.iter().zip(coeffs).map(|(e, c)| *c * *e).sum()
}
