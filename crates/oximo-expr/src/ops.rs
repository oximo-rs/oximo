use crate::degree::{AddDegree, Degree, DivDegree, MulDegree, Nonlinear};
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::arena::{Children, ExprArenaCell, ExprId, ExprNode};
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
}

/// Which flat extrema node an [`ExtremaBuilder`] constructs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Extremum {
    Min,
    Max,
}

impl Extremum {
    pub(crate) fn node(self, children: Children) -> ExprNode {
        match self {
            Self::Min => ExprNode::Min(children),
            Self::Max => ExprNode::Max(children),
        }
    }
}

/// Return the children of a node when it is the requested extrema kind.
pub(crate) fn matching_children(node: &ExprNode, kind: Extremum) -> Option<&[ExprId]> {
    match (kind, node) {
        (Extremum::Min, ExprNode::Min(children)) | (Extremum::Max, ExprNode::Max(children)) => {
            Some(children.as_slice())
        }
        _ => None,
    }
}

/// Streaming builder for flat n-ary minima and maxima.
#[derive(Debug)]
#[doc(hidden)]
pub struct ExtremaBuilder<'a, E: Degree> {
    arena: Option<&'a ExprArenaCell>,
    same_arena: bool,
    kind: Extremum,
    ids: Children,
    _degree: std::marker::PhantomData<fn() -> E>,
}

impl<'a, E: Degree> ExtremaBuilder<'a, E> {
    fn new(kind: Extremum) -> Self {
        Self {
            arena: None,
            same_arena: true,
            kind,
            ids: Children::new(),
            _degree: std::marker::PhantomData,
        }
    }

    /// Start building a flat n-ary minimum.
    #[doc(hidden)]
    pub fn min() -> Self {
        Self::new(Extremum::Min)
    }

    /// Start building a flat n-ary maximum.
    #[doc(hidden)]
    pub fn max() -> Self {
        Self::new(Extremum::Max)
    }

    /// Add one term, flattening matching extrema nodes in place.
    /// Arena ownership is checked by [`Self::finish`] after all terms are pushed.
    #[doc(hidden)]
    pub fn push(&mut self, expr: Expr<'a, E>) {
        match self.arena {
            Some(arena) => self.same_arena &= std::ptr::eq(arena, expr.arena),
            None => self.arena = Some(expr.arena),
        }
        expr.arena.with_ref(|arena| match matching_children(arena.get(expr.id), self.kind) {
            Some(children) => self.ids.extend_from_slice(children),
            None => self.ids.push(expr.id),
        });
    }

    /// Finish the extrema node, returning `None` when no terms were pushed.
    ///
    /// # Panics
    ///
    /// Panics if pushed terms belong to different expression arenas.
    #[doc(hidden)]
    pub fn finish(self) -> Option<Expr<'a, Nonlinear>> {
        let arena = self.arena?;
        assert!(self.same_arena, "expressions belong to different arenas");
        if self.ids.len() == 1 {
            return Some(Expr::from_id(self.ids[0], arena));
        }
        let id = arena.with_mut(|arena| arena.push(self.kind.node(self.ids)));
        Some(Expr::from_id(id, arena))
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
