use crate::degree::{AddDegree, Degree, DivDegree, ExpressionDegree, MulDegree, Nonlinear};
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::arena::{Children, ExprArenaCell, ExprId, ExprNode};
use crate::handle::Expr;
use crate::linear::{
    add_into, add_n, add_with_ownership, div_into, mul_affine_into, mul_into, neg_into, sub_into,
};

/// Construction-local ownership view.
struct Operand<'a> {
    id: ExprId,
    arena: &'a ExprArenaCell,
    owned: bool,
    seal: bool,
}

impl<'a> Operand<'a> {
    #[inline]
    fn owned<D: Degree>(expression: Expr<'a, D>) -> Self {
        Self { id: expression.id, arena: expression.arena, owned: expression.owned, seal: false }
    }

    #[inline]
    fn borrowed<D: Degree>(expression: &Expr<'a, D>) -> Self {
        Self { id: expression.id, arena: expression.arena, owned: false, seal: expression.owned }
    }

    fn assert_same_arena(&self, other: &Self) {
        assert!(std::ptr::eq(self.arena, other.arena), "expressions belong to different arenas");
    }

    fn seal_in(&self, arena: &dyn crate::arena::ArenaAccess) {
        if self.seal {
            arena.seal(self.id);
        }
    }
}

#[inline]
fn build_binary<'a, D: Degree>(
    lhs: Operand<'a>,
    rhs: Operand<'a>,
    construct: impl FnOnce(&mut dyn crate::arena::ArenaAccess, &Operand<'a>, &Operand<'a>) -> ExprId,
) -> Expr<'a, D> {
    lhs.assert_same_arena(&rhs);
    Expr::build(lhs.arena, |arena| {
        lhs.seal_in(arena);
        rhs.seal_in(arena);
        construct(arena, &lhs, &rhs)
    })
}

#[inline]
fn build_unary<'a, D: Degree>(
    input: Operand<'a>,
    construct: impl FnOnce(&mut dyn crate::arena::ArenaAccess, &Operand<'a>) -> ExprId,
) -> Expr<'a, D> {
    Expr::build(input.arena, |arena| {
        input.seal_in(arena);
        construct(arena, &input)
    })
}

#[inline]
fn prefer_right_destination(
    arena: &dyn crate::arena::ArenaAccess,
    lhs: &Operand<'_>,
    rhs: &Operand<'_>,
) -> bool {
    rhs.owned
        && (!lhs.owned
            || matches!((arena.get(lhs.id), arena.get(rhs.id)),
                (ExprNode::Quadratic(left), ExprNode::Quadratic(right))
                    if right.linear.len().saturating_add(right.quadratic.len())
                        > left.linear.len().saturating_add(left.quadratic.len())))
}

#[inline]
fn add_expr<'a, D: Degree>(lhs: Operand<'a>, rhs: Operand<'a>) -> Expr<'a, D> {
    lhs.assert_same_arena(&rhs);

    // Addition can retain an already-owned root even after a checked narrowing
    // to Affine. We preserve its ownership result and we don't reclassify it.
    let (id, owned) = lhs.arena.with_mut(|arena| {
        lhs.seal_in(arena);
        rhs.seal_in(arena);
        let prefer_right = prefer_right_destination(arena, &lhs, &rhs);
        if prefer_right && let Some(id) = arena.accumulate_quadratic(rhs.id, lhs.id, 1.0) {
            return (id, true);
        }
        if lhs.owned
            && let Some(id) = arena.accumulate_quadratic(lhs.id, rhs.id, 1.0)
        {
            return (id, true);
        }
        if rhs.owned
            && !prefer_right
            && let Some(id) = arena.accumulate_quadratic(rhs.id, lhs.id, 1.0)
        {
            return (id, true);
        }
        add_with_ownership(arena, lhs.id, rhs.id, !lhs.owned && !rhs.owned)
    });

    Expr::from_construction(id, owned, lhs.arena)
}

#[inline]
fn sub_expr<'a, D: Degree>(lhs: Operand<'a>, rhs: Operand<'a>) -> Expr<'a, D> {
    lhs.assert_same_arena(&rhs);
    let (id, owned) = lhs.arena.with_mut(|arena| {
        lhs.seal_in(arena);
        rhs.seal_in(arena);
        let prefer_right = prefer_right_destination(arena, &lhs, &rhs);
        if prefer_right && let Some(id) = subtract_into_right(arena, lhs.id, rhs.id) {
            return (id, true);
        }
        if lhs.owned
            && let Some(id) = arena.accumulate_quadratic(lhs.id, rhs.id, -1.0)
        {
            return (id, true);
        }
        if rhs.owned
            && !prefer_right
            && let Some(id) = subtract_into_right(arena, lhs.id, rhs.id)
        {
            return (id, true);
        }
        let id = sub_into(arena, lhs.id, rhs.id);
        (id, crate::degree::may_be_quadratic::<D>() && arena.quadratic_root(id))
    });
    Expr::from_construction(id, owned, lhs.arena)
}

/// Store `lhs - rhs` in the uniquely owned right operand.
fn subtract_into_right(
    arena: &mut dyn crate::arena::ArenaAccess,
    lhs: ExprId,
    rhs: ExprId,
) -> Option<ExprId> {
    arena.scale_quadratic(rhs, -1.0)?;
    if let Some(id) = arena.accumulate_quadratic(rhs, lhs, 1.0) {
        return Some(id);
    }

    arena.scale_quadratic(rhs, -1.0).expect("failed accumulation preserves the unique quadratic");
    None
}

#[inline]
fn mul_expr<'a, D: Degree>(lhs: Operand<'a>, rhs: Operand<'a>) -> Expr<'a, D> {
    build_binary(lhs, rhs, |arena, lhs, rhs| {
        if crate::degree::may_be_quadratic::<D>() {
            if lhs.owned
                && let ExprNode::Const(c) = arena.get(rhs.id)
                && let Some(id) = arena.scale_quadratic(lhs.id, *c)
            {
                return id;
            }
            if rhs.owned
                && let ExprNode::Const(c) = arena.get(lhs.id)
                && let Some(id) = arena.scale_quadratic(rhs.id, *c)
            {
                return id;
            }
            mul_into(arena, lhs.id, rhs.id)
        } else {
            mul_affine_into(arena, lhs.id, rhs.id)
        }
    })
}

#[inline]
fn div_expr<'a, D: Degree>(lhs: Operand<'a>, rhs: Operand<'a>) -> Expr<'a, D> {
    build_binary(lhs, rhs, |arena, lhs, rhs| div_into(arena, lhs.id, rhs.id))
}

#[inline]
fn neg_expr<'a, D: Degree>(input: Operand<'a>) -> Expr<'a, D> {
    build_unary(input, |arena, input| {
        if input.owned
            && let Some(id) = arena.scale_quadratic(input.id, -1.0)
        {
            return id;
        }
        neg_into(arena, input.id)
    })
}

// All ownership combinations use the same construction logic.
// Only consumed operands remain eligible for in-place accumulation.
macro_rules! binary_ops {
    ($trait:ident, $method:ident, $bound:ident, $construct:ident) => {
        impl<'a, L: Degree + $bound<R>, R: Degree> $trait<Expr<'a, R>> for Expr<'a, L> {
            type Output = Expr<'a, <L as $bound<R>>::Output>;
            fn $method(self, rhs: Expr<'a, R>) -> Self::Output {
                $construct(Operand::owned(self), Operand::owned(rhs))
            }
        }
        impl<'a, L: Degree + $bound<R>, R: Degree> $trait<&Expr<'a, R>> for Expr<'a, L> {
            type Output = Expr<'a, <L as $bound<R>>::Output>;
            fn $method(self, rhs: &Expr<'a, R>) -> Self::Output {
                $construct(Operand::owned(self), Operand::borrowed(rhs))
            }
        }
        impl<'a, L: Degree + $bound<R>, R: Degree> $trait<Expr<'a, R>> for &Expr<'a, L> {
            type Output = Expr<'a, <L as $bound<R>>::Output>;
            fn $method(self, rhs: Expr<'a, R>) -> Self::Output {
                $construct(Operand::borrowed(self), Operand::owned(rhs))
            }
        }
        impl<'a, L: Degree + $bound<R>, R: Degree> $trait<&Expr<'a, R>> for &Expr<'a, L> {
            type Output = Expr<'a, <L as $bound<R>>::Output>;
            fn $method(self, rhs: &Expr<'a, R>) -> Self::Output {
                $construct(Operand::borrowed(self), Operand::borrowed(rhs))
            }
        }
    };
}
binary_ops!(Add, add, AddDegree, add_expr);
binary_ops!(Sub, sub, AddDegree, sub_expr);
binary_ops!(Mul, mul, MulDegree, mul_expr);
binary_ops!(Div, div, DivDegree, div_expr);

impl<'a, D: Degree> Neg for Expr<'a, D> {
    type Output = Expr<'a, D::Computed>;
    fn neg(self) -> Self::Output {
        neg_expr(Operand::owned(self))
    }
}
impl<'a, D: Degree> Neg for &Expr<'a, D> {
    type Output = Expr<'a, D::Computed>;
    fn neg(self) -> Self::Output {
        neg_expr(Operand::borrowed(self))
    }
}

// -----------------------------------------------------------------------------
// Expr <op> f64 / f64 <op> Expr, and the same for i32 because `2 * x`
// without type annotation is the most common ergonomic case.
// -----------------------------------------------------------------------------

macro_rules! impl_scalar_ops {
    ($scalar:ty, $to_f64:expr) => {
        impl<'a, D: Degree> Add<$scalar> for Expr<'a, D> {
            type Output = Expr<'a, D::Computed>;
            fn add(self, rhs: $scalar) -> Self::Output {
                Expr::build(self.arena, |arena| {
                    let rhs_id = arena.constant($to_f64(rhs));
                    if self.owned
                        && let Some(id) = arena.accumulate_quadratic(self.id, rhs_id, 1.0)
                    {
                        return id;
                    }
                    add_into(arena, self.id, rhs_id)
                })
            }
        }

        impl<'a, D: Degree> Add<Expr<'a, D>> for $scalar {
            type Output = Expr<'a, D::Computed>;
            fn add(self, rhs: Expr<'a, D>) -> Expr<'a, D::Computed> {
                rhs + self
            }
        }

        impl<'a, D: Degree> Sub<$scalar> for Expr<'a, D> {
            type Output = Expr<'a, D::Computed>;
            fn sub(self, rhs: $scalar) -> Self::Output {
                Expr::build(self.arena, |arena| {
                    let rhs_id = arena.constant($to_f64(rhs));
                    if self.owned
                        && let Some(id) = arena.accumulate_quadratic(self.id, rhs_id, -1.0)
                    {
                        return id;
                    }
                    sub_into(arena, self.id, rhs_id)
                })
            }
        }

        impl<'a, D: Degree> Sub<Expr<'a, D>> for $scalar {
            type Output = Expr<'a, D::Computed>;
            fn sub(self, rhs: Expr<'a, D>) -> Expr<'a, D::Computed> {
                let (id, owned) = rhs.arena.build_expr(
                    crate::degree::may_be_quadratic::<D::Computed>() || rhs.owned,
                    |arena| {
                        let lhs_id = arena.constant($to_f64(self));
                        if rhs.owned
                            && let Some(id) = subtract_into_right(arena, lhs_id, rhs.id)
                        {
                            return id;
                        }
                        sub_into(arena, lhs_id, rhs.id)
                    },
                );
                Expr::from_construction(id, owned, rhs.arena)
            }
        }

        impl<'a, D: Degree> Mul<$scalar> for Expr<'a, D> {
            type Output = Expr<'a, D::Computed>;
            fn mul(self, rhs: $scalar) -> Self::Output {
                Expr::build(self.arena, |arena| {
                    if crate::degree::may_be_quadratic::<D>()
                        && self.owned
                        && let Some(id) = arena.scale_quadratic(self.id, $to_f64(rhs))
                    {
                        return id;
                    }
                    let rhs_id = arena.constant($to_f64(rhs));
                    if crate::degree::may_be_quadratic::<D>() {
                        mul_into(arena, self.id, rhs_id)
                    } else {
                        mul_affine_into(arena, self.id, rhs_id)
                    }
                })
            }
        }

        impl<'a, D: Degree> Mul<Expr<'a, D>> for $scalar {
            type Output = Expr<'a, D::Computed>;
            fn mul(self, rhs: Expr<'a, D>) -> Expr<'a, D::Computed> {
                rhs * self
            }
        }

        impl<'a, D: Degree> Div<$scalar> for Expr<'a, D> {
            type Output = Expr<'a, D::Computed>;
            fn div(self, rhs: $scalar) -> Self::Output {
                assert!($to_f64(rhs) != 0.0, "expression division by zero");
                Expr::build(self.arena, |arena| {
                    let rhs_id = arena.constant($to_f64(rhs));
                    div_into(arena, self.id, rhs_id)
                })
            }
        }

        impl<'a, D: Degree> Div<Expr<'a, D>> for $scalar {
            type Output = Expr<'a, Nonlinear>;
            fn div(self, rhs: Expr<'a, D>) -> Expr<'a, Nonlinear> {
                Expr::build(rhs.arena, |arena| {
                    let lhs_id = arena.constant($to_f64(self));
                    div_into(arena, lhs_id, rhs.id)
                })
            }
        }
    };
}

macro_rules! borrowed_scalar {
    ($scalar:ty) => {
        impl<'a, D: Degree> Add<$scalar> for &Expr<'a, D> {
            type Output = Expr<'a, D::Computed>;
            fn add(self, c: $scalar) -> Self::Output {
                build_unary(Operand::borrowed(self), |arena, input| {
                    let constant = arena.constant(c.into());
                    add_into(arena, input.id, constant)
                })
            }
        }
        impl<'a, D: Degree> Sub<$scalar> for &Expr<'a, D> {
            type Output = Expr<'a, D::Computed>;
            fn sub(self, c: $scalar) -> Self::Output {
                build_unary(Operand::borrowed(self), |arena, input| {
                    let constant = arena.constant(c.into());
                    sub_into(arena, input.id, constant)
                })
            }
        }
        impl<'a, D: Degree> Mul<$scalar> for &Expr<'a, D> {
            type Output = Expr<'a, D::Computed>;
            fn mul(self, c: $scalar) -> Self::Output {
                build_unary(Operand::borrowed(self), |arena, input| {
                    let constant = arena.constant(c.into());
                    if crate::degree::may_be_quadratic::<D>() {
                        mul_into(arena, input.id, constant)
                    } else {
                        mul_affine_into(arena, input.id, constant)
                    }
                })
            }
        }
        impl<'a, D: Degree> Div<$scalar> for &Expr<'a, D> {
            type Output = Expr<'a, D::Computed>;
            fn div(self, c: $scalar) -> Self::Output {
                assert!(f64::from(c) != 0.0, "expression division by zero");
                build_unary(Operand::borrowed(self), |arena, input| {
                    let constant = arena.constant(c.into());
                    div_into(arena, input.id, constant)
                })
            }
        }
        impl<'a, D: Degree> Add<&Expr<'a, D>> for $scalar {
            type Output = Expr<'a, D::Computed>;
            fn add(self, x: &Expr<'a, D>) -> Self::Output {
                x + self
            }
        }
        impl<'a, D: Degree> Sub<&Expr<'a, D>> for $scalar {
            type Output = Expr<'a, D::Computed>;
            fn sub(self, x: &Expr<'a, D>) -> Self::Output {
                build_unary(Operand::borrowed(x), |arena, input| {
                    let constant = arena.constant(self.into());
                    sub_into(arena, constant, input.id)
                })
            }
        }
        impl<'a, D: Degree> Mul<&Expr<'a, D>> for $scalar {
            type Output = Expr<'a, D::Computed>;
            fn mul(self, x: &Expr<'a, D>) -> Self::Output {
                x * self
            }
        }
        impl<'a, D: Degree> Div<&Expr<'a, D>> for $scalar {
            type Output = Expr<'a, Nonlinear>;
            fn div(self, x: &Expr<'a, D>) -> Self::Output {
                build_unary(Operand::borrowed(x), |arena, input| {
                    let constant = arena.constant(self.into());
                    div_into(arena, constant, input.id)
                })
            }
        }
    };
}
borrowed_scalar!(f64);
borrowed_scalar!(i32);

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

impl<'a, D: ExpressionDegree> std::iter::Sum for Expr<'a, D> {
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        let first = iter.next().expect("Expr::sum on empty iterator");
        let ids = sum_children(
            first.id,
            iter.map(|expr| {
                first.assert_same_arena(&expr);
                expr.id
            }),
        );
        if ids.len() == 1 {
            return first;
        }
        let (id, owned) = first.arena.with_mut(|arena| add_n(arena, ids));
        Expr::from_construction(id, owned, first.arena)
    }
}

impl<'a, D: Degree> Expr<'a, D> {
    /// Macro-helper summation that evaluates every term before checking arena
    /// ownership, matching the former collect-then-sum behavior. Returns `None`
    /// for an empty domain so the caller can preserve its own diagnostic.
    #[doc(hidden)]
    pub fn __sum_terms<E: crate::ExprInput<'a, Degree = D>>(
        iter: impl Iterator<Item = E>,
    ) -> Option<Expr<'a, D::Computed>> {
        let mut iter = iter.map(crate::ExprInput::into_expression);
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
        if ids.len() == 1 {
            return Some(first.into_degree());
        }
        let (id, owned) = first.arena.with_mut(|arena| add_n(arena, ids));
        Some(Expr::from_construction(id, owned, first.arena))
    }

    /// Model-anchored summation. Validate all term owners before emitting the
    /// sum, and construct zero in the supplied arena when there are no terms.
    #[doc(hidden)]
    pub fn __sum_terms_in<E: crate::ExprInput<'a, Degree = D>>(
        arena: &'a crate::ExprArenaCell,
        iter: impl Iterator<Item = E>,
    ) -> Expr<'a, D::Computed> {
        let mut iter = iter.map(crate::ExprInput::into_expression);
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
        if ids.len() == 1 {
            return first.into_degree();
        }
        let (id, owned) = arena.with_mut(|arena| add_n(arena, ids));
        Expr::from_construction(id, owned, arena)
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
    pub fn push<I: crate::ExprInput<'a, Degree = E>>(&mut self, expr: I) {
        let expr = expr.as_expression();
        expr.seal();
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

impl<'a, 'b, D: ExpressionDegree> std::iter::Sum<&'b Expr<'a, D>> for Expr<'a, D> {
    fn sum<I: Iterator<Item = &'b Expr<'a, D>>>(iter: I) -> Self {
        iter.cloned().sum()
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
pub fn dot<'a, D: Degree>(exprs: &[Expr<'a, D>], coeffs: &[f64]) -> Expr<'a, D::Computed> {
    assert_eq!(
        exprs.len(),
        coeffs.len(),
        "dot: length mismatch (exprs.len() = {}, coeffs.len() = {})",
        exprs.len(),
        coeffs.len(),
    );
    exprs.iter().zip(coeffs).map(|(e, c)| *c * e).sum()
}

macro_rules! sum_leaves {
    ($leaf:ty => $degree:ty) => {
        impl<'a> std::iter::Sum<Expr<'a, $leaf>> for Expr<'a, $degree> {
            fn sum<I: Iterator<Item = Expr<'a, $leaf>>>(iter: I) -> Self {
                iter.map(Expr::into_degree::<$degree>).sum()
            }
        }
        impl<'a, 'b> std::iter::Sum<&'b Expr<'a, $leaf>> for Expr<'a, $degree> {
            fn sum<I: Iterator<Item = &'b Expr<'a, $leaf>>>(iter: I) -> Self {
                iter.copied().sum()
            }
        }
    };
}
sum_leaves!(crate::VariableDegree => crate::Affine);
sum_leaves!(crate::ParameterDegree => crate::Constant);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExprArena, VarId, evaluate};

    #[test]
    fn subtraction_reuses_the_larger_consumed_quadratic_with_one_guard() {
        for larger_right in [false, true] {
            let arena = ExprArenaCell::new(ExprArena::new());
            let x = Expr::from_var(&arena, VarId(0));
            let y = Expr::from_var(&arena, VarId(1));
            let small = x.square();
            let large = x.square() + y.square() + 2.0 * x + 3.0;
            let root = large.id;
            arena.reset_access_count();
            let difference = if larger_right { small - large } else { large - small };
            assert_eq!(arena.access_count(), 1);
            assert_eq!(difference.id, root);
            assert!(difference.owned);

            let expected: f64 = if larger_right { -16.0 } else { 16.0 };
            assert_eq!(
                evaluate(&arena.borrow(), root, &&[2.0, 3.0][..]).unwrap().to_bits(),
                expected.to_bits()
            );
        }
    }

    #[test]
    fn mixed_subtraction_reuses_only_the_consumed_operand() {
        for borrow_right in [false, true] {
            let arena = ExprArenaCell::new(ExprArena::new());
            let x = Expr::from_var(&arena, VarId(0));
            let y = Expr::from_var(&arena, VarId(1));
            let left = x.square();
            let right = x.square() + y.square() + 2.0 * x + 3.0;

            let (root, borrowed) =
                if borrow_right { (left.id, right.id) } else { (right.id, left.id) };
            arena.reset_access_count();
            let difference = if borrow_right { left - &right } else { &left - right };
            assert_eq!(arena.access_count(), 1);
            assert_eq!(difference.id, root);
            assert!(difference.owned);

            let snapshot = arena.borrow();
            assert_eq!(
                evaluate(&snapshot, root, &&[2.0, 3.0][..]).unwrap().to_bits(),
                (-16.0_f64).to_bits()
            );

            let expected: f64 = if borrow_right { 20.0 } else { 4.0 };
            assert_eq!(
                evaluate(&snapshot, borrowed, &&[2.0, 3.0][..]).unwrap().to_bits(),
                expected.to_bits()
            );
        }
    }

    #[test]
    fn right_subtraction_respects_permanent_sealing_after_a_clone() {
        let arena = ExprArenaCell::new(ExprArena::new());
        let x = Expr::from_var(&arena, VarId(0));
        let y = Expr::from_var(&arena, VarId(1));
        let right = x.square() + y.square();
        let alias = right.clone();
        let difference = x - right;
        assert_ne!(difference.id, alias.id);

        let snapshot = arena.borrow();
        assert_eq!(
            evaluate(&snapshot, alias.id, &&[2.0, 3.0][..]).unwrap().to_bits(),
            13.0_f64.to_bits()
        );
        assert_eq!(
            evaluate(&snapshot, difference.id, &&[2.0, 3.0][..]).unwrap().to_bits(),
            (-11.0_f64).to_bits()
        );
    }

    #[test]
    fn failed_right_accumulation_restores_coefficients_and_constant() {
        for failure in 0..3 {
            let arena = ExprArenaCell::new(ExprArena::new());
            let x = Expr::from_var(&arena, VarId(0));
            let mut builder = crate::QuadraticBuilder::new(&arena);
            let coefficient = if failure == 0 { -0.75 * f64::MAX } else { 1.0 };
            let constant = if failure == 1 { -0.75 * f64::MAX } else { 3.0 };
            builder.add_monomial(coefficient, VarId(0), VarId(0)).unwrap();
            builder.add_monomial(1.0, VarId(1), VarId(1)).unwrap();
            builder.add_constant(constant).unwrap();

            let right = builder.build();
            let root = right.id;
            let before = arena.with_ref(|arena| {
                let ExprNode::Quadratic(q) = arena.get(root) else { panic!("quadratic") };
                q.as_ref().clone()
            });
            let left = match failure {
                0 => ((0.75 * f64::MAX) * x * x).into_dynamic(),
                1 => (x.square() + 0.75 * f64::MAX).into_dynamic(),
                _ => x.sin().into_dynamic(),
            };
            let difference = left - right;
            assert_ne!(difference.id, root);

            let snapshot = arena.borrow();
            let ExprNode::Quadratic(after) = snapshot.get(root) else { panic!("quadratic") };
            assert_eq!(after.linear(), before.linear());
            assert_eq!(after.quadratic(), before.quadratic());
            assert_eq!(after.constant().to_bits(), before.constant().to_bits());

            let value = evaluate(&snapshot, difference.id, &&[0.5, 0.25][..]).unwrap();
            match failure {
                0 => assert!((value / (0.375 * f64::MAX) - 1.0).abs() < f64::EPSILON),
                1 => assert_eq!(value.to_bits(), f64::INFINITY.to_bits()),
                _ => assert_eq!(value.to_bits(), (0.5_f64.sin() - 3.3125).to_bits()),
            }
        }
    }

    #[test]
    fn right_subtraction_preserves_narrowed_affine_ownership_and_coefficients() {
        for scalar in [false, true] {
            let arena = ExprArenaCell::new(ExprArena::new());
            let x = Expr::from_var(&arena, VarId(0));
            let right = (x.square() - x.square() + 2.0 * x + 3.0).into_affine().unwrap();
            let root = right.id;
            let difference = if scalar { 5.0 - right } else { Expr::constant(&arena, 5.0) - right };
            assert_eq!(difference.id, root);
            assert!(difference.owned);

            let square = difference.square();
            let snapshot = arena.borrow();
            let linear = crate::extract_linear(&snapshot, root).unwrap();
            assert_eq!(linear.coeffs.as_ref(), [(VarId(0), -2.0)]);
            assert_eq!(linear.constant.to_bits(), 2.0_f64.to_bits());
            assert_eq!(
                evaluate(&snapshot, square.id, &&[2.0][..]).unwrap().to_bits(),
                4.0_f64.to_bits()
            );
        }
    }

    #[test]
    fn right_subtraction_reuses_worker_roots_with_base_arena_inputs() {
        let arena = ExprArenaCell::new(ExprArena::new());
        let x = Expr::from_var(&arena, VarId(0));
        let y = Expr::from_var(&arena, VarId(1));
        let mut batch = arena.__begin_batch();
        let snapshot = batch.snapshot();
        let mut forks = vec![arena.__with_fork(snapshot.clone(), || {
            let right = x.square() + y.square() + 2.0 * x + 3.0;
            let root = right.id;
            let difference = x - right;
            assert_eq!(difference.id, root);
            assert!(difference.owned);

            difference.id
        })];

        drop(snapshot);
        let remaps = batch.merge(&mut forks);
        let root = remaps[0].apply(forks[0].value);
        drop(batch);
        assert_eq!(
            evaluate(&arena.borrow(), root, &&[2.0, 3.0][..]).unwrap().to_bits(),
            (-18.0_f64).to_bits()
        );
    }

    #[test]
    fn borrowed_binary_operators_use_one_guard_and_seal_both_roots() {
        macro_rules! check {
            ($op:tt, $expected:expr, $fresh_quadratic:expr) => {{
                let arena = ExprArenaCell::new(ExprArena::new());
                let x = Expr::from_var(&arena, VarId(0));
                let y = Expr::from_var(&arena, VarId(1));
                let left = x.square();
                let right = y.square();
                let (left_id, right_id) = (left.id, right.id);
                assert!(left.owned && right.owned);
                arena.reset_access_count();
                let result = &left $op &right;
                assert_eq!(arena.access_count(), 1);
                assert_eq!(result.owned, $fresh_quadratic);
                let result_id = result.id;
                let result = result + x;
                if $fresh_quadratic { assert_eq!(result.id, result_id); }
                let left = left + x;
                let right = right + y;
                assert_ne!(left.id, left_id, "borrowing permanently seals the input");
                assert_ne!(right.id, right_id);
                let snapshot = arena.borrow();
                assert_eq!(evaluate(&snapshot, left_id, &&[2.0, 3.0][..]).unwrap(), 4.0);
                assert_eq!(evaluate(&snapshot, right_id, &&[2.0, 3.0][..]).unwrap(), 9.0);
                assert_eq!(evaluate(&snapshot, result.id, &&[2.0, 3.0][..]).unwrap(), $expected + 2.0);
            }};
        }

        check!(+, 13.0, true);
        check!(-, -5.0, true);
        check!(*, 36.0, false);
        check!(/, 4.0 / 9.0, false);
    }

    #[test]
    fn mixed_addition_keeps_the_consumed_root_eligible_in_both_orders() {
        for borrow_left in [false, true] {
            let arena = ExprArenaCell::new(ExprArena::new());
            let x = Expr::from_var(&arena, VarId(0));
            let y = Expr::from_var(&arena, VarId(1));
            let left = x.square();
            let right = y.square();
            let expected = if borrow_left { right.id } else { left.id };
            arena.reset_access_count();
            let sum = if borrow_left { &left + right } else { left + &right };
            assert_eq!(arena.access_count(), 1);
            assert_eq!(sum.id, expected);
            assert!(sum.owned);
            assert_eq!(evaluate(&arena.borrow(), sum.id, &&[2.0, 3.0][..]).unwrap(), 13.0);
        }
    }

    #[test]
    fn mixed_multiplication_reuses_the_consumed_quadratic() {
        for borrow_left in [false, true] {
            let arena = ExprArenaCell::new(ExprArena::new());
            let x = Expr::from_var(&arena, VarId(0));
            let q = x.square();
            let original = q.id;
            let scale = Expr::constant(&arena, 2.0);
            arena.reset_access_count();
            let product = if borrow_left { &scale * q } else { q * &scale };
            assert_eq!(arena.access_count(), 1);
            assert_eq!(product.id, original);
            assert!(product.owned);
            assert_eq!(evaluate(&arena.borrow(), product.id, &&[3.0][..]).unwrap(), 18.0);
        }
    }

    #[test]
    fn owned_scalar_fast_paths_do_not_allocate_constant_nodes() {
        let arena = ExprArenaCell::new(ExprArena::new());
        let x = Expr::from_var(&arena, VarId(0));
        let q = x.square();
        let root = q.id;
        let nodes = arena.__node_count();
        arena.reset_access_count();
        let q = q * 2.0;
        assert_eq!(arena.access_count(), 1);
        assert_eq!(q.id, root);
        assert_eq!(arena.__node_count(), nodes);

        arena.reset_access_count();
        let q = -q;
        assert_eq!(arena.access_count(), 1);
        assert_eq!(q.id, root);
        assert_eq!(arena.__node_count(), nodes);
        assert_eq!(evaluate(&arena.borrow(), root, &&[3.0][..]).unwrap(), -18.0);
    }

    #[test]
    fn addition_preserves_owned_roots_after_affine_narrowing() {
        let arena = ExprArenaCell::new(ExprArena::new());
        let x = Expr::from_var(&arena, VarId(0));
        let q = x.square() - x.square();
        let q = q.into_affine().unwrap();
        assert!(q.owned);

        let root = q.id;
        arena.reset_access_count();
        let result = q + x;
        assert_eq!(arena.access_count(), 1);
        assert_eq!(result.id, root);
        assert!(result.owned);
        assert_eq!(evaluate(&arena.borrow(), root, &&[3.0][..]).unwrap(), 3.0);
    }

    #[test]
    fn borrowed_unary_and_scalar_operators_use_one_guard() {
        macro_rules! check {
            ($q:ident, $expression:expr, $expected:expr) => {{
                let arena = ExprArenaCell::new(ExprArena::new());
                let x = Expr::from_var(&arena, VarId(0));
                let $q = x.square();
                let original = $q.id;
                arena.reset_access_count();
                let result = $expression;
                assert_eq!(arena.access_count(), 1);
                let after_borrow = $q + x;
                assert_ne!(after_borrow.id, original);
                let snapshot = arena.borrow();
                assert_eq!(evaluate(&snapshot, original, &&[2.0][..]).unwrap(), 4.0);
                assert_eq!(evaluate(&snapshot, result.id, &&[2.0][..]).unwrap(), $expected);
            }};
        }

        check!(q, -&q, -4.0);
        check!(q, &q + 2.0, 6.0);
        check!(q, &q - 2.0, 2.0);
        check!(q, &q * 2.0, 8.0);
        check!(q, &q / 2.0, 2.0);
        check!(q, 2.0 + &q, 6.0);
        check!(q, 2.0 - &q, -2.0);
        check!(q, 2.0 * &q, 8.0);
        check!(q, 2.0 / &q, 0.5);
        check!(q, &q + 2, 6.0);
        check!(q, &q - 2, 2.0);
        check!(q, &q * 2, 8.0);
        check!(q, &q / 2, 2.0);
        check!(q, 2 + &q, 6.0);
        check!(q, 2 - &q, -2.0);
        check!(q, 2 * &q, 8.0);
        check!(q, 2 / &q, 0.5);
    }

    #[test]
    fn borrowed_operations_route_through_worker_forks() {
        let arena = ExprArenaCell::new(ExprArena::new());
        let x = Expr::from_var(&arena, VarId(0));
        let mut batch = arena.__begin_batch();
        let snapshot = batch.snapshot();
        let mut forks = vec![arena.__with_fork(snapshot.clone(), || {
            let left = x.square();
            let right = x.square();
            arena.reset_access_count();
            let sum = &left + &right;
            assert_eq!(arena.access_count(), 1);
            let root = sum.id;
            let sum = sum + x;
            assert_eq!(sum.id, root);
            sum.id
        })];

        drop(snapshot);
        let remaps = batch.merge(&mut forks);
        let root = remaps[0].apply(forks[0].value);
        drop(batch);
        assert_eq!(evaluate(&arena.borrow(), root, &&[3.0][..]).unwrap(), 21.0);
    }
}
