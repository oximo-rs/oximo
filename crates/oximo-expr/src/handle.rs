use crate::arena::{Children, ExprArenaCell, ExprId, ExprNode, ModelId, ParamId, UnaryOp, VarId};
use crate::classify::{ExprClass, classify_access};
use crate::degree::{Affine, Constant, Degree, Dynamic, MulDegree, Nonlinear, Quadratic};

/// Read-only identity exposed by an expression handle.
#[derive(Copy, Clone, Debug)]
pub struct ExprData<'a> {
    pub id: ExprId,
    model_id: ModelId,
    pub arena: &'a ExprArenaCell,
}

/// Copyable arena handle with a conservative static degree.
/// Use [`Self::erase`] to put different degrees in one collection.
#[derive(Copy, Clone)]
pub struct Expr<'a, D: Degree = Dynamic> {
    raw: ExprData<'a>,
    degree: std::marker::PhantomData<D>,
}
impl<'a, D: Degree> std::ops::Deref for Expr<'a, D> {
    type Target = ExprData<'a>;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.raw
    }
}
impl<D: Degree> std::fmt::Debug for Expr<'_, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.raw.fmt(f)
    }
}

/// An expression classified once into a checked static handle.
#[derive(Copy, Clone, Debug)]
pub enum ClassifiedExpr<'a> {
    Affine(Expr<'a, Affine>),
    Quadratic(Expr<'a, Quadratic>),
    Nonlinear(Expr<'a, Nonlinear>),
}
impl<'a> Expr<'a> {
    /// Construct a dynamic handle from a raw arena ID.
    pub fn new(id: ExprId, arena: &'a ExprArenaCell) -> Self {
        Self::from_id(id, arena)
    }

    pub fn constant(arena: &'a ExprArenaCell, value: f64) -> Expr<'a, Constant> {
        Expr::from_id(arena.with_mut(|a| a.constant(value)), arena)
    }

    pub fn from_var(arena: &'a ExprArenaCell, var: VarId) -> Expr<'a, Affine> {
        Expr::from_id(arena.with_mut(|a| a.var(var)), arena)
    }

    /// Make a symbolic, variable-independent parameter handle.
    pub fn from_param(arena: &'a ExprArenaCell, param: ParamId) -> Expr<'a, Constant> {
        Expr::from_id(arena.with_mut(|a| a.push(ExprNode::Param(param))), arena)
    }
}
impl<'a, D: Degree> Expr<'a, D> {
    #[inline]
    pub(crate) fn from_id(id: ExprId, arena: &'a ExprArenaCell) -> Self {
        Self {
            raw: ExprData { id, model_id: arena.model_id(), arena },
            degree: std::marker::PhantomData,
        }
    }

    /// Classify a dynamic expression without copying nodes or extracting coefficients.
    pub fn classified(self) -> ClassifiedExpr<'a> {
        match self.__class() {
            ExprClass::Linear => ClassifiedExpr::Affine(Expr::from_id(self.id, self.arena)),
            ExprClass::Quadratic => ClassifiedExpr::Quadratic(Expr::from_id(self.id, self.arena)),
            ExprClass::Nonlinear => ClassifiedExpr::Nonlinear(Expr::from_id(self.id, self.arena)),
        }
    }

    #[inline]
    pub fn erase(self) -> Expr<'a> {
        Expr::from_id(self.id, self.arena)
    }

    pub fn id(self) -> ExprId {
        self.id
    }

    pub fn arena(self) -> &'a ExprArenaCell {
        self.arena
    }

    /// Compact affine arithmetic once before repeated evaluation or extraction.
    /// Numeric input becomes a `Linear` node or a constant, while parameter-dependent
    /// terms remain symbolic and observe later rebinding.
    ///
    /// Like [`crate::AffineBuilder`], compaction can regroup floating-point
    /// coefficients. Call this after construction.
    pub fn compact(self) -> Expr<'a, Affine>
    where
        D: crate::degree::AddDegree<Affine, Output = Affine>,
    {
        if self.arena.with_ref(|arena| {
            matches!(
                arena.get(self.id),
                ExprNode::Const(_) | ExprNode::Var(_) | ExprNode::Linear { .. }
            )
        }) {
            return Expr::from_id(self.id, self.arena);
        }
        let mut builder = crate::AffineBuilder::new(self.arena);
        builder.add_term(1.0, self);
        builder.build()
    }

    /// Square with a statically determined degree.
    pub fn square(self) -> Expr<'a, <D as MulDegree<D>>::Output>
    where
        D: MulDegree<D>,
    {
        self * self
    }

    /// Check the arena expression before narrowing its static degree.
    ///
    /// # Errors
    /// Returns the actual expression class if it is not affine.
    pub fn try_affine(self) -> Result<Expr<'a, Affine>, ExprClass> {
        let class = self.__class();
        if class == ExprClass::Linear { Ok(Expr::from_id(self.id, self.arena)) } else { Err(class) }
    }

    /// Check that the expression has polynomial degree at most two.
    ///
    /// # Errors
    /// Returns the actual expression class for a nonlinear expression.
    pub fn try_quadratic(self) -> Result<Expr<'a, Quadratic>, ExprClass> {
        let class = self.__class();
        if class <= ExprClass::Quadratic {
            Ok(Expr::from_id(self.id, self.arena))
        } else {
            Err(class)
        }
    }

    pub fn nonlinear(self) -> Expr<'a, Nonlinear> {
        Expr::from_id(self.id, self.arena)
    }

    /// Identity of the model/expression arena that created this handle.
    #[inline]
    #[must_use]
    pub fn model_id(self) -> ModelId {
        self.model_id
    }

    #[inline]
    pub(crate) fn assert_same_arena<E: Degree>(self, other: Expr<'a, E>) {
        assert!(std::ptr::eq(self.arena, other.arena), "expressions belong to different arenas");
    }

    /// If this handle is a bare variable, return its [`VarId`].
    /// `None` for compound expressions (sums, products, constants, ...).
    pub fn var_id(self) -> Option<VarId> {
        self.arena.with_ref(|arena| match arena.get(self.id) {
            ExprNode::Var(id) => Some(*id),
            _ => None,
        })
    }

    /// If this handle is a bare parameter, return its [`ParamId`].
    /// `None` for compound expressions.
    pub fn param_id(self) -> Option<ParamId> {
        self.arena.with_ref(|arena| match arena.get(self.id) {
            ExprNode::Param(id) => Some(*id),
            _ => None,
        })
    }

    /// Re-bind the parameter this handle references to `value`. Takes effect on
    /// the next extraction/evaluation, which read the value straight from the
    /// arena.
    ///
    /// # Panics
    /// Panics if this handle is not a bare parameter (see [`Self::param_id`]).
    pub fn set_param_value(self, value: f64) {
        let id = self.param_id().expect("set_param_value expects a bare parameter handle");
        self.arena.borrow_mut().set_param_value(id, value);
    }

    pub fn pow<E: Degree>(self, exponent: Expr<'a, E>) -> Expr<'a> {
        self.assert_same_arena(exponent);
        let id = self.arena.with_mut(|arena| crate::linear::pow_into(arena, self.id, exponent.id));
        Expr::from_id(id, self.arena)
    }

    pub fn powi(self, n: i32) -> Expr<'a> {
        let id = self.arena.with_mut(|arena| {
            let exp_id = arena.constant(f64::from(n));
            crate::linear::pow_into(arena, self.id, exp_id)
        });
        Expr::from_id(id, self.arena)
    }

    pub fn powf(self, n: f64) -> Expr<'a> {
        let id = self.arena.with_mut(|arena| {
            let exp_id = arena.constant(n);
            crate::linear::pow_into(arena, self.id, exp_id)
        });
        Expr::from_id(id, self.arena)
    }

    fn unary(self, op: UnaryOp) -> Expr<'a, Nonlinear> {
        let id = self.arena.with_mut(|arena| arena.push(ExprNode::Unary(op, self.id)));
        Expr::from_id(id, self.arena)
    }

    /// Unary negation as an explicit expression node.
    /// The `-expr` operator keeps its affine fast path.
    /// Use this method when the public [`UnaryOp`] node is required.
    #[expect(
        clippy::should_implement_trait,
        reason = "explicit node constructor complements Neg::neg"
    )]
    pub fn neg(self) -> Self {
        let id = self.arena.with_mut(|a| a.push(ExprNode::Unary(UnaryOp::Neg, self.id)));
        Expr::from_id(id, self.arena)
    }

    pub fn abs(self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Abs)
    }

    pub fn sqrt(self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Sqrt)
    }

    pub fn cbrt(self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Cbrt)
    }

    pub fn exp(self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Exp)
    }

    pub fn exp2(self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Exp2)
    }

    pub fn expm1(self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Expm1)
    }

    /// Alias for [`Expr::expm1`], matching Rust's `f64::exp_m1` spelling.
    pub fn exp_m1(self) -> Expr<'a, Nonlinear> {
        self.expm1()
    }

    pub fn log(self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Log)
    }

    /// Alias for [`Expr::log`], matching Rust's `f64::ln` spelling.
    pub fn ln(self) -> Expr<'a, Nonlinear> {
        self.log()
    }

    pub fn log2(self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Log2)
    }

    pub fn log10(self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Log10)
    }

    pub fn log1p(self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Log1p)
    }

    /// Alias for [`Expr::log1p`], matching Rust's `f64::ln_1p` spelling.
    pub fn ln_1p(self) -> Expr<'a, Nonlinear> {
        self.log1p()
    }

    pub fn sin(self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Sin)
    }

    pub fn cos(self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Cos)
    }

    pub fn tan(self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Tan)
    }

    pub fn asin(self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Asin)
    }

    pub fn acos(self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Acos)
    }

    pub fn atan(self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Atan)
    }

    pub fn sinh(self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Sinh)
    }

    pub fn cosh(self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Cosh)
    }

    pub fn tanh(self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Tanh)
    }

    pub fn asinh(self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Asinh)
    }

    pub fn acosh(self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Acosh)
    }

    pub fn atanh(self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Atanh)
    }

    /// Two-argument arctangent in Rust's `y.atan2(x)` argument order.
    pub fn atan2<E: Degree>(self, x: Expr<'a, E>) -> Expr<'a, Nonlinear> {
        self.assert_same_arena(x);
        let id = self.arena.with_mut(|arena| arena.push(ExprNode::Atan2(self.id, x.id)));
        Expr::from_id(id, self.arena)
    }

    fn extrema<E: Degree>(self, other: Expr<'a, E>, is_min: bool) -> Expr<'a, Nonlinear> {
        self.assert_same_arena(other);
        let id = self.arena.with_mut(|arena| {
            let mut children = Children::new();
            let left = match arena.get(self.id) {
                ExprNode::Min(existing) if is_min => Some(existing.as_slice()),
                ExprNode::Max(existing) if !is_min => Some(existing.as_slice()),
                _ => None,
            };
            if let Some(existing) = left {
                children.extend_from_slice(existing);
            } else {
                children.push(self.id);
            }
            let right = match arena.get(other.id) {
                ExprNode::Min(existing) if is_min => Some(existing.as_slice()),
                ExprNode::Max(existing) if !is_min => Some(existing.as_slice()),
                _ => None,
            };
            if let Some(existing) = right {
                children.extend_from_slice(existing);
            } else {
                children.push(other.id);
            }
            arena.push(if is_min { ExprNode::Min(children) } else { ExprNode::Max(children) })
        });
        Expr::from_id(id, self.arena)
    }

    /// Pairwise minimum, flattening nested minima into deterministic n-ary nodes.
    pub fn min<E: Degree>(self, other: Expr<'a, E>) -> Expr<'a, Nonlinear> {
        self.extrema(other, true)
    }

    /// Pairwise maximum, flattening nested maxima into deterministic n-ary nodes.
    pub fn max<E: Degree>(self, other: Expr<'a, E>) -> Expr<'a, Nonlinear> {
        self.extrema(other, false)
    }

    #[doc(hidden)]
    pub fn __class(self) -> ExprClass {
        self.arena.with_ref(|arena| classify_access(arena, self.id))
    }
}

impl<'a> From<Expr<'a, Constant>> for Expr<'a> {
    fn from(e: Expr<'a, Constant>) -> Self {
        e.erase()
    }
}

impl<'a> From<Expr<'a, Affine>> for Expr<'a> {
    fn from(e: Expr<'a, Affine>) -> Self {
        e.erase()
    }
}

impl<'a> From<Expr<'a, Quadratic>> for Expr<'a> {
    fn from(e: Expr<'a, Quadratic>) -> Self {
        e.erase()
    }
}

impl<'a> From<Expr<'a, Nonlinear>> for Expr<'a> {
    fn from(e: Expr<'a, Nonlinear>) -> Self {
        e.erase()
    }
}

impl<'a> From<Expr<'a, Constant>> for Expr<'a, Affine> {
    fn from(e: Expr<'a, Constant>) -> Self {
        Expr::from_id(e.id, e.arena)
    }
}

impl<'a> From<Expr<'a, Affine>> for Expr<'a, Quadratic> {
    fn from(e: Expr<'a, Affine>) -> Self {
        Expr::from_id(e.id, e.arena)
    }
}
impl<'a> From<Expr<'a, Constant>> for Expr<'a, Quadratic> {
    fn from(e: Expr<'a, Constant>) -> Self {
        Expr::from_id(e.id, e.arena)
    }
}
#[cfg(test)]
mod tests {
    use super::Expr;
    use crate::arena::{ExprArena, ExprArenaCell};

    #[test]
    fn set_param_value_rebinds_through_handle() {
        let arena = ExprArenaCell::new(ExprArena::new());
        let pid = arena.borrow_mut().new_param(0.05);
        let node = arena.borrow_mut().param(pid);
        let p = Expr::new(node, &arena);

        p.set_param_value(0.2);
        assert!((arena.borrow().param_value(pid) - 0.2).abs() < f64::EPSILON);
    }

    #[test]
    #[should_panic(expected = "bare parameter handle")]
    fn set_param_value_panics_on_non_param() {
        let arena = ExprArenaCell::new(ExprArena::new());
        let c = Expr::constant(&arena, 1.0);
        c.set_param_value(3.0);
    }

    #[test]
    #[should_panic(expected = "different arenas")]
    fn combining_different_arenas_is_rejected() {
        let left_arena = ExprArenaCell::new(ExprArena::new());
        let right_arena = ExprArenaCell::new(ExprArena::new());
        let left = Expr::constant(&left_arena, 1.0);
        let right = Expr::constant(&right_arena, 2.0);
        let _ = left + right;
    }
}
