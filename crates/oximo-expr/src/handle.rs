use crate::arena::{Children, ExprArenaCell, ExprId, ExprNode, ModelId, ParamId, UnaryOp, VarId};
use crate::classify::{ExprClass, classify_access};
use crate::degree::{
    Affine, Constant, Degree, Dynamic, MulDegree, Nonlinear, ParameterDegree, Quadratic,
    VariableDegree,
};

/// An arena expression.
pub struct Expr<'a, D: Degree = Dynamic> {
    pub(crate) id: ExprId,
    pub(crate) arena: &'a ExprArenaCell,
    model_id: ModelId,
    pub(crate) owned: bool,
    degree: std::marker::PhantomData<D>,
}

/// Reusable identity of one decision variable.
pub type VariableHandle<'a> = Expr<'a, VariableDegree>;
/// Reusable identity of one parameter, retaining its live parameter binding.
pub type ParameterHandle<'a> = Expr<'a, ParameterDegree>;

// Only immutable leaf identities can be implicitly copied. Computed expressions
// retain consuming ownership, including affine and constant expressions.
impl<D: crate::degree::LeafDegree> Copy for Expr<'_, D> {}

#[expect(
    clippy::expl_impl_clone_on_copy,
    reason = "the shared Clone implementation also seals non-Copy computed expressions"
)]
impl<D: Degree> Clone for Expr<'_, D> {
    fn clone(&self) -> Self {
        self.seal();
        Self {
            id: self.id,
            arena: self.arena,
            model_id: self.model_id,
            owned: false,
            degree: std::marker::PhantomData,
        }
    }
}
impl<D: Degree> std::fmt::Debug for Expr<'_, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Expr").field("id", &self.id).field("model_id", &self.model_id).finish()
    }
}

/// An expression classified once into a checked static handle.
#[derive(Clone, Debug)]
pub enum ClassifiedExpr<'a> {
    Affine(Expr<'a, Affine>),
    Quadratic(Expr<'a, Quadratic>),
    Nonlinear(Expr<'a, Nonlinear>),
}
impl<'a> Expr<'a> {
    /// Construct a dynamic handle from a raw arena ID.
    pub fn new(id: ExprId, arena: &'a ExprArenaCell) -> Self {
        arena.with_ref(|a| a.seal(id));
        Self::from_id(id, arena)
    }

    #[inline]
    pub fn constant(arena: &'a ExprArenaCell, value: f64) -> Expr<'a, Constant> {
        Expr::from_id(arena.with_mut(|a| a.constant(value)), arena)
    }

    #[inline]
    pub fn from_var(arena: &'a ExprArenaCell, var: VarId) -> VariableHandle<'a> {
        Expr::from_id(arena.with_mut(|a| a.var(var)), arena)
    }

    /// Make a symbolic, variable-independent parameter handle.
    #[inline]
    pub fn from_param(arena: &'a ExprArenaCell, param: ParamId) -> ParameterHandle<'a> {
        Expr::from_id(arena.with_mut(|a| a.push(ExprNode::Param(param))), arena)
    }
}
impl<'a, D: Degree> Expr<'a, D> {
    /// Capture an immutable, owned copy of this expression's reachable graph.
    /// Parameter values are supplied separately when evaluating the capture.
    #[must_use]
    pub fn capture(&self) -> crate::CapturedExpr {
        crate::CapturedExpr::new(self.model_id(), &self.arena().borrow(), self.id())
    }

    #[inline]
    pub(crate) fn from_id(id: ExprId, arena: &'a ExprArenaCell) -> Self {
        Self {
            id,
            arena,
            model_id: arena.model_id(),
            owned: false,
            degree: std::marker::PhantomData,
        }
    }

    /// Record ownership while the construction guard is already held.
    /// Immutable node kinds never need a later arena access when cloned or erased.
    pub(crate) fn build(
        arena: &'a ExprArenaCell,
        construct: impl FnOnce(&mut dyn crate::arena::ArenaAccess) -> ExprId,
    ) -> Self {
        let (id, owned) = arena.build_expr(crate::degree::may_be_quadratic::<D>(), construct);
        Self::from_construction(id, owned, arena)
    }

    pub(crate) fn from_construction(id: ExprId, owned: bool, arena: &'a ExprArenaCell) -> Self {
        let mut result = Self::from_id(id, arena);
        result.owned = owned;
        result
    }

    pub(crate) fn into_degree<E: Degree>(self) -> Expr<'a, E> {
        Expr::from_construction(self.id, self.owned, self.arena)
    }

    /// Consume the handle without publishing an alias or losing unique ownership.
    pub fn into_dynamic(self) -> Expr<'a> {
        self.into_degree()
    }

    /// Consume the handle while widening its static degree.
    pub fn into_nonlinear(self) -> Expr<'a, Nonlinear> {
        self.into_degree()
    }

    /// Check the runtime degree and consume the handle, retaining ownership.
    ///
    /// # Errors
    /// Returns the actual class if it is not affine.
    pub fn into_affine(self) -> Result<Expr<'a, Affine>, ExprClass> {
        let class = self.__class();
        if class == ExprClass::Linear { Ok(self.into_degree()) } else { Err(class) }
    }

    /// Check the runtime degree and consume the handle, retaining ownership.
    ///
    /// # Errors
    /// Returns the actual class if its degree exceeds two.
    pub fn into_quadratic(self) -> Result<Expr<'a, Quadratic>, ExprClass> {
        let class = self.__class();
        if class <= ExprClass::Quadratic { Ok(self.into_degree()) } else { Err(class) }
    }

    /// Classify and consume the handle, retaining unique ownership.
    pub fn into_classified(self) -> ClassifiedExpr<'a> {
        match self.__class() {
            ExprClass::Linear => ClassifiedExpr::Affine(self.into_degree()),
            ExprClass::Quadratic => ClassifiedExpr::Quadratic(self.into_degree()),
            ExprClass::Nonlinear => ClassifiedExpr::Nonlinear(self.into_degree()),
        }
    }

    pub(crate) fn seal(&self) {
        if self.owned {
            self.arena.with_ref(|a| a.seal(self.id));
        }
    }

    /// Classify a dynamic expression without copying nodes or extracting coefficients.
    pub fn classified(&self) -> ClassifiedExpr<'a> {
        self.seal();
        match self.__class() {
            ExprClass::Linear => ClassifiedExpr::Affine(Expr::from_id(self.id, self.arena)),
            ExprClass::Quadratic => ClassifiedExpr::Quadratic(Expr::from_id(self.id, self.arena)),
            ExprClass::Nonlinear => ClassifiedExpr::Nonlinear(Expr::from_id(self.id, self.arena)),
        }
    }

    #[inline]
    /// Borrowed dynamic view that seals unique roots.
    /// Use [`Self::into_dynamic`] when consuming a handle.
    pub fn erase(&self) -> Expr<'a> {
        self.seal();
        Expr::from_id(self.id, self.arena)
    }

    /// Borrowed dynamic view. Sharing permanently seals an owned quadratic.
    pub fn as_dynamic(&self) -> Expr<'a> {
        self.erase()
    }

    /// Borrowed nonlinear view. Use [`Self::into_nonlinear`] to consume a handle.
    pub fn as_nonlinear(&self) -> Expr<'a, Nonlinear> {
        self.nonlinear()
    }

    pub fn id(&self) -> ExprId {
        self.seal();
        self.id
    }

    pub fn arena(&self) -> &'a ExprArenaCell {
        self.arena
    }

    /// Compact affine arithmetic once before repeated evaluation or extraction.
    /// Numeric input becomes a `Linear` node or a constant, while parameter-dependent
    /// terms remain symbolic and observe later rebinding.
    ///
    /// Like [`crate::AffineBuilder`], compaction can regroup floating-point
    /// coefficients. Call this after construction.
    pub fn compact(&self) -> Expr<'a, Affine>
    where
        D: crate::degree::AddDegree<Affine, Output = Affine>,
    {
        self.seal();
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
    pub fn square(&self) -> Expr<'a, <D as MulDegree<D>>::Output>
    where
        D: MulDegree<D>,
    {
        self * self
    }

    /// Preserve the multiplication for evaluation instead of eagerly expanding
    /// numeric affine factors.
    /// Prefer this when coefficient cancellation would change a nonlinear domain.
    pub fn mul_factored<E: ExprInput<'a>>(
        &self,
        rhs: E,
    ) -> Expr<'a, <D as MulDegree<E::Degree>>::Output>
    where
        D: MulDegree<E::Degree>,
    {
        let rhs = rhs.as_expression();
        self.assert_same_arena(rhs);
        let id =
            self.arena.with_mut(|a| a.push(ExprNode::Mul([self.id, rhs.id].into_iter().collect())));
        Expr::from_id(id, self.arena)
    }

    /// Square without expanding the factors, preserving their evaluation order.
    pub fn square_factored(&self) -> Expr<'a, <D as MulDegree<D>>::Output>
    where
        D: MulDegree<D>,
    {
        self.mul_factored(self)
    }

    /// Check the arena expression before narrowing its static degree.
    ///
    /// # Errors
    /// Returns the actual expression class if it is not affine.
    pub fn try_affine(&self) -> Result<Expr<'a, Affine>, ExprClass> {
        self.seal();
        let class = self.__class();
        if class == ExprClass::Linear { Ok(Expr::from_id(self.id, self.arena)) } else { Err(class) }
    }

    /// Check that the expression has polynomial degree at most two.
    ///
    /// # Errors
    /// Returns the actual expression class for a nonlinear expression.
    pub fn try_quadratic(&self) -> Result<Expr<'a, Quadratic>, ExprClass> {
        self.seal();
        let class = self.__class();
        if class <= ExprClass::Quadratic {
            Ok(Expr::from_id(self.id, self.arena))
        } else {
            Err(class)
        }
    }

    /// Borrowed nonlinear view. Use [`Self::into_nonlinear`] to retain ownership.
    pub fn nonlinear(&self) -> Expr<'a, Nonlinear> {
        self.seal();
        Expr::from_id(self.id, self.arena)
    }

    /// Identity of the model/expression arena that created this handle.
    #[inline]
    #[must_use]
    pub fn model_id(&self) -> ModelId {
        self.model_id
    }

    #[inline]
    pub(crate) fn assert_same_arena<E: Degree>(&self, other: &Expr<'a, E>) {
        assert!(std::ptr::eq(self.arena, other.arena), "expressions belong to different arenas");
    }

    /// If this handle is a bare variable, return its [`VarId`].
    /// `None` for compound expressions (sums, products, constants, ...).
    pub fn var_id(&self) -> Option<VarId> {
        self.arena.with_ref(|arena| match arena.get(self.id) {
            ExprNode::Var(id) => Some(*id),
            _ => None,
        })
    }

    /// If this handle is a bare parameter, return its [`ParamId`].
    /// `None` for compound expressions.
    pub fn param_id(&self) -> Option<ParamId> {
        self.arena.with_ref(|arena| match arena.get(self.id) {
            ExprNode::Param(id) => Some(*id),
            _ => None,
        })
    }

    /// Re-bind the parameter this handle references to `value`. Takes effect on
    /// the next extraction/evaluation, which read the value straight from the
    /// arena.
    ///
    /// # Errors
    /// Rejects composed handles, non-finite values, unknown or locked parameters,
    /// and assignments that invert a dependent variable's bounds.
    pub fn set_param_value(&self, value: f64) -> Result<(), crate::ModelError> {
        let id = self.param_id().ok_or(crate::ModelError::InvalidHandle("parameter"))?;
        self.arena.borrow_mut().set_param_value(id, value)
    }

    pub fn pow<E: ExprInput<'a>>(&self, exponent: E) -> Expr<'a> {
        let exponent = exponent.as_expression();
        self.assert_same_arena(exponent);
        self.seal();
        exponent.seal();
        Expr::build(self.arena, |arena| crate::linear::pow_into(arena, self.id, exponent.id))
    }

    pub fn powi(&self, n: i32) -> Expr<'a> {
        self.powf(f64::from(n))
    }

    #[expect(
        clippy::float_cmp,
        reason = "quadratic specialization requires an exact exponent of two"
    )]
    pub fn powf(&self, n: f64) -> Expr<'a> {
        self.seal();
        Expr::build(self.arena, |arena| {
            if n == 2.0
                && let Some(id) = crate::quadratic_builder::product_small(arena, self.id, self.id)
            {
                return id;
            }
            let exponent = arena.constant(n);
            crate::linear::pow_into(arena, self.id, exponent)
        })
    }

    fn unary(&self, op: UnaryOp) -> Expr<'a, Nonlinear> {
        let id = self.arena.with_mut(|arena| {
            if self.owned {
                arena.seal(self.id);
            }
            arena.push_immutable(ExprNode::Unary(op, self.id))
        });
        Expr::from_id(id, self.arena)
    }

    /// Unary negation as an explicit expression node.
    /// The `-expr` operator keeps its affine fast path.
    /// Use this method when the public [`UnaryOp`] node is required.
    pub fn neg(&self) -> Expr<'a, D::Computed> {
        let id = self.arena.with_mut(|a| a.push(ExprNode::Unary(UnaryOp::Neg, self.id)));
        Expr::from_id(id, self.arena)
    }

    pub fn abs(&self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Abs)
    }

    pub fn sqrt(&self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Sqrt)
    }

    pub fn cbrt(&self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Cbrt)
    }

    pub fn exp(&self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Exp)
    }

    pub fn exp2(&self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Exp2)
    }

    pub fn expm1(&self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Expm1)
    }

    /// Alias for [`Expr::expm1`], matching Rust's `f64::exp_m1` spelling.
    pub fn exp_m1(&self) -> Expr<'a, Nonlinear> {
        self.expm1()
    }

    pub fn log(&self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Log)
    }

    /// Alias for [`Expr::log`], matching Rust's `f64::ln` spelling.
    pub fn ln(&self) -> Expr<'a, Nonlinear> {
        self.log()
    }

    pub fn log2(&self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Log2)
    }

    pub fn log10(&self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Log10)
    }

    pub fn log1p(&self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Log1p)
    }

    /// Alias for [`Expr::log1p`], matching Rust's `f64::ln_1p` spelling.
    pub fn ln_1p(&self) -> Expr<'a, Nonlinear> {
        self.log1p()
    }

    pub fn sin(&self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Sin)
    }

    pub fn cos(&self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Cos)
    }

    pub fn tan(&self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Tan)
    }

    pub fn asin(&self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Asin)
    }

    pub fn acos(&self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Acos)
    }

    pub fn atan(&self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Atan)
    }

    pub fn sinh(&self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Sinh)
    }

    pub fn cosh(&self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Cosh)
    }

    pub fn tanh(&self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Tanh)
    }

    pub fn asinh(&self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Asinh)
    }

    pub fn acosh(&self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Acosh)
    }

    pub fn atanh(&self) -> Expr<'a, Nonlinear> {
        self.unary(UnaryOp::Atanh)
    }

    /// Two-argument arctangent in Rust's `y.atan2(x)` argument order.
    pub fn atan2<E: ExprInput<'a>>(&self, x: E) -> Expr<'a, Nonlinear> {
        let x = x.as_expression();
        self.assert_same_arena(x);
        let id = self.arena.with_mut(|arena| arena.push(ExprNode::Atan2(self.id, x.id)));
        Expr::from_id(id, self.arena)
    }

    fn extrema<E: ExprInput<'a>>(
        &self,
        other: E,
        kind: crate::ops::Extremum,
    ) -> Expr<'a, Nonlinear> {
        use crate::ops::matching_children;

        let other = other.as_expression();
        self.assert_same_arena(other);
        let id = self.arena.with_mut(|arena| {
            let mut children = Children::new();
            if let Some(existing) = matching_children(arena.get(self.id), kind) {
                children.extend_from_slice(existing);
            } else {
                children.push(self.id);
            }
            if let Some(existing) = matching_children(arena.get(other.id), kind) {
                children.extend_from_slice(existing);
            } else {
                children.push(other.id);
            }
            arena.push(kind.node(children))
        });
        Expr::from_id(id, self.arena)
    }

    /// Pairwise minimum, flattening nested minima into deterministic n-ary nodes.
    ///
    /// # Panics
    ///
    /// Panics if the operands belong to different expression arenas.
    pub fn min<E: ExprInput<'a>>(&self, other: E) -> Expr<'a, Nonlinear> {
        self.extrema(other, crate::ops::Extremum::Min)
    }

    /// Pairwise maximum, flattening nested maxima into deterministic n-ary nodes.
    ///
    /// # Panics
    ///
    /// Panics if the operands belong to different expression arenas.
    pub fn max<E: ExprInput<'a>>(&self, other: E) -> Expr<'a, Nonlinear> {
        self.extrema(other, crate::ops::Extremum::Max)
    }

    #[doc(hidden)]
    pub fn __class(&self) -> ExprClass {
        self.arena.with_ref(|arena| classify_access(arena, self.id))
    }
}

impl<'a> From<Expr<'a, Constant>> for Expr<'a> {
    fn from(e: Expr<'a, Constant>) -> Self {
        e.into_dynamic()
    }
}

impl<'a> From<Expr<'a, Affine>> for Expr<'a> {
    fn from(e: Expr<'a, Affine>) -> Self {
        e.into_dynamic()
    }
}

impl<'a> From<Expr<'a, Quadratic>> for Expr<'a> {
    fn from(e: Expr<'a, Quadratic>) -> Self {
        e.into_dynamic()
    }
}

impl<'a> From<Expr<'a, Nonlinear>> for Expr<'a> {
    fn from(e: Expr<'a, Nonlinear>) -> Self {
        e.into_dynamic()
    }
}

impl<'a> From<Expr<'a, Constant>> for Expr<'a, Affine> {
    fn from(e: Expr<'a, Constant>) -> Self {
        e.into_degree()
    }
}

impl<'a> From<Expr<'a, Affine>> for Expr<'a, Quadratic> {
    fn from(e: Expr<'a, Affine>) -> Self {
        e.into_degree()
    }
}
impl<'a> From<Expr<'a, Constant>> for Expr<'a, Quadratic> {
    fn from(e: Expr<'a, Constant>) -> Self {
        e.into_degree()
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

        p.set_param_value(0.2).unwrap();
        assert!((arena.borrow().param_value(pid) - 0.2).abs() < f64::EPSILON);
    }

    #[test]
    fn set_param_value_panics_on_non_param() {
        let arena = ExprArenaCell::new(ExprArena::new());
        let c = Expr::constant(&arena, 1.0);
        assert!(c.set_param_value(3.0).is_err());
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

// Reusable borrowed variable/parameter handles cross dynamic APIs.
impl<'a, D: Degree> From<&Expr<'a, D>> for Expr<'a> {
    fn from(value: &Expr<'a, D>) -> Self {
        value.erase()
    }
}

/// Borrow an expression without copying its arena handle.
pub trait ExprInput<'a> {
    type Degree: Degree;
    fn as_expression(&self) -> &Expr<'a, Self::Degree>;
    /// Consume an owned handle, or publish an alias for a borrowed input.
    fn into_expression(self) -> Expr<'a, Self::Degree>
    where
        Self: Sized,
    {
        self.as_expression().clone()
    }
}

impl<'a, D: Degree> ExprInput<'a> for Expr<'a, D> {
    type Degree = D;
    fn into_expression(self) -> Expr<'a, D> {
        self
    }
    fn as_expression(&self) -> &Expr<'a, D> {
        self
    }
}
impl<'a, T: ExprInput<'a> + ?Sized> ExprInput<'a> for &T {
    type Degree = T::Degree;
    fn as_expression(&self) -> &Expr<'a, Self::Degree> {
        (*self).as_expression()
    }
}
macro_rules! borrowed_static_conversion {
    ($($d:ty),*) => { $(impl<'a> From<&Expr<'a,$d>> for Expr<'a,$d> { fn from(e:&Expr<'a,$d>)->Self { e.clone() } })* };
}
borrowed_static_conversion!(Constant, Affine, Quadratic, Nonlinear);

macro_rules! leaf_conversion {
    ($leaf:ty => $($degree:ty),+ $(,)?) => { $(
        impl<'a> From<Expr<'a, $leaf>> for Expr<'a, $degree> {
            fn from(value: Expr<'a, $leaf>) -> Self { value.into_degree() }
        }
    )+ };
}
leaf_conversion!(VariableDegree => Affine, Quadratic, Nonlinear, Dynamic);
leaf_conversion!(ParameterDegree => Constant, Affine, Quadratic, Nonlinear, Dynamic);
