//! Typed mathematical functions and constraint sets.
//!
//! A constraint states that a function's value belongs to a mathematical set
//! `f(x) in S`.
//! For example, `x + 2 * y <= 10` pairs an affine function with
//! [`LessThan(10.0)`](LessThan), while `||[x, y]||_2 <= t` pairs the affine
//! vector `[t, x, y]` with a [`SecondOrderCone`].
//!
//! The typed values are built in the following process:
//!
//! ```text
//! Expr<D> -> Constraint<F, S> -> owned semantic pair -> Model -> translation -> solver IR
//! ```
//!
//! [`FunctionInSet<S>`](FunctionInSet) marks supported function/set pairs at
//! compile time.
//!
//! # Expressions and functions
//!
//! [`Expr`] is the arithmetic DSL, with a conservative degree marker,
//! [`Constant`], [`Affine`], [`Quadratic`], [`Nonlinear`], or [`Dynamic`].
//! Variable handles are affine, while parameter handles are constant with
//! respect to decision variables, while remaining symbolic and rebindable.
//!
//! [`IntoFunction`] converts expressions into canonical function types.
//! Explicit dynamic conversion produces [`AnyScalarFunction`] after checking
//! the expression's structural degree. Scalar relations use
//! [`ScalarDynamicFunction`] for dynamic inputs and defer classification, since
//! every scalar degree supports inequalities and equalities.
//!
//! Use `x.square()` or `x * x` for a statically quadratic expression. Runtime
//! powers such as `x.powi(n)` return [`Dynamic`]. [`Expr::erase`] gives mixed
//! degrees a common type; [`Expr::try_affine`] and [`Expr::try_quadratic`] check
//! the expression before narrowing its degree.
//!
//! # Modeling and validation
//!
//! Unsupported function/set pairs fail at compile time. SOC components and
//! indicator consequents must be statically affine. Dynamic, quadratic and nonlinear
//! inputs require explicit checked narrowing before construction.
//!
//! Lowering and registration panic for invalid numeric bounds, ownership,
//! dimensions, duplicate names, or count overflow before registering a row.
//! NaN bounds, positive-infinite lower bounds, and negative-infinite upper bounds
//! are rejected. Finite inverted intervals represent infeasible constraints and
//! remain valid modeling input. Indexed families are validated before worker
//! nodes are merged or rows registered.
//!
//! # Registration, lowering and extensions
//!
//! Registration retains the function and set identities, symbolic roots, and
//! provenance before creating homogeneous scalar or cone views.
//! [`Function::Owned`] is a lifetime-independent [`crate::SemanticFunction`].
//! Custom functions implement its ownership, root-remapping and model-rebinding
//! contract. Direct registration requires a recognized function/set representation.
//! Built-in scalar lowering recognizes `f64` bound sets. Other parameterizations
//! such as `LessThan<f32>` need an explicit conversion to a supported IR before
//! registration.
//!
//! All ordinary registration APIs, including macros, bulk and indexed families,
//! use the declared function's representation. They never invoke [`LowerConstraint`].
//! To request an alternative transformation, call [`Constraint::into_ir`] and
//! register the result with [`Model::add_lowered_constraint`] or a scalar bulk API.

use crate::{Model, SocConstraintHandle};
use oximo_expr::{
    Affine, Constant, Degree, Dynamic, Expr, ExprClass, ExprId, Nonlinear, Quadratic,
};
use smol_str::SmolStr;

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ScalarBoundsError {
    #[error("constraint has NaN bound")]
    NaN,
    #[error("constraint has invalid infinite bound")]
    InvalidInfinity,
}

/// Validate scalar bound values. Finite inverted intervals represent infeasibility.
///
/// # Errors
/// Rejects NaN and inward-facing infinite bounds.
pub fn validate_scalar_bounds(lower: f64, upper: f64) -> Result<(), ScalarBoundsError> {
    if lower.is_nan() || upper.is_nan() {
        return Err(ScalarBoundsError::NaN);
    }

    if lower == f64::INFINITY || upper == f64::NEG_INFINITY {
        return Err(ScalarBoundsError::InvalidInfinity);
    }

    Ok(())
}

pub(crate) fn validate_bounds(lower: f64, upper: f64) {
    validate_scalar_bounds(lower, upper)
        .unwrap_or_else(|error| panic!("{error} (lower={lower}, upper={upper})"));
}

/// A mathematical function.
pub trait Function {
    type Owned: crate::semantic::SemanticFunction;

    /// Retain symbolic structure without borrowing the expression arena.
    fn semantic_data(&self) -> Self::Owned;
}

/// A mathematical constraint set (distinct from the indexing collection `Set<K>`).
pub trait Set: Clone + std::fmt::Debug + Send + Sync + 'static {}

/// An explicitly supported mathematical function/set pairing.
pub trait FunctionInSet<S: Set>: Function {}

/// A typed constraint. Unsupported pairings cannot be constructed.
#[derive(Clone, Copy, Debug)]
pub struct Constraint<F: FunctionInSet<S>, S: Set> {
    pub function: F,
    pub set: S,
}

impl<F: FunctionInSet<S>, S: Set> Constraint<F, S> {
    /// Construct a compatible pair. Runtime data is checked during registration or lowering.
    #[inline]
    pub fn new(function: F, set: S) -> Self {
        Self { function, set }
    }

    /// Cross the typed boundary without registering the resulting IR record.
    /// Scalar conversions retain the original pair alongside its lowered row.
    ///
    /// # Panics
    /// Panics for invalid source scalar bounds or invalid built-in cone dimensions.
    #[inline]
    pub fn into_ir(self) -> F::Ir
    where
        F: LowerConstraint<S>,
    {
        F::Ir::from_constraint(self)
    }
}

/// The scalar half-line `(-infinity, upper]`.
#[derive(Clone, Copy, Debug)]
pub struct LessThan<T = f64>(pub T);

/// The scalar half-line `[lower, +infinity)`.
#[derive(Clone, Copy, Debug)]
pub struct GreaterThan<T = f64>(pub T);

/// A scalar singleton.
#[derive(Clone, Copy, Debug)]
pub struct EqualTo<T = f64>(pub T);

/// A closed scalar interval. Inverted bounds represent an infeasible constraint.
#[derive(Clone, Copy, Debug)]
pub struct Interval<T = f64> {
    pub lower: T,
    pub upper: T,
}

impl<T: Clone + std::fmt::Debug + Send + Sync + 'static> Set for LessThan<T> {}
impl<T: Clone + std::fmt::Debug + Send + Sync + 'static> Set for GreaterThan<T> {}
impl<T: Clone + std::fmt::Debug + Send + Sync + 'static> Set for EqualTo<T> {}
impl<T: Clone + std::fmt::Debug + Send + Sync + 'static> Set for Interval<T> {}

/// The Lorentz cone where the first component bounds the norm of the remaining ones.
#[derive(Clone, Copy, Debug)]
pub struct SecondOrderCone {
    pub dimension: usize,
}
impl Set for SecondOrderCone {}

macro_rules! scalar_function {
    ($(#[$meta:meta])* $name:ident, $degree:ty) => {
        /// A scalar function retaining its symbolic arena representation.
        $(#[$meta])*
        #[derive(Clone, Copy, Debug)]
        pub struct $name<'a>(Expr<'a, $degree>);

        impl Function for $name<'_> {
            type Owned = crate::semantic::ScalarFunctionData<$degree>;

            fn semantic_data(&self) -> Self::Owned {
                crate::semantic::ScalarFunctionData::new(self.0.model_id(), self.0.id())
            }
        }

        impl crate::objective::ObjectiveFunction for $name<'_> {}
        impl<'a> $name<'a> {
            #[inline]
            pub fn expression(self) -> Expr<'a, $degree> {
                self.0
            }
        }

        impl<'a> From<Expr<'a, $degree>> for $name<'a> {
            #[inline]
            fn from(expr: Expr<'a, $degree>) -> Self {
                Self(expr)
            }
        }
    };
}

scalar_function!(ScalarAffineFunction, Affine);
scalar_function!(ScalarQuadraticFunction, Quadratic);
scalar_function!(ScalarNonlinearFunction, Nonlinear);
scalar_function!(
    /// Its degree is checked only when needed. Scalar inequalities and
    /// equalities accept every degree, so registering them needs no graph walk.
    /// Use [`IntoFunction::into_function`] on its expression for a checked class.
    ScalarDynamicFunction, Dynamic
);

impl<'a> From<Expr<'a, Constant>> for ScalarAffineFunction<'a> {
    fn from(expr: Expr<'a, Constant>) -> Self {
        Self(expr.into())
    }
}
impl<'a> TryFrom<Expr<'a>> for ScalarAffineFunction<'a> {
    type Error = ExprClass;
    fn try_from(expr: Expr<'a>) -> Result<Self, Self::Error> {
        expr.try_affine().map(Self)
    }
}
impl<'a> TryFrom<Expr<'a>> for ScalarQuadraticFunction<'a> {
    type Error = ExprClass;
    fn try_from(expr: Expr<'a>) -> Result<Self, Self::Error> {
        expr.try_quadratic().map(Self)
    }
}

impl<'a> From<Expr<'a, Constant>> for ScalarQuadraticFunction<'a> {
    fn from(e: Expr<'a, Constant>) -> Self {
        Self(e.into())
    }
}
impl<'a> From<Expr<'a, Affine>> for ScalarQuadraticFunction<'a> {
    fn from(e: Expr<'a, Affine>) -> Self {
        Self(e.into())
    }
}
impl<'a> From<Expr<'a, Constant>> for ScalarNonlinearFunction<'a> {
    fn from(e: Expr<'a, Constant>) -> Self {
        Self(e.nonlinear())
    }
}
impl<'a> From<Expr<'a, Affine>> for ScalarNonlinearFunction<'a> {
    fn from(e: Expr<'a, Affine>) -> Self {
        Self(e.nonlinear())
    }
}
impl<'a> From<Expr<'a, Quadratic>> for ScalarNonlinearFunction<'a> {
    fn from(e: Expr<'a, Quadratic>) -> Self {
        Self(e.nonlinear())
    }
}
impl<'a> From<Expr<'a, Dynamic>> for ScalarNonlinearFunction<'a> {
    fn from(e: Expr<'a, Dynamic>) -> Self {
        Self(e.nonlinear())
    }
}

/// Statically affine inputs. Narrow dynamic expressions explicitly with `try_affine`.
pub trait IntoAffineFunction<'a> {
    fn into_affine_function(self) -> ScalarAffineFunction<'a>;
}

impl<'a> IntoAffineFunction<'a> for ScalarAffineFunction<'a> {
    fn into_affine_function(self) -> Self {
        self
    }
}

impl<'a> IntoAffineFunction<'a> for Expr<'a, Affine> {
    fn into_affine_function(self) -> ScalarAffineFunction<'a> {
        self.into()
    }
}

impl<'a> IntoAffineFunction<'a> for Expr<'a, Constant> {
    fn into_affine_function(self) -> ScalarAffineFunction<'a> {
        self.into()
    }
}

/// Runtime-classified scalar function for explicit dynamic expressions.
#[derive(Clone, Copy, Debug)]
pub enum AnyScalarFunction<'a> {
    Affine(ScalarAffineFunction<'a>),
    Quadratic(ScalarQuadraticFunction<'a>),
    Nonlinear(ScalarNonlinearFunction<'a>),
}

impl Function for AnyScalarFunction<'_> {
    type Owned = AnyScalarFunctionData;

    fn semantic_data(&self) -> Self::Owned {
        let expression = self.expression();
        let class = match self {
            Self::Affine(_) => ExprClass::Linear,
            Self::Quadratic(_) => ExprClass::Quadratic,
            Self::Nonlinear(_) => ExprClass::Nonlinear,
        };
        AnyScalarFunctionData {
            expression: crate::semantic::ScalarFunctionData::new(
                expression.model_id(),
                expression.id(),
            ),
            class,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AnyScalarFunctionData {
    pub expression: crate::semantic::ScalarFunctionData<Dynamic>,
    pub class: ExprClass,
}

impl crate::semantic::SemanticFunction for AnyScalarFunctionData {
    fn model_id(&self) -> oximo_expr::ModelId {
        self.expression.model_id
    }

    fn rebind_model(&mut self, model_id: oximo_expr::ModelId) {
        self.expression.model_id = model_id;
    }

    fn visit_expressions(&self, visitor: &mut dyn FnMut(ExprId)) {
        visitor(self.expression.expression);
    }

    fn representation(&self) -> crate::semantic::FunctionRepresentationRef<'_> {
        crate::semantic::FunctionRepresentationRef::Scalar(self.expression.expression)
    }

    fn remap(&mut self, remap: oximo_expr::ExprIdRemap) {
        self.expression.expression = remap.apply(self.expression.expression);
    }
}

impl<'a> AnyScalarFunction<'a> {
    pub fn expression(self) -> Expr<'a> {
        match self {
            Self::Affine(f) => f.0.erase(),
            Self::Quadratic(f) => f.0.erase(),
            Self::Nonlinear(f) => f.0.erase(),
        }
    }
}

/// Conversion from the expression DSL into mathematical functions.
pub trait IntoFunction {
    type Function: Function;
    fn into_function(self) -> Self::Function;
}

/// Scalar function behavior shared by the built-in degree conversions.
pub trait ScalarFunction<'a>:
    Function
    + Copy
    + FunctionInSet<LessThan>
    + FunctionInSet<GreaterThan>
    + FunctionInSet<EqualTo>
    + FunctionInSet<Interval>
{
    /// Borrow the scalar expression without changing its declared function identity.
    fn expression(self) -> Expr<'a>;

    /// Whether this function can use one affine interval row.
    fn is_affine(self) -> bool;
}

/// Maps a sealed expression degree onto its canonical function representation.
pub trait FunctionDegree: Degree {
    type Function<'a>: ScalarFunction<'a>;
    /// Function used by scalar relations, which need no degree restriction.
    type RelationFunction<'a>: ScalarFunction<'a>;
    fn canonicalize(expr: Expr<'_, Self>) -> Self::Function<'_>;
    fn relation_function(expr: Expr<'_, Self>) -> Self::RelationFunction<'_>;
}

macro_rules! degree_function {
    ($degree:ty, $function:ident) => {
        impl FunctionDegree for $degree {
            type Function<'a> = $function<'a>;
            type RelationFunction<'a> = $function<'a>;
            #[inline]
            fn canonicalize(expr: Expr<'_, Self>) -> Self::Function<'_> {
                expr.into()
            }
            #[inline]
            fn relation_function(expr: Expr<'_, Self>) -> Self::RelationFunction<'_> {
                expr.into()
            }
        }
    };
}

degree_function!(Constant, ScalarAffineFunction);
degree_function!(Affine, ScalarAffineFunction);
degree_function!(Quadratic, ScalarQuadraticFunction);
degree_function!(Nonlinear, ScalarNonlinearFunction);

impl FunctionDegree for Dynamic {
    type Function<'a> = AnyScalarFunction<'a>;
    type RelationFunction<'a> = ScalarDynamicFunction<'a>;
    #[inline]
    fn relation_function(expr: Expr<'_, Self>) -> Self::RelationFunction<'_> {
        expr.into()
    }
    #[inline]
    fn canonicalize(expr: Expr<'_, Self>) -> Self::Function<'_> {
        match expr.classified() {
            oximo_expr::ClassifiedExpr::Affine(e) => AnyScalarFunction::Affine(e.into()),
            oximo_expr::ClassifiedExpr::Quadratic(e) => AnyScalarFunction::Quadratic(e.into()),
            oximo_expr::ClassifiedExpr::Nonlinear(e) => AnyScalarFunction::Nonlinear(e.into()),
        }
    }
}

impl<'a, D: FunctionDegree> IntoFunction for Expr<'a, D> {
    type Function = D::Function<'a>;
    #[inline]
    fn into_function(self) -> Self::Function {
        D::canonicalize(self)
    }
}

pub(crate) mod sealed {
    pub trait Sealed {}
    pub trait ScalarSet {}
    pub trait ScalarInput {}
}

/// Recognized lowering outputs. Sealed to the existing heterogeneous IR shapes.
pub trait ConstraintIr: sealed::Sealed {
    type Handle;

    /// Lower a pair using the source-preservation policy of this IR shape.
    #[doc(hidden)]
    fn from_constraint<F, S>(constraint: Constraint<F, S>) -> Self
    where
        Self: Sized,
        F: LowerConstraint<S, Ir = Self>,
        S: Set,
    {
        constraint.function.lower(constraint.set)
    }

    #[doc(hidden)]
    fn register(self, model: &Model, name: SmolStr) -> Self::Handle;
}

/// Optional algebraic conversion for compatible pairs.
///
/// Implementors return a recognized IR output. General semantic registration
/// does not require this trait. An explicit conversion may delegate to a built-in
/// [`Constraint::into_ir`].
pub trait LowerConstraint<S: Set>: FunctionInSet<S> {
    type Ir: ConstraintIr;
    fn lower(self, set: S) -> Self::Ir;

    /// Reuse the source pair when scalar conversion has already captured it.
    #[doc(hidden)]
    fn lower_with_semantic(
        self,
        set: S,
        _semantic: crate::semantic::ScalarConstraintSource,
    ) -> Self::Ir
    where
        Self: Sized,
    {
        self.lower(set)
    }
}

/// A validated pairing lowered to the scalar interval IR, retaining provenance.
#[derive(Clone, Debug)]
pub struct AlgebraicConstraintIr<'a> {
    pub(crate) lhs: Expr<'a>,
    pub(crate) lower: f64,
    pub(crate) upper: f64,
    pub(crate) semantic: crate::semantic::ScalarConstraintSource,
}

impl sealed::Sealed for AlgebraicConstraintIr<'_> {}
impl ConstraintIr for AlgebraicConstraintIr<'_> {
    type Handle = crate::ScalarConstraintHandle;

    #[inline]
    fn from_constraint<F, S>(constraint: Constraint<F, S>) -> Self
    where
        F: LowerConstraint<S, Ir = Self>,
        S: Set,
    {
        let semantic =
            crate::semantic::ScalarConstraintSource::capture(&constraint.function, &constraint.set);
        let mut ir = constraint.function.lower_with_semantic(constraint.set, semantic.clone());
        ir.semantic = semantic;
        ir
    }

    fn register(self, model: &Model, name: SmolStr) -> Self::Handle {
        model.register_algebraic_ir(name, self)
    }
}

/// Checked affine scalar IR for indicator consequents.
#[derive(Clone, Debug)]
pub struct AffineConstraintIr<'a>(pub(crate) AlgebraicConstraintIr<'a>);

impl<'a> AlgebraicConstraintIr<'a> {
    /// Project one scalar declaration without applying an alternative transformation.
    #[inline]
    fn declared(lhs: Expr<'a>, semantic: crate::semantic::ScalarConstraintSource) -> Self {
        let view = semantic.view();
        assert_eq!(semantic.model_id(), lhs.model_id(), "function belongs to a different model");
        assert_eq!(
            view.lhs,
            lhs.id(),
            "conversion changed the declared function; use an explicit lowering"
        );
        Self { lhs, lower: view.lower, upper: view.upper, semantic }
    }

    /// Check erased or custom scalar IR before using it in an indicator.
    ///
    /// # Panics
    /// Panics for invalid bounds or a non-affine expression.
    pub fn into_affine(self) -> AffineConstraintIr<'a> {
        validate_bounds(self.lower, self.upper);
        self.lhs.try_affine().expect("indicator consequent must be affine");
        AffineConstraintIr(self)
    }
}

impl<F, S> Constraint<F, S>
where
    S: ScalarConstraintSet,
    F: FunctionInSet<S>,
{
    /// Check a scalar range side before registering it as an indicator consequent.
    /// Internal range functions come from the sealed expression-degree conversions.
    pub(crate) fn into_checked_affine<'a>(self) -> AffineConstraintIr<'a>
    where
        F: ScalarFunction<'a>,
    {
        let expression = self.function.expression();
        let semantic = crate::semantic::ScalarConstraintSource::capture(&self.function, &self.set);
        AlgebraicConstraintIr::declared(expression, semantic).into_affine()
    }
}

/// Statically affine indicator inputs. Other degrees require explicit narrowing first.
pub trait IntoAffineConstraint<'a> {
    /// # Panics
    /// Panics for invalid bounds.
    fn into_affine_constraint(self) -> AffineConstraintIr<'a>;
}
impl<'a> IntoAffineConstraint<'a> for AffineConstraintIr<'a> {
    fn into_affine_constraint(self) -> Self {
        self
    }
}
impl<'a, F, S> IntoAffineConstraint<'a> for Constraint<F, S>
where
    S: ScalarConstraintSet,
    F: FunctionInSet<S> + IntoAffineFunction<'a>,
{
    fn into_affine_constraint(self) -> AffineConstraintIr<'a> {
        let semantic = crate::semantic::ScalarConstraintSource::capture(&self.function, &self.set);
        let expression = self.function.into_affine_function().expression().erase();
        let ir = AlgebraicConstraintIr::declared(expression, semantic);
        ir.semantic.validate_roots(expression.arena().__node_count());
        AffineConstraintIr(ir)
    }
}

macro_rules! scalar_pairs {
    ($function:ident) => {
        impl FunctionInSet<LessThan> for $function<'_> {}
        impl FunctionInSet<GreaterThan> for $function<'_> {}
        impl FunctionInSet<EqualTo> for $function<'_> {}
        impl FunctionInSet<Interval> for $function<'_> {}
        impl<'a> LowerConstraint<LessThan> for $function<'a> {
            type Ir = AlgebraicConstraintIr<'a>;

            #[inline]
            fn lower(self, set: LessThan) -> Self::Ir {
                let semantic = crate::semantic::ScalarConstraintSource::capture(&self, &set);
                self.lower_with_semantic(set, semantic)
            }

            #[inline]
            fn lower_with_semantic(
                self,
                set: LessThan,
                semantic: crate::semantic::ScalarConstraintSource,
            ) -> Self::Ir {
                validate_bounds(f64::NEG_INFINITY, set.0);
                AlgebraicConstraintIr {
                    lhs: self.expression().erase(),
                    lower: f64::NEG_INFINITY,
                    upper: set.0,
                    semantic,
                }
            }
        }

        impl<'a> LowerConstraint<GreaterThan> for $function<'a> {
            type Ir = AlgebraicConstraintIr<'a>;

            #[inline]
            fn lower(self, set: GreaterThan) -> Self::Ir {
                let semantic = crate::semantic::ScalarConstraintSource::capture(&self, &set);
                self.lower_with_semantic(set, semantic)
            }

            #[inline]
            fn lower_with_semantic(
                self,
                set: GreaterThan,
                semantic: crate::semantic::ScalarConstraintSource,
            ) -> Self::Ir {
                validate_bounds(set.0, f64::INFINITY);
                AlgebraicConstraintIr {
                    lhs: self.expression().erase(),
                    lower: set.0,
                    upper: f64::INFINITY,
                    semantic,
                }
            }
        }

        impl<'a> LowerConstraint<EqualTo> for $function<'a> {
            type Ir = AlgebraicConstraintIr<'a>;

            #[inline]
            fn lower(self, set: EqualTo) -> Self::Ir {
                let semantic = crate::semantic::ScalarConstraintSource::capture(&self, &set);
                self.lower_with_semantic(set, semantic)
            }

            #[inline]
            fn lower_with_semantic(
                self,
                set: EqualTo,
                semantic: crate::semantic::ScalarConstraintSource,
            ) -> Self::Ir {
                validate_bounds(set.0, set.0);
                AlgebraicConstraintIr {
                    lhs: self.expression().erase(),
                    lower: set.0,
                    upper: set.0,
                    semantic,
                }
            }
        }

        impl<'a> LowerConstraint<Interval> for $function<'a> {
            type Ir = AlgebraicConstraintIr<'a>;
            #[inline]
            fn lower(self, set: Interval) -> Self::Ir {
                let semantic = crate::semantic::ScalarConstraintSource::capture(&self, &set);
                self.lower_with_semantic(set, semantic)
            }

            #[inline]
            fn lower_with_semantic(
                self,
                set: Interval,
                semantic: crate::semantic::ScalarConstraintSource,
            ) -> Self::Ir {
                validate_bounds(set.lower, set.upper);
                AlgebraicConstraintIr {
                    lhs: self.expression().erase(),
                    lower: set.lower,
                    upper: set.upper,
                    semantic,
                }
            }
        }
    };
}

scalar_pairs!(ScalarAffineFunction);
scalar_pairs!(ScalarQuadraticFunction);
scalar_pairs!(ScalarNonlinearFunction);
scalar_pairs!(ScalarDynamicFunction);
scalar_pairs!(AnyScalarFunction);

impl<'a> ScalarFunction<'a> for ScalarDynamicFunction<'a> {
    #[inline]
    fn expression(self) -> Expr<'a> {
        ScalarDynamicFunction::expression(self).erase()
    }

    #[inline]
    fn is_affine(self) -> bool {
        self.0.__class() == ExprClass::Linear
    }
}

impl<'a> ScalarFunction<'a> for ScalarAffineFunction<'a> {
    #[inline]
    fn expression(self) -> Expr<'a> {
        ScalarAffineFunction::expression(self).erase()
    }

    #[inline]
    fn is_affine(self) -> bool {
        true
    }
}

impl<'a> ScalarFunction<'a> for ScalarQuadraticFunction<'a> {
    #[inline]
    fn expression(self) -> Expr<'a> {
        ScalarQuadraticFunction::expression(self).erase()
    }

    #[inline]
    fn is_affine(self) -> bool {
        self.0.__class() == ExprClass::Linear
    }
}

impl<'a> ScalarFunction<'a> for ScalarNonlinearFunction<'a> {
    #[inline]
    fn expression(self) -> Expr<'a> {
        ScalarNonlinearFunction::expression(self).erase()
    }

    #[inline]
    fn is_affine(self) -> bool {
        self.0.__class() == ExprClass::Linear
    }
}

impl<'a> ScalarFunction<'a> for AnyScalarFunction<'a> {
    #[inline]
    fn expression(self) -> Expr<'a> {
        AnyScalarFunction::expression(self).erase()
    }

    #[inline]
    fn is_affine(self) -> bool {
        matches!(self, Self::Affine(_))
    }
}

/// Symbolic affine vector. Its first component is stored separately so SOC
/// lowering can move the remaining IDs directly into the existing registry.
#[derive(Clone, Debug)]
pub struct VectorAffineFunction<'a> {
    pub(crate) first: Expr<'a, Affine>,
    pub(crate) rest: Vec<ExprId>,
}

impl Function for VectorAffineFunction<'_> {
    type Owned = crate::semantic::VectorAffineData;

    fn semantic_data(&self) -> Self::Owned {
        Self::Owned {
            model_id: self.first.model_id(),
            components: std::iter::once(self.first.id()).chain(self.rest.iter().copied()).collect(),
        }
    }
}

impl<'a> VectorAffineFunction<'a> {
    pub(crate) fn into_data(mut self) -> crate::VectorAffineData {
        self.rest.insert(0, self.first.id());
        crate::VectorAffineData { model_id: self.first.model_id(), components: self.rest.into() }
    }

    /// Construct a nonempty vector, checking that every component has one owner.
    ///
    /// # Panics
    /// Panics for an empty vector or components belonging to different models.
    pub fn new(components: impl IntoIterator<Item = ScalarAffineFunction<'a>>) -> Self {
        let mut components = components.into_iter();
        let first = components.next().expect("affine vector must be nonempty").0;

        // Leave room for the first component when registration consumes this function.
        let mut rest = Vec::with_capacity(components.size_hint().0.saturating_add(1));
        rest.extend(components.map(|f| {
            assert!(
                std::ptr::eq(first.arena, f.0.arena),
                "affine vector components must belong to the same model"
            );
            f.0.id
        }));
        Self { first, rest }
    }

    pub fn dimension(&self) -> usize {
        self.rest.len() + 1
    }
}

impl FunctionInSet<SecondOrderCone> for VectorAffineFunction<'_> {}

/// A lowered SOC with model provenance and checked dimension.
#[derive(Debug)]
pub struct SocConstraintIr<'a>(pub(crate) VectorAffineFunction<'a>);
impl sealed::Sealed for SocConstraintIr<'_> {}
impl ConstraintIr for SocConstraintIr<'_> {
    type Handle = SocConstraintHandle;
    fn register(self, model: &Model, name: SmolStr) -> Self::Handle {
        model.register_soc_ir(name, self)
    }
}
impl<'a> LowerConstraint<SecondOrderCone> for VectorAffineFunction<'a> {
    type Ir = SocConstraintIr<'a>;
    fn lower(self, set: SecondOrderCone) -> Self::Ir {
        assert!(set.dimension >= 2, "SOC needs at least two components, got {}", set.dimension);
        assert_eq!(self.dimension(), set.dimension, "SOC dimension mismatch");
        SocConstraintIr(self)
    }
}

/// Built-in scalar bound sets accepted by scalar registration APIs.
/// Other sets need an explicitly lowered IR.
pub trait ScalarConstraintSet: Set + sealed::ScalarSet {}

macro_rules! scalar_registration_sets {
    ($($set:ty),* $(,)?) => { $(
        impl sealed::ScalarSet for $set {}
        impl ScalarConstraintSet for $set {}
    )* };
}
scalar_registration_sets!(LessThan, GreaterThan, EqualTo, Interval);

/// Scalar row and its original declaration, prepared without changing coordinates.
#[doc(hidden)]
#[derive(Debug)]
pub struct PreparedScalarConstraint {
    pub(crate) view: crate::semantic::ScalarConstraintView,
    pub(crate) semantic: crate::semantic::ScalarConstraintSource,
}

impl PreparedScalarConstraint {
    #[inline]
    pub(crate) fn declared(semantic: crate::semantic::ScalarConstraintSource) -> Self {
        Self { view: semantic.view(), semantic }
    }

    #[inline]
    pub(crate) fn validate(&self, model_id: oximo_expr::ModelId, node_count: usize) {
        assert_eq!(self.semantic.model_id(), model_id, "function belongs to a different model");
        assert!(self.view.lhs.index() < node_count, "constraint contains an invalid expression ID");
        self.semantic.validate_roots(node_count);
        validate_bounds(self.view.lower, self.view.upper);
    }
}

/// Scalar registration input retaining its signature through indexed construction.
/// Declared pairs use their representation; alternative transformations require `into_ir()`.
/// Sealed so downstream code cannot claim a different registry's handle types.
/// Extend scalar modeling through `Function`, `FunctionInSet`, and `LowerConstraint`.
pub trait ScalarConstraintInput: sealed::ScalarInput {
    type Function: 'static;
    type Set: 'static;

    #[doc(hidden)]
    fn into_scalar_input(self) -> PreparedScalarConstraint;
}

impl<F, S> sealed::ScalarInput for Constraint<F, S>
where
    S: ScalarConstraintSet,
    F: FunctionInSet<S>,
{
}
impl sealed::ScalarInput for AlgebraicConstraintIr<'_> {}
impl sealed::ScalarInput for PreparedScalarConstraint {}

impl<F, S> ScalarConstraintInput for Constraint<F, S>
where
    S: ScalarConstraintSet,
    F: FunctionInSet<S>,
{
    type Function = F::Owned;
    type Set = S;

    #[inline]
    fn into_scalar_input(self) -> PreparedScalarConstraint {
        PreparedScalarConstraint::declared(crate::semantic::ScalarConstraintSource::capture(
            &self.function,
            &self.set,
        ))
    }
}

impl<'a> ScalarConstraintInput for AlgebraicConstraintIr<'a> {
    // Preserve a scalar return view without claiming the source's semantic signature.
    type Function = crate::constraint::ScalarHandleMarker;
    type Set = crate::semantic::Erased;

    #[inline]
    fn into_scalar_input(self) -> PreparedScalarConstraint {
        assert_eq!(
            self.semantic.model_id(),
            self.lhs.model_id(),
            "function belongs to a different model"
        );
        PreparedScalarConstraint {
            view: crate::semantic::ScalarConstraintView {
                lhs: self.lhs.id,
                lower: self.lower,
                upper: self.upper,
            },
            semantic: self.semantic,
        }
    }
}

impl ScalarConstraintInput for PreparedScalarConstraint {
    // As above, this is a return-handle marker.
    type Function = crate::constraint::ScalarHandleMarker;
    type Set = crate::semantic::Erased;

    fn into_scalar_input(self) -> Self {
        self
    }
}

impl<T: crate::SemanticFunction> Function for T {
    type Owned = Self;

    fn semantic_data(&self) -> Self {
        self.clone()
    }
}
macro_rules! owned_scalar_pairs {
    ($($set:ty),*) => { $(impl<D: Degree> FunctionInSet<$set> for crate::ScalarFunctionData<D> {})* };
}
owned_scalar_pairs!(LessThan, GreaterThan, EqualTo, Interval);

impl FunctionInSet<SecondOrderCone> for crate::VectorAffineData {}
impl FunctionInSet<crate::PositiveSemidefiniteCone> for crate::SymmetricAffineData {}

impl<D: Degree> crate::objective::ObjectiveFunction for crate::ScalarFunctionData<D> {}
impl crate::objective::ObjectiveFunction for AnyScalarFunction<'_> {}
impl crate::objective::ObjectiveFunction for AnyScalarFunctionData {}

impl FunctionInSet<LessThan> for AnyScalarFunctionData {}
impl FunctionInSet<GreaterThan> for AnyScalarFunctionData {}
impl FunctionInSet<EqualTo> for AnyScalarFunctionData {}
impl FunctionInSet<Interval> for AnyScalarFunctionData {}
impl FunctionInSet<crate::SpecialOrderedSet> for crate::VariableVectorData {}
impl FunctionInSet<crate::IndicatorSet> for crate::IndicatorFunctionData {}
