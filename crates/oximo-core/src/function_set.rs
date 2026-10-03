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
//! Expr<D> -> Constraint<F, S> -> LowerConstraint -> scalar/SOC/PSD IR -> Model
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
//! indicator consequents must be affine, since dynamic inputs are checked at
//! runtime, while statically quadratic or nonlinear inputs require explicit
//! narrowing.
//!
//! Lowering and registration panic for invalid numeric bounds, ownership,
//! dimensions, duplicate names, or count overflow before registering a row.
//! NaN bounds, positive-infinite lower bounds, and negative-infinite upper bounds
//! are rejected. Finite inverted intervals represent infeasible constraints and
//! remain valid modeling input. Indexed families are validated before worker
//! nodes are merged or rows registered.
//!
//! # Lowering and extensions
//!
//! [`Model`] keeps heterogeneous scalar rows as [`crate::AlgebraicConstraint`],
//! alongside its SOC, PSD, SOS, and indicator registries. Function wrappers retain
//! symbolic arena expressions rather than extracting coefficients or freezing
//! parameter values.
//!
//! [`Function`], [`Set`], and [`FunctionInSet`] are open extension traits. Custom
//! pairs implement [`LowerConstraint`] to return one of the sealed [`ConstraintIr`]
//! shapes, usually by delegating to a built-in [`Constraint::into_ir`]. Model
//! ownership and registration checks still apply to custom pairs.

use crate::{ConstraintHandle, Model, SocConstraintHandle};
use oximo_expr::{
    Affine, Constant, Degree, Dynamic, Expr, ExprClass, ExprId, Nonlinear, Quadratic,
};
use smol_str::SmolStr;

pub(crate) fn validate_bounds(lower: f64, upper: f64) {
    assert!(
        !lower.is_nan() && !upper.is_nan(),
        "constraint has NaN bound (lower={lower}, upper={upper})"
    );
    assert!(
        lower != f64::INFINITY && upper != f64::NEG_INFINITY,
        "constraint has invalid infinite bound (lower={lower}, upper={upper})"
    );
}

/// A mathematical function, distinct from an arithmetic DSL expression.
pub trait Function {}

/// A mathematical constraint set (distinct from the indexing collection `Set<K>`).
pub trait Set {}

/// An explicitly supported mathematical function/set pairing.
///
/// This does not imply convexity, feasibility, or support by any particular solver.
pub trait FunctionInSet<S: Set>: Function {}

/// A typed constraint. Unsupported pairings cannot be constructed.
#[derive(Clone, Copy, Debug)]
pub struct Constraint<F: FunctionInSet<S>, S: Set> {
    pub function: F,
    pub set: S,
}
impl<F: FunctionInSet<S>, S: Set> Constraint<F, S> {
    /// Construct a compatible pair. Runtime data is checked when lowering it.
    #[inline]
    pub fn new(function: F, set: S) -> Self {
        Self { function, set }
    }

    /// Cross the typed boundary without registering the resulting IR record.
    ///
    /// # Panics
    /// Built-in pairs panic for invalid bounds or cone dimensions.
    #[inline]
    pub fn into_ir(self) -> F::Ir
    where
        F: LowerConstraint<S>,
    {
        self.function.lower(self.set)
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
impl<T> Set for LessThan<T> {}
impl<T> Set for GreaterThan<T> {}
impl<T> Set for EqualTo<T> {}
impl<T> Set for Interval<T> {}

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
        impl Function for $name<'_> {}
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

/// Affine inputs accepted by the existing SOC syntax. Dynamic inputs are checked.
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

impl<'a> IntoAffineFunction<'a> for Expr<'a> {
    fn into_affine_function(self) -> ScalarAffineFunction<'a> {
        self.try_into().unwrap_or_else(|class| panic!("expression must be affine, got {class:?}"))
    }
}

impl<'a> IntoAffineFunction<'a> for ScalarDynamicFunction<'a> {
    fn into_affine_function(self) -> ScalarAffineFunction<'a> {
        self.0.into_affine_function()
    }
}

/// Runtime-classified scalar function for explicit dynamic expressions.
#[derive(Clone, Copy, Debug)]
pub enum AnyScalarFunction<'a> {
    Affine(ScalarAffineFunction<'a>),
    Quadratic(ScalarQuadraticFunction<'a>),
    Nonlinear(ScalarNonlinearFunction<'a>),
}

impl Function for AnyScalarFunction<'_> {}
impl<'a> IntoAffineFunction<'a> for AnyScalarFunction<'a> {
    fn into_affine_function(self) -> ScalarAffineFunction<'a> {
        self.expression().into_affine_function()
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
    + LowerConstraint<LessThan, Ir = AlgebraicConstraintIr<'a>>
    + LowerConstraint<GreaterThan, Ir = AlgebraicConstraintIr<'a>>
    + LowerConstraint<EqualTo, Ir = AlgebraicConstraintIr<'a>>
    + LowerConstraint<Interval, Ir = AlgebraicConstraintIr<'a>>
{
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
}

/// Recognized lowering outputs. Sealed to the existing heterogeneous IR shapes.
pub trait ConstraintIr: sealed::Sealed {
    type Handle;
    #[doc(hidden)]
    fn register(self, model: &Model, name: SmolStr) -> Self::Handle;
}

/// Public extension boundary for custom pairs.
///
/// Implementors return a recognized IR output, normally by delegating to a
/// built-in [`Constraint::into_ir`]. Registration still checks model ownership,
/// names, dimensions and numeric bounds.
pub trait LowerConstraint<S: Set>: FunctionInSet<S> {
    type Ir: ConstraintIr;
    fn lower(self, set: S) -> Self::Ir;
}

/// A validated pairing lowered to the scalar interval IR, retaining provenance.
#[derive(Clone, Copy, Debug)]
pub struct AlgebraicConstraintIr<'a> {
    pub(crate) lhs: Expr<'a>,
    pub(crate) lower: f64,
    pub(crate) upper: f64,
}
impl sealed::Sealed for AlgebraicConstraintIr<'_> {}
impl ConstraintIr for AlgebraicConstraintIr<'_> {
    type Handle = ConstraintHandle;
    fn register(self, model: &Model, name: SmolStr) -> Self::Handle {
        model.register_algebraic_ir(name, self)
    }
}
impl<'a, F, S> From<Constraint<F, S>> for AlgebraicConstraintIr<'a>
where
    S: Set,
    F: LowerConstraint<S, Ir = Self>,
{
    #[inline]
    fn from(c: Constraint<F, S>) -> Self {
        c.into_ir()
    }
}

/// Checked affine scalar IR for indicator consequents.
#[derive(Clone, Copy, Debug)]
pub struct AffineConstraintIr<'a>(pub(crate) AlgebraicConstraintIr<'a>);

impl<'a> AlgebraicConstraintIr<'a> {
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

/// Indicator inputs: affine functions or explicitly dynamic functions checked at runtime.
/// Statically quadratic/nonlinear constraints need explicit narrowing first.
pub trait IntoAffineConstraint<'a> {
    /// # Panics
    /// Panics for invalid bounds or a non-affine dynamic consequent.
    fn into_affine_constraint(self) -> AffineConstraintIr<'a>;
}
impl<'a> IntoAffineConstraint<'a> for AffineConstraintIr<'a> {
    fn into_affine_constraint(self) -> Self {
        self
    }
}
impl<'a, F, S> IntoAffineConstraint<'a> for Constraint<F, S>
where
    S: Set,
    F: FunctionInSet<S> + IntoAffineFunction<'a>,
    ScalarAffineFunction<'a>: LowerConstraint<S, Ir = AlgebraicConstraintIr<'a>>,
{
    fn into_affine_constraint(self) -> AffineConstraintIr<'a> {
        let function = self.function.into_affine_function();
        AffineConstraintIr(function.lower(self.set))
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
                self.lower(Interval { lower: f64::NEG_INFINITY, upper: set.0 })
            }
        }
        impl<'a> LowerConstraint<GreaterThan> for $function<'a> {
            type Ir = AlgebraicConstraintIr<'a>;
            #[inline]
            fn lower(self, set: GreaterThan) -> Self::Ir {
                self.lower(Interval { lower: set.0, upper: f64::INFINITY })
            }
        }
        impl<'a> LowerConstraint<EqualTo> for $function<'a> {
            type Ir = AlgebraicConstraintIr<'a>;
            #[inline]
            fn lower(self, set: EqualTo) -> Self::Ir {
                self.lower(Interval { lower: set.0, upper: set.0 })
            }
        }
        impl<'a> LowerConstraint<Interval> for $function<'a> {
            type Ir = AlgebraicConstraintIr<'a>;
            #[inline]
            fn lower(self, set: Interval) -> Self::Ir {
                validate_bounds(set.lower, set.upper);
                AlgebraicConstraintIr {
                    lhs: self.expression().erase(),
                    lower: set.lower,
                    upper: set.upper,
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
    fn is_affine(self) -> bool {
        self.0.__class() == ExprClass::Linear
    }
}
impl<'a> ScalarFunction<'a> for ScalarAffineFunction<'a> {
    #[inline]
    fn is_affine(self) -> bool {
        true
    }
}
impl<'a> ScalarFunction<'a> for ScalarQuadraticFunction<'a> {
    #[inline]
    fn is_affine(self) -> bool {
        self.0.__class() == ExprClass::Linear
    }
}
impl<'a> ScalarFunction<'a> for ScalarNonlinearFunction<'a> {
    #[inline]
    fn is_affine(self) -> bool {
        self.0.__class() == ExprClass::Linear
    }
}
impl<'a> ScalarFunction<'a> for AnyScalarFunction<'a> {
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
impl Function for VectorAffineFunction<'_> {}
impl<'a> VectorAffineFunction<'a> {
    /// Construct a nonempty vector, checking that every component has one owner.
    ///
    /// # Panics
    /// Panics for an empty vector or components belonging to different models.
    pub fn new(components: impl IntoIterator<Item = ScalarAffineFunction<'a>>) -> Self {
        let mut components = components.into_iter();
        let first = components.next().expect("affine vector must be nonempty").0;
        let rest = components
            .map(|f| {
                assert!(
                    std::ptr::eq(first.arena, f.0.arena),
                    "affine vector components must belong to the same model"
                );
                f.0.id
            })
            .collect();
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
