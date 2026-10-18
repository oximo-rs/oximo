//! Typed, solver-independent constraint translation.

use oximo_core::function_set::Set;
use oximo_core::{
    Affine, EqualTo, GreaterThan, Interval, LessThan, Model, ModelId, ScalarFunctionData,
    SemanticFunction,
};

use crate::SolverError;
use crate::prepare::PreparedExpressions;

/// An immutable expression snapshot for typed translators.
#[derive(Debug)]
pub struct TranslationContext {
    model_id: ModelId,
    expressions: PreparedExpressions,
}

impl TranslationContext {
    /// Create a translation context with the model's identity and a snapshot of
    /// its expressions and current parameter values.
    ///
    /// # Errors
    /// Returns an error for unresolved conditional modeling.
    pub fn new(model: &Model) -> Result<Self, SolverError> {
        model.ensure_gdp_reformulated()?;
        Ok(Self {
            model_id: model.id(),
            expressions: PreparedExpressions::new((*model.arena()).clone()),
        })
    }

    /// Return the identity of the model captured by this context.
    pub fn model_id(&self) -> ModelId {
        self.model_id
    }

    /// Borrow the expression snapshot and its lazy coefficient extraction caches.
    pub fn expressions(&self) -> &PreparedExpressions {
        &self.expressions
    }

    /// Validate the function's model identity and expression roots, then translate
    /// the function/set pair using the supplied translator and this snapshot.
    /// A missing trait implementation rejects unsupported typed translations at compile time.
    ///
    /// # Errors
    /// Returns an ownership, numeric validation, or translator error.
    pub fn translate<
        F: SemanticFunction + oximo_core::FunctionInSet<S>,
        S: Set,
        T: TypedConstraintTranslator<F, S>,
    >(
        &self,
        translator: &T,
        function: &F,
        set: &S,
    ) -> Result<T::Output, SolverError> {
        self.validate_function(function)?;

        translator.translate(function, set, self)
    }

    /// Check that the function belongs to this model and its expression roots
    /// exist in the captured arena.
    ///
    /// # Errors
    /// Returns a backend error for a foreign model or roots beyond the snapshot.
    fn validate_function(&self, function: &impl SemanticFunction) -> Result<(), SolverError> {
        if function.model_id() != self.model_id {
            return Err(SolverError::Backend(
                "translation function belongs to a different model".into(),
            ));
        }

        let mut missing = false;
        function.visit_expressions(&mut |id| {
            missing |= id.index() >= self.expressions.arena().len();
        });

        if missing {
            return Err(SolverError::Backend(
                "function contains roots newer than the translation snapshot; refresh TranslationContext".into(),
            ));
        }

        Ok(())
    }
}

/// Translate a supported function/set pair into a solver representation.
pub trait TypedConstraintTranslator<F: SemanticFunction + oximo_core::FunctionInSet<S>, S: Set> {
    type Output;

    /// Build the output representation from the function, set, and expression snapshot.
    ///
    /// # Errors
    /// Returns a numeric validation or unsupported representation error.
    fn translate(
        &self,
        function: &F,
        set: &S,
        context: &TranslationContext,
    ) -> Result<Self::Output, SolverError>;
}

/// A scalar constraint set expressible as lower and upper bounds.
pub trait ScalarBoundSet: Set {
    /// Return `(lower, upper)`, using infinities for absent bounds.
    fn bounds(&self) -> (f64, f64);
}

impl ScalarBoundSet for LessThan {
    /// Return an unbounded lower side and the set's upper bound.
    fn bounds(&self) -> (f64, f64) {
        (f64::NEG_INFINITY, self.0)
    }
}

impl ScalarBoundSet for GreaterThan {
    /// Return the set's lower bound and an unbounded upper side.
    fn bounds(&self) -> (f64, f64) {
        (self.0, f64::INFINITY)
    }
}

impl ScalarBoundSet for EqualTo {
    /// Return the equality value as both bounds.
    fn bounds(&self) -> (f64, f64) {
        (self.0, self.0)
    }
}

impl ScalarBoundSet for Interval {
    /// Return the interval's lower and upper bounds.
    fn bounds(&self) -> (f64, f64) {
        (self.lower, self.upper)
    }
}

/// An affine constraint row with its constant shifted into the bounds.
#[derive(Clone, Debug)]
pub struct AffineRow {
    /// Variable coefficients with a zero constant term.
    pub terms: oximo_expr::LinearTerms<'static>,
    pub lower: f64,
    pub upper: f64,
}

/// This translator accepts affine functions only.
#[derive(Clone, Copy, Debug)]
pub struct AffineRowTranslator;

impl<S: ScalarBoundSet> TypedConstraintTranslator<ScalarFunctionData<Affine>, S>
    for AffineRowTranslator
where
    ScalarFunctionData<Affine>: oximo_core::FunctionInSet<S>,
{
    type Output = AffineRow;

    /// Extract affine coefficients and shift the constant into the set's bounds.
    ///
    /// # Errors
    /// Returns an error for a foreign model, missing expression roots, failed affine
    /// extraction, invalid bounds, non-finite coefficients, or finite-bound overflow.
    fn translate(
        &self,
        function: &ScalarFunctionData<Affine>,
        set: &S,
        context: &TranslationContext,
    ) -> Result<AffineRow, SolverError> {
        context.validate_function(function)?;

        let (lower, upper) = set.bounds();
        check_bounds(lower, upper)?;

        let mut terms = context
            .expressions
            .linear(function.expression())
            .ok_or_else(|| {
                SolverError::Backend("declared affine function failed extraction".into())
            })?
            .into_owned();

        if !terms.constant.is_finite() || terms.coeffs.iter().any(|(_, value)| !value.is_finite()) {
            return Err(SolverError::Backend(
                "affine translation contains non-finite coefficients".into(),
            ));
        }

        let shifted_lower = lower - terms.constant;
        let shifted_upper = upper - terms.constant;

        if (lower.is_finite() && !shifted_lower.is_finite())
            || (upper.is_finite() && !shifted_upper.is_finite())
        {
            return Err(SolverError::Backend("affine translation overflows a finite bound".into()));
        }

        terms.constant = 0.0;

        Ok(AffineRow { lower: shifted_lower, upper: shifted_upper, terms })
    }
}

/// Validate scalar bounds, converting validation failures into backend errors.
fn check_bounds(lower: f64, upper: f64) -> Result<(), SolverError> {
    oximo_core::function_set::validate_scalar_bounds(lower, upper)
        .map_err(|error| SolverError::Backend(error.to_string()))
}
