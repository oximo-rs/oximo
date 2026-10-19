pub use oximo_expr::Bound;
use oximo_expr::{Constant, Expr, ExprArena, ModelError, VarId};
use smol_str::SmolStr;

use crate::domain::Domain;
use crate::model::Model;

/// Variable metadata held by the [`Model`]. Users do not construct this
/// directly, they get an [`Expr`] back from [`VarBuilder::build`] and look up
/// solution values via [`crate::Model`] / `oximo_solver::SolverResult`.
#[derive(Clone, Debug)]
pub struct Variable {
    pub id: VarId,
    pub name: SmolStr,
    pub domain: Domain,
    pub lb: Bound,
    pub ub: Bound,
    pub initial: Option<f64>,
}

impl Variable {
    /// Current numeric endpoints. Bounds never depend on optimization variables.
    pub fn resolved_bounds(&self, arena: &ExprArena) -> (f64, f64) {
        (self.lb.resolve(arena), self.ub.resolve(arena))
    }

    pub fn resolved(&self, arena: &ExprArena) -> ResolvedVariable {
        let (lb, ub) = self.resolved_bounds(arena);
        ResolvedVariable {
            id: self.id,
            name: self.name.clone(),
            domain: self.domain,
            lb,
            ub,
            initial: self.initial,
        }
    }
}

/// Numeric variable metadata captured for a translation or solve.
#[derive(Clone, Debug)]
pub struct ResolvedVariable {
    pub id: VarId,
    pub name: SmolStr,
    pub domain: Domain,
    pub lb: f64,
    pub ub: f64,
    pub initial: Option<f64>,
}

mod sealed {
    pub trait Sealed {}
}

/// Checked, model-aware input to a variable bound.
pub trait IntoBound<'a>: sealed::Sealed {
    /// # Errors
    /// Rejects a foreign handle or a composed parameter expression.
    fn into_bound(self, model: &Model) -> Result<Bound, ModelError>
    where
        Self: Sized,
    {
        self.__into_bound_for_model(model.id())
    }
    #[doc(hidden)]
    fn __into_bound_for_model(self, model: oximo_expr::ModelId) -> Result<Bound, ModelError>;
}

macro_rules! numeric_bounds {
    ($($t:ty),*) => { $(
        impl sealed::Sealed for $t {}
        impl IntoBound<'_> for $t {
            fn __into_bound_for_model(self, _: oximo_expr::ModelId) -> Result<Bound, ModelError> { Ok(Bound::Constant(f64::from(self))) }
        }
    )* };
}
numeric_bounds!(f64, f32, i32, u32, i16, u16, i8, u8);

macro_rules! wide_numeric_bounds {
    ($($t:ty),*) => { $(
        impl sealed::Sealed for $t {}
        impl IntoBound<'_> for $t {
            #[expect(clippy::cast_precision_loss, reason = "variable bounds use f64 numeric precision")]
            fn __into_bound_for_model(self, _: oximo_expr::ModelId) -> Result<Bound, ModelError> {
                Ok(Bound::Constant(self as f64))
            }
        }
    )* };
}
wide_numeric_bounds!(i64, u64, usize, isize);

impl sealed::Sealed for Expr<'_, Constant> {}
impl<'a> IntoBound<'a> for Expr<'a, Constant> {
    fn __into_bound_for_model(self, model: oximo_expr::ModelId) -> Result<Bound, ModelError> {
        if self.model_id() != model {
            return Err(oximo_expr::ModelMismatchError::new(model, self.model_id()).into());
        }
        let id = self.param_id().ok_or(ModelError::InvalidHandle("bound parameter"))?;
        if self.arena().borrow().try_param_value(id).is_none() {
            return Err(ModelError::UnknownParam(id));
        }
        Ok(Bound::Parameter(id))
    }
}

/// Display name of `v` within `vars`, degrading to `variable #<index>` when the
/// id is out of range (a foreign or not-yet-registered [`VarId`]). Used to build
/// human-readable error messages that name the offending variable.
#[must_use]
pub fn var_name<V: VariableName>(vars: &[V], v: VarId) -> String {
    vars.get(v.index())
        .map_or_else(|| format!("variable #{}", v.index()), |x| x.variable_name().to_string())
}

/// Display metadata shared by symbolic and resolved variable records.
pub trait VariableName {
    fn variable_name(&self) -> &str;
}
impl VariableName for Variable {
    fn variable_name(&self) -> &str {
        &self.name
    }
}
impl VariableName for ResolvedVariable {
    fn variable_name(&self) -> &str {
        &self.name
    }
}

/// Validate numeric bounds before a declaration or mutation is accepted.
pub(crate) fn validate_bounds(variable: VarId, lower: f64, upper: f64) -> Result<(), ModelError> {
    if lower.is_nan()
        || upper.is_nan()
        || lower > upper
        || lower == f64::INFINITY
        || upper == f64::NEG_INFINITY
    {
        return Err(ModelError::BoundsInfeasible { variable, lower, upper });
    }
    Ok(())
}

/// To be fixable,  the value is finite, integral when the domain is integer-valued,
/// and within `[lb, ub]`. A semicontinuous/semiinteger variable is 0 or at least its
/// threshold
pub(crate) fn assert_fixable(name: &str, domain: Domain, lb: f64, ub: f64, value: f64) {
    assert!(value.is_finite(), "cannot fix variable {name:?} to the non-finite value {value}");
    assert!(
        !domain.is_integer() || value.fract() == 0.0,
        "cannot fix {domain} variable {name:?} to the fractional value {value}"
    );
    let semi_zero = domain.semi_threshold().is_some() && value == 0.0;
    assert!(
        semi_zero || (lb <= value && value <= ub),
        "cannot fix variable {name:?} to {value}, outside its bounds [{lb}, {ub}]"
    );
    if let Some(threshold) = domain.semi_threshold() {
        assert!(
            value == 0.0 || value >= threshold,
            "cannot fix {domain} variable {name:?} to {value}: it must be 0 or at least {threshold}"
        );
    }
}

/// Builder backing the `variable!` macro. Configure bounds / domain, then call
/// [`Self::build`] to register the variable and obtain an `Expr` handle.
#[must_use = "VarBuilder does nothing until you call .build()"]
pub struct VarBuilder<'a> {
    pub(crate) model: &'a Model,
    pub(crate) name: SmolStr,
    pub(crate) lb: Bound,
    pub(crate) ub: Bound,
    pub(crate) domain: Domain,
    pub(crate) initial: Option<f64>,
}

impl<'a> std::fmt::Debug for VarBuilder<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VarBuilder")
            .field("name", &self.name)
            .field("lb", &self.lb)
            .field("ub", &self.ub)
            .field("domain", &self.domain)
            .finish()
    }
}

impl<'a> VarBuilder<'a> {
    /// # Panics
    /// Panics for a foreign handle or a composed bound expression.
    pub fn lb(mut self, v: impl IntoBound<'a>) -> Self {
        self.lb = v.into_bound(self.model).expect("invalid lower bound");
        self
    }

    /// # Panics
    /// Panics for a foreign handle or a composed bound expression.
    pub fn ub(mut self, v: impl IntoBound<'a>) -> Self {
        self.ub = v.into_bound(self.model).expect("invalid upper bound");
        self
    }

    /// # Panics
    /// Panics for a foreign handle or a composed bound expression.
    pub fn bounds(mut self, lb: impl IntoBound<'a>, ub: impl IntoBound<'a>) -> Self {
        self.lb = lb.into_bound(self.model).expect("invalid lower bound");
        self.ub = ub.into_bound(self.model).expect("invalid upper bound");
        self
    }

    /// Fix the variable to `value`, i.e. `bounds(value, value)`.
    ///
    /// # Panics
    ///
    /// Panics if `value` is not a feasible fixing for the domain and bounds set
    /// so far: non-finite, fractional on an integer domain, outside the bounds,
    /// or inside a semicontinuity gap. A `.domain(..)` or bound call placed after
    /// this one is not re-checked; the `variable!` macro always emits them first.
    pub fn fix(self, value: f64) -> Self {
        let arena = self.model.arena();
        assert_fixable(
            &self.name,
            self.domain,
            self.lb.resolve(&arena),
            self.ub.resolve(&arena),
            value,
        );
        drop(arena);
        self.bounds(value, value)
    }

    pub fn domain(mut self, d: Domain) -> Self {
        self.domain = d;
        self
    }

    pub fn integer(mut self) -> Self {
        self.domain = Domain::Integer;
        self
    }

    pub fn binary(mut self) -> Self {
        self.domain = Domain::Binary;
        self.lb = Bound::Constant(0.0);
        self.ub = Bound::Constant(1.0);
        self
    }

    pub fn initial(mut self, v: f64) -> Self {
        self.initial = Some(v);
        self
    }

    #[inline]
    pub fn build(self) -> Expr<'a, oximo_expr::Affine> {
        self.model.register_var(self)
    }
}
