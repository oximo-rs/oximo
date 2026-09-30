use oximo_expr::{Expr, VarId};
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
    pub lb: f64,
    pub ub: f64,
    pub initial: Option<f64>,
}

/// Display name of `v` within `vars`, degrading to `variable #<index>` when the
/// id is out of range (a foreign or not-yet-registered [`VarId`]). Used to build
/// human-readable error messages that name the offending variable.
#[must_use]
pub fn var_name(vars: &[Variable], v: VarId) -> String {
    vars.get(v.index()).map_or_else(|| format!("variable #{}", v.index()), |x| x.name.to_string())
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
    pub(crate) lb: f64,
    pub(crate) ub: f64,
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
    pub fn lb(mut self, v: f64) -> Self {
        self.lb = v;
        self
    }

    pub fn ub(mut self, v: f64) -> Self {
        self.ub = v;
        self
    }

    pub fn bounds(mut self, lb: f64, ub: f64) -> Self {
        self.lb = lb;
        self.ub = ub;
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
        assert_fixable(&self.name, self.domain, self.lb, self.ub, value);
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
        self.lb = 0.0;
        self.ub = 1.0;
        self
    }

    pub fn initial(mut self, v: f64) -> Self {
        self.initial = Some(v);
        self
    }

    pub fn build(self) -> Expr<'a, oximo_expr::Affine> {
        self.model.register_var(self)
    }
}
