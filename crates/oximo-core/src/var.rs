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

    pub fn fix(self, value: f64) -> Self {
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

    pub fn build(self) -> Expr<'a> {
        self.model.register_var(self)
    }
}
