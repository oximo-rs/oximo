//! Scalar variable and parameter registries and mutations.

use super::{Model, validate_batch_names};
use crate::SymmetricMatrix;
use crate::domain::Domain;
use crate::param::Parameter;
use crate::var::{VarBuilder, Variable};
use oximo_expr::{Affine, Constant, Expr, ModelMismatchError, ParamId, VarId, extract_linear};
use smol_str::SmolStr;
use std::cell::Ref;

#[derive(Debug)]
pub(super) struct PendingVar {
    pub(super) name: SmolStr,
    pub(super) lb: f64,
    pub(super) ub: f64,
}

#[derive(Debug)]
pub(super) struct PendingParam {
    pub(super) name: SmolStr,
    pub(super) value: f64,
}

impl Model {
    // Variables

    /// Macro-facing entry point backing the `variable!` macro. Not part of the
    /// stable public API.
    #[doc(hidden)]
    pub fn __var(&self, name: impl Into<SmolStr>) -> VarBuilder<'_> {
        VarBuilder {
            model: self,
            name: name.into(),
            lb: f64::NEG_INFINITY,
            ub: f64::INFINITY,
            domain: Domain::Real,
            initial: None,
        }
    }

    /// Called by [`VarBuilder::build`]. Pushes the var into the registry and
    /// returns its `Expr` handle.
    pub(crate) fn register_var<'a>(&'a self, b: VarBuilder<'a>) -> Expr<'a, Affine> {
        let mut names = self.var_names.borrow_mut();
        assert!(
            !names.contains_key(&b.name),
            "variable name {:?} is already registered on this model",
            b.name
        );
        let mut vars = self.variables.borrow_mut();
        let id = VarId(u32::try_from(vars.len()).expect("variable count overflow"));
        vars.push(Variable {
            id,
            name: b.name.clone(),
            domain: b.domain,
            lb: b.lb,
            ub: b.ub,
            initial: b.initial,
        });
        names.insert(b.name, id);
        drop(vars);
        drop(names);
        self.cached_kind.set(None);
        Expr::from_var(&self.arena, id)
    }

    pub(super) fn register_vars_batch<'a>(
        &'a self,
        items: &[PendingVar],
        domain: Domain,
    ) -> Vec<Expr<'a, Affine>> {
        let mut names = self.var_names.borrow_mut();
        validate_batch_names(&names, items.iter().map(|item| &item.name), "variable", items.len());
        let mut vars = self.variables.borrow_mut();
        let final_count = vars.len().checked_add(items.len()).expect("variable count overflow");
        if final_count > 0 {
            u32::try_from(final_count - 1).expect("variable count overflow");
        }
        vars.reserve(items.len());
        names.reserve(items.len());

        let mut arena = self.arena.borrow_mut();
        arena.__reserve_nodes(items.len());
        let mut handles = Vec::with_capacity(items.len());
        for item in items {
            let id = VarId(u32::try_from(vars.len()).expect("variable count overflow"));
            vars.push(Variable {
                id,
                name: item.name.clone(),
                domain,
                lb: item.lb,
                ub: item.ub,
                initial: None,
            });
            names.insert(item.name.clone(), id);
            let handle = arena.var_expr(id);
            handles.push(handle);
        }
        self.cached_kind.set(None);
        handles
    }

    pub fn variable_id(&self, name: &str) -> Option<VarId> {
        self.var_names.borrow().get(name).copied()
    }

    /// Create a model-bound expression handle for an existing variable ID.
    ///
    /// This is useful after an ID-preserving model transformation, since source
    /// and transformed models have distinct identities, so the source expression
    /// handle is not accepted by the transformed model or its solver result.
    ///
    /// # Panics
    ///
    /// Panics if `id` is not registered on this model.
    #[must_use]
    pub fn variable_handle(&self, id: VarId) -> Expr<'_, Affine> {
        assert!(id.index() < self.variables.borrow().len(), "unknown variable ID {id:?}");
        Expr::from_var(&self.arena, id)
    }

    pub fn variables(&self) -> Ref<'_, Vec<Variable>> {
        self.variables.borrow()
    }

    pub fn num_variables(&self) -> usize {
        self.variables.borrow().len()
    }

    /// Fix a single-variable expression to `value`.
    /// Convenience over [`Self::fix_var`] for handles from the `variable!` macro
    /// or [`crate::IndexedVar`] indexing.
    ///
    /// # Panics
    ///
    /// Panics if `e` is not a bare variable handle from this model lineage, or on anything
    /// [`Self::fix_var`] rejects.
    ///
    /// # Errors
    ///
    /// Returns [`ModelMismatchError`] if `e` belongs to another model.
    #[inline]
    pub fn fix<'a>(
        &self,
        e: impl Into<Expr<'a>>,
        value: f64,
    ) -> std::result::Result<(), ModelMismatchError> {
        let e: Expr = e.into();
        self.ensure_expr_model(e)?;
        let id = e.var_id().expect("Model::fix expects a single-variable expression");
        self.fix_var(id, value);
        Ok(())
    }

    /// Fix variable `id` to `value` by setting `lb = ub = value`.
    ///
    /// `VarId` is a raw numeric index and carries no model provenance. Prefer
    /// [`Self::fix`] when an expression handle is available.
    ///
    /// # Panics
    ///
    /// Panics if `value` is not a feasible fixing for the variable (non-finite,
    /// fractional on an integer domain, outside its bounds, or inside a
    /// semicontinuity gap), or if its bounds are embedded in a previously
    /// reformulated SOS, indicator, or GDP row.
    pub fn fix_var(&self, id: VarId, value: f64) {
        self.assert_reformulated_bounds_mutable(id);
        let mut vars = self.variables.borrow_mut();
        let v = &mut vars[id.index()];
        crate::var::assert_fixable(&v.name, v.domain, v.lb, v.ub, value);
        v.lb = value;
        v.ub = value;
        drop(vars);
        self.cached_kind.set(None);
    }

    /// Set the initial (warm-start) value of a single-variable expression.
    /// The macro API has no bound-style syntax for warm starts, so this is the
    /// supported way to seed `variable!`-declared variables.
    ///
    /// # Panics
    ///
    /// Panics if `e` is not a bare variable handle from this model lineage.
    ///
    /// # Errors
    ///
    /// Returns [`ModelMismatchError`] if `e` belongs to another model.
    #[inline]
    pub fn set_initial<'a>(
        &self,
        e: impl Into<Expr<'a>>,
        value: f64,
    ) -> std::result::Result<(), ModelMismatchError> {
        let e: Expr = e.into();
        self.ensure_expr_model(e)?;
        let id = e.var_id().expect("Model::set_initial expects a single-variable expression");
        self.variables.borrow_mut()[id.index()].initial = Some(value);
        Ok(())
    }

    /// Restore bounds on variable `id`. Pass `f64::NEG_INFINITY` / `f64::INFINITY`
    /// to restore an unbounded direction.
    ///
    /// # Panics
    ///
    /// Panics if the variable belongs to an SOS or indicator constraint that
    /// has already been reformulated, or a GDP reformulation embeds its bounds.
    /// Binary variables must keep ordered bounds with each endpoint exactly zero or one.
    #[expect(clippy::float_cmp, reason = "binary bounds must be exactly zero or one")]
    pub fn unfix_var(&self, id: VarId, lb: f64, ub: f64) {
        self.assert_reformulated_bounds_mutable(id);
        let mut vars = self.variables.borrow_mut();
        let v = &mut vars[id.index()];
        if v.domain == Domain::Binary {
            assert!(
                (lb == 0.0 || lb == 1.0) && (ub == 0.0 || ub == 1.0) && lb <= ub,
                "binary variable bounds must remain ordered and each be exactly zero or one"
            );
        }
        v.lb = lb;
        v.ub = ub;
        drop(vars);
        self.cached_kind.set(None);
    }

    /// SOS, indicator, and GDP reformulations embed bounds in generated rows.
    /// Changing one of those bounds could make the rows stale.
    fn assert_reformulated_bounds_mutable(&self, id: VarId) {
        #[cfg(feature = "gdp")]
        assert!(
            !self.gdp.borrow().locks_variable(id),
            "cannot change bounds embedded in a GDP reformulation; change the source model before reformulating"
        );
        if let Some(source) = self.sos_constraints.borrow().iter().find(|constraint| {
            !constraint.active && constraint.members.iter().any(|member| member.variable == id)
        }) {
            panic!(
                "cannot change bounds of variable {:?} after SOS constraint {:?} was reformulated; \
                 change bounds before reformulating or create a new reformulated model",
                self.variables.borrow()[id.index()].name,
                source.name,
            );
        }
        let indicator_source = {
            let sources = self.indicator_constraints.borrow();
            let arena = self.arena.borrow();
            sources
                .iter()
                .find(|source| {
                    !source.active
                        && extract_linear(&arena, source.lhs).is_some_and(|terms| {
                            id != source.trigger
                                && terms.coeffs.iter().any(|(variable, coefficient)| {
                                    *variable == id && *coefficient != 0.0
                                })
                        })
                })
                .map(|source| source.name.clone())
        };
        if let Some(source) = indicator_source {
            panic!(
                "cannot change bounds of variable {:?} after indicator constraint {:?} was reformulated; \
                 change bounds before reformulating or create a new reformulated model",
                self.variables.borrow()[id.index()].name,
                source,
            );
        }
    }

    // Parameters

    /// Macro-facing entry point backing the `param!` macro. Not part of the
    /// stable public API.
    ///
    /// Registers a named scalar parameter initialized to `value`, returning an
    /// [`Expr`] handle that references it symbolically. A parameter behaves like a
    /// constant coefficient (`param * var` is linear) but stays symbolic so it can
    /// be re-bound with [`Self::set_param`] / [`Self::set_param_id`] between solves
    /// without rebuilding the model.
    ///
    /// # Panics
    ///
    /// Panics if a parameter with the same name is already registered.
    #[doc(hidden)]
    pub fn __param<'a>(&'a self, name: impl Into<SmolStr>, value: f64) -> Expr<'a, Constant> {
        self.register_param(name.into(), value)
    }

    /// Register one scalar parameter named `name` initialized to `value` and
    /// return its `Expr` handle. Shared by [`Self::__param`] and the indexed
    /// builder.
    ///
    /// # Panics
    ///
    /// Panics if a parameter with the same name is already registered.
    pub(super) fn register_param(&self, name: SmolStr, value: f64) -> Expr<'_, Constant> {
        assert!(
            !self.param_names.borrow().contains_key(&name),
            "parameter name {name:?} is already registered on this model"
        );
        let (id, handle) = {
            let mut a = self.arena.borrow_mut();
            let id = a.new_param(value);
            (id, a.param_expr(id))
        };
        self.parameters.borrow_mut().push(Parameter { id, name: name.clone() });
        self.param_names.borrow_mut().insert(name, id);
        self.cached_kind.set(None);
        handle
    }

    pub(super) fn register_params_batch<'a>(
        &'a self,
        items: &[PendingParam],
    ) -> Vec<Expr<'a, Constant>> {
        let mut names = self.param_names.borrow_mut();
        validate_batch_names(&names, items.iter().map(|item| &item.name), "parameter", items.len());
        let mut params = self.parameters.borrow_mut();
        let mut arena = self.arena.borrow_mut();
        let final_count = params.len().checked_add(items.len()).expect("parameter count overflow");
        if final_count > 0 {
            u32::try_from(final_count - 1).expect("parameter count overflow");
        }
        params.reserve(items.len());
        names.reserve(items.len());
        arena.__reserve_nodes(items.len());

        let mut handles = Vec::with_capacity(items.len());
        for item in items {
            let id = arena.new_param(item.value);
            let handle = arena.param_expr(id);
            params.push(Parameter { id, name: item.name.clone() });
            names.insert(item.name.clone(), id);
            handles.push(handle);
        }
        handles
    }

    /// Re-bind the parameter referenced by handle `p` to `value`.
    ///
    /// # Panics
    ///
    /// Panics if `p` is not a bare parameter handle from this model lineage
    /// (one returned by the `param!` macro).
    ///
    /// # Errors
    ///
    /// Returns [`ModelMismatchError`] if `p` belongs to another model.
    #[inline]
    pub fn set_param<'a>(
        &self,
        p: impl Into<Expr<'a>>,
        value: f64,
    ) -> std::result::Result<(), ModelMismatchError> {
        let p: Expr = p.into();
        self.ensure_expr_model(p)?;
        let id = p.param_id().expect("Model::set_param expects a single-parameter expression");
        self.set_param_id(id, value);
        Ok(())
    }

    /// Re-bind parameter `id` to `value`. Takes effect on the next solve.
    ///
    /// The value is stored only in the expression arena (its single source of
    /// truth); extraction and evaluation read it from there.
    /// `ParamId` is a raw numeric index and carries no model provenance; prefer
    /// [`Self::set_param`] when an expression handle is available.
    ///
    /// # Panics
    /// Panics if the ID is not registered, or a GDP reformulation embeds its value.
    pub fn set_param_id(&self, id: ParamId, value: f64) {
        self.arena.borrow_mut().set_param_value(id, value);
        self.cached_kind.set(None);
    }

    /// Current value bound to parameter `id`.
    ///
    /// # Panics
    ///
    /// Panics if `id` does not belong to a parameter registered on this model.
    pub fn param_value(&self, id: ParamId) -> f64 {
        self.arena.borrow().param_value(id)
    }

    /// Current value of the parameter referenced by handle `p`. Returns
    /// `Ok(None)` if `p` is not a bare parameter handle and
    /// [`ModelMismatchError`] if it belongs to another model.
    ///
    /// # Errors
    ///
    /// Returns [`ModelMismatchError`] if `p` belongs to another model.
    pub fn param_value_of<'a>(
        &self,
        p: impl Into<Expr<'a>>,
    ) -> std::result::Result<Option<f64>, ModelMismatchError> {
        let p: Expr = p.into();
        self.ensure_expr_model(p)?;
        Ok(p.param_id().map(|id| self.param_value(id)))
    }

    pub fn parameter_id(&self, name: &str) -> Option<ParamId> {
        self.param_names.borrow().get(name).copied()
    }

    pub fn parameters(&self) -> Ref<'_, Vec<Parameter>> {
        self.parameters.borrow()
    }

    pub fn num_parameters(&self) -> usize {
        self.parameters.borrow().len()
    }

    /// Declare a continuous symmetric matrix with shared mirrored scalar variables.
    ///
    /// # Panics
    /// Panics for invalid dimensions, duplicate entry names, or count overflow.
    pub fn add_symmetric_variable(
        &self,
        name: &str,
        n: usize,
    ) -> SymmetricMatrix<Expr<'_, Affine>> {
        let len = crate::triangle_len(n);
        let final_count = self.num_variables().checked_add(len).expect("variable count overflow");
        u32::try_from(final_count - 1).expect("variable count overflow");
        let items = SymmetricMatrix::from_upper_fn(n, |i, j| PendingVar {
            name: format!("{name}[{i},{j}]").into(),
            lb: f64::NEG_INFINITY,
            ub: f64::INFINITY,
        });
        let handles = self.register_vars_batch(items.upper_triangle(), Domain::Real);
        SymmetricMatrix::from_upper_triangle(n, handles)
    }
}
