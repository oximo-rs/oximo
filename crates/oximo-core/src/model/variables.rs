//! Scalar variable and parameter registries and mutations.

use super::{Model, validate_batch_names};
use crate::SymmetricMatrix;
use crate::domain::Domain;
use crate::param::Parameter;
use crate::var::{Bound, IntoBound, ResolvedVariable, VarBuilder, Variable};
use oximo_expr::ModelError;
use oximo_expr::{Affine, Constant, Expr, ModelMismatchError, ParamId, VarId, extract_linear};
use rustc_hash::{FxBuildHasher, FxHashMap};
use smol_str::SmolStr;
use std::cell::Ref;

#[derive(Debug)]
pub(super) struct PendingVar {
    pub(super) name: SmolStr,
    pub(super) lb: Bound,
    pub(super) ub: Bound,
}

#[derive(Debug)]
pub(super) struct PendingParam {
    pub(super) name: SmolStr,
    pub(super) value: f64,
}

/// Reuse the validation table as the registry when this is its first family.
fn prepare_batch_names<'a, I: Copy>(
    existing: &FxHashMap<SmolStr, I>,
    items: impl ExactSizeIterator<Item = (&'a SmolStr, I)>,
    kind: &str,
) -> Option<FxHashMap<SmolStr, I>> {
    if !existing.is_empty() {
        let count = items.len();
        validate_batch_names(
            |name| existing.contains_key(name),
            items.map(|(name, _)| name),
            kind,
            count,
        );
        return None;
    }

    let mut names = FxHashMap::with_capacity_and_hasher(items.len(), FxBuildHasher);

    for (name, id) in items {
        let std::collections::hash_map::Entry::Vacant(entry) = names.entry(name.clone()) else {
            panic!("{kind} name {name:?} occurs more than once");
        };
        entry.insert(id);
    }

    Some(names)
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
            lb: Bound::Constant(f64::NEG_INFINITY),
            ub: Bound::Constant(f64::INFINITY),
            domain: Domain::Real,
            initial: None,
        }
    }

    /// Called by [`VarBuilder::build`]. Pushes the var into the registry and
    /// returns its `Expr` handle.
    #[inline]
    pub(crate) fn register_var<'a>(&'a self, b: VarBuilder<'a>) -> Expr<'a, Affine> {
        let mut arena = self.arena.borrow_mut();
        let (lo, hi) = (b.lb.resolve(&arena), b.ub.resolve(&arena));
        crate::var::validate_bounds(
            VarId(self.num_variables().try_into().expect("variable count")),
            lo,
            hi,
        )
        .expect("invalid variable bounds");
        assert!(
            (b.domain != Domain::Binary && b.domain.semi_threshold().is_none())
                || (b.lb.parameter().is_none() && b.ub.parameter().is_none()),
            "binary and semi-variable bounds must be constant"
        );
        let mut names = self.var_names.borrow_mut();
        let std::collections::hash_map::Entry::Vacant(name_entry) = names.entry(b.name.clone())
        else {
            panic!("variable name {:?} is already registered on this model", b.name);
        };

        let mut vars = self.variables.borrow_mut();
        let id = VarId(u32::try_from(vars.len()).expect("variable count overflow"));
        vars.push(Variable {
            id,
            name: b.name,
            domain: b.domain,
            lb: b.lb,
            ub: b.ub,
            initial: b.initial,
        });
        arena.__set_bound_check(id, b.lb, b.ub);
        name_entry.insert(id);
        drop(vars);
        drop(names);
        self.cached_kind.set(None);
        drop(arena);
        Expr::from_var(&self.arena, id)
    }

    pub(super) fn register_vars_batch<'a>(
        &'a self,
        items: &[PendingVar],
        domain: Domain,
    ) -> Vec<Expr<'a, Affine>> {
        if items.is_empty() {
            return Vec::new();
        }

        let mut arena = self.arena.borrow_mut();
        for (i, item) in items.iter().enumerate() {
            crate::var::validate_bounds(
                VarId((self.num_variables() + i).try_into().expect("variable count")),
                item.lb.resolve(&arena),
                item.ub.resolve(&arena),
            )
            .expect("invalid variable bounds");
            assert!(
                (domain != Domain::Binary && domain.semi_threshold().is_none())
                    || (item.lb.parameter().is_none() && item.ub.parameter().is_none()),
                "binary and semi-variable bounds must be constant"
            );
        }
        let mut names = self.var_names.borrow_mut();
        let mut vars = self.variables.borrow_mut();
        let first = vars.len();
        let final_count = first.checked_add(items.len()).expect("variable count overflow");
        u32::try_from(final_count - 1).expect("variable count overflow");
        let prepared_names = prepare_batch_names(
            &names,
            items.iter().enumerate().map(|(offset, item)| {
                (&item.name, VarId(u32::try_from(first + offset).expect("checked variable count")))
            }),
            "variable",
        );
        vars.reserve(items.len());

        if prepared_names.is_none() {
            names.reserve(items.len());
        }

        let expressions = arena.__extend_vars(first..final_count);
        let mut handles = Vec::with_capacity(items.len());

        for (item, expression) in items.iter().zip(expressions) {
            let id = VarId(u32::try_from(vars.len()).expect("variable count overflow"));
            vars.push(Variable {
                id,
                name: item.name.clone(),
                domain,
                lb: item.lb,
                ub: item.ub,
                initial: None,
            });

            if prepared_names.is_none() {
                names.insert(item.name.clone(), id);
            }
            arena.__set_bound_check(id, item.lb, item.ub);
            handles.push(expression);
        }

        if let Some(prepared_names) = prepared_names {
            *names = prepared_names;
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

    /// Capture numeric variable metadata at the current parameter assignment.
    pub fn resolved_variables(&self) -> Vec<ResolvedVariable> {
        let arena = self.arena();
        self.variables.borrow().iter().map(|v| v.resolved(&arena)).collect()
    }

    pub fn num_variables(&self) -> usize {
        self.variables.borrow().len()
    }

    /// Fix a single-variable expression to `value`.
    /// Convenience over [`Self::fix_var`] for handles from the `variable!` macro
    /// or [`crate::IndexedVar`] indexing.
    ///
    /// # Errors
    ///
    /// Rejects foreign or composed handles, non-finite values, and invalid variable writes.
    #[inline]
    pub fn fix<'a>(
        &self,
        e: impl Into<Expr<'a>>,
        value: f64,
    ) -> std::result::Result<(), ModelError> {
        let e: Expr = e.into();
        self.ensure_expr_model(e)?;
        let id = e.var_id().ok_or(ModelError::InvalidHandle("variable"))?;
        self.fix_var(id, value)
    }

    /// Fix variable `id` to a feasible finite value by replacing both bounds.
    ///
    /// # Errors
    /// Rejects unknown or frozen variables, fractional integer fixings, values outside
    /// the current bounds, and values inside a semicontinuity gap.
    pub fn fix_var(&self, id: VarId, value: f64) -> Result<(), ModelError> {
        self.ensure_reformulated_bounds_mutable(id)?;
        let mut arena = self.arena.borrow_mut();
        let mut vars = self.variables.borrow_mut();
        let v = vars.get_mut(id.index()).ok_or(ModelError::UnknownVariable(id))?;
        let (lb, ub) = v.resolved_bounds(&arena);
        let semi_zero = v.domain.semi_threshold().is_some() && value == 0.0;
        if !value.is_finite()
            || (v.domain.is_integer() && value.fract() != 0.0)
            || (!semi_zero && (value < lb || value > ub))
            || v.domain.semi_threshold().is_some_and(|t| value != 0.0 && value < t)
        {
            return Err(ModelError::VariableValue { variable: id, operation: "fix", value });
        }
        v.lb = Bound::Constant(value);
        v.ub = Bound::Constant(value);
        arena.__set_bound_check(id, v.lb, v.ub);
        self.cached_kind.set(None);
        Ok(())
    }

    /// Set the initial (warm-start) value of a single-variable expression.
    /// The macro API has no bound-style syntax for warm starts, so this is the
    /// supported way to seed `variable!`-declared variables.
    ///
    /// # Errors
    ///
    /// Rejects foreign or composed handles, non-finite values, and invalid variable writes.
    #[inline]
    pub fn set_initial<'a>(
        &self,
        e: impl Into<Expr<'a>>,
        value: f64,
    ) -> std::result::Result<(), ModelError> {
        let e: Expr = e.into();
        self.ensure_expr_model(e)?;
        let id = e.var_id().ok_or(ModelError::InvalidHandle("variable"))?;
        if !value.is_finite() {
            return Err(ModelError::VariableValue {
                variable: id,
                operation: "initial value",
                value,
            });
        }
        self.variables
            .borrow_mut()
            .get_mut(id.index())
            .ok_or(ModelError::UnknownVariable(id))?
            .initial = Some(value);
        Ok(())
    }

    /// Restore constant or bare-parameter bounds on an existing variable.
    /// Use constant infinities for unbounded directions. Binary endpoints remain zero or one.
    ///
    /// # Errors
    /// Rejects foreign/unknown parameters, invalid bounds, unsupported parameter-bound
    /// domains, and bounds embedded in a previous reformulation.
    pub fn unfix_var<'a>(
        &self,
        id: VarId,
        lb: impl IntoBound<'a>,
        ub: impl IntoBound<'a>,
    ) -> Result<(), ModelError> {
        self.ensure_reformulated_bounds_mutable(id)?;
        self.replace_bounds(id, lb.into_bound(self)?, ub.into_bound(self)?)
    }

    /// Replace both bounds of a bare variable handle from this model.
    ///
    /// Accepts constants or bare parameter handles; use constant infinities for
    /// unbounded directions. This also restores bounds after [`Self::fix`].
    ///
    /// # Errors
    /// Rejects foreign or composed variable handles and any bounds rejected by
    /// [`Self::unfix_var`], leaving the variable unchanged.
    pub fn set_bounds<'a>(
        &self,
        variable: impl Into<Expr<'a>>,
        lower: impl IntoBound<'a>,
        upper: impl IntoBound<'a>,
    ) -> Result<(), ModelError> {
        let variable = variable.into();
        self.ensure_expr_model(variable)?;
        let id = variable.var_id().ok_or(ModelError::InvalidHandle("variable"))?;
        self.unfix_var(id, lower, upper)
    }

    fn replace_bounds(&self, id: VarId, lb: Bound, ub: Bound) -> Result<(), ModelError> {
        let mut arena = self.arena.borrow_mut();
        let (lo, hi) = (lb.resolve(&arena), ub.resolve(&arena));
        crate::var::validate_bounds(id, lo, hi)?;
        let mut vars = self.variables.borrow_mut();
        let v = vars.get_mut(id.index()).ok_or(ModelError::UnknownVariable(id))?;
        if (v.domain == Domain::Binary || v.domain.semi_threshold().is_some())
            && (lb.parameter().is_some() || ub.parameter().is_some())
        {
            return Err(ModelError::ParameterBoundDomain);
        }
        if v.domain == Domain::Binary && !([0.0, 1.0].contains(&lo) && [0.0, 1.0].contains(&hi)) {
            return Err(ModelError::BoundsInfeasible { variable: id, lower: lo, upper: hi });
        }
        v.lb = lb;
        v.ub = ub;
        drop(vars);
        arena.__set_bound_check(id, lb, ub);
        self.cached_kind.set(None);
        Ok(())
    }

    /// SOS, indicator, and GDP reformulations embed bounds in generated rows.
    /// Changing one of those bounds could make the rows stale.
    fn ensure_reformulated_bounds_mutable(&self, id: VarId) -> Result<(), ModelError> {
        if id.index() >= self.num_variables() {
            return Err(ModelError::UnknownVariable(id));
        }
        #[cfg(feature = "gdp")]
        if self.gdp.borrow().locks_variable(id) {
            return Err(ModelError::FrozenVariable(id));
        }
        let sos = self.sos_constraints.borrow();
        if self
            .sos_reformulations
            .borrow()
            .as_deref()
            .into_iter()
            .flatten()
            .any(|a| sos[a.source.index()].members.iter().any(|m| m.variable == id))
        {
            return Err(ModelError::FrozenVariable(id));
        }
        let indicators = self.indicator_constraints.borrow();
        let arena = self.arena();
        if self.indicator_reformulations.borrow().as_deref().into_iter().flatten().any(|a| {
            let c = &indicators[a.source.index()];
            id != c.trigger
                && extract_linear(&arena, c.lhs)
                    .is_some_and(|t| t.coeffs.iter().any(|&(v, k)| v == id && k != 0.0))
        }) {
            return Err(ModelError::FrozenVariable(id));
        }
        Ok(())
    }

    /// Preserve symbolic bounds consumed by a generated reformulation.
    pub(crate) fn lock_bound_parameters(
        &self,
        variables: impl IntoIterator<Item = VarId>,
        source: &str,
    ) {
        let vars = self.variables.borrow();
        let parameters: Vec<_> = variables
            .into_iter()
            .flat_map(|id| {
                let v = &vars[id.index()];
                [v.lb.parameter(), v.ub.parameter()].into_iter().flatten()
            })
            .collect();
        drop(vars);
        let mut arena = self.arena.borrow_mut();
        for parameter in parameters {
            arena.__lock_bound_parameter(parameter, source.to_owned());
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
        let mut names = self.param_names.borrow_mut();
        let std::collections::hash_map::Entry::Vacant(entry) = names.entry(name.clone()) else {
            panic!("parameter name {name:?} is already registered on this model");
        };

        let (id, handle) = {
            let mut a = self.arena.borrow_mut();
            let id = a.new_param(value);
            (id, a.param_expr(id))
        };
        self.parameters.borrow_mut().push(Parameter { id, name });
        entry.insert(id);
        self.cached_kind.set(None);
        handle
    }

    pub(super) fn register_params_batch<'a>(
        &'a self,
        items: &[PendingParam],
    ) -> Vec<Expr<'a, Constant>> {
        if items.is_empty() {
            return Vec::new();
        }

        let mut names = self.param_names.borrow_mut();
        let mut params = self.parameters.borrow_mut();
        let mut arena = self.arena.borrow_mut();
        let first = arena.num_params();
        let final_count = first.checked_add(items.len()).expect("parameter count overflow");
        u32::try_from(final_count - 1).expect("parameter count overflow");
        let prepared_names = prepare_batch_names(
            &names,
            items.iter().enumerate().map(|(offset, item)| {
                (
                    &item.name,
                    ParamId(u32::try_from(first + offset).expect("checked parameter count")),
                )
            }),
            "parameter",
        );
        params.reserve(items.len());

        if prepared_names.is_none() {
            names.reserve(items.len());
        }

        let parameters = arena.__extend_params(items.iter().map(|item| item.value));

        let mut handles = Vec::with_capacity(items.len());

        for (item, (id, expression)) in items.iter().zip(parameters) {
            params.push(Parameter { id, name: item.name.clone() });

            if prepared_names.is_none() {
                names.insert(item.name.clone(), id);
            }
            handles.push(expression);
        }

        if let Some(prepared_names) = prepared_names {
            *names = prepared_names;
        }
        handles
    }

    /// Re-bind the parameter referenced by handle `p` to `value`.
    ///
    /// # Errors
    ///
    /// Rejects foreign or composed handles and invalid parameter assignments.
    #[inline]
    pub fn set_param<'a>(
        &self,
        p: impl Into<Expr<'a>>,
        value: f64,
    ) -> std::result::Result<(), ModelError> {
        let p: Expr = p.into();
        self.ensure_expr_model(p)?;
        let id = p.param_id().ok_or(ModelError::InvalidHandle("parameter"))?;
        self.set_param_id(id, value)
    }

    /// Re-bind parameter `id` to `value`. Takes effect on the next solve.
    ///
    /// The value is stored only in the expression arena (its single source of
    /// truth); extraction and evaluation read it from there.
    /// `ParamId` is a raw numeric index and carries no model provenance; prefer
    /// [`Self::set_param`] when an expression handle is available.
    ///
    /// # Errors
    /// Rejects unknown or frozen entities and invalid numerical assignments.
    pub fn set_param_id(&self, id: ParamId, value: f64) -> Result<(), ModelError> {
        self.arena.borrow_mut().set_param_value(id, value)?;
        self.cached_kind.set(None);
        Ok(())
    }

    /// Atomically validate and apply a complete parameter batch.
    ///
    /// # Errors
    /// Rejects unknown or frozen entities and invalid numerical assignments,
    /// leaving every value unchanged.
    pub fn set_param_ids(&self, updates: &[(ParamId, f64)]) -> Result<(), ModelError> {
        self.arena.borrow_mut().set_param_values(updates)?;
        self.cached_kind.set(None);
        Ok(())
    }

    /// Checked handle variant of the atomic batch setter.
    ///
    /// # Errors
    /// Rejects foreign or composed handles and invalid assignments.
    pub fn set_params(&self, updates: &[(Expr<'_, Constant>, f64)]) -> Result<(), ModelError> {
        let ids = updates
            .iter()
            .map(|&(p, v)| {
                self.ensure_expr_model(p)?;
                Ok((p.param_id().ok_or(ModelError::InvalidHandle("parameter"))?, v))
            })
            .collect::<Result<Vec<_>, ModelError>>()?;
        self.set_param_ids(&ids)
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
            lb: Bound::Constant(f64::NEG_INFINITY),
            ub: Bound::Constant(f64::INFINITY),
        });
        let handles = self.register_vars_batch(items.upper_triangle(), Domain::Real);
        SymmetricMatrix::from_upper_triangle(n, handles)
    }
}
