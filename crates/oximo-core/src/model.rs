use std::cell::{Cell, Ref, RefCell};
use std::marker::PhantomData;

use oximo_expr::{EvalError, Expr, ExprArena, ExprClass, ExprId, ParamId, VarId, classify};
use rustc_hash::FxHashMap;
use smol_str::SmolStr;

use crate::constraint::{Constraint, ConstraintExpr, ConstraintId, IntoRhs, Relate, Sense};
use crate::domain::Domain;
use crate::error::{Error, Result};
use crate::indexed::{IndexedFamily, IndexedParam, IndexedVar, build_storage};
use crate::objective::{Objective, ObjectiveSense};
use crate::param::Parameter;
use crate::set::{Axis, FromIndexKey, IndexKey, Set};
use crate::soc::{SocConstraint, SocConstraintId, detect_soc};
use crate::var::{VarBuilder, Variable};

/// The kind of mathematical program a `Model` represents.
///
/// This is inferred from the variables and expressions in the model, not set
/// explicitly by the user. See [`Model::kind`] for the exact decision ladder.
///
/// The `MI*` variant of each class is picked when any variable has an integer
/// domain. The continuous classes are, from most to least general:
///
/// - `NLP`: some expression is nonlinear (degree > 2, transcendental, division)
/// - `QCP`: some constraint is quadratic and not recognized as a second-order
///   cone
/// - `SOCP`: second-order cone constraints are present (explicit
///   [`crate::SocConstraint`]s or SOC-shaped quadratic constraints recognized
///   by [`crate::detect_soc`]); the objective may be linear or quadratic
/// - `QP`: quadratic objective, linear constraints
/// - `LP`: everything linear
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ModelKind {
    LP,
    MILP,
    QP,
    MIQP,
    QCP,
    MIQCP,
    SOCP,
    MISOCP,
    NLP,
    MINLP,
}

/// The optimization model. Owns the expression arena, variable/parameter
/// registries, constraints, and (optional) objective.
///
/// `Model` uses interior mutability so the builder API can take `&self`
/// references.
///
/// Variables, constraints, and the objective are added through
/// `RefCell`s under the hood.
pub struct Model {
    pub name: String,
    pub(crate) arena: RefCell<ExprArena>,
    pub(crate) variables: RefCell<Vec<Variable>>,
    pub(crate) var_names: RefCell<FxHashMap<SmolStr, VarId>>,
    pub(crate) parameters: RefCell<Vec<Parameter>>,
    pub(crate) param_names: RefCell<FxHashMap<SmolStr, ParamId>>,
    pub(crate) constraints: RefCell<Vec<Constraint>>,
    pub(crate) constraint_names: RefCell<FxHashMap<SmolStr, ConstraintId>>,
    pub(crate) soc_constraints: RefCell<Vec<SocConstraint>>,
    pub(crate) soc_names: RefCell<FxHashMap<SmolStr, SocConstraintId>>,
    pub(crate) objective: RefCell<Option<Objective>>,
    objective_declared: Cell<bool>,
    cached_kind: Cell<Option<ModelKind>>,
    /// Monotonic counter for auto-naming anonymous constraints registered via
    /// the `constraint!` macro.
    auto_seq: Cell<u32>,
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model")
            .field("name", &self.name)
            .field("vars", &self.variables.borrow().len())
            .field("params", &self.parameters.borrow().len())
            .field("constraints", &self.constraints.borrow().len())
            .field("soc_constraints", &self.soc_constraints.borrow().len())
            .field("has_objective", &self.objective.borrow().is_some())
            .field("feasibility", &self.is_feasibility())
            .finish()
    }
}

impl Model {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            arena: RefCell::new(ExprArena::new()),
            variables: RefCell::new(Vec::new()),
            var_names: RefCell::new(FxHashMap::default()),
            parameters: RefCell::new(Vec::new()),
            param_names: RefCell::new(FxHashMap::default()),
            constraints: RefCell::new(Vec::new()),
            constraint_names: RefCell::new(FxHashMap::default()),
            soc_constraints: RefCell::new(Vec::new()),
            soc_names: RefCell::new(FxHashMap::default()),
            objective: RefCell::new(None),
            objective_declared: Cell::new(false),
            cached_kind: Cell::new(None),
            auto_seq: Cell::new(0),
        }
    }

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
    pub(crate) fn register_var<'a>(&'a self, b: VarBuilder<'a>) -> Expr<'a> {
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

    /// Macro-facing entry point backing the indexed form of the `variable!`
    /// macro. Not part of the stable public API.
    #[doc(hidden)]
    pub fn __indexed_var<'a, K>(
        &'a self,
        name: impl Into<String>,
        set: &Set<K>,
    ) -> IndexedVarBuilder<'a, K> {
        IndexedVarBuilder {
            model: self,
            base_name: name.into(),
            keys: set.iter().collect(),
            axes: set.axes().map(Box::from),
            lb: f64::NEG_INFINITY,
            ub: f64::INFINITY,
            lb_by: None,
            ub_by: None,
            domain: Domain::Real,
            _k: PhantomData,
        }
    }

    pub fn variable_id(&self, name: &str) -> Option<VarId> {
        self.var_names.borrow().get(name).copied()
    }

    pub fn variables(&self) -> Ref<'_, Vec<Variable>> {
        self.variables.borrow()
    }

    pub fn arena(&self) -> Ref<'_, ExprArena> {
        self.arena.borrow()
    }

    pub fn num_variables(&self) -> usize {
        self.variables.borrow().len()
    }

    /// Render an [`EvalError`] using this model's registered variable/parameter
    /// name instead of the bare numeric id it carries.
    /// Use it when surfacing an evaluation failure to a user.
    #[must_use]
    pub fn describe_eval_error(&self, err: &EvalError) -> String {
        match err {
            EvalError::UnboundVar(v) => {
                let name = crate::var::var_name(&self.variables.borrow(), *v);
                format!("variable {name} has no value bound in the evaluation context")
            }
            EvalError::UnboundParam(p) => {
                let name = self.parameters.borrow().iter().find(|par| par.id == *p).map_or_else(
                    || format!("parameter #{}", p.index()),
                    |par| par.name.to_string(),
                );
                format!("parameter {name} has no value bound in the evaluation context")
            }
        }
    }

    /// Fix a single-variable expression to `value`.
    /// Convenience over [`Self::fix_var`] for handles from the `variable!` macro
    /// or [`crate::IndexedVar`] indexing.
    ///
    /// # Panics
    ///
    /// Panics if `e` is not a bare variable handle.
    pub fn fix(&self, e: Expr<'_>, value: f64) {
        let id = e.var_id().expect("Model::fix expects a single-variable expression");
        self.fix_var(id, value);
    }

    /// Fix variable `id` to `value` by setting `lb = ub = value`.
    pub fn fix_var(&self, id: VarId, value: f64) {
        let mut vars = self.variables.borrow_mut();
        let v = &mut vars[id.index()];
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
    /// Panics if `e` is not a bare variable handle.
    pub fn set_initial(&self, e: Expr<'_>, value: f64) {
        let id = e.var_id().expect("Model::set_initial expects a single-variable expression");
        self.variables.borrow_mut()[id.index()].initial = Some(value);
    }

    /// Restore bounds on variable `id`. Pass `f64::NEG_INFINITY` / `f64::INFINITY`
    /// to restore an unbounded direction.
    pub fn unfix_var(&self, id: VarId, lb: f64, ub: f64) {
        let mut vars = self.variables.borrow_mut();
        let v = &mut vars[id.index()];
        v.lb = lb;
        v.ub = ub;
        drop(vars);
        self.cached_kind.set(None);
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
    pub fn __param<'a>(&'a self, name: impl Into<SmolStr>, value: f64) -> Expr<'a> {
        self.register_param(name.into(), value)
    }

    /// Register one scalar parameter named `name` initialized to `value` and
    /// return its `Expr` handle. Shared by [`Self::__param`] and the indexed
    /// builder.
    ///
    /// # Panics
    ///
    /// Panics if a parameter with the same name is already registered.
    fn register_param(&self, name: SmolStr, value: f64) -> Expr<'_> {
        assert!(
            !self.param_names.borrow().contains_key(&name),
            "parameter name {name:?} is already registered on this model"
        );
        let (id, node) = {
            let mut a = self.arena.borrow_mut();
            let id = a.new_param(value);
            (id, a.param(id))
        };
        self.parameters.borrow_mut().push(Parameter { id, name: name.clone() });
        self.param_names.borrow_mut().insert(name, id);
        self.cached_kind.set(None);
        Expr::new(node, &self.arena)
    }

    /// Macro-facing entry point backing the indexed form of the `param!` macro
    /// (`param!(m, cost[i in items] = data[i])`). Registers one scalar parameter
    /// per key, evaluating `value` on the typed key, and returns an
    /// [`IndexedParam`]. Not part of the stable public API.
    ///
    /// # Panics
    ///
    /// Panics if a per-key parameter name collides with one already registered.
    #[doc(hidden)]
    pub fn __indexed_param<'a, K, F>(
        &'a self,
        name: impl Into<String>,
        set: &Set<K>,
        mut value: F,
    ) -> IndexedParam<'a, K>
    where
        K: FromIndexKey,
        F: FnMut(K) -> f64,
    {
        let base = name.into();
        let axes = set.axes().map(Box::from);
        let keys: Vec<IndexKey> = set.iter().collect();
        let make = |key: &IndexKey| -> Expr<'a> {
            let pname: SmolStr = format_index_name(&base, key).into();
            let v = value(K::from_index_key(key));
            self.register_param(pname, v)
        };
        let storage = build_storage(keys, axes, make);
        IndexedFamily { storage, _marker: PhantomData }
    }

    /// Re-bind the parameter at `key` of an indexed family to `value`. Takes
    /// effect on the next solve.
    ///
    /// # Panics
    ///
    /// Panics if `key` is not present in the family, or if `params` was built on
    /// a different `Model`.
    pub fn set_param_idx<K, Q: Into<IndexKey>>(
        &self,
        params: &IndexedParam<'_, K>,
        key: Q,
        value: f64,
    ) {
        let e = params.get(key).expect("set_param_idx: key not present in indexed parameter");
        assert!(
            std::ptr::eq(e.arena, std::ptr::from_ref(&self.arena)),
            "set_param_idx: indexed parameter belongs to a different model"
        );
        let id = e.param_id().expect("indexed parameter entry is not a parameter handle");
        self.set_param_id(id, value);
    }

    /// Current value bound to the parameter at `key` of an indexed family, or
    /// `None` if the key is absent.
    pub fn param_value_idx<K, Q: Into<IndexKey>>(
        &self,
        params: &IndexedParam<'_, K>,
        key: Q,
    ) -> Option<f64> {
        params.get(key).and_then(|e| self.param_value_of(e))
    }

    /// Re-bind the parameter referenced by handle `p` to `value`.
    ///
    /// # Panics
    ///
    /// Panics if `p` is not a bare parameter handle (one returned by the `param!`
    /// macro).
    pub fn set_param(&self, p: Expr<'_>, value: f64) {
        let id = p.param_id().expect("Model::set_param expects a single-parameter expression");
        self.set_param_id(id, value);
    }

    /// Re-bind parameter `id` to `value`. Takes effect on the next solve.
    ///
    /// The value is stored only in the expression arena (its single source of
    /// truth); extraction and evaluation read it from there.
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

    /// Current value of the parameter referenced by handle `p`, or `None` if
    /// `p` is not a bare parameter handle.
    pub fn param_value_of(&self, p: Expr<'_>) -> Option<f64> {
        p.param_id().map(|id| self.param_value(id))
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

    // Constraints

    /// Macro-facing entry point backing the `constraint!` macro. Not part of the
    /// stable public API.
    ///
    /// # Panics
    ///
    /// Panics if a constraint with the same name is already registered, or if
    /// the constraint count exceeds `u32::MAX`.
    #[doc(hidden)]
    pub fn __add_constraint(
        &self,
        name: impl Into<SmolStr>,
        c: ConstraintExpr<'_>,
    ) -> ConstraintId {
        let (lower, upper) = match c.sense {
            Sense::Le => (f64::NEG_INFINITY, c.rhs),
            Sense::Ge => (c.rhs, f64::INFINITY),
            Sense::Eq => (c.rhs, c.rhs),
        };
        self.register_constraint(name.into(), c.lhs.id, lower, upper)
    }

    /// Push a constraint row `lower <= lhs <= upper` into the registry. Shared by
    /// [`Self::__add_constraint`] and the range entry points.
    ///
    /// # Panics
    ///
    /// Panics if a constraint with the same name is already registered, if a
    /// bound is NaN, or if the constraint count exceeds `u32::MAX`.
    fn register_constraint(
        &self,
        name: SmolStr,
        lhs: ExprId,
        lower: f64,
        upper: f64,
    ) -> ConstraintId {
        assert!(
            !lower.is_nan() && !upper.is_nan(),
            "constraint {name:?} has NaN bound (lower={lower}, upper={upper})"
        );
        let mut by_name = self.constraint_names.borrow_mut();
        assert!(!by_name.contains_key(&name), "constraint name {name:?} already registered");
        let mut all = self.constraints.borrow_mut();
        let id = ConstraintId(u32::try_from(all.len()).expect("constraint count overflow"));
        all.push(Constraint { name: name.clone(), lhs, lower, upper, active: true });
        by_name.insert(name, id);
        self.cached_kind.set(None);
        id
    }

    /// A fresh unique auto-name `_c{n}`, skipping any a user already took.
    fn next_auto_name(&self) -> SmolStr {
        loop {
            let n = self.auto_seq.get();
            self.auto_seq.set(n + 1);
            let candidate: SmolStr = format!("_c{n}").into();
            if !self.constraint_names.borrow().contains_key(&candidate) {
                break candidate;
            }
        }
    }

    /// Register an anonymous constraint, deriving a unique name `_c{n}` from an
    /// internal counter. Backs the name-less form of the `constraint!` macro.
    #[doc(hidden)]
    pub fn __add_constraint_auto(&self, c: ConstraintExpr<'_>) -> ConstraintId {
        self.__add_constraint(self.next_auto_name(), c)
    }

    /// Bulk-register constraints. Each entry is `(name, ConstraintExpr)`.
    /// Useful with `.par_iter().map(...).collect()` style construction.
    pub fn add_constraints<'a, I>(&'a self, items: I)
    where
        I: IntoIterator<Item = (SmolStr, ConstraintExpr<'a>)>,
    {
        for (name, c) in items {
            self.__add_constraint(name, c);
        }
    }

    /// Macro-facing entry point backing the indexed-family form of the
    /// `constraint!` macro. The closure receives the index as a typed value `K`
    /// (any [`FromIndexKey`]: `i64`, `i32`, `usize`, `String`, raw `IndexKey`, or
    /// tuples up to arity 4). Not part of the stable public API.
    #[doc(hidden)]
    pub fn __add_constraints_over<'a, K, F>(&'a self, name_prefix: &str, set: &Set<K>, mut rule: F)
    where
        K: FromIndexKey,
        F: FnMut(K) -> ConstraintExpr<'a>,
    {
        for key in set {
            let typed = K::from_index_key(&key);
            let c = rule(typed);
            let name: SmolStr = format_index_name(name_prefix, &key).into();
            self.__add_constraint(name, c);
        }
    }

    /// Macro-facing entry point for a two-sided range `lo <= mid <= hi`.
    ///
    /// Collapses to a single interval [`Constraint`] named `name` only when both
    /// bounds are pure constants and the body is linear (the condition under which
    /// one two-sided row is representable).
    #[doc(hidden)]
    pub fn __add_range<'a, B1, B2>(&'a self, name: &str, mid: Expr<'a>, lo: B1, hi: B2)
    where
        B1: IntoRhs<'a>,
        B2: IntoRhs<'a>,
    {
        if let Some((lower, upper)) = self.collapse_bounds(mid.id, &lo, &hi) {
            self.register_constraint(name.into(), mid.id, lower, upper);
        } else {
            self.__add_constraint(format!("{name}_lo"), mid.ge(lo));
            self.__add_constraint(format!("{name}_hi"), mid.le(hi));
        }
    }

    /// Anonymous form of [`Self::__add_range`] (auto-named rows).
    #[doc(hidden)]
    pub fn __add_range_auto<'a, B1, B2>(&'a self, mid: Expr<'a>, lo: B1, hi: B2)
    where
        B1: IntoRhs<'a>,
        B2: IntoRhs<'a>,
    {
        if let Some((lower, upper)) = self.collapse_bounds(mid.id, &lo, &hi) {
            self.register_constraint(self.next_auto_name(), mid.id, lower, upper);
        } else {
            self.__add_constraint_auto(mid.ge(lo));
            self.__add_constraint_auto(mid.le(hi));
        }
    }

    /// The interval `(lower, upper)` a range collapses to, or `None` (keep two
    /// rows). Requires both bounds to be literal constants and the body `mid` to
    /// be linear.
    fn collapse_bounds<'a>(
        &self,
        mid: ExprId,
        lo: &impl IntoRhs<'a>,
        hi: &impl IntoRhs<'a>,
    ) -> Option<(f64, f64)> {
        let lower = lo.const_bound()?;
        let upper = hi.const_bound()?;
        (classify(&self.arena.borrow(), mid) == ExprClass::Linear).then_some((lower, upper))
    }

    /// Macro-facing entry point for a two-sided range family. One row per key,
    /// each collapsing to a single interval constraint when both bounds are
    /// constant (see [`Self::__add_range`]).
    #[doc(hidden)]
    pub fn __add_range_constraints_over<'a, K, B1, B2, F>(
        &'a self,
        name: &str,
        set: &Set<K>,
        mut rule: F,
    ) where
        K: FromIndexKey,
        B1: IntoRhs<'a>,
        B2: IntoRhs<'a>,
        F: FnMut(K) -> (Expr<'a>, B1, B2),
    {
        for key in set {
            let (mid, lo, hi) = rule(K::from_index_key(&key));
            let row_name = format_index_name(name, &key);
            self.__add_range(&row_name, mid, lo, hi);
        }
    }

    pub fn constraints(&self) -> Ref<'_, Vec<Constraint>> {
        self.constraints.borrow()
    }

    pub fn num_constraints(&self) -> usize {
        self.constraints.borrow().len()
    }

    pub fn constraint_id(&self, name: &str) -> Option<ConstraintId> {
        self.constraint_names.borrow().get(name).copied()
    }

    // Second-order cone constraints

    /// Register the explicit second-order cone constraint
    /// `||terms||_2 <= bound`.
    ///
    /// Every member of `terms` and the `bound` must be affine; the bound is
    /// additionally constrained to be nonnegative by the cone itself, so
    /// backends emit a `bound >= 0` side condition where needed.
    ///
    /// # Panics
    ///
    /// Panics if a SOC constraint with the same name is already registered, if
    /// `terms` is empty, if any term or the bound is not affine, or if the
    /// count exceeds `u32::MAX`.
    pub fn add_soc_constraint<'a>(
        &'a self,
        name: impl Into<SmolStr>,
        terms: impl IntoIterator<Item = Expr<'a>>,
        bound: Expr<'a>,
    ) -> SocConstraintId {
        let name = name.into();
        let arena = self.arena.borrow();
        let terms: Vec<ExprId> = terms
            .into_iter()
            .map(|e| {
                assert!(
                    classify(&arena, e.id) == ExprClass::Linear,
                    "SOC constraint {name:?} has a non-affine term"
                );
                e.id
            })
            .collect();
        assert!(!terms.is_empty(), "SOC constraint {name:?} has no terms");
        assert!(
            classify(&arena, bound.id) == ExprClass::Linear,
            "SOC constraint {name:?} has a non-affine bound"
        );
        drop(arena);

        let mut by_name = self.soc_names.borrow_mut();
        assert!(!by_name.contains_key(&name), "SOC constraint name {name:?} already registered");
        let mut all = self.soc_constraints.borrow_mut();
        let id = SocConstraintId(u32::try_from(all.len()).expect("SOC constraint count overflow"));
        all.push(SocConstraint { name: name.clone(), terms, bound: bound.id, active: true });
        by_name.insert(name, id);
        self.cached_kind.set(None);
        id
    }

    /// A fresh unique auto-name `_soc{n}` in the SOC namespace, skipping any a
    /// user already took. Shares `auto_seq` with [`Self::next_auto_name`]; the
    /// prefixes differ, so the two namespaces never collide.
    fn next_auto_soc_name(&self) -> SmolStr {
        loop {
            let n = self.auto_seq.get();
            self.auto_seq.set(n + 1);
            let candidate: SmolStr = format!("_soc{n}").into();
            if !self.soc_names.borrow().contains_key(&candidate) {
                break candidate;
            }
        }
    }

    /// Register an anonymous SOC constraint, deriving a unique name `_soc{n}`
    /// from an internal counter. Backs the name-less form of the
    /// `soc_constraint!` macro. Not part of the stable public API.
    #[doc(hidden)]
    pub fn __add_soc_constraint_auto<'a>(
        &'a self,
        terms: impl IntoIterator<Item = Expr<'a>>,
        bound: Expr<'a>,
    ) -> SocConstraintId {
        self.add_soc_constraint(self.next_auto_soc_name(), terms, bound)
    }

    /// Macro-facing entry point backing the indexed-family form of the
    /// `soc_constraint!` macro: one cone per key, named `{prefix}[{key}]`. The
    /// closure returns the cone's `(terms, bound)` pair for each typed key.
    /// Not part of the stable public API.
    #[doc(hidden)]
    pub fn __add_soc_constraints_over<'a, K, T, F>(
        &'a self,
        name_prefix: &str,
        set: &Set<K>,
        mut rule: F,
    ) where
        K: FromIndexKey,
        T: IntoIterator<Item = Expr<'a>>,
        F: FnMut(K) -> (T, Expr<'a>),
    {
        for key in set {
            let typed = K::from_index_key(&key);
            let (terms, bound) = rule(typed);
            let name: SmolStr = format_index_name(name_prefix, &key).into();
            self.add_soc_constraint(name, terms, bound);
        }
    }

    pub fn soc_constraints(&self) -> Ref<'_, Vec<SocConstraint>> {
        self.soc_constraints.borrow()
    }

    pub fn num_soc_constraints(&self) -> usize {
        self.soc_constraints.borrow().len()
    }

    pub fn soc_constraint_id(&self, name: &str) -> Option<SocConstraintId> {
        self.soc_names.borrow().get(name).copied()
    }

    /// Whether the model carries any explicit second-order cone constraints.
    pub fn has_cones(&self) -> bool {
        !self.soc_constraints.borrow().is_empty()
    }

    // Objective

    /// Macro-facing entry point backing `objective!(m, Min, ..)`. Not part of the
    /// stable public API.
    #[doc(hidden)]
    pub fn __minimize(&self, expr: Expr<'_>) {
        self.set_objective(expr, ObjectiveSense::Minimize);
    }

    /// Macro-facing entry point backing `objective!(m, Max, ..)`. Not part of the
    /// stable public API.
    #[doc(hidden)]
    pub fn __maximize(&self, expr: Expr<'_>) {
        self.set_objective(expr, ObjectiveSense::Maximize);
    }

    /// Macro-facing entry point backing `objective!(m, Feasibility)`. Declares
    /// the model a feasibility problem (no objective to optimize), clearing any
    /// previously set objective. Not part of the stable public API.
    #[doc(hidden)]
    pub fn __feasibility(&self) {
        *self.objective.borrow_mut() = None;
        self.objective_declared.set(true);
        self.cached_kind.set(None);
    }

    fn set_objective(&self, expr: Expr<'_>, sense: ObjectiveSense) {
        *self.objective.borrow_mut() = Some(Objective { expr: expr.id, sense });
        self.objective_declared.set(true);
        self.cached_kind.set(None);
    }

    /// Whether feasibility was declared explicitly via `objective!(m, Feasibility)`,
    /// as opposed to a model that simply has no objective set.
    pub fn is_feasibility(&self) -> bool {
        self.objective_declared.get() && self.objective.borrow().is_none()
    }

    /// Ensure the model has a solve direction declared: either an objective
    /// (`Min`/`Max`) or an explicit feasibility problem.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NoObjective`] if neither an objective nor
    /// `objective!(m, Feasibility)` was declared.
    pub fn ensure_objective_declared(&self) -> Result<()> {
        if self.objective_declared.get() { Ok(()) } else { Err(Error::NoObjective) }
    }

    pub fn objective(&self) -> Ref<'_, Option<Objective>> {
        self.objective.borrow()
    }

    /// Try to get a cloned copy of the objective.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NoObjective`] if no objective is set on this model.
    pub fn try_objective(&self) -> Result<Objective> {
        self.objective.borrow().clone().ok_or(Error::NoObjective)
    }

    // Classification

    /// Infer the [`ModelKind`] from current variables and expressions.
    /// Result is cached and invalidated whenever variables, constraints, or the
    /// objective change.
    ///
    /// The decision ladder, top-down (any integer variable picks the `MI*`
    /// column):
    ///
    /// 1. any nonlinear expression (objective or constraint) -> `NLP`
    /// 2. any quadratic constraint not recognized as SOC (see
    ///    [`crate::detect_soc`]) -> `QCP`
    /// 3. cones present (explicit or detected) -> `SOCP`
    /// 4. quadratic objective -> `QP`
    /// 5. otherwise -> `LP`
    pub fn kind(&self) -> ModelKind {
        if let Some(k) = self.cached_kind.get() {
            return k;
        }
        let arena = self.arena.borrow();
        let vars = self.variables.borrow();
        let has_int = vars.iter().any(|v| v.domain.is_integer());
        let obj_class = self
            .objective
            .borrow()
            .as_ref()
            .map_or(ExprClass::Linear, |o| classify(&arena, o.expr));

        let mut any_nonlinear = obj_class == ExprClass::Nonlinear;
        // A quadratic constraint that is not SOC-shaped.
        let mut plain_quad_con = false;
        let mut detected_soc = false;
        if !any_nonlinear {
            for c in self.constraints.borrow().iter() {
                match classify(&arena, c.lhs) {
                    ExprClass::Linear => {}
                    ExprClass::Quadratic => {
                        if detect_soc(&arena, &vars, c).is_some() {
                            detected_soc = true;
                        } else {
                            plain_quad_con = true;
                        }
                    }
                    ExprClass::Nonlinear => {
                        any_nonlinear = true;
                        break;
                    }
                }
            }
        }
        let has_soc = detected_soc || !self.soc_constraints.borrow().is_empty();

        let pick = |cont, int| if has_int { int } else { cont };
        let k = if any_nonlinear {
            pick(ModelKind::NLP, ModelKind::MINLP)
        } else if plain_quad_con {
            pick(ModelKind::QCP, ModelKind::MIQCP)
        } else if has_soc {
            pick(ModelKind::SOCP, ModelKind::MISOCP)
        } else if obj_class == ExprClass::Quadratic {
            pick(ModelKind::QP, ModelKind::MIQP)
        } else {
            pick(ModelKind::LP, ModelKind::MILP)
        };
        self.cached_kind.set(Some(k));
        k
    }
}

// IndexedVarBuilder

/// Builder for a collection of scalar variables indexed by a [`Set`].
///
/// For example, `flow[i]` for `i in 0..3` registers `flow[0]`, `flow[1]`, and
/// `flow[2]` as separate scalar variables in the model. Call `.build()` to get
/// an [`IndexedVar`] that maps each key to its [`Expr`] handle. Bounds and
/// domain set here apply uniformly to every scalar in the collection.
type BoundFn<'a> = Box<dyn Fn(&IndexKey) -> f64 + 'a>;

#[must_use = "IndexedVarBuilder does nothing until you call .build()"]
pub struct IndexedVarBuilder<'a, K = IndexKey> {
    model: &'a Model,
    base_name: String,
    keys: Vec<IndexKey>,
    axes: Option<Box<[Axis]>>,
    lb: f64,
    ub: f64,
    lb_by: Option<BoundFn<'a>>,
    ub_by: Option<BoundFn<'a>>,
    domain: Domain,
    _k: PhantomData<fn() -> K>,
}

impl<'a, K> std::fmt::Debug for IndexedVarBuilder<'a, K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndexedVarBuilder")
            .field("base_name", &self.base_name)
            .field("keys", &self.keys.len())
            .field("lb", &self.lb)
            .field("ub", &self.ub)
            .field("per_key_lb", &self.lb_by.is_some())
            .field("per_key_ub", &self.ub_by.is_some())
            .field("domain", &self.domain)
            .finish()
    }
}

impl<'a, K> IndexedVarBuilder<'a, K> {
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
    /// Per-key lower bound. Overrides [`Self::lb`] when both are set.
    ///
    /// The closure receives a typed index value via [`FromIndexKey`].
    /// Annotate the argument to select the projection:
    /// ```ignore
    /// .lb_by(|(p, q): (String, String)| floor_for(&p, &q))
    /// .lb_by(|i: usize| lower_bounds[i])
    /// ```
    pub fn lb_by<F>(mut self, f: F) -> Self
    where
        K: FromIndexKey,
        F: Fn(K) -> f64 + 'a,
    {
        self.lb_by = Some(Box::new(move |k: &IndexKey| f(K::from_index_key(k))));
        self
    }
    /// Per-key upper bound. Overrides [`Self::ub`] when both are set.
    ///
    /// The closure receives a typed index value via [`FromIndexKey`]; annotate
    /// the argument to select the projection:
    /// ```ignore
    /// .ub_by(|(p, q): (String, String)| capacity_for(&p, &q))
    /// .ub_by(|i: usize| upper_bounds[i])
    /// ```
    pub fn ub_by<F>(mut self, f: F) -> Self
    where
        K: FromIndexKey,
        F: Fn(K) -> f64 + 'a,
    {
        self.ub_by = Some(Box::new(move |k: &IndexKey| f(K::from_index_key(k))));
        self
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

    /// Register one scalar variable per key and return the [`IndexedVar`] handle.
    ///
    /// # Panics
    /// Panics if a scalar variable name collides with one already registered.
    pub fn build(self) -> IndexedVar<'a, K> {
        let Self { model, base_name, keys, axes, lb, ub, lb_by, ub_by, domain, _k } = self;

        let make = |key: &IndexKey| -> Expr<'a> {
            let scalar_name: SmolStr = format_index_name(&base_name, key).into();
            let lo = lb_by.as_ref().map_or(lb, |f| f(key));
            let hi = ub_by.as_ref().map_or(ub, |f| f(key));
            model.__var(scalar_name).lb(lo).ub(hi).domain(domain).build()
        };

        let storage = build_storage(keys, axes, make);
        IndexedFamily { storage, _marker: PhantomData }
    }
}

fn format_index_name(base: &str, key: &IndexKey) -> String {
    let mut out = String::with_capacity(base.len() + 4);
    out.push_str(base);
    out.push('[');
    write_key_parts(&mut out, key);
    out.push(']');
    out
}

fn write_key_parts(out: &mut String, key: &IndexKey) {
    use std::fmt::Write;
    match key {
        IndexKey::Int(i) => write!(out, "{i}").unwrap(),
        IndexKey::Str(s) => out.push_str(s),
        IndexKey::Tuple(parts) => {
            for (i, p) in parts.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_key_parts(out, p);
            }
        }
    }
}

/// Public render of an `IndexKey`'s textual form, used when deriving
/// auto-generated names for indexed-family constraints.
pub fn display_index_key(key: &IndexKey) -> String {
    let mut out = String::new();
    write_key_parts(&mut out, key);
    out
}

#[cfg(test)]
mod tests {
    use oximo_expr::extract_linear;

    use super::*;
    use crate::Set;
    use crate::constraint::Relate;

    #[test]
    fn param_times_var_keeps_model_linear() {
        let m = Model::new("p");
        let param = m.__param("param", 4.0);
        let x = m.__var("x").lb(0.0).build();
        m.__minimize(param * x);
        assert_eq!(m.kind(), ModelKind::LP);
    }

    #[test]
    fn param_coeff_resolves_and_rebinds() {
        let m = Model::new("p");
        let param = m.__param("param", 4.0);
        let x = m.__var("x").lb(0.0).build();
        let obj = param * x;

        let coeff = |m: &Model| {
            let arena = m.arena();
            extract_linear(&arena, obj.id).expect("linear").coeffs[0].1
        };
        assert!((coeff(&m) - 4.0).abs() < f64::EPSILON);

        m.set_param(param, 9.0);
        assert!((coeff(&m) - 9.0).abs() < f64::EPSILON);
        assert_eq!(m.parameter_id("param"), Some(param.param_id().unwrap()));
    }

    #[test]
    fn param_value_reads_live_arena_value() {
        let m = Model::new("p");
        let param = m.__param("param", 4.0);
        let id = param.param_id().unwrap();
        assert!((m.param_value(id) - 4.0).abs() < f64::EPSILON);
        assert!((m.param_value_of(param).unwrap() - 4.0).abs() < f64::EPSILON);

        m.set_param(param, 7.5);
        assert!((m.param_value(id) - 7.5).abs() < f64::EPSILON);

        let x = m.__var("x").build();
        assert!(m.param_value_of(x).is_none());
    }

    #[test]
    fn set_param_invalidates_kind_cache() {
        let m = Model::new("p");
        let p = m.__param("p", 1.0);
        let x = m.__var("x").lb(0.0).build();
        m.__add_constraint("c", (p * x).le(10.0));
        assert_eq!(m.kind(), ModelKind::LP);
        m.set_param(p, 2.0);
        assert_eq!(m.kind(), ModelKind::LP);
    }

    #[test]
    #[should_panic(expected = "parameter name \"dup\" is already registered")]
    fn duplicate_param_name_panics() {
        let m = Model::new("p");
        let _a = m.__param("dup", 1.0);
        let _b = m.__param("dup", 2.0);
    }

    #[test]
    fn indexed_param_dense_value_and_per_key_rebind() {
        let m = Model::new("ip");
        let items = Set::range(0..3);
        let data = [10.0, 20.0, 30.0];
        let cost = m.__indexed_param("cost", &items, |i: usize| data[i]);

        assert!(cost.is_dense());
        assert_eq!(cost.len(), 3);
        assert_eq!(m.num_parameters(), 3);
        assert!(m.parameter_id("cost[0]").is_some());
        assert!(m.parameter_id("cost[2]").is_some());
        assert!((m.param_value_idx(&cost, 1usize).unwrap() - 20.0).abs() < f64::EPSILON);

        let x = m.__var("x").lb(0.0).build();
        let obj = cost.at([1]) * x;
        let coeff = |m: &Model| {
            let arena = m.arena();
            extract_linear(&arena, obj.id).expect("linear").coeffs[0].1
        };
        assert!((coeff(&m) - 20.0).abs() < f64::EPSILON);

        m.set_param_idx(&cost, 1usize, 99.0);
        assert!((coeff(&m) - 99.0).abs() < f64::EPSILON);
        assert!((m.param_value_idx(&cost, 1usize).unwrap() - 99.0).abs() < f64::EPSILON);
        assert!((m.param_value_idx(&cost, 0usize).unwrap() - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    #[should_panic(expected = "different model")]
    fn set_param_idx_rejects_foreign_family() {
        let a = Model::new("a");
        let b = Model::new("b");
        let items = Set::range(0..2);
        let pa = a.__indexed_param("p", &items, |_i: usize| 1.0);
        b.set_param_idx(&pa, 0usize, 5.0);
    }

    #[test]
    fn indexed_param_sparse_string_keyed() {
        let m = Model::new("ips");
        let plants = Set::strings(["a", "b"]);
        let price =
            m.__indexed_param("price", &plants, |p: String| if p == "a" { 1.5 } else { 2.5 });
        assert!(!price.is_dense());
        assert_eq!(price.len(), 2);
        assert!((m.param_value_idx(&price, "a").unwrap() - 1.5).abs() < f64::EPSILON);
        assert!((m.param_value_idx(&price, "b").unwrap() - 2.5).abs() < f64::EPSILON);
        assert!(m.param_value_idx(&price, "z").is_none());
    }
}
