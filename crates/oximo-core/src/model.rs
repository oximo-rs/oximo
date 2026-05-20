use std::cell::{Ref, RefCell};

use oximo_expr::{Expr, ExprArena, VarId};
use rustc_hash::FxHashMap;
use smol_str::SmolStr;

use crate::constraint::{Constraint, ConstraintExpr, ConstraintId};
use crate::domain::Domain;
use crate::error::{Error, Result};
use crate::indexed::IndexedVar;
use crate::objective::{Objective, ObjectiveSense};
use crate::set::{FromIndexKey, IndexKey, Set};
use crate::var::{VarBuilder, Variable};

/// The kind of mathematical program a `Model` represents.
///
/// This is inferred from the variables and expressions in the model, not set
/// explicitly by the user. See [`Model::kind`] for details.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ModelKind {
    LP,
    MILP,
    QP,
    MIQP,
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
    pub(crate) constraints: RefCell<Vec<Constraint>>,
    pub(crate) constraint_names: RefCell<FxHashMap<SmolStr, ConstraintId>>,
    pub(crate) objective: RefCell<Option<Objective>>,
    cached_kind: RefCell<Option<ModelKind>>,
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model")
            .field("name", &self.name)
            .field("vars", &self.variables.borrow().len())
            .field("constraints", &self.constraints.borrow().len())
            .field("has_objective", &self.objective.borrow().is_some())
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
            constraints: RefCell::new(Vec::new()),
            constraint_names: RefCell::new(FxHashMap::default()),
            objective: RefCell::new(None),
            cached_kind: RefCell::new(None),
        }
    }

    // Variables

    pub fn var(&self, name: impl Into<SmolStr>) -> VarBuilder<'_> {
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
        assert!(!names.contains_key(&b.name), "variable name {:?} already registered", b.name);
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
        *self.cached_kind.borrow_mut() = None;
        Expr::from_var(&self.arena, id)
    }

    pub fn indexed_var<'a>(&'a self, name: impl Into<String>, set: &Set) -> IndexedVarBuilder<'a> {
        IndexedVarBuilder {
            model: self,
            base_name: name.into(),
            keys: set.iter().collect(),
            lb: f64::NEG_INFINITY,
            ub: f64::INFINITY,
            lb_by: None,
            ub_by: None,
            domain: Domain::Real,
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

    /// Fix variable `id` to `value` by setting `lb = ub = value`.
    pub fn fix_var(&self, id: VarId, value: f64) {
        let mut vars = self.variables.borrow_mut();
        let v = &mut vars[id.index()];
        v.lb = value;
        v.ub = value;
    }

    /// Restore bounds on variable `id`. Pass `f64::NEG_INFINITY` / `f64::INFINITY`
    /// to restore an unbounded direction.
    pub fn unfix_var(&self, id: VarId, lb: f64, ub: f64) {
        let mut vars = self.variables.borrow_mut();
        let v = &mut vars[id.index()];
        v.lb = lb;
        v.ub = ub;
    }

    // Constraints

    /// Register a new constraint.
    ///
    /// # Panics
    ///
    /// Panics if a constraint with the same name is already registered, or if
    /// the constraint count exceeds `u32::MAX`.
    pub fn constraint(&self, name: impl Into<SmolStr>, c: ConstraintExpr<'_>) -> ConstraintId {
        let name = name.into();
        let mut by_name = self.constraint_names.borrow_mut();
        assert!(!by_name.contains_key(&name), "constraint name {name:?} already registered");
        let mut all = self.constraints.borrow_mut();
        let id = ConstraintId(u32::try_from(all.len()).expect("constraint count overflow"));
        all.push(Constraint {
            name: name.clone(),
            lhs: c.lhs.id,
            sense: c.sense,
            rhs: c.rhs,
            active: true,
        });
        by_name.insert(name, id);
        *self.cached_kind.borrow_mut() = None;
        id
    }

    /// Bulk-register constraints. Each entry is `(name, ConstraintExpr)`.
    /// Useful with `.par_iter().map(...).collect()` style construction.
    pub fn add_constraints<'a, I>(&'a self, items: I)
    where
        I: IntoIterator<Item = (SmolStr, ConstraintExpr<'a>)>,
    {
        for (name, c) in items {
            self.constraint(name, c);
        }
    }

    /// Rule-style bulk constraint registration.
    ///
    /// The closure receives the index as a typed value `K`. Any type
    /// implementing [`FromIndexKey`] is accepted. Built-in impls cover `i64`,
    /// `i32`, `usize`, `String`, raw `IndexKey`, and tuples up to arity 4.
    /// The user states the expected shape via the closure-arg annotation.
    ///
    /// # Example
    /// ```ignore
    /// // Scalar set: closure receives a usize directly.
    /// m.add_constraints_over("upper", &i, |i: usize| x[i].le(b[i]));
    ///
    /// // Tuple set: destructure inline.
    /// m.add_constraints_over("blo", &(&tasks * &events), |(t, n): (usize, usize)| {
    ///     (b[(t, n)] - b_min[t] * w[(t, n)]).ge(0.0)
    /// });
    /// ```
    pub fn add_constraints_over<'a, K, F>(&'a self, name_prefix: &str, set: &Set, mut rule: F)
    where
        K: FromIndexKey,
        F: FnMut(K) -> ConstraintExpr<'a>,
    {
        for key in set {
            let typed = K::from_index_key(&key);
            let c = rule(typed);
            let name: SmolStr = format_index_name(name_prefix, &key).into();
            self.constraint(name, c);
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

    // Objective

    pub fn minimize(&self, expr: Expr<'_>) {
        self.set_objective(expr, ObjectiveSense::Minimize);
    }

    pub fn maximize(&self, expr: Expr<'_>) {
        self.set_objective(expr, ObjectiveSense::Maximize);
    }

    fn set_objective(&self, expr: Expr<'_>, sense: ObjectiveSense) {
        *self.objective.borrow_mut() = Some(Objective { expr: expr.id, sense });
        *self.cached_kind.borrow_mut() = None;
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
    pub fn kind(&self) -> ModelKind {
        if let Some(k) = *self.cached_kind.borrow() {
            return k;
        }
        let arena = self.arena.borrow();
        let has_int = self.variables.borrow().iter().any(|v| v.domain.is_integer());
        let nonlinear = self.constraints.borrow().iter().any(|c| has_nonlinear(&arena, c.lhs))
            || self.objective.borrow().as_ref().is_some_and(|o| has_nonlinear(&arena, o.expr));
        let k = match (has_int, nonlinear) {
            (false, false) => ModelKind::LP,
            (true, false) => ModelKind::MILP,
            (false, true) => ModelKind::NLP,
            (true, true) => ModelKind::MINLP,
        };
        *self.cached_kind.borrow_mut() = Some(k);
        k
    }
}

fn has_nonlinear(arena: &ExprArena, id: oximo_expr::ExprId) -> bool {
    use oximo_expr::ExprNode as N;
    match arena.get(id) {
        N::Const(_) | N::Var(_) | N::Param(_) | N::Linear { .. } => false,
        N::Neg(inner) => has_nonlinear(arena, *inner),
        N::Add(children) => children.iter().any(|c| has_nonlinear(arena, *c)),
        N::Mul(children) => {
            let mut nonconst = 0;
            for c in children {
                if !matches!(arena.get(*c), N::Const(_)) {
                    nonconst += 1;
                }
                if has_nonlinear(arena, *c) {
                    return true;
                }
            }
            nonconst >= 2
        }
        N::Pow(_, _) | N::Sin(_) | N::Cos(_) | N::Exp(_) | N::Log(_) => true,
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
pub struct IndexedVarBuilder<'a> {
    model: &'a Model,
    base_name: String,
    keys: Vec<IndexKey>,
    lb: f64,
    ub: f64,
    lb_by: Option<BoundFn<'a>>,
    ub_by: Option<BoundFn<'a>>,
    domain: Domain,
}

impl<'a> std::fmt::Debug for IndexedVarBuilder<'a> {
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

impl<'a> IndexedVarBuilder<'a> {
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
    pub fn lb_by<K, F>(mut self, f: F) -> Self
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
    pub fn ub_by<K, F>(mut self, f: F) -> Self
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

    pub fn build(self) -> IndexedVar<'a> {
        let mut entries = FxHashMap::default();
        for key in self.keys {
            let scalar_name: SmolStr = format_index_name(&self.base_name, &key).into();
            let lb = self.lb_by.as_ref().map_or(self.lb, |f| f(&key));
            let ub = self.ub_by.as_ref().map_or(self.ub, |f| f(&key));
            let expr = self.model.var(scalar_name).lb(lb).ub(ub).domain(self.domain).build();
            entries.insert(key, expr);
        }
        IndexedVar { entries }
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

/// Public render of an `IndexKey`'s textual form, used by helpers like
/// [`Model::add_constraints_over`] to derive constraint names.
pub fn display_index_key(key: &IndexKey) -> String {
    let mut out = String::new();
    write_key_parts(&mut out, key);
    out
}
