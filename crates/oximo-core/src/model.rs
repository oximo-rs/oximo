use std::cell::{Cell, Ref, RefCell};
use std::fmt;
use std::marker::PhantomData;

use oximo_expr::{
    EvalError, Expr, ExprArena, ExprArenaCell, ExprArenaSnapshot, ExprClass, ExprId, ExprIdRemap,
    ParamId, VarId, classify,
};
use rayon::prelude::*;
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};
use smol_str::SmolStr;

use crate::constraint::{
    Constraint, ConstraintExpr, ConstraintId, IntoRhs, RangeConstraintIds, Relate, Sense,
};
use crate::domain::Domain;
use crate::error::{Error, Result};
use crate::indexed::{
    IndexedConstraint, IndexedFamily, IndexedParam, IndexedRangeConstraint, IndexedVar,
    build_storage,
};
use crate::objective::{Objective, ObjectiveSense};
use crate::param::Parameter;
use crate::reformulation::SosReformulationArtifacts;
use crate::set::{Axis, FromIndexKey, IndexKey, Set};
use crate::soc::{SocConstraint, SocConstraintId, is_detected_soc};
use crate::sos::{
    SosConstraint, SosConstraintHandle, SosConstraintId, SosMember, SosType, validate_members,
};
use crate::var::{VarBuilder, Variable};

const PAR_KIND_THRESHOLD: usize = 256;
const PAR_INDEXED_METADATA_THRESHOLD: usize = 1_024;
const PAR_INDEXED_ALGEBRAIC_THRESHOLD: usize = 512;
const PAR_INDEXED_RANGE_THRESHOLD: usize = 512;
const PAR_INDEXED_SOC_THRESHOLD: usize = 256;
const PAR_INDEXED_SOS_THRESHOLD: usize = 512;

fn indexed_parallel(len: usize, forced: Option<bool>, threshold: usize) -> bool {
    forced.unwrap_or(len >= threshold && rayon::current_num_threads() > 1)
}

fn indexed_chunk_size(len: usize) -> usize {
    let chunks = rayon::current_num_threads().saturating_mul(4).max(1);
    len.div_ceil(chunks).max(1)
}

fn arena_key(arena: &ExprArenaCell) -> usize {
    std::ptr::from_ref(arena) as usize
}

fn assert_expr_arena(expr: Expr<'_>, expected: usize) {
    assert_eq!(arena_key(expr.arena), expected, "expression belongs to a different model");
}

fn validate_batch_names<'a, V>(
    existing: &FxHashMap<SmolStr, V>,
    names: impl IntoIterator<Item = &'a SmolStr>,
    kind: &str,
    count: usize,
) {
    let mut batch = FxHashSet::with_capacity_and_hasher(count, FxBuildHasher);
    for name in names {
        assert!(!existing.contains_key(name), "{kind} name {name:?} already registered");
        assert!(batch.insert(name), "{kind} name {name:?} occurs more than once");
    }
}

#[derive(Debug)]
struct PendingVar {
    name: SmolStr,
    lb: f64,
    ub: f64,
}

#[derive(Debug)]
struct PendingParam {
    name: SmolStr,
    value: f64,
}

#[derive(Debug)]
struct PendingConstraint {
    name: SmolStr,
    lhs: ExprId,
    lower: f64,
    upper: f64,
}

impl PendingConstraint {
    fn from_expr(name: SmolStr, constraint: ConstraintExpr<'_>) -> Self {
        let (lower, upper) = match constraint.sense {
            Sense::Le => (f64::NEG_INFINITY, constraint.rhs),
            Sense::Ge => (constraint.rhs, f64::INFINITY),
            Sense::Eq => (constraint.rhs, constraint.rhs),
        };
        assert!(
            !lower.is_nan() && !upper.is_nan(),
            "constraint {name:?} has NaN bound (lower={lower}, upper={upper})"
        );
        Self { name, lhs: constraint.lhs.id, lower, upper }
    }

    fn remap(&mut self, remap: ExprIdRemap) {
        self.lhs = remap.apply(self.lhs);
    }
}

#[derive(Debug)]
struct PendingRangeBatch {
    rows: Vec<PendingConstraint>,
    row_counts: Vec<u8>,
}

fn prepare_range<'a, B1: IntoRhs<'a>, B2: IntoRhs<'a>>(
    name: String,
    mid: Expr<'a>,
    lo: B1,
    hi: B2,
) -> (PendingConstraint, Option<PendingConstraint>) {
    if let (Some(lower), Some(upper)) = (lo.const_bound(), hi.const_bound())
        && mid.__class() == ExprClass::Linear
    {
        assert!(
            !lower.is_nan() && !upper.is_nan(),
            "constraint {name:?} has NaN bound (lower={lower}, upper={upper})"
        );
        (PendingConstraint { name: name.into(), lhs: mid.id, lower, upper }, None)
    } else {
        (
            PendingConstraint::from_expr(format!("{name}_lo").into(), mid.ge(lo)),
            Some(PendingConstraint::from_expr(format!("{name}_hi").into(), mid.le(hi))),
        )
    }
}

#[derive(Debug)]
struct PendingSoc {
    name: SmolStr,
    terms: Vec<ExprId>,
    bound: ExprId,
}

impl PendingSoc {
    fn remap(&mut self, remap: ExprIdRemap) {
        for term in &mut self.terms {
            *term = remap.apply(*term);
        }
        self.bound = remap.apply(self.bound);
    }
}

fn prepare_soc<'a>(
    name: SmolStr,
    terms: impl IntoIterator<Item = Expr<'a>>,
    bound: Expr<'a>,
) -> PendingSoc {
    let terms: Vec<ExprId> = terms
        .into_iter()
        .map(|term| {
            assert!(
                term.__class() == ExprClass::Linear,
                "SOC constraint {name:?} has a non-affine term"
            );
            term.id
        })
        .collect();
    assert!(!terms.is_empty(), "SOC constraint {name:?} has no terms");
    assert!(bound.__class() == ExprClass::Linear, "SOC constraint {name:?} has a non-affine bound");
    PendingSoc { name, terms, bound: bound.id }
}

#[derive(Debug)]
struct PendingSos {
    name: SmolStr,
    members: Vec<SosMember>,
}

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
///   by the model's structural cone predicate). The objective may be linear or quadratic:
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

impl fmt::Display for ModelKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::LP => "LP",
            Self::MILP => "MILP",
            Self::QP => "QP",
            Self::MIQP => "MIQP",
            Self::QCP => "QCP",
            Self::MIQCP => "MIQCP",
            Self::SOCP => "SOCP",
            Self::MISOCP => "MISOCP",
            Self::NLP => "NLP",
            Self::MINLP => "MINLP",
        })
    }
}

/// A borrowed constraint from a [`Model`].
///
/// Algebraic, explicitly declared second-order-cone, and SOS constraints
/// retain their typed IDs and storage. This enum provides a unified inspection
/// boundary without changing either representation.
#[derive(Copy, Clone, Debug)]
pub enum ConstraintRef<'a> {
    Algebraic { id: ConstraintId, constraint: &'a Constraint },
    SecondOrderCone { id: SocConstraintId, constraint: &'a SocConstraint },
    SpecialOrderedSet { id: SosConstraintId, constraint: &'a SosConstraint },
}

/// Unified borrowed view of every constraint declared on a [`Model`].
///
/// The underlying algebraic and explicit-SOC registries remain separate, so
/// backends can iterate a homogeneous slice without a per-constraint branch.
/// [`Self::iter`] visits algebraic constraints in [`ConstraintId`] order,
/// followed by explicit cones and SOS constraints in their respective ID order.
#[derive(Debug)]
pub struct ModelConstraints<'a> {
    algebraic: Ref<'a, Vec<Constraint>>,
    second_order_cones: Ref<'a, Vec<SocConstraint>>,
    special_ordered_sets: Ref<'a, Vec<SosConstraint>>,
}

impl ModelConstraints<'_> {
    /// Algebraic constraints in [`ConstraintId`] order.
    pub fn algebraic(&self) -> &[Constraint] {
        &self.algebraic
    }

    /// Explicit second-order-cone constraints in [`SocConstraintId`] order.
    pub fn second_order_cones(&self) -> &[SocConstraint] {
        &self.second_order_cones
    }

    pub fn special_ordered_sets(&self) -> &[SosConstraint] {
        &self.special_ordered_sets
    }

    /// Iterate over all declared constraints without allocating.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "registration rejects constraint counts above u32::MAX"
    )]
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = ConstraintRef<'_>> + Clone {
        let algebraic = self.algebraic.iter().enumerate().map(|(index, constraint)| {
            ConstraintRef::Algebraic { id: ConstraintId(index as u32), constraint }
        });
        let second_order_cones =
            self.second_order_cones.iter().enumerate().map(|(index, constraint)| {
                ConstraintRef::SecondOrderCone { id: SocConstraintId(index as u32), constraint }
            });
        let special_ordered_sets =
            self.special_ordered_sets.iter().enumerate().map(|(index, constraint)| {
                ConstraintRef::SpecialOrderedSet { id: SosConstraintId(index as u32), constraint }
            });
        algebraic.chain(second_order_cones).chain(special_ordered_sets)
    }

    /// Total number of algebraic, SOC, and SOS constraints.
    pub fn len(&self) -> usize {
        self.algebraic.len() + self.second_order_cones.len() + self.special_ordered_sets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.algebraic.is_empty()
            && self.second_order_cones.is_empty()
            && self.special_ordered_sets.is_empty()
    }
}

/// The optimization model. Owns the expression arena, variable/parameter
/// registries, constraints, and (optional) objective.
///
/// `Model` uses interior mutability so the builder API can take `&self`
/// references.
///
/// Registries use `RefCell`s. The expression arena uses synchronized interior
/// mutability and isolated worker forks while large indexed families are
/// prepared in parallel.
pub struct Model {
    pub name: SmolStr,
    pub(crate) arena: ExprArenaCell,
    pub(crate) variables: RefCell<Vec<Variable>>,
    pub(crate) var_names: RefCell<FxHashMap<SmolStr, VarId>>,
    pub(crate) parameters: RefCell<Vec<Parameter>>,
    pub(crate) param_names: RefCell<FxHashMap<SmolStr, ParamId>>,
    pub(crate) constraints: RefCell<Vec<Constraint>>,
    pub(crate) constraint_names: RefCell<FxHashMap<SmolStr, ConstraintId>>,
    pub(crate) soc_constraints: RefCell<Vec<SocConstraint>>,
    pub(crate) soc_names: RefCell<FxHashMap<SmolStr, SocConstraintId>>,
    pub(crate) sos_constraints: RefCell<Vec<SosConstraint>>,
    pub(crate) sos_names: RefCell<FxHashMap<SmolStr, SosConstraintId>>,
    pub(crate) sos_reformulations: RefCell<Vec<SosReformulationArtifacts>>,
    pub(crate) objective: RefCell<Option<Objective>>,
    objective_declared: Cell<bool>,
    cached_kind: Cell<Option<ModelKind>>,
    /// Monotonic counter for auto-naming anonymous constraints registered via
    /// the `constraint!` macro.
    auto_seq: Cell<u32>,
}

impl Model {
    fn assert_expr_belongs(&self, expr: Expr<'_>) {
        assert_expr_arena(expr, arena_key(&self.arena));
    }

    /// Deep-copy every registry while preserving stable IDs.
    pub(crate) fn clone_preserving_ids_with_capacity(
        &self,
        additional_variables: usize,
        additional_constraints: usize,
        additional_expr_nodes: usize,
    ) -> Self {
        let variables = self.variables.borrow();
        let mut cloned_variables =
            Vec::with_capacity(variables.len().saturating_add(additional_variables));
        cloned_variables.extend_from_slice(&variables);

        let var_names = self.var_names.borrow();
        let mut cloned_var_names = FxHashMap::with_capacity_and_hasher(
            var_names.len().saturating_add(additional_variables),
            FxBuildHasher,
        );
        cloned_var_names.extend(var_names.iter().map(|(name, id)| (name.clone(), *id)));

        let constraints = self.constraints.borrow();
        let mut cloned_constraints =
            Vec::with_capacity(constraints.len().saturating_add(additional_constraints));
        cloned_constraints.extend_from_slice(&constraints);

        let constraint_names = self.constraint_names.borrow();
        let mut cloned_constraint_names = FxHashMap::with_capacity_and_hasher(
            constraint_names.len().saturating_add(additional_constraints),
            FxBuildHasher,
        );
        cloned_constraint_names
            .extend(constraint_names.iter().map(|(name, id)| (name.clone(), *id)));

        Self {
            name: self.name.clone(),
            arena: ExprArenaCell::new(
                self.arena.borrow().__clone_with_additional_capacity(additional_expr_nodes),
            ),
            variables: RefCell::new(cloned_variables),
            var_names: RefCell::new(cloned_var_names),
            parameters: RefCell::new(self.parameters.borrow().clone()),
            param_names: RefCell::new(self.param_names.borrow().clone()),
            constraints: RefCell::new(cloned_constraints),
            constraint_names: RefCell::new(cloned_constraint_names),
            soc_constraints: RefCell::new(self.soc_constraints.borrow().clone()),
            soc_names: RefCell::new(self.soc_names.borrow().clone()),
            sos_constraints: RefCell::new(self.sos_constraints.borrow().clone()),
            sos_names: RefCell::new(self.sos_names.borrow().clone()),
            sos_reformulations: RefCell::new(self.sos_reformulations.borrow().clone()),
            objective: RefCell::new(self.objective.borrow().clone()),
            objective_declared: Cell::new(self.objective_declared.get()),
            cached_kind: Cell::new(self.cached_kind.get()),
            auto_seq: Cell::new(self.auto_seq.get()),
        }
    }
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model")
            .field("name", &self.name)
            .field("vars", &self.variables.borrow().len())
            .field("params", &self.parameters.borrow().len())
            .field("constraints", &self.constraints.borrow().len())
            .field("soc_constraints", &self.soc_constraints.borrow().len())
            .field("sos_constraints", &self.sos_constraints.borrow().len())
            .field("has_objective", &self.objective.borrow().is_some())
            .field("feasibility", &self.is_feasibility())
            .finish()
    }
}

impl Model {
    pub(crate) fn invalidate_kind(&self) {
        self.cached_kind.set(None);
    }

    pub fn new(name: impl Into<SmolStr>) -> Self {
        Self {
            name: name.into(),
            arena: ExprArenaCell::new(ExprArena::new()),
            variables: RefCell::new(Vec::new()),
            var_names: RefCell::new(FxHashMap::default()),
            parameters: RefCell::new(Vec::new()),
            param_names: RefCell::new(FxHashMap::default()),
            constraints: RefCell::new(Vec::new()),
            constraint_names: RefCell::new(FxHashMap::default()),
            soc_constraints: RefCell::new(Vec::new()),
            soc_names: RefCell::new(FxHashMap::default()),
            sos_constraints: RefCell::new(Vec::new()),
            sos_names: RefCell::new(FxHashMap::default()),
            sos_reformulations: RefCell::new(Vec::new()),
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

    /// Construct a constant expression for format readers and other adapters.
    #[doc(hidden)]
    pub fn __constant(&self, value: f64) -> Expr<'_> {
        Expr::constant(&self.arena, value)
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

    fn register_vars_batch<'a>(&'a self, items: &[PendingVar], domain: Domain) -> Vec<Expr<'a>> {
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
            let node = arena.var(id);
            handles.push(Expr::new(node, &self.arena));
        }
        self.cached_kind.set(None);
        handles
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
            parallel: None,
            _k: PhantomData,
        }
    }

    pub fn variable_id(&self, name: &str) -> Option<VarId> {
        self.var_names.borrow().get(name).copied()
    }

    pub fn variables(&self) -> Ref<'_, Vec<Variable>> {
        self.variables.borrow()
    }

    /// Return an immutable copy-on-write snapshot of the expression arena.
    ///
    /// The snapshot is cheap to create, but it is not live.
    /// Subsequent model mutations (including parameter rebinding) are
    /// not visible through a snapshot that is already held.
    pub fn arena(&self) -> ExprArenaSnapshot<'_> {
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
    /// Panics if `e` is not a bare variable handle, or on anything
    /// [`Self::fix_var`] rejects.
    pub fn fix(&self, e: Expr<'_>, value: f64) {
        let id = e.var_id().expect("Model::fix expects a single-variable expression");
        self.fix_var(id, value);
    }

    /// Fix variable `id` to `value` by setting `lb = ub = value`.
    ///
    /// # Panics
    ///
    /// Panics if `value` is not a feasible fixing for the variable (non-finite,
    /// fractional on an integer domain, outside its bounds, or inside a
    /// semicontinuity gap), or if the variable belongs to an SOS constraint that
    /// has already been reformulated, because its bounds are embedded in
    /// generated rows.
    pub fn fix_var(&self, id: VarId, value: f64) {
        self.assert_sos_member_bounds_mutable(id);
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
    /// Panics if `e` is not a bare variable handle.
    pub fn set_initial(&self, e: Expr<'_>, value: f64) {
        let id = e.var_id().expect("Model::set_initial expects a single-variable expression");
        self.variables.borrow_mut()[id.index()].initial = Some(value);
    }

    /// Restore bounds on variable `id`. Pass `f64::NEG_INFINITY` / `f64::INFINITY`
    /// to restore an unbounded direction.
    ///
    /// # Panics
    ///
    /// Panics if the variable belongs to an SOS constraint that has already
    /// been reformulated, because its bounds are embedded in generated rows.
    pub fn unfix_var(&self, id: VarId, lb: f64, ub: f64) {
        self.assert_sos_member_bounds_mutable(id);
        let mut vars = self.variables.borrow_mut();
        let v = &mut vars[id.index()];
        v.lb = lb;
        v.ub = ub;
        drop(vars);
        self.cached_kind.set(None);
    }

    /// Reformulation embeds the member bounds in generated Big-M rows.
    /// Once an SOS has been reformulated, changing one of its member bounds
    /// would make those rows stale and could either truncate or enlarge
    /// the feasible set.
    fn assert_sos_member_bounds_mutable(&self, id: VarId) {
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
    /// Large families may evaluate `value` concurrently, so it must be safe to
    /// call from multiple worker threads.
    ///
    /// # Panics
    ///
    /// Panics if a per-key parameter name collides with one already registered.
    #[doc(hidden)]
    pub fn __indexed_param<'a, K, F>(
        &'a self,
        name: impl Into<String>,
        set: &Set<K>,
        value: F,
    ) -> IndexedParam<'a, K>
    where
        K: FromIndexKey,
        F: Fn(K) -> f64 + Send + Sync,
    {
        self.indexed_param_with(name.into(), set, &value, None)
    }

    fn indexed_param_with<'a, K, F>(
        &'a self,
        base: String,
        set: &Set<K>,
        value: &F,
        forced_parallel: Option<bool>,
    ) -> IndexedParam<'a, K>
    where
        K: FromIndexKey,
        F: Fn(K) -> f64 + Send + Sync,
    {
        let axes = set.axes().map(Box::from);
        let keys: Vec<IndexKey> = set.iter().collect();
        if !indexed_parallel(keys.len(), forced_parallel, PAR_INDEXED_METADATA_THRESHOLD) {
            let handles = keys
                .iter()
                .map(|key| {
                    let name: SmolStr = format_index_name(&base, key).into();
                    self.register_param(name, value(K::from_index_key(key)))
                })
                .collect();
            let storage = build_storage(keys, axes, handles);
            return IndexedFamily { storage, _marker: PhantomData };
        }
        let prepare = |key: &IndexKey| PendingParam {
            name: format_index_name(&base, key).into(),
            value: value(K::from_index_key(key)),
        };
        let prepared: Vec<_> = keys.par_iter().map(prepare).collect();
        let handles = self.register_params_batch(&prepared);
        drop(prepared);
        let storage = build_storage(keys, axes, handles);
        IndexedFamily { storage, _marker: PhantomData }
    }

    fn register_params_batch<'a>(&'a self, items: &[PendingParam]) -> Vec<Expr<'a>> {
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
            let node = arena.param(id);
            params.push(Parameter { id, name: item.name.clone() });
            names.insert(item.name.clone(), id);
            handles.push(Expr::new(node, &self.arena));
        }
        handles
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
        self.assert_expr_belongs(c.lhs);
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

    // Call only after name preflight, with no intervening user callbacks.
    fn register_prevalidated_constraints_batch(
        &self,
        items: Vec<PendingConstraint>,
    ) -> Vec<ConstraintId> {
        let mut names = self.constraint_names.borrow_mut();
        let mut constraints = self.constraints.borrow_mut();
        let final_count =
            constraints.len().checked_add(items.len()).expect("constraint count overflow");
        if final_count > 0 {
            u32::try_from(final_count - 1).expect("constraint count overflow");
        }
        constraints.reserve(items.len());
        names.reserve(items.len());
        let mut ids = Vec::with_capacity(items.len());
        for item in items {
            let id =
                ConstraintId(u32::try_from(constraints.len()).expect("constraint count overflow"));
            constraints.push(Constraint {
                name: item.name.clone(),
                lhs: item.lhs,
                lower: item.lower,
                upper: item.upper,
                active: true,
            });
            names.insert(item.name, id);
            ids.push(id);
        }
        self.cached_kind.set(None);
        ids
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

    /// Register a canonical interval row. This is intentionally hidden from
    /// the public builder API; file readers need to preserve native range rows.
    #[doc(hidden)]
    pub fn __add_constraint_interval(
        &self,
        name: impl Into<SmolStr>,
        lhs: Expr<'_>,
        lower: f64,
        upper: f64,
    ) -> ConstraintId {
        self.assert_expr_belongs(lhs);
        self.register_constraint(name.into(), lhs.id, lower, upper)
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
    pub fn __add_constraints_over<'a, K, F>(
        &'a self,
        name_prefix: &str,
        set: &Set<K>,
        rule: F,
    ) -> IndexedConstraint<K>
    where
        K: FromIndexKey,
        F: Fn(K) -> ConstraintExpr<'a> + Send + Sync,
    {
        self.add_constraints_over_with(name_prefix, set, &rule, None)
    }

    fn add_constraints_over_with<'a, K, F>(
        &'a self,
        name_prefix: &str,
        set: &Set<K>,
        rule: &F,
        forced_parallel: Option<bool>,
    ) -> IndexedConstraint<K>
    where
        K: FromIndexKey,
        F: Fn(K) -> ConstraintExpr<'a> + Send + Sync,
    {
        let keys: Vec<IndexKey> = set.iter().collect();
        if !indexed_parallel(keys.len(), forced_parallel, PAR_INDEXED_ALGEBRAIC_THRESHOLD) {
            let mut ids = Vec::with_capacity(keys.len());
            for key in &keys {
                let constraint = rule(K::from_index_key(key));
                self.assert_expr_belongs(constraint.lhs);
                let name: SmolStr = format_index_name(name_prefix, key).into();
                ids.push(self.__add_constraint(name, constraint));
            }
            return IndexedConstraint::new(keys, set.axes(), ids);
        }

        let arena = &self.arena;
        let expected_arena = arena_key(arena);
        let mut batch = arena.__begin_batch();
        let snapshot = batch.snapshot();
        let chunk_size = indexed_chunk_size(keys.len());
        let mut forks: Vec<_> = keys
            .par_chunks(chunk_size)
            .map(|chunk| {
                arena.__with_fork(snapshot.clone(), || {
                    chunk
                        .iter()
                        .map(|key| {
                            let name = format_index_name(name_prefix, key).into();
                            let constraint = rule(K::from_index_key(key));
                            assert_expr_arena(constraint.lhs, expected_arena);
                            PendingConstraint::from_expr(name, constraint)
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        drop(snapshot);
        {
            let existing = self.constraint_names.borrow();
            validate_batch_names(
                &existing,
                forks.iter().flat_map(|fork| fork.value.iter().map(|item| &item.name)),
                "constraint",
                keys.len(),
            );
        }
        let remaps = batch.merge(&mut forks);
        let mut pending = Vec::with_capacity(keys.len());
        for (mut fork, remap) in forks.into_iter().zip(remaps) {
            for item in &mut fork.value {
                item.remap(remap);
            }
            pending.extend(fork.value);
        }
        drop(batch);
        let ids = self.register_prevalidated_constraints_batch(pending);
        IndexedConstraint::new(keys, set.axes(), ids)
    }

    /// Macro-facing entry point for a two-sided range `lo <= mid <= hi`.
    ///
    /// Collapses to a single interval [`Constraint`] named `name` only when both
    /// bounds are pure constants and the body is linear (the condition under which
    /// one two-sided row is representable).
    #[doc(hidden)]
    pub fn __add_range<'a, B1, B2>(
        &'a self,
        name: &str,
        mid: Expr<'a>,
        lo: B1,
        hi: B2,
    ) -> RangeConstraintIds
    where
        B1: IntoRhs<'a>,
        B2: IntoRhs<'a>,
    {
        self.assert_expr_belongs(mid);
        if let Some((lower, upper)) = self.collapse_bounds(mid.id, &lo, &hi) {
            RangeConstraintIds::Interval(self.register_constraint(
                name.into(),
                mid.id,
                lower,
                upper,
            ))
        } else {
            let lower = self.__add_constraint(format!("{name}_lo"), mid.ge(lo));
            let upper = self.__add_constraint(format!("{name}_hi"), mid.le(hi));
            RangeConstraintIds::Split { lower, upper }
        }
    }

    /// Anonymous form of [`Self::__add_range`] (auto-named rows).
    #[doc(hidden)]
    pub fn __add_range_auto<'a, B1, B2>(
        &'a self,
        mid: Expr<'a>,
        lo: B1,
        hi: B2,
    ) -> RangeConstraintIds
    where
        B1: IntoRhs<'a>,
        B2: IntoRhs<'a>,
    {
        self.assert_expr_belongs(mid);
        if let Some((lower, upper)) = self.collapse_bounds(mid.id, &lo, &hi) {
            RangeConstraintIds::Interval(self.register_constraint(
                self.next_auto_name(),
                mid.id,
                lower,
                upper,
            ))
        } else {
            let lower = self.__add_constraint_auto(mid.ge(lo));
            let upper = self.__add_constraint_auto(mid.le(hi));
            RangeConstraintIds::Split { lower, upper }
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

    /// Macro-facing entry point for a two-sided range family. Each key maps to
    /// one interval row or separate lower/upper rows (see [`Self::__add_range`]).
    #[doc(hidden)]
    pub fn __add_range_constraints_over<'a, K, B1, B2, F>(
        &'a self,
        name: &str,
        set: &Set<K>,
        rule: F,
    ) -> IndexedRangeConstraint<K>
    where
        K: FromIndexKey,
        B1: IntoRhs<'a>,
        B2: IntoRhs<'a>,
        F: Fn(K) -> (Expr<'a>, B1, B2) + Send + Sync,
    {
        self.add_range_constraints_over_with(name, set, &rule, None)
    }

    fn add_range_constraints_over_with<'a, K, B1, B2, F>(
        &'a self,
        name: &str,
        set: &Set<K>,
        rule: &F,
        forced_parallel: Option<bool>,
    ) -> IndexedRangeConstraint<K>
    where
        K: FromIndexKey,
        B1: IntoRhs<'a>,
        B2: IntoRhs<'a>,
        F: Fn(K) -> (Expr<'a>, B1, B2) + Send + Sync,
    {
        let keys: Vec<IndexKey> = set.iter().collect();
        if !indexed_parallel(keys.len(), forced_parallel, PAR_INDEXED_RANGE_THRESHOLD) {
            let mut ids = Vec::with_capacity(keys.len());
            for key in &keys {
                let (mid, lo, hi) = rule(K::from_index_key(key));
                self.assert_expr_belongs(mid);
                ids.push(self.__add_range(&format_index_name(name, key), mid, lo, hi));
            }
            return IndexedRangeConstraint::new(keys, set.axes(), ids);
        }

        let arena = &self.arena;
        let expected_arena = arena_key(arena);
        let mut batch = arena.__begin_batch();
        let snapshot = batch.snapshot();
        let chunk_size = indexed_chunk_size(keys.len());
        let mut forks: Vec<_> = keys
            .par_chunks(chunk_size)
            .map(|chunk| {
                arena.__with_fork(snapshot.clone(), || {
                    let mut prepared = PendingRangeBatch {
                        rows: Vec::with_capacity(chunk.len()),
                        row_counts: Vec::with_capacity(chunk.len()),
                    };
                    for key in chunk {
                        let (mid, lo, hi) = rule(K::from_index_key(key));
                        assert_expr_arena(mid, expected_arena);
                        let (first, upper) =
                            prepare_range(format_index_name(name, key), mid, lo, hi);
                        prepared.row_counts.push(if upper.is_some() { 2 } else { 1 });
                        prepared.rows.push(first);
                        prepared.rows.extend(upper);
                    }
                    prepared
                })
            })
            .collect();
        drop(snapshot);
        let row_count = forks.iter().map(|fork| fork.value.rows.len()).sum();
        {
            let existing = self.constraint_names.borrow();
            validate_batch_names(
                &existing,
                forks.iter().flat_map(|fork| fork.value.rows.iter().map(|item| &item.name)),
                "constraint",
                row_count,
            );
        }
        let remaps = batch.merge(&mut forks);
        let mut pending = Vec::with_capacity(row_count);
        let mut row_counts = Vec::with_capacity(keys.len());
        for (mut fork, remap) in forks.into_iter().zip(remaps) {
            for item in &mut fork.value.rows {
                item.remap(remap);
            }
            row_counts.extend(fork.value.row_counts);
            pending.extend(fork.value.rows);
        }
        drop(batch);
        let mut ids = self.register_prevalidated_constraints_batch(pending).into_iter();
        let groups = row_counts
            .into_iter()
            .map(|count| {
                let first = ids.next().expect("range row ID missing");
                match count {
                    1 => RangeConstraintIds::Interval(first),
                    2 => RangeConstraintIds::Split {
                        lower: first,
                        upper: ids.next().expect("upper range row ID missing"),
                    },
                    _ => unreachable!("range must lower to one or two rows"),
                }
            })
            .collect();
        IndexedRangeConstraint::new(keys, set.axes(), groups)
    }

    /// Unified view of every algebraic, explicit SOC, and SOS constraint
    /// declared on this model.
    pub fn constraints(&self) -> ModelConstraints<'_> {
        ModelConstraints {
            algebraic: self.constraints.borrow(),
            second_order_cones: self.soc_constraints.borrow(),
            special_ordered_sets: self.sos_constraints.borrow(),
        }
    }

    /// Total number of algebraic, SOC, and SOS constraints.
    pub fn num_constraints(&self) -> usize {
        self.constraints.borrow().len()
            + self.soc_constraints.borrow().len()
            + self.sos_constraints.borrow().len()
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
                self.assert_expr_belongs(e);
                assert!(
                    classify(&arena, e.id) == ExprClass::Linear,
                    "SOC constraint {name:?} has a non-affine term"
                );
                e.id
            })
            .collect();
        assert!(!terms.is_empty(), "SOC constraint {name:?} has no terms");
        self.assert_expr_belongs(bound);
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

    // Call only after name preflight, with no intervening user callbacks.
    fn register_prevalidated_soc_batch(&self, items: Vec<PendingSoc>) {
        let mut names = self.soc_names.borrow_mut();
        let mut constraints = self.soc_constraints.borrow_mut();
        let final_count =
            constraints.len().checked_add(items.len()).expect("SOC constraint count overflow");
        if final_count > 0 {
            u32::try_from(final_count - 1).expect("SOC constraint count overflow");
        }
        constraints.reserve(items.len());
        names.reserve(items.len());
        for item in items {
            let id = SocConstraintId(
                u32::try_from(constraints.len()).expect("SOC constraint count overflow"),
            );
            constraints.push(SocConstraint {
                name: item.name.clone(),
                terms: item.terms,
                bound: item.bound,
                active: true,
            });
            names.insert(item.name, id);
        }
        self.cached_kind.set(None);
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
        rule: F,
    ) where
        K: FromIndexKey,
        T: IntoIterator<Item = Expr<'a>>,
        F: Fn(K) -> (T, Expr<'a>) + Send + Sync,
    {
        self.add_soc_constraints_over_with(name_prefix, set, &rule, None);
    }

    fn add_soc_constraints_over_with<'a, K, T, F>(
        &'a self,
        name_prefix: &str,
        set: &Set<K>,
        rule: &F,
        forced_parallel: Option<bool>,
    ) where
        K: FromIndexKey,
        T: IntoIterator<Item = Expr<'a>>,
        F: Fn(K) -> (T, Expr<'a>) + Send + Sync,
    {
        let keys: Vec<IndexKey> = set.iter().collect();
        if !indexed_parallel(keys.len(), forced_parallel, PAR_INDEXED_SOC_THRESHOLD) {
            for key in &keys {
                let name: SmolStr = format_index_name(name_prefix, key).into();
                let (terms, bound) = rule(K::from_index_key(key));
                let terms: Vec<_> = terms.into_iter().collect();
                for &term in &terms {
                    self.assert_expr_belongs(term);
                }
                self.assert_expr_belongs(bound);
                self.add_soc_constraint(name, terms, bound);
            }
            return;
        }

        let arena = &self.arena;
        let expected_arena = arena_key(arena);
        let mut batch = arena.__begin_batch();
        let snapshot = batch.snapshot();
        let chunk_size = indexed_chunk_size(keys.len());
        let mut forks: Vec<_> = keys
            .par_chunks(chunk_size)
            .map(|chunk| {
                arena.__with_fork(snapshot.clone(), || {
                    chunk
                        .iter()
                        .map(|key| {
                            let name = format_index_name(name_prefix, key).into();
                            let (terms, bound) = rule(K::from_index_key(key));
                            assert_expr_arena(bound, expected_arena);
                            let terms: Vec<_> = terms.into_iter().collect();
                            for &term in &terms {
                                assert_expr_arena(term, expected_arena);
                            }
                            prepare_soc(name, terms, bound)
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        drop(snapshot);
        {
            let existing = self.soc_names.borrow();
            validate_batch_names(
                &existing,
                forks.iter().flat_map(|fork| fork.value.iter().map(|item| &item.name)),
                "SOC constraint",
                keys.len(),
            );
        }
        let remaps = batch.merge(&mut forks);
        let mut pending = Vec::with_capacity(keys.len());
        for (mut fork, remap) in forks.into_iter().zip(remaps) {
            for item in &mut fork.value {
                item.remap(remap);
            }
            pending.extend(fork.value);
        }
        drop(batch);
        self.register_prevalidated_soc_batch(pending);
    }

    /// Typed explicit-SOC registry for specialized backend passes.
    ///
    /// Use [`Self::constraints`] when inspecting constraints generically. This
    /// accessor exists so performance-sensitive translators can keep a
    /// homogeneous borrow scoped to one conic pass.
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

    /// Register an explicit SOS1 or SOS2 constraint. Members must be bare
    /// variables belonging to this model and have finite, unique weights.
    ///
    /// # Panics
    ///
    /// Panics when a member belongs to another model, is not a bare variable,
    /// or the name, members, variables, or weights violate SOS invariants.
    pub fn add_sos_constraint<'a>(
        &'a self,
        name: impl Into<SmolStr>,
        sos_type: SosType,
        members: impl IntoIterator<Item = (Expr<'a>, f64)>,
    ) -> SosConstraintHandle<'a> {
        let name = name.into();
        let members: Vec<SosMember> = members
            .into_iter()
            .map(|(expr, weight)| {
                assert!(
                    std::ptr::eq(expr.arena, &raw const self.arena),
                    "SOS member belongs to another model"
                );
                let variable = expr.var_id().expect("SOS members must be bare variables");
                SosMember { variable, weight }
            })
            .collect();
        validate_members(&name, &members);
        let mut by_name = self.sos_names.borrow_mut();
        assert!(!by_name.contains_key(&name), "SOS constraint name {name:?} already registered");
        let mut all = self.sos_constraints.borrow_mut();
        let id = SosConstraintId(u32::try_from(all.len()).expect("SOS constraint count overflow"));
        all.push(SosConstraint { name: name.clone(), sos_type, members, active: true });
        by_name.insert(name, id);
        self.cached_kind.set(None);
        SosConstraintHandle { model: self, id }
    }

    // Call only after name preflight, with no intervening user callbacks.
    fn register_prevalidated_sos_batch(&self, sos_type: SosType, items: Vec<PendingSos>) {
        let mut names = self.sos_names.borrow_mut();
        let mut constraints = self.sos_constraints.borrow_mut();
        let final_count =
            constraints.len().checked_add(items.len()).expect("SOS constraint count overflow");
        if final_count > 0 {
            u32::try_from(final_count - 1).expect("SOS constraint count overflow");
        }
        constraints.reserve(items.len());
        names.reserve(items.len());
        for item in items {
            let id = SosConstraintId(
                u32::try_from(constraints.len()).expect("SOS constraint count overflow"),
            );
            constraints.push(SosConstraint {
                name: item.name.clone(),
                sos_type,
                members: item.members,
                active: true,
            });
            names.insert(item.name, id);
        }
        self.cached_kind.set(None);
    }

    fn register_sos_one(&self, sos_type: SosType, item: PendingSos) {
        let mut names = self.sos_names.borrow_mut();
        assert!(
            !names.contains_key(&item.name),
            "SOS constraint name {:?} already registered",
            item.name
        );
        let mut constraints = self.sos_constraints.borrow_mut();
        let id = SosConstraintId(
            u32::try_from(constraints.len()).expect("SOS constraint count overflow"),
        );
        constraints.push(SosConstraint {
            name: item.name.clone(),
            sos_type,
            members: item.members,
            active: true,
        });
        names.insert(item.name, id);
        self.cached_kind.set(None);
    }

    fn add_pending_sos_over(
        &self,
        keys: Vec<IndexKey>,
        sos_type: SosType,
        forced_parallel: Option<bool>,
        prepare: impl Fn(&IndexKey) -> PendingSos + Send + Sync,
    ) {
        if !indexed_parallel(keys.len(), forced_parallel, PAR_INDEXED_SOS_THRESHOLD) {
            for key in &keys {
                self.register_sos_one(sos_type, prepare(key));
            }
            return;
        }

        let arena = &self.arena;
        let batch = arena.__begin_batch();
        let snapshot = batch.snapshot();
        let chunk_size = indexed_chunk_size(keys.len());
        let forks: Vec<_> = keys
            .par_chunks(chunk_size)
            .map(|chunk| {
                arena.__with_fork(snapshot.clone(), || {
                    chunk.iter().map(&prepare).collect::<Vec<_>>()
                })
            })
            .collect();
        drop(snapshot);
        {
            let existing = self.sos_names.borrow();
            validate_batch_names(
                &existing,
                forks.iter().flat_map(|fork| fork.value.iter().map(|item| &item.name)),
                "SOS constraint",
                keys.len(),
            );
        }
        // Valid SOS members contain only VarIds, so no expression-root remap is needed.
        drop(batch);
        let pending = forks.into_iter().flat_map(|fork| fork.value).collect();
        self.register_prevalidated_sos_batch(sos_type, pending);
    }

    /// Register an SOS1 or SOS2 constraint with consecutive positional
    /// weights `1, 2, ...` inferred from the member order.
    ///
    /// This is a convenience for models where the ordering is already
    /// represented by the iterator order. Use [`Self::add_sos_constraint`]
    /// when the weights are meaningful values that should be preserved.
    ///
    /// # Panics
    ///
    /// Panics under the same conditions as [`Self::add_sos_constraint`], or
    /// when the iterator contains more than `u32::MAX` members.
    pub fn add_sos_constraint_auto_weights<'a>(
        &'a self,
        name: impl Into<SmolStr>,
        sos_type: SosType,
        variables: impl IntoIterator<Item = Expr<'a>>,
    ) -> SosConstraintHandle<'a> {
        self.add_sos_constraint(
            name,
            sos_type,
            variables.into_iter().enumerate().map(|(index, variable)| {
                let weight = index
                    .checked_add(1)
                    .and_then(|index| u32::try_from(index).ok())
                    .expect("SOS member count exceeds positional weight range");
                (variable, f64::from(weight))
            }),
        )
    }

    fn next_auto_sos_name(&self) -> SmolStr {
        loop {
            let n = self.auto_seq.get();
            self.auto_seq.set(n + 1);
            let candidate: SmolStr = format!("_sos{n}").into();
            if !self.sos_names.borrow().contains_key(&candidate) {
                break candidate;
            }
        }
    }

    #[doc(hidden)]
    pub fn __add_sos_constraint_auto<'a>(
        &'a self,
        sos_type: SosType,
        members: impl IntoIterator<Item = (Expr<'a>, f64)>,
    ) -> SosConstraintHandle<'a> {
        self.add_sos_constraint(self.next_auto_sos_name(), sos_type, members)
    }

    #[doc(hidden)]
    pub fn __add_sos_constraint_auto_weights<'a>(
        &'a self,
        sos_type: SosType,
        variables: impl IntoIterator<Item = Expr<'a>>,
    ) -> SosConstraintHandle<'a> {
        self.add_sos_constraint_auto_weights(self.next_auto_sos_name(), sos_type, variables)
    }

    #[doc(hidden)]
    pub fn __add_sos_constraints_over<'a, K, T, F>(
        &'a self,
        name_prefix: &str,
        set: &Set<K>,
        sos_type: SosType,
        rule: F,
    ) where
        K: FromIndexKey,
        T: IntoIterator<Item = (Expr<'a>, f64)>,
        F: Fn(K) -> T + Send + Sync,
    {
        self.add_sos_constraints_over_with(name_prefix, set, sos_type, &rule, None);
    }

    fn add_sos_constraints_over_with<'a, K, T, F>(
        &'a self,
        name_prefix: &str,
        set: &Set<K>,
        sos_type: SosType,
        rule: &F,
        forced_parallel: Option<bool>,
    ) where
        K: FromIndexKey,
        T: IntoIterator<Item = (Expr<'a>, f64)>,
        F: Fn(K) -> T + Send + Sync,
    {
        let keys: Vec<IndexKey> = set.iter().collect();
        let expected_arena = arena_key(&self.arena);
        self.add_pending_sos_over(keys, sos_type, forced_parallel, |key| {
            let name: SmolStr = format_index_name(name_prefix, key).into();
            let members: Vec<_> = rule(K::from_index_key(key))
                .into_iter()
                .map(|(expr, weight)| {
                    assert_expr_arena(expr, expected_arena);
                    let variable = expr.var_id().expect("SOS members must be bare variables");
                    SosMember { variable, weight }
                })
                .collect();
            validate_members(&name, &members);
            PendingSos { name, members }
        });
    }

    #[doc(hidden)]
    pub fn __add_sos_constraints_over_auto_weights<'a, K, T, F>(
        &'a self,
        name_prefix: &str,
        set: &Set<K>,
        sos_type: SosType,
        rule: F,
    ) where
        K: FromIndexKey,
        T: IntoIterator<Item = Expr<'a>>,
        F: Fn(K) -> T + Send + Sync,
    {
        self.add_sos_constraints_over_auto_weights_with(name_prefix, set, sos_type, &rule, None);
    }

    fn add_sos_constraints_over_auto_weights_with<'a, K, T, F>(
        &'a self,
        name_prefix: &str,
        set: &Set<K>,
        sos_type: SosType,
        rule: &F,
        forced_parallel: Option<bool>,
    ) where
        K: FromIndexKey,
        T: IntoIterator<Item = Expr<'a>>,
        F: Fn(K) -> T + Send + Sync,
    {
        let keys: Vec<IndexKey> = set.iter().collect();
        let expected_arena = arena_key(&self.arena);
        self.add_pending_sos_over(keys, sos_type, forced_parallel, |key| {
            let name: SmolStr = format_index_name(name_prefix, key).into();
            let members: Vec<_> = rule(K::from_index_key(key))
                .into_iter()
                .enumerate()
                .map(|(index, expr)| {
                    let weight = index
                        .checked_add(1)
                        .and_then(|index| u32::try_from(index).ok())
                        .expect("SOS member count exceeds positional weight range");
                    assert_expr_arena(expr, expected_arena);
                    let variable = expr.var_id().expect("SOS members must be bare variables");
                    SosMember { variable, weight: f64::from(weight) }
                })
                .collect();
            validate_members(&name, &members);
            PendingSos { name, members }
        });
    }

    pub fn sos_constraints(&self) -> Ref<'_, Vec<SosConstraint>> {
        self.sos_constraints.borrow()
    }

    pub fn num_sos_constraints(&self) -> usize {
        self.sos_constraints.borrow().len()
    }

    pub fn sos_constraint_id(&self, name: &str) -> Option<SosConstraintId> {
        self.sos_names.borrow().get(name).copied()
    }

    pub fn has_sos_constraints(&self) -> bool {
        !self.sos_constraints.borrow().is_empty()
    }

    /// Whether at least one SOS constraint still requires native backend
    /// handling. Reformulation retains source SOS entries but marks them
    /// inactive so their stable IDs and provenance are preserved.
    pub fn has_active_sos_constraints(&self) -> bool {
        self.sos_constraints.borrow().iter().any(|constraint| constraint.active)
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
        self.assert_expr_belongs(expr);
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
    /// 2. any quadratic constraint not recognized by the structural SOC
    ///    predicate -> `QCP`
    /// 3. cones present (explicit or detected) -> `SOCP`
    /// 4. quadratic objective -> `QP`
    /// 5. otherwise -> `LP`
    pub fn kind(&self) -> ModelKind {
        if let Some(k) = self.cached_kind.get() {
            return k;
        }
        let k = self.infer_kind_with(None);
        self.cached_kind.set(Some(k));
        k
    }

    /// Infer the model kind without reading or updating the kind cache.
    ///
    /// Automatic inference keeps the initial classification scan serial, then
    /// parallelizes only a large set of already-known quadratic candidates for
    /// SOC recognition. This keeps LP/NLP models off the Rayon path.
    fn infer_kind_with(&self, parallel: Option<bool>) -> ModelKind {
        self.infer_kind_impl(parallel)
    }

    /// Infer the kind with a forced serial or parallel SOC-recognition pass.
    /// This is used by benchmarks and parity tests.
    #[cfg(any(test, feature = "benchmark-support"))]
    fn infer_kind(&self, parallel: bool) -> ModelKind {
        self.infer_kind_impl(Some(parallel))
    }

    fn infer_kind_impl(&self, parallel: Option<bool>) -> ModelKind {
        let arena = self.arena.borrow();
        let vars = self.variables.borrow();
        let has_int = vars.iter().any(|v| v.domain.is_integer())
            || self.sos_constraints.borrow().iter().any(|constraint| constraint.active);
        let obj_class = self
            .objective
            .borrow()
            .as_ref()
            .map_or(ExprClass::Linear, |o| classify(&arena, o.expr));

        let mut any_nonlinear = obj_class == ExprClass::Nonlinear;
        let mut plain_quad_con = false;
        let mut detected_soc = false;
        if !any_nonlinear {
            let constraints = self.constraints.borrow();
            let arena_ref = &*arena;
            let vars_ref = &*vars;
            let mut quadratic = Vec::new();
            for c in constraints.iter() {
                match classify(arena_ref, c.lhs) {
                    ExprClass::Linear => {}
                    ExprClass::Quadratic => quadratic.push(c),
                    ExprClass::Nonlinear => {
                        any_nonlinear = true;
                        break;
                    }
                }
            }
            if !any_nonlinear {
                let use_parallel = parallel.unwrap_or(
                    quadratic.len() >= PAR_KIND_THRESHOLD && rayon::current_num_threads() > 1,
                );
                if use_parallel {
                    let (has_cone, has_plain) = quadratic
                        .par_iter()
                        .map(|c| {
                            let is_soc = is_detected_soc(arena_ref, vars_ref, c);
                            (is_soc, !is_soc)
                        })
                        .reduce(
                            || (false, false),
                            |left, right| (left.0 || right.0, left.1 || right.1),
                        );
                    detected_soc = has_cone;
                    plain_quad_con = has_plain;
                } else {
                    for c in quadratic {
                        if is_detected_soc(arena_ref, vars_ref, c) {
                            detected_soc = true;
                        } else {
                            plain_quad_con = true;
                        }
                    }
                }
            }
        }
        let has_soc = detected_soc || !self.soc_constraints.borrow().is_empty();

        let pick = |cont, int| if has_int { int } else { cont };
        if any_nonlinear {
            pick(ModelKind::NLP, ModelKind::MINLP)
        } else if plain_quad_con {
            pick(ModelKind::QCP, ModelKind::MIQCP)
        } else if has_soc {
            pick(ModelKind::SOCP, ModelKind::MISOCP)
        } else if obj_class == ExprClass::Quadratic {
            pick(ModelKind::QP, ModelKind::MIQP)
        } else {
            pick(ModelKind::LP, ModelKind::MILP)
        }
    }
}

// IndexedVarBuilder

/// Builder for a collection of scalar variables indexed by a [`Set`].
///
/// For example, `flow[i]` for `i in 0..3` registers `flow[0]`, `flow[1]`, and
/// `flow[2]` as separate scalar variables in the model. Call `.build()` to get
/// an [`IndexedVar`] that maps each key to its [`Expr`] handle. Bounds and
/// domain set here apply uniformly to every scalar in the collection.
type BoundFn<'a> = Box<dyn Fn(&IndexKey) -> f64 + Send + Sync + 'a>;

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
    parallel: Option<bool>,
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
        F: Send + Sync,
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
        F: Send + Sync,
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
        let Self { model, base_name, keys, axes, lb, ub, lb_by, ub_by, domain, parallel, _k } =
            self;

        if !indexed_parallel(keys.len(), parallel, PAR_INDEXED_METADATA_THRESHOLD) {
            let handles = keys
                .iter()
                .map(|key| {
                    let scalar_name: SmolStr = format_index_name(&base_name, key).into();
                    let lo = lb_by.as_ref().map_or(lb, |f| f(key));
                    let hi = ub_by.as_ref().map_or(ub, |f| f(key));
                    model.__var(scalar_name).lb(lo).ub(hi).domain(domain).build()
                })
                .collect();
            let storage = build_storage(keys, axes, handles);
            return IndexedFamily { storage, _marker: PhantomData };
        }

        let prepare = |key: &IndexKey| PendingVar {
            name: format_index_name(&base_name, key).into(),
            lb: lb_by.as_ref().map_or(lb, |f| f(key)),
            ub: ub_by.as_ref().map_or(ub, |f| f(key)),
        };
        let prepared: Vec<_> = keys.par_iter().map(prepare).collect();
        let handles = model.register_vars_batch(&prepared, domain);
        drop(prepared);
        let storage = build_storage(keys, axes, handles);
        IndexedFamily { storage, _marker: PhantomData }
    }

    #[cfg(any(test, feature = "benchmark-support"))]
    fn parallel_for_benchmark(mut self, parallel: bool) -> Self {
        self.parallel = Some(parallel);
        self
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

#[cfg(feature = "benchmark-support")]
#[doc(hidden)]
#[expect(clippy::cast_precision_loss)]
#[allow(clippy::wildcard_imports)]
pub mod benchmark_support {
    use super::*;

    pub const THRESHOLD: usize = PAR_KIND_THRESHOLD;

    pub fn model(rows: usize, degree: usize) -> Model {
        let model = Model::new("kind_bench");
        let x = model.__var("x").build();
        let y = model.__var("y").build();
        let z = model.__var("z").build();
        model.__minimize(x + y + z);
        for i in 0..rows {
            let lhs = match degree {
                1 => x + 2.0 * y - z,
                2 => x.powi(2) + y,
                _ => x * y * z,
            };
            model.__add_constraint_auto(lhs.le(i as f64 + 10.0));
        }
        model
    }

    pub fn soc_model(rows: usize) -> Model {
        let model = Model::new("kind_soc_bench");
        let x = model.__var("x").build();
        let y = model.__var("y").build();
        let t = model.__var("t").lb(0.0).build();
        model.__minimize(t);
        for _ in 0..rows {
            model.__add_constraint_auto((x.powi(2) + y.powi(2) - t.powi(2)).le(0.0));
        }
        model
    }

    pub fn infer(model: &Model, parallel: bool) -> ModelKind {
        model.infer_kind(parallel)
    }

    #[derive(Copy, Clone, Debug)]
    pub enum IndexedBuildCase {
        Variables,
        Parameters,
        Algebraic,
        Range,
        Soc,
        Sos,
    }

    /// Build and immediately inspect a fresh indexed model so Criterion measures
    /// construction rather than solver translation.
    pub fn indexed_build(rows: usize, case: IndexedBuildCase, parallel: bool) -> usize {
        let model = Model::new("indexed_build_bench");
        let keys = Set::range(0..rows);
        match case {
            IndexedBuildCase::Variables => {
                let x = model
                    .__indexed_var("x", &keys)
                    .lb_by(|i: usize| -(i as f64))
                    .ub_by(|i: usize| i as f64 + 1.0)
                    .parallel_for_benchmark(parallel)
                    .build();
                std::hint::black_box(x.len());
            }
            IndexedBuildCase::Parameters => {
                let value = |i: usize| i as f64 + 1.0;
                let p = model.indexed_param_with("p".to_owned(), &keys, &value, Some(parallel));
                std::hint::black_box(p.len());
            }
            IndexedBuildCase::Algebraic => {
                let x = model.__indexed_var("x", &keys).parallel_for_benchmark(parallel).build();
                let rule = |i: usize| (2.0 * x[i] + 1.0).le(i as f64 + 10.0);
                model.add_constraints_over_with("c", &keys, &rule, Some(parallel));
            }
            IndexedBuildCase::Range => {
                let x = model.__indexed_var("x", &keys).parallel_for_benchmark(parallel).build();
                let rule = |i: usize| (x[i] + 1.0, -(i as f64), i as f64 + 10.0);
                model.add_range_constraints_over_with("r", &keys, &rule, Some(parallel));
            }
            IndexedBuildCase::Soc => {
                let x = model.__indexed_var("x", &keys).parallel_for_benchmark(parallel).build();
                let y = model.__indexed_var("y", &keys).parallel_for_benchmark(parallel).build();
                let t = model.__var("t").lb(0.0).build();
                let rule = |i: usize| ([x[i] + 1.0, y[i] - 1.0], t + i as f64 + 1.0);
                model.add_soc_constraints_over_with("q", &keys, &rule, Some(parallel));
            }
            IndexedBuildCase::Sos => {
                let x = model.__indexed_var("x", &keys).parallel_for_benchmark(parallel).build();
                let y = model.__indexed_var("y", &keys).parallel_for_benchmark(parallel).build();
                let rule = |i: usize| [(x[i], 1.0), (y[i], 2.0)];
                model.add_sos_constraints_over_with(
                    "s",
                    &keys,
                    SosType::Sos1,
                    &rule,
                    Some(parallel),
                );
            }
        }
        model.num_variables()
            + model.num_parameters()
            + model.num_constraints()
            + model.arena().len()
    }

    /// Scalar construction guardrail for the parallel-safe arena migration.
    pub fn scalar_build(rows: usize) -> usize {
        let model = Model::new("scalar_build_bench");
        for i in 0..rows {
            let x = model.__var(format!("x{i}")).build();
            model.__add_constraint(format!("c{i}"), (2.0 * x + 1.0).le(i as f64 + 10.0));
        }
        model.num_variables() + model.num_constraints() + model.arena().len()
    }
}

#[cfg(test)]
#[expect(clippy::cast_precision_loss)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use oximo_expr::extract_linear;

    use super::*;
    use crate::Set;
    use crate::constraint::Relate;

    #[test]
    fn model_kind_display_uses_standard_ascii_acronyms() {
        let kinds = [
            ModelKind::LP,
            ModelKind::MILP,
            ModelKind::QP,
            ModelKind::MIQP,
            ModelKind::QCP,
            ModelKind::MIQCP,
            ModelKind::SOCP,
            ModelKind::MISOCP,
            ModelKind::NLP,
            ModelKind::MINLP,
        ];
        let labels: Vec<_> = kinds.into_iter().map(|kind| kind.to_string()).collect();
        assert_eq!(
            labels,
            ["LP", "MILP", "QP", "MIQP", "QCP", "MIQCP", "SOCP", "MISOCP", "NLP", "MINLP"]
        );
        assert!(labels.iter().all(|label| label.is_ascii()));
    }

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
    fn arena_snapshot_is_not_live_after_parameter_rebind() {
        let m = Model::new("snapshot");
        let param = m.__param("p", 1.0);
        let id = param.param_id().unwrap();
        let snapshot = m.arena();
        m.set_param(param, 2.0);
        assert!((snapshot.param_value(id) - 1.0).abs() < f64::EPSILON);
        assert!((m.param_value(id) - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    #[should_panic(expected = "different model")]
    fn indexed_constraints_reject_foreign_expression_arenas() {
        let target = Model::new("target");
        let foreign = Model::new("foreign");
        let x = foreign.__var("x").build();
        let keys = Set::range(0..1024);
        target.add_constraints_over_with("c", &keys, &|_| x.le(1.0), Some(true));
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
    fn unified_constraints_include_inactive_entries() {
        let m = Model::new("inactive");
        let x = m.__var("x").build();
        let t = m.__var("t").lb(0.0).build();
        m.__add_constraint("row", x.le(1.0));
        m.add_soc_constraint("cone", [x], t);
        m.add_sos_constraint("sos", SosType::Sos1, [(x, 1.0)]);
        m.constraints.borrow_mut()[0].active = false;
        m.soc_constraints.borrow_mut()[0].active = false;
        m.sos_constraints.borrow_mut()[0].active = false;

        let constraints = m.constraints();
        assert_eq!(constraints.len(), 3);
        assert!(constraints.iter().all(|entry| match entry {
            ConstraintRef::Algebraic { constraint, .. } => !constraint.active,
            ConstraintRef::SecondOrderCone { constraint, .. } => !constraint.active,
            ConstraintRef::SpecialOrderedSet { constraint, .. } => !constraint.active,
        }));
    }

    #[test]
    fn uncached_kind_inference_leaves_kind_cache_empty() {
        let m = Model::new("uncached_kind");
        let x = m.__var("x").build();
        m.__minimize(x);
        for _ in 0..PAR_KIND_THRESHOLD {
            m.__add_constraint_auto(x.powi(2).le(1.0));
        }

        assert_eq!(m.infer_kind(false), ModelKind::QCP);
        assert_eq!(m.cached_kind.get(), None);
        assert_eq!(m.infer_kind(true), ModelKind::QCP);
        assert_eq!(m.cached_kind.get(), None);

        assert_eq!(m.kind(), ModelKind::QCP);
        assert_eq!(m.cached_kind.get(), Some(ModelKind::QCP));
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

    #[test]
    fn kind_forced_serial_and_parallel_classification_agree() {
        let qcp = Model::new("qcp");
        let x = qcp.__var("x").build();
        let y = qcp.__var("y").build();
        qcp.__minimize(x + y);
        for i in 0..PAR_KIND_THRESHOLD + 3 {
            qcp.__add_constraint_auto((x.powi(2) + y).le(i as f64 + 1.0));
        }
        assert_eq!(qcp.infer_kind(false), ModelKind::QCP);
        assert_eq!(qcp.infer_kind(false), qcp.infer_kind(true));
        assert_eq!(qcp.infer_kind(false), qcp.infer_kind_with(None));

        let socp = Model::new("socp");
        let x = socp.__var("x").build();
        let t = socp.__var("t").lb(0.0).build();
        socp.__minimize(t);
        for _ in 0..PAR_KIND_THRESHOLD + 3 {
            socp.__add_constraint_auto((x.powi(2) - t.powi(2)).le(0.0));
        }
        assert_eq!(socp.infer_kind(false), ModelKind::SOCP);
        assert_eq!(socp.infer_kind(false), socp.infer_kind(true));
        assert_eq!(socp.infer_kind(false), socp.infer_kind_with(None));

        let nlp = Model::new("nlp");
        let x = nlp.__var("x").build();
        let y = nlp.__var("y").build();
        let z = nlp.__var("z").build();
        nlp.__minimize(x + y + z);
        for _ in 0..PAR_KIND_THRESHOLD + 3 {
            nlp.__add_constraint_auto((x * y * z).le(1.0));
        }
        assert_eq!(nlp.infer_kind(false), ModelKind::NLP);
        assert_eq!(nlp.infer_kind(false), nlp.infer_kind(true));
    }

    #[test]
    fn arena_snapshot_allows_nested_model_reads() {
        let model = Model::new("nested_reads");
        let x = model.__var("x").build();
        model.__minimize(x);

        let arena = model.arena();
        assert_eq!(model.kind(), ModelKind::LP);
        assert!(matches!(arena.get(x.id), oximo_expr::ExprNode::Var(_)));
    }

    #[test]
    fn indexed_constraint_handles_match_in_serial_and_parallel() {
        #[derive(Debug, PartialEq, Eq)]
        struct Handles {
            ordinary: Vec<(usize, ConstraintId)>,
            ranges: Vec<(usize, RangeConstraintIds)>,
        }

        fn build(parallel: bool, keys: &Set<usize>) -> Handles {
            let model = Model::new("family_id_parity");
            let x = model.__var("x").build();
            model.__add_constraint("prior", x.ge(0.0));
            let ordinary =
                model.add_constraints_over_with("c", keys, &|_| x.le(5.0), Some(parallel));
            let ranges = model.add_range_constraints_over_with(
                "r",
                keys,
                &|i| (if i % 2 == 0 { x + 1.0 } else { x.powi(2) }, 0.0, 10.0),
                Some(parallel),
            );
            for (key, id) in ordinary.iter() {
                assert_eq!(ordinary.get(key), Some(id));
                assert_eq!(model.constraint_id(&format!("c[{key}]")), Some(id));
            }
            for (key, ids) in ranges.iter() {
                assert_eq!(ranges.get(key), Some(ids));
                match ids {
                    RangeConstraintIds::Interval(id) => {
                        assert_eq!(model.constraint_id(&format!("r[{key}]")), Some(id));
                        let row = &model.constraints.borrow()[id.index()];
                        assert_eq!((row.lower, row.upper), (0.0, 10.0));
                        assert_eq!(classify(&model.arena(), row.lhs), ExprClass::Linear);
                    }
                    RangeConstraintIds::Split { lower, upper } => {
                        assert_eq!(model.constraint_id(&format!("r[{key}]_lo")), Some(lower));
                        assert_eq!(model.constraint_id(&format!("r[{key}]_hi")), Some(upper));
                        let rows = model.constraints.borrow();
                        assert_eq!(rows[lower.index()].lower.to_bits(), 0.0_f64.to_bits());
                        assert_eq!(rows[upper.index()].upper.to_bits(), 10.0_f64.to_bits());
                        assert_eq!(
                            classify(&model.arena(), rows[lower.index()].lhs),
                            ExprClass::Quadratic
                        );
                    }
                }
            }
            Handles { ordinary: ordinary.iter().collect(), ranges: ranges.iter().collect() }
        }

        rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap().install(|| {
            for keys in [Set::range(3..1030), Set::from_ints([8, 1, 6, 3, 4])] {
                assert_eq!(build(false, &keys), build(true, &keys));
            }
        });
    }

    #[test]
    fn indexed_build_forced_parallel_matches_serial_for_every_family() {
        #[derive(Debug, PartialEq)]
        struct Digest {
            variables: Vec<String>,
            parameters: Vec<(String, f64)>,
            rows: Vec<String>,
            typed: Vec<String>,
            arena_len: usize,
        }

        fn digest(parallel: bool) -> Digest {
            let model = Model::new("indexed_parity");
            let keys = Set::range(0..128usize);
            let x = model
                .__indexed_var("x", &keys)
                .lb_by(|i: usize| -(i as f64))
                .ub_by(|i: usize| i as f64 + 10.0)
                .parallel_for_benchmark(parallel)
                .build();
            let y = model.__indexed_var("y", &keys).parallel_for_benchmark(parallel).build();
            let values = |i: usize| i as f64 + 0.5;
            let p = model.indexed_param_with("p".to_owned(), &keys, &values, Some(parallel));

            let algebraic = |i: usize| (2.0 * x[i] + p[i]).le(i as f64 + 20.0);
            model.add_constraints_over_with("c", &keys, &algebraic, Some(parallel));
            let ranges = |i: usize| (y[i] + 1.0, -(i as f64), i as f64 + 5.0);
            model.add_range_constraints_over_with("r", &keys, &ranges, Some(parallel));
            let symbolic_ranges = |i: usize| (x[i] + 2.0, -p[i], p[i] + 3.0);
            model.add_range_constraints_over_with("sr", &keys, &symbolic_ranges, Some(parallel));
            let t = model.__var("t").lb(0.0).build();
            let cones = |i: usize| ([x[i] + 1.0, y[i] - 1.0], t + p[i]);
            model.add_soc_constraints_over_with("q", &keys, &cones, Some(parallel));
            let explicit_sos = |i: usize| [(x[i], 1.0), (y[i], 2.0)];
            model.add_sos_constraints_over_with(
                "s",
                &keys,
                SosType::Sos1,
                &explicit_sos,
                Some(parallel),
            );
            let auto_sos = |i: usize| [x[i], y[i]];
            model.add_sos_constraints_over_auto_weights_with(
                "a",
                &keys,
                SosType::Sos2,
                &auto_sos,
                Some(parallel),
            );

            let variables = model
                .variables()
                .iter()
                .map(|variable| {
                    format!(
                        "{}:{:?}:{}:{}:{}",
                        variable.name, variable.id, variable.lb, variable.ub, variable.domain
                    )
                })
                .collect();
            let parameters = model
                .parameters()
                .iter()
                .map(|parameter| (parameter.name.to_string(), model.param_value(parameter.id)))
                .collect();
            let arena = model.arena();
            let constraints = model.constraints();
            let rows = constraints
                .algebraic()
                .iter()
                .map(|constraint| {
                    let terms = extract_linear(&arena, constraint.lhs).unwrap();
                    format!(
                        "{}:{:?}:{}:{}:{}",
                        constraint.name,
                        terms.coeffs,
                        terms.constant,
                        constraint.lower,
                        constraint.upper
                    )
                })
                .collect();
            let mut typed = constraints
                .second_order_cones()
                .iter()
                .map(|constraint| {
                    let terms: Vec<_> = constraint
                        .terms
                        .iter()
                        .map(|&term| extract_linear(&arena, term).unwrap().into_owned())
                        .collect();
                    let bound = extract_linear(&arena, constraint.bound).unwrap().into_owned();
                    format!("{}:{terms:?}:{bound:?}", constraint.name)
                })
                .collect::<Vec<_>>();
            typed.extend(
                constraints
                    .special_ordered_sets()
                    .iter()
                    .map(|constraint| format!("{constraint:?}")),
            );
            let arena_len = arena.len();
            Digest { variables, parameters, rows, typed, arena_len }
        }

        assert_eq!(digest(false), digest(true));
    }

    #[test]
    fn parallel_duplicate_name_failure_does_not_mutate_model_or_arena() {
        let model = Model::new("indexed_atomicity");
        let x = model.__var("x").build();
        let duplicate_keys = Set::from_ints([0usize, 0]);
        let arena_len = model.arena().len();
        let rule = |i: usize| (x + i as f64).le(1.0);
        let result = catch_unwind(AssertUnwindSafe(|| {
            model.add_constraints_over_with("dup", &duplicate_keys, &rule, Some(true));
        }));
        assert!(result.is_err());
        assert_eq!(model.num_constraints(), 0);
        assert_eq!(model.arena().len(), arena_len);

        model.__add_constraint("after", x.le(2.0));
        assert_eq!(model.constraint_id("after"), Some(ConstraintId(0)));
    }

    #[test]
    fn parallel_range_failure_does_not_mutate_model_or_arena() {
        for duplicate_name in [false, true] {
            let model = Model::new("range_batch_atomicity");
            let x = model.__var("x").build();
            let prior = model.__add_constraint("r[17]_hi", x.le(2.0));
            let arena_len = model.arena().len();
            let keys = Set::range(0..128usize);
            let rule = |i: usize| {
                let body = if i.is_multiple_of(2) { x + 1.0 } else { x.powi(2) };
                assert!(duplicate_name || i != 17, "deliberate range callback panic");
                (body, 0.0, 10.0)
            };
            let result = catch_unwind(AssertUnwindSafe(|| {
                model.add_range_constraints_over_with("r", &keys, &rule, Some(true));
            }));
            assert!(result.is_err());
            assert_eq!(model.num_constraints(), 1);
            assert_eq!(model.constraint_id("r[17]_hi"), Some(prior));
            assert_eq!(model.arena().len(), arena_len);

            // The failed batch must also release the arena for subsequent builds.
            let family = model.add_range_constraints_over_with(
                "after",
                &keys,
                &|_| (x, 0.0, 1.0),
                Some(true),
            );
            assert_eq!(family.get(0), Some(RangeConstraintIds::Interval(ConstraintId(1))));
            assert_eq!(model.num_constraints(), 129);
        }
    }

    #[test]
    fn parallel_callback_panic_does_not_mutate_model_or_arena() {
        let model = Model::new("indexed_callback_atomicity");
        let x = model.__var("x").build();
        let keys = Set::range(0..128usize);
        let arena_len = model.arena().len();
        let rule = |i: usize| {
            let constraint = (x + i as f64).le(1.0);
            assert_ne!(i, 17, "deliberate callback panic");
            constraint
        };
        let result = catch_unwind(AssertUnwindSafe(|| {
            model.add_constraints_over_with("panic", &keys, &rule, Some(true));
        }));
        assert!(result.is_err());
        assert_eq!(model.num_constraints(), 0);
        assert_eq!(model.arena().len(), arena_len);

        model.__add_constraint("after", x.le(2.0));
        assert_eq!(model.constraint_id("after"), Some(ConstraintId(0)));
    }
}
