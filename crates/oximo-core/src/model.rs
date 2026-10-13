mod classify;
mod constraints;
mod indexed;
mod variables;

pub use classify::ModelKind;
pub use constraints::{ConstraintRef, ModelConstraints};
pub use indexed::{IndexedVarBuilder, display_index_key};

#[cfg(feature = "benchmark-support")]
#[doc(hidden)]
#[expect(clippy::cast_precision_loss)]
pub mod benchmark_support;

#[cfg(test)]
#[expect(clippy::cast_precision_loss)]
mod tests;

use crate::constraint::{AlgebraicConstraint, ConstraintId};
use crate::error::{Error, Result};
use crate::indicator::{IndicatorConstraint, IndicatorConstraintId};
use crate::objective::{Objective, ObjectiveSense};
use crate::param::Parameter;
use crate::reformulation::{IndicatorReformulationArtifacts, SosReformulationArtifacts};
use crate::soc::{SocConstraint, SocConstraintId};
use crate::sos::{SosConstraint, SosConstraintId};
use crate::var::Variable;
use crate::{PsdConstraint, PsdConstraintId};
use oximo_expr::{
    Constant, Degree, EvalError, Expr, ExprArena, ExprArenaCell, ExprArenaSnapshot, ModelId,
    ModelMismatchError, ParamId, VarId,
};
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};
use smol_str::SmolStr;
use std::cell::{Cell, Ref, RefCell};

#[cold]
#[inline(never)]
fn model_mismatch(
    expected: ModelId,
    actual: ModelId,
) -> std::result::Result<(), ModelMismatchError> {
    Err(ModelMismatchError::new(expected, actual))
}

fn arena_key(arena: &ExprArenaCell) -> usize {
    std::ptr::from_ref(arena) as usize
}

fn assert_expr_arena<D: Degree>(expr: Expr<'_, D>, expected: usize) {
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
    #[cfg(feature = "gdp")]
    pub(crate) gdp: RefCell<crate::gdp::GdpData>,
    pub name: SmolStr,
    pub(crate) arena: ExprArenaCell,
    pub(crate) variables: RefCell<Vec<Variable>>,
    pub(crate) var_names: RefCell<FxHashMap<SmolStr, VarId>>,
    pub(crate) parameters: RefCell<Vec<Parameter>>,
    pub(crate) param_names: RefCell<FxHashMap<SmolStr, ParamId>>,
    pub(crate) constraints: RefCell<Vec<AlgebraicConstraint>>,
    pub(crate) constraint_names: RefCell<FxHashMap<SmolStr, ConstraintId>>,
    pub(crate) soc_constraints: RefCell<Vec<SocConstraint>>,
    pub(crate) soc_names: RefCell<FxHashMap<SmolStr, SocConstraintId>>,
    pub(crate) psd_constraints: RefCell<Vec<PsdConstraint>>,
    pub(crate) psd_names: RefCell<FxHashMap<SmolStr, PsdConstraintId>>,
    pub(crate) sos_constraints: RefCell<Vec<SosConstraint>>,
    pub(crate) sos_names: RefCell<FxHashMap<SmolStr, SosConstraintId>>,
    pub(crate) indicator_constraints: RefCell<Vec<IndicatorConstraint>>,
    pub(crate) indicator_names: RefCell<FxHashMap<SmolStr, IndicatorConstraintId>>,
    pub(crate) sos_reformulations: RefCell<Vec<SosReformulationArtifacts>>,
    pub(crate) indicator_reformulations: RefCell<Vec<IndicatorReformulationArtifacts>>,
    pub(crate) objective: RefCell<Option<Objective>>,
    objective_declared: Cell<bool>,
    cached_kind: Cell<Option<ModelKind>>,
    /// Monotonic counter for auto-naming anonymous constraints registered via
    /// the `constraint!` macro.
    auto_seq: Cell<u32>,
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = f.debug_struct("Model");
        debug
            .field("name", &self.name)
            .field("vars", &self.variables.borrow().len())
            .field("params", &self.parameters.borrow().len())
            .field("constraints", &self.constraints.borrow().len())
            .field("soc_constraints", &self.soc_constraints.borrow().len())
            .field("psd_constraints", &self.psd_constraints.borrow().len())
            .field("sos_constraints", &self.sos_constraints.borrow().len())
            .field("indicator_constraints", &self.indicator_constraints.borrow().len())
            .field("has_objective", &self.objective.borrow().is_some())
            .field("feasibility", &self.is_feasibility());

        #[cfg(feature = "gdp")]
        {
            let data = self.gdp.borrow();

            debug
                .field("gdp_booleans", &data.booleans.len())
                .field("gdp_rows", &data.rows.len())
                .field("gdp_disjunctions", &data.disjunctions.len())
                .field("gdp_logical_constraints", &data.logical_constraints.len())
                .field("gdp_pending", &data.has_pending());
        }

        debug.finish()
    }
}

impl Model {
    /// Whether GDP components still need explicit reformulation.
    #[inline]
    pub fn has_unreformulated_gdp(&self) -> bool {
        #[cfg(feature = "gdp")]
        {
            self.gdp.borrow().has_pending()
        }
        #[cfg(not(feature = "gdp"))]
        {
            false
        }
    }

    /// Reject unresolved conditional modeling components before lowering.
    ///
    /// # Errors
    #[cfg_attr(
        feature = "gdp",
        doc = "Returns [`Error::UnreformulatedGdp`] when a GDP component is pending."
    )]
    #[cfg_attr(not(feature = "gdp"), doc = "Always succeeds when GDP support is disabled.")]
    pub fn ensure_gdp_reformulated(&self) -> Result<()> {
        #[cfg(feature = "gdp")]
        if self.has_unreformulated_gdp() {
            return Err(Error::UnreformulatedGdp);
        }
        Ok(())
    }

    fn assert_expr_belongs<'a, D: Degree>(&self, expr: Expr<'a, D>) {
        assert_expr_arena(expr, arena_key(&self.arena));
    }

    #[inline]
    fn ensure_model_id(&self, actual: ModelId) -> std::result::Result<(), ModelMismatchError> {
        let expected = self.id();
        if actual == expected { Ok(()) } else { model_mismatch(expected, actual) }
    }

    #[inline]
    fn ensure_expr_model<'a, D: Degree>(
        &self,
        expr: Expr<'a, D>,
    ) -> std::result::Result<(), ModelMismatchError> {
        self.ensure_model_id(expr.model_id())
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

        let cloned = Self {
            name: self.name.clone(),
            #[cfg(feature = "gdp")]
            gdp: RefCell::new(self.gdp.borrow().clone()),
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
            psd_constraints: RefCell::new(self.psd_constraints.borrow().clone()),
            psd_names: RefCell::new(self.psd_names.borrow().clone()),
            sos_constraints: RefCell::new(self.sos_constraints.borrow().clone()),
            sos_names: RefCell::new(self.sos_names.borrow().clone()),
            indicator_constraints: RefCell::new(self.indicator_constraints.borrow().clone()),
            indicator_names: RefCell::new(self.indicator_names.borrow().clone()),
            sos_reformulations: RefCell::new(self.sos_reformulations.borrow().clone()),
            indicator_reformulations: RefCell::new(self.indicator_reformulations.borrow().clone()),
            objective: RefCell::new(self.objective.borrow().clone()),
            objective_declared: Cell::new(self.objective_declared.get()),
            cached_kind: Cell::new(self.cached_kind.get()),
            auto_seq: Cell::new(self.auto_seq.get()),
        };
        #[cfg(feature = "gdp")]
        cloned.rebind_gdp_logic_identity();
        cloned
    }

    pub fn new(name: impl Into<SmolStr>) -> Self {
        Self {
            name: name.into(),
            #[cfg(feature = "gdp")]
            gdp: RefCell::new(crate::gdp::GdpData::default()),
            arena: ExprArenaCell::new(ExprArena::new()),
            variables: RefCell::new(Vec::new()),
            var_names: RefCell::new(FxHashMap::default()),
            parameters: RefCell::new(Vec::new()),
            param_names: RefCell::new(FxHashMap::default()),
            constraints: RefCell::new(Vec::new()),
            constraint_names: RefCell::new(FxHashMap::default()),
            soc_constraints: RefCell::new(Vec::new()),
            soc_names: RefCell::new(FxHashMap::default()),
            psd_constraints: RefCell::new(Vec::new()),
            psd_names: RefCell::new(FxHashMap::default()),
            sos_constraints: RefCell::new(Vec::new()),
            sos_names: RefCell::new(FxHashMap::default()),
            indicator_constraints: RefCell::new(Vec::new()),
            indicator_names: RefCell::new(FxHashMap::default()),
            sos_reformulations: RefCell::new(Vec::new()),
            indicator_reformulations: RefCell::new(Vec::new()),
            objective: RefCell::new(None),
            objective_declared: Cell::new(false),
            cached_kind: Cell::new(None),
            auto_seq: Cell::new(0),
        }
    }

    /// Stable identity carried by this model's expression handles and results.
    #[inline]
    #[must_use]
    pub const fn id(&self) -> ModelId {
        self.arena.model_id()
    }

    /// Accumulate affine terms incrementally without copying every prefix.
    /// Call `build()` to emit an expression and reset the reusable builder.
    pub fn affine_builder(&self) -> oximo_expr::AffineBuilder<'_> {
        oximo_expr::AffineBuilder::new(&self.arena)
    }

    /// Accumulate original numeric affine terms with compensated summation.
    /// This reduces cancellation error, but cannot prevent rounding already
    /// present in a supplied expression or an overflowed partial sum.
    pub fn compensated_affine_builder(&self) -> oximo_expr::AffineBuilder<'_> {
        oximo_expr::AffineBuilder::new_compensated(&self.arena)
    }

    /// Construct a constant expression for format readers and other adapters.
    #[doc(hidden)]
    pub fn __constant(&self, value: f64) -> Expr<'_, Constant> {
        Expr::constant(&self.arena, value)
    }

    /// Expression-only context captured by modeling macros before entering
    /// indexed callbacks.
    #[doc(hidden)]
    pub fn __sum_context(&self) -> &ExprArenaCell {
        &self.arena
    }

    /// Return an immutable copy-on-write snapshot of the expression arena.
    ///
    /// The snapshot is cheap to create, but it is not live.
    /// Subsequent model mutations (including parameter rebinding) are
    /// not visible through a snapshot that is already held.
    pub fn arena(&self) -> ExprArenaSnapshot<'_> {
        self.arena.borrow()
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

    // Objective

    /// Macro-facing entry point backing `objective!(m, Min, ..)`. Not part of the
    /// stable public API.
    #[doc(hidden)]
    pub fn __minimize<'a>(&self, expr: impl Into<Expr<'a>>) {
        let expr: Expr = expr.into();
        self.set_objective(expr, ObjectiveSense::Minimize);
    }

    /// Macro-facing entry point backing `objective!(m, Max, ..)`. Not part of the
    /// stable public API.
    #[doc(hidden)]
    pub fn __maximize<'a>(&self, expr: impl Into<Expr<'a>>) {
        let expr: Expr = expr.into();
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

    fn set_objective<'a>(&self, expr: impl Into<Expr<'a>>, sense: ObjectiveSense) {
        let expr: Expr = expr.into();
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
}
