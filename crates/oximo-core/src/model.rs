mod classify;
mod constraints;
mod indexed;
mod semantic;
mod variables;

pub use classify::ModelKind;
pub use constraints::{ConstraintRef, ModelConstraints};
pub use indexed::{IndexedVarBuilder, display_index_key};
pub use semantic::SemanticConstraints;

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
use crate::shared::SharedCell;
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

fn validate_batch_names<'a>(
    existing: impl Fn(&SmolStr) -> bool,
    names: impl IntoIterator<Item = &'a SmolStr>,
    kind: &str,
    count: usize,
) {
    let mut batch = FxHashSet::with_capacity_and_hasher(count, FxBuildHasher);

    for name in names {
        assert!(!existing(name), "{kind} name {name:?} already registered");
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
    pub(crate) variables: SharedCell<Vec<Variable>>,
    pub(crate) var_names: SharedCell<FxHashMap<SmolStr, VarId>>,
    pub(crate) parameters: SharedCell<Vec<Parameter>>,
    pub(crate) param_names: SharedCell<FxHashMap<SmolStr, ParamId>>,
    pub(crate) semantics: SharedCell<semantic::SemanticRegistry>,
    implicit_scalars: Cell<bool>,
    scalar_declarations: Cell<usize>,
    pub(crate) alternative_scalar_views:
        SharedCell<FxHashMap<crate::SemanticConstraintId, crate::semantic::ScalarConstraintView>>,
    pub(crate) inactive_scalar_views:
        SharedCell<FxHashMap<ConstraintId, crate::semantic::ScalarConstraintView>>,
    pub(crate) constraints: SharedCell<Vec<AlgebraicConstraint>>,
    pub(crate) semantic_names: SharedCell<semantic::ConstraintNames>,
    pub(crate) soc_constraints: SharedCell<Vec<SocConstraint>>,
    pub(crate) psd_constraints: SharedCell<Vec<PsdConstraint>>,
    pub(crate) sos_constraints: SharedCell<Vec<SosConstraint>>,
    pub(crate) indicator_constraints: SharedCell<Vec<IndicatorConstraint>>,
    pub(crate) sos_reformulations: SharedCell<Vec<SosReformulationArtifacts>>,
    pub(crate) indicator_reformulations: SharedCell<Vec<IndicatorReformulationArtifacts>>,
    pub(crate) semantic_objective: RefCell<Option<crate::objective::SemanticObjective>>,
    pub(crate) objective: RefCell<Option<Objective>>,
    objective_declared: Cell<bool>,
    cached_kind: Cell<Option<ModelKind>>,
    cached_kind_parameter_revision: Cell<u64>,
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
    pub(crate) fn ensure_model_id(
        &self,
        actual: ModelId,
    ) -> std::result::Result<(), ModelMismatchError> {
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

    /// Share registries until mutation while preserving stable IDs and independent ownership.
    pub(crate) fn clone_preserving_ids_with_capacity(
        &self,
        additional_variables: usize,
        additional_constraints: usize,
        additional_expr_nodes: usize,
    ) -> Self {
        let cloned = Self {
            name: self.name.clone(),
            #[cfg(feature = "gdp")]
            gdp: RefCell::new(self.gdp.borrow().clone()),
            arena: ExprArenaCell::new(if additional_expr_nodes == 0 {
                (*self.arena.borrow()).clone()
            } else {
                self.arena.borrow().__clone_with_additional_capacity(additional_expr_nodes)
            }),
            variables: self.variables.clone(),
            var_names: self.var_names.clone(),
            parameters: self.parameters.clone(),
            param_names: self.param_names.clone(),
            semantics: self.semantics.clone(),
            implicit_scalars: Cell::new(self.implicit_scalars.get()),
            scalar_declarations: Cell::new(self.scalar_declarations.get()),
            alternative_scalar_views: self.alternative_scalar_views.clone(),
            inactive_scalar_views: self.inactive_scalar_views.clone(),
            constraints: self.constraints.clone(),
            semantic_names: self.semantic_names.clone(),
            soc_constraints: self.soc_constraints.clone(),
            psd_constraints: self.psd_constraints.clone(),
            sos_constraints: self.sos_constraints.clone(),
            indicator_constraints: self.indicator_constraints.clone(),
            sos_reformulations: self.sos_reformulations.clone(),
            indicator_reformulations: self.indicator_reformulations.clone(),
            semantic_objective: RefCell::new(self.semantic_objective.borrow().clone()),
            objective: RefCell::new(self.objective.borrow().clone()),
            objective_declared: Cell::new(self.objective_declared.get()),
            cached_kind: Cell::new(self.cached_kind.get()),
            cached_kind_parameter_revision: Cell::new(self.cached_kind_parameter_revision.get()),
            auto_seq: Cell::new(self.auto_seq.get()),
        };

        if let Some(objective) = cloned.semantic_objective.borrow_mut().as_mut() {
            objective.function.rebind_model(cloned.id());
        }

        if !cloned.semantics.borrow().extensions.is_empty() {
            cloned.semantics.borrow_mut().rebind_extensions(cloned.id());
        }

        if additional_variables != 0 {
            drop(cloned.variables.borrow_mut_with_capacity(additional_variables));
            cloned.var_names.borrow_mut().reserve(additional_variables);
        }

        if additional_constraints != 0 {
            drop(cloned.constraints.borrow_mut_with_capacity(additional_constraints));
            cloned.semantic_names.borrow_mut().reserve(additional_constraints);
        }

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
            variables: SharedCell::new(Vec::new()),
            var_names: SharedCell::new(FxHashMap::default()),
            parameters: SharedCell::new(Vec::new()),
            param_names: SharedCell::new(FxHashMap::default()),
            semantics: SharedCell::new(semantic::SemanticRegistry::default()),
            implicit_scalars: Cell::new(true),
            scalar_declarations: Cell::new(0),
            alternative_scalar_views: SharedCell::new(FxHashMap::default()),
            inactive_scalar_views: SharedCell::new(FxHashMap::default()),
            constraints: SharedCell::new(Vec::new()),
            semantic_names: SharedCell::new(semantic::ConstraintNames::default()),
            soc_constraints: SharedCell::new(Vec::new()),
            psd_constraints: SharedCell::new(Vec::new()),
            sos_constraints: SharedCell::new(Vec::new()),
            indicator_constraints: SharedCell::new(Vec::new()),
            sos_reformulations: SharedCell::new(Vec::new()),
            indicator_reformulations: SharedCell::new(Vec::new()),
            semantic_objective: RefCell::new(None),
            objective: RefCell::new(None),
            objective_declared: Cell::new(false),
            cached_kind: Cell::new(None),
            cached_kind_parameter_revision: Cell::new(0),
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
    pub fn __minimize<'a>(&'a self, expr: impl crate::objective::IntoObjectiveExpression<'a>) {
        self.set_objective(expr, ObjectiveSense::Minimize);
    }

    /// Macro-facing entry point backing `objective!(m, Max, ..)`. Not part of the
    /// stable public API.
    #[doc(hidden)]
    pub fn __maximize<'a>(&'a self, expr: impl crate::objective::IntoObjectiveExpression<'a>) {
        self.set_objective(expr, ObjectiveSense::Maximize);
    }

    /// Macro-facing entry point backing `objective!(m, Feasibility)`. Declares
    /// the model a feasibility problem (no objective to optimize), clearing any
    /// previously set objective. Not part of the stable public API.
    #[doc(hidden)]
    pub fn __feasibility(&self) {
        *self.semantic_objective.borrow_mut() = None;
        *self.objective.borrow_mut() = None;
        self.objective_declared.set(true);
        self.cached_kind.set(None);
    }

    fn set_objective<'a, E: crate::objective::IntoObjectiveExpression<'a>>(
        &'a self,
        expr: E,
        sense: ObjectiveSense,
    ) {
        let expr = expr.into_objective_expression(&self.arena);
        let semantic_function =
            <E::Degree as crate::function_set::FunctionDegree>::relation_function(expr);
        let function =
            crate::FunctionValue::new(crate::Function::semantic_data(&semantic_function));
        self.assert_expr_belongs(expr);
        *self.semantic_objective.borrow_mut() =
            Some(crate::objective::SemanticObjective { function, sense });
        *self.objective.borrow_mut() = Some(Objective { expr: expr.id, sense });
        self.objective_declared.set(true);
        self.cached_kind.set(None);
    }

    /// Whether feasibility was declared explicitly via `objective!(m, Feasibility)`,
    /// as opposed to a model that simply has no objective set.
    pub fn is_feasibility(&self) -> bool {
        self.objective_declared.get() && self.semantic_objective.borrow().is_none()
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

impl Model {
    pub fn semantic_objective(&self) -> Ref<'_, Option<crate::objective::SemanticObjective>> {
        self.semantic_objective.borrow()
    }

    pub fn minimize<F: crate::objective::ObjectiveFunction>(&self, function: F) {
        self.set_semantic_objective(function, ObjectiveSense::Minimize);
    }

    pub fn maximize<F: crate::objective::ObjectiveFunction>(&self, function: F) {
        self.set_semantic_objective(function, ObjectiveSense::Maximize);
    }

    fn set_semantic_objective<F: crate::objective::ObjectiveFunction>(
        &self,
        function: F,
        sense: ObjectiveSense,
    ) {
        let function = crate::FunctionValue::new(function.semantic_data());
        assert_eq!(function.model_id(), self.id(), "objective belongs to a different model");
        function.visit_expressions(&mut |root| {
            assert!(
                root.index() < self.arena.borrow().len(),
                "objective contains invalid expression IDs"
            );
        });

        let view = match function.representation() {
            crate::FunctionRepresentationRef::Scalar(expr) => Objective { expr, sense },
            _ => panic!(
                "objective requires a scalar expression representation; opaque registration is deferred"
            ),
        };
        *self.objective.borrow_mut() = Some(view);
        *self.semantic_objective.borrow_mut() =
            Some(crate::objective::SemanticObjective { function, sense });
        self.objective_declared.set(true);
        self.invalidate_kind();
    }
}

impl Model {
    pub(crate) fn constraint_names(&self) -> semantic::NamesRef<'_, ConstraintId> {
        semantic::NamesRef::new(self.semantic_names.borrow())
    }

    pub(crate) fn constraint_names_mut(&self) -> semantic::NamesMut<'_, ConstraintId> {
        semantic::NamesMut::new(self.semantic_names.borrow_mut())
    }

    pub(crate) fn soc_names(&self) -> semantic::NamesRef<'_, SocConstraintId> {
        semantic::NamesRef::new(self.semantic_names.borrow())
    }

    pub(crate) fn soc_names_mut(&self) -> semantic::NamesMut<'_, SocConstraintId> {
        semantic::NamesMut::new(self.semantic_names.borrow_mut())
    }

    pub(crate) fn psd_names(&self) -> semantic::NamesRef<'_, PsdConstraintId> {
        semantic::NamesRef::new(self.semantic_names.borrow())
    }

    pub(crate) fn psd_names_mut(&self) -> semantic::NamesMut<'_, PsdConstraintId> {
        semantic::NamesMut::new(self.semantic_names.borrow_mut())
    }

    pub(crate) fn sos_names(&self) -> semantic::NamesRef<'_, SosConstraintId> {
        semantic::NamesRef::new(self.semantic_names.borrow())
    }

    pub(crate) fn sos_names_mut(&self) -> semantic::NamesMut<'_, SosConstraintId> {
        semantic::NamesMut::new(self.semantic_names.borrow_mut())
    }

    pub(crate) fn indicator_names(&self) -> semantic::NamesRef<'_, IndicatorConstraintId> {
        semantic::NamesRef::new(self.semantic_names.borrow())
    }

    pub(crate) fn indicator_names_mut(&self) -> semantic::NamesMut<'_, IndicatorConstraintId> {
        semantic::NamesMut::new(self.semantic_names.borrow_mut())
    }
}
