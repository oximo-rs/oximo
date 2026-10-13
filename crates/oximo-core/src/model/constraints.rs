//! Constraint registration and typed registry inspection.

use super::{Model, validate_batch_names};
use crate::constraint::{
    AlgebraicConstraint, ConstraintHandle, ConstraintId, IntoRhs, RangeConstraintHandles, Relate,
};
use crate::domain::Domain;
use crate::function_set::{
    AlgebraicConstraintIr, Constraint, ConstraintIr, GreaterThan, Interval, IntoAffineConstraint,
    IntoAffineFunction, IntoFunction, LessThan, LowerConstraint, ScalarFunction, SecondOrderCone,
    Set as ConstraintSet, SocConstraintIr, VectorAffineFunction, validate_bounds,
};
use crate::indicator::{
    IndicatorConstraint, IndicatorConstraintHandle, IndicatorConstraintId,
    RangeIndicatorConstraintHandles,
};
use crate::soc::{SocConstraint, SocConstraintHandle, SocConstraintId};
use crate::sos::{
    SosConstraint, SosConstraintHandle, SosConstraintId, SosMember, SosType, validate_members,
};
use crate::{
    IntoSymmetricAffineFunction, PositiveSemidefiniteCone, PsdConstraint, PsdConstraintHandle,
    PsdConstraintId, PsdConstraintIr,
};
use oximo_expr::{
    Degree, Expr, ExprClass, ExprId, ExprIdRemap, ModelMismatchError, VarId, classify,
};
use smol_str::SmolStr;
use std::cell::Ref;

#[derive(Debug)]
pub(super) struct PendingConstraint {
    pub(super) name: SmolStr,
    lhs: ExprId,
    lower: f64,
    upper: f64,
}

#[derive(Debug)]
pub(super) struct PendingIndicator {
    name: SmolStr,
    trigger: VarId,
    active_value: bool,
    lhs: ExprId,
    lower: f64,
    upper: f64,
}

impl PendingConstraint {
    pub(super) fn from_expr<'a>(
        name: SmolStr,
        constraint: impl Into<AlgebraicConstraintIr<'a>>,
    ) -> Self {
        let constraint: AlgebraicConstraintIr = constraint.into();
        let (lower, upper) = (constraint.lower, constraint.upper);
        validate_bounds(lower, upper);
        Self { name, lhs: constraint.lhs.id, lower, upper }
    }

    pub(super) fn remap(&mut self, remap: ExprIdRemap) {
        self.lhs = remap.apply(self.lhs);
    }
}

#[derive(Debug)]
pub(super) struct PendingSoc {
    pub(super) name: SmolStr,
    pub(super) terms: Vec<ExprId>,
    pub(super) bound: ExprId,
}

impl PendingSoc {
    pub(super) fn remap(&mut self, remap: ExprIdRemap) {
        for term in &mut self.terms {
            *term = remap.apply(*term);
        }
        self.bound = remap.apply(self.bound);
    }
}

#[derive(Debug)]
pub(super) struct PendingSos {
    pub(super) name: SmolStr,
    pub(super) members: Vec<SosMember>,
}

/// A borrowed constraint from a [`Model`].
///
/// Algebraic, explicitly declared SOC/PSD, SOS, and indicator constraints
/// retain their typed IDs and storage. This enum provides a unified inspection
/// boundary without changing either representation.
#[derive(Copy, Clone, Debug)]
pub enum ConstraintRef<'a> {
    Algebraic { id: ConstraintId, constraint: &'a AlgebraicConstraint },
    PositiveSemidefinite { id: PsdConstraintId, constraint: &'a PsdConstraint },
    SecondOrderCone { id: SocConstraintId, constraint: &'a SocConstraint },
    SpecialOrderedSet { id: SosConstraintId, constraint: &'a SosConstraint },
    Indicator { id: IndicatorConstraintId, constraint: &'a IndicatorConstraint },
}

/// Unified borrowed view of every constraint declared on a [`Model`].
///
/// The underlying algebraic, explicit-SOC, PSD, SOS, and indicator registries remain
/// separate, so backends can iterate a homogeneous slice without a
/// per-constraint branch.
/// [`Self::iter`] visits algebraic constraints in [`ConstraintId`] order,
/// followed by explicit cones, SOS constraints, and indicator constraints in
/// their respective ID order.
#[derive(Debug)]
pub struct ModelConstraints<'a> {
    algebraic: Ref<'a, Vec<AlgebraicConstraint>>,
    second_order_cones: Ref<'a, Vec<SocConstraint>>,
    positive_semidefinite: Ref<'a, Vec<PsdConstraint>>,
    special_ordered_sets: Ref<'a, Vec<SosConstraint>>,
    indicators: Ref<'a, Vec<IndicatorConstraint>>,
}

impl ModelConstraints<'_> {
    /// Algebraic constraints in [`ConstraintId`] order.
    pub fn algebraic(&self) -> &[AlgebraicConstraint] {
        &self.algebraic
    }

    /// Explicit second-order-cone constraints in [`SocConstraintId`] order.
    pub fn second_order_cones(&self) -> &[SocConstraint] {
        &self.second_order_cones
    }

    /// PSD matrix constraints in stable ID order.
    pub fn positive_semidefinite(&self) -> &[PsdConstraint] {
        &self.positive_semidefinite
    }

    pub fn special_ordered_sets(&self) -> &[SosConstraint] {
        &self.special_ordered_sets
    }

    pub fn indicators(&self) -> &[IndicatorConstraint] {
        &self.indicators
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
        let indicators = self.indicators.iter().enumerate().map(|(index, constraint)| {
            ConstraintRef::Indicator { id: IndicatorConstraintId(index as u32), constraint }
        });
        let psd = self.positive_semidefinite.iter().enumerate().map(|(index, constraint)| {
            ConstraintRef::PositiveSemidefinite { id: PsdConstraintId(index as u32), constraint }
        });
        algebraic.chain(second_order_cones).chain(psd).chain(special_ordered_sets).chain(indicators)
    }

    /// Total number of algebraic, SOC, PSD, SOS, and indicator constraints.
    pub fn len(&self) -> usize {
        self.algebraic.len()
            + self.second_order_cones.len()
            + self.positive_semidefinite.len()
            + self.special_ordered_sets.len()
            + self.indicators.len()
    }

    pub fn is_empty(&self) -> bool {
        self.algebraic.is_empty()
            && self.second_order_cones.is_empty()
            && self.positive_semidefinite.is_empty()
            && self.special_ordered_sets.is_empty()
            && self.indicators.is_empty()
    }
}

impl Model {
    /// Register a mathematically compatible function/set pair in the heterogeneous IR.
    pub fn add_constraint<F, S>(
        &self,
        name: impl Into<SmolStr>,
        constraint: Constraint<F, S>,
    ) -> <F::Ir as ConstraintIr>::Handle
    where
        S: ConstraintSet,
        F: LowerConstraint<S>,
    {
        constraint.into_ir().register(self, name.into())
    }

    pub(crate) fn register_algebraic_ir(
        &self,
        name: SmolStr,
        ir: AlgebraicConstraintIr<'_>,
    ) -> ConstraintHandle {
        self.ensure_expr_model(ir.lhs).unwrap_or_else(|error| panic!("{error}"));
        let id = self.register_constraint(name, ir.lhs.id, ir.lower, ir.upper);
        ConstraintHandle::new(id, self.id())
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
    pub fn __add_constraint<'a>(
        &self,
        name: impl Into<SmolStr>,
        c: impl Into<AlgebraicConstraintIr<'a>>,
    ) -> ConstraintHandle {
        let c: AlgebraicConstraintIr = c.into();
        self.assert_expr_belongs(c.lhs);
        let (lower, upper) = (c.lower, c.upper);
        ConstraintHandle::new(
            self.register_constraint(name.into(), c.lhs.id, lower, upper),
            self.id(),
        )
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
        validate_bounds(lower, upper);
        let mut by_name = self.constraint_names.borrow_mut();
        assert!(!by_name.contains_key(&name), "constraint name {name:?} already registered");
        let mut all = self.constraints.borrow_mut();
        let id = ConstraintId(u32::try_from(all.len()).expect("constraint count overflow"));
        all.push(AlgebraicConstraint { name: name.clone(), lhs, lower, upper, active: true });
        by_name.insert(name, id);
        self.cached_kind.set(None);
        id
    }

    // Call only after name preflight, with no intervening user callbacks.
    pub(super) fn register_prevalidated_constraints_batch(
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
            constraints.push(AlgebraicConstraint {
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
    pub fn __add_constraint_auto<'a>(
        &self,
        c: impl Into<AlgebraicConstraintIr<'a>>,
    ) -> ConstraintHandle {
        let c: AlgebraicConstraintIr = c.into();
        self.__add_constraint(self.next_auto_name(), c)
    }

    /// Register a canonical interval row. This is intentionally hidden from
    /// the public builder API; file readers need to preserve native range rows.
    #[doc(hidden)]
    pub fn __add_constraint_interval<'a>(
        &self,
        name: impl Into<SmolStr>,
        lhs: impl Into<Expr<'a>>,
        lower: f64,
        upper: f64,
    ) -> ConstraintId {
        let lhs: Expr = lhs.into();
        self.assert_expr_belongs(lhs);
        self.add_constraint(name, Constraint::new(lhs.into_function(), Interval { lower, upper }))
            .id()
    }

    /// Bulk-register constraints. Each entry is `(name, ConstraintExpr)`.
    /// Useful with `.par_iter().map(...).collect()` style construction.
    pub fn add_constraints<'a, I, C>(&'a self, items: I)
    where
        I: IntoIterator<Item = (SmolStr, C)>,
        C: Into<AlgebraicConstraintIr<'a>>,
    {
        for (name, c) in items {
            self.__add_constraint(name, c);
        }
    }

    /// Macro-facing entry point for a two-sided range `lo <= mid <= hi`.
    ///
    /// Collapses to a single interval [`AlgebraicConstraint`] named `name` only when both
    /// bounds are pure constants and the body is linear (the condition under which
    /// one two-sided row is representable).
    #[doc(hidden)]
    pub fn __add_range<'a, B1, B2, D: crate::function_set::FunctionDegree>(
        &'a self,
        name: &str,
        mid: Expr<'a, D>,
        lo: B1,
        hi: B2,
    ) -> RangeConstraintHandles
    where
        B1: IntoRhs<'a, D>,
        B2: IntoRhs<'a, D>,
    {
        self.assert_expr_belongs(mid);
        if let (Some(lower), Some(upper)) = (lo.const_bound(), hi.const_bound()) {
            validate_bounds(lower, upper);
            let function = mid.into_function();
            if function.is_affine() {
                RangeConstraintHandles::Interval(
                    self.add_constraint(name, Constraint::new(function, Interval { lower, upper })),
                )
            } else {
                let lower = self.__add_constraint(
                    format!("{name}_lo"),
                    Constraint::new(function, GreaterThan(lower)),
                );
                let upper = self.__add_constraint(
                    format!("{name}_hi"),
                    Constraint::new(function, LessThan(upper)),
                );
                RangeConstraintHandles::Split { lower, upper }
            }
        } else {
            let lower = self.__add_constraint(format!("{name}_lo"), mid.ge(lo));
            let upper = self.__add_constraint(format!("{name}_hi"), mid.le(hi));
            RangeConstraintHandles::Split { lower, upper }
        }
    }

    /// Anonymous form of [`Self::__add_range`] (auto-named rows).
    #[doc(hidden)]
    pub fn __add_range_auto<'a, B1, B2, D: crate::function_set::FunctionDegree>(
        &'a self,
        mid: Expr<'a, D>,
        lo: B1,
        hi: B2,
    ) -> RangeConstraintHandles
    where
        B1: IntoRhs<'a, D>,
        B2: IntoRhs<'a, D>,
    {
        self.assert_expr_belongs(mid);
        if let (Some(lower), Some(upper)) = (lo.const_bound(), hi.const_bound()) {
            validate_bounds(lower, upper);
            let function = mid.into_function();
            if function.is_affine() {
                RangeConstraintHandles::Interval(self.add_constraint(
                    self.next_auto_name(),
                    Constraint::new(function, Interval { lower, upper }),
                ))
            } else {
                let lower =
                    self.__add_constraint_auto(Constraint::new(function, GreaterThan(lower)));
                let upper = self.__add_constraint_auto(Constraint::new(function, LessThan(upper)));
                RangeConstraintHandles::Split { lower, upper }
            }
        } else {
            let lower = self.__add_constraint_auto(mid.ge(lo));
            let upper = self.__add_constraint_auto(mid.le(hi));
            RangeConstraintHandles::Split { lower, upper }
        }
    }

    /// Unified view of every algebraic, explicit SOC, SOS, and indicator
    /// constraint declared on this model.
    pub fn constraints(&self) -> ModelConstraints<'_> {
        ModelConstraints {
            algebraic: self.constraints.borrow(),
            second_order_cones: self.soc_constraints.borrow(),
            positive_semidefinite: self.psd_constraints.borrow(),
            special_ordered_sets: self.sos_constraints.borrow(),
            indicators: self.indicator_constraints.borrow(),
        }
    }

    /// Total number of algebraic, SOC, PSD, SOS, and indicator constraints.
    pub fn num_constraints(&self) -> usize {
        self.constraints.borrow().len()
            + self.soc_constraints.borrow().len()
            + self.psd_constraints.borrow().len()
            + self.sos_constraints.borrow().len()
            + self.indicator_constraints.borrow().len()
    }

    pub fn constraint_id(&self, name: &str) -> Option<ConstraintId> {
        self.constraint_names.borrow().get(name).copied()
    }

    /// Return a model-bound handle for a named algebraic constraint.
    pub fn constraint_handle(&self, name: &str) -> Option<ConstraintHandle> {
        self.constraint_id(name).map(|id| ConstraintHandle::new(id, self.id()))
    }

    /// Bind a raw algebraic constraint ID to this model.
    pub fn constraint_handle_from_id(&self, id: ConstraintId) -> Option<ConstraintHandle> {
        (id.index() < self.constraints.borrow().len()).then(|| ConstraintHandle::new(id, self.id()))
    }

    /// Register an explicitly symmetric affine matrix in the real PSD cone.
    ///
    /// # Panics
    /// Panics for invalid ownership, non-affine dynamic entries, duplicate names, or count overflow.
    pub fn add_psd_constraint<'a>(
        &'a self,
        name: impl Into<SmolStr>,
        matrix: impl IntoSymmetricAffineFunction<'a>,
    ) -> PsdConstraintHandle {
        let function = matrix.into_symmetric_affine_function(self);
        let side_dimension = function.matrix().side_dimension();
        self.add_constraint(
            name,
            Constraint::new(function, PositiveSemidefiniteCone { side_dimension }),
        )
    }

    pub(crate) fn register_psd_ir(
        &self,
        name: SmolStr,
        ir: PsdConstraintIr<'_>,
    ) -> PsdConstraintHandle {
        for &entry in ir.0.matrix().upper_triangle() {
            self.assert_expr_belongs(entry);
        }
        let matrix = ir.0.matrix().map(|entry| entry.id());
        let mut names = self.psd_names.borrow_mut();
        validate_batch_names(&names, [&name], "PSD constraint", 1);
        let mut all = self.psd_constraints.borrow_mut();
        let id = PsdConstraintId(u32::try_from(all.len()).expect("PSD count overflow"));
        all.push(PsdConstraint { name: name.clone(), matrix, active: true });
        names.insert(name, id);
        self.invalidate_kind();
        PsdConstraintHandle::new(id, self.id())
    }

    #[doc(hidden)]
    pub fn __add_psd_constraint_auto<'a>(
        &'a self,
        matrix: impl IntoSymmetricAffineFunction<'a>,
    ) -> PsdConstraintHandle {
        loop {
            let n = self.auto_seq.get();
            self.auto_seq.set(n.checked_add(1).expect("constraint name count overflow"));
            let name = format!("_psd{n}");
            if !self.psd_names.borrow().contains_key(name.as_str()) {
                return self.add_psd_constraint(name, matrix);
            }
        }
    }

    pub fn psd_constraints(&self) -> Ref<'_, Vec<PsdConstraint>> {
        self.psd_constraints.borrow()
    }

    pub fn num_psd_constraints(&self) -> usize {
        self.psd_constraints.borrow().len()
    }

    pub fn has_active_psd_constraints(&self) -> bool {
        self.psd_constraints.borrow().iter().any(|c| c.active)
    }

    pub fn psd_constraint_id(&self, name: &str) -> Option<PsdConstraintId> {
        self.psd_names.borrow().get(name).copied()
    }

    pub fn psd_constraint_handle(&self, name: &str) -> Option<PsdConstraintHandle> {
        self.psd_constraint_id(name).map(|id| PsdConstraintHandle::new(id, self.id()))
    }

    pub fn psd_constraint_handle_from_id(
        &self,
        id: PsdConstraintId,
    ) -> Option<PsdConstraintHandle> {
        (id.index() < self.num_psd_constraints()).then(|| PsdConstraintHandle::new(id, self.id()))
    }

    /// Change PSD activity while invalidating the cached model classification.
    ///
    /// # Errors
    /// Returns an ownership error for a foreign model handle.
    ///
    /// # Panics
    /// Panics for an invalid constraint ID.
    pub fn set_psd_active(
        &self,
        handle: PsdConstraintHandle,
        active: bool,
    ) -> std::result::Result<(), ModelMismatchError> {
        self.ensure_model_id(handle.model_id())?;
        self.psd_constraints.borrow_mut()[handle.index()].active = active;
        self.invalidate_kind();
        Ok(())
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
    pub fn add_soc_constraint<'a, A: IntoAffineFunction<'a>, B: IntoAffineFunction<'a>>(
        &'a self,
        name: impl Into<SmolStr>,
        terms: impl IntoIterator<Item = A>,
        bound: B,
    ) -> SocConstraintHandle {
        let function = VectorAffineFunction::new(
            std::iter::once(bound.into_affine_function())
                .chain(terms.into_iter().map(IntoAffineFunction::into_affine_function)),
        );
        let dimension = function.dimension();
        self.add_constraint(name, Constraint::new(function, SecondOrderCone { dimension }))
    }

    pub(crate) fn register_soc_ir(
        &self,
        name: SmolStr,
        ir: SocConstraintIr<'_>,
    ) -> SocConstraintHandle {
        self.ensure_expr_model(ir.0.first).unwrap_or_else(|error| panic!("{error}"));
        let terms = ir.0.rest;
        let bound = ir.0.first;
        let mut by_name = self.soc_names.borrow_mut();
        assert!(!by_name.contains_key(&name), "SOC constraint name {name:?} already registered");
        let mut all = self.soc_constraints.borrow_mut();
        let id = SocConstraintId(u32::try_from(all.len()).expect("SOC constraint count overflow"));
        all.push(SocConstraint { name: name.clone(), terms, bound: bound.id, active: true });
        by_name.insert(name, id);
        self.cached_kind.set(None);
        SocConstraintHandle::new(id, self.id())
    }

    // Call only after name preflight, with no intervening user callbacks.
    pub(super) fn register_prevalidated_soc_batch(&self, items: Vec<PendingSoc>) {
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
    pub fn __add_soc_constraint_auto<'a, A: IntoAffineFunction<'a>, B: IntoAffineFunction<'a>>(
        &'a self,
        terms: impl IntoIterator<Item = A>,
        bound: B,
    ) -> SocConstraintHandle {
        self.add_soc_constraint(self.next_auto_soc_name(), terms, bound)
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

    /// Return a model-bound handle for a named explicit SOC constraint.
    pub fn soc_constraint_handle(&self, name: &str) -> Option<SocConstraintHandle> {
        self.soc_constraint_id(name).map(|id| SocConstraintHandle::new(id, self.id()))
    }

    /// Bind a raw explicit-SOC constraint ID to this model.
    pub fn soc_constraint_handle_from_id(
        &self,
        id: SocConstraintId,
    ) -> Option<SocConstraintHandle> {
        (id.index() < self.soc_constraints.borrow().len())
            .then(|| SocConstraintHandle::new(id, self.id()))
    }

    /// Whether the model carries any explicit SOC or PSD constraints.
    pub fn has_cones(&self) -> bool {
        !self.soc_constraints.borrow().is_empty() || !self.psd_constraints.borrow().is_empty()
    }

    /// Register an explicit SOS1 or SOS2 constraint. Members must be bare
    /// variables belonging to this model and have finite, unique weights.
    ///
    /// # Panics
    ///
    /// Panics when a member belongs to another model, is not a bare variable,
    /// or the name, members, variables, or weights violate SOS invariants.
    pub fn add_sos_constraint<'a, D: Degree>(
        &'a self,
        name: impl Into<SmolStr>,
        sos_type: SosType,
        members: impl IntoIterator<Item = (Expr<'a, D>, f64)>,
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
    pub(super) fn register_prevalidated_sos_batch(
        &self,
        sos_type: SosType,
        items: Vec<PendingSos>,
    ) {
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

    pub(super) fn register_sos_one(&self, sos_type: SosType, item: PendingSos) {
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
    pub fn add_sos_constraint_auto_weights<'a, D: Degree>(
        &'a self,
        name: impl Into<SmolStr>,
        sos_type: SosType,
        variables: impl IntoIterator<Item = Expr<'a, D>>,
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
    pub fn __add_sos_constraint_auto<'a, D: Degree>(
        &'a self,
        sos_type: SosType,
        members: impl IntoIterator<Item = (Expr<'a, D>, f64)>,
    ) -> SosConstraintHandle<'a> {
        self.add_sos_constraint(self.next_auto_sos_name(), sos_type, members)
    }

    #[doc(hidden)]
    pub fn __add_sos_constraint_auto_weights<'a, D: Degree>(
        &'a self,
        sos_type: SosType,
        variables: impl IntoIterator<Item = Expr<'a, D>>,
    ) -> SosConstraintHandle<'a> {
        self.add_sos_constraint_auto_weights(self.next_auto_sos_name(), sos_type, variables)
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

    // Indicator constraints

    pub(super) fn indicator_trigger<'a>(&self, trigger: impl Into<Expr<'a>>) -> VarId {
        let trigger: Expr = trigger.into();
        self.assert_expr_belongs(trigger);
        let id = trigger.var_id().expect("indicator trigger must be a bare binary variable");
        assert!(
            self.variables.borrow()[id.index()].domain == Domain::Binary,
            "indicator trigger must have binary domain"
        );
        id
    }

    fn register_indicator(
        &self,
        name: SmolStr,
        trigger: VarId,
        active_value: bool,
        lhs: ExprId,
        lower: f64,
        upper: f64,
    ) -> IndicatorConstraintId {
        validate_bounds(lower, upper);
        let class = classify(&self.arena.borrow(), lhs);
        assert!(class == ExprClass::Linear, "indicator consequent must be affine, got {class:?}");
        let mut names = self.indicator_names.borrow_mut();
        assert!(
            !names.contains_key(&name),
            "indicator constraint name {name:?} already registered"
        );
        let mut all = self.indicator_constraints.borrow_mut();
        let id = IndicatorConstraintId(
            u32::try_from(all.len()).expect("indicator constraint count overflow"),
        );
        all.push(IndicatorConstraint {
            name: name.clone(),
            trigger,
            active_value,
            lhs,
            lower,
            upper,
            active: true,
        });
        names.insert(name, id);
        self.cached_kind.set(None);
        id
    }

    pub(super) fn prepare_indicator<'a>(
        &self,
        name: SmolStr,
        trigger: VarId,
        active_value: bool,
        consequent: impl IntoAffineConstraint<'a>,
    ) -> PendingIndicator {
        let consequent = consequent.into_affine_constraint().0;
        self.assert_expr_belongs(consequent.lhs);
        let (lower, upper) = (consequent.lower, consequent.upper);
        self.prepare_indicator_interval(
            name,
            trigger,
            active_value,
            consequent.lhs.id,
            lower,
            upper,
        )
    }

    pub(super) fn prepare_indicator_interval(
        &self,
        name: SmolStr,
        trigger: VarId,
        active_value: bool,
        lhs: ExprId,
        lower: f64,
        upper: f64,
    ) -> PendingIndicator {
        validate_bounds(lower, upper);
        assert!(
            classify(&self.arena.borrow(), lhs) == ExprClass::Linear,
            "indicator consequent must be affine"
        );
        PendingIndicator { name, trigger, active_value, lhs, lower, upper }
    }

    pub(super) fn register_indicators_batch(
        &self,
        items: Vec<PendingIndicator>,
    ) -> Vec<IndicatorConstraintId> {
        let mut names = self.indicator_names.borrow_mut();
        validate_batch_names(
            &names,
            items.iter().map(|item| &item.name),
            "indicator constraint",
            items.len(),
        );
        let mut constraints = self.indicator_constraints.borrow_mut();
        let final_count = constraints
            .len()
            .checked_add(items.len())
            .expect("indicator constraint count overflow");
        if final_count > 0 {
            u32::try_from(final_count - 1).expect("indicator constraint count overflow");
        }
        constraints.reserve(items.len());
        names.reserve(items.len());
        let mut ids = Vec::with_capacity(items.len());
        for item in items {
            let id = IndicatorConstraintId(
                u32::try_from(constraints.len()).expect("indicator constraint count overflow"),
            );
            constraints.push(IndicatorConstraint {
                name: item.name.clone(),
                trigger: item.trigger,
                active_value: item.active_value,
                lhs: item.lhs,
                lower: item.lower,
                upper: item.upper,
                active: true,
            });
            names.insert(item.name, id);
            ids.push(id);
        }
        if !ids.is_empty() {
            self.cached_kind.set(None);
        }
        ids
    }

    /// Register `trigger == active_value => consequent`.
    ///
    /// # Panics
    /// Panics for invalid triggers, non-affine dynamic consequents, invalid bounds,
    /// foreign ownership, duplicate names, or count overflow.
    pub fn add_indicator_constraint<'a>(
        &'a self,
        name: impl Into<SmolStr>,
        trigger: impl Into<Expr<'a>>,
        active_value: bool,
        consequent: impl IntoAffineConstraint<'a>,
    ) -> IndicatorConstraintHandle<'a> {
        let trigger = self.indicator_trigger(trigger);
        let consequent = consequent.into_affine_constraint().0;
        self.assert_expr_belongs(consequent.lhs);
        let id = self.register_indicator(
            name.into(),
            trigger,
            active_value,
            consequent.lhs.id,
            consequent.lower,
            consequent.upper,
        );
        IndicatorConstraintHandle { model: self, id }
    }

    #[doc(hidden)]
    pub fn __add_indicator_constraint<'a>(
        &'a self,
        name: impl Into<SmolStr>,
        trigger: impl Into<Expr<'a>>,
        active_value: bool,
        consequent: impl IntoAffineConstraint<'a>,
    ) -> IndicatorConstraintHandle<'a> {
        let trigger: Expr = trigger.into();
        self.add_indicator_constraint(name, trigger, active_value, consequent)
    }

    fn next_auto_indicator_name(&self) -> SmolStr {
        loop {
            let n = self.auto_seq.get();
            self.auto_seq.set(n + 1);
            let candidate: SmolStr = format!("_ind{n}").into();
            if !self.indicator_names.borrow().contains_key(&candidate) {
                break candidate;
            }
        }
    }

    #[doc(hidden)]
    pub fn __add_indicator_constraint_auto<'a>(
        &'a self,
        trigger: impl Into<Expr<'a>>,
        active_value: bool,
        consequent: impl IntoAffineConstraint<'a>,
    ) -> IndicatorConstraintHandle<'a> {
        let trigger: Expr = trigger.into();
        self.add_indicator_constraint(
            self.next_auto_indicator_name(),
            trigger,
            active_value,
            consequent,
        )
    }

    #[doc(hidden)]
    pub fn __add_indicator_interval<'a>(
        &'a self,
        name: impl Into<SmolStr>,
        trigger: impl Into<Expr<'a>>,
        active_value: bool,
        lhs: impl Into<Expr<'a>>,
        lower: f64,
        upper: f64,
    ) -> IndicatorConstraintHandle<'a> {
        let trigger: Expr = trigger.into();
        let lhs: Expr = lhs.into();
        self.assert_expr_belongs(lhs);
        let trigger = self.indicator_trigger(trigger.erase());
        let id = self.register_indicator(name.into(), trigger, active_value, lhs.id, lower, upper);
        IndicatorConstraintHandle { model: self, id }
    }

    #[doc(hidden)]
    pub fn __add_indicator_range<'a, B1: IntoRhs<'a>, B2: IntoRhs<'a>>(
        &'a self,
        name: &str,
        trigger: impl Into<Expr<'a>>,
        active_value: bool,
        mid: impl IntoAffineFunction<'a>,
        lo: B1,
        hi: B2,
    ) -> RangeIndicatorConstraintHandles<'a> {
        let trigger: Expr = trigger.into();
        let mid = mid.into_affine_function().expression().erase();
        self.assert_expr_belongs(mid);
        let trigger = self.indicator_trigger(trigger.erase());
        if let (Some(lower), Some(upper)) = (lo.const_bound(), hi.const_bound())
            && mid.__class() == ExprClass::Linear
        {
            let pending = self.prepare_indicator_interval(
                name.into(),
                trigger,
                active_value,
                mid.id,
                lower,
                upper,
            );
            let id = self.register_indicators_batch(vec![pending])[0];
            RangeIndicatorConstraintHandles::Interval(IndicatorConstraintHandle { model: self, id })
        } else {
            let lower = self.prepare_indicator(
                format!("{name}_lo").into(),
                trigger,
                active_value,
                mid.ge(lo).into_ir().into_affine(),
            );
            let upper = self.prepare_indicator(
                format!("{name}_hi").into(),
                trigger,
                active_value,
                mid.le(hi).into_ir().into_affine(),
            );
            let mut ids = self.register_indicators_batch(vec![lower, upper]).into_iter();
            let lower = IndicatorConstraintHandle {
                model: self,
                id: ids.next().expect("lower indicator range ID missing"),
            };
            let upper = IndicatorConstraintHandle {
                model: self,
                id: ids.next().expect("upper indicator range ID missing"),
            };
            RangeIndicatorConstraintHandles::Split { lower, upper }
        }
    }

    #[doc(hidden)]
    pub fn __add_indicator_range_auto<'a, B1: IntoRhs<'a>, B2: IntoRhs<'a>>(
        &'a self,
        trigger: impl Into<Expr<'a>>,
        active_value: bool,
        mid: impl IntoAffineFunction<'a>,
        lo: B1,
        hi: B2,
    ) -> RangeIndicatorConstraintHandles<'a> {
        let trigger: Expr = trigger.into();
        let mid = mid.into_affine_function().expression().erase();
        let name = self.next_auto_indicator_name();
        self.__add_indicator_range(&name, trigger, active_value, mid, lo, hi)
    }

    pub fn indicator_constraints(&self) -> Ref<'_, Vec<IndicatorConstraint>> {
        self.indicator_constraints.borrow()
    }

    pub fn num_indicator_constraints(&self) -> usize {
        self.indicator_constraints.borrow().len()
    }

    pub fn indicator_constraint_id(&self, name: &str) -> Option<IndicatorConstraintId> {
        self.indicator_names.borrow().get(name).copied()
    }

    pub fn has_indicator_constraints(&self) -> bool {
        !self.indicator_constraints.borrow().is_empty()
    }

    pub fn has_active_indicator_constraints(&self) -> bool {
        self.indicator_constraints.borrow().iter().any(|c| c.active)
    }
}
