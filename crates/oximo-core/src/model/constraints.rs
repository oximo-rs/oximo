//! Constraint registration and typed registry inspection.

use super::{Model, validate_batch_names};
use crate::constraint::{
    AlgebraicConstraint, ConstraintHandle, ConstraintId, IntoRhs, RangeConstraintHandles, Relate,
};
use crate::domain::Domain;
use crate::function_set::{
    AlgebraicConstraintIr, Constraint, GreaterThan, Interval, IntoAffineConstraint,
    IntoAffineFunction, IntoFunction, LessThan, ScalarFunction, Set as ConstraintSet,
    SocConstraintIr, validate_bounds,
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
    IntoSymmetricAffineFunction, PsdConstraint, PsdConstraintHandle, PsdConstraintId,
    PsdConstraintIr,
};
use oximo_expr::{
    Degree, Expr, ExprClass, ExprId, ExprIdRemap, ModelMismatchError, VarId, classify,
};
use smol_str::SmolStr;
use std::cell::Ref;

/// Append an algebraic row, growing storage in chunks when capacity is exhausted.
#[inline]
pub(super) fn push_algebraic(rows: &mut Vec<AlgebraicConstraint>, row: AlgebraicConstraint) {
    if rows.len() == rows.capacity() {
        grow_algebraic(rows);
    }
    rows.push(row);
}

/// Grow algebraic row storage in larger chunks for individual insertions.
#[cold]
#[inline(never)]
fn grow_algebraic(rows: &mut Vec<AlgebraicConstraint>) {
    let additional = rows.capacity().checked_mul(3).expect("constraint capacity overflow").max(8);
    rows.reserve(additional);
}

#[derive(Debug)]
enum PendingSemantic {
    Direct(super::semantic::ScalarSignature),
    Extension(Box<crate::SemanticPair>),
}

impl PendingSemantic {
    /// Return the compact source signature when the row uses a built-in representation.
    fn signature(&self) -> Option<super::semantic::ScalarSignature> {
        match self {
            Self::Direct(signature) => Some(*signature),
            Self::Extension(_) => None,
        }
    }

    /// Keep a compact signature for direct rows, or retain the full source pair.
    #[inline]
    fn new(
        source: crate::semantic::ScalarConstraintSource,
        view: crate::semantic::ScalarConstraintView,
    ) -> Self {
        if let Some(signature) = source.direct_signature(view) {
            Self::Direct(signature)
        } else {
            Self::Extension(Box::new(source.into_pair()))
        }
    }

    /// Record an appended row's source function/set identity and provenance.
    #[inline]
    fn record(
        self,
        writer: &mut super::semantic::SemanticWriter<'_>,
        id: ConstraintId,
        name: &SmolStr,
        origin: PendingOrigin,
        alternative: bool,
    ) {
        match self {
            Self::Direct(_) => match origin {
                PendingOrigin::Declared => {
                    writer.record_declared(id.into(), super::semantic::Representation::Scalar);
                }
                PendingOrigin::RangeName => {
                    writer.record_row_range(id.into(), super::semantic::Representation::Scalar);
                }
                origin @ PendingOrigin::Range(_) => writer.record_native(
                    id.into(),
                    name.clone(),
                    super::semantic::Representation::Scalar,
                    origin.into_origin(name),
                ),
            },
            Self::Extension(pair) => {
                writer.record_pair(
                    id.into(),
                    name.clone(),
                    *pair,
                    origin.into_origin(name),
                    !alternative,
                );
            }
        }
    }

    /// Record a directly declared row using its compact signature or retained source pair.
    #[inline]
    fn record_declared(self, model: &Model, id: ConstraintId, name: &SmolStr, alternative: bool) {
        match self {
            Self::Direct(_) => {
                model.record_declared_scalar(id);
            }
            Self::Extension(pair) => model.semantic_writer().record_pair(
                id.into(),
                name.clone(),
                *pair,
                crate::ConstraintOrigin::Declared,
                !alternative,
            ),
        }
    }

    /// Remap expression IDs in a retained source pair after merging a worker arena.
    fn remap(&mut self, remap: ExprIdRemap) {
        if let Self::Extension(pair) = self {
            pair.remap(remap);
        }
    }

    /// Check whether the stored row differs from the original function/set pair.
    #[inline]
    fn alternative(&self, view: crate::semantic::ScalarConstraintView) -> bool {
        match self {
            Self::Direct(_) => false,
            Self::Extension(pair) => !view.matches_pair(pair),
        }
    }
}

#[derive(Debug)]
pub(super) enum PendingOrigin {
    Declared,
    RangeName,
    Range(std::sync::Arc<SmolStr>),
}

impl PendingOrigin {
    /// Resolve a pending declaration or range origin to its stored provenance.
    fn into_origin(self, name: &SmolStr) -> crate::ConstraintOrigin {
        match self {
            Self::Declared => crate::ConstraintOrigin::Declared,
            Self::RangeName => crate::ConstraintOrigin::Range(name.clone()),
            Self::Range(name) => {
                crate::ConstraintOrigin::Range(std::sync::Arc::unwrap_or_clone(name))
            }
        }
    }
}

#[derive(Debug)]
pub(super) struct PendingConstraint {
    pub(super) name: SmolStr,
    lhs: ExprId,
    lower: f64,
    upper: f64,
    semantic: PendingSemantic,
    alternative: bool,
    pub(super) origin: PendingOrigin,
}

#[derive(Debug)]
pub(super) struct PendingIndicator {
    semantic: crate::SemanticPair,
    alternative: bool,
    name: SmolStr,
    trigger: VarId,
    active_value: bool,
    lhs: ExprId,
    lower: f64,
    upper: f64,
}

impl PendingConstraint {
    /// Validate scalar bounds and retain the row and source pair for deferred registration.
    pub(super) fn from_expr(
        name: SmolStr,
        constraint: impl crate::function_set::ScalarConstraintInput,
    ) -> Self {
        let constraint = constraint.into_scalar_input();
        let (lower, upper) = (constraint.view.lower, constraint.view.upper);
        validate_bounds(lower, upper);

        Self {
            name,
            lhs: constraint.view.lhs,
            lower,
            upper,
            semantic: PendingSemantic::new(
                constraint.semantic,
                crate::semantic::ScalarConstraintView { lhs: constraint.view.lhs, lower, upper },
            ),
            alternative: false,
            origin: PendingOrigin::Declared,
        }
    }

    /// Remap the stored row and its source pair, then refresh the lowering comparison.
    pub(super) fn remap(&mut self, remap: ExprIdRemap) {
        self.lhs = remap.apply(self.lhs);
        self.semantic.remap(remap);
        self.alternative = self.semantic.alternative(crate::semantic::ScalarConstraintView {
            lhs: self.lhs,
            lower: self.lower,
            upper: self.upper,
        });
    }
}

#[derive(Debug)]
pub(super) struct PendingSoc {
    pub(super) name: SmolStr,
    pub(super) components: crate::SharedSlice<ExprId>,
}

impl PendingSoc {
    /// Collect the affine SOC bound and terms while checking shared model ownership.
    pub(super) fn new<'a, A: IntoAffineFunction<'a>, B: IntoAffineFunction<'a>>(
        name: SmolStr,
        terms: impl IntoIterator<Item = A>,
        bound: B,
        expected_arena: usize,
    ) -> Self {
        let bound = bound.into_affine_function().expression();
        super::assert_expr_arena(bound, expected_arena);
        let components: crate::SharedSlice<_> = std::iter::once(bound.id())
            .chain(terms.into_iter().map(|term| {
                let term = term.into_affine_function().expression();
                assert_eq!(
                    super::arena_key(term.arena),
                    expected_arena,
                    "SOC components must belong to the same model"
                );
                term.id()
            }))
            .collect();
        assert!(components.len() >= 2, "SOC needs at least two components");
        Self { name, components }
    }

    /// Remap every SOC component after merging a worker arena.
    pub(super) fn remap(&mut self, remap: ExprIdRemap) {
        for root in &mut self.components {
            *root = remap.apply(*root);
        }
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

/// Borrowed homogeneous algebraic, cone, SOS and indicator views of a [`Model`].
///
/// The underlying algebraic, explicit-SOC, PSD, SOS, and indicator registries remain
/// separate, so backends can iterate a homogeneous slice without a
/// per-constraint branch.
/// Use [`Model::semantic_constraints`] to inspect the original function/set identity.
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

    /// Borrow SOS constraints in their registry ID order, including inactive sources.
    pub fn special_ordered_sets(&self) -> &[SosConstraint] {
        &self.special_ordered_sets
    }

    /// Borrow indicator constraints in their registry ID order, including inactive sources.
    pub fn indicators(&self) -> &[IndicatorConstraint] {
        &self.indicators
    }

    /// Iterate over all common constraint views without allocating.
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

    /// Return whether all constraint registries are empty.
    pub fn is_empty(&self) -> bool {
        self.algebraic.is_empty()
            && self.second_order_cones.is_empty()
            && self.positive_semidefinite.is_empty()
            && self.special_ordered_sets.is_empty()
            && self.indicators.is_empty()
    }
}

impl Model {
    /// Register a semantic pair with a recognized algebraic, conic, SOS, or indicator view.
    ///
    /// # Panics
    /// Panics for an unsupported representation, invalid data, or a duplicate registry name.
    pub fn add_constraint<F, S>(
        &self,
        name: impl Into<SmolStr>,
        constraint: Constraint<F, S>,
    ) -> ConstraintHandle<F::Owned, S>
    where
        S: ConstraintSet,
        F: crate::FunctionInSet<S>,
    {
        use crate::semantic::ScalarConstraintSource;
        let name = name.into();
        let scalar =
            match ScalarConstraintSource::capture_unshared(&constraint.function, &constraint.set) {
                Ok(source) => crate::function_set::PreparedScalarConstraint::declared(source),
                Err(pair) => {
                    let id = self.register_semantic_pair(name, pair);
                    return ConstraintHandle::new(id, self.id());
                }
            };
        let handle = self.__add_constraint(name, scalar);
        ConstraintHandle::new(handle.id(), self.id())
    }

    /// Register an explicitly lowered representation. Scalar IR retains its source declaration.
    ///
    /// # Panics
    /// Panics for invalid data, ownership, or duplicate names.
    pub fn add_lowered_constraint<I: crate::function_set::ConstraintIr>(
        &self,
        name: impl Into<SmolStr>,
        ir: I,
    ) -> I::Handle {
        ir.register(self, name.into())
    }

    /// Check model ownership and register a lowered scalar constraint.
    pub(crate) fn register_algebraic_ir(
        &self,
        name: SmolStr,
        ir: AlgebraicConstraintIr<'_>,
    ) -> crate::ScalarConstraintHandle {
        self.__add_constraint(name, ir)
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
    #[inline]
    pub fn __add_constraint<C: crate::function_set::ScalarConstraintInput>(
        &self,
        name: impl Into<SmolStr>,
        c: C,
    ) -> ConstraintHandle<C::Function, C::Set> {
        let c = c.into_scalar_input();
        c.validate(self.id(), self.arena.__node_count());
        let id = self.register_constraint(
            name.into(),
            c.view.lhs,
            c.view.lower,
            c.view.upper,
            c.semantic,
        );
        ConstraintHandle::new(id, self.id())
    }

    /// Push a constraint row `lower <= lhs <= upper` into the registry. Shared by
    /// [`Self::__add_constraint`] and the range entry points.
    ///
    /// # Panics
    ///
    /// Panics if a constraint with the same name is already registered, if a
    /// bound is NaN, or if the constraint count exceeds `u32::MAX`.
    #[inline]
    fn register_constraint(
        &self,
        name: SmolStr,
        lhs: ExprId,
        lower: f64,
        upper: f64,
        semantic: crate::semantic::ScalarConstraintSource,
    ) -> ConstraintId {
        assert_eq!(semantic.model_id(), self.id(), "function belongs to a different model");
        validate_bounds(lower, upper);
        let view = crate::semantic::ScalarConstraintView { lhs, lower, upper };
        let source = PendingSemantic::new(semantic, view);
        let alternative = source.alternative(view);
        let mut by_name = self.constraint_names_mut();
        let mut all = self.constraints.borrow_mut_for_push();
        let id = ConstraintId(u32::try_from(all.len()).expect("constraint count overflow"));
        by_name.insert_unique(name.clone(), id);
        push_algebraic(
            &mut all,
            AlgebraicConstraint {
                name,
                lhs,
                lower,
                upper,
                active: true,
                signature: source.signature(),
            },
        );
        source.record_declared(self, id, &all[id.index()].name, alternative);

        if alternative {
            self.alternative_scalar_views.borrow_mut().insert(id.into(), view);
        }

        self.cached_kind.set(None);
        id
    }

    /// Append prevalidated scalar rows and their source metadata, returning IDs in input order.
    ///
    /// Call only after name preflight, with no intervening user callbacks.
    pub(super) fn register_prevalidated_constraints_batch(
        &self,
        items: impl IntoIterator<Item = PendingConstraint>,
        count: usize,
    ) -> Vec<ConstraintId> {
        let mut names = self.constraint_names_mut();
        let mut constraints = self.constraints.borrow_mut();
        let final_count = constraints.len().checked_add(count).expect("constraint count overflow");

        if final_count > 0 {
            u32::try_from(final_count - 1).expect("constraint count overflow");
        }
        constraints.reserve(count);
        names.reserve(count);
        let mut ids = Vec::with_capacity(count);
        let mut semantics = None;

        for item in items {
            let id =
                ConstraintId(u32::try_from(constraints.len()).expect("constraint count overflow"));
            constraints.push(AlgebraicConstraint {
                name: item.name.clone(),
                lhs: item.lhs,
                lower: item.lower,
                upper: item.upper,
                active: true,
                signature: item.semantic.signature(),
            });

            if item.alternative {
                self.alternative_scalar_views.borrow_mut().insert(
                    id.into(),
                    crate::semantic::ScalarConstraintView {
                        lhs: item.lhs,
                        lower: item.lower,
                        upper: item.upper,
                    },
                );
            }
            if semantics.is_none()
                && self.implicit_scalars.get()
                && matches!(item.semantic, PendingSemantic::Direct(_))
                && matches!(item.origin, PendingOrigin::Declared)
            {
                self.record_declared_scalar(id);
            } else {
                item.semantic.record(
                    semantics.get_or_insert_with(|| self.semantic_writer()),
                    id,
                    &item.name,
                    item.origin,
                    item.alternative,
                );
            }
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
            use std::fmt::Write;
            let mut candidate = smol_str::SmolStrBuilder::new();
            write!(&mut candidate, "_c{n}").expect("format constraint name");
            let candidate = candidate.finish();

            if !self.constraint_names().contains_key(&candidate) {
                break candidate;
            }
        }
    }

    /// Register an anonymous constraint, deriving a unique name `_c{n}` from an
    /// internal counter. Backs the name-less form of the `constraint!` macro.
    #[doc(hidden)]
    pub fn __add_constraint_auto<C: crate::function_set::ScalarConstraintInput>(
        &self,
        c: C,
    ) -> ConstraintHandle<C::Function, C::Set> {
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
            .row_id()
    }

    /// Bulk-register constraints. Each entry is `(name, ConstraintExpr)`.
    /// Useful with `.par_iter().map(...).collect()` style construction.
    #[inline]
    pub fn add_constraints<'a, I, C>(&'a self, items: I)
    where
        I: IntoIterator<Item = (SmolStr, C)>,
        C: crate::function_set::ScalarConstraintInput,
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
        let handles = if let (Some(lower), Some(upper)) = (lo.const_bound(), hi.const_bound()) {
            validate_bounds(lower, upper);
            let function = mid.into_function();

            if function.is_affine() {
                let name = SmolStr::from(name);
                let mut names = self.constraint_names_mut();
                assert!(!names.contains_key(&name), "constraint name {name:?} already registered");
                let mut rows = self.constraints.borrow_mut();
                let id =
                    ConstraintId(u32::try_from(rows.len()).expect("constraint count overflow"));
                let signature = super::semantic::ScalarSignature {
                    function: super::semantic::storage::ScalarKind::of(&crate::FunctionValue::new(
                        crate::Function::semantic_data(&function),
                    ))
                    .expect("built-in scalar function"),
                    set: super::semantic::storage::BoundKind::Interval,
                };
                push_algebraic(
                    &mut rows,
                    AlgebraicConstraint {
                        name: name.clone(),
                        lhs: mid.id(),
                        lower,
                        upper,
                        active: true,
                        signature: Some(signature),
                    },
                );
                self.semantic_writer().record_native(
                    id.into(),
                    name.clone(),
                    super::semantic::Representation::Scalar,
                    crate::ConstraintOrigin::Range(name.clone()),
                );
                names.insert(name, id);
                self.invalidate_kind();
                return RangeConstraintHandles::Interval(ConstraintHandle::new(id, self.id()));
            }

            let lower = self.__add_constraint(
                format!("{name}_lo"),
                Constraint::new(function, GreaterThan(lower)),
            );
            let upper = self
                .__add_constraint(format!("{name}_hi"), Constraint::new(function, LessThan(upper)));
            RangeConstraintHandles::Split { lower: lower.scalar_view(), upper: upper.scalar_view() }
        } else {
            let lower = self.__add_constraint(format!("{name}_lo"), mid.ge(lo));
            let upper = self.__add_constraint(format!("{name}_hi"), mid.le(hi));
            RangeConstraintHandles::Split { lower: lower.scalar_view(), upper: upper.scalar_view() }
        };

        let origin_name = name.to_owned();

        match handles {
            RangeConstraintHandles::Interval(handle) => self.set_constraint_origin(
                handle.id(),
                crate::ConstraintOrigin::Range(origin_name.into()),
            ),
            RangeConstraintHandles::Split { lower, upper } => {
                self.set_constraint_origin(
                    lower.id(),
                    crate::ConstraintOrigin::Range(origin_name.clone().into()),
                );
                self.set_constraint_origin(
                    upper.id(),
                    crate::ConstraintOrigin::Range(origin_name.into()),
                );
            }
        }
        handles
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
        let handles = if let (Some(lower), Some(upper)) = (lo.const_bound(), hi.const_bound()) {
            validate_bounds(lower, upper);
            let function = mid.into_function();

            if function.is_affine() {
                RangeConstraintHandles::Interval(
                    self.__add_constraint(
                        self.next_auto_name(),
                        Constraint::new(function, Interval { lower, upper }),
                    )
                    .scalar_view(),
                )
            } else {
                let lower =
                    self.__add_constraint_auto(Constraint::new(function, GreaterThan(lower)));
                let upper = self.__add_constraint_auto(Constraint::new(function, LessThan(upper)));
                RangeConstraintHandles::Split {
                    lower: lower.scalar_view(),
                    upper: upper.scalar_view(),
                }
            }
        } else {
            let lower = self.__add_constraint_auto(mid.ge(lo));
            let upper = self.__add_constraint_auto(mid.le(hi));
            RangeConstraintHandles::Split { lower: lower.scalar_view(), upper: upper.scalar_view() }
        };

        let origin_name = format!("anonymous_range_{}", self.auto_seq.get());

        match handles {
            RangeConstraintHandles::Interval(handle) => self.set_constraint_origin(
                handle.id(),
                crate::ConstraintOrigin::Range(origin_name.into()),
            ),
            RangeConstraintHandles::Split { lower, upper } => {
                self.set_constraint_origin(
                    lower.id(),
                    crate::ConstraintOrigin::Range(origin_name.clone().into()),
                );
                self.set_constraint_origin(
                    upper.id(),
                    crate::ConstraintOrigin::Range(origin_name.into()),
                );
            }
        }
        handles
    }

    /// Unified view of every algebraic, explicit SOC/PSD, SOS, and indicator
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
        if self.implicit_scalars.get() {
            self.scalar_declarations.get()
        } else {
            self.semantics.borrow().as_deref().map_or(0, super::semantic::SemanticRegistry::len)
        }
    }

    /// Look up an algebraic constraint ID by its registry name.
    pub fn constraint_id(&self, name: &str) -> Option<ConstraintId> {
        self.constraint_names().get(name)
    }

    /// Look up a constraint by name across registries and bind its handle to this model.
    ///
    /// Returns `None` if the name is unknown or shared by more than one registry.
    #[inline]
    pub fn constraint_handle(&self, name: &str) -> Option<ConstraintHandle> {
        self.semantic_names.borrow().unique_id(name).map(|id| ConstraintHandle::new(id, self.id()))
    }

    /// Bind a raw algebraic constraint ID to this model.
    pub fn constraint_handle_from_id(&self, id: ConstraintId) -> Option<ConstraintHandle> {
        (id.index() < self.constraints.borrow().len()).then(|| ConstraintHandle::new(id, self.id()))
    }

    /// Register an explicitly symmetric affine matrix in the real PSD cone.
    ///
    /// # Panics
    /// Panics for invalid ownership, duplicate names, or count overflow.
    pub fn add_psd_constraint<'a>(
        &'a self,
        name: impl Into<SmolStr>,
        matrix: impl IntoSymmetricAffineFunction<'a>,
    ) -> PsdConstraintHandle {
        let function = matrix.into_symmetric_affine_function(self);
        self.register_psd_ir(name.into(), PsdConstraintIr(function))
    }

    /// Register an affine PSD matrix and its semantic metadata under the given name.
    pub(crate) fn register_psd_ir(
        &self,
        name: SmolStr,
        ir: PsdConstraintIr<'_>,
    ) -> PsdConstraintHandle {
        self.assert_expr_belongs(ir.0.matrix()[(0, 0)]);
        let mut names = self.psd_names_mut();
        let mut rows = self.psd_constraints.borrow_mut();
        let id = PsdConstraintId(u32::try_from(rows.len()).expect("PSD count overflow"));
        names.insert_unique(name.clone(), id);
        rows.push(PsdConstraint {
            name: name.clone(),
            matrix: ir.0.matrix().map(|entry| entry.id()),
            active: true,
        });
        self.record_declared_native(id.into(), super::semantic::Representation::Psd);
        self.invalidate_kind();
        PsdConstraintHandle::new(id, self.id())
    }

    /// Register a PSD constraint with an automatically generated unique name.
    #[doc(hidden)]
    pub fn __add_psd_constraint_auto<'a>(
        &'a self,
        matrix: impl IntoSymmetricAffineFunction<'a>,
    ) -> PsdConstraintHandle {
        loop {
            let n = self.auto_seq.get();
            self.auto_seq.set(n.checked_add(1).expect("constraint name count overflow"));
            let name = format!("_psd{n}");

            if !self.psd_names().contains_key(name.as_str()) {
                return self.add_psd_constraint(name, matrix);
            }
        }
    }

    /// Borrow the PSD registry, including inactive constraints.
    pub fn psd_constraints(&self) -> Ref<'_, Vec<PsdConstraint>> {
        self.psd_constraints.borrow()
    }

    /// Count all PSD declarations, including inactive constraints.
    pub fn num_psd_constraints(&self) -> usize {
        self.psd_constraints.borrow().len()
    }

    /// Return whether at least one PSD constraint is active.
    pub fn has_active_psd_constraints(&self) -> bool {
        self.psd_constraints.borrow().iter().any(|c| c.active)
    }

    /// Look up a PSD constraint ID by name.
    pub fn psd_constraint_id(&self, name: &str) -> Option<PsdConstraintId> {
        self.psd_names().get(name)
    }

    /// Return a named PSD handle only when its semantic signature is the built-in pair.
    /// Custom wrappers remain accessible through `typed_constraint_handle` or an erased handle.
    pub fn psd_constraint_handle(&self, name: &str) -> Option<PsdConstraintHandle> {
        self.psd_constraint_handle_from_id(self.psd_constraint_id(name)?)
    }

    /// Bind a raw PSD ID only when it has the built-in function/set signature.
    pub fn psd_constraint_handle_from_id(
        &self,
        id: PsdConstraintId,
    ) -> Option<PsdConstraintHandle> {
        self.try_typed_handle(ConstraintHandle::new(id, self.id())).ok().flatten()
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
        let pending = PendingSoc::new(name.into(), terms, bound, super::arena_key(&self.arena));
        self.register_soc_components(pending.name, pending.components)
    }

    /// Check the lowered SOC function's ownership and register its affine components.
    pub(crate) fn register_soc_ir(
        &self,
        name: SmolStr,
        ir: SocConstraintIr<'_>,
    ) -> SocConstraintHandle {
        let data = ir.0.into_data();
        assert!(data.components.len() >= 2, "SOC needs at least two components");
        self.ensure_model_id(data.model_id).unwrap_or_else(|error| panic!("{error}"));
        self.register_soc_components(name, data.components)
    }

    /// Register SOC components with the bound first, followed by the norm terms.
    fn register_soc_components(
        &self,
        name: SmolStr,
        components: crate::SharedSlice<ExprId>,
    ) -> SocConstraintHandle {
        let mut names = self.soc_names_mut();
        let mut rows = self.soc_constraints.borrow_mut();
        let id = SocConstraintId(u32::try_from(rows.len()).expect("SOC count overflow"));
        names.insert_unique(name.clone(), id);
        let bound = components[0];
        let dimension = components.len();
        let terms = components.into_slice(1..dimension);
        rows.push(SocConstraint { name: name.clone(), bound, terms, active: true });
        self.record_declared_native(id.into(), super::semantic::Representation::Soc);
        self.invalidate_kind();
        SocConstraintHandle::new(id, self.id())
    }

    /// Append prevalidated SOC constraints and their semantic metadata in input order.
    ///
    /// Call only after name preflight, with no intervening user callbacks.
    pub(super) fn register_prevalidated_soc_batch(
        &self,
        items: impl IntoIterator<Item = PendingSoc>,
        count: usize,
    ) {
        let mut names = self.soc_names_mut();
        let mut constraints = self.soc_constraints.borrow_mut();
        let final_count =
            constraints.len().checked_add(count).expect("SOC constraint count overflow");
        if final_count > 0 {
            u32::try_from(final_count - 1).expect("SOC constraint count overflow");
        }
        constraints.reserve(count);
        names.reserve(count);
        let mut semantics = self.semantic_writer();

        for item in items {
            let id = SocConstraintId(
                u32::try_from(constraints.len()).expect("SOC constraint count overflow"),
            );
            let components = item.components;
            let bound = components[0];
            let dimension = components.len();
            let terms = components.into_slice(1..dimension);
            semantics.record_native(
                id.into(),
                item.name.clone(),
                super::semantic::Representation::Soc,
                crate::ConstraintOrigin::Declared,
            );
            constraints.push(SocConstraint { name: item.name.clone(), terms, bound, active: true });
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

            if !self.soc_names().contains_key(&candidate) {
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

    /// Return whether at least one explicit SOC constraint is active.
    pub fn has_active_soc_constraints(&self) -> bool {
        self.soc_constraints.borrow().iter().any(|c| c.active)
    }

    /// Count all explicit SOC declarations, including inactive constraints.
    pub fn num_soc_constraints(&self) -> usize {
        self.soc_constraints.borrow().len()
    }

    /// Look up an explicit SOC constraint ID by name.
    pub fn soc_constraint_id(&self, name: &str) -> Option<SocConstraintId> {
        self.soc_names().get(name)
    }

    /// Return a named SOC handle only when its semantic signature is the built-in pair.
    /// Custom wrappers remain accessible through `typed_constraint_handle` or an erased handle.
    pub fn soc_constraint_handle(&self, name: &str) -> Option<SocConstraintHandle> {
        self.soc_constraint_handle_from_id(self.soc_constraint_id(name)?)
    }

    /// Bind a raw SOC ID only when it has the built-in function/set signature.
    pub fn soc_constraint_handle_from_id(
        &self,
        id: SocConstraintId,
    ) -> Option<SocConstraintHandle> {
        self.try_typed_handle(ConstraintHandle::new(id, self.id())).ok().flatten()
    }

    /// Return whether at least one explicit SOC or PSD constraint is active.
    pub fn has_cones(&self) -> bool {
        self.has_active_soc_constraints() || self.has_active_psd_constraints()
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
    ) -> SosConstraintHandle {
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
        let mut by_name = self.sos_names_mut();
        assert!(!by_name.contains_key(&name), "SOS constraint name {name:?} already registered");
        let mut all = self.sos_constraints.borrow_mut();
        let id = SosConstraintId(u32::try_from(all.len()).expect("SOS constraint count overflow"));
        self.record_sos_semantic(id, name.clone());
        all.push(SosConstraint { name: name.clone(), sos_type, members, active: true });
        by_name.insert(name, id);
        self.cached_kind.set(None);
        SosConstraintHandle::new(id, self.id())
    }

    /// Append prevalidated SOS constraints and their semantic metadata in input order.
    ///
    /// Call only after name preflight, with no intervening user callbacks.
    pub(super) fn register_prevalidated_sos_batch(
        &self,
        sos_type: SosType,
        items: impl IntoIterator<Item = PendingSos>,
        count: usize,
    ) {
        let mut names = self.sos_names_mut();
        let mut constraints = self.sos_constraints.borrow_mut();
        let final_count =
            constraints.len().checked_add(count).expect("SOS constraint count overflow");
        if final_count > 0 {
            u32::try_from(final_count - 1).expect("SOS constraint count overflow");
        }
        constraints.reserve(count);
        names.reserve(count);
        let mut semantics = self.semantic_writer();

        for item in items {
            let id = SosConstraintId(
                u32::try_from(constraints.len()).expect("SOS constraint count overflow"),
            );
            semantics.record_sos(id, item.name.clone());
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

    /// Register one prepared SOS constraint and its source metadata.
    pub(super) fn register_sos_one(&self, sos_type: SosType, item: PendingSos) {
        let mut names = self.sos_names_mut();
        assert!(
            !names.contains_key(&item.name),
            "SOS constraint name {:?} already registered",
            item.name
        );
        let mut constraints = self.sos_constraints.borrow_mut();
        let id = SosConstraintId(
            u32::try_from(constraints.len()).expect("SOS constraint count overflow"),
        );
        self.record_sos_semantic(id, item.name.clone());
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
    ) -> SosConstraintHandle {
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

    /// Generate a unique `_sos{n}` name in the SOS registry.
    fn next_auto_sos_name(&self) -> SmolStr {
        loop {
            let n = self.auto_seq.get();
            self.auto_seq.set(n + 1);
            let candidate: SmolStr = format!("_sos{n}").into();

            if !self.sos_names().contains_key(&candidate) {
                break candidate;
            }
        }
    }

    /// Register an SOS constraint with supplied weights and an automatically generated name.
    #[doc(hidden)]
    pub fn __add_sos_constraint_auto<'a, D: Degree>(
        &'a self,
        sos_type: SosType,
        members: impl IntoIterator<Item = (Expr<'a, D>, f64)>,
    ) -> SosConstraintHandle {
        self.add_sos_constraint(self.next_auto_sos_name(), sos_type, members)
    }

    /// Register an SOS constraint with automatic weights and an automatically generated name.
    #[doc(hidden)]
    pub fn __add_sos_constraint_auto_weights<'a, D: Degree>(
        &'a self,
        sos_type: SosType,
        variables: impl IntoIterator<Item = Expr<'a, D>>,
    ) -> SosConstraintHandle {
        self.add_sos_constraint_auto_weights(self.next_auto_sos_name(), sos_type, variables)
    }

    /// Borrow the SOS registry, including inactive and reformulated sources.
    pub fn sos_constraints(&self) -> Ref<'_, Vec<SosConstraint>> {
        self.sos_constraints.borrow()
    }

    /// Count all SOS declarations, including inactive and reformulated sources.
    pub fn num_sos_constraints(&self) -> usize {
        self.sos_constraints.borrow().len()
    }

    /// Look up an SOS constraint ID by name.
    pub fn sos_constraint_id(&self, name: &str) -> Option<SosConstraintId> {
        self.sos_names().get(name)
    }

    /// Return whether any SOS declarations exist, including inactive sources.
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

    /// Check that the trigger belongs to this model and is a bare binary variable.
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

    /// Validate and register an affine indicator consequent with its source metadata.
    fn register_indicator(
        &self,
        name: SmolStr,
        trigger: VarId,
        active_value: bool,
        lhs: ExprId,
        bounds: (f64, f64),
        semantic: Option<crate::SemanticPair>,
    ) -> IndicatorConstraintId {
        let (lower, upper) = bounds;
        validate_bounds(lower, upper);
        let class = classify(&self.arena.borrow(), lhs);
        assert!(class == ExprClass::Linear, "indicator consequent must be affine, got {class:?}");
        let mut names = self.indicator_names_mut();
        assert!(
            !names.contains_key(&name),
            "indicator constraint name {name:?} already registered"
        );
        let mut all = self.indicator_constraints.borrow_mut();
        let id = IndicatorConstraintId(
            u32::try_from(all.len()).expect("indicator constraint count overflow"),
        );
        self.record_indicator_semantic(
            id,
            name.clone(),
            trigger,
            active_value,
            semantic.unwrap_or_else(|| crate::SemanticPair {
                function: crate::FunctionValue::new(
                    crate::ScalarFunctionData::<oximo_expr::Affine>::new(self.id(), lhs),
                ),
                set: crate::SetValue::new(Interval { lower, upper }),
            }),
            crate::semantic::ScalarConstraintView { lhs, lower, upper },
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

    /// Prepare an affine indicator consequent while retaining its original function/set pair.
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
        let mut pending = self.prepare_indicator_interval(
            name,
            trigger,
            active_value,
            consequent.lhs.id,
            lower,
            upper,
        );
        pending.semantic = consequent.semantic.into_pair();

        pending
    }

    /// Validate an affine interval consequent and prepare its indicator source data.
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
        PendingIndicator {
            alternative: false,
            name,
            trigger,
            active_value,
            lhs,
            lower,
            upper,
            semantic: crate::SemanticPair {
                function: crate::FunctionValue::new(
                    crate::ScalarFunctionData::<oximo_expr::Affine>::new(self.id(), lhs),
                ),
                set: crate::SetValue::new(Interval { lower, upper }),
            },
        }
    }

    /// Check batch names and append prepared indicators, returning IDs in input order.
    pub(super) fn register_indicators_batch(
        &self,
        mut items: Vec<PendingIndicator>,
    ) -> Vec<IndicatorConstraintId> {
        let mut names = self.indicator_names_mut();
        validate_batch_names(
            |name| names.contains_key(name),
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

        for item in &mut items {
            item.alternative = !crate::semantic::ScalarConstraintView {
                lhs: item.lhs,
                lower: item.lower,
                upper: item.upper,
            }
            .matches_pair(&item.semantic);
        }

        constraints.reserve(items.len());
        names.reserve(items.len());
        let mut ids = Vec::with_capacity(items.len());
        let mut semantics = None;

        for item in items {
            let id = IndicatorConstraintId(
                u32::try_from(constraints.len()).expect("indicator constraint count overflow"),
            );

            if item.alternative {
                self.alternative_scalar_views.borrow_mut().insert(
                    id.into(),
                    crate::semantic::ScalarConstraintView {
                        lhs: item.lhs,
                        lower: item.lower,
                        upper: item.upper,
                    },
                );
            }
            semantics.get_or_insert_with(|| self.semantic_writer()).record_indicator(
                id,
                item.name.clone(),
                self.id(),
                (item.trigger, item.active_value),
                item.semantic,
                !item.alternative,
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
    /// Panics for invalid triggers or bounds,
    /// foreign ownership, duplicate names, or count overflow.
    pub fn add_indicator_constraint<'a>(
        &'a self,
        name: impl Into<SmolStr>,
        trigger: impl Into<Expr<'a>>,
        active_value: bool,
        consequent: impl IntoAffineConstraint<'a>,
    ) -> IndicatorConstraintHandle {
        let trigger = self.indicator_trigger(trigger);
        let consequent = consequent.into_affine_constraint().0;
        self.assert_expr_belongs(consequent.lhs);
        let id = self.register_indicator(
            name.into(),
            trigger,
            active_value,
            consequent.lhs.id,
            (consequent.lower, consequent.upper),
            Some(consequent.semantic.into_pair()),
        );
        IndicatorConstraintHandle::new(id, self.id())
    }

    /// Register a named indicator constraint on behalf of the modeling macro.
    #[doc(hidden)]
    pub fn __add_indicator_constraint<'a>(
        &'a self,
        name: impl Into<SmolStr>,
        trigger: impl Into<Expr<'a>>,
        active_value: bool,
        consequent: impl IntoAffineConstraint<'a>,
    ) -> IndicatorConstraintHandle {
        let trigger: Expr = trigger.into();
        self.add_indicator_constraint(name, trigger, active_value, consequent)
    }

    /// Generate a unique `_ind{n}` name in the indicator registry.
    fn next_auto_indicator_name(&self) -> SmolStr {
        loop {
            let n = self.auto_seq.get();
            self.auto_seq.set(n + 1);
            let candidate: SmolStr = format!("_ind{n}").into();

            if !self.indicator_names().contains_key(&candidate) {
                break candidate;
            }
        }
    }

    /// Register an indicator constraint with an automatically generated unique name.
    #[doc(hidden)]
    pub fn __add_indicator_constraint_auto<'a>(
        &'a self,
        trigger: impl Into<Expr<'a>>,
        active_value: bool,
        consequent: impl IntoAffineConstraint<'a>,
    ) -> IndicatorConstraintHandle {
        let trigger: Expr = trigger.into();
        self.add_indicator_constraint(
            self.next_auto_indicator_name(),
            trigger,
            active_value,
            consequent,
        )
    }

    /// Register an indicator whose affine consequent lies between numeric bounds.
    #[doc(hidden)]
    pub fn __add_indicator_interval<'a>(
        &'a self,
        name: impl Into<SmolStr>,
        trigger: impl Into<Expr<'a>>,
        active_value: bool,
        lhs: impl Into<Expr<'a>>,
        lower: f64,
        upper: f64,
    ) -> IndicatorConstraintHandle {
        let trigger: Expr = trigger.into();
        let lhs: Expr = lhs.into();
        self.assert_expr_belongs(lhs);
        let trigger = self.indicator_trigger(trigger.erase());
        let id = self.register_indicator(
            name.into(),
            trigger,
            active_value,
            lhs.id,
            (lower, upper),
            None,
        );
        IndicatorConstraintHandle::new(id, self.id())
    }

    /// Register a two-sided indicator range as one interval or two scalar consequents.
    #[doc(hidden)]
    pub fn __add_indicator_range<'a, B1: IntoRhs<'a>, B2: IntoRhs<'a>>(
        &'a self,
        name: &str,
        trigger: impl Into<Expr<'a>>,
        active_value: bool,
        mid: impl IntoAffineFunction<'a>,
        lo: B1,
        hi: B2,
    ) -> RangeIndicatorConstraintHandles {
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
            RangeIndicatorConstraintHandles::Interval(IndicatorConstraintHandle::new(id, self.id()))
        } else {
            let lower = self.prepare_indicator(
                format!("{name}_lo").into(),
                trigger,
                active_value,
                mid.ge(lo).into_checked_affine(),
            );
            let upper = self.prepare_indicator(
                format!("{name}_hi").into(),
                trigger,
                active_value,
                mid.le(hi).into_checked_affine(),
            );
            let mut ids = self.register_indicators_batch(vec![lower, upper]).into_iter();
            let lower = IndicatorConstraintHandle::new(
                ids.next().expect("lower indicator range ID missing"),
                self.id(),
            );
            let upper = IndicatorConstraintHandle::new(
                ids.next().expect("upper indicator range ID missing"),
                self.id(),
            );
            RangeIndicatorConstraintHandles::Split { lower, upper }
        }
    }

    /// Register a two-sided indicator range with an automatically generated base name.
    #[doc(hidden)]
    pub fn __add_indicator_range_auto<'a, B1: IntoRhs<'a>, B2: IntoRhs<'a>>(
        &'a self,
        trigger: impl Into<Expr<'a>>,
        active_value: bool,
        mid: impl IntoAffineFunction<'a>,
        lo: B1,
        hi: B2,
    ) -> RangeIndicatorConstraintHandles {
        let trigger: Expr = trigger.into();
        let mid = mid.into_affine_function().expression();
        let name = self.next_auto_indicator_name();
        self.__add_indicator_range(&name, trigger, active_value, mid, lo, hi)
    }

    /// Borrow the indicator registry, including inactive and reformulated sources.
    pub fn indicator_constraints(&self) -> Ref<'_, Vec<IndicatorConstraint>> {
        self.indicator_constraints.borrow()
    }

    /// Count all indicator declarations, including inactive and reformulated sources.
    pub fn num_indicator_constraints(&self) -> usize {
        self.indicator_constraints.borrow().len()
    }

    /// Look up an indicator constraint ID by name.
    pub fn indicator_constraint_id(&self, name: &str) -> Option<IndicatorConstraintId> {
        self.indicator_names().get(name)
    }

    /// Return whether any indicator declarations exist, including inactive sources.
    pub fn has_indicator_constraints(&self) -> bool {
        !self.indicator_constraints.borrow().is_empty()
    }

    /// Return whether at least one indicator constraint is active.
    pub fn has_active_indicator_constraints(&self) -> bool {
        self.indicator_constraints.borrow().iter().any(|c| c.active)
    }
}
