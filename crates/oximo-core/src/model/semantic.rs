mod names;
mod positions;
pub(crate) use names::{ConstraintNames, NamesMut, NamesRef};
pub(crate) mod storage;
mod views;

pub(crate) use storage::{Representation, ScalarSignature, SemanticRegistry};
pub use views::SemanticConstraints;

use super::Model;
use crate::function_set::{SecondOrderCone, validate_bounds};
use crate::semantic::{
    ConstraintOrigin, FunctionRepresentationRef, NativeConstraintRef, SemanticConstraint,
    SemanticConstraintId, SemanticPair,
};
use crate::{ConstraintHandle, PositiveSemidefiniteCone, PsdConstraint, SocConstraint};
use smol_str::SmolStr;
use std::cell::RefMut;

pub(super) struct SemanticWriter<'a> {
    pub(super) records: RefMut<'a, SemanticRegistry>,
}

impl SemanticWriter<'_> {
    /// Record a declaration, using native storage when its signature permits.
    pub(super) fn record(
        &mut self,
        id: SemanticConstraintId,
        name: SmolStr,
        pair: SemanticPair,
        origin: ConstraintOrigin,
    ) {
        self.record_pair(id, name, pair, origin, true);
    }

    /// Record a pair, retaining its full payload when native storage cannot describe it.
    pub(super) fn record_pair(
        &mut self,
        id: SemanticConstraintId,
        name: SmolStr,
        pair: SemanticPair,
        origin: ConstraintOrigin,
        direct: bool,
    ) {
        self.records.record(SemanticConstraint { id, name, pair, active: true, origin }, direct);
    }

    /// Record a built-in SOS declaration using its native registry entry.
    pub(super) fn record_sos(&mut self, id: crate::SosConstraintId, name: SmolStr) {
        self.record_native(id.into(), name, Representation::Sos, ConstraintOrigin::Declared);
    }

    /// Record native metadata with the shared declared-origin marker.
    #[inline]
    pub(super) fn record_declared(
        &mut self,
        id: SemanticConstraintId,
        representation: Representation,
    ) {
        self.records.push_declared(id, representation);
    }

    /// Record a range whose provenance uses its native row name.
    #[inline]
    pub(super) fn record_row_range(
        &mut self,
        id: SemanticConstraintId,
        representation: Representation,
    ) {
        self.records.push_row_range(id, representation);
    }

    /// Record native declaration metadata and its provenance.
    #[inline]
    pub(super) fn record_native(
        &mut self,
        id: SemanticConstraintId,
        name: SmolStr,
        representation: Representation,
        origin: ConstraintOrigin,
    ) {
        if matches!(origin, ConstraintOrigin::Declared) {
            self.records.push_declared(id, representation);
        } else {
            self.records.push_named(id, representation, origin, &name);
        }
    }

    /// Record an indicator's trigger, consequent, and semantic signature.
    pub(super) fn record_indicator(
        &mut self,
        id: crate::IndicatorConstraintId,
        name: SmolStr,
        model_id: crate::ModelId,
        activation: (oximo_expr::VarId, bool),
        consequent: SemanticPair,
        direct: bool,
    ) {
        if let Some(signature) = ScalarSignature::of(&consequent).filter(|_| direct) {
            self.record_native(
                id.into(),
                name,
                Representation::Indicator(signature),
                ConstraintOrigin::Declared,
            );
            return;
        }

        let (trigger, active_value) = activation;
        self.record_pair(
            id.into(),
            name,
            SemanticPair {
                function: crate::FunctionValue::new(crate::IndicatorFunctionData {
                    model_id,
                    trigger,
                    consequent: consequent.function,
                }),
                set: crate::SetValue::new(crate::IndicatorSet {
                    active_value,
                    consequent: consequent.set,
                }),
            },
            ConstraintOrigin::Declared,
            direct,
        );
    }
}

impl Model {
    /// Borrow semantic metadata for mutation, initializing any implicit scalar prefix.
    #[cold]
    #[inline(never)]
    pub(super) fn semantic_writer(&self) -> SemanticWriter<'_> {
        let mut records = self.semantics.borrow_mut();

        if self.implicit_scalars.replace(false) {
            records.initialize_scalar_prefix(self.scalar_declarations.get());
        }
        SemanticWriter { records }
    }

    /// Record a scalar declaration without allocating metadata while its order stays implicit.
    #[inline]
    pub(super) fn record_declared_scalar(&self, id: crate::ConstraintId) {
        if self.implicit_scalars.get() {
            debug_assert_eq!(id.index(), self.scalar_declarations.get());
            self.scalar_declarations.set(id.index() + 1);
        } else {
            self.semantic_writer().record_declared(id.into(), Representation::Scalar);
        }
    }

    /// Append native declaration metadata after registering any implicit scalar rows.
    #[inline]
    pub(super) fn record_declared_native(
        &self,
        id: SemanticConstraintId,
        representation: Representation,
    ) {
        let mut records = self.semantics.borrow_mut();

        if self.implicit_scalars.replace(false) {
            records.initialize_scalar_prefix(self.scalar_declarations.get());
        }
        records.push_declared(id, representation);
    }

    /// Visit constraints whose stored scalar view differs from their semantic pair.
    /// Their row multipliers require an explicit mapping to semantic coordinates.
    #[inline]
    pub fn visit_alternative_lowerings(&self, visitor: impl FnMut(SemanticConstraintId)) {
        self.alternative_scalar_views.borrow().keys().copied().for_each(visitor);
    }

    /// Return the stored scalar row or indicator consequent selected by an explicit lowering.
    /// The view uses native coordinates; it does not replace the original declaration.
    pub fn alternative_lowering_view(
        &self,
        id: SemanticConstraintId,
    ) -> Option<crate::ScalarConstraintView> {
        self.alternative_scalar_views.borrow().get(&id).copied()
    }

    /// Reserve metadata space for additions to one native registry.
    pub(super) fn reserve_semantics(&self, registry: SemanticConstraintId, additional: usize) {
        let mut records = self.semantics.borrow_mut();
        records.reserve(registry, additional);
    }

    /// Record a function/set pair as an original model declaration.
    pub(super) fn record_semantic(
        &self,
        id: SemanticConstraintId,
        name: SmolStr,
        pair: SemanticPair,
    ) {
        self.record_semantic_with_origin(id, name, pair, ConstraintOrigin::Declared);
    }

    /// Record a pair and its provenance, keeping ordinary scalar declarations implicit.
    pub(super) fn record_semantic_with_origin(
        &self,
        id: SemanticConstraintId,
        name: SmolStr,
        pair: SemanticPair,
        origin: ConstraintOrigin,
    ) {
        if self.implicit_scalars.get()
            && matches!(origin, ConstraintOrigin::Declared)
            && let SemanticConstraintId::Algebraic(id) = id
            && ScalarSignature::of(&pair).is_some()
        {
            self.record_declared_scalar(id);
            return;
        }

        self.semantic_writer().record(id, name, pair, origin);
    }

    /// Add semantic metadata for an existing built-in SOS constraint.
    pub(super) fn record_sos_semantic(&self, id: crate::SosConstraintId, name: SmolStr) {
        self.semantic_writer().record_sos(id, name);
    }

    /// Record an indicator source and track any difference from its lowered consequent.
    pub(super) fn record_indicator_semantic(
        &self,
        id: crate::IndicatorConstraintId,
        name: SmolStr,
        trigger: oximo_expr::VarId,
        active_value: bool,
        consequent: SemanticPair,
        view: crate::semantic::ScalarConstraintView,
    ) {
        let alternative = !view.matches_pair(&consequent);

        self.semantic_writer().record_indicator(
            id,
            name,
            self.id(),
            (trigger, active_value),
            consequent,
            !alternative,
        );

        if alternative {
            self.alternative_scalar_views.borrow_mut().insert(id.into(), view);
        }
    }

    /// Validate an erased pair and register its native view with its original semantics.
    #[expect(
        clippy::too_many_lines,
        reason = "ownership preflight and shape-specific registration form one atomic boundary"
    )]
    pub(super) fn register_semantic_pair(
        &self,
        name: SmolStr,
        pair: SemanticPair,
    ) -> SemanticConstraintId {
        assert_eq!(pair.function.model_id(), self.id(), "function belongs to a different model");

        {
            let arena = self.arena.borrow();
            pair.function.visit_expressions(&mut |id| {
                assert!(id.index() < arena.len(), "function contains an invalid expression ID");
            });
        }

        if let Some((lower, upper)) = pair.set.scalar_bounds() {
            validate_bounds(lower, upper);
        }

        let native = pair.native_constraint().expect(
            "constraint requires an algebraic, conic, SOS, or indicator representation; opaque registration is deferred",
        );

        let id = match native {
            NativeConstraintRef::Scalar { .. } => {
                let scalar = crate::function_set::PreparedScalarConstraint::declared(
                    crate::semantic::ScalarConstraintSource::from_pair(pair),
                );
                return self.__add_constraint(name, scalar).id();
            }
            NativeConstraintRef::Soc { components, dimension } => {
                assert!(dimension >= 2, "SOC needs at least two components");
                assert_eq!(components.len(), dimension, "SOC dimension mismatch");
                {
                    let arena = self.arena.borrow();
                    assert!(
                        components.iter().all(|&root| oximo_expr::classify(&arena, root)
                            == oximo_expr::ExprClass::Linear),
                        "SOC components must be affine"
                    );
                }
                assert!(
                    !self.soc_names().contains_key(&name),
                    "SOC name {name:?} already registered"
                );

                let mut rows = self.soc_constraints.borrow_mut();
                let id =
                    crate::SocConstraintId(u32::try_from(rows.len()).expect("SOC count overflow"));
                rows.push(SocConstraint {
                    name: name.clone(),
                    bound: components[0],
                    terms: pair.function.soc_terms(components),
                    active: true,
                });
                self.soc_names_mut().insert(name.clone(), id);
                SemanticConstraintId::SecondOrderCone(id)
            }
            NativeConstraintRef::Psd { matrix, dimension } => {
                assert_eq!(matrix.side_dimension(), dimension, "PSD cone dimension mismatch");
                {
                    let arena = self.arena.borrow();
                    assert!(
                        matrix
                            .upper_triangle()
                            .iter()
                            .all(|&root| oximo_expr::classify(&arena, root)
                                == oximo_expr::ExprClass::Linear),
                        "PSD components must be affine"
                    );
                }
                assert!(
                    !self.psd_names().contains_key(&name),
                    "PSD name {name:?} already registered"
                );

                let mut rows = self.psd_constraints.borrow_mut();
                let id =
                    crate::PsdConstraintId(u32::try_from(rows.len()).expect("PSD count overflow"));
                rows.push(PsdConstraint {
                    name: name.clone(),
                    matrix: matrix.clone(),
                    active: true,
                });
                self.psd_names_mut().insert(name.clone(), id);
                SemanticConstraintId::PositiveSemidefinite(id)
            }
            NativeConstraintRef::Sos { variables, set } => {
                let members = self.sos_members(&name, variables, &set.weights);
                assert!(!self.sos_names().contains_key(&name), "SOS name already registered");

                let mut rows = self.sos_constraints.borrow_mut();
                let id =
                    crate::SosConstraintId(u32::try_from(rows.len()).expect("SOS count overflow"));
                rows.push(crate::SosConstraint {
                    name: name.clone(),
                    sos_type: set.kind,
                    members,
                    active: true,
                });
                self.sos_names_mut().insert(name.clone(), id);
                id.into()
            }
            NativeConstraintRef::Indicator { trigger, consequent, set } => {
                let lhs = self.checked_indicator_function(trigger, consequent);
                let (lower, upper) =
                    set.consequent.scalar_bounds().expect("indicator requires scalar bounds");
                validate_bounds(lower, upper);
                assert!(
                    !self.indicator_names().contains_key(&name),
                    "indicator name already registered"
                );

                let mut rows = self.indicator_constraints.borrow_mut();
                let id = crate::IndicatorConstraintId(
                    u32::try_from(rows.len()).expect("indicator count overflow"),
                );
                rows.push(crate::IndicatorConstraint {
                    name: name.clone(),
                    trigger,
                    active_value: set.active_value,
                    lhs,
                    lower,
                    upper,
                    active: true,
                });
                self.indicator_names_mut().insert(name.clone(), id);

                id.into()
            }
        };

        self.record_semantic(id, name, pair);
        self.invalidate_kind();

        id
    }

    /// Borrow semantic declarations and canonical native payloads without cloning.
    pub fn semantic_constraints_ref(&self) -> SemanticConstraints<'_> {
        SemanticConstraints::new(self)
    }

    /// Owned semantic records in declaration order.
    pub fn semantic_constraints(&self) -> Vec<SemanticConstraint> {
        self.semantic_constraints_ref().iter().map(|c| c.to_owned()).collect()
    }

    /// Find a registered ID's declaration position, panicking if it is unknown.
    fn semantic_position(&self, id: SemanticConstraintId) -> usize {
        self.semantic_position_opt(id).expect("unknown semantic handle")
    }

    /// Look up an ID's declaration position without expanding implicit scalar metadata.
    fn semantic_position_opt(&self, id: SemanticConstraintId) -> Option<usize> {
        if self.implicit_scalars.get() {
            match id {
                SemanticConstraintId::Algebraic(id)
                    if id.index() < self.scalar_declarations.get() =>
                {
                    Some(id.index())
                }
                _ => None,
            }
        } else {
            self.semantics.borrow().position(id)
        }
    }

    /// Return an owned copy of a constraint's semantic declaration, if registered.
    ///
    /// # Errors
    /// Returns an ownership error for a handle from another model.
    pub fn semantic_constraint<F, S>(
        &self,
        handle: ConstraintHandle<F, S>,
    ) -> Result<Option<SemanticConstraint>, oximo_expr::ModelMismatchError> {
        self.ensure_model_id(handle.model_id())?;
        let position = self.semantic_position_opt(handle.id());

        Ok(position.and_then(|p| self.semantic_constraints_ref().get(p).map(|c| c.to_owned())))
    }

    /// Read the semantic scalar expression and bounds together, without owning a pair.
    /// Returns `None` for non-scalar or opaque functions, including alternative
    /// lowerings whose original expression cannot be evaluated.
    ///
    /// # Errors
    /// Returns an ownership error for a handle from another model.
    #[inline]
    pub fn scalar_constraint_view<F, S>(
        &self,
        handle: ConstraintHandle<F, S>,
    ) -> Result<Option<crate::ScalarConstraintView>, oximo_expr::ModelMismatchError> {
        self.ensure_model_id(handle.model_id())?;

        if let SemanticConstraintId::Algebraic(id) = handle.id() {
            let rows = self.constraints.borrow();
            let Some(row) = rows.get(id.index()) else { return Ok(None) };

            if row.signature.is_some() {
                return Ok(Some(self.inactive_scalar_views.borrow().get(&id).copied().unwrap_or(
                    crate::ScalarConstraintView {
                        lhs: row.lhs,
                        lower: row.lower,
                        upper: row.upper,
                    },
                )));
            }
        }

        Ok(self.extension_scalar_view(handle.id()))
    }

    /// Recover a scalar expression and bounds from a retained extension pair.
    #[cold]
    fn extension_scalar_view(
        &self,
        id: SemanticConstraintId,
    ) -> Option<crate::ScalarConstraintView> {
        let position = self.semantic_position_opt(id)?;
        let records = self.semantic_constraints_ref();
        let record = records.get(position)?;
        let FunctionRepresentationRef::Scalar(lhs) = record.pair.function.representation() else {
            return None;
        };
        let (lower, upper) = record.pair.set.scalar_bounds()?;
        Some(crate::ScalarConstraintView { lhs, lower, upper })
    }

    /// Return whether any constraint registry contains the name.
    pub fn contains_constraint_name(&self, name: &str) -> bool {
        self.semantic_names.borrow().contains_key(name)
    }

    /// Check identity before recovering a typed handle; no type-name comparisons are used.
    pub fn typed_constraint_handle<F: crate::SemanticFunction, S: crate::function_set::Set>(
        &self,
        name: &str,
    ) -> Option<ConstraintHandle<F, S>> {
        let names = self.semantic_names.borrow();
        let records = self.semantic_constraints_ref();
        names
            .ids(name)
            .filter_map(|id| self.semantic_position_opt(id))
            .filter_map(|p| records.get(p))
            .find(|c| c.pair.function.signature().is::<F>() && c.pair.set.signature().is::<S>())
            .map(|c| ConstraintHandle::new(c.id, self.id()))
    }

    /// Recover types by identity, independently of names shared by different registries.
    ///
    /// # Errors
    /// Returns an ownership error for a foreign handle; absent or mismatched pairs return `None`.
    pub fn try_typed_handle<F: crate::SemanticFunction, S: crate::function_set::Set>(
        &self,
        handle: ConstraintHandle,
    ) -> Result<Option<ConstraintHandle<F, S>>, oximo_expr::ModelMismatchError> {
        self.ensure_model_id(handle.model_id())?;
        let Some(position) = self.semantic_position_opt(handle.id()) else { return Ok(None) };
        let records = self.semantic_constraints_ref();
        let Some(record) = records.get(position) else { return Ok(None) };
        let pair = &record.pair;

        Ok((pair.function.signature().is::<F>() && pair.set.signature().is::<S>())
            .then(|| ConstraintHandle::new(handle.id(), self.id())))
    }

    /// Enable or disable a constraint while retaining its definition and identity.
    ///
    /// # Errors
    /// Rejects foreign handles, reformulated SOS/indicator sources,
    /// or rows used by a GDP reformulation.
    ///
    /// # Panics
    /// Panics for an unknown handle or an incompatible representation.
    pub fn set_constraint_active<F, S>(
        &self,
        handle: ConstraintHandle<F, S>,
        active: bool,
    ) -> Result<(), crate::ConstraintUpdateError> {
        self.ensure_model_id(handle.model_id())?;
        self.ensure_constraint_not_reformulated(handle.id())?;

        if self.semantic_constraint(handle)?.is_some_and(|c| c.active == active) {
            return Ok(());
        }

        match handle.id() {
            SemanticConstraintId::Algebraic(id) => {
                let record = self.semantic_constraint(handle)?.expect("unknown semantic handle");
                let (lhs, lower, upper) = if !active {
                    (self.__constant(0.0).id(), f64::NEG_INFINITY, f64::INFINITY)
                } else if let Some(view) = self.alternative_scalar_views.borrow().get(&handle.id())
                {
                    (view.lhs, view.lower, view.upper)
                } else {
                    let lhs =
                        scalar_root(&record.pair.function).expect("scalar view lost its function");
                    let set = &record.pair.set;
                    let bounds = set.scalar_bounds().expect("scalar view lost its set");
                    (lhs, bounds.0, bounds.1)
                };

                let mut rows = self.constraints.borrow_mut();
                let mut inactive = self.inactive_scalar_views.borrow_mut();

                if active {
                    inactive.remove(&id);
                } else {
                    let row = &rows[id.index()];
                    inactive.insert(
                        id,
                        crate::semantic::ScalarConstraintView {
                            lhs: row.lhs,
                            lower: row.lower,
                            upper: row.upper,
                        },
                    );
                }
                rows[id.index()].lhs = lhs;
                rows[id.index()].lower = lower;
                rows[id.index()].upper = upper;
                rows[id.index()].active = active;
            }
            SemanticConstraintId::SecondOrderCone(id) => {
                self.soc_constraints.borrow_mut()[id.index()].active = active;
            }
            SemanticConstraintId::PositiveSemidefinite(id) => {
                self.psd_constraints.borrow_mut()[id.index()].active = active;
            }
            SemanticConstraintId::SpecialOrderedSet(id) => {
                self.sos_constraints.borrow_mut()[id.index()].active = active;
            }
            SemanticConstraintId::Indicator(id) => {
                self.indicator_constraints.borrow_mut()[id.index()].active = active;
            }
        }

        let representation = if self.implicit_scalars.get() {
            Representation::Scalar
        } else {
            self.semantics
                .borrow()
                .entry(self.semantic_position(handle.id()))
                .expect("registered position")
                .representation
        };

        if let Representation::Extension(index) = representation {
            self.semantics.borrow_mut().extensions[index as usize].active = active;
        }

        self.invalidate_kind();

        Ok(())
    }

    /// Update a set's values while preserving its signature and semantic identity.
    ///
    /// # Errors
    /// Rejects foreign handles, invalid bounds or dimensions, reformulated sources, GDP dependencies,
    /// or alternative lowerings without an update mapping.
    ///
    /// # Panics
    /// Panics for an unknown handle or an incompatible representation.
    pub fn set_constraint_set<F: crate::SemanticFunction, S: crate::function_set::Set>(
        &self,
        handle: ConstraintHandle<F, S>,
        set: S,
    ) -> Result<(), crate::ConstraintUpdateError> {
        self.ensure_model_id(handle.model_id())?;
        self.ensure_constraint_updatable(handle.id())?;

        let record = self.semantic_constraint(handle)?.expect("unknown semantic handle");
        assert!(
            record.pair.function.signature().is::<F>() && record.pair.set.signature().is::<S>(),
            "handle signature mismatch"
        );

        let value = crate::SetValue::new(set);
        let bounds = value.scalar_bounds();

        if let Some((lower, upper)) = bounds {
            crate::function_set::validate_scalar_bounds(lower, upper)?;

            if let SemanticConstraintId::Algebraic(id) = handle.id() {
                let mut rows = self.constraints.borrow_mut();
                let row = &mut rows[id.index()];

                if row.active {
                    row.lower = lower;
                    row.upper = upper;
                } else if let Some(view) = self.inactive_scalar_views.borrow_mut().get_mut(&id) {
                    view.lower = lower;
                    view.upper = upper;
                }
            }
        } else {
            match handle.id() {
                SemanticConstraintId::SpecialOrderedSet(id) => {
                    let updated =
                        value.downcast_ref::<crate::SpecialOrderedSet>().expect("SOS set");
                    let FunctionRepresentationRef::Variables(variables) =
                        record.pair.function.representation()
                    else {
                        panic!("SOS function")
                    };
                    check_dimension(variables.len(), updated.weights.len())?;
                    let members = self.sos_members(&record.name, variables, &updated.weights);
                    let mut rows = self.sos_constraints.borrow_mut();
                    rows[id.index()].sos_type = updated.kind;
                    rows[id.index()].members = members;
                }
                SemanticConstraintId::Indicator(id) => {
                    let updated =
                        value.downcast_ref::<crate::IndicatorSet>().expect("indicator set");
                    let (lower, upper) = updated
                        .consequent
                        .scalar_bounds()
                        .expect("indicator consequent requires scalar bounds");
                    crate::function_set::validate_scalar_bounds(lower, upper)?;
                    let mut rows = self.indicator_constraints.borrow_mut();
                    rows[id.index()].active_value = updated.active_value;
                    rows[id.index()].lower = lower;
                    rows[id.index()].upper = upper;
                }
                SemanticConstraintId::SecondOrderCone(_) => {
                    check_dimension(
                        record
                            .pair
                            .set
                            .downcast_ref::<SecondOrderCone>()
                            .expect("SOC set")
                            .dimension,
                        value.downcast_ref::<SecondOrderCone>().expect("SOC set").dimension,
                    )?;
                }
                SemanticConstraintId::PositiveSemidefinite(_) => {
                    check_dimension(
                        record
                            .pair
                            .set
                            .downcast_ref::<PositiveSemidefiniteCone>()
                            .expect("PSD set")
                            .side_dimension,
                        value
                            .downcast_ref::<PositiveSemidefiniteCone>()
                            .expect("PSD set")
                            .side_dimension,
                    )?;
                }
                SemanticConstraintId::Algebraic(_) => panic!("scalar view requires scalar bounds"),
            }
        }

        let mut record = record;
        record.pair.set = value;

        self.replace_semantic_record(record);
        self.invalidate_kind();

        Ok(())
    }

    /// Set the source mapping for a declared or generated constraint.
    ///
    /// # Panics
    /// Panics if the semantic ID is not registered.
    pub fn set_constraint_origin(&self, id: SemanticConstraintId, origin: ConstraintOrigin) {
        let position = self.semantic_position(id);

        if !self.implicit_scalars.get() || !matches!(origin, ConstraintOrigin::Declared) {
            self.semantic_writer().records.set_origin(position, origin);
        }
    }
}

impl Model {
    /// Independent structural copy preserving source identities and symbolic parameter references.
    pub fn fork(&self) -> Self {
        self.clone_preserving_ids_with_capacity(0, 0, 0)
    }
}

impl Model {
    /// Register an erased pair with a recognized native representation.
    /// Ownership, registered names and built-in dimensions/bounds are still checked.
    #[doc(hidden)]
    pub fn __register_semantic_pair(
        &self,
        name: impl Into<SmolStr>,
        pair: SemanticPair,
    ) -> ConstraintHandle {
        ConstraintHandle::new(self.register_semantic_pair(name.into(), pair), self.id())
    }

    /// Bind a registered semantic ID to an erased handle owned by this model.
    pub fn semantic_constraint_handle(&self, id: SemanticConstraintId) -> Option<ConstraintHandle> {
        self.semantic_position_opt(id).map(|_| ConstraintHandle::new(id, self.id()))
    }
}

impl Model {
    /// Replace a function's value/structure while retaining its exact semantic signature.
    ///
    /// # Errors
    /// Rejects foreign handles, invalid bounds or dimensions, reformulated sources,
    /// GDP dependencies, or alternative lowerings without an update mapping.
    ///
    /// # Panics
    /// Panics for an unknown handle or an incompatible representation.
    pub fn set_constraint_function<F: crate::SemanticFunction, S: crate::function_set::Set>(
        &self,
        handle: ConstraintHandle<F, S>,
        function: impl crate::Function<Owned = F>,
    ) -> Result<(), crate::ConstraintUpdateError> {
        self.ensure_model_id(handle.model_id())?;
        let function = function.semantic_data();
        self.ensure_model_id(function.model_id())?;
        self.assert_function_roots(&function);

        self.ensure_constraint_updatable(handle.id())?;

        let mut record = self.semantic_constraint(handle)?.expect("unknown semantic handle");
        assert!(
            record.pair.function.signature().is::<F>() && record.pair.set.signature().is::<S>(),
            "handle signature mismatch"
        );

        record.pair.function = crate::FunctionValue::new(function);

        match (handle.id(), record.pair.native_constraint()) {
            (
                SemanticConstraintId::Algebraic(id),
                Some(NativeConstraintRef::Scalar { root: lhs, .. }),
            ) => {
                if record.active {
                    self.constraints.borrow_mut()[id.index()].lhs = lhs;
                } else if let Some(view) = self.inactive_scalar_views.borrow_mut().get_mut(&id) {
                    view.lhs = lhs;
                }
            }
            (
                SemanticConstraintId::SecondOrderCone(id),
                Some(NativeConstraintRef::Soc { components, dimension }),
            ) => {
                check_dimension(dimension, components.len())?;
                assert!(
                    components.iter().all(|&root| oximo_expr::classify(&self.arena.borrow(), root)
                        == oximo_expr::ExprClass::Linear),
                    "SOC function must be affine"
                );
                let mut rows = self.soc_constraints.borrow_mut();
                rows[id.index()].bound = components[0];
                rows[id.index()].terms = record.pair.function.soc_terms(components);
            }
            (
                SemanticConstraintId::PositiveSemidefinite(id),
                Some(NativeConstraintRef::Psd { matrix, dimension }),
            ) => {
                check_dimension(dimension, matrix.side_dimension())?;
                assert!(
                    matrix
                        .upper_triangle()
                        .iter()
                        .all(|&root| oximo_expr::classify(&self.arena.borrow(), root)
                            == oximo_expr::ExprClass::Linear),
                    "PSD function must be affine"
                );
                self.psd_constraints.borrow_mut()[id.index()].matrix = matrix.clone();
            }
            (
                SemanticConstraintId::SpecialOrderedSet(id),
                Some(NativeConstraintRef::Sos { variables, set }),
            ) => {
                check_dimension(set.weights.len(), variables.len())?;
                let members = self.sos_members(&record.name, variables, &set.weights);
                self.sos_constraints.borrow_mut()[id.index()].members = members;
            }
            (
                SemanticConstraintId::Indicator(id),
                Some(NativeConstraintRef::Indicator { trigger, consequent, .. }),
            ) => {
                let lhs = self.checked_indicator_function(trigger, consequent);
                let mut rows = self.indicator_constraints.borrow_mut();
                rows[id.index()].trigger = trigger;
                rows[id.index()].lhs = lhs;
            }
            _ => panic!("function no longer matches its registered representation"),
        }

        self.replace_semantic_record(record);
        self.invalidate_kind();

        Ok(())
    }

    /// Update stored semantics and the native scalar signature after a constraint edit.
    fn replace_semantic_record(&self, record: SemanticConstraint) {
        if let SemanticConstraintId::Algebraic(id) = record.id {
            let signature = ScalarSignature::of(&record.pair);
            self.constraints.borrow_mut()[id.index()].signature = signature;

            if self.implicit_scalars.get()
                && signature.is_some()
                && matches!(record.origin, ConstraintOrigin::Declared)
            {
                return;
            }
        }

        let position = self.semantic_position(record.id);
        self.semantic_writer().records.replace_record(position, record);
    }
}

impl Model {
    /// Check a binary trigger and return its affine consequent's scalar root.
    fn checked_indicator_function(
        &self,
        trigger: oximo_expr::VarId,
        consequent: &crate::FunctionValue,
    ) -> oximo_expr::ExprId {
        assert!(
            self.variables
                .borrow()
                .get(trigger.index())
                .is_some_and(|v| v.domain == crate::Domain::Binary),
            "indicator trigger must be binary"
        );
        assert_eq!(consequent.model_id(), self.id(), "foreign indicator consequent");
        let lhs = scalar_root(consequent).expect("indicator consequent must be scalar");
        assert!(
            oximo_expr::classify(&self.arena.borrow(), lhs) == oximo_expr::ExprClass::Linear,
            "indicator consequent must be affine"
        );
        lhs
    }

    /// Pair registered variables with weights and validate the resulting SOS members.
    fn sos_members(
        &self,
        name: &str,
        variables: crate::VariableVectorRef<'_>,
        weights: &[f64],
    ) -> Vec<crate::SosMember> {
        assert_eq!(variables.len(), weights.len(), "SOS dimension mismatch");
        let count = self.variables.borrow().len();
        assert!(variables.iter().all(|v| v.index() < count), "unknown SOS variable");
        let members: Vec<_> = variables
            .iter()
            .zip(weights)
            .map(|(variable, &weight)| crate::SosMember { variable, weight })
            .collect();
        crate::sos::validate_members(name, &members);
        members
    }
}

impl Model {
    /// Reject edits to reformulated sources or alternative lowerings without an update mapping.
    fn ensure_constraint_updatable(
        &self,
        id: SemanticConstraintId,
    ) -> Result<(), crate::ConstraintUpdateError> {
        self.ensure_constraint_not_reformulated(id)?;

        if self.alternative_scalar_views.borrow().contains_key(&id) {
            return Err(crate::ConstraintUpdateError::AlternativeLowering(id));
        }

        Ok(())
    }

    /// Reject edits to constraints used by GDP, SOS, or indicator reformulations.
    fn ensure_constraint_not_reformulated(
        &self,
        id: SemanticConstraintId,
    ) -> Result<(), crate::ConstraintUpdateError> {
        #[cfg(feature = "gdp")]
        if let SemanticConstraintId::Algebraic(row) = id
            && self.gdp.borrow().locks_constraint(row)
        {
            return Err(crate::ConstraintUpdateError::GdpDependency(id));
        }

        let reformulated = match id {
            SemanticConstraintId::SpecialOrderedSet(id) => {
                self.sos_reformulations.borrow().iter().any(|a| a.source == id)
            }
            SemanticConstraintId::Indicator(id) => {
                self.indicator_reformulations.borrow().iter().any(|a| a.source == id)
            }
            _ => false,
        };

        if reformulated {
            Err(crate::ConstraintUpdateError::ReformulatedSource(id))
        } else {
            Ok(())
        }
    }
}

/// Return the expression root when the function has a scalar representation.
fn scalar_root(function: &crate::FunctionValue) -> Option<oximo_expr::ExprId> {
    match function.representation() {
        FunctionRepresentationRef::Scalar(root) => Some(root),
        _ => None,
    }
}

/// Return a dimension error when the expected and actual sizes differ.
fn check_dimension(expected: usize, actual: usize) -> Result<(), crate::ConstraintUpdateError> {
    if expected == actual {
        Ok(())
    } else {
        Err(crate::ConstraintUpdateError::DimensionMismatch { expected, actual })
    }
}

impl Model {
    /// Check that every expression root referenced by the function exists in this arena.
    fn assert_function_roots(&self, function: &impl crate::SemanticFunction) {
        let arena = self.arena.borrow();
        function.visit_expressions(&mut |id| {
            assert!(id.index() < arena.len(), "invalid expression ID");
        });
    }
}
