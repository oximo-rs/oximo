use std::cell::Ref;

use rustc_hash::FxHashMap;

use super::storage::{Representation, SemanticRegistry};
use crate::semantic::{
    FunctionRef, ScalarConstraintView, SemanticConstraintRef, SemanticPairRef, SetRef,
};
use crate::{ConstraintId, Model, ModelConstraints, SemanticConstraintId};

/// A read guard over semantic declarations and their canonical native payloads.
#[derive(Debug)]
pub struct SemanticConstraints<'a> {
    model_id: crate::ModelId,
    records: Ref<'a, SemanticRegistry>,
    native: ModelConstraints<'a>,
    inactive: Ref<'a, FxHashMap<ConstraintId, ScalarConstraintView>>,
    implicit_len: Option<usize>,
}

impl<'a> SemanticConstraints<'a> {
    pub(super) fn new(model: &'a Model) -> Self {
        Self {
            model_id: model.id(),
            records: model.semantics.borrow(),
            native: model.constraints(),
            inactive: model.inactive_scalar_views.borrow(),
            implicit_len: model.implicit_scalars.get().then(|| model.scalar_declarations.get()),
        }
    }

    /// Count all declarations, including inactive constraints.
    pub fn len(&self) -> usize {
        self.implicit_len.unwrap_or_else(|| self.records.len())
    }

    /// Return whether the model has no constraint declarations.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Stable source IDs without constructing function/set views.
    ///
    /// # Panics
    /// Panics during iteration if internal declaration metadata is inconsistent.
    pub fn ids(
        &self,
    ) -> impl ExactSizeIterator<Item = SemanticConstraintId> + DoubleEndedIterator + Clone + '_
    {
        (0..self.len()).map(|position| self.entry(position).expect("registered position").id)
    }

    /// Read metadata at a declaration position without expanding an implicit scalar prefix.
    fn entry(&self, position: usize) -> Option<super::storage::SemanticEntry> {
        if let Some(len) = self.implicit_len {
            (position < len).then(|| SemanticRegistry::declared_scalar(position))
        } else {
            self.records.entry(position)
        }
    }

    /// Iterate over semantic views in declaration order, including inactive constraints.
    ///
    /// # Panics
    /// Panics during iteration if declaration metadata or native constraint storage is inconsistent.
    pub fn iter(
        &self,
    ) -> impl ExactSizeIterator<Item = SemanticConstraintRef<'_>> + DoubleEndedIterator {
        (0..self.len()).map(|position| self.get(position).expect("registered position"))
    }

    /// Build a semantic view at a declaration position, borrowing its registered payloads.
    ///
    /// # Panics
    /// Panics if declaration metadata or native constraint storage is inconsistent.
    pub fn get(&self, position: usize) -> Option<SemanticConstraintRef<'_>> {
        use SemanticConstraintId as Id;
        let entry = self.entry(position)?;
        let model_id = self.model_id;
        let (name, active, function, set) = match (entry.id, entry.representation) {
            (_, Representation::Extension(index)) => {
                let source = &self.records.extensions[index as usize];
                let mut view = source.as_ref();
                view.active = match entry.id {
                    Id::Algebraic(id) => self.native.algebraic()[id.index()].active,
                    Id::SecondOrderCone(id) => self.native.second_order_cones()[id.index()].active,
                    Id::PositiveSemidefinite(id) => {
                        self.native.positive_semidefinite()[id.index()].active
                    }
                    Id::SpecialOrderedSet(id) => {
                        self.native.special_ordered_sets()[id.index()].active
                    }
                    Id::Indicator(id) => self.native.indicators()[id.index()].active,
                };
                return Some(view);
            }
            (Id::Algebraic(id), Representation::Scalar) => {
                let row = &self.native.algebraic()[id.index()];
                let signature = row.signature.expect("canonical scalar signature");
                let (root, lower, upper) = self
                    .inactive
                    .get(&id)
                    .map_or((row.lhs, row.lower, row.upper), |v| (v.lhs, v.lower, v.upper));
                (
                    &row.name,
                    row.active,
                    FunctionRef::Scalar(signature.function.function(model_id, root)),
                    SetRef::Inline(signature.set.set(lower, upper)),
                )
            }
            (Id::SecondOrderCone(id), Representation::Soc) => {
                let row = &self.native.second_order_cones()[id.index()];
                (
                    &row.name,
                    row.active,
                    FunctionRef::Vector { model_id, components: row.terms.with_prefix(1) },
                    SetRef::Inline(crate::SetValue::new(crate::SecondOrderCone {
                        dimension: row.terms.len() + 1,
                    })),
                )
            }
            (Id::PositiveSemidefinite(id), Representation::Psd) => {
                let row = &self.native.positive_semidefinite()[id.index()];
                (
                    &row.name,
                    row.active,
                    FunctionRef::Symmetric { model_id, matrix: &row.matrix },
                    SetRef::Inline(crate::SetValue::new(crate::PositiveSemidefiniteCone {
                        side_dimension: row.matrix.side_dimension(),
                    })),
                )
            }
            (Id::SpecialOrderedSet(id), Representation::Sos) => {
                let row = &self.native.special_ordered_sets()[id.index()];
                (
                    &row.name,
                    row.active,
                    FunctionRef::Variables { model_id, members: &row.members },
                    SetRef::Sos(row),
                )
            }
            (Id::Indicator(id), Representation::Indicator(signature)) => {
                let row = &self.native.indicators()[id.index()];
                (
                    &row.name,
                    row.active,
                    FunctionRef::Indicator {
                        model_id,
                        trigger: row.trigger,
                        consequent: signature.function.function(model_id, row.lhs),
                    },
                    SetRef::Indicator {
                        active_value: row.active_value,
                        consequent: signature.set.set(row.lower, row.upper),
                    },
                )
            }
            _ => unreachable!("native representation must match its registry"),
        };

        Some(SemanticConstraintRef {
            id: entry.id,
            name,
            active,
            pair: SemanticPairRef { function, set },
            origin: self.records.origin(&entry, name),
        })
    }
}
