use std::cell::Ref;

use rustc_hash::FxHashMap;

use super::storage::{Representation, SemanticEntry, SemanticEntryIter, SemanticRegistry};
use crate::semantic::{
    FunctionRef, ScalarConstraintView, SemanticConstraintRef, SemanticPairRef, SetRef,
};
use crate::{ConstraintId, Model, ModelConstraints, SemanticConstraintId};

#[derive(Clone)]
struct SemanticIter<'a, F> {
    entries: SemanticEntryIter<'a>,
    view: F,
}

impl<T, F: FnMut(SemanticEntry) -> T> Iterator for SemanticIter<'_, F> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.entries.next().map(&mut self.view)
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.entries.nth(n).map(&mut self.view)
    }

    fn last(mut self) -> Option<Self::Item> {
        self.entries.last().map(&mut self.view)
    }

    fn count(self) -> usize {
        self.entries.count()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.entries.size_hint()
    }
}

impl<T, F: FnMut(SemanticEntry) -> T> DoubleEndedIterator for SemanticIter<'_, F> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.entries.next_back().map(&mut self.view)
    }

    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.entries.nth_back(n).map(&mut self.view)
    }
}

impl<T, F: FnMut(SemanticEntry) -> T> ExactSizeIterator for SemanticIter<'_, F> {}

/// A read guard over semantic declarations and their canonical native payloads.
#[derive(Debug)]
pub struct SemanticConstraints<'a> {
    model_id: crate::ModelId,
    records: Ref<'a, Option<std::sync::Arc<SemanticRegistry>>>,
    native: ModelConstraints<'a>,
    inactive: Ref<'a, Option<std::sync::Arc<FxHashMap<ConstraintId, ScalarConstraintView>>>>,
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
        self.implicit_len
            .unwrap_or_else(|| self.records.as_deref().map_or(0, SemanticRegistry::len))
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
        SemanticIter { entries: self.entries(), view: |entry: SemanticEntry| entry.id }
    }

    /// Read metadata at a declaration position without expanding an implicit scalar prefix.
    fn entry(&self, position: usize) -> Option<super::storage::SemanticEntry> {
        if let Some(len) = self.implicit_len {
            (position < len).then(|| SemanticRegistry::declared_scalar(position))
        } else {
            self.records.as_deref()?.entry(position)
        }
    }

    /// Iterate over semantic views in declaration order, including inactive constraints.
    ///
    /// # Panics
    /// Panics during iteration if declaration metadata or native constraint storage is inconsistent.
    pub fn iter(
        &self,
    ) -> impl ExactSizeIterator<Item = SemanticConstraintRef<'_>> + DoubleEndedIterator {
        SemanticIter { entries: self.entries(), view: |entry| self.view(entry) }
    }

    /// Build a semantic view at a declaration position, borrowing its registered payloads.
    ///
    /// # Panics
    /// Panics if declaration metadata or native constraint storage is inconsistent.
    pub fn get(&self, position: usize) -> Option<SemanticConstraintRef<'_>> {
        Some(self.view(self.entry(position)?))
    }

    fn entries(&self) -> super::storage::SemanticEntryIter<'_> {
        self.records.as_deref().filter(|_| self.implicit_len.is_none()).map_or_else(
            || super::storage::SemanticEntryIter::implicit(self.implicit_len.unwrap_or(0)),
            SemanticRegistry::iter,
        )
    }

    #[inline]
    fn view(&self, entry: super::storage::SemanticEntry) -> SemanticConstraintRef<'_> {
        use SemanticConstraintId as Id;
        let model_id = self.model_id;
        let (name, active, function, set) = match (entry.id, entry.representation) {
            (_, Representation::Extension(index)) => {
                let source = &self.records.as_deref().expect("registered metadata").extensions
                    [index as usize];
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
                return view;
            }
            (Id::Algebraic(id), Representation::Scalar) => {
                let row = &self.native.algebraic()[id.index()];
                let signature = row.signature.expect("canonical scalar signature");
                let (root, lower, upper) = self
                    .inactive
                    .as_deref()
                    .and_then(|views| views.get(&id))
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

        SemanticConstraintRef {
            id: entry.id,
            name,
            active,
            pair: SemanticPairRef { function, set },
            origin: self.records.as_deref().map_or(
                std::borrow::Cow::Borrowed(&crate::ConstraintOrigin::Declared),
                |records| records.origin(&entry, name),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn seeking_projects_only_returned_entries() {
        let visits = Cell::new(0);
        let make_iter = || SemanticIter {
            entries: SemanticEntryIter::implicit(10),
            view: |entry: SemanticEntry| {
                visits.set(visits.get() + 1);
                entry.id.index()
            },
        };
        let mut iter = make_iter();
        assert_eq!(iter.nth(3), Some(3));
        assert_eq!(visits.get(), 1);
        assert_eq!(iter.nth_back(2), Some(7));
        assert_eq!(visits.get(), 2);
        assert_eq!(iter.next(), Some(4));
        assert_eq!(iter.next_back(), Some(6));
        assert_eq!(iter.len(), 1);
        assert_eq!(iter.last(), Some(5));
        assert_eq!(visits.get(), 5);

        assert_eq!(make_iter().count(), 10);
        let mut empty = make_iter();
        assert_eq!(empty.nth(usize::MAX), None);
        assert_eq!(empty.last(), None);
        assert_eq!(make_iter().nth_back(usize::MAX), None);
        assert_eq!(visits.get(), 5);
    }
}
