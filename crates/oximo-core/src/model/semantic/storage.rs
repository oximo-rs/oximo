pub(crate) use crate::semantic::scalar::{BoundKind, ScalarKind, ScalarSignature};
use crate::semantic::{SemanticConstraint, SemanticPair};
use crate::{ConstraintOrigin, SemanticConstraintId};
use oximo_expr::ModelId;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Representation {
    Scalar,
    Soc,
    Psd,
    Sos,
    Indicator(ScalarSignature),
    Extension(u32),
}

impl Representation {
    /// Identify the native metadata form when the pair has its registry's built-in signature.
    fn of(id: SemanticConstraintId, pair: &SemanticPair) -> Option<Self> {
        use SemanticConstraintId as Id;
        Some(match id {
            Id::Algebraic(_) => {
                ScalarSignature::of(pair)?;
                Self::Scalar
            }
            Id::SecondOrderCone(_)
                if pair.function.signature().is::<crate::VectorAffineData>()
                    && pair.set.signature().is::<crate::SecondOrderCone>() =>
            {
                Self::Soc
            }
            Id::PositiveSemidefinite(_)
                if pair.function.signature().is::<crate::SymmetricAffineData>()
                    && pair.set.signature().is::<crate::PositiveSemidefiniteCone>() =>
            {
                Self::Psd
            }
            Id::SpecialOrderedSet(_)
                if pair.function.signature().is::<crate::VariableVectorData>()
                    && pair.set.signature().is::<crate::SpecialOrderedSet>() =>
            {
                Self::Sos
            }
            Id::Indicator(_) => {
                let f = pair.function.downcast_ref::<crate::IndicatorFunctionData>()?;
                let s = pair.set.downcast_ref::<crate::IndicatorSet>()?;
                Self::Indicator(ScalarSignature {
                    function: ScalarKind::of(&f.consequent)?,
                    set: BoundKind::of(&s.consequent)?,
                })
            }
            _ => return None,
        })
    }
}

const DECLARED: u32 = u32::MAX;
const ROW_NAME: u32 = u32::MAX - 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SemanticEntry {
    pub(crate) id: SemanticConstraintId,
    pub(crate) representation: Representation,
    origin: u32,
}

impl SemanticEntry {
    /// Check whether this entry continues a uniform run at the given offset.
    fn follows(self, first: Self, offset: usize) -> bool {
        if self.origin != first.origin
            || std::mem::discriminant(&self.id) != std::mem::discriminant(&first.id)
            || self.id.index().checked_sub(first.id.index()) != Some(offset)
        {
            return false;
        }

        match (self.representation, first.representation) {
            (Representation::Extension(index), Representation::Extension(base)) => {
                (index as usize).checked_sub(base as usize) == Some(offset)
            }
            (current, first) => current == first,
        }
    }

    /// Derive an entry at an offset within a uniform run, returning `None` on overflow.
    fn offset(self, offset: usize) -> Option<Self> {
        use SemanticConstraintId as Id;
        let offset = u32::try_from(offset).ok()?;
        let index = u32::try_from(self.id.index()).ok()?.checked_add(offset)?;
        let id = match self.id {
            Id::Algebraic(_) => Id::Algebraic(crate::ConstraintId(index)),
            Id::SecondOrderCone(_) => Id::SecondOrderCone(crate::SocConstraintId(index)),
            Id::PositiveSemidefinite(_) => Id::PositiveSemidefinite(crate::PsdConstraintId(index)),
            Id::SpecialOrderedSet(_) => Id::SpecialOrderedSet(crate::SosConstraintId(index)),
            Id::Indicator(_) => Id::Indicator(crate::IndicatorConstraintId(index)),
        };

        let representation = match self.representation {
            Representation::Extension(index) => {
                Representation::Extension(index.checked_add(offset)?)
            }
            representation => representation,
        };

        Some(Self { id, representation, origin: self.origin })
    }
}

#[derive(Clone, Copy, Debug)]
struct SemanticRun {
    first: SemanticEntry,
    start: usize,
    len: usize,
}

#[derive(Clone, Debug, Default)]
struct SemanticEntries {
    runs: smallvec::SmallVec<[SemanticRun; 1]>,
    overrides: rustc_hash::FxHashMap<usize, SemanticEntry>,
    len: usize,
}

impl SemanticEntries {
    fn base(&self, position: usize) -> Option<SemanticEntry> {
        if position >= self.len {
            return None;
        }
        let run = &self.runs[self.runs.partition_point(|run| run.start <= position) - 1];
        run.first.offset(position - run.start)
    }

    fn get(&self, position: usize) -> Option<SemanticEntry> {
        self.overrides.get(&position).copied().or_else(|| self.base(position))
    }

    #[inline]
    fn push(&mut self, entry: SemanticEntry) {
        if let Some(last) = self.runs.last_mut()
            && entry.follows(last.first, last.len)
        {
            last.len += 1;
        } else {
            self.runs.push(SemanticRun { first: entry, start: self.len, len: 1 });
        }
        self.len += 1;
    }

    fn replace(&mut self, position: usize, entry: SemanticEntry) {
        let base = self.base(position).expect("registered position");
        debug_assert_eq!(base.id, entry.id);
        if base == entry {
            self.overrides.remove(&position);
        } else {
            self.overrides.insert(position, entry);
        }
    }
}

#[derive(Clone)]
pub(super) struct SemanticEntryIter<'a> {
    entries: Option<&'a SemanticEntries>,
    front: usize,
    back: usize,
    front_run: usize,
    back_run: usize,
}

impl SemanticEntryIter<'_> {
    pub(super) fn implicit(len: usize) -> Self {
        Self { entries: None, front: 0, back: len, front_run: 0, back_run: 0 }
    }

    #[inline]
    fn at(&self, position: usize, run: usize) -> SemanticEntry {
        self.entries.map_or_else(
            || SemanticRegistry::declared_scalar(position),
            |entries| {
                entries.overrides.get(&position).copied().unwrap_or_else(|| {
                    let run = &entries.runs[run];
                    run.first.offset(position - run.start).expect("registered ID")
                })
            },
        )
    }
}

impl Iterator for SemanticEntryIter<'_> {
    type Item = SemanticEntry;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.front == self.back {
            return None;
        }
        if let Some(entries) = self.entries {
            let run = &entries.runs[self.front_run];
            if self.front == run.start + run.len {
                self.front_run += 1;
            }
        }
        let result = self.at(self.front, self.front_run);
        self.front += 1;
        Some(result)
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if n >= self.len() {
            self.front = self.back;
            return None;
        }
        if n != 0 {
            self.front += n;
            if let Some(entries) = self.entries {
                self.front_run = entries.runs.partition_point(|run| run.start <= self.front) - 1;
            }
        }
        self.next()
    }

    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    fn count(self) -> usize {
        self.len()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.back - self.front;
        (len, Some(len))
    }
}

impl DoubleEndedIterator for SemanticEntryIter<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front == self.back {
            return None;
        }
        self.back -= 1;
        if let Some(entries) = self.entries
            && self.back < entries.runs[self.back_run].start
        {
            self.back_run -= 1;
        }
        Some(self.at(self.back, self.back_run))
    }

    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        if n >= self.len() {
            self.back = self.front;
            return None;
        }
        if n != 0 {
            self.back -= n;
            if let Some(entries) = self.entries {
                self.back_run = entries.runs.partition_point(|run| run.start < self.back) - 1;
            }
        }
        self.next_back()
    }
}

impl ExactSizeIterator for SemanticEntryIter<'_> {}

#[derive(Clone, Debug, Default)]
pub(crate) struct SemanticRegistry {
    positions: super::positions::SemanticPositions,
    entries: SemanticEntries,
    pub(crate) extensions: Vec<SemanticConstraint>,
    origins: Vec<ConstraintOrigin>,
    origin_overrides: rustc_hash::FxHashMap<SemanticConstraintId, ConstraintOrigin>,
}

impl SemanticRegistry {
    /// Initialize the existing scalar prefix in constant space and time.
    pub(super) fn initialize_scalar_prefix(&mut self, len: usize) {
        debug_assert_eq!(self.len(), 0);
        if len != 0 {
            let first = Self::declared_scalar(0);
            self.entries.runs.push(SemanticRun { first, start: 0, len });
            self.entries.len = len;
            self.positions.insert_run(first.id, 0, len);
        }
    }

    pub(super) fn declared_scalar(position: usize) -> SemanticEntry {
        SemanticEntry {
            id: crate::ConstraintId(u32::try_from(position).expect("registered ID")).into(),
            representation: Representation::Scalar,
            origin: DECLARED,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len
    }

    pub(crate) fn entry(&self, position: usize) -> Option<SemanticEntry> {
        self.entries.get(position)
    }

    pub(super) fn iter(&self) -> SemanticEntryIter<'_> {
        SemanticEntryIter {
            entries: Some(&self.entries),
            front: 0,
            back: self.len(),
            front_run: 0,
            back_run: self.entries.runs.len().saturating_sub(1),
        }
    }

    pub(crate) fn position(&self, id: SemanticConstraintId) -> Option<usize> {
        self.positions.get(id)
    }

    #[inline]
    fn push_entry(&mut self, entry: SemanticEntry) {
        self.positions.insert_run(entry.id, self.len(), 1);
        self.entries.push(entry);
    }

    /// Store native metadata when possible, otherwise retain the full semantic record.
    pub(crate) fn record(&mut self, record: SemanticConstraint, direct: bool) {
        let representation = direct.then(|| Representation::of(record.id, &record.pair)).flatten();

        if let Some(representation) = representation {
            self.push_named(record.id, representation, record.origin, &record.name);
        } else {
            let index = u32::try_from(self.extensions.len()).expect("extension count overflow");
            self.push(record.id, Representation::Extension(index), ConstraintOrigin::Declared);
            self.extensions.push(record);
        }
    }

    /// Append native metadata with an explicit origin.
    pub(crate) fn push(
        &mut self,
        id: SemanticConstraintId,
        representation: Representation,
        origin: ConstraintOrigin,
    ) {
        self.push_named(id, representation, origin, "");
    }

    /// Append a declaration using the shared declared-origin marker.
    #[inline]
    pub(crate) fn push_declared(
        &mut self,
        id: SemanticConstraintId,
        representation: Representation,
    ) {
        self.push_entry(SemanticEntry { id, representation, origin: DECLARED });
    }

    /// Append a range entry whose origin comes from the native row name.
    #[inline]
    pub(crate) fn push_row_range(
        &mut self,
        id: SemanticConstraintId,
        representation: Representation,
    ) {
        self.push_entry(SemanticEntry { id, representation, origin: ROW_NAME });
    }

    /// Append metadata, reusing the row name for matching range provenance.
    pub(crate) fn push_named(
        &mut self,
        id: SemanticConstraintId,
        representation: Representation,
        origin: ConstraintOrigin,
        name: &str,
    ) {
        let origin = if matches!(&origin, ConstraintOrigin::Range(range) if range == name) {
            ROW_NAME
        } else {
            self.store_origin(origin)
        };
        self.push_entry(SemanticEntry { id, representation, origin });
    }

    /// Encode provenance, reusing the declared marker or the most recent matching range.
    fn store_origin(&mut self, origin: ConstraintOrigin) -> u32 {
        if matches!(origin, ConstraintOrigin::Declared) {
            DECLARED
        } else {
            if let (ConstraintOrigin::Range(name), Some(ConstraintOrigin::Range(previous))) =
                (&origin, self.origins.last())
                && name == previous
            {
                return u32::try_from(self.origins.len() - 1).expect("provenance count overflow");
            }

            let index = u32::try_from(self.origins.len()).expect("provenance count overflow");
            assert!(index < ROW_NAME, "provenance count overflow");
            self.origins.push(origin);
            index
        }
    }

    /// Retain isolated edits without expanding declaration runs or inverse indexes.
    fn replace_entry(&mut self, position: usize, entry: SemanticEntry) {
        self.entries.replace(position, entry);
    }

    /// Recover provenance from an extension, a native row name, or the origin table.
    pub(crate) fn origin<'a>(
        &'a self,
        entry: &SemanticEntry,
        name: &smol_str::SmolStr,
    ) -> std::borrow::Cow<'a, ConstraintOrigin> {
        if let Some(origin) = self.origin_overrides.get(&entry.id) {
            return std::borrow::Cow::Borrowed(origin);
        }
        self.base_origin(entry, name)
    }

    fn base_origin<'a>(
        &'a self,
        entry: &SemanticEntry,
        name: &smol_str::SmolStr,
    ) -> std::borrow::Cow<'a, ConstraintOrigin> {
        if let Representation::Extension(index) = entry.representation {
            std::borrow::Cow::Borrowed(&self.extensions[index as usize].origin)
        } else if entry.origin == DECLARED {
            std::borrow::Cow::Borrowed(&ConstraintOrigin::Declared)
        } else if entry.origin == ROW_NAME {
            std::borrow::Cow::Owned(ConstraintOrigin::Range(name.clone()))
        } else {
            std::borrow::Cow::Borrowed(&self.origins[entry.origin as usize])
        }
    }

    /// Change an entry's provenance while preserving its declaration position.
    pub(crate) fn set_origin(
        &mut self,
        position: usize,
        origin: ConstraintOrigin,
        name: &smol_str::SmolStr,
    ) {
        let entry = self.entry(position).expect("registered position");

        if let Representation::Extension(index) = entry.representation {
            self.extensions[index as usize].origin = origin;
        } else {
            let base = self.base_origin(&entry, name);
            let equal = match (base.as_ref(), &origin) {
                (ConstraintOrigin::Declared, ConstraintOrigin::Declared) => true,
                (ConstraintOrigin::Range(a), ConstraintOrigin::Range(b)) => a == b,
                (
                    ConstraintOrigin::Generated { sources: a, transformation: at },
                    ConstraintOrigin::Generated { sources: b, transformation: bt },
                ) => a == b && at == bt,
                _ => false,
            };
            if equal {
                self.origin_overrides.remove(&entry.id);
            } else {
                self.origin_overrides.insert(entry.id, origin);
            }
        }
    }

    /// Update a record, retaining native metadata or moving its payload to extension storage.
    pub(crate) fn replace_record(&mut self, position: usize, record: SemanticConstraint) {
        let mut entry = self.entry(position).expect("registered position");

        if let Representation::Extension(index) = entry.representation {
            self.extensions[index as usize] = record;
        } else if let Some(representation) = Representation::of(entry.id, &record.pair) {
            entry.representation = representation;
        } else {
            let index = u32::try_from(self.extensions.len()).expect("extension count overflow");
            entry.representation = Representation::Extension(index);
            self.origin_overrides.remove(&entry.id);
            self.extensions.push(record);
        }
        self.replace_entry(position, entry);
    }

    /// Rebind retained extension functions to another model identity.
    pub(crate) fn rebind_extensions(&mut self, model_id: ModelId) {
        for record in &mut self.extensions {
            record.pair.function.rebind_model(model_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeking_matches_dense_iteration_across_runs_overrides_and_exhaustion() {
        fn check(iter: SemanticEntryIter<'_>, expected: &[SemanticEntry]) {
            for front in 0..=expected.len() {
                for back in 0..=expected.len() - front {
                    let mut remaining = iter.clone();
                    for _ in 0..front {
                        remaining.next();
                    }
                    for _ in 0..back {
                        remaining.next_back();
                    }
                    let expected = &expected[front..expected.len() - back];
                    assert_eq!(remaining.clone().count(), expected.len());
                    assert_eq!(remaining.clone().last(), expected.last().copied());
                    for n in (0..=expected.len()).chain([usize::MAX]) {
                        let mut forward = remaining.clone();
                        let mut backward = remaining.clone();
                        let mut dense_forward = expected.iter().copied();
                        let mut dense_backward = expected.iter().copied();
                        assert_eq!(forward.nth(n), dense_forward.nth(n));
                        assert_eq!(backward.nth_back(n), dense_backward.nth_back(n));
                        assert_eq!(forward.len(), dense_forward.len());
                        assert_eq!(backward.len(), dense_backward.len());
                        assert_eq!(forward.clone().last(), dense_forward.clone().last());
                        assert_eq!(backward.clone().last(), dense_backward.clone().last());
                        assert_eq!(forward.next_back(), dense_forward.next_back());
                        assert_eq!(backward.next(), dense_backward.next());
                        assert_eq!(forward.collect::<Vec<_>>(), dense_forward.collect::<Vec<_>>());
                        assert_eq!(
                            backward.collect::<Vec<_>>(),
                            dense_backward.collect::<Vec<_>>()
                        );
                    }
                }
            }
        }

        let mut records = SemanticRegistry::default();
        records.initialize_scalar_prefix(3);
        for index in 0..2 {
            records.push_declared(crate::SocConstraintId(index).into(), Representation::Soc);
        }
        for index in 3..7 {
            records.push_row_range(crate::ConstraintId(index).into(), Representation::Scalar);
        }
        for position in [2, 3, 7] {
            let mut entry = records.entry(position).unwrap();
            entry.representation = Representation::Extension(0);
            records.replace_entry(position, entry);
        }
        let expected: Vec<_> = (0..records.len()).map(|i| records.entry(i).unwrap()).collect();
        check(records.iter(), &expected);

        let implicit: Vec<_> = (0..9).map(SemanticRegistry::declared_scalar).collect();
        check(SemanticEntryIter::implicit(implicit.len()), &implicit);
        check(SemanticRegistry::default().iter(), &[]);
    }

    #[test]
    fn sparse_origins_do_not_expand_a_million_scalar_declarations() {
        let mut records = SemanticRegistry::default();
        records.initialize_scalar_prefix(1_000_000);
        for position in [0, 500_000, 999_999] {
            records.set_origin(position, ConstraintOrigin::Range("edited".into()), &"row".into());
        }
        assert_eq!(records.entries.runs.len(), 1);
        assert_eq!(records.entries.overrides.len(), 0);
        assert_eq!(records.origin_overrides.len(), 3);
        assert!(records.origins.is_empty());
        for position in [0, 500_000, 999_999] {
            assert_eq!(
                records.position(crate::ConstraintId(position).into()),
                Some(position as usize)
            );
            records.set_origin(position as usize, ConstraintOrigin::Declared, &"row".into());
        }
        assert!(records.origin_overrides.is_empty());
        assert_eq!(records.entries.runs.len(), 1);
    }

    #[test]
    fn mixed_runs_sparse_representations_and_bidirectional_iteration_keep_positions() {
        let mut records = SemanticRegistry::default();
        records.initialize_scalar_prefix(100_000);
        records.push_declared(crate::SocConstraintId(0).into(), Representation::Soc);
        for index in 100_000..200_000 {
            records.push_row_range(crate::ConstraintId(index).into(), Representation::Scalar);
        }
        assert_eq!(records.entries.runs.len(), 3);
        assert_eq!(records.positions.get(crate::ConstraintId(199_999).into()), Some(200_000));

        let mut changed = records.entry(50_000).unwrap();
        changed.representation = Representation::Extension(0);
        records.replace_entry(50_000, changed);
        assert_eq!(records.entries.overrides.len(), 1);
        assert_eq!(records.entry(50_000), Some(changed));
        records.replace_entry(50_000, SemanticRegistry::declared_scalar(50_000));
        assert!(records.entries.overrides.is_empty());

        let expected: Vec<_> = (0..records.len()).map(|i| records.entry(i).unwrap().id).collect();
        assert_eq!(records.iter().map(|entry| entry.id).collect::<Vec<_>>(), expected);
        assert_eq!(
            records.iter().rev().map(|entry| entry.id).collect::<Vec<_>>(),
            expected.iter().rev().copied().collect::<Vec<_>>()
        );
        let mut iter = records.iter();
        let mut front = 0;
        let mut back = expected.len();
        while front < back {
            assert_eq!(iter.next().unwrap().id, expected[front]);
            front += 1;
            if front < back {
                back -= 1;
                assert_eq!(iter.next_back().unwrap().id, expected[back]);
            }
            assert_eq!(iter.len(), back - front);
        }
        assert!(iter.next().is_none());
        assert!(iter.next_back().is_none());
    }
}
