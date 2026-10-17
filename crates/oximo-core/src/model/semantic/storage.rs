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

#[derive(Clone, Debug, Default)]
enum SemanticEntries {
    #[default]
    Empty,
    Scalars {
        len: usize,
        origin: u32,
        tail: Vec<SemanticEntry>,
    },
    Uniform {
        first: SemanticEntry,
        len: usize,
    },
    Dense(Vec<SemanticEntry>),
}

impl SemanticEntries {
    /// Count entries across compact prefixes and explicitly stored declarations.
    fn len(&self) -> usize {
        match self {
            Self::Empty => 0,
            Self::Scalars { len, tail, .. } => *len + tail.len(),
            Self::Uniform { len, .. } => *len,
            Self::Dense(entries) => entries.len(),
        }
    }

    /// Read one declaration entry without expanding compact storage.
    fn get(&self, position: usize) -> Option<SemanticEntry> {
        match self {
            Self::Empty => None,
            Self::Scalars { len, origin, tail } => {
                if position < *len {
                    Some(SemanticEntry {
                        id: crate::ConstraintId(u32::try_from(position).expect("registered ID"))
                            .into(),
                        representation: Representation::Scalar,
                        origin: *origin,
                    })
                } else {
                    tail.get(position - len).copied()
                }
            }
            Self::Uniform { first, len } => {
                (position < *len).then(|| first.offset(position)).flatten()
            }
            Self::Dense(entries) => entries.get(position).copied(),
        }
    }

    /// Expand compact entries into a mutable vector.
    fn materialize(&mut self) -> &mut Vec<SemanticEntry> {
        if matches!(self, Self::Scalars { .. }) {
            *self = Self::Dense(
                (0..self.len())
                    .map(|position| self.get(position).expect("registered position"))
                    .collect(),
            );
        } else if let Self::Uniform { first, len } = *self {
            *self = Self::Dense(
                (0..len).map(|offset| first.offset(offset).expect("registered ID")).collect(),
            );
        } else if matches!(self, Self::Empty) {
            *self = Self::Dense(Vec::new());
        }

        let Self::Dense(entries) = self else { unreachable!("materialized entries") };
        entries
    }

    /// Append an entry while retaining compact scalar or uniform runs when possible.
    #[inline]
    fn push(&mut self, entry: SemanticEntry) {
        if let (SemanticConstraintId::Algebraic(id), Representation::Scalar) =
            (entry.id, entry.representation)
        {
            match self {
                Self::Empty if id.index() == 0 => {
                    *self = Self::Scalars { len: 1, origin: entry.origin, tail: Vec::new() };
                    return;
                }
                Self::Scalars { len, origin, tail }
                    if tail.is_empty() && *origin == entry.origin && id.index() == *len =>
                {
                    *len += 1;
                    return;
                }
                _ => {}
            }
        }

        match self {
            Self::Empty => *self = Self::Uniform { first: entry, len: 1 },
            Self::Scalars { tail, .. } => tail.push(entry),
            Self::Uniform { first, len } if entry.follows(*first, *len) => *len += 1,
            _ => self.materialize().push(entry),
        }
    }

    /// Replace one entry, expanding compact storage only when needed.
    fn replace(&mut self, position: usize, entry: SemanticEntry) {
        if let Self::Scalars { len, tail, .. } = self
            && position >= *len
        {
            tail[position - *len] = entry;
            return;
        }

        if let Self::Scalars { origin, .. } = self
            && entry.origin == *origin
            && matches!(entry.id, SemanticConstraintId::Algebraic(id) if id.index() == position)
            && let Representation::Scalar = entry.representation
        {
            return;
        }

        if self.get(position) != Some(entry) {
            if let Self::Scalars { len, tail, .. } = self
                && position.checked_add(1) == Some(*len)
            {
                *len -= 1;
                tail.insert(0, entry);
                return;
            }
            self.materialize()[position] = entry;
        }
    }

    /// Reserve existing entry buffers without expanding compact runs.
    fn reserve(&mut self, additional: usize) {
        match self {
            Self::Scalars { tail, .. } if !tail.is_empty() => tail.reserve(additional),
            Self::Dense(entries) => entries.reserve(additional),
            _ => {}
        }
    }

    /// Count scalar entries stored as an implicit prefix.
    fn scalar_prefix(&self) -> usize {
        match self {
            Self::Scalars { len, .. } => *len,
            _ => 0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct SemanticRegistry {
    positions: Option<Box<super::positions::SemanticPositions>>,
    entries: SemanticEntries,
    pub(crate) extensions: Vec<SemanticConstraint>,
    origins: Vec<ConstraintOrigin>,
}

impl SemanticRegistry {
    /// Initialize metadata for scalar rows already registered in native storage.
    pub(super) fn initialize_scalar_prefix(&mut self, len: usize) {
        debug_assert_eq!(self.len(), 0);

        if len != 0 {
            self.entries = SemanticEntries::Scalars { len, origin: DECLARED, tail: Vec::new() };
        }
    }

    /// Build metadata for a declared scalar row at its native index.
    pub(super) fn declared_scalar(position: usize) -> SemanticEntry {
        SemanticEntry {
            id: crate::ConstraintId(u32::try_from(position).expect("registered ID")).into(),
            representation: Representation::Scalar,
            origin: DECLARED,
        }
    }

    /// Return the number of semantic entries.
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    /// Read declaration metadata by its position.
    pub(crate) fn entry(&self, position: usize) -> Option<SemanticEntry> {
        self.entries.get(position)
    }

    /// Find an ID's declaration position using compact runs or the position index.
    pub(crate) fn position(&self, id: SemanticConstraintId) -> Option<usize> {
        match &self.entries {
            SemanticEntries::Scalars { len, .. } => match id {
                SemanticConstraintId::Algebraic(id) if id.index() < *len => Some(id.index()),
                _ => self.positions.as_ref()?.get(id),
            },
            SemanticEntries::Uniform { first, len }
                if std::mem::discriminant(&id) == std::mem::discriminant(&first.id) =>
            {
                id.index().checked_sub(first.id.index()).filter(|&offset| offset < *len)
            }
            SemanticEntries::Empty | SemanticEntries::Uniform { .. } => None,
            SemanticEntries::Dense(_) => self.positions.as_ref()?.get(id),
        }
    }

    /// Build the ID-to-position index for entries outside the implicit scalar prefix.
    fn index_dense(&mut self) {
        let start = self.entries.scalar_prefix();

        if self.positions.is_none()
            && start < self.len()
            && matches!(self.entries, SemanticEntries::Dense(_) | SemanticEntries::Scalars { .. })
        {
            let mut positions = Box::<super::positions::SemanticPositions>::default();

            for position in start..self.len() {
                let entry = self.entries.get(position).expect("registered position");
                positions.insert(entry.id, position);
            }
            self.positions = Some(positions);
        }
    }

    /// Append metadata and update any required ID-to-position index.
    #[inline]
    fn push_entry(&mut self, entry: SemanticEntry) {
        let position = self.len();
        self.entries.push(entry);

        if position < self.entries.scalar_prefix() {
            return;
        }

        if let Some(positions) = &mut self.positions {
            positions.insert(entry.id, position);
        } else if matches!(
            self.entries,
            SemanticEntries::Dense(_) | SemanticEntries::Scalars { .. }
        ) {
            self.index_dense();
        }
    }

    /// Reserve entry and position-index space for additions to one registry.
    pub(crate) fn reserve(&mut self, registry: SemanticConstraintId, additional: usize) {
        self.entries.reserve(additional);

        if let Some(positions) = &mut self.positions {
            positions.reserve(registry, additional);
        }
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

    /// Replace metadata and rebuild position indexes if the compact scalar prefix changes.
    fn replace_entry(&mut self, position: usize, entry: SemanticEntry) {
        let prefix = self.entries.scalar_prefix();
        self.entries.replace(position, entry);

        if prefix != self.entries.scalar_prefix() {
            self.positions = None;
        }

        self.index_dense();
    }

    /// Recover provenance from an extension, a native row name, or the origin table.
    pub(crate) fn origin<'a>(
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
    pub(crate) fn set_origin(&mut self, position: usize, origin: ConstraintOrigin) {
        let mut entry = self.entries.get(position).expect("registered position");

        if let Representation::Extension(index) = entry.representation {
            self.extensions[index as usize].origin = origin;
        } else {
            entry.origin = self.store_origin(origin);
            self.replace_entry(position, entry);
        }
    }

    /// Update a record, retaining native metadata or moving its payload to extension storage.
    pub(crate) fn replace_record(&mut self, position: usize, record: SemanticConstraint) {
        let mut entry = self.entries.get(position).expect("registered position");

        if let Representation::Extension(index) = entry.representation {
            self.extensions[index as usize] = record;
        } else if let Some(representation) = Representation::of(entry.id, &record.pair) {
            entry.representation = representation;
        } else {
            let index = u32::try_from(self.extensions.len()).expect("extension count overflow");
            entry.representation = Representation::Extension(index);
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
