use std::cell::{Ref, RefMut};
use std::marker::PhantomData;

use rustc_hash::FxHashMap;
use smol_str::SmolStr;

use crate::{
    ConstraintId, IndicatorConstraintId, PsdConstraintId, SemanticConstraintId, SocConstraintId,
    SosConstraintId,
};

const COLLISION: u64 = 1 << 63;

/// Unique names cost one packed ID.
#[derive(Clone, Default)]
pub(crate) struct ConstraintNames {
    entries: FxHashMap<SmolStr, u64>,
    collisions: Vec<Vec<SemanticConstraintId>>,
}

/// Pack a constraint's registry tag and native index into one name-table value.
fn encode(id: SemanticConstraintId) -> u64 {
    use SemanticConstraintId as Id;
    let (tag, value) = match id {
        Id::Algebraic(v) => (0, v.0),
        Id::SecondOrderCone(v) => (1, v.0),
        Id::PositiveSemidefinite(v) => (2, v.0),
        Id::SpecialOrderedSet(v) => (3, v.0),
        Id::Indicator(v) => (4, v.0),
    };
    (tag << 32) | u64::from(value)
}

/// Recover a semantic ID from a packed name-table value.
fn decode(value: u64) -> SemanticConstraintId {
    let index = u32::try_from(value & u64::from(u32::MAX)).expect("packed constraint ID");

    match value >> 32 {
        0 => ConstraintId(index).into(),
        1 => SocConstraintId(index).into(),
        2 => PsdConstraintId(index).into(),
        3 => SosConstraintId(index).into(),
        4 => IndicatorConstraintId(index).into(),
        _ => unreachable!("registered ID tag"),
    }
}

impl ConstraintNames {
    /// Return the ID only when the name identifies exactly one constraint.
    #[inline]
    pub(crate) fn unique_id(&self, name: &str) -> Option<SemanticConstraintId> {
        let value = *self.entries.get(name)?;
        (value & COLLISION == 0).then(|| decode(value))
    }

    /// Iterate over every constraint registered under the name.
    pub(crate) fn ids(&self, name: &str) -> impl Iterator<Item = SemanticConstraintId> + '_ {
        let binding = self.entries.get(name).copied();
        let single = binding.filter(|value| value & COLLISION == 0).map(decode);
        let multiple = binding
            .filter(|value| value & COLLISION != 0)
            .map(|value| {
                self.collisions[usize::try_from(value & !COLLISION).expect("collision index")]
                    .as_slice()
            })
            .unwrap_or_default();
        single.into_iter().chain(multiple.iter().copied())
    }

    /// Return whether the name exists in any constraint registry.
    pub(crate) fn contains_key(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// Reserve space for additional distinct constraint names.
    pub(crate) fn reserve(&mut self, additional: usize) {
        self.entries.reserve(additional);
    }

    /// Register an ID under a name.
    pub(crate) fn insert(&mut self, name: SmolStr, id: SemanticConstraintId) {
        use std::collections::hash_map::Entry;

        match self.entries.entry(name) {
            Entry::Vacant(entry) => {
                entry.insert(encode(id));
            }
            Entry::Occupied(mut entry) => {
                let old = *entry.get();

                if old & COLLISION != 0 {
                    let ids = &mut self.collisions
                        [usize::try_from(old & !COLLISION).expect("collision index")];
                    assert!(!ids.contains(&id), "constraint name already registered");
                    ids.push(id);
                } else {
                    let old = decode(old);
                    assert_ne!(old, id, "constraint name already registered");
                    let index =
                        u64::try_from(self.collisions.len()).expect("collision count overflow");
                    self.collisions.push(vec![old, id]);
                    entry.insert(COLLISION | index);
                }
            }
        }
    }
}

pub(crate) trait RegistryId: Copy + Into<SemanticConstraintId> {
    /// Return the native ID when the semantic tag matches this registry.
    fn recover(id: SemanticConstraintId) -> Option<Self>;
}

macro_rules! registry_id {
    ($($ty:ty => $variant:ident),* $(,)?) => {
        $(
            impl RegistryId for $ty {
                /// Extract the native ID for this registry's semantic variant.
                fn recover(id: SemanticConstraintId) -> Option<Self> {
                    if let SemanticConstraintId::$variant(id) = id {
                        Some(id)
                    } else {
                        None
                    }
                }
            }
        )*
    };
}

registry_id!(
    ConstraintId => Algebraic,
    SocConstraintId => SecondOrderCone,
    PsdConstraintId => PositiveSemidefinite,
    SosConstraintId => SpecialOrderedSet,
    IndicatorConstraintId => Indicator
);

pub(crate) struct NamesRef<'a, I> {
    names: Ref<'a, ConstraintNames>,
    marker: PhantomData<I>,
}

impl<'a, I: RegistryId> NamesRef<'a, I> {
    /// Restrict a shared name-table borrow to one native registry.
    pub(crate) fn new(names: Ref<'a, ConstraintNames>) -> Self {
        Self { names, marker: PhantomData }
    }

    /// Find the named constraint in this native registry.
    pub(crate) fn get(&self, name: &str) -> Option<I> {
        self.names.ids(name).find_map(I::recover)
    }

    /// Return whether this native registry contains the name.
    pub(crate) fn contains_key(&self, name: &str) -> bool {
        self.get(name).is_some()
    }
}

pub(crate) struct NamesMut<'a, I> {
    names: RefMut<'a, ConstraintNames>,
    marker: PhantomData<I>,
}

impl<'a, I: RegistryId> NamesMut<'a, I> {
    /// Restrict a mutable name-table borrow to one native registry.
    pub(crate) fn new(names: RefMut<'a, ConstraintNames>) -> Self {
        Self { names, marker: PhantomData }
    }

    /// Return whether this native registry contains the name.
    pub(crate) fn contains_key(&self, name: &str) -> bool {
        self.names.ids(name).any(|id| I::recover(id).is_some())
    }

    /// Add this registry's typed ID to the shared name index.
    pub(crate) fn insert(&mut self, name: SmolStr, id: I) {
        self.names.insert(name, id.into());
    }

    /// Register a name, rejecting duplicates within this registry.
    #[inline]
    pub(crate) fn insert_unique(&mut self, name: SmolStr, id: I) {
        use std::collections::hash_map::Entry;
        let ConstraintNames { entries, collisions } = &mut *self.names;

        match entries.entry(name) {
            Entry::Vacant(entry) => {
                entry.insert(encode(id.into()));
            }
            Entry::Occupied(entry) => insert_existing_name(entry, collisions, id),
        }
    }

    /// Reserve shared name-table space for additional names.
    pub(crate) fn reserve(&mut self, additional: usize) {
        self.names.reserve(additional);
    }
}

/// Extend a shared name binding, rejecting a second constraint from the same registry.
#[cold]
#[inline(never)]
fn insert_existing_name<I: RegistryId>(
    mut entry: std::collections::hash_map::OccupiedEntry<'_, SmolStr, u64>,
    collisions: &mut Vec<Vec<SemanticConstraintId>>,
    id: I,
) {
    let old = *entry.get();

    if old & COLLISION == 0 {
        let previous = decode(old);
        assert!(
            I::recover(previous).is_none(),
            "constraint name {:?} already registered",
            entry.key()
        );
        let index = u64::try_from(collisions.len()).expect("collision count overflow");
        collisions.push(vec![previous, id.into()]);
        entry.insert(COLLISION | index);
    } else {
        let ids = &mut collisions[usize::try_from(old & !COLLISION).expect("collision index")];
        assert!(
            !ids.iter().any(|&old| I::recover(old).is_some()),
            "constraint name {:?} already registered",
            entry.key()
        );
        ids.push(id.into());
    }
}
