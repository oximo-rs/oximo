use crate::SemanticConstraintId;

const MISSING: usize = usize::MAX;

#[derive(Clone, Debug, Default)]
struct NativePositions {
    first: usize,
    values: Vec<usize>,
}

impl NativePositions {
    /// Map a native index to its declaration position, storing only the used index range.
    fn insert(&mut self, index: usize, position: usize) {
        if self.values.is_empty() {
            self.first = index;
        } else if index < self.first {
            self.values.splice(0..0, std::iter::repeat_n(MISSING, self.first - index));
            self.first = index;
        }

        let offset = index - self.first;

        if offset >= self.values.len() {
            self.values.resize(offset.checked_add(1).expect("constraint ID overflow"), MISSING);
        }
        self.values[offset] = position;
    }

    /// Look up a native index, returning `None` for gaps or indexes outside the stored range.
    fn get(&self, index: usize) -> Option<usize> {
        index
            .checked_sub(self.first)
            .and_then(|offset| self.values.get(offset))
            .copied()
            .filter(|&position| position != MISSING)
    }

    /// Return whether no native positions have been stored.
    fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct SemanticPositions {
    native: [NativePositions; 5],
    identity: Option<(usize, usize)>,
}

impl SemanticPositions {
    /// Select the position table belonging to a semantic ID's native registry.
    fn registry(id: SemanticConstraintId) -> usize {
        match id {
            SemanticConstraintId::Algebraic(_) => 0,
            SemanticConstraintId::SecondOrderCone(_) => 1,
            SemanticConstraintId::PositiveSemidefinite(_) => 2,
            SemanticConstraintId::SpecialOrderedSet(_) => 3,
            SemanticConstraintId::Indicator(_) => 4,
        }
    }

    /// Expand an implicit identity mapping into the selected registry's position table.
    fn materialize_identity(&mut self) {
        if let Some((registry, len)) = self.identity.take() {
            let positions = &mut self.native[registry];
            positions.first = 0;
            positions.values.extend(0..len);
        }
    }

    /// Reserve only the registry selected by `id`.
    pub(super) fn reserve(&mut self, id: SemanticConstraintId, additional: usize) {
        let registry = Self::registry(id);

        if let Some((existing, _)) = self.identity {
            if existing == registry {
                return;
            }
            self.materialize_identity();
        } else if self.native.iter().all(NativePositions::is_empty) {
            self.identity = Some((registry, 0));
            return;
        }
        self.native[registry].values.reserve(additional);
    }

    /// Store an ID's declaration position.
    #[inline]
    pub(super) fn insert(&mut self, id: SemanticConstraintId, position: usize) {
        let registry = Self::registry(id);

        if let Some((existing, len)) = self.identity.as_mut() {
            if registry == *existing && id.index() == *len && position == *len {
                *len += 1;
                return;
            }
            self.materialize_identity();
        } else if position == 0
            && id.index() == 0
            && self.native.iter().all(NativePositions::is_empty)
        {
            self.identity = Some((registry, 1));
            return;
        }

        self.native[registry].insert(id.index(), position);
    }

    /// Find an ID's declaration position through its implicit or stored mapping.
    #[inline]
    pub(super) fn get(&self, id: SemanticConstraintId) -> Option<usize> {
        if let Some((registry, len)) = self.identity {
            return (Self::registry(id) == registry && id.index() < len).then(|| id.index());
        }

        self.native[Self::registry(id)].get(id.index())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ConstraintId, IndicatorConstraintId, PsdConstraintId, SocConstraintId, SosConstraintId,
    };

    #[test]
    fn native_tail_indexes_do_not_allocate_the_implicit_prefix() {
        let mut positions = SemanticPositions::default();
        positions.insert(ConstraintId(100_000).into(), 100_001);
        positions.insert(ConstraintId(100_002).into(), 100_003);
        assert_eq!(positions.native[0].values.len(), 3);
        assert_eq!(positions.get(ConstraintId(99_999).into()), None);
        assert_eq!(positions.get(ConstraintId(100_001).into()), None);

        positions.insert(ConstraintId(99_999).into(), 100_000);
        assert_eq!(positions.get(ConstraintId(99_999).into()), Some(100_000));
        assert_eq!(positions.get(ConstraintId(100_000).into()), Some(100_001));
        assert_eq!(positions.get(ConstraintId(100_002).into()), Some(100_003));
        assert_eq!(positions.get(ConstraintId(u32::MAX).into()), None);
    }

    #[test]
    fn homogeneous_prefix_survives_reservations_mixing_and_independent_forks() {
        let mut positions = SemanticPositions::default();
        positions.reserve(ConstraintId(0).into(), 64);

        for id in 0..64 {
            positions.insert(ConstraintId(id).into(), id as usize);
        }

        let mut fork = positions.clone();
        positions.reserve(SocConstraintId(0).into(), 2);
        positions.insert(SocConstraintId(0).into(), 64);
        positions.insert(ConstraintId(64).into(), 65);
        fork.insert(ConstraintId(64).into(), 64);

        for id in 0..64 {
            assert_eq!(positions.get(ConstraintId(id).into()), Some(id as usize));
            assert_eq!(fork.get(ConstraintId(id).into()), Some(id as usize));
        }
        assert_eq!(positions.get(ConstraintId(64).into()), Some(65));
        assert_eq!(fork.get(ConstraintId(64).into()), Some(64));
        assert_eq!(positions.get(SocConstraintId(0).into()), Some(64));
        assert_eq!(fork.get(SocConstraintId(0).into()), None);
        assert_eq!(fork.get(ConstraintId(u32::MAX).into()), None);
    }

    #[test]
    fn registry_ids_have_independent_positions() {
        let mut positions = SemanticPositions::default();
        let ids = [
            SemanticConstraintId::SecondOrderCone(SocConstraintId(0)),
            SemanticConstraintId::Algebraic(ConstraintId(0)),
            SemanticConstraintId::PositiveSemidefinite(PsdConstraintId(0)),
            SemanticConstraintId::SpecialOrderedSet(SosConstraintId(0)),
            SemanticConstraintId::Indicator(IndicatorConstraintId(0)),
            SemanticConstraintId::Algebraic(ConstraintId(2)),
        ];

        for (position, id) in ids.into_iter().enumerate() {
            positions.insert(id, position);
        }

        let mut fork = positions.clone();
        fork.insert(ConstraintId(1).into(), 6);

        for (position, id) in ids.into_iter().enumerate() {
            assert_eq!(positions.get(id), Some(position));
            assert_eq!(fork.get(id), Some(position));
        }
        assert_eq!(positions.get(ConstraintId(1).into()), None);
        assert_eq!(fork.get(ConstraintId(1).into()), Some(6));
        assert_eq!(positions.get(ConstraintId(u32::MAX).into()), None);
        assert_eq!(positions.native[0].values.len(), 3);
    }
}
