use crate::SemanticConstraintId;

#[derive(Clone, Copy, Debug)]
struct PositionRun {
    first: usize,
    position: usize,
    len: usize,
}

#[derive(Clone, Debug, Default)]
struct NativePositions {
    runs: smallvec::SmallVec<[PositionRun; 1]>,
}

impl NativePositions {
    fn insert_run(&mut self, first: usize, position: usize, len: usize) {
        if let Some(last) = self.runs.last_mut()
            && last.first + last.len == first
            && last.position + last.len == position
        {
            last.len += len;
            return;
        }
        let index = self.runs.partition_point(|run| run.first < first);
        self.runs.insert(index, PositionRun { first, position, len });
        if index + 1 < self.runs.len() {
            let next = self.runs[index + 1];
            let current = &mut self.runs[index];
            if current.first + current.len == next.first
                && current.position + current.len == next.position
            {
                current.len += next.len;
                self.runs.remove(index + 1);
            }
        }
    }

    fn get(&self, index: usize) -> Option<usize> {
        let run =
            self.runs.get(self.runs.partition_point(|run| run.first <= index).checked_sub(1)?)?;
        let offset = index - run.first;
        (offset < run.len).then(|| run.position + offset)
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct SemanticPositions {
    native: [NativePositions; 5],
}

impl SemanticPositions {
    fn registry(id: SemanticConstraintId) -> usize {
        match id {
            SemanticConstraintId::Algebraic(_) => 0,
            SemanticConstraintId::SecondOrderCone(_) => 1,
            SemanticConstraintId::PositiveSemidefinite(_) => 2,
            SemanticConstraintId::SpecialOrderedSet(_) => 3,
            SemanticConstraintId::Indicator(_) => 4,
        }
    }

    pub(super) fn insert_run(&mut self, id: SemanticConstraintId, position: usize, len: usize) {
        self.native[Self::registry(id)].insert_run(id.index(), position, len);
    }

    #[cfg(test)]
    fn insert(&mut self, id: SemanticConstraintId, position: usize) {
        self.insert_run(id, position, 1);
    }

    pub(super) fn get(&self, id: SemanticConstraintId) -> Option<usize> {
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
        assert_eq!(positions.native[0].runs.len(), 2);
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

        for id in 0..64 {
            positions.insert(ConstraintId(id).into(), id as usize);
        }

        let mut fork = positions.clone();
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
        assert_eq!(positions.native[0].runs.len(), 2);
    }
}
