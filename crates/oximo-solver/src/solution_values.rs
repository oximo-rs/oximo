//! Scalar solution values indexed by model coordinates.

use std::collections::TryReserveError;
use std::iter::FusedIterator;
use std::marker::PhantomData;
use std::ops::Range;
use std::slice;

use oximo_core::{ConstraintId, SocConstraintId, VarId};

mod sealed {
    pub trait Sealed {}
}

/// A scalar solution coordinate.
pub trait SolutionValueId: sealed::Sealed + Copy + std::fmt::Debug {
    #[doc(hidden)]
    fn index(self) -> usize;
    #[doc(hidden)]
    fn from_index(index: usize) -> Self;
}

macro_rules! solution_ids {
    ($($id:ident),+ $(,)?) => {$(
        impl sealed::Sealed for $id {}
        impl SolutionValueId for $id {
            fn index(self) -> usize { self.index() }
            fn from_index(index: usize) -> Self {
                Self(u32::try_from(index).expect("solution coordinate exceeds u32"))
            }
        }
    )+};
}
solution_ids!(VarId, ConstraintId, SocConstraintId);

/// Dense scalar values with explicit missing slots.
///
/// IDs address slots and are never renumbered. `set` and `try_set` keep the
/// slot count fixed, while `insert`, `try_insert`, and `extend` can grow it.
///
/// Coordinate kinds cannot be mixed:
///
/// ```compile_fail
/// use oximo_core::{ConstraintId, VarId};
/// use oximo_solver::SolutionValues;
/// let primal = SolutionValues::<VarId>::from_dense(vec![1.0]);
/// primal.get(ConstraintId(0));
/// ```
#[derive(Clone, Debug)]
pub struct SolutionValues<I: SolutionValueId> {
    values: Vec<f64>,
    presence: Option<Vec<u64>>,
    len: usize,
    id: PhantomData<I>,
}

impl<I: SolutionValueId> Default for SolutionValues<I> {
    fn default() -> Self {
        Self::from_dense(Vec::new())
    }
}

/// A coordinate outside the fixed solution slot count.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("solution ID {index} is outside slot count {slot_count}")]
pub struct SolutionIndexError {
    pub index: usize,
    pub slot_count: usize,
}

impl<I: SolutionValueId> SolutionValues<I> {
    /// Adopt a complete array in ID order without copying its allocation.
    ///
    /// # Panics
    /// Panics if the slot count cannot be represented by the ID type.
    #[must_use]
    pub fn from_dense(values: Vec<f64>) -> Self {
        Self::check_slot_count(values.len());
        Self { len: values.len(), values, presence: None, id: PhantomData }
    }

    /// Allocate a fixed slot count with every slot missing.
    ///
    /// # Panics
    /// Panics if the slot count cannot be represented by the ID type.
    #[must_use]
    pub fn missing(slot_count: usize) -> Self {
        Self::check_slot_count(slot_count);
        if slot_count == 0 {
            return Self::default();
        }
        Self {
            values: vec![0.0; slot_count],
            presence: Some(vec![0; slot_count.div_ceil(64)]),
            len: 0,
            id: PhantomData,
        }
    }

    /// Fill a known slot count directly.
    ///
    /// # Errors
    /// Returns the first out-of-range coordinate. No partial result escapes.
    ///
    /// # Panics
    /// Panics if the slot count cannot be represented by the ID type.
    pub fn try_from_pairs(
        slot_count: usize,
        entries: impl IntoIterator<Item = (I, f64)>,
    ) -> Result<Self, SolutionIndexError> {
        let mut result = Self::missing(slot_count);
        for (id, value) in entries {
            result.try_set(id, Some(value))?;
        }
        Ok(result)
    }

    fn check_slot_count(slot_count: usize) {
        assert!(
            slot_count == 0 || u32::try_from(slot_count - 1).is_ok(),
            "solution slot count exceeds u32"
        );
    }

    /// Number of present values.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of addressable slots, including missing values.
    #[must_use]
    pub fn slot_count(&self) -> usize {
        self.values.len()
    }

    #[inline]
    fn present(&self, index: usize) -> bool {
        self.presence.as_ref().is_none_or(|bits| bits[index / 64] & (1 << (index % 64)) != 0)
    }

    /// Read a present value.
    /// Missing and out-of-range IDs return `None`.
    #[inline]
    #[must_use]
    pub fn get(&self, id: I) -> Option<f64> {
        let index = id.index();
        self.values.get(index).filter(|_| self.present(index)).copied()
    }

    /// Set or remove a slot, returning its previous value.
    ///
    /// # Panics
    /// Panics if `id` is outside [`Self::slot_count`].
    #[track_caller]
    pub fn set(&mut self, id: I, value: Option<f64>) -> Option<f64> {
        self.try_set(id, value).unwrap_or_else(|error| {
            panic!("solution ID {id:?} outside slot count {}", error.slot_count)
        })
    }

    /// Set or remove a slot without growing the slot count.
    ///
    /// # Errors
    /// Returns an out-of-range error without changing the container.
    #[must_use = "handle the coordinate error or explicitly discard the previous value"]
    pub fn try_set(
        &mut self,
        id: I,
        value: Option<f64>,
    ) -> Result<Option<f64>, SolutionIndexError> {
        let index = self.checked_index(id)?;
        Ok(self.replace_at(index, value))
    }

    fn checked_index(&self, id: I) -> Result<usize, SolutionIndexError> {
        let index = id.index();
        if index >= self.slot_count() {
            Err(SolutionIndexError { index, slot_count: self.slot_count() })
        } else {
            Ok(index)
        }
    }

    fn replace_at(&mut self, index: usize, value: Option<f64>) -> Option<f64> {
        let present = self.present(index);
        let previous = present.then_some(self.values[index]);
        match (present, value.is_some()) {
            (false, true) => self.len += 1,
            (true, false) => self.len -= 1,
            _ => {}
        }

        if let Some(value) = value {
            self.values[index] = value;
        } else if self.presence.is_none() {
            self.presence = Some(full_bitmap(self.slot_count()));
        }

        if let Some(bits) = &mut self.presence {
            let mask = 1 << (index % 64);
            if value.is_some() {
                bits[index / 64] |= mask;
            } else {
                bits[index / 64] &= !mask;
            }
        }

        if self.len == self.slot_count() {
            self.presence = None;
        }
        previous
    }

    /// Insert a value, growing the slot count through `id` when necessary.
    /// Prefer bounded `try_set` for external IDs.
    ///
    /// # Panics
    /// Panics if the larger buffers cannot be allocated.
    #[track_caller]
    pub fn insert(&mut self, id: I, value: f64) -> Option<f64> {
        self.try_insert(id, value).unwrap_or_else(|error| panic!("cannot insert {id:?}: {error}"))
    }

    /// Insert a value.
    ///
    /// # Errors
    /// Returns an allocation/capacity error without changing values or slot count.
    pub fn try_insert(&mut self, id: I, value: f64) -> Result<Option<f64>, TryReserveError> {
        let index = id.index();
        let old_count = self.slot_count();
        if index < old_count {
            return Ok(self.replace_at(index, Some(value)));
        }

        let slot_count = index.saturating_add(1);
        self.values.try_reserve(slot_count - old_count)?;
        if index == old_count {
            if let Some(bits) = &mut self.presence {
                bits.try_reserve(slot_count.div_ceil(64) - bits.len())?;
                bits.resize(slot_count.div_ceil(64), 0);
                bits[index / 64] |= 1 << (index % 64);
            }

            self.values.push(value);
        } else {
            let words = slot_count.div_ceil(64);
            if let Some(bits) = &mut self.presence {
                bits.try_reserve(words - bits.len())?;
                bits.resize(words, 0);
            } else {
                let mut bits = Vec::new();
                bits.try_reserve_exact(words)?;
                bits.resize(old_count.div_ceil(64), u64::MAX);
                if !old_count.is_multiple_of(64) {
                    bits[old_count / 64] &= low_mask(old_count % 64);
                }
                bits.resize(words, 0);
                self.presence = Some(bits);
            }

            self.values.resize(slot_count, 0.0);
            self.values[index] = value;
            if let Some(bits) = &mut self.presence {
                bits[index / 64] |= 1 << (index % 64);
            }
        }
        self.len += 1;

        if self.len == self.slot_count() {
            self.presence = None;
        }

        Ok(None)
    }

    /// Borrow a present value mutably.
    /// Missing or out-of-range IDs return `None`.
    #[must_use]
    pub fn get_mut(&mut self, id: I) -> Option<&mut f64> {
        let index = id.index();
        if index < self.slot_count() && self.present(index) {
            self.values.get_mut(index)
        } else {
            None
        }
    }

    /// Add to a slot, initializing a missing value to zero.
    ///
    /// # Errors
    /// Returns an out-of-range error without changing the container.
    pub fn try_accumulate(&mut self, id: I, value: f64) -> Result<(), SolutionIndexError> {
        let index = self.checked_index(id)?;
        if let Some(bits) = &mut self.presence {
            let word = &mut bits[index / 64];
            let mask = 1 << (index % 64);

            if *word & mask == 0 {
                *word |= mask;
                self.values[index] = 0.0;
                self.len += 1;
                if self.len == self.values.len() {
                    self.presence = None;
                }
            }
        }

        self.values[index] += value;
        Ok(())
    }

    /// Whether an ID has an available value, including an available zero.
    #[must_use]
    pub fn contains(&self, id: I) -> bool {
        self.get(id).is_some()
    }

    /// Whether every slot has a value.
    /// Complete-empty containers return `true`.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.len == self.slot_count()
    }

    /// Complete values in ID order, or `None` when any slot is missing.
    /// Check the expected element count when interpreting model arrays.
    #[must_use]
    pub fn as_dense(&self) -> Option<&[f64]> {
        (self.len == self.values.len()).then_some(self.values.as_slice())
    }

    /// Recover a complete owned array without copying.
    /// Completeness covers only this container's slot count.
    /// An empty container succeeds even when an external model has coordinates.
    ///
    /// # Errors
    /// Returns the original container, unchanged, when any slot is missing.
    #[must_use = "use the recovered buffer or handle the partial solution"]
    pub fn into_dense(self) -> Result<Vec<f64>, Self> {
        if self.len == self.values.len() { Ok(self.values) } else { Err(self) }
    }

    /// Present entries in ascending ID order, with exact remaining length.
    #[must_use]
    pub fn iter(&self) -> SolutionIter<'_, I> {
        let inner = if self.is_complete() {
            EntryIter::Dense(self.values.iter().enumerate())
        } else {
            EntryIter::Sparse { values: &self.values, slots: self.sparse_slots() }
        };
        SolutionIter { inner, id: PhantomData }
    }

    /// Present values directly. Complete storage delegates to slice iteration.
    #[must_use]
    pub fn values(&self) -> SolutionValuesIter<'_> {
        let inner = if self.is_complete() {
            ValueIter::Dense(self.values.iter())
        } else {
            ValueIter::Sparse { values: &self.values, slots: self.sparse_slots() }
        };
        SolutionValuesIter { inner }
    }

    /// Present IDs in ascending order, without reading stored values.
    #[must_use]
    pub fn keys(&self) -> SolutionKeys<'_, I> {
        let inner = if self.is_complete() {
            KeyIter::Dense(0..self.slot_count())
        } else {
            KeyIter::Sparse(self.sparse_slots())
        };

        SolutionKeys { inner, id: PhantomData }
    }

    fn sparse_slots(&self) -> SparseSlots<'_> {
        SparseSlots {
            bits: self.presence.as_deref().unwrap_or(&[]),
            front: 0,
            back: self.slot_count(),
            remaining: self.len,
        }
    }

    /// Remove rejected entries, preserving IDs.
    pub fn retain(&mut self, mut keep: impl FnMut(I, f64) -> bool) {
        if self.len == 0 {
            return;
        }

        if let Some(bits) = &mut self.presence {
            for (word_index, word) in bits.iter_mut().enumerate() {
                let mut pending = *word;
                let mut kept = pending;
                while pending != 0 {
                    let bit = pending.trailing_zeros() as usize;
                    let mask = 1 << bit;
                    let index = word_index * 64 + bit;
                    if !keep(I::from_index(index), self.values[index]) {
                        kept &= !mask;
                    }
                    pending &= pending - 1;
                }

                // Commit a whole word after callbacks
                self.len -= (*word ^ kept).count_ones() as usize;
                *word = kept;
            }
        } else {
            for (index, &value) in self.values.iter().enumerate() {
                if !keep(I::from_index(index), value) {
                    let bits = self.presence.get_or_insert_with(|| full_bitmap(self.values.len()));
                    bits[index / 64] &= !(1 << (index % 64));
                    self.len -= 1;
                }
            }
        }
    }

    /// Remove all values while preserving the slot count.
    pub fn clear(&mut self) {
        if self.len == 0 {
            return;
        }
        self.presence.get_or_insert_with(|| vec![0; self.values.len().div_ceil(64)]).fill(0);
        self.len = 0;
    }

    /// Discard a suffix without changing remaining IDs.
    pub fn truncate(&mut self, slot_count: usize) {
        let old_count = self.values.len();
        if slot_count >= old_count {
            return;
        }

        if let Some(bits) = &self.presence {
            if slot_count < old_count - slot_count {
                self.len = count_bits(bits, 0, slot_count);
            } else {
                self.len -= count_bits(bits, slot_count, old_count);
            }
        } else {
            self.len = slot_count;
        }

        self.values.truncate(slot_count);
        if self.len == slot_count {
            self.presence = None;
        } else if let Some(bits) = &mut self.presence {
            bits.truncate(slot_count.div_ceil(64));
            if !slot_count.is_multiple_of(64) {
                bits[slot_count / 64] &= low_mask(slot_count % 64);
            }
        }
    }

    /// Release excess allocation without changing the slot count or values.
    pub fn compact(&mut self) {
        self.values.shrink_to_fit();
        if let Some(bits) = &mut self.presence {
            bits.shrink_to_fit();
        }
    }

    // Avoid copying ordinary small spare buffers.
    pub(crate) fn compact_if_excess(&mut self) {
        let capacity = self.values.capacity();
        if capacity.saturating_sub(self.values.len()) >= 128 * 1024
            && self.values.len() <= capacity / 2
        {
            self.compact();
        }
    }
}

fn full_bitmap(slot_count: usize) -> Vec<u64> {
    let mut bits = vec![u64::MAX; slot_count.div_ceil(64)];
    if !slot_count.is_multiple_of(64) {
        bits[slot_count / 64] &= low_mask(slot_count % 64);
    }
    bits
}

fn low_mask(count: usize) -> u64 {
    if count == 64 { u64::MAX } else { (1_u64 << count) - 1 }
}

fn masked_word(bits: &[u64], word: usize, front: usize, back: usize) -> u64 {
    let mut value = bits[word];
    if word == front / 64 {
        value &= u64::MAX << (front % 64);
    }

    if word == (back - 1) / 64 {
        value &= low_mask((back - 1) % 64 + 1);
    }

    value
}

fn count_bits(bits: &[u64], front: usize, back: usize) -> usize {
    if front == back {
        return 0;
    }
    (front / 64..=(back - 1) / 64)
        .map(|word| masked_word(bits, word, front, back).count_ones() as usize)
        .sum()
}

#[derive(Clone, Debug)]
struct SparseSlots<'a> {
    bits: &'a [u64],
    front: usize,
    back: usize,
    remaining: usize,
}
impl SparseSlots<'_> {
    fn exhaust(&mut self) -> Option<usize> {
        self.front = self.back;
        self.remaining = 0;
        None
    }

    fn nth(&mut self, mut n: usize) -> Option<usize> {
        if n >= self.remaining {
            return self.exhaust();
        }

        loop {
            let word = self.front / 64;
            let mut value = masked_word(self.bits, word, self.front, self.back);
            let count = value.count_ones() as usize;
            if n < count {
                for _ in 0..n {
                    value &= value - 1;
                }
                let index = word * 64 + value.trailing_zeros() as usize;
                self.front = index + 1;
                self.remaining -= n + 1;
                return Some(index);
            }
            n -= count;
            self.remaining -= count;
            self.front = ((word + 1) * 64).min(self.back);
        }
    }

    fn nth_back(&mut self, mut n: usize) -> Option<usize> {
        if n >= self.remaining {
            return self.exhaust();
        }
        loop {
            let word = (self.back - 1) / 64;
            let mut value = masked_word(self.bits, word, self.front, self.back);
            let count = value.count_ones() as usize;
            if n < count {
                for _ in 0..n {
                    value &= !(1 << value.ilog2());
                }
                let index = word * 64 + value.ilog2() as usize;
                self.back = index;
                self.remaining -= n + 1;
                return Some(index);
            }
            n -= count;
            self.remaining -= count;
            self.back = (word * 64).max(self.front);
        }
    }

    fn fold<B>(self, mut acc: B, mut fold: impl FnMut(B, usize) -> B) -> B {
        if self.remaining == 0 {
            return acc;
        }
        for word in self.front / 64..=(self.back - 1) / 64 {
            let mut pending = masked_word(self.bits, word, self.front, self.back);
            while pending != 0 {
                let index = word * 64 + pending.trailing_zeros() as usize;
                acc = fold(acc, index);
                pending &= pending - 1;
            }
        }
        acc
    }
}

#[derive(Clone, Debug)]
enum EntryIter<'a> {
    Dense(std::iter::Enumerate<slice::Iter<'a, f64>>),
    Sparse { values: &'a [f64], slots: SparseSlots<'a> },
}

/// Present `(ID, value)` pairs.
#[derive(Clone, Debug)]
pub struct SolutionIter<'a, I: SolutionValueId> {
    inner: EntryIter<'a>,
    id: PhantomData<I>,
}
impl<I: SolutionValueId> Iterator for SolutionIter<'_, I> {
    type Item = (I, f64);

    fn next(&mut self) -> Option<Self::Item> {
        self.nth(0)
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        match &mut self.inner {
            EntryIter::Dense(iter) => {
                iter.nth(n).map(|(index, &value)| (I::from_index(index), value))
            }
            EntryIter::Sparse { values, slots } => {
                slots.nth(n).map(|index| (I::from_index(index), values[index]))
            }
        }
    }

    fn count(self) -> usize {
        self.len()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = match &self.inner {
            EntryIter::Dense(iter) => iter.len(),
            EntryIter::Sparse { slots, .. } => slots.remaining,
        };
        (len, Some(len))
    }

    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    fn fold<B, F: FnMut(B, Self::Item) -> B>(self, init: B, mut fold: F) -> B {
        match self.inner {
            EntryIter::Dense(iter) => {
                iter.fold(init, |acc, (index, &value)| fold(acc, (I::from_index(index), value)))
            }
            EntryIter::Sparse { values, slots } => {
                slots.fold(init, |acc, index| fold(acc, (I::from_index(index), values[index])))
            }
        }
    }
}
impl<I: SolutionValueId> DoubleEndedIterator for SolutionIter<'_, I> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.nth_back(0)
    }
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        match &mut self.inner {
            EntryIter::Dense(iter) => {
                iter.nth_back(n).map(|(index, &value)| (I::from_index(index), value))
            }
            EntryIter::Sparse { values, slots } => {
                slots.nth_back(n).map(|index| (I::from_index(index), values[index]))
            }
        }
    }
}
impl<I: SolutionValueId> ExactSizeIterator for SolutionIter<'_, I> {}
impl<I: SolutionValueId> FusedIterator for SolutionIter<'_, I> {}

#[derive(Clone, Debug)]
enum ValueIter<'a> {
    Dense(slice::Iter<'a, f64>),
    Sparse { values: &'a [f64], slots: SparseSlots<'a> },
}

/// Present values without constructing IDs.
#[derive(Clone, Debug)]
pub struct SolutionValuesIter<'a> {
    inner: ValueIter<'a>,
}
impl Iterator for SolutionValuesIter<'_> {
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        self.nth(0)
    }

    fn nth(&mut self, n: usize) -> Option<f64> {
        match &mut self.inner {
            ValueIter::Dense(iter) => iter.nth(n).copied(),
            ValueIter::Sparse { values, slots } => slots.nth(n).map(|index| values[index]),
        }
    }

    fn count(self) -> usize {
        self.len()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = match &self.inner {
            ValueIter::Dense(iter) => iter.len(),
            ValueIter::Sparse { slots, .. } => slots.remaining,
        };
        (len, Some(len))
    }

    fn last(mut self) -> Option<f64> {
        self.next_back()
    }

    fn fold<B, F: FnMut(B, f64) -> B>(self, init: B, mut fold: F) -> B {
        match self.inner {
            ValueIter::Dense(iter) => iter.copied().fold(init, fold),
            ValueIter::Sparse { values, slots } => {
                slots.fold(init, |acc, index| fold(acc, values[index]))
            }
        }
    }
}
impl DoubleEndedIterator for SolutionValuesIter<'_> {
    fn next_back(&mut self) -> Option<f64> {
        self.nth_back(0)
    }

    fn nth_back(&mut self, n: usize) -> Option<f64> {
        match &mut self.inner {
            ValueIter::Dense(iter) => iter.nth_back(n).copied(),
            ValueIter::Sparse { values, slots } => slots.nth_back(n).map(|index| values[index]),
        }
    }
}
impl ExactSizeIterator for SolutionValuesIter<'_> {}
impl FusedIterator for SolutionValuesIter<'_> {}

#[derive(Clone, Debug)]
enum KeyIter<'a> {
    Dense(Range<usize>),
    Sparse(SparseSlots<'a>),
}

/// Present IDs without reading values.
#[derive(Clone, Debug)]
pub struct SolutionKeys<'a, I: SolutionValueId> {
    inner: KeyIter<'a>,
    id: PhantomData<I>,
}
impl<I: SolutionValueId> Iterator for SolutionKeys<'_, I> {
    type Item = I;

    fn next(&mut self) -> Option<I> {
        self.nth(0)
    }

    fn nth(&mut self, n: usize) -> Option<I> {
        match &mut self.inner {
            KeyIter::Dense(iter) => iter.nth(n),
            KeyIter::Sparse(slots) => slots.nth(n),
        }
        .map(I::from_index)
    }

    fn count(self) -> usize {
        self.len()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = match &self.inner {
            KeyIter::Dense(iter) => iter.len(),
            KeyIter::Sparse(slots) => slots.remaining,
        };
        (len, Some(len))
    }

    fn last(mut self) -> Option<I> {
        self.next_back()
    }

    fn fold<B, F: FnMut(B, I) -> B>(self, init: B, mut fold: F) -> B {
        match self.inner {
            KeyIter::Dense(iter) => iter.fold(init, |acc, index| fold(acc, I::from_index(index))),
            KeyIter::Sparse(slots) => {
                slots.fold(init, |acc, index| fold(acc, I::from_index(index)))
            }
        }
    }
}
impl<I: SolutionValueId> DoubleEndedIterator for SolutionKeys<'_, I> {
    fn next_back(&mut self) -> Option<I> {
        self.nth_back(0)
    }

    fn nth_back(&mut self, n: usize) -> Option<I> {
        match &mut self.inner {
            KeyIter::Dense(iter) => iter.nth_back(n),
            KeyIter::Sparse(slots) => slots.nth_back(n),
        }
        .map(I::from_index)
    }
}
impl<I: SolutionValueId> ExactSizeIterator for SolutionKeys<'_, I> {}
impl<I: SolutionValueId> FusedIterator for SolutionKeys<'_, I> {}

impl<'a, I: SolutionValueId> IntoIterator for &'a SolutionValues<I> {
    type Item = (I, f64);
    type IntoIter = SolutionIter<'a, I>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<I: SolutionValueId> Extend<(I, f64)> for SolutionValues<I> {
    fn extend<T: IntoIterator<Item = (I, f64)>>(&mut self, iter: T) {
        for (id, value) in iter {
            // Ordered input needs only Vec's ordinary append path
            if id.index() == self.values.len() {
                let index = self.values.len();
                self.values.push(value);
                self.len += 1;
                if let Some(bits) = &mut self.presence {
                    bits.resize((index + 1).div_ceil(64), 0);
                    bits[index / 64] |= 1 << (index % 64);
                    if self.len == self.values.len() {
                        self.presence = None;
                    }
                }
            } else {
                self.insert(id, value);
            }
        }
    }
}

impl<I: SolutionValueId> FromIterator<(I, f64)> for SolutionValues<I> {
    fn from_iter<T: IntoIterator<Item = (I, f64)>>(iter: T) -> Self {
        let entries = iter.into_iter();
        let mut result = Self::default();
        // Bound speculative allocation from a large or incorrect hint
        result.values.reserve(entries.size_hint().0.min(1_048_576));
        result.extend(entries);
        result.compact_if_excess();
        result
    }
}

/// Equality compares slot count, presence, and the bits of available values.
impl<I: SolutionValueId> PartialEq for SolutionValues<I> {
    fn eq(&self, other: &Self) -> bool {
        self.slot_count() == other.slot_count()
            && self.len == other.len
            && self
                .iter()
                .zip(other.iter())
                .all(|((a, x), (b, y))| a.index() == b.index() && x.to_bits() == y.to_bits())
    }
}
impl<I: SolutionValueId> Eq for SolutionValues<I> {}

impl oximo_expr::EvalContext for SolutionValues<VarId> {
    #[inline]
    fn var(&self, id: VarId) -> Option<f64> {
        self.get(id)
    }

    fn param(&self, _: oximo_expr::ParamId) -> Option<f64> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_retention_preserves_slots_without_calling_the_predicate() {
        let mut values = SolutionValues::<VarId>::missing(1_000_000);
        for _ in 0..2 {
            values.retain(|_, _| panic!("missing values must not call retain's predicate"));
            assert_eq!(values.slot_count(), 1_000_000);
            assert_eq!(values.len(), 0);
        }
    }

    #[test]
    fn collection_limits_speculative_reservation_from_a_lying_hint() {
        struct WrongHint(bool);
        impl Iterator for WrongHint {
            type Item = (VarId, f64);
            fn next(&mut self) -> Option<Self::Item> {
                std::mem::take(&mut self.0).then_some((VarId(0), 7.0))
            }
            fn size_hint(&self) -> (usize, Option<usize>) {
                (usize::MAX, None)
            }
        }

        let values: SolutionValues<_> = WrongHint(true).collect();
        assert_eq!(values.as_dense(), Some([7.0].as_slice()));
        assert_eq!(values.values.capacity(), 1);
    }

    #[test]
    fn growing_insert_and_extend_preserve_holes_and_bitmap_padding() {
        for old in [0, 1, 63, 64, 65, 127, 128, 129] {
            let mut values = SolutionValues::<VarId>::from_dense(vec![2.0; old]);
            if old != 0 {
                values.set(VarId(0), None);
            }
            let last = u32::try_from(old + 70).unwrap();
            assert_eq!(values.try_insert(VarId(last), 4.0).unwrap(), None);
            assert_eq!(values.slot_count(), old + 71);

            for index in old..old + 70 {
                assert_eq!(values.get(VarId(u32::try_from(index).unwrap())), None);
            }

            values.extend([(VarId(last - 1), 3.0), (VarId(last), 5.0), (VarId(0), 1.0)]);
            assert_eq!(values.get(VarId(last)), Some(5.0));
            assert_eq!(values.get(VarId(last - 1)), Some(3.0));
            assert_eq!(values.get(VarId(0)), Some(1.0));
            assert_eq!(values.iter().count(), values.len());
            assert_eq!(values.keys().count(), values.len());
        }

        let mut ordered = SolutionValues::<VarId>::default();
        ordered.extend((0..130).map(|index| (VarId(index), f64::from(index))));
        assert!(ordered.is_complete());
        assert!(ordered.presence.is_none());
        assert_eq!(ordered.insert(VarId(64), 9.0), Some(64.0));
    }

    #[test]
    fn failed_growing_insert_leaves_contents_and_slot_count_unchanged() {
        #[derive(Clone, Copy, Debug)]
        struct HugeId(usize);
        impl sealed::Sealed for HugeId {}
        impl SolutionValueId for HugeId {
            fn index(self) -> usize {
                self.0
            }
            fn from_index(index: usize) -> Self {
                Self(index)
            }
        }

        let mut values = SolutionValues::<HugeId>::from_dense(vec![3.0]);
        assert!(values.try_insert(HugeId(usize::MAX), 1.0).is_err());
        assert_eq!(values.slot_count(), 1);
        assert_eq!(values.get(HugeId(0)), Some(3.0));
        assert!(values.is_complete());
    }

    #[test]
    fn mutable_values_and_accumulation_keep_presence_consistent() {
        let mut values = SolutionValues::<ConstraintId>::missing(3);
        assert_eq!(values.get_mut(ConstraintId(1)), None);

        values.try_accumulate(ConstraintId(1), 2.0).unwrap();
        values.try_accumulate(ConstraintId(1), -0.5).unwrap();
        *values.get_mut(ConstraintId(1)).unwrap() = 4.0;
        assert_eq!(values.get(ConstraintId(1)), Some(4.0));
        assert_eq!(values.len(), 1);
        assert!(values.contains(ConstraintId(1)));
        assert!(!values.contains(ConstraintId(3)));

        let before = values.clone();
        assert!(values.try_accumulate(ConstraintId(3), 8.0).is_err());
        assert_eq!(values, before);

        values.try_accumulate(ConstraintId(0), -0.0).unwrap();
        values.try_accumulate(ConstraintId(2), 0.0).unwrap();
        assert!(values.is_complete());
        assert_eq!(values.get(ConstraintId(0)).unwrap().to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn equality_uses_available_float_bits_and_ignores_missing_payloads() {
        let nan = f64::from_bits(0x7ff8_0000_0000_1234);
        let a = SolutionValues::<VarId>::from_dense(vec![nan, -0.0]);
        assert_eq!(a, a.clone());
        assert_ne!(a, SolutionValues::from_dense(vec![f64::NAN, -0.0]));
        assert_ne!(a, SolutionValues::from_dense(vec![nan, 0.0]));

        let mut cleared = a;
        cleared.clear();
        assert_eq!(cleared, SolutionValues::missing(2));
        assert_ne!(cleared, SolutionValues::missing(3));
    }

    #[test]
    fn sparse_retain_visits_present_values_and_keeps_counts_on_panic() {
        let mut values = SolutionValues::<VarId>::missing(1_000_000);
        values.extend([(VarId(0), 1.0), (VarId(65), 2.0), (VarId(999_999), 3.0)]);
        let mut visited = Vec::new();
        values.retain(|id, _| {
            visited.push(id);
            id != VarId(65)
        });
        assert_eq!(visited, [VarId(0), VarId(65), VarId(999_999)]);
        assert_eq!(values.len(), 2);

        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            values.retain(|id, _| {
                assert_ne!(id, VarId(999_999), "callback failure");
                false
            });
        }));
        assert!(failed.is_err());
        assert_eq!(values.len(), 1);
        assert_eq!(values.len(), values.keys().collect::<Vec<_>>().len());
        assert_eq!(values.get(VarId(999_999)), Some(3.0));
    }

    #[test]
    fn folds_and_keys_respect_mixed_iterator_consumption() {
        for sparse in [false, true] {
            let mut values = SolutionValues::<VarId>::from_dense((0..257).map(f64::from).collect());
            if sparse {
                values.retain(|id, _| id.index() % 13 == 0);
            }

            let expected: Vec<_> = (0..257_u32).filter(|&i| !sparse || i % 13 == 0).collect();
            let mut iter = values.iter();
            let mut raw = values.values();
            let mut keys = values.keys();
            assert_eq!(iter.nth(1).unwrap().0, VarId(expected[1]));
            assert_eq!(raw.nth(1), Some(f64::from(expected[1])));
            assert_eq!(keys.nth(1), Some(VarId(expected[1])));
            assert_eq!(iter.nth_back(1).unwrap().0, VarId(expected[expected.len() - 2]));
            assert_eq!(raw.nth_back(1), Some(f64::from(expected[expected.len() - 2])));
            assert_eq!(keys.nth_back(1), Some(VarId(expected[expected.len() - 2])));

            let wanted = expected[2..expected.len() - 2].iter().copied().sum::<u32>();
            assert_eq!(iter.fold(0.0, |sum, (_, v)| sum + v), f64::from(wanted));
            assert_eq!(raw.sum::<f64>(), f64::from(wanted));
            assert_eq!(keys.fold(0, |sum, id| sum + id.0), wanted);
        }
    }

    #[test]
    fn seeking_and_values_do_not_project_skipped_ids() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static PROJECTED: AtomicUsize = AtomicUsize::new(0);
        #[derive(Clone, Copy, Debug)]
        struct CountedId(usize);
        impl sealed::Sealed for CountedId {}
        impl SolutionValueId for CountedId {
            fn index(self) -> usize {
                self.0
            }
            fn from_index(index: usize) -> Self {
                PROJECTED.fetch_add(1, Ordering::Relaxed);
                Self(index)
            }
        }

        let values = SolutionValues::<CountedId>::from_dense(vec![1.0; 1_000_000]);
        let mut entries = values.iter();
        assert_eq!(entries.nth(900_000).unwrap().0.index(), 900_000);
        assert_eq!(entries.nth_back(50_000).unwrap().0.index(), 949_999);
        assert_eq!(entries.count(), 49_998);
        assert_eq!(values.values().nth(999_999), Some(1.0));
        assert_eq!(values.values().count(), 1_000_000);
        assert_eq!(PROJECTED.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn bounded_pairs_and_checked_writes_preserve_holes_duplicates_and_errors() {
        let mut values = SolutionValues::try_from_pairs(
            65,
            [(VarId(64), 1.0), (VarId(0), 0.0), (VarId(64), 2.0)],
        )
        .unwrap();
        assert_eq!(values.len(), 2);
        assert_eq!(values.get(VarId(1)), None);
        assert_eq!(values.get(VarId(64)), Some(2.0));
        assert_eq!(
            values.try_set(VarId(65), Some(9.0)),
            Err(SolutionIndexError { index: 65, slot_count: 65 })
        );
        assert_eq!(values.len(), 2);
        assert_eq!(
            SolutionValues::try_from_pairs(0, [(VarId(0), 1.0)]).unwrap_err(),
            SolutionIndexError { index: 0, slot_count: 0 }
        );

        let dense = SolutionValues::<VarId>::try_from_pairs(0, []).unwrap().into_dense().unwrap();
        assert_eq!(dense, [] as [f64; 0]);
    }

    #[test]
    fn owned_recovery_and_compaction_preserve_bits_and_identity() {
        let input = vec![1.0, -0.0, f64::NAN];
        let pointer = input.as_ptr();
        let values = SolutionValues::<VarId>::from_dense(input);
        let output = values.into_dense().unwrap();
        assert_eq!(output.as_ptr(), pointer);
        assert_eq!(output[1].to_bits(), (-0.0_f64).to_bits());

        let mut partial = SolutionValues::<VarId>::from_dense(vec![1.0; 1_000_000]);
        partial.set(VarId(0), None);
        let partial = partial.into_dense().unwrap_err();
        assert_eq!(partial.len(), 999_999);

        let mut partial = partial;
        partial.truncate(65);
        partial.compact_if_excess();
        assert_eq!(partial.values.capacity(), 65);
        assert_eq!(partial.presence.as_ref().unwrap().capacity(), 2);
        assert_eq!(partial.len(), 64);
        assert_eq!(partial.get(VarId(64)), Some(1.0));

        let mut complete = SolutionValues::<VarId>::from_dense(vec![2.0; 1_000_000]);
        complete.truncate(1);
        complete.compact_if_excess();
        assert_eq!(complete.values.capacity(), 1);
        assert_eq!(complete.into_dense().unwrap(), [2.0]);
    }

    #[test]
    fn ordered_pair_collection_needs_only_the_value_buffer() {
        let values: SolutionValues<_> = (0..1_000_000).map(|i| (VarId(i), f64::from(i))).collect();
        assert_eq!(values.values.capacity(), 1_000_000);
        assert_eq!(values.len(), 1_000_000);
        assert!(values.presence.is_none());
        assert_eq!(values.get(VarId(999_999)), Some(999_999.0));
    }

    #[test]
    fn streaming_pairs_preserve_gaps_when_growing_across_bitmap_words() {
        for count in [0_u32, 1, 63, 64, 65, 129, 257] {
            let pairs: Vec<_> =
                (0..count).filter(|i| i % 3 != 1).rev().map(|i| (VarId(i), f64::from(i))).collect();
            let values: SolutionValues<_> = pairs.iter().copied().collect();
            let mut expected = pairs;
            expected.reverse();
            assert_eq!(values.iter().collect::<Vec<_>>(), expected);
        }
    }

    #[test]
    fn dense_and_sparse_iterators_match_mixed_front_back_seeking() {
        use std::collections::VecDeque;
        for count in [0, 1, 63, 64, 65, 129, 257, 16_385] {
            for sparse in [false, true] {
                let mut values = SolutionValues::<VarId>::from_dense(vec![3.0; count]);
                if sparse {
                    values.retain(|id, _| id.index() % 131 == 0 || id.index() % 257 == 64);
                }
                let mut expected: VecDeque<_> = (&values).into_iter().collect();
                let mut entries = values.iter();
                let mut raw = values.values();
                let mut state = 17_usize;
                loop {
                    assert_eq!(entries.len(), expected.len());
                    assert_eq!(entries.size_hint(), (expected.len(), Some(expected.len())));
                    assert_eq!(entries.clone().count(), expected.len());
                    assert_eq!(raw.len(), expected.len());
                    assert_eq!(raw.clone().count(), expected.len());

                    if expected.is_empty() {
                        break;
                    }

                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    let n = state % 9;
                    let from_back = state.is_multiple_of(2);
                    for _ in 0..n {
                        if from_back {
                            expected.pop_back();
                        } else {
                            expected.pop_front();
                        }
                    }

                    let want = if from_back { expected.pop_back() } else { expected.pop_front() };
                    let got = if from_back { entries.nth_back(n) } else { entries.nth(n) };
                    let value = if from_back { raw.nth_back(n) } else { raw.nth(n) };
                    assert_eq!(got, want);
                    assert_eq!(value, want.map(|(_, v)| v));
                }
                assert_eq!(entries.nth(usize::MAX), None);
                assert_eq!(entries.nth_back(usize::MAX), None);
                assert_eq!(entries.next(), None);
                assert_eq!(entries.next_back(), None);
                assert_eq!(raw.next(), None);
                assert_eq!(raw.next_back(), None);
            }
        }
    }

    #[test]
    fn truncation_counts_partial_words_and_can_restore_dense_storage() {
        for old in [1, 63, 64, 65, 129, 257] {
            let mut values = SolutionValues::<VarId>::from_dense(vec![1.0; old]);
            values.retain(|id, _| id.index() % 7 == 0);
            for length in 0..=old {
                let mut truncated = values.clone();
                truncated.truncate(length);
                assert_eq!(truncated.len(), (0..length).filter(|i| i % 7 == 0).count());
                assert_eq!(truncated.iter().count(), truncated.len());
                assert_eq!(truncated.slot_count(), length);
            }
        }
    }

    #[test]
    fn complete_storage_adopts_allocation_and_preserves_float_bits() {
        let values = vec![0.0, -0.0, f64::from_bits(0x7ff8_0000_0000_1234), f64::INFINITY];
        let address = values.as_ptr();
        let bits: Vec<_> = values.iter().map(|v| v.to_bits()).collect();
        let values = SolutionValues::<VarId>::from_dense(values);
        assert_eq!(values.as_dense().unwrap().as_ptr(), address);
        assert!(values.presence.is_none());
        assert_eq!(values.len(), 4);
        assert_eq!(values.iter().map(|(_, v)| v.to_bits()).collect::<Vec<_>>(), bits);
        assert_eq!(values.get(VarId(4)), None);
        assert_eq!(values.get(VarId(u32::MAX)), None);
    }

    #[test]
    fn partial_values_cross_bitmap_words_and_restore_complete_storage() {
        let mut values = SolutionValues::<VarId>::from_dense(vec![2.0; 130]);
        for index in [0, 63, 64, 127, 129] {
            assert_eq!(values.set(VarId(index), None), Some(2.0));
            assert_eq!(values.set(VarId(index), None), None);
        }

        assert_eq!(values.len(), 125);
        assert_eq!(values.slot_count(), 130);
        assert!(values.as_dense().is_none());
        assert_eq!(values.get(VarId(65)), Some(2.0));
        assert_eq!(values.iter().next(), Some((VarId(1), 2.0)));
        assert_eq!(values.iter().next_back(), Some((VarId(128), 2.0)));

        let mut cloned = values.clone();
        cloned.set(VarId(64), Some(-0.0));
        assert_eq!(values.get(VarId(64)), None);
        assert_eq!(cloned.get(VarId(64)).unwrap().to_bits(), (-0.0_f64).to_bits());

        for index in [0, 63, 64, 127, 129] {
            assert_eq!(values.set(VarId(index), Some(0.0)), None);
        }

        assert_eq!(values.len(), 130);
        assert!(values.presence.is_none());
        assert_eq!(values.set(VarId(0), Some(3.0)), Some(0.0));
    }

    #[test]
    fn missing_clear_retain_and_truncate_preserve_ids_and_counts() {
        let mut values = SolutionValues::<ConstraintId>::missing(130);
        assert!(values.is_empty());

        for index in [0, 63, 64, 129] {
            values.set(ConstraintId(index), Some(f64::from(index)));
        }
        let mut visited = Vec::new();
        values.retain(|id, _| {
            visited.push(id);
            id != ConstraintId(63)
        });
        assert_eq!(
            visited,
            [ConstraintId(0), ConstraintId(63), ConstraintId(64), ConstraintId(129)]
        );
        assert_eq!(values.len(), 3);

        values.truncate(65);
        assert_eq!(values.len(), 2);
        assert_eq!(values.slot_count(), 65);
        assert_eq!(values.get(ConstraintId(64)), Some(64.0));

        values.clear();
        assert!(values.is_empty());
        assert_eq!(values.slot_count(), 65);
        assert_eq!(values.values().count(), 0);

        values.set(ConstraintId(64), Some(1.0));
        values.truncate(64);
        assert!(values.is_empty());

        values.truncate(0);
        assert_eq!(values.as_dense(), Some([].as_slice()));
    }

    #[test]
    fn collection_is_ordered_with_last_duplicate_winning() {
        let values: SolutionValues<SocConstraintId> =
            [(SocConstraintId(64), 1.0), (SocConstraintId(0), 0.0), (SocConstraintId(64), 2.0)]
                .into_iter()
                .collect();
        assert_eq!(values.slot_count(), 65);
        assert_eq!(values.len(), 2);
        assert_eq!(
            values.iter().collect::<Vec<_>>(),
            [(SocConstraintId(0), 0.0), (SocConstraintId(64), 2.0)]
        );
        assert_eq!(values.get(SocConstraintId(63)), None);
    }

    #[test]
    fn empty_values_are_complete_and_do_not_allocate_a_bitmap() {
        let mut values = SolutionValues::<VarId>::missing(0);
        values.clear();
        values.truncate(100);
        assert!(values.is_empty());
        assert_eq!(values.slot_count(), 0);
        assert_eq!(values.as_dense(), Some([].as_slice()));
        assert_eq!(values.iter().count(), 0);
    }

    #[test]
    #[should_panic(expected = "VarId(2) outside slot count 2")]
    fn writes_cannot_grow_storage() {
        SolutionValues::<VarId>::missing(2).set(VarId(2), Some(1.0));
    }
}
