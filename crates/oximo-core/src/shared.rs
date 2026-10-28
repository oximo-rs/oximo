//! Shared component storage for semantic and native views of a function.

use std::fmt;
use std::ops::{Deref, DerefMut, Range};
use std::sync::Arc;

use std::cell::{Cell, Ref, RefCell, RefMut};

enum CellStorage<T> {
    Owned(T),
    Shared(Arc<T>),
}

/// Ordinary construction has no reference counting or allocation.
///
/// Forking promotes the registry to shared storage where its first write detaches only that registry.
pub(crate) struct SharedCell<T>(RefCell<CellStorage<T>>, Cell<u64>);

impl<T> SharedCell<T> {
    #[inline]
    pub(crate) fn new(value: T) -> Self {
        Self(RefCell::new(CellStorage::Owned(value)), Cell::new(0))
    }

    pub(crate) fn revision(&self) -> u64 {
        self.1.get()
    }

    /// Preflight coupled writes without detaching storage or advancing its revision.
    pub(crate) fn assert_mutable(&self) {
        drop(self.0.borrow_mut());
    }

    #[inline]
    pub(crate) fn borrow(&self) -> Ref<'_, T> {
        Ref::map(self.0.borrow(), |storage| match storage {
            CellStorage::Owned(value) => value,
            CellStorage::Shared(value) => value,
        })
    }
}

impl<T: Clone> SharedCell<T> {
    #[inline]
    pub(crate) fn borrow_mut(&self) -> RefMut<'_, T> {
        self.borrow_mut_with(Clone::clone)
    }

    #[inline]
    fn borrow_mut_with(&self, clone: impl FnOnce(&T) -> T) -> RefMut<'_, T> {
        let storage = self.0.borrow_mut();
        self.1.set(self.1.get().checked_add(1).expect("registry revision overflow"));
        RefMut::map(storage, |storage| match storage {
            CellStorage::Owned(value) => value,
            shared @ CellStorage::Shared(_) => detach_registry(shared, clone),
        })
    }
}

#[cold]
#[inline(never)]
fn detach_registry<T>(storage: &mut CellStorage<T>, clone: impl FnOnce(&T) -> T) -> &mut T {
    if let CellStorage::Shared(value) = storage
        && Arc::get_mut(value).is_none()
    {
        let detached = clone(value);
        *storage = CellStorage::Owned(detached);
    }

    match storage {
        CellStorage::Owned(value) => value,
        CellStorage::Shared(value) => Arc::get_mut(value).expect("unique registry"),
    }
}

impl<T: Clone> SharedCell<Vec<T>> {
    #[inline]
    pub(crate) fn borrow_mut_for_push(&self) -> RefMut<'_, Vec<T>> {
        self.borrow_mut_with(|source| {
            let capacity = source.len().checked_add(1).expect("registry count overflow");
            let mut detached = Vec::with_capacity(capacity);
            detached.extend_from_slice(source);
            detached
        })
    }

    /// Detach and reserve in one allocation when a fork will append known rows.
    pub(crate) fn borrow_mut_with_capacity(&self, additional: usize) -> RefMut<'_, Vec<T>> {
        let mut values = self.borrow_mut_with(|source| {
            let capacity = source.len().checked_add(additional).expect("registry count overflow");
            let mut detached = Vec::with_capacity(capacity);
            detached.extend_from_slice(source);
            detached
        });
        values.reserve(additional);
        values
    }
}

impl<T: Clone + Default> Clone for SharedCell<T> {
    fn clone(&self) -> Self {
        let Ok(mut storage) = self.0.try_borrow_mut() else {
            return Self::new(self.borrow().clone());
        };

        let shared = match &mut *storage {
            CellStorage::Owned(value) => {
                let shared = Arc::new(std::mem::take(value));
                *storage = CellStorage::Shared(Arc::clone(&shared));
                shared
            }
            CellStorage::Shared(value) => Arc::clone(value),
        };
        Self(RefCell::new(CellStorage::Shared(shared)), Cell::new(self.1.get()))
    }
}

pub(crate) struct LazySharedCell<T> {
    value: RefCell<Option<Arc<T>>>,
    revision: Cell<u64>,
}

impl<T> Default for LazySharedCell<T> {
    fn default() -> Self {
        Self { value: RefCell::new(None), revision: Cell::new(0) }
    }
}

impl<T> LazySharedCell<T> {
    pub(crate) fn borrow(&self) -> Ref<'_, Option<Arc<T>>> {
        self.value.borrow()
    }

    pub(crate) fn revision(&self) -> u64 {
        self.revision.get()
    }

    /// Initialize a read cache without recording a logical model mutation.
    pub(crate) fn borrow_or_init(&self, init: impl FnOnce() -> T) -> Ref<'_, T> {
        if self.value.borrow().is_none() {
            *self.value.borrow_mut() = Some(Arc::new(init()));
        }
        Ref::map(self.value.borrow(), |value| value.as_deref().expect("initialized registry"))
    }

    fn mark_changed(&self) {
        self.revision.set(self.revision.get().checked_add(1).expect("registry revision overflow"));
    }

    pub(crate) fn clear(&self) {
        let mut value = self.value.borrow_mut();
        self.mark_changed();
        *value = None;
    }
}

impl<T: Clone + Default> LazySharedCell<T> {
    pub(crate) fn borrow_mut(&self) -> RefMut<'_, T> {
        let value = self.value.borrow_mut();
        self.mark_changed();
        RefMut::map(value, |value| Arc::make_mut(value.get_or_insert_with(Arc::default)))
    }
}

impl<T> Clone for LazySharedCell<T> {
    fn clone(&self) -> Self {
        Self {
            value: RefCell::new(self.value.borrow().clone()),
            revision: Cell::new(self.revision.get()),
        }
    }
}

/// A slice that shares its backing buffer and detaches on mutation.
pub struct SharedSlice<T> {
    storage: SliceStorage<T>,
    range: Range<usize>,
}

enum SliceStorage<T> {
    Inline(smallvec::SmallVec<[T; 4]>),
    Vector(Arc<Vec<T>>),
    Slice(Arc<[T]>),
}

impl<T> SharedSlice<T> {
    /// Restrict an owned slice without cloning its backing storage.
    ///
    /// # Panics
    /// Panics if the range lies outside this slice.
    pub fn into_slice(mut self, range: Range<usize>) -> Self {
        assert!(range.start <= range.end && range.end <= self.len(), "invalid shared slice range");

        self.range = self.range.start + range.start..self.range.start + range.end;
        self
    }
}

impl<T: Clone> SharedSlice<T> {
    pub(crate) fn with_prefix(&self, count: usize) -> Self {
        assert!(count <= self.range.start, "missing shared prefix");
        let mut slice = self.clone();
        slice.range.start -= count;
        slice
    }

    /// Share a subrange of this slice.
    ///
    /// # Panics
    /// Panics if the range lies outside this slice.
    pub fn slice(&self, range: Range<usize>) -> Self {
        assert!(range.start <= range.end && range.end <= self.len(), "invalid shared slice range");

        self.clone().into_slice(range)
    }
}

impl<T: Clone> Clone for SharedSlice<T> {
    fn clone(&self) -> Self {
        let storage = match &self.storage {
            SliceStorage::Inline(value) => SliceStorage::Inline(value.clone()),
            SliceStorage::Vector(value) => SliceStorage::Vector(Arc::clone(value)),
            SliceStorage::Slice(value) => SliceStorage::Slice(Arc::clone(value)),
        };
        Self { storage, range: self.range.clone() }
    }
}

impl<T> From<Vec<T>> for SharedSlice<T> {
    fn from(value: Vec<T>) -> Self {
        let len = value.len();
        let storage = if len <= 4 {
            SliceStorage::Inline(smallvec::SmallVec::from_vec(value))
        } else {
            SliceStorage::Vector(Arc::new(value))
        };
        Self { storage, range: 0..len }
    }
}

impl<T> FromIterator<T> for SharedSlice<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();

        if iter.size_hint().1.is_some_and(|upper| upper <= 4) {
            let value: smallvec::SmallVec<[T; 4]> = iter.collect();
            let len = value.len();
            Self { storage: SliceStorage::Inline(value), range: 0..len }
        } else {
            let value: Arc<[_]> = iter.collect();
            let len = value.len();
            Self { storage: SliceStorage::Slice(value), range: 0..len }
        }
    }
}

impl<T> Deref for SharedSlice<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        let values = match &self.storage {
            SliceStorage::Inline(value) => value.as_slice(),
            SliceStorage::Vector(value) => value.as_slice(),
            SliceStorage::Slice(value) => value,
        };
        &values[self.range.clone()]
    }
}

impl<T: Clone> DerefMut for SharedSlice<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        let values = match &mut self.storage {
            SliceStorage::Inline(value) => value.as_mut_slice(),
            SliceStorage::Vector(value) => Arc::make_mut(value).as_mut_slice(),
            SliceStorage::Slice(value) => Arc::make_mut(value),
        };
        &mut values[self.range.clone()]
    }
}

impl<'a, T> IntoIterator for &'a SharedSlice<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T: Clone> IntoIterator for &'a mut SharedSlice<T> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T: fmt::Debug> fmt::Debug for SharedSlice<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.deref().fmt(f)
    }
}

impl<T: PartialEq> PartialEq for SharedSlice<T> {
    fn eq(&self, other: &Self) -> bool {
        self[..] == other[..]
    }
}

impl<T: PartialEq, const N: usize> PartialEq<[T; N]> for SharedSlice<T> {
    fn eq(&self, other: &[T; N]) -> bool {
        &self[..] == other
    }
}

impl<T: Eq> Eq for SharedSlice<T> {}

#[cfg(test)]
mod tests {
    use super::SharedSlice;

    #[test]
    fn collected_and_vector_buffers_preserve_copy_on_write_subranges() {
        for source in [SharedSlice::from(vec![0, 1, 2, 3, 4, 5]), (0..6).collect()] {
            let mut middle = source.slice(1..4);
            middle[0] = 9;
            assert_eq!(source, [0, 1, 2, 3, 4, 5]);
            assert_eq!(middle, [9, 2, 3]);
        }
    }
}
