use std::ops::Mul;

use num_traits::PrimInt;
use rayon::prelude::*;

// Note: I used `Box<[IndexKey]>` over:
//   - `Vec<IndexKey>`: saves one word
//     (no capacity field) since tuples are never mutated after construction.
//
//   - `SmallVec<[IndexKey; N]>` is rejected by the compiler: recursive size
//      cycle

/// Heap-allocated tuple key payload. Immutable after construction.
pub type IndexTuple = Box<[IndexKey]>;

/// A finite, ordered index set.
///
/// Supports integer ranges, string lists, and arbitrary tuple lists (built via
/// [`Set::product`] / the `&a * &b` operator, or constructed directly).
#[derive(Clone, Debug)]
pub enum Set {
    Range(Vec<i64>),
    Strings(Vec<String>),
    Tuples(Vec<IndexTuple>),
}

impl Set {
    /// Build an integer index set from a range over any primitive integer
    /// type. Accepts `Range<i64>`, `Range<i32>`, `Range<usize>`, etc.
    ///
    /// The untyped literal form `Set::range(0..5)` defaults to `i32` and
    /// works without annotation.
    ///
    /// # Panics
    /// Panics if either range bound does not fit in `i64`.
    pub fn range<T: PrimInt>(r: std::ops::Range<T>) -> Self {
        let start = r.start.to_i64().expect("range start out of i64 range");
        let end = r.end.to_i64().expect("range end out of i64 range");
        Self::Range((start..end).collect())
    }

    /// Build an integer index set from an iterator of any primitive integer
    /// type. Useful when keys are sparse or computed (e.g. only even indices).
    ///
    /// # Panics
    /// Panics if any element does not fit in `i64`.
    pub fn from_ints<T, I>(iter: I) -> Self
    where
        T: PrimInt,
        I: IntoIterator<Item = T>,
    {
        Self::Range(
            iter.into_iter().map(|v| v.to_i64().expect("element out of i64 range")).collect(),
        )
    }

    pub fn strings<I, S>(iter: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::Strings(iter.into_iter().map(Into::into).collect())
    }

    /// Build a tuple set directly from an iterator of keys.
    pub fn tuples<I, T>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<IndexTuple>,
    {
        Self::Tuples(iter.into_iter().map(Into::into).collect())
    }

    /// Cartesian product of two sets. Inner tuple keys are flattened so a
    /// product `(a * b) * c` yields 3-element tuples, not nested 2-tuples.
    pub fn product(a: &Set, b: &Set) -> Self {
        let mut out = Vec::with_capacity(a.len() * b.len());
        for ka in a {
            for kb in b {
                let mut parts: Vec<IndexKey> = Vec::new();
                push_flat(&mut parts, ka.clone());
                push_flat(&mut parts, kb);
                out.push(parts.into_boxed_slice());
            }
        }
        Self::Tuples(out)
    }

    /// Filter keys with a predicate. Preserves the original variant where
    /// possible, filtered `Range`/`Strings` stay in their native variant.
    pub fn filter<F>(&self, mut f: F) -> Self
    where
        F: FnMut(&IndexKey) -> bool,
    {
        match self {
            Self::Range(v) => {
                Self::Range(v.iter().copied().filter(|i| f(&IndexKey::Int(*i))).collect())
            }
            Self::Strings(v) => Self::Strings(
                v.iter().filter(|s| f(&IndexKey::Str((*s).clone()))).cloned().collect(),
            ),
            Self::Tuples(v) => Self::Tuples(
                v.iter().filter(|t| f(&IndexKey::Tuple((*t).clone()))).cloned().collect(),
            ),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Range(v) => v.len(),
            Self::Strings(v) => v.len(),
            Self::Tuples(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

fn push_flat(dst: &mut Vec<IndexKey>, k: IndexKey) {
    match k {
        IndexKey::Tuple(inner) => dst.extend(inner.into_vec()),
        other => dst.push(other),
    }
}

fn make_tuple<I: IntoIterator<Item = IndexKey>>(items: I) -> IndexTuple {
    let mut v: Vec<IndexKey> = Vec::new();
    for k in items {
        push_flat(&mut v, k);
    }
    v.into_boxed_slice()
}

impl Mul<&Set> for &Set {
    type Output = Set;
    fn mul(self, rhs: &Set) -> Set {
        Set::product(self, rhs)
    }
}

/// A serializable index key from a [`Set`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum IndexKey {
    Int(i64),
    Str(String),
    Tuple(IndexTuple),
}

impl IndexKey {
    /// Build a tuple key from any iterable of convertible items. Nested tuple
    /// keys are flattened.
    pub fn tuple<I, T>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<IndexKey>,
    {
        Self::Tuple(make_tuple(iter.into_iter().map(Into::into)))
    }

    pub fn as_i64(&self) -> Option<i64> {
        if let Self::Int(v) = self { Some(*v) } else { None }
    }

    pub fn as_str(&self) -> Option<&str> {
        if let Self::Str(s) = self { Some(s.as_str()) } else { None }
    }

    pub fn as_tuple(&self) -> Option<&[IndexKey]> {
        if let Self::Tuple(t) = self { Some(&t[..]) } else { None }
    }
}

impl From<i64> for IndexKey {
    fn from(v: i64) -> Self {
        Self::Int(v)
    }
}

impl From<i32> for IndexKey {
    fn from(v: i32) -> Self {
        Self::Int(i64::from(v))
    }
}

impl From<usize> for IndexKey {
    fn from(v: usize) -> Self {
        Self::Int(i64::try_from(v).expect("usize -> i64 overflow"))
    }
}

impl From<&str> for IndexKey {
    fn from(s: &str) -> Self {
        Self::Str(s.to_owned())
    }
}

impl From<String> for IndexKey {
    fn from(s: String) -> Self {
        Self::Str(s)
    }
}

impl<A, B> From<(A, B)> for IndexKey
where
    A: Into<IndexKey>,
    B: Into<IndexKey>,
{
    fn from(t: (A, B)) -> Self {
        Self::Tuple(make_tuple([t.0.into(), t.1.into()]))
    }
}

impl<A, B, C> From<(A, B, C)> for IndexKey
where
    A: Into<IndexKey>,
    B: Into<IndexKey>,
    C: Into<IndexKey>,
{
    fn from(t: (A, B, C)) -> Self {
        Self::Tuple(make_tuple([t.0.into(), t.1.into(), t.2.into()]))
    }
}

impl<A, B, C, D> From<(A, B, C, D)> for IndexKey
where
    A: Into<IndexKey>,
    B: Into<IndexKey>,
    C: Into<IndexKey>,
    D: Into<IndexKey>,
{
    fn from(t: (A, B, C, D)) -> Self {
        Self::Tuple(make_tuple([t.0.into(), t.1.into(), t.2.into(), t.3.into()]))
    }
}

/// Typed projection of an [`IndexKey`]. Implementations panic when the
/// key's shape does not match the target type, the same contract as
/// [`crate::indexed::IndexedVar`] indexing on a missing key.
///
/// Used by [`crate::model::Model::add_constraints_over`] (and similar rule
/// helpers) to give the closure typed indices directly:
///
/// ```ignore
/// m.add_constraints_over("supply", &(&plants * &markets), |(p, m): (String, String)| {
///     // p, m are native String, no manual unpack
///     ...
/// });
/// ```
pub trait FromIndexKey: Sized {
    fn from_index_key(k: &IndexKey) -> Self;
}

impl FromIndexKey for IndexKey {
    fn from_index_key(k: &IndexKey) -> Self {
        k.clone()
    }
}

impl FromIndexKey for i64 {
    fn from_index_key(k: &IndexKey) -> Self {
        k.as_i64().unwrap_or_else(|| panic!("expected Int key, got {k:?}"))
    }
}

impl FromIndexKey for i32 {
    fn from_index_key(k: &IndexKey) -> Self {
        let v = i64::from_index_key(k);
        i32::try_from(v).unwrap_or_else(|_| panic!("key {v} out of i32 range"))
    }
}

impl FromIndexKey for usize {
    fn from_index_key(k: &IndexKey) -> Self {
        let v = i64::from_index_key(k);
        usize::try_from(v).unwrap_or_else(|_| panic!("negative key {v} cannot be usize"))
    }
}

impl FromIndexKey for String {
    fn from_index_key(k: &IndexKey) -> Self {
        k.as_str().unwrap_or_else(|| panic!("expected Str key, got {k:?}")).to_owned()
    }
}

fn tuple_parts<'a>(k: &'a IndexKey, expected: usize) -> &'a [IndexKey] {
    let p = k.as_tuple().unwrap_or_else(|| panic!("expected Tuple key, got {k:?}"));
    assert_eq!(p.len(), expected, "expected tuple of arity {expected}, got arity {}", p.len());
    p
}

impl<A, B> FromIndexKey for (A, B)
where
    A: FromIndexKey,
    B: FromIndexKey,
{
    fn from_index_key(k: &IndexKey) -> Self {
        let p = tuple_parts(k, 2);
        (A::from_index_key(&p[0]), B::from_index_key(&p[1]))
    }
}

impl<A, B, C> FromIndexKey for (A, B, C)
where
    A: FromIndexKey,
    B: FromIndexKey,
    C: FromIndexKey,
{
    fn from_index_key(k: &IndexKey) -> Self {
        let p = tuple_parts(k, 3);
        (A::from_index_key(&p[0]), B::from_index_key(&p[1]), C::from_index_key(&p[2]))
    }
}

impl<A, B, C, D> FromIndexKey for (A, B, C, D)
where
    A: FromIndexKey,
    B: FromIndexKey,
    C: FromIndexKey,
    D: FromIndexKey,
{
    fn from_index_key(k: &IndexKey) -> Self {
        let p = tuple_parts(k, 4);
        (
            A::from_index_key(&p[0]),
            B::from_index_key(&p[1]),
            C::from_index_key(&p[2]),
            D::from_index_key(&p[3]),
        )
    }
}

impl<'a> IntoIterator for &'a Set {
    type Item = IndexKey;
    type IntoIter = SetIter<'a>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl Set {
    pub fn iter(&self) -> SetIter<'_> {
        SetIter { set: self, pos: 0 }
    }

    pub fn par_iter(&self) -> impl ParallelIterator<Item = IndexKey> + '_ {
        match self {
            Self::Range(v) => v.par_iter().map(|i| IndexKey::Int(*i)).collect::<Vec<_>>(),
            Self::Strings(v) => v.par_iter().map(|s| IndexKey::Str(s.clone())).collect::<Vec<_>>(),
            Self::Tuples(v) => v.par_iter().map(|t| IndexKey::Tuple(t.clone())).collect::<Vec<_>>(),
        }
        .into_par_iter()
    }
}

#[derive(Debug)]
pub struct SetIter<'a> {
    set: &'a Set,
    pos: usize,
}

impl<'a> Iterator for SetIter<'a> {
    type Item = IndexKey;
    fn next(&mut self) -> Option<Self::Item> {
        let out = match self.set {
            Set::Range(v) => v.get(self.pos).copied().map(IndexKey::Int),
            Set::Strings(v) => v.get(self.pos).cloned().map(IndexKey::Str),
            Set::Tuples(v) => v.get(self.pos).cloned().map(IndexKey::Tuple),
        };
        if out.is_some() {
            self.pos += 1;
        }
        out
    }
}
