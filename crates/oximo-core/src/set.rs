use std::marker::PhantomData;
use std::ops::Mul;

use num_traits::PrimInt;
use rayon::prelude::*;
use smol_str::SmolStr;

// Note: I used `Box<[IndexKey]>` over:
//   - `Vec<IndexKey>`: saves one word
//     (no capacity field) since tuples are never mutated after construction.
//
//   - `SmallVec<[IndexKey; N]>` is rejected by the compiler: recursive size
//      cycle

/// Heap-allocated tuple key payload. Immutable after construction.
pub type IndexTuple = Box<[IndexKey]>;

/// Runtime representation of a [`Set`]. The key type is tracked only at the type
/// level (via [`Set`]'s phantom parameter), so the stored representation is
/// identical for every `K` and carries no per-key type information.
#[derive(Clone, Debug)]
enum SetRepr {
    Range(Vec<i64>),
    Strings(Vec<SmolStr>),
    Tuples(Vec<IndexTuple>),
}

impl SetRepr {
    fn len(&self) -> usize {
        match self {
            Self::Range(v) => v.len(),
            Self::Strings(v) => v.len(),
            Self::Tuples(v) => v.len(),
        }
    }
}

/// A single contiguous integer axis of a dense index grid.
/// Carried by [`Set`] (see [`Set::axes`]) so an [`crate::IndexedVar`]
/// built over a range can store its scalars densely and map
/// a key to a flat offset without hashing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Axis {
    pub start: i64,
    pub len: usize,
}

/// A finite, ordered index set, parameterized by the type `K` its keys decode to.
///
/// `K` is a phantom marker: the runtime representation is the same erased
/// payload regardless of `K`, so carrying the key type costs nothing at runtime.
/// It exists so the `variable!`/`constraint!`/`sum!` macros can infer the
/// closure parameter type from the set instead of requiring an annotation.
///
/// Supports integer ranges (`K = usize`), string lists (`K = String`), and
/// arbitrary tuple lists (built via [`Set::product`] / the `&a * &b` operator.
///
/// `axes` is `Some` exactly when the set is a dense integer grid
/// and records the per-axis extents.
/// It is `None` for string sets, sparse/`from_ints` sets, and any
/// `filter`ed set.
pub struct Set<K = IndexKey> {
    repr: SetRepr,
    axes: Option<Box<[Axis]>>,
    _k: PhantomData<fn() -> K>,
}

// Manual `Clone`/`Debug` so they hold for every `K`
impl<K> Clone for Set<K> {
    fn clone(&self) -> Self {
        Self { repr: self.repr.clone(), axes: self.axes.clone(), _k: PhantomData }
    }
}

impl<K> std::fmt::Debug for Set<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.repr, f)
    }
}

impl<K> Set<K> {
    fn from_repr(repr: SetRepr) -> Self {
        Self { repr, axes: None, _k: PhantomData }
    }

    fn from_repr_with_axes(repr: SetRepr, axes: Box<[Axis]>) -> Self {
        Self { repr, axes: Some(axes), _k: PhantomData }
    }

    /// Per-axis extents when this set is a dense integer grid (a range or a
    /// product of ranges), else `None`. Read by the `IndexedVar` builder to pick
    /// dense vs sparse storage.
    pub(crate) fn axes(&self) -> Option<&[Axis]> {
        self.axes.as_deref()
    }

    /// Build a tuple set directly from an iterator of keys.
    pub fn tuples<I, T>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<IndexTuple>,
    {
        Self::from_repr(SetRepr::Tuples(iter.into_iter().map(Into::into).collect()))
    }

    /// Filter keys with a predicate. Preserves the original variant where
    /// possible.
    #[must_use]
    pub fn filter<F>(&self, mut f: F) -> Self
    where
        F: FnMut(&IndexKey) -> bool,
    {
        let repr = match &self.repr {
            SetRepr::Range(v) => {
                SetRepr::Range(v.iter().copied().filter(|i| f(&IndexKey::Int(*i))).collect())
            }
            SetRepr::Strings(v) => SetRepr::Strings(
                v.iter()
                    .filter_map(|s| {
                        let key = IndexKey::Str(s.clone());
                        f(&key).then(|| s.clone())
                    })
                    .collect(),
            ),
            SetRepr::Tuples(v) => SetRepr::Tuples(
                v.iter()
                    .filter_map(|t| {
                        let key = IndexKey::Tuple(t.clone());
                        f(&key).then(|| match key {
                            IndexKey::Tuple(owned) => owned,
                            _ => unreachable!(),
                        })
                    })
                    .collect(),
            ),
        };
        Self::from_repr(repr)
    }

    /// Filter keys with a predicate over the typed, by-value decoded key.
    ///
    /// Unlike [`Self::filter`] (which hands the closure a raw [`IndexKey`]), the
    /// key is decoded to `K` first, so a product set yields native tuples and no
    /// manual `as_tuple().unwrap()` unpacking is needed. The receiver's `K` pins
    /// the closure parameter, so it usually needs no annotation.
    ///
    /// ```
    /// use oximo_core::Set;
    /// let plants = Set::strings(["seattle", "san-diego"]);
    /// // No-self-loop arcs; keys decoded to `(String, String)`.
    /// let arcs = (&plants * &plants).filter_typed(|(p, q)| p != q);
    /// assert_eq!(arcs.len(), 2);
    /// ```
    #[must_use]
    pub fn filter_typed<F>(&self, mut pred: F) -> Self
    where
        K: FromIndexKey,
        F: FnMut(K) -> bool,
    {
        self.filter(|k| pred(K::from_index_key(k)))
    }

    pub fn len(&self) -> usize {
        self.repr.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether the backing representation is the integer-range variant.
    pub fn is_range(&self) -> bool {
        matches!(self.repr, SetRepr::Range(_))
    }

    /// Whether the backing representation is the string-list variant.
    pub fn is_strings(&self) -> bool {
        matches!(self.repr, SetRepr::Strings(_))
    }

    /// Whether the backing representation is the tuple-list variant.
    pub fn is_tuples(&self) -> bool {
        matches!(self.repr, SetRepr::Tuples(_))
    }

    /// Cartesian product of two sets. Inner tuple keys are flattened so a
    /// product `(a * b) * c` yields 3-element tuples, not nested 2-tuples.
    ///
    /// # Panics
    /// Panics if `a.len() * b.len()` overflows `usize`.
    #[must_use]
    pub fn product<B>(a: &Set<K>, b: &Set<B>) -> Set<<K as KeyCat<B>>::Out>
    where
        K: KeyCat<B>,
    {
        let a_len = a.len();
        let b_len = b.len();
        let total = a_len.checked_mul(b_len).expect("Set::product size overflow");

        let axes = match (a.axes(), b.axes()) {
            (Some(aa), Some(bb)) => {
                let mut v = Vec::with_capacity(aa.len() + bb.len());
                v.extend_from_slice(aa);
                v.extend_from_slice(bb);
                Some(v.into_boxed_slice())
            }
            _ => None,
        };

        // Below this size, rayon dispatch overhead may dominate, so we stay serial.
        // TODO: benchmark and tune this threshold.
        const PAR_THRESHOLD: usize = 4096;
        let out: Vec<IndexTuple> = if total < PAR_THRESHOLD {
            let mut out = Vec::with_capacity(total);
            for ka in a {
                for kb in b {
                    let mut parts: Vec<IndexKey> = Vec::new();
                    push_flat(&mut parts, ka.clone());
                    push_flat(&mut parts, kb);
                    out.push(parts.into_boxed_slice());
                }
            }
            out
        } else {
            let a_keys: Vec<IndexKey> = a.iter().collect();
            let b_keys: Vec<IndexKey> = b.iter().collect();
            (0..total)
                .into_par_iter()
                .map(|i| {
                    let mut parts: Vec<IndexKey> = Vec::new();
                    push_flat(&mut parts, a_keys[i / b_len].clone());
                    push_flat(&mut parts, b_keys[i % b_len].clone());
                    parts.into_boxed_slice()
                })
                .collect()
        };

        match axes {
            Some(axes) => Set::from_repr_with_axes(SetRepr::Tuples(out), axes),
            None => Set::from_repr(SetRepr::Tuples(out)),
        }
    }
}

impl Set<usize> {
    /// Build an integer index set from a range over any primitive integer
    /// type. Accepts `Range<i64>`, `Range<i32>`, `Range<usize>`, etc.
    ///
    /// The keys decode to `usize`.
    /// Negative elements are accepted into the payload but panic when
    /// decoded to `usize`.
    ///
    /// # Panics
    /// Panics if either range bound does not fit in `i64`.
    #[must_use]
    pub fn range<T: PrimInt>(r: std::ops::Range<T>) -> Self {
        let start = r.start.to_i64().expect("range start out of i64 range");
        let end = r.end.to_i64().expect("range end out of i64 range");
        Self::dense_i64(start, end)
    }

    /// Build a dense contiguous integer set from an `i64` half-open range,
    /// recording the single axis so the resulting [`IndexedVar`] stores densely.
    /// Shared by [`Self::range`] and the `RangeInclusive` `IntoSet` path.
    pub(crate) fn dense_i64(start: i64, end: i64) -> Self {
        let vals: Vec<i64> = (start..end).collect();
        let len = vals.len();
        Self::from_repr_with_axes(SetRepr::Range(vals), Box::from([Axis { start, len }]))
    }

    /// Build an integer index set from an iterator of any primitive integer
    /// type. Useful when keys are sparse or computed.
    ///
    /// # Panics
    /// Panics if any element does not fit in `i64`.
    pub fn from_ints<T, I>(iter: I) -> Self
    where
        T: PrimInt,
        I: IntoIterator<Item = T>,
    {
        Self::from_repr(SetRepr::Range(
            iter.into_iter().map(|v| v.to_i64().expect("element out of i64 range")).collect(),
        ))
    }
}

impl Set<String> {
    pub fn strings<I, S>(iter: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<SmolStr>,
    {
        Self::from_repr(SetRepr::Strings(iter.into_iter().map(Into::into).collect()))
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

impl<A, B> Mul<&Set<B>> for &Set<A>
where
    A: KeyCat<B>,
{
    type Output = Set<<A as KeyCat<B>>::Out>;
    fn mul(self, rhs: &Set<B>) -> Self::Output {
        Set::product(self, rhs)
    }
}

/// Type-level concatenation of index key types, mirroring the runtime tuple
/// flattening in [`Set::product`]. The arity ceiling is 4, matching the
/// [`FromIndexKey`]/`From<(...)>` tuple implementations.
#[diagnostic::on_unimplemented(
    message = "cannot form a Cartesian product index key from `{Self}` and `{Rhs}`",
    label = "no product key for `{Self}` * `{Rhs}`",
    note = "`&a * &b` composes scalar keys (`usize`/`i64`/`i32`/`String`) into flat tuples up to arity 4. A 5th axis or a non-scalar operand is unsupported"
)]
pub trait KeyCat<Rhs> {
    type Out;
}

/// Marker for non-tuple ("scalar") index key types. Lets [`KeyCat`] distinguish
/// the scalar base case from the tuple-extension cases without overlap.
pub trait ScalarKey {}
impl ScalarKey for usize {}
impl ScalarKey for i32 {}
impl ScalarKey for i64 {}
impl ScalarKey for String {}
impl ScalarKey for IndexKey {}

impl<A: ScalarKey, B: ScalarKey> KeyCat<B> for A {
    type Out = (A, B);
}

impl<A, B, C: ScalarKey> KeyCat<C> for (A, B) {
    type Out = (A, B, C);
}

impl<A, B, C, D: ScalarKey> KeyCat<D> for (A, B, C) {
    type Out = (A, B, C, D);
}

// Right-associated / tuple-on-the-right products (`a * (b * c)`,
// `(a * b) * (c * d)`). The macros only ever left-fold with a scalar right
// operand, but `Set::product` flattens both sides, so we keep manual
// products associative up to the arity-4 ceiling.
impl<A: ScalarKey, B, C> KeyCat<(B, C)> for A {
    type Out = (A, B, C);
}

impl<A: ScalarKey, B, C, D> KeyCat<(B, C, D)> for A {
    type Out = (A, B, C, D);
}

impl<A, B, C, D> KeyCat<(C, D)> for (A, B) {
    type Out = (A, B, C, D);
}

/// A serializable index key from a [`Set`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum IndexKey {
    Int(i64),
    Str(SmolStr),
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
        Self::Str(SmolStr::new(s))
    }
}

impl From<String> for IndexKey {
    fn from(s: String) -> Self {
        Self::Str(SmolStr::from(s))
    }
}

impl From<&String> for IndexKey {
    fn from(s: &String) -> Self {
        Self::Str(SmolStr::new(s.as_str()))
    }
}

// Reference conversions.
impl From<&usize> for IndexKey {
    fn from(v: &usize) -> Self {
        Self::from(*v)
    }
}

impl From<&i64> for IndexKey {
    fn from(v: &i64) -> Self {
        Self::Int(*v)
    }
}

impl From<&i32> for IndexKey {
    fn from(v: &i32) -> Self {
        Self::Int(i64::from(*v))
    }
}

impl From<&&str> for IndexKey {
    fn from(s: &&str) -> Self {
        Self::Str(SmolStr::new(*s))
    }
}

impl From<&&String> for IndexKey {
    fn from(s: &&String) -> Self {
        Self::Str(SmolStr::new(s.as_str()))
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
/// Used by the indexed-family `constraint!` macro (and similar rule helpers) to
/// give the closure typed indices directly:
///
/// ```ignore
/// constraint!(m, supply[(p, m) in &plants * &markets], {
///     // p, m are native String, no manual unpack
///     ...
/// });
/// ```
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a valid index key type",
    label = "cannot be decoded from an index key",
    note = "index keys decode to `usize`, `i64`, `i32`, `String`, `IndexKey`, or a tuple of those up to arity 4",
    note = "annotate the binding to one of these (e.g. `for k: usize in set`) or match the `Set`'s key type"
)]
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
        usize::try_from(v).unwrap_or_else(|_| panic!("key {v} out of usize range"))
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

impl<'a, K> IntoIterator for &'a Set<K> {
    type Item = IndexKey;
    type IntoIter = SetIter<'a>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<K> Set<K> {
    pub fn iter(&self) -> SetIter<'_> {
        SetIter { repr: &self.repr, pos: 0 }
    }

    pub fn par_iter(&self) -> impl ParallelIterator<Item = IndexKey> + '_ {
        match &self.repr {
            SetRepr::Range(v) => v.par_iter().map(|i| IndexKey::Int(*i)).collect::<Vec<_>>(),
            SetRepr::Strings(v) => {
                v.par_iter().map(|s| IndexKey::Str(s.clone())).collect::<Vec<_>>()
            }
            SetRepr::Tuples(v) => {
                v.par_iter().map(|t| IndexKey::Tuple(t.clone())).collect::<Vec<_>>()
            }
        }
        .into_par_iter()
    }
}

#[derive(Debug)]
pub struct SetIter<'a> {
    repr: &'a SetRepr,
    pos: usize,
}

impl<'a> Iterator for SetIter<'a> {
    type Item = IndexKey;
    fn next(&mut self) -> Option<Self::Item> {
        let out = match self.repr {
            SetRepr::Range(v) => v.get(self.pos).copied().map(IndexKey::Int),
            SetRepr::Strings(v) => v.get(self.pos).cloned().map(IndexKey::Str),
            SetRepr::Tuples(v) => v.get(self.pos).cloned().map(IndexKey::Tuple),
        };
        if out.is_some() {
            self.pos += 1;
        }
        out
    }
}
