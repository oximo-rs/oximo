use std::marker::PhantomData;
use std::ops::Index;

use oximo_expr::Expr;
use rustc_hash::FxHashMap;

use crate::constraint::{ConstraintId, RangeConstraintIds};
use crate::set::{Axis, FromIndexKey, IndexKey};

/// Owned, ordered IDs with a domain-specific lookup index.
#[derive(Clone, Debug)]
struct ConstraintFamilyStorage<T> {
    entries: Vec<(IndexKey, T)>,
    lookup: ConstraintFamilyLookup,
}

#[derive(Clone, Debug)]
enum ConstraintFamilyLookup {
    Dense(Box<[Axis]>),
    Sparse(FxHashMap<IndexKey, usize>),
}

impl<T: Copy> ConstraintFamilyStorage<T> {
    fn new(keys: Vec<IndexKey>, axes: Option<&[Axis]>, values: Vec<T>) -> Self {
        assert_eq!(keys.len(), values.len(), "constraint family key/value length mismatch");
        let lookup = axes.map_or_else(
            || {
                ConstraintFamilyLookup::Sparse(
                    keys.iter().cloned().enumerate().map(|(i, key)| (key, i)).collect(),
                )
            },
            |axes| ConstraintFamilyLookup::Dense(axes.into()),
        );
        Self { entries: keys.into_iter().zip(values).collect(), lookup }
    }

    fn get(&self, key: &IndexKey) -> Option<T> {
        let position = match &self.lookup {
            ConstraintFamilyLookup::Dense(axes) => grid_offset(axes, key)?,
            ConstraintFamilyLookup::Sparse(positions) => *positions.get(key)?,
        };
        self.entries.get(position).map(|(_, value)| *value)
    }
}

macro_rules! constraint_family {
    ($(#[$doc:meta])* $name:ident, $value:ty) => {
        $(#[$doc])*
        pub struct $name<K = IndexKey> {
            storage: ConstraintFamilyStorage<$value>,
            _marker: PhantomData<fn() -> K>,
        }

        impl<K> Clone for $name<K> {
            fn clone(&self) -> Self {
                Self { storage: self.storage.clone(), _marker: PhantomData }
            }
        }

        impl<K> std::fmt::Debug for $name<K> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct(stringify!($name)).field("entries", &self.storage.entries).finish()
            }
        }

        impl<K> $name<K> {
            pub(crate) fn new(keys: Vec<IndexKey>, axes: Option<&[Axis]>, values: Vec<$value>) -> Self {
                Self { storage: ConstraintFamilyStorage::new(keys, axes, values), _marker: PhantomData }
            }

            /// Number of domain entries (not the number of lowered rows).
            pub fn len(&self) -> usize {
                self.storage.entries.len()
            }

            pub fn is_empty(&self) -> bool {
                self.storage.entries.is_empty()
            }

            /// Look up an entry, returning `None` for missing or filtered-out keys.
            pub fn get<Q: Into<IndexKey>>(&self, key: Q) -> Option<$value> {
                self.storage.get(&key.into())
            }
        }

        impl<K: FromIndexKey> $name<K> {
            /// Iterate typed keys and copied IDs in the original domain order.
            pub fn iter(&self) -> impl Iterator<Item = (K, $value)> + '_ {
                self.storage.entries.iter().map(|(key, value)| (K::from_index_key(key), *value))
            }
        }
    };
}

constraint_family!(
    /// Owned handle returned by an indexed single-relation `constraint!` declaration.
    ///
    /// IDs refer to rows of the originating model.
    /// `get` supports integer, string, and tuple keys, including sparse domains.
    IndexedConstraint,
    ConstraintId
);

constraint_family!(
    /// Owned handle returned by an indexed two-sided range `constraint!` declaration.
    ///
    /// Each key maps to one interval ID or separate lower/upper IDs. Entries can
    /// have different lowering forms within the same family.
    IndexedRangeConstraint,
    RangeConstraintIds
);

/// Backing storage for an [`IndexedFamily`].
///
/// `Dense` is used when the domain is a contiguous integer grid (a range, or a
/// product of ranges). `Sparse` (string sets, sparse `from_ints`, or any
/// `filter`ed family) keeps the original hash map.
#[derive(Clone)]
pub(crate) enum Storage<'a> {
    Dense { data: Vec<Expr<'a>>, keys: Vec<IndexKey>, axes: Box<[Axis]> },
    Sparse(FxHashMap<IndexKey, Expr<'a>>),
}

mod sealed {
    pub trait Sealed {}
}

/// Marker selecting which kind of indexed family an [`IndexedFamily`] is.
///
/// Sealed implementation detail: implemented only for [`VarFamily`] and
/// [`ParamFamily`]. It exists so [`IndexedVar`] and [`IndexedParam`] are distinct
/// types (only a parameter family can be re-bound) while sharing one
/// implementation.
#[doc(hidden)]
pub trait Family: sealed::Sealed {
    /// Type name used in [`Debug`](std::fmt::Debug) output.
    const NAME: &'static str;
}

/// Marker for an indexed family of decision variables ([`IndexedVar`]).
#[doc(hidden)]
#[derive(Debug)]
pub struct VarFamily;

/// Marker for an indexed family of parameters ([`IndexedParam`]).
#[doc(hidden)]
#[derive(Debug)]
pub struct ParamFamily;

impl sealed::Sealed for VarFamily {}
impl sealed::Sealed for ParamFamily {}
impl Family for VarFamily {
    const NAME: &'static str = "IndexedVar";
}
impl Family for ParamFamily {
    const NAME: &'static str = "IndexedParam";
}

/// Indexed family: maps an `IndexKey` to a single-element `Expr` (a variable or a
/// parameter), tagged with the key type `K` its domain decodes to and the family
/// kind `F`.
///
/// You normally name this through the [`IndexedVar`]/[`IndexedParam`] aliases,
/// constructed by the indexed form of the `variable!`/`param!` macros.
///
/// When the domain is a contiguous integer range (or a Cartesian product of
/// ranges) the family is stored densely (see the internal `Storage`).
/// String, sparse, and `filter`ed families fall back to a hash map.
pub struct IndexedFamily<'a, K = IndexKey, F = VarFamily> {
    pub(crate) storage: Storage<'a>,
    pub(crate) _marker: PhantomData<fn() -> (K, F)>,
}

/// Indexed family of decision variables; see [`IndexedFamily`].
pub type IndexedVar<'a, K = IndexKey> = IndexedFamily<'a, K, VarFamily>;

/// Indexed family of re-bindable parameters; see [`IndexedFamily`].
///
/// Re-bind a single entry with [`Model::set_param_idx`](crate::Model::set_param_idx).
pub type IndexedParam<'a, K = IndexKey> = IndexedFamily<'a, K, ParamFamily>;

impl<'a, K, F> Clone for IndexedFamily<'a, K, F> {
    fn clone(&self) -> Self {
        Self { storage: self.storage.clone(), _marker: PhantomData }
    }
}

impl<'a, K, F: Family> std::fmt::Debug for IndexedFamily<'a, K, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct(F::NAME).field("len", &self.len()).field("dense", &self.is_dense()).finish()
    }
}

impl<'a, K, F> IndexedFamily<'a, K, F> {
    pub fn len(&self) -> usize {
        match &self.storage {
            Storage::Dense { data, .. } => data.len(),
            Storage::Sparse(m) => m.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether this family is stored densely (domain was a range or product of
    /// ranges).
    pub fn is_dense(&self) -> bool {
        matches!(self.storage, Storage::Dense { .. })
    }

    /// Per-axis lengths when stored densely, else `None`.
    pub fn shape(&self) -> Option<Box<[usize]>> {
        match &self.storage {
            Storage::Dense { axes, .. } => Some(axes.iter().map(|a| a.len).collect()),
            Storage::Sparse(_) => None,
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&IndexKey, &Expr<'a>)> + '_ {
        let it: Box<dyn Iterator<Item = (&IndexKey, &Expr<'a>)>> = match &self.storage {
            Storage::Dense { data, keys, .. } => Box::new(keys.iter().zip(data.iter())),
            Storage::Sparse(m) => Box::new(m.iter()),
        };
        it
    }

    pub fn get<Q: Into<IndexKey>>(&self, key: Q) -> Option<Expr<'a>> {
        match &self.storage {
            Storage::Sparse(m) => m.get(&key.into()).copied(),
            Storage::Dense { data, axes, .. } => {
                grid_offset(axes, &key.into()).map(|off| data[off])
            }
        }
    }

    /// Zero-allocation typed index by integer coordinates.
    /// On a dense family this maps straight to a flat offset with no
    /// `IndexKey` built. On a sparse family it falls back to building a key.
    ///
    /// # Panics
    /// Panics if the coordinates are out of range/not present.
    pub fn at<const N: usize>(&self, coords: [usize; N]) -> Expr<'a> {
        *self.get_ref(&coords).expect("indexed family: coordinates not present")
    }

    /// Fallible form of [`Self::at`].
    pub fn get_at<const N: usize>(&self, coords: [usize; N]) -> Option<Expr<'a>> {
        self.get_ref(&coords).copied()
    }

    fn get_ref(&self, coords: &[usize]) -> Option<&Expr<'a>> {
        match &self.storage {
            Storage::Dense { data, axes, .. } => {
                grid_offset_coords(axes, coords).map(|off| &data[off])
            }
            Storage::Sparse(m) => m.get(&coords_to_key(coords)),
        }
    }
}

impl<'a, K: FromIndexKey, F> IndexedFamily<'a, K, F> {
    /// Iterate the family's entries with each key decoded to the typed `K`.
    pub fn keys(&self) -> impl Iterator<Item = (K, Expr<'a>)> + '_ {
        self.iter().map(|(k, e)| (K::from_index_key(k), *e))
    }
}

impl<'a, K, F, Q: Into<IndexKey>> Index<Q> for IndexedFamily<'a, K, F> {
    type Output = Expr<'a>;
    fn index(&self, key: Q) -> &Self::Output {
        match &self.storage {
            Storage::Sparse(m) => m.get(&key.into()).expect("indexed family: key not present"),
            Storage::Dense { data, axes, .. } => {
                let off = grid_offset(axes, &key.into()).expect("indexed family: key not present");
                &data[off]
            }
        }
    }
}

impl<'a, K, F> Index<&IndexKey> for IndexedFamily<'a, K, F> {
    type Output = Expr<'a>;
    fn index(&self, key: &IndexKey) -> &Self::Output {
        match &self.storage {
            Storage::Sparse(m) => m.get(key).expect("indexed family: key not present"),
            Storage::Dense { data, axes, .. } => {
                let off = grid_offset(axes, key).expect("indexed family: key not present");
                &data[off]
            }
        }
    }
}

impl<'a, K, F, const N: usize> Index<[usize; N]> for IndexedFamily<'a, K, F> {
    type Output = Expr<'a>;
    fn index(&self, coords: [usize; N]) -> &Self::Output {
        self.get_ref(&coords).expect("indexed family: coordinates not present")
    }
}

/// Build [`Storage`] from ordered keys and their already-registered handles.
pub(crate) fn build_storage<'a>(
    keys: Vec<IndexKey>,
    axes: Option<Box<[Axis]>>,
    values: Vec<Expr<'a>>,
) -> Storage<'a> {
    assert_eq!(keys.len(), values.len(), "indexed storage key/value length mismatch");
    if let Some(axes) = axes {
        if keys.iter().enumerate().all(|(i, key)| grid_offset(&axes, key) == Some(i)) {
            return Storage::Dense { data: values, keys, axes };
        }
        let total = keys.len();
        let mut data: Vec<Option<Expr<'a>>> = vec![None; total];
        let mut kept: Vec<Option<IndexKey>> = vec![None; total];
        for (key, expr) in keys.into_iter().zip(values) {
            let off = grid_offset(&axes, &key).expect("dense grid key out of range");
            data[off] = Some(expr);
            kept[off] = Some(key);
        }
        let data = data.into_iter().map(|o| o.expect("dense grid had a gap")).collect();
        let kept = kept.into_iter().map(|o| o.expect("dense grid had a gap")).collect();
        Storage::Dense { data, keys: kept, axes }
    } else {
        let entries = keys.into_iter().zip(values).collect::<FxHashMap<_, _>>();
        Storage::Sparse(entries)
    }
}

/// Position of a key value along one axis, or `None` if out of `[start, start+len)`.
fn axis_index(a: &Axis, v: i64) -> Option<usize> {
    let d = v.checked_sub(a.start)?;
    let u = usize::try_from(d).ok()?;
    (u < a.len).then_some(u)
}

/// Row-major flat offset (axis 0 outermost) of an `IndexKey` in a dense grid, or
/// `None` if the key's shape does not match the axes or is out of range.
pub(crate) fn grid_offset(axes: &[Axis], key: &IndexKey) -> Option<usize> {
    match (axes, key) {
        ([a], IndexKey::Int(v)) => axis_index(a, *v),
        (axes, IndexKey::Tuple(parts)) if parts.len() == axes.len() => {
            let mut off = 0usize;
            for (a, p) in axes.iter().zip(parts.iter()) {
                off = off.checked_mul(a.len)?.checked_add(axis_index(a, p.as_i64()?)?)?;
            }
            Some(off)
        }
        _ => None,
    }
}

/// Row-major flat offset from raw integer coordinates (key values).
fn grid_offset_coords(axes: &[Axis], coords: &[usize]) -> Option<usize> {
    if coords.len() != axes.len() {
        return None;
    }
    let mut off = 0usize;
    for (a, &c) in axes.iter().zip(coords) {
        off = off * a.len + axis_index(a, i64::try_from(c).ok()?)?;
    }
    Some(off)
}

/// Build the `IndexKey` a coordinate array would hash to (sparse fallback for
/// [`IndexedFamily::get_ref`]).
fn coords_to_key(coords: &[usize]) -> IndexKey {
    if let [single] = coords {
        IndexKey::from(*single)
    } else {
        IndexKey::Tuple(coords.iter().map(|&c| IndexKey::from(c)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Model;

    #[test]
    fn dense_storage_keeps_key_value_pairs_when_ordered_or_shuffled() {
        let model = Model::new("storage_order");
        let x = model.__var("x").build();
        let y = model.__var("y").build();
        for (keys, values) in [
            (vec![IndexKey::Int(-1), IndexKey::Int(0)], vec![x, y]),
            (vec![IndexKey::Int(0), IndexKey::Int(-1)], vec![y, x]),
        ] {
            let storage = build_storage(keys, Some(Box::new([Axis { start: -1, len: 2 }])), values);
            let Storage::Dense { data, keys, .. } = storage else {
                panic!("expected dense storage")
            };
            assert_eq!(data.iter().map(|expr| expr.id).collect::<Vec<_>>(), [x.id, y.id]);
            assert_eq!(keys, [IndexKey::Int(-1), IndexKey::Int(0)]);
        }
    }
}
