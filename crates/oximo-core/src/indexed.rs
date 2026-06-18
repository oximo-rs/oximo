use std::marker::PhantomData;
use std::ops::Index;

use oximo_expr::Expr;
use rustc_hash::FxHashMap;

use crate::set::{Axis, FromIndexKey, IndexKey};

/// Backing storage for an [`IndexedVar`].
///
/// `Dense` is used when the domain is a contiguous integer grid (a range, or a
/// product of ranges). `Sparse` (string sets, sparse `from_ints`, or any
/// `filter`ed family) keeps the original hash map.
#[derive(Clone)]
pub(crate) enum Storage<'a> {
    Dense { data: Vec<Expr<'a>>, keys: Vec<IndexKey>, axes: Box<[Axis]> },
    Sparse(FxHashMap<IndexKey, Expr<'a>>),
}

/// Indexed variable: maps an `IndexKey` to a single-variable `Expr`, tagged with
/// the key type `K` its domain decodes to.
///
/// Constructed by [`crate::Model::indexed_var`] or the `variable!` macro. `K` is
/// a phantom marker, carried so the type of an indexed family is visible and
/// typed iteration via [`Self::keys`] is possible.
///
/// When the domain is a contiguous integer range (or a Cartesian product of
/// ranges) the family is stored densely (see [`Storage`]).
/// String, sparse, and `filter`ed families fall back to a hash map.
pub struct IndexedVar<'a, K = IndexKey> {
    pub(crate) storage: Storage<'a>,
    pub(crate) _k: PhantomData<fn() -> K>,
}

impl<'a, K> Clone for IndexedVar<'a, K> {
    fn clone(&self) -> Self {
        Self { storage: self.storage.clone(), _k: PhantomData }
    }
}

impl<'a, K> std::fmt::Debug for IndexedVar<'a, K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndexedVar")
            .field("len", &self.len())
            .field("dense", &self.is_dense())
            .finish()
    }
}

impl<'a, K> IndexedVar<'a, K> {
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
        *self.get_ref(&coords).expect("IndexedVar: coordinates not present")
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

impl<'a, K: FromIndexKey> IndexedVar<'a, K> {
    /// Iterate the family's entries with each key decoded to the typed `K`.
    pub fn keys(&self) -> impl Iterator<Item = (K, Expr<'a>)> + '_ {
        self.iter().map(|(k, e)| (K::from_index_key(k), *e))
    }
}

impl<'a, K, Q: Into<IndexKey>> Index<Q> for IndexedVar<'a, K> {
    type Output = Expr<'a>;
    fn index(&self, key: Q) -> &Self::Output {
        match &self.storage {
            Storage::Sparse(m) => m.get(&key.into()).expect("IndexedVar: key not present"),
            Storage::Dense { data, axes, .. } => {
                let off = grid_offset(axes, &key.into()).expect("IndexedVar: key not present");
                &data[off]
            }
        }
    }
}

impl<'a, K> Index<&IndexKey> for IndexedVar<'a, K> {
    type Output = Expr<'a>;
    fn index(&self, key: &IndexKey) -> &Self::Output {
        match &self.storage {
            Storage::Sparse(m) => m.get(key).expect("IndexedVar: key not present"),
            Storage::Dense { data, axes, .. } => {
                let off = grid_offset(axes, key).expect("IndexedVar: key not present");
                &data[off]
            }
        }
    }
}

impl<'a, K, const N: usize> Index<[usize; N]> for IndexedVar<'a, K> {
    type Output = Expr<'a>;
    fn index(&self, coords: [usize; N]) -> &Self::Output {
        self.get_ref(&coords).expect("IndexedVar: coordinates not present")
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
/// [`IndexedVar::get_ref`]).
fn coords_to_key(coords: &[usize]) -> IndexKey {
    if let [single] = coords {
        IndexKey::from(*single)
    } else {
        IndexKey::Tuple(coords.iter().map(|&c| IndexKey::from(c)).collect())
    }
}
