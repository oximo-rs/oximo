use std::marker::PhantomData;
use std::ops::Index;

use oximo_expr::Expr;
use rustc_hash::FxHashMap;

use crate::set::{Axis, FromIndexKey, IndexKey};

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

/// Build [`Storage`] from a family's keys and optional dense axes, registering
/// each element through `make`.
pub(crate) fn build_storage<'a>(
    keys: Vec<IndexKey>,
    axes: Option<Box<[Axis]>>,
    mut make: impl FnMut(&IndexKey) -> Expr<'a>,
) -> Storage<'a> {
    if let Some(axes) = axes {
        let total = keys.len();
        let mut data: Vec<Option<Expr<'a>>> = vec![None; total];
        let mut kept: Vec<Option<IndexKey>> = vec![None; total];
        for key in keys {
            let expr = make(&key);
            let off = grid_offset(&axes, &key).expect("dense grid key out of range");
            data[off] = Some(expr);
            kept[off] = Some(key);
        }
        let data = data.into_iter().map(|o| o.expect("dense grid had a gap")).collect();
        let kept = kept.into_iter().map(|o| o.expect("dense grid had a gap")).collect();
        Storage::Dense { data, keys: kept, axes }
    } else {
        let mut entries = FxHashMap::default();
        for key in keys {
            let expr = make(&key);
            entries.insert(key, expr);
        }
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
