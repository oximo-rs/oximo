use std::marker::PhantomData;
use std::ops::Index;

use oximo_expr::Expr;
use rustc_hash::FxHashMap;

use crate::set::{FromIndexKey, IndexKey};

/// Sparse indexed variable: maps an `IndexKey` to a single-variable `Expr`,
/// tagged with the key type `K` its domain decodes to.
///
/// Constructed by [`crate::Model::indexed_var`] or the `variable!` macro. `K` is
/// a phantom marker, carried so the type of an indexed family is visible and
/// typed iteration via [`Self::keys`] is possible.
pub struct IndexedVar<'a, K = IndexKey> {
    pub(crate) entries: FxHashMap<IndexKey, Expr<'a>>,
    pub(crate) _k: PhantomData<fn() -> K>,
}

impl<'a, K> Clone for IndexedVar<'a, K> {
    fn clone(&self) -> Self {
        Self { entries: self.entries.clone(), _k: PhantomData }
    }
}

impl<'a, K> std::fmt::Debug for IndexedVar<'a, K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndexedVar").field("entries", &self.entries.len()).finish()
    }
}

impl<'a, K> IndexedVar<'a, K> {
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&IndexKey, &Expr<'a>)> {
        self.entries.iter()
    }

    pub fn get<Q: Into<IndexKey>>(&self, key: Q) -> Option<Expr<'a>> {
        self.entries.get(&key.into()).copied()
    }
}

impl<'a, K: FromIndexKey> IndexedVar<'a, K> {
    /// Iterate the family's entries with each key decoded to the typed `K`.
    pub fn keys(&self) -> impl Iterator<Item = (K, Expr<'a>)> + '_ {
        self.entries.iter().map(|(k, e)| (K::from_index_key(k), *e))
    }
}

impl<'a, K, Q: Into<IndexKey>> Index<Q> for IndexedVar<'a, K> {
    type Output = Expr<'a>;
    fn index(&self, key: Q) -> &Self::Output {
        self.entries.get(&key.into()).expect("IndexedVar: key not present")
    }
}

impl<'a, K> Index<&IndexKey> for IndexedVar<'a, K> {
    type Output = Expr<'a>;
    fn index(&self, key: &IndexKey) -> &Self::Output {
        self.entries.get(key).expect("IndexedVar: key not present")
    }
}
