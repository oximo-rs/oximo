//! Runtime support items the `oximo-macros` procedural macros expand into.
//!
//! This module is NOT part of the stable public API. It exists so that the
//! `variable!`/`constraint!`/`objective!`/`sum!`/`param!` macros have a
//! fully-qualified path (`<crate>::__macro_support::...`) to reference
//! regardless of how the user brought the macros into scope. Items here
//! are re-exports of the real modeling surface plus a couple of helpers.

pub use crate::constraint::Relate;
pub use crate::domain::Domain;
pub use crate::set::{FromIndexKey, KeyCat, Set};
pub use crate::sum::__sum_over as sum_over;
pub use crate::sum::SumDomain;

/// Flatten already-collected `sum!` terms into one expression (a single n-ary
/// `Add`).
#[must_use]
pub fn sum_terms<'a>(terms: Vec<oximo_expr::Expr<'a>>) -> oximo_expr::Expr<'a> {
    terms.into_iter().sum()
}

/// Filter a [`Set`] by a typed predicate over its decoded keys. Backs the
/// filtered family form `name[i in dom if cond]` of `variable!`/`constraint!`.
pub fn filter_keys<K, F>(set: &Set<K>, pred: F) -> Set<K>
where
    K: FromIndexKey,
    F: FnMut(K) -> bool,
{
    set.filter_typed(pred)
}

/// Typed key iterator over a sum/constraint domain. Backs the filtered form of
/// the `sum!` macro (`sum!(.. for i in dom if cond)`), which iterates and skips
/// non-matching keys rather than summing zero terms.
pub fn keys_of<K, D: SumDomain<K> + ?Sized>(d: &D) -> impl Iterator<Item = K> + '_ {
    d.keys()
}

use num_traits::PrimInt;

/// Conversion into an owned [`Set`], used by the `variable!`/`constraint!`
/// macros to turn an index domain (`i in 0..n`, `i in some_set`) into the
/// [`Set`] the indexed `variable!` form expects. The associated `Key` is
/// the type the domain's keys decode to, so the macro can infer the closure
/// parameter type. Integer ranges decode to `usize`. The macro always passes
/// the domain by reference; the blanket impl for `&S` forwards through, so a
/// domain that is itself a `&Set` (e.g. a function parameter) also works.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a valid index domain for a `variable!`/`constraint!` family",
    label = "not an index domain",
    note = "use a `Set`, an integer range `a..b` / `a..=b`, or a reference to one"
)]
pub trait IntoSet {
    type Key;
    fn into_set(self) -> Set<Self::Key>;
}

impl<K> IntoSet for Set<K> {
    type Key = K;
    fn into_set(self) -> Set<K> {
        self
    }
}

impl<T: PrimInt> IntoSet for std::ops::Range<T> {
    type Key = usize;
    fn into_set(self) -> Set<usize> {
        Set::range(self)
    }
}

impl<T: PrimInt> IntoSet for std::ops::RangeInclusive<T> {
    type Key = usize;
    fn into_set(self) -> Set<usize> {
        let start = self.start().to_i64().expect("range start out of i64 range");
        let end = self.end().to_i64().expect("range end out of i64 range");
        Set::dense_i64(start, end.checked_add(1).expect("inclusive range end overflows i64"))
    }
}

impl<S: IntoSet + Clone> IntoSet for &S {
    type Key = S::Key;
    fn into_set(self) -> Set<S::Key> {
        (*self).clone().into_set()
    }
}

/// Normalize a macro index domain into an owned [`Set`], preserving its key type.
pub fn as_set<S: IntoSet>(s: S) -> Set<S::Key> {
    s.into_set()
}

/// Cartesian product of two index domains, composing their key types. Used by
/// the `variable!`/`constraint!` macros to combine multiple `pat in domain`
/// clauses.
#[must_use]
pub fn product<A, B>(a: &Set<A>, b: &Set<B>) -> Set<<A as KeyCat<B>>::Out>
where
    A: KeyCat<B>,
{
    Set::product(a, b)
}
