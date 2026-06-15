//! Runtime support items the `oximo-macros` procedural macros expand into.
//!
//! This module is NOT part of the stable public API. It exists so that the
//! `variable!`/`constraint!`/`objective!`/`sum!`/`param!` macros have a
//! fully-qualified path (`<crate>::__macro_support::...`) to reference
//! regardless of how the user brought the macros into scope. Items here
//! are re-exports of the real modeling surface plus a couple of helpers.

pub use crate::constraint::Relate;
pub use crate::set::{FromIndexKey, Set};
pub use crate::sum::__sum_over as sum_over;
pub use crate::sum::SumDomain;

/// Typed key iterator over a sum/constraint domain. Backs the filtered form of
/// the `sum!` macro (`sum!(.. for i in dom if cond)`), which iterates and skips
/// non-matching keys rather than summing zero terms.
pub fn keys_of<K, D: SumDomain<K> + ?Sized>(d: &D) -> impl Iterator<Item = K> + '_ {
    d.keys()
}

use num_traits::PrimInt;

/// Conversion into an owned [`Set`], used by the `variable!`/`constraint!`
/// macros to turn an index domain (`i in 0..n`, `i in some_set`) into the
/// [`Set`] that [`crate::Model::indexed_var`] expects. The macro always passes
/// the domain by reference; the blanket impl for `&S` forwards through, so a
/// domain that is itself a `&Set` (e.g. a function parameter) also works.
pub trait IntoSet {
    fn into_set(self) -> Set;
}

impl IntoSet for Set {
    fn into_set(self) -> Set {
        self
    }
}

impl<T: PrimInt> IntoSet for std::ops::Range<T> {
    fn into_set(self) -> Set {
        Set::range(self)
    }
}

impl<T: PrimInt> IntoSet for std::ops::RangeInclusive<T> {
    fn into_set(self) -> Set {
        let start = self.start().to_i64().expect("range start out of i64 range");
        let end = self.end().to_i64().expect("range end out of i64 range");
        Set::from_ints(start..=end)
    }
}

impl<S: IntoSet + Clone> IntoSet for &S {
    fn into_set(self) -> Set {
        (*self).clone().into_set()
    }
}

/// Normalize a macro index domain into an owned [`Set`].
pub fn as_set<S: IntoSet>(s: S) -> Set {
    s.into_set()
}

/// Cartesian product of two index domains. Used by the `variable!`/
/// `constraint!` macros to combine multiple `pat in domain` clauses.
#[must_use]
pub fn product(a: &Set, b: &Set) -> Set {
    Set::product(a, b)
}
