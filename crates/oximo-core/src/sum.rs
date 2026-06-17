use oximo_expr::Expr;

use crate::set::{FromIndexKey, Set};

/// Domain over which [`sum_over`] iterates. Lets a single `sum_over` call
/// accept either a [`Set`] (with typed key decoding via [`FromIndexKey`])
/// or a borrowed slice of `Copy` keys, without intermediate conversions.
///
/// Returns an iterator (rather than taking a callback) so the trait method
/// monomorphizes through to the loop body in [`sum_over`], allowing inlining
/// in hot sums. Implementations are typically one line.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a valid index domain over key type `{K}`",
    label = "the domain's keys are not `{K}`",
    note = "the loop/closure binding type must match the domain's keys",
    note = "for a `Set<T>` write `for x in set` (the key type is inferred) or annotate `for x: T in set`. Integer ranges yield `usize`/`i64`/`i32`. A slice/`Vec`/array yields its element type"
)]
pub trait SumDomain<K> {
    fn keys(&self) -> impl Iterator<Item = K> + '_;
}

// A typed set yields exactly its own key type.
// The single `SumDomain` impl for `Set<K>`, so `sum!`/`constraint!`
// can infer the closure parameter without an annotation
// (the erased `Set` defaulted to `Set<IndexKey>`).
impl<K: FromIndexKey> SumDomain<K> for Set<K> {
    fn keys(&self) -> impl Iterator<Item = K> + '_ {
        self.iter().map(|k| K::from_index_key(&k))
    }
}

impl<K: Copy> SumDomain<K> for [K] {
    fn keys(&self) -> impl Iterator<Item = K> + '_ {
        self.iter().copied()
    }
}

impl<K: Copy> SumDomain<K> for Vec<K> {
    fn keys(&self) -> impl Iterator<Item = K> + '_ {
        self.iter().copied()
    }
}

impl<K: Copy, const N: usize> SumDomain<K> for [K; N] {
    fn keys(&self) -> impl Iterator<Item = K> + '_ {
        self.iter().copied()
    }
}

// Forward through a reference, so a domain that is itself a reference (e.g. a
// `&Set` function parameter passed to `sum!`/`constraint!`) is accepted.
impl<K, D: SumDomain<K> + ?Sized> SumDomain<K> for &D {
    fn keys(&self) -> impl Iterator<Item = K> + '_ {
        (**self).keys()
    }
}

// Integer ranges as sum domains. Iteration is lazy, so `sum!(x[i] for i in 0..n)`
// allocates nothing. Provided for the common integer types the `sum!`/`constraint!`
// macros default to.
impl SumDomain<usize> for std::ops::Range<usize> {
    fn keys(&self) -> impl Iterator<Item = usize> + '_ {
        self.clone()
    }
}

impl SumDomain<i64> for std::ops::Range<i64> {
    fn keys(&self) -> impl Iterator<Item = i64> + '_ {
        self.clone()
    }
}

impl SumDomain<i32> for std::ops::Range<i32> {
    fn keys(&self) -> impl Iterator<Item = i32> + '_ {
        self.clone()
    }
}

/// Sum an expression over every element of a domain.
///
/// Reads as the mathematical `sum_{k in domain} f(k)`. The closure parameter is
/// either decoded from the domain's [`crate::set::IndexKey`] via [`FromIndexKey`] (when
/// the domain is a [`Set`]) or yielded directly (when the domain is a slice
/// of `Copy` keys).
///
/// # Panics
/// Panics if `domain` is empty, the resulting expression has no arena to
/// attach to.
#[deprecated(
    since = "0.3.0",
    note = "use the `sum!` macro, the builder API is scheduled for removal in 0.4.0"
)]
pub fn sum_over<'a, K, D, F>(domain: &D, f: F) -> Expr<'a>
where
    D: SumDomain<K> + ?Sized,
    F: FnMut(K) -> Expr<'a>,
{
    __sum_over(domain, f)
}

/// Macro-facing entry point behind [`sum_over`]. Backs the `sum!` macro. Not
/// part of the stable public API.
#[doc(hidden)]
pub fn __sum_over<'a, K, D, F>(domain: &D, mut f: F) -> Expr<'a>
where
    D: SumDomain<K> + ?Sized,
    F: FnMut(K) -> Expr<'a>,
{
    let mut iter = domain.keys();
    let first = f(iter.next().expect("sum_over on empty domain"));
    iter.fold(first, |acc, k| acc + f(k))
}

#[cfg(test)]
// exercises the builder API directly until its 0.4.0 removal
#[allow(deprecated)]
mod tests {
    use oximo_expr::extract_linear;

    use super::*;
    use crate::model::Model;

    #[test]
    fn sum_over_scalar_set() {
        let m = Model::new("scalar");
        let items = Set::range(0..4);
        let x = m.indexed_var("x", &items).lb(0.0).build();

        let total = sum_over(&items, |i: usize| x[i]);
        let arena = m.arena();
        let terms = extract_linear(&arena, total.id).expect("linear");
        assert_eq!(terms.coeffs.len(), 4);
        assert!(terms.coeffs.iter().all(|(_, c)| (c - 1.0).abs() < f64::EPSILON));
    }

    #[test]
    fn sum_over_tuple_set() {
        let m = Model::new("tuple");
        let plants = Set::strings(["seattle", "san-diego"]);
        let markets = Set::strings(["nyc", "chicago", "topeka"]);
        let routes = &plants * &markets;
        let x = m.indexed_var("x", &routes).lb(0.0).build();

        let total = sum_over(&routes, |(p, q): (String, String)| x[(p, q)]);
        let arena = m.arena();
        let terms = extract_linear(&arena, total.id).expect("linear");
        assert_eq!(terms.coeffs.len(), 6);
    }

    #[test]
    fn nested_sum_over_double_sum() {
        let m = Model::new("nested");
        let plants = Set::strings(["a", "b"]);
        let markets = Set::strings(["x", "y", "z"]);
        let routes = &plants * &markets;
        let x = m.indexed_var("x", &routes).lb(0.0).build();

        let total = sum_over(&plants, |p: String| sum_over(&markets, |q: String| x[(&p, q)]));
        let arena = m.arena();
        let terms = extract_linear(&arena, total.id).expect("linear");
        assert_eq!(terms.coeffs.len(), 6);
    }

    #[test]
    fn sum_over_passes_typed_usize_key() {
        let m = Model::new("usizekey");
        let items = Set::range(0..3);
        let x = m.indexed_var("x", &items).lb(0.0).build();

        let total = sum_over(&items, |i: usize| x[i]);
        let arena = m.arena();
        let terms = extract_linear(&arena, total.id).expect("linear");
        assert_eq!(terms.coeffs.len(), 3);
    }

    #[test]
    fn sum_over_slice_of_usize() {
        let m = Model::new("slice");
        let items = Set::range(0..5);
        let x = m.indexed_var("x", &items).lb(0.0).build();

        let picked: &[usize] = &[0, 2, 4];
        let total = sum_over(picked, |i: usize| x[i]);
        let arena = m.arena();
        let terms = extract_linear(&arena, total.id).expect("linear");
        assert_eq!(terms.coeffs.len(), 3);
    }

    #[test]
    fn sum_over_vec_of_usize() {
        let m = Model::new("vec");
        let items = Set::range(0..5);
        let x = m.indexed_var("x", &items).lb(0.0).build();

        let picked: Vec<usize> = vec![1, 3];
        let total = sum_over(&picked, |i: usize| x[i]);
        let arena = m.arena();
        let terms = extract_linear(&arena, total.id).expect("linear");
        assert_eq!(terms.coeffs.len(), 2);
    }

    #[test]
    fn sum_over_array_of_usize() {
        let m = Model::new("array");
        let items = Set::range(0..5);
        let x = m.indexed_var("x", &items).lb(0.0).build();

        let picked: [usize; 4] = [0, 1, 2, 3];
        let total = sum_over(&picked, |i: usize| x[i]);
        let arena = m.arena();
        let terms = extract_linear(&arena, total.id).expect("linear");
        assert_eq!(terms.coeffs.len(), 4);
    }

    #[test]
    #[should_panic(expected = "sum_over on empty domain")]
    fn sum_over_empty_set_panics() {
        let m = Model::new("empty");
        let empty = Set::range(0..0);
        let _x = m.indexed_var("x", &Set::range(0..1)).lb(0.0).build();
        let _ = sum_over(&empty, |_: usize| panic!("closure should not run"));
    }

    #[test]
    #[should_panic(expected = "sum_over on empty domain")]
    fn sum_over_empty_slice_panics() {
        let m = Model::new("empty_slice");
        let _x = m.indexed_var("x", &Set::range(0..1)).lb(0.0).build();
        let empty: &[usize] = &[];
        let _ = sum_over(empty, |_: usize| panic!("closure should not run"));
    }
}
