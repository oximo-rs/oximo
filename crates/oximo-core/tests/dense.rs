//! Dense-storage fast path for `IndexedVar` over integer-range grids.
//!
//! Range / product-of-range domains store densely; string, sparse `from_ints`,
//! and `filter`ed families stay sparse.

#![allow(clippy::float_cmp)]

use oximo_core::prelude::*;

#[test]
fn range_family_is_dense_1d() {
    let m = Model::new("d1");
    variable!(m, x[i in 0..5] >= 0.0);
    assert!(x.is_dense());
    assert_eq!(x.shape().as_deref(), Some(&[5usize][..]));
    assert_eq!(x.len(), 5);
}

#[test]
fn product_family_is_dense_2d() {
    let m = Model::new("d2");
    variable!(m, b[i in 0..3, n in 0..4] >= 0.0);
    assert!(b.is_dense());
    assert_eq!(b.shape().as_deref(), Some(&[3usize, 4][..]));
    assert_eq!(b.len(), 12);

    // The universal key path, the `at` method, and the array `Index` all resolve
    // to the same scalar.
    for i in 0..3 {
        for n in 0..4 {
            let via_key = b[(i, n)];
            let via_at = b.at([i, n]);
            let via_arr = b[[i, n]];
            assert_eq!(via_key.var_id(), via_at.var_id());
            assert_eq!(via_key.var_id(), via_arr.var_id());
        }
    }
    assert!(b.get_at([3, 0]).is_none());
    assert!(b.get_at([0, 4]).is_none());
}

#[test]
fn range_inclusive_is_dense() {
    let m = Model::new("dinc");
    variable!(m, st[s in 0..2, n in 0..=3] >= 0.0);
    assert!(st.is_dense());
    assert_eq!(st.shape().as_deref(), Some(&[2usize, 4][..]));
    assert_eq!(st.len(), 8);
    assert_eq!(st[(1, 3)].var_id(), st.at([1, 3]).var_id());
    assert!(st.get_at([0, 4]).is_none());
}

#[test]
fn nonzero_start_offsets_correctly() {
    let m = Model::new("dstart");
    variable!(m, x[i in 2..5] >= 0.0);
    assert!(x.is_dense());
    assert_eq!(x.shape().as_deref(), Some(&[3usize][..]));
    // Coordinates are key values: `at([2])` and key `2` hit the same scalar.
    assert_eq!(x[2].var_id(), x.at([2]).var_id());
    assert_eq!(x[4].var_id(), x.get_at([4]).unwrap().var_id());
    assert!(x.get(5).is_none());
    assert!(x.get_at([5]).is_none());
}

#[test]
fn iter_and_keys_preserve_order_when_dense() {
    let m = Model::new("diter");
    variable!(m, x[i in 0..3] >= 0.0);
    // Dense iteration is deterministic and in key order.
    let got: Vec<i64> = x.iter().map(|(k, _)| k.as_i64().unwrap()).collect();
    assert_eq!(got, vec![0, 1, 2]);
    let typed: Vec<usize> = x.keys().map(|(k, _)| k).collect();
    assert_eq!(typed, vec![0, 1, 2]);
}

#[test]
fn string_family_is_sparse() {
    let m = Model::new("sstr");
    let plants = Set::strings(["a", "b", "c"]);
    variable!(m, x[p in plants] >= 0.0);
    assert!(!x.is_dense());
    assert_eq!(x.shape(), None);
    assert_eq!(x.len(), 3);
    assert!(x.get("a").is_some());
    assert!(x.get("z").is_none());
}

#[test]
fn sparse_from_ints_is_not_dense() {
    let m = Model::new("ssparse");
    let s = Set::from_ints(vec![0usize, 2, 4]);
    variable!(m, x[i in s] >= 0.0);
    assert!(!x.is_dense());
    assert_eq!(x.len(), 3);
    assert!(x.get(0).is_some());
    assert!(x.get(2).is_some());
    assert!(x.get(1).is_none());
}

#[test]
fn filtered_family_is_not_dense() {
    let m = Model::new("sfilter");
    variable!(m, x[i in 0..5 if i % 2 == 0] >= 0.0);
    // Filtering breaks contiguity, so the family falls back to sparse storage.
    assert!(!x.is_dense());
    assert_eq!(x.shape(), None);
    assert_eq!(x.len(), 3);
    assert_eq!(x[0].var_id(), x.get(0).unwrap().var_id());
    assert!(x.get(1).is_none());
}

#[test]
#[should_panic(expected = "key not present")]
fn dense_index_out_of_range_panics() {
    let m = Model::new("dpanic");
    variable!(m, b[i in 0..2, n in 0..2] >= 0.0);
    let _ = b[(2, 0)];
}
