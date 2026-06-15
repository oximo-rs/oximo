#![allow(clippy::float_cmp)]
// builder API exercised in tests until its 0.4.0 removal
#![allow(deprecated)]

use oximo_core::prelude::*;

#[test]
fn range_set_iteration_order() {
    let s = Set::range(0..3);
    let keys: Vec<_> = s.iter().collect();
    assert_eq!(keys, vec![IndexKey::Int(0), IndexKey::Int(1), IndexKey::Int(2)]);
    assert_eq!(s.len(), 3);
}

#[test]
fn range_accepts_any_primitive_int_type() {
    // Default untyped literals -> i32.
    assert_eq!(Set::range(0..3).len(), 3);

    // Range<usize> (common when keys come from a `.len()`).
    let len: usize = 4;
    assert_eq!(Set::range(0..len).len(), 4);

    // Range<i64> (explicit).
    assert_eq!(Set::range(0_i64..5).len(), 5);

    // Range<u32>, Range<i8>, also work via PrimInt.
    assert_eq!(Set::range(0_u32..2).len(), 2);
    assert_eq!(Set::range(0_i8..6).len(), 6);
}

#[test]
fn from_ints_with_sparse_iterator() {
    let evens: Vec<usize> = vec![0, 2, 4, 6, 8];
    let s = Set::from_ints(evens);
    let keys: Vec<i64> = s.iter().map(|k| k.as_i64().unwrap()).collect();
    assert_eq!(keys, vec![0, 2, 4, 6, 8]);
}

#[test]
fn strings_set_iteration_order() {
    let s = Set::strings(["a", "b", "c"]);
    let keys: Vec<_> = s.iter().collect();
    assert_eq!(keys.len(), 3);
    assert_eq!(keys[0].as_str(), Some("a"));
    assert_eq!(keys[2].as_str(), Some("c"));
}

#[test]
fn cartesian_product_yields_all_pairs() {
    let i = Set::range(0..2);
    let j = Set::range(0..3);
    let ij = &i * &j;
    assert_eq!(ij.len(), 6);
    let keys: Vec<_> = ij.iter().collect();
    // First key is (0,0).
    let first = keys[0].as_tuple().unwrap();
    assert_eq!(first.len(), 2);
    assert_eq!(first[0].as_i64(), Some(0));
    assert_eq!(first[1].as_i64(), Some(0));
    // Last key is (1,2).
    let last = keys[5].as_tuple().unwrap();
    assert_eq!(last[0].as_i64(), Some(1));
    assert_eq!(last[1].as_i64(), Some(2));
}

#[test]
fn product_flattens_nested_tuples() {
    let a = Set::range(0..2);
    let b = Set::range(0..2);
    let c = Set::range(0..2);
    let abc = &(&a * &b) * &c;
    let keys: Vec<_> = abc.iter().collect();
    assert_eq!(keys.len(), 8);
    for k in &keys {
        let parts = k.as_tuple().unwrap();
        assert_eq!(parts.len(), 3, "tuple must be 3-arity, not nested");
        for p in parts {
            assert!(p.as_i64().is_some(), "all parts must be scalars after flattening");
        }
    }
}

#[test]
fn filter_keeps_variant_for_range() {
    let s = Set::range(0..10).filter(|k| k.as_i64().unwrap() % 2 == 0);
    assert!(matches!(s, Set::Range(_)));
    assert_eq!(s.len(), 5);
}

#[test]
fn filter_keeps_variant_for_tuples() {
    let i = Set::range(0..3);
    let j = Set::range(0..3);
    let filtered = (&i * &j).filter(|k| {
        let p = k.as_tuple().unwrap();
        p[0].as_i64() != p[1].as_i64()
    });
    assert!(matches!(filtered, Set::Tuples(_)));
    assert_eq!(filtered.len(), 6);
}

#[test]
fn filter_on_product() {
    let i = Set::range(0..3);
    let j = Set::range(0..3);
    let diag = (&i * &j).filter(|k| {
        let p = k.as_tuple().unwrap();
        p[0].as_i64() == p[1].as_i64()
    });
    assert_eq!(diag.len(), 3);
}

#[test]
fn tuple_key_from_pair_literal() {
    let k: IndexKey = (1, 2).into();
    let parts = k.as_tuple().unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0].as_i64(), Some(1));
    assert_eq!(parts[1].as_i64(), Some(2));
}

#[test]
fn indexed_var_over_product_creates_named_scalars() {
    let m = Model::new("net");
    let plants = Set::strings(["seattle", "san-diego"]);
    let markets = Set::strings(["nyc", "chi", "topeka"]);
    let flow = m.indexed_var("flow", &(&plants * &markets)).lb(0.0).build();
    assert_eq!(flow.len(), 6);
    assert!(m.variable_id("flow[seattle,nyc]").is_some());
    assert!(m.variable_id("flow[san-diego,topeka]").is_some());
}

#[test]
fn indexed_var_per_key_bounds() {
    let m = Model::new("ub_by");
    let set = Set::range(0..3);
    let _x = m
        .indexed_var("x", &set)
        .lb(0.0)
        .ub_by(|k: i64| {
            #[allow(clippy::cast_precision_loss)]
            {
                k as f64
            }
        })
        .build();
    let vars = m.variables();
    assert_eq!(vars[0].ub, 0.0);
    assert_eq!(vars[1].ub, 1.0);
    assert_eq!(vars[2].ub, 2.0);
}

#[test]
fn indexed_var_tuple_indexing() {
    let model = Model::new("idx");
    let rows = Set::range(0..2);
    let cols = Set::range(0..2);
    let x = model.indexed_var("x", &(&rows * &cols)).lb(0.0).build();
    let by_tuple = x[(0, 1)];

    let key: IndexKey = (1, 0).into();
    let by_ref = x[&key];

    // IDs must differ, the two scalars are not the same variable.
    assert_ne!(by_tuple.id, by_ref.id);
}

#[test]
fn add_constraints_over_scalar_typed_closure() {
    let m = Model::new("rule");
    let set = Set::range(0..3);
    let x = m.indexed_var("x", &set).lb(0.0).build();
    m.add_constraints_over("c", &set, |i: usize| x[i].le(10.0));
    assert_eq!(m.num_constraints(), 3);
    assert!(m.constraint_id("c[0]").is_some());
    assert!(m.constraint_id("c[2]").is_some());
}

#[test]
fn add_constraints_over_tuple_set_typed_closure() {
    let m = Model::new("rule_tup");
    let rows = Set::range(0..2);
    let cols = Set::strings(["a", "b"]);
    let ij = &rows * &cols;
    let x = m.indexed_var("x", &ij).lb(0.0).build();
    m.add_constraints_over("c", &ij, |(i, j): (i64, String)| x[(i, j)].le(5.0));
    assert_eq!(m.num_constraints(), 4);
    assert!(m.constraint_id("c[0,a]").is_some());
    assert!(m.constraint_id("c[1,b]").is_some());
}

#[test]
fn add_constraints_over_raw_key_escape_hatch() {
    let m = Model::new("rule_raw");
    let set = Set::range(0..2);
    let x = m.indexed_var("x", &set).lb(0.0).build();
    m.add_constraints_over("c", &set, |k: IndexKey| x[&k].le(1.0));
    assert_eq!(m.num_constraints(), 2);
}

#[test]
fn from_index_key_scalar_impls() {
    let k = IndexKey::Int(42);
    assert_eq!(i64::from_index_key(&k), 42_i64);
    assert_eq!(usize::from_index_key(&k), 42_usize);
    assert_eq!(i32::from_index_key(&k), 42_i32);

    let s = IndexKey::Str("hi".into());
    assert_eq!(String::from_index_key(&s), "hi");

    // Identity: round-trips full key.
    assert_eq!(IndexKey::from_index_key(&k), k);
}

#[test]
fn from_index_key_tuple_impls() {
    let k2: IndexKey = (1, "a").into();
    let (a, b): (i64, String) = FromIndexKey::from_index_key(&k2);
    assert_eq!(a, 1);
    assert_eq!(b, "a");

    let k3: IndexKey = (1, 2, 3).into();
    let (a, b, c): (usize, usize, usize) = FromIndexKey::from_index_key(&k3);
    assert_eq!((a, b, c), (1, 2, 3));

    let k4: IndexKey = (1, 2, 3, 4).into();
    let t: (i64, i64, i64, i64) = FromIndexKey::from_index_key(&k4);
    assert_eq!(t, (1, 2, 3, 4));
}

#[test]
fn lb_by_ub_by_override_binary_defaults() {
    // binary() sets lb=0, ub=1 for all keys.
    // lb_by/ub_by must win per-key: key 0 fixed to 1, key 2 fixed to 0,
    // key 1 left as free binary (lb=0, ub=1).
    let m = Model::new("fix");
    let set = Set::range(0..3);
    let _x = m
        .indexed_var("x", &set)
        .binary()
        .lb_by(|k: i64| if k == 0 { 1.0 } else { 0.0 })
        .ub_by(|k: i64| if k == 2 { 0.0 } else { 1.0 })
        .build();
    let vars = m.variables();
    assert_eq!(vars[0].lb, 1.0); // fixed to 1
    assert_eq!(vars[0].ub, 1.0);
    assert_eq!(vars[1].lb, 0.0); // free binary
    assert_eq!(vars[1].ub, 1.0);
    assert_eq!(vars[2].lb, 0.0); // fixed to 0
    assert_eq!(vars[2].ub, 0.0);
}

#[test]
fn tuple_product_associativity_shape() {
    let a = Set::range(0..2);
    let b = Set::range(0..2);
    let c = Set::range(0..2);

    let left = &(&a * &b) * &c;
    let right = &a * &(&b * &c);

    let left_keys: Vec<_> = left.iter().collect();
    let right_keys: Vec<_> = right.iter().collect();

    assert_eq!(left_keys, right_keys);
}

#[test]
#[allow(clippy::cast_possible_wrap)]
fn large_product_preserves_lex_order() {
    // Crosses the rayon threshold in Set::product (4096) so the parallel
    // path runs. Order must still be lex (outer = a, inner = b).
    let a = Set::range(0..100_i64);
    let b = Set::range(0..100_i64);
    let ab = &a * &b;
    assert_eq!(ab.len(), 10_000);
    let keys: Vec<_> = ab.iter().collect();
    for i in 0..100 {
        for j in 0..100 {
            let parts = keys[i * 100 + j].as_tuple().unwrap();
            assert_eq!(parts[0].as_i64(), Some(i as i64));
            assert_eq!(parts[1].as_i64(), Some(j as i64));
        }
    }
}

#[test]
fn duplicates_are_preserved() {
    // Set is an ordered list, not a mathematical set. Duplicates survive.
    let s = Set::from_ints(vec![1_i32, 1, 2]);
    assert_eq!(s.len(), 3);
    let keys: Vec<i64> = s.iter().map(|k| k.as_i64().unwrap()).collect();
    assert_eq!(keys, vec![1, 1, 2]);

    let t = Set::strings(["a", "a", "b"]);
    assert_eq!(t.len(), 3);
    assert_eq!(t.iter().filter(|k| k.as_str() == Some("a")).count(), 2);
}

#[test]
#[should_panic(expected = "expected tuple of arity 2")]
fn from_index_key_panics_on_arity_mismatch() {
    let k: IndexKey = (1, 2, 3).into();
    let _: (i64, i64) = FromIndexKey::from_index_key(&k);
}

#[test]
#[should_panic(expected = "out of usize range")]
fn from_index_key_panics_on_negative_usize() {
    let k = IndexKey::Int(-1);
    let _ = usize::from_index_key(&k);
}
