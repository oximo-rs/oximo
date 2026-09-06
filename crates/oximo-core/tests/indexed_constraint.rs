#![deny(unused_variables)]

use oximo_core::prelude::*;

#[test]
fn scalar_relations_still_return_ids() {
    let m = Model::new("scalar_ids");
    variable!(m, x);
    let named: ConstraintId = constraint!(m, capacity, x <= 10.0);
    let anonymous: ConstraintId = constraint!(m, x >= 0.0);
    let computed: ConstraintId = constraint!(m, name = format!("fixed_{}", 2), x == 2.0);
    assert_eq!(m.constraint_id("capacity"), Some(named));
    assert_eq!(m.constraint_id("_c0"), Some(anonymous));
    assert_eq!(m.constraint_id("fixed_2"), Some(computed));
}

#[test]
fn dense_family_returns_actual_ids_and_owns_its_entries() {
    let cover = {
        let m = Model::new("dense_ids");
        variable!(m, x);
        constraint!(m, prior, x >= 0.0);
        let cover: IndexedConstraint<usize> = constraint!(m, cover[_i in 3..6], x <= 10.0);
        assert_eq!(cover.len(), 3);
        assert!(!cover.is_empty());
        for (i, cid) in cover.iter() {
            assert_eq!(m.constraint_id(&format!("cover[{i}]")), Some(cid));
            assert_eq!(cover.get(i), Some(cid));
        }
        cover
    };
    assert_eq!(
        cover.iter().collect::<Vec<_>>(),
        vec![(3, ConstraintId(1)), (4, ConstraintId(2)), (5, ConstraintId(3)),]
    );
    assert_eq!(cover.get(2), None);
    assert_eq!(cover.get(-1), None);
    assert_eq!(cover.get(6), None);
    assert_eq!(cover.get((3, 0)), None);
    assert_eq!(cover.get("3"), None);
    assert_eq!(cover.clone().iter().collect::<Vec<_>>(), cover.iter().collect::<Vec<_>>());
    assert!(format!("{cover:?}").contains("IndexedConstraint"));
}

#[test]
fn dense_tuple_lookup_uses_each_axis_offset() {
    let m = Model::new("grid_ids");
    variable!(m, x);
    let grid: IndexedConstraint<(usize, usize)> =
        constraint!(m, grid[_i in 2..4, _j in 5..8], x <= 1.0);
    let expected = [(2, 5), (2, 6), (2, 7), (3, 5), (3, 6), (3, 7)];
    assert_eq!(grid.iter().map(|(key, _)| key).collect::<Vec<_>>(), expected);
    for (key, cid) in grid.iter() {
        assert_eq!(grid.get(key), Some(cid));
        assert_eq!(m.constraint_id(&format!("grid[{},{}]", key.0, key.1)), Some(cid));
    }
    assert_eq!(grid.get((2, 8)), None);
    assert_eq!(grid.get((1, 5)), None);
    assert_eq!(grid.get(2), None);
}

#[test]
fn sparse_and_filtered_domains_preserve_order() {
    let m = Model::new("sparse_ids");
    variable!(m, x);
    let indices = Set::from_ints([7, 3, 20]);
    let sparse: IndexedConstraint<usize> = constraint!(m, sparse[_i in indices], x <= 5.0);
    assert_eq!(sparse.iter().map(|(i, _)| i).collect::<Vec<_>>(), [7, 3, 20]);
    for (i, cid) in sparse.iter() {
        assert_eq!(sparse.get(i), Some(cid));
        assert_eq!(m.constraint_id(&format!("sparse[{i}]")), Some(cid));
    }
    assert_eq!(sparse.get(0), None);
    let filtered = constraint!(m, filtered[i in 0..6 if i % 2 == 0], x >= 0.0);
    assert_eq!(filtered.iter().map(|(i, _)| i).collect::<Vec<_>>(), [0, 2, 4]);
    assert_eq!(filtered.get(1), None);
    assert_eq!(filtered.get(4), m.constraint_id("filtered[4]"));
}

#[test]
fn string_and_filtered_tuple_domains_support_lookup() {
    let m = Model::new("string_ids");
    variable!(m, x);
    let plants = Set::strings(["west", "skip", "east"]);
    let supply: IndexedConstraint<String> =
        constraint!(m, supply[p in plants if p != "skip"], x <= 5.0);
    assert_eq!(supply.iter().map(|(p, _)| p).collect::<Vec<_>>(), ["west", "east"]);
    assert_eq!(supply.get("skip"), None);
    assert_eq!(supply.get("east"), m.constraint_id("supply[east]"));
    let grid: IndexedConstraint<(String, usize)> =
        constraint!(m, grid[p in plants, i in 1..3 if p != "skip" && i == 2], x >= 0.0);
    assert_eq!(grid.len(), 2);
    for ((p, i), cid) in grid.iter() {
        assert_eq!(grid.get((p.as_str(), i)), Some(cid));
        assert_eq!(m.constraint_id(&format!("grid[{p},{i}]")), Some(cid));
    }
    assert_eq!(grid.get(("west", 1)), None);
}

#[test]
fn empty_domains_return_empty_handles() {
    let m = Model::new("empty_ids");
    variable!(m, x);
    let empty = constraint!(m, empty[_i in 0..0], x <= 1.0);
    let filtered = constraint!(m, filtered[i in 0..3 if i > 5], x <= 1.0);
    for family in [empty, filtered] {
        assert!(family.is_empty());
        assert_eq!(family.len(), 0);
        assert_eq!(family.iter().count(), 0);
        assert_eq!(family.get(0), None);
    }
    let ranges: IndexedRangeConstraint<usize> = constraint!(m, ranges[_i in 0..0], 0.0 <= x <= 1.0);
    assert!(ranges.is_empty());
    assert_eq!(ranges.iter().count(), 0);
    assert_eq!(ranges.get(0), None);
    assert_eq!(m.num_constraints(), 0);
}

#[test]
fn scalar_ranges_return_named_and_anonymous_groups() {
    let m = Model::new("range_ids");
    variable!(m, x);
    param!(m, lo = 0.0);
    let band: RangeConstraintIds = constraint!(m, band, 0.0 <= x <= 4.0);
    assert_eq!(band, RangeConstraintIds::Interval(m.constraint_id("band").unwrap()));
    let reversed = constraint!(m, name = format!("band_{}", 2), 4.0 >= x >= 0.0);
    assert_eq!(reversed, RangeConstraintIds::Interval(m.constraint_id("band_2").unwrap()));
    let auto = constraint!(m, 0.0 <= x <= 4.0);
    assert_eq!(auto, RangeConstraintIds::Interval(m.constraint_id("_c0").unwrap()));
    let split = constraint!(m, symbolic, lo <= x <= 4.0);
    assert_eq!(
        split,
        RangeConstraintIds::Split {
            lower: m.constraint_id("symbolic_lo").unwrap(),
            upper: m.constraint_id("symbolic_hi").unwrap(),
        }
    );
    let nonlinear = constraint!(m, name = "nonlinear".to_owned(), 4.0 >= x.powi(2) >= 0.0);
    assert_eq!(
        nonlinear,
        RangeConstraintIds::Split {
            lower: m.constraint_id("nonlinear_lo").unwrap(),
            upper: m.constraint_id("nonlinear_hi").unwrap(),
        }
    );
    let auto_split = constraint!(m, lo <= x <= 4.0);
    assert_eq!(
        auto_split,
        RangeConstraintIds::Split {
            lower: m.constraint_id("_c1").unwrap(),
            upper: m.constraint_id("_c2").unwrap(),
        }
    );
}

#[test]
fn range_family_can_mix_interval_and_split_entries() {
    let m = Model::new("mixed_range_ids");
    variable!(m, x);
    constraint!(m, prior, x >= 0.0);
    let bodies = [x, x.powi(2), x + 1.0];
    let ranges: IndexedRangeConstraint<usize> =
        constraint!(m, ranges[i in 0..3], 4.0 >= bodies[i] >= 0.0);
    assert_eq!(ranges.len(), 3);
    assert_eq!(m.num_constraints(), 5);
    assert_eq!(
        ranges.iter().collect::<Vec<_>>(),
        vec![
            (0, RangeConstraintIds::Interval(ConstraintId(1))),
            (1, RangeConstraintIds::Split { lower: ConstraintId(2), upper: ConstraintId(3) }),
            (2, RangeConstraintIds::Interval(ConstraintId(4))),
        ]
    );
    assert_eq!(ranges.get(3), None);
    for (i, ids) in ranges.iter() {
        assert_eq!(ranges.get(i), Some(ids));
    }
    assert_eq!(ranges.clone().iter().collect::<Vec<_>>(), ranges.iter().collect::<Vec<_>>());
    assert!(format!("{ranges:?}").contains("IndexedRangeConstraint"));

    let keys = Set::strings(["last", "skip", "first"]);
    let sparse = constraint!(m, sparse[k in keys if k != "skip"], 0.0 <= x <= 4.0);
    assert_eq!(sparse.iter().map(|(k, _)| k).collect::<Vec<_>>(), ["last", "first"]);
    assert_eq!(sparse.get("skip"), None);
    assert_eq!(
        sparse.get("first"),
        Some(RangeConstraintIds::Interval(m.constraint_id("sparse[first]").unwrap()))
    );
}

#[test]
fn index_used_only_in_format_string_still_binds() {
    let m = Model::new("fmt_ids");
    variable!(m, x);
    let f: IndexedConstraint<usize> =
        constraint!(m, f[i in 0..2], x <= format!("{i}").parse::<f64>().unwrap());
    assert_eq!(f.len(), 2);
    let g: IndexedConstraint<usize> =
        constraint!(m, g[i in 0..4 if !format!("{i}").is_empty()], x <= 1.0);
    assert_eq!(g.len(), 4);
}

#[test]
fn tuple_bindings_can_move_string_keys_after_generated_borrow() {
    let m = Model::new("moved_keys");
    variable!(m, x);
    let keys = Set::strings(["2", "3"]);
    let domain = &keys * &Set::range(0..2);
    // `key` is unused in the filter, `guard` is unused in the rule, and
    // into_bytes consumes the String after the generated acknowledgement.
    let moved = constraint!(m, moved[(key, guard) in domain if guard == 1],
        x <= f64::from(key.into_bytes()[0] - b'0'));
    assert_eq!(moved.len(), 2);
    let rows = m.constraints();
    for (key, expected) in [("2", 2.0), ("3", 3.0)] {
        let id = moved.get((key, 1)).unwrap();
        assert!((rows.algebraic()[id.index()].upper - expected).abs() < f64::EPSILON);
    }
}

#[test]
fn sums_acknowledge_unused_bindings_and_allow_moves() {
    let m = Model::new("sum_bindings");
    variable!(m, x);
    let keys = Set::strings(["2", "3"]);
    let domain = &keys * &Set::range(0..2);
    // Deliberately use a non-underscore name for the unused index: denying
    // unused_variables makes this fail if generated acknowledgements regress.
    let unfiltered = sum!(f64::from(key.into_bytes()[0] - b'0') * x for (key, unused) in domain);
    let filtered =
        sum!(f64::from(key.into_bytes()[0] - b'0') * x for (key, unused) in domain if true);
    let nested =
        sum!(f64::from(key.into_bytes()[0] - b'0') * x for unused in 0..2, key in keys if true);
    let arena = m.arena();
    for total in [unfiltered, filtered, nested] {
        let linear = oximo_expr::extract_linear(&arena, total.id).unwrap();
        assert_eq!(linear.coeffs.len(), 1);
        assert!((linear.coeffs[0].1 - 10.0).abs() < f64::EPSILON);
    }
}
