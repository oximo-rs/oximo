use oximo_core::prelude::*;

#[test]
fn scalar_ranges_and_display() {
    let m = Model::new("indicators");
    variable!(m, b, Binary);
    variable!(m, x);
    let le = indicator_constraint!(m, cap, b == 1 => x <= 4.0);
    let range = indicator_constraint!(m, band, b == 0 => -2.0 <= x <= 3.0);
    assert_eq!(le.id(), IndicatorConstraintId(0));
    assert!(matches!(range, RangeIndicatorConstraintHandles::Interval(_)));
    assert_eq!(m.num_indicator_constraints(), 2);
    assert_eq!(m.num_constraints(), 2);
    assert_eq!(m.display_indicator(le.id()).to_string(), "cap: b = 1 -> x <= 4");
}

#[test]
fn anonymous_computed_and_indexed_forms() {
    let m = Model::new("families");
    variable!(m, b[i in 0..3], Binary);
    variable!(m, x[i in 0..3]);
    indicator_constraint!(m, b[0] == 1 => x[0] == 2.0);
    let name = "dynamic";
    indicator_constraint!(m, name = name, b[1] == 0 => x[1] >= 1.0);
    let family = indicator_constraint!(m, cap[i in 0..3 if i != 1], b[i] == 1 => x[i] <= f64::from(u32::try_from(i).unwrap()));
    assert_eq!(family.len(), 2);
    assert!(family.get(1usize).is_none());
    assert_eq!(m.indicator_constraint_id("dynamic"), Some(IndicatorConstraintId(1)));
    assert_eq!(m.num_indicator_constraints(), 4);
}

#[test]
#[should_panic(expected = "binary domain")]
fn rejects_non_binary_trigger() {
    let m = Model::new("bad");
    variable!(m, b);
    variable!(m, x);
    indicator_constraint!(m, bad, b == 1 => x <= 1.0);
}

#[test]
#[should_panic(expected = "affine")]
fn rejects_nonlinear_consequent() {
    let m = Model::new("bad");
    variable!(m, b, Binary);
    variable!(m, x);
    indicator_constraint!(m, bad, b == 1 => x.powi(2) <= 1.0);
}

#[test]
fn split_range_name_collision_does_not_partially_register() {
    let m = Model::new("atomic_split");
    variable!(m, b, Binary);
    variable!(m, x);
    variable!(m, lo);
    indicator_constraint!(m, band_hi, b == 1 => x <= 10.0);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        indicator_constraint!(m, band, b == 1 => lo <= x <= 5.0);
    }));

    assert!(result.is_err());
    assert_eq!(m.num_indicator_constraints(), 1);
    assert_eq!(m.indicator_constraint_id("band_lo"), None);
    assert_eq!(m.indicator_constraint_id("band_hi"), Some(IndicatorConstraintId(0)));
}

#[test]
fn indexed_name_collision_does_not_partially_register() {
    let m = Model::new("atomic_family");
    variable!(m, b[i in 0..3], Binary);
    variable!(m, x[i in 0..3]);
    indicator_constraint!(m, name = "cap[1]", b[1] == 1 => x[1] <= 10.0);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        indicator_constraint!(m, cap[i in 0..3], b[i] == 1 => x[i] <= 5.0);
    }));

    assert!(result.is_err());
    assert_eq!(m.num_indicator_constraints(), 1);
    assert_eq!(m.indicator_constraint_id("cap[0]"), None);
    assert_eq!(m.indicator_constraint_id("cap[1]"), Some(IndicatorConstraintId(0)));
    assert_eq!(m.indicator_constraint_id("cap[2]"), None);
}

#[test]
fn indexed_split_range_collision_does_not_partially_register() {
    let m = Model::new("atomic_range_family");
    variable!(m, b[i in 0..3], Binary);
    variable!(m, x[i in 0..3]);
    variable!(m, lo[i in 0..3]);
    indicator_constraint!(m, name = "band[1]_hi", b[1] == 1 => x[1] <= 10.0);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        indicator_constraint!(m, band[i in 0..3], b[i] == 1 => lo[i] <= x[i] <= 5.0);
    }));

    assert!(result.is_err());
    assert_eq!(m.num_indicator_constraints(), 1);
    for name in ["band[0]_lo", "band[0]_hi", "band[1]_lo", "band[2]_lo", "band[2]_hi"] {
        assert_eq!(m.indicator_constraint_id(name), None, "unexpected row {name}");
    }
    assert_eq!(m.indicator_constraint_id("band[1]_hi"), Some(IndicatorConstraintId(0)));
}
