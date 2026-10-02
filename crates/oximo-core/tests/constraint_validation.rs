use oximo_core::prelude::*;
use oximo_expr::{ExprClass, extract_linear};
use std::panic::{AssertUnwindSafe, catch_unwind};

fn assert_panics_with<R>(f: impl FnOnce() -> R, expected: &str) {
    let error = catch_unwind(AssertUnwindSafe(f)).err().expect("invalid input must panic");
    let message = error
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| error.downcast_ref::<&str>().copied())
        .expect("panic must have a message");
    assert!(message.contains(expected), "expected {expected:?}, got {message:?}");
}

#[test]
fn invalid_bounds_are_rejected_without_consuming_names_or_ids() {
    let m = Model::new("bounds");
    variable!(m, x);
    for (lower, upper) in [
        (f64::NAN, 1.0),
        (0.0, f64::NAN),
        (f64::INFINITY, 1.0),
        (0.0, f64::NEG_INFINITY),
        (f64::INFINITY, f64::INFINITY),
        (f64::NEG_INFINITY, f64::NEG_INFINITY),
    ] {
        assert_panics_with(
            || {
                constraint!(m, row, lower <= x <= upper);
            },
            "bound",
        );
        assert_eq!(m.num_constraints(), 0);
        assert!(m.constraint_handle("row").is_none());
    }
    assert_panics_with(|| constraint!(m, row, x <= f64::NEG_INFINITY), "bound");
    assert_panics_with(|| constraint!(m, row, x >= f64::INFINITY), "bound");
    assert_panics_with(|| constraint!(m, row, x == f64::INFINITY), "bound");
    assert_panics_with(|| constraint!(m, row, x == f64::NEG_INFINITY), "bound");
    let row = constraint!(m, row, 3.0 <= x <= 2.0);
    let RangeConstraintHandles::Interval(row) = row else {
        panic!("literal affine range must register one interval row");
    };
    assert_eq!(row.index(), 0);
    assert_panics_with(|| constraint!(m, row, x <= 1.0), "already registered");
    constraint!(m, upper_unbounded, x <= f64::INFINITY);
    constraint!(m, lower_unbounded, x >= f64::NEG_INFINITY);
    m.add_constraint(
        "free",
        Constraint::new(
            x.into_function(),
            Interval { lower: f64::NEG_INFINITY, upper: f64::INFINITY },
        ),
    );
    assert_eq!(m.num_constraints(), 4);
}

#[test]
fn foreign_scalar_and_soc_registration_panic_before_registering() {
    let m = Model::new("local");
    let other = Model::new("foreign");
    variable!(m, x);
    variable!(other, y);
    assert_panics_with(|| constraint!(m, row, y <= 1.0), "model");
    assert_panics_with(|| soc_constraint!(m, cone, [y] <= x), "same model");
    assert_panics_with(|| VectorAffineFunction::new([]), "nonempty");
    let vector = VectorAffineFunction::new([y.into_function(), y.into_function()]);
    assert_panics_with(
        || m.add_constraint("cone", Constraint::new(vector, SecondOrderCone { dimension: 2 })),
        "model",
    );
    assert_eq!(m.num_constraints(), 0);
    assert_eq!(other.num_constraints(), 0);
}

#[test]
fn soc_checks_dimensions_and_dynamic_degree_before_registering() {
    let m = Model::new("cones");
    variable!(m, x);
    variable!(m, t);
    let vector = VectorAffineFunction::new([t.into_function(), x.into_function()]);
    assert_panics_with(
        || {
            m.add_constraint(
                "cone",
                Constraint::new(vector.clone(), SecondOrderCone { dimension: 3 }),
            )
        },
        "dimension mismatch",
    );
    assert_panics_with(
        || m.add_constraint("cone", Constraint::new(vector, SecondOrderCone { dimension: 1 })),
        "at least two",
    );
    assert_panics_with(|| soc_constraint!(m, cone, [x.square().erase()] <= t), "affine");
    assert_panics_with(
        || m.add_soc_constraint("cone", std::iter::empty::<Expr<'_, Affine>>(), t),
        "at least two",
    );
    assert_eq!(m.num_soc_constraints(), 0);
    let cone = soc_constraint!(m, cone, [x.powi(1)] <= t);
    assert_eq!(cone.index(), 0);
    assert_panics_with(|| soc_constraint!(m, cone, [x] <= t), "already registered");
    assert_eq!(m.num_soc_constraints(), 1);
}

#[test]
fn affine_powers_extract_and_rebind_without_freezing_parameters() {
    let m = Model::new("affine powers");
    variable!(m, x);
    param!(m, p = 2.0);
    let expressions = [x.powi(0), x.powi(1), p.powi(3) * x.powi(1), x.sin().powi(0)];
    for parameter in [2.0, 3.0] {
        m.set_param(p, parameter).unwrap();
        for (expr, coefficient, constant) in [
            (expressions[0], 0.0, 1.0),
            (expressions[1], 1.0, 0.0),
            (expressions[2], parameter.powi(3), 0.0),
            (expressions[3], 0.0, 1.0),
        ] {
            let affine = expr.try_affine().unwrap();
            let arena = m.arena();
            let terms = extract_linear(&arena, affine.id()).unwrap();
            assert_eq!(terms.coeffs.iter().map(|(_, value)| value).sum::<f64>(), coefficient);
            assert_eq!(terms.constant, constant);
        }
    }
    for exponent in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(p.powf(exponent).try_affine().unwrap_err(), ExprClass::Nonlinear);
    }
    assert!(x.powi(2).try_affine().is_err());
}

#[test]
fn scalar_zero_division_is_rejected_before_emitting_nodes() {
    let m = Model::new("division");
    variable!(m, x);
    let quadratic = x.square();
    let before = m.arena().len();
    for divisor in [0.0, -0.0] {
        assert!(catch_unwind(AssertUnwindSafe(|| x / divisor)).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| quadratic / divisor)).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| x.erase() / divisor)).is_err());
    }
    assert!(catch_unwind(AssertUnwindSafe(|| x / 0)).is_err());
    assert_eq!(m.arena().len(), before);
    let e: Expr<'_, Affine> = x / 2.0;
    assert_eq!(e.__class(), ExprClass::Linear);
}

#[test]
fn indicator_inputs_are_checked_before_registration() {
    let m = Model::new("indicators");
    let other = Model::new("other");
    variable!(m, b, Binary);
    variable!(m, x);
    variable!(other, y);
    assert_panics_with(|| indicator_constraint!(m, row, x == 1 => x <= 1.0), "binary");
    assert_panics_with(|| indicator_constraint!(m, row, b == 1 => y <= 1.0), "model");
    assert_panics_with(|| indicator_constraint!(m, row, y == 1 => x <= 1.0), "model");
    assert_panics_with(
        || indicator_constraint!(m, row, b == 1 => x.square().erase() <= 1.0),
        "affine",
    );
    assert_panics_with(|| indicator_constraint!(m, row, b == 1 => x <= f64::NEG_INFINITY), "bound");
    assert_eq!(m.num_indicator_constraints(), 0);
    let checked = x.nonlinear().try_affine().unwrap();
    let row = indicator_constraint!(m, row, b == 1 => checked <= 1.0);
    assert_eq!(row.id().0, 0);
    assert_panics_with(|| indicator_constraint!(m, row, b == 1 => x <= 1.0), "already registered");
    assert_eq!(m.num_indicator_constraints(), 1);
}

#[test]
fn indexed_invalid_infinity_rolls_back_worker_nodes_and_rows() {
    rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap().install(|| {
        let m = Model::new("batch bounds");
        variable!(m, x[i in 0..2048]);
        let mut bounds = [1.0; 2048];
        bounds[1000] = f64::NEG_INFINITY;
        let before = m.arena().len();
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                constraint!(m, rows[i in 0..2048], x[i] + 1.0 <= bounds[i]);
            }))
            .is_err()
        );
        assert_eq!(m.num_constraints(), 0);
        assert_eq!(m.arena().len(), before);
    });
}
