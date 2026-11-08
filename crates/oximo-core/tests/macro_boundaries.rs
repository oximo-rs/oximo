use oximo_core::prelude::*;
use oximo_expr::{evaluate, extract_linear};

#[test]
fn borrowed_sums_match_with_and_without_filters_and_anchors() {
    let m = Model::new("borrowed sum inputs");
    variable!(m, x);
    let terms = [x + 1.0, x + 2.0];
    let sums = [
        sum!(m, &terms[i] for i in 0..2),
        sum!(m, &terms[i] for i in 0..2 if true),
        sum!(&terms[i] for i in 0..2),
        sum!(&terms[i] for i in 0..2 if true),
    ];
    for expression in sums {
        let arena = m.arena();
        let linear = extract_linear(&arena, expression.id()).unwrap();
        assert_eq!(linear.coeffs.as_ref(), [(x.var_id().unwrap(), 2.0)]);
        assert_eq!(linear.constant, 3.0);
    }

    for expression in [sum!(m, &terms[i] for i in 0..0), sum!(m, &terms[i] for i in 0..2 if false)]
    {
        assert_eq!(evaluate(&m.arena(), expression.id(), &&[2.0][..]).unwrap(), 0.0);
    }
    assert_eq!(evaluate(&m.arena(), terms[0].id(), &&[2.0][..]).unwrap(), 3.0);
    assert_eq!(evaluate(&m.arena(), terms[1].id(), &&[2.0][..]).unwrap(), 4.0);

    let single = sum!(m, &terms[i] for i in 0..1);
    assert_eq!(single.id(), terms[0].id());
}

#[test]
fn ordinary_rust_arithmetic_preserves_consumption_and_side_effects() {
    struct Owned(f64);
    impl std::ops::Add for Owned {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            Self(self.0 + rhs.0)
        }
    }

    let m = Model::new("ordinary Rust values");
    variable!(m, x);
    let mut visits = 0;
    objective!(m, Min, {
        let prefix = String::from("a");
        let name = prefix + "b";
        let left = Owned(2.0);
        let right = Owned(3.0);
        let total = {
            visits += 1;
            left + right
        };
        (f64::from(u32::try_from(name.len()).unwrap()) + total.0) * x
    });
    assert_eq!(visits, 1);

    let root = m.objective().as_ref().unwrap().expr;
    assert_eq!(evaluate(&m.arena(), root, &&[2.0][..]).unwrap(), 14.0);

    struct MoveOnly<'a>(Expr<'a, Affine>);
    impl<'a> oximo_core::objective::IntoObjectiveExpression<'a> for MoveOnly<'a> {
        type Degree = Affine;
        fn into_objective_expression(self, _: &'a oximo_expr::ExprArenaCell) -> Expr<'a, Affine> {
            self.0
        }
    }

    objective!(m, Min, MoveOnly(x + 1.0));
    let root = m.objective().as_ref().unwrap().expr;
    assert_eq!(evaluate(&m.arena(), root, &&[2.0][..]).unwrap(), 3.0);
}

#[test]
fn computed_expressions_follow_normal_ownership_inside_blocks() {
    let m = Model::new("explicit expression ownership");
    variable!(m, x);
    variable!(m, y);
    objective!(m, Min, {
        let shared = x + y;
        let left = &shared + x;
        let right = shared + y;
        left + right
    });
    let root = m.objective().as_ref().unwrap().expr;
    assert_eq!(evaluate(&m.arena(), root, &&[2.0, 3.0][..]).unwrap(), 15.0);

    let before = m.__sum_context().__node_count();
    objective!(m, Min, {
        let quadratic = x.square();
        quadratic + y
    });
    assert_eq!(
        m.__sum_context().__node_count(),
        before + 1,
        "consuming the fresh quadratic must reuse its root"
    );
}

#[test]
fn explicit_borrows_work_for_scalar_and_indexed_ranges_and_indicators() {
    let m = Model::new("borrowed macro inputs");
    variable!(m, x);
    variable!(m, b, Bin);
    let e = x + 1.0;
    let trigger = b.as_dynamic();
    constraint!(m, interval, 0.0 <= &e <= 5.0);
    constraint!(m, intervals[i in 0..512], 0.0 <= &e <= f64::from(u32::try_from(i).unwrap()) + 5.0);
    indicator_constraint!(m, single, &trigger == 1 => &e <= 4.0);
    indicator_constraint!(m, rows[i in 0..2], &trigger == 1 => &e <= f64::from(u32::try_from(i).unwrap()) + 4.0);
    indicator_constraint!(m, ranges[i in 0..2], &trigger == 1 => 0.0 <= &e <= f64::from(u32::try_from(i).unwrap()) + 4.0);
    objective!(m, Min, e);
    assert_eq!(m.num_constraints(), 518);
}
