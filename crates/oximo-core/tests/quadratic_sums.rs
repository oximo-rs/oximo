#![expect(clippy::float_cmp, reason = "exact polynomial coefficients and IEEE regressions")]

use std::cell::RefCell;
use std::ops::Mul;
use std::panic::{AssertUnwindSafe, catch_unwind};

use oximo_core::prelude::*;
use oximo_expr::{Degree, ExprNode, QuadraticTerms, evaluate, extract_quadratic};

fn terms<D: Degree>(expression: &Expr<'_, D>) -> QuadraticTerms {
    extract_quadratic(&expression.arena().borrow(), expression.id()).unwrap()
}

fn value<D: Degree>(expression: &Expr<'_, D>, point: &[f64]) -> f64 {
    evaluate(&expression.arena().borrow(), expression.id(), &point).unwrap()
}

fn one_node<'a, D: Degree>(m: &Model, build: impl FnOnce() -> Expr<'a, D>) -> Expr<'a, D> {
    let before = m.arena().len();
    let result = build();
    assert_eq!(m.arena().len(), before + 1);

    result
}

fn same_coefficients(left: &QuadraticTerms, right: &QuadraticTerms) {
    assert_eq!(left.constant(), right.constant());
    assert_eq!(left.linear(), right.linear());
    assert_eq!(left.quadratic(), right.quadratic());
}

#[test]
fn common_spellings_emit_one_node_and_match_the_builder() {
    for count in [1_usize, 8, 1024] {
        let m = Model::new("monomial fusion");
        variable!(m, x[i in 0..8]);
        let mut builder = m.quadratic_builder();
        for i in 0..count {
            builder.add_monomial(2.0, x[i % 8], x[(i + 1) % 8]).unwrap();
        }
        let expected = builder.build_terms();

        let expressions: [Expr<'_, Quadratic>; 8] = [
            one_node(&m, || sum!(m, 2.0 * x[i % 8] * x[(i+1) % 8] for i in 0..count)),
            one_node(&m, || sum!(m, x[i % 8] * 2 * x[(i+1) % 8] for i in 0..count)),
            one_node(&m, || sum!(m, x[i % 8] * x[(i+1) % 8] * 2.0 for i in 0..count)),
            one_node(&m, || sum!(m, 2 * (x[i % 8] * x[(i+1) % 8]) for i in 0..count)),
            one_node(&m, || sum!(m, (2.0 * &x[i % 8]) * &x[(i+1) % 8] for i in 0..count)),
            one_node(&m, || sum!(m, &x[i % 8] * (2.0 * &x[(i+1) % 8]) for i in 0..count)),
            one_node(&m, || sum!(2 * x[i % 8] * x[(i+1) % 8] for i in 0..count)),
            one_node(&m, || sum!(m, 2 * x[i % 8] * x[(i+1) % 8] for i in 0..count if true)),
        ];

        for expression in expressions {
            assert!(matches!(m.arena().get(expression.id()), ExprNode::Quadratic(_)));

            same_coefficients(&terms(&expression), &expected);
            assert_eq!(
                value(&expression, &[1.0; 8]),
                2.0 * f64::from(u32::try_from(count).unwrap())
            );
        }
    }
}

#[test]
fn affine_offsets_borrows_and_consuming_inputs_expand_without_product_nodes() {
    let m = Model::new("affine factors");
    variable!(m, x);
    variable!(m, y);
    let left = x + 3.0;
    let right = 2.0 * y - 1.0;
    let mut builder = m.quadratic_builder();
    builder.add_product(6.0, &left, &right).unwrap();
    let expected = builder.build_terms();

    let fused = one_node(&m, || sum!(m, 2.0 * &left * &right for _i in 0..3));
    same_coefficients(&terms(&fused), &expected);

    let reversed = one_node(&m, || sum!(m, &right * &left * 2 for _i in 0..3));
    same_coefficients(&terms(&reversed), &expected);

    let temporary = sum!(m,
        2.0 * std::convert::identity(&(x + 3.0)) * std::convert::identity(&(2.0 * y - 1.0))
        for _i in 0..3);
    same_coefficients(&terms(&temporary), &expected);
    let mut owned = Some(left);
    let moved = sum!(m, 6.0 * owned.take().unwrap() * &right for _i in 0..1);
    same_coefficients(&terms(&moved), &expected);
    assert!(owned.is_none());
}

#[test]
fn pair_orientation_diagonals_and_cancellation_match_numeric_builder() {
    let m = Model::new("pairs");
    variable!(m, x[i in 0..3]);
    let data = [(2.0, 0, 1), (-2.0, 1, 0), (3.0, 2, 2), (0.0, 0, 0)];
    let fused = one_node(&m, || sum!(m, data[i].0 * x[data[i].1] * x[data[i].2] for i in 0..4));
    assert_eq!(terms(&fused).quadratic(), &[(oximo_expr::VarId(2), oximo_expr::VarId(2), 3.0)]);

    let plain = one_node(&m, || sum!(m, x[i] * x[i] for i in 0..3));
    assert_eq!(value(&plain, &[1.0, 2.0, 3.0]), 14.0);
}

#[test]
fn domains_empty_sums_and_enclosing_macros_keep_their_behavior() {
    let m = Model::new("domains");
    variable!(m, x[i in 0..4]);
    let flat = one_node(&m, || sum!(m, 2.0 * x[i] * x[j] for i in 0..4, j in 0..4));
    assert_eq!(value(&flat, &[1.0; 4]), 32.0);

    let filtered = one_node(&m, || sum!(m, x[i] * x[j] for i in 0..4, j in 0..i if j % 2 == 0));
    assert_eq!(value(&filtered, &[1.0; 4]), 4.0);

    let empty: Expr<'_, Quadratic> = one_node(&m, || sum!(m, 2 * x[i] * x[i] for i in 0..0));
    assert_eq!(value(&empty, &[]), 0.0);

    let filtered = sum!(m, x[i] * x[i] for i in 0..4 if false);
    assert_eq!(value(&filtered, &[]), 0.0);
    assert!(catch_unwind(AssertUnwindSafe(|| sum!(x[i] * x[i] for i in 0..0))).is_err());

    let nested = sum!(m, sum!(x[i] * x[j] for j in 0..i) for i in 0..4);
    assert_eq!(value(&nested, &[1.0; 4]), 6.0);

    objective!(m, Min, sum!(2 * x[i] * x[i] for i in 0..4));
    constraint!(m, row, sum!(x[i] * x[i] for i in 0..4) <= 8.0);
}

#[test]
fn static_degrees_and_parameters_remain_live_across_numeric_segments() {
    let m = Model::new("symbolic products");
    variable!(m, x);
    param!(m, p = 2.0);
    let constant: Expr<'_, Constant> = sum!(m, 2.0 * p * p for _i in 0..2);
    let affine: Expr<'_, Affine> = sum!(m, 2.0 * p * x for _i in 0..2);
    let factors = [x + 1.0, p * x + 1.0, x + 2.0];
    let quadratic: Expr<'_, Quadratic> = sum!(m, 2.0 * &factors[i] * x for i in 0..3);
    let zero_weight = sum!(m, 0.0 * p * x for _i in 0..2);
    for parameter in [2.0, -4.0] {
        p.set_param_value(parameter).unwrap();
        assert_eq!(value(&constant, &[]), 4.0 * parameter * parameter);
        assert_eq!(value(&affine, &[3.0]), 12.0 * parameter);
        assert_eq!(value(&quadratic, &[3.0]), 18.0 * (2.0 + parameter) + 24.0);
        assert_eq!(value(&zero_weight, &[3.0]), 0.0);
    }

    // Syntactic param under p^0 must allow numeric extraction.
    let power = p.powi(0).into_affine().unwrap();
    let numeric = one_node(&m, || sum!(m, 2.0 * &power * x for _i in 0..2));
    assert_eq!(value(&numeric, &[3.0]), 12.0);
}

#[test]
fn numeric_rejection_falls_back_without_panics_or_losing_prefixes() {
    let m = Model::new("numeric fallback");
    variable!(m, x);

    for weights in [
        [1.0, f64::INFINITY, -2.0],
        [f64::INFINITY, f64::NEG_INFINITY, 1.0],
        [1.0, f64::NAN, 2.0],
        [f64::MAX, f64::MAX, -f64::MAX],
        [f64::MIN_POSITIVE, f64::from_bits(1), -f64::MIN_POSITIVE],
    ] {
        let fused = sum!(m, weights[i] * x * x for i in 0..3);
        let ordinary = sum!(m, unfused: weights[i] * x * x for i in 0..3);
        for point in [0.0, 1.0, 2.0] {
            let actual = value(&fused, &[point]);
            let expected = value(&ordinary, &[point]);
            assert!(actual.to_bits() == expected.to_bits() || actual.is_nan() && expected.is_nan());
        }
    }

    let large = 1e308 * x;
    let fused = sum!(m, 2.0 * &large * x for _i in 0..2);
    let ordinary = sum!(m, unfused: 2.0 * &large * x for _i in 0..2);
    assert_eq!(value(&fused, &[1.0]), value(&ordinary, &[1.0]));
}

#[test]
fn foreign_factors_finish_callbacks_without_emitting_collector_nodes() {
    let m = Model::new("local");
    let other = Model::new("foreign");
    variable!(m, x);
    variable!(other, y);
    param!(m, p = 2.0);
    let symbolic = p * x;

    for foreign_left in [false, true] {
        for filtered in [false, true] {
            let before = (m.arena().len(), other.arena().len());
            let mut visited = Vec::new();

            let result = catch_unwind(AssertUnwindSafe(|| {
                if filtered {
                    sum!(m, (if foreign_left && i == 2 { y } else { x })
                        * { visited.push(i); if !foreign_left && i == 2 { y } else { x } }
                        for i in 0..4 if true)
                } else {
                    sum!(m, (if foreign_left && i == 2 { y } else { x })
                        * { visited.push(i); if !foreign_left && i == 2 { y } else { x } }
                        for i in 0..4)
                }
            }));
            assert!(result.is_err());
            assert_eq!(visited, [0, 1, 2, 3]);
            assert_eq!((m.arena().len(), other.arena().len()), before);
        }
    }

    let numeric = x + 0.0;
    let before = m.arena().len();
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            sum!(m, (if i == 1 { &symbolic } else { &numeric })
            * (if i == 2 { y } else { x }) for i in 0..3)
        }))
        .is_err()
    );
    assert_eq!(m.arena().len(), before);
    assert!(catch_unwind(AssertUnwindSafe(|| sum!(m, 0.0 * x * y for _i in 0..1))).is_err());
}

#[test]
fn custom_inner_multiplication_runs_before_the_third_operand_and_guard_drop() {
    struct Guard<'a>(&'a RefCell<Vec<&'static str>>);
    struct Weight<'a>(&'a RefCell<Vec<&'static str>>);
    impl Guard<'_> {
        fn weight(&self) -> Weight<'_> {
            self.0.borrow_mut().push("weight");
            Weight(self.0)
        }
    }
    impl Drop for Guard<'_> {
        fn drop(&mut self) {
            self.0.borrow_mut().push("drop");
        }
    }
    impl<'a> Mul<VariableHandle<'a>> for Weight<'_> {
        type Output = Expr<'a, Affine>;
        fn mul(self, x: VariableHandle<'a>) -> Self::Output {
            self.0.borrow_mut().push("mul");
            2.0 * x
        }
    }

    let m = Model::new("custom stages");
    variable!(m, x);
    let events = RefCell::new(Vec::new());
    let fused = sum!(m, Guard(&events).weight() * x * { events.borrow_mut().push("third"); x }
        for _i in 0..2);
    assert_eq!(*events.borrow(), ["weight", "mul", "third", "drop"].repeat(2));
    assert_eq!(value(&fused, &[3.0]), 36.0);
}

#[test]
fn deferred_left_products_materialize_before_custom_rhs_acquires_an_arena_guard() {
    struct Rhs<'a>(oximo_expr::ExprArenaWriteGuard<'a>);

    impl<'a, D: oximo_expr::ExpressionDegree> Mul<Rhs<'a>> for Expr<'a, D> {
        type Output = Self;
        fn mul(self, rhs: Rhs<'a>) -> Self {
            drop(rhs.0);
            self
        }
    }

    let m = Model::new("materialization before RHS");
    variable!(m, x);
    let rhs = || Rhs(m.__sum_context().borrow_mut());
    let reference = sum!(m, unfused: 2.0 * x * rhs() for _i in 0..2);
    assert_eq!(value(&reference, &[3.0]), 12.0);

    let affine = sum!(m, 2.0 * x * rhs() for _i in 0..2);
    assert_eq!(value(&affine, &[3.0]), 12.0);

    let quadratic = sum!(m, x * x * rhs() for _i in 0..2);
    assert_eq!(value(&quadratic, &[3.0]), 18.0);

    let filtered = sum!(m, 2.0 * x * rhs() for _i in 0..2 if true);
    assert_eq!(value(&filtered, &[3.0]), 12.0);

    let flat = sum!(m, x * x * rhs() for _i in 0..2, _j in 0..2);
    assert_eq!(value(&flat, &[3.0]), 36.0);

    struct BorrowedRhs<'a>(RefCell<Option<oximo_expr::ExprArenaWriteGuard<'a>>>);
    impl<'a, D: oximo_expr::ExpressionDegree> Mul<&BorrowedRhs<'a>> for Expr<'a, D> {
        type Output = Self;
        fn mul(self, rhs: &BorrowedRhs<'a>) -> Self {
            drop(rhs.0.borrow_mut().take());
            self
        }
    }
    let borrowed_rhs = || BorrowedRhs(RefCell::new(Some(m.__sum_context().borrow_mut())));
    let temporary = sum!(m, 2.0 * x * std::convert::identity(&borrowed_rhs()) for _i in 0..2);
    assert_eq!(value(&temporary, &[3.0]), 12.0);

    fn generic<'a, R>(
        m: &'a Model,
        x: VariableHandle<'a>,
        mut rhs: impl FnMut() -> R,
    ) -> Expr<'a, Affine>
    where
        Expr<'a, Affine>: Mul<R, Output = Expr<'a, Affine>>,
    {
        sum!(m, 2.0 * x * rhs() for _i in 0..2)
    }
    assert_eq!(value(&generic(&m, x, rhs), &[3.0]), 12.0);
}

#[test]
fn filtered_product_control_flow_keeps_its_original_loop_scope() {
    let m = Model::new("product break/continue");
    variable!(m, x);
    let mut visited = Vec::new();
    let total = sum!(m, 2.0 * x * {
        visited.push(i);
        if i == 1 { continue; }
        if i == 3 { break; }
        x
    } for i in 0..8 if true);
    assert_eq!(visited, [0, 1, 2, 3]);
    assert_eq!(value(&total, &[3.0]), 36.0);
}

#[test]
fn nonlinear_dynamic_long_chains_and_unfused_keep_ordinary_results() {
    let m = Model::new("fallback");
    variable!(m, x);
    let nonlinear: Expr<'_, Nonlinear> = sum!(m, 2.0 * x * x.sin() for _i in 0..2);
    let cubic: Expr<'_, Nonlinear> = sum!(m, x * x * x for _i in 0..2);
    let dynamic: Expr<'_, Dynamic> = sum!(m, 2.0 * x.as_dynamic() * x for _i in 0..2);
    let long = sum!(m, 2.0 * x * x * 3.0 for _i in 0..2);
    assert_eq!(value(&nonlinear, &[0.0]), 0.0);
    assert_eq!(value(&cubic, &[3.0]), 54.0);
    assert_eq!(value(&dynamic, &[3.0]), 36.0);
    assert_eq!(value(&long, &[3.0]), 108.0);

    let before = m.arena().len();
    let unfused = sum!(m, unfused: 2.0 * x * x for _i in 0..8);
    assert!(m.arena().len() > before + 1);
    assert_eq!(value(&unfused, &[3.0]), 144.0);
}

#[test]
fn two_weights_keep_the_existing_affine_accumulation_path() {
    let m = Model::new("two weights");
    variable!(m, x);
    param!(m, p = 2.0);
    let before = m.arena().len();
    let first: Expr<'_, Affine> = sum!(m, 2.0 * x * 3 for _i in 0..4);
    let first_nodes = m.arena().len() - before;
    let before = m.arena().len();
    let reference = sum!(m, 3 * (2.0 * x) for _i in 0..4);
    assert_eq!(m.arena().len() - before, first_nodes);
    assert_eq!(value(&first, &[3.0]), 72.0);
    assert_eq!(value(&reference, &[3.0]), 72.0);

    let constant: Expr<'_, Constant> = sum!(m, 2 * p * 3 for _i in 0..2);
    p.set_param_value(-4.0).unwrap();
    assert_eq!(value(&constant, &[]), -48.0);
}

#[test]
fn quadratic_sums_work_in_serial_and_parallel_forks() {
    for threads in [1, 4] {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| {
            let m = Model::new("forks");
            variable!(m, x[j in 0..4]);
            constraint!(m, rows[i in 0..1024], sum!(2.0 * x[j] * x[j] for j in 0..i % 5) <= 10.0);
            let arena = m.arena();
            let rows = m.constraints();
            for (i, row) in rows.algebraic().iter().enumerate() {
                let coefficients = extract_quadratic(&arena, row.lhs).unwrap();
                assert_eq!(coefficients.quadratic().len(), i % 5);
                assert!(coefficients.quadratic().iter().all(|&(a, b, c)| a == b && c == 2.0));
            }
        });
    }
}
