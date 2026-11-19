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
fn hinted_and_unhinted_streams_match_bits_for_sparse_duplicate_and_cancelled_outputs() {
    let m = Model::new("conservative quadratic hints");
    variable!(m, x[i in 0..1024]);

    for (coefficient, weight, distinct, cancel) in [
        (1.0, 0.0, 1024, false),
        (1.0, -0.0, 1024, false),
        (0.0, 1.0, 1024, false),
        (1e-200, 1.0, 1024, false),
        (1e-160, 1.0, 1, false),
        (1.0, 1.0, 16, false),
        (1.0, 1.0, 113, true),
        (1.0, 1.0, 1024, true),
    ] {
        let left: Vec<_> = x.iter().map(|(_, x)| coefficient * x).collect();
        let right: Vec<_> = x.iter().map(|(_, x)| coefficient * x).collect();
        let weights: Vec<_> =
            (0..2048).map(|i| if cancel && i >= 1024 { -weight } else { weight }).collect();

        let hinted = one_node(&m, || {
            sum!(m,
            weights[i] * &left[i % distinct] * &right[i % distinct]
            for i in 0..2048)
        });
        let unhinted = one_node(&m, || {
            sum!(m,
            weights[i] * &left[i % distinct] * &right[i % distinct]
            for i in 0..2048 if true)
        });

        let bits = |q: QuadraticTerms| {
            (
                q.constant().to_bits(),
                q.linear().iter().map(|&(v, c)| (v, c.to_bits())).collect::<Vec<_>>(),
                q.quadratic().iter().map(|&(a, b, c)| (a, b, c.to_bits())).collect::<Vec<_>>(),
            )
        };
        assert_eq!(bits(terms(&hinted)), bits(terms(&unhinted)));
    }
}

#[test]
fn quadratic_hints_keep_temporary_and_dependent_domains_evaluated_once() {
    let m = Model::new("quadratic hint domains");
    variable!(m, x);
    let domains = RefCell::new(Vec::new());
    let visited = RefCell::new(Vec::new());
    let flat = sum!(m, {
        visited.borrow_mut().push((i, j));
        2.0 * x * x
    } for i in { domains.borrow_mut().push("outer"); vec![0, 1, 2, 3] }, j in 0..3);
    assert_eq!(*domains.borrow(), ["outer"]);
    assert_eq!(visited.borrow().len(), 12);
    assert_eq!(value(&flat, &[2.0]), 96.0);

    domains.borrow_mut().clear();
    visited.borrow_mut().clear();
    let dependent = sum!(m, {
        visited.borrow_mut().push((i, j));
        2.0 * x * x
    } for i in { domains.borrow_mut().push("outer"); 0..4 },
        j in { domains.borrow_mut().push("inner"); 0..i });
    assert_eq!(*domains.borrow(), ["outer", "inner", "inner", "inner", "inner"]);
    assert_eq!(*visited.borrow(), [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)]);
    assert_eq!(value(&dependent, &[2.0]), 48.0);
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

    let reversed = sum!(m, &right * &left * 2 for _i in 0..3);
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
fn shared_non_monomial_squares_retain_their_factors() {
    let m = Model::new("shared affine squares");
    variable!(m, x);
    variable!(m, y);

    let point = [100_000_001.0, -100_000_000.0];
    for (factor, equal) in [(x - 100_000_000.0, x - 100_000_000.0), (x + y, x + y)] {
        let reference = &factor * &factor;
        assert_eq!(value(&reference, &point), 1.0);

        let square = sum!(m, &factor * &factor for _i in 0..1);
        assert_eq!(value(&square, &point), 1.0);
        assert_eq!(value(&square.ln(), &point), 0.0);
        assert!(matches!(m.arena().get(square.id()), ExprNode::Mul(_)));

        let repeated = sum!(m, &factor * &factor for _i in 0..3);
        assert_eq!(value(&repeated, &point), 3.0);

        let unit = 0.0 * x + 1.0;
        let mixed = [unit.clone(), factor.clone(), unit];
        let surrounded = sum!(m, &mixed[i] * &mixed[i] for i in 0..3);
        assert_eq!(value(&surrounded, &point), 3.0);

        for weighted in [
            sum!(m, 2.0 * &factor * &factor for _i in 0..1),
            sum!(m, &factor * 2.0 * &factor for _i in 0..1),
            sum!(m, &factor * &factor * 2.0 for _i in 0..1),
            sum!(m, 2.0 * (&factor * &factor) for _i in 0..1),
            sum!(m, &factor * (2.0 * &factor) for _i in 0..1),
        ] {
            assert_eq!(value(&weighted, &point), 2.0);
        }

        let equivalent = sum!(m, &factor * &equal for _i in 0..1);
        assert_eq!(value(&equivalent, &point), 1.0);
    }

    let simple = one_node(&m, || sum!(m, x * x for _i in 0..3));
    assert_eq!(value(&simple, &[3.0]), 27.0);

    let scaled = 2.0 * x;
    let monomial = one_node(&m, || sum!(m, &scaled * &scaled for _i in 0..3));
    assert_eq!(value(&monomial, &[3.0]), 108.0);
}

#[test]
fn independently_built_add_factors_with_identical_ordered_terms_stay_factored() {
    let m = Model::new("independent compound factors");
    variable!(m, x[i in 0..2]);
    let left = sum!(m, x[i] for i in 0..2);
    let right = sum!(m, x[i] for i in 0..2);
    assert_ne!(left.id(), right.id());
    assert!(matches!(m.arena().get(left.id()), ExprNode::Add(_)));
    assert!(matches!(m.arena().get(right.id()), ExprNode::Add(_)));

    let point = [100_000_001.0, -100_000_000.0];
    let ordinary = &left * &right;
    assert_eq!(value(&ordinary, &point), 1.0);

    let fused = sum!(m, &left * &right for _i in 0..1);
    assert_eq!(value(&fused, &point), 1.0);
    assert!(matches!(m.arena().get(fused.id()), ExprNode::Mul(_)));
    assert_eq!(value(&fused.ln(), &point), 0.0);

    let weighted = sum!(m, 2.0 * &left * &right for _i in 0..1);
    assert_eq!(value(&weighted, &point), 2.0);

    let compact = right.compact();
    let mixed = sum!(m, 2.0 * &left * &compact for _i in 0..1);
    assert_eq!(value(&mixed, &point), 2.0);

    let reversed = sum!(m, x[1-i] for i in 0..2);
    let reordered = one_node(&m, || sum!(m, &left * &reversed for _i in 0..1));
    assert!(matches!(m.arena().get(reordered.id()), ExprNode::Quadratic(_)));
    assert_eq!(value(&reordered, &[2.0, 3.0]), 25.0);

    let monomial_left = sum!(m, x[0] for _i in 0..2);
    let monomial_right = sum!(m, x[0] for _i in 0..2);
    let monomial = one_node(&m, || sum!(m, &monomial_left * &monomial_right for _i in 0..1));
    assert_eq!(value(&monomial, &[3.0]), 36.0);
}

#[test]
fn square_checks_ignore_zero_coefficients_without_reordering_factors() {
    let m = Model::new("zero coefficients in square factors");
    variable!(m, x[i in 0..4]);
    let indices = [2, 0, 3, 1, 2, 3];
    let weights = [1.0, 1.0, 1.0, 1.0, -1.0, -1.0];
    let left = sum!(m, unfused: weights[i] * x[indices[i]] for i in 0..indices.len());
    let right = left.compact();
    assert!(matches!(m.arena().get(left.id()), ExprNode::Add(_)));

    let point = [100_000_001.0, -100_000_000.0, 2.0, 3.0];
    let ordinary = sum!(m, unfused: &left * &right for _i in 0..1);
    assert_eq!(value(&ordinary, &point), 1.0);

    for (a, b) in [(&left, &right), (&right, &left)] {
        let fused = sum!(m, a * b for _i in 0..1);
        assert_eq!(value(&fused, &point), 1.0);
        assert!(matches!(m.arena().get(fused.id()), ExprNode::Mul(_)));
        assert_eq!(value(&fused.ln(), &point), 0.0);
        let weighted = sum!(m, 2.0 * a * b for _i in 0..1);
        assert_eq!(value(&weighted, &point), 2.0);
    }

    let reversed = sum!(m, x[1-i] for i in 0..2);
    let reordered = one_node(&m, || sum!(m, &left * &reversed for _i in 0..1));
    assert!(matches!(m.arena().get(reordered.id()), ExprNode::Quadratic(_)));
    assert_eq!(value(&reordered, &[2.0, 3.0, 4.0, 5.0]), 25.0);

    let indices = [1, 0, 1];
    let weights = [1.0, 1.0, -1.0];
    let monomial_left = sum!(m, unfused: weights[i] * x[indices[i]] for i in 0..3);
    let monomial_right = monomial_left.compact();
    let monomial = one_node(&m, || sum!(m, &monomial_left * &monomial_right for _i in 0..1));
    assert!(matches!(m.arena().get(monomial.id()), ExprNode::Quadratic(_)));
    assert_eq!(value(&monomial, &[3.0; 4]), 9.0);
}

#[test]
fn matched_squares_evaluate_their_normalized_factors() {
    let m = Model::new("normalized square factors");
    variable!(m, x[i in 0..2]);
    let indices = [0, 0, 0, 1];
    let weights = [f64::MAX, -f64::MAX, 1.0, 1.0];
    let left = sum!(m, unfused: weights[i] * x[indices[i]] for i in 0..4);
    let equal = sum!(m, unfused: weights[i] * x[indices[i]] for i in 0..4);
    let compact = left.compact();
    let point = [2.0, 3.0];
    assert!(value(&left, &point).is_nan(), "the original Add root overflows");
    assert_eq!(value(&compact, &point), 5.0);

    for (a, b) in [(&left, &left), (&left, &equal), (&left, &compact), (&compact, &left)] {
        let ordinary = sum!(m, unfused: a * b for _i in 0..1);
        assert_eq!(value(&ordinary, &point), 25.0);
        let fused = sum!(m, a * b for _i in 0..1);
        assert_eq!(value(&fused, &point), 25.0);
        assert!(matches!(m.arena().get(fused.id()), ExprNode::Mul(_)));

        for weighted in [
            sum!(m, 2.0 * a * b for _i in 0..1),
            sum!(m, a * (2.0 * b) for _i in 0..1),
            sum!(m, a * b * 2.0 for _i in 0..1),
            sum!(m, 2.0 * (a * b) for _i in 0..1),
        ] {
            assert_eq!(value(&weighted, &point), 50.0);
        }
    }

    let reversed = sum!(m, x[1-i] for i in 0..2);
    let unmatched = one_node(&m, || sum!(m, &left * &reversed for _i in 0..1));
    assert!(matches!(m.arena().get(unmatched.id()), ExprNode::Quadratic(_)));
    assert_eq!(value(&unmatched, &point), 25.0);
}

#[test]
fn normalized_squares_preserve_weight_placement() {
    let m = Model::new("normalized square weight placement");
    variable!(m, x[i in 0..2]);
    let weights = [f64::MAX, -f64::MAX];
    let factor = sum!(m, unfused: weights[i] * x[i] for i in 0..2);
    let point = [1.0, 1.0];

    for weighted_factor in [
        sum!(m, 2.0 * &factor * &factor for _i in 0..1),
        sum!(m, &factor * (2.0 * &factor) for _i in 0..1),
    ] {
        assert!(value(&weighted_factor, &point).is_nan(), "scaling the factor overflows");
    }
    for weighted_product in [
        sum!(m, 2.0 * (&factor * &factor) for _i in 0..1),
        sum!(m, &factor * &factor * 2.0 for _i in 0..1),
    ] {
        assert_eq!(value(&weighted_product, &point), 0.0);
    }
}

#[test]
fn oversized_weighted_factors_keep_ordinary_scalar_overflow_and_underflow() {
    for (coefficients, weight, point) in [
        (
            [f64::MAX, -f64::MAX, 1.0, 1.0, 1.0, 1.0],
            2.0,
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ),
        ([f64::from_bits(1); 6], 0.5, [2.0; 12]),
    ] {
        let m = Model::new("oversized scalar multiplication order");
        variable!(m, x[i in 0..12]);
        let factor = sum!(m, unfused: coefficients[i] * x[i] for i in 0..6);
        let other = sum!(m, x[i] for i in 6..12);
        for (actual, reference) in [
            (
                sum!(m, weight * &factor * &other for _i in 0..1),
                sum!(m, unfused: weight * &factor * &other for _i in 0..1),
            ),
            (
                sum!(m, &factor * weight * &other for _i in 0..1),
                sum!(m, unfused: &factor * weight * &other for _i in 0..1),
            ),
            (
                sum!(m, &other * (weight * &factor) for _i in 0..1),
                sum!(m, unfused: &other * (weight * &factor) for _i in 0..1),
            ),
            (
                sum!(m, &other * (&factor * weight) for _i in 0..1),
                sum!(m, unfused: &other * (&factor * weight) for _i in 0..1),
            ),
        ] {
            let expected = value(&reference, &point);
            let actual_value = value(&actual, &point);
            if weight > 1.0 {
                assert!(expected.is_nan());
            } else {
                assert_eq!(expected, 0.0);
            }
            assert!(
                actual_value.to_bits() == expected.to_bits()
                    || actual_value.is_nan() && expected.is_nan(),
                "expected {expected}, got {actual_value}"
            );
            assert!(matches!(m.arena().get(actual.id()), ExprNode::Mul(_)));
        }
    }
}

#[test]
fn oversized_products_reuse_normalized_unweighted_affine_factors() {
    for mode in 0..3 {
        let m = Model::new("oversized shared factors");
        variable!(m, x[i in 0..13]);
        let (left, right, normalized_roots, expected) = match mode {
            0 => (sum!(m, x[i] for i in 0..8), sum!(m, x[i] for i in 8..13), 2, 6080.0),
            1 => (sum!(m, 1.0 * x[i] for i in 0..8), sum!(m, 1.0 * x[i] for i in 8..13), 0, 6080.0),
            _ => {
                let factor = sum!(m, x[i] for i in 0..6);
                (factor.clone(), factor, 1, 5472.0)
            }
        };
        let weights: Vec<_> = (2..18).map(f64::from).collect();
        let linear_nodes = || {
            m.arena().nodes().iter().filter(|node| matches!(node, ExprNode::Linear { .. })).count()
        };

        for spelling in 0..7 {
            let before = linear_nodes();
            let expression = match spelling {
                0 => sum!(m, weights[i] * &left * &right for i in 0..weights.len()),
                1 => sum!(m, &left * weights[i] * &right for i in 0..weights.len()),
                2 => sum!(m, &left * &right * weights[i] for i in 0..weights.len()),
                3 => sum!(m, weights[i] * (&left * &right) for i in 0..weights.len()),
                4 => sum!(m, &left * (weights[i] * &right) for i in 0..weights.len()),
                5 => sum!(weights[i] * &left * &right for i in 0..weights.len()),
                _ => sum!(m, &left * &right for _i in 0..weights.len()),
            };
            let new_linear_nodes = if matches!(spelling, 0 | 1 | 4 | 5) {
                weights.len() + usize::from(mode != 1)
            } else {
                normalized_roots
            };
            assert_eq!(linear_nodes(), before + new_linear_nodes);

            let expected = if spelling == 6 { expected / 152.0 * 16.0 } else { expected };
            assert_eq!(value(&expression, &[1.0; 13]), expected);
            assert!(!m.arena().nodes().iter().any(|node| matches!(node, ExprNode::Quadratic(_))));
        }
    }
}

#[test]
fn oversized_nonfinite_weights_keep_the_existing_symbolic_fallback() {
    let m = Model::new("oversized nonfinite fallback");
    variable!(m, x[i in 0..13]);
    let left = sum!(m, x[i] for i in 0..8);
    let right = sum!(m, x[i] for i in 8..13);

    for weight in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        let ordinary = sum!(m, unfused: weight * &left * &right for _i in 0..2);
        let fused = sum!(m, weight * &left * &right for _i in 0..2);
        let expected = value(&ordinary, &[1.0; 13]);
        let actual = value(&fused, &[1.0; 13]);
        assert!(actual.to_bits() == expected.to_bits() || actual.is_nan() && expected.is_nan());
    }
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
fn right_weighted_products_preserve_coefficient_overflow_and_underflow() {
    let m = Model::new("right-weighted coefficient order");
    variable!(m, x);
    variable!(m, y);

    for (a, b, weight, expected) in
        [(0.5, f64::MAX, 2.0, f64::INFINITY), (f64::MAX, f64::from_bits(1), 0.5, 0.0)]
    {
        let left = a * x;
        let right = b * y;
        let ordinary = sum!(m, unfused: &left * (weight * &right) for _i in 0..1);
        assert_eq!(value(&ordinary, &[1.0, 1.0]), expected);

        let fused = sum!(m, &left * (weight * &right) for _i in 0..1);
        assert_eq!(value(&fused, &[1.0, 1.0]), expected);
    }

    let left = 0.5 * x;
    let right = f64::MAX * y;
    let left_weighted = one_node(&m, || sum!(m, 2.0 * &left * &right for _i in 0..1));
    assert_eq!(value(&left_weighted, &[1.0, 1.0]), f64::MAX);
}

#[test]
fn outer_weights_do_not_mask_inner_coefficient_overflow_or_underflow() {
    let m = Model::new("outer-weighted coefficient order");
    variable!(m, x);
    variable!(m, y);

    for (a, b, weight, expected) in
        [(f64::MAX, 2.0, 0.5, f64::INFINITY), (f64::from_bits(1), 0.5, f64::MAX, 0.0)]
    {
        let left = a * x;
        let right = b * y;
        let ordinary = sum!(m, unfused: weight * (&left * &right) for _i in 0..1);
        assert_eq!(value(&ordinary, &[1.0, 1.0]), expected);

        let prefix = sum!(m, weight * (&left * &right) for _i in 0..1);
        let suffix = sum!(m, &left * &right * weight for _i in 0..1);
        assert_eq!(value(&prefix, &[1.0, 1.0]), expected);
        assert_eq!(value(&suffix, &[1.0, 1.0]), expected);
    }
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
