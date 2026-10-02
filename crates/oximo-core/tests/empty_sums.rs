use std::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind};

use oximo_core::prelude::*;
use oximo_expr::{evaluate, extract_linear};

fn value<D: oximo_expr::Degree>(expr: Expr<'_, D>) -> f64 {
    evaluate(&expr.arena.borrow(), expr.id, &&[1.0, 2.0, 3.0][..]).unwrap()
}

#[test]
fn empty_domains_and_filters_do_not_evaluate_the_body() {
    fn forbidden<'a>() -> Expr<'a> {
        panic!("empty sum body must not run");
    }
    let m = Model::new("empty");
    let empty: &[usize] = &[];
    let terms = [
        sum!(m, forbidden() for _i in 0..0),
        sum!(&m, forbidden() for _i in empty),
        sum!(m, forbidden() for i in 0..3 if i > 10),
    ];
    for expr in terms {
        assert_eq!(value(expr), 0.0);
        assert!(std::ptr::eq(expr.arena, m.__sum_context()));
    }
}

#[test]
fn anchored_sums_reject_foreign_terms_before_emitting_nodes() {
    let m = Model::new("local");
    let other = Model::new("foreign");
    variable!(m, x);
    variable!(other, y);
    for filtered in [false, true] {
        for mixed in [false, true] {
            let before_local = m.arena().len();
            let before_foreign = other.arena().len();
            let mut visited = Vec::new();
            let result = catch_unwind(AssertUnwindSafe(|| {
                if filtered {
                    sum!(m, { visited.push(i); if mixed && i == 0 { x } else { y } }
                        for i in 0..3 if true)
                } else {
                    sum!(m, { visited.push(i); if mixed && i == 0 { x } else { y } }
                        for i in 0..3)
                }
            }));
            assert!(result.is_err());
            assert_eq!(visited, [0, 1, 2]);
            assert_eq!(m.arena().len(), before_local);
            assert_eq!(other.arena().len(), before_foreign);
        }
    }
    assert!(catch_unwind(AssertUnwindSafe(|| sum!(m, y for _i in 0..1))).is_err());
}

#[test]
fn generated_identifiers_do_not_capture_caller_bindings() {
    let m = Model::new("hygiene");
    variable!(m, x);
    let __oximo_sum_model = x;
    let __terms = x;
    let __oximo_model_arena = x;
    let __oximo_model_receiver = x;
    assert_eq!(value(sum!(m, __oximo_sum_model for _i in 0..2)), 2.0);
    assert_eq!(value(sum!(m, __terms for i in 0..3 if i < 2)), 2.0);
    assert_eq!(value(sum!(__terms for i in 0..3 if i < 2)), 2.0);
    objective!(m, Min, sum!(__oximo_model_arena + __oximo_model_receiver for _i in 0..2));
    let objective = m.try_objective().unwrap();
    assert_eq!(evaluate(&m.arena(), objective.expr, &&[1.0][..]).unwrap(), 4.0);
}

#[test]
fn nested_and_multi_index_sums_inherit_the_anchor() {
    let m = Model::new("nested");
    variable!(m, x[j in 0..3]);
    let flat = sum!(m, x[j] for i in 0..4, j in 0..i);
    let nested = sum!(m, sum!(x[j] for j in 0..i) for i in 0..4);
    let explicit = sum!(m, sum!(&m, x[j] for j in 0..i) for i in 0..4);
    let block = sum!(m, { let end = i; sum!(x[j] for j in 0..end) } for i in 0..4);
    let filtered = sum!(m, sum!(x[j] for j in 0..3 if j < i) for i in 0..4);
    for expr in [flat, nested, explicit, filtered, block] {
        assert_eq!(value(expr), 10.0);
    }
    assert_eq!(value(sum!(m, x[j] for _i in 0..3, j in 0..0)), 0.0);
}

#[test]
fn model_macros_supply_context_including_qualified_sums() {
    let m = Model::new("implicit");
    variable!(m, x);
    let named = constraint!(m, named, sum!(x for _i in 0..0) <= 3.0);
    let anonymous = constraint!(m, sum!(x for i in 0..2 if i > 10) == 0.0);
    let computed = constraint!(m, name = "computed", sum!(x for _i in 0..0) >= -1.0);
    let range = constraint!(m, band, 0.0 <= sum!(x for _i in 0..0) <= 1.0);
    objective!(m, Min, 2.0 + oximo_core::sum!(sum!(x for _j in 0..0) for _i in 0..2));
    let arena = m.arena();
    let constraints = m.constraints();
    for id in [named, anonymous, computed] {
        let terms = extract_linear(&arena, constraints.algebraic()[id.index()].lhs).unwrap();
        assert!(terms.coeffs.is_empty());
    }
    assert!(matches!(range, RangeConstraintHandles::Interval(_)));
    assert_eq!(evaluate(&arena, m.try_objective().unwrap().expr, &&[][..]).unwrap(), 2.0);
}

#[test]
#[expect(clippy::float_cmp, reason = "these sums have exactly representable unit coefficients")]
fn indexed_sums_work_with_serial_and_parallel_arena_forks() {
    for threads in [1, 4] {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| {
            let m = Model::new("families");
            variable!(m, x[j in 0..3]);
            let alias = &m;
            let implicit = constraint!(m, implicit[i in 0..1024],
                sum!(x[j] for j in 0..i % 4) <= 10.0);
            let explicit = constraint!(m, explicit[i in 0..1024],
                sum!(&m, x[j] for j in 0..i % 4) <= 10.0);
            let aliased = constraint!(m, aliased[i in 0..1024],
                sum!(alias, x[j] for j in 0..i % 4) <= 10.0);
            let ranged = constraint!(m, ranged[i in 0..1024],
                0.0 <= sum!(x[j] for j in 0..3 if j < i % 4) <= 10.0);
            soc_constraint!(m, cones[i in 0..512],
                [sum!(m, x[j] for j in 0..i % 4)] <= 10.0 * x[2]);
            let arena = m.arena();
            let constraints = m.constraints();
            for i in 0..1024_usize {
                let RangeConstraintHandles::Interval(range) = ranged.get(i).unwrap() else {
                    panic!("expected an interval row");
                };
                for id in [
                    implicit.get(i).unwrap(),
                    explicit.get(i).unwrap(),
                    aliased.get(i).unwrap(),
                    range,
                ] {
                    let terms =
                        extract_linear(&arena, constraints.algebraic()[id.index()].lhs).unwrap();
                    assert_eq!(terms.coeffs.len(), i % 4);
                    assert!(terms.coeffs.iter().all(|(_, coefficient)| *coefficient == 1.0));
                }
            }
        });
    }
}

#[test]
fn model_expressions_are_evaluated_once_and_bodies_keep_order() {
    let m = Model::new("evaluation_order");
    variable!(m, x);
    let calls = Cell::new(0);
    let get_model = || {
        calls.set(calls.get() + 1);
        &m
    };
    let mut visited = Vec::new();
    let total = sum!(get_model(), { visited.push(i); x } for i in 0..3);
    assert_eq!(calls.get(), 1);
    assert_eq!(visited, [0, 1, 2]);
    assert_eq!(value(total), 3.0);
    constraint!(get_model(), c[i in 0..4], sum!(x for _j in 0..i) <= 10.0);
    assert_eq!(calls.get(), 2);
    objective!(get_model(), Min, sum!(x for _j in 0..0));
    assert_eq!(calls.get(), 3);
}

#[test]
fn explicit_model_anchors_respect_local_shadowing() {
    let m = Model::new("outer");
    let other = Model::new("inner");
    variable!(m, x);
    variable!(other, y);
    let inner_owned_by_other = Cell::new(false);
    let result = catch_unwind(AssertUnwindSafe(|| {
        sum!(m, {
            let m = &other;
            let inner = sum!(m, y for _j in 0..1);
            inner_owned_by_other.set(std::ptr::eq(inner.arena, y.arena));
            inner
        } for _i in 0..1)
    }));
    let panic = result.expect_err("the local m must not resolve to the outer model");
    assert!(inner_owned_by_other.get());
    let message = panic.downcast_ref::<&str>().copied().unwrap_or("");
    assert!(message.contains("different model"));
    assert_eq!(value(sum!(m, x for _i in 0..1)), 1.0);
}

#[test]
fn pattern_bindings_do_not_shadow_let_initializers_or_for_iterators() {
    let m = Model::new("pattern_scope");
    variable!(m, x);

    let total = sum!(m, {
        let from_let = {
            let m = sum!(m, x for _j in 0..1);
            m
        };
        for m in std::iter::once(sum!(m, x for _j in 0..1)) {
            let _ = m;
        }
        from_let
    } for _i in 0..1);

    assert_eq!(value(total), 1.0);
}
