#![expect(clippy::float_cmp, reason = "exact coefficients and preserved IEEE arithmetic")]

use std::cell::{Cell, RefCell};
use std::ops::Mul;
use std::panic::{AssertUnwindSafe, catch_unwind};

use oximo_core::prelude::*;
use oximo_expr::{Degree, ExprNode, evaluate, extract_linear};

fn value<D: Degree>(expr: &Expr<'_, D>, values: &[f64]) -> f64 {
    evaluate(&expr.arena().borrow(), expr.id(), &values).unwrap()
}

#[test]
fn numeric_products_emit_one_node_in_both_orders() {
    let m = Model::new("weighted");

    variable!(m, x[i in 0..128]);
    let weights: Vec<_> = (0..128).map(|i| f64::from(i + 1)).collect();
    let before = m.arena().len();
    let left: Expr<'_, Affine> = sum!(m, weights[i] * x[i] for i in 0..128);
    assert_eq!(m.arena().len(), before + 1);

    let before = m.arena().len();
    let right: Expr<'_, Affine> = sum!((x[i] * weights[i]) for i in 0..128);
    assert_eq!(m.arena().len(), before + 1);

    let arena = m.arena();
    assert!(matches!(arena.get(left.id()), ExprNode::Linear { .. }));

    let left_terms = extract_linear(&arena, left.id()).unwrap();
    let right_terms = extract_linear(&arena, right.id()).unwrap();
    assert_eq!(left_terms.coeffs, right_terms.coeffs);
    assert_eq!(left_terms.constant, right_terms.constant);
    assert_eq!(value(&left, &[1.0; 128]), 8256.0);
}

#[test]
fn affine_owned_borrowed_and_temporary_inputs_use_the_builder() {
    let m = Model::new("affine operands");

    variable!(m, x);
    let input = x + 3.0;
    let before = m.arena().len();
    let borrowed: Expr<'_, Affine> = sum!(m, 2 * &input for _i in 0..3);
    assert_eq!(m.arena().len(), before + 1);

    let reverse = sum!(m, &input * 2 for _i in 0..3);
    let owned = sum!(m, 2.0 * (x + 3.0) for _i in 0..3);
    let temporary = sum!(m, 2.0 * &(x + 3.0) for _i in 0..3);
    let coefficients = || vec![2.0];
    let borrowed_coefficient = sum!(m, coefficients()[0] * x for _i in 0..3);
    for expr in [borrowed, reverse, owned, temporary] {
        assert_eq!(value(&expr, &[5.0]), 48.0);
    }

    assert_eq!(value(&input, &[5.0]), 8.0);
    assert_eq!(value(&borrowed_coefficient, &[5.0]), 30.0);
}

#[test]
fn function_returned_borrows_keep_their_temporary_owners_alive() {
    let m = Model::new("temporary borrows through functions");

    variable!(m, x);
    let left = sum!(m, 2.0 * std::convert::identity(&(x + 1.0)) for _i in 0..2);
    let right = sum!(m, std::convert::identity(&(x + 1.0)) * 2.0 for _i in 0..2);
    let filtered = sum!(m, 2.0 * std::convert::identity(&(x + 1.0)) for _i in 0..2 if true);
    let flat = sum!(m, 2.0 * std::convert::identity(&(x + 1.0))
        for _i in 0..2, _j in 0..2);

    for expression in [left, right, filtered] {
        assert_eq!(value(&expression, &[3.0]), 16.0);
    }
    assert_eq!(value(&flat, &[3.0]), 32.0);
}

#[test]
fn custom_multiplication_runs_before_scrutinee_guards_drop() {
    struct Guard<'a> {
        alive: &'a Cell<bool>,
        events: &'a RefCell<Vec<&'static str>>,
    }

    struct Weight<'a> {
        alive: &'a Cell<bool>,
        events: &'a RefCell<Vec<&'static str>>,
    }

    impl<'a> Guard<'a> {
        fn new(alive: &'a Cell<bool>, events: &'a RefCell<Vec<&'static str>>) -> Self {
            alive.set(true);
            events.borrow_mut().push("create");
            Self { alive, events }
        }
        fn weight(&self) -> Weight<'a> {
            Weight { alive: self.alive, events: self.events }
        }
    }

    impl Drop for Guard<'_> {
        fn drop(&mut self) {
            self.events.borrow_mut().push("drop");
            self.alive.set(false);
        }
    }

    impl<'a> Mul<VariableHandle<'a>> for Weight<'_> {
        type Output = Expr<'a, Affine>;
        fn mul(self, x: VariableHandle<'a>) -> Self::Output {
            assert!(self.alive.get(), "the guard must outlive multiplication");
            self.events.borrow_mut().push("mul");
            2.0 * x
        }
    }

    impl<'a> Mul<Weight<'_>> for VariableHandle<'a> {
        type Output = Expr<'a, Affine>;
        fn mul(self, weight: Weight<'_>) -> Self::Output {
            weight * self
        }
    }

    let m = Model::new("temporary guard order");

    variable!(m, x);
    let alive = Cell::new(false);
    let events = RefCell::new(Vec::new());
    let expected = ["create", "mul", "drop", "create", "mul", "drop"];
    let left = sum!(m, Guard::new(&alive, &events).weight() * x for _i in 0..2);
    assert_eq!(*events.borrow(), expected);

    events.borrow_mut().clear();
    let right = sum!(m, x * Guard::new(&alive, &events).weight() for _i in 0..2 if true);
    assert_eq!(*events.borrow(), expected);
    assert!(!alive.get());
    assert_eq!(value(&left, &[3.0]), value(&right, &[3.0]));
}

#[test]
fn flattened_callbacks_preserve_custom_input_conversion_and_drop_order() {
    struct Weight<'s>(&'s RefCell<Vec<&'static str>>);
    struct Input<'a, 's> {
        expression: Expr<'a, Affine>,
        events: &'s RefCell<Vec<&'static str>>,
    }

    impl<'a> ExprInput<'a> for Input<'a, '_> {
        type Degree = Affine;
        fn as_expression(&self) -> &Expr<'a, Affine> {
            self.events.borrow_mut().push("convert");
            &self.expression
        }
    }

    impl Drop for Input<'_, '_> {
        fn drop(&mut self) {
            self.events.borrow_mut().push("drop");
        }
    }

    impl<'a, 's> Mul<VariableHandle<'a>> for Weight<'s> {
        type Output = Input<'a, 's>;
        fn mul(self, x: VariableHandle<'a>) -> Self::Output {
            self.0.borrow_mut().push("mul");
            Input { expression: 2.0 * x, events: self.0 }
        }
    }

    let m = Model::new("custom input conversion order");

    variable!(m, x);
    let events = RefCell::new(Vec::new());
    let flat = sum!(m, Weight(&events) * x for _i in 0..2, _j in 0..2);
    assert_eq!(*events.borrow(), ["mul", "convert", "drop"].repeat(4));

    events.borrow_mut().clear();
    let filtered = sum!(m, Weight(&events) * x for _i in 0..2, _j in 0..2 if true);
    let mut expected = vec!["mul"; 4];
    expected.extend(["convert", "drop"].repeat(4));
    assert_eq!(*events.borrow(), expected);
    assert_eq!(value(&flat, &[3.0]), value(&filtered, &[3.0]));
}

#[test]
fn flattened_callbacks_own_noncopy_keys_until_before_input_conversion() {
    type Events = RefCell<Vec<(&'static str, usize)>>;
    struct Key<'s> {
        id: usize,
        events: &'s Events,
    }
    impl Drop for Key<'_> {
        fn drop(&mut self) {
            self.events.borrow_mut().push(("key", self.id));
        }
    }

    struct Keys<'s>(&'s Events);
    impl<'s> SumDomain<Key<'s>> for Keys<'s> {
        fn keys(&self) -> impl Iterator<Item = Key<'s>> + '_ {
            (0..2).map(|id| Key { id, events: self.0 })
        }
    }

    struct Input<'a, 's> {
        expression: Expr<'a, Affine>,
        id: usize,
        events: &'s Events,
    }
    impl<'a> ExprInput<'a> for Input<'a, '_> {
        type Degree = Affine;
        fn as_expression(&self) -> &Expr<'a, Affine> {
            self.events.borrow_mut().push(("convert", self.id));
            &self.expression
        }
    }
    impl Drop for Input<'_, '_> {
        fn drop(&mut self) {
            self.events.borrow_mut().push(("input", self.id));
        }
    }

    let m = Model::new("owned index keys");

    variable!(m, x);
    let events = Events::new(Vec::new());
    let domain = Keys(&events);
    let flat = sum!(m, {
        events.borrow_mut().push(("body", key.id));
        Input { expression: 2.0 * x, id: key.id, events: &events }
    } for _i in 0..2, key in domain);

    let observed = events.borrow().clone();
    events.borrow_mut().clear();
    let unfused = sum!(m, unfused: {
        events.borrow_mut().push(("body", key.id));
        Input { expression: 2.0 * x, id: key.id, events: &events }
    } for _i in 0..2, key in domain);

    let expected: Vec<_> = (0..2)
        .flat_map(|_| {
            (0..2).flat_map(|id| [("body", id), ("key", id), ("convert", id), ("input", id)])
        })
        .collect();
    assert_eq!(observed, expected);
    assert_eq!(*events.borrow(), expected);
    assert_eq!(value(&flat, &[3.0]), 24.0);
    assert_eq!(value(&unfused, &[3.0]), 24.0);
}

#[test]
fn capacity_hints_preserve_temporary_domains_and_noncopy_outer_keys() {
    type Events = RefCell<Vec<&'static str>>;
    struct Domain<'s>(&'s Events);
    impl<'s> Domain<'s> {
        fn new(events: &'s Events) -> Self {
            events.borrow_mut().push("domain");
            Self(events)
        }
    }
    impl Drop for Domain<'_> {
        fn drop(&mut self) {
            self.0.borrow_mut().push("domain drop");
        }
    }

    struct Key<'s>(&'s Events);
    impl Drop for Key<'_> {
        fn drop(&mut self) {
            self.0.borrow_mut().push("key drop");
        }
    }
    impl<'s> SumDomain<Key<'s>> for Domain<'s> {
        fn keys(&self) -> impl Iterator<Item = Key<'s>> + '_ {
            self.0.borrow_mut().push("keys");
            (0..2).map(|_| Key(self.0))
        }
    }

    struct Input<'a, 's>(Expr<'a, Affine>, &'s Events);
    impl<'a> ExprInput<'a> for Input<'a, '_> {
        type Degree = Affine;
        fn as_expression(&self) -> &Expr<'a, Affine> {
            self.1.borrow_mut().push("convert");
            &self.0
        }
    }
    impl Drop for Input<'_, '_> {
        fn drop(&mut self) {
            self.1.borrow_mut().push("input drop");
        }
    }

    let m = Model::new("capacity hint lifetimes");

    variable!(m, x);
    let events = Events::new(Vec::new());
    let flat = sum!(m, {
        key.0.borrow_mut().push("body");
        Input(2.0 * x, &events)
    } for key in Domain::new(&events), _j in 0..3);

    let observed = events.borrow().clone();
    events.borrow_mut().clear();
    let unfused = sum!(m, unfused: {
        key.0.borrow_mut().push("body");
        Input(2.0 * x, &events)
    } for key in Domain::new(&events), _j in 0..3);

    let mut expected = vec!["domain", "keys"];
    for _ in 0..2 {
        expected.extend(["body", "convert", "input drop"].repeat(3));
        expected.push("key drop");
    }

    expected.push("domain drop");
    assert_eq!(observed, expected);
    assert_eq!(*events.borrow(), expected);
    assert_eq!(value(&flat, &[3.0]), 36.0);
    assert_eq!(value(&flat, &[3.0]), value(&unfused, &[3.0]));
}

#[test]
fn capacity_planning_keeps_dependent_domains_lazy_and_evaluated_once() {
    let m = Model::new("dependent capacity hints");

    variable!(m, x);
    let domains = RefCell::new(Vec::new());
    let visited = RefCell::new(Vec::new());
    let total = sum!(m, {
        visited.borrow_mut().push((i, j));
        2.0 * x
    } for i in { domains.borrow_mut().push("outer"); 0..4 },
        j in { domains.borrow_mut().push("inner"); 0..i });
    assert_eq!(*domains.borrow(), ["outer", "inner", "inner", "inner", "inner"]);
    assert_eq!(*visited.borrow(), [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)]);
    assert_eq!(value(&total, &[3.0]), 36.0);

    domains.borrow_mut().clear();
    let empty = sum!(m, 2.0 * x for _i in 0..0,
        _j in { domains.borrow_mut().push("inner"); 0..8 });
    assert!(domains.borrow().is_empty());
    assert_eq!(value(&empty, &[]), 0.0);

    let cube = sum!(m, 2.0 * x for _i in 0..2, _j in 1..3, _k in 0..4);
    assert_eq!(value(&cube, &[1.0]), 32.0);

    let wide = sum!(m, 2.0 * x for _i in 0..0, _j in 0..65536, _k in 0..65536);
    assert_eq!(value(&wide, &[]), 0.0);
}

#[test]
fn local_loop_exits_in_inner_domains_keep_sum_lowering_flat() {
    let m = Model::new("local loop exits in sum domains");
    variable!(m, x);
    let unanchored = sum!(2.0 * x for i in 0..3, _j in {
        let mut length = 0;
        loop {
            if length == i { break 0..length; }
            length += 1;
        }
    });
    assert_eq!(value(&unanchored, &[3.0]), 18.0);

    let before = m.arena().len();
    let anchored = sum!(m, 2.0 * x for i in 0..3, _j in {
        let mut length = 0;
        loop {
            if length == i { break 0..length; }
            length += 1;
        }
    });
    assert_eq!(m.arena().len(), before + 1);
    assert_eq!(value(&anchored, &[3.0]), 18.0);

    let before = m.arena().len();
    let continued = sum!(2.0 * x for i in 0..4, _j in {
        let mut length = 0;
        for k in 0..i {
            if k % 2 == 0 { continue; }
            length += 1;
        }
        0..length
    });
    assert_eq!(m.arena().len(), before + 1);
    assert_eq!(value(&continued, &[3.0]), 12.0);
}

#[test]
fn multiple_indices_share_one_accumulator_and_skip_empty_inner_domains() {
    let m = Model::new("flat multi-index sum");

    variable!(m, x[j in 0..3]);
    let before = m.__sum_context().__node_count();
    let flat = sum!(m, 2.0 * x[j] for _i in 0..4, j in 0..3);
    assert_eq!(m.__sum_context().__node_count(), before + 1);

    let before = m.__sum_context().__node_count();
    let filtered = sum!(m, 2.0 * x[j] for _i in 0..4, j in 0..3 if true);
    assert_eq!(m.__sum_context().__node_count(), before + 1);
    assert_eq!(value(&flat, &[1.0, 2.0, 3.0]), 48.0);
    assert_eq!(value(&flat, &[1.0, 2.0, 3.0]), value(&filtered, &[1.0, 2.0, 3.0]));

    let before = m.__sum_context().__node_count();
    let nested = sum!(m, unfused: 2.0 * x[j] for _i in 0..4, j in 0..3);
    let arena = m.arena();
    assert_eq!(arena.len(), before + 29);
    assert_eq!(
        arena.nodes()[before..].iter().filter(|node| matches!(node, ExprNode::Add(_))).count(),
        5
    );
    assert_eq!(value(&nested, &[1.0, 2.0, 3.0]), 48.0);

    let skipped = sum!(2.0 * x[j] for i in 0..3, j in 0..i);
    assert_eq!(value(&skipped, &[1.0, 2.0, 3.0]), 8.0);

    let empty = sum!(m, 2.0 * x[j] for _i in 0..3, j in 0..0);
    assert_eq!(value(&empty, &[]), 0.0);
}

#[test]
fn multi_index_callback_returns_and_domain_returns_keep_their_scope() {
    let m = Model::new("multi-index returns");

    variable!(m, x);
    let local = sum!(m, ({ if j == 1 { return 3.0 * x; } 2.0 }) * x
        for _i in 0..2, j in 0..3);
    assert_eq!(value(&local, &[1.0]), 14.0);

    let domain = sum!(m, 2.0 * x for i in 0..3, _j in {
        if i == 1 { return 3.0 * x; }
        0..2
    });
    assert_eq!(value(&domain, &[1.0]), 11.0);
}

#[test]
fn typed_weighted_terms_make_generic_rounding_explicit() {
    fn ordinary<'a, W>(m: &'a Model, weights: &[W], x: VariableHandle<'a>) -> Expr<'a, Affine>
    where
        W: Copy + Mul<VariableHandle<'a>, Output = Expr<'a, Affine>>,
    {
        sum!(m, weights[i] * x for i in 0..weights.len())
    }
    fn typed<'a, W: Copy + Into<f64>, E: ExprInput<'a, Degree = Affine>>(
        m: &'a Model,
        weights: &[W],
        input: E,
    ) -> Expr<'a, Affine> {
        sum!(m, weighted_term(weights[i], &input) for i in 0..weights.len())
    }

    let m = Model::new("generic rounding");

    variable!(m, x);
    let weights = [0.1, 0.2];
    let concrete = sum!(m, weights[i] * x for i in 0..2);
    let generic = ordinary(&m, &weights, x);
    let explicit = typed(&m, &weights, x + 0.0);
    assert_eq!(value(&generic, &[10.0]), 3.0);
    assert_eq!(value(&concrete, &[10.0]), 3.000_000_000_000_000_4);
    assert_eq!(value(&explicit, &[10.0]).to_bits(), value(&concrete, &[10.0]).to_bits());

    let explicit_unfused = sum!(m, unfused: weighted_term(weights[i], x) for i in 0..2);
    assert_eq!(value(&explicit_unfused, &[10.0]).to_bits(), value(&concrete, &[10.0]).to_bits());
    let integers = typed(&m, &[2, 3], x + 1.0);
    assert_eq!(value(&integers, &[10.0]), 55.0);
    let extremes = typed(&m, &[i32::MAX, i32::MIN, 2], x + 1.0);
    assert_eq!(value(&extremes, &[10.0]), 11.0);
}

#[test]
fn constants_keep_their_static_degree_and_parameters_stay_live() {
    let m = Model::new("symbolic");
    variable!(m, x);
    param!(m, p = 2.0);
    let constant: Expr<'_, Constant> = sum!(m, 3.0 * p for _i in 0..2);
    let affine = p * x + 1.0;
    let symbolic: Expr<'_, Affine> = sum!(m, 2.0 * &affine for _i in 0..3);

    let parameter_weight = sum!(m, p * x for _i in 0..2);
    for parameter in [2.0, -4.0] {
        p.set_param_value(parameter).unwrap();
        assert_eq!(value(&constant, &[]), 6.0 * parameter);
        assert_eq!(value(&symbolic, &[5.0]), 30.0 * parameter + 6.0);
        assert_eq!(value(&parameter_weight, &[5.0]), 10.0 * parameter);
    }
}

#[test]
fn fused_numeric_rules_match_manual_builder() {
    let m = Model::new("numeric rules");

    variable!(m, x);
    for weights in [
        [1e16, 1.0, -1e16],
        [0.0, -0.0, 2.0],
        [f64::INFINITY, 1.0, f64::NEG_INFINITY],
        [f64::NAN, 1.0, 0.0],
        [f64::MIN_POSITIVE, f64::from_bits(1), -f64::MIN_POSITIVE],
    ] {
        let fused = sum!(m, weights[i] * x for i in 0..3);
        let mut builder = m.affine_builder();
        for weight in weights {
            builder.add_term(weight, x);
        }
        let manual = builder.build();
        let actual = value(&fused, &[2.0]);
        let expected = value(&manual, &[2.0]);
        assert!(actual.to_bits() == expected.to_bits() || actual.is_nan() && expected.is_nan());
    }
}

#[test]
fn unfused_retains_product_nodes_and_nested_sum_grouping() {
    let m = Model::new("preserve");

    variable!(m, x);
    let weights = [1e16, 1.0, -1e16];
    let before = m.arena().len();
    let preserved = sum!(m, unfused: weights[i] * x for i in 0..3);
    assert_eq!(m.arena().len(), before + 7);

    let ordinary = oximo_core::sum::__sum_over(&(0..3_usize), |i| weights[i] * x);
    assert_eq!(value(&preserved, &[1.0]).to_bits(), value(&ordinary, &[1.0]).to_bits());
    assert!(matches!(m.arena().get(preserved.id()), ExprNode::Add(_)));

    let before = m.arena().len();
    let nested = sum!(m, unfused: sum!(2.0 * x for _j in 0..3) for _i in 0..2);
    assert_eq!(m.arena().len(), before + 15);
    assert_eq!(value(&nested, &[1.0]), 12.0);

    let unanchored = sum!(unfused: 2 * x for _i in 0..2);
    let filtered = sum!(m, unfused: 2 * x for i in 0..4 if i % 2 == 0);
    assert_eq!(value(&unanchored, &[1.0]), 4.0);
    assert_eq!(value(&filtered, &[1.0]), 4.0);

    let rounding_weights = [0.1, 0.2];
    let preserved = sum!(m, unfused: rounding_weights[i] * x for i in 0..2);
    let fused = sum!(m, rounding_weights[i] * x for i in 0..2);
    assert_eq!(value(&preserved, &[10.0]), 3.0);
    assert_eq!(value(&fused, &[10.0]), (0.1 + 0.2) * 10.0);
    assert_ne!(value(&preserved, &[10.0]).to_bits(), value(&fused, &[10.0]).to_bits());
}

#[test]
fn filters_domains_contexts_and_empty_sums_work() {
    let m = Model::new("domains");

    variable!(m, x[i in 0..3]);
    let selected = sum!(m, 2 * x[j] for i in 0..3, j in 0..i if j % 2 == 0);
    assert_eq!(value(&selected, &[1.0; 3]), 4.0);

    let nested = sum!(m, sum!(2 * x[j] for j in 0..i) for i in 0..3);
    assert_eq!(value(&nested, &[1.0; 3]), 6.0);

    let empty: Expr<'_, Affine> = sum!(m, 2 * x[i] for i in 0..0);
    let filtered = sum!(m, 2 * x[i] for i in 0..3 if false);
    assert_eq!(value(&empty, &[]), 0.0);
    assert_eq!(value(&filtered, &[]), 0.0);

    let preserved_empty = sum!(m, unfused: 2 * x[i] for i in 0..0);
    assert_eq!(value(&preserved_empty, &[]), 0.0);
    assert!(catch_unwind(AssertUnwindSafe(|| sum!(2 * x[i] for i in 0..0))).is_err());

    objective!(m, Min, sum!(2 * x[i] for i in 0..3));
    constraint!(m, rows[i in 0..3], sum!(2 * x[j] for j in 0..i) <= 10.0);
    assert_eq!(m.num_constraints(), 3);
}

#[test]
fn operands_execute_once_in_order_and_do_not_capture_generated_names() {
    let m = Model::new("evaluation order");

    variable!(m, x);
    let visited = RefCell::new(Vec::new());
    let __oximo_product_operands = 2.0;
    let __oximo_product_tag = x;
    let total = sum!(m,
        { visited.borrow_mut().push((i, "left")); __oximo_product_operands }
        * { visited.borrow_mut().push((i, "right")); __oximo_product_tag }
        for i in 0..3);
    assert_eq!(
        *visited.borrow(),
        [(0, "left"), (0, "right"), (1, "left"), (1, "right"), (2, "left"), (2, "right")]
    );
    assert_eq!(value(&total, &[1.0]), 6.0);

    let __oximo_sum_key = x;
    let __oximo_sum_keys = x;
    let flat = sum!(m, (__oximo_sum_key + __oximo_sum_keys) for _i in 0..2, _j in 0..2);
    assert_eq!(value(&flat, &[1.0]), 8.0);
}

#[test]
fn custom_noncopy_products_and_generic_products_keep_mul() {
    struct Weight<'a>(&'a RefCell<Vec<&'static str>>);
    impl<'a> Mul<VariableHandle<'a>> for Weight<'_> {
        type Output = Expr<'a, Affine>;
        fn mul(self, rhs: VariableHandle<'a>) -> Self::Output {
            self.0.borrow_mut().push("mul");
            7.0 * rhs
        }
    }
    impl Drop for Weight<'_> {
        fn drop(&mut self) {
            self.0.borrow_mut().push("drop");
        }
    }

    fn generic<'a, W: Mul<VariableHandle<'a>, Output = Expr<'a, Affine>>>(
        m: &'a Model,
        weight: W,
        x: VariableHandle<'a>,
    ) -> Expr<'a, Affine> {
        let mut weight = Some(weight);
        sum!(m, weight.take().unwrap() * x for _i in 0..1)
    }

    let m = Model::new("custom");

    variable!(m, x);
    let calls = RefCell::new(Vec::new());
    let total = sum!(m, Weight(&calls) * x for _i in 0..2);
    assert_eq!(*calls.borrow(), ["mul", "drop", "mul", "drop"]);
    assert_eq!(value(&total, &[1.0]), 14.0);
    assert_eq!(value(&generic(&m, Weight(&calls), x), &[1.0]), 7.0);
}

#[test]
fn ordinary_inner_products_keep_array_to_slice_rhs_coercions() {
    struct Scale<'a>(VariableHandle<'a>);
    impl<'a> Mul<&[f64]> for Scale<'a> {
        type Output = Expr<'a, Affine>;
        fn mul(self, weights: &[f64]) -> Self::Output {
            weights.iter().sum::<f64>() * self.0
        }
    }

    let m = Model::new("coercing product operands");
    variable!(m, x);
    let weights = [2.0, 3.0];
    let reference = sum!(m, unfused: Scale(x) * &weights * x for _i in 0..2);
    assert_eq!(value(&reference, &[3.0]), 90.0);

    let array = sum!(m, Scale(x) * &weights * x for _i in 0..2);
    assert_eq!(value(&array, &[3.0]), 90.0);

    let temporary = sum!(m, Scale(x) * &[2.0, 3.0] * x for _i in 0..2);
    assert_eq!(value(&temporary, &[3.0]), 90.0);

    let direct = sum!(m, Scale(x) * &weights for _i in 0..2);
    assert_eq!(value(&direct, &[3.0]), 30.0);

    let filtered = sum!(m, Scale(x) * &weights * x for _i in 0..2, _j in 0..2 if true);
    assert_eq!(value(&filtered, &[3.0]), 180.0);

    fn generic<'a, S>(m: &'a Model, scale: S, x: VariableHandle<'a>) -> Expr<'a, Quadratic>
    where
        S: for<'b> Mul<&'b [f64], Output = Expr<'a, Affine>>,
    {
        let mut scale = Some(scale);
        sum!(m, scale.take().unwrap() * &[2.0, 3.0] * x for _i in 0..1)
    }

    assert_eq!(value(&generic(&m, Scale(x), x), &[3.0]), 45.0);
}

#[test]
fn non_affine_and_dynamic_products_remain_ordinary_expressions() {
    let m = Model::new("fallback");

    variable!(m, x);
    let quadratic: Expr<'_, Quadratic> = sum!(m, 2.0 * x.square() for _i in 0..2);
    let nonlinear: Expr<'_, Nonlinear> = sum!(m, 2.0 * x.sin() for _i in 0..2);
    let dynamic: Expr<'_, Dynamic> = sum!(m, 2.0 * x.as_dynamic() for _i in 0..2);
    assert_eq!(value(&quadratic, &[3.0]), 36.0);
    assert_eq!(value(&nonlinear, &[0.0]), 0.0);
    assert_eq!(value(&dynamic, &[3.0]), 12.0);
}

#[test]
fn foreign_products_finish_callbacks_before_rejection() {
    let m = Model::new("local");
    let other = Model::new("foreign");

    variable!(m, x);
    variable!(other, y);
    for filtered in [false, true] {
        let before = m.arena().len();
        let foreign_before = other.arena().len();
        let mut visited = Vec::new();
        let result = catch_unwind(AssertUnwindSafe(|| {
            if filtered {
                sum!(m, 2.0 * { visited.push(i); if i == 1 { y } else { x } }
                    for i in 0..3 if true)
            } else {
                sum!(m, 2.0 * { visited.push(i); if i == 1 { y } else { x } }
                    for i in 0..3)
            }
        }));
        assert!(result.is_err());
        assert_eq!(visited, [0, 1, 2]);
        assert_eq!(m.arena().len(), before);
        assert_eq!(other.arena().len(), foreign_before);
    }

    let mut visited = Vec::new();
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            sum!(m, 2.0 * { visited.push((i,j)); if i == 0 && j == 1 { y } else { x } }
            for i in 0..3, j in 0..2)
        }))
        .is_err()
    );
    assert_eq!(visited, [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]);
}

#[test]
fn callback_returns_keep_the_original_expression_type() {
    let m = Model::new("returns");

    variable!(m, x);
    let total = sum!(m, ({ if i == 1 { return 3.0 * x; } 2.0 }) * x for i in 0..3);
    assert_eq!(value(&total, &[1.0]), 7.0);

    macro_rules! coefficient {
        ($i:expr) => {{
            if $i == 1 {
                return 3.0 * x;
            }
            2.0
        }};
    }

    let opaque = sum!(m, coefficient!(i) * x for i in 0..3);
    assert_eq!(value(&opaque, &[1.0]), 7.0);
}

#[test]
fn unfused_is_scoped_and_reaches_nested_domains() {
    let m = Model::new("preserve scope");

    variable!(m, x);
    objective!(m, Min, {
        let before = m.arena().len();
        let preserved = sum!(unfused: 2.0 * x for _j in 0..3);
        assert_eq!(m.arena().len(), before + 7);
        let before = m.arena().len();
        let fused = sum!(2.0 * x for _j in 0..3);
        assert_eq!(m.arena().len(), before + 1);
        preserved + fused
    });

    let before = m.arena().len();
    let total = sum!(m, unfused: 2.0 * x for _i in 0..{
        let inner = sum!(2.0 * x for _j in 0..3);
        assert_eq!(value(&inner, &[1.0]), 6.0);
        1
    });
    assert_eq!(m.arena().len(), before + 9);
    assert_eq!(value(&total, &[1.0]), 2.0);

    let before = m.__sum_context().__node_count();
    let nested = sum!(m, unfused: sum!(2.0 * x for _i in 0..2, _j in 0..2) for _k in 0..1);
    assert_eq!(m.__sum_context().__node_count(), before + 11);
    assert_eq!(value(&nested, &[1.0]), 8.0);
}

#[test]
fn weighted_sums_work_in_serial_and_parallel_arena_forks() {
    for threads in [1, 4] {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| {
            let m = Model::new("forks");

            variable!(m, x[j in 0..4]);
            constraint!(m, rows[i in 0..1024], sum!(2.0 * x[j] for j in 0..i % 5) <= 10.0);
            let arena = m.arena();
            let rows = m.constraints();
            for (i, row) in rows.algebraic().iter().enumerate() {
                let terms = extract_linear(&arena, row.lhs).unwrap();
                assert_eq!(terms.coeffs.len(), i % 5);
                assert!(terms.coeffs.iter().all(|(_, coefficient)| *coefficient == 2.0));
            }
        });
    }
}
