use std::panic::{AssertUnwindSafe, catch_unwind};

use oximo_expr::{
    Affine, AffineBuilder, Expr, ExprArena, ExprArenaCell, ExprClass, ExprNode, VarId, classify,
    evaluate, extract_linear, extract_quadratic, render_expr, split_linear,
};

fn variables(arena: &ExprArenaCell, n: u32) -> Vec<Expr<'_, Affine>> {
    (0..n).map(|i| Expr::from_var(arena, VarId(i))).collect()
}

#[test]
fn builder_merges_numeric_terms_and_reuses_without_mutating_outputs() {
    let arena = ExprArenaCell::new(ExprArena::new());
    let vars = variables(&arena, 3);
    let before = arena.borrow().len();
    let mut builder = AffineBuilder::with_capacity(&arena, 4);
    builder.add_term(2.0, vars[2]).add_term(4.0, vars[0]);
    builder.add_term(-2.0, vars[2]).add_term(3.0, vars[1]).add_constant(7.0);
    assert_eq!(arena.borrow().len(), before, "adding terms writes no nodes");
    let first = builder.build();
    let snapshot = arena.borrow();
    assert_eq!(snapshot.len(), before + 1);
    let terms = extract_linear(&snapshot, first.id()).unwrap();
    assert_eq!(terms.coeffs.as_ref(), &[(VarId(0), 4.0), (VarId(1), 3.0)]);
    assert!((terms.constant - 7.0).abs() < f64::EPSILON);
    assert!(matches!(snapshot.get(first.id()), ExprNode::Linear { .. }));
    builder.add_term(6.0, vars[2]);
    let second = builder.build();
    let latest = arena.borrow();
    assert_eq!(extract_linear(&latest, second.id()).unwrap().coeffs.as_ref(), &[(VarId(2), 6.0)]);
    assert_eq!(extract_linear(&latest, first.id()).unwrap().coeffs, terms.coeffs);
    assert_eq!(snapshot.len(), before + 1, "snapshots stay immutable");
    let zero = builder.build();
    assert!(evaluate(&arena.borrow(), zero.id(), &&[][..]).unwrap().abs() < f64::EPSILON);
    builder.add_term(1.0, vars[0]).add_constant(9.0);
    builder.clear();
    let zero = builder.build();
    assert!(extract_linear(&arena.borrow(), zero.id()).unwrap().coeffs.is_empty());
}

#[test]
fn builder_keeps_symbolic_coefficients_constants_and_first_seen_order() {
    let arena = ExprArenaCell::new(ExprArena::new());
    let vars = variables(&arena, 3);
    let param = arena.borrow_mut().new_param(2.0);
    let p = Expr::from_param(&arena, param);
    let px = p * vars[2];
    let before = arena.borrow().len();
    let mut builder = AffineBuilder::new(&arena);
    builder.add_term(2.0, vars[1]).add_term(3.0, px);
    builder.add_term(1.0, vars[0]).add_term(4.0, p).add_constant(5.0);
    assert_eq!(arena.borrow().len(), before);
    p.set_param_value(7.0); // rebinding before emission also stays symbolic
    let expr = builder.build();
    for value in [7.0, -3.0] {
        p.set_param_value(value);
        let snapshot = arena.borrow();
        let terms = extract_linear(&snapshot, expr.id()).unwrap();
        assert_eq!(
            terms.coeffs.as_ref(),
            &[(VarId(1), 2.0), (VarId(2), 3.0 * value), (VarId(0), 1.0)]
        );
        assert!((terms.constant - (5.0 + 4.0 * value)).abs() < f64::EPSILON);
        let quadratic = extract_quadratic(&snapshot, expr.id()).unwrap();
        assert_eq!(quadratic.hessian, Vec::new());
        assert_eq!(classify(&snapshot, expr.id()), ExprClass::Linear);
        let actual = evaluate(&snapshot, expr.id(), &&[1.0, 2.0, 3.0][..]).unwrap();
        assert!((actual - (10.0 + 13.0 * value)).abs() < f64::EPSILON);
    }
}

#[test]
fn foreign_term_panics_before_changing_builder_or_arenas() {
    let arena = ExprArenaCell::new(ExprArena::new());
    let foreign = ExprArenaCell::new(ExprArena::new());
    let x = variables(&arena, 1)[0];
    let y = variables(&foreign, 1)[0];
    let mut builder = AffineBuilder::new(&arena);
    builder.add_term(2.0, x);
    let before = arena.borrow().len();
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            builder.add_term(1.0, y);
        }))
        .is_err()
    );
    assert_eq!(arena.borrow().len(), before);
    assert_eq!(foreign.borrow().len(), 1);
    let expr = builder.build();
    assert_eq!(
        extract_linear(&arena.borrow(), expr.id()).unwrap().coeffs.as_ref(),
        &[(VarId(0), 2.0)]
    );
}

#[test]
fn operator_thresholds_and_saved_prefixes() {
    let arena = ExprArenaCell::new(ExprArena::new());
    let vars = variables(&arena, 65);
    let prefix = vars[..32].iter().copied().reduce(|a, b| a + b).unwrap();
    assert!(matches!(arena.borrow().get(prefix.id()), ExprNode::Linear { .. }));
    let large = prefix + vars[32];
    assert!(matches!(arena.borrow().get(large.id()), ExprNode::Add(_)));
    for expr in [large * 2.0, -large, large / 2.0, large - vars[0]] {
        assert!(!matches!(arena.borrow().get(expr.id()), ExprNode::Linear { .. }));
        assert!(extract_linear(&arena.borrow(), expr.id()).is_some());
    }
    assert_eq!(extract_linear(&arena.borrow(), prefix.id()).unwrap().coeffs.len(), 32);
    // More than 64 visited nodes, even though there is only one distinct variable.
    let mut id = vars[0].id();
    {
        let mut writer = arena.borrow_mut();
        for _ in 0..65 {
            id = writer.push(ExprNode::Unary(oximo_expr::UnaryOp::Neg, id));
        }
    }
    let deep = Expr::new(id, &arena).try_affine().unwrap() + vars[1];
    assert!(matches!(arena.borrow().get(deep.id()), ExprNode::Add(_)));
    assert_eq!(
        extract_linear(&arena.borrow(), deep.id()).unwrap().coeffs.as_ref(),
        &[(VarId(0), -1.0), (VarId(1), 1.0)]
    );
}

#[test]
fn small_nested_sums_preserve_coefficient_rounding() {
    let arena = ExprArenaCell::new(ExprArena::new());
    let x = variables(&arena, 1)[0];
    // Flattening this small nested sum would change its coefficient from 0 to 1.
    let inner: Expr<'_, Affine> = [-1e16 * x, x].into_iter().sum();
    let nested: Expr<'_, Affine> = [1e16 * x, inner].into_iter().sum();
    let snapshot = arena.borrow();
    let terms = extract_linear(&snapshot, nested.id()).unwrap();
    assert_eq!(terms.coeffs[0].1.to_bits(), 0.0_f64.to_bits());
    assert_eq!(extract_quadratic(&snapshot, nested.id()).unwrap().linear, Vec::new());
    drop(snapshot);
    let eager = 1e16 * x + inner;
    assert!(matches!(arena.borrow().get(eager.id()), ExprNode::Const(_)));
}

#[test]
fn deep_left_and_right_sums_store_linear_space_and_extract_in_order() {
    for right in [false, true] {
        let arena = ExprArenaCell::new(ExprArena::new());
        let vars = variables(&arena, 20_000);
        let expr = if right {
            vars.iter().rev().copied().reduce(|a, b| b + a).unwrap()
        } else {
            vars.iter().copied().reduce(|a, b| a + b).unwrap()
        };
        let snapshot = arena.borrow();
        assert_eq!(snapshot.len(), vars.len() * 2 - 1);
        let stored: usize = (0..snapshot.len())
            .map(|i| match snapshot.get(oximo_expr::ExprId(u32::try_from(i).unwrap())) {
                ExprNode::Linear { coeffs, .. } => coeffs.len(),
                _ => 0,
            })
            .sum();
        assert_eq!(stored, (2..=32).sum::<usize>(), "large prefixes are not copied");
        let terms = extract_linear(&snapshot, expr.id()).unwrap();
        let expected: Vec<_> = (0..20_000).map(|i| (VarId(i), 1.0)).collect();
        assert_eq!(terms.coeffs.as_ref(), expected);
        assert_eq!(extract_quadratic(&snapshot, expr.id()).unwrap().linear, expected);
        assert_eq!(split_linear(&snapshot, expr.id()).1, Vec::new());
        assert!(
            (evaluate(&snapshot, expr.id(), &&vec![1.0; vars.len()][..]).unwrap() - 20_000.0).abs()
                < f64::EPSILON
        );
    }
}

#[test]
fn shared_symbolic_dag_is_memoized_and_rebindable() {
    let arena = ExprArenaCell::new(ExprArena::new());
    let x = variables(&arena, 1)[0];
    let param = arena.borrow_mut().new_param(3.0);
    let p = Expr::from_param(&arena, param);
    let mut expr = p * x;
    for _ in 0..40 {
        expr = expr + expr;
    }
    for value in [3.0, 5.0] {
        p.set_param_value(value);
        let snapshot = arena.borrow();
        let expected = value * 2.0_f64.powi(40);
        assert_eq!(
            extract_linear(&snapshot, expr.id()).unwrap().coeffs.as_ref(),
            &[(VarId(0), expected)]
        );
        assert_eq!(
            extract_quadratic(&snapshot, expr.id()).unwrap().linear,
            vec![(VarId(0), expected)]
        );
    }
}

#[test]
fn builder_and_deferred_sums_handle_cancellation_and_nonfinite_values() {
    let arena = ExprArenaCell::new(ExprArena::new());
    let vars = variables(&arena, 80);
    let total: Expr<'_, Affine> = vars.iter().copied().sum();
    let cancelled = total - total;
    let mut builder = AffineBuilder::new(&arena);
    builder.add_term(1.0, total).add_term(-1.0, total);
    let compact = builder.build();
    assert!(matches!(arena.borrow().get(compact.id()), ExprNode::Const(_)));
    assert!(
        extract_linear(&arena.borrow(), cancelled.id())
            .unwrap()
            .coeffs
            .iter()
            .all(|(_, c)| c.abs() < f64::EPSILON)
    );
    let param = arena.borrow_mut().new_param(f64::INFINITY);
    let p = Expr::from_param(&arena, param);
    builder.add_term(0.0, p);
    let symbolic = builder.build();
    assert!(evaluate(&arena.borrow(), symbolic.id(), &&[][..]).unwrap().is_nan());
    p.set_param_value(2.0);
    assert!(evaluate(&arena.borrow(), symbolic.id(), &&[][..]).unwrap().abs() < f64::EPSILON);
    builder.add_constant(f64::INFINITY).add_constant(f64::NEG_INFINITY);
    let nan = builder.build();
    assert!(extract_linear(&arena.borrow(), nan.id()).unwrap().constant.is_nan());
}

#[test]
fn wide_duplicate_prefix_merges_to_small_support_in_first_seen_order() {
    let arena = ExprArenaCell::new(ExprArena::new());
    let vars = variables(&arena, 16);
    let prefix: Expr<'_, Affine> = (0..4096).map(|i| vars[i % 16]).sum();
    let before = arena.borrow();
    let combined = prefix + vars[0];
    let snapshot = arena.borrow();
    assert!(matches!(snapshot.get(combined.id()), ExprNode::Linear { .. }));
    let terms = extract_linear(&snapshot, combined.id()).unwrap();
    assert_eq!(terms.coeffs.len(), 16);
    assert_eq!(terms.coeffs[0], (VarId(0), 257.0));
    assert_eq!(extract_linear(&before, prefix.id()).unwrap().coeffs[0], (VarId(0), 256.0));
    let right = vars[1] + prefix;
    assert_eq!(extract_linear(&arena.borrow(), right.id()).unwrap().coeffs[0], (VarId(1), 257.0));
}

#[test]
fn numeric_cancellation_does_not_raise_product_or_power_degree() {
    let arena = ExprArenaCell::new(ExprArena::new());
    let vars = variables(&arena, 100);
    let left = vars.iter().copied().reduce(|a, b| a + b).unwrap();
    let right = vars.iter().rev().copied().reduce(|a, b| a + b).unwrap();
    for cancelled in [left - left, left - right] {
        for expr in [
            (cancelled * vars[0]).erase(),
            (vars[0] * cancelled).erase(),
            cancelled.powi(3),
            cancelled.powf(2.0),
            cancelled.square().erase(),
        ] {
            assert!(expr.try_affine().is_ok());
            let snapshot = arena.borrow();
            let linear = extract_linear(&snapshot, expr.id()).unwrap();
            assert!(linear.coeffs.is_empty());
            assert_eq!(linear.constant, 0.0);
            assert_eq!(evaluate(&snapshot, expr.id(), &&vec![1.0; 100][..]).unwrap(), 0.0);
        }
        let constant = cancelled + 3.0;
        let affine = constant * vars[0];
        assert!(affine.try_affine().is_ok());
        assert_eq!(
            extract_linear(&arena.borrow(), affine.id()).unwrap().coeffs.as_ref(),
            &[(VarId(0), 3.0)]
        );
        let large_product = cancelled * left;
        assert!(large_product.try_affine().is_ok());
        assert!(extract_linear(&arena.borrow(), large_product.id()).unwrap().coeffs.is_empty());
        assert_eq!(evaluate(&arena.borrow(), large_product.id(), &&[][..]).unwrap(), 0.0);
    }
    let parameter = arena.borrow_mut().new_param(0.0);
    let p = Expr::from_param(&arena, parameter);
    let symbolic = p * left;
    let product = symbolic * vars[0];
    assert_eq!(product.__class(), ExprClass::Quadratic);
    p.set_param_value(2.0);
    assert_eq!(evaluate(&arena.borrow(), product.id(), &&vec![1.0; 100][..]).unwrap(), 200.0);
    let scalar = p + p;
    let before = arena.borrow().len();
    let scaled = left * scalar;
    assert_eq!(
        arena.borrow().len(),
        before + 1,
        "symbolic scaling does not compact the large prefix"
    );
    assert!(matches!(arena.borrow().get(scaled.id()), ExprNode::Mul(_)));
    // NaN coefficients must survive, rather than simplifying s - s by ID.
    let nan = f64::INFINITY * left;
    let product = (nan - nan) * vars[0];
    assert!(evaluate(&arena.borrow(), product.id(), &&vec![1.0; 100][..]).unwrap().is_nan());
}

#[test]
fn numeric_cancellation_under_symbolic_parents_stays_affine_after_rebinding() {
    let arena = ExprArenaCell::new(ExprArena::new());
    let vars = variables(&arena, 64);
    let sum = vars.iter().copied().reduce(|a, b| a + b).unwrap();
    let parameter = arena.borrow_mut().new_param(2.0);
    let p = Expr::from_param(&arena, parameter);
    let cancelled = sum - sum;
    let symbolic = cancelled + p;
    let snapshot = arena.borrow();
    let expressions = [
        (((cancelled * p) * vars[0]).erase(), false),
        ((symbolic * vars[0]).erase(), true),
        ((-symbolic * vars[0]).erase(), true),
        ((symbolic.powi(3) * vars[0]).erase(), true),
    ];
    // Rewriting never changes the original handles or the saved snapshot.
    assert!(matches!(snapshot.get(symbolic.id()), ExprNode::Add(_)));
    for value in [2.0_f64, -3.0, 0.0] {
        p.set_param_value(value);
        let current = arena.borrow();
        for (index, &(expression, has_parameter)) in expressions.iter().enumerate() {
            assert!(expression.try_affine().is_ok(), "case {index}");
            let terms = extract_linear(&current, expression.id()).unwrap();
            let coefficient = match index {
                2 => -value,
                3 => value.powi(3),
                _ => value,
            };
            let expected = if has_parameter { 4.0 * coefficient } else { 0.0 };
            assert_eq!(evaluate(&current, expression.id(), &&vec![4.0; 64][..]).unwrap(), expected);
            assert!(terms.coeffs.iter().all(|&(var, _)| var == VarId(0)));
        }
    }
}

#[test]
fn compaction_is_explicit_immutable_and_preserves_parameter_rebinding() {
    let arena = ExprArenaCell::new(ExprArena::new());
    let vars = variables(&arena, 100);
    let original = vars.iter().copied().reduce(|a, b| a + b).unwrap();
    let snapshot = arena.borrow();
    let compact = original.compact();
    let latest = arena.borrow();
    assert!(matches!(latest.get(original.id()), ExprNode::Add(_)));
    assert!(matches!(latest.get(compact.id()), ExprNode::Linear { .. }));
    assert_eq!(compact.compact().id(), compact.id());
    assert_eq!(
        extract_linear(&snapshot, original.id()).unwrap().coeffs,
        extract_linear(&latest, compact.id()).unwrap().coeffs
    );
    let parameter = arena.borrow_mut().new_param(2.0);
    let p = Expr::from_param(&arena, parameter);
    let symbolic = (p * original + original).compact();
    for value in [2.0, 4.0] {
        p.set_param_value(value);
        assert_eq!(
            evaluate(&arena.borrow(), symbolic.id(), &&vec![1.0; 100][..]).unwrap(),
            100.0 * (value + 1.0)
        );
    }
}

#[test]
fn compensated_builder_recovers_small_numeric_terms_and_constants() {
    let arena = ExprArenaCell::new(ExprArena::new());
    let vars = variables(&arena, 2);
    let mut compensated = AffineBuilder::with_capacity_compensated(&arena, 4);
    let mut ordinary = AffineBuilder::new(&arena);
    for weight in [1e16, 1.0, -1e16] {
        compensated.add_term(weight, vars[1]).add_constant(weight);
        ordinary.add_term(weight, vars[1]).add_constant(weight);
    }
    compensated.add_term(2.0, vars[0]);
    let first = compensated.build();
    let plain = ordinary.build();
    let snapshot = arena.borrow();
    let terms = extract_linear(&snapshot, first.id()).unwrap();
    assert_eq!(terms.coeffs.as_ref(), &[(VarId(1), 1.0), (VarId(0), 2.0)]);
    assert_eq!(terms.constant, 1.0);
    assert!(extract_linear(&snapshot, plain.id()).unwrap().coeffs.is_empty());
    assert_eq!(extract_linear(&snapshot, plain.id()).unwrap().constant, 0.0);
    let empty = compensated.build();
    assert!(extract_linear(&arena.borrow(), empty.id()).unwrap().coeffs.is_empty());
    compensated.add_term(4.0, vars[1]);
    let second = compensated.build();
    assert_eq!(
        extract_linear(&arena.borrow(), second.id()).unwrap().coeffs.as_ref(),
        &[(VarId(1), 4.0)]
    );
    assert_eq!(extract_linear(&snapshot, first.id()).unwrap().coeffs, terms.coeffs);
}

#[test]
fn compensated_builder_keeps_symbolic_segments_live_and_nonfinite_arithmetic_observable() {
    let arena = ExprArenaCell::new(ExprArena::new());
    let x = variables(&arena, 1)[0];
    let parameter = arena.borrow_mut().new_param(2.0);
    let p = Expr::from_param(&arena, parameter);
    let mut builder = AffineBuilder::new_compensated(&arena);
    for weight in [1e16, 1.0, -1e16] {
        builder.add_term(weight, x);
    }
    builder.add_term(1.0, p * x);
    let expr = builder.build();
    for value in [2.0, 4.0] {
        p.set_param_value(value);
        assert_eq!(evaluate(&arena.borrow(), expr.id(), &&[1.0][..]).unwrap(), value + 1.0);
    }
    builder.add_constant(f64::INFINITY);
    let infinity = builder.build();
    assert_eq!(evaluate(&arena.borrow(), infinity.id(), &&[][..]).unwrap(), f64::INFINITY);
    builder.add_constant(f64::INFINITY).add_constant(f64::NEG_INFINITY);
    let nan = builder.build();
    assert!(evaluate(&arena.borrow(), nan.id(), &&[][..]).unwrap().is_nan());
    // Compensation does not recover an overflowed partial sum.
    builder.add_constant(1e308).add_constant(1e308).add_constant(-1e308);
    let overflow = builder.build();
    assert_eq!(evaluate(&arena.borrow(), overflow.id(), &&[][..]).unwrap(), f64::INFINITY);
    builder.add_term(1e16, x).add_constant(3.0);
    builder.clear();
    let cleared = builder.build();
    assert_eq!(evaluate(&arena.borrow(), cleared.id(), &&[][..]).unwrap(), 0.0);
}

#[test]
fn regrouping_overflow_retries_original_grouping_for_finite_inputs() {
    let arena = ExprArenaCell::new(ExprArena::new());
    let vars = variables(&arena, 64);
    let terms: Vec<_> = [1e308, 1e308, -1e308]
        .into_iter()
        .flat_map(|weight| vars.iter().map(move |&var| weight * var))
        .collect();
    let right = terms.iter().rev().copied().reduce(|a, b| b + a).unwrap();
    let left = terms.iter().copied().reduce(|a, b| a + b).unwrap();
    let snapshot = arena.borrow();
    let linear = extract_linear(&snapshot, right.id()).unwrap();
    assert!(linear.coeffs.iter().all(|(_, c)| c.to_bits() == 1e308_f64.to_bits()));
    let quadratic = extract_quadratic(&snapshot, right.id()).unwrap();
    assert!(quadratic.linear.iter().all(|(_, c)| c.to_bits() == 1e308_f64.to_bits()));
    // Original left grouping overflows too; this guard does not hide it.
    assert!(
        extract_linear(&snapshot, left.id()).unwrap().coeffs.iter().all(|(_, c)| c.is_infinite())
    );
    let mut values = vec![0.0; 64];
    values[0] = 1.0;
    assert_eq!(evaluate(&snapshot, right.id(), &&values[..]).unwrap(), 1e308);
}

#[test]
fn mixed_shared_affine_region_is_extracted_without_expanding_occurrences() {
    let arena = ExprArenaCell::new(ExprArena::new());
    let vars = variables(&arena, 64);
    let mut shared = vars.iter().copied().reduce(|a, b| a + b).unwrap();
    for _ in 0..40 {
        shared = shared + shared;
    }
    let sine = vars[0].sin();
    let expression = shared + sine + shared;
    let snapshot = arena.borrow();
    let (linear, residual) = split_linear(&snapshot, expression.id());
    assert_eq!(linear.coeffs.len(), 64);
    assert!(linear.coeffs.iter().all(|(_, c)| c.to_bits() == 2f64.powi(41).to_bits()));
    assert_eq!(residual, vec![oximo_expr::SignedExpr { id: sine.id(), neg: false }]);
}

#[test]
fn mixed_deep_additive_chain_and_nested_rendering_are_stack_safe() {
    let arena = ExprArenaCell::new(ExprArena::new());
    let vars = variables(&arena, 20_000);
    let sum = vars.iter().copied().reduce(|a, b| a + b).unwrap();
    let nonlinear = sum.sin();
    let rendered = render_expr(&arena.borrow(), nonlinear.id(), &|v| format!("x{}", v.0));
    let names = (0..20_000).map(|i| format!("x{i}")).collect::<Vec<_>>().join(" + ");
    assert_eq!(rendered, format!("sin({names})"));
    let mixed = vars.iter().copied().fold(vars[0].sin(), |a, b| a + b);
    let snapshot = arena.borrow();
    let (linear, residual) = split_linear(&snapshot, mixed.id());
    assert_eq!(linear.coeffs.len(), 20_000);
    assert!(linear.coeffs.iter().all(|(_, c)| c.to_bits() == 1.0_f64.to_bits()));
    assert_eq!(residual.len(), 1);
}

#[test]
fn regrouping_does_not_hide_original_coefficient_or_constant_overflow() {
    let arena = ExprArenaCell::new(ExprArena::new());
    let vars = variables(&arena, 64);
    let sum = vars.iter().copied().reduce(|a, b| a + b).unwrap();
    let coefficient_overflow = -1e308 * sum + (1e308 * sum + 1e308 * sum);
    let constant_overflow = (sum - 1e308) + ((sum + 1e308) + (sum + 1e308));
    let q = vars[0] * vars[1];
    let hessian_overflow = (-1e308 * q + (1e308 * q + 1e308 * q)) + sum;
    let snapshot = arena.borrow();
    let linear = extract_linear(&snapshot, coefficient_overflow.id()).unwrap();
    assert!(linear.coeffs.iter().all(|(_, c)| *c == f64::INFINITY));
    let quadratic = extract_quadratic(&snapshot, coefficient_overflow.id()).unwrap();
    assert!(quadratic.linear.iter().all(|(_, c)| *c == f64::INFINITY));
    assert_eq!(extract_linear(&snapshot, constant_overflow.id()).unwrap().constant, f64::INFINITY);
    assert_eq!(
        extract_quadratic(&snapshot, constant_overflow.id()).unwrap().constant,
        f64::INFINITY
    );
    let quadratic = extract_quadratic(&snapshot, hessian_overflow.id()).unwrap();
    assert_eq!(quadratic.hessian, vec![(VarId(1), VarId(0), f64::INFINITY)]);
    assert!(quadratic.linear.iter().all(|(_, c)| c.to_bits() == 1.0_f64.to_bits()));
}

#[test]
fn hidden_overflow_guard_uses_current_parameter_bindings() {
    let arena = ExprArenaCell::new(ExprArena::new());
    let vars = variables(&arena, 64);
    let sum = vars.iter().copied().reduce(|a, b| a + b).unwrap();
    let parameter = arena.borrow_mut().new_param(1e308);
    let p = Expr::from_param(&arena, parameter);
    let expression = -p * sum + (p * sum + p * sum);
    for value in [1e308, 2.0, 1e308] {
        p.set_param_value(value);
        let snapshot = arena.borrow();
        let expected = if value > 1e300 { f64::INFINITY } else { 2.0 };
        assert!(
            extract_linear(&snapshot, expression.id())
                .unwrap()
                .coeffs
                .iter()
                .all(|(_, c)| c.to_bits() == expected.to_bits())
        );
        assert!(
            extract_quadratic(&snapshot, expression.id())
                .unwrap()
                .linear
                .iter()
                .all(|(_, c)| c.to_bits() == expected.to_bits())
        );
    }
}

#[test]
fn cancellation_under_a_deep_symbolic_spine_stays_affine_after_rebinding() {
    std::thread::Builder::new()
        .stack_size(512 * 1024)
        .spawn(|| {
            let arena = ExprArenaCell::new(ExprArena::new());
            let vars = variables(&arena, 64);
            let sum = vars.iter().copied().reduce(|a, b| a + b).unwrap();
            let cancelled = sum - sum;
            let pid = arena.borrow_mut().new_param(2.0);
            let p = Expr::from_param(&arena, pid);
            let mut root = cancelled.id();
            {
                let mut storage = arena.borrow_mut();
                for _ in 0..20_000 {
                    root = storage.push(ExprNode::Add(vec![root, p.id()].into()));
                }
            }
            let expression = Expr::new(root, &arena) * vars[0];
            assert!(expression.try_affine().is_ok());
            for value in [2.0, -1.0, 0.0] {
                p.set_param_value(value);
                let snapshot = arena.borrow();
                let terms = extract_linear(&snapshot, expression.id()).unwrap();
                assert!(terms.coeffs.iter().all(|(var, _)| *var == VarId(0)));
                assert_eq!(terms.coeffs.iter().map(|(_, c)| c).sum::<f64>(), 20_000.0 * value);
                assert_eq!(terms.constant, 0.0);
            }
        })
        .unwrap()
        .join()
        .unwrap();
}
