use oximo_expr::{
    Children, EvalContext, EvalError, ExprArena, ExprNode, ParamId, ParameterSnapshot, VarId,
    WithParameters, evaluate,
};

struct Variables;

#[test]
fn snapshot_prefix_preserves_bits_and_shares_storage_when_unchanged() {
    let mut arena = ExprArena::new();
    let zero = arena.new_param(-0.0);
    let extra = arena.new_param(7.0);
    let saved = arena.parameter_snapshot();
    assert_eq!(saved.prefix(2).values().as_ptr(), saved.values().as_ptr());
    assert_eq!(saved.prefix(3).values().as_ptr(), saved.values().as_ptr());

    let prefix = saved.prefix(1);
    assert_eq!(prefix.value(zero).unwrap().to_bits(), (-0.0_f64).to_bits());
    assert_eq!(prefix.value(extra), None);
    assert_eq!(saved.prefix(0).values(), []);
    assert_eq!(saved.value(extra), Some(7.0));
}

impl EvalContext for Variables {
    fn var(&self, _: VarId) -> Option<f64> {
        Some(3.0)
    }
    fn param(&self, _: ParamId) -> Option<f64> {
        Some(99.0)
    }
}

#[test]
fn parameter_snapshot_survives_rebinding_append_cloning_and_arena_drop() {
    let saved = {
        let mut arena = ExprArena::new();
        let p = arena.new_param(2.0);
        let snapshot = arena.parameter_snapshot();
        assert_eq!(snapshot.values().as_ptr(), snapshot.clone().values().as_ptr());
        let mut copy = arena.clone();
        arena.set_param_value(p, 4.0).unwrap();
        copy.set_param_value(p, 6.0).unwrap();
        let q = arena.new_param(5.0);
        assert_eq!(snapshot.value(p), Some(2.0));
        assert_eq!(snapshot.value(q), None);
        snapshot
    };
    assert_eq!(saved.values(), &[2.0]);
    assert_eq!(ParameterSnapshot::default().values(), []);
}

#[test]
fn identical_parameter_assignments_keep_shared_storage_and_distinct_bits_detach() {
    let finite = 1.25_f64;
    for (initial, changed) in
        [(2.0, 4.0), (0.0, -0.0), (finite, f64::from_bits(finite.to_bits() + 1))]
    {
        let mut arena = ExprArena::new();
        let p = arena.new_param(initial);
        let saved = arena.parameter_snapshot();
        arena.set_param_value(p, initial).unwrap();
        assert_eq!(saved.values().as_ptr(), arena.parameter_snapshot().values().as_ptr());
        arena.set_param_value(p, changed).unwrap();
        assert_ne!(saved.values().as_ptr(), arena.parameter_snapshot().values().as_ptr());
        assert_eq!(saved.value(p).unwrap().to_bits(), initial.to_bits());
        assert_eq!(arena.param_value(p).to_bits(), changed.to_bits());
    }
}

#[test]
fn identical_parameter_assignment_still_rejects_locked_parameters() {
    let mut arena = ExprArena::new();
    let p = arena.new_param(2.0);
    arena.__lock_parameters([p]);
    assert!(arena.set_param_value(p, 2.0).is_err());
}

#[test]
fn snapshot_context_never_falls_back_in_leaf_shallow_array_or_large_arena_queries() {
    let mut arena = ExprArena::new();
    let p = arena.new_param(2.0);
    let parameters = arena.parameter_snapshot();
    let saved = WithParameters::new(&Variables, &parameters);
    let q = arena.new_param(5.0);
    let p_expr = arena.param(p);
    let q_expr = arena.param(q);
    let x = arena.var(VarId(0));
    let product = arena.push(ExprNode::Mul(Children::from_slice(&[p_expr, x])));
    let missing_product = arena.push(ExprNode::Mul(Children::from_slice(&[q_expr, x])));
    let shared = arena.push(ExprNode::Add(Children::from_slice(&[product, product])));
    let missing_shared =
        arena.push(ExprNode::Add(Children::from_slice(&[missing_product, missing_product])));
    arena.set_param_value(p, 4.0).unwrap();
    for large in [false, true] {
        if large {
            // Large arenas select the hashed traversal for compound roots.
            for i in 0..65_537 {
                arena.constant(f64::from(i));
            }
        }
        for (expr, expected) in [(p_expr, 2.0), (product, 6.0), (shared, 12.0)] {
            assert_eq!(evaluate(&arena, expr, &saved).unwrap(), expected);
        }
        for expr in [q_expr, missing_product, missing_shared] {
            assert!(
                matches!(evaluate(&arena, expr, &saved), Err(EvalError::UnboundParam(id)) if id == q)
            );
        }
        let current: &[f64] = &[3.0];
        assert_eq!(evaluate(&arena, shared, &current).unwrap(), 24.0);
        assert_eq!(evaluate(&arena, missing_shared, &current).unwrap(), 30.0);
    }
}

#[test]
fn snapshot_bit_comparison_preserves_signed_zero_rejected_nonfinite_values_and_length() {
    let mut arena = ExprArena::new();
    let p = arena.new_param(0.0);
    let positive_zero = arena.parameter_snapshot();
    assert!(positive_zero.same_bits(&positive_zero.clone()));
    arena.set_param_value(p, -0.0).unwrap();
    assert!(!positive_zero.same_bits(&arena.parameter_snapshot()));
    arena.set_param_value(p, 0.0).unwrap();
    assert!(positive_zero.same_bits(&arena.parameter_snapshot()));

    let saved = arena.parameter_snapshot();
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(arena.set_param_value(p, value).is_err());
        assert!(saved.same_bits(&arena.parameter_snapshot()));
    }
    arena.new_param(1.0);
    assert!(!saved.same_bits(&arena.parameter_snapshot()));
    assert!(ParameterSnapshot::default().same_bits(&ParameterSnapshot::default()));
}

#[test]
fn changed_parameter_ids_include_value_and_length_differences() {
    let mut arena = ExprArena::new();
    let p = arena.new_param(1.0);
    let q = arena.new_param(-0.0);
    let saved = arena.parameter_snapshot();
    assert_eq!(saved.changed_parameters(&saved), Vec::new());

    arena.set_param_value(p, 2.0).unwrap();
    assert_eq!(saved.changed_parameters(&arena.parameter_snapshot()), vec![p]);

    arena.set_param_value(p, 1.0).unwrap();
    arena.set_param_value(q, 0.0).unwrap();
    assert_eq!(saved.changed_parameters(&arena.parameter_snapshot()), vec![q]);

    let added = arena.new_param(3.0);
    assert_eq!(saved.changed_parameters(&arena.parameter_snapshot()), vec![q, added]);
    assert_eq!(arena.parameter_snapshot().changed_parameters(&saved), vec![q, added]);
}

#[test]
fn snapshot_equality_compares_bits_for_shared_and_independent_storage() {
    let capture = |values: &[f64]| {
        let mut arena = ExprArena::new();
        for &value in values {
            arena.new_param(value);
        }
        arena.parameter_snapshot()
    };
    let finite = 1.25_f64;
    let saved = capture(&[1.0, -0.0, finite, f64::MAX]);
    assert_eq!(saved, saved.clone());
    assert_eq!(saved, capture(&[1.0, -0.0, finite, f64::MAX]));
    assert_ne!(saved, capture(&[1.0, 0.0, finite, f64::MAX]));
    assert_ne!(saved, capture(&[1.0, -0.0, f64::from_bits(finite.to_bits() + 1), f64::MAX]));
    assert_ne!(saved, capture(&[1.0, -0.0, finite]));
    assert_ne!(saved, capture(&[2.0, -0.0, finite, f64::MAX]));
    assert_eq!(ParameterSnapshot::default(), capture(&[]));
}
