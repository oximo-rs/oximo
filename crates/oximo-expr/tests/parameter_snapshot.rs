use oximo_expr::{
    Children, EvalContext, EvalError, ExprArena, ExprNode, ParamId, ParameterSnapshot, VarId,
    WithParameters, evaluate,
};

struct Variables;

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
        arena.set_param_value(p, 4.0);
        copy.set_param_value(p, 6.0);
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
    let nan = f64::from_bits(0x7ff8_0000_0000_0001);
    for (initial, changed) in [(2.0, 4.0), (0.0, -0.0), (nan, f64::from_bits(nan.to_bits() + 1))] {
        let mut arena = ExprArena::new();
        let p = arena.new_param(initial);
        let saved = arena.parameter_snapshot();
        arena.set_param_value(p, initial);
        assert_eq!(saved.values().as_ptr(), arena.parameter_snapshot().values().as_ptr());
        arena.set_param_value(p, changed);
        assert_ne!(saved.values().as_ptr(), arena.parameter_snapshot().values().as_ptr());
        assert_eq!(saved.value(p).unwrap().to_bits(), initial.to_bits());
        assert_eq!(arena.param_value(p).to_bits(), changed.to_bits());
    }
}

#[test]
#[should_panic(expected = "cannot change a parameter embedded in a reformulation")]
fn identical_parameter_assignment_still_rejects_locked_parameters() {
    let mut arena = ExprArena::new();
    let p = arena.new_param(2.0);
    arena.__lock_parameters([p]);
    arena.set_param_value(p, 2.0);
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
    arena.set_param_value(p, 4.0);
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
fn snapshot_bit_comparison_preserves_signed_zero_nan_payloads_and_length() {
    let mut arena = ExprArena::new();
    let p = arena.new_param(0.0);
    let positive_zero = arena.parameter_snapshot();
    assert!(positive_zero.same_bits(&positive_zero.clone()));
    arena.set_param_value(p, -0.0);
    assert!(!positive_zero.same_bits(&arena.parameter_snapshot()));
    arena.set_param_value(p, 0.0);
    assert!(positive_zero.same_bits(&arena.parameter_snapshot()));

    let nan = f64::from_bits(0x7ff8_0000_0000_0001);
    arena.set_param_value(p, nan);
    let saved_nan = arena.parameter_snapshot();
    assert!(saved_nan.same_bits(&saved_nan.clone()));
    arena.set_param_value(p, nan);
    assert!(saved_nan.same_bits(&arena.parameter_snapshot()));
    arena.set_param_value(p, f64::from_bits(nan.to_bits() + 1));
    assert!(!saved_nan.same_bits(&arena.parameter_snapshot()));
    arena.set_param_value(p, nan);
    arena.new_param(1.0);
    assert!(!saved_nan.same_bits(&arena.parameter_snapshot()));
    assert!(ParameterSnapshot::default().same_bits(&ParameterSnapshot::default()));
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
    let nan = f64::from_bits(0x7ff8_0000_0000_0001);
    let saved = capture(&[1.0, -0.0, nan, f64::INFINITY]);
    assert_eq!(saved, saved.clone());
    assert_eq!(saved, capture(&[1.0, -0.0, nan, f64::INFINITY]));
    assert_ne!(saved, capture(&[1.0, 0.0, nan, f64::INFINITY]));
    assert_ne!(saved, capture(&[1.0, -0.0, f64::from_bits(nan.to_bits() + 1), f64::INFINITY]));
    assert_ne!(saved, capture(&[1.0, -0.0, nan]));
    assert_ne!(saved, capture(&[2.0, -0.0, nan, f64::INFINITY]));
    assert_eq!(ParameterSnapshot::default(), capture(&[]));
}
