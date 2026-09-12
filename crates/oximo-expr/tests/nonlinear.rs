#![expect(clippy::float_cmp)]

use oximo_expr::{
    Expr, ExprArena, ExprArenaCell, ExprClass, ExprNode, UnaryOp, VarId, classify, evaluate,
    render_expr, simplify,
};

fn value(expr: Expr<'_>) -> f64 {
    let values: &[f64] = &[];
    evaluate(&expr.arena.borrow(), expr.id, &values).unwrap()
}

#[test]
fn every_unary_method_uses_the_public_operation_enum() {
    let arena = ExprArenaCell::new(ExprArena::new());
    let x = Expr::constant(&arena, 1.25);
    let expressions = [
        (x.neg(), UnaryOp::Neg),
        (x.abs(), UnaryOp::Abs),
        (x.sqrt(), UnaryOp::Sqrt),
        (x.cbrt(), UnaryOp::Cbrt),
        (x.exp(), UnaryOp::Exp),
        (x.exp2(), UnaryOp::Exp2),
        (x.expm1(), UnaryOp::Expm1),
        (x.log(), UnaryOp::Log),
        (x.log2(), UnaryOp::Log2),
        (x.log10(), UnaryOp::Log10),
        (x.log1p(), UnaryOp::Log1p),
        (x.sin(), UnaryOp::Sin),
        (x.cos(), UnaryOp::Cos),
        (x.tan(), UnaryOp::Tan),
        ((x / 2.0).asin(), UnaryOp::Asin),
        ((x / 2.0).acos(), UnaryOp::Acos),
        (x.atan(), UnaryOp::Atan),
        (x.sinh(), UnaryOp::Sinh),
        (x.cosh(), UnaryOp::Cosh),
        (x.tanh(), UnaryOp::Tanh),
        (x.asinh(), UnaryOp::Asinh),
        ((x + 1.0).acosh(), UnaryOp::Acosh),
        ((x / 2.0).atanh(), UnaryOp::Atanh),
    ];
    let snapshot = arena.borrow();
    for (expr, expected) in expressions {
        assert!(matches!(snapshot.get(expr.id), ExprNode::Unary(op, _) if *op == expected));
        let expected_class =
            if expected == UnaryOp::Neg { ExprClass::Linear } else { ExprClass::Nonlinear };
        assert_eq!(classify(&snapshot, expr.id), expected_class);
    }
}

#[test]
fn rust_spelling_aliases_produce_the_same_operations() {
    let arena = ExprArenaCell::new(ExprArena::new());
    let x = Expr::constant(&arena, 0.25);
    for (modeling, rust, expected) in [
        (x.log(), x.ln(), UnaryOp::Log),
        (x.log1p(), x.ln_1p(), UnaryOp::Log1p),
        (x.expm1(), x.exp_m1(), UnaryOp::Expm1),
    ] {
        let snapshot = arena.borrow();
        assert!(matches!(snapshot.get(modeling.id), ExprNode::Unary(op, _) if *op == expected));
        assert!(matches!(snapshot.get(rust.id), ExprNode::Unary(op, _) if *op == expected));
    }
}

#[test]
fn unary_values_and_constant_folding_follow_f64() {
    let mut arena = ExprArena::new();
    for op in [
        UnaryOp::Neg,
        UnaryOp::Abs,
        UnaryOp::Sqrt,
        UnaryOp::Cbrt,
        UnaryOp::Exp,
        UnaryOp::Exp2,
        UnaryOp::Expm1,
        UnaryOp::Log,
        UnaryOp::Log2,
        UnaryOp::Log10,
        UnaryOp::Log1p,
        UnaryOp::Sin,
        UnaryOp::Cos,
        UnaryOp::Tan,
        UnaryOp::Asin,
        UnaryOp::Acos,
        UnaryOp::Atan,
        UnaryOp::Sinh,
        UnaryOp::Cosh,
        UnaryOp::Tanh,
        UnaryOp::Asinh,
        UnaryOp::Acosh,
        UnaryOp::Atanh,
    ] {
        let input = arena.constant(0.5);
        let node = arena.push(ExprNode::Unary(op, input));
        let folded = simplify(&mut arena, node);
        let ExprNode::Const(actual) = arena.get(folded) else { panic!("not folded") };
        assert_eq!(actual.to_bits(), op.apply(0.5).to_bits(), "{op}");
    }
}

#[test]
fn invalid_domains_are_deferred_to_ieee_evaluation() {
    let arena = ExprArenaCell::new(ExprArena::new());
    assert!(value(Expr::constant(&arena, -1.0).sqrt()).is_nan());
    assert!(value(Expr::constant(&arena, -1.0).log()).is_nan());
    assert!(value(Expr::constant(&arena, 0.5).acosh()).is_nan());
    assert!(value(Expr::constant(&arena, 2.0).atanh()).is_nan());
    assert_eq!(value(Expr::constant(&arena, -8.0).cbrt()), -2.0);
}

#[test]
fn atan2_uses_y_then_x_and_preserves_quadrants() {
    let arena = ExprArenaCell::new(ExprArena::new());
    let cases = [(1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0)];
    for (y, x) in cases {
        let expr = Expr::constant(&arena, y).atan2(Expr::constant(&arena, x));
        assert_eq!(value(expr).to_bits(), y.atan2(x).to_bits());
    }
}

#[test]
fn pairwise_min_and_max_flatten_both_sides_in_stable_order() {
    let arena = ExprArenaCell::new(ExprArena::new());
    let a = Expr::from_var(&arena, VarId(0));
    let b = Expr::from_var(&arena, VarId(1));
    let c = Expr::from_var(&arena, VarId(2));
    let d = Expr::from_var(&arena, VarId(3));
    let minimum = a.min(b).min(c.min(d));
    let maximum = a.max(b).max(c.max(d));
    let snapshot = arena.borrow();
    assert!(
        matches!(snapshot.get(minimum.id), ExprNode::Min(children) if children.as_slice() == [a.id, b.id, c.id, d.id])
    );
    assert!(
        matches!(snapshot.get(maximum.id), ExprNode::Max(children) if children.as_slice() == [a.id, b.id, c.id, d.id])
    );
    assert_eq!(render_expr(&snapshot, minimum.id, &|v| format!("x{}", v.0)), "min(x0, x1, x2, x3)");
    assert_eq!(render_expr(&snapshot, maximum.id, &|v| format!("x{}", v.0)), "max(x0, x1, x2, x3)");
}

#[test]
fn extrema_term_helpers_return_the_only_flattened_child() {
    let arena = ExprArenaCell::new(ExprArena::new());
    let (empty_min, empty_max, child) = {
        let mut nodes = arena.borrow_mut();
        (
            nodes.push(ExprNode::Min(Default::default())),
            nodes.push(ExprNode::Max(Default::default())),
            nodes.push(ExprNode::Const(7.0)),
        )
    };
    let empty_min = Expr::new(empty_min, &arena);
    let empty_max = Expr::new(empty_max, &arena);
    let child = Expr::new(child, &arena);

    let minimum = Expr::__min_terms([empty_min, child].into_iter()).unwrap();
    let maximum = Expr::__max_terms([empty_max, child].into_iter()).unwrap();

    assert_eq!(minimum.id, child.id);
    assert_eq!(maximum.id, child.id);
    assert_eq!(value(minimum), 7.0);
    assert_eq!(value(maximum), 7.0);
    assert_eq!(Expr::__min_terms([child].into_iter()).unwrap().id, child.id);
    assert_eq!(Expr::__max_terms([child].into_iter()).unwrap().id, child.id);
}
