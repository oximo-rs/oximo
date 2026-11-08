use oximo_expr::{
    Affine, Expr, ExprArena, ExprArenaCell, ExprClass, ExprNode, QuadraticBuilder, VarId, evaluate,
    extract_quadratic,
};

fn square<'a>(factor: &Expr<'a, Affine>, construction: usize) -> Expr<'a> {
    match construction {
        0 => factor.square().into_dynamic(),
        1 => factor.powi(2),
        2 => factor.powf(2.0),
        3 => factor.pow(Expr::constant(factor.arena(), 2.0)),
        4 => (factor * factor).into_dynamic(),
        _ => (factor.clone() * factor.clone()).into_dynamic(),
    }
}

#[test]
fn compound_affine_squares_keep_their_factors_and_expose_quadratic_coefficients() {
    let arena = ExprArenaCell::new(ExprArena::new());
    let x = Expr::from_var(&arena, VarId(0));
    let y = Expr::from_var(&arena, VarId(1));
    let factor = 2.0 * x - 3.0 * y + 1.0;
    for construction in 0..6 {
        let expression = square(&factor, construction);
        assert_eq!(expression.__class(), ExprClass::Quadratic);

        let snapshot = arena.borrow();
        assert!(matches!(snapshot.get(expression.id()), ExprNode::Mul(children)
            if children.len() == 2 && children[0] == children[1]));

        let coefficients = extract_quadratic(&snapshot, expression.id()).unwrap();
        assert_eq!(
            coefficients.quadratic(),
            [(VarId(0), VarId(0), 4.0), (VarId(1), VarId(0), -12.0), (VarId(1), VarId(1), 9.0)]
        );
        assert_eq!(coefficients.linear(), [(VarId(0), 4.0), (VarId(1), -6.0)]);
        assert_eq!(coefficients.constant().to_bits(), 1.0_f64.to_bits());

        for point in [[0.0, 0.0], [1.0, 1.0], [2.0, -1.0]] {
            let value = evaluate(&snapshot, factor.id(), &&point[..]).unwrap();
            assert_eq!(
                evaluate(&snapshot, expression.id(), &&point[..]).unwrap().to_bits(),
                (value * value).to_bits()
            );
        }
    }

    let mut builder = QuadraticBuilder::new(&arena);
    builder.add_product(1.0, &factor, &factor).unwrap();
    let expanded = builder.build();
    assert!(matches!(arena.borrow().get(expanded.id()), ExprNode::Quadratic(_)));
}

#[test]
fn shifted_squares_preserve_small_positive_values_inside_nonlinear_functions() {
    let arena = ExprArenaCell::new(ExprArena::new());
    let x = Expr::from_var(&arena, VarId(0));
    let factor = x - 100_000_000.0;
    for construction in 0..6 {
        let expression = square(&factor, construction).ln();
        let value = evaluate(&arena.borrow(), expression.id(), &&[100_000_001.0][..]).unwrap();
        assert_eq!(value.to_bits(), 0.0_f64.to_bits(), "construction={construction}");
    }
}

#[test]
fn scaled_squares_preserve_values_when_the_expanded_coefficient_underflows() {
    let arena = ExprArenaCell::new(ExprArena::new());
    let x = Expr::from_var(&arena, VarId(0));
    let factor = 1e-200 * x;
    for construction in 0..6 {
        let expression = square(&factor, construction).ln();
        let value = evaluate(&arena.borrow(), expression.id(), &&[1e200][..]).unwrap();
        assert_eq!(value.to_bits(), 0.0_f64.to_bits(), "construction={construction}");
    }
}
