use oximo_expr::{Expr, ExprArena, ExprArenaCell, ExprNode, VarId, evaluate, extract_quadratic};

#[test]
fn right_growing_subtraction_retains_linear_coefficient_storage() {
    for count in [256_u32, 257, 512, 1024, 1025] {
        let arena = ExprArenaCell::new(ExprArena::new());
        let variables: Vec<_> = (0..count).map(|i| Expr::from_var(&arena, VarId(i))).collect();
        let mut expression = variables[0].square();
        for variable in &variables[1..] {
            expression = variable.square() - expression;
        }
        let snapshot = arena.borrow();
        let retained: usize = snapshot
            .nodes()
            .iter()
            .map(|node| match node {
                ExprNode::Quadratic(terms) => terms.linear().len() + terms.quadratic().len(),
                _ => 0,
            })
            .sum();
        assert!(retained <= 3 * count as usize, "{count} terms retained {retained} coefficients");

        let allocated: usize = snapshot
            .nodes()
            .iter()
            .map(|node| match node {
                ExprNode::Quadratic(terms) => terms.coefficient_storage_bytes(),
                _ => 0,
            })
            .sum();
        assert!(
            allocated <= 128 * count as usize,
            "{count} terms retained {allocated} coefficient bytes"
        );

        let terms = extract_quadratic(&snapshot, expression.id()).unwrap();
        assert_eq!(terms.quadratic().len(), count as usize);

        for &(row, column, coefficient) in terms.quadratic() {
            assert_eq!(row, column);
            let expected: f64 = if (count - 1 - row.0).is_multiple_of(2) { 1.0 } else { -1.0 };
            assert_eq!(coefficient.to_bits(), expected.to_bits());
        }

        let values = vec![1.0; count as usize];
        let expected: f64 = if count.is_multiple_of(2) { 0.0 } else { 1.0 };
        assert_eq!(
            evaluate(&snapshot, expression.id(), &&values[..]).unwrap().to_bits(),
            expected.to_bits()
        );
    }
}
