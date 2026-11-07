use oximo_core::prelude::*;
use oximo_expr::{ExprClass, ExprNode, QuadraticBuildError, evaluate, extract_quadratic};

#[test]
fn coefficients_expand_affine_products_and_reuse_builder() {
    let m = Model::new("numeric quadratic");
    variable!(m, x);
    variable!(m, y);
    let left = 2.0 * x + 3.0;
    let right = y - 4.0;
    let before = m.arena().len();
    let mut builder = m.quadratic_builder();
    builder.add_product(5.0, &left, &right).unwrap();
    builder.add_product(2.0, y, x).unwrap();
    builder.add_product(3.0, x, x).unwrap();
    builder.add_term(7.0, x).unwrap();
    builder.add_constant(8.0).unwrap();
    assert_eq!(m.arena().len(), before);
    let expression = builder.build();
    let snapshot = m.arena();
    assert_eq!(snapshot.len(), before + 1);
    let q = extract_quadratic(&snapshot, expression.id()).unwrap();
    assert_eq!(
        q.hessian_entries().collect::<Vec<_>>(),
        [
            (x.var_id().unwrap(), x.var_id().unwrap(), 6.0),
            (y.var_id().unwrap(), x.var_id().unwrap(), 12.0)
        ]
    );
    assert_eq!(q.linear(), [(x.var_id().unwrap(), -33.0), (y.var_id().unwrap(), 15.0)]);
    assert_eq!(q.constant(), -52.0);
    assert_eq!(evaluate(&snapshot, expression.id(), &&[2.0, 3.0][..]).unwrap(), 11.0);
    builder.add_product(1.0, x, y).unwrap();
    let second = builder.build();
    assert_eq!(evaluate(&m.arena(), second.id(), &&[2.0, 3.0][..]).unwrap(), 6.0);
    assert_eq!(evaluate(&m.arena(), expression.id(), &&[2.0, 3.0][..]).unwrap(), 11.0);
    assert_eq!(builder.build().__class(), ExprClass::Linear);
}

#[test]
fn validation_is_atomic_and_parameters_are_never_snapshotted() {
    let m = Model::new("validation");
    let other = Model::new("foreign");
    variable!(m, x);
    variable!(other, y);
    param!(m, p = 2.0);
    let symbolic = p * x;
    let mut builder = m.quadratic_builder();
    builder.add_product(2.0, x, x).unwrap();
    assert_eq!(builder.add_product(1.0, x, y).unwrap_err(), QuadraticBuildError::ForeignArena);
    assert_eq!(
        builder.add_product(1.0, x, &symbolic).unwrap_err(),
        QuadraticBuildError::ParameterizedInput
    );
    assert_eq!(builder.add_term(1.0, p).unwrap_err(), QuadraticBuildError::ParameterizedInput);
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            builder.add_product(value, x, x).unwrap_err(),
            QuadraticBuildError::NonFiniteInput
        );
    }
    assert_eq!(
        builder.add_product(f64::MAX, 2.0 * x, x).unwrap_err(),
        QuadraticBuildError::ProductOverflow
    );
    let expression = builder.build();
    let q = extract_quadratic(&m.arena(), expression.id()).unwrap();
    assert_eq!(
        q.hessian_entries().collect::<Vec<_>>(),
        [(x.var_id().unwrap(), x.var_id().unwrap(), 4.0)]
    );
    builder.add_constant(f64::MAX).unwrap();
    assert_eq!(
        builder.add_constant(f64::MAX).unwrap_err(),
        QuadraticBuildError::AccumulationOverflow
    );
    let constant = builder.build();
    assert_eq!(extract_quadratic(&m.arena(), constant.id()).unwrap().constant(), f64::MAX);
}

#[test]
fn macro_arithmetic_reuses_handles_without_caller_clones() {
    let m = Model::new("ergonomics");
    variable!(m, x);
    variable!(m, y);
    param!(m, p = 2.0);
    objective!(m, Min, p * x * x + x);
    constraint!(m, first, p * x * x + x <= 4.0);
    constraint!(m, second, x * x + y <= 4.0);
    let reusable = x * x + y;
    objective!(m, Min, &reusable);
    constraint!(m, reused, &reusable + &reusable <= 10.0);
    constraint!(m, again, reusable <= 4.0);
    p.set_param_value(3.0).unwrap();
    let arena = m.arena();
    let q = extract_quadratic(&arena, m.constraints().algebraic()[0].lhs).unwrap();
    assert_eq!(q.hessian_entries().collect::<Vec<_>>()[0].2, 6.0);
}

#[test]
fn cancellation_and_large_products_follow_construction_policy() {
    let m = Model::new("thresholds");
    variable!(m, x[i in 0..34]);
    let mut builder = m.quadratic_builder();
    builder.add_product(2.0, x[0], x[1]).unwrap();
    builder.add_product(-2.0, x[1], x[0]).unwrap();
    builder.add_term(3.0, x[0]).unwrap();
    let affine = builder.build();
    assert!(matches!(m.arena().get(affine.id()), ExprNode::Linear { .. }));
    for count in [31, 32, 33] {
        let lhs = x.iter().take(count).map(|(_, v)| v).sum::<Expr<'_, Affine>>().compact();
        let product = &lhs * x[33];
        assert_eq!(matches!(m.arena().get(product.id()), ExprNode::Quadratic(_)), count <= 32);
        builder.add_product(1.0, &lhs, x[33]).unwrap();
        let explicit = builder.build();
        assert!(matches!(m.arena().get(explicit.id()), ExprNode::Quadratic(_)));
        assert_eq!(extract_quadratic(&m.arena(), explicit.id()).unwrap().quadratic().len(), count);
    }
}

#[test]
fn macro_sums_keep_fresh_quadratics_unique_and_reused_roots_identical() {
    let m = Model::new("sum ownership");
    variable!(m, x);
    for filtered in [false, true] {
        let before = m.arena().len();
        let q = if filtered {
            sum!(m,x.powi(2) for _i in 0..1 if true)
        } else {
            sum!(m,x.powi(2) for _i in 0..1)
        };
        let q = q + x;
        assert_eq!(
            m.arena().len(),
            before + 1,
            "sum! must preserve ownership of its fresh single term"
        );
        assert_eq!(
            extract_quadratic(&m.arena(), q.id()).unwrap().linear(),
            [(x.var_id().unwrap(), 1.0)]
        );
    }
    let q = x.square();
    let root = q.id();
    let summed = sum!(m,&q for _i in 0..1);
    assert_eq!(summed.id(), root);
    let again = sum!(&q for _i in 0..1);
    assert_eq!(again.id(), root);
}

#[test]
fn builders_ingest_hessians_and_check_dynamic_affine_inputs_without_nodes() {
    let m = Model::new("coefficient inputs");
    variable!(m, x);
    let tiny = f64::from_bits(1);
    let v = x.var_id().unwrap();
    let seed =
        oximo_expr::QuadraticTerms::from_coefficients(vec![(v, v, tiny)], vec![(v, 2.0)], 3.0)
            .unwrap();
    let dynamic = (2.0 * x + 1.0).erase();
    let before = m.arena().len();
    let mut builder = m.quadratic_builder();
    builder.add_quadratic(1.0, &seed).unwrap().add_expr(2.0, &dynamic).unwrap();
    let terms = builder.build_terms();
    assert_eq!(m.arena().len(), before);
    assert_eq!(terms.quadratic()[0].2.to_bits(), 1);
    assert_eq!(terms.linear(), [(v, 6.0)]);
    assert_eq!(terms.constant(), 5.0);
    assert!(builder.build_terms().quadratic().is_empty());
    let compound = x.square().erase();
    assert_eq!(builder.add_expr(1.0, &compound).unwrap_err(), QuadraticBuildError::NotAffine);
    param!(m, p = 2.0);
    assert_eq!(
        builder.add_expr(0.0, (p * x).erase()).unwrap_err(),
        QuadraticBuildError::ParameterizedInput
    );
    builder.add_quadratic(1.0, &terms).unwrap();
    let huge =
        oximo_expr::QuadraticTerms::from_hessian(Vec::new(), vec![(v, f64::MAX)], 0.0).unwrap();
    assert_eq!(
        builder.add_quadratic(2.0, &huge).unwrap_err(),
        QuadraticBuildError::ProductOverflow
    );
    let result = builder.build_terms();
    assert_eq!(result.linear(), terms.linear());
    assert_eq!(
        result.hessian_entries().collect::<Vec<_>>(),
        terms.hessian_entries().collect::<Vec<_>>()
    );
    assert_eq!(result.constant(), terms.constant());
}
