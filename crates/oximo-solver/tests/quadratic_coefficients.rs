use std::sync::Arc;

use oximo_core::prelude::*;
use oximo_expr::ExprNode;
use oximo_solver::prepare::{Extracted, PreparedExpressions};

#[test]
fn direct_coefficients_are_shared_even_above_the_extraction_cache_budget() {
    let model = Model::new("large direct quadratic");
    variable!(model, x[i in 0..2048]);
    let mut builder = model.quadratic_builder();
    for i in 0..2048 {
        builder.add_product(2.0, x[i], x[i]).unwrap();
    }

    let expression = builder.build();
    let root = expression.id();
    let arena = model.arena();
    let ExprNode::Quadratic(stored) = arena.get(root) else { panic!("compact node") };
    let prepared = PreparedExpressions::new(arena.clone());

    for terms in [prepared.quadratic(root), prepared.quadratic_once(root), prepared.quadratic(root)]
    {
        let Some(Extracted::Shared(terms)) = terms else { panic!("shared coefficients") };
        assert!(Arc::ptr_eq(stored, &terms));
        assert_eq!(terms.quadratic().len(), 2048);
        assert_eq!(terms.hessian_entries().collect::<Vec<_>>()[0].2, 4.0);
    }
    assert!(prepared.linear(root).is_none());
}

#[test]
fn symbolic_parents_keep_revision_valid_values_and_numeric_children_immutable() {
    let model = Model::new("parameter around compact child");
    variable!(model, x);
    param!(model, p = 2.0);

    let mut builder = model.quadratic_builder();
    builder.add_product(3.0, x, x).unwrap();
    let child = builder.build();
    let expression = p * &child + x;
    let root = expression.id();
    let old = PreparedExpressions::new(model.arena().clone());
    p.set_param_value(-4.0).unwrap();
    let new = PreparedExpressions::new(model.arena().clone());
    assert_eq!(old.quadratic(root).unwrap().hessian_entries().collect::<Vec<_>>()[0].2, 12.0);
    assert_eq!(new.quadratic(root).unwrap().hessian_entries().collect::<Vec<_>>()[0].2, -24.0);
    assert_eq!(new.quadratic(child.id()).unwrap().hessian_entries().collect::<Vec<_>>()[0].2, 6.0);
}

#[test]
fn parameterized_parent_snapshots_preserve_the_smallest_hessian_diagonal() {
    let model = Model::new("subnormal parent");
    variable!(model, x);
    param!(model, p = 1.0);
    let tiny = f64::from_bits(1);
    let terms = oximo_expr::QuadraticTerms::from_coefficients(
        vec![(x.var_id().unwrap(), x.var_id().unwrap(), tiny)],
        Vec::new(),
        0.0,
    )
    .unwrap();
    let mut builder = model.quadratic_builder();
    builder.add_quadratic(1.0, &terms).unwrap();
    let child = builder.build();
    let root = (p * &child).id();
    let old = PreparedExpressions::new(model.arena().clone());
    p.set_param_value(-1.0).unwrap();

    let new = PreparedExpressions::new(model.arena().clone());
    assert_eq!(old.quadratic(root).unwrap().quadratic()[0].2.to_bits(), tiny.to_bits());
    assert_eq!(new.quadratic(root).unwrap().quadratic()[0].2.to_bits(), (-tiny).to_bits());
}
