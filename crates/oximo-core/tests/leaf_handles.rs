use oximo_core::prelude::*;
use oximo_expr::{ExprNode, evaluate, extract_linear, extract_quadratic};

fn copies<T: Copy>(value: T) -> (T, T) {
    (value, value)
}

#[test]
fn scalar_and_indexed_leaves_copy_without_allocating_nodes() {
    let m = Model::new("copy leaves");
    variable!(m, x);
    param!(m, p = 2.0);
    variable!(m, z[i in 0..3, j in 0..2]);
    param!(m, cost[i in 0..3] = f64::from(u32::try_from(i).unwrap()) + 1.0);

    let before = m.__sum_context().__node_count();
    let (first, second): (VariableHandle<'_>, _) = copies(x);
    let (parameter, _): (ParameterHandle<'_>, _) = copies(p);
    let entry = z[(2, 1)];
    let unit = cost[1];
    let _ = copies(entry);
    let _ = copies(unit);
    assert_eq!(first.id(), second.id());
    assert_eq!(parameter.id(), p.id());
    assert_eq!(z.at([2, 1]).id(), entry.id());
    assert_eq!(z.get((2, 1)).unwrap().id(), entry.id());
    assert_eq!(m.__sum_context().__node_count(), before);

    m.set_param(unit, 7.0).unwrap();
    assert_eq!(evaluate(&m.arena(), cost[1].id(), &&[][..]).unwrap(), 7.0);

    let expression = unit * entry * entry + first + second;
    let terms = extract_quadratic(&m.arena(), expression.id()).unwrap();
    assert_eq!(terms.quadratic(), [(entry.var_id().unwrap(), entry.var_id().unwrap(), 7.0)]);
    assert_eq!(terms.linear(), [(x.var_id().unwrap(), 2.0)]);
}

#[test]
fn sparse_filtered_and_named_entries_are_copy_handles() {
    let m = Model::new("sparse leaves");
    let names = Set::strings(["a", "b"]);
    variable!(m, named[k in names]);
    variable!(m, filtered[i in 0..4 if i % 2 == 0]);
    param!(m, price[k in names] = 2.0);

    let x = named["a"];
    let y = filtered[2];
    let p = price["a"];
    let _ = copies(x);
    let _ = copies(y);
    let _ = copies(p);
    assert!(filtered.get(1).is_none());

    m.fix(x, 1.0).unwrap();
    m.set_initial(y, 2.0).unwrap();
    constraint!(m, row, p * x + y <= 4.0);
    objective!(m, Min, p * x * x + y);
    m.set_params(&[(p, 3.0)]).unwrap();
    assert_eq!(
        extract_quadratic(&m.arena(), m.objective().as_ref().unwrap().expr).unwrap().quadratic()[0]
            .2,
        3.0
    );
}

#[test]
fn arithmetic_and_sums_return_computed_degrees() {
    let m = Model::new("computed expressions");
    variable!(m, x[i in 0..3]);
    param!(m, p = 2.0);
    let _: Expr<'_, Affine> = x[0] + 1.0;
    let _: Expr<'_, Affine> = -x[0];
    let _: Expr<'_, Affine> = x[0].neg();
    let _: Expr<'_, Constant> = p + 1.0;
    let _: Expr<'_, Constant> = -p;
    let _: Expr<'_, Quadratic> = x[0] * x[0];
    let sum: Expr<'_, Affine> = x.iter().map(|(_, value)| *value).sum();
    assert_eq!(extract_linear(&m.arena(), sum.id()).unwrap().coeffs.len(), 3);

    let sum: Expr<'_, Affine> = sum!(x[i] for i in 0..3);
    assert_eq!(extract_linear(&m.arena(), sum.id()).unwrap().coeffs.len(), 3);

    let empty: Expr<'_, Affine> = sum!(m, x[i] for i in 0..0);
    assert_eq!(evaluate(&m.arena(), empty.id(), &&[1.0, 2.0, 3.0][..]).unwrap(), 0.0);
}

#[test]
fn copied_leaves_preserve_unique_quadratic_accumulation_and_model_checks() {
    let m = Model::new("unique quadratic");
    variable!(m, x);
    let (first, second) = copies(x);
    let quadratic = first * second;
    let before = m.__sum_context().__node_count();
    let quadratic = quadratic + x;
    assert_eq!(m.__sum_context().__node_count(), before);

    let alias = quadratic.clone();
    let before = m.__sum_context().__node_count();
    let _ = quadratic + x;
    assert_eq!(m.__sum_context().__node_count(), before + 1);
    assert!(matches!(m.arena().get(alias.id()), ExprNode::Quadratic(_)));
    assert_eq!(extract_quadratic(&m.arena(), alias.id()).unwrap().linear().len(), 1);

    let other = Model::new("foreign");
    assert!(other.fix(x, 1.0).is_err());
}

#[test]
fn matrix_leaves_copy_and_trace_produces_an_affine_expression() {
    let m = Model::new("matrix leaves");
    let matrix = m.add_symmetric_variable("X", 2);
    let entry = matrix[(0, 1)];
    let _ = copies(entry);
    let trace: Expr<'_, Affine> = matrix.trace();
    assert_eq!(extract_linear(&m.arena(), trace.id()).unwrap().coeffs.len(), 2);

    let product: Expr<'_, Quadratic> = matrix.frobenius(&matrix);
    assert_eq!(extract_quadratic(&m.arena(), product.id()).unwrap().quadratic().len(), 3);
}
