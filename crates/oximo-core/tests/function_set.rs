#![expect(clippy::float_cmp)]
#![expect(clippy::many_single_char_names)]

use oximo_core::function_set::{AlgebraicConstraintIr, IntoAffineFunction};
use oximo_core::prelude::*;
use oximo_expr::{ExprClass, evaluate, extract_linear, extract_quadratic};

#[test]
fn operators_and_families_carry_static_degree() {
    let m = Model::new("degrees");
    variable!(m, x[i in 0..3]);
    param!(m, p[i in 0..3] = 2.0);
    let _: Expr<'_, Affine> = p[0] * x[0] + x[1] + 3.0;
    let _: Expr<'_, Quadratic> = x[0] * x[1] + x[2];
    let _: Expr<'_, Quadratic> = x[0].square();
    let _: Expr<'_, Nonlinear> = x[0] * x[1] * x[2];
    let _: Expr<'_, Nonlinear> = x[0].sin();
    let _: Expr<'_, Dynamic> = x[0].powi(2);
    let _: Expr<'_, Affine> = sum!(m, p[i] * x[i] for i in 0..3);
    let _: Expr<'_, Quadratic> = sum!(m, x[i].square() for i in 0..3);
    let _: Expr<'_, Affine> = dot(&[x[0], x[1]], &[1.0, 2.0]);
    let mixed = [x[0].erase(), x[1].square().erase(), x[2].sin().erase()];
    assert!(matches!(mixed[0].into_function(), AnyScalarFunction::Affine(_)));
    assert!(matches!(mixed[1].into_function(), AnyScalarFunction::Quadratic(_)));
    assert!(matches!(mixed[2].into_function(), AnyScalarFunction::Nonlinear(_)));
    assert_eq!(mixed[1].try_affine().unwrap_err(), ExprClass::Quadratic);
    assert_eq!(mixed[2].try_quadratic().unwrap_err(), ExprClass::Nonlinear);
}

#[test]
fn dynamic_relations_retain_expressions_and_checked_classification() {
    let m = Model::new("dynamic relations");
    variable!(m, x);
    let expressions = [x.erase(), x.square().erase(), x.sin().erase()];
    let before = m.arena().len();
    for expr in expressions {
        let le: Constraint<ScalarDynamicFunction<'_>, LessThan> = expr.le(3.0);
        let ge: Constraint<ScalarDynamicFunction<'_>, GreaterThan> = expr.ge(-2.0);
        let eq: Constraint<ScalarDynamicFunction<'_>, EqualTo> = expr.eq(1.0);
        assert_eq!(le.function.expression().id(), expr.id());
        let handles =
            [m.__add_constraint_auto(le), m.__add_constraint_auto(ge), m.__add_constraint_auto(eq)];
        let rows = m.constraints();
        for (handle, expected) in
            handles.into_iter().zip([(f64::NEG_INFINITY, 3.0), (-2.0, f64::INFINITY), (1.0, 1.0)])
        {
            let row = &rows.algebraic()[handle.index()];
            assert_eq!(row.lhs, expr.id());
            assert_eq!((row.lower, row.upper), expected);
        }
    }
    assert_eq!(m.arena().len(), before);
    assert!(matches!(expressions[0].into_function(), AnyScalarFunction::Affine(_)));
    assert!(matches!(expressions[1].into_function(), AnyScalarFunction::Quadratic(_)));
    assert!(matches!(expressions[2].into_function(), AnyScalarFunction::Nonlinear(_)));
    assert_eq!(m.kind(), ModelKind::NLP);
}

#[test]
fn indexed_dynamic_relations_remap_local_roots_and_keep_parameters_live() {
    let m = Model::new("dynamic indexed relations");
    variable!(m, x[i in 0..2048]);
    param!(m, p = 2.0);
    let rows = constraint!(
        m, rows[i in 0..2048], (x[i].square() + p * x[i]).erase() <= 10.0
    );
    assert_eq!(m.num_constraints(), 2048);
    assert_eq!(m.kind(), ModelKind::QCP);
    let point = vec![2.0; 2048];
    for parameter in [2.0, 5.0] {
        m.set_param(p, parameter).unwrap();
        let constraints = m.constraints();
        let arena = m.arena();
        for key in [0_usize, 1024, 2047] {
            let handle = rows.get(key).unwrap();
            let row = &constraints.algebraic()[handle.index()];
            assert_eq!(m.constraint_handle(&format!("rows[{key}]")), Some(handle));
            assert_eq!(evaluate(&arena, row.lhs, &&point[..]).unwrap(), 4.0 + 2.0 * parameter);
            assert_eq!(row.upper, 10.0);
        }
    }
}

#[test]
fn static_lowering_reuses_nodes_and_matches_macro_rows() {
    let m = Model::new("rows");
    variable!(m, x);
    variable!(m, y);
    let affine = x + 2.0 * y + 3.0;
    let quadratic = x.square() + 4.0 * x * y + affine;
    let nonlinear = x.sin() + y;
    let before = m.arena().len();
    let a = m.add_constraint("a", Constraint::new(affine.into_function(), LessThan(10.0)));
    let q = m.add_constraint("q", Constraint::new(quadratic.into_function(), EqualTo(11.0)));
    let n = m.add_constraint("n", Constraint::new(nonlinear.into_function(), GreaterThan(0.0)));
    assert_eq!(m.arena().len(), before);
    let am = constraint!(m, am, affine <= 10.0);
    let qm = constraint!(m, qm, quadratic == 11.0);
    let nm = constraint!(m, nm, nonlinear >= 0.0);
    let rows = m.constraints();
    for (direct, generated) in [(a, am), (q, qm), (n, nm)] {
        let left = &rows.algebraic()[direct.index()];
        let right = &rows.algebraic()[generated.index()];
        assert_eq!((left.lhs, left.lower, left.upper), (right.lhs, right.lower, right.upper));
    }
    let arena = m.arena();
    let terms = extract_linear(&arena, rows.algebraic()[a.index()].lhs).unwrap();
    assert_eq!(terms.constant, 3.0);
    assert_eq!(&*terms.coeffs, &[(x.var_id().unwrap(), 1.0), (y.var_id().unwrap(), 2.0)]);
    let terms = extract_quadratic(&arena, rows.algebraic()[q.index()].lhs).unwrap();
    assert!(terms.hessian.contains(&(x.var_id().unwrap(), x.var_id().unwrap(), 2.0)));
    assert_eq!(evaluate(&arena, rows.algebraic()[n.index()].lhs, &&[0.0, 2.0][..]).unwrap(), 2.0);
}

#[test]
fn symbolic_rhs_and_parameter_coefficients_remain_live() {
    let m = Model::new("parameters");
    variable!(m, x);
    param!(m, p = 2.0);
    let c = constraint!(m, row, p * x + 3.0 <= p);
    for value in [2.0, 5.0] {
        m.set_param(p, value).unwrap();
        let rows = m.constraints();
        let row = &rows.algebraic()[c.index()];
        let arena = m.arena();
        let terms = extract_linear(&arena, row.lhs).unwrap();
        assert_eq!(&*terms.coeffs, &[(x.var_id().unwrap(), value)]);
        assert_eq!(terms.constant, 3.0 - value);
        assert_eq!(row.upper, 0.0);
    }
}

#[test]
fn constant_arithmetic_agrees_with_checked_degrees_and_extraction() {
    let m = Model::new("constant arithmetic");
    variable!(m, x);
    let c = Expr::constant(x.arena(), 2.0);
    for coefficient in [-c, c + 3.0, c * c, c / 2.0, c.neg()] {
        let affine: Expr<'_, Affine> = coefficient * x;
        let quadratic: Expr<'_, Quadratic> = affine * x;
        assert_eq!(affine.__class(), ExprClass::Linear);
        assert_eq!(quadratic.__class(), ExprClass::Quadratic);
        assert!(affine.erase().try_affine().is_ok());
        assert!(quadratic.erase().try_quadratic().is_ok());
        let arena = m.arena();
        assert!(extract_linear(&arena, affine.id).is_some());
        assert!(extract_quadratic(&arena, quadratic.id).is_some());
    }
}

#[test]
fn intervals_keep_macro_split_policy_and_direct_ir_bounds() {
    let m = Model::new("intervals");
    variable!(m, x);
    assert!(matches!(constraint!(m, 0.0 <= x <= 2.0), RangeConstraintHandles::Interval(_)));
    assert!(matches!(
        constraint!(m, 0.0 <= x.square() <= 2.0),
        RangeConstraintHandles::Split { .. }
    ));
    assert!(matches!(constraint!(m, 0.0 <= x.erase() <= 2.0), RangeConstraintHandles::Interval(_)));
    assert!(matches!(
        constraint!(m, 0.0 <= x.square().erase() <= 2.0),
        RangeConstraintHandles::Split { .. }
    ));
    let c = m.add_constraint(
        "direct",
        Constraint::new(x.square().into_function(), Interval { lower: 3.0, upper: 2.0 }),
    );
    let rows = m.constraints();
    let row = &rows.algebraic()[c.index()];
    assert!(row.is_range());
    assert_eq!((row.lower, row.upper), (3.0, 2.0));
}

#[test]
fn typed_and_macro_soc_share_the_registry() {
    let m = Model::new("soc");
    variable!(m, x);
    variable!(m, t);
    let vector = VectorAffineFunction::new([t.into_function(), x.into_function()]);
    let direct =
        m.add_constraint("direct", Constraint::new(vector, SecondOrderCone { dimension: 2 }));
    let generated = soc_constraint!(m, generated, [x] <= t);
    let cones = m.soc_constraints();
    assert_eq!(cones[direct.index()].terms, cones[generated.index()].terms);
    assert_eq!(cones[direct.index()].bound, cones[generated.index()].bound);
    let _ = x.powi(1).into_affine_function();
}

#[test]
fn custom_pair_delegates_to_existing_ir() {
    struct CustomFunction<'a>(ScalarAffineFunction<'a>);
    struct CustomSet(f64);
    impl Function for CustomFunction<'_> {}
    impl ConstraintSet for CustomSet {}
    impl FunctionInSet<CustomSet> for CustomFunction<'_> {}
    impl<'a> LowerConstraint<CustomSet> for CustomFunction<'a> {
        type Ir = AlgebraicConstraintIr<'a>;
        fn lower(self, set: CustomSet) -> Self::Ir {
            Constraint::new(self.0, EqualTo(set.0)).into_ir()
        }
    }
    let m = Model::new("extension");
    variable!(m, x);
    let c = m.add_constraint(
        "custom",
        Constraint::new(CustomFunction(x.into_function()), CustomSet(4.0)),
    );
    assert_eq!(m.constraints().algebraic()[c.index()].as_single(), Some((Sense::Eq, 4.0)));
}

#[test]
fn invalid_runtime_data_cannot_register_rows() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let m = Model::new("local");
    let other = Model::new("other");
    variable!(m, x);
    variable!(other, y);
    assert!(
        catch_unwind(AssertUnwindSafe(
            || m.add_constraint("foreign", Constraint::new(y.into_function(), LessThan(1.0)))
        ))
        .is_err()
    );
    assert!(
        catch_unwind(AssertUnwindSafe(
            || m.add_constraint("nan", Constraint::new(x.into_function(), LessThan(f64::NAN)))
        ))
        .is_err()
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| m.add_constraint("foreign dynamic", y.erase().le(1.0))))
            .is_err()
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| m.add_constraint("nan dynamic", x.erase().le(f64::NAN))))
            .is_err()
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| m.add_soc_constraint(
            "nonlinear dynamic",
            [x.square().erase()],
            x.erase()
        )))
        .is_err()
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| m.add_constraint(
            "dimension",
            Constraint::new(
                VectorAffineFunction::new([x.into_function(), x.into_function()]),
                SecondOrderCone { dimension: 3 }
            )
        )))
        .is_err()
    );
    assert_eq!(m.num_constraints(), 0);
    assert_eq!(m.num_soc_constraints(), 0);
}
