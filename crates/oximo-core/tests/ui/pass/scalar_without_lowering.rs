use oximo_core::prelude::*;
use oximo_core::function_set::ScalarFunction;

#[derive(Clone, Copy)]
struct Scalar<'a>(Expr<'a, Affine>);

impl Function for Scalar<'_> {
    type Owned = ScalarFunctionData<Affine>;
    fn semantic_data(&self) -> Self::Owned {
        ScalarFunctionData::from_expression(self.0)
    }
}
impl FunctionInSet<LessThan> for Scalar<'_> {}
impl FunctionInSet<GreaterThan> for Scalar<'_> {}
impl FunctionInSet<EqualTo> for Scalar<'_> {}
impl FunctionInSet<Interval> for Scalar<'_> {}

// A scalar function needs no LowerConstraint implementation.
impl<'a> ScalarFunction<'a> for Scalar<'a> {
    fn expression(self) -> Expr<'a> { self.0.erase() }
    fn is_affine(self) -> bool { true }
}

fn register<'a, F: ScalarFunction<'a>>(model: &Model, function: F) {
    assert!(function.is_affine());
    let row = model.__add_constraint("row", Constraint::new(function, LessThan(2.0)));
    assert_eq!(model.constraints().algebraic()[row.index()].lhs, function.expression().id());
}

fn main() {
    let model = Model::new("scalar without lowering");
    variable!(model, x);
    register(&model, Scalar(x));
}
