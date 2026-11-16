use oximo_core::prelude::*;

struct Input<'a>(Expr<'a, Affine>);

impl<'a> ExprInput<'a> for Input<'a> {
    type Degree = Affine;
    fn as_expression(&self) -> &Expr<'a, Affine> {
        &self.0
    }
}

struct Weight;

impl<'a> std::ops::Mul<VariableHandle<'a>> for Weight {
    type Output = Input<'a>;
    fn mul(self, x: VariableHandle<'a>) -> Self::Output {
        Input(2.0 * x)
    }
}

fn main() {
    let m = Model::new("custom expression input");
    variable!(m, x);
    let input = Input(x + 1.0);
    let _: Expr<'_, Affine> = sum!(m, &input for _i in 0..2);
    let _: Expr<'_, Affine> = sum!(m, &input for _i in 0..2 if true);
    let _: Expr<'_, Affine> = sum!(m, &input for _i in 0..2, _j in 0..2);
    let _: Expr<'_, Affine> = sum!(m, weighted_term(2, &input) for _i in 0..2);
    let _: Expr<'_, Affine> = sum!(m, Weight * x for _i in 0..2);
    let mut builder = m.affine_builder();
    builder.add_term(2.0, &input);
    let _: Expr<'_, Affine> = builder.finish();
}
