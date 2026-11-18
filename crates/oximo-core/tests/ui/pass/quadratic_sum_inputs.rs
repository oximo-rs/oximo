use oximo_core::prelude::*;

fn main() {
    let m = Model::new("quadratic inputs");
    variable!(m, x[i in 0..2]);
    param!(m, p = 2.0);
    let affine = [x[0] + 1.0, x[1] + 2.0];
    let _: Expr<'_, Quadratic> = sum!(m, 2.0 * &affine[i] * x[i] for i in 0..2);
    let _: Expr<'_, Quadratic> = sum!(m, x[i] * x[i] * 2 for i in 0..2);
    let _: Expr<'_, Quadratic> = sum!(m, 2 * (x[i] * x[i]) for i in 0..2);
    let _: Expr<'_, Quadratic> =
        sum!(m, std::convert::identity(&(x[i] + 1.0)) * x[i] for i in 0..2);
    let _: Expr<'_, Affine> = sum!(m, 2 * p * x[i] for i in 0..2);
    let _: Expr<'_, Constant> = sum!(m, 2 * p * p for _i in 0..0);
    let _: Expr<'_, Nonlinear> = sum!(m, x[i] * x[i] * x[i] for i in 0..2);
    let _: Expr<'_, Dynamic> = sum!(m, 2 * x[i].as_dynamic() * x[i] for i in 0..2);
    let _: Expr<'_, Quadratic> = sum!(m, unfused: 2 * x[i] * x[i] for i in 0..2);
}
