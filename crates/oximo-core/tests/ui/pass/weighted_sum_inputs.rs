use oximo_core::prelude::*;

fn main() {
    let m = Model::new("weighted input types");
    variable!(m, x);
    param!(m, p = 2.0);
    let weights = [1.0, 2.0];
    let integers = [1, 2];
    let affine = x + 1.0;
    let _: Expr<'_, Affine> = sum!(m, weights[i] * x for i in 0..2);
    let _: Expr<'_, Affine> = sum!(m, integers[i] * x for i in 0..2);
    let _: Expr<'_, Affine> = sum!(m, 2 * &affine for _i in 0..2);
    let _: Expr<'_, Affine> = sum!(m, &(x + 1.0) * 2.0 for _i in 0..2);
    let _: Expr<'_, Constant> = sum!(m, 2.0 * p for _i in 0..2);
    let _: Expr<'_, Constant> = sum!(m, 2 * m.__constant(3.0) for _i in 0..2);
    let _: Expr<'_, Affine> = sum!(m, unfused: weights[i] * &affine for i in 0..2);
    let _: Expr<'_, Affine> = sum!(unfused: weights[i] * x for i in 0..2);
    let _: Expr<'_, Constant> = sum!(m, 2 * p for _i in 0..0);
    let mut owned = Some(x + 1.0);
    let _: Expr<'_, Affine> = sum!(m, 2.0 * owned.take().unwrap() for _i in 0..1);
    let unfused = &m;
    let _: Expr<'_, Affine> = sum!(unfused, 2.0 * x for _i in 0..2);
    mod unfused { pub const WEIGHT: f64 = 2.0; }
    let _: Expr<'_, Affine> = sum!(unfused::WEIGHT * x for _i in 0..2);
    let _: Expr<'_, Affine> = sum!(m, 2.0 * std::convert::identity(&(x + 1.0)) for _i in 0..2);
    let _: Expr<'_, Affine> = sum!(m, std::convert::identity(&(x + 1.0)) * 2.0 for _i in 0..2);
    let _: Expr<'_, Constant> = sum!(m, weighted_term(2.0, p) for _i in 0..2);
    let _: Expr<'_, Affine> = sum!(m, weighted_term(2.0, &affine) for _i in 0..2);
}
