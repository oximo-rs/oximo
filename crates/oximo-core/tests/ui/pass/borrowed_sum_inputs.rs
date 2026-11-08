use oximo_core::prelude::*;

fn main() {
    let m = Model::new("borrowed sums");
    variable!(m, x);
    let terms = [x + 1.0, x + 2.0];
    let _: Expr<'_, Affine> = sum!(m, &terms[i] for i in 0..2);
    let _: Expr<'_, Affine> = sum!(m, &terms[i] for i in 0..2 if true);
    let _: Expr<'_, Affine> = sum!(&terms[i] for i in 0..2);
    let _: Expr<'_, Affine> = sum!(&terms[i] for i in 0..2 if true);
    let _: Expr<'_, Affine> = sum!(m, &terms[i] for i in 0..0);
    let _: Expr<'_, Affine> = sum!(m, &terms[i] for i in 0..2 if false);
}
