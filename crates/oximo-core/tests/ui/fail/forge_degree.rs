use oximo_core::prelude::*;
fn main() {
    let m = Model::new("invalid");
    variable!(m, x);
    let _: Expr<'_, Affine> = x * x;
}
