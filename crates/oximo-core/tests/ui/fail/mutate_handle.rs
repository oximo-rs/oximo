use oximo_core::prelude::*;
fn main() {
    let m = Model::new("invalid");
    variable!(m, x);
    let mut expr = x;
    expr.id = (x * x).id;
}
