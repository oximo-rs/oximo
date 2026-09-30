use oximo_core::prelude::*;
fn main() {
    let m = Model::new("invalid indicator");
    variable!(m, b, Binary);
    variable!(m, x);
    indicator_constraint!(m, bad, b == 1 => x.square() <= 1.0);
}
