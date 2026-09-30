use oximo_core::prelude::*;
fn main() {
    let m = Model::new("invalid indicator range");
    variable!(m, b, Binary);
    variable!(m, x);
    indicator_constraint!(m, bad, b == 1 => 0.0 <= x.square() <= 1.0);
}
