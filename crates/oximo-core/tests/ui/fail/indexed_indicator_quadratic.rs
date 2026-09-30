use oximo_core::prelude::*;
fn main() {
    let m = Model::new("invalid indexed indicator");
    variable!(m, b[i in 0..2], Binary);
    variable!(m, x[i in 0..2]);
    indicator_constraint!(m, bad[i in 0..2], b[i] == 1 => x[i].square() <= 1.0);
}
