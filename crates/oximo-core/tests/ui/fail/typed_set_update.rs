use oximo_core::prelude::*;
fn main() {
    let m = Model::new("typed update");
    variable!(m, x);
    let c = constraint!(m, x <= 1.0);
    m.set_constraint_set(c, GreaterThan(0.0)).unwrap();
}
