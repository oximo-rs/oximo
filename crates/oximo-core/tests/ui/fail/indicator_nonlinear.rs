use oximo_core::prelude::*;
fn main() {
    let m = Model::new("invalid indicator");
    variable!(m, b, Binary);
    variable!(m, x);
    m.add_indicator_constraint("bad", b, true, x.sin().le(1.0));
}
