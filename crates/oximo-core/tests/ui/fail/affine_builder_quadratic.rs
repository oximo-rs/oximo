use oximo_core::prelude::*;
fn main() {
    let m = Model::new("invalid builder term");
    variable!(m, x);
    m.affine_builder().add_term(1.0, x.square());
}
