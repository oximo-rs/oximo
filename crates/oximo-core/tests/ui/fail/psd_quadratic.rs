use oximo_core::prelude::*;
fn main() {
    let m = Model::new("invalid");
    variable!(m, x);
    psd_constraint!(m, SymmetricMatrix::from_upper_triangle(1, [x.square()]));
}
