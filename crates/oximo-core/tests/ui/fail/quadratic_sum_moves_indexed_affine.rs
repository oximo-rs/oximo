use oximo_core::prelude::*;

fn main() {
    let m = Model::new("consuming quadratic factors");
    variable!(m, x);
    let affine = [x + 1.0, x + 2.0];
    let _ = sum!(m, 2.0 * affine[i] * x for i in 0..2);
}
