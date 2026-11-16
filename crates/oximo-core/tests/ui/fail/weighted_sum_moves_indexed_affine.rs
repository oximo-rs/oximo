use oximo_core::prelude::*;

fn main() {
    let m = Model::new("normal indexed ownership");
    variable!(m, x);
    let terms = [x + 1.0, x + 2.0];
    let _ = sum!(m, 2.0 * terms[i] for i in 0..2);
}
