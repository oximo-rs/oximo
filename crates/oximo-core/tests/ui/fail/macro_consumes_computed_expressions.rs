use oximo_core::prelude::*;

fn main() {
    let m = Model::new("normal ownership");
    variable!(m, x);
    variable!(m, y);
    let e = x + y;
    objective!(m, Min, e + x);
    let _ = e + y;

    let f = x + y;
    objective!(m, Min, f);
    let _ = f + x;
}
