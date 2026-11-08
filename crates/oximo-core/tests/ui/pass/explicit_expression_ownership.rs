use oximo_core::prelude::*;

fn main() {
    let m = Model::new("explicit sharing");
    variable!(m, x);
    variable!(m, y);
    let e = x + y;
    let a = &e + x;
    let b = e + y;
    objective!(m, Min, &a + &b);
    constraint!(m, shared, &a <= 4.0);
    constraint!(m, consumed, a + b <= 8.0);

    let terms = [x + 1.0, x + 2.0];
    let _ = sum!(m, &terms[i] for i in 0..2);
    let _ = min!(&terms[i] for i in 0..2);
}
