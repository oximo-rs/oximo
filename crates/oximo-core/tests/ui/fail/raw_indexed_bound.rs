#![allow(unused_imports, unused_variables, dead_code)]

use oximo_core::prelude::*;

fn main() {
    let model = Model::new("contract");
    variable!(model, x);
    variable!(model, y[i in 0..2] <= Bound::Constant(i as f64));
}
