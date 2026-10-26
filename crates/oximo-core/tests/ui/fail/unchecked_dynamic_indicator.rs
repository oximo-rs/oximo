#![allow(unused_imports, unused_variables, dead_code)]

use oximo_core::prelude::*;

fn main() {
    let model = Model::new("contract");
    variable!(model, x);
    variable!(model, b, Bin);
    indicator_constraint!(model, b == 1 => x.erase() <= 1.0);
}
