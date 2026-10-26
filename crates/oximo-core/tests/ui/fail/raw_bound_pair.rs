#![allow(unused_imports, unused_variables, dead_code)]

use oximo_core::prelude::*;

fn main() {
    let model = Model::new("contract");
    variable!(model, x);
    model.__var("y").bounds(Bound::Constant(0.0), 1.0).build();
}
