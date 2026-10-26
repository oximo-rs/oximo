#![allow(unused_imports, unused_variables, dead_code)]

use oximo_core::prelude::*;

fn main() {
    let model = Model::new("contract");
    variable!(model, x);
    psd_constraint!(model, SymmetricMatrix::from_upper_triangle(1, [x.sin()]));
}
