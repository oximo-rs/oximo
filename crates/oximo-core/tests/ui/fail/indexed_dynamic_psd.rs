#![allow(unused_imports, unused_variables, dead_code)]

use oximo_core::prelude::*;

fn main() {
    let model = Model::new("contract");
    variable!(model, x);
    psd_constraint!(model, cones[i in 0..2], SymmetricMatrix::from_upper_triangle(1, [x.powi(i as i32)]));
}
