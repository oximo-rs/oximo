#![allow(unused_imports, unused_variables, dead_code)]

use oximo_core::prelude::*;

fn main() {
    let model = Model::new("contract");
    variable!(model, x);
    variable!(model, b, Bin);
    indicator_constraint!(model, rows[i in 0..2], b == 1 => 0.0 <= x.powi(i as i32) <= 1.0);
}
