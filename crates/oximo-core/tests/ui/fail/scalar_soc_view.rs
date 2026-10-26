#![allow(unused_imports, unused_variables, dead_code)]

use oximo_core::prelude::*;

fn main() {
    let model = Model::new("contract");
    variable!(model, x);
    let row = constraint!(model, x <= 1.0);
    let _ = row.soc_id();
}
