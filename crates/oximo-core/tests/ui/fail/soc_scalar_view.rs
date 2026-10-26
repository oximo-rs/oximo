#![allow(unused_imports, unused_variables, dead_code)]

use oximo_core::prelude::*;

fn main() {
    let model = Model::new("contract");
    variable!(model, x);
    let cone = soc_constraint!(model, [x] <= x);
    let _ = cone.row_id();
}
