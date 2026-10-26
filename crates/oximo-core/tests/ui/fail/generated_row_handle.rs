#![allow(unused_imports, unused_variables, dead_code)]

use oximo_core::prelude::*;

fn main() {
    let model = Model::new("contract");
    variable!(model, x);
    let id = GeneratedConstraintId::from_raw(ConstraintId(0));
    let _ = model.constraint_handle_from_id(id);
}
