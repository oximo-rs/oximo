#![allow(unused_imports, unused_variables, dead_code)]

use oximo_core::prelude::*;

fn main() {
    let model = Model::new("contract");
    variable!(model, x);
    let id = GeneratedVarId::from_raw(VarId(0));
    let _ = model.variable_handle(id);
}
