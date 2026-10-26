#![allow(unused_imports, unused_variables, dead_code)]

use oximo_core::prelude::*;

fn main() {
    let model = Model::new("contract");
    variable!(model, x);
    let row = constraint!(model, x <= 1.0);
    let _ = row.row_id();
    let erased = row.erase();
    let checked = model.require_scalar_dual_handle(erased).unwrap();
    let _ = checked.row_id();
    let cone = soc_constraint!(model, [x] <= x);
    let _ = cone.soc_id();
    assert!(model.scalar_dual_handle(cone).unwrap().is_none());
    let _ = erased.try_row_id();
}
