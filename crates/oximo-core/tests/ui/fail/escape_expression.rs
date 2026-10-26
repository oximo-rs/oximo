#![allow(unused_imports, unused_variables, dead_code)]

use oximo_core::prelude::*;

fn main() {
    let escaped;
    {
        let model = Model::new("temporary");
        variable!(model, x);
        escaped = x;
    }
    let _ = escaped + 1.0;
}
