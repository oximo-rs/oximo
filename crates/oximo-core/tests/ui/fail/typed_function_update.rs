use oximo_core::prelude::*;
fn main() {
    let m = Model::new("typed update");
    variable!(m, x);
    let c = constraint!(m, x <= 1.0);
    m.set_constraint_function(c, Function::semantic_data(&x.square().into_function())).unwrap();
}
