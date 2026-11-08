use oximo_core::prelude::*;

fn main() {
    let model = Model::new("degree check");
    variable!(model, x);
    model.quadratic_builder().add_product(1.0, x, &x.sin()).unwrap();
}
