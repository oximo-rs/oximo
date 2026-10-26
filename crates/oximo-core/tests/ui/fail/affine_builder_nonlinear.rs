use oximo_core::prelude::*;
fn main() {
    let model = Model::new("nonlinear builder");
    variable!(model, x);
    model.affine_builder().add_term(1.0, x.sin());
}
