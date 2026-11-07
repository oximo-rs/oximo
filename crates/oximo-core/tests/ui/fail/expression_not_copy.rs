use oximo_core::prelude::*;

fn main() {
    let model = Model::new("ownership");
    variable!(model, x);
    let quadratic = x * x;
    let first = quadratic;
    let alias = quadratic;
    drop((first, alias));
}
