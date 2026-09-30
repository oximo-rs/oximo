use oximo_core::prelude::*;
fn main() {
    let m = Model::new("invalid");
    variable!(m, x);
    let affine: ScalarAffineFunction = x.into_function();
    let _ = Constraint::new(affine, SecondOrderCone { dimension: 2 });
}
