use oximo_core::prelude::*;
fn main() {
    let m = Model::new("invalid");
    variable!(m, x);
    let vector = VectorAffineFunction::new([x.into_function(), x.into_function()]);
    let _ = Constraint::<VectorAffineFunction<'_>, LessThan<f64>>::new(vector, LessThan(1.0));
}
