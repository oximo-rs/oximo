use oximo_core::prelude::*;
fn main() {
    let m = Model::new("invalid dynamic SOC");
    variable!(m, x);
    let function = x.square().erase().le(1.0).function;
    let _ = VectorAffineFunction::new([function]);
}
