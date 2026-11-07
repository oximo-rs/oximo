use oximo_core::prelude::*;

fn require_copy<T: Copy>(_: T) {}

fn main() {
    let model = Model::new("only leaves copy");
    variable!(model, x);
    param!(model, p = 2.0);
    require_copy(x);
    require_copy(p);
    require_copy(x + 1.0);
    require_copy(p + 1.0);
}
