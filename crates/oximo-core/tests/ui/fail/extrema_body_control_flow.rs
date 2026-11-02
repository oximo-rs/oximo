use oximo_core::prelude::*;

fn main() {
    let model = Model::new("extrema_control_flow");
    variable!(model, x);
    let _ = min!({ break; x } for i in 0..3);
    let _ = max!({ continue; x } for i in 0..3);
    let _ = min!({ break; x } for i in 0..3 if i > 0);
    let _ = max!({ continue; x } for i in 0..3 if i > 0);
}
