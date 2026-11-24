use oximo_core::prelude::*;
fn main() {
    let m = Model::new("invalid rotated SOC syntax");
    variable!(m, x);
    variable!(m, u);
    variable!(m, v);
    // The old tuple head is no longer a rotated cone.
    soc_constraint!(m, tuple, [x] <= (u, v));
    soc_constraint!(m, unknown, [x] in cone(u, v));
    soc_constraint!(m, arity, [x] in rotated(u));
    soc_constraint!(m, soc_arity, [x] in soc(u, v));
}
