#![allow(non_snake_case)]
use oximo_core::prelude::*;
fn main() {
    let m = Model::new("sdp");
    symmetric_variable!(m, X[2]);
    let _cone = psd_constraint!(m, positivity, &X);
    constraint!(m, X[0, 1] == 1.0);
    objective!(m, Min, X.trace());
}
