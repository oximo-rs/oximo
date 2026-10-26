#![allow(unused_imports, unused_variables, dead_code)]

use oximo_core::prelude::*;

fn main() {
    let model = Model::new("contract");
    variable!(model, x);
    variable!(model, b, Bin);
    let affine = x.erase().try_affine().unwrap();
    soc_constraint!(model, cones[i in 0..2], [affine + i as f64] <= x);
    psd_constraint!(model, cones_psd[i in 0..2], SymmetricMatrix::from_upper_triangle(1, [affine + i as f64]));
    indicator_constraint!(model, rows[i in 0..2], b == 1 => affine <= i as f64);
    indicator_constraint!(model, bands[i in 0..2], b == 1 => 0 <= affine <= i as f64);
    variable!(model, bounded[i in 0..2], lb = 0, ub = i + 1);
    param!(model, limit = 2.0);
    variable!(model, capped <= limit);
}
