#![allow(unused_variables)]

use oximo_core::prelude::*;

fn main() {
    let model = Model::new("missing_components");
    variable!(model, pair[i in 0..2, j in 0..2]);
    variable!(model, binary[i in 0..2, j in 0..2], Bin);
    let weights = &SymmetricMatrix::from_upper_triangle(2, [1.0, 2.0, 3.0]);
    let (i, j) = (0, 1);

    objective!(model, Min, pair[,i,j]);
    objective!(model, Min, pair[i,,j]);
    objective!(model, Min, pair[i,j,,]);
    objective!(model, Min, pair[0,,1]);
    constraint!(model, pair[i,,j] <= 1.0);
    soc_constraint!(model, [pair[i,,j]] <= pair[i,j] + 1.0);
    sos_constraint!(model, SOS1, [pair[i,,j]]);
    sos_constraint!(model, SOS1, [(pair[i,,j], 1.0)]);
    sos_constraint!(model, SOS1, [(pair[i,j], weights[i,,j])]);
    param!(model, value = weights[i,,j]);
    param!(model, values[k in 0..2] = weights[k,,j]);
    variable!(model, lower >= weights[i,,j]);
    variable!(model, upper <= weights[i,,j]);
    variable!(model, started, initial = weights[i,,j]);
    variable!(model, fixed, fix = weights[i,,j]);
    indicator_constraint!(model, binary[i,,j] == 1 => pair[i,j] <= 1.0);
    psd_constraint!(model, SymmetricMatrix::from_upper_triangle(1, [pair[i,,j]]));
    let _ = sum!(pair[i,,j] for i in 0..2);
    let _ = min!(pair[i,,j] for i in 0..2);
    let _ = max!(pair[i,,j] for i in 0..2);
    objective!(model, Min, sum!(pair[i,,j] for i in 0..2));
}
