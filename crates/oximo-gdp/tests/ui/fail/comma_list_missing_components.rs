#![allow(unused_variables)]

use oximo_gdp::prelude::*;

fn main() {
    let model = Model::new("missing_gdp_entries");
    boolean_variable!(model, selected[i in 0..2, j in 0..2]);
    variable!(model, x[i in 0..2, j in 0..2]);

    boolean_variable!(model,, flag);
    disjunct!(model, |_d| {},,);
    disjunction!(model,, [selected[(0, 0)]]);
    logical_constraint!(model, selected[0,,1]);
    disjunct_constraint!(model, selected[0,,1], x[0,1] <= 1.0);
    disjunct_constraint!(model, selected[0,1], x[0,,1] <= 1.0);
    disjunction!(model, [selected[0,0]], parent = selected[0,,1]);
}
