#![allow(unused_variables)]

use oximo_core::prelude::*;

fn main() {
    let model = Model::new("missing_list_entries");
    variable!(model, x);
    variable!(model, y);
    variable!(model, active, Bin);
    let matrix = SymmetricMatrix::from_upper_triangle(1, [x]);

    variable!(,model, z);
    variable!(model, z,, lb = 0.0);
    variable!(model, z,,);
    param!(,model, limit = 1.0);
    param!(model,, limit = 1.0);
    param!(model, limit = 1.0,,);

    soc_constraint!(model, [,x,y] <= x + 1.0);
    soc_constraint!(model, [x,,y] <= x + 1.0);
    soc_constraint!(model, [x,y,,] <= x + 1.0);
    soc_constraint!(model, [] <= x + 1.0);
    sos_constraint!(model, SOS1, [,x,y]);
    sos_constraint!(model, SOS1, [x,,y]);
    sos_constraint!(model, SOS1, [x,y,,]);
    sos_constraint!(model, SOS1, []);
    sos_constraint!(model, SOS1, [(,1.0)]);
    sos_constraint!(model, SOS1, [(x,,1.0)]);
    sos_constraint!(model, SOS1, [(x,1.0,,)]);

    constraint!(model,, x <= 1.0);
    indicator_constraint!(model,, active == 1 => y <= 1.0);
    psd_constraint!(model,, &matrix);
    symmetric_variable!(model, matrix_var[2],,);
    objective!(model, Min, x,,);
    objective!(model, Feasibility,,);
}
