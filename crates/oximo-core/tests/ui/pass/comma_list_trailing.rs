use oximo_core::prelude::*;

fn main() {
    let model = Model::new("trailing_commas");
    let weights = SymmetricMatrix::from_upper_triangle(2, [1.0, 2.0, 3.0]);
    let weights = &weights;
    variable!(model, x, lb = 0.0, ub = weights[1,1,], initial = weights[0,0,],);
    variable!(model, y, fix = weights[0,1,],);
    variable!(model, z >= weights[0,0,],);
    variable!(model, active, Bin,);
    param!(model, limit = weights[0,1,],);
    param!(model, limits[i in 0..2, j in 0..2,] = weights[i,j,],);

    constraint!(model, x + y + z <= limit,);
    constraint!(model, cap, x <= limits[0,1,],);
    indicator_constraint!(model, active == 1 => x <= limit,);
    soc_constraint!(model, [x, y,] <= x + 10.0,);
    soc_constraint!(model, cone, [x,] <= x + 10.0,);
    sos_constraint!(model, SOS1, [x, y,],);
    sos_constraint!(model, weighted, SOS2, [(x, weights[0,0,],), (y, weights[0,1,],),],);
    symmetric_variable!(model, matrix[1],);
    psd_constraint!(model, &matrix,);
    psd_constraint!(model, positive, &matrix,);
    objective!(model, Min, x + y,);
    objective!(model, Feasibility,);

    assert_eq!(model.num_sos_constraints(), 2);
    assert_eq!(
        model.sos_constraints()[1].members.iter().map(|member| member.weight).collect::<Vec<_>>(),
        [1.0, 2.0],
    );
    assert_eq!(limits.len(), 4);
}
