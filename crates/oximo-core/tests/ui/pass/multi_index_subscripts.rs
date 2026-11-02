use oximo_core::prelude::*;

fn main() {
    let model = Model::new("multi_index_subscripts");
    variable!(model, pair[i in 0..2, j in 0..2]);
    variable!(model, triple[i in 0..2, j in 0..2, k in 0..2]);
    let (i, j, k) = (0, 1, 0);
    let weights = [1.0, 2.0];
    let matrix = [[1.0, 2.0], [3.0, 4.0]];

    objective!(model, Min, pair[i,j,] + triple[i,j,k,]);
    objective!(model, Min, pair[&i,j,] + pair[&i,&j,]);
    objective!(model, Min, pair[(i,j)] + pair[(&i,&j)]);
    objective!(model, Min, weights[i] * pair[i,j] + matrix[i][j] * pair[i,j]);
    objective!(model, Min, [1.0, 2.0,][i] * pair[i,j]);
    objective!(model, Min, vec![1.0, 2.0,][i] * pair[i,j]);
    constraint!(model, pair[i,j,] <= 10.0);
    objective!(model, Min, sum!(pair[i,j,] for i in 0..2, j in 0..2));
}
