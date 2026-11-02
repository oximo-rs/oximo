use oximo_core::prelude::*;

macro_rules! coefficient {
    ($($input:tt)*) => { 2.0 };
}

macro_rules! with_syntax {
    (syntax[$($syntax:tt)*], $expr:expr) => { $expr };
}

fn main() {
    let model = Model::new("macro_input_subscripts");
    variable!(model, pair[i in 0..2, j in 0..2]);

    objective!(model, Min, coefficient!(data[0,,1]) * pair[0,1]);
    objective!(model, Min, coefficient![data[,0,1]] * pair[0,1]);
    objective!(model, Min, coefficient! { data[0,1,,] } * pair[0,1]);
    objective!(model, Min, with_syntax!(syntax[0,,1], pair[0,1]));
    objective!(model, Min, with_syntax!(syntax[0,,1], sum!(pair[i,j] for i in 0..2, j in 0..2)));
    objective!(model, Min, sum!(coefficient!(data[i,,j]) * pair[i,j] for i in 0..2, j in 0..2));
}
