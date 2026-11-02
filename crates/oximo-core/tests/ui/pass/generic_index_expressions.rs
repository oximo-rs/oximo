use oximo_core::prelude::*;

fn index<A, B>() -> usize {
    usize::from(std::mem::size_of::<A>() > std::mem::size_of::<B>())
}

fn main() {
    let model = Model::new("generic_index_expressions");
    let weights = [2.0];
    param!(model, weight = weights[index::<u8, u16>()]);
    variable!(model, x);
    variable!(model, pair[i in 0..1, j in 0..1]);

    constraint!(model, x <= weight);
    sos_constraint!(model, SOS1, [(x, weights[index::<u8, u16>()])]);
    objective!(model, Min, pair[index::<u8, u16>(), index::<u8, u16>(),]);

    assert_eq!(model.sos_constraints()[0].members[0].weight, 2.0);
}
