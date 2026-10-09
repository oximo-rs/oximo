use oximo_gdp::prelude::*;
fn main() {
    let model = Model::new("selectors");
    boolean_variable!(model, a);
    boolean_variable!(model, b);
    disjunction!(model, [a, b], ExactlyOne, ExactlyOne);
    disjunction!(model, choice, [a, b], AtLeastOne, AtLeastOne);
    disjunction!(model, [a, b], ExactlyOne, AtLeastOne);
    disjunction!(model, choice, [a, b], AtLeastOne, ExactlyOne);
    disjunction!(model, [a, b], parent = a, parent = b);
}
