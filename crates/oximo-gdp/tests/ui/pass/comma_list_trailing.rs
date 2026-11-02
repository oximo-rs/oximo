use oximo_gdp::prelude::*;

fn main() {
    let model = Model::new("gdp_trailing_commas");
    boolean_variable!(model, selected[i in 0..2, j in 0..2,],);
    variable!(model, 0.0 <= x <= 2.0,);
    logical_constraint!(model, selected[0,1,],);
    disjunct_constraint!(model, selected[0,1,], x <= 1.0,);
    let branch = disjunct!(model, |_d| {},);
    disjunction!(model, [branch], ExactlyOne,);
    disjunction!(model, [selected[0,0,]], parent = selected[0,1,],);
    assert_eq!(selected.len(), 4);
    assert_eq!(model.gdp().disjunctions.len(), 2);
}
