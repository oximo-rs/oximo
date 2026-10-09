use algebra::prelude::*;
fn main() {
    let m = Model::new("renamed");
    variable!(m, 0.0 <= x <= 5.0);
    boolean_variable!(m, y[i in 0..2]);
    disjunct_constraint!(m, y[i], row[i in 0..2], x <= 2.0);
    disjunction!(m, [y[0], y[1]]);
    m.reformulate_gdp(BigM::default()).unwrap();
}
