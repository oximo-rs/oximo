use oximo_gdp::prelude::*;

fn main() {
    let model = Model::new("scalar tuple indicators");
    variable!(model, 0.0 <= x[i in 0..2, j in 0..2] <= 10.0);
    boolean_variable!(model, selected[i in 0..2, j in 0..2]);
    param!(model, lower = 0.5);
    param!(model, upper = 3.5);

    disjunct_constraint!(model, selected[0, 1], x[0, 1] <= 2.0);
    disjunct_constraint!(model, selected[1, 0], capacity, x[1, 0] <= 3.0);
    disjunct_constraint!(model, selected[1, 1], name = "floor", x[1, 1] >= 1.0);
    disjunct_constraint!(model, selected[0, 0], 0.0 <= x[0, 0] <= 4.0);
    disjunct_constraint!(model, selected[0, 1], range, 0.0 <= x[0, 1] <= 3.0);
    disjunct_constraint!(model, selected[1, 0], lower <= x[1, 0] <= upper);

    let expected = [
        selected[(0, 1)].id(), selected[(1, 0)].id(), selected[(1, 1)].id(),
        selected[(0, 0)].id(), selected[(0, 1)].id(),
        selected[(1, 0)].id(), selected[(1, 0)].id(),
    ];
    assert_eq!(model.gdp().rows.iter().map(|row| row.indicator).collect::<Vec<_>>(), expected);
    model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
}
