use oximo_gdp::prelude::*;

fn main() {
    let m = Model::new("indexed parents");
    boolean_variable!(m, p[i in 0..2, j in 0..2]);
    boolean_variable!(m, a[i in 0..2, j in 0..2]);
    boolean_variable!(m, b[i in 0..2, j in 0..2]);
    let choices = disjunction!(m, choice[i in 0..2, j in 0..2 if i == j],
        [a[i, j], b[i, j]], parent = p[i, j]);
    assert_eq!(choices.len(), 2);
    assert_eq!(m.gdp().disjunctions[0].parent, Some(p[(0, 0)].id()));
    assert_eq!(m.gdp().disjunctions[1].parent, Some(p[(1, 1)].id()));

    // Anonymous names still belong to the parent scope and the parent is
    // evaluated once per declaration.
    let calls = std::cell::Cell::new(0);
    disjunction!(m, [a[0, 1], b[0, 1]], parent = {
        calls.set(calls.get() + 1);
        p[0, 1]
    });
    assert_eq!(calls.get(), 1);
    assert_eq!(m.gdp().disjunctions[2].name, "p[0,1]::_gdp0");
    m.reformulate_gdp(GdpReformulationOptions::default()).unwrap();

    let m = Model::new("single index parents");
    boolean_variable!(m, p[i in 0..2]);
    boolean_variable!(m, a[i in 0..2]);
    boolean_variable!(m, b[i in 0..2]);
    let choices = disjunction!(m, choice[i in 0..2], [a[i], b[i]], parent = p[i]);
    assert_eq!(choices.len(), 2);
    assert_eq!(m.gdp().disjunctions[0].parent, Some(p[0].id()));
    assert_eq!(m.gdp().disjunctions[1].parent, Some(p[1].id()));
    m.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
}
