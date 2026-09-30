use oximo_core::prelude::*;
fn main() {
    let m = Model::new("valid");
    variable!(m, x);
    let a: ScalarAffineFunction = (x + 2.0).into_function();
    let q: ScalarQuadraticFunction = (x * x).into_function();
    m.add_constraint("le", Constraint::new(a, LessThan(4.0)));
    m.add_constraint("eq", Constraint::new(a, EqualTo(3.0)));
    m.add_constraint("q", Constraint::new(q, LessThan(4.0)));
    let v = VectorAffineFunction::new([a, a]);
    m.add_constraint("cone", Constraint::new(v, SecondOrderCone { dimension: 2 }));
    constraint!(m, x.powi(2) <= 4.0);
    variable!(m, b, Binary);
    indicator_constraint!(m, b == 1 => x <= 4.0);
    indicator_constraint!(m, b == 0 => x.erase() >= 0.0);
    let checked = x.nonlinear().try_affine().unwrap();
    indicator_constraint!(m, checked_row, b == 1 => checked <= 1.0);
}
