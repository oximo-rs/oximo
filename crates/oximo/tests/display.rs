//! End-to-end checks of the human-readable model display built through the
//! macro API.

use oximo::prelude::*;

#[test]
fn model_display_matches_grammar() {
    let m = Model::new("diet");
    variable!(m, x >= 0.0);
    variable!(m, y >= 0.0);
    constraint!(m, c1, x + 2.0 * y <= 14.0);
    constraint!(m, c2, 3.0 * x - y >= 0.0);
    objective!(m, Min, 3.0 * x + 4.0 * y);

    let expected = "Model 'diet' (LP)\n\
                    min 3 x + 4 y\n\
                    s.t.\n\
                    \x20 c1: x + 2 y <= 14\n\
                    \x20 c2: 3 x - y >= 0\n\
                    vars\n\
                    \x20 x >= 0\n\
                    \x20 y >= 0\n";
    assert_eq!(m.to_string(), expected);
}

#[test]
fn display_covers_range_soc_and_domains() {
    let m = Model::new("mix");
    variable!(m, x >= 0.0);
    variable!(m, y, Bin);
    variable!(m, t >= 0.0);
    constraint!(m, band, 1.0 <= x + t <= 4.0);
    soc_constraint!(m, disk, [x, t] <= x + 1.0);
    objective!(m, Max, x + y);

    let out = m.to_string();
    assert!(out.contains("max x + y\n"), "{out}");
    assert!(out.contains("  band: 1 <= x + t <= 4\n"), "{out}");
    assert!(out.contains("  disk: ||x, t|| <= x + 1\n"), "{out}");
    assert!(out.contains("  0 <= y <= 1, binary\n"), "{out}");
}

#[test]
fn display_expr_and_constraint_adapters() {
    let m = Model::new("adapters");
    variable!(m, x);
    variable!(m, y);
    constraint!(m, c, x * y - y <= 3.0);
    let c = m.constraint_id("c").unwrap();
    assert_eq!(m.display_constraint(c).to_string(), "c: -y + x * y <= 3");
    assert_eq!(m.display_expr(2.0 * x - y).to_string(), "2 x - y");
}

#[test]
fn params_render_current_binding() {
    let m = Model::new("params");
    variable!(m, x >= 0.0);
    param!(m, price = 4.0);
    objective!(m, Min, price * x);

    let out = m.to_string();
    assert!(out.contains("min 4 x\n"), "{out}");
    assert!(out.contains("params\n  price = 4\n"), "{out}");

    m.set_param(price, 7.5);
    let out = m.to_string();
    assert!(out.contains("min 7.5 x\n"), "{out}");
    assert!(out.contains("params\n  price = 7.5\n"), "{out}");
}
