//! Snapshot tests for the AMPL `.nl` writer.
//!
//! Build small models, render to a string, and compare to an expected literal.
//! Covers LP, MILP, NLP, MINLP.

use oximo_core::prelude::*;
use oximo_io::to_nl_string;

#[test]
fn nl_pure_lp() {
    // min  x + 2y
    // s.t. x + y >= 3
    // x, y free
    let m = Model::new("tinylp");
    let x = m.var("x").build();
    let y = m.var("y").build();
    m.minimize(x + 2.0 * y);
    m.constraint("c0", (x + y).ge(3.0));

    let s = to_nl_string(&m).expect("nl writer");
    let expected = "\
g3 1 1 0\t# problem tinylp
 2 1 1 0 0\t# vars, constraints, objectives, ranges, eqns
 0 0\t# nonlinear constraints, objectives
 0 0\t# network constraints: nonlinear, linear
 0 0 0\t# nonlinear vars in constraints, objectives, both
 0 0 0 1\t# linear network variables; functions; arith, flags
 0 0 0 0 0\t# discrete variables: binary, integer, nonlinear (b,c,o)
 2 2\t# nonzeros in Jacobian, gradients
 0 0\t# max name lengths: constraints, variables
 0 0 0 0 0\t# common exprs: b,c,o,c1,o1
C0
n0
O0 0
n0
r
2 3
b
3
3
k1
1
J0 2
0 1
1 1
G0 2
0 1
1 2
";
    assert_eq!(s, expected);
}

#[test]
fn nl_milp() {
    // min  z + y
    // s.t. z + y <= 4
    // z continuous, y binary
    let m = Model::new("milp");
    let z = m.var("z").lb(0.0).ub(10.0).build();
    let y = m.var("y").binary().build();
    m.minimize(z + y);
    m.constraint("c0", (z + y).le(4.0));

    let s = to_nl_string(&m).expect("nl writer");
    // Variables: z (id 0, linear continuous, bucket LinC), y (id 1, linear binary, bucket LinB).
    // Permuted: z -> v0, y -> v1.
    let expected = "\
g3 1 1 0\t# problem milp
 2 1 1 0 0\t# vars, constraints, objectives, ranges, eqns
 0 0\t# nonlinear constraints, objectives
 0 0\t# network constraints: nonlinear, linear
 0 0 0\t# nonlinear vars in constraints, objectives, both
 0 0 0 1\t# linear network variables; functions; arith, flags
 1 0 0 0 0\t# discrete variables: binary, integer, nonlinear (b,c,o)
 2 2\t# nonzeros in Jacobian, gradients
 0 0\t# max name lengths: constraints, variables
 0 0 0 0 0\t# common exprs: b,c,o,c1,o1
C0
n0
O0 0
n0
r
1 4
b
0 0 10
0 0 1
k1
1
J0 2
0 1
1 1
G0 2
0 1
1 1
";
    assert_eq!(s, expected);
}

#[test]
fn nl_nlp_rosenbrock() {
    // min (1 - x)^2 + 100 (y - x^2)^2
    // x, y free
    let m = Model::new("rosen");
    let x = m.var("x").build();
    let y = m.var("y").build();
    m.minimize((1.0 - x).powi(2) + 100.0 * (y - x.powi(2)).powi(2));

    let s = to_nl_string(&m).expect("nl writer");
    // Both vars are nonlinear in obj only -> bucket NlOnlyO, var_order = [x, y].
    // No constraints. Header line 5: nl_vars_in_c=0, in_o=2, both=0.
    assert!(s.starts_with("g3 1 1 0\t# problem rosen\n"));
    assert!(s.contains(" 2 0 1 0 0\t# vars, constraints, objectives, ranges, eqns\n"));
    assert!(s.contains(" 0 1\t# nonlinear constraints, objectives\n"));
    assert!(s.contains(" 0 2 0\t# nonlinear vars in constraints, objectives, both\n"));
    // No J segments.
    assert!(!s.contains("\nJ"));
    // G has 2 entries with coef 0 (both vars nonlinear in obj).
    assert!(s.contains("\nG0 2\n0 0\n1 0\n"));
    // Objective body starts with O0 0 followed by an expression.
    assert!(s.contains("O0 0\n"));
}

#[test]
fn nl_abs_objective() {
    // min |x|, x free
    let m = Model::new("absobj");
    let x = m.var("x").build();
    m.minimize(x.abs());

    let s = to_nl_string(&m).expect("nl writer");
    // abs is nonlinear in the objective and lowers to the OPABS opcode o15.
    assert!(s.contains(" 0 1\t# nonlinear constraints, objectives\n"), "header:\n{s}");
    assert!(s.contains("O0 0\no15\nv0\n"), "nl body:\n{s}");
}

#[test]
fn nl_minlp() {
    // min  x*x + 3*y
    // s.t. x + y >= 1
    // x continuous free, y integer in [0, 5]
    let m = Model::new("mi");
    let x = m.var("x").build();
    let y = m.var("y").integer().lb(0.0).ub(5.0).build();
    m.minimize(x * x + 3.0 * y);
    m.constraint("c0", (x + y).ge(1.0));

    let s = to_nl_string(&m).expect("nl writer");
    // split_linear separates `3*y` (linear) from `x*x` (residual). Only x
    // sits in oV; y stays in the linear-integer bucket.
    // var_order: x (NlOnlyO, continuous) -> v0, y (LinI) -> v1.
    assert!(s.contains(" 2 1 1 0 0\t# vars, constraints, objectives, ranges, eqns\n"));
    assert!(s.contains(" 0 1\t# nonlinear constraints, objectives\n"));
    assert!(s.contains(" 0 1 0\t# nonlinear vars in constraints, objectives, both\n"));
    // y is a linear integer (not nonlinear) -> nl_int_o = 0.
    assert!(s.contains(" 0 1 0 0 0\t# discrete variables: binary, integer, nonlinear (b,c,o)\n"));
    // Constraint is linear: J0 should list both vars with coef 1.
    assert!(s.contains("\nJ0 2\n0 1\n1 1\n"));
    // Objective body: residual is `x*x`; linear part `3*y` -> G has x (coef 0) and y (coef 3).
    assert!(s.contains("\nG0 2\n0 0\n1 3\n"));
    // b segment: x is free (line "3"), y has bounds [0, 5] (line "0 0 5").
    assert!(s.contains("\nb\n3\n0 0 5\n"));
}

#[test]
fn nl_nonlinear_integer_order() {
    // x continuous, nonlinear in a constraint only            -> NlOnlyC
    // y integer,    nonlinear in BOTH objective and constraint -> NlBothI
    //
    // Gay Table 4 groups the nonlinear block by appearance (both, con-only,
    // obj-only), continuous before integer within each group. So NlBothI (y)
    // precedes NlOnlyC (x): var_order = [y, x], i.e. y -> v0, x -> v1.
    let m = Model::new("minlp_nl_int");
    let x = m.var("x").build();
    let y = m.var("y").integer().lb(0.0).ub(5.0).build();
    m.minimize(y * y);
    m.constraint("c0", (x * x + y * y).le(10.0));

    let s = to_nl_string(&m).expect("nl writer");
    // Nonlinear vars: in constraints = 2 (x, y), in objective = 1 (y), both = 1 (y).
    assert!(
        s.contains(" 2 1 1\t# nonlinear vars in constraints, objectives, both\n"),
        "header line 5:\n{s}"
    );
    // Discrete line: nonlinear integer in both (nlvbi) = 1.
    assert!(
        s.contains(" 0 0 1 0 0\t# discrete variables: binary, integer, nonlinear (b,c,o)\n"),
        "discrete line:\n{s}"
    );
    // The b segment is emitted in NL variable order. y (bounds [0,5]) before
    // x (free) proves the interleaved ordering: NlBothI precedes NlOnlyC.
    assert!(s.contains("\nb\n0 0 5\n3\n"), "y must precede x in var order:\n{s}");
}
