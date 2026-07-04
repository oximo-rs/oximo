//! Every text format rejects models with explicit second-order cone
//! constraints: LP/MPS/NL have no conic representation.

use oximo_core::prelude::*;
use oximo_io::{IoError, to_lp_string, to_mps_string, to_nl_string};

fn soc_model() -> Model {
    let m = Model::new("socp");
    variable!(m, x);
    variable!(m, y);
    variable!(m, t >= 0.0);
    m.add_soc_constraint("cone", [x, y], t);
    constraint!(m, c0, x + y >= 1.0);
    objective!(m, Min, t);
    m
}

#[test]
fn lp_writer_rejects_soc_constraints() {
    assert!(matches!(to_lp_string(&soc_model()), Err(IoError::Conic)));
}

#[test]
fn mps_writer_rejects_soc_constraints() {
    assert!(matches!(to_mps_string(&soc_model()), Err(IoError::Conic)));
}

#[test]
fn nl_writer_rejects_soc_constraints() {
    assert!(matches!(to_nl_string(&soc_model()), Err(IoError::Conic)));
}
