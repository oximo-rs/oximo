//! The LP/MPS writers name where a nonlinear term appears and render the
//! offending sub-expression, instead of a bare "nonlinear" error.

use oximo_core::prelude::*;
use oximo_io::{IoError, to_lp_string, to_mps_string};

fn bilinear_model() -> Model {
    let m = Model::new("bilinear");
    variable!(m, x);
    variable!(m, y);
    objective!(m, Min, x + y);
    constraint!(m, capacity, x * y <= 1.0);
    m
}

fn nonlinear_objective_model() -> Model {
    let m = Model::new("nlobj");
    variable!(m, theta);
    objective!(m, Min, theta.sin());
    constraint!(m, c0, theta >= 0.0);
    m
}

#[test]
fn lp_names_the_constraint_and_renders_the_term() {
    match to_lp_string(&bilinear_model()) {
        Err(IoError::Nonlinear { location, term }) => {
            assert_eq!(location, "constraint \"capacity\"");
            assert_eq!(term, "x * y");
        }
        other => panic!("expected a Nonlinear error, got {other:?}"),
    }
}

#[test]
fn lp_nonlinear_message_is_user_facing() {
    let msg = to_lp_string(&bilinear_model()).unwrap_err().to_string();
    assert_eq!(
        msg,
        "expected an affine expression in constraint \"capacity\", found nonlinear term: x * y"
    );
}

#[test]
fn lp_names_the_objective() {
    match to_lp_string(&nonlinear_objective_model()) {
        Err(IoError::Nonlinear { location, term }) => {
            assert_eq!(location, "the objective");
            assert_eq!(term, "sin(theta)");
        }
        other => panic!("expected a Nonlinear error, got {other:?}"),
    }
}

#[test]
fn mps_names_the_constraint_and_renders_the_term() {
    match to_mps_string(&bilinear_model()) {
        Err(IoError::Nonlinear { location, term }) => {
            assert_eq!(location, "constraint \"capacity\"");
            assert_eq!(term, "x * y");
        }
        other => panic!("expected a Nonlinear error, got {other:?}"),
    }
}
