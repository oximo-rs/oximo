//! Two semidefinite models:
//! - A minimum-trace PSD matrix,
//! - A largest-eigenvalue problem expressed as the affine matrix inequality tI - A >= 0.
//!
//! Their optimal objective values are 2 and 3, respectively.
//!
//! Run with a Clarabel SDP linear algebra provider, for example:
//! `cargo run -p oximo --example sdp --features clarabel-sdp-mkl`

#![allow(non_snake_case)]
use oximo::prelude::*;
use oximo::solvers::Clarabel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Minimum trace problem");
    minimum_trace()?;
    println!("\nLargest eigenvalue problem");
    largest_eigenvalue()
}

fn minimum_trace() -> Result<(), Box<dyn std::error::Error>> {
    let m = Model::new("minimum trace");

    // A 2x2 symmetric matrix has three independent scalar variables.
    symmetric_variable!(m, X[2]);

    // Require nonnegative eigenvalues.
    let positivity = psd_constraint!(m, positivity, &X);
    constraint!(m, X[0, 1] == 1.0);

    // With the off-diagonal fixed to 1, PSD requires nonnegative diagonal entries
    // and X[0, 0] * X[1, 1] >= 1. Their sum (the trace) is minimized.
    objective!(m, Min, X.trace());

    let result = Clarabel.solve(&m, &ClarabelOptions::default())?;
    println!("{}", result.report(&m)?);

    if let Some(primal) = result.value_of_matrix(&X)? {
        println!("X = {primal}");
    }

    // Y is the dual matrix for the PSD constraint, separate from the scalar
    // equality multiplier. At optimality, X and Y satisfy trace(X*Y) = 0
    // up to numerical tolerances.
    if let Some(dual) = result.psd_dual_of(positivity)? {
        println!("Y = {dual}");
    }
    Ok(())
}

fn largest_eigenvalue() -> Result<(), Box<dyn std::error::Error>> {
    let m = Model::new("largest eigenvalue");
    variable!(m, t);

    // Dense rows follow the usual matrix notation.
    let a = SymmetricMatrix::try_from_rows([[2.0, 1.0], [1.0, 2.0]])?;
    let identity = SymmetricMatrix::<f64>::identity(2);

    // The eigenvalues of tI - A are t minus the eigenvalues of A.
    let shifted = t * identity - &a;
    let bound = psd_constraint!(m, eigenvalue_bound, &shifted);
    objective!(m, Min, t);

    let result = Clarabel.solve(&m, &ClarabelOptions::default())?;
    println!("A = {a}");
    println!("{}", result.report(&m)?);
    if let Some(value) = result.value_of(t)? {
        println!("t = {value}");
    }

    if let Some(primal) = result.value_of_matrix(&shifted)? {
        println!("tI - A = {primal}");
    }

    if let Some(dual) = result.psd_dual_of(bound)? {
        println!("Y = {dual}");
    }
    Ok(())
}
