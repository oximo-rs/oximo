//! Native binary-triggered affine constraints with Gurobi.
//!
//! Indicators constraints are natively supported by Gurobi and MOSEK;
//! GAMS requires an explicitly selected COPT, CPLEX, Gurobi, SCIP, or Xpress
//! subsolver.
//!
//! Run with:
//! `cargo run -p oximo --example indicator --features gurobi`

use oximo::prelude::*;
use oximo::{GurobiOptions, solvers::Gurobi};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Model::new("indicator_example");
    variable!(model, enabled, Binary);
    variable!(model, 0.0 <= production <= 20.0);

    indicator_constraint!(model, capacity, enabled == 1 => production <= 8.0);
    indicator_constraint!(model, shutdown, enabled == 0 => production == 0.0);
    objective!(model, Max, production - 2.0 * enabled);

    let result = Gurobi.solve(&model, &GurobiOptions::default())?;
    println!("enabled = {}", result.value_of(enabled)?.unwrap_or_default());
    println!("production = {}", result.value_of(production)?.unwrap_or_default());
    Ok(())
}
