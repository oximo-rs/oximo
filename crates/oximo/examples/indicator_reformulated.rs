//! Indicator constraints explicitly reformulated and solved with HiGHS.
//!
//! Run with: `cargo run -p oximo --example indicator_reformulated --features highs`

use oximo::prelude::*;
use oximo::{HighsOptions, solvers::Highs};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Model::new("indicator_reformulated_example");
    variable!(model, produce, Binary);
    variable!(model, 0.0 <= quantity <= 10.0);
    indicator_constraint!(model, capacity, produce == 1 => quantity <= 7.0);
    indicator_constraint!(model, shutdown, produce == 0 => quantity == 0.0);
    objective!(model, Max, quantity + 2.0 * produce);

    // Each row receives a tight Big-M derived from the finite quantity bounds.
    let artifacts = model.reformulate_indicators(IndicatorReformulationOptions::default())?;
    println!("generated rows for {} indicators", artifacts.len());

    let result = Highs.solve(&model, &HighsOptions::default())?;
    println!("termination = {:?}", result.termination);
    println!("produce = {:.6}", result.value_of(produce)?.unwrap_or_default());
    println!("quantity = {:.6}", result.value_of(quantity)?.unwrap_or_default());
    Ok(())
}
