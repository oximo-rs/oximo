//! Nonlinear GDP with disjunct blocks, reformulated in place at solve time.
//!
//! Select exactly one operating mode, then maximize output minus its fixed cost.
//! Each Boolean disjunct groups an exponential capacity law and an operating
//! cost. The high-capacity mode also requires a minimum output of 1.
//!
//! ```text
//! maximize       x - cost
//! subject to     [ Y_low       ] XOR [ Y_high      ]
//!                [ exp(x) <= 2 ]     [ exp(x) <= 4 ]
//!                [ cost = 0    ]     [ x >= 1      ]
//!                                    [ cost = 0.25 ]
//!                0 <= x <= 3
//!                0 <= cost <= 0.25
//!                Y_low, Y_high in {true, false}
//! ```
//!
//! XOR selects exactly one Boolean disjunct. Big-M values are inferred from
//! the global variable bounds. Selecting Big-M with `with_gdp` reformulates
//! the model in place into a MINLP before solving it with SCIP.
//!
//! Run: `cargo run -p oximo --example gdp_nonlinear --features gdp,scip`

use oximo::prelude::*;
use oximo::{ScipOptions, solvers::Scip};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Model::new("nonlinear choices");

    variable!(model, 0.0 <= x <= 3.0);
    variable!(model, 0.0 <= cost <= 0.25);

    let low = disjunct!(model, low_capacity, |d| {
        constraint!(d, capacity, x.exp() <= 2.0);
        constraint!(d, fixed_cost, cost == 0.0);
    });
    let high = disjunct!(model, high_capacity, |d| {
        constraint!(d, capacity, x.exp() <= 4.0);
        constraint!(d, minimum_output, x >= 1.0);
        constraint!(d, fixed_cost, cost == 0.25);
    });
    disjunction!(model, operating_mode, [low, high]);
    objective!(model, Max, x - cost);

    let result =
        Scip::default().with_gdp(BigM::default()).solve(&model, &ScipOptions::default())?;

    println!("termination: {:?}", result.termination);
    println!("objective: {:?}", result.objective());
    println!("x: {:?}", result.value_of(x)?);
    println!("cost: {:?}", result.value_of(cost)?);
    let high_selected = result.boolean_value_of(high)?.ok_or("missing selection value")?;
    println!("high-capacity mode selected: {high_selected}");

    Ok(())
}
