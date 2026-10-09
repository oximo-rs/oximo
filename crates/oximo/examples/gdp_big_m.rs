//! GDP modeled with blocks, reformulated in place and solved with HiGHS.
//!
//! Select exactly one operating mode. Each disjunct groups an operating range
//! and its fixed cost under a Boolean decision. The large unit requires a
//! minimum flow of 5 and has a fixed cost of 2.
//!
//! ```text
//! maximize       flow - cost
//! subject to     [ Y_small   ] XOR [ Y_large        ]
//!                [ flow <= 3 ]     [ 5 <= flow <= 8 ]
//!                [ cost = 0  ]     [ cost = 2       ]
//!                0 <= flow <= 10
//!                0 <= cost <= 2
//!                Y_small, Y_large in {true, false}
//! ```
//!
//! XOR selects exactly one disjunct. The Boolean decisions are represented by
//! disjunct handles, while binary counterparts are used by the algebraic reformulation.
//! Big-M values are inferred from the global variable bounds.
//!
//! Run: `cargo run -p oximo --example gdp_big_m --features gdp,highs`

use oximo::prelude::*;
use oximo::{HighsOptions, solvers::Highs};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Model::new("unit selection");

    variable!(model, 0.0 <= flow <= 10.0);
    variable!(model, 0.0 <= cost <= 2.0);

    let small = disjunct!(model, small, |d| {
        constraint!(d, capacity, flow <= 3.0);
        constraint!(d, fixed_cost, cost == 0.0);
    });
    let large = disjunct!(model, large, |d| {
        constraint!(d, operating_range, 5.0 <= flow <= 8.0);
        constraint!(d, fixed_cost, cost == 2.0);
    });
    disjunction!(model, unit, [small, large]);

    objective!(model, Max, flow - cost);

    let report = model.reformulate_gdp(BigM::default())?;

    let result = Highs.solve(&model, &HighsOptions::default())?;

    println!("{} conditional rows reformulated", report.rows.len());
    println!("termination: {:?}", result.termination);
    println!("objective: {:?}", result.objective());
    println!("flow: {:?}", result.value_of(flow)?);
    println!("cost: {:?}", result.value_of(cost)?);
    let large_selected = result.boolean_value_of(large)?.ok_or("missing selection value")?;
    println!("large unit selected: {large_selected}");

    Ok(())
}
