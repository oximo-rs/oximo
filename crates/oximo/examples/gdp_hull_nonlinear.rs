//! Maximize a flow by choosing exactly one of two operating modes:
//!
//! ```text
//! maximize       flow
//! subject to     [ Y_active          ] XOR [ Y_low       ]
//!                [ ln(flow) <= ln(3) ]     [ flow <= 1.5 ]
//!                1 <= flow <= 4
//! ```
//!
//! The logarithmic mode permits flow up to 3, so the optimal solution selects
//! that mode and has `flow = 3`.
//!
//! Hull represents the Boolean decisions with binary variables `y_active` and
//! `y_low` and introduces a shifted copy for each mode, using reference coordinate
//! `a = 1`. At binary values, `z_mode = y_mode * (flow - 1)`.
//! With `d(y) = epsilon + (1 - epsilon) * y`, the transformed constraints are:
//!
//! ```text
//! y_active + y_low = 1, with y_active, y_low in {0, 1}
//! flow = 1 + z_active + z_low
//!
//! 0 <= z_active <= 3 * y_active
//! 0 <= z_low <= 3 * y_low
//!
//! d(y_active) * ln(1 + z_active / d(y_active)) <= y_active * ln(3)
//! z_low <= 0.5 * y_low
//! ```
//!
//! Since `epsilon > 0` and the copies are nonnegative, the logarithm's argument
//! is always at least 1, even when its mode is inactive. A zero reference would
//! require evaluating `ln(0)` and is rejected during reformulation.
//!
//! Run: `cargo run -p oximo --example gdp_hull_nonlinear --features gdp,scip`

use oximo::prelude::*;
use oximo::{ScipOptions, solvers::Scip};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Model::new("shifted nonlinear Hull");

    variable!(model, 1.0 <= flow <= 4.0);
    let active = disjunct!(model, active, |d| {
        constraint!(d, flow.ln() <= 3.0_f64.ln());
    });
    let low = disjunct!(model, low, |d| {
        constraint!(d, flow <= 1.5);
    });
    disjunction!(model, [active, low]);
    objective!(model, Max, flow);

    let hull = GdpReformulationOptions::new(Hull::default()).with_hull_reference_point(flow, 1.0);
    let result = Scip::default().with_gdp(hull).solve(&model, &ScipOptions::default())?;

    println!(
        "termination: {:?}, flow: {:?}, active: {:?}",
        result.termination,
        result.value_of(flow)?,
        result.boolean_value_of(active)?
    );

    Ok(())
}
